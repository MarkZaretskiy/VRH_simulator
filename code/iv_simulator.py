from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import MatrixRankWarning

try:
    from .sim import RRNSolver, contact_nodes_from_x, make_random_sites
    from .plotting import (
        format_box_plot_annotation,
        plot_conductance_vs_temperature,
        plot_iv_curves,
        resolve_plot_output_dir,
    )
    from .utils import (
        build_sweep_values,
        load_mapping_file,
        normalize_config_mapping,
        parse_cli_args,
        parse_float_values,
        parse_index_array,
        parse_optional_array,
    )
except ImportError:
    from sim import RRNSolver, contact_nodes_from_x, make_random_sites
    from plotting import (
        format_box_plot_annotation,
        plot_conductance_vs_temperature,
        plot_iv_curves,
        resolve_plot_output_dir,
    )
    from utils import (
        build_sweep_values,
        load_mapping_file,
        normalize_config_mapping,
        parse_cli_args,
        parse_float_values,
        parse_index_array,
        parse_optional_array,
    )


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "iv_simulator_results.csv"
DEFAULT_REFERENCE_VOLTAGE = 1.0
DEFAULT_MAX_GENERATED_SITES = 3000
DEFAULT_CONTACT_LAYER_THICKNESS_NM = 5.0
DEFAULT_NEAREST_NEIGHBOR_STATS_K = 3
NM_TO_CM = 1.0e-7
SUPPORTED_ENERGY_DISTRIBUTIONS = frozenset({"gaussian", "uniform"})


def validate_positive_params(
    params: dict[str, float | int],
    *,
    suffix: str = "must be > 0",
) -> None:
    for name, value in params.items():
        if value <= 0:
            raise ValueError(f"{name} {suffix}")


def validate_positions_and_energies(
    positions_array: np.ndarray | None,
    energies_array: np.ndarray | None,
) -> None:
    if (positions_array is None) != (energies_array is None):
        raise ValueError("positions and energies must be provided together")
    if positions_array is not None and energies_array is not None:
        if len(positions_array) != len(energies_array):
            raise ValueError("positions and energies must have the same length")


def validate_contact_node_arrays(
    left_nodes_array: np.ndarray | None,
    right_nodes_array: np.ndarray | None,
) -> None:
    if (left_nodes_array is None) != (right_nodes_array is None):
        raise ValueError("left_nodes and right_nodes must be provided together")
    if left_nodes_array is not None and right_nodes_array is not None:
        if np.intersect1d(left_nodes_array, right_nodes_array).size > 0:
            raise ValueError("left_nodes and right_nodes must not overlap")


def validate_energy_distribution(energy_distribution: str) -> None:
    if energy_distribution not in SUPPORTED_ENERGY_DISTRIBUTIONS:
        supported = ", ".join(sorted(SUPPORTED_ENERGY_DISTRIBUTIONS))
        raise ValueError(
            f"energy_distribution must be one of: {supported}"
        )


def sample_site_energies(
    *,
    rng: np.random.Generator,
    energy_std: float,
    energy_distribution: str,
    n_sites: int,
) -> np.ndarray:
    validate_energy_distribution(energy_distribution)
    if energy_distribution == "gaussian":
        return rng.normal(0.0, energy_std, size=n_sites)
    half_span = 0.5 * energy_std
    return rng.uniform(-half_span, half_span, size=n_sites)


def resolve_contact_region_width() -> float:
    return DEFAULT_CONTACT_LAYER_THICKNESS_NM


def nearest_boundary_node(
    positions: np.ndarray,
    target_x: float,
    *,
    exclude: int | None = None,
) -> int:
    candidate_order = np.argsort(np.abs(positions[:, 0] - float(target_x)))
    for candidate_index in candidate_order:
        if exclude is None or int(candidate_index) != exclude:
            return int(candidate_index)
    raise ValueError("could not select a distinct nearest boundary node")


@dataclass(slots=True)
class SimulationConfig:
    positions: str | list[list[float]] | np.ndarray | None = None
    energies: str | list[float] | np.ndarray | None = None
    concentration_cm3: float | None = 1.8e20
    temperatures_k: (
        str | float | int | list[float] | tuple[float, ...] | np.ndarray | None
    ) = None
    xi: float = 0.35
    G0: float = 1.0
    kB: float = 8.617333262145e-5
    cutoff_distance: float | None = 20.0
    max_neighbors: int | None = 100
    min_conductance: float = 0.0
    left_nodes: str | list[int] | np.ndarray | None = None
    right_nodes: str | list[int] | np.ndarray | None = None
    t_min_k: float = 5.0
    t_max_k: float = 250.0
    t_step_k: float = 25.0
    v_min: float = -1.0
    v_max: float = 1.0
    v_step: float = 0.1
    output: str | None = str(DEFAULT_OUTPUT_PATH)
    no_output: bool = False
    plot: bool = False
    plot_output_dir: str | None = None
    show_plots: bool = False
    n_realizations: int = 4
    n_jobs: int | None = None
    device_length_nm: float = 120.0
    device_width_nm: float = 30.0
    device_thickness_nm: float = 3.0
    max_generated_sites: int = DEFAULT_MAX_GENERATED_SITES
    n_sites: int = 300
    length: float = 20.0
    dim: int = 1
    energy_std: float = 0.4
    energy_distribution: str = "gaussian"
    seed: int | None = 42

    positions_array: np.ndarray | None = field(init=False, repr=False)
    energies_array: np.ndarray | None = field(init=False, repr=False)
    left_nodes_array: np.ndarray | None = field(init=False, repr=False)
    right_nodes_array: np.ndarray | None = field(init=False, repr=False)
    temperature_values_k: np.ndarray = field(init=False, repr=False)
    voltage_values_v: np.ndarray = field(init=False, repr=False)
    resolved_n_jobs: int = field(init=False)

    def __post_init__(self) -> None:
        self.positions_array = parse_optional_array(
            self.positions,
            name="positions",
            ndim=2,
        )
        self.energies_array = parse_optional_array(
            self.energies,
            name="energies",
            ndim=1,
        )
        self.left_nodes_array = parse_index_array(
            self.left_nodes,
            name="left_nodes",
        )
        self.right_nodes_array = parse_index_array(
            self.right_nodes,
            name="right_nodes",
        )

        validate_positive_params(
            {
                "n_realizations": self.n_realizations,
                "xi": self.xi,
                "G0": self.G0,
                "energy_std": self.energy_std,
            }
        )
        validate_energy_distribution(self.energy_distribution)

        validate_positions_and_energies(
            self.positions_array,
            self.energies_array,
        )
        if self.positions_array is not None and self.energies_array is not None:
            if self.n_realizations != 1:
                raise ValueError(
                    "n_realizations > 1 requires a generated disorder network; "
                    "for explicit positions/energies use n_realizations=1"
                )
        elif self.concentration_cm3 is not None:
            validate_positive_params(
                {
                    "concentration_cm3": self.concentration_cm3,
                    "device_length_nm": self.device_length_nm,
                    "device_width_nm": self.device_width_nm,
                    "device_thickness_nm": self.device_thickness_nm,
                    "max_generated_sites": self.max_generated_sites,
                }
            )

        validate_contact_node_arrays(
            self.left_nodes_array,
            self.right_nodes_array,
        )

        self.temperature_values_k = build_temperature_values(
            temperatures_k=self.temperatures_k,
            t_min_k=self.t_min_k,
            t_max_k=self.t_max_k,
            t_step_k=self.t_step_k,
        )
        self.voltage_values_v = build_sweep_values(self.v_min, self.v_max, self.v_step)
        self.resolved_n_jobs = resolve_n_jobs(
            n_jobs=self.n_jobs,
            n_realizations=self.n_realizations,
        )


SIMULATION_CONFIG_FIELD_NAMES = frozenset(
    config_field.name for config_field in fields(SimulationConfig) if config_field.init
)
SIMULATION_CONFIG_BOOL_FIELDS = frozenset(
    config_field.name
    for config_field in fields(SimulationConfig)
    if config_field.init and isinstance(config_field.default, bool)
)


def load_config_dict(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    data = load_mapping_file(config_path)
    return normalize_config_mapping(
        data,
        source_name=f"config file {config_path}",
        allowed_keys=SIMULATION_CONFIG_FIELD_NAMES,
    )


def build_config_from_mapping(data: Mapping[str, object]) -> SimulationConfig:
    normalized_data = normalize_config_mapping(
        data,
        source_name="configuration",
        allowed_keys=SIMULATION_CONFIG_FIELD_NAMES,
    )
    return SimulationConfig(**normalized_data)


def build_config_from_file(
    path: str | Path,
) -> SimulationConfig:
    config_data = load_config_dict(path)
    return SimulationConfig(**config_data)


def cli_usage() -> str:
    return (
        "Usage:\n"
        "  python code/iv_simulator.py [--field value ...]\n"
        "  python code/iv_simulator.py --config CONFIG_PATH\n\n"
        "Notes:\n"
        "  - field names match SimulationConfig, for example --xi, --v_step, --plot\n"
        "  - boolean flags accept true/false, or can be passed without a value to mean true\n"
        "  - when --config is provided, all simulation parameters are read from the file\n"
    )


def build_temperature_values(
    temperatures_k: (
        str | float | int | list[float] | tuple[float, ...] | np.ndarray | None
    ),
    t_min_k: float,
    t_max_k: float,
    t_step_k: float,
) -> np.ndarray:
    parsed_temperatures = parse_float_values(temperatures_k, name="temperatures_k")
    if parsed_temperatures is not None:
        if np.any(parsed_temperatures <= 0.0):
            raise ValueError("all temperatures must be > 0")
        return parsed_temperatures
    return build_sweep_values(t_min_k, t_max_k, t_step_k)


def write_iv_csv(
    temperatures_k: np.ndarray,
    voltages_v: np.ndarray,
    current_mean_a: np.ndarray,
    current_std_a: np.ndarray,
    conductance_mean_s: np.ndarray,
    conductance_std_s: np.ndarray,
    n_realizations: int,
    non_conductive_counts: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_index = np.repeat(np.arange(temperatures_k.size), voltages_v.size)
    df = pd.DataFrame(
        {
            "temperature_K": temperatures_k[row_index],
            "voltage_V": np.tile(voltages_v, temperatures_k.size),
            "current_mean_A": current_mean_a.reshape(-1),
            "current_std_A": current_std_a.reshape(-1),
            "conductance_mean_S": conductance_mean_s[row_index],
            "conductance_std_S": conductance_std_s[row_index],
            "n_realizations": np.full(row_index.size, n_realizations, dtype=int),
            "non_conductive_realizations": non_conductive_counts[row_index],
        }
    )
    df.to_csv(output_path, index=False)


def build_network(
    config: SimulationConfig,
    *,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if config.positions_array is not None and config.energies_array is not None:
        return (
            config.positions_array,
            config.energies_array,
            {
                "mode": "explicit",
                "n_sites": int(len(config.energies_array)),
                "dim": int(config.positions_array.shape[1]),
                "x_min": float(np.min(config.positions_array[:, 0])),
                "x_max": float(np.max(config.positions_array[:, 0])),
            },
        )

    if config.concentration_cm3 is not None:
        volume_nm3 = (
            config.device_length_nm
            * config.device_width_nm
            * config.device_thickness_nm
        )
        volume_cm3 = volume_nm3 * (NM_TO_CM**3)
        expected_n_sites = config.concentration_cm3 * volume_cm3
        generated_n_sites = max(2, int(round(expected_n_sites)))
        if generated_n_sites > config.max_generated_sites:
            raise ValueError(
                "concentration_cm3 and box geometry generate "
                f"{generated_n_sites} sites, which is too large for this explicit O(N^2) solver; "
                "reduce geometry/concentration or raise max_generated_sites intentionally"
            )
        rng = np.random.default_rng(seed)
        positions_array = np.column_stack(
            [
                rng.uniform(0.0, config.device_length_nm, size=generated_n_sites),
                rng.uniform(0.0, config.device_width_nm, size=generated_n_sites),
                rng.uniform(0.0, config.device_thickness_nm, size=generated_n_sites),
            ]
        )  # TODO: change z from uniform to exponential to mimick 2d
        energies_array = sample_site_energies(
            rng=rng,
            energy_std=config.energy_std,
            energy_distribution=config.energy_distribution,
            n_sites=generated_n_sites,
        )
        actual_concentration_cm3 = generated_n_sites / volume_cm3
        return (
            positions_array,
            energies_array,
            {
                "mode": "box",
                "n_sites": int(generated_n_sites),
                "dim": 3,
                "device_length_nm": float(config.device_length_nm),
                "device_width_nm": float(config.device_width_nm),
                "device_thickness_nm": float(config.device_thickness_nm),
                "expected_n_sites": float(expected_n_sites),
                "concentration_cm3": float(config.concentration_cm3),
                "actual_concentration_cm3": float(actual_concentration_cm3),
                "energy_scale": float(config.energy_std),
                "energy_distribution": str(config.energy_distribution),
                "x_min": 0.0,
                "x_max": float(config.device_length_nm),
            },
        )

    positions_array, energies_array = make_random_sites(
        n_sites=config.n_sites,
        length=config.length,
        dim=config.dim,
        energy_std=config.energy_std,
        energy_distribution=config.energy_distribution,
        seed=seed,
    )
    return (
        positions_array,
        energies_array,
        {
            "mode": "random",
            "n_sites": int(config.n_sites),
            "dim": int(config.dim),
            "length": float(config.length),
            "energy_scale": float(config.energy_std),
            "energy_distribution": str(config.energy_distribution),
            "x_min": 0.0,
            "x_max": float(config.length),
        },
    )


def summarize_samples(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_realizations, n_temperatures)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(samples, axis=0), np.nanstd(samples, axis=0, ddof=0)


def summarize_neighbor_distances(
    positions: np.ndarray,
    *,
    k: int = DEFAULT_NEAREST_NEIGHBOR_STATS_K,
) -> dict[str, object]:
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.ndim != 2:
        raise ValueError("positions must have shape (N, d)")
    if positions_array.shape[0] < 2:
        return {
            "neighbor_rank_means_nm": np.empty(0, dtype=float),
            "pooled_mean_nm": np.nan,
        }

    dummy_energies = np.zeros(positions_array.shape[0], dtype=float)
    solver = RRNSolver(
        positions=positions_array,
        energies=dummy_energies,
        temperature=300.0,
        xi=1.0,
    )
    stats = solver.nearest_neighbor_distance_statistics(k=k)
    per_neighbor = stats["per_neighbor"]
    pooled = stats["pooled"]

    return {
        "neighbor_rank_means_nm": np.asarray(
            [neighbor_stats.mean for neighbor_stats in per_neighbor],
            dtype=float,
        ),
        "pooled_mean_nm": (
            float(pooled.mean)
            if pooled is not None
            else np.nan
        ),
    }


def realization_seed(base_seed: int | None, realization_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + realization_index


def resolve_n_jobs(
    n_jobs: int | None,
    n_realizations: int,
) -> int:
    if n_jobs is None:
        return max(1, min(n_realizations, os.cpu_count() or 1))
    validate_positive_params(
        {"n_jobs": n_jobs},
        suffix="must be > 0 when provided",
    )
    return min(n_jobs, n_realizations)


def configure_parallel_process_env() -> None:
    # to prevent overscripting, might be reworked in the future, if we will have more thoughts and desire at how to solve our equations
    #  numerically more stable and efficient
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def iter_realization_results(
    n_realizations: int,
    resolved_n_jobs: int,
    config: SimulationConfig,
) -> Iterator[dict[str, object]]:
    if resolved_n_jobs == 1:
        for realization_index in range(n_realizations):
            yield run_realization(realization_index, config)
        return

    configure_parallel_process_env()
    with ProcessPoolExecutor(
        max_workers=resolved_n_jobs,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                run_realization,
                realization_index,
                config,
            )
            for realization_index in range(n_realizations)
        ]
        for future in as_completed(futures):
            yield future.result()


def print_network_metadata(result: dict[str, object]) -> None:
    print(
        f"Network: n_sites={int(result['n_sites'])}, dim={int(result['dim'])}, "
        f"left_contacts={int(result['left_contacts'])}, right_contacts={int(result['right_contacts'])}"
    )
    network_metadata = result["network_metadata"]
    if not isinstance(network_metadata, dict):
        return

    if network_metadata["mode"] == "box":
        print(
            "Box geometry: "
            f"L={network_metadata['device_length_nm']:.3g} nm, "
            f"W={network_metadata['device_width_nm']:.3g} nm, "
            f"T={network_metadata['device_thickness_nm']:.3g} nm"
        )
        print(
            "Concentration: "
            f"target={network_metadata['concentration_cm3']:.3e} cm^-3, "
            f"expected_n_sites={network_metadata['expected_n_sites']:.3f}, "
            f"actual={network_metadata['actual_concentration_cm3']:.3e} cm^-3"
        )
        print(
            "Energy distribution: "
            f"{network_metadata['energy_distribution']}, "
            f"W_E={network_metadata['energy_scale']:.3g} eV"
        )
    elif network_metadata["mode"] == "random":
        print(
            f"Random cloud: length={network_metadata['length']:.3g}, "
            f"dim={int(network_metadata['dim'])}, "
            f"energy_distribution={network_metadata['energy_distribution']}, "
            f"W_E={network_metadata['energy_scale']:.3g} eV"
        )


def store_realization_result(
    result: dict[str, object],
    conductance_samples: np.ndarray,
    non_conductive_mask: np.ndarray,
) -> None:
    realization_index = int(result["realization_index"])
    conductance_samples[realization_index] = np.asarray(
        result["conductance_curve"],
        dtype=float,
    )
    non_conductive_mask[realization_index] = np.asarray(
        result["non_conductive_curve"],
        dtype=bool,
    )


def print_realization_progress(
    result: dict[str, object],
    n_realizations: int,
) -> None:
    non_conductive_curve = np.asarray(result["non_conductive_curve"], dtype=bool)
    print(
        f"Completed realization {int(result['realization_index']) + 1}/{n_realizations} "
        f"(seed={result['seed']}, "
        f"non_conductive_t={int(np.count_nonzero(non_conductive_curve))}/{len(non_conductive_curve)})"
    )


def run_realization(
    realization_index: int,
    config: SimulationConfig,
) -> dict[str, object]:
    current_seed = realization_seed(config.seed, realization_index)
    positions_array, energies_array, network_metadata = build_network(
        config,
        seed=current_seed,
    )
    left_nodes_array, right_nodes_array = build_contacts(
        config,
        positions=positions_array,
        x_min=float(network_metadata["x_min"]),
        x_max=float(network_metadata["x_max"]),
    )
    neighbor_distance_summary = summarize_neighbor_distances(
        positions_array,
        k=DEFAULT_NEAREST_NEIGHBOR_STATS_K,
    )

    conductance_curve = np.full(config.temperature_values_k.size, np.nan, dtype=float)
    non_conductive_curve = np.zeros(config.temperature_values_k.size, dtype=bool)
    for temperature_index, temperature_k in enumerate(config.temperature_values_k):
        try:
            conductance_s = compute_conductance_for_temperature(
                positions=positions_array,
                energies=energies_array,
                temperature_k=float(temperature_k),
                xi=config.xi,
                left_nodes=left_nodes_array,
                right_nodes=right_nodes_array,
                g0=config.G0,
                kB=config.kB,
                cutoff_distance=config.cutoff_distance,
                max_neighbors=config.max_neighbors,
                min_conductance=config.min_conductance,
                reference_voltage=DEFAULT_REFERENCE_VOLTAGE,
            )
        except (MatrixRankWarning, FloatingPointError, ValueError, RuntimeError) as exc:
            if isinstance(exc, ValueError) and "no conductive path connects" not in str(
                exc
            ):
                raise
            if isinstance(exc, RuntimeError) and "factorize matrix" not in str(exc):
                raise
            conductance_s = np.nan
            non_conductive_curve[temperature_index] = True
        conductance_curve[temperature_index] = conductance_s

    return {
        "realization_index": realization_index,
        "seed": current_seed,
        "conductance_curve": conductance_curve,
        "non_conductive_curve": non_conductive_curve,
        "network_metadata": network_metadata,
        "n_sites": int(len(energies_array)),
        "dim": int(positions_array.shape[1]),
        "left_contacts": int(len(left_nodes_array)),
        "right_contacts": int(len(right_nodes_array)),
        "neighbor_distance_summary": neighbor_distance_summary,
    }


def build_contacts(
    config: SimulationConfig,
    positions: np.ndarray,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if config.left_nodes_array is not None and config.right_nodes_array is not None:
        return config.left_nodes_array, config.right_nodes_array

    if x_min is None:
        x_min = float(np.min(positions[:, 0]))
    if x_max is None:
        x_max = float(np.max(positions[:, 0]))
    contact_region_width = resolve_contact_region_width()
    left_nodes_array, right_nodes_array = contact_nodes_from_x(
        positions,
        contact_width=contact_region_width,
        x_min=x_min,
        x_max=x_max,
    )
    if left_nodes_array.size == 0:
        left_nodes_array = np.asarray(
            [nearest_boundary_node(positions, x_min)],
            dtype=int,
        )
    if right_nodes_array.size == 0:
        excluded_index = (
            int(left_nodes_array[0]) if left_nodes_array.size == 1 else None
        )
        right_nodes_array = np.asarray(
            [nearest_boundary_node(positions, x_max, exclude=excluded_index)],
            dtype=int,
        )
    if np.intersect1d(left_nodes_array, right_nodes_array).size > 0:
        raise ValueError(
            "automatic contact selection could not assign distinct left/right contact nodes"
        )
    return left_nodes_array, right_nodes_array


def compute_conductance_for_temperature(
    positions: np.ndarray,
    energies: np.ndarray,
    temperature_k: float,
    xi: float,
    left_nodes: np.ndarray,
    right_nodes: np.ndarray,
    g0: float,
    kB: float,
    cutoff_distance: float | None,
    max_neighbors: int | None,
    min_conductance: float,
    reference_voltage: float,
) -> float:
    solver = RRNSolver(
        positions=positions,
        energies=energies,
        temperature=temperature_k,
        xi=xi,
        G0=g0,
        kB=kB,
        cutoff_distance=cutoff_distance,
        max_neighbors=max_neighbors,
        min_conductance=min_conductance,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        result = solver.solve(
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            V_left=reference_voltage,
            V_right=0.0,
        )

    conductance_s = float(result.effective_conductance)
    if not np.isfinite(conductance_s):
        raise FloatingPointError("non-finite effective conductance")
    if conductance_s < 0.0:
        conductance_s = 0.0
    return conductance_s


def run(config: SimulationConfig) -> None:
    temperatures_k = config.temperature_values_k
    voltages_v = config.voltage_values_v
    # number of realizations we will average over
    n_realizations = config.n_realizations
    resolved_n_jobs = config.resolved_n_jobs

    conductance_samples = np.full(
        (n_realizations, temperatures_k.size),
        np.nan,
        dtype=float,
    )
    non_conductive_mask = np.zeros_like(conductance_samples, dtype=bool)

    print(
        f"Running linear I-V sweep for {len(temperatures_k)} temperatures "
        f"and {len(voltages_v)} voltages"
    )
    print(f"Ensemble realizations: {n_realizations}, base_seed={config.seed}")
    print(f"Parallel workers: {resolved_n_jobs}")

    metadata_printed = False
    for result in iter_realization_results(
        n_realizations=n_realizations,
        resolved_n_jobs=resolved_n_jobs,
        config=config,
    ):
        store_realization_result(
            result,
            conductance_samples,
            non_conductive_mask,
        )
        if not metadata_printed:
            print_network_metadata(result)
            metadata_printed = True
        print_realization_progress(result, n_realizations)

    conductance_mean_s, conductance_std_s = summarize_samples(conductance_samples)
    current_mean_a = conductance_mean_s[:, None] * voltages_v[None, :]
    current_std_a = conductance_std_s[:, None] * np.abs(voltages_v[None, :])
    zero_voltage_mask = np.isclose(voltages_v, 0.0)
    if np.any(zero_voltage_mask):
        current_mean_a[:, zero_voltage_mask] = 0.0
        current_std_a[:, zero_voltage_mask] = 0.0
    non_conductive_counts = non_conductive_mask.sum(axis=0)
    output_path = (
        None
        if config.no_output
        else (Path(config.output) if config.output is not None else DEFAULT_OUTPUT_PATH)
    )

    print("")
    print("Ensemble summary:")
    for temperature_index, temperature_k in enumerate(temperatures_k):
        print(
            f"T={temperature_k:6.1f} K  "
            f"G_mean={conductance_mean_s[temperature_index]:.6e} S  "
            f"G_std={conductance_std_s[temperature_index]:.6e} S  "
            f"non_conductive={int(non_conductive_counts[temperature_index])}/{n_realizations}  "
            f"I_mean({voltages_v[-1]:.3g} V)={current_mean_a[temperature_index, -1]:.6e} A"
        )

    if output_path is not None:
        write_iv_csv(
            temperatures_k=temperatures_k,
            voltages_v=voltages_v,
            current_mean_a=current_mean_a,
            current_std_a=current_std_a,
            conductance_mean_s=conductance_mean_s,
            conductance_std_s=conductance_std_s,
            n_realizations=n_realizations,
            non_conductive_counts=non_conductive_counts,
            output_path=output_path,
        )
        print(f"\nSaved I-V table to {output_path}")

    if config.plot:
        plot_dir = resolve_plot_output_dir(
            plot_output_dir=config.plot_output_dir,
            output_path=output_path,
        )
        plot_annotation = format_box_plot_annotation(
            concentration_cm3=config.concentration_cm3,
            device_length_nm=config.device_length_nm,
            device_width_nm=config.device_width_nm,
            device_thickness_nm=config.device_thickness_nm,
        )
        print("")
        plot_iv_curves(
            temperatures_k=temperatures_k,
            voltages_v=voltages_v,
            current_mean_a=current_mean_a,
            current_std_a=current_std_a,
            output_dir=plot_dir,
            n_realizations=n_realizations,
            plot_annotation=plot_annotation,
            show_plots=config.show_plots,
        )
        plot_conductance_vs_temperature(
            temperatures_k=temperatures_k,
            conductance_mean_s=conductance_mean_s,
            conductance_std_s=conductance_std_s,
            output_dir=plot_dir,
            n_realizations=n_realizations,
            plot_annotation=plot_annotation,
            show_plots=config.show_plots,
        )


def main(
    positions: str | list[list[float]] | np.ndarray | None = None,
    energies: str | list[float] | np.ndarray | None = None,
    concentration_cm3: float | None = 1.8e20,
    temperatures_k: (
        str | float | int | list[float] | tuple[float, ...] | np.ndarray | None
    ) = None,
    xi: float = 0.35,
    G0: float = 1.0,
    kB: float = 8.617333262145e-5,
    cutoff_distance: float | None = 20.0,
    max_neighbors: int | None = 100,
    min_conductance: float = 0.0,
    left_nodes: str | list[int] | np.ndarray | None = None,
    right_nodes: str | list[int] | np.ndarray | None = None,
    t_min_k: float = 5.0,
    t_max_k: float = 250.0,
    t_step_k: float = 25.0,
    v_min: float = -1.0,
    v_max: float = 1.0,
    v_step: float = 0.1,
    output: str | None = str(DEFAULT_OUTPUT_PATH),
    no_output: bool = False,
    plot: bool = False,
    plot_output_dir: str | None = None,
    show_plots: bool = False,
    n_realizations: int = 4,
    n_jobs: int | None = None,
    device_length_nm: float = 120.0,
    device_width_nm: float = 30.0,
    device_thickness_nm: float = 3.0,
    max_generated_sites: int = DEFAULT_MAX_GENERATED_SITES,
    n_sites: int = 300,
    length: float = 20.0,
    dim: int = 1,
    energy_std: float = 0.4,
    energy_distribution: str = "gaussian",
    seed: int | None = 42,
) -> None:

    #  we need it here because class SimulationConfig has logic with checking validity of argument
    config = SimulationConfig(
        positions=positions,
        energies=energies,
        concentration_cm3=concentration_cm3,
        temperatures_k=temperatures_k,
        xi=xi,
        kB=kB,
        G0=G0,
        cutoff_distance=cutoff_distance,
        max_neighbors=max_neighbors,
        min_conductance=min_conductance,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        t_min_k=t_min_k,
        t_max_k=t_max_k,
        t_step_k=t_step_k,
        v_min=v_min,
        v_max=v_max,
        v_step=v_step,
        output=output,
        no_output=no_output,
        plot=plot,
        plot_output_dir=plot_output_dir,
        show_plots=show_plots,
        n_realizations=n_realizations,
        n_jobs=n_jobs,
        device_length_nm=device_length_nm,
        device_width_nm=device_width_nm,
        device_thickness_nm=device_thickness_nm,
        max_generated_sites=max_generated_sites,
        n_sites=n_sites,
        length=length,
        dim=dim,
        energy_std=energy_std,
        energy_distribution=energy_distribution,
        seed=seed,
    )
    run(config)


def main_from_file(config_path: str | Path) -> None:
    config = build_config_from_file(config_path)
    run(config)


def cli(argv: list[str] | None = None) -> None:
    command = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in command):
        print(cli_usage(), end="")
        return
    try:
        config_path, params = parse_cli_args(
            command,
            option_names=SIMULATION_CONFIG_FIELD_NAMES,
            bool_fields=SIMULATION_CONFIG_BOOL_FIELDS,
        )
        if config_path is not None and params:
            raise ValueError(
                "cannot combine --config with other simulation options"
            )
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\n{cli_usage()}") from exc

    if config_path is None:
        main(**params)
    else:
        main_from_file(config_path)


if __name__ == "__main__":
    cli()
