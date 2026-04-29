from __future__ import annotations

import csv
import contextlib
import json
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .sim import RRNSolver
except ImportError:
    from sim import RRNSolver


NM_TO_CM = 1.0e-7
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("1d_simulator_conductivity.csv")
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "1d_simulator_config.json"
)


@dataclass(frozen=True)
class OneDRRNConfig:
    """
    Parameters for the regular 1D Random Resistor Network described in
    arXiv:2601.01243v1, Sec. 3A.

    Energies are sampled from a uniform density of states in
    [-energy_span_ev / 2, energy_span_ev / 2]. Spatial sites are placed on a
    regular 1D grid with spacing delta_min_nm.
    """

    n_sites: int = 100
    delta_min_nm: float = 1.0
    energy_span_ev: float = 0.4
    xi_nm: float = 0.15
    temperatures_k: tuple[float, ...] = tuple(np.arange(100.0, 401.0, 10.0))
    cross_section_area_nm2: float = 1.0
    n_realizations: int = 1
    outlier_iqr_factor: float | None = 1.5
    seed: int = 42
    g0: float = 1.0
    min_conductance: float = 0.0
    max_neighbors: int | None = None

    @property
    def length_nm(self) -> float:
        return self.delta_min_nm * (self.n_sites - 1)


@dataclass(frozen=True)
class ConductivitySweep:
    temperatures_k: np.ndarray
    conductance_s: np.ndarray
    conductance_std_s: np.ndarray
    typical_conductance_s: np.ndarray
    mean_ln_conductance: np.ndarray
    std_ln_conductance: np.ndarray
    conductivity_s_per_cm: np.ndarray
    conductivity_std_s_per_cm: np.ndarray
    typical_conductivity_s_per_cm: np.ndarray
    mean_ln_conductivity: np.ndarray
    std_ln_conductivity: np.ndarray
    n_realizations_used: np.ndarray
    conductance_samples_s: np.ndarray
    filtered_conductance_samples_s: np.ndarray
    outlier_mask: np.ndarray
    realization_positions_nm: np.ndarray
    realization_energies_ev: np.ndarray
    positions_nm: np.ndarray
    energies_ev: np.ndarray


def parse_temperatures(
    value: str | float | int | list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)

    if isinstance(value, str) and ":" in value and "," not in value:
        start_text, stop_text, step_text = (part.strip() for part in value.split(":"))
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        if step <= 0.0:
            raise ValueError("range step must be > 0")
        values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    elif isinstance(value, str):
        values = np.array(
            [float(part.strip()) for part in value.split(",") if part.strip()],
            dtype=float,
        )
    else:
        values = np.array(value, dtype=float)

    if values.size == 0:
        raise ValueError("range must contain at least one value")
    return tuple(float(value) for value in values)


def validate_config(config: OneDRRNConfig) -> None:
    if config.n_sites < 2:
        raise ValueError("n_sites must be >= 2")
    if config.delta_min_nm <= 0.0:
        raise ValueError("delta_min_nm must be > 0")
    if config.energy_span_ev <= 0.0:
        raise ValueError("energy_span_ev must be > 0")
    if config.xi_nm <= 0.0:
        raise ValueError("xi_nm must be > 0")
    if config.cross_section_area_nm2 <= 0.0:
        raise ValueError("cross_section_area_nm2 must be > 0")
    if config.n_realizations <= 0:
        raise ValueError("n_realizations must be > 0")
    if config.outlier_iqr_factor is not None and config.outlier_iqr_factor < 0.0:
        raise ValueError("outlier_iqr_factor must be >= 0 when provided")
    if config.g0 <= 0.0:
        raise ValueError("g0 must be > 0")
    if config.min_conductance < 0.0:
        raise ValueError("min_conductance must be >= 0")
    if config.max_neighbors is not None and config.max_neighbors <= 0:
        raise ValueError("max_neighbors must be > 0 when provided")
    if len(config.temperatures_k) == 0:
        raise ValueError("temperatures_k must contain at least one value")
    if any(temperature <= 0.0 for temperature in config.temperatures_k):
        raise ValueError("all temperatures must be > 0")


def make_regular_1d_network(
    config: OneDRRNConfig,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed if seed is None else seed)
    positions_nm = (np.arange(config.n_sites, dtype=float) * config.delta_min_nm)[
        :, None
    ]
    half_span = 0.5 * config.energy_span_ev
    energies_ev = rng.uniform(-half_span, half_span, size=config.n_sites)
    return positions_nm, energies_ev


def conductance_to_conductivity_s_per_cm(
    conductance_s: float,
    *,
    length_nm: float,
    area_nm2: float,
) -> float:
    length_cm = length_nm * NM_TO_CM
    area_cm2 = area_nm2 * NM_TO_CM * NM_TO_CM
    return conductance_s * length_cm / area_cm2


def stabilize_positive_values(values: np.ndarray) -> np.ndarray:
    stabilized = np.asarray(values, dtype=float).copy()
    positive_values = stabilized[np.isfinite(stabilized) & (stabilized > 0.0)]
    if positive_values.size == 0:
        stabilized[~np.isfinite(stabilized) | (stabilized <= 0.0)] = np.nextafter(
            0.0,
            1.0,
        )
        return stabilized

    floor_value = max(np.nextafter(0.0, 1.0), 0.5 * float(np.min(positive_values)))
    invalid_mask = ~np.isfinite(stabilized) | (stabilized <= 0.0)
    stabilized[invalid_mask] = floor_value
    return stabilized


def build_outlier_mask_iqr(
    samples: np.ndarray,
    *,
    iqr_factor: float | None,
    temperatures_k: np.ndarray,
) -> np.ndarray:
    outlier_mask = np.zeros(samples.shape, dtype=bool)
    if iqr_factor is None or samples.shape[0] < 4:
        return outlier_mask

    for column_index in range(samples.shape[1]):
        column = samples[:, column_index]
        q1, q3 = np.percentile(column, [25.0, 75.0])
        iqr = q3 - q1
        if iqr <= 0.0:
            continue

        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        keep = (column >= lower) & (column <= upper)
        if np.any(keep):
            outlier_indices = np.flatnonzero(~keep)
            if outlier_indices.size > 0:
                outlier_mask[outlier_indices, column_index] = True
                print(
                    "Outliers removed: "
                    f"T={temperatures_k[column_index]:g} K, "
                    f"count={outlier_indices.size}, "
                    f"realizations={outlier_indices.tolist()}"
                )

    return outlier_mask


def apply_outlier_mask(samples: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
    filtered = np.asarray(samples, dtype=float).copy()
    filtered[outlier_mask] = np.nan
    return filtered


def nan_log_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_values = np.where(values > 0.0, values, np.nan)
    log_values = np.log(positive_values)
    mean_log = np.nanmean(log_values, axis=0)
    std_log = np.nanstd(log_values, axis=0, ddof=0)
    typical_values = np.exp(mean_log)
    return typical_values, mean_log, std_log


def simulate_single_conductance_curve(
    config: OneDRRNConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures_k = np.asarray(config.temperatures_k, dtype=float)
    positions_nm, energies_ev = make_regular_1d_network(config, seed=seed)
    left_nodes = np.array([0], dtype=int)
    right_nodes = np.array([config.n_sites - 1], dtype=int)

    conductance_s = np.empty(temperatures_k.size, dtype=float)
    for index, temperature_k in enumerate(temperatures_k):
        solver = RRNSolver(
            positions=positions_nm,
            energies=energies_ev,
            temperature=float(temperature_k),
            xi=config.xi_nm,
            G0=config.g0,
            max_neighbors=config.max_neighbors,
            min_conductance=config.min_conductance,
        )
        result = solver.solve(left_nodes=left_nodes, right_nodes=right_nodes)
        conductance_s[index] = result.effective_conductance

    return stabilize_positive_values(conductance_s), positions_nm, energies_ev


def simulate_conductivity_sweep(config: OneDRRNConfig) -> ConductivitySweep:
    """
    Return ensemble-averaged Conductivity(T) for a 1D RRN.

    The pair conductance is Eq. (1) of arXiv:2601.01243v1:
        G_ij = (G0 / T) exp(-2 r_ij / xi) exp(-epsilon_ij / kB T)
    with epsilon_ij from Eq. (2). The effective two-terminal conductance is
    obtained by solving Kirchhoff equations with Dirichlet contacts at the
    first and last 1D sites.
    """

    validate_config(config)
    temperatures_k = np.asarray(config.temperatures_k, dtype=float)
    master_rng = np.random.default_rng(config.seed)
    realization_seeds = master_rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=config.n_realizations,
        dtype=np.int64,
    )

    conductance_samples = np.empty(
        (config.n_realizations, temperatures_k.size),
        dtype=float,
    )
    realization_positions = np.empty(
        (config.n_realizations, config.n_sites, 1),
        dtype=float,
    )
    realization_energies = np.empty(
        (config.n_realizations, config.n_sites),
        dtype=float,
    )
    positions_nm = np.empty((0, 1), dtype=float)
    energies_ev = np.empty(0, dtype=float)
    for realization_index, realization_seed in enumerate(realization_seeds):
        conductance_curve, positions_nm, energies_ev = simulate_single_conductance_curve(
            config,
            seed=int(realization_seed),
        )
        conductance_samples[realization_index] = conductance_curve
        realization_positions[realization_index] = positions_nm
        realization_energies[realization_index] = energies_ev

    outlier_mask = build_outlier_mask_iqr(
        conductance_samples,
        iqr_factor=config.outlier_iqr_factor,
        temperatures_k=temperatures_k,
    )
    filtered_conductance_samples = apply_outlier_mask(
        conductance_samples,
        outlier_mask,
    )
    conductance_s = np.nanmean(filtered_conductance_samples, axis=0)
    conductance_std_s = np.nanstd(filtered_conductance_samples, axis=0, ddof=0)
    typical_conductance_s, mean_ln_conductance, std_ln_conductance = nan_log_stats(
        filtered_conductance_samples
    )
    n_realizations_used = np.sum(
        np.isfinite(filtered_conductance_samples),
        axis=0,
        dtype=int,
    )
    conductivity_s_per_cm = np.array(
        [
            conductance_to_conductivity_s_per_cm(
                conductance,
                length_nm=config.length_nm,
                area_nm2=config.cross_section_area_nm2,
            )
            for conductance in conductance_s
        ],
        dtype=float,
    )
    conductivity_std_s_per_cm = np.array(
        [
            conductance_to_conductivity_s_per_cm(
                conductance_std,
                length_nm=config.length_nm,
                area_nm2=config.cross_section_area_nm2,
            )
            for conductance_std in conductance_std_s
        ],
        dtype=float,
    )
    typical_conductivity_s_per_cm = np.array(
        [
            conductance_to_conductivity_s_per_cm(
                conductance,
                length_nm=config.length_nm,
                area_nm2=config.cross_section_area_nm2,
            )
            for conductance in typical_conductance_s
        ],
        dtype=float,
    )
    mean_ln_conductivity = np.log(typical_conductivity_s_per_cm)
    std_ln_conductivity = std_ln_conductance.copy()

    return ConductivitySweep(
        temperatures_k=temperatures_k,
        conductance_s=conductance_s,
        conductance_std_s=conductance_std_s,
        typical_conductance_s=typical_conductance_s,
        mean_ln_conductance=mean_ln_conductance,
        std_ln_conductance=std_ln_conductance,
        conductivity_s_per_cm=conductivity_s_per_cm,
        conductivity_std_s_per_cm=conductivity_std_s_per_cm,
        typical_conductivity_s_per_cm=typical_conductivity_s_per_cm,
        mean_ln_conductivity=mean_ln_conductivity,
        std_ln_conductivity=std_ln_conductivity,
        n_realizations_used=n_realizations_used,
        conductance_samples_s=conductance_samples,
        filtered_conductance_samples_s=filtered_conductance_samples,
        outlier_mask=outlier_mask,
        realization_positions_nm=realization_positions,
        realization_energies_ev=realization_energies,
        positions_nm=positions_nm,
        energies_ev=energies_ev,
    )


def write_conductivity_csv(sweep: ConductivitySweep, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "temperature_K",
                "mean_conductance_S",
                "std_conductance_S",
                "typical_conductance_S",
                "mean_ln_conductance",
                "std_ln_conductance",
                "mean_conductivity_S_per_cm",
                "std_conductivity_S_per_cm",
                "typical_conductivity_S_per_cm",
                "mean_ln_conductivity",
                "std_ln_conductivity",
                "n_realizations_used",
                "inverse_temperature_1_per_K",
                "inverse_sqrt_temperature_1_per_sqrt_K",
            ]
        )
        for (
            temperature_k,
            conductance_s,
            conductance_std_s,
            typical_conductance_s,
            mean_ln_conductance,
            std_ln_conductance,
            conductivity_s_per_cm,
            conductivity_std_s_per_cm,
            typical_conductivity_s_per_cm,
            mean_ln_conductivity,
            std_ln_conductivity,
            n_realizations_used,
        ) in zip(
            sweep.temperatures_k,
            sweep.conductance_s,
            sweep.conductance_std_s,
            sweep.typical_conductance_s,
            sweep.mean_ln_conductance,
            sweep.std_ln_conductance,
            sweep.conductivity_s_per_cm,
            sweep.conductivity_std_s_per_cm,
            sweep.typical_conductivity_s_per_cm,
            sweep.mean_ln_conductivity,
            sweep.std_ln_conductivity,
            sweep.n_realizations_used,
            strict=True,
        ):
            writer.writerow(
                [
                    temperature_k,
                    conductance_s,
                    conductance_std_s,
                    typical_conductance_s,
                    mean_ln_conductance,
                    std_ln_conductance,
                    conductivity_s_per_cm,
                    conductivity_std_s_per_cm,
                    typical_conductivity_s_per_cm,
                    mean_ln_conductivity,
                    std_ln_conductivity,
                    n_realizations_used,
                    1.0 / temperature_k,
                    1.0 / np.sqrt(temperature_k),
                ]
            )


def load_json_config(config_path: Path) -> tuple[OneDRRNConfig, Path | None]:
    with config_path.open(encoding="utf-8") as handle:
        raw_config = json.load(handle)

    output = raw_config.pop("output", str(DEFAULT_OUTPUT_PATH))
    if output in (None, ""):
        output_path = None
    else:
        output_path = Path(output)

    if "temperatures_k" in raw_config:
        raw_config["temperatures_k"] = parse_temperatures(raw_config["temperatures_k"])

    config = OneDRRNConfig(**raw_config)
    return config, output_path


def build_default_config_for_temperatures(
    temperatures_k: str | float | int | list[float] | tuple[float, ...],
) -> OneDRRNConfig:
    config, _ = load_json_config(DEFAULT_CONFIG_PATH)
    return OneDRRNConfig(
        n_sites=config.n_sites,
        delta_min_nm=config.delta_min_nm,
        energy_span_ev=config.energy_span_ev,
        xi_nm=config.xi_nm,
        temperatures_k=parse_temperatures(temperatures_k),
        cross_section_area_nm2=config.cross_section_area_nm2,
        n_realizations=config.n_realizations,
        outlier_iqr_factor=config.outlier_iqr_factor,
        seed=config.seed,
        g0=config.g0,
        min_conductance=config.min_conductance,
        max_neighbors=config.max_neighbors,
    )


def simulate_default_conductivity_for_temperatures(
    temperatures_k: Any,
) -> dict[str, list[float]]:
    config = build_default_config_for_temperatures(temperatures_k)
    with contextlib.redirect_stdout(io.StringIO()):
        sweep = simulate_conductivity_sweep(config)

    return {
        "temperature_k": [float(value) for value in sweep.temperatures_k],
        "conductivity": [
            float(value) for value in sweep.typical_conductivity_s_per_cm
        ],
        "ln(conductivity)": [float(value) for value in sweep.mean_ln_conductivity],
    }


def write_default_config(config_path: Path) -> None:
    example_config = {
        "n_sites": 100,
        "delta_min_nm": 1.0,
        "energy_span_ev": 0.4,
        "xi_nm": 0.15,
        "temperatures_k": "100:400:10",
        "cross_section_area_nm2": 1.0,
        "n_realizations": 10,
        "outlier_iqr_factor": 1.5,
        "seed": 42,
        "g0": 1.0,
        "min_conductance": 0.0,
        "max_neighbors": None,
        "output": str(DEFAULT_OUTPUT_PATH),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(example_config, handle, indent=2)
        handle.write("\n")


def main() -> ConductivitySweep:
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python code/1d_simulator.py [config.json]")

    config_path = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        write_default_config(config_path)
        print(f"Wrote example config to {config_path}")

    config, output_path = load_json_config(config_path)
    sweep = simulate_conductivity_sweep(config)

    if output_path is not None:
        write_conductivity_csv(sweep, output_path)
        print(f"Wrote {output_path}")

    print(
        "Conductivity(T): "
        f"{sweep.temperatures_k.size} points, "
        f"sigma range {np.min(sweep.conductivity_s_per_cm):.6e}.."
        f"{np.max(sweep.conductivity_s_per_cm):.6e} S/cm"
    )
    return sweep


if __name__ == "__main__":
    main()
