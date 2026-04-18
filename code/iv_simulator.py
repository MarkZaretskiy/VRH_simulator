from __future__ import annotations

import ast
import csv
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import fire
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

try:
    from .sim import RRNSolver, contact_nodes_from_x, make_random_sites
except ImportError:
    from sim import RRNSolver, contact_nodes_from_x, make_random_sites


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "iv_simulator_results.csv"
DEFAULT_PLOT_OUTPUT_DIR = Path(__file__).with_name("plots")
DEFAULT_REFERENCE_VOLTAGE = 1.0
DEFAULT_MAX_GENERATED_SITES = 3000
NM_TO_CM = 1.0e-7


def parse_optional_array(
    value: str | list[float] | list[list[float]] | tuple[float, ...] | np.ndarray | None,
    name: str,
    ndim: int,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a Python-like literal, for example "
                f"'[0.1, -0.2, 0.3]' or '[[0.0], [1.0], [2.0]]'"
            ) from exc
        array = np.asarray(parsed, dtype=float)
    else:
        array = np.asarray(value, dtype=float)

    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    return array


def parse_index_array(
    value: str | list[int] | tuple[int, ...] | np.ndarray | None,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=int)
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a Python-like literal, for example '[0, 1, 2]'"
            ) from exc
        array = np.asarray(parsed, dtype=int)
    else:
        array = np.asarray(value, dtype=int)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def validate_positive_params(
    params: dict[str, float | int],
    *,
    suffix: str = "must be > 0",
) -> None:
    for name, value in params.items():
        if value <= 0:
            raise ValueError(f"{name} {suffix}")


@dataclass(slots=True)
class SimulationConfig:
    positions: str | list[list[float]] | np.ndarray | None = None
    energies: str | list[float] | np.ndarray | None = None
    concentration_cm3: float | None = 1.8e20
    temperatures_k: str | float | int | list[float] | tuple[float, ...] | np.ndarray | None = None
    xi: float = 0.35
    G0: float = 1.0
    kB: float = 8.617333262145e-5
    cutoff_distance: float | None = 20.0
    max_neighbors: int | None = 100
    min_conductance: float = 0.0
    left_nodes: str | list[int] | np.ndarray | None = None
    right_nodes: str | list[int] | np.ndarray | None = None
    contact_width: float = 1.5
    contact_depth_nm: float | None = None
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
            }
        )

        if (self.positions_array is None) != (self.energies_array is None):
            raise ValueError("positions and energies must be provided together")
        if self.positions_array is not None and self.energies_array is not None:
            if len(self.positions_array) != len(self.energies_array):
                raise ValueError("positions and energies must have the same length")
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

        if (self.left_nodes_array is None) != (self.right_nodes_array is None):
            raise ValueError("left_nodes and right_nodes must be provided together")
        if self.left_nodes_array is not None and self.right_nodes_array is not None:
            if np.intersect1d(self.left_nodes_array, self.right_nodes_array).size > 0:
                raise ValueError("left_nodes and right_nodes must not overlap")
        else:
            effective_contact_width = (
                float(self.contact_depth_nm)
                if self.contact_depth_nm is not None
                else float(self.contact_width)
            )
            validate_positive_params({"contact width/depth": effective_contact_width})

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


def build_sweep_values(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    n_steps = int(np.floor((stop - start) / step))
    values = start + step * np.arange(n_steps + 1, dtype=float)
    if values.size == 0:
        raise ValueError("sweep must contain at least one point")
    if not np.isclose(values[-1], stop):
        values = np.append(values, float(stop))
    return np.unique(np.round(values, 12))


def parse_float_values(
    value: str | float | int | list[float] | tuple[float, ...] | np.ndarray | None,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        if ":" in value and "," not in value:
            try:
                start_text, stop_text, step_text = (
                    token.strip() for token in value.split(":")
                )
                array = build_sweep_values(
                    float(start_text),
                    float(stop_text),
                    float(step_text),
                )
            except ValueError as exc:
                raise ValueError(
                    f"{name} range must look like '5:250:25'"
                ) from exc
        else:
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            if not tokens:
                raise ValueError(f"{name} must contain at least one value")
            array = np.asarray([float(token) for token in tokens], dtype=float)
    elif isinstance(value, (float, int)):
        array = np.asarray([float(value)], dtype=float)
    else:
        array = np.asarray(value, dtype=float)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence of floats")
    return np.unique(np.sort(array))


def build_temperature_values(
    temperatures_k: str | float | int | list[float] | tuple[float, ...] | np.ndarray | None,
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "temperature_K",
                "voltage_V",
                "current_mean_A",
                "current_std_A",
                "conductance_mean_S",
                "conductance_std_S",
                "n_realizations",
                "non_conductive_realizations",
            ]
        )
        for row_index, (temperature_k, conductance_mean_s_value, conductance_std_s_value, current_mean_row, current_std_row) in enumerate(
            zip(
                temperatures_k,
                conductance_mean_s,
                conductance_std_s,
                current_mean_a,
                current_std_a,
            )
        ):
            for voltage_v, current_mean_value, current_std_value in zip(
                voltages_v,
                current_mean_row,
                current_std_row,
            ):
                writer.writerow(
                    [
                        float(temperature_k),
                        float(voltage_v),
                        float(current_mean_value),
                        float(current_std_value),
                        float(conductance_mean_s_value),
                        float(conductance_std_s_value),
                        int(n_realizations),
                        int(non_conductive_counts[row_index]),
                    ]
                )


def resolve_plot_output_dir(
    plot_output_dir: str | None,
    output_path: Path | None,
) -> Path:
    if plot_output_dir is not None:
        return Path(plot_output_dir)
    if output_path is not None:
        return output_path.parent
    return DEFAULT_PLOT_OUTPUT_DIR


def format_box_plot_annotation(
    concentration_cm3: float | None,
    device_length_nm: float,
    device_width_nm: float,
    device_thickness_nm: float,
) -> str | None:
    if concentration_cm3 is None:
        return None
    return (
        f"L = {device_length_nm:.3g} nm\n"
        f"W = {device_width_nm:.3g} nm\n"
        f"T = {device_thickness_nm:.3g} nm\n"
        f"n = {concentration_cm3:.3e} cm^-3"
    )


def plot_iv_curves(
    temperatures_k: np.ndarray,
    voltages_v: np.ndarray,
    current_mean_a: np.ndarray,
    current_std_a: np.ndarray | None,
    output_dir: Path,
    n_realizations: int,
    plot_annotation: str | None = None,
    show_plots: bool = False,
) -> None:
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import colormaps, colors

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    has_temperature_sweep = temperatures_k.size > 1
    if has_temperature_sweep:
        temperature_norm = colors.Normalize(
            vmin=float(np.min(temperatures_k)),
            vmax=float(np.max(temperatures_k)),
        )
    else:
        temperature_norm = None
    temperature_cmap = colormaps["viridis"]

    for row_index, (temperature_k, current_row) in enumerate(
        zip(temperatures_k, current_mean_a)
    ):
        if has_temperature_sweep:
            curve_color = temperature_cmap(temperature_norm(float(temperature_k)))
            curve_label = None
        else:
            curve_color = "tab:blue"
            curve_label = f"T = {float(temperature_k):.1f} K"
        ax.plot(
            voltages_v,
            current_row,
            color=curve_color,
            linewidth=1.6,
            label=curve_label,
        )
        if current_std_a is not None and n_realizations > 1:
            current_std_row = current_std_a[row_index]
            ax.fill_between(
                voltages_v,
                current_row - current_std_row,
                current_row + current_std_row,
                color=curve_color,
                alpha=0.12,
                linewidth=0.0,
            )

    if has_temperature_sweep:
        scalar_mappable = plt.cm.ScalarMappable(
            norm=temperature_norm,
            cmap=temperature_cmap,
        )
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax)
        colorbar.set_label("Temperature (K)")
    else:
        ax.legend()

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    if n_realizations > 1:
        ax.set_title(f"Linear I-V curves vs temperature (mean ± std, n={n_realizations})")
    else:
        ax.set_title("Linear I-V curves vs temperature")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    if plot_annotation is not None:
        ax.text(
            0.98,
            0.98,
            plot_annotation,
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.9,
            },
        )
    fig.tight_layout()

    plot_path = output_dir / "iv_curves_vs_temperature.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_conductance_vs_temperature(
    temperatures_k: np.ndarray,
    conductance_mean_s: np.ndarray,
    conductance_std_s: np.ndarray | None,
    output_dir: Path,
    n_realizations: int,
    plot_annotation: str | None = None,
    show_plots: bool = False,
) -> None:
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    positive_mask = conductance_mean_s > 0.0
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    gt_ax = axes[0, 0]

    if np.any(positive_mask):
        plot_temperatures_k = temperatures_k[positive_mask]
        plot_conductance_mean_s = conductance_mean_s[positive_mask]
        gt_ax.plot(
            plot_temperatures_k,
            plot_conductance_mean_s,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color="tab:blue",
        )
        if conductance_std_s is not None and n_realizations > 1:
            plot_conductance_std_s = conductance_std_s[positive_mask]
            lower_bound_s = plot_conductance_mean_s - plot_conductance_std_s
            upper_bound_s = plot_conductance_mean_s + plot_conductance_std_s
            band_mask = lower_bound_s > 0.0
            if np.any(band_mask):
                gt_ax.fill_between(
                    plot_temperatures_k[band_mask],
                    lower_bound_s[band_mask],
                    upper_bound_s[band_mask],
                    color="tab:blue",
                    alpha=0.14,
                    linewidth=0.0,
                )
        gt_ax.set_yscale("log")
    else:
        plot_temperatures_k = temperatures_k
        plot_conductance_mean_s = conductance_mean_s
        gt_ax.plot(
            temperatures_k,
            conductance_mean_s,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color="tab:blue",
        )

    gt_ax.set_xlabel("Temperature (K)")
    gt_ax.set_ylabel("Conductance (S)")
    if n_realizations > 1:
        gt_ax.set_title(f"Conductance vs temperature (mean ± std, n={n_realizations})")
    else:
        gt_ax.set_title("Conductance vs temperature")
    gt_ax.grid(True, alpha=0.3, which="both")

    transformed_axes = [
        (
            axes[0, 1],
            "Activated",
            -1.0,
            r"$1/T$ (K$^{-1}$)",
        ),
        (
            axes[0, 2],
            "1D Mott-like",
            -0.5,
            r"$T^{-1/2}$ (K$^{-1/2}$)",
        ),
        (
            axes[1, 0],
            "2D Mott-like",
            -(1.0 / 3.0),
            r"$T^{-1/3}$ (K$^{-1/3}$)",
        ),
        (
            axes[1, 1],
            "3D Mott-like",
            -0.25,
            r"$T^{-1/4}$ (K$^{-1/4}$)",
        ),
    ]

    ln_conductance = (
        np.log(conductance_mean_s[positive_mask])
        if np.any(positive_mask)
        else np.array([], dtype=float)
    )
    positive_temperatures_k = (
        temperatures_k[positive_mask]
        if np.any(positive_mask)
        else np.array([], dtype=float)
    )
    for axis, regime_label, exponent, x_label in transformed_axes:
        if positive_temperatures_k.size > 0:
            transformed_x = np.power(positive_temperatures_k, exponent)
            axis.scatter(
                transformed_x,
                ln_conductance,
                s=28,
                color="tab:blue",
            )
            axis.set_title(f"{regime_label}: ln(G)")
        else:
            axis.text(
                0.5,
                0.5,
                "No positive conductance values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(f"{regime_label}: ln(G)")
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\ln G$")
        axis.grid(True, alpha=0.3)

    note_ax = axes[1, 2]
    note_ax.axis("off")
    note_text = "\n".join(
        [
            "Regime guide:",
            r"1/T: activated transport",
            r"$T^{-1/2}$: 1D Mott-like",
            r"$T^{-1/3}$: 2D Mott-like",
            r"$T^{-1/4}$: 3D Mott-like",
            "",
            "The panel with the most linear",
            "ln(G) trend is the best match.",
        ]
    )
    if plot_annotation is not None:
        note_text = f"{plot_annotation}\n\n{note_text}"
    note_ax.text(
        0.02,
        0.98,
        note_text,
        ha="left",
        va="top",
        transform=note_ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()

    plot_path = output_dir / "conductance_vs_temperature.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def build_network(
    positions: str | list[list[float]] | np.ndarray | None,
    energies: str | list[float] | np.ndarray | None,
    concentration_cm3: float | None,
    device_length_nm: float,
    device_width_nm: float,
    device_thickness_nm: float,
    max_generated_sites: int,
    n_sites: int,
    length: float,
    dim: int,
    energy_std: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    parsed_positions = parse_optional_array(positions, name="positions", ndim=2)
    parsed_energies = parse_optional_array(energies, name="energies", ndim=1)

    if (parsed_positions is None) != (parsed_energies is None):
        raise ValueError("positions and energies must be provided together")

    if parsed_positions is not None and parsed_energies is not None:
        if len(parsed_positions) != len(parsed_energies):
            raise ValueError("positions and energies must have the same length")
        return parsed_positions, parsed_energies, {
            "mode": "explicit",
            "n_sites": int(len(parsed_energies)),
            "dim": int(parsed_positions.shape[1]),
        }

    if concentration_cm3 is not None:
        validate_positive_params(
            {
                "concentration_cm3": concentration_cm3,
                "device_length_nm": device_length_nm,
                "device_width_nm": device_width_nm,
                "device_thickness_nm": device_thickness_nm,
            }
        )

        volume_nm3 = device_length_nm * device_width_nm * device_thickness_nm
        volume_cm3 = volume_nm3 * (NM_TO_CM**3)
        expected_n_sites = concentration_cm3 * volume_cm3
        generated_n_sites = max(2, int(round(expected_n_sites)))
        validate_positive_params({"max_generated_sites": max_generated_sites})
        if generated_n_sites > max_generated_sites:
            raise ValueError(
                "concentration_cm3 and box geometry generate "
                f"{generated_n_sites} sites, which is too large for this explicit O(N^2) solver; "
                "reduce geometry/concentration or raise max_generated_sites intentionally"
            )
        rng = np.random.default_rng(seed)
        positions_array = np.column_stack(
            [
                rng.uniform(0.0, device_length_nm, size=generated_n_sites),
                rng.uniform(0.0, device_width_nm, size=generated_n_sites),
                rng.uniform(0.0, device_thickness_nm, size=generated_n_sites),
            ]
        )
        energies_array = rng.normal(0.0, energy_std, size=generated_n_sites)
        actual_concentration_cm3 = generated_n_sites / volume_cm3
        return positions_array, energies_array, {
            "mode": "box",
            "n_sites": int(generated_n_sites),
            "dim": 3,
            "device_length_nm": float(device_length_nm),
            "device_width_nm": float(device_width_nm),
            "device_thickness_nm": float(device_thickness_nm),
            "expected_n_sites": float(expected_n_sites),
            "concentration_cm3": float(concentration_cm3),
            "actual_concentration_cm3": float(actual_concentration_cm3),
        }

    positions_array, energies_array = make_random_sites(
        n_sites=n_sites,
        length=length,
        dim=dim,
        energy_std=energy_std,
        seed=seed,
    )
    return positions_array, energies_array, {
        "mode": "random",
        "n_sites": int(n_sites),
        "dim": int(dim),
        "length": float(length),
    }


def summarize_samples(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_realizations, n_temperatures)")
    return samples.mean(axis=0), samples.std(axis=0, ddof=0)


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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def run_realization(
    realization_index: int,
    positions: str | list[list[float]] | np.ndarray | None,
    energies: str | list[float] | np.ndarray | None,
    concentration_cm3: float | None,
    device_length_nm: float,
    device_width_nm: float,
    device_thickness_nm: float,
    max_generated_sites: int,
    n_sites: int,
    length: float,
    dim: int,
    energy_std: float,
    seed: int | None,
    left_nodes: str | list[int] | np.ndarray | None,
    right_nodes: str | list[int] | np.ndarray | None,
    contact_width: float,
    contact_depth_nm: float | None,
    temperatures_k: np.ndarray,
    xi: float,
    G0: float,
    kB: float,
    cutoff_distance: float | None,
    max_neighbors: int | None,
    min_conductance: float,
) -> dict[str, object]:
    current_seed = realization_seed(seed, realization_index)
    positions_array, energies_array, network_metadata = build_network(
        positions=positions,
        energies=energies,
        concentration_cm3=concentration_cm3,
        device_length_nm=device_length_nm,
        device_width_nm=device_width_nm,
        device_thickness_nm=device_thickness_nm,
        max_generated_sites=max_generated_sites,
        n_sites=n_sites,
        length=length,
        dim=dim,
        energy_std=energy_std,
        seed=current_seed,
    )
    left_nodes_array, right_nodes_array = build_contacts(
        positions=positions_array,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        contact_width=contact_width,
        contact_depth_nm=contact_depth_nm,
    )

    conductance_curve = np.empty(temperatures_k.size, dtype=float)
    non_conductive_curve = np.zeros(temperatures_k.size, dtype=bool)
    for temperature_index, temperature_k in enumerate(temperatures_k):
        try:
            conductance_s = compute_conductance_for_temperature(
                positions=positions_array,
                energies=energies_array,
                temperature_k=float(temperature_k),
                xi=xi,
                left_nodes=left_nodes_array,
                right_nodes=right_nodes_array,
                g0=G0,
                kB=kB,
                cutoff_distance=cutoff_distance,
                max_neighbors=max_neighbors,
                min_conductance=min_conductance,
                reference_voltage=DEFAULT_REFERENCE_VOLTAGE,
            )
        except (MatrixRankWarning, FloatingPointError, ValueError) as exc:
            if isinstance(exc, ValueError) and "no conductive path connects" not in str(exc):
                raise
            conductance_s = 0.0
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
    }


def build_contacts(
    positions: np.ndarray,
    left_nodes: str | list[int] | np.ndarray | None,
    right_nodes: str | list[int] | np.ndarray | None,
    contact_width: float,
    contact_depth_nm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    parsed_left_nodes = parse_index_array(left_nodes, name="left_nodes")
    parsed_right_nodes = parse_index_array(right_nodes, name="right_nodes")

    if parsed_left_nodes is not None and parsed_right_nodes is not None:
        return parsed_left_nodes, parsed_right_nodes
    if parsed_left_nodes is not None or parsed_right_nodes is not None:
        raise ValueError("left_nodes and right_nodes must be provided together")

    effective_contact_width = (
        float(contact_depth_nm) if contact_depth_nm is not None else float(contact_width)
    )
    validate_positive_params({"contact width/depth": effective_contact_width})
    return contact_nodes_from_x(positions, contact_width=effective_contact_width)


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


def main(
    positions: str | list[list[float]] | np.ndarray | None = None,
    energies: str | list[float] | np.ndarray | None = None,
    concentration_cm3: float | None = 1.8e20,
    temperatures_k: str | float | int | list[float] | tuple[float, ...] | np.ndarray | None = None,
    xi: float = 0.35,
    G0: float = 1.0,
    kB: float = 8.617333262145e-5,
    cutoff_distance: float | None = 20.0,
    max_neighbors: int | None = 100,
    min_conductance: float = 0.0,
    left_nodes: str | list[int] | np.ndarray | None = None,
    right_nodes: str | list[int] | np.ndarray | None = None,
    contact_width: float = 1.5,
    contact_depth_nm: float | None = None,
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
    seed: int | None = 42,
) -> None:
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
        contact_width=contact_width,
        contact_depth_nm=contact_depth_nm,
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
        seed=seed,
    )
    positions = config.positions_array
    energies = config.energies_array
    left_nodes = config.left_nodes_array
    right_nodes = config.right_nodes_array
    temperatures_k = config.temperature_values_k
    voltages_v = config.voltage_values_v
    resolved_n_jobs = config.resolved_n_jobs

    conductance_samples = np.empty((n_realizations, temperatures_k.size), dtype=float)
    non_conductive_mask = np.zeros_like(conductance_samples, dtype=bool)

    print(
        f"Running linear I-V sweep for {len(temperatures_k)} temperatures "
        f"and {len(voltages_v)} voltages"
    )
    print(f"Ensemble realizations: {n_realizations}, base_seed={seed}")
    print(f"Parallel workers: {resolved_n_jobs}")

    metadata_printed = False
    if resolved_n_jobs == 1:
        for realization_index in range(n_realizations):
            result = run_realization(
                realization_index=realization_index,
                positions=positions,
                energies=energies,
                concentration_cm3=concentration_cm3,
                device_length_nm=device_length_nm,
                device_width_nm=device_width_nm,
                device_thickness_nm=device_thickness_nm,
                max_generated_sites=max_generated_sites,
                n_sites=n_sites,
                length=length,
                dim=dim,
                energy_std=energy_std,
                seed=seed,
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                contact_width=contact_width,
                contact_depth_nm=contact_depth_nm,
                temperatures_k=temperatures_k,
                xi=xi,
                G0=G0,
                kB=kB,
                cutoff_distance=cutoff_distance,
                max_neighbors=max_neighbors,
                min_conductance=min_conductance,
            )
            conductance_samples[result["realization_index"]] = result["conductance_curve"]
            non_conductive_mask[result["realization_index"]] = result["non_conductive_curve"]
            if not metadata_printed:
                print(
                    f"Network: n_sites={result['n_sites']}, dim={result['dim']}, "
                    f"left_contacts={result['left_contacts']}, right_contacts={result['right_contacts']}"
                )
                network_metadata = result["network_metadata"]
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
                elif network_metadata["mode"] == "random":
                    print(
                        f"Random cloud: length={network_metadata['length']:.3g}, "
                        f"dim={int(network_metadata['dim'])}"
                    )
                metadata_printed = True
            print(
                f"Completed realization {result['realization_index'] + 1}/{n_realizations} "
                f"(seed={result['seed']}, "
                f"non_conductive_t={int(np.count_nonzero(result['non_conductive_curve']))}/{len(temperatures_k)})"
            )
    else:
        configure_parallel_process_env()
        with ProcessPoolExecutor(
            max_workers=resolved_n_jobs,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    run_realization,
                    realization_index,
                    positions,
                    energies,
                    concentration_cm3,
                    device_length_nm,
                    device_width_nm,
                    device_thickness_nm,
                    max_generated_sites,
                    n_sites,
                    length,
                    dim,
                    energy_std,
                    seed,
                    left_nodes,
                    right_nodes,
                    contact_width,
                    contact_depth_nm,
                    temperatures_k,
                    xi,
                    G0,
                    kB,
                    cutoff_distance,
                    max_neighbors,
                    min_conductance,
                ): realization_index
                for realization_index in range(n_realizations)
            }
            for future in as_completed(futures):
                result = future.result()
                conductance_samples[result["realization_index"]] = result["conductance_curve"]
                non_conductive_mask[result["realization_index"]] = result["non_conductive_curve"]
                if not metadata_printed:
                    print(
                        f"Network: n_sites={result['n_sites']}, dim={result['dim']}, "
                        f"left_contacts={result['left_contacts']}, right_contacts={result['right_contacts']}"
                    )
                    network_metadata = result["network_metadata"]
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
                    elif network_metadata["mode"] == "random":
                        print(
                            f"Random cloud: length={network_metadata['length']:.3g}, "
                            f"dim={int(network_metadata['dim'])}"
                        )
                    metadata_printed = True
                print(
                    f"Completed realization {result['realization_index'] + 1}/{n_realizations} "
                    f"(seed={result['seed']}, "
                    f"non_conductive_t={int(np.count_nonzero(result['non_conductive_curve']))}/{len(temperatures_k)})"
                )

    conductance_mean_s, conductance_std_s = summarize_samples(conductance_samples)
    current_mean_a = conductance_mean_s[:, None] * voltages_v[None, :]
    current_std_a = conductance_std_s[:, None] * np.abs(voltages_v[None, :])
    non_conductive_counts = non_conductive_mask.sum(axis=0)

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

    if not no_output:
        output_path = Path(output) if output is not None else DEFAULT_OUTPUT_PATH
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

    if plot:
        output_path = None
        if not no_output:
            output_path = Path(output) if output is not None else DEFAULT_OUTPUT_PATH
        plot_dir = resolve_plot_output_dir(
            plot_output_dir=plot_output_dir,
            output_path=output_path,
        )
        plot_annotation = format_box_plot_annotation(
            concentration_cm3=concentration_cm3,
            device_length_nm=device_length_nm,
            device_width_nm=device_width_nm,
            device_thickness_nm=device_thickness_nm,
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
            show_plots=show_plots,
        )
        plot_conductance_vs_temperature(
            temperatures_k=temperatures_k,
            conductance_mean_s=conductance_mean_s,
            conductance_std_s=conductance_std_s,
            output_dir=plot_dir,
            n_realizations=n_realizations,
            plot_annotation=plot_annotation,
            show_plots=show_plots,
        )


if __name__ == "__main__":
    fire.Fire(main)
