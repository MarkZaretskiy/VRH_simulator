from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np

try:
    from .sim import RRNSolver
except ImportError:
    from sim import RRNSolver


DEFAULT_XI_VALUES_NM = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("1d_experiment_results.csv")
DEFAULT_PLOT_OUTPUT_DIR = Path(__file__).with_name("plots")
NM_TO_CM = 1.0e-7
FIT_RESPONSE_REGULARIZATION = 1.0e-12
AVERAGED_CURVE_FLOOR_FACTOR = 0.5
VRH_COLORMAP_MAX_MILLI = 15.0
PLOT_HIGHLIGHT_XI_COLORS = (
    (0.1, "blue"),
    (0.15, "red"),
    (0.3, "green"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    n_sites: int
    delta_min_nm: float
    energy_span_ev: float
    cross_section_area_nm2: float
    xi_values_nm: tuple[float, ...]
    temperatures_k: np.ndarray
    n_realizations: int
    seed: int
    g0: float
    min_conductance: float
    output_path: Path | None

    @property
    def length_nm(self) -> float:
        if self.n_sites < 2:
            return self.delta_min_nm
        return self.delta_min_nm * (self.n_sites - 1)


@dataclass(frozen=True)
class AveragedCurve:
    xi_nm: float
    temperatures_k: np.ndarray
    mean_conductance: np.ndarray
    std_conductance: np.ndarray
    mean_conductivity: np.ndarray
    std_conductivity: np.ndarray


@dataclass(frozen=True)
class VRHFitResult:
    xi_nm: float
    tx_temperatures_k: np.ndarray
    epsilon_vrh: np.ndarray
    slopes_ln_sigma_vs_inv_sqrt_t: np.ndarray
    intercepts_ln_sigma: np.ndarray
    last_threshold_compliant_temperature_k: float | None
    vrh_fit_max_temperature_k: float
    vrh_fit_epsilon: float
    vrh_fit_is_threshold_compliant: bool
    selected_fit_slope_ln_sigma_vs_inv_sqrt_t: float
    selected_fit_intercept_ln_sigma: float
    transition_temperature_k: float | None
    transition_epsilon: float | None
    epsilon_at_t_max: float


def parse_xi_values(
    xi_values_nm: str | float | int | list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    if isinstance(xi_values_nm, str):
        if ":" in xi_values_nm and "," not in xi_values_nm:
            start_text, stop_text, step_text = (
                token.strip() for token in xi_values_nm.split(":")
            )
            start = float(start_text)
            stop = float(stop_text)
            step = float(step_text)
            if step <= 0:
                raise ValueError("xi range step must be > 0")

            values = np.arange(start, stop + 0.5 * step, step, dtype=float)
            if values.size == 0:
                raise ValueError("xi range must contain at least one value")
            return tuple(np.round(values, 10))

        xi_tokens = [token.strip() for token in xi_values_nm.split(",") if token.strip()]
        if not xi_tokens:
            raise ValueError("at least one xi value is required")
        return tuple(float(token) for token in xi_tokens)

    if isinstance(xi_values_nm, (int, float)):
        return (float(xi_values_nm),)

    xi_list = tuple(float(value) for value in xi_values_nm)
    if not xi_list:
        raise ValueError("at least one xi value is required")
    return xi_list


def parse_energy_span_values(
    energy_span_ev: str | float | int | list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    if isinstance(energy_span_ev, str):
        if ":" in energy_span_ev and "," not in energy_span_ev:
            start_text, stop_text, step_text = (
                token.strip() for token in energy_span_ev.split(":")
            )
            start = float(start_text)
            stop = float(stop_text)
            step = float(step_text)
            if step <= 0:
                raise ValueError("energy span range step must be > 0")

            values = np.arange(start, stop + 0.5 * step, step, dtype=float)
            if values.size == 0:
                raise ValueError("energy span range must contain at least one value")
            return tuple(np.round(values, 10))

        tokens = [token.strip() for token in energy_span_ev.split(",") if token.strip()]
        if not tokens:
            raise ValueError("at least one energy span value is required")
        return tuple(float(token) for token in tokens)

    if isinstance(energy_span_ev, (int, float)):
        return (float(energy_span_ev),)

    energy_span_list = tuple(float(value) for value in energy_span_ev)
    if not energy_span_list:
        raise ValueError("at least one energy span value is required")
    return energy_span_list


def format_energy_span_suffix(energy_span_ev: float) -> str:
    return f"{energy_span_ev:g}".replace(".", "p")


def make_energy_span_output_path(
    base_path: Path,
    energy_span_ev: float,
    multiple_energy_spans: bool,
) -> Path:
    if not multiple_energy_spans:
        return base_path

    suffix = format_energy_span_suffix(energy_span_ev)
    return base_path.with_name(f"{base_path.stem}_we_{suffix}eV{base_path.suffix}")


def make_energy_span_plot_output_dir(
    base_output_dir: Path,
    energy_span_ev: float,
    multiple_energy_spans: bool,
) -> Path:
    if not multiple_energy_spans:
        return base_output_dir

    suffix = format_energy_span_suffix(energy_span_ev)
    return base_output_dir / f"we_{suffix}eV"


def build_config(
    n_sites: int = 100,
    delta_min_nm: float = 1.0,
    energy_span_ev: float = 0.4,
    cross_section_area_nm2: float = 1.0,
    xi_values_nm: str | float | int | list[float] | tuple[float, ...] = DEFAULT_XI_VALUES_NM,
    t_min_k: float = 100.0,
    t_max_k: float = 400.0,
    t_step_k: float = 2.0,
    n_realizations: int = 100,
    seed: int = 42,
    g0: float = 1.0,
    min_conductance: float = 0.0,
    output: str | None = str(DEFAULT_OUTPUT_PATH),
    no_output: bool = False,
) -> ExperimentConfig:
    xi_values_nm = parse_xi_values(xi_values_nm)

    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")
    if delta_min_nm <= 0:
        raise ValueError("delta_min_nm must be > 0")
    if energy_span_ev <= 0:
        raise ValueError("energy_span_ev must be > 0")
    if cross_section_area_nm2 <= 0:
        raise ValueError("cross_section_area_nm2 must be > 0")
    if t_step_k <= 0:
        raise ValueError("t_step_k must be > 0")
    if t_max_k < t_min_k:
        raise ValueError("t_max_k must be >= t_min_k")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be > 0")
    if any(xi <= 0 for xi in xi_values_nm):
        raise ValueError("all xi values must be > 0")
    if min_conductance < 0:
        raise ValueError("min_conductance must be >= 0")

    temperatures_k = np.arange(
        t_min_k,
        t_max_k + 0.5 * t_step_k,
        t_step_k,
        dtype=float,
    )

    output_path = Path(output) if output is not None else DEFAULT_OUTPUT_PATH
    if no_output:
        output_path = None

    return ExperimentConfig(
        n_sites=n_sites,
        delta_min_nm=delta_min_nm,
        energy_span_ev=energy_span_ev,
        cross_section_area_nm2=cross_section_area_nm2,
        xi_values_nm=tuple(xi_values_nm),
        temperatures_k=temperatures_k,
        n_realizations=n_realizations,
        seed=seed,
        g0=g0,
        min_conductance=min_conductance,
        output_path=output_path,
    )


def make_uniform_1d_chain(
    n_sites: int,
    delta_min_nm: float,
    energy_span_ev: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positions = (np.arange(n_sites, dtype=float) * delta_min_nm)[:, None]
    half_span = 0.5 * energy_span_ev
    energies = rng.uniform(-half_span, half_span, size=n_sites)
    return positions, energies


def conductance_to_conductivity(
    effective_conductance: float,
    length_nm: float,
    area_nm2: float,
) -> float:
    length_cm = length_nm * NM_TO_CM
    area_cm2 = area_nm2 * (NM_TO_CM**2)
    return effective_conductance * length_cm / area_cm2


def run_single_realization(
    config: ExperimentConfig,
    xi_nm: float,
    realization_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(realization_seed)
    positions, energies = make_uniform_1d_chain(
        n_sites=config.n_sites,
        delta_min_nm=config.delta_min_nm,
        energy_span_ev=config.energy_span_ev,
        rng=rng,
    )

    left_nodes = np.array([0], dtype=int)
    right_nodes = np.array([config.n_sites - 1], dtype=int)

    conductance_curve = np.empty_like(config.temperatures_k)
    conductivity_curve = np.empty_like(config.temperatures_k)

    for idx, temperature_k in enumerate(config.temperatures_k):
        solver = RRNSolver(
            positions=positions,
            energies=energies,
            temperature=temperature_k,
            xi=xi_nm,
            G0=config.g0,
            min_conductance=config.min_conductance,
        )
        result = solver.solve(
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            V_left=1.0,
            V_right=0.0,
        )
        conductance_curve[idx] = result.effective_conductance
        conductivity_curve[idx] = conductance_to_conductivity(
            result.effective_conductance,
            length_nm=config.length_nm,
            area_nm2=config.cross_section_area_nm2,
        )

    return conductance_curve, conductivity_curve


def simulate_averaged_curves(config: ExperimentConfig) -> list[AveragedCurve]:
    master_rng = np.random.default_rng(config.seed)
    realization_seeds = master_rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=config.n_realizations,
        dtype=np.int64,
    )

    curves: list[AveragedCurve] = []
    for xi_nm in config.xi_values_nm:
        print(
            f"Simulating xi={xi_nm:.3f} nm "
            f"with {config.n_realizations} realizations..."
        )

        conductance_samples = np.empty(
            (config.n_realizations, config.temperatures_k.size),
            dtype=float,
        )
        conductivity_samples = np.empty_like(conductance_samples)

        for idx, realization_seed in enumerate(realization_seeds):
            conductance_curve, conductivity_curve = run_single_realization(
                config=config,
                xi_nm=xi_nm,
                realization_seed=int(realization_seed),
            )
            conductance_samples[idx] = conductance_curve
            conductivity_samples[idx] = conductivity_curve

        mean_conductance = conductance_samples.mean(axis=0)
        mean_conductivity = conductivity_samples.mean(axis=0)
        mean_conductance, conductance_replacements, conductance_floor = (
            stabilize_positive_curve(mean_conductance)
        )
        mean_conductivity, conductivity_replacements, conductivity_floor = (
            stabilize_positive_curve(mean_conductivity)
        )

        total_replacements = conductance_replacements + conductivity_replacements
        if total_replacements > 0:
            warnings.warn(
                "Replaced non-positive averaged transport values for "
                f"xi={xi_nm:.3f} nm with floors of "
                f"{conductance_floor:.3e} (conductance) and "
                f"{conductivity_floor:.3e} S/cm (conductivity). "
                "This occurs when the linear solve returns values extremely "
                "close to zero with small numerical sign errors.",
                RuntimeWarning,
                stacklevel=2,
            )

        curves.append(
            AveragedCurve(
                xi_nm=xi_nm,
                temperatures_k=config.temperatures_k.copy(),
                mean_conductance=mean_conductance,
                std_conductance=conductance_samples.std(axis=0, ddof=0),
                mean_conductivity=mean_conductivity,
                std_conductivity=conductivity_samples.std(axis=0, ddof=0),
            )
        )

    return curves


def stabilize_positive_curve(values: np.ndarray) -> tuple[np.ndarray, int, float]:
    stabilized_values = np.asarray(values, dtype=float).copy()
    positive_values = stabilized_values[
        np.isfinite(stabilized_values) & (stabilized_values > 0.0)
    ]
    if positive_values.size == 0:
        raise ValueError("curve must contain at least one positive finite value")

    floor_value = max(
        np.nextafter(0.0, 1.0),
        AVERAGED_CURVE_FLOOR_FACTOR * positive_values.min(),
    )
    invalid_mask = ~np.isfinite(stabilized_values) | (stabilized_values <= 0.0)
    stabilized_values[invalid_mask] = floor_value
    return stabilized_values, int(np.count_nonzero(invalid_mask)), float(floor_value)


def compute_vrh_fit_error(
    temperatures_k: np.ndarray,
    observed_log_sigma: np.ndarray,
    fitted_log_sigma: np.ndarray,
) -> float:
    """
    Evaluate Eq. (10) in the same space used for the VRH linear fit:
    ln(sigma) as a function of T^(-1/2).

    The paper denotes the fitted curve as g(T) and the data as y(T). Since the
    VRH fit is performed on ln(sigma), both quantities are evaluated in that
    space here. The denominator uses |g(T)| to keep the relative error stable
    when the fitted response is negative.
    """
    if temperatures_k.size < 2:
        return 0.0

    fitted_magnitude = np.maximum(
        np.abs(fitted_log_sigma),
        FIT_RESPONSE_REGULARIZATION,
    )
    relative_squared_error = (
        (fitted_log_sigma - observed_log_sigma) / fitted_magnitude
    ) ** 2

    return float(
        np.sqrt(
            np.trapezoid(relative_squared_error, temperatures_k)
            / (temperatures_k[-1] - temperatures_k[0])
        )
    )


def fit_vrh_transition(
    curve: AveragedCurve,
    error_threshold: float = 5.0e-3,
    min_fit_points: int = 3,
) -> VRHFitResult:
    if error_threshold <= 0:
        raise ValueError("error_threshold must be > 0")
    if min_fit_points < 2:
        raise ValueError("min_fit_points must be >= 2")
    if curve.temperatures_k.size < min_fit_points:
        raise ValueError("not enough temperature points for the requested fit window")
    if np.any(curve.mean_conductivity <= 0.0):
        raise ValueError(
            f"VRH fit requires positive conductivity values for xi={curve.xi_nm:.3f} nm"
        )

    temperatures_k = curve.temperatures_k
    sigma = curve.mean_conductivity
    log_sigma = np.log(sigma)
    inv_sqrt_temperature = 1.0 / np.sqrt(temperatures_k)

    tx_indices = np.arange(min_fit_points - 1, temperatures_k.size, dtype=int)
    epsilon_vrh = np.empty(tx_indices.size, dtype=float)
    slopes = np.empty_like(epsilon_vrh)
    intercepts = np.empty_like(epsilon_vrh)

    for scan_idx, end_idx in enumerate(tx_indices):
        fit_slice = slice(0, end_idx + 1)
        slope, intercept = np.polyfit(
            inv_sqrt_temperature[fit_slice],
            log_sigma[fit_slice],
            deg=1,
        )
        fitted_log_sigma = slope * inv_sqrt_temperature[fit_slice] + intercept
        tx_temperatures = temperatures_k[fit_slice]

        epsilon_vrh[scan_idx] = compute_vrh_fit_error(
            temperatures_k=tx_temperatures,
            observed_log_sigma=log_sigma[fit_slice],
            fitted_log_sigma=fitted_log_sigma,
        )
        slopes[scan_idx] = slope
        intercepts[scan_idx] = intercept

    within_threshold = epsilon_vrh <= error_threshold
    compliant_prefix_length = int(np.argmax(~within_threshold)) if np.any(~within_threshold) else len(within_threshold)

    if compliant_prefix_length == 0:
        transition_temperature_k = None
        transition_epsilon = None
        last_threshold_compliant_temperature_k = None
        selected_scan_idx = 0
    elif compliant_prefix_length < len(within_threshold):
        transition_scan_idx = compliant_prefix_length
        transition_temperature_k = float(temperatures_k[tx_indices[transition_scan_idx]])
        transition_epsilon = float(epsilon_vrh[transition_scan_idx])
        last_threshold_compliant_temperature_k = float(
            temperatures_k[tx_indices[transition_scan_idx - 1]]
        )
        selected_scan_idx = transition_scan_idx - 1
    else:
        transition_temperature_k = None
        transition_epsilon = None
        last_threshold_compliant_temperature_k = float(temperatures_k[tx_indices[-1]])
        selected_scan_idx = len(tx_indices) - 1

    return VRHFitResult(
        xi_nm=curve.xi_nm,
        tx_temperatures_k=temperatures_k[tx_indices].copy(),
        epsilon_vrh=epsilon_vrh,
        slopes_ln_sigma_vs_inv_sqrt_t=slopes,
        intercepts_ln_sigma=intercepts,
        last_threshold_compliant_temperature_k=last_threshold_compliant_temperature_k,
        vrh_fit_max_temperature_k=float(temperatures_k[tx_indices[selected_scan_idx]]),
        vrh_fit_epsilon=float(epsilon_vrh[selected_scan_idx]),
        vrh_fit_is_threshold_compliant=bool(
            epsilon_vrh[selected_scan_idx] <= error_threshold
        ),
        selected_fit_slope_ln_sigma_vs_inv_sqrt_t=float(slopes[selected_scan_idx]),
        selected_fit_intercept_ln_sigma=float(intercepts[selected_scan_idx]),
        transition_temperature_k=transition_temperature_k,
        transition_epsilon=transition_epsilon,
        epsilon_at_t_max=float(epsilon_vrh[-1]),
    )


def analyze_vrh_transitions(
    curves: list[AveragedCurve],
    error_threshold: float = 5.0e-3,
    min_fit_points: int = 3,
) -> list[VRHFitResult]:
    return [
        fit_vrh_transition(
            curve=curve,
            error_threshold=error_threshold,
            min_fit_points=min_fit_points,
        )
        for curve in curves
    ]


def write_curves_csv(curves: list[AveragedCurve], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "xi_nm",
                "temperature_K",
                "mean_conductance",
                "std_conductance",
                "mean_sigma_S_per_cm",
                "std_sigma_S_per_cm",
            ]
        )
        for curve in curves:
            for row in zip(
                curve.temperatures_k,
                curve.mean_conductance,
                curve.std_conductance,
                curve.mean_conductivity,
                curve.std_conductivity,
            ):
                writer.writerow([curve.xi_nm, *row])


def make_derived_output_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def write_vrh_fit_summary_csv(
    fit_results: list[VRHFitResult],
    output_path: Path,
    error_threshold: float,
    min_fit_points: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "xi_nm",
                "fit_error_threshold",
                "min_fit_points",
                "last_threshold_compliant_temperature_K",
                "vrh_fit_max_temperature_K",
                "vrh_fit_epsilon",
                "vrh_fit_is_threshold_compliant",
                "transition_temperature_K",
                "transition_epsilon",
                "epsilon_at_t_max",
                "selected_fit_slope_ln_sigma_vs_inv_sqrt_t",
                "selected_fit_intercept_ln_sigma",
            ]
        )
        for fit_result in fit_results:
            writer.writerow(
                [
                    fit_result.xi_nm,
                    error_threshold,
                    min_fit_points,
                    fit_result.last_threshold_compliant_temperature_k,
                    fit_result.vrh_fit_max_temperature_k,
                    fit_result.vrh_fit_epsilon,
                    fit_result.vrh_fit_is_threshold_compliant,
                    fit_result.transition_temperature_k,
                    fit_result.transition_epsilon,
                    fit_result.epsilon_at_t_max,
                    fit_result.selected_fit_slope_ln_sigma_vs_inv_sqrt_t,
                    fit_result.selected_fit_intercept_ln_sigma,
                ]
            )


def write_vrh_fit_scan_csv(
    fit_results: list[VRHFitResult],
    output_path: Path,
    error_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "xi_nm",
                "tx_temperature_K",
                "epsilon_vrh",
                "fit_error_threshold",
                "within_threshold",
                "slope_ln_sigma_vs_inv_sqrt_t",
                "intercept_ln_sigma",
            ]
        )
        for fit_result in fit_results:
            for row in zip(
                fit_result.tx_temperatures_k,
                fit_result.epsilon_vrh,
                fit_result.slopes_ln_sigma_vs_inv_sqrt_t,
                fit_result.intercepts_ln_sigma,
            ):
                tx_temperature_k, epsilon_vrh, slope, intercept = row
                writer.writerow(
                    [
                        fit_result.xi_nm,
                        tx_temperature_k,
                        epsilon_vrh,
                        error_threshold,
                        epsilon_vrh <= error_threshold,
                        slope,
                        intercept,
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


def compute_axis_bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if values.size == 1:
        half_width = 0.5
        return np.array([values[0] - half_width, values[0] + half_width], dtype=float)

    deltas = np.diff(values)
    left_edge = values[0] - 0.5 * deltas[0]
    right_edge = values[-1] + 0.5 * deltas[-1]
    centers = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[left_edge], centers, [right_edge]])


def plot_curves(
    curves: list[AveragedCurve],
    output_dir: Path,
    energy_span_ev: float,
    show_plots: bool = False,
) -> None:
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        (
            lambda temperature_k: 1.0e3 / temperature_k,
            r"$1 / T$ ($10^{-3}$ K$^{-1}$)",
            rf"$\langle \sigma \rangle$ (S/cm) vs $1/T$ ($W_E$ = {energy_span_ev:.3g} eV)",
            (2.0, 10.0),
        ),
        (
            lambda temperature_k: 1.0e3 / np.sqrt(temperature_k),
            r"$1 / \sqrt{T}$ ($10^{-3}$ K$^{-1/2}$)",
            rf"$\langle \sigma \rangle$ (S/cm) vs $1/\sqrt{{T}}$ ($W_E$ = {energy_span_ev:.3g} eV)",
            (50.0, 100.0),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, (transform, xlabel, title, x_limits) in zip(axes, plot_specs):
        has_data = False

        for target_xi_nm, color in PLOT_HIGHLIGHT_XI_COLORS:
            curve = next(
                (
                    candidate_curve
                    for candidate_curve in curves
                    if np.isclose(candidate_curve.xi_nm, target_xi_nm)
                ),
                None,
            )
            if curve is None:
                continue

            x_values = transform(curve.temperatures_k)
            y_values = curve.mean_conductivity
            positive_mask = y_values > 0.0
            if np.any(positive_mask):
                ax.plot(
                    x_values[positive_mask],
                    y_values[positive_mask],
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.0,
                    label=f"xi = {curve.xi_nm:.3f} nm",
                )
                has_data = True

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle \sigma \rangle$ (S/cm)")
        ax.set_title(title)
        ax.set_xlim(*x_limits)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    f"{int(np.round(np.log10(value)))}"
                    if value > 0.0 and np.isclose(value, 10.0 ** np.round(np.log10(value)))
                    else ""
                )
            )
        )
        ax.grid(True, alpha=0.3, which="both")
        if has_data:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No positive sigma values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.tight_layout()

    plot_path = output_dir / "sigma_vs_inverse_temperature_and_inverse_sqrt_temperature.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_vrh_fit_map(
    fit_results: list[VRHFitResult],
    output_dir: Path,
    error_threshold: float,
    energy_span_ev: float,
    show_plots: bool = False,
) -> None:
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(fit_results, key=lambda fit_result: fit_result.xi_nm)
    xi_values_nm = np.array(
        [fit_result.xi_nm for fit_result in sorted_results],
        dtype=float,
    )
    tx_temperatures_k = sorted_results[0].tx_temperatures_k.copy()
    epsilon_matrix = np.vstack(
        [fit_result.epsilon_vrh for fit_result in sorted_results]
    )
    epsilon_vrh_milli = np.clip(1.0e3 * epsilon_matrix, 0.0, VRH_COLORMAP_MAX_MILLI)
    error_threshold_milli = 1.0e3 * error_threshold

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    heatmap = ax.pcolormesh(
        compute_axis_bin_edges(tx_temperatures_k),
        compute_axis_bin_edges(xi_values_nm),
        epsilon_vrh_milli,
        cmap="coolwarm",
        shading="auto",
        vmin=0.0,
        vmax=VRH_COLORMAP_MAX_MILLI,
    )

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label(r"$\epsilon_{\mathrm{VRH}}(T_x)$ ($10^{-3}$)")
    colorbar.ax.axhline(error_threshold_milli, color="white", linestyle="--", linewidth=1.2)
    colorbar.ax.text(
        1.1,
        error_threshold_milli,
        "threshold",
        color="white",
        va="center",
        ha="left",
        transform=colorbar.ax.get_yaxis_transform(),
    )

    ax.set_xlabel(r"$T_x$ (K)")
    ax.set_ylabel(r"$\xi$ (nm)")
    ax.set_title(
        rf"$T_x$ vs $\xi$ colored by $\epsilon_{{\mathrm{{VRH}}}}(T_x)$ "
        rf"($W_E$ = {energy_span_ev:.3g} eV)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "xi_vs_tx_colored_by_epsilon_vrh.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def print_summary(
    curves: list[AveragedCurve],
    cross_section_area_nm2: float,
) -> None:
    print("")
    print(
        "Averaged conductivity summary "
        f"(S/cm, cross_section_area={cross_section_area_nm2:.6g} nm^2):"
    )
    for curve in curves:
        sigma_low_t = curve.mean_conductivity[0]
        sigma_high_t = curve.mean_conductivity[-1]
        print(
            f"  xi={curve.xi_nm:.3f} nm: "
            f"<sigma>({curve.temperatures_k[0]:.0f} K)={sigma_low_t:.6e}, "
            f"<sigma>({curve.temperatures_k[-1]:.0f} K)={sigma_high_t:.6e}"
        )


def print_vrh_fit_summary(
    fit_results: list[VRHFitResult],
    error_threshold: float,
    min_fit_points: int,
) -> None:
    print("")
    print(
        "VRH fit summary "
        f"(fit: ln(sigma) vs T^(-1/2), epsilon threshold={error_threshold:.3e}, "
        f"min_fit_points={min_fit_points}):"
    )
    for fit_result in fit_results:
        if (
            fit_result.last_threshold_compliant_temperature_k is None
            and not fit_result.vrh_fit_is_threshold_compliant
        ):
            print(
                f"  xi={fit_result.xi_nm:.3f} nm: "
                "no threshold-compliant VRH window detected, "
                f"first scan at T_x={fit_result.vrh_fit_max_temperature_k:.0f} K "
                f"already exceeds threshold with epsilon={fit_result.vrh_fit_epsilon:.3e}"
            )
            continue

        if fit_result.transition_temperature_k is None:
            print(
                f"  xi={fit_result.xi_nm:.3f} nm: "
                f"no transition detected up to {fit_result.tx_temperatures_k[-1]:.0f} K, "
                f"epsilon(T_max)={fit_result.epsilon_at_t_max:.3e}"
            )
            continue

        if fit_result.last_threshold_compliant_temperature_k is None:
            print(
                f"  xi={fit_result.xi_nm:.3f} nm: "
                f"T_c={fit_result.transition_temperature_k:.0f} K, "
                "no threshold-compliant scan before transition, "
                f"fallback fit uses T_x={fit_result.vrh_fit_max_temperature_k:.0f} K, "
                f"epsilon(T_c)={fit_result.transition_epsilon:.3e}"
            )
            continue

        print(
            f"  xi={fit_result.xi_nm:.3f} nm: "
            f"T_c={fit_result.transition_temperature_k:.0f} K, "
            f"last threshold-compliant T_x={fit_result.last_threshold_compliant_temperature_k:.0f} K, "
            f"epsilon(T_c)={fit_result.transition_epsilon:.3e}"
        )


def main(
    n_sites: int = 100,
    delta_min_nm: float = 1.0,
    energy_span_ev: str | float | int | list[float] | tuple[float, ...] = "0.4",
    cross_section_area_nm2: float = 1.0,
    xi_values_nm: str | float | int | list[float] | tuple[float, ...] = ",".join(
        str(value) for value in DEFAULT_XI_VALUES_NM
    ),
    t_min_k: float = 100.0,
    t_max_k: float = 400.0,
    t_step_k: float = 2.0,
    n_realizations: int = 100,
    seed: int = 42,
    g0: float = 1.0,
    min_conductance: float = 0.0,
    fit_error_threshold: float = 5.0e-3,
    fit_min_points: int = 3,
    output: str | None = str(DEFAULT_OUTPUT_PATH),
    no_output: bool = False,
    plot: bool = False,
    plot_output_dir: str | None = None,
    show_plots: bool = False,
) -> None:
    energy_span_values_ev = parse_energy_span_values(energy_span_ev)
    multiple_energy_spans = len(energy_span_values_ev) > 1
    base_output_path = Path(output) if output is not None else DEFAULT_OUTPUT_PATH

    for index, single_energy_span_ev in enumerate(energy_span_values_ev):
        if index > 0:
            print("")

        print(f"Running 1D experiment for W_E = {single_energy_span_ev:.3g} eV")

        span_output = make_energy_span_output_path(
            base_output_path,
            single_energy_span_ev,
            multiple_energy_spans=multiple_energy_spans,
        )
        config = build_config(
            n_sites=n_sites,
            delta_min_nm=delta_min_nm,
            energy_span_ev=single_energy_span_ev,
            cross_section_area_nm2=cross_section_area_nm2,
            xi_values_nm=xi_values_nm,
            t_min_k=t_min_k,
            t_max_k=t_max_k,
            t_step_k=t_step_k,
            n_realizations=n_realizations,
            seed=seed,
            g0=g0,
            min_conductance=min_conductance,
            output=str(span_output),
            no_output=no_output,
        )
        curves = simulate_averaged_curves(config)
        fit_results = analyze_vrh_transitions(
            curves,
            error_threshold=fit_error_threshold,
            min_fit_points=fit_min_points,
        )

        if config.output_path is not None:
            write_curves_csv(curves, config.output_path)
            print(f"\nSaved averaged curves to {config.output_path}")
            fit_summary_output = make_derived_output_path(
                config.output_path,
                "vrh_fit_summary",
            )
            fit_scan_output = make_derived_output_path(
                config.output_path,
                "vrh_fit_scan",
            )
            write_vrh_fit_summary_csv(
                fit_results,
                fit_summary_output,
                error_threshold=fit_error_threshold,
                min_fit_points=fit_min_points,
            )
            write_vrh_fit_scan_csv(
                fit_results,
                fit_scan_output,
                error_threshold=fit_error_threshold,
            )
            print(f"Saved VRH fit summary to {fit_summary_output}")
            print(f"Saved VRH fit scan to {fit_scan_output}")

        print_summary(curves, cross_section_area_nm2=config.cross_section_area_nm2)
        print_vrh_fit_summary(
            fit_results,
            error_threshold=fit_error_threshold,
            min_fit_points=fit_min_points,
        )

        if plot:
            base_plot_output_dir = resolve_plot_output_dir(
                plot_output_dir=plot_output_dir,
                output_path=config.output_path,
            )
            output_dir = make_energy_span_plot_output_dir(
                base_plot_output_dir,
                single_energy_span_ev,
                multiple_energy_spans=multiple_energy_spans,
            )
            print("")
            plot_curves(
                curves,
                output_dir=output_dir,
                energy_span_ev=single_energy_span_ev,
                show_plots=show_plots,
            )
            plot_vrh_fit_map(
                fit_results,
                output_dir=output_dir,
                error_threshold=fit_error_threshold,
                energy_span_ev=single_energy_span_ev,
                show_plots=show_plots,
            )


if __name__ == "__main__":
    fire.Fire(main)
