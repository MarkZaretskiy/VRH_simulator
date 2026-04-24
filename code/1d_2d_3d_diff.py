from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .sim import RRNSolver
except ImportError:
    from sim import RRNSolver


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("1d_2d_3d_diff_results")
NM_TO_CM = 1.0e-7


REGIMES = (
    ("nnh", "NNH", -1.0, "#f28e2b"),
    ("vrh_1d", "1D VRH", -0.5, "#8cd17d"),
    ("vrh_2d", "2D VRH", -1.0 / 3.0, "#4e79a7"),
    ("vrh_3d", "3D VRH", -0.25, "#1f3a93"),
)


@dataclass(frozen=True)
class Config:
    nx: int
    ny: int
    nz: int
    dx_nm: float
    y_spacing_factor: float
    z_spacing_factor: float
    energy_span_ev: float
    xi_values_nm: np.ndarray
    temperatures_k: np.ndarray
    n_realizations: int
    n_jobs: int
    seed: int
    max_neighbors: int
    min_fit_points: int
    g0: float
    min_conductance: float
    selected_temperature_k: float
    output_dir: Path

    @property
    def dy_nm(self) -> float:
        return self.dx_nm * self.y_spacing_factor

    @property
    def dz_nm(self) -> float:
        return self.dx_nm * self.z_spacing_factor

    @property
    def length_nm(self) -> float:
        return self.dx_nm * (self.nx - 1)

    @property
    def area_nm2(self) -> float:
        return max(self.dy_nm * (self.ny - 1), self.dy_nm) * max(
            self.dz_nm * (self.nz - 1),
            self.dz_nm,
        )


def parse_float_range(text: str) -> np.ndarray:
    if ":" in text and "," not in text:
        start_text, stop_text, step_text = (part.strip() for part in text.split(":"))
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        if step <= 0:
            raise ValueError("range step must be > 0")
        values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    else:
        values = np.array(
            [float(part.strip()) for part in text.split(",") if part.strip()],
            dtype=float,
        )
    if values.size == 0:
        raise ValueError("range must contain at least one value")
    return values


def build_anisotropic_grid(config: Config) -> np.ndarray:
    x_values = np.arange(config.nx, dtype=float) * config.dx_nm
    y_values = np.arange(config.ny, dtype=float) * config.dy_nm
    z_values = np.arange(config.nz, dtype=float) * config.dz_nm
    xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def contact_nodes_for_grid(config: Config) -> tuple[np.ndarray, np.ndarray]:
    yz_count = config.ny * config.nz
    left_nodes = np.arange(yz_count, dtype=int)
    right_start = (config.nx - 1) * yz_count
    right_nodes = np.arange(right_start, right_start + yz_count, dtype=int)
    return left_nodes, right_nodes


def random_energies(
    rng: np.random.Generator,
    n_sites: int,
    energy_span_ev: float,
) -> np.ndarray:
    half_span = 0.5 * energy_span_ev
    return rng.uniform(-half_span, half_span, size=n_sites)


def conductance_to_conductivity_cm(
    conductance: float,
    *,
    length_nm: float,
    area_nm2: float,
) -> float:
    length_cm = length_nm * NM_TO_CM
    area_cm2 = area_nm2 * NM_TO_CM * NM_TO_CM
    return conductance * length_cm / area_cm2


def simulate_curves(config: Config) -> dict[float, dict[str, np.ndarray]]:
    positions = build_anisotropic_grid(config)
    left_nodes, right_nodes = contact_nodes_for_grid(config)
    curves: dict[float, dict[str, np.ndarray]] = {}

    tasks = []
    for xi_index, xi_nm in enumerate(config.xi_values_nm):
        for realization_index in range(config.n_realizations):
            realization_seed = (
                config.seed
                + 1_000_003 * xi_index
                + 9_176 * realization_index
            )
            tasks.append(
                (
                    float(xi_nm),
                    realization_seed,
                    positions,
                    left_nodes,
                    right_nodes,
                    config.temperatures_k,
                    config.energy_span_ev,
                    config.g0,
                    config.max_neighbors,
                    config.min_conductance,
                    config.length_nm,
                    config.area_nm2,
                )
            )

    sigma_by_xi: dict[float, list[np.ndarray]] = {
        float(xi_nm): [] for xi_nm in config.xi_values_nm
    }

    if config.n_jobs == 1:
        for task in tasks:
            xi_nm, sigmas = simulate_realization(task)
            sigma_by_xi[xi_nm].append(sigmas)
    else:
        with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
            futures = [executor.submit(simulate_realization, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                xi_nm, sigmas = future.result()
                sigma_by_xi[xi_nm].append(sigmas)
                completed += 1
                if completed == len(futures) or completed % max(config.n_jobs, 1) == 0:
                    print(f"Completed {completed}/{len(futures)} realization curves")

    for xi_nm, sigma_by_realization in sigma_by_xi.items():
        sigma_matrix = np.asarray(sigma_by_realization, dtype=float)
        curves[xi_nm] = {
            "mean_sigma": np.mean(sigma_matrix, axis=0),
            "median_sigma": np.median(sigma_matrix, axis=0),
            "std_sigma": np.std(sigma_matrix, axis=0, ddof=0),
        }

    return curves


def simulate_realization(
    task: tuple[
        float,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        int,
        float,
        float,
        float,
    ],
) -> tuple[float, np.ndarray]:
    (
        xi_nm,
        seed,
        positions,
        left_nodes,
        right_nodes,
        temperatures_k,
        energy_span_ev,
        g0,
        max_neighbors,
        min_conductance,
        length_nm,
        area_nm2,
    ) = task
    rng = np.random.default_rng(seed)
    energies = random_energies(rng, len(positions), energy_span_ev)
    sigmas = []
    for temperature_k in temperatures_k:
        solver = RRNSolver(
            positions=positions,
            energies=energies,
            temperature=float(temperature_k),
            xi=float(xi_nm),
            G0=g0,
            max_neighbors=max_neighbors,
            min_conductance=min_conductance,
        )
        result = solver.solve(left_nodes=left_nodes, right_nodes=right_nodes)
        sigmas.append(
            conductance_to_conductivity_cm(
                result.effective_conductance,
                length_nm=length_nm,
                area_nm2=area_nm2,
            )
        )
    return xi_nm, np.asarray(sigmas, dtype=float)


def fit_error(
    temperatures_k: np.ndarray,
    sigma: np.ndarray,
    exponent: float,
) -> float:
    valid = np.isfinite(sigma) & (sigma > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.inf

    x = temperatures_k[valid] ** exponent
    y = np.log(sigma[valid])
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted_y = slope * x + intercept
    denominator_floor = 1.0e-30
    relative_residual = (y - fitted_y) / np.where(
        np.abs(y) > denominator_floor,
        y,
        np.sign(y) * denominator_floor + (y == 0.0) * denominator_floor,
    )
    if x.size == 1:
        return float(abs(relative_residual[0]))

    squared_relative_residual = relative_residual * relative_residual
    t_window = temperatures_k[valid]
    temperature_span = float(t_window[-1] - t_window[0])
    if temperature_span <= 0.0:
        return float(np.sqrt(np.mean(squared_relative_residual)))

    integrated_error = np.trapezoid(squared_relative_residual, t_window)
    return float(np.sqrt(integrated_error / temperature_span))


def classify_windows(
    config: Config,
    curves: dict[float, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    n_xi = len(config.xi_values_nm)
    n_t = len(config.temperatures_k)
    regime_index = np.full((n_xi, n_t), -1, dtype=int)
    errors = np.full((n_xi, n_t, len(REGIMES)), np.nan, dtype=float)

    for xi_index, xi_nm in enumerate(config.xi_values_nm):
        sigma = curves[float(xi_nm)]["median_sigma"]
        for tx_index in range(config.min_fit_points - 1, n_t):
            window_t = config.temperatures_k[: tx_index + 1]
            window_sigma = sigma[: tx_index + 1]
            regime_errors = [
                fit_error(window_t, window_sigma, exponent)
                for _, _, exponent, _ in REGIMES
            ]
            errors[xi_index, tx_index, :] = regime_errors
            regime_index[xi_index, tx_index] = int(np.argmin(regime_errors))

    return regime_index, errors


def write_curves_csv(
    config: Config,
    curves: dict[float, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "xi_nm",
                "temperature_k",
                "mean_sigma_s_per_cm",
                "median_sigma_s_per_cm",
                "std_sigma_s_per_cm",
            ]
        )
        for xi_nm in config.xi_values_nm:
            curve = curves[float(xi_nm)]
            for temperature_k, mean_sigma, median_sigma, std_sigma in zip(
                config.temperatures_k,
                curve["mean_sigma"],
                curve["median_sigma"],
                curve["std_sigma"],
                strict=True,
            ):
                writer.writerow(
                    [xi_nm, temperature_k, mean_sigma, median_sigma, std_sigma]
                )


def write_classification_csv(
    config: Config,
    regime_index: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["xi_nm", "tx_temperature_k", "best_regime"]
        header.extend(f"error_{regime_id}" for regime_id, _, _, _ in REGIMES)
        writer.writerow(header)
        for xi_index, xi_nm in enumerate(config.xi_values_nm):
            for tx_index, tx_temperature_k in enumerate(config.temperatures_k):
                best = regime_index[xi_index, tx_index]
                label = "" if best < 0 else REGIMES[best][0]
                writer.writerow(
                    [
                        xi_nm,
                        tx_temperature_k,
                        label,
                        *errors[xi_index, tx_index, :],
                    ]
                )


def plot_results(
    config: Config,
    regime_index: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap([color for _, _, _, color in REGIMES])
    masked_regimes = np.ma.masked_where(regime_index < 0, regime_index)

    fig, (map_ax, error_ax) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    map_ax.imshow(
        masked_regimes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(REGIMES) - 0.5,
        extent=[
            float(config.temperatures_k[0]),
            float(config.temperatures_k[-1]),
            float(config.xi_values_nm[0]),
            float(config.xi_values_nm[-1]),
        ],
    )
    map_ax.set_title("Best linearized transport regime")
    map_ax.set_xlabel(r"Fit-window upper temperature $T_x$ (K)")
    map_ax.set_ylabel(r"Localization length $\xi$ (nm)")
    map_ax.grid(True, color="white", alpha=0.2, linewidth=0.5)
    map_ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for _, label, _, color in REGIMES
        ],
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    selected_index = int(
        np.argmin(np.abs(config.temperatures_k - config.selected_temperature_k))
    )
    for regime_index_value, (_, label, _, color) in enumerate(REGIMES):
        error_ax.plot(
            config.xi_values_nm,
            errors[:, selected_index, regime_index_value],
            marker="o",
            linewidth=1.5,
            color=color,
            label=label,
        )
    error_ax.set_title(
        rf"Fit error at $T_x$ = {config.temperatures_k[selected_index]:.0f} K"
    )
    error_ax.set_xlabel(r"Localization length $\xi$ (nm)")
    error_ax.set_ylabel(r"relative fit error in $\ln(\sigma)$")
    error_ax.grid(True, alpha=0.3)
    error_ax.legend(fontsize=9)

    fig.suptitle(
        (
            f"Anisotropic 3D grid: dx={config.dx_nm:g} nm, "
            f"dy={config.dy_nm:g} nm, dz={config.dz_nm:g} nm, "
            f"WE={config.energy_span_ev:g} eV"
        ),
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_config(args: argparse.Namespace) -> Config:
    xi_values_nm = parse_float_range(args.xi_values_nm)
    temperatures_k = parse_float_range(args.temperatures_k)

    if args.nx < 2:
        raise ValueError("nx must be >= 2")
    if args.ny < 1 or args.nz < 1:
        raise ValueError("ny and nz must be >= 1")
    if args.dx_nm <= 0:
        raise ValueError("dx_nm must be > 0")
    if args.y_spacing_factor <= 0 or args.z_spacing_factor <= 0:
        raise ValueError("spacing factors must be > 0")
    if args.energy_span_ev <= 0:
        raise ValueError("energy_span_ev must be > 0")
    if args.n_realizations <= 0:
        raise ValueError("n_realizations must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    if args.max_neighbors <= 0:
        raise ValueError("max_neighbors must be > 0")
    if args.min_fit_points < 2:
        raise ValueError("min_fit_points must be >= 2")

    return Config(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx_nm=args.dx_nm,
        y_spacing_factor=args.y_spacing_factor,
        z_spacing_factor=args.z_spacing_factor,
        energy_span_ev=args.energy_span_ev,
        xi_values_nm=xi_values_nm,
        temperatures_k=temperatures_k,
        n_realizations=args.n_realizations,
        n_jobs=args.n_jobs,
        seed=args.seed,
        max_neighbors=args.max_neighbors,
        min_fit_points=args.min_fit_points,
        g0=args.g0,
        min_conductance=args.min_conductance,
        selected_temperature_k=args.selected_temperature_k,
        output_dir=Path(args.output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate an anisotropic 3D RRN lattice and plot the fitted "
            "NNH/1D/2D/3D VRH regime map, following the dimensionality "
            "transition setup in arXiv:2601.01243."
        )
    )
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--y-spacing-factor", type=float, default=5.0)
    parser.add_argument("--z-spacing-factor", type=float, default=5.0)
    parser.add_argument("--energy-span-ev", type=float, default=0.4)
    parser.add_argument("--xi-values-nm", default="0.05:0.30:0.025")
    parser.add_argument("--temperatures-k", default="100:400:10")
    parser.add_argument("--n-realizations", type=int, default=8)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max((os.cpu_count() or 2) - 1, 1),
        help="Number of worker processes for independent realization curves.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=120)
    parser.add_argument("--min-fit-points", type=int, default=8)
    parser.add_argument("--g0", type=float, default=1.0)
    parser.add_argument("--min-conductance", type=float, default=0.0)
    parser.add_argument("--selected-temperature-k", type=float, default=250.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Simulating "
        f"{config.nx * config.ny * config.nz} sites, "
        f"{len(config.xi_values_nm)} xi values, "
        f"{len(config.temperatures_k)} temperatures, "
        f"{config.n_realizations} realizations, "
        f"{config.n_jobs} worker processes..."
    )
    curves = simulate_curves(config)
    regime_index, errors = classify_windows(config, curves)

    curves_path = config.output_dir / "anisotropic_grid_curves.csv"
    classification_path = config.output_dir / "regime_classification.csv"
    plot_path = config.output_dir / "regime_transition_map.png"

    write_curves_csv(config, curves, curves_path)
    write_classification_csv(config, regime_index, errors, classification_path)
    plot_results(config, regime_index, errors, plot_path)

    print(f"Saved curves to {curves_path}")
    print(f"Saved classification table to {classification_path}")
    print(f"Saved regime map to {plot_path}")


if __name__ == "__main__":
    main()
