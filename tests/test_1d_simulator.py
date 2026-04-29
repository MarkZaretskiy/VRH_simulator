from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
PLOT_DIR = ROOT_DIR / "tests" / "images" / "1d_simulator"
SUMMARY_PLOT_PATH = PLOT_DIR / "conductivity_summary.png"
CHAIN_PLOT_PATH = PLOT_DIR / "chain_geometry.png"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def load_1d_simulator_module():
    module_path = CODE_DIR / "1d_simulator.py"
    spec = importlib.util.spec_from_file_location("one_d_simulator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


one_d_simulator = load_1d_simulator_module()


def compute_integral_fit_error(
    temperatures_k: np.ndarray,
    observed_ln_sigma: np.ndarray,
    fitted_ln_sigma: np.ndarray,
) -> float:
    denominator_floor = 1.0e-8
    # denominator = np.maximum(np.abs(fitted_ln_sigma), denominator_floor) bug?
    
    # In the paper, the formula uses `fitted_ln_sigma` in the denominator.
    # However, this may be a mistake — it seems more logical to use
    # `observed_ln_sigma` instead.
    #
    # For example:
    # if observed = 1 and fitted = 100:
    # using fitted in the denominator → (100 - 1) / 100 = 0.99
    # using observed in the denominator → (100 - 1) / 1 = 99
    #
    # The second result better reflects the magnitude of the error,
    # so using `observed_ln_sigma` appears more reasonable.

    denominator = np.maximum(np.abs(observed_ln_sigma), denominator_floor)  #observed ln sigma typical values are in [-60, 20] range
    relative_squared_error = ((fitted_ln_sigma - observed_ln_sigma) / denominator) ** 2

    temperature_span = float(temperatures_k[-1] - temperatures_k[0])
    if temperature_span <= 0.0:
        return float(np.sqrt(np.mean(relative_squared_error)))

    return float(
        np.sqrt(np.trapezoid(relative_squared_error, temperatures_k) / temperature_span)
    )


def scan_vrh_fit_error(
    temperatures_k: np.ndarray,
    mean_ln_sigma: np.ndarray,
    *,
    start_temperature_k: float = 100.0,
    min_fit_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    start_index = int(np.searchsorted(temperatures_k, start_temperature_k, side="left"))
    tx_indices = np.arange(
        start_index + min_fit_points - 1,
        temperatures_k.size,
        dtype=int,
    )
    tx_temperatures_k = temperatures_k[tx_indices]
    fit_errors = np.empty(tx_indices.size, dtype=float)

    inv_sqrt_temperature = 1.0 / np.sqrt(temperatures_k)
    for scan_index, tx_index in enumerate(tx_indices):
        fit_slice = slice(start_index, tx_index + 1)
        x = inv_sqrt_temperature[fit_slice]
        y = mean_ln_sigma[fit_slice]
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted_y = slope * x + intercept
        fit_errors[scan_index] = compute_integral_fit_error(
            temperatures_k[fit_slice],
            y,
            fitted_y,
        )

    return tx_temperatures_k, fit_errors


class OneDSimulatorPlotTests(unittest.TestCase):
    def test_default_config_draws_sigma_vs_temperature_plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config, _ = one_d_simulator.load_json_config(
            one_d_simulator.DEFAULT_CONFIG_PATH
        )
        sweep = one_d_simulator.simulate_conductivity_sweep(config)

        self.assertEqual(sweep.temperatures_k.size, len(config.temperatures_k))
        self.assertTrue(np.all(np.isfinite(sweep.conductivity_s_per_cm)))
        self.assertTrue(np.all(sweep.conductivity_s_per_cm > 0.0))
        self.assertTrue(np.all(sweep.n_realizations_used > 0))

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PLOT_PATH.unlink(missing_ok=True)
        CHAIN_PLOT_PATH.unlink(missing_ok=True)

        conductivity_scale = config.length_nm / config.cross_section_area_nm2 * 1.0e7
        filtered_sigma_samples = (
            sweep.filtered_conductance_samples_s * conductivity_scale
        )
        mean_sigma = np.nanmean(filtered_sigma_samples, axis=0)
        std_sigma = np.nanstd(filtered_sigma_samples, axis=0, ddof=0)
        log_sigma_samples = np.log(filtered_sigma_samples)
        mean_log_sigma = np.nanmean(log_sigma_samples, axis=0)
        typical_sigma = np.exp(mean_log_sigma)
        np.testing.assert_allclose(mean_sigma, sweep.conductivity_s_per_cm)
        np.testing.assert_allclose(std_sigma, sweep.conductivity_std_s_per_cm)
        np.testing.assert_allclose(typical_sigma, sweep.typical_conductivity_s_per_cm)
        np.testing.assert_allclose(mean_log_sigma, sweep.mean_ln_conductivity)

        tx_temperatures_k, vrh_fit_errors = scan_vrh_fit_error(
            sweep.temperatures_k,
            sweep.mean_ln_conductivity,
            start_temperature_k=100.0,
            min_fit_points=3,
        )
        self.assertTrue(np.all(np.isfinite(vrh_fit_errors)))

        fig, axes = plt.subplots(3, 1, figsize=(7.4, 11.0))
        sigma_ax, vrh_ax, error_ax = axes

        sigma_ax.semilogy(
            sweep.temperatures_k,
            typical_sigma,
            marker="o",
            linewidth=1.5,
            label="typical = exp(mean ln sigma)",
        )
        sigma_ax.set_xlabel("Temperature T (K)")
        sigma_ax.set_ylabel("Conductivity sigma (S/cm)")
        sigma_ax.set_title("1D RRN conductivity sweep")
        sigma_ax.grid(True, which="both", alpha=0.3)
        sigma_ax.legend()

        inv_sqrt_temperature = 1.0 / np.sqrt(sweep.temperatures_k)
        vrh_ax.plot(
            inv_sqrt_temperature,
            sweep.mean_ln_conductivity,
            marker="o",
            linewidth=1.5,
            label="mean ln(sigma)",
        )
        vrh_ax.set_xlabel("T^(-1/2) (K^(-1/2))")
        vrh_ax.set_ylabel("ln(sigma), sigma in S/cm")
        vrh_ax.set_title("1D VRH linearization")
        vrh_ax.grid(True, alpha=0.3)
        vrh_ax.legend()

        error_ax.plot(
            tx_temperatures_k,
            vrh_fit_errors,
            marker="o",
            linewidth=1.5,
        )
        error_ax.set_xlabel("Fit-window upper temperature Tx (K)")
        error_ax.set_ylabel("Integral fit error")
        error_ax.set_title("1D VRH fit error from T0 = 100 K to Tx")
        error_ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(SUMMARY_PLOT_PATH, dpi=200)
        plt.close(fig)

        self.assertTrue(SUMMARY_PLOT_PATH.exists())
        self.assertGreater(SUMMARY_PLOT_PATH.stat().st_size, 0)

        n_realizations = sweep.realization_energies_ev.shape[0]
        ncols = 1
        nrows = n_realizations
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10.5, 1.15 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        energy_abs_max = float(np.max(np.abs(sweep.realization_energies_ev)))
        scatter = None
        for realization_index in range(n_realizations):
            ax = axes[realization_index][0]
            positions_x_nm = sweep.realization_positions_nm[realization_index, :, 0]
            energies_ev = sweep.realization_energies_ev[realization_index]
            ax.plot(
                positions_x_nm,
                np.zeros_like(positions_x_nm),
                color="0.65",
                linewidth=0.8,
                zorder=1,
            )
            scatter = ax.scatter(
                positions_x_nm,
                np.zeros_like(positions_x_nm),
                c=energies_ev,
                cmap="coolwarm",
                vmin=-energy_abs_max,
                vmax=energy_abs_max,
                edgecolors="black",
                linewidths=0.25,
                s=22,
                zorder=2,
            )
            ax.scatter(
                [positions_x_nm[0], positions_x_nm[-1]],
                [0.0, 0.0],
                marker="s",
                s=38,
                color="black",
                zorder=3,
            )
            outlier_temperatures = sweep.temperatures_k[
                sweep.outlier_mask[realization_index]
            ]
            if outlier_temperatures.size:
                outlier_label = (
                    f"outliers: {outlier_temperatures.size}T, "
                    f"{outlier_temperatures[0]:.0f}-{outlier_temperatures[-1]:.0f} K"
                )
            else:
                outlier_label = "no outliers"
            ax.set_title(f"R{realization_index:02d}: {outlier_label}", fontsize=8)
            ax.set_yticks([])
            ax.margins(x=0.02, y=0.5)

        for ax in axes[-1]:
            if ax.has_data():
                ax.set_xlabel("x (nm)")

        fig.suptitle("1D RRN chain geometries by realization", y=1.0)
        colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), pad=0.01)
        colorbar.set_label("Site energy epsilon (eV)")
        fig.savefig(CHAIN_PLOT_PATH, dpi=200)
        plt.close(fig)

        self.assertTrue(CHAIN_PLOT_PATH.exists())
        self.assertGreater(CHAIN_PLOT_PATH.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
