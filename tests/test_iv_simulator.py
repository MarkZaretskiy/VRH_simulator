from __future__ import annotations

import io
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse.linalg import MatrixRankWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
PERSISTENT_PLOT_DIR = ROOT_DIR / "tmp" / "test_iv_simulator_plots"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import iv_simulator
import sim


EXPECTED_TEMPERATURES_K = [float(value) for value in range(10, 201, 10)]


class IVSimulatorConcentrationSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_concentration_sweep(self, **overrides: object) -> None:
        params: dict[str, object] = {
            "concentration_cm3": 1.8e20,
            "temperatures_k": None,
            "t_min_k": 10.0,
            "t_max_k": 200.0,
            "t_step_k": 10.0,
            "v_min": -0.5,
            "v_max": 0.5,
            "v_step": 0.5,
            "n_realizations": 1,
            "n_jobs": 1,
            "device_length_nm": 20.0,
            "device_width_nm": 10.0,
            "device_thickness_nm": 3.0,
            "cutoff_distance": None,
            "max_neighbors": None,
            "min_conductance": 0.0,
            "seed": 42,
            "show_plots": False,
        }
        params.update(overrides)
        with redirect_stdout(io.StringIO()):
            iv_simulator.main(**params)

    def test_main_writes_iv_table_for_concentration_temperature_sweep(self) -> None:
        output_path = self.temp_path / "iv_box_concentration.csv"

        self.run_concentration_sweep(
            no_output=False,
            output=str(output_path),
            plot=False,
        )

        self.assertTrue(output_path.exists())
        table = pd.read_csv(output_path)

        self.assertEqual(len(table), len(EXPECTED_TEMPERATURES_K) * 3)

        temperatures = sorted(table["temperature_K"].unique().tolist())
        voltages = sorted(table["voltage_V"].unique().tolist())
        self.assertEqual(temperatures, EXPECTED_TEMPERATURES_K)
        self.assertEqual(voltages, [-0.5, 0.0, 0.5])

        for temperature_k in temperatures:
            temperature_rows = table[table["temperature_K"] == temperature_k].set_index(
                "voltage_V"
            )
            non_conductive_count = int(
                temperature_rows.loc[0.0, "non_conductive_realizations"]
            )
            negative_current = float(temperature_rows.loc[-0.5, "current_mean_A"])
            zero_current = float(temperature_rows.loc[0.0, "current_mean_A"])
            positive_current = float(temperature_rows.loc[0.5, "current_mean_A"])

            for row in temperature_rows.itertuples():
                self.assertEqual(int(row.n_realizations), 1)
                self.assertEqual(
                    int(row.non_conductive_realizations),
                    non_conductive_count,
                )

            self.assertAlmostEqual(zero_current, 0.0)
            self.assertAlmostEqual(positive_current, -negative_current)
            if non_conductive_count == 0:
                self.assertGreater(positive_current, 0.0)
                self.assertLess(negative_current, 0.0)
            else:
                self.assertEqual(non_conductive_count, 1)
                self.assertAlmostEqual(positive_current, 0.0)
                self.assertAlmostEqual(negative_current, 0.0)

    def test_main_writes_plot_artifacts_for_concentration_temperature_sweep(self) -> None:
        plot_dir = PERSISTENT_PLOT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

        iv_plot_path = plot_dir / "iv_curves_vs_temperature.png"
        conductance_plot_path = plot_dir / "conductance_vs_temperature.png"
        iv_plot_path.unlink(missing_ok=True)
        conductance_plot_path.unlink(missing_ok=True)

        self.run_concentration_sweep(
            no_output=True,
            plot=True,
            plot_output_dir=str(plot_dir),
        )

        self.assertTrue(iv_plot_path.exists())
        self.assertGreater(iv_plot_path.stat().st_size, 0)
        self.assertTrue(conductance_plot_path.exists())
        self.assertGreater(conductance_plot_path.stat().st_size, 0)


class RRNSolverRobustnessTests(unittest.TestCase):
    def test_solver_ignores_disconnected_islands_when_contacts_are_connected(self) -> None:
        positions = np.array([[0.0], [1.0], [1000.0]], dtype=float)
        energies = np.zeros(3, dtype=float)
        solver = sim.RRNSolver(
            positions=positions,
            energies=energies,
            temperature=300.0,
            xi=1.0,
            G0=1.0,
            cutoff_distance=2.0,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            result = solver.solve(
                left_nodes=np.array([0], dtype=int),
                right_nodes=np.array([1], dtype=int),
                V_left=1.0,
                V_right=0.0,
            )

        expected_conductance = (1.0 / 300.0) * np.exp(-2.0)
        self.assertAlmostEqual(
            result.effective_conductance,
            expected_conductance,
            places=15,
        )


class ContactSelectionTests(unittest.TestCase):
    def test_build_contacts_uses_fixed_device_bounds_instead_of_sample_extrema(self) -> None:
        positions = np.array([[1.2], [2.2], [60.0], [119.0]], dtype=float)
        config = iv_simulator.SimulationConfig(
            positions=[[1.2], [2.2], [60.0], [119.0]],
            energies=[0.0, 0.0, 0.0, 0.0],
            n_realizations=1,
        )

        left_nodes, right_nodes = iv_simulator.build_contacts(
            config,
            positions=positions,
            x_min=0.0,
            x_max=120.0,
        )

        np.testing.assert_array_equal(left_nodes, np.array([0], dtype=int))
        np.testing.assert_array_equal(right_nodes, np.array([3], dtype=int))

    def test_contact_region_width_uses_absolute_layer_thickness_constant(self) -> None:
        self.assertAlmostEqual(
            iv_simulator.resolve_contact_region_width(),
            iv_simulator.DEFAULT_CONTACT_LAYER_THICKNESS_NM,
        )

    def test_build_contacts_falls_back_to_nearest_boundary_nodes_when_strip_is_empty(self) -> None:
        positions = np.array([[0.8], [60.0], [119.2]], dtype=float)
        config = iv_simulator.SimulationConfig(
            positions=positions.tolist(),
            energies=[0.0, 0.0, 0.0],
            xi=0.35,
            n_realizations=1,
        )

        left_nodes, right_nodes = iv_simulator.build_contacts(
            config,
            positions=positions,
            x_min=0.0,
            x_max=120.0,
        )

        np.testing.assert_array_equal(left_nodes, np.array([0], dtype=int))
        np.testing.assert_array_equal(right_nodes, np.array([2], dtype=int))

    def test_build_network_reports_fixed_x_bounds_for_generated_box_network(self) -> None:
        config = iv_simulator.SimulationConfig(
            concentration_cm3=1.8e20,
            device_length_nm=20.0,
            device_width_nm=10.0,
            device_thickness_nm=3.0,
            max_generated_sites=1000,
            length=5.0,
            dim=1,
            energy_std=0.4,
            seed=42,
        )

        _, _, metadata = iv_simulator.build_network(
            config,
            seed=42,
        )

        self.assertEqual(metadata["x_min"], 0.0)
        self.assertEqual(metadata["x_max"], 20.0)


if __name__ == "__main__":
    unittest.main()
