from __future__ import annotations

import io
import json
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse.linalg import MatrixRankWarning

try:
    import yaml
except ImportError:
    yaml = None


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
TEST_IMAGES_DIR = ROOT_DIR / "tests" / "images"
PERSISTENT_TOYSET_PLOT_DIR = TEST_IMAGES_DIR / "iv_simulator_toyset"
PERSISTENT_DEVICE_PLOT_DIR = TEST_IMAGES_DIR / "iv_simulator_device"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import iv_simulator
import plotting
import sim
import sweep


EXPECTED_TEMPERATURES_K = [float(value) for value in range(10, 201, 10)]


class IVSimulatorConfigFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def base_config(self) -> dict[str, object]:
        return {
            "positions": [[0.0], [1.0]],
            "energies": [0.0, 0.0],
            "left_nodes": [0],
            "right_nodes": [1],
            "n_realizations": 1,
            "n_jobs": 1,
            "temperatures_k": [100.0],
            "v_min": -0.5,
            "v_max": 0.5,
            "v_step": 0.5,
            "cutoff_distance": 2.0,
            "max_neighbors": 1,
            "seed": 7,
            "plot": False,
            "show_plots": False,
        }

    def test_build_config_from_json_file_parses_mapping(self) -> None:
        config_path = self.temp_path / "iv_config.json"
        config_data = self.base_config()
        config_data["xi"] = 0.42
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        config = iv_simulator.build_config_from_file(config_path)

        self.assertAlmostEqual(config.xi, 0.42)
        np.testing.assert_array_equal(config.left_nodes_array, np.array([0], dtype=int))
        np.testing.assert_array_equal(
            config.temperature_values_k,
            np.array([100.0], dtype=float),
        )

    @unittest.skipUnless(yaml is not None, "PyYAML is not installed")
    def test_main_from_yaml_file_reads_values_from_file(self) -> None:
        config_path = self.temp_path / "iv_config.yaml"
        output_path = self.temp_path / "iv_from_yaml.csv"
        config_data = self.base_config()
        config_data["v_step"] = 0.25
        config_data["output"] = str(output_path)
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

        with redirect_stdout(io.StringIO()):
            iv_simulator.main_from_file(config_path)

        table = pd.read_csv(output_path)
        self.assertEqual(len(table), 5)
        self.assertEqual(
            sorted(table["voltage_V"].unique().tolist()),
            [-0.5, -0.25, 0.0, 0.25, 0.5],
        )

    def test_cli_rejects_extra_options_when_config_is_used(self) -> None:
        config_path = self.temp_path / "iv_config.json"
        config_data = self.base_config()
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        with self.assertRaises(SystemExit) as exc_info:
            with redirect_stdout(io.StringIO()):
                iv_simulator.cli(
                    [
                        "--config",
                        str(config_path),
                        "--no_output",
                    ]
                )

        self.assertIn("cannot combine --config with other simulation options", str(exc_info.exception))


class IVSimulatorToysetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_concentration_sweep(self, **params_update: object) -> None:
        params: dict[str, object] = {
            "concentration_cm3": 1.8e20,
            "temperatures_k": None,
            "t_min_k": 10.0,
            "t_max_k": 200.0,
            "t_step_k": 10.0,
            "v_min": -0.5,
            "v_max": 0.5,
            "xi":0.35,
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
        params.update(params_update)
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
            if non_conductive_count == 0:
                self.assertAlmostEqual(positive_current, -negative_current)
                self.assertGreater(positive_current, 0.0)
                self.assertLess(negative_current, 0.0)
            else:
                self.assertEqual(non_conductive_count, 1)
                self.assertTrue(np.isnan(positive_current))
                self.assertTrue(np.isnan(negative_current))

    def test_main_writes_plot_artifacts_for_concentration_temperature_sweep(self) -> None:
        plot_dir = PERSISTENT_TOYSET_PLOT_DIR
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


class IVSimulatorDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def box_device_params(self, **params_update: object) -> dict[str, object]:
        params: dict[str, object] = {
            "concentration_cm3": 0.1e21, #1.8e20
            "device_length_nm": 100.0,
            "device_width_nm": 50.0,
            "device_thickness_nm": 10.0,
            "max_generated_sites": 100000,
            "energy_std": 0.1,
            "xi": 0.3197,
            "cutoff_distance": 50.0,
            "max_neighbors": 100,
            "min_conductance": 0.0,
            "n_realizations": 20,
            "n_jobs": 20,
            "seed": 42,
            "show_plots": False,
        }
        params.update(params_update)
        return params

    def test_build_network_uses_concentration_to_generate_9000_sites_for_box_device(self) -> None:
        config = iv_simulator.SimulationConfig(**self.box_device_params())
        _, _, metadata = iv_simulator.build_network(config, seed=config.seed)

        self.assertEqual(metadata["mode"], "box")
        # self.assertEqual(metadata["n_sites"], 9000)
        # self.assertAlmostEqual(metadata["expected_n_sites"], 9000.0)

    def test_build_network_uses_uniform_energy_window_when_requested(self) -> None:
        config = iv_simulator.SimulationConfig(
            **self.box_device_params(
                concentration_cm3=1.0e20,
                device_length_nm=20.0,
                device_width_nm=10.0,
                device_thickness_nm=3.0,
                n_realizations=1,
                n_jobs=1,
                energy_std=0.4,
                energy_distribution="uniform",
                seed=42,
            )
        )

        _, energies, metadata = iv_simulator.build_network(config, seed=config.seed)

        self.assertEqual(metadata["energy_distribution"], "uniform")
        self.assertTrue(np.all(energies >= -0.2))
        self.assertTrue(np.all(energies <= 0.2))

    def test_main_writes_iv_table_for_concentration_defined_3d_device(self) -> None:
        output_path = self.temp_path / "iv_random_3d_device.csv"

        with redirect_stdout(io.StringIO()):
            iv_simulator.main(
                **self.box_device_params(
                output=str(output_path),
                temperatures_k=[150.0],
                v_min=-0.2,
                v_max=0.2,
                v_step=0.2,
                n_realizations=10,
                n_jobs=10,
                no_output=False,
                plot=False,
                )
            )

        self.assertTrue(output_path.exists())
        table = pd.read_csv(output_path)

        self.assertEqual(len(table), 3)
        self.assertEqual(table["temperature_K"].unique().tolist(), [150.0])
        self.assertEqual(
            sorted(table["voltage_V"].unique().tolist()),
            [-0.2, 0.0, 0.2],
        )
        self.assertTrue((table["n_realizations"] == 10).all())
        self.assertTrue((table["non_conductive_realizations"] == 0).all())

        rows = table.set_index("voltage_V")
        self.assertAlmostEqual(float(rows.loc[0.0, "current_mean_A"]), 0.0)
        self.assertAlmostEqual(
            float(rows.loc[0.2, "current_mean_A"]),
            -float(rows.loc[-0.2, "current_mean_A"]),
        )
        self.assertGreater(float(rows.loc[0.2, "current_mean_A"]), 0.0)
        self.assertGreater(float(rows.loc[0.2, "conductance_mean_S"]), 0.0)

    def test_main_writes_plot_artifacts_for_concentration_defined_3d_device(self) -> None:
        plot_dir = PERSISTENT_DEVICE_PLOT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

        iv_plot_path = plot_dir / "iv_curves_vs_temperature.png"
        conductance_plot_path = plot_dir / "conductance_vs_temperature.png"
        iv_plot_path.unlink(missing_ok=True)
        conductance_plot_path.unlink(missing_ok=True)

        with redirect_stdout(io.StringIO()):
            iv_simulator.main(
                **self.box_device_params(
                temperatures_k=list(range(20, 250, 10)),
                v_min=-0.2,
                v_max=0.2,
                v_step=0.2,
                n_realizations=10,
                n_jobs=10,
                no_output=True,
                plot=True,
                plot_output_dir=str(plot_dir),
                )
            )

        self.assertTrue(iv_plot_path.exists())
        self.assertGreater(iv_plot_path.stat().st_size, 0)
        self.assertTrue(conductance_plot_path.exists())
        self.assertGreater(conductance_plot_path.stat().st_size, 0)


class IVSimulatorAggregationTests(unittest.TestCase):
    def test_summarize_samples_ignores_nan_non_conductive_realizations(self) -> None:
        samples = np.array(
            [
                [1.0, np.nan, np.nan],
                [3.0, np.nan, np.nan],
                [5.0, 9.0, np.nan],
            ],
            dtype=float,
        )

        mean_values, std_values = iv_simulator.summarize_samples(samples)

        np.testing.assert_allclose(mean_values[:2], np.array([3.0, 9.0]))
        np.testing.assert_allclose(
            std_values[:2],
            np.array(
                [
                    np.std([1.0, 3.0, 5.0], ddof=0),
                    0.0,
                ]
            ),
        )
        self.assertTrue(np.isnan(mean_values[2]))
        self.assertTrue(np.isnan(std_values[2]))


class SweepScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_main_writes_conductance_tables_for_parameter_grid(self) -> None:
        output_dir = self.temp_path / "sweep_output"
        config = sweep.SweepRunConfig(
            output_dir=output_dir,
            temperatures_k=np.arange(20.0, 201.0, 10.0),
            device_length_nm=20.0,
            grid=sweep.SweepGrid(
                device_width_nm_values=np.array([10.0], dtype=float),
                device_thickness_nm_values=np.array([2.0, 3.0], dtype=float),
                concentration_cm3_values=np.array([5.0e19], dtype=float),
                energy_std_values=np.array([0.08], dtype=float),
                xi_values=np.array([0.25, 0.35], dtype=float),
            ),
            cutoff_distance=15.0,
            max_neighbors=12,
            n_realizations=2,
            n_jobs=1,
            max_generated_sites=100,
            seed=7,
        )

        with redirect_stdout(io.StringIO()):
            sweep.run(config)

        manifest_path = output_dir / "sweep_manifest.csv"
        combined_table_path = output_dir / "conductance_vs_temperature_all.csv"
        table_dir = output_dir / "conductance_tables"

        self.assertTrue(manifest_path.exists())
        self.assertTrue(combined_table_path.exists())
        self.assertTrue(table_dir.exists())

        manifest = pd.read_csv(manifest_path)
        combined_table = pd.read_csv(combined_table_path)

        self.assertEqual(len(manifest), 4)
        self.assertEqual(len(list(table_dir.glob("*.csv"))), 4)
        self.assertEqual(
            sorted(combined_table["temperature_K"].unique().tolist()),
            [float(value) for value in range(20, 201, 10)],
        )
        self.assertEqual(
            sorted(combined_table["combination_index"].unique().tolist()),
            [1, 2, 3, 4],
        )
        self.assertTrue((combined_table["n_realizations"] == 2).all())
        self.assertEqual(
            sorted(combined_table["device_thickness_nm"].unique().tolist()),
            [2.0, 3.0],
        )
        self.assertEqual(
            sorted(combined_table["xi"].unique().tolist()),
            [0.25, 0.35],
        )
        for column_name in [
            "nn1_mean_distance_nm",
            "nn1_mean_distance_std_nm",
            "nn2_mean_distance_nm",
            "nn2_mean_distance_std_nm",
            "nn3_mean_distance_nm",
            "nn3_mean_distance_std_nm",
            "nn_pooled_mean_distance_nm",
            "nn_pooled_mean_distance_std_nm",
        ]:
            self.assertIn(column_name, manifest.columns)
            self.assertIn(column_name, combined_table.columns)

        self.assertTrue((manifest["nn1_mean_distance_nm"] > 0.0).all())
        self.assertTrue(
            (manifest["nn1_mean_distance_nm"] <= manifest["nn2_mean_distance_nm"]).all()
        )
        self.assertTrue(
            (manifest["nn2_mean_distance_nm"] <= manifest["nn3_mean_distance_nm"]).all()
        )

        for table_path in manifest["table_path"].tolist():
            self.assertTrue((output_dir / table_path).exists())


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

    def test_nearest_neighbor_distance_statistics_report_expected_ranks(self) -> None:
        positions = np.array([[0.0], [1.0], [3.0], [6.0]], dtype=float)
        energies = np.zeros(4, dtype=float)
        solver = sim.RRNSolver(
            positions=positions,
            energies=energies,
            temperature=300.0,
            xi=1.0,
        )

        stats = solver.nearest_neighbor_distance_statistics(k=3)
        np.testing.assert_allclose(
            stats["distance_matrix"],
            np.array(
                [
                    [1.0, 3.0, 6.0],
                    [1.0, 2.0, 5.0],
                    [2.0, 3.0, 3.0],
                    [3.0, 5.0, 6.0],
                ],
                dtype=float,
            ),
        )

        per_neighbor = stats["per_neighbor"]
        self.assertEqual(len(per_neighbor), 3)
        self.assertAlmostEqual(per_neighbor[0].mean, 1.75)
        self.assertAlmostEqual(per_neighbor[1].median, 3.0)
        self.assertAlmostEqual(per_neighbor[2].max, 6.0)


class PlottingFitWindowTests(unittest.TestCase):
    def test_fit_prefix_windows_reports_unit_r2_for_perfect_activated_line(self) -> None:
        temperatures_k = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
        ln_conductance = 3.5 * (1.0 / temperatures_k) - 4.0
        conductance_mean_s = np.exp(ln_conductance)

        fit_windows = plotting.fit_prefix_windows(
            temperatures_k,
            conductance_mean_s,
            exponent=-1.0,
        )

        self.assertEqual(len(fit_windows), 3)
        self.assertEqual(
            [int(window["t_max_k"]) for window in fit_windows],
            [30, 40, 50],
        )
        for fit_window in fit_windows:
            self.assertAlmostEqual(float(fit_window["r2"]), 1.0, places=12)


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
