from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
PERSISTENT_PLOT_DIR = ROOT_DIR / "tmp" / "test_iv_simulator_plots"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import iv_simulator


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
            "contact_width": 3.0,
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
        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), len(EXPECTED_TEMPERATURES_K) * 3)

        temperatures = sorted({float(row["temperature_K"]) for row in rows})
        voltages = sorted({float(row["voltage_V"]) for row in rows})
        self.assertEqual(temperatures, EXPECTED_TEMPERATURES_K)
        self.assertEqual(voltages, [-0.5, 0.0, 0.5])

        for temperature_k in temperatures:
            temperature_rows = {
                float(row["voltage_V"]): row
                for row in rows
                if float(row["temperature_K"]) == temperature_k
            }
            non_conductive_count = int(
                temperature_rows[0.0]["non_conductive_realizations"]
            )
            negative_current = float(temperature_rows[-0.5]["current_mean_A"])
            zero_current = float(temperature_rows[0.0]["current_mean_A"])
            positive_current = float(temperature_rows[0.5]["current_mean_A"])

            for row in temperature_rows.values():
                self.assertEqual(int(row["n_realizations"]), 1)
                self.assertEqual(
                    int(row["non_conductive_realizations"]),
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


if __name__ == "__main__":
    unittest.main()
