from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = ROOT_DIR / "mcp" / "server.py"


def load_mcp_server_module():
    spec = importlib.util.spec_from_file_location("vrh_mcp_server", MCP_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {MCP_SERVER_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OneDMcpWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_mcp_server_module()
        self.original_simulate = (
            self.module.simulate_default_conductivity_for_temperatures
        )
        self.received_temperatures = None

        def fake_simulate(temperatures_k):
            self.received_temperatures = temperatures_k
            if isinstance(temperatures_k, str):
                temperature_values = [100.0, 105.0, 110.0]
            elif isinstance(temperatures_k, list):
                temperature_values = [float(value) for value in temperatures_k]
            else:
                temperature_values = [float(temperatures_k)]
            return {
                "temperature_k": temperature_values,
                "conductivity": [1.0 for _ in temperature_values],
                "ln(conductivity)": [0.0 for _ in temperature_values],
            }

        self.module.simulate_default_conductivity_for_temperatures = fake_simulate

    def tearDown(self) -> None:
        self.module.simulate_default_conductivity_for_temperatures = (
            self.original_simulate
        )

    def assert_response_shape(self, response: dict[str, object], expected_len: int) -> None:
        self.assertEqual(
            set(response),
            {"temperature_k", "conductivity", "ln(conductivity)"},
        )
        self.assertEqual(len(response["temperature_k"]), expected_len)
        self.assertEqual(len(response["conductivity"]), expected_len)
        self.assertEqual(len(response["ln(conductivity)"]), expected_len)

    def test_payload_accepts_single_temperature(self) -> None:
        response = self.module.simulate_1d_conductivity_payload(
            {"temperatures_k": 225}
        )

        self.assert_response_shape(response, expected_len=1)
        self.assertEqual(response["temperature_k"], [225.0])
        self.assertEqual(self.received_temperatures, 225)

    def test_payload_accepts_temperature_list(self) -> None:
        response = self.module.simulate_1d_conductivity_payload(
            {"temperatures_k": [100, 150, 200]}
        )

        self.assert_response_shape(response, expected_len=3)
        self.assertEqual(response["temperature_k"], [100.0, 150.0, 200.0])
        self.assertEqual(self.received_temperatures, [100, 150, 200])

    def test_payload_accepts_temperature_range_string(self) -> None:
        response = self.module.simulate_1d_conductivity_payload(
            {"temperatures_k": "100:110:5"}
        )

        self.assert_response_shape(response, expected_len=3)
        self.assertEqual(response["temperature_k"], [100.0, 105.0, 110.0])
        self.assertEqual(self.received_temperatures, "100:110:5")

    def test_payload_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValueError):
            self.module.simulate_1d_conductivity_payload(
                {"temperatures_k": 250, "extra": 1}
            )


if __name__ == "__main__":
    unittest.main()
