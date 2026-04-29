from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

original_sys_path = sys.path.copy()
try:
    sys.path = [
        path_text
        for path_text in original_sys_path
        if Path(path_text or ".").resolve() != PROJECT_ROOT
    ]
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None
finally:
    sys.path = original_sys_path

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
simulate_default_conductivity_for_temperatures = import_module(
    "1d_simulator"
).simulate_default_conductivity_for_temperatures


server = FastMCP(
    name="simulator",
    instructions=(
        "MCP server for the simulator of semiconductor device. "
        "Call simulate_conductivity with a strict JSON object containing "
        "only temperatures_k. temperatures_k may be a range string like "
        "'100:400:5', a numeric list like [100, 150, 200], or a single number "
        "like 225. The response contains temperature_k, conductivity, and "
        "ln(conductivity). Conductivity is the typical conductivity "
        "exp(mean(ln(sigma))) in S/cm from the default JSON simulator config "
        "with only temperatures_k overridden."
    ),
) if FastMCP is not None else None


def get_temperatures_from_request(request: Dict[str, Any]) -> Any:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    expected_keys = {"temperatures_k"}
    request_keys = set(request)
    if request_keys != expected_keys:
        missing = sorted(expected_keys - request_keys)
        extra = sorted(request_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"unexpected keys: {extra}")
        raise ValueError("; ".join(details))

    return request["temperatures_k"]


def simulate_1d_conductivity_payload(request: Dict[str, Any]) -> Dict[str, Any]:
    temperatures_k = get_temperatures_from_request(request)
    return simulate_default_conductivity_for_temperatures(temperatures_k)


def simulate_conductivity(request: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate conductivity of the device for requested temperatures.

    Request must be a strict JSON object with exactly one field:
    - temperatures_k: range string ("100:400:5"), numeric list
      ([100, 150, 200]), or a single number (225).

    Returns:
    - temperature_k: list of temperatures in K.
    - conductivity: typical conductivity exp(mean(ln(sigma))) in S/cm.
    - ln(conductivity): mean ln(sigma), consistent with conductivity.
    """
    return simulate_1d_conductivity_payload(request)


if server is not None:
    simulate_conductivity = server.tool()(simulate_conductivity)


def main() -> None:
    if server is None:
        raise SystemExit(
            "Missing dependency: install the `mcp` package to run this server."
        )
    server.run()


if __name__ == "__main__":
    main()
