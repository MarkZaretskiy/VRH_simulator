# VRH Simulator

Random Resistor Network tooling for temperature-dependent hopping transport.

The maintained entry points are:

- `code/sim.py`: low-level RRN solver.
- `code/1d_simulator.py`: regular 1D RRN conductivity sweep using the model from `literature/2601.01243v1.pdf`.


## Environment

This repo uses `pixi`.

- [Pixi](https://pixi.sh)

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

Basic checks:

```bash
pixi run python code/sim.py
pixi run python code/1d_simulator.py
```


## `code/sim.py`

`sim.py` contains the core network solver.

Main pieces:

- `RRNSolver`
  Builds the hopping conductance matrix, assembles the graph Laplacian,
  applies Dirichlet boundary conditions, solves node voltages, and computes
  effective conductance.
- `RRNResult`
  Bundles voltages, total current, effective conductance/resistance, and the
  final sparse matrices.
- `make_random_sites(...)`
  Generates a random cloud of sites and on-site energies.
- `contact_nodes_from_x(...)`
  Defines left and right contacts from the x coordinate.

Physics implemented in `RRNSolver`:

```text
G_ij = (G0 / T) * exp(-2 r_ij / xi) * exp(-eps_ij / (kB T))
eps_ij = 0.5 * (|eps_i| + |eps_j| + |eps_i - eps_j|)
```

Units used by the current code:

- distances and `xi`: same spatial units, typically nm;
- energies and `eps_ij`: eV;
- `kB = 8.617333262145e-5 eV/K`.

Numerical stability:

- conductances are assembled in log-space first;
- the largest active `log(G_ij)` is subtracted before exponentiation;
- the solve is performed on the globally scaled matrix;
- current and effective conductance are rescaled afterwards.

Run the built-in example:

```bash
pixi run python code/sim.py
```


## `code/1d_simulator.py`

`1d_simulator.py` builds a regular 1D chain, assigns random site energies from
a uniform distribution, solves the RRN over requested temperatures, and returns
`Conductivity(T)`.

The default config is:

```text
configs/1d_simulator_config.json
```

Run with the default config:

```bash
pixi run python code/1d_simulator.py
```

Run with an explicit config:

```bash
pixi run python code/1d_simulator.py configs/1d_simulator_config.json
```

The config supports `temperatures_k` as:

```json
"100:400:5"
```

or:

```json
[100, 150, 200]
```

or a single value:

```json
250
```

The simulator averages over `n_realizations`. Before averaging, it can remove
outliers per temperature using an IQR filter:

```json
"outlier_iqr_factor": 1.5
```

Set this field to `null` to disable outlier removal.

The output CSV path is configured by the `output` field. The current default is:

```text
code/1d_simulator_conductivity.csv
```

Important CSV fields:

- `temperature_K`
- `mean_conductivity_S_per_cm`: arithmetic mean of `sigma`
- `typical_conductivity_S_per_cm`: `exp(mean(ln(sigma)))`
- `mean_ln_conductivity`: mean `ln(sigma)`, useful for VRH fits
- `std_ln_conductivity`: spread in log-conductivity
- `n_realizations_used`: realizations remaining after outlier filtering
- `inverse_temperature_1_per_K`
- `inverse_sqrt_temperature_1_per_sqrt_K`

For 1D VRH analysis, prefer `mean_ln_conductivity` vs
`inverse_sqrt_temperature_1_per_sqrt_K`.


## Test Plots

The 1D simulator test creates diagnostic plots:

```bash
pixi run python -m unittest tests.test_1d_simulator
```

Outputs:

- `tests/images/1d_simulator/conductivity_summary.png`
  Contains `sigma(T)`, `mean ln(sigma)` vs `T^(-1/2)`, and the integral VRH fit error vs `Tx`.
- `tests/images/1d_simulator/chain_geometry.png`
  Shows every disorder realization as a 1D chain colored by site energy, with outlier information in subplot titles.


## MCP Wrapper

The MCP wrapper is in:

```text
mcp/server.py
```

Visible tool:

```text
simulate_conductivity
```

It accepts a strict JSON object with only `temperatures_k` and calls the
default 1D simulator config internally.

Example request:

```json
{"temperatures_k": "100:400:5"}
```

Response fields:

- `temperature_k`
- `conductivity`
- `ln(conductivity)`


## Notes

- Very small conductance values are expected in hopping transport.
- Low-temperature 1D chains can have large realization-to-realization spread.
- `typical_conductivity_S_per_cm = exp(mean(ln(sigma)))` is often more stable
  than the arithmetic mean for disorder-dominated low-temperature regimes.
