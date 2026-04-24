# VRH Simulator

This repository contains a small Random Resistor Network simulator for
temperature-dependent hopping transport.

The two main entry points are:

- `code/sim.py`: low-level network solver and helper utilities
- `code/iv_simulator.py`: temperature sweep and linear I-V workflow with CSV and plots
- `code/sweep.py`: grid sweep over device parameters with conductance-vs-temperature tables


## Environment

This repo uses `pixi`.

Basic check:

```bash
pixi run python code/sim.py
```


## `code/sim.py`

`sim.py` contains the core solver.

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

- Pairwise hopping conductance:
  `G_ij = (G0 / T) * exp(-2 r_ij / xi) * exp(-eps_ij / (kB T))`
- Energy cost:
  `eps_ij = 0.5 * (|eps_i| + |eps_j| + |eps_i - eps_j|)`
- Kirchhoff solve:
  the reduced Laplacian is solved with fixed left and right contact voltages

Numerical stability:

- conductances are assembled in log-space first
- the largest active `log(G_ij)` is subtracted before exponentiation
- the solve is performed on the globally scaled matrix
- current and effective conductance are rescaled afterwards

Run the built-in example:

```bash
pixi run python code/sim.py
```


## `code/iv_simulator.py`

`iv_simulator.py` builds a disorder realization or a box-populated network,
solves the conductance for each temperature, and converts the result into a
linear I-V sweep.

What it does:

1. Builds a network from explicit `positions` and `energies`, from a box
   concentration, or from a random site cloud.
2. Selects left and right contact nodes automatically or uses explicit contact
   indices.
3. Solves the effective conductance for each temperature.
4. Averages over `n_realizations`.
5. Writes a CSV table and optional plots.

### CLI Run

Example run with direct CLI parameters:

```bash
pixi run python code/iv_simulator.py \
  --concentration_cm3 1.8e20 \
  --t_min_k 10 \
  --t_max_k 200 \
  --t_step_k 10 \
  --v_min -0.5 \
  --v_max 0.5 \
  --v_step 0.1 \
  --n_realizations 4 \
  --n_jobs 4 \
  --plot true \
  --output tmp/iv_run.csv
```

### Config Run

`code/iv_simulator.py` can load a top-level JSON or YAML mapping.

Example `tmp/iv_config.yaml`:

```yaml
concentration_cm3: 1.8e20
temperatures_k: [10, 20, 30, 50, 80, 100, 150, 200]
xi: 0.35
G0: 1.0
kB: 8.617333262145e-5
cutoff_distance: 20.0
max_neighbors: 100
min_conductance: 0.0
v_min: -0.5
v_max: 0.5
v_step: 0.1
n_realizations: 4
n_jobs: 4
device_length_nm: 120.0
device_width_nm: 30.0
device_thickness_nm: 3.0
max_generated_sites: 3000
energy_std: 0.4
seed: 42
plot: true
plot_output_dir: tmp/iv_plots
show_plots: false
output: tmp/iv_config_run.csv
```

Run it:

```bash
pixi run python code/iv_simulator.py --config tmp/iv_config.yaml
```

When `--config` is used, all simulation parameters are read from the file.

### Output

The main CSV output contains:

- `temperature_K`
- `voltage_V`
- `current_mean_A`
- `current_std_A`
- `conductance_mean_S`
- `conductance_std_S`
- `n_realizations`
- `non_conductive_realizations`

If `plot: true`, the script also writes plots for:

- I-V curves vs temperature
- conductance vs temperature


## `code/sweep.py`

`code/sweep.py` runs a grid sweep for a box-defined device and saves
conductance-vs-temperature tables for each parameter combination.

It uses the same device baseline as `IVSimulatorDeviceTests`:

- `device_length_nm = 100`
- `device_width_nm = 50`
- temperatures are fixed to `20..200 K` with step `10`

The swept grid dimensions are:

- `device_thickness_nm`
- `concentration_cm3`
- `energy_std`
- `xi`

The sweep grid is hardcoded directly in [code/sweep.py](/home/mark/Desktop/Projects/ARIA/vrh_simulator/code/sweep.py).
Edit these constants when you want to change the production run:

- `SWEEP_DEVICE_THICKNESS_GRID_NM`
- `SWEEP_CONCENTRATION_GRID_CM3`
- `SWEEP_ENERGY_STD_GRID_EV`
- `SWEEP_XI_GRID_NM`

Run it:

```bash
pixi run sweep
```

Outputs:

- `sweep_manifest.csv`: one row per parameter combination
- `conductance_vs_temperature_all.csv`: combined long table for all combinations
- `conductance_tables/*.csv`: one conductance-vs-temperature table per combination


## Notes

- Very small conductance values are expected in hopping transport.
- The main numerical difficulty is the dynamic range of the conductance matrix.
- That is why `sim.py` assembles conductances in log-space before solving.
