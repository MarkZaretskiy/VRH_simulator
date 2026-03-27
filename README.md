# VRH Simulator

This repository contains a small Random Resistor Network (RRN) simulator for
temperature-dependent hopping transport, together with a 1D experiment script
used to reproduce the type of analysis shown in the paper
`2601.01243v1.pdf` https://arxiv.org/pdf/2601.01243.

The code focuses on:

- solving hopping transport on a resistor network
- scanning temperature and localization length
- fitting the low-temperature VRH regime
- detecting the temperature where the fit stops being VRH-like
- generating CSV outputs and plots


## Files

### `code/sim.py`

`sim.py` contains the network solver.

Main pieces:

- `RRNSolver`
  Builds the hopping conductance matrix, assembles the graph Laplacian,
  applies Dirichlet boundary conditions, solves for node voltages, and computes
  the effective conductance.

- `RRNResult`
  Bundles the voltages, total current, effective conductance/resistance, and
  the final matrices.

- `make_random_sites(...)`
  Helper for generating random site clouds.

- `contact_nodes_from_x(...)`
  Helper for defining left/right contacts from the x coordinate.

Physics implemented in `RRNSolver`:

- Pairwise hopping conductance:
  `G_ij = (G0 / T) * exp(-2 r_ij / xi) * exp(-eps_ij / (kB T))`

- Energy cost:
  `eps_ij = 0.5 * (|eps_i| + |eps_j| + |eps_i - eps_j|)`

- Kirchhoff solve:
  the reduced Laplacian is solved with fixed left/right contact voltages

Numerical stability in `sim.py`:

- conductances are assembled in log-space first
- the largest active `log(G_ij)` is subtracted before exponentiation
- the network is solved on the globally scaled matrix
- current and effective conductance are rescaled back afterwards

This keeps node voltages unchanged while making the solve much more stable.


### `code/1d_experiment.py`

`1d_experiment.py` builds a regular 1D chain and repeats the simulation over:

- temperature
- localization length `xi`
- multiple Monte Carlo realizations of the disorder

What it does:

1. Build a regular 1D chain with spacing `delta_min_nm`
2. Assign random site energies in the window `[-W_E/2, W_E/2]`
3. Solve the network for every temperature
4. Average the conductivity curves over realizations
5. Fit `ln(sigma)` vs `T^(-1/2)` on `[T_min, T_x]`
6. Compute `epsilon_VRH(T_x)` for each scan window
7. Detect the first `T_x` where the fit exceeds the threshold
8. Write CSV files and optional plots

Important helpers:

- `build_config(...)`
- `simulate_averaged_curves(...)`
- `fit_vrh_transition(...)`
- `plot_curves(...)`
- `plot_vrh_fit_map(...)`

Numerical guardrails in `1d_experiment.py`:

- averaged curves are stabilized if a tiny non-physical negative value appears
  from the linear solve near numerical zero
- the VRH fit is only reported when there is a threshold-compliant VRH window
- the `epsilon_VRH` heatmap is clipped to `[0, 15]` in plot units


## Environment

This repo uses `pixi`.

Basic environment usage:

```bash
pixi run python code/sim.py
```


## How To Run

### 1. Default 1D experiment

```bash
pixi run python code/1d_experiment.py --plot
```

This writes:

- `code/1d_experiment_results.csv`
- `code/1d_experiment_results_vrh_fit_summary.csv`
- `code/1d_experiment_results_vrh_fit_scan.csv`
- plots in the default output directory


### 2. Dense xi sweep

You can pass `xi` as a range:

```bash
pixi run python code/1d_experiment.py \
  --xi_values_nm 0.05:0.30:0.01 \
  --n_realizations 100 \
  --plot
```

Range syntax means:

- `start:stop:step`
- here: `0.05, 0.06, ..., 0.30`


### 3. Run different energy spans

Single energy span:

```bash
pixi run python code/1d_experiment.py --energy_span_ev 0.3 --plot
```

Multiple energy spans in one command (takes ~30min):

```bash
pixi run python code/1d_experiment.py \
  --xi_values_nm 0.05:0.30:0.01 \
  --n_realizations 100 \
  --plot \
  --energy_span_ev 0.4,0.3,0.2
```

If you include spaces, quote the value:

```bash
pixi run python code/1d_experiment.py \
  --energy_span_ev "0.4, 0.3, 0.2" \
  --plot
```

For multi-span runs, outputs are separated automatically, for example:

- `..._we_0p4eV.csv`
- `..._we_0p3eV.csv`
- `..._we_0p2eV.csv`

and plots go into per-span directories like:

- `we_0p4eV/`
- `we_0p3eV/`
- `we_0p2eV/`


### 4. Small quick test run

Useful when checking code changes:

```bash
pixi run python code/1d_experiment.py \
  --n_sites 20 \
  --n_realizations 2 \
  --xi_values_nm 0.1,0.15,0.3 \
  --t_min_k 100 \
  --t_max_k 200 \
  --t_step_k 50 \
  --plot
```


## Plots

The script produces two main plot types.

### Conductivity curves

`sigma_vs_inverse_temperature_and_inverse_sqrt_temperature.png`

This contains:

- `sigma` vs `1/T`
- `sigma` vs `1/sqrt(T)`

Only three highlighted curves are shown in the subplot styling:

- `xi = 0.1` in blue
- `xi = 0.15` in red
- `xi = 0.3` in green


### VRH fit map

`xi_vs_tx_colored_by_epsilon_vrh.png`

This is a heatmap of:

- x-axis: `T_x`
- y-axis: `xi`
- color: `epsilon_VRH(T_x)`

Plot-specific notes:

- the color map is clipped to `[0, 15]` in units of `10^-3`
- the threshold line is shown on the colorbar
- the `T_c` curve is not overlaid on the heatmap


## Outputs

### `*_results.csv`

Contains:

- `xi_nm`
- `temperature_K`
- mean/std conductance
- mean/std conductivity


### `*_vrh_fit_summary.csv`

Contains one row per `xi` with:

- the selected fit window
- whether the fit is threshold-compliant
- transition temperature if one is detected
- `epsilon_VRH(T_max)`


### `*_vrh_fit_scan.csv`

Contains the full scan over `T_x` for each `xi`:

- `epsilon_VRH(T_x)`
- fit slope/intercept
- whether each scanned point is within threshold


## Notes

- Very small conductivity values are expected in hopping transport.
- Small `sigma` by itself is not the main numerical problem.
- The hard part is the huge dynamic range in the conductance matrix.
- That is why `sim.py` now assembles conductances in log-space first.

If you want the figure to look more like the paper, the most important thing is
to use a dense `xi` sweep rather than the default sparse set.
