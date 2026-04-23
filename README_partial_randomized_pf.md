# Partial Randomized PF

This is the first-stage research implementation for the simplified partially randomized product-formula comparison described in the project discussion.

## Scope

- Use the Pauli LCU form of the JW Hamiltonian.
- Sort Pauli terms by descending `|c_l|`.
- Define `H_D` from the top `L_D` terms and `H_R` from the tail.
- Compute `lambda_R(L_D) = sum_{l > L_D} |c_l|`.
- Fit `C_gs^(p)(L_D)` from perturbative ground-state error scaling on `H_D`.
- Treat `C_gs` as a surrogate for the deterministic part only. It is not a rigorous error constant for the full partially randomized PF on the full Hamiltonian.

## Cost Model

For a fixed PF label `p`, deterministic cutoff `L_D`, total error budget `eps`, `q = eps_qpe / eps`, and `0 < q < 1`,

- `eps_qpe = q * eps`
- `eps_trot = eps * sqrt(1 - q^2)`

The deterministic contribution is

- `G_det = A_p(L_D) * (C_gs^(p)(L_D))^(1/p) / (eps_qpe * eps_trot^(1/p))`

The randomized prefactor is no longer an external constant. It is derived from `kappa`:

- `B(kappa) = B0 * kappa * exp(2 / kappa)`
- `B0 = (280/9) * G_rand * gamma * (0.1 * pi)^2`

Defaults:

- `randomized_method = "qdrift"` gives `gamma = 1`
- `randomized_method = "rte"` gives `gamma = 2`
- `G_rand = 1.0`

The randomized contribution is

- `G_rand = B(kappa) * lambda_R(L_D)^2 / eps_qpe^2`

The total cost is

- `G_total = G_det + G_rand`

## Modes

- `kappa_mode = optimize`
  Main analysis mode. For each `(p, L_D)`, the code numerically optimizes both `q` and `kappa`.
- `kappa_mode = fixed`
  Reference mode. Use a fixed `kappa`, with `kappa = 2` as the default baseline.
- `kappa_mode = sweep`
  Sensitivity analysis mode. Evaluate a kappa grid, optimize `q` at each grid point, and store the sweep table for CSV export.

If `lambda_R == 0`, the randomized side is turned off explicitly:

- `kappa_opt = None`
- `B_opt = 0`
- `G_rand = 0`

## Reused Code

- Hamiltonian generation: `src/trotterlib/chemistry_hamiltonian.py`
- PF definitions: `src/trotterlib/product_formula.py`
- PF step ordering: `src/trotterlib/pf_decomposition.py`
- Perturbative state-evolution path: `src/trotterlib/qiskit_time_evolution_ungrouped.py`
- Fixed-order log-log coefficient fit: `src/trotterlib/analysis_utils.py`

## Outputs

Each candidate stores:

- `c_gs`, `fit_coeff_fixed_order`
- `fit_slope`, `fit_coeff`
- `q_opt`, `eps_qpe_opt`, `eps_trot_opt`
- `kappa_opt`, `b_opt`
- `boundary_hit_q`, `boundary_hit_kappa`
- `randomized_method`, `g_rand_input`, `b0`
- `G_det`, `G_rand`, `G_total`

Here `fit_coeff_fixed_order` is the coefficient obtained with the log-log slope fixed to the PF order `p`, while `fit_slope` and `fit_coeff` come from the free log-log fit used only as diagnostics.

If `kappa_mode = sweep`, each candidate also stores the full `kappa_sweep` table.

Boundary diagnostics:

- `boundary_hit_kappa = True` means the best `kappa` is close to the lower or upper search boundary.
- If `kappa` sticks to the upper boundary, the randomized tail may already be nearly irrelevant and the result may be close to the deterministic limit.

## C_gs Cache

- `C_gs^(p)(L_D)` fits are cached automatically in `artifacts/partial_randomized_pf/cgs_fit_cache.json`.
- The cache stores the fitted values and perturbation-fit diagnostics, not the explicit contents of `H_D`.
- Cache keys are built from the full sorted Hamiltonian hash, `PF`, `L_D`, the perturbation-fit time grid, and the surrogate-definition version.

## Run

Create an environment with the existing project requirements, then run for a small system first.

```bash
python3 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
```

Main mode with kappa optimization:

```bash
python scripts/run_partial_randomized_pf.py \
  --molecule-type 3 \
  --epsilon-total 1e-3 \
  --pf-labels 2nd,4th(new_2),8th(Morales) \
  --kappa-mode optimize
```

Reference mode with `kappa = 2`:

```bash
python scripts/run_partial_randomized_pf.py \
  --molecule-type 3 \
  --epsilon-total 1e-3 \
  --kappa-mode fixed \
  --kappa-value 2
```

Sensitivity analysis with CSV export:

```bash
python scripts/run_partial_randomized_pf.py \
  --molecule-type 3 \
  --epsilon-total 1e-3 \
  --kappa-mode sweep \
  --kappa-grid 1,2,4,8,16,32 \
  --export-kappa-sweep-csv
```

The script prints the best candidate and writes the full JSON result to `artifacts/partial_randomized_pf/`. In sweep mode it can also write a CSV table for the kappa sensitivity scan.

## Backward Compatibility

The old fixed-`B` path is still available in the library through `optimize_error_budget(...)`, and the CLI keeps a deprecated `--random-prefactor` option for legacy comparisons. The main path now uses `B(kappa)` instead of a user-supplied external constant.
