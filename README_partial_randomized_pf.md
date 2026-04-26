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

## DF-native C_gs,D GPU path

DF representation を使う本流では、Pauli 項へ戻さず DF fragment のまま
`H_D/H_R` を分割する。`src/trotterlib/df_partial_randomized_pf.py` がそのための
DF-native な `C_gs,D` fit path を提供する。

- `rank_df_fragments(...)`: DF fragment を固定の重み規則で降順に並べる。
- `split_df_hamiltonian_by_ld(...)`: `L_D` 個の DF fragment を deterministic 側に置く。
- `fit_df_cgs_with_perturbation(...)`: DF block circuit builder で `H_D` の Trotter 回路を作り、GPU statevector で perturbation error を計算して `C_gs,D` を fit する。
- `get_or_compute_cached_df_cgs_fit(...)`: DF 用 cache key で GPU 実行結果を再利用する。
- `get_or_compute_cached_df_ground_state(...)`: `H_D` と physical sector と solver 条件ごとに基底状態・基底エネルギーを `artifacts/partial_randomized_pf/df_ground_state_cache/*.npz` に保存して再利用する。
- `df_deterministic_step_rz_cost(...)`: DF project と同じ U/D 分解ベースで `total_ref_rz_depth` を数え、`C_gs,D` の cache record の `metadata.df_step_cost` に保存する。

GPU runner は `src/trotterlib/df_gpu_statevector.py` にあり、
`qiskit-aer-gpu` の `AerSimulator(method="statevector", device="GPU")` を使う。
default では symbolic time parameter `t` を持つ回路を一度だけ GPU backend 向けに
transpile し、各 `t_values` では parameter bind だけを行って実行する。
`gpu_ids=("0", "1", ...)` のように複数 GPU を渡すと、`t_values` の各点を
1 GPU ずつ round-robin に割り当て、別プロセスで並列実行する。
旧来のように各 `t` で個別に circuit build/transpile したい場合は
`use_parameterized_template=False` を指定するか、
`scripts/run_h5_df_cgs_gpu_slopes.py --no-parameterized-template` を使う。
基底状態 cache を使わず毎回 solve したい場合は `use_ground_state_cache=False`
または `scripts/run_h5_df_cgs_gpu_slopes.py --no-ground-state-cache` を使う。
GPU 環境では以下を使う。

```bash
pip install -r requirements-gpu.txt
```

この path で得る `C_gs,D` は、DF `H_D` に対する deterministic surrogate であり、
full partial-randomized scheme 全体の厳密な誤差係数ではない。

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

## Matrix-Free Ground State Solver

The `H_D` ground state used in the `C_gs` surrogate is computed with a matrix-free
`LinearOperator` by default. When `numba` is installed, Pauli strings are compiled
to bit masks and the statevector index loop is parallelized with OpenMP through
`numba.prange`.

- `--matrix-free-backend auto` uses the numba backend when available and falls back to the pure-Python tensor backend otherwise.
- `--matrix-free-backend numba` requires numba and fails if it is unavailable.
- By default, the numba backend detects the CPUs available to the process and uses that thread count.
- `--matrix-free-threads N` sets the numba thread count for the matrix-free matvec; `0` means auto-detect.
- `--ground-state-ncv N` passes `ncv` to ARPACK's `eigsh`; smaller values reduce memory pressure, while larger values may improve convergence.

This avoids materializing the sparse Hamiltonian matrix, but it still uses full
statevectors. It does not remove the exponential memory scaling of exact full-space
ground-state calculations.

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
  --kappa-mode optimize \
  --matrix-free-backend auto \
  --matrix-free-threads 4
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
