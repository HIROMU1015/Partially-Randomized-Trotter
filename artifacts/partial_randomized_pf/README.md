# Partial Randomized PF Artifacts

This directory is organized by artifact role.

- `df_cgs_cost_table.json`: aggregate lightweight Cgs/cost input table used by screening.
- `df_cgs_cost_tables/`: split lightweight Cgs/cost input tables by molecule family, system size, and PF label.
- `df_cgs_fit_cache.json`: local Cgs fit cache used by the DF Cgs pipeline.
- `df_ground_state_cache/`: local DF ground-state cache.
- `df_cgs/raw/`: raw detailed DF Cgs batch outputs.
- `df_cgs/diagnostics/`: Cgs slope checks, profiles, and scaling diagnostics.
- `screening_results/`: reduced-model cost minimization outputs.
- `legacy_pauli/`: older Pauli-based or early simplified partial-randomized outputs.
- `logs/`: tmux/runtime logs.

Files directly under this directory should be limited to canonical tables, active partial outputs, and caches that current scripts read by default. Active tmux jobs may temporarily write partial JSON files here until they finish.

Git policy:

- Keep this README, `df_cgs_cost_table.json`, `df_cgs_cost_tables/`, and the canonical `epsilon_total=1e-4` screening result in git.
- Do not keep raw Cgs outputs, diagnostics, ground-state caches, fit caches, runtime logs, active partial JSON files, or legacy Pauli outputs in git. These are local/recomputable artifacts.
