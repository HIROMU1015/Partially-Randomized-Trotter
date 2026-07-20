# Partial Randomized PF Artifacts

> [!CAUTION]
> The committed screening output
> `screening_results/df_screening_cost_minimization_eps_1.000e-04.json` is
> **quarantined and must not be used for research claims**. It was generated
> from Cgs tables that were later removed in commit `98f960c` after a
> ground-state mismatch was found. The screening output has not been
> regenerated from corrected inputs. See
> [`../../VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md) and
> [`../validation_manifest.json`](../validation_manifest.json).

This directory is organized by artifact role.

- `df_cgs_cost_table.json`: expected aggregate lightweight Cgs/cost input table used by screening; **currently missing**.
- `df_cgs_cost_tables/`: expected split lightweight Cgs/cost input tables; **currently missing**.
- `df_cgs_fit_cache.json`: local Cgs fit cache used by the DF Cgs pipeline.
- `df_ground_state_cache/`: local DF ground-state cache.
- `df_cgs/raw/`: raw detailed DF Cgs batch outputs.
- `df_cgs/diagnostics/`: Cgs slope checks, profiles, and scaling diagnostics.
- `screening_results/`: reduced-model cost minimization outputs; the currently committed JSON is quarantined.
- `legacy_pauli/`: older Pauli-based or early simplified partial-randomized outputs.
- `logs/`: tmux/runtime logs.

Files directly under this directory should be limited to canonical tables,
active partial outputs, and caches that current scripts read by default. Active
tmux jobs may temporarily write partial JSON files here until they finish.

## Current repository state

| Artifact | State | Interpretation |
|---|---|---|
| `df_cgs_cost_table.json` | Missing | Corrected screening input has not been published. |
| `df_cgs_cost_tables/` | Missing | Per-molecule/PF Cgs evidence has not been published. |
| `screening_results/df_screening_cost_minimization_eps_1.000e-04.json` | Quarantined | Internally inspectable, but its removed inputs prevent validation or reproduction. |
| Raw Cgs fits, diagnostics, and logs | Not committed | No repository-only audit of the underlying fits is possible. |

## Git policy

- Keep this README, corrected `df_cgs_cost_table.json`, corrected
  `df_cgs_cost_tables/`, and a screening result regenerated from those exact
  inputs in the same commit.
- Each canonical result must record the producing Git commit, command,
  dependency/environment information, input hashes, and run timestamp.
- Do not keep raw Cgs outputs, diagnostics, ground-state caches, fit caches, runtime logs, active partial JSON files, or legacy Pauli outputs in git. These are local/recomputable artifacts.
- If raw outputs remain outside Git, publish a stable archive reference and
  checksums so the compact committed result can be audited.
