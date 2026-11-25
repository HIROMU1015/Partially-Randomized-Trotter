# H4 PR-DF requested maximum-width continuation (2026-11-25)

Run: `search-78ee25e2f10dcaccee5c`. This is an operational/scientific preparation extension, not a final comparison of optimized total RZ.

## Scope and mapper

The existing H4 line geometry (1 Å), STO-3G, saved DF10 with tolerance-only rank policy (`1e-8`), saved state and Hamiltonian, `M=12`, S2, and `K=2` remain fixed. The requested width is an exact multiple of the stored binary64 `h=0.1` for ratios 1–10; the kappa grid is `1, 2, 4, 8, 16, 32, 64`. The old mapper had supported nine families at width `h` for `L_D=1..9`. The explicit `A_PR_DF_REQUESTED_MAX_WIDTH_V1` adapter adds 81 wider families for `L_D=1..9` and ten `L_D=0` families, giving 637 newly reachable logical configurations. The one-body matrix remains in the deterministic head at `L_D=0`; the randomized tail is all DF blocks, not the full-H pure-RTE route.

For each round, the mapper computes `q=max(1,ceil(T/requested_width))` on exact rationals. Both exact mathematical `T/q` and its binary64 kernel representation are recorded. The existing finite-RTE signal and Q3 statistical kernels remain pinned. The price rule, actual-step registry and native calibration request metadata recognize this explicit adapter without changing old fixed-grid validation.

New `L_D=0` tail descriptor: [DF_TAIL_DESCRIPTOR_LD0.json](DF_TAIL_DESCRIPTOR_LD0.json). Its paired STARTED record preserves the scientific SOURCE and input provenance. Mapper SOURCE: `06c50ce3628d7ce8b98c0cd04a54eb0c749557c8`. Controller SOURCE: `5a3f582f6ffc2120af2e39894c53ef5f19cf27c5`.

## Transport and recovery

The original signal provider checkout is never replaced with the controller checkout. The selected private caller and all its relative metadata dependencies are verified against their pinned SOURCE. A common Git bundle is reused and each independent signal request explicitly carries its own frozen input commit/path/hashes. In a two-request technical check, the common bundle was about 199.7 MB, while the candidate-specific bundles were about 1.1 MB each; this is not a network speed measurement.

Four returned, zero-round, pre-numerical Git input failures (`prep-dc0899c7a0d81fa65e68f2ff`, `prep-b244e2c3ca209a3e66fadfe5`, `prep-9b1588086a0ec7246cbba0ea`, `prep-f6dfd282043e07862535e900`) are retained. The two additional failures came from restored pre-dispatch readiness records; such records are now retired before queue restoration only if they have no attempt. The technical continuation selection limits each to one successor with the same scientific input and added missing locator. Other scientific STOPs and owned attempts are unchanged. Unstarted, cancelled reservations can be reissued with complete locator transport, without a previous scientific attempt.

The ordinary Resume checks and existing queue ownership/acceptance logic remain in use. Price inputs for the new adapter use the original native compiler and sampler; the new metadata stage does not perform compilation. Price uncertainty must not block independent Q3; approximate cost is a shortlist criterion and not a qualified full-wrapper cost expectation.

PBS has a 96-CPU allocation policy. Following observed per-round signals of at most about 251 MiB RSS, the selected signal profile reserves 13 CPUs for the 13 independent rounds and 6 GiB per job, including process overhead. This is a scheduling reservation, not a scientific call limit.

The exploratory maximum-width path now admits a candidate's signals to Q3 without waiting for a new price model. Existing price attempts retain their ownership and are reconciled; the ordinary MSE-feasible shortlist price preparation resolves missing prices for the candidates that require ranking. Explicitly selected signal predecessor recovery is scheduled before new grid candidates, with the existing ownership and failure checks still in force. WSL remains eligible for Q3, but the currently configured placement excludes it from PF/PR-DF heavy price compilation because its 15 GiB memory and OS OOM logs demonstrated pressure from concurrent compiler workers. No failed scientific attempt is silently resubmitted. Server1 and server2 remain administratively disabled.

The focused mock and mapper tests passed (42 tests, three subtests). A broader technical run passed 148 tests and seven subtests; one old fixed-AST comparison failed in the unchanged `rpe_method_optimizer.py`/test pair. This failure also applies to the predecessor SOURCE and is not a pass of that particular invariant test. After the numerical-input deduplication implementation, 149 tests and seven subtests passed with that unchanged comparison deselected. The extended dedicated alias test also passed with one assignment and one adoption over repeated submission and Resume.

## Live continuation evidence (2026-11-25 JST)

The ordinary run ledger contains 100 PR-DF families at `READY_METADATA` and 637 requests for the new mapper (91 families × seven kappas). The four specifically selected, pre-numerical failures have the following single successors, all `ACCEPTED` and adopted:

| Original | Successor | Execution host |
|---|---|---|
| `prep-dc0899c7a0d81fa65e68f2ff` | `prep-ca875937107cd30a4ef21d55` | RomuTail |
| `prep-b244e2c3ca209a3e66fadfe5` | `prep-80f5135c8da1369f1de87338` | WSL |
| `prep-9b1588086a0ec7246cbba0ea` | `prep-821a837c0b178a938291065b` | RomuTail |
| `prep-f6dfd282043e07862535e900` | `prep-1a175a35b70e400e22452f43` | RomuTail |

One independent endpoint configuration (`50061e6592bd462b8bf5`, `L_D=0`, requested width `6h`) returned 12 valid rounds and failed the existing round-11 `RAW_CORRECTED_CLOSURE_GATE` (closure approximately `2.10e-9`, gate `1e-9`). Its raw output and no-retry disposition are preserved. This is a numerical admissibility hold, not mapper non-support or an operational retry request. No scientific conclusion is drawn from it.

A stronger candidate-input audit found one historical duplicate signal execution, candidate `5e54b45a560cdd3bea11`. Two PBS requests (`prep-c75cf9c7c2cdc9497f0e5180`, `prep-6b685d6155ca91b14770c7c3`) had identical frozen input bytes, path, provider and configuration, but different bundle manifests. Each performed 12 new rounds (m1–m12); only the second request was adopted. Both original raw records remain preserved. The queue now reuses the durable logical owner by frozen input content, provider, adapter and contract even if its bundle or Git publication reference differs. Unclaimed duplicate readiness/queue rows are retired, and the candidate retains the exact original owner request for intake. A comparison against this actual historical pair selects the already-adopted owner. The duplicate cannot be retroactively removed from the historical execution count.

At 16:19 JST on 2026-11-25, the run had 140 new-mapper candidates ready for search (164 candidate preparation records in total). The accepted, adopted new-mapper signal counts were RomuTail 55, WSL 12 and PBS 9; Q3 adoption counts were RomuTail 58 and WSL 11. Four PBS calibration and four corresponding local model-freeze jobs had already been adopted in the earlier normal runs. These counts are operational observations, not final scientific results or an optimized RZ comparison.

The user temporarily paused PBS admission because Akeno would become unavailable, and explicitly requested that the existing compute node be left unchanged. The run-owned admission was set to `accepting=false` for `pbs-0` only. Existing PBS allocation and attempts were not cancelled, stopped, or reissued. RomuTail and WSL continued; WSL's routine verified-copy maintenance reduced the storage measurement below its limit and restored three admission slots.

The common implementation branch `93adfaec48591e0fb131f1fb278aa71d2501efe1` diverged at `bc945e9b20c6ee2c1cdd7e4bcef692d1ececa3c2`. Its source changes are integrated on top of the later run-controller safeguards rather than replacing them. The old run keeps its selected worker policy, explicit recovery selection, and scientific SOURCE; common policy support is available for future releases without automatically changing the production current release. Worker-specific new receipt and resource classifications become effective only when the corresponding operational workers have been deployed; existing resource reports retain the old classification.
