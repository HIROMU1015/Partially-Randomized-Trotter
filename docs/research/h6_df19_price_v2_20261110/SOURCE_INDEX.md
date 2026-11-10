# SOURCE and reuse map

All references are immutable Git objects in HIROMU1015/Partially-Randomized-Trotter unless described as local raw.

| Asset | Fixed source / use |
|---|---|
| V1 saved predictions, descriptors, direct prices, prior mismatch | `a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4:docs/research/h6_df19_price_v1_20261110/` (174 axes; no recount) |
| V1 Accounting kernel | `960e8f91140cce37a0eb9091a65a41e18f6e54a3:src/trottertracks/h6_df19_price_v1.py` and `track_d_native_resources.py`; preserved |
| Price design / boundary-method applicability | `4c965ff69e3619a7b2f8313a2093cde679f542b8:docs/research/h6_df19_price_design_20261110/REPORT.md`, especially sections 3–7 |
| H4 paired fit/evaluation and C7-4 generic fit | `cefc01622a3cd75dd16d30de7b9fb68274f3eddd:src/trottertracks/h4_df10_price_v2.py`, `track_d_c7_4_models.py`; function reuse only, no H4 coefficients |
| Original H6 native / preparation / sampling / schedule helpers | `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py`, `h46_new2_native.py`, `h46_new2_core.py`, `h46_pr_pilot_repair.py`; `src/trotterlib/h46_m5_plan.py` |
| Actual provider, builder, compiler, snapshot loader, finite RTE law | `c31ef424175c02379eafebe8fc6c7548c1d9aabe`; matching read-only private provider. Scientific function bodies are reused through AST extraction; historical launch/grant functions are not called. Imports binding these pure functions are adapted locally. |
| q2 candidate input | `95d01b6ab1d820921fcd2521324e41e82389713a:docs/research/h6_df19_rpe_d2_preparation_20261110/PLAN.json`; Full19 `3f813a46e17461bfe484`, PR7/R16/K2 `72255c43facdfb8b4bee` |
| q4 candidate input / existing N result | `18fcd90f1614e3bf7b91bb02d8254f968e048a94:docs/research/h6_df19_rpe_m6_preparation_20261110/PLAN.json` and `h6_df19_rpe_m6_results_20261110/run02/REPORT.md`; Full19 `201a8e365ab6f4af6ebc`, PR7/R16/K2 `ef41b65942d56977d6cd`; N/signal not recalculated |
| V2 saved-fit scientific source | `105a4632ea14a3efe0541bf80a0558f0f4529466`; inputs/source hashes in SAVED_CORRECTIONS.json |
| V2 pilot input / q2 acquisition source | `699dc0cbd48f3192af560b23fdbd0f1c1b99d75a`; PILOT_INPUTS.json, new private scientific inputs/events/results |
| V2 local diagnostics / q analysis | Script SHA256 recorded in LOCAL_DIAGNOSTICS.json and Q4_FROZEN_MODELS.json; final committed bytes must match. q4 coefficients and script hash are frozen before q4 prices. |

New private raw root: `/home/AbeHiromu/projects/h6-df19-price-v2/20261110`. Seeds and event sequences remain private scientific input. Original long-time compiled cache is consulted only at exact newly constructed wrapper fingerprints and never modified. No historical grant/ledger/marker/STOP is reused or changed.

V1's saved PF q2/T0.8 auxiliary point has delta0.4 and is not evidence for the delta1.6 q direction. PR_A p5/R32 and PR_B p7/R16 retain separate coefficients, cohorts and populations. Track D Strong/Reference comparisons are control constructions, not PF/PR comparisons.

q4 uses `scripts/run_h6_df19_price_v2_q4.py`: the same sampled whole-path evolution feeds independent forked X/Y pricing workers. No scientific word, seed or compiler change; its committed execution SOURCE and frozen executor hash are recorded in Q4_DIRECT.json / Q4_FROZEN_MODELS.json.
