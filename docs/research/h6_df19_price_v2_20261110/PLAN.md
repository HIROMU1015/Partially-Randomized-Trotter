# H6 DF19 price V2 — fixed execution scope

Continue `codex/resource-eval-h6-df19-price-design`; preserve V1 and historical STOPs. The user's V2 instruction explicitly permits this complete limited study; no new approval mechanism is required.

1. Reuse V1 carrier a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4 saved predictions/compiled RZ. Fit only Moffset (PF, PR_A, PR_B) and Mbasis (PR_A, PR_B), paired X/Y. Reuse H4 S1 fit/evaluation helpers and C7-4 multiresponse fit, without H4 coefficients.
2. Exploratory model selection: smallest maximum-axis RMSE on the V1-used known cohorts, ties favor fewer parameters. These data are not fresh holdout. PF's one-point fit has no trajectory SE. Nominal OLS covariance does not cover systematic or finite-bit-law errors. q1 does not identify q slope; Moffset remains a constant offset when applied at other q.
3. Review existing local compiler/boundary methods. Limit new diagnostics to those directly useful; do not chase exact-zero residuals.
4. Check saved prices and exact-fingerprint cache read-only. Unless a fixed-delta q law is established, execute delta=1.6, m5_best, Full19 PF and prefix7/R16/K2 PR: q2/T3.2 calibration and q4/T6.4 independent verification. Maximum 36 full-wrapper prices (PF one pair plus PR eight independent pairs at each q). Freeze candidates, cohort labels, seeds, allocation, order and compiler before acquisition in the pilot input artifact. No modification of shared cache.
5. Freeze selected V2 and q1/q2 Affine coefficients and comparison metrics before inspecting q4 prices. Do not refit q4. q1 PR anchor is the saved PR_B EXPLORATION n9, not pooled confirmation n33.

Scientific conditions: saved H6 linear 1 Angstrom/STO-3G/tol-only 1e-8 DF19 weighted-Hermitian completion; 12 system modes, saved sector/state; original fragment order, 11 signed m5 pieces; symmetric_directional control; P=0; full measured X/Y Hadamard wrappers. Qiskit 1.3.0, rz/sx/x/cx, opt1, seed17; compiler hash 1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b. Original finite K2 RTE law and identity/scalar conventions apply.

No new signal/N/G, state/Hamiltonian regeneration, q>=8, cohort top-up, candidate search, main claim change or detailed scientific review. Complete publication and remote verification, then mandatory STOP. Synthetic checks are implementation evidence only.
