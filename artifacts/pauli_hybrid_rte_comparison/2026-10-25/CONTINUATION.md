Observation timestamp: 2026-10-25T12:55:29.288161+00:00

# Hybrid PR / full-randomized RTE execution continuation

The user permitted ending the chat while the current calculation continues.
This note is a continuation aid, not a scientific completion or STOP record.

## Active run: do not restart

- Worktree: `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/pauli-hybrid-rte-execution-20261025`
- Branch: `codex/pauli-hybrid-rte-execution-20261025`
- Published scientific SOURCE: `b19ed83c6b4b64821965fa0b756cd2691e34eda8`
- Approved preparation: `25d248a1f2ed88de62b0e621c2d1cea8532804c4`
- Native execution session: `90479`; stdout/stderr redirected to this directory's `engine.stdout.log` and `engine.stderr.log`.
- Engine: `scripts/run_pauli_hybrid_rte_comparison.py run`.
- Interpreter: `/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python`.
- Initial batch: 48 paired trajectories per candidate, 144 total. Last saved counts: H4 `p001_q8_r8_K4` 48/48; H6 `PR_p0000_q8_r4_K2` 48/48; H6 `PR_p0000_q8_r4_K4` 24/48. These are observations at note creation, not final counts.
- `STUDY_STOP.json` was absent and stderr empty at observation.
- The engine applies the fixed extension rule after the initial batch, writes final comparison tables and `STUDY_STOP.json`, then exits. Maximum approved new trajectories 2688 / axis compiles 5376; no new state action.
- No artificial resource cap: available CPU was almost fully occupied; engine selected one low-priority worker from observed available resources. No other user's process or shared settings changed.

Do not rerun `analyze` or `run`, interrupt the current process, alter its frozen SOURCE/inputs, or reopen an old STOP. Poll the existing session if available, otherwise inspect its saved logs/markers. A lost native session is not evidence of native exit 0.

## Remaining work after the engine finishes

1. Read actual `STUDY_STOP.json`, stage native records and logs; collect root native exit only if directly observable. Preserve failures and do not fabricate exit evidence.
2. Run `scripts/check_pauli_hybrid_rte_results.py` using the interpreter above. It verifies existing input/SOURCE hashes, copied N/B/bias values, exact G identities, saved seeds, raw price means, QPY contents/counts and native child exits. It does not compile or perform state calculations.
3. Run `scripts/pack_pauli_hybrid_rte_evidence.py` once. It packs H4/H6/stages losslessly into split archives; it preserves original raw files. The output namespace's `.gitignore` excludes raw directories and unsplit tar files, not archive parts.
4. Restore `data_archives.json` with the existing `scripts/restore_pauli_h2_h6_evidence.py` into a fresh empty directory and record actual SHA-verification success. Do not replace original raw.
5. Run `scripts/report_pauli_hybrid_rte_comparison.py` once after completion verification passes. It fills `<!-- RESULTS_AFTER_COMPLETION -->` in `docs/pauli_hybrid_rte_comparison_20261025.md` and writes `comparison_scope_status.csv`. Read and finish the report; do not publish the placeholder as a completed result.
6. Update only relevant discovery indexes/current research entry. Stage intended new outputs, helper scripts/report and directly relevant indexes; preserve old SOURCE/raw/grant/ledger/marker/STOP.
7. Commit and publish final results on the execution branch and verify local HEAD / GitHub ref SHA. The connected GitHub Git data API was used successfully for SOURCE publication; native push has not been attempted. Do not force-overwrite an unexpectedly changed remote ref.
8. Mandatory STOP after result publication; no new science, major-claim adoption, detailed GPT science review, paper writing or population guarantee proof.

The new checker, packer and report formatter are saved locally but are not yet run or committed as final results. Do not alter the frozen execution driver or kernels.

## Interpretation boundaries to retain

- Separate exploration and independent confirmation; no pooling. n is classical trajectory-price sample size, N is copied conditional shot accounting.
- All G comparisons remain conditional on rho=0 and empirical finite-cohort mean RZ prices, excluding state preparation. PR whole-law residual and population price tail remain unproved.
- Missing/null G is not zero or infinity. Preserve unresolved IDs in comparisons.
- H2 saved independent confirmation reverses exploration point ordering; reuse it, do not resample H2.
- H4 new p001/q8/r8/K4 completed48 diagnostic is separately saved in `H4_completed_initial48_diagnostic.json`; all five epsilon extension checks were false. This is not the final multi-molecule study verification.
- H6 RTE registered candidates use r4, whereas Hybrid includes r2/r4/r8; finite-register comparison cannot isolate prefix alone. Some H6 RTE N/G are null (safe-integer/numerical conditions). Report any winner within the computable finite subset and disclose incomplete registered-domain comparison.
- Nominal two-standard-error separation is descriptive, not a rigorous post-selection or population guarantee. The five epsilon comparisons share trajectory prices and are not independent experiments.
- Use final saved rare-order counts; never assign zero population cost to unobserved events.

The full user-authorized remaining verification/publication scope remains in effect; no repeated scientific execution permission is needed within it.

## Completion update (2026-10-25 JST)

On return the original session 90479 returned native exit 0. All 3 candidates completed 48 paired trajectories; no extension was selected. Scientific STOP and 15-row engine verification PASS are saved. The separate saved-artifact checker passed for all 144 trajectories / 288 compiled QPY. Final packaging, restore verification and publication follow; do not rerun the scientific engine.

Packaging and restoration completed: 3 archives / 1020 original files, all SHA verified. The comparison report and scope CSV are completed. Scientific calculations remain stopped; only commit/publication and remote SHA verification remain.
