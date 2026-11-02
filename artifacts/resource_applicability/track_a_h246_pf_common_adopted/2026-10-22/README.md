# H246 common PF: adopted fixed-domain research dataset

Status: **FORMALLY_ADOPTED_FIXED_DOMAIN_POINT_RESULTS; NOT_FINAL_EVIDENCE_FREEZE** (2026-10-22 JST).
Fixed result: `90f1cd55a0d963953696bd0a27d9d041ff939bb7`.
This bundle transcribes saved fields and joins provenance. New science, numerical reanalysis, reranking, sampling and compile: **0**.

- [Adoption report](../../../../docs/research/track_a_h246_pf_common_adoption_20261022.md) and [protocol v2 supplement](../../../../docs/research/track_a_h246_pf_common_protocol_v2_supplement_20261022.md).
- [dataset.json.gz](dataset.json.gz): 588 candidates, original saved signal records, 2940 saved paired precision endpoints (shots, native prices, total RZ), per-system scientific identity/compiler/native references. Decode with the Python standard library, without importing science packages.
- [candidate_index.csv](candidate_index.csv): formula/method/prefix/q, eligibility, acquisition type and JSON pointers. `signal=REUSED` is the original common-stage label; `signal_provenance` and the full saved record distinguish H6 metadata observations and new11 raw signals.
- [All-formula comparisons (15)](comparisons_all_formulas_15.json) and [within-formula comparisons (45)](comparisons_within_formula_45.json): exact unchanged costs and PR mean/variance/paired SE/n. PR is saved exploration outer32/K4 with prefixes1/3/5; not independent confirmation. outer32 is not sample size n.
- [Saved precision rankings (15)](precision_rankings_15.json), [old vs common comparisons (15)](old_vs_common_15.json), [saved denominator decomposition (10)](denominator_decomposition_10.json): copied without reranking or new ratios.
- [Paper manifest](paper_manifest.json), [paper data manifest](paper_data_manifest.csv), [Claim–Evidence Matrix](claim_evidence_matrix.md) / [CSV](claim_evidence_matrix_v2.csv), [evidence ledger v4](evidence_ledger_v4.csv).
- [Fixed source catalog](source_catalog.json): commit, Git blob, SHA256, byte count and immutable URL, including prior main table and all34 Figure2 files; no original paper assets are replaced.
- [Adoption record](adoption_record.json), [artifact SHA inventory](artifact_manifest.json), [structural validation](technical_validation.json), [mandatory STOP](MANDATORY_STOP.json).

Linear H2/H4/H6, adjacent1 Å, STO-3G, tol-onlyDF1e-8 saved ranks3/10/19; T=.8/P=0.
B0/B0I prefixes0..2/1..9/5..15 and Full3/10/19; q1/2/4/8, delta.8/.4/.2/.1; S2/Yoshida4/m5_best.
Usual signed bias, empirical allowance, five precisions .05/.01/.005/.001/.0005, alpha_axis.025, full cos/sin wrapper, symmetric-directional control, Qiskit1.3.0 rz/sx/x/cx opt1 seed17.

All588 evaluated, unresolved0 does not mean all588 compiled: unnecessary ineligible candidates remain uncompiled. Saved prices may exist for currently ineligible candidates; `eligible_epsilons` distinguishes possession from use. H4 saved28 pairs include22 used and6 inactive; H6 saved36 include30 used and6 inactive.
H6 old COST187 complete pairs were accepted through saved-byte/vector/gate audit; fresh22 pairs completed with a separate grant. Original parent exit/reap/cause/time remain unknown. No success receipt is fabricated for the interrupted stage.

Adopted all-formula point result: H2 PF lower at all five precisions; H4/H6 PR lower at .001/.0005, PF lower at the other three. These are finite registered-grid results under empirical allowance, not population ranking, global optimum, general PR advantage, rigorous bias or full energy/RPE cost guarantees. Unobserved rare-event bounds remain null.
Old independent confirmation, old q16/DF12 studies, SOURCE/raw/grant/ledger/marker/STOP and prior paper claims are retained in their own scope. AD03 remains deferred.

Technical check (stdlib only): `python scripts/adopt_h246_pf_common_results.py`.
No execution grant is created by this bundle. **MANDATORY STOP.**
