# Remaining 106 PF axes: repaired control reference binding

See [report](../../../../docs/research/track_a_pf_direct_compile_retry106_20261020.md).

Exact scope: H2 DF3 46 axes and H4 DF10 60 axes, 53 pairs. Existing 682 accepted observations are preserved. `retry_axes.csv`, `scope_and_reference_bindings.json`, `user_approval_receipt.json`, `repair_diagnosis.json` and `technical_validation.json` describe the fixed preparation. Source/seal/launch and actual results follow separately.

Twenty H2 prefix0/S4 axes (B0 and B0I) use already-saved exact_truncated because the original merged single-term signal schedule is inconsistent with registered T. Their direct gate measurements require explicit saved-classical-signal reconciliation flags. No saved signals/ranking are updated here. New publication is not yet approved.

## Final: all 106 repaired axes accepted

SOURCE `9c65fb312891d8334037f30ef953336c287bc262`; 106 success, 0 failed/unperformed/unknown, 53 owned workers reaped, final byte gate39522 PASS. Existing682 rows preserved, total788 saved task acceptances. The 20 old-classical-signal qualifications and original572 global final-gate limitation remain. See results01/, final_structure_validation.json, terminal_exit_evidence.json and final_resource_summary.json. All science is now in mandatory STOP. Publication remains unapproved.
