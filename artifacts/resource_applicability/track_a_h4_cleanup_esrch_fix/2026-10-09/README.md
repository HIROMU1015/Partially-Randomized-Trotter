# H4 cleanup ESRCH修正・独立再review bundle

一つの人向け入口は[最終報告](../../../../docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`、branch `track-a-h4-cleanup-esrch-fix-20261009`。

- [旧/new closure33](source_freeze_v3.json) と [事前限定人工計画](ARTIFICIAL_TEST_PLAN_v3.json)。
- [author39件PASS](focused_test_results_v3.json)、[独立13件PASSとbinding再review](independent_rereview_v3.json)、[外部raw証拠hash](local_artificial_evidence_manifest_v3.json)。
- [候補環境](environment_profile_v2.json)、[compiler](compiler_profile_v2.json)、[環境差](environment_evaluation_v2.json)。
- [入力未受領](input_receipt_v3.json)、[native停止未受領](stop_receipt_v3.json)、[binding](binding_update_v3.json)。
- [plan](plan_draft_v3.json)、[auth](authorization_draft_v3.json)、[review草案](review_draft_v3.json)、[未実行command](unexecuted_launch_proposal_v3.json)。
- [容量](storage_projection_v2.json)、[静的actual budget](static_budget_v2.json)、[残条件・承認案](final_approval_status_v4.json)。
- [commit対象](commit_inventory_v3.json)、[bundle hash inventory](artifact_manifest_v3.json)。

P1_SCOPE_TECHNICAL_PASS / draft binding PASS。凍結入力・native停止証拠はNOT_EVALUABLE。
sealed=false、approved=false、runtime_authorization=false、allowed_cpus=[]。独立reviewは実行許可を発行しない。
現actual cap74784・carry20不変、74804は未適用案。追加transpile/本計算/共有設定変更0でSTOP。
raw logs・NPZ・実runtime/control・credential・内部SSH情報は収録しない。actual REVIEW/remote SHAはcommit後の外部publication receiptと最終報告で固定する。
