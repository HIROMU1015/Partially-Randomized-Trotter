# H4全byte受領・carry・最終binding bundle

一つの人向け入口は[最終報告](../../../../docs/research/track_a_h4_byte_receipt_binding_20261009.md)。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a` 不変、branch `track-a-h4-receipt-reseal-20261009`。

- [全2150files/既知9SHA](receipt_audit_v4.json)、[6入力/freeze](input_receipt_v4.json)、[carry native監査](carry_audit_v2.json)。
- [native停止評価](native_stop_assessment_v2.json)、[不足する最小旧host proof](NATIVE_TERMINAL_PROOF_REQUIRED_v1.md)、[byte/control receipt](stop_receipt_v6.json)。
- [control82 mapping](control_binding_map_v6.json)、[草案訂正履歴](draft_corrections_v2.json)、[現正常配置](materialization_audit_v2.json)。
- [SOURCE33](source_freeze_v4.json)、[environment](environment_profile_v2.json)、[compiler](compiler_profile_v2.json)、[旧環境との差](environment_evaluation_v2.json)。
- [plan v6](plan_draft_v6.json)、[auth v6](authorization_draft_v6.json)、[review草案 v6](review_draft_v6.json)、[binding](binding_update_v6.json)。
- [独立最終整合review](independent_final_consistency_review_v2.json)、[容量](storage_projection_v4.json)、[静的budget](static_budget_v2.json)。
- [残承認事項](final_approval_status_v5.json)、[未実行command](unexecuted_launch_proposal_v6.json)、[固定前監査](bundle_verification_v4.json)。
- [commit対象](commit_inventory_v4.json)、[bundle hash inventory](artifact_manifest_v4.json)。

all bytes/freeze/carry/P1とv6整合は合格範囲。run05の全owned terminal proofが未評価なので再sealしない。
sealed=false、approved=false、runtime_authorization=false、allowed_cpus=[]。actual cap74784不変、74804は未適用案。
旧materialization_audit_v1は初期hardlinkのbyte-check履歴でruntime admissionではない。現在v2のregular nlink1 copiesとflat basename control mapが有効。
未公開v4/v5草案と失敗準備utility/logは外部homeに保持。NPZ・raw runtime/control・raw receipt・credentials・内部SSH情報をcommitしない。
source/tests/schema変更0、科学array/新science actual/追加transpile/本番起動/共有設定変更0でSTOP。
