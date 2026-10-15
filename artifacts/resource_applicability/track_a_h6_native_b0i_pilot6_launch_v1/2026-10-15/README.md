# H6 native B0+I pilot6 run01 — FAIL_CLOSED_STOP

SOURCE `7300b39df4c53d854d5bc8934430af19fbdb0085`。利用者の明示承認後に一度起動し、Phase AのB0_BLUEPRINT_FINGERPRINT_MISMATCHで停止。native build/control/compile0。自動retryなし、mandatory STOP。

- [研究記録・停止理由・次の条件](../../../../docs/research/track_a_h6_native_b0i_pilot6_run01_20261015.md)
- [停止結果・集計・SOURCE/input不変](stop_result_v1.json)
- [first STOP/監視履歴](terminal_audit_v1.json) / [native exit2](native_exit_audit_v1.json) / [親wait/回収](launcher_wait_audit_v1.json)
- [承認文](user_approval_raw_v1.txt) / [文脈](approval_evidence_v1.txt) / [固定実行契約](final_contract_v1.json)
- [独立最終binding所見](final_binding_review_v1.md) / [認可](authorization_v1.json) / [最終seal](final_seal_v1.json)
- [入力/profile/source照合](byte_profile_binding_v1.json) / [carry参照結合](carry_binding_evidence_v1.json)
- [初期資源](initial_resource_gate_v1.json) / [直前fresh gate](fresh_gate_immediately_before_launch_v1.json)
- [launcher identity](launcher_identity_v1.json) / [最終起動観測](launch_observation_v1.json)
- [公開状態](publication_receipt_v1.json) / [成果物manifest](artifact_manifest_v1.json)

固定契約を変えず、終了後もSOURCE/inputのbytes一致を確認。NPZ・実cache/checkpoint・private input mapping・認証情報はcommitしない。
