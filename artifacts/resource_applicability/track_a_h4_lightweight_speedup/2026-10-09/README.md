# H4軽量高速化・限定同等性確認

一つの入口は[変更・検証・binding・再実行条件](../../../../docs/research/track_a_h4_lightweight_speedup_20261009.md)。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38、branch `track-a-h4-lightweight-speedup-20261009`。

- [source blob/hash](source_freeze_v1.json)、[48限定人工結果と事前計画](limited_tests_v1.json)、[速度改善scope](speedup_scope_v1.json)。
- [plan](plan_draft_v10.json)、[auth](authorization_draft_v10.json)、[review](review_draft_v10.json)、[binding](binding_audit_v10.json)。
- [library cache profile](library_cache_profile_v1.json)、[停止・carry receipt](stop_receipt_v9.json)、[再実行案](relaunch_proposal_v1.json)。
- [独立review](independent_speedup_review_v1.json)、[commit対象](commit_inventory_v1.json)、[artifact検査](artifact_check_v1.json)。

driver共通準備13×218→13回/距離（旧数は静的）、全prepare218→10種類、ledger deltaは変更keysだけ。
全218prepのbytes、代表8paired wrappers、dense256人工wrapper1、代表4signals、ledger13file一致。
実Gaussian/compiler速度と全H4完了は未検証。monitor/charge capsは変更しない。
累積worst12.261694GiB>現10GiB、未seal/approved=false/runtime_authorization=false、再起動0。
既承認environment/CPU/observer/actual74804を保持し、carryを返却しない。absolute_launch_command=null。
