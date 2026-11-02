# H8 B2 execution readiness / 2026-10-17

B1正本83a3c4a6…/model8420b9f2…を使う固定4軸のS4検証準備。実H8 compile/fit/有効grant0。B0 p15/q1の両軸→B0I p21/q8の両軸、同時compile1、各軸2%以上で即STOP。Cへ自動進行しない。

- `execution_contract_UNAPPROVED_v1.json` / `grant_INVALID_DRAFT_v1.json`: 固定scope・call/資源上限、approved=false/CPU[]/command null。
- `source_profile_input_binding_v1.json`: SOURCE sealと21 source/265 provider/34,273 runtime/15 inputの照合。
- `B2_input_binding_v1.json`: B1正本、固定model/予測・比較条件と入力hash。
- `prefix_preparation_binding_v1.json`: full p28からp15/p21のprefix view。p15の保存root hashはnullのまま。
- `saved_control_targets_v1.json`: 保存corrected/independent-forward/full-H/allowanceとp21 scalarの転記。
- `instruction_allocation_bounds_v1.json`: 静的native配列確保上限。compiled費用の推定/実測ではない。
- `technical_validation_v1.json`: 新19＋既存100の119人工tests、2 qubit S4接続11項目・人工4compile。
- `parallel_preparation_history_v1.json`: 別チャットの許可済み1worktree更新と旧/新baseline。科学retryではない。
- `preservation_receipt_v1.json`: 旧683保護files・更新後122 worktree・B0/B1原本を保全。
- `publication_files_v1.json`: 自身を除く公開bytes/SHA inventory。

`python3 -B verify_readiness.py`は公開bytesと保存metadataだけを照合する。科学API・resource gate・compile・fitは起動しない。実行承認待ち、mandatory STOP。
