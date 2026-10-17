# H8主要3構成・6軸：実行準備のみ

入口は[実装・承認待ち報告](../../../../docs/research/track_a_h8_key_compile_readiness_20261017.md)。

- `execution_plan_UNAPPROVED_v1.json`：優先順、6軸、2%基準、禁止事項。
- `source_profile_input_binding_v1.json`：SOURCE22 files、保存入力11件のbyte照合、固定profile。
- `grant_INVALID_DRAFT_v1.json`：無効grant。承認・実行権限なし、CPU未固定。
- `artificial_tests_v1.json` / `artificial_test_output.txt`：人工22件。実transpile・実H8構築なし。
- `resource_proposal_v1.json`：2並列準備/直列compile、上限を新設しない資源案と観測。
- `author_technical_check_v1.json`：担当自身の技術確認。独立した別担当reviewではない。

NPZ・private config・有効grant・実QPY/runtime/ledgerは収録しない。実compileと実測値による再会計は別々の承認が必要。
