# H8 B2 recovery and parallel preparation readiness

原run01のSTOPと未settle予約1を保持。保存B0の2軸受入＋残るB0Iの2軸一回取得に限定した新scope。実H8 compile/fit/有効grant0。

- `execution_contract_UNAPPROVED_v1.json` / `grant_INVALID_DRAFT_v1.json`: prepare worker2/compile1、5core/112.3125GiB案、未承認・CPU[]・command null。
- `saved_B0_acceptance_registry_v1.json`: 原STOPを保持した保存2軸の証拠と別途承認が必要な受入条件。
- `source_profile_input_binding_v1.json`: 31 source/265 provider/34,273 runtime/16 inputのseal。
- `repair_summary_v1.json`: PSI raw・detached native wait・axis marker・QPY/並列化の実装範囲。
- `technical_validation_v1.json`: 34人工tests、2 qubit旧新compiled一致・4人工compile・4並列worker全回収。
- `preservation_receipt_v1.json`: 旧687保護・124worktree・B2原45files不変。
- `publication_files_v1.json`: 公開SHA inventory（自身を除く）。

`python3 -B verify_readiness.py` は公開bytesと保存metadataだけを確認する。実H8科学計算・resource gate・compile・fitを起動しない。新scopeの明示実行承認待ち、mandatory STOP。
