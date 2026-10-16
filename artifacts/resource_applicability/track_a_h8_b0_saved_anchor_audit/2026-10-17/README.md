# H8 B0 saved-anchor audit / 2026-10-17

保存8軸はB1入力候補として構造的に再利用可能。元run02の契約STOPを維持し、過去の違反process未記録を補完しない。新科学計算/compile/fit/grantは0。

- `saved_anchor_audit_v1.json`: 15 SOURCE/265 provider/34,273 runtime/8 inputと8軸の読取専用照合。
- `repair_SOURCE_binding_v1.json`: 別の修正SOURCEと16 files。旧SOURCE差替え・新実行ではない。
- `technical_validation_v1.json`: 新14＋既存83、固定Pythonの97人工tests。独立レビュー/CIではない。
- `administrative_git_probe_v1.json`: 現在のGit metadata probe。過去の原因再現ではない。
- `preservation_v1.json`: 今回baselineの旧681保護files/118worktreeを保持。
- `original_STOP_reference_v1.json`: 元STOPへの不変参照。
- `publication_files_v1.json`: 公開bytes/SHA。自己参照を避けて自身を除く。

`python3 -B verify_saved_audit.py`は公開bytesとmetadataだけを検証し、科学計算・実資源gateを起動しない。B1 fit/B2/Cは別承認、mandatory STOP。
