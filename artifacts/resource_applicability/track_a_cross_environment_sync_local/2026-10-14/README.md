# Track A ローカル環境照合bundle

2026-10-14 JST。資料専用branch `track-a-cross-environment-sync-local-20261014`、親server commit `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c`。
新科学計算0・H8操作0・新grant0。`INTEGRATION_CANDIDATE_NOT_CANONICAL` / `DRAFT_NOT_AUTHORIZATION` / `next_stage_authorized=false`。

- [ローカルinventory](../../../../docs/research/track_a_local_inventory.md)
- [差分・競合](../../../../docs/research/track_a_environment_sync_differences.md)
- [統合候補](../../../../docs/research/track_a_shared_research_status_candidate.md)
- [source/result/資料索引](../../../../docs/research/track_a_local_sync_reference_index.md)
- [local metadata](local_inventory_v1.json)：HEAD/dirty/path/Git blob/SHA、重複含む265実体・93内容
- [GitHub取得照合](github_reference_verification_v1.json)：37固定ファイル、うち主要19証拠
- [local-only](local_only_materials_v1.json)：16内容/44実体、全履歴不存在の証明ではない
- [論点・未解決](difference_registry_v1.json)：7分類、scope競合と未確認
- [公開履歴registry](historical_reference_registry_v1.json)：既存14資料の固定commit/path/hash、再commitなし
- [old-export原terminal](old_environment_export_terminal_original.json) / [provenance](old_environment_export_provenance_v1.json)：330 bytes exact copy、原因未特定

- [静的照合・保全監査](static_verification_v1.json) / [公開asset hash](asset_manifest_v1.json)

private preservation baselineは公開せず、source/raw/runtime/archive/private設定を複製しない。
共有server入口・inventory・policyと原稿v0.1/v0.4、旧STOP/grantは原bytesで保持。既存root branch/ref/remote/dirty/untrackedは変更しない。
本bundleの検査はstdlib/Git/文書hash・リンクに限定し、科学的再現試験やCIと呼ばない。
公開後mandatory STOP。後続は両環境の資料追加照合と候補の採択判断であり、新計算を自動認可しない。
