# H6 server preparation v1 — 資料入口

2026-10-12 JST。準備完了、Phase A実行未認可。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` / `next_stage_authorized=false`。
科学run0、approved grant0、新grant未消費。H4は変更せず継続する。

[移行準備契約・環境・資源・未実行command・次の承認事項](../../../../docs/research/track_a_h6_server_migration_preparation_v1.md)が今回の全体説明。

| 固定項目 | Identity |
|---|---|
| branch | `track-a-h6-server-migration-preparation-20261011` |
| migration SOURCE（公開済み） | `61cadec5b5b2c8f1138828f43f7780d36319ccc3` |
| science SOURCE（224 paths不変） | `5a903aea2492c2d211c1e15e7737df0d50694c12` |
| 起点の公開証拠 | `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` |
| manifest digest | `43672daed236ff47033458d7c7de1771ae2607c62a7ca83e6b07031b83207403` |
| environment fingerprint | `cbaf068eb8a0dbb5a6843eeb054e7cb367c6eee2951b04baaed322b746478561` |
| Phase A | index447 / `B1_p19_q8_4th` / replica0 / seed17585593421502146214 |

SOURCEの子に軽量準備資料をREVIEW_BUNDLE commitとして固定する。commit自身の40文字SHAは最終報告または`git rev-parse HEAD`で確認する。
利用者のpush指示変更に従い、REVIEWはローカル固定のみ。remote branchは現在SOURCEまでで、REVIEW公開は未完了。

- [sealed manifest](sealed_manifest_server_v1.json)：科学plan・231 runtime paths・input/reuse/environment/outputとPhase Aだけのresource schema。
- [grant draft](authorization_draft_server_v1.json)：`approved_by_user=false`、`server_usage_approved_by_user=false`。実行に使えない。
- [受領・再利用receipt](receipt_server_v1.json)：33入力、旧raw5220、449 group/902 wrapper、92 signal/84 B3、missing143。
- [environment/compiler profile](environment_compiler_profile_v1.json)、[wheel inventory](dependency_wheel_inventory_v1.json)、[hash付きrequirements](requirements-server.lock)、[build inputs](environment_build_inputs_v1.json)。
- [synthetic plan](synthetic_test_plan_v1.json)、[検査結果56件](synthetic_validation_server_v1.json)、[38件XML](server-tests-release-final.xml)、[18件XML](compatibility-tests.xml)。
- [binding audit](final_binding_audit_server_v1.json)、[最終整合性review](final_consistency_review_server_v1.json)、[他worktree保全](preservation_audit_server_v1.json)。
- [fresh資源観測](fresh_resource_observation_v1.json)、[サーバー利用とH4分離](server_usage_and_h4_separation_v1.json)、[容量/block/inode](storage_projection_server_v1.json)。
- [SOURCE remote照合](source_remote_verification_v1.json)、[bundle bytes/SHA inventory](review_bundle_inventory_v1.json)。

CPU候補20–23、cost worker1（affinity20/21）とcoordinator（22/23）、内部thread1。
worker AS32 GiB、coordinator soft2/hard32 GiB、aggregate RSS40 GiB、job RAM予算64 GiB。
空きRAMの観測は予約を意味しない。正式な運用規則・scheduler/割当の確認と新しい一回実行承認が残る。
実H6 q8のpeak/完走、旧native binary bit同一性、実compiled output完全一致は未検証。
実行承認後のfresh検査は必須。成功/失敗のどちらもmandatory STOP、retry/resume・残り142 group・独立32確認へ自動進行しない。

専用Python：`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`（3.11.0rc1）。
専用worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011`。
今回のSOURCE/REVIEWに新NPZ・実runtime/cache/checkpoint・venv・credential・内部SSH情報を含めていない。
