# H6 Phase A v2 — 資料入口

最新利用者指示によるnative依存修正と、新しい一回index447実行。旧grant・旧run・旧環境は保全。
[修正/固定環境/検査/資源/一回範囲/公開状態](../../../../docs/research/track_a_h6_server_native_dependency_fix_v2.md)。

SOURCE `6a924e52cf1b6541e2780e9d1b756b8049683166`、science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`。REVIEWはこのbundleを固定したcommit（起動receipt参照）。

- [SOURCE変更理由・対象](source_change_receipt_v2.json)、[限定51検査](limited_validation_v2.json)、[XML](tests_v2.xml)、[人工予算](synthetic_test_plan_v2.json)。
- [環境profile](environment_compiler_profile_v2.json)、[環境差](environment_delta_v2.json)、[旧環境保全](old_environment_preservation_v2.json)、[wheel inventory](dependency_wheel_inventory_v2.json)、[requirements lock](requirements-server-v2.lock)。
- [manifest](sealed_manifest_server_v1.json)、[受領/reuse](receipt_server_v1.json)、[新grant](authorization_approved_server_v2.json)、[承認根拠](authorization_basis_v2.json)、[draft](authorization_draft_server_v1.json)。
- [起動前限定整合review/fresh資源](prelaunch_consistency_v2.json)、[remote起点照合と新SOURCEの公開制限](remote_base_verification_v2.json)。
- 起動後の`launch_receipt_v2.json`、原ログ/diagnostics/資源journalは `../../track_a_h6_server_phase_a_v2/2026-10-12/launch_server_v2/` を参照。

worker1 CPU20/21 AS32GiB、coordinator22/23 soft2/hard32、RSS40GiB/RAM64GiB/internal thread1。
実32GiB q8完走・旧output equivalenceは未確認。成功/失敗でSTOP、retry/resume/Phase B/確認32/H8へ進まない。
新公開用資料へNPZ・実runtime/cache/venv・credentialをstageしない。

利用者が公開する場合のcommand（REVIEW/結果commit後にHEADを公開）：
```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-native-dependency-fix-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-native-dependency-fix-20261012
```
