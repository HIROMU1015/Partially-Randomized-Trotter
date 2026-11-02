# H4 worker error修正・review入口

[変更・検証・未確認事項と再起動条件](../../../../docs/research/track_a_h4_worker_error_fix_20261009.md)を最初に読む。

SOURCE `d3bb388401bfd25d31c53e7f785b533a7753ddb3`、branch `track-a-h4-worker-error-fix-20261009`。
39限定人工PASS。productionの元compiler/IPC原因は記録欠落のため未特定、本計算再起動0。

- [closure41のactual blob/hash](source_freeze_v1.json)
- [profile/input/carry・未認可binding](preparation_binding_v1.json)
- [39人工test結果](limited_tests_v1.json)と[上限・所有plan](artificial_test_plan_v2.json)
- [256matrix・9qubit人工IPC probe](artificial_ipc_probe_v1.json)と[cold worker import](cold_worker_import_v1.json)
- [失敗を含む検証原本hash](verification_inventory_v1.json)
- [独立review](independent_review_v1.json)
- [軽量commit inventory](commit_inventory_v1.json)と[bundle byte manifest](artifact_manifest_v1.json)

approved=false/runtime_authorization=false/allowed_cpus=[]/sealed=false/absolute command=null。
carry21/8,692,723,164B/5,766.582514658794s upperを返却せず保持。13GiB/74,804 caps不変。
shared/venv/他job変更、追加transpile、科学入力読込、実worker/GPU/affinity操作0。
