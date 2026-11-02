# H6共用サーバー Phase A v1 — 結果入口

2026-10-12 JST。**FAILURE / mandatory STOP**。
index447 `B1_p19_q8_4th` replica0 / seed17585593421502146214を一度起動。
qiskit-nature不足によりGaussian block準備でImportError、full-wrapper/transpile未到達。
新group0/wrapper0、旧449/592・missing143のまま。AS32 GiBの完走可否は未判定。

- [結果報告・原因・資源・次の判断](../../../../docs/research/track_a_h6_server_phase_a_result_v1.md)
- [機械可読summary](result_summary_v1.json)
- [原raw output](../../track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1/execution_terminal.json)
- [worker原STOP](../../track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1/cost_exploration_0447/worker_terminal.json)、[failure diagnostics](../../track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1/cost_exploration_0447/failure_diagnostics.json)
- [resource原summary](../../track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1/phase_a_resource_summary.json)、[原monitor journal](../../track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1/cost_exploration_0447_resource_observations.jsonl)
- [新grantとresource/source binding](../../track_a_h6_server_phase_a_authorization_v1/2026-10-12/authorization_binding_v1.json)
- [raw bytes/SHA監査](raw_byte_audit_v1.json)、[旧source/input/reuse/profile post-STOP監査](phase_a_post_stop_binding_audit_v1.json)
- [preflight完了receipt](preflight_completion_receipt_v1.json)、[運用条件の利用者説明](server_operational_clarification_v1.json)、[必要な3検査](limited_preflight_tests.xml)
- [H4継続read-only記録](h4_post_run_read_only.log)、[公開対象bytes/SHA一覧](publication_inventory_v1.json)

Science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`、migration SOURCE `61cadec5b5b2c8f1138828f43f7780d36319ccc3`。
起点REVIEW `e51f19b649a7feb3f7f84d1e22cf49fc57e0e412`は起動前originと一致。
今回の新grant SHA `c420691bf857ef2e860b64c6724e41770af16eaabde7be7459879e25c4fa80d0`はone-shot消費済み。
成果commit自身の40文字SHAは最終報告/`git rev-parse HEAD`で確認する。

worker peak RSS約0.352 GiB/VmSize約0.757 GiB、worker監視wall4.012秒。memory cap到達ではない。
coordinatorの個別peak/正確な全wall/exit codeは原記録がないため補完しない。
own coordinator/worker残存0、wait/reap・log/journal終了確認、H4継続・CPU重複なし。
新science input生成0、再実行0、環境install/変更0。
missing143のうち447のretryと他142 group、独立32確認、H8は今回のgrantで実行できない。
