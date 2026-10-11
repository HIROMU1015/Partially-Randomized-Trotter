# 2026-10-12 H6共用サーバー Phase A一回実行

利用者が固定index447/B1 S4 q8・CPU20–23/workerAS32/coordinator soft2-hard32/RSS40/job RAM64/thread1を一回承認。origin REVIEW e51f19b6・migration SOURCE61cadec5・科学SOURCE5a903aeaを再照合した。

利用規則が未提供だったため運用情報を照会し、利用者から「規則/schedulerは特にない。他利用者へ著しく迷惑を掛けない範囲で可」と指示変更を受領。CPU/RAMの承認と監視guardを維持し、新grant c420691bへ結合した。

必要な継承/CPU/thread/grant検査3件PASS。初回test driverのFSIZE hard16MiB設定がfixtureの2GiB要求を拒否した原証拠も保持し、driverだけsoft16MiB/hard2GiBへ直した。科学source/production制限の変更やscience retryではない。

Phase Aを一度起動。workerはlimits/CPU設定後、qiskit-nature不足により12-mode Gaussian準備でImportError。transpile未到達、新group/wrapper0、missing143。旧449/902/92/84を再照合した。

準備時に固定package gateの照合に加えて必要なoptional lowering dependencyを確認すべきだった。小規模人工fixtureはdense fallbackで通過し、この不足を検出できなかった。requirementsにはqiskit-nature0.7.2がある。既存profileや原PASS記録を改変せず、今回の依存不足と検査範囲の限界を別記録に残した。

worker peak RSS約0.352GiB/VmSize約0.757GiB、監視wall4.012秒、exit1。coordinator個別memory/正確な全wall/exit codeは未記録、退出後補完なし。AS32GiBの完走可否は未判定。

own wait/reap/log/journal終了、coordinator/worker残存0。H4は継続し、shared/private H4環境変更なし。新grant一回消費済み、追加install/retry/resumeなし、Phase B/独立確認/H8未認可でmandatory STOP。

[最新結果・原ログ・監査・次の判断](../track_a_h6_server_phase_a_result_v1.md)へ戻す。
