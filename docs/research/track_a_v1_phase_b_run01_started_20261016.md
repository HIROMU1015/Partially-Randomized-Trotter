# V1 Phase B run01：S4検証compileを起動

終了追記：[登録全4軸PASS・正常終了STOP](track_a_v1_phase_b_run01_results_20261016.md)を確認した。以下は起動時点の履歴であり、現在の完了状態は終了報告を参照。

2026-10-16 14:10:37 JSTのsnapshotで、B0 p11 q1 S4 cosineの実compile開始を確認した。実行SOURCEは `fcf37e6794e2e51fc95e958a75fc8f75fb049423`。準備内容のcommit・push後にremote SHAとlocal tree一致を確認し、利用者の「計算を実際に開始するところまで進めて」に基づく独立one-shot grantとfresh gateを発行した。

[起動receipt](../../artifacts/resource_applicability/track_a_v1_phase_b_run01/2026-10-16/start_receipt.json)、[固定モデルと実行契約](track_a_v1_phase_b_readiness_20261016.md)。これは起動時刻付きの記録であり、全軸PASSや完了の記録ではない。

対象はlinear H6/1Å/STO-3G/DF19/12 modes/sector400、T=.8、B0 p11 q1とB0+I p15 q8のS4各cosine/sine、最大4軸。compilerはQiskit1.3.0・rz/sx/x/cx・opt1・seed17・symmetric_directional。scalarと経験的allowanceはPhase Aから保持し、保守的bias certificateは未解決。

fresh gateはSOURCE・provider230ファイル・runtime34,273ファイル・9入力bytes/SHA・既存grant/ledger/carry等650ファイル・他job・資源条件を確認してPASS。idle CPU2をdriver、CPU9をobserverに固定し、各thread1で起動した。snapshot時に650ファイルを再hashして変更0。

driver PID2037397のstarttime/parent/UIDを照合し、監視processの生存と最初のcompile開始記録を確認した。snapshot時の予約はsnapshot1、archive decode1、native evolution1、control probe3、wrapper1、compile1。完了軸はまだ0。入力回路25,468命令、fingerprint `3fb2d450d55b843b7a7015bb19ba922d7aab03a5b02ca3da993b3cea13603d7b`。固定予測RZは325984/5で変更していない。

実行中は1秒間隔・最大gap5秒の外部監視を継続する。各軸で相対誤差2%以上なら、その軸の実測を保存して即STOPし、次軸を開始しない。driver/observerのみをowned cleanupの対象とし、再fit・追加compile・自動retry・resumeはない。正常完了でもmandatory STOPし、5精度費用再評価・PR削減率再計算・V2への自動移行はしない。

この記録commitは起動状況の文書だけを追加する。実行SOURCE、固定model、入力、compiler、grant、資源上限を変更しない。準備時の57人工tests PASSを再利用し、新しい科学的結論や母集団費用保証は主張しない。最終判定はprivate run `track-a-v1-phase-b-run01` の`output01/terminal.json`、`output01/validation_complete.json`、`control01/launcher_terminal.json`で確認し、終了後の別報告へ保存する。
