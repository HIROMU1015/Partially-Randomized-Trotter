# Q1–Q9・最終server schemaへ接続する箇所

**Q1：global verifierは。** 実装済み。独立core＋公開人工版adapter＋cost＋CLI。旧pair/F01/cross verifierを再利用し、既存SOURCEは不変。

**Q2：対応形式・証拠は。** R1_PARALLEL_8PAIR_ARTIFICIAL_RESULT_V1、b3_parallel_artificial_v1 PLAN/control、uint8 statefile、pair/summary JSONL、completion receipt、safe_json worker report。公開正常1＋障害17rawを取得・元bytes/SHA照合して検証。

**Q3：call/inflight/STOP/failureは。** planned/reserved/ENTER observed/completed/failed/cancelled/unknown/not-startedを区別。F01/state/rawを結合し、inflightや完全未commit行を成功加算しない。core操作logではSTOP後開始・二重予約・二重完了・不正補充を拒否しdrainを許す。公開P1のSTOP順序は操作履歴がなくpending。

**Q4：独立検証できない範囲は。** statefileだけの全状態遷移/STOP時刻、ENTER→実関数本体entry、compactからの全event/連結trace復元、元private SQLiteのatomic claim、当時の物理process終了、process間例外全順序、未保存費用。申告値照合と独立再計算を分ける。

**Q5：費用二重計上は。** host親＋全worker CPUだけを合計。pair・launch・IPC・finalizationは内包、wallは外側global wall、coarse/fineは別partition、X/Y共有setupは一度。unknownはnull。失敗prefixは診断だけで通常比較CSVへ入れない。並列CPUの科学主指標採択・研究rankingはなし。

**Q6：PASS件数は。** 既存360＋新77＝**437 PASS、FAIL/ERROR/SKIP0**。公開正常1はVERIFIED_COMPLETE、障害17はTECHNICAL_FAILURE。旧339/356を加算していない。

**Q7：fused受入は。** 固定ad3117d2の実helperを人工loadし、exact bytes、Unicode/surrogate/DEL、key collision、int/Fraction、event/evidence/angle、F01型failure、完全SOURCE SHAを照合。公開CachedObserverもlabel/call count/exclusive partition/path・既存profiler拒否を検査。新server統合版を検証したとはしない。

**Q8：次schemaへの変更点は。** 新version adapterで以下だけを明示接続する。

1. 新settings/PLAN/run identity→独立task/offset/global identity。
2. 新state codeとcounter、coalesced ACK/commit/receiptのsnapshot境界→既存core accounting。
3. trial操作証拠が保存される場合、その実形式→normalized check_operations。保存されない項目はpendingを維持。
4. worker failure/F01 wrapper・複数failure/exit/phase/STOP/join→既存pair/F01/control checks。
5. CachedObserver/fused/worker/host全SOURCE pin、function label、cost boundary→別versionの受入・cost adapter。
6. 新正式契約digest/独立approval/atomic claimのbindingと非再利用証拠。これは新SOURCE固定後の別人工統合作業。

共通coreや元pair verifierを新schemaの推測で書き換えない。FutureIntegratedAdapterはPENDING_SERVER_SCHEMA、旧PendingParallelAdapterも従来のpendingを維持。

**Q9：実science等は。** 実G7/F3 import/sampling、新科学計算、実科学結果の数値再解析、実authorization/grant/marker/claim発行消費は0。親/他worktree/既存SOURCE・契約・結果・ledger/STOPに干渉していない。

次の担当は**ユーザー**：server統合版が公開されたら固定commit/schema/契約変更案を指定し、adapter接続・人工統合受入を依頼するか判断する。
その指示後の**Codex**：上記6接続と新SOURCE/pin/observer/failure/STOP会計を人工回帰し公開する。後続提案であり今回未着手。
正式並列契約採択・科学SOURCE変更・実correctness256/31,710,848 calls・実claim・科学レビューへ進まず、公開・保全完了後 **mandatory STOP**。
