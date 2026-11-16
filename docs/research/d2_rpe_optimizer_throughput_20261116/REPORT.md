# RPE探索共通scheduler：D2実装・技術検証

完了済みB2ログの観測区間では18 slotの利用率は99.955%、平均同時稼働は17.992件だった。主時間はDPであり、現行PR-Pauli κ探索の速度改善は今回測定していない。4方式の独立探索を同時に進めるopt-in実装を追加し、同じ人工入力・開始状態で最安適合点、採否に使った配分順、停止理由の一致を確認した。4方式の人工遅延では中央値2.537秒→0.722秒、対比較の速度比中央値3.52倍。単一方式でslotが埋まる人工ケースでは1.043秒→1.179秒となり、高速化しなかった。

実装SOURCEは `c3bb2ad2c409c48ab73708ca61aeab5b9323b317`。基点は実行SOURCE `1e88cd85cc8d2739a4dce43d138b94b153309333`／公開 `98b6f8e1ac6e68285d2c68a8e35125d02d6edd9c`。専用branch `codex/d2-rpe-optimizer-throughput-20261116`・専用worktreeで作業した。**新規実Q3・信号・sampling・回路build/compile・fitは0件**。B2の進行中run、SOURCE、checkpoint、実行directoryには書き込んでいない。今回の検査はD2実装者自身による技術検査である。

## 既存成果・進行中作業の照合

着手時と公開前に関連branch・Git履歴をscheduler/parallel/throughputの範囲で確認した。B2の現行runnerとAの受入branchは存在したが、同じ共通dispatcher改修は確認できなかった。既存の `Executor`（request/identity/contract照合、route adapter、collect）、`run_by_configuration` の独立構成並列、既存のlookahead、固定DP acceleratorのworkerパラメータ、保存C catalog readerを再利用した。新規部分は方法をまたぐ実行queue・起動前の予約・採用計測と、それらの人工検証に限定した。

既存手法集は [PERFORMANCE_METHODS](../../PERFORMANCE_METHODS.md) と固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を照合した。新しい科学kernel、別の探索規則、別の候補選定は実装していない。別系統の既存 `b1dba4b6f` の人工R1 coordinator（`algorithm_codesign/g6_w2c_b3_parallel_candidate.py`）も確認したが、8-pair F01 driver専用で今回のRPE探索・既存routeに適合しないため移植しなかった。

## 完了ログから分かったこと

読取対象は [OBSERVED_LOGS.json](OBSERVED_LOGS.json) に固定した1 run。snapshot UTCは `2026-11-16T03:32:01.914181+00:00`、観測区間は約23,967秒。`PROCESS_RETURN.json` がある3,353 jobの運用記録だけを解析し、OUTCOME/科学RESULTの数値や進行中checkpointは読まなかった。PROGRESSの完結行から同時稼働を復元し、区間末尾で未完の18 jobは右打切りとして扱った。

|量|保存ログで確認できた値|
|---|---:|
|設定slot／最大同時稼働|18／18|
|時間加重平均稼働／slot利用率|17.99195／99.95526%|
|空きslot秒|193.0（区間全体の0.04474%）|
|1 job wall 平均／中央値|128.328秒／129.586秒|
|起動→科学call開始 平均|5.674秒（平均wallの約4.42%）|
|metadata 平均|2.621秒|
|DP 平均|117.601秒（平均wallの約91.64%）|
|DP worker設定|読んだ5件すべて8、outcome_batch=8|
|process tree CPU秒／wall秒 平均|1.592|
|個別jobの標本peak RSS 中央値／最大|3,323,992 KiB／4,808,840 KiB|
|候補待ちに帰属する空き時間|不明：旧logにready/enqueue情報なし|
|先読み結果の実探索採用率|不明：旧logにADOPT記録なし、最終ROUND_RESULT未公開|

CPU/メモリの飽和原因までは特定できない。個別peak RSSを足して全体peakとはしていない。利用可能CPU affinityは64だったが、共有ホストの稼働条件は隔離していない。既存18 slot・lookahead 3を増やす根拠は得られなかった。DP worker数の最適値も実Q3では未測定なので、初期導入時は既存値を維持する。

## 実装と実際の呼出し経路

`run_rpe_method_optimizer.py --round … --run-dir …` → `round_mode` → `rpe_optimizer_concurrent.optimize` → 元 `O.optimize`／`run_method` → 共通 `ConcurrentExecutor` → 元route adapter → 固定worker、という経路を使う。`executor.scheduler="concurrent"` で選び、未指定は旧版のまま。

- methodごとの探索を同時に進め、構成ごとの既存探索に共通queueを接続する。要求中の回答を優先し、同順位では直近に起動していない構成を優先する。lookahead登録はslot待ちで探索threadを塞がない。
- 結果の取込は元の探索が選んだ順だけ。先に完了しても次の配分を先に採否しない。古くなった未起動の先読みだけをqueueから除く。起動済みjobは中断しない。
- job identityは既存のmethod/configuration/vector/route.kind/評価契約digestを保つ。同じrunで重複要求を共有し、保存OUTCOMEと実行中jobを再利用する。異なるrun directoryに同じ科学callを新規発行しないため、後続roundでは従来どおり保存結果を `resume_from` で引き継ぐ。
- 有限budgetでは予約段階で全起動予定を計上する。既存の有限budget時の構成順序と先読み抑止を維持する。未起動queueの取消時だけ予約を返し、起動済みの未採用先読みも件数に含める。
- 再開時のlive workerも共有slotに数える。完了結果は再起動しない。Popen前のSTARTINGとLAUNCHの間で停止した曖昧なjobは再投入せず、既存collectと同様に要確認とする。同dirのfast driver二重起動をlocal file lockで拒否する。起動・保存の失敗では新規queueを停止し、Popen済みworkerはslotを保持して終了を待つ。失敗をUNKNOWNとして残し、保存できたrawを保全する。legacy driverとの同時使用はしない。
- `new_calls` は今回の実起動件数、`adopted_new_calls` は従来の採用時call計数。`executor_metrics` は起動・採用・先読み採用率・未採用完了・最大同時数・待ち時間・標本RSSを区別する。構成別の既存traceは採用順と従来のcallsを保つ。以前のjobの `spent_before` は別欄。
- ENQUEUE/LAUNCH/ADOPT/CANCEL/RESUME/EXITを同じPROGRESSに記録し、候補のない空きslot秒とqueueがある空きslot秒を分離する。子終了code/signalと人工worker段階logを保存する。faulthandler timerや他processへの操作は使わない。

`dp_workers` は既存CASE_INPUTS.acceleration.workers_per_configurationだけを変更する。decoder、target、loss、exact τ、AST期待hash、kernel bytes、outcome_batch、精度・状態数等の科学条件は変えない。変更後の値はREQUEST／CASE_INPUTS／SOURCEに残る。実routeのPython 3.11.0rc1で元Accelerator constructorをworkers=1/2/4/8で呼び、元AST hash・置換数guardの通過を確認した（[CONSTRUCTOR_CHECK](CONSTRUCTOR_CHECK.json)）。kernel評価・科学callは0件。実数値の同等性や速度をその検査から主張しない。

## 同じ開始状態での人工比較

基準schedulerは固定1e88cd85cのoptimizerを直接loadし、両版で同じ人工route adapter、開始時の保存V0、候補、価格、MSE応答、探索規則を使用した。4方式各1人工構成と、1方式9人工構成を比較した。後者も架空入力であり、実κ探索の複製ではない。各3反復、下表はwallとメモリの中央値、倍率は同反復の旧版との比の中央値。起動数・最大同時数は3反復の範囲。全21対比較で、exact最安点・X/Y配分・採否順・構成別と方式別STOPが一致した。

|人工シナリオ|scheduler|slot/DP worker|wall秒|速度比|起動/採用Q3|最大同時|標本RSS MiB|
|---|---|---:|---:|---:|---:|---:|---:|
|4方式 遅延|旧版|18/8|2.537|1.00×|60/52|4〜5|45.0|
|4方式 遅延|共通|18/8|0.722|3.52×|60/52|12〜14|180.0|
|4方式 遅延|共通|8/8|0.944|2.66×|57/52|8|120.0|
|4方式 fork/整数CPU|旧版|18/8|3.940|1.00×|60/52|4〜5|357.8|
|4方式 fork/整数CPU|共通|18/8|1.105|3.56×|60/52|12〜13|1126.9|
|4方式 fork/整数CPU|共通|18/4|1.060|3.72×|60/52|12〜14|696.4|
|4方式 fork/整数CPU|共通|18/1|0.968|4.05×|60/52|12〜13|222.9|
|4方式 fork/整数CPU|共通|8/2|1.426|2.76×|57/52|8|289.0|
|1方式・9人工構成 遅延|旧版|18/8|1.043|1.00×|135/117|18|270.0|
|1方式・9人工構成 遅延|共通|18/8|1.179|0.89×|127/117|18|269.0|

全件ARTIFICIALで、実Q3速度とは区別する。fork/CPU例は短い整数演算・fork・2 MiB確保であり、実DPの有理数演算を模擬していない。workers=1の好結果は実DPでの最適値の証拠にならない。slotを8に抑えると人工4方式でも遅くなる場合があり、単にslotやlookaheadを増やす案は採らなかった。単一方式の飽和ケースでは共通queue・監視のoverheadも現れたため、その形の次runで高速版の優位は未確認である。

RSSは両版で同じ50ms観測器が所有する人工job treeだけを標本観測した値。共有pageの重複計上、短いpeakの見逃しがあり、PSS/厳密最大メモリではない。新実装の内蔵監視は100ms。共有ホストでB2を稼働させたままの計測であり、速度倍率は技術fixtureでの観測範囲に限定する。

新版の計測例として、4方式遅延・18/8・反復0では先読み起動52件のうち44件（84.62%）を採用し、未採用完了は8件だった。queueなし空き3.819 slot秒とqueueあり空き2.243 slot秒も分離して保存した。これらは人工例であり、旧実Q3の不明な採用率を置き換える値ではない。

個別反復、exact人工最安点、採用vector列、STOP、起動・未採用計数は [BENCHMARKS.json](BENCHMARKS.json)、集約は [BENCHMARK_SUMMARY.json](BENCHMARK_SUMMARY.json)。実科学費用の再評価や最安更新には使用しない。

## 検証と保存結果の再生

[TESTS.txt](TESTS.txt)：新規人工15件、関連する既存Search/ExecutorJobs/Inputs 17件がPASS。通常入口、4方式、有限予約、先読みの逆順完了→元順採用、重複要求、完了再利用、live再開の容量計上、曖昧起動STOP、UNKNOWN停止、METHOD scope、実CLIから人工adapter subprocessへの到達と、子起動直後のLAUNCH保存失敗でも追加投入せず子終了を待つ経路を確認した。`proposals/method_state/run_method/run_by_configuration` は基準SOURCEとAST一致を検査した。実H4/P0数値suiteは起動していない。

[SAVED_REPLAY.json](SAVED_REPLAY.json) は固定C catalogだけを再生した。candidate generationと外部価格quoteを無効にし、科学budget 0・science_approved=falseで既存各構成の保存適合seedから比較した。4方式の保存最安点・STOPは一致、未保存の次提案で両版ともNEW_SCIENCE_NOT_APPROVED。新しい回答を採用したtraceは0件なので、これは保存入力の接続確認であり探索全軌道の検証は人工fixtureが担う。metadata準備約5.6秒、保存再生は数十msで、thread overheadによる小さな増加も記録した。保存価格・MSEは再計算していない。

## 引渡し・残る確認

[HANDOFF.md](HANDOFF.md) に次run用の起動commandと実行profileを記載した。複数方式の独立したready候補がある次の承認済み探索で使用できる。今回の結果は実Q3のworker最適値や、現行κ探索の完了時間を予測しない。実Q3での速度確認は、後続の承認済み探索に計測を付けて行う。旧runやcheckpointを変更して導入しない。

ロードマップへの差分は、共通探索基盤にopt-in schedulerと計測が加わったことだけ。価格資格、科学的最安点、Q3結果、LD/κの選択、精度契約は更新しない。旧STOPは保全し、公開・remote照合後は新たな科学探索へ進まず終了する。
