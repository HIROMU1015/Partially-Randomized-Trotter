# 旧P5 worker異常終了：証拠と未確定事項

対象は旧C1-5A final `b022f2ea421649015a4c21ea3c904fe619d0c198`、SOURCE `900a5278c604b4c58b794ca891720d9dea293919`、authorization `079ba428257444997d89ee2a975251563d8d16b0`。旧結果・STOPは保持する。この調査段階ではH4 modelを生成・復元・再実行していない。Codex主エージェントによる静的調査であり、別査読者による独立レビューではない。

|項目|確認できたこと|不明・仮説|
|---|---|---|
|setup|[保存setup](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/closed_P5_full/setup.json)は104,977群、setup CPU 283.087432224秒、wall 283.099552773秒、max RSS 1,770,984 KiBを記録。KiBからGiBでは約1.689。|setup成功はworker安全性の保証ではない。|
|メモリ構造|[P5Closed](../../../src/trottertracks/algorithm_codesign/g9_p5.py)はgroup tuple、root interval、dyadic Fraction law、completion閉包/cache、p等を保持。群数L²+1=104,977。全ordered四文字語の表は作らない。|旧workerごとのprivate/PSS/COW増加は未取得。fork後に1.689 GiB×21が全て物理複製されたとはいえない。|
|初期化順序|旧[diagnostic](../track_b_c1_5a_h4_limited_resource_pilot_20261021/diagnostic.py)のNumPy/Qiskit import・basis/model復元・sampler setup後、[run.coordinator](../track_b_c1_5a_h4_limited_resource_pilot_20261021/run.py)がfork contextで21 workerを作る。samplerはglobal Gで継承、batch indexだけsubmit。|multithreaded native library状態を継承するforkは調査候補。segfault/deadlockが発生した証拠はない。|
|群選択|[dyadic_index](../../../src/trottertracks/algorithm_codesign/return_aggregation.py)はlaw合計確認、全Fractionのinteger count化、listと走査をdrawごとに行う。旧P5では104,977群が対象。|大量の再走査・一時allocation・参照count更新はCPU/COW候補。worker終了の直接原因とは未確定。|
|出力|旧workerはpathをmemoryのrecords listに蓄え、batch完了時gzip保存。Python例外handlerはprefix保存する。event/angleはcontent-addressed個別保存。|突然のsignal/killはPython handlerを通らず、完走batch未保存でも処理済trial数は不明。0 accepted/0費用と補完できない。|
|失敗点|[stderr](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/closed_P5_full.stderr.log)と[terminal](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/closed_P5_full/terminal.json)はas_completed→future.resultのBrokenProcessPoolを示す。|個別worker exit code/signal・最初の異常PID・直接のOS原因は記録なし。|
|cleanup|旧coordinatorはabort Event、future cancel、存命の自分のpool worker terminate、shutdownを行う。rootも自分のmethod process groupだけ停止する。|cleanup後に観測したexitから元のfailure signalを復元できない。|
|資源|旧[post_failure_cgroup_observation](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/post_failure_cgroup_observation.json)のsession/user memory.eventsはoom/oom_kill=0。|上位cgroupのpeakは他jobを含む集計。job専用OOM、OS kill、allocator問題、共有memory/serialization、同時負荷を断定する材料はない。OOM否定の完全な証拠でもない。|

結論：**根本原因UNRESOLVED**。確定したのはworker突然終了と観測不足まで。P5固有のfork継承、巨大lawの反復走査、一時allocation、native初期化状態を今回の技術対策で避け、終了状態を保存する。原因完全特定は今回の進行条件ではないが、新方式の安全性を数学的に証明したとはしない。

旧rawを新たに展開・科学再解析していない。保存summary、setup、terminal、原ログ、SOURCEと既存archive indexのみを読んだ。既存raw archiveのbyte保全照合は科学再解析とは分ける。
