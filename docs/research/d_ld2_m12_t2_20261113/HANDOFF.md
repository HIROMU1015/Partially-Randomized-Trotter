# T2 → B/C：単一pathの価格と実行可能性

2026-11-13 JST。**T2はscalar binding不一致でSTOP。sampling/path/event/新compileはすべて0、X/Y S/C/Dはnull。** [REPORT](REPORT.md)／[T2_PATH_RESULT](run01/T2_PATH_RESULT.json)／[WORKER_STOP](run01/WORKER_STOP.json)／[CHECKS](run01/CHECKS.json)。実行SOURCEは8291f16ceb6b88cc8a43dbe508db419b9a26f373。A a9ab固定PR-LD2/m12、cohort D_LONG_TIME_FEASIBILITY_V1/index0/X,Y/上限2 compileの一attemptだけを起動し、retryしていない。

今回供給するのは**失敗したrequestと部分STOP情報**であり、完成した`PR_CANDIDATE_PAIRED_PATH_PRICE`ではない。親が保存したT2_PATH_RESULTのstatus=PARTIAL_STOP、partial={}、mean/CI=null、qualified_expectation=falseを維持する。元Bのmean又はCの正常pathとしてmappingしない。必要なら既存D.pending_price_records(request)の未取得状態を使い、今回は値付きB packetを作らない。

requestはconfiguration/physical ID、q/R/piece/K/time・law、expected scalar、全profile/lowering、入力/実行SOURCEを保持する。保存H/state/snapshot復元後、NativeRuntimeの `float(actual scalar).hex()==float(expected scalar).hex()` が不成立となった。expected=0.2223759094107018、actual値と差の大きさは未保存/UNKNOWN。これはC5ccのkeyword forwarding問題より前の**D実preparation適合性**である。Cにscalar転送を追加するだけで解消したり、旧LD1価格をLD2へ流用したりしない。

観測wallは失敗attempt全体4.694852秒、worker peak RSS354,508,800 bytes。input復元2.306109秒、contextは途中停止。巨大native/hash/full compileのtime/RAMは未測定である。この短い時間・RSSをm12 compile性能や速度改善の証拠にしない。

新private領域 `/home/AbeHiromu/projects/d-long-time-price-t2/20261113/run01` は空で保全。元入力2ファイルのSHAと科学実行コードは停止後も一致。旧event/cacheの読取り段階にも未到達で、cache hit/miss/fingerprintを補完しない。

次はユーザーが[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)のpreparation-only診断を判断する。実値・差・preparation/hash/定義が確認されるまでD/Cは正式m12価格をUNKNOWNに保ち、Bは価格欠測をGへ0代入しない。診断後にもT2再試行・平均取得は別承認である。C5cc scalar bindingの旧差分とA/BのN/claim課題も残る。最終公開照合後 **mandatory STOP**。
