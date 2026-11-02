# 接続・計測の実装と限界

新driverは標準libraryと既存準備helperだけを通常importする。CLIはpreflightのみ。
science entry `execute` は入力・固定policy・16 cell・SOURCE pin・未決条件を検査し、
外部の別途承認済みatomic admission portなしでは科学moduleをimport／構築しない。
boolや候補編集だけでは起動できない。外部portは新claimと採択契約digestを結ぶ境界であり、
本準備はportのauthority発行／実claim消費を実装・実行していない。悪意ある任意Pythonのattestationではない。

Layer Aは元G7 `make_generator/sample`。所有instanceのpacket/eventだけを透明callbackで包み、
同じ引数・返却値・draw順を保つ。Fullのpacket未到達Noneはraw zero、到達後Noneはcoin reject。
canonical eventではgroup indexと元raw wordを捕捉し、後の既存event再queryへ渡す。
追加のselector・root・law・乱数drawで分類しない。最小observerも実費に含める。

Layer B Fullは既存F3 `prepare/sample/angle_for_event` を呼ぶ。P0–P5、SOURCE/state guard、
TraceBits、event照合、angle時のevent→packet再queryを残す。全方式に共通AuditParityを追加し、
canonicalは実objectの既存root/law witnessで有限preflight、実保持state seal、共通event invariantと
angle時の原event再query＋既存F3 enclosureを用いる。Fullに固有の監査とcanonicalの監査は別の実処理であり、
同一手順／同一費用と偽装しない。canonical未取得費用を0にしない。

## 排他的な費用座標

`Meter` はnested childのinclusive時間を親から一度引く。固定件数のaggregateだけを保持する。
粗いpartitionはsetup、R/C/A別generator+audit、accepted angle/requery、R/C/A別evidence、
serialize／chain hash／write／file hash／flush／finalize、memory observer。
instrumented passは固定SOURCEのPython call/returnをfilename:function単位でも排他的に記録する。
parent、全child、law/sort、root、selector、reduction、weight、canonical conditional law、F3 guard/hashを
実呼出として追う。CレベルFraction/GCD等はそのPython呼出元に含まれ、正確bit演算数ではない。
coin rejectでもpacket内の全child law/sortがcoin前に払われる元順序を保持する。

粗いstage合計と細かいfunction合計を**足さない**。それぞれown childの外側process CPU/monotonic wall total以下を検査し、
差をunattributed/observer residualとして残す。hook、時計、callback、trace、証拠・保存自体の効果を除去済みとは言わない。
同じseed/Nのprofile-disabled baselineを別passで実行し、instrumented totalと並べてobserver効果を見る。
両者の時間一致や非負差を要求せず、差から純粋なoverheadを認証／差引きしない。両passの科学callを明示予算へ含める。

各arm/layer/phase-passを新しい所有Linux fork childへ分離し、HWMの前arm混入を避ける。
X/Yは一つのobjectを共有し、初期・X前・Y前・最後にsource/state identityを照合する。
parentの共有source確認／budget load／module load、dispatch CPU、最後のoutputは別座標。
dispatch wallはchild wallを含むため加算しない。最後の自己receipt書込時間はnullとして明示し、無料としない。
arm単独cold取得費とは別の共有scheduleである。OS file cacheをcoldと保証しない。

memoryはown child RSSを4096 trialsごとに観測し、own HWM/final RSS、最大行bytes、output bytesを別欄へ残す。
RSS samplingは連続peakではない。HWMはそのchildのbaseline/source/importも含む。runtime memo cacheは実際に無効なので0bytes、
未知OS cacheやallocator全量は0と置かない。unbuffered writeでPython file bufferを省き、ページcacheを無視したRAM値とはしない。

## W2c-Aへの接続

同じ層・pass・CPU単位で、観測online合計T_onlineからbar_c=T_online/Nを作る将来の保存後解析を区別する。
Sは実setup、Ωはonlineに含めないfinal output、Δ=S_F−S_a+Ω_F−Ω_a。
`(N_a−N_F) bar_c_a >= N_F(bar_c_F−bar_c_a)+Δ_a` は完了した当該workload総額比較の書換えである。
X/Yでは各axisのN bar_cを合計し、共有setupを一度だけ課金する。
これはW2c-Aの母平均μのcertificateではない。旧α_axisをCPU CIへ流用しない。
実行後の値の縮約・比較は承認scopeで行い、今回保存科学値を再集計しない。

CPU秒、wall、bit上界/random bits、RSS、output/I/Oは別vector。CPU→bitやCPU+Tの任意scalarは作らない。
F2理論上界、旧B1/B2期待係数、実観測費用を混ぜて総量子／End-to-End優位を宣言しない。

## 資源と保全

静的読解で支配項はFull局所有理packet、BのSOURCE/state serialization/hash、per-accepted角度有理級数・再query。
既存科学lawを変えるvector化／packet memo／conditional law cacheを追加せず、保持state共有、有限aggregate、
all-N streaming、既存保存Nを利用する。比較worker同士の競合と最初の失敗後の追加callを避ける固定順序scheduleを候補とする。
これはCPU/RAM/worker/wall上限ではない。速度向上倍率は未測定。人工検査・公開照合は独立に並列化した。
観測時点CPU32、RAM62GiB/available約49GiB、disk空き約375GiB、CPU/memory pressureのavg10=0。
実科学launch前には改めて空き資源／I/O競合を確認し、他jobや設定を変更しない。

新科学経路は実行前承認待ち。mock成功は実G7 correctnessやCPU優位の証拠ではない。mandatory STOP。
