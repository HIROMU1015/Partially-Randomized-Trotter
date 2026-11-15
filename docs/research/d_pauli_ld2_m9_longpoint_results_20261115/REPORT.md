# PR-Pauli LD2 m9：独立2 pathの記述的価格比較

fresh 2 path・新規4 compileの取得と元実装gateが完了し、m9-X/Yの2件を記述比較した。両軸とも観測平均は2,679,913.5 RZ、固定予測の符号付き相対差は**+0.274418%**（表示丸め）。モデルPASS/FAILは付けず、独立有限cohortの差として公開する。

対象はH4直線1.00 Å/STO-3G/tol-only DF10、LD2（one-body＋DF fragment0,1）、S2、実step0.1、K2 normal。
m9はq=512/R=526/T=51.2、exact T=`3602879701896397/70368744177664`。
physical ID=`RMSE:978923356c4e83c897d03442fb488997d32047237444f95757658f61321b2729`。
PR_STEP_V1の順序は`[one,pf_step_count,event_count]`、値は`[1,512,269312]`。

採択済み案Bの目的はS526とR=526を共有する長いq=512での有限価格比較。
価格は状態準備を除いた測定付きfull controlled Hadamard wrapper全体のcompiled RZ数。
元LD2辞書`dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7`、
Pauli D2、symmetric-directional、c_H＋保存s_Rを総Tで一回、
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/P0を維持した。

## 観測前固定と取得

凍結モデルb74eのX/Y別係数を再fitせず使用し、各軸の予測は同じexact値
`242777084125865/90343469` RZ。別々のaxis/model locatorを保存している。
予測正本は`3f6e6604a4877dc882fd1ee7538a854464160a5b:PREDICTIONS.json`（本directory）。
作成11:24:05 UTC、commit11:24:23 UTC。pushとremote blob一致を確認した後、
同日11:24:55 UTCに取得を開始した。実行入口自身もremote ancestorと予測blob一致を確認する。

SOURCEは`650d8d98486c6ff479629ffa0daf93e6e976b48f:scripts/run_pauli_ld2_m9.py`、
元caller/core/receiptは13abのC2専用detached checkoutから無変更で利用。
旧m0/m5/m6 validatorを緩和せず、m9のcanonical request・SOURCE・axis・seedへ別途束縛した。
[準備・重複照合](SOURCE_PREPARATION.md)、[人工4 test](TESTS.txt)は公開済み。
追加のcanonical m9 guard人工照合では正しい2軸を受理しwrong-axisを拒否、実H4取得0。
人工試験は科学的な価格精度の証拠ではない。

§10.5の関連公開履歴・保存先・稼働job metadata照合では同等m9予測・receiptを確認できず、
独立holdoutとして再利用可能な価格は0。旧W/CAL/holdoutは元用途を維持。
新cohort=`D_PAULI_LD2_M9_LONG_NORMAL_N2_V1`、index0/1各512 stepを元seed規則で固定。
1024新seedはcohort内unique、旧CAL/holdout/W seedとのoverlap0。
各pathはnormal samplerを1回だけ使用し、X/Yは同一pathの別軸で、独立path数は2。

件数：予定fresh path2/個別軸4、再利用compile0、新規compile4、cache hit0、有効path2/個別軸4、欠測・無効0。再sampling/retry/追加path/追加compile/再fitは全て0。

## 個別価格・gateとexact比較

| path | X RZ | Y RZ | 取得・実装gate |
|---|---:|---:|---|
| 0 | 2678974 | 2678974 | 元7項目とbit-order成立 |
| 1 | 2680853 | 2680853 | 元7項目とbit-order成立 |

全wrapper/compiled QPYは専用private cacheに保持し、4 receiptに元record・所在・hashを保存。各pathの全7 gateの最大誤差は4.33e−10以下（元閾値：compiled-vector 1e−7、他1e−9）。canonical guardは2軸を`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`として受理。これは科学的価格精度や全law資格の認定ではない。

`P_a`は予測正本、`O_a=(C_a,path0+C_a,path1)/2`。
`d_a=P_a−O_a`、`|d_a|`、`d_a/O_a`、`|d_a|/O_a`をexact正本とし、%のみ表示時丸め。
一方のpathが無効・欠測ならその軸は比較不能、1 pathで代用しない。O=0の相対誤差は未定義。

| axis | P | O | P−O | 絶対誤差 | (P−O)/O | 絶対相対誤差 |
|---|---|---|---|---|---|---|
| X | 242777084125865/90343469 | 5359827/2 | 1328803831867/180686938 | 1328803831867/180686938 | 1328803831867/484225364419863 | 1328803831867/484225364419863 |
| Y | 242777084125865/90343469 | 5359827/2 | 1328803831867/180686938 | 1328803831867/180686938 | 1328803831867/484225364419863 | 1328803831867/484225364419863 |

[COMPARISON.json](COMPARISON.json)のexact分子/分母が正本。両軸の符号付き・絶対相対差の表示は+0.274418% / 0.274418%。

数値許容差・モデルPASS/FAILを付けない。実装gateの受入と価格精度判定を区別する。
決定論PFの2%や95%を流用せず、同じ観測から後で選ぶ閾値を事前独立検証と扱わない。

## 資源実績と制約

全体の取得wallは3825.209秒（約63分45秒）。path0/1のworker wallは3820.240/3806.267秒、peak RSSは7,538,860/7,601,256 KiB、双方exit0/signalなし。

| 段階（秒） | path0 | path1 |
|---|---:|---:|
| sampling | 24.461 | 24.478 |
| full-space oracle | 36.467 | 39.241 |
| compose | 377.791 | 382.505 |
| index0 checks（内部simulate1〜4を含む） | 1050.956 | 1037.480 |
| X compile | 453.391 | 463.252 |
| Y compile | 449.297 | 436.002 |
| raw保存 | 24.621 | 24.756 |

両pathのevent-matrix cacheは各184 miss/269128 hit。追加した並列公開照合は別PIDでpath0/1が4.756/4.719秒、公開全体7.310秒。公開rawはgzip16,376,709/16,394,299 bytesで、hashと元receiptを照合した。

初期の並列・cache対策は適用したが、path内X/YとQiskit Statevector.evolveは逐次であり、最大限の高速化が済んだとはいえない。利用者の追加指示で、稼働SOURCEを変えず[2 processのraw公開照合](PUBLICATION_ACCELERATION.md)を実装・人工2件で検証し、実取得結果にも適用した。重いevent配列は親processへ転送しない。これによる全体速度比は未測定。

独立2 pathをprocess並列、X/Yは固定coreの順に処理。worker内の純粋event_matrix memoと
既存の同一byte fingerprint writerを利用。RNG、event順、D2、simulator、compiler、gateは変更なし。
[手法索引](../../PERFORMANCE_METHODS.md)。比較用の未高速化m9を走らせていないためspeedup比は不明。
協調stage logと親observerを使い、CPU生存だけで完了・gate成立と判断しない。

自然eventは各path269312件すべてorder0で、order2は0件。各512 stepのD2 aggregate source global phaseも非ゼロ0件。元event/選択label/D2 phase metadataをlossless rawに保持した。実LD2のorder2 product・非自明位相の科学的観測は依然UNKNOWN。希少eventを探索・強制生成していない。

結論はこの新m9有限cohortの予測差まで。母平均・CI・全13roundのE[C]・N/G・winnerは未評価。
m5/m6の既存独立比較を更新せず、b74eの係数と4予測、raw、旧STOPを保全。
希少event調査、追加sampling/compile、再fit、別round・LD1・Full-Hへの移送は行わず、公開後mandatory STOP。

固定raw：`a7fdd200da8d499e93965d63b5b2e5b01e0b8940`。canonical価格/evidence：`c9986c82f866bfa57a7142124746f89fc42f9838`。詳細locator・読取方法・ロードマップ差分は[HANDOFF](HANDOFF.md)。
