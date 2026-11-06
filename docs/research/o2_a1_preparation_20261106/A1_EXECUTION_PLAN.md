# A1：一roundの最小科学検証計画（未固定・未実行）

2026-11-06 JST。[preflight](A1_PREFLIGHT.md)で確認した保存入力・SOURCEを用いる提案。
**今回の許可は実装・人工テスト・公開まで。以下のtrain/cal/prod、実lawの科学入力への適用、variance/N/費用評価、build/compileは開始していない。**
ユーザーの明示的な科学実行指示が別途必要であり、承認ファイルや新grantは要らない。

## 1. 固定する対象と実行判断の空欄

| 条件 | 推奨・既存参照 | 実行前に必要な判断 |
|---|---|---|
| H/model/state | linear H4 1.00 Å STO-3G legacy DF12 L_D3、保存snapshot・4電子physical ground | 未近似Hへの結合を閉じるか、保存DF model限定の診断を明示して採るか。H_D/PF control groundへ置換しない |
| round | opt2 m0/q1、δ=.02、r1/K2、X/Y両軸 | この一roundを採るか。代替m1/m2はpreflight表の既存条件のみ。全18roundへ拡張しない |
| target/weight | 元finite attenuated event-mean、元law/normalization/phase、canonical raw weight | actual-q対algebraic target差、PF/RTE/native・centering biasの扱いと残余幅を固定。黙ってweightを変更しない |
| circuit/profile | 元support_run_le_1、全D halves/中央RTE/control/ancilla/readout、opt2の既存compiler条件 | measurement engineと同policyの価格取得経路を指定。標準full-basis価格との混在不可 |
| resources | 元opt2主指標compiled RZ countを参照。CX/depth等は補助、T/1Q取得状況は別欄 | RZを今回も主指標とするか。Tが必要なら別のnative価格scopeが必要で、今回未採択 |
| precision/confidence | m0保存epsilon=.266146579872968、alpha/axis=2.8897922699779505e-08等を参照 | model条件付きの旧許容幅を採るか。raw/strata/CV、X/Y、cal/prod・価格不確かさへの失敗配分。新しい精度・成功率を自動採択しない |
| controls | 結果前に決めるlocal label/pair bank、order-onlyは対照 | 実label ID、occurrence、positions、order、有限集合とfit規則を固定。後からfeatureや分子を探索しない |
| finite測定scope | X/Y別の有限train、独立cal、fresh prod | train/cal数、層別pilot/各層割当の学習法、productionの十分N計算法と許可される測定・価格取得総範囲を固定。新数値を本書で発行しない |

旧N=510/axisは旧±1 Hoeffdingの保存結果で、pilot予算、最適N、CVの十分Nではない。
保存alpha/epsilonを使う場合でも、calibration等の失敗予算を追加して総額を超えないよう分割する必要がある。
従来のepsilonが物理Hへの保証を持つとはしない。単位半径仮定を維持した**条件付き座標診断**と、正当な半径/biasを戻したRPE保証を区別する。

## 2. 科学実行に先立つ技術接続

追加した[adapter](../../../src/trottertracks/algorithm_codesign/a1_control_variates.py)は実行runnerではない。
将来の明示scope内で次を接続する。量子builderやsamplerを再実装する必要はない。

1. 保存snapshot・元H/stateの参照をmeasurement engineへ渡す。元groundの利用と、完全state vectorを推定器へ公開することは別。新しいH/state生成が必要ならその科学処理も実行scopeへ明記する。
2. 元samplerの実order/label CDFを同じ演算順で取得し、law_from_choice_cdf→OccurrenceLawへ渡す。support、実qとfinite targetのbias、factorized random-word仮定を確認する。sampler置換で解決しない。
3. 元pathのevent metadataをproducts→rotation順に保持し、wrapper/trajectory fingerprint、axis、trial/stream、実bit、元weightをbind_wrapper_measurementへ渡す。target/round/選択policyと出所の対応も保つ。
4. train/cal/prodとX/Yの独立取得を接続する。異なるstream名だけで独立性を認定しない。元RNG経路・fresh trialの規約を守る。X/Yの価格較正に同pathを共有することを、測定noiseの共有と混同しない。
5. 同じ情報で実行できる条件付きstrata取得・事前割当を接続する。古典pathの測定前棄却なら棄却回数・生成費を記録し、空層を無視しない。条件付き取得は固定baseline用で、新law最適化ではない。
6. confidence計算と実full価格のadapterを追加する。将来の科学呼出しは許可後のみ。既存price rawの範囲外に必要なbuild/compile、native synthesisを自動許可しない。

実装されたデータ分離は再利用検出の必要条件であり、PRNG独立性や物理入力を認証する管理機構ではない。
CDF・control平均・正当なcentering/bias条件が閉じなければ、科学測定へ進まない。

## 3. train → freeze → independent calibration → production

**学習pilot**：各axisについて、固定bankと実Hadamard bit/weightだけを使う。fit_and_freezeは既存NumPy lstsqによるintercept付きOLSを行い、slopesを凍結する。
未知mean/path信号/最適lambdaを渡さない。interceptは学習のためだけに使い、CV targetに加算しない。
rank不足は最小norm解という技術挙動に過ぎず、有効性を保証しない。固定条件を外れた不安定性・range膨張は縮小/中止材料とし、無許可のridge/feature探索で救済しない。
計算されたbinary64係数をexact Fractionとして保持するため、range/biasもその実係数に対応する。大きなFractionの古典費・memoryを無料にしない。

**係数凍結後の較正**：固定されたcalibration数でfresh outcomeを取得する。λ固定条件の下でIIDな補正値を評価する。
補正後rangeは `[-W,W] - sum lambda_j([0,1]-nu_j)` の保守box enclosureを使う。joint制約で狭める場合は別途正当化が必要。
未知の真varianceや1−mu²をoracleとして入力しない。

**production**：calibrationから十分予算Nを決めた後、freshな独立trialで平均する。
calibrationをproduction meanに混ぜない。同じデータで学習・較正・成功判定を無補正で兼用しない。
許可範囲を超えるN/価格callが必要なら、実行を自動拡張しない。
各段階の独立性は統計設計であり、段階ごとの新承認・証明書・grant発行は要求しない。

保存path期待値は実bitの代用品にしない。simulationを後日許可する場合も、measurement engine内部のstate/path期待値をfitやcontrol選択へ漏らさない。
simulationの古典実費と、推定される量子interrogation費は別座標に記録する。

## 4. confidenceとshot会計の適用条件

これは[既知のbounded標本保証](https://arxiv.org/pdf/0907.3740v1)の尺度変換と実行条件の整理であり、新定理・新Nの数値計算ではない。
一axis・一方式でλを凍結し、出力range長をD、独立calibration数をn_cal≥2、不偏sample varianceをs²とする。
Theorem10の上側標準偏差boundから、失敗δ_cal以下で

`V_upper = min(D²/4, (sqrt(s²) + D sqrt(2 log(1/δ_cal)/(n_cal−1)))²)`

をvariance上界として使える（IIDなのでE[V_n]=Var）。calibration sample varianceそのものは上界ではない。
meanのcentering誤差は `b_center ≤ sum |lambda_j| e_j`。
PF/RTE/native/finite-law等も含め、正しく換算した共通targetのcoordinate残余をε_rem>0として別に保持する。
保存epsilonを採る場合、既に予算化されたPF/RTE分を二重控除しない。追加centering/実law/native残差のうち未包含の部分だけを戻し、元Hへの保証の不足は別に保持する。

fresh productionに対し、bounded Bernsteinの二側十分条件として

`N_suff = ceil(log(2/δ_prod) [2 V_upper/ε_rem² + 2D/(3 ε_rem)])`

等を用いる計画。δ_cal+δ_prodと他の失敗配分を該当axis予算内に戻す。
このNは**十分shot数**であり、最小shot数ではない。float実装時の丸めと対数/平方根の扱いは必要な保証水準に合わせる。
今回これらの関数実装・科学値入力による計算はしていない。

rawにも同じcalibration＋Bernsteinを認め、trainは不要。独立calibrationなしの事前固定N＋Theorem4の両側empirical Bernsteinを選ぶ道も認める。
後者は `sqrt(2 s² log(4/δ)/n) + 7D log(4/δ)/(3(n−1))` の固定n boundで、観測後の無補正peeking/任意停止には使わない。
CVだけに新しいconfidenceを与えず、rawに不要な学習費を課さない。

strataでは正の全層を既知質量p_hで合成する。独立calibrationと固定allocationの下で各層の誤差・失敗を有効に配分し、全座標誤差へ `sum p_h ε_h`、失敗へunion boundを戻す等の正当な方式を固定する。
経験的なNeyman/cost-aware配分を学ぶなら、その独立pilotも課金し、配分凍結後のcal/prodを分ける。真varianceからの最適配分は与えない。
層別により厳密な条件付き取得が必要になることと、元targetをp_hで維持することを区別する。

補正値は非binaryである。旧V2の二項PMF/tailをCVのfinite-confidenceへ使わない。
既存decoderの決定論部分は同じaxis meanへ戻す後段として再利用できるが、一roundの診断からbranch/energy成功を認定しない。

## 5. 強いbaselineと全費用

| 方式 | 必要な取得・条件 | 課金 |
|---|---|---|
| raw canonical | 同じq・回路・target、共通に利用可能なvariance-sensitive保証 | 必要なcal/prod、または正当な事前固定production。不要なtrainを課さない |
| order-only対照 | 既知order質量、正層の条件付き測定、固定または有料学習した配分 | 全層の取得・古典選別・必要pilot/cal/prod |
| 強いorder/label strata | A1と同じ情報、必要ならorderでさらに細分、取得可能な小分割の飽和推定・cost-aware配分 | 層平均/配分を無料で知らない。取得・学習・cal/prod・古典費を実際に計上 |
| A1 | 既知factorized bank、実outcome pilot fit、独立cal、凍結CV prod | train+cal+prodのfull価格、moment/fit/補正・記録・memory |

既存IS/Closed/CTSの利益をCVへ二重計上しない。適合する既知対照を排除しないが、今回新law・RA実験を追加しない。
indicator bankが小さな層別表に還元される利益はCV独自の追加利益と呼ばない。

各resource aの費用は、方式ごとに `G_train^(a)+G_cal^(a)+G_prod^(a)`。
full no-prep interrogationにはD half-steps、中央RTE、制御、相対phase、ancilla、X/Y readout、追加操作を含む。
full価格にwrapperが含まれる場合、H/Sdg/readoutを再加算しない。状態準備のみ除外する。
標本数がcalibrationに依存する場合は条件付きproduction予算と費用から会計し、任意停止で安易にE[N]E[C]とはしない。

主指標RZ、CX、depth等の元6指標、T、1Q、古典setup/online秒、moment/fit、取得engineの古典実費、memoryは区別する。
T/1Q等が未取得ならUNKNOWNであり0ではない。depthのshot合計は集計work量で、並列wall-timeとは違う。
共有実験データを複数方式の比較に再利用する場合、実験全体の取得費と各方式を単独運用する費用を分けて記録する。
TrialCost/account_costsは同方式内のtrial重複を拒否する小接口。承認台帳ではない。全方式のraw/pilot費を混ぜた一つの分母を作らない。
classical-only setup/棄却費は外部の実測欄へ一回計上する。quantum trialを捏造して載せない。

## 6. 継続・不改善・中止・STOP

**継続材料**：同じtarget/誤差/失敗確率と情報アクセス、最も強い実行可能strata後にも、pilot/calibration・range・価格不確かさ・古典費を含む総費用差が残ること。
sample covariance/varianceだけの低下や、旧range-only rawとの差だけでは不足。新規性・全PR優位・全18round利益は別判断。

**不改善／縮小材料**：量子conditional noise支配、within-order情報が弱い、fit/range/cal費が節約を相殺、strataと同等以下、centering取得が過大、未知state情報が必要。
結果後のfeature/precision/分子/r/K/time探索で救済しない。必要なら現在の候補を終了し、新scopeをユーザーが判断する。

**科学取得前のSTOP**：H/stateの許された保証範囲が不明、実q/centering/finite-target bias未処理、比較target不一致、正残余幅なし、未知価格を0扱い、confidenceや有限測定scopeが未固定。
**科学取得中のSTOP**：入力破損、SOURCE/条件の不一致、同測定の混入、対応付け破損、実異常・継続不能な資源不足、許可scopeの超過。
一時的CPU競合・PSI上昇、無関係worktree更新だけで正常計算を停止/再起動しない。共有設定・他ユーザーjob/filesには操作しない。

**今回のmandatory STOP**：preflight、最小adapter、人工テスト、計画、専用branch公開・再取得照合まで。
S2/R1再実行、C1解凍、旧grant再利用、新測定/N評価/compile、重要科学レビュー、科学的GOは行わない。

## 7. 次の担当

**ユーザー／GPT**：§1の対象・保証範囲・control/strata・精度/失敗配分・測定数と価格取得範囲を判断する。保存model限定でよいかを特に明示する。
**Codex**：上記条件を固定した明示許可の後だけ、不足する小adapterと当該scopeの科学取得・統計・費用評価を進める。新しい重要科学レビューの開始許可は科学実行許可と別である。
本書への同意やcommit/push許可を科学実行許可とはしない。各統計段階ごとの再承認や新しい承認発行手続きは求めない。
