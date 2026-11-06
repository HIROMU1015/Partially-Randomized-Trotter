# A1対B1-OL：縮小予備検証の科学実行契約草案 v1

**提案／未採択／実科学未実行。** 今回許可されたのは技術調査・設計・人工test・公開だけ。現行jobの停止、科学条件の正式変更、新測定/compile/再実行は許可されていない。

## 1. 固定targetと入力

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y、理想gate/保存DFモデル/独立uniform53-bitモデル限定。
N1 744e1a09661ebba6841fe5faf456f88c41c47780、N2 5e9bd08068738cdf1070bdc43bee8194b113ae33、engine fdf00cee3c87d804845497afc89ac6a71f3b3f59の保存identity・proof・state・actual Q・selected full policyを再利用する。
一次targetは同じ複素平均。N1準備/CDF/ν、N2 solve/spectral、engine認証は再計算しない。保存厳密b_engineを使い、float/GPUへ置換しない。

候補campaign ID=`A1-N3-PILOT-H4-m0-q1-20261106-v1`、候補seed=`20261108`。いずれも今回未採択・未生成。旧seed20261106/20261107の消費済みstream/word/bit/係数を移さない。
SOURCE/環境/入力/compiler/stream一覧とfresh outputは、実行許可後に人工検証・公開照合して実入力前に固定する。
現行jobとの関係は先に明示する。停止許可を別途得て保全する案を推奨するが、今回は停止しない。現行jobが自然終了すれば、その状態・費用をそのまま保持する。

## 2. Step 1：学習と技術成立

旧397件および新BのX/Y各320学習は参考資料だけにする。候補既存係数のコピー採択やraw bitsの混入はしない。
新pilotでは3-control固定、各軸の5正層から**16 fresh conditional shots/層**、計80/軸、160 shots。既存OLSを各軸一回だけ使い、intercept込みrank4とD_CV=2+Σabs(λ)≤4を確認して係数をexact binary64 rationalとして凍結する。
D_CV≤4は今回も推奨する有限range guardで、精度達成や有望性の証明ではない。失敗時はSTOPし、control削除、追加train、再学習、clip、seed変更で救済しない。
5層による4種類のcontrol pattern被覆はrankの構造条件を与えるが、応答・係数精度・真のQでの最適OLSを保証しない。
各bitは既存facadeと認証済みexact engineで取得し、full selected no-prep RZをtrainにも課金する。値/時間/RSS/rank/係数/rangeを保存し、次段階前に凍結する。

## 3. Step 2：独立probeと固定サイズ評価

| stage | A1 | B1-OL | 両軸合計 |
|---|---|---|---:|
| 学習 | 5層×16/軸 | なし | 160 |
| 配分probe | なし | 5層×8/軸 | 80 |
| fresh評価 | unconditional Qで128/軸 | 全5層に合計128/軸 | 512 |
| 合計 | 416 | 336 | **752** |

A1とB1-OLの評価shot数は各軸で同じ128。両方式の学習/probe費は実際の固有費用として別記する。RZ予算が同額であるとは称さず、shot数だけで優位性を判定しない。
probeのraw bitとfull RZだけから各層の不偏sample variance s_h²と平均RZ cbar_hを取得する。真のBorn/path信号/真variance/state/E0/gapは入力にしない。
probe8/層はvariance/priceに少数の反復を与える低予算案。高精度分散推定や全希少事象の観測を保証しない。

### 質量・分散・価格を使うB1-OL配分案

p_hはN1 actual-Qの保存全5正層質量で、厳密和1。現在この草案では値を再計算・再配分していない。
各軸でprobeだけから以下を固定し、評価結果で選び直さない。

```text
v_proxy,h = (1 + min(s_h², 1))/2
c_proxy,h = max(1, cbar_h)
w_h = p_h sqrt(v_proxy,h / c_proxy,h)
n_h = 4 + largest_remainder_h(108; normalized w)
```

Σn_h=128、全n_h≥4。v_proxyは二値±1のvariance上限1との固定1/2 shrinkageで、**信頼上界ではない**。不偏sample varianceは有限nで1を超え得るので、proxy用capとraw s²を別々に保存する。s²=0でもzero varianceと認定しない。
c_proxyのfloor1は配分scoreのためだけで、観測価格を書き換えない。RZ0を実測した場合も請求値は0のまま保持する。
旧return_aggregation.root_intervalの128-bit dyadic enclosureの中点をdeterministic scoreへ使い、largest remainderのtieはN1固定cell順。これは配置heuristicであり、厳密最適quota/真のNeyman最適性を主張しない。
重みは分散×費用の連続proxyに対応する。固定shot数で分散だけを最小にする配分と同一とは扱わない。整数floor4による希少層の診断overheadも全費用に含める。
正層を削除・無断再正規化せず、取得lawは常にQ(ω|h)。quota配分はQ自体の変更ではない。
**実p_h/probe/価格へこの式を適用する数値評価は、別途のpilot実行許可に含めて初めて行う。** 今回は人工入力だけで検査する。

### 評価と診断指標

凍結λでZ=B−λᵀ(g−ν)。A1はfresh unconditional QのZ平均、B1-OLはΣp_h Bbar_h。ν/質量はN1のexact値、X/Yとbit0→+1・bit1→−1、位相/orderを保持する。
評価のsample varianceから次を**診断用plug-in**として保存する。

```text
Vhat_mean,A1 = s_Z² / 128
Vhat_mean,B1 = Σ_h p_h² s_eval,h² / n_h
C_eval,m    = 評価stageで実際に取得したfull RZの和
score_m     = Vhat_mean,m × C_eval,m
```

scoreは期待費用×母分散の不偏推定量や精度保証ではない。分散/価格の不確かさと有限quotaの影響を併記し、raw値・層別n/s²/costを隠さない。
A1/B1の平均、sample variance、診断standard error、RZを軸別に報告する。平均が一致するだけで改善とはしない。
両軸でA1のscoreが小さい傾向ならStep 3検討候補、一軸ずつ異なる/比が未定義/zero分散推定で不安定ならINCONCLUSIVE、有利傾向が見られなければNO_FAVOURABLE_SIGNALとする。全て探索的分類で、わずかな差を有意差と呼ばない。
s²=0/分母0、all-equal bits、極小層・floorによる影響は必ずフラグを残す。zero varianceだけを根拠に有望と分類しない。小pilotは結論保留を許す設計であり、結論を出すための追加shotsを自動で足さない。
精度ε/同時αはこのpilotでは採択しない。旧ε=1/8・α=1/20、各単位512 cal、8192 cap、十分N/信頼区間を自動継承せず、`PILOT_DIAGNOSTIC_ONLY / NOT_PRECISION_CERTIFIED`を明示する。

## 4. 独立性・再利用の境界

method×axis×stage×cell×path/bitごとの独立domainを、全取得前に固定する。候補はA1 train10、B1 probe10、A1 eval2、B1 eval10=32 domains/64子stream。
既存Domainのtrainは学習/probe、prodはpilot評価に対応させ、外側ledgerにsemantic stageとcampaign/attempt IDを付ける。`prod`という内部型名を保証付きproductionの意味へ読み替えない。
旧require_disjointはstage/stratum streamの衝突を検出するが、外側campaign provenanceまで証明しない。future runnerは旧campaign/欠落campaign IDのrecordを取り込み前に拒否する必要がある。
A1 train、B1 probe、両方式評価、X/Y間のデータ・乱数を共有しない。学習/配分freeze前に評価を使わない。真の信号/state/確率をCVへ渡さない。
旧データや価格を無料の新pilot入力にしない。共通N1/N2/engine証明の再利用と、旧測定の再利用は別である。

## 5. 費用・有限call scope・保全

一条件・2方式×X/Y・一pilot、logical trials≤752、exact engine/Born≤752、path≤64word/trial・別bit word1、word全体≤48880。
実際に取得したtrialだけ同selected full no-prep価格を取得し、cache missのみcompile。compile上限も752、selected full buildは測定用と価格用で計1504回以下。全supportの列挙や未取得pathのcompileなし。保存bundle decodeは一回、N1/N2/認証の新計算0。
Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、backend/layout/routing/coupling指定なし。状態準備だけ除外する。
cache hitでもshot RZを課金し、新run専用cacheを使う。学習、probe、評価、共通setup、generation、fit、配分、engine、compile、cache/lock、wall/RSSを別記する。
各trialは開始前journal→bit fsync→価格の順序を維持する。価格欠測はUNKNOWN、既知RZ和を完成費用としない。中断attemptの古典費も残す。
全stage RZは方式固有startup込みで報告し、評価だけのscore/費用と区別する。旧d97e、新B、今回pilotの研究全体累積費用も別列で保持する。不同精度の実現RZ差を同精度の期待改善としない。

identity/source/engine/matrix/probability/stream/word/価格不一致、rank<4、D_CV>4、quota不能、非干渉不成立でSTOP。追加shots、seed変更、再fit、control変更、float/GPU、他molecule/round、18round、C1解除、科学retryはscope外。
CPU/RAM/workerに恣意的固定上限を設けず、実際の空き資源で既存poolを再利用する。独立taskを同stage内で充填し、学習/配分/評価の情報依存とSTOPを保つ。
成功・中断とも全結果/費用/STOPを公開照合してmandatory STOP。Step 3へ自動移行しない。

## 6. Step 3と次の採択事項

有望性の診断後だけ、別の独立cal/prod、精度/失敗確率/予算/全費用保証の範囲を新契約として提案する。pilot bitを本productionへ混ぜず、pilotのsample varianceを真variance/保証上界にしない。
次に必要なのは、現行jobの停止と保全への明示許可（自然終了していなければ）、本草案の752-shot条件/seed/配分/判定の採択、新SOURCE/薄いpilot driver・serializerの人工確認と公開固定、有限な実科学実行への明示許可。
今回、これらの停止・正式採択・実行はしていない。
