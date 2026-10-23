# H4非DF/DF：numerical error・bias・normalizationの保証手法設計

2026-10-23 JST。段階Cの設計と人工検証。**実H4の高精度・区間演算・certificate生成は0。元N/Gはnull、全40軸UNDETERMINED、元native終了証拠は未回収。** Bの必要誤差予算は達成可能性やcertificateではない。

## 1. 固定対象とBから受け取る要求

対象はlinear H4/隣接1.00 Å/STO-3G、Pauli-JW184項（PR prefix27）、DF tol-only1e-8/rank10（PR prefix1＋one-body）、共通非DF Nα=Nβ=2 sector36の保存状態、8 system qubit。S2/T=.8/q8/δ=.1、PR r4/K4、ε_sig=.05/.01/.005/.001/.0005。準備P=0、ordinary-controlled wrapper、旧compiler/候補/標本数/guardは変更しない。旧科学SOURCEは`0c329b30a81accb35c212be8defa66fee0dd0f12`。

Bの技術SOURCEは`2bd212288e8530e519c4a8135257df37f7608662`、入力shaはB_run01に固定した。BはFractionのexact binary64解釈と別経路Decimal100で40軸を一回診断した。必要u上限は全40軸で約2.45e-6〜3.50e-4。R群は約1.53〜90.1倍、H群は約107〜2823倍のu改善が必要。固定点biasだけで不可能となる軸は0だが、biasの数値不確かさは未証明である。[40軸の表・実行証拠](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/diagnostics.md)。

各軸の設計目標は、Bが記録した保守的なu上限の1/2。これはguardを変える値ではなく、証拠生成側の余裕である。残り半分は境界丸めやモデル対応誤差を見落とさないために確保する。Bの予算と手法の対応は[C_precision_method_map.csv](../artifacts/non_df_h4_precision_bc/2026-10-23/C_precision_method_map.csv) / [JSON](../artifacts/non_df_h4_precision_bc/2026-10-23/C_precision_method_map.json)に全40軸を記録した。

## 2. 保証する対象を先に固定する

第一の対象は、保存された係数・state・法則を持つ**固定モデルの数値計算誤差**である。SCF/DF/FCI/状態最適化をやり直して物理モデルを変えない。未丸めの真のMO積分や、真の分子ground stateに対する誤差保証まで取得したとはしない。

JSON/NPZのbinary64値は整数比へ変換し、そのexact値を球・区間へ取り込む。decimalで見やすく書き直して別の入力へ置換しない。Pythonのfloat→Decimalのexact変換と、FLINTのball入力の意味を区別する。[Python decimal](https://docs.python.org/3.12/library/decimal.html)、[FLINT arb仕様](https://flintlib.org/doc/arb.html)。

次の対応誤差を独立の証明対象にする。

- 保存stateの正規化：保存vectorを無断で正規化して置換しない。`n=ψ†ψ`を包含し、同じrayを物理的なnormalized stateとして使う場合の差を証拠予算へ含める。legacyの未再正規化signalと区別し、n>0を証明する。これを実装できなければSTATE_NORMALIZATION_UNRESOLVED。
- Pauli/DF operator：保存係数、qubit順序、核・抽出identity、one-body、λとgから構成する演算子を固定する。DFの`hermitized()`やbasis diagonalizationを高精度で勝手に置換しない。元のg/one-bodyから準備されたnative表現との残差も包含する。丸め込みでHermiticity/unitarityを仮定しない。
- DF basisと全component：sampled eventだけから未観測componentを補わない。保存snapshotから旧runtimeのdeterministic preparationを復元し、元preparation/hash・basis・identity/prefixを照合する。新SCF/DF生成・sampling・compileではないが、**次回Cのoperator復元予算**へ明記する。完全なregistryや同一hashを再現できなければMISSING_FROZEN_PREPARATIONで停止する。
- 物理sector：DFの個々の因子について不変性をexactに証明した場合だけsector最適化を許す。Pauli個別項が粒子数を保存するとは限らないため、共通状態がsector36にあることだけで全因子を36次元へ射影しない。初期実装は全256次元を基準にする。

## 3. 候補手法の比較

| 手法 | 保証するもの | 保存入力/新しい計算 | 必要精度への見通し | 残る問題・検証 | 実装/計算量 |
|---|---|---|---|---|---|
| 高精度float二系統の一致 | 収束・一致の診断。単独では包含を保証しない | 同じH/stateのreference/law meanを再評価 | Bの数e-6予算を下回る可能性はあるが未実測 | cancellation/共通bug/モデル差は一致だけで除外不可 | 比較的容易。密行列なら概ねO(n³) |
| **Arb/Acb ball＋検証済み行列指数** | 入力ball内の演算結果を丸め・近似剰余込みで包含 | 保存binary入力からmatrix/信号/有限lawを包含。現段階は人工例だけ | 64 bits開始、必要な場合だけ96/128 bitsへ。実H4での成功は未確認 | 対応誤差・状態・確率・basisを別に束縛する必要 | 有力な基準経路。dense exp O(n³)、記憶O(n²) |
| forward/backward error解析 | 証明済みkernelの丸め誤差と残差から包含 | operation graph/BLAS/expアルゴリズムの特定が必要 | 低精度を維持できる可能性はあるが、旧work proxyが粗いことだけでは保証を得ない | gamma_n等の前提、FMA/集計順、exp残差、normを検証 | 証明負荷が大きい。改良候補として保持 |
| exact rational interval prototype | 小規模人工例の演算・平方根/log剰余・Taylor剰余を数学的に包含 | 人工1×1/2×2だけ。H4 loaderなし | B/Cの式と誤差伝播のoracleに利用 | Fraction分母増大のため256次元の基準本実装には不向き | 小例は軽量、H4では計算量・整数長が増大 |

推奨はball経路である。FLINTは`approx`系と保証付き演算を区別し、`acb_mat_exp`はTaylorとscaling/squaringに基づく誤差境界を返す。**approx eig/exp、radiusを落とす変換、midpointだけの受理は禁止**。[FLINT acb_mat](https://flintlib.org/doc/acb_mat.html)。python-flintのmatrix exponentialと局所precision contextを人工入力で接続検査した。[python-flint matrix](https://python-flint.readthedocs.io/en/latest/acb_mat.html)、[context](https://python-flint.readthedocs.io/en/latest/general.html)。

## 4. reference・law mean・biasの包含アルゴリズム

### 4.1 reference

各表現について全Hを構成し、`E_ref = exp(-i T H)`をball行列指数で包含する。保存state上のcomplex signalを縮約し、Re/Imを包含する。referenceは同じ表現のPF/PRで共有し、重複計算を避ける。

人工rational prototypeでは、scaling後の行列Aに対してTaylor次数mの部分和をexactに計算し、submultiplicative normの上界αを使い、剰余を

`α^(m+1)/(m+1)! / (1 - α/(m+2))`（α/(m+2)<1）

で覆う。各成分へ剰余を加えた区間をsquaringし、誤差をそのまま伝播する。実H4のrational計算を実施したものではなく、FLINT接続を検証する人工oracleである。

### 4.2 固定S2とpaired finite law

科学SOURCEの因子順序を維持する。deterministic blocksを正順にstateへ適用し、PRはtail作用をr回適用し、逆順のblocksとscalar phaseを適用する操作をq回繰り返す。paired-even cutoff K=4は**通常Taylorのdegree5まで**保持する。新しいdegree4打切りやexact tail expへの置換は法則変更なので行わない。

nominal corrected tailは `P5(-i δ H_tail/r)^r`。Horner/forwardの別実装は一致診断として使い、両者とも包含演算にする。非unitaryな有限多項式のnormを1と仮定せず、たとえば因子norm M_j・誤差d_jに対する積の誤差を `E_j=(M_j+d_j)E_(j-1)+d_j*prod_(k<j)M_k` として伝播する。

### 4.3 frozen lawの期待値との対応

nominal polynomialだけを再評価してcertificateが完成したとはしない。完全なcomponent P_j、保存されたeffective probability p_j、order probability π_k、rotation角φ_kを復元する。`M=sum p_j P_j`、`R_k=sum p_j V_j(φ_k)`とすると、sourceの「productsを先に、rotationを最後に適用」という順序では、短step期待値を

`F = sum_(k=0,2,4) π_k (-1)^(k/2) R_k M^k`

に因数分解できる。全component組合せを列挙せずに、保持された法則の期待値を包含する。native basis/gate対応の誤差がある場合はR_k/P_jの区間へ含める。`B_K*F`とnominal P5の差をゼロと仮定せず、law実装差として最後のbias証拠へ加える。人工2componentでは完全列挙42eventと因数分解・degree5 polynomialの一致を確認した。

NumPy1.26.4の`choice`は保存pからCDFを作り、その末尾で割り、uniform値をright-side searchする。PCG64のdouble生成は53bit格子に写す。このため、保存pを単に高精度へ変換して真の実装確率だとしない。[choice source](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx)、[PCG64](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx)、[double変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd)。

次回Cで、旧runtimeに結合したCDF計算をrounding graphごと復元し、CDF端点c_iに対するmassを `ceil(2^53 c_i)-ceil(2^53 c_(i-1))` / `2^53` で包含/算出する設計とする。RNGを引かず、既存seedもreplayしない。これはuniform wordをIIDとする既存の統計モデルの下での浮動小数点mapping証拠であり、PCG64列の独立性自体の証明ではない。統計仮定は残し、無条件の母平均保証へ格上げしない。

### 4.4 signed biasと旧点biasの誤差

`b_interval = corrected_mean_interval - reference_interval`を作る。保存biasを無断で新しいmidpointへ置換せず、各軸で `e_bias = sup |b_interval - b_saved|` を求める。state normalization、operator/law対応等の追加誤差も包含したtotalが設計目標以内かを判定する。実際のbias自体が要求εを超える可能性と、旧biasの計算誤差とは別の項目として保存する。

受理は区間の上端を使い、`total_error_upper <= Bのu_budget/2`を厳密な比較で満たす場合だけとする。表示値や丸めた equality を受理に使わない。N/Gの再計算・正式な旧判定変更は、さらに後の明示承認された会計段階に分ける。

## 5. PR normalization certificate

固定有限lawに対するnominal式は、`tau=lambda_r*delta/r`、`w_k=|tau|^k/k!*sqrt(1+(tau/(k+1))²)`、`B_K=sum w_k`、`B=B_K^(q*r)`、`log_B=q*r*log(B_K)`、k=0,2,4である。平方根とlogを外向き包含する。C prototypeはsqrtのexact二乗不等式、logのatanh級数と剰余境界で人工的に検証した。

Pauli lambdaは固定tail係数絶対値和を包含できる。DFは全symbolic tail componentの絶対係数和であり、fragment順位proxyや観測componentだけの和で代用しない。保存lambda/τ/weightsと、この包含で得る数学量の差を記録する。高精度値で置換して旧法則を変更しない。

保存された`log_B_upper=log_B_float+empirical_margin`については、次回Cで `log_B_upper_saved >= upper(log_B_interval)` が成り立つかを検証する。満たさなければその保存上界をcertifyしない。実装されたprobability/angle/basisを含むlawとの対応、各eventの測定範囲±Bとの関係も別gateで確認する。

normalization誤差がcorrected signalへ与える影響は、`|raw|<=1`等の範囲が証明された上で `B_upper*(exp(delta_logB)-1)` などの包含で結合する。小さいdeltaに対する近似を無断で等号として用いない。既存の`log_B_margin`やnormalization falseフラグは今回変更せず、将来の証拠は別versionへ保存する。

PFはstochastic normalizationがなくB=1/log_B=0の構造証明で足りるが、reference/law mean/biasの数値誤差証拠は引き続き必要である。

## 6. 必要精度に応じた実装・計算量

作業精度の予定は64→96→128 bits。最初に全固定対象を64bitsで評価し、同じ表現内で必要な軸の包含幅だけが不足したときに、未充足case/referenceだけを次の精度へ進める。これは契約内で事前登録した有限precision scheduleで、例外後のretryではない。モデル不一致、入力/環境変化、非有限値、例外が出た場合は次精度へ進めずSTOPする。128bitsでも不足ならUNDETERMINEDを保持し、無断の精度・候補・予算拡大をしない。

Pauliのdeterministic factorsはP²=Iを使うcos/sin＋signed permutation作用で実装でき、各因子O(n)のstate作用を目指す。DF因子は同じstepの指数をcacheしPF/PR間で再利用する。referenceは2表現だけで共用する。密な基準経路はO(n³) exp/O(n²) storage、行列vector作用はO(n²)。Pauli/DFの独立process、precision-local private cache、十分な空き資源を利用し、mutable precision contextをthread共有しない。

全caseに一律の高precisionを強制しない一方、最も厳しい数e-6のu予算を満たせない証拠を緩いεの成功で補わない。半予算の設計目標は約1.23e-6〜1.75e-4。これを満たすことやH4のruntimeはまだ実測していない。入力/model対応誤差が予算を超えるときは、precisionを上げても解消しないので別の未確定理由として保存する。

## 7. 次回Cの完成契約候補

[C_execution_contract_DRAFT.json](../artifacts/non_df_h4_precision_bc/2026-10-23/C_execution_contract_DRAFT.json)に固定対象、入力sha、algorithm、precision schedule、論理call予算、proof gates、failure/監視/保全/出力範囲を記した。**execution_authorized/grant_valid/dispatch_enabledはfalse。C grantは未発行・未消費。**

最大予算はoperator/preparation復元2 batches、probability対応2 batches、reference包含6、nominal＋実装law mean包含18、PR normalization包含6。dense matrix exponentialはcacheを含む基準経路で最大39 requests。SCF/FCI/新DF生成/状態最適化/MC sampling/compile/shot/G計算は0。固定precision tierを使い切っても、同じgrantで別候補やrunへ移らない。

設計と人工prototypeは今回完成したが、**実H4のproduction assembler/certificate runnerは未実装**。C実行前には、この設計に従う実装・回帰を完成し、その実装commitとruntimeを新SOURCEへsealする必要がある。現時点のB SOURCEを実H4 Cの実装SOURCEと偽って使わない。契約候補の未実装statusを隠さず、dispatchは不可能にしている。

## 8. 検証・保全・残存事項

人工・回帰33検査PASS。exact closed guard境界・nextafter・内向き表示、独立Decimal、40軸missing/duplicate、旧会計の人工回帰、one-shot・変更拒否・partial・C拒否、rational包含・有限degree5・非可換順序・norm増幅・sector射影・state norm欠損を検証した。

private環境のpython-flint0.9.0/FLINT3.6.0で人工backend13項目PASS。依存wheelと実際のlibrary SHAはinstall/runtime記録へ固定した。共有venvは変更しない。初回の人工test harnessのmock問題、および独立区間とbackend ballの幅関係を誤ったcontainment testの修正履歴も残した。これらは実H4 runの失敗・retryではない。

元SOURCE/raw/input/grant/ledger/marker/STOPと調査commitの資料を保全する。新しいBの正常終了は元H4 native exit/reap/STOPの証明ではない。元監視消失原因UNKNOWN・native未証明を維持する。B/C結果保存・終了監査・local commit後mandatory STOP。次の担当はユーザーによるC実装/実行範囲の承認判断であり、今回実H4のcertificateやN/G再会計へ進まない。
