# O2-A：Sampling・測定統計・有限confidenceの技術照合

2026-11-06 JST。対象は既存SOURCE・保存文書・一次文献の限定照合である。新しいsampling法の設計、科学レビュー、数値計算、保存値の再集計、テスト実行は行っていない。科学的GO、新規性、論文化可能性は判定しない。

**残るのは、既知のimportance sampling（IS）を、実際に取得できるwhole-path価格と有限confidence保証へ接続した場合の適用可能性である。** `p/√c`、Parent/Euler再配分、履歴law、return集約自体は既知・既存であり、新機構として提案しない。主RPEのHoeffding接口に加え、既存G7・R1・Parentにはすでに二次momentとrangeを使うBernstein十分shot会計がある。未接続なのは、それらを元H・同精度・全RPE・full-wrapper価格で比較できる条件である。

## 1. 共通targetと証拠の入口

候補とbaselineは、元の物理Hamiltonianの厳密基底状態 `H|E₀⟩=E₀|E₀⟩` を同じ情報条件で利用し、状態準備費を除く。同じ元Hのenergy精度・成功確率を目指す。

\[
G_{\rm total}^{(a)}=\sum_{m,b\in\{X,Y\}}N_{m,b}\,
\mathbb E[C_{m,b}^{\rm full,no\text{-}prep,(a)}].
\]

RZ、CX、T、depth、古典CPU時間、memoryは別座標である。真のE₀、gap、完全state vector、exact path expectation、exact native priceを無料のalgorithm入力にしない。厳密元H groundであることは、PF/finite-RTE meanの固有状態・unit radius・alias-freeを自動的に与えない。

指定した[3環境統合5資料（固定3571477）][INTEGRATION]と[文献調査6資料（固定e8b19ff）][LITSURVEY]を入口とした。統合台帳の `IDX-RB-04/13/14/18`、`SAMPLING-FREEDOM`、`C1-PAUSE`、`SUP-RPE-*` と、再利用profile `RTE-BASE/ALLOCATION/RA-KERNEL/RA-NATIVE/RPE-COST` を参照した。台帳行は実験数でなく、旧UNKNOWN・後続完了・凍結を併読する。主checkoutの現行AGENTSと正式共通指示2026-10-25の§8.1/8.2/8.4/10.1/10.3も確認した。

| 実読した固定資料・SOURCE | locator・役割 | その証拠から移送しないもの |
|---|---|---|
| [Sampling freedom survey][FREEDOM]（公開a776aa8） | §2–6の現paired finite law、実±1統計、Parent、相関、access、§8の既存人工test記録 | `NO_DISTINCT_FREEDOM` は限定調査の結論。全lawのno-go、元H条件での分子性能、今回のPASSではない |
| [rte.py][RTE]（静読版e8b19ff） | `RTEEvent`、`finite_rte_distribution`、`_make_event`、`iter_sample_rte_events`、`event_unitary`、`finite_taylor_operator`、`finite_rte_attenuation` | binary64/PCG64の実装存在から、理想実数law・一般IID・丸め誤差認証を導かない |
| [parent_allocation.py][PARENT]（静読版e8b19ff） | `digital_proposal/reweight_events/event_statistics/error_accounting`。cost取得は外部。finite tableのc、実dyadic q、wを分離 | explicit-table lawを安価な全path samplerと呼ばない |
| [訂正S4結果][S4RESULT]（science SOURCE `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`） | 訂正sqrt(C) KKT、旧proxyと訂正版、50 control＋20新lawを区別。元development finite-P5条件 | exact native最適点、分子/DF、全roundの性能にしない |
| [parent_experiment/design.py][DESIGN]（静読版e8b19ff） | `proposals/ledger`。実qのM2/range、bias控除、Bernstein十分N、prep/readout slope、online `NOT_MEASURED` | 内部価格proxyのKKT、finite-confidence N最適化、実shotsを同一視しない |
| [rte_reallocation/accounting.py][R1ACCOUNT]（静読版e8b19ff） | `canonical_profile/confidence_budget`。canonical正規化、variance上界B²、centered range 2B、strict synthesis bias | canonical-onlyの接口が一般ISや全RPEにそのまま対応するとはしない |
| [g7_generator.py][G7]（静読版e8b19ff） | `budget`。finite-bit膨張付きM2/range、common bias、全trial十分N。full-returnは未知の真B_newを使わない | 元の固定16軸confidence配分を任意の全RPE軸へ移さない |
| [RPE accounting][RPE]（静読版e8b19ff） | `_hoeffding_shots/evaluate_rpe_round_candidate/build_rpe_resource_summary`、独立sampling policy、軸費用、alpha和 | 既存unit-radius/effective-PF-eigenstate仮定を厳密元H groundだけから正当化しない |
| [相関法の既存照合][CORR]（固定4301c0f） | §3–5：PRHS/qSHIFTのtask差、marginal反例、実±1 M2、Bernstein式、finite-bit留保 | 今回PRHS/qSHIFT全定理を再証明・再レビューした意味ではない |
| [D1–D3引継ぎ][D123]、[Parent終了記録][CLOSE] | 同finite target/同情報/同native条件、強対照、D＋E採択と旧数値探索終了 | 同じdevelopment fixtureの延長許可を移さない |

静的に読んだtestは、[sampling freedom人工test][FREEDOMTEST]の `test_probability_change_needs_weights_and_keeps_measurement_noise`、`test_affine_dictionary_dependence_allows_unweighted_probability_change`、`test_parent_split_is_euler_identity_and_common_mass_minimum`、`test_correlated_marginals_change_ordered_mean_and_path_is_restores_it`、[Parent test][PARENTTEST]の `ProposalControls`、[R1 accounting test][R1TEST]の `test_canonical_probability_and_moment_are_charged_consistently`、[RPE test][RPETEST]の軸・union-bound・MC件数の区別である。今回再実行していない。静読版のcommitはSOURCEの所在を示し、過去runのexecution SOURCEを代替しない。

## 2. 固定するもの・変更できるもの

元lawの対応は既存survey§2を再利用する。signed involution `P_l`、`p_l=|a_l|/λ_R`、`R=Σp_lP_l` とし、even order nの正係数は `w_n=|τ|^n/n! √(1+(τ/(n+1))²)`。`B_K=Σ_even n≤K w_n` で、eventの位相・順序を含めた平均は `E_p V=T_(K+1)(τR)/B_K` となる。SOURCEのcutoff Kは最大even orderであり、多項式degreeはK＋1。[rte.py `_make_event/event_unitary/finite_taylor_operator`][RTE] に対応する。K0はangle `atan(τ)` のP1で、P3/P5 returnを一microstepへ直接代入できない。

| 手法 | 固定するもの | 変更と統計への影響 | 既知対応・採用上の境界 |
|---|---|---|---|
| 元paired PR law | 元c、V、位相、順序、有限target、D/PF | c/Bでsample。raw ±1、correctedはB倍。独立反復でattenuationが積になる | PR原著App.Aと既存RTE。`H_D` groundへ状態を変更しない |
| 固定辞書IS | 同じ実装pathと目標平均、同じ有限target | qへ変更しp/qまたはc/qを実測値に掛ける。meanは保持、range/M2/価格分布は変化 | resource-optimal IS。biasは同じ実装pathの再重み付けだけでは減らない |
| resource-optimal IS | 固定の正価格c(ω)、元path係数、情報条件 | `q∝p/√c` はcost×worst-case M2 proxyの最適law | 文献[IS][IS] §II Thm.1。有限confidenceのceil付きN・RPE成功・生成費込み最適ではない |
| cost-aware qDRIFT | 当該simulation/channelのtarget・誤差task | `q_j∝h_j/C_j` とstep/experiment数を交換 | [Kiss et al.][KISS] §4.1 Eqs.50–55。paired PRのfixed-path ISと目的が異なり、式の差を新規性としない |
| whole-path IS | 順序付き全V_ω、D interleaving、位相、全trial target | path全体の非加法価格でqとlikelihoodを変える | ISのparameter vectorをwhole-pathとする既知適用。eventからpathへの名称変更だけでは追加手法でない |
| Parent/S4 | 同finite parent `aI−iΣb_iP_i`、全target | identity share z_iからangle・係数mass・native priceが同時に変わる。さらにISを組合せ可 | Euler分割＋既知KKT/IS。commonは当該familyでL1最小。旧proxyと訂正KKTを混ぜない |
| correlated/history law | 同targetを求めるならjoint operator mean | marginal保存だけでは不十分。全path P/q補正ならそのweightを戻す。別高次targetならbias比較を別に要する | PRHS/qSHIFT/既存相関audit。履歴の利用そのものは既知 |
| affine辞書の無重みlaw変更 | `Σ(q−p)V=0` とΣq=1を満たす同normalized mean | 理想±1の各軸marginal/varianceは同じ。価格分布は変更し得る | 既知線形係数/dictionary最適化。元H分子辞書での安価な関係取得・利益は未確認 |
| Return Aggregation＋sampling | 同じ補正finite operatorと位相 | 集約で辞書・B・実q・acceptanceを変更し得る。raw/weighted/zero-fillの測定量をそれぞれ定義 | G6–G10で既存。無条件全trial費用とNへ戻す。詳細は[O2-B](O2_B_RETURN_AGGREGATION.md) |

Parent familyの `c_i=√(z_i²+b_i²)`、`Σz_i=a` では、common `z_i=ab_i/Σb` がfamily内の係数mass最小である。これは[既存survey§4.3][FREEDOM]と[訂正S4レビュー§4][S4REVIEW]の帰属付き再掲で、一般unitary辞書のL1最小性ではない。安いangleへ移すことで増えたnormalizationを、多stepの量子費用が支払う可能性を残す。D＋Eの現fixture探索終了は維持する。

## 3. mean・実測分布・range・M2の区別

一round、一軸で元whole-path lawをP、実±1 Hadamard測定をY_b、同じnormalized raw target平均をμ_bとする。固定law・独立whole-path shot、同じreadoutを仮定した既存の会計は

\[
X_b=\frac{P(\omega)}{q(\omega)}Y_b,\quad
\mathbb E_qX_b=\mu_b,\quad
M_2=\mathbb E_qX_b^2=\sum_\omega\frac{P(\omega)^2}{q(\omega)},\quad
\operatorname{Var}X_b=M_2-\mu_b^2,\quad
|X_b|\le L=\max_\omega P/q.
\]

補正targetには元normalization B_totをweightへ戻し、M2にB_tot²、rangeにB_totを戻す。旧path自身が非一定weightを持つ場合はP/qだけでなく元weightも保持する。[相関audit§4.4][CORR]、[sampling survey§4.1/5][FREEDOM] の既存式に対応する。

`Σq(w E[Y|ω])²` はpathごとの期待値のmomentで、実測値 `(wY)²` のmomentではない。厳密stateや正確なpath meanを古典的に知ると仮定しても、実行するHadamard測定の条件付き量子ノイズは消えない。正のnormalized Pに対する単純ISではM2は元lawの1を下回らないが、単価との交換余地は別にある。

一方、**unweighted ±1測定で同じraw complex mean**を保ち、readoutとshot独立性も同じなら、各軸のBernoulli marginalとvarianceは同じである。この限定条件なら同じN規則を使える。「補正平均が同じ」「同じH」「channelが同じ」だけではこの条件を満たさない。Bが変わる場合、raw radius低下とcorrected weight増大は同じ負担の別表記で、二重計上しない。

history依存は二つに分ける。一shot内部のjoint lawは、全順序付きoperator和または正しいP/qが必要。shot間にもhistory適応を入れるなら、同じtargetの条件付きmean、predictable probability/weight、bounded increment等を確認して適用可能な既知martingale保証を選ぶ。IID Hoeffding/empirical Bernsteinをそのまま使わない。単純persistenceのmarginal反例と、channelでは同じU/−Uがcontrolled signalでは異なる反例は[旧surveyの人工test][FREEDOMTEST]で既確認である。

## 4. range-only障害のscopeと、残る既知variance-aware経路

### 4.1 ユーザーが示した前回結論との照合

ユーザー指示§4.2Cは、同じpath集合・同じ目標平均・最大weightによるrange-only十分shot評価では単純再重み付けに障害がある、と既存科学レビューの結論を示す。その**旧レビュー原本の固定path/commitは今回の指定台帳・関連資料から特定できていない**。ここはその帰属と限界を明示した技術照合で、新しいno-go定理の証明ではない。

主RPEの[_hoeffding_shots][RPE]は±1座標に `ceil(2 log(2/α)/ε_coord²)` を使う。weighted outcomeの幅をmaxweightで拡大し、同じtarget座標精度へ戻すrange-only評価は、平均単価の低下だけでは改善を裏付けない。対象は同path/同target/同じ残り統計幅でのrange-onlyの連続十分予算係数である。整数ceilによる微小差、別のstatistical rule、別辞書/return、zero-fill、異なるbias許容幅へ一般no-goを拡張しない。これは必要最小shot数の下界でも、全ISの総費用不可能性でもない。

### 4.2 すでに存在するBernstein接口

独立whole-path shot、`|X|≤L`、`E X²≤M2`、bias控除後の統計幅s>0に対し、既存[相関audit§4.4][CORR]は次の保守的十分式を記録する。

\[
N_{m,b}\ge
\left\lceil\log\frac{2}{\alpha_{m,b}}
\left(\frac{2M_{2,m,b}}{s_{m,b}^2}
+\frac{4L_{m,b}}{3s_{m,b}}\right)\right\rceil.
\]

[Parent `ledger`][DESIGN]は実dyadic qのM2/range、[R1 `confidence_budget`][R1ACCOUNT]はcanonical B²/2B、[G7 `budget`][G7]はfinite-bitの上界を用いてこの型を既に実装している。測定済みvarianceやexact signalは入力でない。**variance-aware保証を初めて導入する候補ではない。** M2上界はvariance上界として有効だが、真のvarianceを最適化したことにもならない。

固定targetのISで改善可能性が残るためには、平均full-wrapper価格低下が、この式のM2項だけでなくrange項、bias余裕、整数N、全軸・全roundの負担を支払った後にも残る必要がある。正価格のp/√cはleading proxyの参考で、有限confidence最適の保証でない。[Resource-optimal IS §II/Thm.1][IS] の目的はM2×価格で、§II.1の長積での利益縮小は独立・加法価格に限る。非加法whole-path価格へ一般no-goとして拡張しない。D/PF backboneやwrapperの共通費が大きければ、tailだけの価格差が全価格の差へ現れにくい。この効果も既存費用境界で確認する。

### 4.3 実測varianceを利用する既知保証

| 既知手法 | 必要情報と利用可否 | 今回のPR/RPEへ残る接続 |
|---|---|---|
| M2上界によるBernstein | 実q、元c、weight range、M2上界、bias余裕が取得できれば、未知E₀/gap/path meanなしで利用可能 | Parent/G7から全roundとfull-wrapperへ適合確認。既存対照も同じ保証を使う |
| 固定nのempirical Bernstein | [Maurer–Pontil (2009), Thm.4][EB]はIID bounded observationのsample varianceを使う。既知rangeで尺度変換し、両側・X/Y・roundの失敗配分を行う既知手法 | 実量子outcomeまたは許可済みの同等な検証データが必要。保存path期待値・少数sampler出力・旧人工PASSをsample varianceに代入しない。今回は取得していない |
| sequential confidence sequence | [Howard et al., confidence sequences][CS]は時点一様のcoverageを扱う既知対照。固定n保証を観測後の任意停止へ流用しない | 今回は要約確認のみでPRへの適用定理・stopping ruleは選択していない。random Nなら期待停止費用も会計し、無条件に `E[N]E[C]` としない |

最小の入口は既存の固定n/M2会計である。empirical/sequential法へ広げることは必須でない。baselineが高いraw signalで本来小さいvarianceを持つ場合、baselineにも同じvariance-aware処理を与える必要がある。候補だけに真varianceや良いpilot情報を与えれば公平な利益比較にならない。実測variance低下、二次moment最適化、十分N低下、必要Nの下界、総費用保証はすべて別である。

## 5. finite-bit・normalizer・古典accessの照合

| 確認点 | 既存資産から分かること | 追加で必要な最小情報 |
|---|---|---|
| path生成 | 元RTEはorderとlabelを生成し、metadataにbasis/符号/順序を保持。Parentの`reweight_events`はexplicit tableである | whole-path proposalの実際の条件付き確率、乱数bit数、逐次状態、全supportでの正確なlikelihood |
| path価格 | basis共有、controlled phase、D境界、native synthesisで非加法になり得る。event単価和は全wrapper価格でない | 同じprecision/compiler/optimization後の価格、または保証付き価格access。未compile pathのexact price oracleを仮定しない |
| optimal-law normalizer | p/√Cの規格化和が数式にあっても、全pathで計算可能とは限らない | 安価なfactorization/再帰/計算可能enclosureが既存接口にあるか。なければsupport列挙/価格取得を必要費用として扱う |
| enumeration | 元RTEの列挙は小規模reference用。Parent lawのtable長は明示的。一般local returnとclosed-P5は異なるaccess | 元supportを列挙せずsampleできる具体的入口とnormalizer/weight評価。数式上の最適lawだけで可とはしない |
| finite-bit q | Parentはroot/probability precisionとdyadic qを持ち、event weightを実qからc/qへ戻す | ideal q*との近さだけでなく実qのsupport/M2/range。zeroへ丸めたpositive coefficientは未補正biasになり得る |
| 係数/角度誤差 | exact qでc/qを評価しても、実装係数cと理想係数との差、angle/native誤差は残る | coefficient/angle/native/controlのmean誤差上界を別に戻す。likelihoodの近似評価も独立に確認 |
| zero-fill | General full returnはknown envelopeから全trial lawを構成する既存方式。accepted-only normalizerを無料には得られない | rejectを含む全trial分母、accepted量子費用＋全trial生成費、known returnを引ける実signal条件 |
| cache | basis/符号/phase/角度/precision/tool/compiler/入力を保つ既存key・actual adjoint資産がある | cold取得・warm再利用・missの費用と範囲。異なるlawのtrace依存hit率を旧値から予測しない |
| setup/online | 既存S4でonline `NOT_MEASURED`。G8等の有限interface計測は分子全Nの母平均証拠でない | 古典前処理・selector・normalizer・lookup・online生成・serializationの必要部分。CPU秒をT/RZへ無根拠換算しない |

probability丸めの影響は二経路ある。実qと正確なc/qが既知ならmeanをcへ保てても、M2/rangeは実qで再評価する。c自体やweight/angleを近似する場合は目標meanへのbiasを別枠で支払う。この区別は[Parent `event_statistics/error_accounting`][PARENT]、[G7 `budget`][G7]、[S4 `ledger`][DESIGN]の既存仕様であり、新しいsamplingアクセス保証を取得したものではない。

資源指標によってはpositive support上の価格が0になり得る。既存`digital_proposal`はその場合のanalytic ISを拒否し、S4 `proposals`は正常不適用として分ける。恣意的な正価格floorを実回路価格として追加しない。既存regularized lawのparameterも、実際に支払う物理資源価格とは別である。

## 6. X/Y・全round・energy成功条件への接続

主[RPE accounting][RPE]は `rho_lb=attenuation*(reference_radius−epsilon_z)`、`epsilon_coord=rho_lb sin(beta_stat)/√2` を使い、X/Yごとにalphaを配分する。SOURCEのreference radiusは1のeffective-PF-eigenstate仮定であり、現在の元H厳密ground条件の下で無条件に使わない。必要なのは対象stateと実meanに適合するradius/phase allowanceである。

ISでnormalized raw targetを保つなら、同じ座標精度にweighted concentrationを適用する。corrected targetを使うなら誤差幅とradiusを同じ単位へ戻す。qがX/Yで異なっても両平均のtarget/scaleを一致させ、各軸のNと価格を別に会計する。基準meanが保持されても、PF、finite-RTE、finite-bit、native synthesisのbias allowanceが異なれば同じNにはならない。

全RPEには各roundの位相誤差予算、alias/branch条件、初期energy区間、最長時間、最終energy変換、`Σ_m,b α_m,b` によるfamily成功条件が要る。union boundそのものはround間独立を要求しないが、各軸のconcentration式はその前提を要する。RMSE、固定時間のcomplex精度、energy失敗確率を置換しない。RPE原著の訂正による旧分枝条件の変更は[文献調査SOURCESのL18][LITSOURCES]に帰属させ、今回旧decoderの合否を再審査していない。

統合台帳の `SUP-RPE-V3R1/ENERGY` は、H4 linear 1.0 Å、STO-3G、legacy DF12、L_D=3対12、δ=.02、K=2、round別r、18round/36軸の条件付き成功とenergy接続を既に記録する。直接compile q≤32と長q外挿、理想IID/数学的decoderと実装全体を区別する。これを「全RPE未着手」へ戻さず、今回のweighted/return変更への移植が未接続とする。最終費用優越は既存でも `UNDETERMINED` である。[統合Gap RQ-06][GAP]

## 7. 条件付き候補・反証・最小追加検証

| 方向 | 残す理由・既知法との差 | 継続に必要な最小確認 | 終了を検討する条件 |
|---|---|---|---|
| A-V：既知IS＋既存Bernsteinをfull-wrapper価格に適用 | sampling単独の適用候補。未知signalを使わずM2/range上界で会計できる。独立新規性は主張しない | ユーザーが選んだ同一H/state/精度・confidence・一資源指標で、既存対照と同じbias/N規則、実q、取得可能なwhole価格を結ぶ。まず一つの代表roundで成立条件を見て、全round移送はその後の判断 | 価格取得が非現実的、range/bias項で利益消失、強baselineへ同じ最適化を与えると差が消失。単なるproxy小差だけなら追加大規模計算を必須にしない |
| A-E：同じestimatorへの既知empirical-Bernstein適用 | sampling変更なしでも有限confidenceの過剰なNを減らす可能性。統計手法の適用候補 | bounded actual outcome、fixed n、round/軸配分、bias/radius接口。既存双方に同じ保証を与える。実測の新取得・再解析は別実行許可 | varianceを正当化できない、既知range/M2会計との差が小さい、探索/停止費用で利益を失う。adaptive法へ自動拡張しない |
| A-R：return構造を使える非列挙whole-path access | 組合せ候補として残る問いは、同target/全trial統計と非加法価格を、支払えるnormalizer/生成費で利用できるクラスの有無 | 既存G6/G7/Parentの入力接口に、どのreturn/価格/likelihood情報が既にあるかを限定する。その後必要なら別許可の短い同target比較。正式samplerは今回設計しない | 無料exact-price/normalizer/state情報を足さないと成立しない、既知table法/closed P5を再命名しただけ、生成費を無視して初めて有利になる |
| A-weak：maxweightだけで安いpathを増やす単純再重み付け | ユーザー提示の限定range-only障害。mean保存だけでは十分でない | 現条件のままの追加分子samplingは提案しない | 同path/同target/range-only条件を維持する限り根拠が弱い。異なる保証の検討はA-Vへ分ける |
| A-weak：既済Parent/common、p/√c、相関marginal、全path P/qの再命名 | 既知Euler/IS/PRHS/qSHIFT、旧 `NO_DISTINCT_FREEDOM` とD＋Eのscopeを維持 | 新しい具体的access/保証差が提示されたときだけ限定照合 | 式・変数・名称の追加だけ、measurement noiseの省略、K0へのP5無断移植、旧toy延長でしかない場合 |

表は性能順位・研究採択・GO/NO-GOでない。追加科学の前にユーザー／GPTが必要性と比較taskを選ぶ。Codexの次の科学作業は、別途許可された場合にだけ既存SOURCEを適合させて行う。新しい非自明な統計定理を必要とする方向は、今回の最小技術確認へ取り込まない。

## 8. 実施していない作業とSTOP

文献は関連節の部分確認であり、全論文精読・新規性網羅審査・独立定理証明ではない。今回直接webで確認したのはresource-optimal IS §II–II.1/Thm.2、Kiss論文§4.1 Thm.4/5のsimulation費とexperiment数の区別、Maurer–Pontil Thm.3/4、Howard confidence-sequence要約である。PRHS/qSHIFTの詳細は固定相関auditの確認範囲を継承した。

新しい科学計算・数値再解析・保存値の集計/ランキング/予測、sampling/RNG replay、Hamiltonian/state/reference生成、build/compile/native synthesis、新sampling実装、scientific test/simulation、C1再実行、重要科学レビューは行っていない。過去のsynthetic PASSを今回の科学PASSへ変更していない。新精度・分子・compiler・資源指標・実行契約は固定していない。

公開と取得照合をroot担当が完了したらmandatory STOP。旧C1、Parent、SAMPLING-FREEDOM、G系列その他のSOURCE/raw/result/grant/ledger/marker/STOPは維持する。次の担当はユーザー／GPT：表のどの条件確認を必要とするか判断する。新科学実行と重要レビューはそれぞれ別途の明示許可が必要で、自動的に開始する次段階はない。

[INTEGRATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[LITSURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[LITSOURCES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/SOURCES.md
[GAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_GAP_MATRIX.md
[FREEDOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a776aa84aefc128cc317e33075415140916013a0/docs/research/research_c_sampling_law_freedom_survey_20261030.md
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rte.py
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/parent_allocation.py
[S4RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md
[S4REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/parent_experiment/design.py
[R1ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/g7_generator.py
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[D123]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[CLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[FREEDOMTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/tests/management/test_research_c_sampling_law_freedom.py
[PARENTTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/tests/tracks/algorithm_codesign/test_parent_allocation.py
[R1TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/tests/tracks/algorithm_codesign/test_r1_rte_reallocation_accounting_launch.py
[RPETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/tests/test_rpe_resource_accounting.py
[IS]: https://arxiv.org/html/2603.13495v1#S2
[KISS]: https://arxiv.org/pdf/2212.05952v2
[EB]: https://arxiv.org/pdf/0907.3740
[CS]: https://arxiv.org/abs/1810.08240v9
