# 次期検証の選定と数値誤差上界の最小計画

2026-11-03 JST。**第一候補は、samplerとの接続を明示したm17周辺平均の数値誤差上界の限定評価。**
B-1b→B-2という連続実行は推奨しない。本書は検証の提案であり、研究方針の正式採択や科学実行の許可ではない。
新しい実H4のnorm、CDF、scalar誤差、行列、平均、上界は一切計算していない。

## 1. 三候補の比較

| 項目 | B-1b（component平均） | B-2（整合した定義式入力） | 直接の数値誤差上界 |
|---|---|---|---|
| 入力 | 完全な保存成分入力なし。SOURCE replay variantは可能 | π/θ/Bは保存τから新生成可能。係数/Aを再構成する場合はB-1bと同じ不足 | 保存modelの包含だけなら既存A/F/R/ψで可能。sampler接続には成分/basis exportが必要 |
| 新情報 | A構成算術を変更したときのmean変化 | 整合scalarへ変えたときの差。共通Aでの一致は既に代数的に予想される | νと保存model μ0の差が固定予算に収まるか、上界として判断可能 |
| 独立性 | 係数生成・basis SOURCEを共有。共通の生成誤りは検出しにくい | 同じTaylor関数への統一は禁止。独立Law式とTaylor式を保持 | 独立した包含backendと明示した誤差bridgeが必要。同じSOURCEの意味の誤りは包含でも消えない |
| 主な計算量 | SOURCE replay、basis構成、集約、70次元のr/q乗 | scalar生成は軽いがmeanには70次元r/q乗。2個の保存Aを残す場合は同値性は強制されない | 成分/basisの1回のexportと構成、少数の包含meanまたは摂動bound。trajectory列挙は不要 |
| RPEへの寄与 | 原因調査。真値誤差上界ではない | 原因調査・実装control。元binary64の保証ではない | 成立するbridgeの範囲内で座標予算の十分条件へ直接使える |
| 残る限界 | π/θ/B・sampler・評価丸めは別 | 新入力での一致は元実装の保証ではない | backend、SOURCE対応、確率モデル、状態/compiled範囲を明示する必要 |

B-1bには再生成値を保存原本と誤認する危険があり、現時点で第一候補にしない。
B-2は保存π/θ/Bの丸めの効果を調べる診断として価値があるが、1e-10への一致を目的に先行させない。
元条件のA_probとA_coeffを同じ行列へ置き換えるcontrolは、独立した人工/方法検査として区別する。
実研究のLawをTaylorに置き換える修正は行わない。

## 2. 必要な誤差の両端

記号を次のように分ける。

- μ_ideal：厳密な数式の有限RTE平均。物理的指数関数との差には既存PF/RTE予算が関係する。
- μ_64：保存binary64 A/π/τ/B等を厳密数として解釈した形式的な演算子平均。πのunit-sum等を自動補正しない。
- μ_hat：その参照式を浮動小数点で評価した値。RPEで固定する保存値をμ0と書く。
- ν_cat：指定したcomponent/angle/gateとcategorical lawでのideal Hadamard周辺平均。抽選のIID仮定を含む。
- ν_exec：仮想shotの浮動小数点信号/clip/確率比較、またはcompiled回路・実機等、実際の実行形態に対応する平均。

RPEの固定座標事象へ直接必要なのは、**実際に採用するν_execと保存μ0の各座標の差の上界**。
Law/referenceの観測差、μ_hatの精度間収束、μ_idealとμ_64の差だけでは代用できない。
ν_catを第一の実装対象とする場合でも、ν_execとのbridgeが未確立なら結果はその条件付き範囲に留める。

各軸aについて、包含I_aがν_catを含むなら

    U_cat,a = sup_{x∈I_a} |x − μ0,a|,
    U_exec,a ≤ U_cat,a + η_exec←cat,a.

I_aに全構成/評価丸めを含めれば、別に同じ評価誤差を再加算しない。
保存Aだけの包含I_64を使うなら、η_cat←64も必要で、未評価のbridgeを0としない。
旧B-1a gapを一律にこれらのηへ置くこともできない。
保存HP operatorのdecimal round-tripは近似値の保存であり、真値を含む区間ではない。
それをradius=0の真の演算子として流用せず、必要なら別に包含した対象のmidpoint候補としてだけ使う。

固定N/α/εに対し、独立±1測定のHoeffding半幅s_aを使う保守的十分条件は

    U_exec,a + s_a < ε_a,  s_a = sqrt(2 log(2/α_a)/N_a)   （両軸）.

これは[既存の残余設計](../rpe_mean_tolerance_validation_selection_20261103/RPE_MEAN_ERROR_REQUIREMENTS.md)と同じ条件。
元m17の約8.60e-5という統計残余は既存結果の引用で、今回再評価していない。
εやβ_statを追加誤差に満額使わず、PF/RTE誤差と二重会計しない。
この局所条件が成立しても、他round、物理状態/半径、branch条件、PF/RTE側の保証まで自動的に成立するわけではない。

## 3. 上界手法の見込み

### 演算子摂動と縮小性

任意のsubmultiplicative normについて、非可換でも

    X^q − Y^q = Σ_{j=0}^{q−1} X^(q−1−j)(X−Y)Y^j,
    ||X^q−Y^q|| ≤ η Σ_{j=0}^{q−1} κ_X^(q−1−j) κ_Y^j,

ただしη≥||X−Y||、κ_X≥||X||、κ_Y≥||Y||とする。
両方が縮小ならqη。一般にqηと置くことはできない。
exact unitaryの確率mixtureとexact unitary F/Rから作る対象は縮小だが、保存binary64行列にその性質を転用しない。
一方だけ縮小、もう一方のnorm≤1+ηなら幾何和で扱える。
中間normを大きなdimension依存の粗い値に置くと、そのq乗で判定力を失うため、
近unitary性・縮小性と局所残差を保持する。厳密化にはこれらのnorm上端の外向き評価が必要。

tail Tとhalfの差は例えば

    R T F − R0 T0 F0 = (R−R0)TF + R0(T−T0)F + R0T0(F−F0)

から合成できる。stateもψ0+eとして両側の期待値誤差を加える。
保存ψを無言で再正規化せず、物理的に正規化したstateを対象にする場合はその差もbridgeに入れる。
||E||から||E^r||へ、さらにF/Rとqへという段階を保持し、qrを無条件に掛けるだけの上界にしない。

### 確率・CDF・normalization

同じunitary eventに対する二つのlawの期待値差は2 TV以下。
同じcomponent operatorsの平均差はΣ_i |p_i−p'_i|で上から評価できる。
CDFの差δ_jを使うとTV≤Σ_{j=1}^{J−1}|δ_j|。
有限uniform gridによる差はCDF境界ごとに1/2^53未満だが、元p→float CDFの丸めも別に含める。
独立eventへ転送する際は、order条件付きのcomponent抽選数も数え、必要なら期待個数で鋭くする。
固定seedのPRNG列のIID性はこのTV評価から導けない。

理想係数の正規化bには区間の正の下端が必要。b^rの影響はlog/expの外向き評価等で扱える。
実装categorical lawを直接評価する場合は、その確率が既に正規化されているため、さらにbで除算しない。
有限Taylor truncation boundはμ_idealの指数関数近似を扱い、CDF/係数表現/FP積の丸めの代用ではない。
SOURCEの既存bound関数もfloat返値であり、厳密上界に使うなら外向きの実装または丸め余裕の証明が必要。

### 包含演算と安定性

推奨は、scalar・小さな構成段階にはArb/Acb包含、長qには包含付きbinary powerまたは
midpointとglobal norm radiusの伝播を組み合わせる方法。70次元のentrywise intervalを素朴に何度も乗算すると
依存性を失って幅が広がる可能性があり、単なる高精度化だけで解決すると約束しない。
通常の丸め誤差モデルγ_k=ku/(1−ku)は、underflow/overflowや演算kernelの前提を確認した上で補助に使えるが、
transcendentalや未知のbasis誤差を機械epsilonだけで覆ったとはしない。

FLINTのball演算は包含を返す。曖昧な区間比較をFalse=反証と解釈せず、外向きendpointで判定する。
approximate/中心値のみの関数やradiusを捨てる処理を保証経路へ混ぜない。
[python-flintの包含と比較規則](https://python-flint.readthedocs.io/en/latest/general.html#inexact-numbers-and-numerical-evaluation)、
[Acb行列](https://python-flint.readthedocs.io/en/latest/acb_mat.html)

## 4. 第一科学検証の最小計画（SOURCE準備前の案）

**現時点で科学実行可能とはしない。** flint backendが現在の実行環境に存在せず、
sampler接続用の全成分/basis exportと新しい包含adapterも未作成である。
これらの技術準備と人工試験を終え、公開SOURCEを固定してから、次の範囲をユーザーへ提示する。

1. **対象を固定する。** m17の両軸、H4/1Å/STO-3G/legacy DF12/B state/sector70/L_D3/δ=.02/q131072/r32/K2。
   μ0はV2保存mean、N411/axisと既存α/β/εを保持する。原保存値のbinary64と、数式上の定義値を区別する。
   orderのπ/τ/Bは元m17 RESULT、θ64はB-1a scalar_traceから固定し、B-2型の整合入力再生成は含めない。
   第一対象は指定SOURCE・categorical law・ideal未compile Hadamard平均ν_cat。
   元の仮想shotやcompiled実行そのものを保証するにはη_exec←catも必要だと明記する。
2. **入力bridgeを用意する。** [入力表](INPUT_AVAILABILITY.md)の固定snapshot/state/μ0とSOURCEを取得。
   元成分原本がないため、許可対象に成分/basis/half定義のSOURCE replayを明示的に含め、1回だけexportする。
   SCF・DF因子化・state solveは行わない。replayは新入力として保存し、218成分・ID順・符号・support・
   係数/λ/確率・basis descriptor・local matrixの対応、既存preparation/tail/hashを照合する。
   不一致は記録してSTOP。元実行入力の完全同一性が確認できない場合に、その誤差を0と認定しない。
3. **sampler semanticsを固定する。** NumPy1.26.4のp→CDF演算をRNGなしで評価し、生成CDFを保存。
   53-bit一様IID仮定のbucket massをexact rationalとして取得する。componentとorderの両方を対象とする。
   fresh-IIDは仮定として保持し、固定PCG streamの独立性証明やRNG replayは行わない。
   旧event_probability metadataとchoiceのbucket massを混同しない。
4. **周辺平均を包含する。** 同じ固定gate/basisのideal作用を包含し、component平均をbasis単位で集約。
   rotation→productのLawを独立に組み、r、F/R、q、ψへ伝播する。
   sectorへの厳密な不変性またはleakage上界が必要で、元の数値leakage=0.0だけでは代用しない。
   保存ψの扱い、sector補正、SOURCE replayと過去入力の対応、virtual/compiledとの差のうち、
   未保証項はUNKNOWNと記録する。未対応gateやbasisを黙って近似して進まない。
5. **必要なcallだけ行う。** 1成分/basis replayとCDF構成、1つのcomplex周辺平均の包含を初期96 bitで評価。
   区間幅で判定が決まらない場合だけ160、次いで224 bitへ進む。最大3包含meanで終了し、
   収束確認のための無条件な二重評価はしない。Taylor meanやB-1aを再計算する必要はない。
   直接包含が広すぎるときのnorm-radius算法は、SOURCE準備時に人工例で選択・固定する。
   実行中に別算法を次々試す計画にはしない。
6. **判定する。** ν_catの包含からU_cat,aを外向きに得て、必要bridge上端を合算。
   s_aも固定N/αから外向きに評価する。両軸でU_exec,a+s_a<ε_aが確実なら、指定scopeだけの十分条件を確認。
   有効な下界で予算外が確定した場合も「この十分条件では満たせない」であり、RPE失敗の証明とはしない。
   区間幅不足・入力/bridge未確立ならUNDETERMINED。診断用1e-10への一致は実行目的・合格条件にしない。
7. **原本保存・終了。** SOURCE/input/replay、CDFとmass、包含endpoint、各誤差項の対象、精度/call数、
   条件付き仮定・欠測・判定・資源・終了状況を保存し、正常終了/中断/予算判定のいずれでもmandatory STOP。
   B-1b/B-2/他round/sampling/compile/費用・成功確率再評価へ移行しない。

科学実行の許可は、この新しいSOURCEとreplay/包含範囲の固定後に一括して受ける。
承認証明のファイル・ID・grant・署名は不要。過去のB-1a実行許可は新しい計算へ転用しない。
元artifactやSOURCEを変更せず、新結果が得られても過去のu_num=nullや旧INCONSISTENTを遡って書き換えない。

## 5. 再利用・性能・実装負荷

Task Aのdyadic入力変換・binary power、B-1aの独立Law構成と記録方式、元m17のbasis集約を再利用候補とする。
既存h4_ball_evidence.pyのexact-rational→Arb変換、Frobenius/norm上端、
h4_arb_bridge.pyのmidpoint+global radiusの考え方も参照できる。
ただし既存entrypointは別研究の保護scope/STOPを持ち、今回のreal H4へ流用してguardを緩めない。
必要なら純粋な数値helperを新しいadapterへ限定移植し、その違いを人工試験する。grant/controllerは新設・転用しない。

現在のLOCAL環境はmpmathあり、python-flint/gmpy2とnative flint/arbは未検出。
今回はinstallや実H4の包含をしていない。次の技術準備で専用環境へbackendを導入し、versionをSOURCEと共に固定する。
性能見込みはbasis共有と有限orderの集約で、218^(n+1)のevent列挙を避け、qもbinary powerで扱えることに基づく。
実際の包含幅・速度・必要RAMは未測定で、MP計算の既存収束や約12.65秒を保証計算の性能と見なさない。
独立なbasis/export作業の並列化、FLINT内部threadとprocessの調整を実資源に合わせる。
CPU/RAM/worker/wall timeの人工capは設けず、共有環境・他jobを変更しない。
