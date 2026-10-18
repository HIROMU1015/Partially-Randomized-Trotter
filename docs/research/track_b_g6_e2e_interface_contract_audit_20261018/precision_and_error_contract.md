# 精度・誤差：既存保証と未認証の接続条件

既存M1〜M12と固定SOURCEにある保証の対応であり、新しい誤差定理・数値certificate・科学条件の採択ではない。実データへ値を代入せず、SOURCEも実行していない。

## Targetの層と単位

**保存raw DF**（constant/one_body/lambdas/g_matrices）、**float抽出成分とbasis gate列が定義するH**、**G6の有理係数とQで定義するH**、**finite polynomial M**、**digital M̃**、**実native回路のmean**を分ける。DF係数生成・対角化・basis構築は数値演算なので、raw DFと抽出成分が丸め誤差0で同じとは認証されていない。

c_a、Λ_R、入力operator誤差はエネルギー、hは逆エネルギー、x=Λ_R abs(h)、p、η、ρ、weightは無次元、angle/errorはrad又は無次元operator norm。normalizationの同名fieldと、物理R／normalized R_G6を混同しない。保存binary64を正確な有理数として表す場合にも、それが認証するのは**表現された値**で、元の実数Hamiltonianではない。

どの層を保証対象にするか、元Hからの誤差をどの既存規則で信号へ移すかは未採択。既存float pの個別Fraction化、再正規化、原DFからのexact係数再構成は異なるtarget/provenanceを持ち得る。今回routeを選ばない。

## 九つの誤差項目

| 項目 | 既存保証・SOURCE | 接続に必要な記録／認証状態 |
|---|---|---|
| 1 入力モデル・係数 | S04のthreshold L1 metadata、成分・basis hash、S05のD/R identity | raw DF→数値対角化/係数/basis→有理G6の誤差は未認証。threshold boundというfloat値を完全な丸めcertificateにしない。入力target、変換policy、区間/残差証拠が必要 |
| 2 Taylor truncation | S06 `step_taylor_truncation_residual_bound`：even Kはdegree K+1まで。Mの最高次数m=K+1、scalar exponential tailの既存理論 | 関数はfloat/log/lgammaで実装され、厳密な外向き有理enclosureとは記載されていない。同じx/m・time/単位・比較targetへのbindingが必要。今回は評価しない |
| 3 Return aggregation | M1/M5。同じ有理p/x、Q²=I、符号・順序の下でfinite operatorの恒等式 | 代数上のRA自体に近似誤差を加えない。一方、adapter・型変換・phase/orderがこれを保持したという実装証拠は未取得 |
| 4 Finite-bit係数 | M9/M10、S01：c̃のroot近似によりbias≤ρB_RA≤ρB_o | root interval、d̃、ρ、L1/operator保証scopeを保存。元MとM̃を別targetにする |
| 5 Proposal丸め | πW=c̃は実dyadic πと正確な有理補正による恒等式 | 確率丸めによるtarget変更と係数biasを混ぜない。Wをfloatで集計するならそのpostprocessing誤差は別。π・Wを再正規化／clipしない |
| 6 Angle | S01はsymbolic tangent、G6 finite-bit監査§6にangle enclosureの処理可能性 | 一般的なatan enclosure selectorのSOURCE接続は未実装。float atanを誤差0とせず、符号σ・support sign・rad interval・丸め証拠が必要 |
| 7 Controlled/native | S03 exact CQ仮定下の2 Rz error、S17の`2*row_error+n_provider*delta`。非Pauli監査のweighted mean error会計 | exact CQ仮定と物理δ certificateは別。DF basis・central・control・scalar・actual adjoint・合成誤差のstrict証拠が不足。global-phase最小化残差をcontrol位相証明へ使わない |
| 8 Statistics | M7/M8/M10、S02：U、digital moment/range、残余幅、十分N | 全trial estimator・IID policy・axis/family confidenceへの接続未完成。G7のα=1/320／16 axesを新DF taskへ自動採択しない |
| 9 PF/multi-step/RPE | S06のtruncation composition、S13のordinary repeated metadata、S15の条件付きphase/成功会計 | G6の入力/bit/native biasを含む全wrapper budget未完成。経験的Cを厳密PF上界へ格上げしない。state品質、visibility/branch、RPE統計は別条件 |

## Root・probability・precisionの受入条件

M10は、正の有理p、0<x≤1、有限odd m、指定η/ρ、正supportのもとで十分なbit数が存在するという既存保証である。SOURCEはisqrt interval、largest-remainder dyadic law、acceptanceの下向き丸めとrelative-error検査を持ち、不足ならfail closedする。固定K_root/Hが任意DF入力に成功する保証ではない。

将来のprecision interfaceに必要なのは、入力identity、p/x/m、η/ρ、目的誤差、bit-policy IDを受け取り、**sampling前**に固定されたH/K_rootと、適用した既存下界・enclosure条件を返すことである。新selectorのアルゴリズムや数値を今回選ばない。query中のsupport喪失、enclosure不成立、非正残余幅はtechnical failureであり、zero trialや無料retryに変換しない。

accepted eventはπ、c̃、W、coefficient/root/angle証拠、input/source/policy identityを失わずに保存する必要がある。πW=c̃を満たすことだけで、M̃=M、angle/native誤差0、元DF精度認証を宣言しない。

## U・moment・rangeとbudget

既存idealのB_RA≤U≤B_oを維持する。S02 `upper_normalizer`の区間は**Uのenclosure**であり、B_RAの両側enclosureではない。上界にはhiを使い、点値・midpointや未知の実B_RA/Zへ置き換えない。

G7の既存digital予算はκ=(1+ρ)²/(1−η)^(m+2)、moment upper=κ B_o^+ U^+（full return）、range upper=(1+ρ)B_o^+/(1−η)^(m+2)。`budget`は固定confidenceとcommon-bias規約を持つ。G8は有限provider δとresource-failureを別に組み込むが、δはhypotheticalで、DF providerの実証値ではない。

これらを新taskへ接続する場合、input/Taylor/angle/native等のbiasと統計残余幅を共通targetへ戻す仕様が必要。G7の保守的factor2を片側だけ除去したり、別のHadamard conventionへ無条件に移したりしない。新しい十分N・最適Nは算定していない。

## 保留する判断

保証target、input誤差のcoverage、全taskの精度・confidence/family、direct DF回転かhelper providerか、single-stepからmulti-stepへ移す保証のscopeは未採択。自動precision処理、exact sidecar、native証拠の形式等の技術作業は、これらが固定された後に実装可能である。新保証が必要になればCodex Aの課題順位を変更せず、独立の科学判断としてユーザーへ戻す。
