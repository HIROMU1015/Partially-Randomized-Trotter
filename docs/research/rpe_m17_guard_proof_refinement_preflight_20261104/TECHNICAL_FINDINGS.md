# RPE m17：現行kernelでguard証明を鋭くするための技術調査

2026-11-04 JST。**保存norm²の再計算は通常不要。現行kernelでも構造を保てば上界改善の余地はあるが、
H4の全support guardを認定できる見込みはまだ条件付きである。**
次は行ごとの丸め式と符号付きenergy伝播を人工系で接続する技術準備を推奨する。
単に積型係数を少し小さくすること、norm²をexactに再計算することを第一策にしない。

基点：`6d4b1434945bad2d16c91ce87dbeb58d6946e16c`の[証明限界](../rpe_m17_norm_only_adapter_preparation_20261104/PROOF_LIMIT.md)。
固定対象はlinear H4 / 隣接1.00 Å / STO-3G / 保存legacy DF12 / B raw physical state /
8 qubit・4電子sector70、Partial L_D3 / 単一δ=.02 / m17 q131072 r32 K2。
extract_identity_phase、coefficient_atol=1e-12、保存N411/axis・α・β・ε、現行NumPy/OpenBLAS profileを維持する。
数学的IID uniform53下のModel IIを対象とし、固定PCG64のIIDは保証しない。
本調査は静的SOURCE・byte identity・数式・人工検証に限定し、新H4計算・保存値の数値再解析は0。

## 1. 保存norm²は同じraw stateを包含している

包含結果commit `34840ce9bf81d985b32764d9bc9d77645ed52e59`、科学SOURCE
`18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`、96 bitの保存state recordを確認した。

| 対応する原本 | 固定commit / 処理 |
|---|---|
| B `physical_state.npz:state.npy` | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`、保存raw state |
| 包含 `source_replay.json:state` | `34840ce9…`、同じvectorをreal/imagのbinary64 hexとしてexport |
| 包含 `RESULT.json:enclosures[0].state.saved_vector_norm_squared` | `34840ce9…`、上記hex vectorの96 bit包含 |
| bridge `numeric_context.npz:psi.npy` | `81fa6813194e4e14189092626f66dd10793225e1`、科学SOURCE c3a5cc387の保存context |

5原本（上表の4種とbridge RESULT）を固定Git blob・SHA256へ照合した。
BとcontextのNPY headerは`<c16`、shape70、C orderで、データpayloadを数値arrayへdecodeせず比較。
export hexをlittle-endianの実/虚binary64 bytesへ戻した表現とも一致した。
**3表現の1,120 bytesがsigned zeroを含めて同一**。
payload SHA256は`a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。
NPZ containerのhashと、NPYのデータpayload hashは区別する。

保存norm²区間の原文は次のとおり。新しい和・幅・小数変換・閾値比較を行っていない。

    lower = 39614081257132153570591926899/39614081257132168796771975168
    upper = 39614081257132153570591926901/39614081257132168796771975168

[ball_adapter.expectation](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/ball_adapter.py)は
まず`norm2=(psi†psi).real`を包含し、次に期待値だけをnorm2で除算してnormalized rayを扱う。
state recordの`normalized ray; exact division, no raw-vector rewrite`は期待値の意味であり、
保存norm²が正規化後の1を意味するものではない。
[export_adapter](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/export_adapter.py)はB state fingerprintを確認し、
vector順を変更せずhexへexportする。
[bridge_pipeline](../rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/bridge_pipeline.py)も
B raw bytesとexportを照合し、そのψをcontextへコピーし、包含結果のnorm²区間を既に再利用している。

したがって同一raw stateのノルムを**上から／下から包含する用途には、保存区間をそのまま使える**。
norm²は入力成分の二次形式で、包含されたideal meanのq乗やoperator半径を再評価する必要はない。
任意の非unitary state preparationや別stateへの保証を与えるものではない。

前段のexact norm²診断は、exact値自体が必要な目的、または将来の許可済み比較で保存区間が
判定閾値をまたぐ場合に限って検討すればよい。今回はその比較も行っていない。
通常のguard上界検討のために、6d4b1434の`--run`を先に実行する必要はない。
旧SOURCE・旧科学実行計画は未実行履歴として保全し、内容を変更しない。

## 2. genericな積型係数を短くするだけでは足りない理由

旧上界は理想unitaryからの作用誤差をnorm増加へ変換し、Frobenius/Cauchy・非負項・
worst component/orderを用いて、長列をaffine mapの積として覆う。
前段で、絶対欠損`||A†A−I||`を用いる置換にも現行κの積み重ねが残ると示した。

さらに、仮に各F/Rの係数を`g=1+u`まで小さくできても、同じ独立・非負の積型方式なら、
unit normでF/Rだけの2q回から

    (1+u)^(2q) >= 1+2q*u = 1+2^-35,
    coordinate_limit = 1+563/2^49,   2^-35 = 16384/2^49

となる。単なるdepth短縮やFrobenius係数の改善だけでは、座標guard認定に届かない場合が残る。
これは**各作用へ非負のu以上の余裕を課す方式についての条件付き限界**である。
実kernelの誤差が常にu以上、という主張ではない。exactに作用する人工例は後述する。

norm-only仮定`||e_j||≤η||v_j||`だけでは、誤差を毎回外向きにそろえるモデルを排除できない。
そのモデルでは`||v_L||=(1+η)^L||v_0||`となり、√L累積や二次誤差だけへの置換は導けない。
これは抽象誤差モデルの反例で、実GEMVのA=Iがその誤差を出すという意味ではない。
categorical抽選がIIDでも、状態依存の決定論的丸め誤差が独立・平均ゼロになるとは限らない。
平均・高確率・有限samplingでの正常完了は、正のmassを持つ全supportのguard証明を代用しない。

## 3. 現行kernelで使える改善と、その追加条件

### 行ごとの丸め式：導出済みforestをnorm化する前に使う

現行SOURCEの[forest導出](../rpe_m17_openblas_kernel_bound_preflight_20261104/ROUNDOFF_BOUND_DERIVATION.md)は
各実出力について`γ_D Σ|real product| + Jη/(1−Du)`を与える。
`D=2n+16, J=4n+16, η=2^-1075`、binary/dispatch/FP/no-overflow条件は旧profileのまま。
複素行列の実表現を`B=[[Re A,-Im A],[Im A,Re A]]`、vector実表現をxとすると

    |e_i(x)| <= γ_D (|B| |x|)_i + τ,   τ=Jη/(1−Du).

この段階を保持すれば、すぐκ||A||F||x||へ潰す必要はない。
既存係数から`γ_D=mv_rel/2`、`τ=dot_abs/2`を再利用できる。
人工例では同じ係数のまま、行ごとの情報を残した方が旧Frobenius boundより小さくなった。

ただし実H4での改善量は未評価。全到達xの包絡が必要で、典型stateで代用できない。
成分ごとのboxだけでは`|B|`の反復が人工的に膨らむ場合もあり、energyや相関を残す表現が必要になる。
depthをさらに短くするなら、同じbinaryの実graphを根拠に証明し直す。係数をfitしない。
underflow、scale/subtract、最後のvdotを忘れず、kernel自体は変更しない。

### 符号付きenergy：radial誤差を別に証明する

正定値H、数値作用`f(v)=Av+e(v)`に対して、恒等式

    E_H(f(v)) = E_H(v) + v†(A†HA−H)v
               + 2 Re((Av)†H e(v)) + e(v)†H e(v)

を使う。H=Iなら通常のnorm²。
signed driftとsigned radial項を全到達vで証明できれば、位相誤差をnorm膨張として扱わずに済む。
H=I、A=I、e=(i−1)vの人工例では、radial=-||v||²と二次項2||v||²が相殺してnormを保存する。
作用誤差は非zeroのままであり、平均用の旧作用誤差を0にしてはいけない。

**||e||の上端だけからradial≤0やradial=0は導けない。**
符号付き丸めcell、exactなzero/sign/swap、到達集合上の代数など、新たな根拠が必要。
Hが計算で得られる場合は正定値性と各不等式を包含で認定し、経験的なHのfitだけで保証しない。
unitaryという性質だけでは、全vでのstrictな収縮は得られない。

数値rotation `T=cI−isP`のdriftには

    T†HT−H = (c²+s²−1)H + s²(P†HP−H) + i c s(P†H−HP)

を使える。数値Pを未証明のままH-Hermitianとして最後の項を落とさない。
実rotationのmatvec・二つのscale・subtractの丸めはeに含める。
理想θとのc/sの位相差は平均用の作用誤差側に残す。

### F/R相関と作用順序：証明のblock化とプログラム変更を分ける

実際の順序は`R E_r ... E_1 F`で、F/Rの間にeventsが入る。
R=F^-1がexactに成り立っても、一般にeventsをまたいで消去できない。
人工例F=R=X、event=ZではR E F=−Zとなる。
eventsなしでも、`f_R(f_F(v))`の誤差は`R e_F(v)+e_R(Fv+e_F(v))`であり、逆行列関係だけでは相殺しない。

一方、同じ作用順序のblockを証明上まとめ、jointな丸め・状態の相関を保持することは可能。
全event supportで証明できることが必要で、列を平均してから作用させない。
実行を`fl((R E F)v)`へ置換したり、乗算順を変える最適化は今回の案に含めない。

### 収縮・weighted normにはprefixとobservableへの戻しが必要

認定されたsigned spectral/energy boundからg<1を使う場合、旧`Growth(g>=1)`へ押し込まない。
最終normが小さくても中間が大きいことがある。人工例「4倍→1/8倍」のblockは最終1/2倍だが、prefixは4倍。
finite screenには全prefixとrotation内部の一時vectorの上端が必要。

norm-only affine mapのfinal/prefix包絡を分ければ、合成・binary powerで長列を扱える。
prefixを`p*x+b`で覆い、after/beforeを合成すると、prefix係数の安全な上端は

    p = max(p_before, p_after*g_before)
    b = max(b_before, p_after*a_before+b_after).

Hを使った後も最後のsignalには

    |psi†v| <= sqrt(psi†H^-1 psi) sqrt(v†Hv)

と実vdotの丸めが必要。初期stateを含むmetric prefactorは、一般にraw norm²より大きい。
Hのcondition numberの損失だけで余裕を失うこともあり、energy収縮だけでguard GOにしない。

### 構造で改善できる存在例と、H4で未確認の点

signed permutation / ±i位相の人工GEMVは、各実rowに非zeroの±1 productが高々一つ。
finite入力、gradual underflow、既存graphの条件下では、zeroとの加算・sign/swapはexact。
基底vectorの有限orbitが閉じ、任意長の全support列でもnormを保存する例を作れる。
したがってgeneric κの積型下限は、現行kernelのあらゆる証明に対する不可能性ではない。
人工テストはorbitの閉性と既存guardへの数学的接続を確認したもので、実runtime認定ではない。

実H4に同じexact構造や有利なsigned radial boundがあるかは未調査・未評価。
今回P/F/Rの成分や新しいmatrix normを調べて、そのような性質を主張していない。

## 4. 最小prototypeと次段

[energy_prototype.py](energy_prototype.py)に、旧係数を使う実row誤差式、signed energy恒等式、
g<1と全prefixを扱うnorm-only affine演算を追加した。loader/RNG/科学CLIはない。
これは人工用のproof primitiveであり、実到達集合・drift/radial boundを生成する科学adapterではない。
旧SOURCEは変更せず、旧作用誤差とinterfaceも分けた。

新人工12＋既存25、計37件PASS。詳細は[VALIDATION.txt](VALIDATION.txt)。
改善量の定量保証やH4 guardのGOは得ていない。
`U_virtual_IID_upper=null`、`U_exec_upper=null`、全体`UNDETERMINED`を維持する。

次段案は[最小検証計画](MINIMAL_VALIDATION_PLAN.md)に記載。
**Codex**が次の依頼後に人工系のsigned-energy/row graph接続を準備し、
実データ評価が必要になればSOURCE・入力・精度・call範囲を固定して公開する。
**ユーザー**はその公開SOURCEに対する新科学実行を別途判断・承認する。
今回、旧norm診断の実行や新科学計算・重要な独立科学レビューは開始しない。
専用branchへ公開・remote照合した後に**mandatory STOP**する。
