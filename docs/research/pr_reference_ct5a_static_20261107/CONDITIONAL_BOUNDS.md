# CT-5A：T価格差の条件式と、追加証拠に必要な境界

**技術的な静的整理のみ。** 全support strict word policyも全support T優越も証明していない。固定条件は[REPORT](REPORT.md)とCT-4Bに従う。

## 1. Policyと対象をまず固定する

`P_epsilon`を、compiler A後の**literal binary64** `RZ(theta)`へ、同一precision `epsilon_G=10^-11`のstrict wordを返し、その実T/T†数を`t_epsilon(theta)`として記録する仮想の取得policyとする。符号はactual adjoint、Wと全joint relative phaseは保持する。wordを返せない角度がある限り、全support上で`t_epsilon`は定義されていない。

CT-4Bで定義・取得済みなのは固定2 trajectoryに必要な299 positive keyの保存wordである。欠けた入力に対し`t=0`、隣接binary角、理想pi角、固定cohort最大Tを代入しない。CT-2/3Aの`THEORETICAL_ONLY`はこの欠落を埋めない。

各`arm a in {Bprime,reference}`、axis`x`、trajectory`omega`について、認定可能な実native RZ多重度を`n_{a,x,omega}(theta)`とし、

[
T_{a,x}^{P}(\omega)
 = \sum_{\theta} n_{a,x,\omega}(\theta)t_{P}(\theta)
]

と定義する。等式は実際にそのwordを各物理gateへ使い、すべての出現・相殺を正しく数えられたときに成立する。`t_P(-theta)=t_P(theta)`はactual adjointを選ぶ場合だけ成立。期待値より前に、`P`の全到達key取得・strict guard・termination資格が必要。

## 2. Roleと固定差分

V3 `dparts(arm,q)` から、Dを`D_a(q;P)`（D未融合 + signed D境界補正）、Reference固定16 CSWAPを`C_{sw}(P)`と書く。D境界補正はT-costの符号付き*台帳差分*であって、負の物理operationでも負のoperator誤差でもない。

他のroleを次のように定義する。

- `F_a(omega;P)`：RTE basis frame/inverseのT価格。
- `R_a(omega;P)`：RTE中央回転。
- `Q_a(omega;P)`：RTE product。
- `E_B(omega;P)`：B′ event符号位相。Referenceでは既存roleのevent phaseは0だがvacuum Pへ反映される。
- `S_B(q;P)`：B′の固定controlled scalar P価格。
- `V_R(omega;P)`：Referenceのtrajectory-dependent vacuum補正P価格。
- `W_a(x;P)`：測定軸wrapperの価格。
- `A(omega,x;P)`：固定role投影と実native出現・cross-role cancellationの対応が全supportで未証明な場合の**未資格残差**。これは実行される新しいgateではない。

source/wholeとの同一性が成立するなら、

[
\begin{aligned}
\Delta T_x(\omega)
&:=T_{B',x}^{P}(\omega)-T_{R,x}^{P}(\omega)\\
&=\underbrace{D_{B'}(q;P)-D_R(q;P)-C_{sw}(P)
+S_B(q;P)+W_{B'}(x;P)-W_R(x;P)}_{C_{q,x}(P):\ \text{固定価格差}}\\
&\quad+\underbrace{[F_{B'}-F_R]+[R_{B'}-R_R]+[Q_{B'}-Q_R]
+E_B - V_R(\omega;P)+A(\omega,x;P)}_{Z_x(\omega;P):\ \text{trajectory依存／未資格差}}
\end{aligned}
]

と整理できる。この式は**役割ラベルの構造的会計式**である。実nativeの`A=0`を全supportに認定した式ではない。

### CT-4Bの有限価格との対応

[CT-4B REPORT](../pr_reference_ct4b_20261107/REPORT.md)と[ROLE_T_COUNTS](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)では、固定2 trajectoryについて、DのB′比削減177,576、Reference CSWAP寄与17,472を報告している。RTE frame価格は両方式で一致し、RTE中央・vacuum価格に有限trajectory差がある。CT-4Bの8 nativeとroleの一致が確認された範囲では`A=0`となる。

**これらは`q=128,r=1,epsilon=10^-11`の有限採用word価格**であり、`D(q;P)`および`C_sw(P)`の全support資格がまだないpolicyに対する数値保証ではない。旧CT-1（q1,e=-9）と価格を混合しない。

## 3. 必要な保証は「価格そのもの」より「差分の下界」

### Pointwiseな全support優越

保存finite-bit lawのsupportを`Omega`とする。全`omega in Omega`でwordを返すstrict policyが実際に構成・資格されたうえで、

[
\inf_{\omega\in\Omega}\Delta T_x(\omega)>0
]

なら、そのaxis・固定q/r/epsilonでpointwiseなReference T優越を示せる。より保守的な十分条件は、

[
C_{q,x}^{\rm lower}
+\inf_{\omega\in\Omega}\bigl(Z_x(\omega;P)\bigr)>0.
]

`Z`の全項を独立に最悪化する境界は安全な場合もあるが、同じevent列・vacuum角の**相関を捨てると粗くなり得る**。特に`V_R`だけの上界を先に求める場合、

[
\inf_{\omega} \bigl(C_{q,x}+Z_x(\omega;P)+V_R(\omega;P)\bigr)
>
\sup_{\omega}V_R(\omega;P)
]

は十分だが必要条件ではない。右辺にCT-4Bの2 vacuum価格の最大を使うことは不可。

### 期待Tの優越（pointwiseより弱い目標）

canonical lawでの期待が有限に定義でき、

[
C_{q,x}(P)+\mathbb E[Z_x(\omega;P)]>0
]

を保証できれば、固定round/axisの期待T削減を保証できる。全supportに非負T価格が定義されていることが前提。期待値に必要なのは差の**期待価格**であり、pointwise優越は必須ではない。

有限bitの有理multiplicityを再利用できる場合も、trajectory依存vacuum角の価格とevent変数との結合が残り得る。全support`V_R`のpointwise上界`F_vac`を別に得る方法は**一つの十分条件**であり、唯一の方法ではない。差分の平均を直接安全に囲えるならそれでもよい。一般的なtypical synthesis価格を保証なしで挿入しない。

### RPE成功保証との独立性

たとえ価格差だけを挟むことができても、各回路のword取得・strict guard、whole-joint誤差、CT-0のraw disk/headroom、保存N=199/axisなどを全supportで満たした証明にはならない。逆に成功保証が得られてもT優越は別問題。**費用優越とCT版RPE成功は二つの独立した資格**。

## 4. 最小の技術的証拠の順序

| 優先 | 未解決条件 | 既存資産 | 追加作業の境界 |
|---|---|---|---|
| 1 | D項のliteral native値・相殺contextが全supportで固定 | V3 `dparts`・保存catalogue、CT-4B role照合 | SOURCE/contextの静的技術検査。数値再解析・compileが必要なら別承認 |
| 2 | Paired frameが同一T価格となる条件 | V3 `pair`・`saved_path_parts`・`expected_parts` | eventごとの共通multiset/actual adjointとcompiler境界を照合 |
| 3 | RTE中央差の有限key閉包 | V3 `rotation`、CT-0有限bitとCT-4B word | すべての到達event角を漏れなく識別。新規key/価格計算は別承認 |
| 4 | B′ event位相とproduct価格 | CT-0の有限superset、V3 `integer_pi_phase`／`fixed` | 浮動加算順、CZ/Z loweringとsource cancellationを確認 |
| 5 | Reference vacuum P価格の上界・期待価格 | CT-0 `PHASE_SUPPORT`、CT-1/4有限word、CT-2/3A停止性調査 | 全support strict word policyと価格cap／または安全な期待差の保証。今は未資格 |
| 6 | 実word policyとraw-disk成功 | strict guard/emitter、CT-0誤差・保存N、CT-4B誤差 | 全keyで合成可能/有限停止・guard・joint誤差を証明。pilot PASSを代用しない |

最初の4項目を絞ると、**具体的にどの未知差分にどの程度のT価格境界が必要か**が判定できる。CT-3BのSK初期netを全自動構築するのは現段階で必須とは確定しない。

## 5. 今回は行っていないこと

- 未取得keyを含むT価格の新規数値評価、差分の極値・期待値の再計算
- 既存rawからの新ランキング・試験的threshold値の取得
- 新規合成／再guard／native compile／simulation
- 新しいepsilon／alpha／shotの採択、最適化
- `A=0`、`F_{B'}=F_R`の全support証明
- Reference vacuum Pの全support word価格保証
- CT版RPE成功資格、全18round期待総T、T-depth

**CT-5A静的調査としてmandatory STOP。** 次の実行scopeは未固定。旧SOURCE・raw・result・grant・ledger・marker・STOPは変更しない。
