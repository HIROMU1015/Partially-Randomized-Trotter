# RPE m17：guard上界の保守性に関する静的検討

2026-11-04 JST。**作用誤差とノルム増幅を分離する証明改善には余地がある。
ただし、現行kernelの丸め項も残るため、guard成立の見込みは未確定。**
本書はSOURCE・数式・人工例の検討であり、実H4の新しい数値評価や保存結果の再解析ではない。
既存の`ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED`を変更しない。

## 根拠と固定範囲

基点は結果公開commit `81fa6813194e4e14189092626f66dd10793225e1`、
科学SOURCEは`c3a5cc387b8962b63782dc71ab175147f2d87e33`。
この検討で旧SOURCE・runner・rawを変更せず、旧runnerを実行しない。

対象はlinear H4 / 隣接1.00 Å / STO-3G / 保存legacy DF12 / Bの保存raw physical state /
8 qubit・4電子sector70 / Partial L_D3 / 単一δ=.02 / m17 q131072 r32 K2。
extract_identity_phase、coefficient_atol=1e-12、保存N411/axis・α・β・εを維持する。
保証対象は独立一様53-bit categorical/measurement wordを仮定するModel IIの局所float64仮想測定。
固定PCG64の数学的IID性、実回路、全18round、費用は対象外。

| 静的に確認したSOURCE | 関連する処理 |
|---|---|
| [virtual_bridge_adapter.py](../rpe_m17_virtual_measurement_bridge_preparation_20261104/virtual_bridge_adapter.py) | action_growth / rotation_growth / trajectory_growth / signal_error / guard_completion |
| [bridge_pipeline.py](../rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/bridge_pipeline.py) | finite screen、complex abs、guard、平均誤差を使うstrict判定の接続 |
| [kernel_bounds.py](../rpe_m17_openblas_kernel_bound_preflight_20261104/kernel_bounds.py)、[導出](../rpe_m17_openblas_kernel_bound_preflight_20261104/ROUNDOFF_BOUND_DERIVATION.md) | 現行binary/dispatch/FP条件付きのmatvec・scale・sub・dot・abs上界 |
| [元state-action](../../../src/trotterlib/rpe_hadamard_failure_validation.py)、[測定adapter](../rpe_fresh_iid_measurement_a_preparation_20261103/fresh_iid_adapter.py) | product→rotation→phase、F→events→R、vdot、検査→clip→Bernoulli |
| [直前の実行報告](../rpe_m17_virtual_bridge_scientific_execution_20261104/REPORT.md) | 保存された結果・限界の引用のみ |

## 保存された結果から言えること

直前報告に記載された概数は、平均作用誤差majorant約8.0111580176e-8、
一様作用誤差majorant約2.5546158311e-6、signal magnitude一様上端約1.00000255461583。
今回の転記でnorm・差・比・予算・成分別寄与を新しく計算していない。
finiteは確認済みだが、clip前の座標閾値`1.+1e-12`とcomplex abs閾値`1.+1e-10`を
一様上端では認定できなかった。これは実際のabort trajectoryの発見ではない。

保存結果はaggregateと成分残差/Frobeniusを含むが、各項を分離した全伝播の寄与表ではない。
以下は**保守性を生む構造の特定**であり、実H4での寄与の順位・支配項の断定ではない。

## 保守性の発生箇所

1. `Growth(g,a)`は理想unitaryからの作用誤差を`(g−1)||v||+a`で覆い、同時にノルムを`g||v||+a`で覆う。
   位相のずれなど、ノルムを増やさない誤差までノルム増幅として長列に伝播する。
   例えば数値作用が`−I`、参照が`I`なら、作用距離は2でも数値作用のノルムは1である。
2. 行列残差とkernel丸めの両方をノルムで三角不等式により加算する。
   matvecの`κ_mv ||A64||F`は、A64が厳密unitaryであっても消えない。
   Frobenius上界はspectral normや到達状態上の作用より大きくなることがある。
3. `bound_context`は成分別残差の最大値とFrobeniusの最大値を別々に取る。
   同じ成分が両方の最大を達成するとは限らない。さらにrotationの三角不等式、
   order別gain/additiveの別々の最大化、q/rの反復が保守性を重ねる。
4. 平均伝播は確率加重できるが、全support guardは正のmassを持つ全order/componentを覆う必要がある。
   小さい正の確率は除外理由にならない。元のlawで確率0と証明された要素だけ除外できる。
5. `guard_completion`は一つの円板`|z64|≤M`でRe/+Imの両座標も覆う。
   座標ごとの全support包絡があれば改善し得るが、平均値が内側にあることから
   各trajectoryの座標余裕は導けない。座標閾値とabs閾値も異なる。

## 改善案：作用誤差と独立のノルム伝播

以下の導出は有限・no-overflow・既存kernel/FP条件を前提とする。
`A`は保存binary64行列を**厳密なdyadic行列として解釈**したもの。
`ρ_A ≥ ||A†A−I||₂`を包含演算で証明できれば、任意のvについて

    ||A||₂ ≤ sqrt(1+ρ_A) = p_A
    ||fl(A v)|| ≤ (p_A + κ_mv ||A||F) ||v|| + a_mv.

理想行列との距離δ_Aをノルム用係数へ入れる必要はない。
`ρ_A`はFrobenius欠損で上から覆ってもよいが、数値Aをunitaryと仮定して0にしてはならない。
成分P、forward、reverseの**すべて**について別々に必要。
sectorを切り出した数値basisからの構成だけで厳密unitarity/Hermiticityを仮定しない。

### Rotationの結合構造と丸め項

保存された実数binary64のc,sと同じ数値Pに対し、厳密行列`T=cI−isP`を定義する。
数値実行をこの一行列へ置換するのではなく、証明上だけ使う。

    T†T−I = (c²+s²−1)I + s²(P†P−I) + i c s(P†−P)
    ||T†T−I||₂ ≤ D_T := |c²+s²−1| + s²ρ_P + |c s| h_P,
    h_P ≥ ||P†−P||₂.

欠損の認定上界としてρ_T=D_Tを選べる。あるいはT†T−Iを直接包含評価してρ_Tを求めてもよい。
数値PがHermitianと未証明のまま最後の項を落としてはいけない。
θに対するc/sの位相誤差と、c²+s²のノルムへの影響は異なる。
ideal θとの誤差は平均用の作用誤差側に保持する。

実行順は元どおり`c*v − i*s*fl(P@v)`で、scale/subtractの丸めを残す。
以下、c₀=|c|、s₀=|s|、p≥||P||₂、f≥||P||F、d=κ_mv f、
k=κ_scale、t=κ_sub、a_m/a_s/a_tを各加法項とする。

    A₀ = (1+k)c₀
    B₀ = (1+k)s₀(p+d)
    E_rel = k c₀ + s₀ d + k s₀(p+d) + t(A₀+B₀)
    E_abs = (1+t)((1+k)s₀ a_m + 2a_s) + a_t
    ||rotation_fp(v)|| ≤ (sqrt(1+ρ_T)+E_rel)||v|| + E_abs.

これは`rotation_growth`と同じ二つのscale・matvec・subtractを覆う。
finiteなphase=±1、−iのsign/swapの扱いも変更しない。
underflow加法項をfloatに落として0にしない。

### 合成と最後のsignal

ノルム用affine mapを`H_N(x)=g_N x+a_N`とし、既存と同じ合成・binary powerの数式を使う。
product→rotation→phase、F→r events→Rの順序、正のmassを持つ全supportを維持する。
`n=||ψ_raw||`、全列のノルム上界`H_N(n)`、dot係数をκ_dot,a_dotとすると

    M_N = n H_N(n) + κ_dot n H_N(n) + a_dot

がraw float signalの絶対値を覆う。nは保存norm²区間を上向きに扱い、ψを正規化しない。
finite screenとcomplex absの誤差上界も併せて、既存のbinary64閾値に比較する。
座標別の独立包絡がない限り、同じM_Nを両座標へ適用する。

**ノルム用gainを既存`signal_error`へ代入してはいけない。**
同関数は`gain−1`を理想作用からの誤差として使うため、位相差を捨てると平均誤差を過小評価する。
平均用の旧作用誤差とguard用の新ノルム上界を明示的に分け、
`classify_axes`には引き続き証明された平均作用誤差を渡す必要がある。
将来実装する際は同じ`Growth`型の意味を黙って変更しない。

## 改善しても残る障害と低コストな確認順

ノルム欠損ρが0でも、現行κとunderflow項は残る。
一般の人工モデルで、raw norm=1、L個のunitary行列作用、各Frobenius上界f、
各matvec相対項κf、dot相対項κ_d、加法項0なら、構成される上端は

    M_floor = (1+κ_d)(1+κ f)^L ≥ 1+κ_d+Lκ f.

これは**このmajorant方式の上端の下限**であり、実際のFP誤差・signalの下限ではない。
この楽観的上端でも閾値を超えるなら、同じ係数・同じ積型ノルム証明でρだけを鋭くしても足りない。
実H4のn/f/L・rotationを代入した新しいfloor評価は今回は行っていない。

したがって次の順が合理的である。

1. 次の技術準備では、実行順と最小不可避項から**現行係数での楽観的な証明限界**を整理する。
   新しい実データのnorm/予算数値評価は別の科学実行範囲に固定する。
2. その方式で成立の余地があれば、同じ保存contextに対するρ_P/h_P/ρ_F/ρ_R/ρ_Tの包含と
   成分ごとの係数を準備する。成分別にδ_i+κf_iを評価してからmaxを取る改善も併用できる。
3. 不足する場合は、同じ実kernelについて演算順・行ごとのroundoff・到達状態の構造を使う
   さらに鋭い**証明**が必要か検討する。kernelや算術モデルの変更とは区別する。
   構造が使えることも、全supportに適用できることも自動的には保証されない。

Arb precisionの増加だけでは、意図して加えたκ・Frobenius化・三角不等式・worst supportの項は消えない。
既存96 bit包含の半径が定量的に支配したとは判定していないため、追加precisionを第一策にしない。
sampling増加、guard緩和、状態のrenormalization、失敗shotの除外/置換、α・β・ε変更も採用しない。

## 到達点

旧コードの数式は、作用誤差とnormの両方の十分上界を与える設計である。
今回の発見はその**正当な上界の保守性**であり、バグや旧計算の無効性を示すものではない。
ノルム専用の証明は改善候補だが、実H4でguardを満たすかは未評価。
固定PCG64のIID保証も得られない。

[人工例](test_guard_bound_examples.py)はこの区別と反例を検証する。
[次段案とSTOP](NEXT_STAGE_DECISION.md)、[検証記録](VALIDATION.txt)を参照。
