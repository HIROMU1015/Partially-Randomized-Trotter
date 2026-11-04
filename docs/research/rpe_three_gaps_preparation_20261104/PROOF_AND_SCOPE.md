# RPE残存3課題：数学的条件、SOURCE、保証の限界

2026-11-04 JST。**技術準備・人工検証のみ。実H4の新しい数値判定は未実行。**
固定対象：linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D3／δ=.02、
保存physical state ray・4電子sector70、m0..17の保存q/r/K/N/α/β/ε/E_ref。
旧D結果89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f、旧D SOURCE6942e5176d2dec115ac3c271e48f84377abda915を変更しない。

## A. 保存αを変更しない直接Hoeffding上界

ν_m,aをideal categorical ±1測定の真の平均、μ₀_m,aを保存binary64のexact中心とする。
旧36軸の包含から|ν−μ₀|≤U_cat_upperが保証される。εは旧包含と同じexact decimal。
各軸内で独立な[-1,1]有界shotの標本平均hについて、t=ε−U_cat_upper>0なら

    {|h−μ₀|≥ε} ⊂ {|h−ν|≥t}
    Pr(|h−μ₀|≥ε) ≤ 2 exp(−N t²/2).

従ってG={全36軸で|h−μ₀|<ε}に対して

    Pr(G) ≥ max(0,1−Σ 2 exp(−N(ε−U_cat_upper)²/2)).

U_cat_upperは既存包含に対する有効な上界であり、それを使うこの評価も全包含内のνに一様。
ここではUをεから一度引く。phase側へUを再加算せず、PF/RTE予算も再び加えない。
Uを新たに小さく推定する処理やmean包含の再計算はない。

旧Σαは保存された配分の和。新しいp_upperは同じN/ε/Uから直接得る失敗確率の上界であり、
**新しいα配分ではない**。Nの整数化による余裕があり得ても、Σp_upper≤.05の成立は事前に断定しない。
旧α・N・ε・U_cat・成功目標を変更しない。

`alpha_adapter.failure_upper`はArbでtと指数関数を包含し、確定した外向きdyadic上端を返す。
`aggregate`はその上端36個をexact rationalとして足し、exact target1/20と比較する。
**≤**が判定条件であり、許容差を足さない。tのstrict正値が不明、入力不正、上界不明はUNKNOWN/STOP。
上端和が目標を超えた場合は十分条件の不通過であり、実際の失敗率>.05の証明ではない。
float underflowの0を数学的確率0とせず、Arbの指数・端点を保持する。

union boundにround間・軸間の独立性は不要。各軸のHoeffdingにはshot内独立性・有界性が必要。
fixed PRNGのIIDや実回路の測定完了は証明しない。V2の保存中心点の二項確率は使用しない。
既存Dで確認したG→数学的branch→literal energyの含意はそのまま再利用する。

## B. Float64 decoder：何を調べ、何が未解決か

既存`reconstruct_rpe_phase`と関連testを読み、旧Dの数値誤差条件を再利用した。
q₀=1、q_m=2q_(m−1)、共通φに対する真の測定位相誤差b_mを仮定し、
a_mを入力平均→angle→principal-wrapの円周誤差、c_mを全rootのbase-phase誤差、
d_mを全computed distanceの絶対誤差とすると、十分条件は

    B_m=b_m+a_m
    π−[2B_(m−1)+B_m+q_m(c_(m−1)+c_m+d_m)] > 0.

最終実装位相誤差はB_17/q_17+c_17。
全候補と理想rootの対応・有限性・欠落なしも必要で、argminがtieで一つ返すだけでは不十分。
既存Dの`branch_conditions`がこの条件を評価する。a/c/dが欠測ならUNKNOWNのまま。

### SOURCEで分かった演算経路

| 段階 | SOURCEと問題 |
|---|---|
| 標本平均 | `_axis_distribution`／`_marginal_branch_monte_carlo`は`2.0*k/N−1.0`。fresh-IID診断の`math.fsum(±1)/N`とは丸めが異なる。人工N3で差を確認 |
| phase入力 | `np.angle`。NumPy1.26.4のPython SOURCEでは実部・虚部を`arctan2`へ渡す |
| wrap | `(x+pi64) % (2*pi64)−pi64`。seam、剰余、加減算と符号付きzeroを含む |
| root生成 | `wrap((principal+2*pi64*k)/q)`。π64・乗算・加算・除算・wrapの誤差が必要 |
| 距離 | `abs(angle(exp(i*(candidate−previous))))`。下側C/ufunc経路の超越関数誤差が残る |
| 選択 | finite距離のstrictな順序が保たれればargminは一意。np.partitionは診断だけ |

公式[NumPy1.26 angle SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/lib/function_base.py)と
[arctan2説明](https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html)も確認した。
後者は下層C libraryへの対応と特殊値を説明する。
**本調査でruntime全域に適用できる検証済みulp/絶対誤差契約は取得できなかった**。
これは保証が不可能という証明ではなく、必要な数値誤差証拠が未取得という判断である。
ローカル観測はNumPy1.26.4／glibc2.35。versionだけから正しく丸められると仮定しない。
scalarとvector ufuncのdispatch、meanの生成経路、rounding modeまで同じ契約にする必要がある。
停止済みの仮想float guard/kernel系列は再開していない。

### 有限測定格子の活用と計算量

各軸のexactな実現値は(2k−N)/N、0≤k≤N。全ての格子点について
|h−μ₀|<εをexact rationalで選べば、成功域の有限集合を漏れなく扱える。
μ₀には保存binary64のexact値を使う必要があり、表示decimalを無断代入しない。
単一roundのangle誤差aは、全成功格子について実装値とArb真値を包含比較すれば、
libmの一様定理がなくても**その有限入力集合**では証明可能である。
少数の高精度比較や経験的最大値とは、全実現値のcoverageが違う。

しかしangleだけで全decoderは証明できない。P_mを成功cos/sin格子数とすると、
単純な全履歴列挙は∏P_m、全root生成はΣP_m q_m、前estimateとの全距離は更に組合せが増える。
保存Nが数百、q_max=131072の全18roundへそのまま使う方法は実用的なSOURCEとして採用しない。
今回、実H4のP_mや成功格子を新たに数えていない。

理想rootの等間隔性2π/qにより、誤選択を防ぐ距離差は旧Dのstrict marginから評価できる。
これと一様c/dを組み合わせれば全候補の明示列挙を避けられる。
ただし、一様dや計算距離の順序保存が未証明の状態で「近い2候補だけでよい」として
他のcomputed candidateを省いてはならない。
π64/基本演算/剰余の解析、あるいは到達する全距離引数の保証付き被覆が次の未解決部分である。

### 最小の人工実装の範囲

`decoder_finite.py`は、与えた小さい人工prefixの**全成功count履歴**について、既存decoderを無変更で実行する。
Arbのcos(root−φ)によるstrict oracleと選択branchを比較する。cosを使うのは、誤rootがπで
Argのcutをまたぐと距離の下端を失うため。既存decoder自体は変更しない。
人工2round、各round6成功格子、全36履歴を検証し、意図的な誤branch・tie・originも検出した。
mean変換・signed zero・subnormalも人工検査した。

このcheckerの保証は**供給された有限prefix・scalar呼出し経路**だけ。
H4の全域、一様a/c/d、vectorized decoder、最終energyまでの保証ではない。
本packageはBの実H4実行入口を提供しない。無制限のadapter追加でこれを埋めようとしない。

最小の代替案は、対象mean/ufunc経路を固定した上で、bounded primitive誤差と
全computed distanceの被覆・順序保存を静的に証明すること。
それが揃う前に新しいH4全域検証SOURCEを実行可能とはしない。

## C. 保存E_refと最小固有値の包含

### 由来と対象の区別

保存snapshotはc、one_body8×8、lambda12、g12×8×8のbinary64値を含む。
**保存された70×70 dense Hamiltonianそのものはこのsnapshotにはない**。
state NPZには70-vectorとbasis_indicesがある。今回はZIP/NPY headerだけを確認し、数値arrayをdecodeしていない。

既存`finite_rte_signal_validation`は同snapshotの4電子sector dense行列を作り、
`np.linalg.eigh`の最小列をphysical state/E_refとして使う。
Bの保存結果e22d7718c47358ff650dbc04e8e72f6ab1c8bf41にも同じE_ref・state fingerprintが記録される。
`solve_df_ground_state`の別経路はeigsh(which='SA')、residual等を返すが、その要求・小残差だけを最小性証明にしない。
調べた経路に保証付きspectral-gap証拠は見つからず、本経路ではgapを仮定しない。

| 対象 | 本SOURCEでの扱い |
|---|---|
| 1. 保存DF Hamiltonian | 保存binary64係数をexactとするH=cI+T(h)+Σλ_l T(g_l)²の4電子sector最小固有値λ_minを優先 |
| materializeしたFloat64 dense行列 | 上のexact polynomialとは区別。旧dense演算の丸め済み行列自体を新しいtargetに置き換えない |
| 2. upstream構築誤差を含むHamiltonian | 本SOURCEは保存係数からsector行列への演算丸めを包含する。元積分・DF構築の誤差を含めるには別のΔ_HとWeyl型移送が必要 |
| 3. 元分子・実際の物理energy | basis/model/DF誤差、他sector、実state preparation等を含めずUNKNOWN |

### exact係数からのsector行列

snapshotの係数が**exactなHermitian**かを、実行時に各binary64のreal/imag rationalで検査する。
HermitianでなければUNKNOWN/STOP。許容差で丸めたりsymmetrizeして通さない。
全4電子basisが同じsorted整数列で完備かも確認する。
既存`_before_mode_masks`、`_build_excitation_table`のFunctionDefを変更せず再利用し、
fermion符号・OpenFermionのmode/basis規約を保持する。

T(h)、T(g_l)をAcb行列へ持ち上げ、12回のnative行列積でHを包含する。
旧NumPy matvec・対角化・分子生成・PF/RTE回路構築は呼ばない。
exact targetのHermitian性を係数から証明した後に限り、下三角の包含と共役上三角、real対角で
`HermitianEnclosure`を作る。これは元targetの構造を用いた包含であり、target変更ではない。
rawの非Hermitian行列をground証明へ黙って入れるAPIは拒否する。

### 最小性を識別する証明

事前固定したtrial lower shiftをL=lower(E_ref−10^−8 Ha)とする。
これはη_ref=10^−8を**仮定する操作ではない**。H−LIの正定値性を検証するためのtrialである。
interval LDL*の各pivotを

    d_j = H_jj − L − Σ_(k<j)|l_jk|² d_k
    l_ij = (H_ij − Σ_(k<j)l_ik d_k conj(l_jk))/d_j

で包含し、全pivotの下端がstrict正ならH−LI≻0、従ってλ_min>L。
正値が不明・zero/負pivot・非有限ではUNKNOWNとし、pivot変更・shift調整・精度追加を自動で行わない。
これは負pivotから失敗機構を推定するinertia診断ではなく、最小固有値の下界の十分証明である。

保存raw state v≠0について、正確なnormalized-ray Rayleigh商を包含して

    R = upper(v* H v / (v* v)),  λ_min≤R
    η_ref = upper max(|E_ref−L|, |E_ref−R|)

を得る。vが基底stateだという仮定は不要。
下側の正定値証明が最小固有値の同定を担い、Rayleigh商だけや小residualだけに依存しない。
退化基底でも、Lが全固有値より低ければ同じ証明が使える。
係数のHermitian性、実H4の全pivot正値、Rayleigh商、η_refの小ささは**今回未評価**。

### energy・aliasへの接続

旧D RESULTのliteral energy上端u_Eを再利用し、u_E+η_refを一度だけ外向き評価する。
qδ η_refを各round位相へ重複加算しない。保存energy目標は元binary64のままstrict比較する。
δ η_refが旧alias境界margin下端より小さいことも判定する。
この移送先は数学的decoderの**保存DF sector ground energy**である。
Float64 decoderが未保証なら、その誤差を0にせず、実装energy上界をnullに保つ。
本証明はphysical stateとPF有効固有stateの同一性や実state preparationを要求・証明するものではない。

## 統合・再利用・現状

AはGの確率、Bは実装と数学的decoderの接続、Cはreferenceのtarget移送であり、判定を独立に保存する。
A/Cが両方通っても、Bや実測定bridgeを自動昇格しない。
既存D/Arb/V2 loaderと整数excitation tableは再利用し、新規部分はAの指数上界、
Cのinterval正定値＋Rayleigh、Bの小さい有限prefix checkerに限定した。
既存の一般eigsh/float-eighやArb全固有値分離APIは、単独で目的の最小性証明に適さないため実行しない。
FLINTの[eigenvalue API](https://python-flint.readthedocs.io/en/latest/acb_mat.html)も調べたが、
全固有値の分離や重複固有値の扱いを本計画の必須条件に増やさず、単一shiftの正定値経路を選んだ。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
既存bridge `U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。今回の28人工PASSは科学的GOではない。
