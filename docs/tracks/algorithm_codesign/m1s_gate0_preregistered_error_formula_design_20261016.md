# M1-S Gate 0：結果前誤差算定式とcoverage条件（候補、2026-10-16）

## 状態と対象

**技術準備のみ。8項目の算定式候補を具体化し、1項目の科学的品質係数は未決。正式採択0、新規科学計算0、数値correctness test0、算定式の数値評価0。mandatory STOP。**

基点は `4019d045cbfad94ba9333b544f7950a6d10d467d`。適用共通指示は
[2026-10-16統合正式版](../../research/research_project_instructions_unified_2026-10-16.md)。
[旧17項目設計](m1s_gate0_precision_error_budget_design_20261016.md)と
[v2契約候補](m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md)は履歴として保全する。
今回の既知誤差解析の適用は独立した科学レビューでも新M1-S定理でもない。

対象は採択済みの限定scopeだけ：linear H4、1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、
保存順prefix G3/G6/G9、delta windowなし（有限step評価なし）。JW signed aggregation、identityをHDへ移す、
保存数値oracle、W/v/ρの診断、同じideal task/conceptual event positionsの比較を保持する。
DF-native law・finite-step dressing・native shots/費用・full-Fock ground/nondegeneracy・T1cは評価しない。

## 何を誤差のtargetにするか

NPZ内binary64値を**正確な入力値**とみなし、その値から数学的に定まるHermitian Hと、
非冗長signed Pauli lawをtargetにする。DF近似・入力生成誤差・化学精度・真の基底状態への距離は含まない。
状態targetは保存vectorをその正確なnormで割って期待値を取る数学的oracleであり、保存vectorは書き換えない。
残差から状態距離へ変換するには別のgap等が必要で、今回その証明も数値評価もしない。

## 共通の条件付きモデル

u=2^-53、γ_k=ku/(1-ku)、ku<1。通常の演算ごとの相対誤差モデルと
matrix productの絶対値積によるforward errorを用いる。
[Higham–Mary, §1, Eq. (1.1)–(1.2)](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf)の
**古典的γ_k**だけを使い、確率的なsqrt(k)への置換はしない。
[NumPy matmul](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)はBLAS依存であり、
version/build hashの一致だけではその演算誤差保証を証明しない。
[CPython math.fsum](https://docs.python.org/3.11/library/math.html#math.fsum)にもdouble roundingの注意がある。
したがって以下のcoverageは**条件付きengineering envelope**であり、この段階でrigorous certificateと呼ばない。

必要条件は次のとおり。

1. binary64、round-to-nearest、複素積/BLAS dot等が保守的なγ_(8k+32)モデル以内。
2. fsumの絶対誤差がγ_(m+32)×入力絶対値和以内。correct roundingは要求しない。
3. 未会計のoverflow/underflow/FTZがない。nonfiniteはSOURCEが停止するが、BLAS内部のsilent underflowを全検出する実装ではない。
4. 入力one_bodyと各gがexact Hermitian（新v3のformula pathではarray equalityで確認）。repairや新しい対称化はしない。
5. frozen source・入力hash・runtime/build/thread・call数上限が一致。compiler/reduction順序を変えない。

これらは未検証の仮定を明示したもの。`coverage_model_adopted=false`を維持し、ユーザーがこの条件付き評価の扱いを採択するまで実行不可。
条件が成立しない場合に係数や閾値を広げてcoverage済みにすることは禁止する。

## 記号、演算数、入力尺度

d=256、sqrt(d)=16、M=65536（8-qubit canonical Pauli table上限）、C=1000000（既存contribution上限）。
静的支持数はone-bodyで高々121、既存SOURCEの総combine contribution上界は706837でC以内。
複素積の実演算数≤8、degree≤4のJW polynomial、符号・±i・1/2のexact operation、
grouped real/imag fsumとblock scalingを保守的に合算し、全coefficient DAGをK=24C+64 real-operation budgetで覆う。
fsumについては上記の別条件を置く。この全体countはかなり保守的で、runtimeで実際に予算が小さくなるとは主張しない。

入力だけから、F=Σ_pq(|Re h_pq|+|Im h_pq|)、G_l=Σ_pq(|Re g_lpq|+|Im g_lpq|)、
S=|c|+F+Σ_l|λ_l|G_l² を作る。FとSはHartree、G_lはdimensionless。
JW ladderのcoefficient L1=1より、Hのoperator normと任意prefix/tailのcoefficient L1をSで覆える。
この尺度は観測された再構成差・残差・W/v/ρから作らない。

`budget.py`は入力floatをFractionへexact変換し、すべての予算式を有理数で計算してから
最終binary64値を上向きに変換する。算定式自体の丸めによる過小評価を避ける設計である。
今回この処理は一度も呼んでいない。

以下、g=γ_(8d+32)、t=γ_8、n=γ_(8d²+32)、j=γ_(8M+32)、k=γ_(8·64+32)。
a0=γ_K S、a=a0(1+γ_(M+32))、α=2a と置く。
a0は投影前coefficient L1誤差、aはimaginary L1の測定丸めも含む上限。
αはreal projection・identity移送後の各sideに使う保守的coefficient誤差である。
exact floating zeroの削除で消えた真の微小係数もαに含み、近似cutoffは導入しない。

## referenceとPauli matrixの予算

入力のbit-reference one-bodyを作る誤差は B_one=16kF。
各DF blockでは b_l=16kG_l とし、

```
q_l = 2 G_l b_l + b_l² + g(16G_l+b_l)²
B0  = B_one + Σ_l |λ_l| [q_l + t((16G_l+b_l)²+q_l)]
Bref = B0 + γ_64(16S+B0)
m = t(2S+2α)
Bmap = 16[α + m + j(S+2α)]
D = γ_32(64S+2Bmap+2Bref)
```

2G_lb_l+b_l²はexact A_l²とapproximate reference squareの差、g項はmatrix product、t項はλ scaling。
Brefはfull Hと任意prefix/tailのbit-reference Frobenius誤差を同時に覆う。
Bmapはcoefficient誤差、identity scalarのfsum、Pauli matrixの要素別加算を覆う。
係数のoperator L1誤差→Frobeniusではsqrt(d)=16を掛ける。
Dは比較前のmatrix加減算を覆い、norm測定には1+nを掛ける。
単位の異なるg factor誤差へHartree予算を流用しない。

## 未決9項目の今回の処理

| 項目 | 採択候補の算定式 | 入力・coverage・残る状態 |
|---|---|---|
| eps_H | (1+n)(2Bmap+2Bref+D) | full/HD/HRのFrobenius再構成差。固定H4 raw尺度＋演算数。条件付き式を具体化 |
| eps_hermiticity_H | (1+n)(2(Bmap+Bref)+D) | skew差のtriangle inequality。formula pathの入力h/g exact Hermiticityが前提。条件付き式を具体化 |
| eps_phase | eps_Hと同じ保守的envelope | identity移送のnet matrix ledger、Hartree。有限Tのphase/signal保証ではない。条件付き式を具体化 |
| eps_state_residual | η_oracle S + B_res | B_resは演算誤差だけ。**η_oracle未決**。roundoffからoracleの真の残差許容値を決められない |
| eps_imag_coeff | a | signed aggregateのimaginary L1、Hartree。observed imaginary partからthresholdを作らない。条件付き式を具体化 |
| eps_imag_energy | Eerr（下記） | exact Hermitian HのRayleigh quotientがrealであること＋reference/matvec/dot/quotient誤差。条件付き式を具体化 |
| W_error_budget | eW（下記） | canonical coefficients、λ合計、入力scale。λ下端>0が必要。条件付き式を具体化 |
| v_error_budget | ev（下記） | normalized signed-event lawのL1誤差＋Pauli expectation＋mean/variance算術。条件付き式を具体化 |
| W_floor | ((eW+ev)/2)(1+t)/(eps_rho−t) | eps_rho>t。既存rho resolutionから導く分母screenで、任意の固定floorや研究GO/STOP閾値ではない。式を具体化 |

8件は正式解決・正式採択ではなく**事前登録できる条件付き候補の技術具体化**。
残差だけは丸め計算では決まらない科学的係数を残す。η=2^-32等を勝手に埋めない。

## Rayleigh energyと残差の算術成分

保存stateのexact norm²をNとすれば、dotによるnorm²の絶対誤差≤gN。
Nはquotientの誤差伝播で相殺され、observed energyやresidualは入力不要。

```
Yerr = Bref + g(16S+Bref)
Eerr = [Yerr + g(S+Yerr) + Sg + t(1+g)(S+Yerr)]/(1−g)
B_res = (1+n)[Yerr+Eerr+γ_32(2S+Yerr+Eerr)]/(1−g)
```

exact normのsqrtに対する1/sqrt(1−g)も、より大きい1/(1−g)で覆う。
Eerrはimaginary energyの許容値にできるが、真のresidualを丸め誤差扱いしてはならない。
η_oracle Sは別の品質基準であり、合格してもexact eigenstate、ground、非縮退性、状態距離を証明しない。

## W/v予算をlawから作る順序と式

以下は将来別途実行承認が出た後、canonical lawを作った時点で、hpsi・Pauli期待値・W/v/ρを評価する**前**に一回だけ解決する。
`global_limits()`のraw尺度と、`law_limits(coefficients, lambda_sum, precision)`だけを使う。
formula selectorはfrozen configから各splitへ独立に適用し、他splitの解決済みscalarを流用しない。

L=Σ|ĉ_P|（Fractionでexact sum）、λhat=SOURCEのfsum出力、
s=γ_(M+32)L、δλ=α+s、λlo=λhat−δλ。
λlo≤0ならW/v未評価で`PRECISION_UNRESOLVED_BEFORE_METRICS`。
λhat=0はnumerical empty-tail NAとして報告し、exact target tailが空と証明しない。

```
r = L/λhat
ec = α/λlo + δλ L/(λlo λhat)
Bassembly = 16jL
y = ec + Bassembly/λhat + g(16L+Bassembly)/λhat
    + t(1+g)(16L+Bassembly)/λhat
A = (1+y)²
Qup = (1+t)(1+g)A/(1−g)
eW = 2y+y² + [2g+t(1+g)]A/(1−g) + u(1+Qup)

ep = ec + tr
defect = (γ_(M+32)+t)r
ea = [2g+t(1+g)]/(1−g)
gm = γ_(4M+32)
em = ep+(1+defect)ea+gm(1+defect)(1+ea)
am = (1+gm)(1+defect)(1+ea)
ev = 4ep+(1+defect)[4(ea+em)+(ea+em)²]
     + γ_(16M+32)(1+defect)(1+ea+am)²
```

ecはexact normalized signed coefficientsとpoint-lawのL1距離。
signが反転した微小係数も、(Pauli word, ±sign)をevent空間にすれば同じL1 boundに含む。
epは確率division、defectはprobability mass defect、eaはPauli quotient、emはweighted meanを含む。
|a|≤1、|μ|≤1、(a−μ)²≤4より、lawの変動に4epを掛け、
centerの誤差ea+emに4δ+δ²を掛ける。最後の項はcentered varianceのsubtraction/square/product/fsumを覆う。
vを不安定な二つの近いmomentの差へ置き換えない。

**eW>eps_W または ev>eps_vなら、そのsplitはW/v/ρ未評価でprecision unresolved。**
保守的なγ_K/γ_Mは実際の誤差より大幅に大きい可能性があり、全splitがunresolvedとなる可能性もある。
実測に合わせたgamma削減、scalarへの自動fallback、precision ceiling拡大は実装しない。
固定scalarを人が将来採択する場合、そのcoverageは別途明示したengineering assumptionであり、この式の証明に読み替えない。

## 分母floorとrho区間

exact v≤W、|W_hat−W|≤eW、|v_hat−v|≤ev、wlo>0なら、
point quotientの厳密算術におけるρ差は A0/wlo 以下（A0=(eW+ev)/2）。
point rho演算の丸めを t(1+A0/wlo) 以下で覆うと、
wlo>A0(1+t)/(eps_rho−t) がその合計をeps_rho未満にする十分条件となる。
この式がW_floor候補で、追加の恣意的倍率はない。

これはfeasible-domainを用いたpoint感度のscreen。
既存の保守的rectangle intervalの半幅を自動的に保証するものではなく、最終rho interval判定は引き続き必要。
v/Wのrectangle端点・W/v端点・不確かさは新v3でnextafterにより外向きに丸める。
raw pointをclampせず、rho∉[1/2,1]やrho precision unresolvedから研究GO/STOPを作らない。

## 既存17項目との対応

| 区分 | 項目 | 今回の扱い |
|---|---|---|
| exact quality候補2 | eps_basis=0、eps_hermiticity_factors=0 | unchanged、正式採択なし |
| scalar quality候補3 | eps_state_norm=2^-40、eps_imag_expectation=2^-40、eps_prob=2^-46 | unchanged、正式採択なし。forward boundがこのquality gate内に必ず収まるとは主張しない |
| 報告resolution候補3 | eps_W=eps_v=2^-24、eps_rho=2^-12 | unchanged、正式採択なし。結果から調整しない |
| 条件付きformula候補8 | 上表の残差以外8項目 | scalar又はallowlisted formula ID。任意expression・追加引数・未知IDはschema拒否 |
| 科学的品質未決1 | eps_state_residual | formula skeletonのみ。η_oracle=nullで実行不可 |

## SOURCE、runtime、保全、残る承認

[v3 SOURCE](../../../src/trottertracks/algorithm_codesign/m1s_gate0_v3)は新namespace。
[runner](../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v3.py)はdefault/status/help/schema/validate-configで科学処理を行わない。
source<adopted-contract<separate-authorizationの3段階commit、clean checkout、scope/raw-input/source/runtime/build/thread identity guardを維持。
JSONでadopted=trueにする操作はユーザーの実際の採択の代行ではない。

[既存runtime案](m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)のCPU1 worker、BLAS/OpenMP等1thread、
4 GiB address space、output/log各256 MiB、context4 MiB、M65536、C1000000、wall-time capなしを維持。
新resolverはraw8×8×12 scaleと各canonical lawを1回走査するだけ。Pauli map/reference共有、mask cache、各split独立性を保つ。
Fractionの追加コストは未benchmark、上記memory/instruction/output caps内で停止する。speedupを主張しない。
既存の実数・複素numpy計算を起動していないため、数値coverage・性能・実験完遂は未検証。

必要な科学判断はη_oracle（又は同等のresidual quality基準）、条件付きcoverageを受理するか、
従来8scalar候補と今回8formulaを含む最終17項目・runtime・契約全体の採否。
次はユーザーによる最終契約採択。その後もGate 0の科学計算には別途明示承認が必要。
新しいauthorization/markerは作成しない。mandatory STOP。
