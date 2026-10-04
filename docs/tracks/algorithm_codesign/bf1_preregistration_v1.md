# BF-1 結果前契約 v1

日付: 2026-10-05 JST。status: `PREREGISTRATION_CONTENT_FROZEN_EXECUTION_NOT_AUTHORIZED`。
execution ID: `bf1-20261005-development-v1`。

利用者はcommit `15465b0b856d80f9cfde495fd3d22434825f1cc8`の四文書をreviewし、
`PASS_FOR_BF1_PREREGISTRATION`と判定した。これはBF-1実行承認ではない。
本書はそのreview条件を閉じる結果前の仕様であり、science inputは開かない。
BF-0の三本文・review requestはreviewされた版のまま保存する。本書がBF-1実装の規範となる。

## 1. RQ、primary route、非claim

同じ五stage symmetric fourth-order family、同じI2 access、同じ32係数評価で、
ordinary PF設計 O、既知leading-tail設計 L、finite task設計 F を比較する。
**新規性に関係する主対照はF対L**。Oはtail-aware設計を入れる効果を見る診断ablationであり、
最強physical baselineとは呼ばない。

primaryはepsilon_sig=0.01、alpha=0.05のcorrected coherent signalに対するnative action proxy。

\[
\rho=\frac{\min_{w\in F}G_{\rm finite}(w)}
 {\min_{w\in O\cup L\cup\text{four fixed refs}}G_{\rm finite}(w)}\le0.95.
\]

各集合は**選ばれた一点でなく、各armが評価した全係数**。共通q/R/K policyでfinite再採点する。
q、feasibility、Pareto membershipの変化はsecondaryであり、primaryの代替routeにしない。
係数の違いだけでBF-Cにしない。F対Oだけの改善で新規性を主張しない。

BF-1はknown development inputでのI2機構検査。最良の四次PF、新allocation法、compiled/RZ/FT cost、
最終総cost、oracle-free method、general第四次・processed・SPRINTに対する優位、transfer、held-out、
independent replicationを主張しない。BF-Cも正式な主研究として再設計する現象候補の判定まで。

## 2. Inputと保存済みstate

H4 linear 1.00 Å / STO-3G / 8 system qubits / DF rank12 / T=0.8 /
**generation-prefix L_D=3**のみ。B-F用の新しいHamiltonian/stateは作らない。
BF-0 proposalの「weight-ordered prefix」は、Aのcurrent native sourceに合わせて保存順prefixへ具体化する。
既存G/W一致のevidenceをBの新結果とせず、実装policyそのものを固定する。

`L_D`はDF fragment数。決定論generatorはone-body、fragment 0、1、2の**4個**であり、
数学契約§2の和の添字は`n_D`に読み替える。tailはfragment 3–11のidentity抽出後の残差。
identity phaseはHamiltonian constantとtail extracted identityを一度ずつ加える。
`coefficient_atol=0.0`、`diagonal_sort=descending_abs`、generator順序は上記のまま。

保存済みsnapshotのfull stateを用いる。recorded state identity/phaseを維持し、OpenFermion→Qiskitの
bit reversalとroundoff範囲のnormalizationだけを行う。新しいground-state solverやrandom startは使わない。
targetは保存state上の`exp(-iTH)`を評価し、Rayleigh energy一つからのphaseへ置き換えない。

| identity | 文書・sourceに記録されている値 |
|---|---|
| literal read source | `/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928` |
| literal input path | `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| raw SHA-256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| Hamiltonian | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| state | `31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd` |
| state vector | `c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63` |

identityの参照元はM2 result commit `b6e65c6123475add5e620ec1064f361378bead95`に含まれる
`pr2_matched_accuracy_m1_contract.py`と`pr2_new_series_amendment_v4.md`。NPZ実物を確認した値ではない。
raw bytesは選定した正本文書に値がないため未記載とし、immutable raw SHAで同定する。
この段階のresolve/stat/hash/loadは全て0。M2 H4 1.30 Åは使用しない。

## 3. Numerical domainのfreeze

\(w=(a,b,1-2(a+b),b,a)\)、sum w=1、sum w³=0、max|w|≤2、実数。
\(s=a+b,d=a-b\)、\(g(s)=5s^2-8s+4-2/(3s)\)、\(d=\pm\sqrt{g(s)}\)。

| 枝 | sの閉区間 | 接続 |
|---|---|---|
| negative、d<0 | [-1/2, s_-] | 独立component 0 |
| negative、d>0 | [-1/2, s_-] | 独立component 1 |
| positive、d<0 | [s_+, 3/2] | positive d>0とs_+で接続 |
| positive、d>0 | [s_+, 3/2] | 同じcomponent 2 |

`s_+=2/(4-cuberoot(4))`。`s_-`は`6s³−24s²−18s−1=0`の[-1/2,0)内の根。
端点のdはnegative下端±sqrt(127/12)、negative上端±(s_-+4)、positive上端±sqrt(101)/6。
これで**4 chart / 3 connected components**を尽くす。

Euclidean `||dw||`によるarc lengthはnegative各`1.739213918708686`、positive全体`4.547503609322151`。
positive chartは`r=sqrt(s−s_+)`でcuspを除いて積分する。
SciPy quadのepsabs/epsrelは1e-11、reported errorが1e-9を超えればpreparation STOP。
arc inversionはbrentq、xtol=5e-15。branch、端点、arc、16初期点の全identityを
`domain_manifest.json`へ結果前固定する。科学signalでbranchを発見・追加しない。

共通初期点はanalytic Suzuki5、zero-embedded Yoshida3と14層化点。
componentへ最低1点、その後11点を長さ比例・largest remainderで割り当て、tieはcomponent順。
層化点数は**[4,3,7]**。各stratum中央を使い、Suzukiと重複するpositive中央だけ同じstratumの1/4点へ移す。
Yoshidaは`a=1/(2-cuberoot(2)), b=0`、Suzukiは`a=b=s_+/2`。

各armはさらに16回refineする。隣接intervalのmin endpoint objective、component、左s、左dの順。
全endpoint scoreがinfeasibleなら最長interval、同tie順。未採点のdomain端点はobjective=+infinityの
virtual boundaryとして端部intervalにも含める。signalを評価せず、midpoint評価を通常の32点budgetに数える。
重複、branch/order screen違反は無断replacementせずSTOP。global optimumは証明しない。

係数はexact定義と80-digit loweringを併記する。generic pointのsはdecimal rational、dは代数的平方根。
named pointはcubic algebraic basis。時間はrational affine ledgerで加算し、exact zeroのみ消す。
binary64 screenはeta1≤1e-12、eta3≤1e-12。近ゼロを切らない。

## 4. F adapter、allocation、K

exact chronological stage listを展開→同一generatorの隣接exact exponentialを融合→exactゼロを除去→
残ったtail occurrenceにfinite meanを入れる。controlled relative phaseを保持する。
finite化後のcircuit simplificationやS constructionへ置き換えない。
Kは最大偶数event次数なのでK2=P3、K4=P5。

一stepのtail ledgerが全q-step listのtail ledgerをq回繰り返したものと一致することを必須gateにする。
DF境界の融合はfull listで数え、tailがstepを越えて融合した場合は契約reviewへ戻す。
4-generator backbone、初期点、四fixed refsはsyntheticで検査する。将来のrefinementにも同じguardを適用する。

allocationはBF-0§6どおり`1+floor((R_bud−n)|gamma_j|/Gamma)`、残りをfractional part降順、
tieはchronological occurrence index。個別r_j、個別K_jを探索しない。
`q={1,2,4,8}`、`R_bud={5,10,20,40,80}`。
K4はK2 tail bound>epsilon_axis/4、かつ**guarded ideal両axis bias≤epsilon_axis/2**のcellだけ。
primaryとbridgeのeligibilityは各epsilonで同じ規則。epsilon=0.05は同じ係数の再採点のみ。

## 5. O/L/Fと四固定参照

Oはideal PF bias、B=1、融合後のnative DF/tail exponential数による通常PF診断objective。
Lはideal axis bias+保守的Taylor tail bound、leading log Bとleading workを使うnovelty対照。
Fはcorrected finite mean、exact finite B_K、期待event actionsを使う。
O/Lのsearch順位へF scoreを渡さない。設計時に同じI2情報へaccessできてもobjectiveを混ぜない。
詳細な式はBF-0 proposal§4と数学契約§5–7を維持する。

fixed referenceはnative S2、Yoshida3、Suzuki5、Morales arXiv:2210.15817v3 Table I左列・21-stage unprocessed
eighth-order。MoralesはEq.16の`w10,...,w1,w0,w1,...,w10`、`w0=1−2sum(w1...w10)`を転記する。
published decimal係数を無断で再最適化しない。noncommutative A/B word seriesの次数8までの残差を
80-digitで検査する。これは転記検査であり、finite algorithmの次数・DF controlled wrapper検証ではない。
21-tail参照でR<21が不適格でも同じbudgetのまま記録する。
legacy new_4th_m2、general fourth/processed/SPRINT、新familyの追加は禁止。
この限定で最強published PF一般に勝つとは主張しない。

## 6. 数値biasとshot判定

`u_axis`は結果に合わせた定数でなく、sourceに固定したforward guardで計算する。
Hermitian spectral decompositionのreconstruction residual、Q†Q−I、polar-unitary距離、
generator perturbationのLipschitz項、binary64 time lowering、matrix-vector roundoff、
target/state inner product、DF matrix assemblyの誤差を合算する。
unitary phaseは1/8以下へscaleした18次Taylorとrepeated squaringを用い、remainderとroundoffを別計上する。
finite polynomialとr乗は明示的乗算でroundoffを追跡する。

これはIEEE binary64と標準BLAS forward-error仮定に条件付きの数値guardであり、
物理HamiltonianのDF truncation bound、interval arithmeticの認証、C_useの厳密上界ではない。
DF assemblyには8 orbitalsのone-body hopping≤64項、12 fragmentの二回作用・加算に対する
65,536 operationsの保守的gamma allowance、source reconstruction discrepancy、DF eigensystem residualを加える。
係数相殺でscaleを過小評価しないよう、入力one-body係数の絶対和と`sum |lambda_l| (sum |G_l,ij|)^2`を使う。
このboundの導出と実装の妥当性は**実行前source reviewの対象**であり、synthetic一致だけから証明済みとしない。

Hoeffdingの両axis配分はepsilon/√2、alpha/2。`s=epsilon_axis−bias−u_axis>0`、
`N=ceil(2B² log(2/alpha_axis)/s²)`。`u_axis>0.01 epsilon_axis`は不適格として保存し、数値guardを緩めない。
optimistic lower scoreもbias−uから作り、primary ratio intervalを報告する。
BF-Cにはpoint ratio≤.95に加えupper ratio≤.95、interval width≤.01を要求する。
浮動小数点ceilとaction workにもsource固定gamma allowanceを置く。
閾値を跨ぐ場合はINCONCLUSIVE。無限大・overflowは巨大な有限scoreに変換しない。

## 7. Outcomeの結果前定義

| outcome | operational rule | 結果後の方針再評価 |
|---|---|---|
| BF-A | 完全な比較でprimary ratio≥.99。5%差がなく、finite resource decision利得は1%未満 | B-F main line STOPを第一候補。係数差などsecondaryは保存 |
| BF-B | BF-Cに届かず、BF-Aの1%未満域でもない | mechanism note等へ縮小する価値をreview |
| BF-C | primary ratioとupper≤.95、uncertainty≤.01、F bestがLの全評価集合にない、±2% D/random重みの四cornerで改善方向維持 | 限定familyの現象候補。主研究へ再設計する価値をreview |
| INCONCLUSIVE | F/L/対照の適格点がない、threshold interval跨ぎ、数値guard未解決、最良F/対照/Lが係数cap（1e-12以内）・q=8・R=80上限、resource cap、失敗、中断 | 結論を出さず停止。retry/grid追加を認可しない |

5% route以外のBF-C判定を追加しない。BF-Aの1%はこのpilotのnegative分類のための事前閾値であり、
method同一性の証明ではない。BF-Bでも5%は通ったが機構/guardが通らない場合があり得る。
F bestのL集合外という条件は、未知methodの証明ではなく、共有初期点の採点一致をGOとしないguard。
finite bias、finite B、integer allocation、Gamma、ideal bias、native workとoptimizer routeの説明は全caseで必要。
候補のmetadata、q、allocation、Paretoへの変化をsecondaryとして保存する。
secondary Paretoは適格な評価済みcellのguarded Re bias、guarded Im bias、log B、deterministic actions、
expected random actionsの五軸で定義し、exact componentwise dominanceを使う。compiled metric frontierとは呼ばない。

**全outcomeでmandatory STOP**。automatic_next_stage=null、BF2_authorized=false、science_retry_authorized=false。
結果がA/B/Cのどれでも、RQ、新規性、strong baseline、oracle-free designの必要性、actual compile、
independent validation、論文の最小着地点を全面再評価する。旧P-D/R3/FRのSTOPは解除しない。

## 8. 上限、source、authorization

係数評価32/arm、合計96。fixed refs4。distinct ideal cells≤400、finite K2/K4合計≤4000。
同一B source/input/state/係数/q/R/KだけB内memoizeし、A cache/runtimeを使わない。
1 worker、BLAS/Numba thread=1（proposalのmax2以内）。CPU≤8h、wall≤4h、RSS/AS≤4GiB、aggregate≤8GiB。
B専用output≤128MiB、GPU query/use=0、trajectory/circuit/compile=0、新system/geometry/state=0。

準備artifactは`artifacts/track_b_bf1_preparation/2026-10-05/v1/`に新規作成し、上書きしない。
domain、synthetic report、transitive shared text-source inventory、B source、環境、plan fingerprintと
**science_execution_authorized=falseのauthorization draft**を収録する。
source共有はfile参照で行い、track別にtrotterlibをコピーしない。B codeは`src/trottertracks/algorithm_codesign/`。

science runnerは別authorizationのreview verdict、明示的利用者実行指示、plan/domain/test identities、
source commitに収録された全sourceの一致、環境一致を**NPZ pathname操作より前に**確認する。
さらにB専用one-shot registryをgit common directoryへexclusive作成してから初めてinputを開く。
失敗後もregistryを消して再実行しない。authorization flag一つの変更では実行を許さない。

現段階はuncommitted source-content sealである。commit-bound sourceと別実行authorizationは未完了。
利用者のreviewはpreregistration準備を許しただけなので、science run、新input開封、commit/pushへ自動進行しない。
