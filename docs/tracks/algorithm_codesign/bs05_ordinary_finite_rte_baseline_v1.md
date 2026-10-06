# BS-0.5：ordinary finite-RTE baselineの形式仕様

2026-10-06 JST。docs-only。**数式・会計仕様案を閉じる文書で、実gate manifest/合成結果は未取得。**
[監査入口](bs05_method_target_design_audit_v1.md)と[pilot amendment](block_synthesis_pilot_amendment_v2.md)を併読する。
数値precision/bias/capの採用、実装、semantic tests、科学実行はいずれも未認可。

## sourceと表現規約

固定source S=`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`の
[rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/src/trotterlib/rte.py)をtextだけ参照。
`_paired_order_weight`、`finite_rte_distribution`、`_make_event`、`event_unitary`、`finite_taylor_operator`の式と作用順を照合した。
import、event列挙、sample、dense matrix、DF circuit生成は実行していない。
Kはpaired even cutoffであり、K=2は**degree3**のP3を意味する。

Pauli stringのq0,q1順は左tensor因子から。|a,q0,q1>をjoint basisの表示規約とし、control |1>で作用する。
R_P(θ)=exp(−iθP/2)、V_P(φ)=exp(−iφP)=R_P(2φ)。sourceのevent角φとlogical rotation θを混同しない。
係数c_lの符号はQ_l=sign(c_l)P_lへ吸収、p_l=|c_l|/λ、λ=Σl |c_l|、Rhat=Σl p_l Q_l。
Q_l²=I。toy案ではp=(1,u)/(1+u)、λ=1、r=1。

## K=2,r=1：event分布を完全に指定

signed dimensionless x=τ（boundaryではτ/2）、k∈{0,2}とする。

\[
w_0(x)=\sqrt{1+x^2},\quad
w_2(x)=\frac{x^2}{2}\sqrt{1+x^2/9},\quad B_2(x)=w_0(x)+w_2(x).
\]

まずkをw_k/B2で選び、rotation index l0とproduct indices l1,…,lkをそれぞれ独立にp_lで選ぶ。
順序付きtuple e=(k,l0,l1,…,lk)について

\[
p_e=\frac{w_k}{B_2}\prod_{j=0}^{k}p_{l_j},\qquad
U_e=(-1)^{k/2}\exp[-i\phi_k Q_{l_0}]Q_{l_k}\cdots Q_{l_1},\qquad
\phi_k=\arctan\frac{x}{k+1}.
\]

右から作用する。**product l1,…,lk → rotation l0**のchronological順で、phaseはk=0で+1、k=2で−1。
負時間でもw_k/B2は同じ、φ_kの符号を保持する。phaseをsamplingの符号へ移す別algorithmをordinaryに混ぜない。
x=0ではw2=0、k=0のみ。zero-probability eventをsampling supportから除く。
二componentなら形式的tuple数は2+2³=10。これは組合せ式で、今回eventsを生成・数値採点していない。

\[
B_2\,\mathbb E_e[U_e]
=I-ix\hat R-\frac{x^2}{2}\hat R^2+\frac{i x^3}{6}\hat R^3
=P_3(-ix\hat R)=M.
\]

paired恒等式 `w_k exp(−i φ_k Q)=|x|^k/k!·(I−ixQ/(k+1))`と独立samplingから得る代数的定義。
raw meanはM/B2。Hadamard Re/Im outcome Y∈{−1,+1}のcorrected estimatorはX=B2 Y。
weight second momentはV2=B2²、absolute range Mmax=B2。exact signalでvarianceを小さくしない。

boundaryは独立e1/e2、p=p_e1 p_e2、
`U_path=U_e2 W U_e1`、`W=R_XZ(π/2)`、`b=B2(τ/2)²`。
corrected meanはP3(−iτR2/2) W P3(−iτR1/2)、X=bY、V2=b²、Mmax=b。
形式的10²のtuple supportを実trajectory列挙と呼ばない。二つのfinite化を一つのTaylor多項式へ融合しない。

## phase込みcontrolled列と共通lowering

Ctrl(U_e)のchronological列は
`Ctrl(Q_l1),…,Ctrl(Q_lk), Ctrl(R_Ql0(2φk)), phase-on-control((-1)^(k/2))`。
phase-on-control(−1)はZ_a。boundaryはCtrl(U_e1)→Ctrl(W)→Ctrl(U_e2)。
HadamardのRe軸はH_a→列→H_a→Z測定、Im軸はH_a→列→S_a†→H_a→Z測定。
unknown state準備は共通明示context、|+>準備と軸readoutもClifford欄へ含める。

Q=±Pは符号をrotation角とPauli scalarへ戻す。
Ctrl(ζP)=phase-on-control(ζ) Ctrl(P)。ζ∈{1,−1,i,−i}ではI/Z/S/S†をcontrolへ作用する。
ordered Pauli productはζPへexact集約してよい。−1と±iを捨てるprojective集約は禁止。
同じphase込みUに一致したeventのpは合算可能。Uと−Uのprobabilityを消し合わせると別LCU分布になるためordinaryには使わない。
このexact Clifford簡約と共通cacheをordinaryにも与え、未集約product列を弱い対照にしない。

\[
\mathrm{Ctrl}(R_P(\theta))
=R_{I_aP}(\theta/2)R_{Z_aP}(-\theta/2).
\]

二つは可換だがsource manifestには固定chronological順を保存する。
θ=2φならnative角は±φ。system-only合成を後からcontrolせず、joint Pauli rotationを合成する。
P=Iのscalar exp(−iφ)はdiag(1,exp(−iφ)) on aであり、phaseを消さない専用列とする。
literal primitive identityにはsigned角・joint word・phase・precision・synthesizer version/seedを含める。

joint Pauli Qのrotationは、nonidentity supportにbasis changeを行い、最後のactive qubitへ昇順CX ladder、RZ(θ)、inverse ladder、inverse basisとする案。
XはH、Yはchronological S†→H、ZはI。ladder rootは最大番号、inactive qubitは触らない。
RZ=exp(−iθZ/2)のscalar phaseもoperator residualへ戻す。
θ=π/4ならRZ=e^(−iπ/8)T等のscalarを保存する。pairのscalar cancellationはexactに確認できた場合だけ記録する。
共通canonical reductionは隣接inverse Clifford、同word隣接rotationの角加算、exact phase積までの案。
arm/roleに依存せず適用し、raw/lowered/reducedの三identityと全資源を保存する。
実装でこの列が正しいことは未テスト。将来backend endianへの変換もsemantic test対象。

## precisionとgate費用：未取得値を0としない

task-tuned普通合成はη∈{10^-3,10^-4,10^-6}の**三precision案**を全native synthesis keyへ与える。
各ηについて全branchのoperator error guardを取得し、補正後biasへ戻す。適格ηのresource vectorを全保存し、Pareto envelopeをbaselineとする。
coarsest ηやT最小ηを無条件に一つ選ばない。不適格precisionを無理に細かくして追加探索しない。
task ε=.05 / familywise α=.05は旧案の未承認候補。数値予算・η採用・tool/wheel/source hash・seed/key capは**未freeze**。
SP-0.5/SP-1のsaved sequenceを新角度の合成結果とみなさない。

各branch countはc_T=c_T+ + c_T−、c_CX、one-qubit Clifford c_1Cを別欄で保存。
raw additive countと共通簡約後countを分け、generic whole-unitary最適compile済みとは呼ばない。

| primitive | 形式的ledger（標準列のraw count） |
|---|---|
| control scalar ζ=±1,±i | T=0、CX=0、非identity phaseならone-qubit Clifford1 |
| Ctrl(P) | 各XはCX1、各ZはH-CX-HでCX1＋1C2、各YはS†-CX-SでCX1＋1C2。identity supportは0 |
| joint R_Q(θ)、support h | ladder CX=2(h−1)、basis 1C=2 n_X+4 n_Y、native RZのT/T†/Cliffordを加える |
| general native RZ(θ) | T/T†/Clifford/error/sequence hashは合成後の値。**現段階null** |
| Hadamard軸context | H準備＋H readout（Re）、さらにS†（Im）。state preparation/測定costは共通context別欄 |
| workspace | system2、Hadamard ancilla1、追加workspace0案、total live qubits3。hidden synthesis ancillaは禁止 |

multi-Pauli productのζが元Q符号とevent phaseを合成する。符号/phasegateの二重計上をしない。
Ctrl(W)も同loweringで二つのnative joint ±π/4。exact templateの費用案であり、sequence検証済みatomではない。

実装枝eのoperator誤差をδ_e≤Σν δ_(e,ν)で抑える（branch内因子はunitary）。ordinaryの補正後biasは
`b_impl≤b Σe p_e δ_e`。共通worst-case上界も保存する。
boundary全列にも同じbranch内unitary telescopeが使える。
一方finite M同士の接続normは1とは限らず、block-LCU residual伝播には別のnorm付き積boundを使う。
operator residualから各軸biasへのboundは全ρの|Tr(ρΔM)|≤||ΔM||。coherent relative phaseを無視したgate errorは使わない。

## sufficient shotsと全資源

各軸a∈{Re,Im}の統計余裕を
`s_a=ε_a−b_sim,a−b_block,a−b_impl,a−u_a>0`とする。ordinaryはb_block=0。
δsim=||M−Uideal||、numerical margin、全arm同じε/α配分を結果前固定する必要がある。
phase/channel biasは符号付き推定weightを含めて戻す。oracle exact signal/varianceはdiagnostic専用。

independent bounded samples、E[X²]≤V2、|X|≤Mmaxを使う共通Bernstein **十分数**案は

\[
n_a=\left\lceil\frac{2V_2+(4/3)M_{max}s_a}{s_a^2}\log\frac{2}{\alpha_a}\right\rceil.
\]

centered rangeを2Mmaxで抑えた保守的規則。最小必要shotsや下界と呼ばない。
ε_Re²+ε_Im²≤ε²、全target/arm/η/axisのΣα_a≤α_familyを固定する。
shot cap超過は不適格として保存し、cap値へ切って成功扱いしない。cap自体は未固定。
fractional shot count、weight1 toy外挿、異なるconfidenceでのT比較は使わない。

\[
N_{shots}=\sum_a n_a,\qquad
G_Q=C_{Q,once}+\sum_a n_a(C_{Q,context,a}+\sum_e p_e c_{Q,e}),\quad Q\in\{T,CX,1C\}.
\]

shotごとのstate準備はC_context、一度だけのsetupはC_once。知らないcontextを0と断定せず共通記号と感度を表示する。
C_classicalはinput取得、sparse M、辞書、全η合成、solver、独立residual、sampling/postprocessを含むtime/memory/callsのledger。
実branch event生成費用もgeneric LCUと公平に数える。warm cache値だけを片armの新取得costとしない。

primaryは(G_T,Nshots,G_CX,G_1C,workspace,C_classical)の同時比較。
native logical resource・sufficient-confidence会計の案であり、hardware wall timeやactual compiled quantum advantageを示さない。
ここまでの式・列・ledgerは定義可能だが、actual synthesis counts/guard/tool identityとaccuracy/capsの採用が未了。
したがって**ordinary baselineは形式仕様済み／実行contract未完成**と報告する。
