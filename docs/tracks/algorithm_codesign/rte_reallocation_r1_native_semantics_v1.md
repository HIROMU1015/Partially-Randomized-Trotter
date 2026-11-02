# R1 native builder, phase and numerical semantics

2026-10-06 JST. [v2 contract](rte_reallocation_r1_preregistration_v2.md)の技術仕様。
synthetic two-qubit circuit descriptionのみ。controlled-DF wrapper検証ではない。

## Algebraic orderとodd complement

word indices Q1,...,Qkをcircuit timeで順に作用させ、Q0 rotationを最後に作用させる。
matrix orderは exp(-i sigma phi Q0) Qk ... Q1。
phase (-i sigma)^kは別のexact scalar i^qとして保持する。
odd Aでは theta=atan(rho)とし、

\[
e^{-i\sigma(\pi/2-\theta)Q_0}
=(-i\sigma)Q_0e^{+i\sigma\theta Q_0}.
\]

追加Q0をwordの最後に作用させ、rotation signを反転し、phaseを(-i sigma)^(k+1)へ変更する。
Q0とそのrotationだけが可換であることを使い、異なるbasisのwordを自由に並べ替えない。
Pauli wordはi^q Pへ縮約するがi^qを落とさない。
controlled eventのphase i^qはancilla S / Z / Sdagで実装する。
ordinary circuitのGLOBALは数学的phaseとして保存。ordinaryのsystem phaseをcontrol前に捨てない。

## Fixed native gate construction

system wires0,1、control ancilla wire2。
Pauli product rotationはsingle-qubit basis change→parity CX→RZ→inverse parity/basisで構成する。
controlled rotationはtargetにRZ(theta/2)、CX(control,target)、RZ(-theta/2)、CXを作用させる。
controlled X/Y/Zもexact CliffordとCXへlowerし、CZを無料gateとして数えない。
RZ(theta)=exp(-i theta Z/2)の規約。

V=exp(-i pi X0X1/16)はH0 H1、CX01、RZ1(pi/8)、CX01、H0 H1で実装する。
Q=V†Z1Vではcircuit timeでV→Z1→V†。
controlled Qのconjugatorsはuncontrolled VとV†であり、そのcost/errorを含める。
known basis circuitから構成するので、未知black-box Qを無料でcontrolled化するclaimはない。

shared exact adjacent-inverse cancellationは全armで同一。
signed pi/8 RZのexact inverseも同じ規則に含める。
arbitrary RZ-angle fusionやcompiled gate-string間の新しいoptimizerは含めない。
TとTdag各1、CX各1、物理single-qubit Clifford/T各1。
global Wはsequence identityに残すが、joint lowering後のfull-circuit global scalarとしてphysical gate cost0。
workspaceは二systemの他にcontrolled時ancilla1、ordinary時0。odd stage用の追加ancillaなし。

## Numerical / synthesis guard

mpmath intervalとpointは固定100 dps。angleはpi rationalまたはatan(rational)のsymbolic key。
native target角をintervalで評価し、pygridsynthへ渡すpoint角との差も、同じideal targetに対するguardに含める。
sequence written-product orderは固定toolの規約。H/T/t/S/X/W全てをinterval行列として積算する。
phaseをminimizeせず、target RZとのFrobenius差のoutward上端を保存する。
これがstrict operator error上界になり、各precision epsilon以下を要求する。
旧SP-0.5のprojective phase witnessを用いない。全signed anglesは独立keyで一度だけ合成する。

各eventのstrict joint operator errorはnative primitive boundsの和。
approximate controlled loweringはancilla-0を厳密identityへ戻すとは限らないので、coherent observableでは
unitary-channel差≤2 delta_eventを使う。
coefficient midpointの変位も加え、bias upperを
Σ|alpha_mid-alpha_exact| + 2Σalpha_mid delta_eventとする。
operator mean boundとcoherent-measurement biasを別fieldへ保存する。

Bernstein会計はbounded corrected outcome±B_mid、v≤B_mid²、centered range≤2B_mid。
remaining axis accuracy=e_axis-biasが正なら

\[
n\ge \frac{[2B_{\rm mid}^2+(4/3)B_{\rm mid}\epsilon_{\rm stat}]
\log(2/\alpha_{\rm axis})}{\epsilon_{\rm stat}^2}
\]

のoutward上端をceilする。測定mean/varianceのoracleを使わない。
confidence/resourceは理想logical gate modelのforecastであり、実shotやhardware noiseのvalidationではない。

## Focused evidence scope

登録x={1/8,1/4}のcost/synthesisは開かず、off-domain x=1/3のsmall-matrix semantic testsを用いた。
全branch/time sign/contextのliteral operatorとnative loweringを照合し、有限meanも別のP3 polynomialと比較。
位相を捨てたcontrolled mutantとbasis inverseを欠いたmutantを検出する。
strict guardがscalar W誤差を隠さないこと、group rounding後のorder→IID law、coherent biasのfactor2、
zero-cost baseline、bias/shot cap、source/direct-child/one-shot拒否を確認する。
実synthesizer呼び出しはmockを含めて0。local testsでありimmutable CI・外部再現ではない。

次は[GPT source review](rte_reallocation_r1_source_review_request_20261006.md)。新科学実行なし、mandatory STOP。
