# Track B：DF少数軌道・希薄三次補正付きPRの構成設計と研究判断

- 作成日：2026-10-11 JST
- 作業区分：ユーザー承認に基づくGPTの数理構成・科学的成立性・新規性候補の検討。
- 承認：「ではそのGPTの作業をできるだけ丁寧に行って」
- 基点：`HIROMU1015/Partially-Randomized-Trotter`、`22a0d032717b4139cdfa806850643e93ba25624f`。
- 旧系列：G10技術ノートv1.0完成・終了判断を保持。旧source、結果、contract、authorization、STOP、markerは変更しない。
- 本書の結論：**具体的な数学的候補は構成できた。独立した数式・入力アクセス・回路分解監査へ進める価値がある。ただし、新規アルゴリズムとしての優先性、native資源優位、分子上の実用GOは未認定。**
- 暫定呼称：「DF少数軌道・希薄三次補正付きPR」。既存の正式なalgorithm名、または新規性確立済みの名称ではない。

## 0. 今回何を実際に行ったか

実際のpartial-S2と有限RTEのsource契約を読み、決定論ブロック内部の分割誤差を含む三次生成子を導いた。その補正を、DF由来の最大6軌道の局所演算子と、低確率の正のunitary mixtureへ落とし込んだ。平均演算子、有限step誤差、長時間誤差、既存RTEの測定重み、期待補正回数、有限精度の課金を記述した。2–4ブロックの有理数形式語と、小さな合成行列による自己検算を実行した。

これは新しい分子データ、native合成表、量子shot、provider実行の取得ではない。以下を区別する。

1. **既存証拠**：[R1–R4]のsource・API・入力構造。
2. **今回の導出**：以下の命題・確率・誤差式。一般的な証明を記述し、有限例でも自己検算した。
3. **今回の数値**：合成入力の浮動小数点確認。独立Codex認証でも性能実験でもない。
4. **既知の理論**：symmetric corrector、NCC、qDRIFT・確率的疎化、Gaussian基底変換など。これらを新規性としない。
5. **未確定**：結合構成の独立新規性、中心回転の安価なnative分解、比較手法に対する総資源優位。

## 1. 研究課題と今回の選択

元の課題は、PR内部の実装を変え、同じ物理的目標と成功条件に必要な資源を減らすことである。G10の「同じ有限Taylor多項式を別のlawで実装する」比較は終了済みとし、今回はその結果を再採点しない。

今回の中間taskを、任意の入力状態ρについて

\[
z(t)=\operatorname{Tr}[\rho e^{-itH}]
\]

を所定の複素誤差・失敗確率で推定することとする。未知のground state、exact-ground weight、oracleの真の誤差を設計入力に用いない。operator normの誤差保証から任意ρの信号biasを上から押さえる。これはそれだけでground-state energy、RPE全round、QPE全体の保証ではない。

採用した設計は、実際の決定論ブロック順序を保った二次PRに、**三次誤差の補正回転を前後から稀に挿入する**もの。random tailには少なくともdegree 5の有限Taylor平均を使用し、補正が消した三次誤差より低い次数のtail誤差を残さない。

これは、初めに例示した「PF誤差とTaylor誤差を一つの補正LCUで同時相殺する」案そのものではない。実際の次数を調べると、P3 tailの四次誤差は三次補正では消えない。今回はtailをP5にする費用を正面から課金する構成を選んだ。この選択を元PRの既定条件へ黙って追加しない。

## 2. sourceから確認した実装条件

[R1]のpartial-S2は、採用済み順序の決定論ブロックについて

```text
D0(δ/2), D1(δ/2), ..., D(L−1)(δ/2)
finite RTE occurrence(δ)
D(L−1)(δ/2), ..., D1(δ/2), D0(δ/2)
```

を実行する。D全体の厳密発展を二つ置く回路ではない。一体項と定数は決定論側にあり、DF fragmentの前後順序を勝手に最適化しない。controlled版では基底変換を制御せず、中央の操作と相対位相を制御する。

[R2]のRTEはHermitian involutionの分解に基づき、一般のDF fragmentそのものをinvolutionとしてsamplingしない。`finite_taylor_order`は偶数cutoffであり、cutoff 4のpaired構成はdegree 5のTaylor平均に対応する。

[R3–R4]にはGaussian基底のoperation列、I/Z/ZZ support、係数と符号、DF modelの軌道行列等がある。ただし、現`DFBasisRegistry`が本書の6軌道補正unitaryを既に生成できるとは言わない。軌道列の取得・orientation一致、新しい局所回転のloweringは次の技術監査対象である。

必要な既存の資料は取得できた。取得不能を未pushとした事例はない。新しい候補のsourceが存在しないことは、旧研究資料の欠損とは別である。

## 3. 入力クラスを明示する

\[
H=\sum_{j=0}^{L}H_j,\qquad H_L=R,
\]

とし、j<Lは実際の決定論ブロックである。既知の定数項はcontrolled相対位相として分離し、以下のcommutatorからは外せる。

各ブロックを

\[
H_j=\sum_{a\in I_j}h_a Q_a,\qquad h_a\ge0,
\]

と書く。符号はQに吸収し、DF diagonal primitiveについて

\[
Q_a=\sigma_a\prod_{v\in\mathcal U_a}(I-2n_v),\qquad
\sigma_a\in\{\pm1\},\quad |\mathcal U_a|\le2.
\]

ここでn_v=a†(v)a(v)、同じprimitive内の軌道は正規直交である。I成分は既知の位相として扱える。全fragmentがrank 2、電子数が6以下、分子全体が6軌道、という条件ではない。**一つのZ/ZZ由来primitiveの作用軌道数が1又は2**という、既存DF表現の構造である。

一般black-box involutionでは以降の少数軌道構成を利用できるとは限らない。一般Pauli blockを自動的に本クラスへ含めない。元のHやDF分解、軌道最適化問題を変更する前処理研究でもない。

以下では有限次元、δ>0を扱い、ħ=1とする。δ=0はidentity。負時間の独立契約・乱数couplingは今回の対象外である。

## 4. 決定論側内部を含む三次誤差

解析上のideal referenceとして

\[
S_2(\delta)=e^{-i\delta H_0/2}\cdots e^{-i\delta H_{L-1}/2}
 e^{-i\delta R}
 e^{-i\delta H_{L-1}/2}\cdots e^{-i\delta H_0/2}
\]

を置く。中央のexact Rは誤差を分けるための比較対象であり、実装へ無料で与えない。

\[
B_j=\sum_{\ell>j}H_\ell
\]

とすると、三次のHermitian生成子は

\[
\boxed{K_3=\sum_{j=0}^{L-1}\left(
\frac1{24}[H_j,[H_j,B_j]]+
\frac1{12}[B_j,[H_j,B_j]]\right)}. \tag{1}
\]

この符号規約では

\[
\log S_2(\delta)=-i\delta H-i\delta^3K_3+O(\delta^5). \tag{2}
\]

### 導出

A、Bをanti-Hermitianな一次生成子とすると、対称積の有限次数展開は

\[
\log(e^{A/2}e^Be^{A/2})
=A+B-\frac1{24}[A,[A,B]]-\frac1{12}[B,[A,B]]+O(5).
\]

外側ブロックからこの式を再帰的に適用する。内側の既存三次項は三次係数へ加算され、一次との追加commutatorはこの係数を変えない。A=−iδH_j、B=−iδB_jを代入すると(1)–(2)を得る。有限次数の恒等式として使用し、全BCH級数の大域収束は仮定しない。

**R=0でもD内が非可換ならK3は一般に非zeroである。** mixed D/R項だけを補正すると、実際のPR回路に残るpure-D誤差を見落とす。

## 5. 理想correctorと既知部分

\[
C_\delta=\exp(+i\delta^3K_3/2) \tag{3}
\]

を前後に置くと

\[
C_\delta S_2(\delta)C_\delta=e^{-i\delta H}+O(\delta^5). \tag{4}
\]

三次係数が消え、積全体の対称性から四次のlog項もない。(4)は既知のsymmetric-correctorの考え方であり、新規性の根拠にしない。[P1,P4]

問題は、K3全体のexponentialを無料で使えないことである。K3は一般に大きなnon-Gaussian operatorであり、「式を定義した」だけでは実装がない。次節以降で、この問題を個別の三重交換子へ分解する。

## 6. 全三重表を入力しない係数生成

(1)へ各Hjのprimitive分解を代入する。

\[
K_3=\sum_\tau c_\tau T_\tau,\qquad
T_\tau=[Q_a,[Q_b,Q_c]],\qquad c_\tau\ge0. \tag{5}
\]

二つの型を、重複を含むまま保つ。

- 型I：a,b∈Ij、c∈I>j、係数ha hb hc/24。
- 型II：a,c∈I>j、b∈Ij、係数ha hb hc/12。

符号はQに含まれる。別tupleの同一演算子への集約を前提にせず、強いbaselineにも同じ既知の相殺処理を認める。

\[
\lambda_j=\sum_{a\in I_j}h_a,\qquad
\ell_j=\sum_{\ell>j}\lambda_\ell,
\]

\[
A=\sum_\tau c_\tau=
\sum_j\left(\frac{\lambda_j^2\ell_j}{24}
+\frac{\lambda_j\ell_j^2}{12}\right),\qquad
\mathcal B=4A. \tag{6}
\]

まず(j,型)を(6)の項に比例して選び、型に応じて三つのprimitiveをブロックまたはsuffixから係数比例で選べばqτ=cτ/Aとなる。全tupleを列挙する必要はない。係数リストが与えられた後のprefix/alias前処理は入力primitive数Mに線形、tree方式のqueryはO(log M)算術操作で記述できる。入力DFの構築やbit精度の費用は別に課金する。

Hermitian unitary Qに対しTτはHermitianで、\|Tτ\|≤4。以下、

\[
\|T_\tau\|\le\kappa_\tau\le4,\quad
\Lambda_3=\sum_\tau c_\tau\kappa_\tau\le\mathcal B,
\quad\|K_3\|\le\Lambda_3. \tag{7}
\]

κτは認証済み上界でよい。普遍的なκτ=4でも構成は成立する。小さな局所行列から良い上界を得れば受理率を下げられるが、未知のglobal Λ3をsamplerへ入力する必要はない。κ=0とするのはT=0を確認した場合のみ。浮動小数点の小さいnormを、根拠なくzeroにしない。

## 7. 三重交換子は最大6軌道に厳密に閉じる

### 命題：DF primitive三つの共通作用部分空間

三つのprimitiveに含まれる一粒子軌道のspanを

\[
\mathcal S_\tau=\operatorname{span}(\mathcal U_a\cup\mathcal U_b\cup\mathcal U_c),
\qquad r_\tau=\dim\mathcal S_\tau\le6
\]

とする。最初のrτ軌道をこのspanへ写すorbital unitary Wτと、その第二量子化Gτにより

\[
T_\tau=G_\tau(T_\tau^{(r_\tau)}\otimes I)G_\tau^\dagger. \tag{8}
\]

従って、

\[
\|T_\tau\|=\|T_\tau^{(r_\tau)}\|,
\]

\[
\boxed{
V_\tau(\theta)=e^{i\theta T_\tau/\kappa_\tau}
=G_\tau\bigl(e^{i\theta T_\tau^{(r_\tau)}/\kappa_\tau}\otimes I\bigr)G_\tau^\dagger.
} \tag{9}
\]

### 証明と意味

各n_vはv∈Sτの生成消滅演算子だけで書ける。Qはこれらの偶パリティ・粒子数保存多項式であり、和・積・交換子も同じ代数に属する。Sτを最初のrτ modesへ写せば、spectator側にはidentityとして作用する。偶パリティなのでspectatorに未処理のJordan–Wigner stringを残さない。Fock空間のtensor順序は実装と一致させる。数式はactive/spectatorの論理的tensor順序で書いた。添付コードはmode 0を最下位bitとするため、行列では`I_spectator ⊗ T_active`を使う。

これは電子状態をactive spaceへ近似する操作ではない。分子全体の状態は任意でよく、spectatorとentangleしていても(8)はoperator恒等式として成立する。局所dimensionは最大64。粒子数sector分解を使う場合、最大sector dimensionはC(6,3)=20である。

局所matrixの構築・対角化はこの固定dimensionで行える。全2^n×2^n行列を必要としない。ただし次の費用は残る。

- 実際のDF basis operationから軌道列を取り出し、orientation・位相・符号を一致させる。
- n×rのisometryをGaussian/Givens操作で実装する。rを固定すればO(nr)のGivens構成を利用できるが、native精度とroutingは別である。
- 中央のr-qubit unitaryは一般に**非Gaussian**。小さい行列であることを「数個のGaussian gateで足りる」と読み替えない。一般分解の定数は大きくなり得る。
- 全n軌道の補集合を明示構築するO(n²)作業を必須にせず、必要なisometry列のみを生成する実装が必要。

Controlled Vは、GとG†を制御せず中央unitaryのみ制御する。control=0 branchで基底対が打ち消し合う。既存Hadamard ancillaを利用する経路はあるが、中央制御のnative費用と相対位相を保存する。追加workspaceなしの一般分解は有限に存在しても、安価とは限らない。

(8)–(9)の構造を今回導いたが、少数supportを共通基底へ移す原理自体の優先性は主張しない。既知のGaussian基底変換・低rank構成との違いは、**三重交換子の回転に対する具体的な閉包とnative lowering**に絞って監査する。[P7–P8]

## 8. 低確率補正の具体的なsampler

γ>0を、結果を見る前に指定する正のrateとする。γは時間の逆数の次元を持つ。全入力に対する最適値γ=1を主張しない。

\[
\eta_\delta=\min(1,\gamma\delta),\qquad
\theta_\delta=\frac{\mathcal B\delta^3}{2\eta_\delta}. \tag{10}
\]

一つのhalf-correctorの生成手順は次のとおり。

```text
if A == 0: return identity
with probability 1 − ηδ: return identity
sample tuple τ with probability cτ / A
construct active orbital span, local Tτ and a certified bound κτ ≤ 4
if κτ == 0: return identity
with probability 1 − κτ/4: return identity
return Vτ(θδ)
```

普遍的κ=4ならnorm-based coinは常に受理する。normを求めるためのexact oracleは不要。良いκが得られる実装では、最後のcoinでcommutatorの小ささを利用できる。

**不受理は「identity補正」**である。PR本体と測定は行う。shot全体を捨てたりzero estimatorに置き換えたり、受理まで再試行したりしてはいけない。

左右のhalf-corrector、各step、中央RTEは独立に生成する。サンプルした一つの回路の誤差次数を高次化したと主張しない。

## 9. 平均恒等式と補正誤差の証明

τが実行される無条件確率は

\[
p_\tau=\eta_\delta\frac{c_\tau\kappa_\tau}{\mathcal B}.
\]

平均operatorは

\[
\bar C_\delta=I+
\sum_\tau\frac{\eta_\delta c_\tau\kappa_\tau}{\mathcal B}
\left(e^{i\theta_\delta T_\tau/\kappa_\tau}-I\right). \tag{11}
\]

一次項を展開すると

\[
\bar C_\delta=I+i\delta^3K_3/2+R_C. \tag{12}
\]

Hermitian G、\|G\|≤1について

\[
\|e^{i\theta G}-I-i\theta G\|\le\theta^2/2
\]

なので

\[
\|R_C\|\le
\frac{\delta^6\mathcal B\Lambda_3}{8\eta_\delta}.
\]

理想Cδの二次以降を同様に課金すれば

\[
\boxed{
\|\bar C_\delta-C_\delta\|
\le e_C(\delta):=
\frac{\delta^6}{8}
\left(\frac{\mathcal B\Lambda_3}{\eta_\delta}+\|K_3\|^2\right).
} \tag{13}
\]

γδ≤1なら第一項はδ5 BΛ3/(8γ)、第二項はO(δ6)。従って本samplerの平均correctorは、(4)の四次精度を壊さない精度を持つ。粗いが取得可能な保証ではΛ3と\|K3\|をBで置き換えてよい。

これは確率的Hamiltonian疎化・qDRIFT型の一次matchingを、サイズδ3のcorrectorへ適用した構成である。[P2,P3] **samplingの基本原理を新発見とはしない。**

また、Cbarは正のunitary mixtureなので\|Cbar\|≤1。非自明なunitaryを全く異なるunitaryの正mixtureで厳密再現したという主張ではない。残るbiasは(13)へ明示的に課金した。operatorの一次平均とquantum channel平均の一致は要求しておらず、coherent signalのHadamard推定に使用する。

## 10. P3をそのまま使うと四次精度にならない

中央のRTE平均がP3なら

\[
P_3(-i\delta R)-e^{-i\delta R}
=-\delta^4R^4/24+O(\delta^5).
\]

三次correctorを前後から掛けても、この四次項は消えない。全体は一般にlocal O(δ4)、global O(δ3)に留まる。Rが小さいと有限δで五次項が支配して見えることがあるが、それはleading orderを変えない。

今回の具体案では中央に

\[
P_5(-i\delta R)=\sum_{k=0}^5(-i\delta R)^k/k!
\]

を使う。既存ordinary paired RTEで生成できるdegreeであり、G10の全return集約や特殊closed-P5を必須にしない。既存P5が有利なら対照にも同じ実装を与え、tail次数の変更による利益を三次補正の利益に混ぜない。

## 11. P5有限RTEと全体の明示誤差bound

R側をs≥1個のmicrostepに分ける場合、

\[
\widehat U_R(\delta)=P_5(-i\delta R/s)^s.
\]

xR=λRδ/s、g=e^{xR}xR^6/720と置くと

\[
e_R(\delta)\le s g(1+g)^{s-1}. \tag{14}
\]

P5は一般にunitaryでもcontractionでもない。そのためnorm増幅を(14)で保持する。P5†をP5の逆とみなさない。

corrected ideal S2に対しても、記号O(δ5)だけでなく有限δのboundを与える。ΛH=Σjλj、x=ΛHδ、y=Bδ3とし、

\[
\begin{aligned}
b_{\mathrm{sym}}(\delta)=&\ 2\left(e^x-\sum_{k=0}^4x^k/k!\right)\\
&+y(e^x-1-x)+e^x(e^y-1-y).
\end{aligned} \tag{15}
\]

(4)でdegree 4までの係数が一致することを使い、それ以降の係数のnormを、正係数majorant e^{x+y}とe^xのtailで押さえた式である。各exponentialの有界性と有限Taylor係数から導くため、BCH級数の大域収束を必要としない。(15)はcommuting系でも非常に緩くなり得る上界であり、resource headroomの不可能性証明ではない。

実際のweighted one-step meanをMbarδとすると

\[
\bar M_\delta=\bar C_\delta\,M_{\mathrm{PR},5}(\delta)\,\bar C_\delta,
\]

\[
\boxed{\|\bar M_\delta-e^{-i\delta H}\|
\le b_{\mathrm{loc}}=b_{\mathrm{sym}}+2e_C+e_R.} \tag{16}
\]

Cbar、C、ideal S2のnorm≤1を用いて二つのcorrector差を課金し、中央RTE差を独立に加えた。

n=t/δが整数で各stepを独立生成する場合、期待operatorはMbarδのn乗である。telescopingから

\[
\boxed{\|\bar M_\delta^n-e^{-itH}\|
\le(1+b_{\mathrm{loc}})^n-1
\le n b_{\mathrm{loc}}e^{n b_{\mathrm{loc}}}.} \tag{17}
\]

固定入力・固定γ・固定t、γδ≤1で(16)はO(δ5)、(17)はO(tδ4)。native synthesis・係数丸め・DF threshold誤差はこの理想式に含まれず、次節のように別に加える。

## 12. 有限精度をどこへ課金するか

### 12.1 浮動小数点normを認証と呼ばない

数値対角化で得た\|T\|をそのまま厳密上界にしない。intervalまたは明示的residualを使うか、常に安全なκ=4へfallbackする。線形依存に近い軌道をrank dropする場合はoperator誤差を課金する。既知のexact linear dependency以外を、零の条件へ丸め込まない。

### 12.2 同じGaussian circuitとそのinverseを使う利点

(9)のBを近似Btildeへ置き換え、同じcompiled circuitの逆列を使う。\|Btilde−B\|≤εBなら

\[
\|\tilde B V\tilde B^\dagger-BVB^\dagger\|
\le2\epsilon_B\|V-I\|
\le2\epsilon_B\min(2,|\theta|). \tag{18}
\]

BIB†=Iを先に消して差を展開することで得る。中心近似誤差εVは別に加える。独立に合成したforward/backwardで完全inverseにならない場合、この小角抑制を無条件で使わない。既知の誤差伝播であり、対照側にも同じ処理を認める。

### 12.3 確率・係数・局所unitaryの有限bit誤差

理想受理確率pτと実際p̃τ、unitary誤差ετに対して一つのhalfslotの平均誤差は、例えば

\[
\sum_\tau |\tilde p_\tau-p_\tau|\min(2,|\theta|)
+\sum_\tau\tilde p_\tau\epsilon_\tau
\]

で押さえる。これは有限bitで厳密平均を維持したという意味ではなく、biasの追加課金である。比較するRTE本体にも同じfinite-bit policyを与える。

Qの入力誤差は\|ΔT\|≤4(εQa+εQb+εQc)等のsafe boundで伝播できる。係数や角度の誤差も別途含める。最終coherent signalでは既存RTEの重みΓtotを含む実際のweighted meanで誤差を評価する。例えば一つのRTE occurrenceの回路合成誤差が平均εRなら、そのweighted meanへの寄与はΓR εR以下であり、εRだけを足してはいけない。決定論half sweepの合成誤差も、同じweighted full-stepとの差として課金する。

## 13. 何がsampling負担を増やさないのか

補正は正確率のunitary又はidentityであり、補正用のsigned weightやnormalization倍率を追加しない。**固定δ、同じRTE実装であれば、補正追加による新しい測定重み倍率は1**である。

ただしRTE自体のnormalizationは残る。ordinary paired P5で、x≥0に対し

\[
B_5(x)=\sum_{k\in\{0,2,4\}}\frac{x^k}{k!}
\sqrt{1+\left(\frac{x}{k+1}\right)^2}. \tag{19}
\]

これはTaylor k,k+1のpairを、k個のinvolution積と一つの回転へまとめて導ける。RTE microstep数sならstep倍率ΓR=B5(λRδ/s)^s、軌道全体の倍率Γtot=ΓR^n。

Hadamard testのX又はY outcomeをY∈{−1,+1}とするとZ=Γtot Yで、

\[
|Z|=\Gamma_{\mathrm{tot}},\qquad E[Z^2]=\Gamma_{\mathrm{tot}}^2. \tag{20}
\]

ρについて全体operator誤差をbと評価でき、各axisの許容残余がεstat>0なら、Bernstein型の十分shot数として

\[
N\ge\log(2/\alpha)\left(
\frac{2\Gamma_{\mathrm{tot}}^2}{\epsilon_{\mathrm{stat}}^2}
+\frac{4\Gamma_{\mathrm{tot}}}{3\epsilon_{\mathrm{stat}}}
\right) \tag{21}
\]

を使える。複素誤差と全失敗確率の契約では、axisごとのε、αを明示してunion boundを取る。例としてoperator bias bを先に引き、各axisのstatistical誤差を(ε−b)/√2以下、失敗確率α/2以下にすればよい。

**「追加倍率1」≠「shot数が常に同じ」**である。P3からP5への変更、δを大きくすること、残余biasの違いでΓtotとNは変わる。γやδを変えればsampling errorとgate costも変わる。

## 14. 期待補正回数はstep数と同じ速さでは増えない

halfslotの受理確率はηδΛ3/Bであり、2n個のslotに対して

\[
E[J]=2n\eta_\delta\Lambda_3/\mathcal B. \tag{22}
\]

γδ≤1、n=t/δなら

\[
\boxed{E[J]=2t\gamma\Lambda_3/\mathcal B\le2t\gamma.} \tag{23}
\]

Λ3は採用したκ上界に基づく。exact normを使う理想形では非zero回転の受理回数となる。粗いκでidentityも受理する場合、(23)は回転要求回数であり、実際のno-op除去後はさらに小さくなり得る。

outer coinの後だけ局所tupleを評価するので、局所行列を作る回数の期待値も2nηδ=2tγ。全slotで6軌道行列を作ってから稀なcoinを引く実装は、この利点を失う。

これは、**固定γでstepを細分化しても、追加の補正要求数がO(n)に増えない**という構造である。次を意味しない。

- 決定論PR本体のn回実行が不要になる。
- 各補正のT価格がδや合成精度に依存しない。
- 期待回数をworst-case上限とみなせる。
- 稀な長回路を条件付けで捨てても推定量が変わらない。

最悪要求数は2n。capを置くならそのtail確率とweighted biasを追加課金するか、契約上のfailureとして扱う。都合のよい受理pathだけで総費用を作らない。既知TE-PAI・確率的疎化にも稀なgateの発想はあり、(23)を「初めてgate数を細分化から切り離した」と主張しない。[P3,P6]

## 15. 資源改善の機構と、実際に勝つための条件

additiveな一次モデルでは、一trajectoryの費用は

\[
\bar C_T=h_{\mathrm{prep/read}}+n\bar c_{\mathrm{PR},5}
+E[J]\bar c_{\mathrm{corr}},
\]

二axes総費用はGT=2N CbarTである。実回路ではD/R/補正境界のreuseやroutingが非加法的なので、最終比較はsame compilerのwhole wrapperで行う。

改善の可能性は、三次biasを消して許容δを広げ、D側の高いsweep回数を減らしつつ、追加回転は稀にすることにある。単に「小さいcommutatorを見つける」から一歩進み、(9)でそのexponentialを実際の有限support回路へ結び付けた。

しかしγを小さくすると補正要求数は減る一方、(13)の主要項は1/γで増える。native中心unitaryが高価なら最適γでも負け得る。Bはblock係数の積の上界なので、大きなDF入力では保守性が強くなる。厳密な小ささ\|K3\|を、全tupleの取得なしに無料で使わない。

必要な条件は、同じ物理精度・成功条件で

\[
N_{\rm cand}\bigl(h+n_{\rm cand}\bar c_{\rm PR,5}
+E[J]\bar c_{\rm corr}\bigr)
< N_{\rm base}\bigl(h+n_{\rm base}\bar c_{\rm base}\bigr).
\]

今回この不等式を実入力・native synthesisで確認したわけではない。

固定t・入力・γ、合成log因子を一旦外すと、biasに必要なnが二次式のεb^(−1/2)から候補のεb^(−1/4)になる。ただし既知四次PFも四次精度であり、NCC/HNCCにはさらに強い精度理論がある。この次数差を、全既知手法に対する優位とはしない。

## 16. 先行研究との対応と、残る差候補

| 項目 | 既知内容・今回の位置付け |
|---|---|
| S2三次誤差、対称corrector | 既知のBCH・corrected PFの範囲。[P4] 独立新規性としない |
| PF誤差をランダムunitaryで補償 | NCC/PTSCで既知。[P1]一般HamiltonianもAppendix Eで対象にしている |
| 稀な正unitaryで小さい生成子を近似 | qDRIFT、stochastic sparsificationの既知機構。[P2,P3] |
| 高次nested-commutator補償 | HNCCがchannel-levelで強い理論を提示。[P5] 本候補が高次精度理論で優位とはしない |
| Gaussian基底の低rank化 | 電子構造の既知方法。[P7,P8] 単一primitiveのsupport縮約も新規性としない |
| 今回の差候補 | DF primitiveの三重交換子全体を6軌道へ閉じ、global Pauli展開や全tuple norm表を入力せず、coherent-mean用の希薄補正へ接続する具体的native経路 |

NCC論文Appendix E、Proposition 13は一般Hamiltonianに対するnested-commutator補償を扱う。したがって「既知NCCはlatticeだけなのでDFには適用できない」は不正確である。ただし一般式があることと、同じDF入力から同じ費用で局所corrector回路が生成できることは別である。次の比較は、この実装・情報条件まで具体化する。

HNCCはoperator一次平均とは異なるchannel推定を扱うが、controlled Hamiltonianを拡大系で使う経路を含めて比較できる。単にchannelという理由で除外しない。control、補助qubit、Pauli locality、global展開・sampling費用を戻す。

現在、結合構成全体を同一の入力・保証・native手順で提示した先行研究は、この限定調査では特定していない。しかし、これは不存在や優先性の証明ではない。特に(8)の単純なsupport論は、既知のfermionic少数mode処理と同値である可能性がある。**研究差候補は残るが、独立新規アルゴリズムとして採択した段階ではない。**

## 17. 検討したが主候補にしなかった案

### 単純なadditive誤差LCU

Eδ=e^(−iδH)−Mδを定義して、そのまま足す案は実装を与えない。有限語まで展開してLCU化すれば既知補償に近く、係数質量の増加を避ける別機構が必要。今回の主候補にはしない。

### δ3補正とP3のδ4 tailを同時に消すと言う案

二つは異なる次数・Hermitian構造を持つ。三次のHermitian生成子のunitary補正だけで四次R^4を消したことにはならない。本構成ではP5を明示的に選び、その費用を課金する。

### normが小さいcommutatorを、直ちに安いPauli LCUとみなす案

operator normとPauli coefficient L1は異なる。今回も(7)だけでは回路がない。その問題に対する具体的な答えが(8)–(9)だが、局所中心unitaryが高価という限界は残る。

### D全体をexactとしてinteraction pictureへ移る案

既存PRのDはordered DF blocksであり、D全体のexact time evolutionを仮定するのは別のアクセス条件になる。本候補では使わない。THRIFT等との差を名前だけで作らない。

### 全tuple表を作って有利なκや角度を選ぶ案

有限例の探索は可能だが、全表取得を避ける方法を主張しながら、その表に依存した最適化結果だけ示すのは循環する。今回の基本samplerは(6)とlocal queryだけで成立する。

## 18. 今回の自己検算

コード：`validate_design.py`。Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0。全数値は合成入力。provider、新規分子、native synthesis、量子shot、repo runnerは使っていない。

### 18.1 Fractionによる非可換形式語

2、3、4ブロックについて、degree 6までの積を有理数で直接展開した。

- S2の三次係数が(1)に一致。
- 正しい符号のcorrector後、degree 4までexact time evolutionと完全一致。
- 誤符号では三次誤差が二倍となり、negative controlを検出。
- P3を中央に残すとdegree 4に−R^4/24が残ることをexactに確認。
- 形式語の有限checkだけで任意ブロック数の定理を証明したとはしない。一般証明は第4–5節の再帰と有限次数展開。

### 18.2 6軌道support恒等式

8 fermionic modes、各2軌道を持つ三primitive、共通span 6軌道。seed=20261011。全Fock 256次元とlocal 64次元を別々に生成し、第二量子化したorbital unitaryで照合した。

| 確認 | 残差 |
|---|---:|
| 全系commutatorとlocal lift | 5.0752172842×10^-15 |
| operator normの差 | 表示精度で0 |
| 全系exponentialとlocal exponential lift | 1.7557606042×10^-15 |
| 第二量子化basisのunitarity | 1.6441785972×10^-15 |

これはcontrolled native circuitをcompileした結果ではない。diag(I,U)へ拡張した場合の行列差は同じになるが、実際のcontrolled gate loweringは未実装である。

### 18.3 誤差次数と期待補正回数

3 modes、D0,D1,Rの三ブロック、rank 1/2/1、係数0.7/0.5/0.2、seed=740112。t=1、γ=1（この合成入力の単位）で計算。

| δ | 補正なしP5 local誤差 | 補正付きP5 local誤差 | t=1補正付き誤差 |
|---:|---:|---:|---:|
| 0.2 | 5.48447214e-4 | 9.34533551e-6 | 3.96920249e-5 |
| 0.1 | 6.86941602e-5 | 2.93104183e-7 | 2.47797342e-6 |
| 0.05 | 8.59109557e-6 | 9.16916597e-9 | 1.54862754e-7 |
| 0.025 | 1.07402216e-6 | 2.86633704e-10 | 9.67979682e-9 |
| 0.0125 | 1.34256995e-7 | 8.95848076e-12 | 6.05038380e-10 |

最後の二点のslopeは補正なしlocal≈2.99995、補正付きlocal≈4.99981、global≈3.99988。期待補正回数は全δで1.2539617939。これは確率分布を使った期待値の計算であり、Monte Carlo頻度の測定ではない。

明示bound(13)、(16)、(17)はこの例の誤差を上から覆った。γδ≥1のsaturation branchも別途確認した。R=0でpure-D三次誤差が残るcontrol、全Qが同じ軌道基底でcommuteしてT=0となるcontrolも確認した。

### 18.4 失敗・修正履歴を隠さない

最初の自己検算コードでは、誤符号controlの平均を正符号のideal Cと比較したまま、正符号用の近似boundをassertしていた。control側idealの符号も合わせるよう修正した。候補本体の符号を結果に合わせて変更したのではない。

次に、P3 controlが指定された有限δで既に四次のasymptotic slopeを示すと仮定したassertが失敗した。弱いRでは−δ4R4/24が小さく、高次項がこの範囲を支配する。入力やδ範囲を都合よく変更せず、P3の観測slope約4.99→4.963を保存し、四次残差は有理数形式語で直接確認する基準へ改めた。

最終整理では、local liftをcontrolled circuitの実行と誤読させる出力名を修正し、明示bound・saturation branchを追加した。初期scriptを履歴として保持した。final scriptのPASSを「最初から全検査がPASS」と報告しない。

## 19. 研究判断

### 採択する範囲

**Codexによる独立数式・DFアクセス・局所回路構成監査へ進める。** 理由は、未知のUを含む形式的な補正定義で止まらず、既存DF情報から生成できるtuple、有限support回転、正確率、有限δ boundまで具体化できたためである。

### 採択しない範囲

- 一般PRより安いという性能結論。
- NCC/HNCC/既知correctorに対する新規性の確定。
- 分子系production、native gate catalogueの大規模生成、最良γ探索。
- 全RPE/QPE・ground-state energyまでの実用GO。
- 旧G10、RA-D0、BM、SPのrunner再開。

「理論が成立する」ことだけで実用GOを出さない。一方、native性能がまだ測れていないことだけを理由に、ここで構成そのものを無価値と決めない。

## 20. 次のCodex作業：一つの独立監査packet

以下をまとめて実施できる技術作業として渡す。日々の細かな補助testごとにGPT承認を要求しない。

### 目的

本書の構成が正しいか、実際のDF入力から情報・回路取得が閉じるか、既知の同条件構成へそのまま帰着するかを判定する。勝つparameterを探索する作業ではない。

### 対象

1. **式監査**：(1)のblock順序・係数・符号、(4)の次数、P3残差、(11)–(17)の平均・有限δ誤差。
2. **support監査**：complex orbital、sign、I/Z/ZZ、rank degeneracy、spectator parity、control相対位相を含む(8)–(9)。full-Fock matrixは小さな合成検証だけに限定。
3. **source bridge**：[R1–R4]から全D blockとR primitiveを取得し、U_opsと軌道列のorientationを一致させる仕様。実分子providerを新規起動せず、既存/syntheticデータで確認。
4. **sampler監査**：q=c/Aの階層生成、κ=4 fallback、norm-based coin、identity帰結、左右・step独立、有限bit誤差、Γ不変の条件。
5. **native経路の監査**：6-mode中心unitaryの具体的分解方法、controlled実装、同一compiled basis逆列、Gaussian O(nr)構成が実際のAPIで閉じるか。網羅的native合成はせず、有限のgate構造と粗い費用を提示。
6. **先行研究の同値性表**：NCC一般Hamiltonian Appendix E、corrected PF、qDRIFT・疎化、既知fermionic support処理との式・入力・取得手順対応。独立差がなければその具体的対応を示す。

### 返すもの

独立した導出・反例、自己検算scriptと結果、入力/source identity、未解決事項、native価格取得の最小見積りを一つのpacketにする。本GPTのscriptをそのまま再実行しただけで独立証明としない。生成物は新branchへcommit・pushし、remote再取得とref一致を確認する。旧artifactを上書きしない。

### 停止・判定

- 数式反例、controlled位相不一致、DF supportの破綻：理由を保存してSTOP。修正が主構成を変える場合はGPTへ戻す。
- 普遍的κ=4でも生成できない、または未取得の情報をoracle入力としていた：access不成立としてSTOP。
- 既知法と入力・処理・保証・native生成が具体的に同じ：new-method差なしとしてSTOP。応用研究への転換は別判断。
- 正しいが中心native価格が未確定：数学・access通過と性能未確定を分けて返す。未確定を自動GOにしない。
- 全てが通っても分子pilotを自動実行しない。次の重要な科学判断は結果資料の取得確認とユーザーの開始承認を経てGPTで行う。

今回の承認対象はGPTの構成設計であり、ここからCodexが実際に外部環境で作業を開始したわけではない。上記は次に渡せる具体的な作業範囲である。

## 21. 限界と結論

本構成は、DFのprimitive supportを利用し、少数modeの非Gaussian回転を稀に使うことで、実際のordered-D PRの三次誤差を平均として補償する。P5 tailを明示的に採用すればlocal O(δ5)、global O(δ4)の数学的候補となる。追加補正は正のmixtureなので新しいsigned normalizationを加えず、固定γでは期待補正要求数が細分化nに比例しない。

一方、中心回転のnative価格、DF係数sum上界の保守性、γと残余biasのtrade-off、同条件の四次PF/NCC/HNCCとの差が未確定である。6 modesという小ささだけで回路が安いとは言えない。新規性も確定していない。小さな試験でboundが通ったことは、任意入力のfinite-bit証明やnative資源の優位性を認証したことにはならない。

**最終判定：具体的構成あり・独立監査へ進める。性能上の主method採択、分子production、論文化GOはまだ出さない。**

## 22. 参照資料と確認範囲

### 固定repository

基点URL：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/`

- [R1] `docs/df_partial_s2_compiled_cost.md`。実際のblock順序、phase、finite RTE、費用階層。Git blob `28fcdb50e98496178bba0d03a0f19020d72f3b96`。
- [R2] `src/trotterlib/rte.py`、読取行1–210。involution input、paired even cutoff。Git blob `53601b7c524653f5e1966b3955b9f51373e54002`。
- [R3] `src/trotterlib/df_rte_tail.py`、読取行1–240。symbolic supportとbasis registry。Git blob `499d6f9c70609891375cc232a29ad45ae5a29f05`。
- [R4] `src/trotterlib/df_trotter/model.py`、読取行1–175。DF model、blocks、orbitals。Git blob `6e6408ae6ce3e367bc8a4fcab79b038b6171b6b4`。
- [R5] ユーザー共有の `track_b_overall_research_strategy_review_20261011.md`。前段の研究判断。一次実験結果の代用にはしていない。

### 外部一次資料

- [P1] P. Zeng, J. Sun, L. Jiang, Q. Zhao, *Simple and High-Precision Hamiltonian Simulation by Compensating Trotter Error with Linear Combination of Unitary Operations*, PRX Quantum 6, 010359 (2025). DOI `10.1103/PRXQuantum.6.010359`; arXiv `2212.04566`。本文のNCC構成・sampler・一般Hamiltonian Appendix E/Proposition 13を確認。
- [P2] E. Campbell, *A random compiler for fast Hamiltonian simulation*, arXiv `1811.08017`。qDRIFTの一次資料。原理の関連確認であり、同じcoherent-mean契約をそのまま引用したわけではない。
- [P3] Y. Ouyang, D. R. White, E. T. Campbell, *Compilation by stochastic Hamiltonian sparsification*, Quantum 4, 235 (2020), DOI `10.22331/q-2020-02-27-235`, arXiv `1910.06255v2`。確率的疎化の機構・scopeを確認。
- [P4] *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, PRX Quantum 7, 033018 (2026), DOI `10.1103/phk5-slkl`。symmetric/symplectic correctorと実装について関連部分を確認。
- [P5] X. Wang et al., *Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas*, arXiv `2607.11856v2` (23 July 2026)。Theorem 1、channel-level構成、入力classを確認。
- [P6] TE-PAI関連一次資料、arXiv `2410.16850v2`、DOI `10.1088/2058-9565/ae1160`。離散rotation・sampling経路との比較上の注意。全文の独立再導出はしていない。
- [P7] M. Motta et al., *Low rank representations for quantum simulation of electronic structure*, arXiv `1808.02625`。低rank/Gaussian表現の既知背景。
- [P8] S. McArdle, E. Campbell, Y. Su, *Exploiting fermion number in factorized decompositions of the electronic structure Hamiltonian*, arXiv `2107.07238`。factorized commutator評価の最接近背景。今回のsupport構成との全式同値性までは未監査。
- [P9] *Phase Estimation with Partially Randomized Time Evolution*, arXiv `2503.05647`、PRX Quantum 7, 020332 (2026), DOI `10.1103/ynxb-p2xq`。PRの背景。今回のsource条件は[R1–R4]を優先して確認。

これは全関連文献の網羅的優先性調査ではない。NCC PDFの画像取得をweb screenshotで複数回試したが内部エラーとなり、local downloadもDNS接続不能だった。数式の参照は取得できた本文抽出と出版社本文に基づく。図表から新しい定量結果を採用していない。この取得障害をrepository資料の未pushと混同しない。

## 23. 再現資料

ZIPには本書、final自己検算script、保存JSON、stdout、初期script、修正履歴、source identity manifestを含める。外部論文全文やフォントは同梱しない。実行コマンド、versions、seeds、hashは同梱READMEとmanifestを参照。

native price、新しい分子結果、実際の量子shotを含むように誤読しないこと。次の独立監査で誤りが見つかれば、本書の命題・候補を修正または終了する。現在のproof sketchと自己検算が、独立レビューを不要にするという位置付けではない。
