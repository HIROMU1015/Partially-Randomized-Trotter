# C1未達原因・異符号相殺機構：静的技術監査 v1

2026-10-13 JST。基準結果commit `3dd94637e8da77ce6e1f724f12f6a67dd4ded7ed`、独立branch `qc-c1-sign-mechanism-audit-20261013`。
ユーザーの[今回指示原文](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/user_instructions_original.txt)に基づくCodex技術監査である。
新しいGPT科学レビュー、C1の採否・終了判断、C4への移行判断ではない。
[基準結果](qc_c1_macrocycle_pilot_v1_results_20261013.md)、[元設計](qc_c1_macrocycle_technical_design_v1_20261013.md)、[事前登録](qc_c1_macrocycle_preregistration_v1_20261013.md)、[完了選定レビュー](qc_four_algorithm_candidates_scientific_selection_review_2026-10-13.md)を使用した。

今回、保存literal/pointerを読むことと、係数未代入の数式恒等式を調べることに限定した。
新しい数値Hamiltonian/ground solver/PF性能/費用再計算/数値的事後集計/保存vectorからの新しい物理量/最適化/compileは全てzero。
保存結果を再生成・書換えず、消費済み940-call認可とretry0/Mandatory STOPを保持する。

技術上の主要結論は次である。

- **SUPPORTED**：真kernel dは保存4入力で全て正なので、正時間macroの真の主要項相殺はこの固定scopeでは起きない。
- **SUPPORTED**：S1_weak/S4_strongでは近似stateの第三kernel期待値だけが負で、真値の正と食い違う。S4_strongでは近似macro期待値zeroを選ぶが真の主要項は非zero。符号情報の問題を具体的に確認できる。
- **CONDITIONAL**：同じPauli族の十分弱い結合領域では、3kernelに共通の正の二次coupling項から同符号性を証明できる。ただし固定4点へ新しい係数代入をしていないので、その十分条件が4点全てを覆うとは主張しない。
- **SUPPORTED（限定された構造定理）**：今回のPauli族ではABC/ACBの二つが同時に負になることはない。各々が正になることや、3kernel全ての同符号性を保証する定理ではない。
- **UNDETERMINED**：現Pauli族内で、係数条件だけから異符号を実現できるか。一般の三fragment辞書では形式的に可能だが、その反例はpilot族/native templateの外にある。
- **UNDETERMINED**：未選択macro全catalogの最良有限h性能、selector regretの大きさ、有限h/情報保証を変えた費用利益。必要な新しいposthoc集計を今回は実行していない。

## 1. 監査のscopeと証拠区分

保存pilotは非分子2qubit Pauli4H、3代表S2/24macro、39方法、6h、h=1/(8Λ2^s),s=0…5、Λ=fragment係数L1。
energy target10^-5/10^-6はmodel単位、controlled価格3/6、geometry/basis/DF rank-policy/L_D/固定粒子sectorは適用なし。
940eig一回、保存監査936unitary/8064比較行PASS、通常16比較の全3selectorで5%改善0は既存結果として引用する。
今回の新しい数学はstatic identity/conditional proofであり、新しい数値性能resultではない。

[読み取ったliteral/pointer](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/saved_literal_index.json)は保存fieldsをそのまま抜粋したもの。
新しい平均、最大最小、cost ratio、oracle frontier、期待値、gapを生成していない。
[形式恒等式検査](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/symbolic_identity_checks.json)は有理係数多項式とPauli文字積だけを使用し、登録係数を代入せず、行列・固有値を生成しない。
weak-couplingの剰余boundは以下の解析的導出として保存し、数値calibration済みとしない。

## 2. ground-stateに限って成り立つ主要項の分解

H=A+B+C、Hψ0=E0ψ0とする。Hermitian Xについて

\[
G_X=\langle\psi_0|X(H-E_0)X|\psi_0\rangle\ge0,
\quad \langle[X,[X,H]]\rangle_0=-2G_X.
\]

任意のKで⟨[H,K]⟩0=0。これらはground variational/spectral positivityから得る標準恒等式である。
physical convention S=exp[−ih(H+h²D+…)]では、outer Strang項

\[
\frac{[X,[X,H-X]]}{24}+\frac{[H-X,[X,H-X]]}{12}
=-\frac{[X,[X,H]]}{24}+\frac{[H,[X,H]]}{12}
\]

のground期待値はG_X/12となる。inner Strang誤差は、一般には同じgroundを持たない二fragmentの誤差なので、その符号をこのpositivityで置き換えられない。

\[
\begin{aligned}
d_{ABC}&=G_A/12+\langle[B,[B,C]]\rangle_0/24+\langle[C,[B,C]]\rangle_0/12,\\
d_{ACB}&=G_A/12-\langle[C,[B,C]]\rangle_0/24-\langle[B,[B,C]]\rangle_0/12,\\
d_{BAC}&=G_B/12+\langle[A,[A,C]]\rangle_0/24+\langle[C,[A,C]]\rangle_0/12.
\end{aligned}
\]

したがってG_X≥0だけでは3つのdの符号は確定しない。kernel間の差は

\[
D_{ABC}-D_{ACB}=\tfrac18[B+C,[B,C]],\qquad
d_{ABC}-d_{ACB}=-\tfrac18\langle[A,[B,C]]\rangle_0.
\]

後者はground条件を使った期待値等式で、演算子等式ではない。
**近似stateのdtildeへ、このground-gauge分解をそのまま適用しない。** 元selectorは完全なDを保持しており、その契約を変更しない。
またTr Dj=0である。非zero Hermitian Djは全Hilbert空間でindefiniteなので、d_j>0をDj≽0と混同しない。

## 3. pilot族の明示的Pauli構造と符号条件

identity shiftを省き、A=−aZI−bIZ−jZZ、B=uXI+vIX、C=pXX+qYY。
pilotの対応はa,b,j>0、u=λ、v=λ/3、p=μ、q=κμ。以下は全て係数をsymbolのまま扱う。

\[
\begin{aligned}
[B,[B,C]]&=4q\{(u^2+v^2)YY-2uvZZ\},\\
[C,[B,C]]&=-4q^2B,\\
[A,[B,C]]&=-4q\{buZX+avXZ+j(uIX+vXI)\}.
\end{aligned}
\]

\[
\begin{aligned}
[A,[A,C]]={}&4\{[(a^2+b^2)p-2abq]XX+[(a^2+b^2)q-2abp]YY\},\\
[C,[A,C]]={}&4\{[a(p^2+q^2)-2bpq]ZI+[b(p^2+q^2)-2apq]IZ\}.
\end{aligned}
\]

各恒等式は、二qubit Pauli語と形式係数の独立文字積により検査した。
qの符号だけでは、未知のground correlations ⟨YY⟩0,⟨ZZ⟩0,⟨B⟩0等が入るdの符号は決まらない。
jはinner B/C exchangeに直接現れなくても、G_A/G_Bおよびground correlationsに影響する。

G=ZZはA,Cを保存しBを反転する。ground variational原理から

\[
E_0\le\langle\psi_0|G H G|\psi_0\rangle=E_0-2\langle B\rangle_0,
\quad \langle B\rangle_0\le0.
\]

非縮退やgapを要求しない。そこで

\[
\alpha=G_A/12\ge0,\quad
x=\tfrac q6\{(u^2+v^2)\langle YY\rangle_0-2uv\langle ZZ\rangle_0\},\quad
y=-\tfrac{q^2}3\langle B\rangle_0\ge0
\]

とおくとdABC=α+x+y、dACB=α−2x−y/2。

\[
2d_{ABC}+d_{ACB}=3\alpha+\tfrac32 y\ge0.
\]

これは**両方が負となることを除外する**構造的結果で、個別の符号反転を除外しない。
二つが共に非負である必要十分条件は

\[
-\alpha-y\le x\le\alpha/2-y/4.
\]

strict opposite signsにはx<−α−yまたはx>α/2−y/4が必要十分。
third kernelについては、上記explicit A/C nested式の期待値が−G_B/12以上という別条件が必要。
したがって、この条件だけで全族の同符号性や異符号点の存在を断定しない。
係数のみの条件へ閉じるには、G_A/G_Bとcorrelationの有効bounds、または別の構造証明が不足している。
保存ground vectorからこれらを新計算することは今回禁止されているため、固定4点での式の数値分解は未実施。

## 4. 十分弱い結合領域の共通正符号：条件付き証明

Aのground |00⟩は一意で、positive excitation gaps
Δ1=2(a+j)、Δ2=2(b+j)、Δ12=2(a+b)を持つ。
固定real B0=u0XI+v0IX、C0=p0XX+q0YYについてH(ζ)=A+ζ(B0+C0)とする。
これは解析用symbol ζであり、新しい数値inputを作成していない。
B0はsingle flip、C0はdouble flipのみで|00⟩から結合する。
一次のground correctionはsymbolically

\[
|\psi(\zeta)\rangle=|00\rangle-\zeta\left(
\frac{u_0}{\Delta_1}|10\rangle+\frac{v_0}{\Delta_2}|01\rangle+
\frac{p_0-q_0}{\Delta_{12}}|11\rangle\right)+O(\zeta^2).
\]

三代表のphysical BCHをこの形式展開へ入れると

\[
d_j(\zeta)=\zeta^2 L+O(\zeta^3),\quad
L=\frac{\Delta_1u_0^2+\Delta_2v_0^2+\Delta_{12}(p_0-q_0)^2}{12}.
\]

ABC/ACB：outer Dの二次seed項はgap-weightの1/6、一次state補正は−1/12、inner B/Cはcubic以降。
BAC：B項はouter seed −1/12とstate補正+1/6、C項はinner seed+1/6とstate補正−1/12。
single/double couplingがdisjointなので二次cross項はzero、各々1/12に揃う。
L>0なら十分小さい非zero ζで全3dがstrict positive。q0の符号が異なっても(p0−q0)²からこの領域は残る。

存在論だけにしないため、以下の保守的なexplicit係数boundも得られる。
A0=A−E_A I、α0=max(Δ1,Δ2,Δ12)、γA=min(Δ1,Δ2,Δ12)、
β=|u0|+|v0|+|p0|+|q0|>0、r=γA/(4β)、M=2(α0+rβ)^3。

\[
0<|\zeta|\le\min\left\{r/2,\frac{Lr^3}{4M}\right\}
\quad\Longrightarrow\quad d_j(\zeta)\ge\tfrac12 L\zeta^2>0.
\]

証明：複素ζで|ζ|≤r、Riesz contour |z|=γA/2上のresolventをNeumann級数でboundすると||P(ζ)||≤2、rank P=1。
各nested commutatorを||[X,[Y,Z]]||≤4||X||||Y||||Z||でboundすると||Dj(ζ)||≤(α0+rβ)^3。
従って|Tr(PDj)|≤M、Cauchyの係数boundより、|ζ|≤r/2で三次以降のtail≤2M|ζ|³/r³。
既に求めた二次係数Lから上記lower boundが従う。数値calibration・外部査読済みの新規定理とは呼ばない。

**適用範囲の限界**：この条件は十分条件で、符号変化の必要条件ではない。
固定4点がこの保守的な領域に入るか、新しい係数代入による判定はしていない。
特にS2/S4等の強い結合で観測した正を、このweak theoremだけで説明済みとはしない。
固定4点の完全な係数的正符号証明は `UNDETERMINED`。

## 5. 一般辞書の異符号可能性と現族の未決着を分ける

「正時間の三fragment対称S2は必ずground d≥0」という一般化は成り立たない。
係数未代入の形式例A=−X+(τ−1)Z、B=X、C=Z、0<τ<1を考える。
H=τZの既知Pauli ground |1⟩の恒等式から

\[
d_{ABC}=d_{BAC}=(\tau-1)/6<0,\qquad d_{ACB}=(\tau+2)/6>0.
\]

全てのpair commutatorは非zero、係数τは一度も数値化していない。
自由BCHとPauli文字積で多項式等式を検査しただけで、新しいground solver・eigensolve・時間発展を実行していない。
これは辞書一般に対する数学的反例であり、**二qubit pilotの対角density A/commuting-Pauli native templateの例ではない**。
H自体が単一Pauliで直接実装できるため、資源削減算法の実用例にもならない。
この族から数値pointや次pilot入力を選ばない。現pilot族での異符号存在はなお `UNDETERMINED`。

reverse-isospectralなABC/CBA等を再導入して「相殺候補を増やす」ことはしない。
一般にprocessorによるunitary conjugationはスペクトルを変えず、Djの[H,K]差も真のeigenstate期待値を変えない。
演算子表示や近似state期待値だけの差を独立したeigenphase機構と数えない。

## 6. 正時間macroの相殺条件と運用条件

0<a<1/2、b=1−2a>0でdθ=2a³dj+b³dk。
dj,dkがnonzeroならstrict opposite signsがexact zeroの必要十分条件。
r_c=(|dk|/(2|dj|))^(1/3)とすれば一意のzero係数a=r_c/(1+2r_c)、b=1/(1+2r_c)となる。
一方がzeroで他方nonzeroならstrict positive timesではexact zeroにならず、両方zeroの場合は別の剰余解析が必要。

登録された4時間だけでexact cancelする条件は、係数未代入の代数式として

| 登録a | 必要なdk/dj |
|---|---:|
| 1/8 | −1/108 |
| 1/4 | −1/4 |
| 1/3 | −2 |
| 3/8 | −27/4 |

異符号が存在しても、この離散辞書がrootを含むとは限らない。root formulaを使って新しいaを最適化してはいない。
真値なしの運用には、dtilde_j±ηj、dtilde_k±ηkがzeroを跨がない符号certificateか、同等の安い構造情報が必要。
例えばdtilde_j−ηj>0、dtilde_k+ηk<0。joint状態幅は相殺後のDθに適用するが、真のratioを無料で与えない。
確実な幅eθ、finite-h remainder/枝/overlap、controlled費用gθ、追加情報費用Iθを含めて対照と比べる。
主要量gθ√eθが小さいことは必要な手掛かりで、最終費用削減の証明ではない。
exact dθ=0でもDj演算子がzeroとは限らず、O(h^4)||Dθ||²/gapとoverlap負担が残る。
S4/C2も四次biasを得る既知強対照なので、次数の改善だけでは独立算法として採択しない。

## 7. 保存証拠が実際に支える未達原因

以下の数値は全て[保存literal索引](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/saved_literal_index.json)の固定scopeからの引用である。

### 7.1 相殺されなかった真の主要項：SUPPORTED

全4Hの真d_j>0は基準結果の既存観測である。従って全登録正時間macroに真のleading zeroはない。
uniform m subdivisionでもd/m²となるため、同符号positive主要項の減少だけで特別な相殺効果を得たとしない。
固定加法価格かつ同じ合法fusionでは、gθ≥Σg_j,steady。単一反復にもouter boundary fusionを与えるとsame-sign Hölder制約を回避しない。
このasymptotic制約を有限hでの全macro不能に拡張しない。

### 7.2 近似stateの符号誤認：SUPPORTED、macro全体の不能性は未判定

S1_weakのsealではS2_m1(BAC)のdtilde=−145/221184、保存真dBACは正。
empiricalはM02_1_4を選ぶ。S4_strongでは同kernelのdtilde=−5/72、真dBACは正で、
M20_1_3のdtildeはexact zero。3真kernel全て正なので、このmacroの真dθは正である。
後者の保存true_leading_dは0.0034541795699245116であり、これは今回再計算した値ではない。
従って「近似state上の相殺を真のground誤差相殺と扱った」ことが具体的な未達要因として支持される。
S4_strongのsaved dρ=2/gap=null、S1_weakでもdρ=2/11が非zeroで、符号を保証する入力契約ではない。
近似stateを真のgroundに置換して候補を選ぶ診断・新物理量計算は実行しない。

### 7.3 Controlled費用・fusion・強い対照：SUPPORTED in registered model

保存sealのS1_weak/価格3のliteralを引用する。

| method | RZ | CX | basis Clifford | 保存g_hypothesis |
|---|---:|---:|---:|---:|
| S0_m1 ABC | 19 | 30 | 20 | 129 |
| S1_m1 ACB | 19 | 38 | 28 | 161 |
| S2_m1 BAC | 22 | 36 | 16 | 146 |
| M02_1_4 | 71 | 110 | 60 | 461 |
| M02_1_3 | 71 | 110 | 60 | 461 |

このtableは費用sourceを再実行した再計算ではない。周期single fusion、macro内部・周期boundaryの隣接fusion後の保存countsである。
scalar controlled phaseとbasis Cliffordを含め、noncommutingな再順序化や全compiler最適化はしていない。
同じa族でstage種が変わらない限り費用はa非依存というモデルで、微小angleを安く合成するbackendの結果ではない。
S4の負時間/五stage、C2の二推定/statistical weightsは元sourceが会計済み。
価格3/6の両登録比較で5%改善0は既存保存集約を参照し、別価格のfrontierを計算しない。
T/depth/routing/full QPE/RPE費用はUNKNOWN、登録モデル外のprice感度は `UNDETERMINED`。

### 7.4 有限h・branch・oracle：保存例でSUPPORTED、全体帰属はUNDETERMINED

保存例S1_weak/M02_1_4/h_index1：h=0.030973451327433628、bias=1.4672949544269898e−7、保存finite-h remainder=1.1451732022079869e−11。
S4_strong/M20_1_3/h_index2：h=0.011290322580645162、bias=4.403108142536638e−7、保存remainder=1.7639771260272903e−12。
これら二例は「大きい高次剰余が相殺を台無しにした」説明を支持せず、nonzero真主要項が残ったことを支持する。
全936点での残差割合や新しいh依存fitは集計していない。他の未選択候補とgrid外の性能は未決着。
保存branch-valid936は基準集約の引用。通常stateの未達をbranch失敗に帰す証拠はない。
bad-state controlのoverlap不適格は別の原因で、通常stateのcost未達と混ぜない。
reference frontierがprecision hを真biasで選ぶことはsourceに明記され、運用のcoefficient-only hとは異なる。

## 8. Selector・情報保証・macro性能の分離

| 問い | 判定 | 保存証拠/不足 |
|---|---|---|
| empiricalが近似期待値の誤った符号・zeroを使ったか | SUPPORTED | 上記S1/S4のsealと保存true d |
| selected C1が強い対照へ5%勝ったか | NOT_SUPPORTED_IN_FIXED_SCOPE | 通常16比較、全3selector改善0という既存集約 |
| 未選択catalogに有利なmacroがないか | UNDETERMINED | macro全catalogの新oracle集計は未実施 |
| certified sign情報へ変えると利益を回収するか | UNDETERMINED | 取得費用とfinite-h比較が不足 |
| coefficient-only6valid/26nullが全てmacro性能失敗か | NOT_SUPPORTED | 数学保証のeligibilityと価格/phase performanceは別 |
| S2_balanced/S4_strongのnullは何で起きるか | SUPPORTED（このsource） | saved gamma=null、sourceはgapなしでboundをnullにする |
| S3_signedのstrict target nullはgapなしが原因か | NOT_SUPPORTED | gamma=19/24は保存済み。どのbound項がdominantかは新分解禁止のためUNDETERMINED |
| strong-known-compositionとの生成規則差があるか | NOT_SUPPORTED | 同catalog/情報/score/sharing、min ruleが同一 |

S1_weak/価格3/target10^-6の保存frontierでは、empiricalはM02_1_4/h1、jointはM02_1_3/h0、single対照はS0_m3/h0。
既存集約tableではjointも最良対照への5%改善なし。selector間の差が実在しても、macro全体の不能性や運用成功には直結しない。
certified PF精度の6/32は、S1/S3の一部でs5の十分条件が通った件数で、通常16比較の費用改善0と別に保持する。

## 9. 判断を変え得る情報と、今回実行しない最小案

現在の数式・保存証拠で不足する事項を順序付きで列挙する。以下は計画だけで認可/実装/実行ではない。

1. **数学を先に**：現Pauli族のG_A/G_B・x/y制約を係数で閉じ、異符号を許す領域または排除領域を証明する。
   数値符号scanを代替物にしない。weak cone外の解析的構造が見つからなければ、異符号pilotを設計済みと表示しない。
2. **保存データが必要なら別認可**：既存4H/24macro/6hだけでoracle catalog frontierを取得し、保存selectorとの差を調べる。
   保存比較行/既存countsだけのscopeでfresh eig/compileなし。4H×24×6の既存rowsを読む数MiB規模、1worker/1thread・memory/output/call caps、wall capなしという最小案。
   これは新しい数値的事後集計なので、今は実行しない。元結果を上書きせず別namespaceが必要。
3. **有効な符号情報の取得案**：full Djに対する近似state期待値と測定/trace幅でzero-crossingを排除する。
   希望誤差ηに対するmeasurement費用は一般に係数scale²/η²とstate preparation/confidenceに依存する。
   η<|d|が必要なsmall-dでは高価になり得る。H4/H6や新stateを今生成しない。
4. **剰余・価格の切り分け**：もし科学判断に必要なら、保存bound各項/既存native countsの新posthoc分解を別scopeとして認可する。
   新しいPF hや価格scanを含めず、取得情報が具体的に判断を変えるかを先に問う。

all-positiveの現4点へ正時間aを増やしてもleading exact cancellationは得られない。
reverse-isospectral対を戻す、同sharingをsingleへ与えない、同じknown-min selectorを別名にすることは判断を変えにくい。
新しいmath mechanismが特定できなくても探索を延長しない。正式継続/終了/C4/中心themeはGPT科学判断へ残す。

## 10. 先行研究との関係・claim境界

effective-Hamiltonian/commutatorの固有state期待値と、state fidelity・energy errorを分ける観点は
[Yi–Crosson, npj Quantum Information 8,37 (2022)](https://www.nature.com/articles/s41534-022-00548-w)の関連節を確認した。
本稿のground commutator identityはこの標準的なスペクトル的考え方に沿う導出で、processor表示の変更をenergy改善としない。
diagonal BCH/approximate-state error estimationとrecursive symmetric BCHについて
[Maxwell et al., arXiv:2606.30738v1, §III.1–III.2](https://arxiv.org/html/2606.30738v1)を確認した。
ordering/basis sequenceの相殺構成は[Kronenberger et al., arXiv:2602.18913v1, §2.5/§4.3](https://arxiv.org/html/2602.18913v1)に既知の比較対象がある。
対象節だけを確認し、一般的な新規性survey・新版同一視・科学レビューを再開していない。

今回のPauli縮約、partial sign condition、weak-cone十分条件、係数未代入の形式反例はCodexのstatic技術導出として保存する。
独立新規性や論文claimを認定せず、完全なpilot族同符号定理/族内opposite example/資源優位を得たと主張しない。
[GPT引継ぎ](qc_c1_sign_mechanism_review_handoff_v1_20261013.md)の材料が揃った段階でMandatory STOP。
