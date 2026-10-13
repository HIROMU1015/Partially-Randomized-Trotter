# C1残存数学課題の限定技術調査 v1

2026-10-14 JST。基準commit `61d50b09cd46eef8b2f570f65b34587e60ae5931`、独立branch `qc-c1-residual-math-audit-20261014`。
[今回指示原文](../../artifacts/qc_c1_residual_math_audit_v1/2026-10-14/user_instructions_original.txt)に基づく一回の限定Codex技術調査。
[前回監査](qc_c1_sign_mechanism_technical_audit_v1_20261013.md)と[引継ぎ](qc_c1_sign_mechanism_review_handoff_v1_20261013.md)の恒等式・保存事実を既知のlemmaとして使用し、旧21check、pilot、保存数値集約を再実行していない。
C1継続/終了/再設計/C4/論文化/正式採択を判断するGPT科学レビューではない。

**追加で閉じた範囲**：三代表のground誤差和の二次形式、可換境界での非負性、現Pauli族内の強いC・弱いBの対称領域に対する係数だけの同符号十分条件。
**閉じていない範囲**：現族全体の同符号定理、族内の異符号存在、未知groundを使わない有利な選択、未選択catalogの有限h最良費用。
新しい異符号例・独立した資源削減算法は具体化できなかった。可能性だけを新成果へ昇格させない。

## 1. 対象・認可・証拠の意味

現族はA=−aZI−bIZ−jZZ、B=uXI+vIX、C=pXX+qYY、a,b,j>0。
pilotの関係u=λ、v=λ/3、p=μ、q=κμを維持する。以下の十分条件はこれと両立する形式係数領域であり、新しい数値point・kernel・入力を登録するものではない。
可換境界の解析では、元validatorがpairwise非可換を要求することを明示する。

保存pilotは非分子2qubit4H/39方法/6h、h=1/(8Λ2^s),s=0…5、Λ=fragment係数L1、target10^-5/10^-6 model units、controlled価格3/6。
geometry/basis/DF rank-policy/L_D/固定粒子sectorは適用なし。元940eig一回/独立監査/通常16比較改善0/6 of 32 precision validは、[基準結果](qc_c1_macrocycle_pilot_v1_results_20261013.md)の既存事実を参照する。
今回の新しい数値性能評価・係数代入・ground/eigen solver・費用再計算・保存数値posthocはzero。

[新8形式恒等式](../../artifacts/qc_c1_residual_math_audit_v1/2026-10-14/new_symbolic_checks.json)はFraction多項式の等式だけを検査した。
旧checker/library/pilot結果をimportしない。manualなPSD/摂動/剰余の証明と、等式checkerの守備範囲を分ける。immutable CI・外部研究者の再現・新規性の証明ではない。

## 2. ground二次形式で整理できる追加の構造

前回のG_X=⟨X(H−E0)X⟩0≥0を使用する。T=Re⟨B(H−E0)C⟩0、S=G_B+G_Cとおけば

\[
|T|\le\sqrt{G_BG_C}\le S/2,\qquad G_A=S+2T.
\]

最後の等式は(H−E0)ψ0=0を使う。t_B=⟨[B,[B,C]]⟩0、t_C=⟨[C,[B,C]]⟩0とすると、前回のABC/ACB式に加えて

\[
d_{BAC}=\frac{S-T}{12}+\frac{t_B-t_C}{24}.
\]

証明：⟨[B,[H,C]]⟩0=2T、⟨[C,[H,C]]⟩0=2G_C、⟨[H,[A,C]]⟩0=0から
⟨[C,[A,C]]⟩0=2G_C−t_C、⟨[A,[A,C]]⟩0=−2T+t_B−2G_C+t_Cを得て、前回のBAC式に代入する。
従って、任意のHermitian三fragmentについて

\[
\boxed{d_{ABC}+d_{ACB}+d_{BAC}
=\frac{G_A+G_B+G_C}{8}=\frac{S+T}{4}\ge\frac S8\ge0.}
\]

**SUPPORTED（構造関係）**。個々のdが正となる定理ではない。前回の一般辞書の異符号形式例とも矛盾しない。
Pauli族でt_B=24x、t_C=12yであることを使う場合も、未知の相関量を無料で知ったとはしない。

## 3. 可換境界での符号：全域と区別する

一組のfragmentが可換なら、三代表のうち二つは二fragment Strangのground係数G_Z/12になる。
残る一つは、可換なX,Yを使って

\[
\frac{G_X+G_Y-\operatorname{Re}\langle X(H-E_0)Y\rangle_0}{12}
\ge\frac{G_X+G_Y}{24}\ge0.
\]

例えば[B,C]=0ならABC/ACBはG_A/12、BACは(S−T)/12。
[A,C]=0ならACB/BACはG_B/12、ABCが残るGram式。
[A,B]=0ならABC/BACはG_C/12、ACBが残るGram式。
非縮退/gapを要求しない。非負とstrict positiveを区別する。

このPauli族でu,vがnonzeroなら[B,C]=0はq=0。
a,b>0、p,q nonzeroの場合、[A,C]=0はa=bかつp=q。
従ってこれらの境界ではstrict opposite signsが排除される。
**元pilotはpairwise非可換を要求するため、この境界を有効な新pilot入力・族全体の定理として数えない。**
近傍へ結論を延長するには、strict positivityのmarginと摂動boundが必要となる。

## 4. 現族内の新しい係数十分条件：強いC・弱いB

前回のA-dominated weak-coupling領域とは別に、次の対称な部分領域を解析する。

\[
a=b=s>0,\quad j>0,\quad p>j,\quad q>j,
\quad r=p-j,\ t=q-j,\ \Delta=rt-s^2>0,
\]

\[
B=\varepsilon B_0,
\quad B_0=u_0XI+v_0IX,\quad u_0=3v_0>0,
\quad W=(u_0-v_0)^2>0.
\]

ε,u0,v0,s,j,p,qに数値を与えない。p≠qかつε≠0なら、三fragmentはpairwise非可換であり元native classに入る。
ε=0は解析の基準となる極限で、zero fragmentをpilotへ登録するものではない。
p=μ、κ=q/p、λ=εu0として元族と対応する。signed q<0やa≠bの領域はこの十分条件に含めない。

### 4.1 未知ground solverを使わない基準点の係数判定

ε=0で既知Bell vector |Ψ−⟩=(|01⟩−|10⟩)/√2はA,C共通の固有vectorで、形式energy E*=j−p−qを持つ。
他のodd vectorの差は2(p+q)。even basis (|Φ−⟩,|Φ+⟩)でH0−E*の形式作用は

\[
K=2\begin{pmatrix}t&-s\\-s&r\end{pmatrix},\qquad
K^{-1}e_1=\frac{(r,s)^T}{2\Delta}.
\]

これは係数未代入の2×2PSD・逆作用の式で、数値Hamiltonianや固有値solverを生成していない。
r,t>0とΔ>0からK≻0なので|Ψ−⟩が一意groundとなる。
係数だけの安全なgap lower boundは

\[
\gamma=\min\{2(p+q),\ 2\Delta/(r+t)\}>0.
\]

Kの最小固有値を数値化せず、det(K)/Tr(K)下界を用いた。
ε≠0の真groundは取得しない。以下の符号certificateはこの係数情報から導く。

### 4.2 主要係数の二次展開

B0|Ψ−⟩=−(u0−v0)|Φ−⟩。
一階state補正のeven成分は+(u0−v0)K^-1 e1、A_even−jI=−2[[j,s],[s,j]]である。

\[
F=t(jr+s^2)^2-2s^2p(jr+s^2)+rs^2p^2\ge0.
\]

この非負性は、正定値Kによる二次形式から得る。前回のα,x,y式をlemmaとして用いると

\[
\alpha=\frac{W F}{6\Delta^2}\varepsilon^2+O(\varepsilon^4),\quad
x=-\frac{qW}{6}\varepsilon^2+O(\varepsilon^4),\quad
y=\frac{q^2Wr}{3\Delta}\varepsilon^2+O(\varepsilon^4).
\]

G=ZZによるB→−Bの共役対称性でground d_j(ε)はevenなので、三次項はない。
ρ=r/t>0、k=s²/(rt)∈(0,1)とおくと

\[
\begin{aligned}
d_j(\varepsilon)&=\varepsilon^2 L_j+O(\varepsilon^4),\\
L_{ABC}&=W\left[\frac F{6\Delta^2}-\frac q6+\frac{q^2r}{3\Delta}\right]>0,\\
L_{ACB}&=\frac{Wt}{6(1-k)^2} f_A(\rho,k),\\
L_{BAC}&=\frac{Wt}{6(1-k)^2} f_B(\rho,k),\\
f_A&=1+(\rho^2-3)k+(3-2\rho)k^2,\\
f_B&=1+\rho(\rho-3)k+\rho k^2.
\end{aligned}
\]

ABCはr/Δ>1/tとq>j>0により、x+yの二次係数が既にstrict positive。
ACBの分子はF+2qΔ²−q²rΔ。BACはG_Bの二次項2Wtと、even空間上のinner A/C誤差から得る。
inner誤差は|Ψ−⟩をannihilateするので、二次state補正との交差項は不要。
BAC分子はtΔ²+s²(r−t)(r²−2rt+s²)となる。

### 4.3 f_A、f_Bの符号を係数だけで閉じる

\[
f_A=(1-k)^2+(\rho-1)k(\rho+1-2k).
\]

ρ≥1ではこの式からstrict positive。0<ρ≤1では

\[
4(3-2\rho)f_A=
[2(3-2\rho)k+\rho^2-3]^2+(1-\rho)^3(\rho+3).
\]

0<ρ<1では第二項が正、ρ=1ではf_A=(1−k)²>0。
また

\[
4\rho f_B=[2\rho k+\rho(\rho-3)]^2+
\rho(4-\rho)(\rho-1)^2.
\]

0<ρ<3で正（ρ=1は(1−k)²）、ρ≥3では元のf_Bの全項が非負でconstant1が残る。
従って全ρ>0、0<k<1でL_ACB,L_BAC>0。
多項式等式は新checkerが検査したが、範囲条件・PSD・摂動論の正当化は本証明に依存する。

### 4.4 「十分小さい」を明示的な係数条件にする

β=|u0|+|v0|、R=γ/(4β)、Λ0=2s+j+p+q、M=2(Λ0+Rβ)^3、L_min=min_j L_j>0とする。

\[
\boxed{0<|\varepsilon|\le
\min\{R/2,\ R^2\sqrt{L_{\min}/(4M)}\}
\quad\Longrightarrow\quad
 d_j(\varepsilon)\ge\tfrac12 L_j\varepsilon^2>0\ \ (j=ABC,ACB,BAC).}
\]

証明：ground-isolating Riesz contourの半径γ/2を用い、|ε|≤Rで||P(ε)||≤2。
前回と同じnested-commutator normの方法から||D_j(ε)||≤(Λ0+Rβ)^3、従って|Tr(PDj)|≤M。
even analyticity、d_j(0)=0と既に求めたL_jから、|ε|≤R/2の四次以降のtailは2M|ε|^4/R^4以下。
上記ε²の条件がtail≤L_min ε²/2を与える。
運用上の制限：基準Bell groundは元情報state |00⟩と直交する。形式摂動から|00⟩とのoverlapはε→0でO(ε²)となり、十分弱いBでは固定の正overlap gateに届かない。ここで未知のfull-H groundをselectorへ入力していない。このconeを元state情報での性能改善領域とはしない。

このradius/gap boundは保守的な十分条件。固定4入力での適用判定・数値calibration・剰余fitは一切行っていない。

**CONDITIONAL（数学的に有効な係数十分条件）**：現native族の非可換な部分領域でも同符号を保証できる。
全領域で異符号が不可能とも、外側で異符号が存在するとも結論しない。
前回のA-dominated coneと今回のC-dominated coneで排除されない領域の存在は、異符号点の存在証明ではない。

## 5. 異符号が仮に成立しても残る運用条件

[前回第6節](qc_c1_sign_mechanism_technical_audit_v1_20261013.md#6-正時間macroの相殺条件と運用条件)の正時間rootと固定4aのratio条件を再計算せず参照する。
真値なしで使うには、full Dに対する取得可能な期待値と有効な不確かさが必要である。
例えば0<Lj≤dj≤Uj、0<Lk≤−dk≤Ukを認証できれば

\[
(|d_k|/(2|d_j|))^{1/3}\in
[(L_k/(2U_j))^{1/3},\ (U_k/(2L_j))^{1/3}].
\]

a=r_c/(1+2r_c)へmonotoneに写すが、今回は新しいaを生成・最適化しない。
固定aで近似期待値がzeroでも、真の|dθ|はboxでは2a³ηj+(1−2a)³ηkまで残る。
joint幅を使う場合もfull Dθのstate/measurement不確かさと情報取得費用は残る。

必要な条件は、単一PF/細分化/S4/C2/既知ordering構成と同じ情報品質・confidence・H・energy精度の下で、
追加情報とcontrolled回路費用を回収できること。
finite-stepのD4、Dθによるbranch mixing/gap、overlap、powerのalias/controlを別に保証する。
leading zeroや符号certificateの存在だけでは総ground-energy costの改善にならない。
今回の新しい十分条件は**異符号を排除する条件**なので、有利なopposite-sign selectorを与えない。

## 6. 残る四機構と既存controlとの差

| 機構 | 具体的内容・必要条件 | 現時点の支持・既知法との差・障害 |
|---|---|---|
| 異符号相殺 | full-ground dが異符号、fixed aのratio/residual幅、安い符号情報 | 固定4点ではなし。今回の追加条件も非相殺領域。族内存在はUNDETERMINED。同catalogの既知compositionにも同じ相殺を与えられる。 |
| 非自明な回路共有 | 強い対照へ同じ最適化を与えても残るsharing | 現合法fusionには前回のno-goが適用。角度合成等のbackendを変更せず、追加の具体的sharing差は特定できない。 |
| 有限stepの構造的利益 | D4/高次/枝・overlapを含む同費用利益 | 未選択catalogのoracleは未集計。仮に利益があっても同じVθ(h)を既知compositionが生成するため、現factoryの独立差にはならない。 |
| 取得可能state情報の選択 | full Dθのjoint幅、情報費用を含む非自明規則 | joint幅はboxより鋭くなり得るが既存controlも同じ規則を使用。同品質・同価格の独立した選択差は未特定。 |

ここで「具体的な方法差を特定できない」は研究の正式終了判断ではない。
新kernel・新input・数値scan・別backendを自動導入してこの結論を回避しない。

## 7. 保存posthocの情報価値と未認可の最小案

[機械可読の未実行案](../../artifacts/qc_c1_residual_math_audit_v1/2026-10-14/posthoc_information_value_plan.json)。
static source inspectionにより、元comparisons.jsonのrowsは全method/hのeligibleとconditional_quantum_costを既に保存することを確認した。
今回はそのrowsを新集計せず、sourceの保存schemaだけを確認した。

C_Mを24macroだけの保存oracle最小費用、C_Bを元single/subdivision/S4/C2対照の最小費用とする。
selector regretには同じC1 catalog（single/subdivision＋24macro）のC_K=min(C_single_subdivision,C_M)を使う。
C_saved_selector/C_K−1は**method選択についての事後診断**。selected method内のhは既に真biasで最良点を選ぶため、完全な運用precision selectorのregretではない。

| 未実行posthocで得る可能な結果 | 新たに分かること | 変わらないこと |
|---|---|---|
| oracleにも5%優位macroなし | この有限catalog/grid/価格/精度/状態では、selector改良だけでは登録利益を得られない | 全族/全PF/分子の不能性、正式終了は導けない |
| oracleには優位、保存selectorは選べない | selected failureとcatalog-level failureを分離できる | 運用可能な安い情報取得、保証、総費用は未実証 |
| oracle優位がknown compositionでも同じ | 有限templateの適用利益・既知構成のtradeoffになり得る | 現factoryの独立新規性にはならない |
| eligible点なし/null/数値境界/欠落 | 現保存情報では判定不能である理由 | 閾値変更、追加h、solver再実行を正当化しない |

**提案だけ**：固定4H/24macro/6h、元32contextsを保持、主比較は通常state16、bad-state16は既存controlとして保持。
既存比較行のeligible/cost/frontierのみを読み、既存Hamiltonian/vector/solver/費用sourceを再実行しない。
元価格3/6、target10^-5/10^-6、5%gate、overlap/branch/energy marginを維持。
新しい値はC_M,C_K,ratio,method-regretとpointerだけ。bias/期待値/gap/費用の再生成はzero。
最大既存8064比較行・32contexts、1worker/1thread、memory256MiB、output2MiB、log1MiB、eigensolver/PF/compile calls0、phase/total wall capなし。
入力SHA不一致、missing/nonfinite/null cost、method不整合、cap超過はfailure/UNDETERMINEDを保存し、retry/修復/追加scopeを行わない。
future runner/checker/source/input SHAを別scopeへ固定してから、一回/retry0の**別明示実行承認**が必要。今回はrunnerを実装・実行していない。

この診断は、どの失敗を説明できるかを変え得るが、既知factoryとの同値性を変えない。
費用が小さい診断でも、研究判断に必要かはGPTが判断する。追加計算を必須の次作業としない。

## 8. 新規性・論文化判断へ渡す技術的区分

| 区分 | 確認できること | 認定していないこと |
|---|---|---|
| 既知PF composition適用 | current catalog/min-selector/controlは同じ生成規則 | 独立した新生成算法 |
| ordering相殺 | opposite signsなら正時間root/残差幅を記述できる | 現族での実現、free ground情報なしの利益 |
| 新しい生成手順候補 | 具体的な差は未特定 | 名前・候補数変更を新規性とすること |
| 取得可能情報による選択 | state/measurement幅を含む条件付き運用契約 | 同品質・同価格のstrong controlより優れる規則 |
| 追加の数学 | ground和/可換境界/strong-C coneの符号条件 | 外部査読済み新規定理・論文主claim・研究採択 |
| 分子への接続 | native commuting fragments、利用可能state/RDM、measurement/control/uncomputeが新たに必要 | Pauli toyの構造/gap/sign/価格をDF分子へ無料移送 |
| full-ground-energy資源 | preparation/overlap/confidence/情報取得/control/合成を含めた公平比較が必要 | conditional Heisenberg proxyを全QPE/RPE費用とすること |

近接文献の確認範囲は[前回の文献記録](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/literature_check.json)と[完了選定レビュー](qc_four_algorithm_candidates_scientific_selection_review_2026-10-13.md)を参照する。
今回は一般文献調査や同じ新規性レビューを反復していない。未読を差分の証拠としない。

## 9. 限定調査の終了点

解析的に追加で導出できた範囲と残存課題、四機構の具体性、保存posthocの情報価値を整理した。
族内異符号存在の問題を完全には解決できなかったが、一回の限定scopeを延長しない。
[GPT判断資料](qc_c1_residual_math_review_handoff_v1_20261014.md)へ引継ぐ。
旧科学source/結果/認可/STOP/global manifest FAIL、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template終了を保持する。
公開・remote取得一致後Mandatory STOP。新しいGPT科学判断の開始承認、新数値posthocの実行承認は独立に必要となる。
