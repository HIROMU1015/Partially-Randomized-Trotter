# C2・C3最終限定技術監査：既知手順への還元と資源上の未解決点

2026-10-14 JST。ユーザーが指定した一回の技術監査。C2の二点PF固有値外挿、C3のself-energy Schur-rootは既知手順へ還元できる。現案について、既知法と区別できる低費用の情報取得法・回路生成法は特定できなかった。詳細実装・pilot・追加候補探索へ進まない。この技術的所見は、研究の正式採択・継続・終了・論文化判断ではない。

研究目的は、元の量子化学Hamiltonianのground energyを、精度・成功確率を維持して低費用に取得すること。今回、係数を代入した性能計算は行わない。新規Hamiltonian、固有値、PF、保存データの数値集計、sampling、compile、T合成、C1/C4再監査、GPT科学レビューはすべて0。

## 1. 固定資料と判定語

元案は `fc063696026b2500d9608abe447849fd76274ed5` の [4候補技術資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md)。本branchの基準はC4完了 `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38`。固定本文を実際に取得し、[参照identity](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/reference_identity.json)にcommit・path・SHA256・bytes・GitHub照合を保存した。

[一次文献registry](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/literature_registry.json)、[scope・原指示SHA](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/scope.json)、[claim状態](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/method_status.json)、[研究全体の引継ぎ](qc_research_integrated_gpt_decision_handoff_v1_20261014.md)。文献全文をrepositoryへ転載しない。

`SUPPORTED` は条件付き数式・固定本文に支持される所見、`REDUCES_TO_KNOWN` は原案の操作が既知手順に対応する所見、`CONDITIONAL` は入力・access・保証があれば成立する構成、`UNDETERMINED` は優位性・独立方法差など未確定の事項。実gate単価・全費用の未取得値は `UNKNOWN` とする。未解決事項があること自体を新規性と数えない。

## 2. 一次本文との直接対応

| 一次資料・確認箇所 | 既知の内容／必要情報 | 現案への対応と限界 |
| --- | --- | --- |
| [Rendon–Watkins–Wiebe, Chebyshev interpolation, v4](https://arxiv.org/pdf/2212.14144v4), Lemma1, Theorem7, Lemma11–12, Theorem15/17, AppendixD.5–6 | PFの有効Hamiltonianと固有値の補間。small-step・孤立branch・gapの条件、perturbed groundの準備費用、推定精度配分も扱う | C2外挿、branch確認、費用配分を追加しただけでは独立差にならない。定理ごとのoracular state、gap下界、term-exponential accessを保持する |
| [Mehendaleら, PF eigenvalue-error prediction, 2025出版版](https://pubs.rsc.org/ja-jp/content/articlepdf/2025/dd/d5dd00185d?page=search), II Eqs.(5)–(9), III.B, IV, AppendicesC/D/F | 対称PFのBCH主要項を近似stateで予測し、QPE費用を見積る。予測値は厳密上界ではない | 「stateを用いたenergy bias予測」は既知。C2に必要な同branch・有限剰余のcertificateとは区別する |
| [Childsら, A Theory of Trotter Error, v3](https://arxiv.org/pdf/1912.08854), Theorem11/Corollary12 | commutatorを用いたoperator誤差とstep数の保証 | energy-specific予測やbranch同定を自動的には与えない。既知の保守的対照として利用可能 |
| [Dusson–Sigal–Stamm, Feshbach–Schur](https://arxiv.org/pdf/2105.02058), Theorem1.2, Corollary1.3, Eq.(1.16), Proposition1.4 | 外部逆作用素、self-energy、非線形固有値root、単調性 | C3の基本構成に直接対応。外部逆作用素の低費用な量子accessは別問題 |
| [Exact downfolding and its perturbative approximation, v4](https://arxiv.org/html/2507.16916v4), I–II, AppendixF | partial trace/path integralでtarget observablesを保存するeffective action | 温度依存・retarded作用を持つ方式。energy-dependent Schur固有値問題と同じ数式と断定しない |
| [Bauman–Low–Kowalski, DUCC](https://arxiv.org/pdf/1909.06404), II.A–C | external anti-Hermitian変換によるHermitian active-space Hamiltonian。exact性は外部amplitudeを含む条件付き | C3はE依存resolvent、DUCCは変換後のHamiltonianという既知の相違。実用amplitude・commutator切断の誤差／費用は残る |
| [Bravyi–DiVincenzo–Loss, SW v1](https://arxiv.org/pdf/1105.0675v1), II–III, IV.4 Theorem3 | gap・摂動条件の下での低エネルギー変換と切断誤差 | C3のself-energyを保持する発想だけを新規としない。local-spin定理の条件を一般分子へ無断移送しない |
| [Gilyénら, QSVT v1](https://arxiv.org/pdf/1806.01838), Definition43, Lemma40/Theorem41, Lemma52–53 | scaled inverse、LCU、block積、normalizationと誤差伝播 | C3 resolventの関数実装は既知access手順。入力encoding・normalization・制御の単価は必要 |
| [Berryら, truncated Taylor/LCU v1](https://arxiv.org/pdf/1412.4687), SELECT/PREPAREとTaylor構成 | 展開係数とunitary accessから関数を実装 | polynomialが存在してもSELECT/PREPAREが無償にはならない |
| [DFTHC/SOS spectrum amplification v1](https://arxiv.org/html/2502.15882v1), II–IV, AppendixB；[SOSSA v1](https://arxiv.org/html/2505.01528v1), II–III, Table1, AppendixA | SOS下界、square-root encodingと低エネルギーのquery削減。gate費用はaccess構成に依存 | C3と同じactive eliminationではないが、元Hのground-energyを改善する強い対照。低い下界やquery数だけで全gate優位を認定しない |

RSCの直接PDF/HTML取得は403だったが、publisherの12ページPDFをweb経由で取得し、関連本文・式を確認した。旧first-order preprintで代用しない。web本文にローカルSHAを捏造せず、URL・pages・取得状態をregistryへ記録した。他10本文はローカル取得bytes/SHAで固定。必須比較本文の未取得は0。これは関連節の限定照合で、全証明の再証明・先行性の網羅調査ではない。

## 3. C2：二点外挿とground branch

### 3.1 既知原理への還元

対称PF `S(h)` について、同じlog branch上の `H_eff(h)=(i/h)log S(h)` がh=0近傍で解析的なら、孤立固有値branchはeven展開を持つ。元案は既知のRichardson cancellationである。

\[
E(s)=E_0+c_2s^2+R(s),\quad |R(s)|\le C s^4\quad(s=h,h/2),
\]
\[
E_{\rm ext}=\frac{4E(h/2)-E(h)}3,
\qquad |E_{\rm ext}-E_0|\le\frac5{12}C h^4.
\]

係数未代入で式を展開すると二次項は消え、残る `4R(h/2)/3−R(h)/3` の三角不等式が上式を与える。二点・三点の数値fitからCやbranchを証明することはできない。少数点の一致を「剰余保証」としない。

RWW Theorem7は対象eigenstateのoracular準備、全区間の非zero gap、Lemma1のsmall-stepを仮定する。Lemma12はgapと有効Hの近接から解析半径を制限する。Theorem15/AppendixD.5はgap下界を使う準備費用を、Theorem17/AppendixD.6は推定・精度配分を扱う。したがって「先行法はbranchや準備を扱わず、C2が追加する」というclaimは支持されない。[RWW本文](https://arxiv.org/pdf/2212.14144v4)

これらはterm-exponential accessの条件付き結果であり、未知groundの安価な初期overlap・gap証拠・分子native controlled gate価格を一括で与えるという解釈はしない。C2もそれらを未取得のまま優位性へ変換できない。

### 3.2 原案の操作列と入力依存

| 操作 | 必要な取得情報 | 技術判定 |
| --- | --- | --- |
| 分解・対称PF・二つの幅を固定 | 同じ元H、scalar、ordering、synthesis精度 | 既知構成。kernel生成の独立差なし |
| unwrap可能なwindowを固定 | 係数norm上界、全幅に共通なenergy shift、logの解析domain | 既知条件。shiftのcontrolled relative phaseを落とさない |
| groundに連なるbranchを同定 | gap下界、ground識別情報、overlap、推定interval | CONDITIONAL。安い情報取得手順は未特定 |
| 各branch energyを推定 | controlled S(s)の整数powers、失敗率配分、状態準備 | 既知QPE等。任意の得られたeigenvalueをgroundとしない |
| 剰余intervalと統計intervalを伝播 | 有効H近接・解析半径等の保証、誤差予算 | 既知区間演算。少数点fitでは代替不能 |
| weighted extrapolation・古典処理 | 上の保証を満たす二値と精度 | REDUCES_TO_KNOWN |

例えば係数から `||H−H_eff(s)||≤η(s)` を保証し、元ground gapの**保証下界** gが `g>2η(s)` を満たせば、Weylによって低いbranchの分離を維持できる。logのdomain、gap取得、測定値がそのbranchである証拠は別途必要。近似状態のresidualだけではgroundを識別しない。正確な励起固有状態でもresidualは0になる。真gap・真groundをselector入力にすることは許されない。

RSC出版版は対称PFの主要energy係数を近似stateから評価しQPE資源を予測する既知の方法を扱う。本文はperturbative estimateを厳密上界と区別し、state overlap・準備の制限を述べる。C2にとって有用な対照であるが、finite-h剰余・same-ground branchの保証をその経験的予測で置換しない。[出版本文 II/IV/F](https://pubs.rsc.org/ja-jp/content/articlepdf/2025/dd/d5dd00185d?page=search)

### 3.3 精度配分と費用

統計intervalの半幅をε1,ε2とすると、独立性を仮定せずunion boundと三角不等式で `(4ε1+ε2)/3` を制御できる。準備失敗・branch誤認・各測定失敗へ共通の総失敗率αを配分する。varianceを用いる別のpolicyは明示契約が必要。

`w=(4/3,1/3)`、各推定の費用が条件付きで `A_i/ε_i^p` に比例し、A_iを精度によらない既知単価と仮定すれば、`Σw_iε_i≤ε_stat` の最小費用は標準convex allocationで

\[
\epsilon_i\propto (A_i/w_i)^{1/(p+1)},\qquad
G_{\min}=\frac{\bigl(\sum_i A_i^{1/(p+1)}w_i^{p/(p+1)}\bigr)^{p+1}}
{\epsilon_{\rm stat}^{p}}.
\]

p=1のHeisenberg proxyでは `(Σ√(A_iw_i))²/ε_stat`。p=2のshot型と取り違えない。これは既知最適化の静的再記述で、値の計算やpilot最適化は行っていない。A_iにはnative controlled primitive、powers、overlap/成功率、confidenceを反映する必要がある。合成精度の依存・整数ceil・取得費用があれば、単純式がそのまま最適とは限らない。

| 費用成分 | 必要な接続 | 現状 |
| --- | --- | --- |
| 複数幅controlled time evolution | 同ordering/control/synthesisの単価、長時間integer powers | 値UNKNOWN |
| energy推定・統計増幅 | 外挿weights、alpha、整数schedule | 条件付き式のみ |
| branch/gap/window取得 | groundの同定と保証を真値なしで構成 | 低費用な固有手順未特定 |
| 初期state・overlap | nodeごとの準備/成功確率、再試行 | gate実費UNKNOWN |
| finite-step remainder | 有効H/log bound、解析半径または検証済み証拠 | C2固有の安い取得法なし |
| 古典後処理 | 証拠生成、配分、interval、履歴管理 | 小さな線形結合と高価な証拠取得を分離 |

公平な比較は、最良単一PF/細分化、S4、RWW Chebyshev/Richardsonへ同じgap・state・価格情報を許す。各法のbranch・alias・剰余・統計・synthesisを同じ元energy targetへ戻す。Chebyshevのfractional query/丸めの前提も、実整数scheduleと同一と扱わない。異なるPF平均signalを足すだけでは、一つのunitaryを使うcoherent phase estimationにはならない。

**C2の所見**：外挿と配分はREDUCES_TO_KNOWN。保証されたbranchと剰余を備える構成はCONDITIONAL。先行法を超える安価な情報取得・操作列・全費用改善はUNDETERMINEDで、現案に具体的な独立生成手順は特定できなかった。

## 4. C3：self-energy Schur-root

### 4.1 正確な還元とground root

元Hの選択sectorで

\[
H=\begin{pmatrix}A&B\\B^\dagger&D\end{pmatrix},\quad
L\le E_0(H)\le U<c\le\lambda_{\min}(D),\quad
\lambda_{\max}(D)\le d,\quad\|B\|\le\beta.
\]

E<cではD−EIは正定値。Schur inertiaで

\[
K(E)=A-B(D-EI)^{-1}B^\dagger,
\qquad f(E)=\lambda_{\min}(K(E))-E
\]

の符号はH−EIの最小固有値の符号と対応する。ground rootはf(E0)=0であり、区間に一意に存在する。`K'(E)=−B(D−EI)^{-2}B†≼0` なので、E2>E1なら `f(E2)≤f(E1)−(E2−E1)`。固有値の縮退があってもこの順序不等式で扱える。これらは既知Feshbach/Schur構成の静的確認である。[Theorem1.2/Corollary1.3/Proposition1.4](https://arxiv.org/pdf/2105.02058)

**各Eで必要なのはK(E)の最小energy interval**。任意のQPE出力固有値でrootを解くと励起rootを選ぶ可能性がある。active-space solverの準備・ground同定問題は消えない。近似Kのpoly切断が単調であることや任意の数値rootの正しさを自動的に仮定しない。

Lは係数norm/SOS等の有効下界、Uはcertified trial energy等で得られる。限定classではcに非負densityを落とした一体固有値和を利用できる。下界が存在することと、鋭いwindowを安く取得することは別。新たな一体/多体固有値計算は今回行っていない。`U<c` を得られない場合、原案のこの保証経路は使えない。

### 4.2 resolventの一様誤差

`m_E=(c+d)/2−E`、`X_E=(D−(c+d)I/2)/m_E`、`ρ_E=(d−c)/(d+c−2E)` とする。`0≤ρ_E<1` で

\[
R_\ell(E)=\frac1{m_E}\sum_{k=0}^{\ell}(-X_E)^k,\quad
\|(D-EI)^{-1}-R_\ell(E)\|
\le\frac{\rho_E^{\ell+1}}{m_E(1-\rho_E)}.
\]

`Δ=c−U>0`、`ρ_U=(d−c)/(d+c−2U)` により区間全体でself-energy誤差は

\[
\|K(E)-\widetilde K_\ell(E)\|
\le\frac{\beta^2\rho_U^{\ell+1}}\Delta.
\]

ηを許すには、0<ρ_U<1で右辺が非自明なら `ℓ+1≥log(β²/(Δη))/(-log ρ_U)` が十分。ρ_U=0はD=cI、β=0は結合なしとして別扱いする。弱い外部分離ではρ_U→1となる。係数未代入の標準Neumann解析であり、新しい生成則・scaling優位の証明ではない。

### 4.3 QSVT/LCU accessの費用を戻す

block-encoding scaleをα_A,α_B,α_Dとする。`D−EI` の既知LCU構成ではα_Eを例えば `α_D+|E|` と取り、normalized singular gapは `δ_E=Δ/α_E` 以下で固定する。QSVTのscaled inverseは `δ_E/(2x)` を近似するので、resolventのscaleは概ね `α_R=2/Δ`、degreeは

\[
d_{\rm inv}=O\!\left(\frac{\alpha_E}{\Delta}\log\frac1{\epsilon_{\rm enc}}\right)
\]

となる。これはencoding accessが与えられ、定理のprecision/domain条件を満たす場合のquery式。δ_E>1/2等はscaleを増やして定理domainへ入れる。`ε_enc` はnormalized encoding誤差であり、energy許容幅ηそのものではない。[QSVT Lemma40/Theorem41](https://arxiv.org/pdf/1806.01838)

block積とLCUでself-energy、Kを構成すると、標準的な選択scaleは `α_Σ=α_B²α_R`、`α_K=α_A+α_Σ`。BやDの誤差もscale付きで伝播する。例えば逆作用素のnorm誤差ε_Rだけでもself-energy誤差にβ²ε_Rを割り当てる必要がある。encoding誤差をε_Rへ直す際はα_Rを掛ける。SELECT/PREPARE、Bの二回access、Dの多項式反復、controls、adjoint/uncompute、QSP角度のfinite precisionを実gateへ換算する必要がある。[QSVT Lemma52–53](https://arxiv.org/pdf/1806.01838)、[LCU construction](https://arxiv.org/pdf/1412.4687)

`||X_E||≤ρ_E` は、費用の小さいscale=1 encodingが既にあることを意味しない。係数LCUのnormalizationがm_Eより大きければ、Neumannのdegreeだけを見た実装費用は過少評価になる。一般化学系でα_Bをβと同一に置くことも許さない。

### 4.4 term/body rank、mode削減、root費用

| 成分 | 静的な障害／必要条件 | gate等の状態 |
| --- | --- | --- |
| 明示展開 | 一外部mode・一体hopping Bなら、Dが二体のとき `B D^k B†` の最大active body rankは2k+1以下。raw term数は `O(m_B²m_D^k)` へ増え得る | 小さなDF rankの有効Hが保たれる保証なし |
| 一般二体coupling | 外部modeを含むdensity-assisted hopping等でBはactiveの三fermion演算子を含み、上の単純classと異なる。raw body rank上限は2k+3へ増える | 一外部削減のN3条件を無断適用しない |
| polynomial/encoding | 明示term展開を避けるaccessでもdegree・α_K・SELECT/PREPARE費用が残る | 実gate費用UNKNOWN |
| active/Q sector | Pは `binom(n−1,ν)`、Qは `binom(n−1,ν−1)`。rectangular Bを共通registerへ埋込む必要 | 取り除いたoccupation flag相当のworkspaceやinverse ancillaが戻り得る |
| controlled access | B/Dのoracle、phase、projection flag、QSP controls、uncompute | native/T/depth/ancilla実費UNKNOWN |
| root探索 | Kの**ground** intervalを毎候補Eで取得。正規化、準備、成功率、alphaを反復へ戻す | root countだけで優位としない |
| 誤差保証 | window/resolvent/encoding/energy推定のintervalを元HのE0へ戻す | exact resolvent式の存在と費用削減は別 |

`m_B,m_D` はそれぞれ明示項数。normal ordering・対称性・構造化評価で小さくなる特別例はあり得るが、一般fillingで多項式かつ安い有効Hamiltonianを生成する手順は現案にない。

noisy bisectionを用いるなら、`f̂(E)±χ` の符号が明確なときのみ区間を半分にする。0を含む場合、上のslope不等式から `|E−E0|≤|f̂(E)|+χ≤2χ` を用いて終了区間を構成できる。区間幅W=U−Lに対する `O(log(W/ε_E))` は必要な最小energy intervalを取得できる場合の反復数であり、solver費用を隠すものではない。単純な符号不明時retryを無限に続けない。

exact groundのactive成分には `||ψ_Q||≤(β/Δ)||ψ_P||`、したがって `||ψ_P||²≥1/(1+(β/Δ)²)` という条件付き関係がある。しかしこれは未知groundの射影normの関係であり、安い準備回路や各EのK-ground overlapを生成する方法ではない。

### 4.5 限定classと既知強対照

diagonal D・安いoccupation評価・sparse known B・広いcertified Δが揃うclassなら、可逆算術やstandard sparse/QSVT accessでresolventを扱う余地はある。しかし逆関数、coherent occupation処理、controlled implementationを含めた全費用差はUNKNOWN。既知access法へ還元できる範囲と、低gate-costの新constructorがあることは別である。

DUCCはexact external amplitudesがあればenergy-independent Hermitian effective Hを得る。SWはgap/摂動と切断条件を伴う。C3がE依存でそれらと異なるというだけで、既知Schur/Feshbachを超える算法差にはならない。trace-based exact downfoldingを同じSchur-rootと扱わない。[DUCC](https://arxiv.org/pdf/1909.06404)、[SW](https://arxiv.org/pdf/1105.0675v1)、[trace-based AppendixF](https://arxiv.org/html/2507.16916v4)

SOS/DFTHC/SOSSAは同じ元Hの低energy taskに対する別の強いbaselineである。比較には同じ係数・精度・成功率・利用できるclassical情報を与え、SOS下界、encoding norm、factorization誤差、preprocessing、gate/depth/ancilla/準備を含める。文献の別分子の速度倍率を本案へ転記しない。[DFTHC-SOS](https://arxiv.org/html/2502.15882v1)、[SOSSA](https://arxiv.org/html/2505.01528v1)

**C3の所見**：Schur-rootはREDUCES_TO_KNOWN。windowとaccessが与えられればresolvent保証・量子関数実装はCONDITIONAL。具体的な安い有効H生成、全gate/ancilla/ground準備/反復込み利益、一般分子への移送はUNDETERMINED。N3の終了済み保証成果をC3の新結果として数えない。

## 5. 最終技術境界と停止

C2・C3の現案に、既知法を超える操作列・情報取得則・生成規則・全資源の改善機構は特定できなかった。Phase A/Bの限定照合を完了し、Phase Cの[統合判断資料](qc_research_integrated_gpt_decision_handoff_v1_20261014.md)へ戻す。追加実装、候補探索、第五候補、pilot、C1/C4再調査を自動で開始しない。

GPT判断を変え得る情報は、C2では同branch/剰余/状態情報を同品質で安く取得する具体的な手順、C3では利用可能windowと低normalization/低gate/少ancillaの明示access、同情報強対照を上回る全費用根拠。それらを今回の後続作業として取得する認可はない。科学的新規性・採否・終了は未判定のまま渡す。

[既存374filesの保全](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/preservation_index.json)、旧144manifest entries/non-result fields/global description欠落FAILを維持する。N1 STOP、N2 HOLD、N3 CLOSED、THRIFT現template CLOSED、C1/C4旧STOP・結果・source・消費済み認可は不変。[公開再取得receipt](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/publication/receipt.json)の照合後Mandatory STOP。次はユーザーの別明示開始承認を受けたGPTの重要科学判断。
