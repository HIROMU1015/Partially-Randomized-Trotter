# 量子化学ground-energy算法候補：Codex技術事前調査

2026-10-13 JST。status `TECHNICAL_CANDIDATE_DOSSIER_COMPLETE_AWAITING_GPT_START_APPROVAL`。
研究目的は**元の量子化学Hamiltonianの基底エネルギー推定を、同じ精度・confidenceで低コスト化すること**。
ユーザーの最新指示により、終了したmixed-access系列の後にCodexが過去成果・一次文献を整理し、4件の具体的構成と技術的条件を用意した。候補の新規性認定・正式採択・優先順位の科学判断はGPTに残す。

今回の数学は既知のBCH、摂動、外挿、Schur補元、importance samplingを用いた条件付き設計の整理である。新しい性能実験・新規定理の認定・独立GPTレビューではない。
新分子生成、SCF/FCI、固有値計算、数値的性能評価、IR結合・相殺の登録評価、certificate再認証、compile/T合成、sampling、旧batch再実行は0。
N1停止・N2保留・N3現系列終了、THRIFT現template終了、各Trackの既存source/result/authorization/STOPを保持する。

## 1. 過去成果から残る技術的な問題

branch間の成果を一つの時系列HEADへ混ぜない。固定commit・SHAは[証拠索引](../../artifacts/qc_algorithm_candidates/2026-10-13/existing_evidence_index.json)。下表は保存文書の読取りで、新しい集計ではない。

| 系列・保存scope | 肯定的な成果 | 限界・否定的な結果 | 今回抽出する問題 |
|---|---|---|---|
| PR/PF/DFの旧H4：linear1Å/STO-3G、rank12、L_D3/12、S2、各資料のδ/q/compiler窓 | finite-RTE、phase/radius、controlled wrapper、配分・費用proxyを局所接続 | 共通配分最適化・compiler・state-prepを戻すと頑健順位未確立。C_useは経験的窓envelope | 誤差係数・gate単価・shot・状態準備を同じtaskで扱う必要 |
| Track A H6：linear1Å/STO-3G、12 modes、3α/3β sector400、DF tol1e−8/rank19、prefix5、T=.8、q1/2/4/8、r4/4/4/2、K2/4 | 独立32 trajectoriesの選択集合でPRのfull-wrapper RZ点推定が低い | 希少Taylor次数の未観測、経験的allowance、CONDITIONAL_ORACLE会計、最終ground-energy全費用未認定 | rare-tail平均費用の保証と運用情報の取得費用。有限時間complex-signal精度をenergy精度に読み替えない |
| Track B D1：3-system-qubit synthetic involution、p=(1/5,3/10,1/2)、τ5/7、state000、P5、r1/2、六方式、axisε1/200、familyα1/20、strict primitive1e−6 | 固定finite-RTE表現のnormalization・native単価のtrade-offを保存。rごとに費用差の内訳が異なる | 分子/DFではない。全wordの安いPauli取得・加法native費用は一般化できず、online classical費用・whole-circuit順位は未知 | 表現を保ったproposalと回路列費用の結合。D1の新D2比較は別契約で、今回起動しない |
| N1：合成JW4/ν2/signed DF rank2、L_Dなし、S2/T=.6、q1/2/4/8、δ=.6/.3/.15/.075、εcomplex=.04、固定44factor/178pair | decode復旧後、誤差込みselectorのexact fallbackが機能 | 全contextでexact/ungaugedを選択し、現有限domainのobjective改善なし。追加探索停止 | 安い基底変換だけでなく元H誤差と最終task費用を評価 |
| N2：固定occupation4・k≤2 signed-charge fit、T=.7、係数L1予算.0021 | 固定SのL1整合性修正を保存 | signed-charge回路はdirectより高価、追加検証は保留 | 係数fitの成立と効率的回路の成立を区別 |
| N3：非負density・固定粒子数・一外部mode、有限有理入力 | occupation方向を保持するcertificate、旧族との理論分離、有限保証幅の差 | 強いnorm/SDP/SOS対照を戻すと主研究としての有用差未確定、現系列終了 | 保証幅の差を、実際に減らす演算・推定taskへ結び付ける新しい用途が必要 |
| THRIFT型H4：linear1Å/STO-3G realRHF8mode/ν4、DF16全保持、rank10 reportingのみ、L_Dなし/PRなし、T1/4、r倍増1…1024、εU=.001 | R4のideal PF step低下を保存 | 現bitwise occupation template・κ3/6/12仮定の273 eligible frontierに正余白なし。function実費用はUNKNOWN | step削減を追加算術・control費用が回収する条件。現templateのbackend追加ではこの障壁を解決しない |

主に残るのは、(i) oracleなしでenergy誤差を予測・保証できる情報、(ii) coherent eigenphaseと平均signalの意味の違い、(iii) sequence依存の費用と統計の交換、(iv) 縮約で失うvirtual couplingを小さい表現で扱う条件、である。
これは全領域に未解決だと断定したものではない。以下の一次研究には既に対応する原理・解法があり、差は限定する必要がある。

調査中に到着したTrack A H4 geometry資料も、[固定結果文書](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)をremoteから取得して文脈へ追加した。linear H4/STO-3G/DF12、0.70/.80/.90/1.10/1.40/1.60Å、L_D0/3/4/5/6/9/12、S2、T=.8・q1/2/4/8、δ=T/q、32paired標本、complex ε=.005〜.1、ordinary controlled structured-Givens/P=0という登録条件で、PRの低費用側の点推定が報告されている。
この資料も最終energy taskや全PFとの優位は保証しない。到着した完了レビューのS4不在・B0 scalar保持対照への注意を、強いbaselineが必要な理由として記録しただけで、原稿統合・H8・新診断を今回採択/起動しない。zip内算術や全raw/projectionを再実行していない。

## 2. 一次文献調査の結果と取得範囲

検索日2026-10-13、同日までの公開資料を対象。題名・著者・版と読んだ箇所は[literature registry](../../artifacts/qc_algorithm_candidates/2026-10-13/literature_registry.json)。2026年8–9月のpreprintも確認した。網羅的優先性証明ではない。

| ID | 調査対象・確認範囲 | 今回の設計に課す既知の制約 |
|---|---|---|
| L1 | [SPRINT/GRADE, 2606.30741v1](https://arxiv.org/html/2606.30741v1)、III.2–3、V、関連付録 | norm階層、near-integrability、processing、symmetry、回転実装を既に組み合わせる。単なる「構造を使うPF」は独立差にならない |
| L2 | [BCH実用推定, 2606.30738v1](https://arxiv.org/html/2606.30738v1)、III.1–2 | diagonal BCHとcompact表現は既知。漸近式を有限step保証に無条件転用しない |
| L3 | [orbital/order cancellation, 2602.18913v1](https://arxiv.org/html/2602.18913v1)、2.5/4.3 | 異なるbasis/orderの組合せによるenergy誤差相殺は既知。低1-normだけで符号や良いorderingを選べない |
| L4 | [Mehendaleら、Digital Discovery出版版](https://pubs.rsc.org/en/content/articlepdf/2025/dd/d5dd00185d)、abstract/本文検索取得 | 近似stateでのenergy摂動推定・partition比較は既知。旧arXiv2312.13282のfirst-order版と出版版second-orderを同一視しない |
| L5 | [Chebyshev Trotter外挿](https://arxiv.org/pdf/2212.14144v4)、§2–3/Appendix Bの関連箇所 | eigenvalue外挿、gap/準備費用を扱う既存対照。外挿そのものを新法としない |
| L6 | [cost-aware stochastic simulation](https://quantum-journal.org/papers/q-2023-04-13-977/)、abstract/出版情報 | 個別演算費用を使うimportance samplingは既知。今回のpath-level構成との差は本文追加照合を要する |
| L7 | [exact downfolding, 2507.16916v4](https://arxiv.org/html/2507.16916v4)、導入/形式と適用条件の関連箇所 | 外部自由度を積分するeffective modelとcontrolled approximationは既知。これと単一固有値のSchur rootを同じ算法としない |
| L8 | [DUCC/PE active-space](https://pubs.aip.org/aip/jcp/article-abstract/151/23/234114/315571/Quantum-simulations-of-excited-states-with-active)、abstract | active-space PE・外部相関回復は既知。C3には近接する強い対照として残す |
| L9 | [DFTHC-SOS/spectrum amplification](https://arxiv.org/html/2502.15882v1)、II–IVの関連箇所 | 下界・SOS・factorizationをground-energy費用削減へ接続済み。shift/下界だけを新しい原理としない |
| L10 | [SOSSA](https://arxiv.org/abs/2505.01528v1)、abstract/書誌確認 | low-energy amplificationは既知の強い比較対象。query削減を未知block-encoding費用と混同しない |
| L11 | [PR現行v2](https://arxiv.org/abs/2503.05647v2)、abstract/版履歴 | PRによるphase estimationは既知。v2全文の独立監査はしていない |
| L12 | [PRHS, 2608.29756v1](https://arxiv.org/abs/2608.29756v1)、abstract/版履歴 | correlated-slice/quasi-probabilityは近接候補。query費用をcontrolled gate費用へ外挿しない |
| L13 | [average-error randomization, 2609.36694v1](https://arxiv.org/abs/2609.36694v1)、abstract/版履歴 | Haar平均の確率最適化はground-state保証と異なる。平均改善を特定基底stateへ適用しない |

RSC landing/HTMLは403等で直接取得できず、検索で得た出版版PDF/本文部分を使用した。L5のHTMLは404でPDFを確認。L6/L8/L10–13は全文理論の監査ではない。この制限を研究差が残った証拠にしない。

## 3. 共通の比較契約

全候補のtargetは同じ元H、粒子数sector、ground energy E0、absolute energy誤差εE、family failure α。source/input/係数精度を固定し、state-prep成功確率、phase wrap、branch誤同定を会計へ入れる。
εEはH近似・PF/外挿/縮約・回転合成・推定統計に分ける。operator誤差やcomplex-signal誤差と換算せずに比較しない。

量子費用、古典前処理/online費用、depth、workspace、state preparationは別の量として報告する。Tが1Qに含まれる場合は足してscalar scoreを作らない。異なる費用を足す場合は価格モデルを先に固定する。
各候補にはexact/full構成へ戻る選択肢を置く。真E0・真ground state・真gap・未知compiled費用で候補を選ばない。経験的幅をcertificateとしない。
PFを主対照とする場合もDF-native高次PF、既知spectral/extrapolation、SPRINT、利用条件を満たすDF/THC/SOS qubitizationを別枠で確認する。単に各primitiveをunit costで数える比較で終えない。

## 4. C1：近似stateの不確かさを入れる固定macrocycle

### 数学的構成

同じHを近似する対称kernelの有限辞書をS_j(h)=exp{−ih[H+h²D_j+O(h⁴)]}とする。basisを変えたkernelは共通frameへ戻し、変換費用を含める。
palindromeの列(j_l,a_l)についてa_l≥0、Σa_l=1を課し、V(h)=∏_l S_{j_l}(a_l h)を固定する。全kernelの一次生成子は同じHなのでBCHのh³項は

\[
D_\theta=\sum_l a_l^3D_{j_l},\qquad
H_{\rm eff,\theta}=H+h^2D_\theta+O(h^4).
\]

palindromeはlogの偶数次を消す。非縮退ground branchが追跡できる領域ではenergy主項はh²〈ψ0|Dθ|ψ0〉。これはL2–4および前の方向レビューの構成を具体化したもので、新規性認定ではない。
取得可能な近似state ρでtilde d_j=Tr(ρD_j)を得て、正当な誤差幅η_jがある場合だけ

\[
b_\theta=|\sum_l a_l^3\tilde d_{j_l}|+\sum_l a_l^3\eta_{j_l}
\]

をselectorの誤差側へ入れる。η_jの取得が未成立ならempirical designと表示し、保証selectorと呼ばない。
例えばtrace-distance下界でなく**上界**d≥||ρ−|ψ0〉〈ψ0|||1が得られればη_j=||D_j||dを使える。残差から作る場合はground branch同定とgap下界が必要で、真gapを入力しない。

### 手順と費用

1. ordering/basisが単なる共役・巡回類似でない辞書を作り、S2/S4と同じcost予算へ接続する。
2. compact BCH/normal orderingでD_jの取得費用と必要RDM次数を算定する。全stateの古典保存を要求する方法は避ける。
3. 有限palindromeと有理a_lの候補を用い、Bθと回路費用gθ、追加情報取得Qinfoを合わせた費用を比較する。現在は最適化を実行していない。
4. V(h)を固定後、controlled powersはV(h)^(2^k)。roundごとに別kernelを選ばない。

energy予算へはh²bθに、有限h log剰余、off-diagonal摂動、回転合成の幅を加える必要がある。これらを数式上定義できても、安い有効certificateを得られるとは未確認。
小step領域を十分条件hΣ||H_l||<1等で制限すればlog級数を扱えるが、粗い領域では利益を失う。phase誤差を原Hのground energyへ戻すground weightも必要。
必要費用はQinfo+Σround N_k(Cprep+2^k gθ+Cmeasure)。gθは追加basis/control/negative-timeがある場合その費用も含む。

### 自明な改善を除く条件と障害

P^−1SPは固有値不変。二項StrangのA/2–B–A/2とB/2–A–B/2もAB/BAの巡回類似でisospectralなので、反対符号辞書に数えない。
全d_jが同符号なら正係数では消せない。h²ηが残る場合は無条件のenergy四次収束ではない。単なる小step細分化は同gate予算のuniform subdivisionと比較する。
L1の純processorはoverlapには効いてもkernel固有位相を変えない。L3の相殺・L4の近似state利用を超える差は、費用を抑えた生成・不確かさ・finite branch保証の具体的接続に限って問う。

**技術状態**：構成・主要項は条件付きで閉じる。安いηとfinite-h/ground branch保証が未取得。未認証情報を無料とすれば成立を主張できない。
**最小将来検証案**：三つ以上の非可換fragmentの登録外例で、同gate数uniform subdivision・S4・既知ordering混合と比べる。近似stateを故意に悪くしたcontrolを含む。単なる二項processor例は使わない。数値実行は別承認。

## 5. C2：ground branchを確認する有限PF固有値外挿

### 構成と誤差

同じ対称kernelの固定unitary S(h)とS(h/2)で、それぞれのground-connected固有値E(h),E(h/2)を推定し、古典出力を

\[
\widehat E_{\rm ext}=\frac{4\widehat E(h/2)-\widehat E(h)}3
\]

とする。unitaryを負重みでcoherently足す方法ではなく、二つのphase推定結果の後処理である。L5の既知外挿を強い対照に置く。
同じ枝でE(h)=E0+c2h²+R(h)、|R(s)|≤Cs⁴が両stepで成立する場合、c2を知らずに二次項が消え、外挿bias≤(5/12)Ch⁴。
二点だけではCも枝同一性も認証できない。三点が同じ曲線に見えることだけで厳密保証とはしない。

各energy confidence interval I_sとfinite-h perturbation boundを用い、候補枝の同定とalias-free範囲を確認する。coeff-derived global spectral windowでphase wrapは制限できるが、ground枝をexcited枝から分離するgap/overlapの情報は別に必要。
branch certificateがない場合は明示したempirical extrapolationか未判定として残す。timeごとにmeanから別unitaryの位相を混ぜない。

### 資源構成

二つの推定幅をε_i、重みの絶対値を|w_i|とするとΣ|w_i|ε_i≤εstat。量子costが条件付きでA_i/ε_iなら、Cauchy–Schwarzから最小costは

\[
\frac{(\sum_i\sqrt{A_i|w_i|})^2}{\epsilon_{\rm stat}},
\quad \epsilon_i\propto\sqrt{A_i/|w_i|}.
\]

A_iにはstep primitive cost/h_i、ground成功確率、confidence overheadを含める。この仮定が変わる推定器へ式をそのまま使わない。state preparationとbranch確認は加算される。
classical costは二点なら小さいが、後処理精度・error widthのbit長も記録する。負重みの統計増幅を無視しない。PR-3のbackbone/variance悪化を再び隠さない。

**既知との差の候補**：native費用に基づくstep/精度配分と、真値なしのbranch確認・有限step剰余を扱う具体的実装。外挿とHeisenberg精度の原理はL5に既にあり、単なる組合せなら新規算法として採択しない。
**技術状態**：energy推定後処理は構成可能。有限step剰余とbranch情報の費用が未確定。C1と独立のbaselineにもなるが、新規性は特に重複を精査する必要がある。
**最小将来検証案**：一つの保存H・一つの対称kernel・結果前固定の二step、branch誤対応controlで止める。ground oracleをselectorから隠し、S4とChebyshev外挿にも同じ情報・費用を与える。H4/H6を今起動しない。

## 6. C3：外部modeのself-energyを残すSchur-root energy推定

### 狭い成立範囲と構成

最初の設計はN3で扱った実one-body＋非負density、一外部mode、固定粒子数νの範囲。一般二体分子Hへの成立・拡張を認定しない。
external-vacuum Pとexternal-occupied Qに対して

\[
H=\begin{pmatrix}A&B\\B^\dagger&D\end{pmatrix},\quad
K(E)=A-B(D-EI)^{-1}B^\dagger .
\]

coefficientsまたは有効なcertificateからL≤E0≤U<c≤λmin(D)、λmax(D)≤d、||B||≤βを取得する。UはAのvariational upper、Lはfull-H coefficient lowerでよく、真E0・真gapは不要。
一外部hoppingではCARからBのnormをvのEuclidean normで評価でき、非負densityの場合cには外部一体energy＋active one-bodyのν−1個の最低固有値和を利用できる。ただし同じ情報で強化した既存下界を比較に含める。

Schur congruenceでH−EIのinertiaはD−EIとK(E)−EIへ分かれる。E<cでは外部blockが正なので、ground energyはf(E)=λmin(K(E))−Eの零点である。
K'(E)=−B(D−EI)^−2B†≼0より、fは少なくとも傾き−1で減少し、上記window内のground rootは一意。これは標準Schur/Feshbach原理の条件付き帰結で、N3を新規算法と再認定するものではない。

### 係数で閉じるresolvent近似

m_E=(c+d)/2−E、X_E=(D−(c+d)I/2)/m_E、ρ_E=(d−c)/(d+c−2E)<1と置き、

\[
R_\ell(E)=\frac1{m_E}\sum_{k=0}^{\ell}(-X_E)^k,
\quad \|(D-EI)^{-1}-R_\ell(E)\|
\le \frac{\rho_E^{\ell+1}}{m_E(1-\rho_E)}.
\]

Ktilde=A−BR_lB†はHermitian、self-energy誤差幅は右辺×β²。E≤Uでworst-case幅を固定できる。
Ktildeの**最小固有値**をenergy推定し、self-energy/PF/統計幅を加えたfのinterval符号だけでbracketを更新する。符号不明ならbranch/budgetを未判定として止める。元Hのground eigensolveでrootを選ばない。
一様operator幅ηと最小固有値推定幅εがあれば、fの単調性からrootのenergy幅はη+εとbracket幅へ戻せる。ただしKtildeのrootを無条件に一意としない。各Eでexact fをencloseする方式を用いる。

### 回路・費用と重大な障害

未知resolvent oracleを与えず、A/B/Dのblock-encodingまたは短い演算子表現、R_lのLCU/QSVT、coef精度・controlled access・uncomputeを設計する必要がある。
query費用は少なくともlに依存するD accessと二つのB access、normalization、各bracket点のenergy推定、state prepを含む。P/Qの粒子数が異なるためBによるsector間写像の補助制御も必要。
poly(D)を直接fermionic termへ展開するとbody rankやterm数が増え、指数的に重くなり得る。外部mode一つを削除してもQSVT ancillaやwalk費用が上回ることがある。ρ_U≈1なら次数も大きくなり、下界の正しさだけでは有利にならない。
L7/L8のdownfolding、DUCC、Schrieffer–Wolff、N3強対照との比較が必要。L9/L10のSOS下界活用も既知。差があるとすれば、explicit coefficient windowと元H誤差幅を保つ小さいself-energy回路の生成法であり、現時点では未実装・費用未証明。

**技術状態**：限定classでroot/近似誤差の形式は閉じる。効率的controlled実装・state準備・元分子への適用が未成立。N3の現benchmarkは再開しない。
**最小将来検証案**：一外部modeの小さい有理係数例で、bare active A・既知Schur近似・poly self-energyを同window/誤差で比較し、ρが悪いcontrolも含める。先にbody rank/normalization/ancillaの静的上限を出し、fullより重ければ数値勝利を追わない。別承認で新scopeを固定する。

## 7. C4：回路列の費用を扱うfinite-RTE proposal生成

### meanを変えない構成

固定finite approximationをF=Σ_γ a_γU_γとする。γはwhole path、係数phaseはcontrolled回路へ保持し、以下では|a_γ|を使う。
q_γ>0でdrawし、Hadamard出力Y∈{−1,+1}へsigned/complex weight a_γ/q_γを掛けると、meanは元Fのsignal。変更するのはproposalで、event演算子・Taylor bias・元H・normalization意味論を変えない。
g_γ>0をcontrolled whole path＋state-prep/readoutの特定資源費用とし、二次moment上界M(q)=Σ|a_γ|²/q_γ、期待costG(q)=Σq_γg_γを使う。

\[
M(q)G(q)\ge(\sum_\gamma|a_\gamma|\sqrt{g_\gamma})^2,
\quad q_\gamma^*\propto |a_\gamma|/\sqrt{g_\gamma}.
\]

これは既知のcost-aware importance原理のCauchy–Schwarz帰結で、新規定理ではない。改善候補は、全pathを列挙せず、sequence依存のgを入れる生成器を作れる限定classにある。
この式だけでD1のBernstein十分shot数が減るとは言えない。max|a/q|のrange項、finite-bit rounding、resource-tail、零drawのclassical費用も再評価する別契約が必要。

### 生成法の具体案と条件

費用が状態sを持つ有限transition表でgγ=g0+Σ_t c(s_t,e_t)と正確に表せるclassをまず限定する。全native IRのinverse stackを小さいsで表せるとは仮定しない。
係数絶対値|aγ|も、取得可能な非負transition weightsの積と終端値へ因子化できることを要求する。wordを集約した後の|Σa|がこの形になるとは限らず、符号相殺を勝手に捨てない。
1/sqrt(g)=(1/sqrtπ)∫_0^∞u^−1/2 exp(−ug)duをpositive有限quadratureで近似する。各uではexp(−ugγ)がtransition因子へ分かれ、backward DPでsubtree massを計算できる。
positive有理quadrature係数・有理transition近似を用いて、実際に生成するlawを完全に固定する。ideal q*への誤差は性能の問題であり、実lawをweightへ使用すればmeanの恒等式を維持できる。係数・phase近似は別のbiasである。
whole-path defensive mixture q=(1−ζ)q_DP+ζq0、0<ζ≤1を置けばq≥ζq0でsupport/rangeを保護できる。独立stepごとのmixingを同じwhole-path保証と呼ばない。
有限状態S・quadrature J・steps rならdense transition DPの計算はO(JrS²)、保存はO(JrS)。ただしSがword長とともに指数増大なら、この利点はない。実moment/rangeの安いcertificationも別途必要。

### 既存法との差、回路意味論、障害

L6、SCU/CTS、D1の六表現、L12のcorrelated slicesが対照。proposal変更とoperator channel変更を混ぜず、finite-RTE平均振幅をrandom-unitary channelのboundで保証しない。
controlled scalar phase、draw先行、zero-fill、uncomputeを保持する。compileや新angle費用を無料oracleにしない。full enumeration/optimal qは診断対照で、運用selector情報には使わない。
generic Gaussian DFではbasis共有・global synthesis・inverse cancellationの状態が大きい可能性がある。元D1のexact adjacent passはstackを使うため、定数サイズinterfaceへ縮約できる保証はない。
gが少数のkey価格へ依存するだけなら、既知importanceの再記述に留まる。状態空間圧縮または元情報contractで取得可能な厳密cost表が独立差の必要条件である。R3の一般best-arm探索を名前だけ変えて再開しない。

**技術状態**：mean/optimal-law/DPは限定classの条件付き構成。一般DF circuitと安いmoment/range保証への接続は未成立。online classical費用UNKNOWN、実装利益未測定。
**最小将来検証案**：小さいfinite-state event列で、q0・termwise cost law・全path oracle・DP/defensive lawを同mean/精度で比較。stateが増大する反例を含める。D1保存r2の新IR評価を今回実行せず、採択後に必要性を判断する。

## 8. 4件の技術比較と未解決事項

| ID | 直接減らす対象 | 原理として既知 | 独立差が残る可能性を調べる部分 | 主な未解決点 |
|---|---|---|---|---|
| C1 | 同energy biasに必要なPF step/gate | BCH、ordering/basis相殺、state期待値 | 安い状態幅＋固有位相が異なるkernel辞書＋native costの生成規則 | η/finite-h/branch情報が高価・辞書符号が揃う |
| C2 | 単一高精度PFのcoherent step負担 | Richardson/Chebyshev eigenvalue外挿 | 実費用配分と真値なしbranch/剰余確認 | 既存L5との重複、準備・精度増幅 |
| C3 | 明示的外部自由度を扱うenergy task | Schur/Feshbach、downfolding、resolvent polynomial | 係数windowを保つ安いself-energy constructor | 高body rank、QSVT normalization/ancilla、元分子への適用 |
| C4 | shot×whole-path回路費用 | cost-aware importance/DP/Laplace | sequence costの有限状態圧縮とfinite-bit保証 | generic IR stack増大、range項、online古典費用 |

4件は探索対象の上限で、四つのGO判定ではない。独立新規法として成立すると認定した候補は0。C2は既知手法の強い対照にもなり、C3/C4は適用classが狭い。既知原理の寄せ集めである可能性を残してGPTへ渡す。
今回は優先順位を採択しない。GPTは、未解決の差を証明・構成できる見込み、必要情報の取得費用、元ground-energy taskへの接続と論文化claimを比較する。

## 9. 再利用と次の検証準備の境界

読み取った既存APIとfixed sourceは[再利用索引](../../artifacts/qc_algorithm_candidates/2026-10-13/reuse_and_future_validation.json)。import/実行はしていない。
C1/C2ではPhysicalSector、DF-native S2/S4、controlled Hadamard wrapper、既存normal orderingを再利用できる候補。ただしBCH/RDM constructorとbranch certificateは新規準備が必要。
C3のN3有理PSD checkerは取得したc等の照合に利用できる候補で、self-energy回路や量子最小固有値selectorの実装ではない。
C4のD1 law/moment/IR/sequence源は別commitにある。現baseに存在しないmoduleを実装済みとして扱わず、新sourceへ取り込む場合のAPI/情報契約を別途固定する。

次の数値検証は候補採択後に最小scopeへ固定する。今は実行planではなく比較用案であり、科学source/input SHA、精度、call-count、memory/output/log caps、独立checkerを備えた新契約が揃うまでlaunch不可。
既存timingではTrack A compile、D1ではguard/serializationも負担になり得る。将来run前に実profileを確認し、sector/matrix-free/cache/streamingを使い、独立caseをthread制限付きprocessで並列化する。CPU32・host RAM約62.5GiBという今回のread-only inventoryは、専有資源や将来空きRAMの保証ではない。GPU/runtimeは今回未検査。
新契約にphase/total wall capを既定追加しない。memory/output/log/call-count/instruction limitsは保持し、worker合計RSSとBLAS/OpenMP/Numba threadsを調整する。数式上の効率と実backend speedupを同一視しない。

## 10. GPTへの判断事項とMandatory STOP

次のGPTレビューの入力は、本資料、literature registry、四候補card、固定過去証拠。開始には別途明示承認が必要。
問いは、(1) 差が既知の直接帰結を超えるか、(2) 狭いclassにも研究価値があるか、(3) 情報・回路費用の障害を解く具体的構成が必要か、(4) どの候補を一つの最小検証へ進めるか、である。
全部が弱ければ採択0でもよい。成功候補が出るまで同型探索を延長しない。今回の終了地点は4件の具体化・技術条件・比較資料・公開確認が揃った段階。

**重要な更新**：mixed-access closure文書の「次はGPT」は終了記録時点の履歴。最新ユーザー指示で、GPT開始前の本Codex技術調査を追加した。旧closure/STOP/source/resultを改変せず、この後続記録へ進行を反映する。
IDEのActive file/Open tabsを対象・認可に使わず、明示された今回の研究全体の事前調査scopeに従った。
現時点で新しい研究方針は未採択、GPT科学レビュー未開始、科学計算未認可、Mandatory STOP。


## 11. 公開・取得確認と今回の作業完了

技術資料payloadは`5caf8de4c45f120ce0ef5a8f96f05a5ed5cc8ad8`、branch `qc-algorithm-candidates-prevalidation-20261013`。
[固定commitの候補比較資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5caf8de4c45f120ce0ef5a8f96f05a5ed5cc8ad8/docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md)、[取得receipt](../../artifacts/qc_algorithm_candidates/2026-10-13/publication/receipt.json)。
独立GitHub取得（shallow bare cache、local alternatesなし）で16変更blobs/1,160,390bytes、旧229files、旧128manifest entries/non-result fields、14追加relative linksを照合した。過去15固定資料は独立取得でSHA一致（H4結果文書はbounded raw取得）。
これは文書・source/evidenceの取得性確認で、外部科学再現・数値性能結果・immutable CIではない。
rootの既存127filesを保持し、別作業中に到着したH4レビュー/zipはstage・変更していない。旧global description欠落FAILも保持する。
四候補の技術調査と比較資料作成は完了し、探索を停止。新規性・研究採否・優先順位のGPT判断は開始承認待ち。科学計算・compileは別の固定scope認可を必要とし、Mandatory STOP。
