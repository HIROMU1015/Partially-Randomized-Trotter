# DF少数軌道・希薄三次補正付きPR：独立監査

2026-10-11。利用者の限定監査認可による技術証拠。科学的採択・次実行認可ではない。

## 結果と入力identity

数学的反例は今回の導出・登録したsynthetic検査では発見しなかった。BCHと理想sampler平均は成立する。
最大6軌道という支持の主張は、各primitiveがnumber-conserving Gaussian基底での高々2軌道のparity積である場合に成立する。
controlled局所回路は実装できたが、汎用分解の費用は大きい。production用の薄い基底変換、認証付き確率・軌道縮約、T合成は未実装／未評価。
先行法との具体的な生成手順差はあるが、独立新規性・総資源優位は確定していない。mandatory STOP、GPTへ返す。

| Phase | 判定 | 証拠と限界 |
|---|---|---|
| A | 独立検証PASS | exact rational free-word log、1〜5 blocks、24 checks。帰納証明とsource順序照合 |
| B | 条件付きPASS／production DFアクセス未解決 | even CAR支持の証明、非直交rank6を含む12 checks。数値rank落としを認可しない |
| C | 理想数学PASS／digital実装は条件付き | 平均、有限step境界、階層確率、退化、正時間。negative time未証明 |
| D | 条件付きPASS | 有界固定入力、独立draw、中央P5以上、digital/native誤差予算が必要。C/D合計23 checks |
| E | 小規模実装PASS／native T費用未解決 | source bridge＋5代表compile、18 checks。安価な6軌道実装という証拠ではない |
| F | 指定一次文献の構成比較完了／独立新規性未解決 | [claim比較](df_sparse_correction_prior_art_comparison_20261011.md)。同一DF access・同一誤差budgetでの強いbaseline費用比較は未実施 |

入力はbase `22a0d032717b4139cdfa806850643e93ba25624f`から独立worktreeへ固定し、監査開始前にcommit
`ae558cfbbcf6e222f1c84afc59ba533476a12384`をpush、remoteからMD/ZIP/intakeを取得して完全一致を確認した。

| 原本 | bytes | SHA256 |
|---|---:|---|
| [設計書](../../research/track_b_df_sparse_correction_design_review_20261011.md) | 41,234 | `e6cec20eb2e4abe4f00d0e62fd05d28180132888857cd110edb368b2d4ba1c03` |
| [ZIP](../../../artifacts/track_b_df_sparse_correction_audit/2026-10-11/inputs/track_b_df_sparse_correction_design_20261011_reproducibility.zip) | 34,941 | `75851dff8fb1f7851ad48b0dcd31853cb650bde148b59d54825b798193dca29b` |

ZIPの8 memberはCRC・内部manifest byte/hash照合PASS、path traversal/symlink/encryptionなし、外部MDとarchived MDはbyte一致。
付属コード・最終検算・初期→最終diff・execution historyを読んだ。GPT自己検算は独立証拠に数えていない。
GPT履歴にwrong-sign対照とweak-R有限grid slopeの検査修正がある。P3残差をexact四次係数へ変更した履歴も保持した。
付属scriptの実行／importは0。Codexの別実装はformal logとCAR actionで検証した。

## A：符号・順序・再帰

`src/trotterlib/df_partial_s2.py`のbuildはordered DF half-sweep→central tail→逆half-sweep。
operatorとして回路appendの順序は反転するがpalindromeは同じ並びになる。block内部は同じGaussian基底で可換、identity phaseはcontrol ancillaに保持する。
順序を変更するとK3が変わるため、blockのソート変更は本監査の適用外。

一般のnoncommuting X,Yについて

\[
\log(e^{X/2}e^Ye^{X/2})=X+Y-\frac{[X,[X,Y]]}{24}-\frac{[Y,[X,Y]]}{12}+O(5).
\]

X=−iδHj、Y=−iδBjを代入すると候補の−iδ³K3となる。
内側のpalindromeの三次誤差はこの三次項に加算される。これを内側から帰納すれば、各suffix Bjに対する候補の総和が得られる。
形式的な対称性S2(−δ)=S2(δ)⁻¹でlogの偶数次数が消える。
C=exp(+iδ³K3/2)を左右に置けばlogの三次が相殺され、C(−δ)=C(δ)⁻¹も保たれる。
新しい五次係数の小ささまでは証明しない。

exact free-word検査は有理数で次数5までlog(I+X)を計算し、次数4までcorrected stepがidealと一致した。
逆符号correctorでは三次残差が倍になるnegative controlも通った。
可換quotient・定数identity・pure-D内部非可換をfocused testsで別途確認した。R=0でもD内部K3は一般に非zero。
中央をP3とすると−δ⁴R⁴/24が残る。これはcandidateへの反例ではなく、設計書が除外しているP3の既知障害である。
`rte.py`の`finite_taylor_operator`はeven cutoff Kに対してdegree K+1までを含む。cutoff4はP5、cutoff2はP3である。
synthetic operatorでsource P5と独立Taylorを照合した。RTE単独の平均operator、正規化B5、実sample unitaryは別の量。

## B：支持とsourceへの橋渡し

各Qa=σa∏(I−2nv)のorbital columnsはprimitive内では直交、primitive間では直交不要。
三つのcolumn集合のspan Sのdim rは6以下。各QaはS上のeven CAR algebraに入り、その積・交換子も閉じる。
Sを先頭r modeへ写すnumber-conserving Gaussian Gを用いれば、spectator上にidentityを残してTがliftされる。
JWのlittle endianでは行列表示はG(I_spectator⊗Tlocal)G†である。fermionic gradingの符号はeven性によってspectatorに残らない。
これはDF Hamiltonian全体が6 modeであるという主張ではない。

独立Fock実装はbit-string JW actionとone-particle logのsecond quantizationを使用した（GPTのminor実装とは別）。
n=7/r=6、n=5/r=3、n=4/r=2で非直交・identity・degenerate span・符号を検査。r=6では64×64中央行列、最大particle sectorは20次元。
rank6を勝手に数値thresholdで落とさず、fixtureの構成からspanを固定した。

source bridgeはsynthetic n=4 complex Uを`U_to_qiskit_ops_jw_givens`に与え、adjacent Givensの局所1粒子blockからWを回収した。
必要な局所gate行列は2×2/4×4、全Fock行列を使わない軌道取得が可能。one-particle作用WはUと残差8.39e−16で一致し、U.T/conjugate/daggerとは不一致。
source registryで得たGによるZ/ZZ/identityと、回収columnによるoccupation parityをsmall Fockで照合（残差8.78e−16以下）。
negative DF λ・ηを含むI/Z/ZZ展開も照合した。vacuum phaseはG†UGで打ち消すが、central controlled phaseは捨てない。

このaccess経路はsource-generated adjacent Gaussian opsに限定する。任意のbasis registryはGaussian性を保証しない。
RY、number-preservingだが非GaussianなCZは拒否するcontrolを保存した。
一般nonadjacent gate sequenceのlocal-block判定だけではspectator parityを認証できない。
`DFBlock`はeigenvectorを直接保持しないため、既存U_opsの検証済み作用から取得するか、元Gを別途diagonalizeして向きのidentityを照合する必要がある。
productionのinterval rank・near-degenerate stability・係数取得費用・thin n×r compilerは未実装。
現在のsource APIはsquare Uを受ける。既知O(nr)部分基底変換を本repositoryで検証済みと扱わない。

## C：平均と有限step境界の独立導出

h≥0は符号をHermitian involution Qへ吸収した規約。type Iは(a,b∈Ij,c∈suffix)/24、type IIは(a,c∈suffix,b∈Ij)/12。
したがってc≥0、A=Σj(λj²ℓj/24+λjℓj²/12)。typeとjのmassを選び、各hに比例して3 indexを独立選択すればqτ=cτ/A。
prefix構築O(M)、index query O(log M)のarithmetic模型。DF/係数取得、bit精度、rank処理、局所行列の費用は別途必要。
全tuple列挙は検算のsmall fixtureだけで行い、production samplerは実装していない。

κτは**証明済み**norm upper≤4を必要とする。本監査は普遍κ=4を使い、floating normを証明に流用しない。
A=0ならcorrector=I。T=0の判明tupleはκ=0としてidentityへ戻せるが、近似norm zeroでdropしない。
B=4A、Λ3=Σcκ≤B、pτ=ηcκ/B、pI=1−Σpτ≥0。identity選択時も元PRを実行し、postselectionしない。

\[
\bar C=I+\sum_\tau p_\tau(e^{i\theta T_\tau/\kappa_\tau}-I),\qquad
\sum_\tau p_\tau\frac{i\theta T_\tau}{\kappa_\tau}=\frac{i\delta^3K_3}{2}.
\]

Hermitian generatorに対するintegral remainderは∥exp(iZ)−I−iZ∥≤∥Z∥²/2。
各sampleへの適用からremainder≤ηΛ3θ²/(2B)=δ⁶BΛ3/(8η)。
ideal Cの一次近似との差≤δ⁶∥K3∥²/8を加えると、指定の有限step境界そのものが得られる。
固定γ>0、δ>0、δ→0ではη=γδ、θ=Bδ²/(2γ)でlocal O(δ5)。saturation η=1でも境界は有効だが、大stepが高精度になる保証はない。
tiny ηの実装はunsaturated formulaで0/0を避ける必要がある。80桁の小δをexact rationalで検算した。

finite-bit確率をp̃で置換するbiasは
Στ|p̃τ−pτ|min(2,|θ|)以下（identityとの差を使う）。
別途unitary error≤Στp̃τετを加算する。これは有限bit samplerを実装・認証したという意味ではない。
independent left/right/steps/central drawでのみ積のfirst momentがfactorizeする。同じdrawを左右で使う反例controlをtestsに保存した。

追加corrector ensembleの正規化は1、weight倍率も1。しかし元RTE weight Γは残り、shots=1を意味しない。
またbar Cはexact unitaryでなく近似平均であり、λ=1でexact correctionを実現したと主張しない。
weighted Hadamard信号ではE[WVsample]を使う。E[VρV†]をそのfirst momentへ読み替えない。
P3→P5、δ/step数変更後のRTE Γ・confidence taskは別途比較が必要。

## D：有限時間と六つの誤差台帳

固定有界Hermitian blocks、固定γ>0、正step、理想independent draws、exact coefficients、中央P5以上が条件。
x=δΣ∥Hj∥の代わりにsource-accessible Σh bound、y=Bδ³を使える。
formal cancellation後の全系matrixを作らないmajorantは

\[
b_{sym}=2\left(e^x-\sum_{k=0}^4x^k/k!\right)
+y(e^x-1-x)+e^x(e^y-1-y).
\]

S2とidealの次数≥5のtailを各e^xで押さえ、corrector一次×S2次数≥2、corrector二次以上を順に押さえた式。
eCを上記有限sampler境界、eRを中央finite RTE平均誤差とすればlocal b=b_sym+2eC+eR。
bar Cはconvex unitary平均でnorm≤1なので2eCの係数に追加amplificationは不要。
中央s microsteps、g=e^(λRδ/s)(λRδ/s)^6/720ではeR≤sg(1+g)^(s−1)。
各step norm≤1+bよりglobal≤(1+b)^n−1≤nb e^(nb)。t=nδ固定、b=O(δ5)ならO(tδ4)。
このformal boundは保守的で、finiteδで実用的なmarginや有利なstep数を保証しない。

| 台帳 | 扱い |
|---|---|
| PF | b_sym。固定順序、D内部とcutの両方 |
| finite RTE | eR。cutoff4/P5条件、元normalization保持 |
| corrector平均 | 2eC。exact probability・generatorの理想境界 |
| finite-bit sampling | 確率距離×unitary-I、係数roundingやrank誤差は取得誤差へ |
| native synthesis | weighted meanの各gate norm error、strict controlled phase。Γを落とさない |
| DF/input threshold | HからH̃への差。少なくともt∥H−H̃∥を別計上、既存thresholdを再最適化しない |

sampleごとのoperatorは4次PFではない。物理channelのdiamond norm certificateやRPE/QPE energy精度は本監査から出ない。
負時間・γがδに依存する再設計・相関draw・P3中央はこのorder証明の適用外。

## E：実際の限定compileと障害

固定synthetic seed `7101107`、θ=1/32、κ=4、3/6 active modes、各Qはcomplex rank2 parity。
Qiskit1.3.0、all-to-all、basis rz/sx/x/cx、optimization0、transpiler seed20261011。
existing `transpile_and_measure_cost`を使用。探索・compiler最適化・native catalogue構築は0。
controlled centerはdiag(I,U)をexplicitに構成し、ancillaをMSBへ置いた。embeddingはuncontrolled G†→controlled center→G。
global phaseを含むOperatorのstrict normで照合し、projective equivalenceを使っていない。

| active r | control | CX | RZ | 1Q合計 | depth | strict残差 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | no | 20 | 120 | 200 | 125 | 9.26e−15 |
| 3 | yes | 90 | 453 | 755 | 549 | 1.43e−14 |
| 6 | no | 1,868 | 11,256 | 18,760 | 13,981 | 2.13e−13 |
| 6 | yes | 5,299 | 27,153 | 45,255 | 35,310 | 5.08e−13 |

4mode full G†/controlled center/G代表はCX399、RZ2069、SX1302、depth2465、strict残差5.58e−14。
center4例＋embedding1例＝compile5 calls。workspace0、controlled ancilla1。state preparation/readout/routing/元RTEは含めていない。
countsはこの汎用分解のscreeningであり、最小cost、lower bound、DF-native全体のcostではない。
compileのunitarity/phase誤差はdouble precision検査でありinterval certificateではない。

**Clifford+T synthesisは0、T countは未解決。** rz/sx/xとcountされたRZをTと呼ばない。
T費用は固定target precisionで各RZのε配分とstrict phase-preserving synthesisを行わなければ数値化できない。
形式的にはΣl T_RZ(φl,εl)＋必要なphase cost、誤差≤Σlεl。controlled strict phaseを無視する合成は許されない。
particle-sector分解を使う将来実装やknown corrector compilationで安くなる可能性はあるが、今回未評価で、次実行も未認可。

補正slotは2n。accepted request probability p=ηΛ3/Bで独立ならJ~Binomial(2n,p)、E[J]=2nηΛ3/B≤2tγ（small-step域）。
κ=4でzero Tをまだ証明していない場合、これはrequest数の上限側会計で、実非identity回転数を過大に含む。
outer coin通過後のlocal生成requestは2nη≤2tγ、worst hard capは2n。
P(J≥k)≤inf_{s>0}exp(−sk)(1−p+pe^s)^(2n)。実行tail capでdropする場合は追加bias/confidenceを予算化する必要がある。
sampleごとのcost ceiling、prep/readout、original RTEのΓ・required shots込みをまだ比較していない。定数頻度だけで総資源優位とはしない。

## 検査と実行範囲

77独立check（A24+B12+CD23+E18）と15 focused pytest controls PASS。
Python3.11.0rc1/NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0、既存 `.venv311`、BLAS/OpenMP/Rayon1 thread、32 CPU affinity。
小行列主体のworkloadでprocess poolを使わず、独立phaseを分けてmemoryを解放。GPU query/use0、wall capなし。
各phase wallは0.707/0.134/1.249/15.824s、最大観測RSS557.465MiB。監査の1536MiB post-check guardを超過しなかった。
このguardはallocation中のhard terminationを保証しない。G10の512MiB contractを変更／再利用していない。
native scriptは初回に存在しないAPI名をimportして起動前失敗した。合成・検査開始前のharness typoを正しい既存API名へ修正して開始した履歴を保存。
数学的反例、旧one-shotのretry、science source修正ではない。

本番runner、新規分子/NPZ/DF input、LP、GPU、quantum shot/trajectory、full test suite、parameter optimizationは0。
library APIは変更なし。原input、旧G10/RA-D0結果・source・authorization・marker・STOPは保持。
記録は [artifact directory](../../../artifacts/track_b_df_sparse_correction_audit/2026-10-11/)、scripts/testはindexから辿れる。

## GPTへ返す未決事項

production DF access/rank/digital sampler、precision付きT合成と六つのerror budgetが未固定。
generic centerの大きなCX/RZ負担を払っても独立の価値があるか、同じDF backendを与えたNCC/HNCC/known corrector/sparsificationとの差が残るかは未解決。
この監査は限定理想構成と代表実装を支持する技術証拠。algorithm採択、科学的GO、次の検証の必要性・範囲はGPT/利用者が判断する。
