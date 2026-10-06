# RA-RTE数学設計の統合技術監査 v1

2026-10-06 JST。GPTの[設計書原文](inputs/ra_rte_mathematical_design_after_r1p5_20261006_user_input.md)と
[利用者添付summary](inputs/ra_rte_mathematical_design_user_summary_20261006.txt)をbyte-exactで保持して監査した。
原文・研究方針を変更せず、採用可能な数学と必要な仮定修正を返す。
**DOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。科学実行・合成・solver・R1資源再採点は0。**

基点はR1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942`。
設計書SHA256は`4d8e448fa7d2386a545fdc7b9204a4738b917605bb4a7c240dd07f0c196b6826`、
bookkeeping script SHA256は`c6aa6cbc2848921080fae154f85fc766e5e70f7763834e56720492b26305d56e`。
R0/R0.5/R1/R1.5の既存分類、source、authorization、result、marker、Track Aを保持する。
証拠は[manifest](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)に記録する。

## 判定

**理想的一block・固定有限表・canonical samplingの固定n LPはPROVED_UNDER_ASSUMPTIONS。**
新規性、実資源改善、実装承認、R2 authorizationはUNRESOLVED／未認可のまま。
無条件のcap保存grid保証には反例がある。実行contract前に次の四点を明文化する。

1. gridのfactor-r資源近似と、固定total-resource capのfeasibilityを分ける。
2. `ell / kappa_n`もoutward上界で認証し、実際のq/y・sampler lawの丸めを再検証する。
3. 二角度quotientは`0 < Delta < pi`で使い、Delta=0は同一columnへ直接配分する。
4. workspace上限はpeak必要量で扱う。期待workspaceを容量制約へ転記しない。

方針の全面再設計は技術上要求しない。この修正付きmodelを採用するか、次の実装・pilotが必要かはGPT判断。

## 1. 命題別readout

| ID | 命題 | 判定と必要な仮定 |
|---|---|---|
| M1 | mixed-column finite mean、phase、negative time、終端 | PROVED_UNDER_ASSUMPTIONS：同じpの独立indices、順序固定、scalar phase保持、terminalはphi=0 |
| M2 | 元one-angle familyとの区別、ordinary/A/PTSC包含、B_star下界 | PROVED_UNDER_ASSUMPTIONS：実際の端点columnsと同じ実装を表に含む。mixed anglesは別class |
| M3 | 二角度matchingとsec normalization倍率 | PROVED_UNDER_ASSUMPTIONS：c>=0、0<Delta<pi、bracketing。Delta=0 quotientはCOUNTEREXAMPLE（未定義） |
| M4 | q/yの一対一対応、kappa、固定n LP | PROVED_UNDER_ASSUMPTIONS：e>0、d>=0、n>=1、0<alpha<1、固定D/t/d/C、exact real arithmetic |
| M5 | numerical residual／interval LP | PROVED_UNDER_ASSUMPTIONS：qは実装lawと一致、sum q=1、y>0、outward bounds。log/root認証・sampler条件は実装contractで未固定 |
| M6 | dual／疎解／shot grid／outer bound | dual・疎解はPROVED_UNDER_ASSUMPTIONS。gridのcore近似はPROVED。固定cap保存への無条件拡張はCOUNTEREXAMPLE |
| M7 | fixed-ensemble ISとjoint SOCP | PROVED_UNDER_ASSUMPTIONS：positive supportまたは閉包規約、fixed n／range cap／bias budget、固定cost/error |
| M8 | adjoint-paired controlled formと誤差 | PROVED_UNDER_ASSUMPTIONS：同じ実装列のexact adjointとexact CX、phase-preserving comparison。新native実装は未実施 |
| M9 | multi-block mean／channel伝播 | PROVED_UNDER_ASSUMPTIONS：mean式はexact controlled formとideal contractions、channel式はdiamond normと独立抽出 |
| M10 | 既知法とのclaim差 | 既知部品を確認。LP統合を独立論文貢献とできるか、優先性、取得cost差、実益はUNRESOLVED |

PROVEDは以下の一般的導出を指す。50件の有限bookkeepingだけから一般定理を外挿していない。

## 2. Meanとclass：M1–M3

Hermitian involution Qに対しexp(-i sigma phi Q)=cos(phi)I-i sigma sin(phi)Q。
独立indicesなら、非可換性を仮定せず

\[
\mathbb E[Q_k\cdots Q_1]=\widehat R^k,\qquad
\mathbb E[V_j]=\cos\phi_j F_{k_j}+\sin\phi_j F_{k_j+1}.
\]

従ってDw=tはformal degree matchingとして十分。Qに特別な代数関係がある場合の必要条件ではない。
terminal phi=0でF_(m+1)を作らず、sigmaの両符号に同じ式が成立する。
rotationを非可換word内で移動するbranch identityは使用していない。

元familyの非零c_kに対応する角度・precision・loweringを表に置けばw_j=c_kで再現できる。
複数の異なる角度へ同じdegreeの正weightを置くmixed解は元の一角度制約を満たさない。
そのclassでの最適性を旧classの最適性と呼ばない。

columnをeven kで(cos,sin)、odd kで(sin,cos)のunit vectorとして写す。
sum w_j vector=(E_d,O_d)なのでtriangle inequalityでB>=sqrt(E_d^2+O_d^2)。
A columnsを含むなら旧R0のattaining解があり、normalization optimumはB_starのまま。
resource optimumへの含意はない。

二方向分解は2x2係数系の逆行列、det=sin(Delta)から原文のlambdaを得る。
sum lambda/c=cos(phi-mid)/cos(Delta/2)<=sec(Delta/2)。
ただし原文の許容記述phi_-<=phi<=phi_+にはDelta=0も含まれる。
phi_-=phi=phi_+=0では0/0となるため、同一方向の一columnをcで使う分岐が必要。
このnorm倍率からnative cost／bias／G倍率は導かない。

## 3. 固定nのLPとdual：M4・M6

q=w/B、y=1/BによりDq=yt、sum q=1。
s=e-d^T w、h=s/B=ey-d^Tqなので

\[
n\ge\ell(2+4h/3)/h^2
\quad\Longleftrightarrow\quad
nh^2-(4\ell/3)h-2\ell\ge0,
\]

h>0で正根kappa_n以上という原文の条件になる。
d>=0、e>0、kappa_n>0からy>0。逆写像w=q/yが存在し、sum w=1/y。
したがって**元の指定Bernstein十分条件との同値性**は成立する。
真の最小shotsとの同値性ではない。fixed nではmin C^Tqとlinear budgetsがLPになる。

物理的なcolumnに対しsum_r D_rj=cos(phi_j)+sin(phi_j)は[1,sqrt(2)]。
exact modelでは1/sum t<=y<=sqrt(2)/sum tとなり、q simplexと合わせて有限なfeasible domainが得られる。
LP最適性の主張は固定表・固定error upper・固定cost modelに限る。

dualは原文どおり

\[
\max\zeta+\lambda\kappa_n,\quad
D_j^Tu+\zeta\le C_j+\lambda d_j,\quad t^Tu\ge\lambda e,\quad\lambda\ge0.
\]

q/yの非負性からweak dualityが直接成立する。u/ zetaはfree。
人工表ではprimal=dual=10/3のexact certificateを照合した。solverによる探索はしていない。
追加制約やrobust variablesを使う場合はこのbasic dualをそのまま証明書にしない。

独立equalitiesは高々m+2、confidenceと追加r制約のslacksを含むBFSでpositive変数は高々m+3+r。
y>0が一つを占めるため、q support<=m+2+rの最適基本解を選べる。
これは最適極点の存在下の上界で、すべての最適解やsolver出力の疎性保証ではない。

## 4. Shot gridの修正と反例

nを増やすとkappa_nは減る。**nに依存するhard total capを付けないcore集合**はnested。
n<=hat n<=rnなら同じ(q,y)を使え、全非負shot費用と非負once費用はfactor r以内。
全fixed-n frontを保持する理想解法でのcoverageと、有限weighted objectivesの出力を分けた原文は正しい。

固定total capは保存されない。人工的なD=(1)、t=1、q=y=1、d=0、e=ell=1、C=1を考える。
n=4はBernstein条件4>=10/3を満たし、二軸G=8でtotal cap=8も満たす。
gridへhat n=8と切り上げるとconfidenceは満たすが、G=16で同じcapを超える。
倍率保証2は成立し、元capでのfeasibilityは成立しない。

従って設計§7.2には、元cap保存を主張しないという条件を§7.3と同様に明記する。
fixed hard capsで運用するなら、そのcapを各grid点で直接検査し、grid未解決を元整数問題のinfeasibilityと呼ばない。
原文のv(n)非増加・outer boundも、非負costかつn-dependent total capsなしの範囲でのみ使える。

## 5. 数値certificate・資源model：M5

実装するqは非負でsum q=1、y>0であることをexact rational等で検査する。
r=Dq-ytから、||Rhat||<=1、各phaseの絶対値1より、normalized operator residual<=||r||_1。
corrected residual biasは||r||_1/y以下。
従ってey-d^Tq-xi>=kappa_n、xi>=||r||_1は正しい十分条件。
xi<=y delta_numを事前固定して、許容numerical差と厳密なideal mean保存を区別する。

interval版はq/y非負なら、各行で
abs(D_mid q-y t_mid)+D_rad q+y t_radがworst-case residual upper。
原文の二本のz制約とsum z<=xiは正しい保守的LP。ただしbox correlationsを捨てる。

**追加の実装条件：ellとkappaも認証する。** 原文§8はD/t/d/Cを扱うが、log/rootの丸め方向を明記していない。
ell_upper>=log(2/alpha)を取り、その正根のoutward上端kappa_upperを使用する。
kappaの過小丸めを許すと、表示LPはfeasibleでもBernstein条件を満たさない点が生じる。
nominal LP optimumとcertified operating point／gapを別保存する。

実際のsamplerが記録q/pと異なる場合、正規化式だけの証明を移送できない。
exact rational samplerを仕様化するか、lawのズレによるmean/biasを別に戻す。
有限bitのsampler、IID law、phase、signed loweringは将来のfocused semantic gateで検証する。
本監査はsamplerやsolverを実装・実行していない。

workspaceは平均costで扱う容量ではない。workspace=100のcolumnを確率1/100で使ってもcapacity=2では実行不可。
workspace cap Wを使うなら必要量>Wのcolumnsを禁止する等のsupport条件を固定する。
同様にclassical acquisition、worst-case time、once costが選択に依存する場合の扱いを定義する。
point expected gate countsしかないLPをactual-resource certificateとは呼ばない。

## 6. IS・baseline・controlled実装：M7–M9

固定wとrに対しV=sum w_j^2/r_j、R=max w_j/r_jは正しいsecond-moment/range upper。
bias=sum w_j d_jは抽出確率を変えても不変。
2V+(4/3)sR<=ns^2/ellはfixed nでconvex、V epigraphはquadratic-over-linear、R epigraphはw_j<=R r_j。
joint SOCPもrange capとbias budgetを外から固定すれば成立する。
それらを同時にfreeにしたglobal convexityを主張しない。zero-weight/supportの閉包規約は要固定。

leading product optimum=(sum w_j sqrt(C_j))^2、representation側のweighted LPは既知ISの直接帰結。
finite-confidence canonical LPの比較からこれを省いて、sampling最適化を新しい差と呼ばない。

完成ensembleのcanonical混合ではconditional profileを固定しD_r=t/B_r、y=sum q_r/B_r。
degree columnsへの自由配分は、そのprofile比を保持するwhole-ensemble混合を含む広いclassとなる。
同じ表・precision choices・lowering・confidence model・取得費用を与える条件が必要。
全改善がprecision-onlyまたはwhole-mixtureでも再現できる可能性を除外していない。

controlled adjoint pairは0 branch=A_dag A=I、1 branch=X A_dag X A。
ideal half-RZ Uに対する差はunitary productのtriangle inequalityで<=2 delta_A。
conjugatorも同じ列とexact adjointなら0 branch identityを保つ。
一般R1の独立符号合成と異なるnative optionであり、旧count/errorを置換しない。
一般joint approximationのsafe 2deltaと、exact-controlled-formのmean operator boundを分ける原文は妥当。

mean伝播はideal ||M_j||<=1とlocal e_jから、telescopeでproduct(1+e_j)-1。
exact Ctrl(tilde U)なら平均積を使える。general joint実装ではCPTPのdiamond telescopeを用い、
corrected bias<=product B_j * sum_j bar d_jとなる。
固定observableだけのerror upperはdiamond boundとして使えない。
I/-I ensembleはoperator first mean=0だがchannelはidentityなので、この二つのtargetを同一視しない。
product yとcross-block costがあるため、全block global LPへは拡張されない。

## 7. Claim単位の既知性・差分

以下は読んだ一次資料の範囲での監査であり、publication priorityの不存在証明ではない。

| Claim | 既知の具体箇所 | 本案に残る差／非claim |
|---|---|---|
| dictionary＋convex coefficients＋error/overhead交換 | [Koczor v2](https://arxiv.org/html/2402.15550v2)、II.1–II.3、Eqs.1–6：process/PTMをtargetにする | 辞書やconvex optimization自体は既知。RAのformal finite operator meanはchannel targetと異なるが、これだけで新規性は確定しない |
| cost×second-moment optimum、bias invariance | [Cugini–Atif–Subasi v1](https://arxiv.org/html/2603.13495v1)、II Eqs.7–13、III Eqs.21–24 | fixed protocol ISは既知。leading joint weighted LPはその直接帰結。finite bias/rangeを含む限定modelの意義は要評価 |
| identity pairing／common angle | [Zeng et al. v2](https://arxiv.org/pdf/2212.04566v2)、IV.B Eqs.73–79、R0.5 same-target audit | Euler/common-angleを新原理としない。restricted adjacent/mixed degree classと取得costを明示する |
| CTS collection／partial expansion | [Peetz–Smart–Narang v2](https://arxiv.org/html/2407.21095v2)、IV.2、Supplementary Note 3 | Pauli-domainの強い既知対照。layeringのnorm増と情報取得を省かない。toyでPauli展開を禁止しただけのI0 advantageは主張しない |
| term別hardware-aware angle | [Structure-Aware Variance Reduction v1](https://arxiv.org/html/2606.23544v1)、III.1 Remark、III.2 | cost-aware angleの一般原理は既知。finite-table precisionとphase-preserving RTE会計の統合差は未確定 |
| coherent/random LCU groupingの交換 | [Wada et al. v1](https://arxiv.org/html/2512.06260v1)、Introductionのsubgroup coherent LCU構成 | grouping/mixture自体は既知。本案のcanonical one-event degree配分とは同じalgorithmではないが、一般的sampling-resource交換を新規としない |
| fractional normalization、LP dual・疎解 | [Charnes–Cooper (1962)](https://doi.org/10.1002/nav.3800090303)、出版社書誌を確認 | q/y変換は古典手法に関連する。全文を新たに精読したとはしない。ここでは式を独立に導出した |

Zionts DOIの追加取得はtool errorで未確認。引用網全体・全文の横断的不在確認は未完了。
generic LPへ同じD/d/Cを渡したときの値一致は、correctnessの期待動作。
その一致を手法の実益や新規性の証拠とも、直ちに研究STOPの証拠とも扱わない。
数学の正しさと論文貢献の採択は別の判断である。

## 8. 採用可能な最小modelと、未承認の次案

技術上閉じる最小modelは、一block、finite ideal dictionary、固定conditional IID law、phase-preserving native table、
nonnegative canonical weights、fixed symmetric n、fixed epsilon/alpha/bias reserve、nonnegative additive resources。
exact ideal theoremとfinite-bit residual付きcertificateを別modelにする。
workspace、acquisition、各cap、integer gridのscopeを上記の条件で固定する。

次の小型pilotが反証する問いの**未承認案**は一つ：
「同じ有限表・confidence条件で、degree配分の自由化により、precision-only／whole-ensemble mixtureが既に作れるfrontへ追加の資源点が生じるか」。
実資源優位を主張する前にはfixed-ensemble ISと利用可能なCTSにも照合する。
candidate table生成規則／上限、input、precision、n-grid/caps、backend identity、acquisition会計、
comparison budgets、numerical certification、sourceとauthorizationは未固定。
本監査はその表・角度・etaを生成せず、pilotの必要性やscience scopeを採択しない。

## 9. 実施記録とSTOP

[事前固定protocol](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_protocol_v1.json)、
[stdlib checker](../../../scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)、
[全50 checks](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)。
自由wordの係数、signed phase、終端、q/y対応、shot polynomial、dual gap、二方向分解、
grid cap反例、既知IS恒等式、抽象adjoint、故意のphase/IID/bias/probability mutationsを確認した。
これはoff-domainの数学bookkeepingで、一般証明は本書の導出に基づく。

LP/SOCP solver 0、実sampler 0、matrix/circuit評価0、science/synthesis/compile/NPZ/GPU 0。
R1/R1.5の再分類や再採点0、旧artifact移動／削除／再生成0、共通API変更0。
local technical evidenceであり、immutable CIや外部再現ではない。
commit/push後、[GPT handoff](ra_rte_mathematical_audit_gpt_handoff_20261006.md)へ戻す。
**mandatory STOP。algorithm採択、次実装、R2、追加synthesis、DF接続は未認可。**
