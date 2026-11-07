# Track C（O3）PRアルゴリズム改善研究の統合報告

2026年11月8日。対象repository：`HIROMU1015/Partially-Randomized-Trotter`。収集基点：`94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9`。本書は既存の固定SOURCE、REPORT、JSON/CSV、raw索引を照合した文献・保存結果の統合である。**本総括では新しい科学計算、回路build/compile、sampling、MC、探索、詳細GPT科学レビューを行っていない。** 各研究の過去のSTOPと未資格事項は維持する。

## 1. 研究の結論と位置づけ

Track Cは、部分ランダム化Product Formula（PR）の時間発展を改善し、同じenergy精度・成功条件で必要な量子資源を減らす研究である。作用素誤差だけを小さくするのではなく、基底状態の位相bias、有限RTEの信号減衰、shot数、実装回路費用を接続して評価した。

主要な科学的成果候補は、**決定論部分Dの内部PF誤差を単独で消すのではなく、D/R混合誤差の基底状態期待値と相殺する係数選択**である。既知の対称合成族を用い、固定H4の保存基底状態からBCH係数を選ぶと、R側へ負時間を導入せず大きいD刻みを使えた。登録済みRTE分割数を固定した比較で、18roundのshotsは8598→8498、推定総RZは26.195%、総CXは30.687%減少した。これは一般の4次作用素公式の発見ではなく、固定状態・固定Hamiltonianに対する主要energy補正の相殺である。根拠は[§5](#5-d内部bch調整主要な科学的成果候補)と[D研究の原報告][D]・[登録比較][DREG]。

追加の回路成果として、singleton partial QR、same-basis runの共同isometry、同一basis内の対角演算の融合を接続した。共同isometryは従来singleton方式との独立paired比較で総RZを3.355%、CXを1.789%削減した一方、serial depth上界は0.696%増えた。対角融合のRZ利益は、既知の効率的な単一controlを基準にすると小さく、RTE run固有の追加効果は全PRの約0.135%だった。[共同isometry原報告][J]、[対角融合原報告][F]。

**最終的な同条件比較では強Full PFが低コストだった。** 対角合成を双方に与えたRZ優先policyの条件付き18round会計は、PRがRZ `1.42343e12`・CX `1.97230e11`、FullがRZ `5.66091e11`・CX `1.01463e11`。PR内部の改善は成立したが、PR全体の優越は成立していない。既知controlの導入はFullにも大きく効き、最終のPR/Full比はRZ約2.514、CX約1.944になった。[§8](#8-最終比較同条件の強full-pfは低コスト)、[保存最終値][FFINAL]。

## 2. 固定科学条件・記号・評価対象

### 2.1 変更しなかった研究条件

| 項目 | Track Cの固定条件と読み方 |
|---|---|
| 物理入力 | Linear H4、隣接1.00 Å、STO-3G、neutral singlet。保存DF12の生成順を使用 |
| 系・状態 | 8 spin orbital / 8 system qubit、4電子sectorは70次元。元Hの厳密基底状態を利用可能と仮定し、状態準備費は除外 |
| 分割 | Partial LD3：one-bodyと先頭3 DF fragmentのD4 block、残差Rは218 signed Z/ZZ component・7 basis。Full LD12はtail-freeの13 block |
| RPE | m=0,…,17、18round、δ0=.02、T_m=.02×2^m、最長q=131072 |
| 目標 | energy精度1.5936e-4 Ha、失敗確率上限.05、全位相予算.4 rad |
| finite RTE | 既存canonical K2/P3 mean、component確率、符号、normalization、return aggregationを保持。既存resolutionの選択変更とlaw変更を区別 |
| 最終native profile | Qiskit1.3.0、Track C reference compiler A、basis rz/sx/x/cx、seed17 profile。定義によるloweringと固定inverse cancellation、微小角cutoff・近似交換最適化なし |
| 除外 | 状態準備、実機noise/routing、誤り訂正、最終方式のClifford+T合成、実機shot実行 |

条件の出典：[初期pilot][PILOT]、[保証A][GA]、[D研究][D]、[最終融合][F]。後続研究で再利用したcommon cacheのSHA256は `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。公開入力参照とprivate cacheの区別は[DのFINAL][DFINAL]に残る。今回、SCF・DF fit・ground solveの再実行やprivate cacheの代替生成は行っていない。

「Track C（O3）」は、旧「研究C：Hamiltonian表現・PR-aware DF」やTrack B、B2 P1とは別系列である。既存DF/RTE、Hadamard、QR/Givens、decoder、価格処理の再利用を、それら別系列の科学成果の再帰属とは扱わない。B2のopt1/seed7と、初期Track Cのopt2は最終compiler A価格に混ぜない。旧PR資源研究は、event単位の単純和がbasis共有を失うこと、Hamiltonian snapshot・compilerごとに価格modelを区別すること、短round fitの移送にholdoutが要ることの背景として用いた。[研究全体の旧概要](../研究概要・現状.md)、[統一索引](../RESEARCH_RESULTS_INDEX.md)、[再利用の原報告][K]。

### 2.2 位相誤差からshot・総資源への接続

分割は `H=H_D+H_R`、Dの正確な発展を `D(t)=exp(-iH_D t)`、Rを `R(t)=exp(-iH_R t)` と書く。実装のDはfragment PF、Rは有限RTEであり、理想的な二項分割との違いを明示する。scalarとcontrolled相対位相も費用・誤差に含む。

量子Hadamard測定のraw複素信号を `z_m` とすると、X/Yはその実部・虚部を測る。normalizer `B_total` による減衰を一度だけ反映し、全SOURCE/native/finite-bit diskを含む半径下限を `rho_m`、.4 radから系統位相予算を引いた統計幅を `beta_stat,m` とする。各軸に対する十分shot数は

$$
N_{m,b}\geq\left\lceil\frac{4\log(2/\alpha_{m,b})}
 {\rho_m^2\sin^2\beta_{\mathrm{stat},m}}\right\rceil,
\qquad \sum_{m,b}\alpha_{m,b}\leq .05.
$$

一様配分は `alpha=.05/36=1/720` なので対数項は `log(1440)`。後続のcost-sensitive α配分は、高価な長roundへの割当を増やし、安い短roundにより多くのshotsを置く。**総shotsの増加と総ゲート費の減少は両立する。** 最小shot数やglobal最適配分の証明ではない。[保証A][GA]、[資源配分][K]。

既存alias条件、各round位相幅.4、`3×.4<pi` によりbranchを制御し、最終energy幅は

$$
\frac{.4}{.02\,2^{17}}=1.52587890625\times10^{-4}\ \mathrm{Ha}
 <1.5936\times10^{-4}\ \mathrm{Ha}.
$$

保存Hermitian-H・そのexact ground・理想native gate・fresh独立finite bitsの条件で、保証を閉じた後続policyは成功率≥.95の十分条件を満たす。E0は係数設計・結果照合に利用するが、decoderへ渡して未知biasやaliasを補正しない。canonical平均信号からのbranch復元と、有限shotを実行したときの十分条件は別の証拠である。[GA]、[D]、[J]、[F]。

資源jは個別に

$$
G_j=\sum_{m=0}^{17}\sum_{b\in\{X,Y\}}
 N_{m,b}\,\mathbb E[C^{\mathrm{no\text{-}prep}}_{j,m,b}],
\qquad j\in\{\mathrm{RZ,CX,depth,size}\}
$$

で会計する。RZ・CX・depth・size・Tを合算しない。後続の価格対象は、compile済みD区間と**whole RTE occurrence**、wrapperを連結した具体的modular native回路である。そのRZ/CX/sizeは実装の定義上加算できる。部品depthの和はserial上界で、m17回路全体の実DAG depthではない。短whole回路との一致は、その具体的検査範囲の観測であり、全長の最適whole価格の定理ではない。[D]、[K]、[J]、[F]。

### 2.3 証拠の種類と比較の規則

| 証拠・不確かさ | この研究での扱い |
|---|---|
| BCH恒等式・既知公式 | 保存SOURCEの代数展開・既知対称合成。有限h・有限RTEでの資源改善は別途検証 |
| float64診断 | 位相bias、dominant overlap、折返し分類、mesh則。区間包含・一般定理と区別 |
| 数値保証 | directed interval、SOURCE生成、compiler、finite-bit、normalization、aliasの条件付き包含。元の連続積分Hの無条件包含は未資格 |
| 価格MC標準誤差 | 条件付きstratumの確率を戻した推定値。paired差はcovarianceを使う。MC SEだけで価格model誤差や数値diskを覆わない |
| 経験的長round model | 初期q1/q2 affine fit→q8 holdoutはm17保証ではない。後続の具体的modular実装へ読み替えない |
| 実機・外部再現 | 実機保証・noise評価・第三者再現・immutable CIとは呼ばない。人工test PASSは科学性能PASSではない |

以下の改善率は**各研究内の指定対照**に対する値であり、研究間で掛け合わせた累積削減率を作らない。新holdoutで同じ旧回路を再測定したMC推定値の差は、アルゴリズムの追加変更と数えない。

## 3. 仮説の変遷：どの結果が次の研究を促したか

| 段階 | 改善仮説 | 保存結果と次の問い |
|---|---|---|
| C-N | 高次PFとwhole native回路が強加法対照より有利か | compiler Aで2304 paired S2標本の全4価格が一致。次数・whole化だけの利益は未確立。PF誤差の構造へ進む |
| C-S | coarse回転と残差RTEに分け、合成Tを減らせるか | 登録policyはnormalizationが極端に増大。誤差削減とshot負担を同時に見る必要 |
| C-P | 固定processorでkernelの位相性能を改善できるか | processorがほぼidentity、信号・shot改善なし |
| compiler A・保証 | native省略誤差を隠さず18roundを比較できるか | 初期opt2を棄却し、保存行列モデルの条件付き保証を整備。価格移送の未資格は別に残す |
| C2-A | roundごとにPF族を切り替える利益があるか | 3round診断ではS2、交差未観測。全18round優位・一般no-goは未確立 |
| C2-B | 共役blockの誤差構造を用い合成精度を配分できるか | 有限T policy差は約.514%、強対照の最適性gapを超えず未確定 |
| 混合公式 | D/R交換子を減らす2MN等でPRを改善できるか | 作用素誤差は減るがground bias・shotsは同コスト半刻みS2を下回らない。D内部誤差へ着目 |
| D内部BCH | 内部誤差と混合誤差をground期待値で共同相殺できるか | 係数をBCHで固定し、位相・shots・native Gの改善を確認。強Fullには届かない |
| bottleneck / singleton | 残るRTE basis費を、共有を壊さず削減できるか | always-partialは悪化、singleton限定は改善。長round配分も共同選択 |
| 共同isometry | 複数applicationの必要列をrunでまとめられるか | PR総RZ3.355%減、depth上界は増加。残る中心対角演算へ |
| 対角融合 | run内Z/ZZの可換構造でさらに削減できるか | 既知control後のPR固有RZ利益は小さい。Fullも改善し、差は縮まらない |

原報告は順に[PILOT]、[GB]、[A2]、[B2]、[M]、[D]、[K]、[J]、[F]。この表の矢印は研究上の経緯であり、後続完了によって旧研究のSTOPや旧未資格を上書きしない。

## 4. 初期候補・C2とcompiler基盤

### 4.1 C-N：高次PF・native価格の強対照

初期C-NはS2、Yoshida4、公開Morales Y8m10b、h=.02/.01、LD3/12の12構成を評価した。ideal/finite meanの18round診断は得たが、指定opt2の未計上native誤差で価格を停止した。初期Q価格は63/306セル、T価格は24/108セルまでであり、欠損を0で補わない。[PILOT]

compiler A導入後の独立cohortでは、Partial S2のr=1,2,4,8,16,32、q1/2較正・q8 holdoutを取得した。強対照にはR occurrence間の最大D run融合とwrapperを与え、弱いhalf-run単純和を使わなかった。**2304 paired S2標本でfull wrapperと強対照のRZ/CX/実DAG depth/sizeが全て一致**した。Full Y4の3点の一致は対照の定義によるもので、独立した性能発見とは分ける。[GB]、[C-N保存paired検査][CNPAIR]

q8の最大点誤差6.0905%、最大予測相対95%半幅7.2624%は、それぞれ既存目安5%・2%を通らなかった。保存shot8598/6912で得たRZ点model `4.52054e12 / 1.18703e13` は**長q系統誤差不明の条件付き外挿**であり、後の直接occurrence modular価格と同列に順位づけしない。保証Aの更新Full shots6914と、価格Bの固定6912も区別する。初期C-Nは「強FullよりPRが安い」という確定結論を与えない。[GB]

### 4.2 C-S：残差ランダム化・合成の負結果

通常S0、D+R残差S1、Dのみ残差S2、合成cap .01/.02、残差cap .01を比較した。登録pi/4 coarse・最小r_res=1では、D halfの116 eligible回転全てがcoarse index0だった。終端log(range)は通常 `.5573908260` に対し `386.8029033 / 386.2455126`。全18roundのliteral gate数とrangeから、合成精度は通常1e-13、残差方式1e-181となった。初期uniform診断の終端log10(shots/軸)は約338.3/337.8である。[PILOT]、[初期機械可読結果][PRESULT]

これはcoarse回転の軽減よりnormalization負担が大きい登録policyの負結果であり、全残差法・別r・別lawのno-goではない。strict Clifford+T guardがPASSでも、その前のnative opt2省略誤差を補正しない。最終T優位は未資格。実現ごとのweightとglobal rangeは一度ずつ扱い、回路価格へ重ねてabs(weight)を掛けないという会計規約を保持した。[PILOT]

### 4.3 C-P：Processingの無改善

固定D-contextの公開YP8m8 kernel、h=.02、nD=1、固定processor Pを用いた。PとIの作用素距離は約1.270e-12、P0/P1のcorrected複素信号差は最大7.664e-15、36軸のuniform shotsは全て同数だった。512次元joint空間でX/Yの測定等価loweringは照合したが、丸めを超えるspectral改善は観測しなかった。これは当該固定processorの負結果であり、processing一般の不可能性ではない。[PILOT]

### 4.4 C2-A：round依存PF

LD3/12を別々に、S2/Y4、h=.02/.01、LD3 resolution1,…,32、m=0,8,17の84信号を調べた。q1/2・q4の60短価格セルはstratum代表pathでありMC母平均の精密推定ではない。3診断round全てでS2・h=.02、LD3のrは1/1/32が選ばれ、PF族切替の交差は未観測。q4のRZ/CX最大点誤差15.3821%、新構成native envelope未資格のため全18round資源へ進まなかった。高次PFが全roundで無用という一般結論ではない。[A2]、[C2-A保存結果][A2RESULT]

### 4.5 C2-B：構造的合成誤差と強baseline

対象はPR LD3ではなく**Full Y4・LD12・h=.02固定**のClifford+T合成である。共役block `B D B†` にactual adjointを用い、basis合成誤差epsilon_Bに対する通常 `2 epsilon_B+epsilon_D` の代わりに

$$
\min(2,|\tau|W)\,\epsilon_B+\epsilon_D
$$

を適用した。Wは全256占有状態のdiagonal spectral幅。帰りのbasis費は実際に課金し、負時間もabs(τ)で会計した。これは既知の共役誤差評価の適用である。[B2P]

保存7603 strict word、各caseの登録2精度、2capを再利用した強対照検証では、同等のMILP強化・α/cap再選択を与えた実現総Tは通常 `493413295481472`、構造 `490876793817392`、差 **0.514072%**。shotsは11616/11610。通常の厳密緩和下界 `486238229821631` が構造の実現値を下回るため、改善の最適性は未確定である。72case中27のみ厳密case最適性を確認した。旧0.547022%と後続0.514072%は異なる比較policyで、矛盾ではない。[B2]、[C2-B保存比較・下界][B2RESULT]

このT数を、後続native RZ/CXの削減へ加算・換算しない。対象も固定Fullであり、partial PR一般への利益や強Fullに対するPR優位の証拠にはしない。

### 4.6 compiler Aと18round保証：性能の前提を整備

初期opt2ではCRZ(1e-5)が空回路になり、joint共通phaseを除いても作用素差約5e-6。保存Full Y4 q1のX差は約1.4077e-5だった。静的調査はRustの微小回転交換判定と別の合算角cutoffを区別し、H4差の全量を一経路だけに帰属しなかった。[STATIC]

reference Aは定義に基づいてlowerし、数値的な近似交換や角度切捨てを避ける。44人工ケースとH4 q1の全512次元already-controlled operator・X/Y・負時間を検査し、旧X差は約1.45e-12へ低下した。許すのは**joint回路全体に一つの共通global phase**であり、active/inactive branchに別phaseをfitしたり、lower後に新しくcontrolを付けたりしない。[ACOR]

保証は段階的である。128 context・1302局所loweringの18round評価時点では36条件の残余予算は正でも、SOURCE→mean・alias・旧数値diskが未完だった。次の保証AでSOURCE生成、有限RTE、spectral residual・polar bound、ground/alias、finite-bit CDFを区間評価し、保存行列モデルで条件を閉じた。保存ground energy区間は `[-2.166387448635427, -2.1663874486340977] Ha`、state距離上界は約1.46e-11。これは元の連続積分Hの厳密包含や実機保証ではない。[A18]、[GA]

後続D角度・basis completion・対角角度和の変更では、必要なsource/native diskを追加した。過去のq1 PASSや旧角度の保証を無条件に新公式へ移していない。

## 5. D内部BCH調整：主要な科学的成果候補

### 5.1 外側混合公式だけでは改善しなかった理由

既知5段族

$$
U_a(h)=D(ah)R(h/2)D((1-2a)h)R(h/2)D(ah)
$$

について、A=−iH_D、B=−iH_Rとすると保存代数展開は

$$
\log U_a=h(A+B)+h^3\{\alpha[A,[A,B]]+\beta[B,[A,B]]\}+O(h^5),
\quad\alpha=\frac{1-6a+6a^2}{12},\quad\beta=\frac{1-6a}{24}.
$$

`C_D=[H_D,[H_D,H_R]]`、`C_R=[H_R,[H_D,H_R]]` に対して `H_eff=H−h²(alpha C_D+beta C_R)+O(h⁴)`。exact groundでは

$$
\langle C_D\rangle+\langle C_R\rangle=0,\qquad
\langle C_D\rangle=-2\sum_n(E_n-E_0)|\langle n|H_D|0\rangle|^2\leq0,
$$

$$
\alpha-\beta=\frac1{96}+\frac12(a-\tfrac14)^2>0.
$$

したがって、**正確なD/R二項分割のこの5段族では、ground主要混合biasの最小はa=1/4**であり、一般にゼロにできない。負のaでもこの式は変わらない。a=1/4は半刻みS2二回と有限RTEまで同値で、新方式の利益に数えない。2MN `a=.1931833275037836` やalpha=0は既知候補である。[混合公式の理論・先行文献][M]、[保存代数・局所結果][MTHEORY]

固定H4の期待値は `<C_D>=−.07277680776710108`、`<C_R>=+.07277680776709988`。実際のmultiblock PFではD内部誤差と追加mixed寄与もある。2MNのH2作用素normは約.00685863、半刻み対照は.01560067で約56%減ったが、ground H2は.00385183対.00333506で悪化した。作用素normの改善は基底位相の改善を保証しない。[M]

全18round3564行（実条件3244）の同一h/resolution比較で、2MN/alpha=0の双方feasibleな各576条件のうち、半刻みS2よりshotsが減った条件は0。96 paired価格標本・576 measured wrappersのRZ/CX/depth/sizeは同一だった。partial Y4は負R時間によりm17 logBが約2.4544、半径約.0859、shots/軸26090へ増加した。この研究の新角度native18round保証・絶対Gは未資格のまま負結果として保存した。[M]、[全round保存表][MROUND]

### 5.2 D構成の自由度と共同相殺

D内部研究は、LD3の4 fragmentを保った24順序、対称な単一fragment時間再配分、D自己対称S2、D-only Yoshida4、Dの対称3段合成を調べた。fragment0はone-body、1〜3はDF prefixであり、Hamiltonian・partition自体は変えていない。[D]

元の片側sweepからなるPRでは、順序0123のground H2 `.01334023255` に対し0321は `.01230116103`。単一fragment再配分の最良例はfragment0・a≈.2518184061、係数.00611798241だったが、この族に実根はなかった。順序変更の全24係数は正で、その正時間合成の三次係数和だけではゼロにできなかった。[D]

D自身の対称S2を

$$
S_D(t)=\exp[-it(H_D+t^2K_D+O(t^4))]
$$

と定義し、既知対称合成

$$
D_v(t)=S_D(vt)S_D((1-2v)t)S_D(vt),\qquad
\eta(v)=2v^3+(1-2v)^3
$$

を外側 `D_v(h/2) R(h) D_v(h/2)` に置く。その主要effective-Hamiltonian誤差は

$$
H_{2,\mathrm{PR}}=H_{2,\mathrm{outer}}+\frac{\eta(v)}4K_D,
\quad H_{2,\mathrm{outer}}=-\alpha C_D-\beta C_R,
\quad (\alpha,\beta)=(-1/24,-1/12).
$$

基底期待値から

$$
\eta(v)=-\frac{4\langle H_{2,\mathrm{outer}}\rangle}{\langle K_D\rangle}
$$

を満たすvを選ぶ。D-only Yoshida4はeta=0でD内部だけを消すが、今回は**eta<0の内部biasを用いて正の混合biasも消す**。係数はBCHで選び、長round位相や価格へのfitで選んだものではない。[D]、[D固定SOURCEと入力][DFINAL]

| 構成 | D内部ground H2 | mixed ground H2 | 合計 | D指数/outer step（literal） |
|---|---:|---:|---:|---:|
| 元partial S2 0123 | .01194830121 | .00139193134 | .01334023255 | 8 |
| 順序0321 | .01052108376 | .00178007726 | .01230116103 | 8 |
| D自己対称S2 sandwich | .00298707530 | .00303236699 | .00601944229 | 14 |
| D-only Yoshida4 | 約0 | .00303236699 | .00303236699 | 38 |
| D調整0123 | −.00303236699 | +.00303236699 | −2.04e-14 | 38 |
| D調整0321 | −.00303236699 | +.00303236699 | −1.70e-15 | 38 |

出典：[D原報告のBCH表][D]。元PRと新sandwichではDの組み方が異なり、同じ「混合寄与」という名でも値を付け替えない。旧混合研究の「外側D/R・D内部・追加mixed」の三分解とも区別する。D指数の表はstep内部のliteral数で、全反復後の境界結合済み数ではない。

D調整0123は `v=1.4768778287800015, eta=−1.0151625527143378`、0321は `v=1.490624664134615, eta=−1.1528724828732642`。中央D時間は負だがR(h)は正一回のまま。全fragmentのnet時間、controlled scalar、負時間・追加D呼出費を保持した。構成はYoshida型既知合成の再利用であり、新しい一般高次PFの発明とはしない。[D]

### 5.3 何が消え、何が残るか

調整0123/0321のH2作用素normは約.03146/.03177で**非ゼロ**。消えたのは特定基底状態における主要energy補正であり、作用素4次精度ではない。孤立dominant ground branchのenergy補正がh²からh⁴に移る機構である。係数を選んだ厳密基底状態は本研究の利用可能資源として仮定しており、利用不可を理由に結果を棄却しない。一方、別状態・別分子・近似状態へ係数が移る保証はない。[D]

新規性の候補は、PR内部Dの誤差を混合誤差と共同で制御し、Rの負時間によるnormalization増を避けながら、有限RTE・native回路・RPE会計まで効果を検証した点にある。BCH期待値の利用、状態依存誤差設計、対称合成自体には先行研究があり、世界的新規性・最適性は未確定である。先行文献の入口は[Dの参考リンク][D]（Yoshida、Maxwellほか、Childsほか）。本総括は新たな網羅的先行研究調査を実施していない。

### 5.4 有限RTE・登録対照・資源結果

初期11988信号行（実条件10434）、Full追加7452行、D価格410セル、33 RTE occurrence条件×2層×16＝1056価格標本を保存した。新角度・SOURCE生成・compiler・finite-bit等は219 unique論理条件を資格化した。Fullにも最初の4 blockの同じ24順序、S2/Y4、同じh候補を与え、全13!順序の最適化とはしていない。[D]、[DFINAL]

まず、**登録済み元PRのr_mをそのまま固定**した主比較である。調整Dはh=.16、元はh=.02。短roundは実Tに応じた刻みを使う。

| 登録対照 | 全18round shots | 総RZ（MC 1 SE） | 総CX（MC 1 SE） |
|---|---:|---:|---:|
| 元PR S2 | 8598 | 4.77920e12（1.2893e11） | 1.09324e12（1.4076e10） |
| D調整0123 | 8498 | 3.52729e12（6.5269e10） | 7.57756e11（7.0997e9） |

RZ26.195%・CX30.687%・shots1.163%減。paired差のSEはRZ1.2298e11・CX1.3406e10。m17のPF位相上界は.01400660→.00232990、shots/軸626→592。同じRTE4194304 event、logB=.557390826を保ちながら、PF stepは131072→16384、結合後D指数は917505→606209となった。平均信号energy誤差は約5.34e-6→8.84e-7 Ha。±10%刻みのholdoutは有限位相差 `.0015203/.0023168/.0033907 rad` とh⁴則に整合し、再fitしていない。[DREG]、[D]

次に、有限menuからroundごとに資源優先で選んだ**別policy**では、折返し非依存の改良PRが11572shots、RZ3.34845e12・CX7.21658e11、強Fullが8828shots、RZ1.20943e12・CX3.80934e11だった。登録比較の8498shots・3.52729e12と混同しない。[非共鳴保存比較][DNR]

粗いhでwrapped終端位相だけが小さくなる例もあった。Full S2順序1320・h=1.28のm17はdominant energy bias約.023996 Ha、累積bias約62.90 radなのにwrapped PF幅約.06984 radだった。後続比較は両armに

$$
|\Delta E_{\rm dominant}|T+arcsin((1-p)/p)<.4
$$

の追加十分条件を課し、折返しによる偶然の改善に依存しないpolicyを用いた。この分類はfloat診断であり、別のdirected native保証と区別する。[D]、[DNR]

### 5.5 同じ主要仮説内の負結果

D-only Yoshida4は内部誤差を消しても混合biasが残り、source-shot価格診断の総RZは約6.00e12と悪化した。fragment1/2/3の再配分も主候補を上回らず、順序だけの改善は限定的だった。全体partial Y4は負R時間・normalization増の負担が残る。**D内部精度を単独で最大化することと、PR全体を最小費用にすることは同じでない**という結論が、その後の費用分解の動機となった。[D]

## 6. 資源分解からsingleton・共同isometryへ

### 6.1 改良後も残った支配要因

以下はD研究後、α一様・追加basis最適化前の折返し非依存policyの分解である。D/D融合のsigned補正を既にDへ含み、境界費を二重加算しない。

| policy | D RZ | RTE RZ | wrapper RZ | 総RZ | 総CX |
|---|---:|---:|---:|---:|---:|
| 元PR・同じmenu | 2.27239e12 | 2.17157e12 | 61325 | 4.44397e12 | 1.01929e12 |
| D改良PR・出発点 | 1.45233e12 | 1.89612e12 | 63646 | 3.34845e12 | 7.21658e11 |
| 強Full | 1.20943e12 | 0 | 48554 | 1.20943e12 | 3.80934e11 |

出典：[K]、[成分別・round別CSV][KROUND]。出発PRのRZはD43.37%・RTE56.63%、m17が72.2%、m16約19.0%、最後2round約91%。RTE RZの91.44%がbasis変換だった。露出するD/R両側basisは総RZの約1.97%でbasis費の部分集合。D/D融合補正RZ−2.71769e10・CX−3.51185e9は既に含まれる。[K]、[template成分][KTEMPLATE]

固定shotsでD全体を無料にしてもRTE1.8961e12、RTEを無料にしてもD1.4523e12で、当時Full1.2094e12を超える。一方、normalizationだけを理想的に除く感度ではPR RZ1.11853e12、PF biasだけを0にする感度では3.23473e12だった。これは**実現済みアルゴリズムでも一般的下界でもなく、固定条件の反実仮想感度**である。D精度・境界だけの追加改善より、RTE basis費と減衰の共同問題が大きい。[K]、[感度保存値][KSENS]

### 6.2 singleton partial QRとstep/resolution・α

既知のpartial-column QR/Givensを用い、1/2列だけ必要なZ/ZZを最大7/13 Givens（8モード）で作れる。しかし全applicationを個別partial化するとsame-basis共有が失われ、m17代表cellでRZ1.682倍、CX1.509倍、DAG depth2.621倍へ悪化した。**always-partialは棄却**し、最大same-basis runの長さ1だけを置換した。長いrunはfull-basis共有を残し、trajectory全体で実RZが安い既存回路へfallbackする。[K]

218成分のFock involutionを照合し、数値誤差最大約1.98e-14。128 context・3394 native loweringと全trajectory置換diskを追加した。列spanだけの一致ではなく各列projectorを保持する。旧WP06a/b、Gaussian/Givens合成、既存stable QRを再利用したもので、その一般原理をTrack C独自発明とはしない。[K]、[WP06a][WP06A]、[WP06b][WP06B]

α最適化だけなら出発点PR RZ3.34845e12→2.40298e12、Fullも1.20943e12→9.13529e11。さらに既存h/resolution共同選択では、支配m17にresolution64を追加し4信号・3価格セル96標本を取得した。最終m17はD調整0321・h=.32・resolution64、8192 step、8388608 RTE event、logB≈.27869543、半径下限≈.756675。RTE eventを倍にして減衰を軽減するtradeoffを実価格で評価しており、sampling law変更や追加RTEの無料化ではない。[K]

候補・α・Nを凍結した独立384 holdoutで、singleton方式のRZ優先policyは17054shots、RZ1.79060e12（SE1.69635e10）、CX3.64850e11（SE1.85495e9）。強Fullは14228shots、RZ9.13529e11・CX2.87925e11。出発点からの46.52%/49.44%削減にはbasis方式、step/resolution、α配分が全て含まれ、singletonだけの因果効果ではない。固定点・固定Nのpaired basis削減はRZ4.33865e11（差SE9.06979e9）として別保存した。[K]、[KFINAL]

### 6.3 共同isometry：必要列をrunで一緒に作る

最大same-source-basis runのZ/ZZ support和集合をSとする。元one-particle basis Uの各列U[:,i]を保つunitary completion Vを作り、supportを入力mode f(i)へ写す。各列projectorの一致からFock空間で

$$
\Gamma(U)n_i\Gamma(U)^\dagger
=\Gamma(V)n_{f(i)}\Gamma(V)^\dagger\qquad(i\in S)
$$

を保つため、個々のZ、ZZ、その積・回転も保持する。占有部分空間のspan一致だけでは不十分である。Vは隣接Givens/RZの全8モードunitary積であり、長方形行列をgateとして扱わない。run全体で一組のV/V†を共有する。[J]

昇順/降順×先頭/末尾配置の4layoutに限定し、639登録集合の625を採用。要求rank1〜6が主対象、rank7/8・未知集合・高価な構成は既存full共有へ戻す。trainingに前研究の384 holdoutを再利用したことを明記し、**別の新384 holdout**で評価した。物理確率で重み付けしたm17 occurrenceは約278.63 run、うち複数application118.50、共同partialを使える複数run約68.44だった。[J]、[run分析][JRUN]

| application数 / 必要列数 | full共有RZ | 個別partial RZ | 共同isometry RZ |
|---|---:|---:|---:|
| 2 / 3 | 1020 | 976 | 686 |
| 3 / 4 | 1044 | 1472 | 856 |
| 7 / 6 | 1125 | 3441 | 1121 |

出典：[代表run保存値][JREP]。孤立runの価格であり、この表をそのまま全Gへ足さない。高rankで利益が消える傾向が見える。

625集合の全256次元Fock Z作用とunitarity、11短runの全512次元controlled operator・X/Y、128 context・5380 native loweringを検査した。代表的なsource/native operator差はjoint共通phaseを除いて最大約1.90e-13、X/Y差約1.17e-14。全trajectory一様の追加diskに従来fallbackも含め、古い未supplement資格へ一度だけ追加した。[J]、[区間証拠][JCERT]

同じ新holdout上の旧singletonはRZ1.79034e12・CX3.64837e11、新共同方式はRZ1.73026e12・CX3.58311e11、17054→17024shots。paired RZ削減6.00742e10（差SE2.25175e9）、CX6.52632e9（差SE2.41336e8）。総RZ−3.355%、CX−1.789%、size−3.190%に対しserial depth上界は+0.696%。固定新Nのablationでは、singleton layout変更RZ3.29588e9に対し、multi-application共同共有の追加利益5.56399e10で、basis改善の約94.4%を占めた。[J]、[JFINAL]

D係数・h/resolution・canonical信号・normalizationは固定しており、これはPF位相や減衰の改善ではなく回路構成の改善である。Fullの全13 blockにも同じ4layoutを提示したが、全8列が必要でRZ削減は0件。PR/Full比は当時RZ1.894・CX1.244へ縮まったが、逆転はしなかった。[J]

## 7. 対角gate融合：既知controlとPR固有効果を分離

### 7.1 既に行っていた処理

same-basis共有、共同isometry、event相対位相のoccurrence全体集約、compiler Aの固定inverse cancellationはreferenceへ含めた。DF diagonal primitiveも同じZ/ZZを既に集約し、exact zeroを除いていた。これらを今回の新利益として再計上しない。[F]

従来C builderのgeneric controlled-RZZに対し、既存の汎用部品 `h4_d2_pauli_control._rotation` にある `CX–CRZ–CX` は大きく安い。これは別Trackの科学条件を移したものではなく、**既知の単一Pauli controlを両armの強化共通baselineにした**コード再利用である。[F]

### 7.2 追加した同一basis対角合成

比較方式はreference、既知term_parity、同一Pauli角度集約merge、共同parity networkの4種。controlled rotationを

$$
\mathrm C[RZ_S(\theta)]
=RZ_S(\theta/2)RZ_{\{a\}\cup S}(-\theta/2)
$$

へ写し、binary64入力角度をexact rationalで集約、最終係数を一度だけ丸める。CZ productはwireごとの偶奇で整理し、負の係数・Taylor符号・ancilla相対位相を保持した。異なるsource basisの間を可換扱いしない。[F]

既存Qiskit GraySynthはT placeholderでGF(2) parity topologyだけを取り出し、要求maskが一度ずつ現れ最終線形変換がidentityになることを確認してRZへ置換する。Qiskit1.3数値角度経路のmod πを通さず、controlled 2πの相対−1も保持した。各trajectory/D区間は実native価格の辞書式RZ→CX→depth→sizeで**一つの回路**を選び、指標ごとの別回路minを合成しない。[F]

33 run・9 Dの全joint operator、8短whole X/Yを検査し、run source/native差は最大約1.12e-14 / 1.73e-13、whole測定差約5.59e-13。以前の全128 context証拠を再利用し、角度集約の追加誤差 `2uN+N×2^-1074 (u=2^-53)` を保守会計した。raw複素disk最大約5.27e-9を一度だけ追加、29 D価格条件のhalf-angle和の丸め誤差はexactに0だった。物理モデル外側の保証は増やしていない。[F]、[FFINAL]

### 7.3 改善余地とablation

m17代表occurrence（t=.32、n=1024）の物理重み付き独立holdoutでは、元1024回転に対し集約後supportは約853.81、重複は約16.6%。しかし既知control後の回転関連RZは約1708にすぎず、総RZ約174800の約.98%。basis費が残るので、回転部分だけの削減は小さい。[F]、[改善余地保存値][FOPP]

| 固定した新shotsでの追加処理 | 全PR総RZ削減（差MC SE） | 全PR総CX削減（差MC SE） |
|---|---:|---:|
| reference→既知term_parity | 3.16449e11（5.49939e7） | 1.44449e11（2.91441e7） |
| term→同一Pauli集約・CZ偶奇 | 1.91824e9（4.06832e7） | 3.22761e9（8.00356e7） |
| merge→共同parity network | 0 | 1.49900e10（4.69616e7） |

出典：[FFINAL]。networkのCX利益の内訳はD1.46161e10、RTE3.73875e8。したがって、既知termを基準とした**RTE run固有**の全PR追加利益はRZ約.135%・CX約1.67%。Dも含む共通parity共有の追加利益はRZ約.135%・CX約8.47%。Fullのunique DF primitiveではmergeだけのRZ/CX削減は0、networkはFullにも有効だった。

全変更のPR総RZ−18.196%、CX−45.118%の大部分は既知単一controlである。これを「PR固有の新融合でRZ18%減」と表現しない。Fullにも同じ合成を与えると改善が大きく、相対差は広がった。追加の局所対角RZ探索でFull逆転を期待する根拠は弱い。[F]

## 8. 最終比較：同条件の強Full PFは低コスト

### 8.1 最終RZ優先policyの4指標

以下は**対角融合研究の同じ独立384価格標本**によるpaired旧/新PRと、同じ共通合成機会を与えた決定論Full。選択・α・Nはholdout前に固定した。h/resolution・DのBCH係数・RTE law・canonical平均信号は保持し、αは各価格policyで再配分した。全て18round・同じenergy/成功条件・no-prep・compiler Aのmodular実装である。

| policy | shots X+Y | 総RZ | 総CX | serial depth上界合計 | size合計 |
|---|---:|---:|---:|---:|---:|
| 旧共同isometry PR（今回paired再評価） | 17024 | 1.74005256e12 | 3.59371297e11 | 8.94597516e11 | 2.61365100e12 |
| 既知control＋融合 PR | 17164 | 1.42342991e12 | 1.97230369e11 | 5.46129894e11 | 2.07779668e12 |
| 旧強Full | 14228 | 9.13529189e11 | 2.87924683e11 | 5.98001479e11 | 1.43616078e12 |
| 同じ既知control＋融合 Full | 14232 | 5.66090697e11 | 1.01463437e11 | 2.09331937e11 | 8.39091129e11 |

出典：[F]、[FFINAL]、[全round保存CSV][FROUND]。新PRのMC 1 SEは順にRZ `1.37133e10`、CX `1.57153e9`、depth上界 `5.76988e9`、size `1.97994e10`。Fullは決定論価格でprice MC SE=0だが、物理モデル条件・native数値誤差まで0という意味ではない。PR旧/新差のpaired SEはRZ1.37368e8、CX1.07064e8。各armのSEを独立として足していない。

最終PR/Full比はRZ **2.51449**、CX **1.94386**。Fullは4指標で安い。PR総shotsは少なくなく、局所PF改善やbasis削減を、そのまま全体優位へ置き換えられない。[FFINAL]

CX優先の別policyも保存されている。PRは16760shots、RZ1.42448e12/CX1.97283e11、Fullは14278shots、RZ5.66137e11/CX1.01468e11。主表のRZ優先policyから都合のよい列を入れ替えていない。[F]

### 8.2 研究ごとの到達点は条件付きで読む

| 段階 | 改良PR 総RZ / 総CX | 対応する強Full 総RZ / 総CX | 条件の差 |
|---|---|---|---|
| D内部後の非折返し選択 | 3.34845e12 / 7.21658e11 | 1.20943e12 / 3.80934e11 | α一様、有限menu選択。登録r固定の主比較とは別 |
| singleton＋共同選択 | 1.79060e12 / 3.64850e11 | 9.13529e11 / 2.87925e11 | α・step/resolutionを共同選択、新384 holdout |
| 共同isometry | 1.73026e12 / 3.58311e11 | 9.13529e11 / 2.87925e11 | D/h/resolution固定、別の新384 holdout、旧方式をpaired再評価 |
| 既知control＋対角融合 | 1.42343e12 / 1.97230e11 | 5.66091e11 / 1.01463e11 | 両armの合成を強化、さらに別の新384 holdout |

出典：[DNR]、[KFINAL]、[JFINAL]、[FFINAL]。これは経緯の表であり、同一MC母標本に対する累積ablationではない。共同研究時の1.73026e12が次研究referenceでは1.74005e12になったのは新holdout推定の差で、保存SOURCE・Nを勝手に変更した結果ではない。

比較の公平性には限界もある。FullへS2/Y4、同じh候補、最初の4 blockの24順序、α、partial QR/layout、対角合成を与えた一方、全13!順序、任意のground-specific Full係数、全精度・routing等を探索したわけではない。従って一般的な最適Fullとの差やPR全体のno-goは主張しない。**取得済みの実行可能な強Fullが今回の改良PRより安い**、という負結果は明確である。[K]、[J]、[F]

## 9. 主張・根拠・適用条件・限界の対応

| 主張 | 根拠 | 適用条件 | 主張しないこと |
|---|---|---|---|
| 初期opt2価格は性能比較に採用できない | [PILOT]、[STATIC] | 保存CRZ反例・H4短回路 | 全H4差が一つのpassだけに由来すること |
| compiler Aの条件付きRPE十分条件は閉じる | [A18]→[GA]、後続追加disk | 保存H/ground・ideal native・fresh bits・alias条件 | 連続積分Hの無条件包含、実機保証 |
| C-N whole化の価格利益は未観測 | [GB]、[CNPAIR] | 2304 paired S2、A・短q・指定r | 未実行長q・別compilerの一般定理 |
| C-S/C-Pは登録構成で不成功 | [PILOT]、[PRESULT] | coarse/residual policy、固定processor | 残差法・processing全体のno-go |
| C2-AのPF切替利益は未確立 | [A2] | 3診断round、h=.02/.01、S2/Y4 | 全18round最適性、一般no-go |
| C2-Bは有限policy差.514%、最適性未確定 | [B2]、[B2RESULT] | Full Y4、登録2精度、T価格 | native PRへの同率利益、厳密優越 |
| 5段二項族のground mixed最小はa=1/4 | [M]、[MTHEORY] | 正確なD/R、exact ground、leading BCH | 内部PFを含む全族の最適性 |
| D内部と混合のground主要biasを相殺できた | [D]、[DREG] | 固定H4 LD3・保存ground・BCH係数 | 作用素4次、他状態/系の一般優位 |
| singleton限定なら共有を保って安くなる | [K]、[KFINAL] | 最大run長1・whole回路fallback | always-partialが有利であること |
| 共同isometryはPR資源を追加削減 | [J]、[JFINAL]、[JCERT] | 625集合・4layout・独立paired holdout | depthも改善、QR原理の発明 |
| RTE固有対角融合のRZ利益は小さい | [F]、[FFINAL]、[FOPP] | 既知term controlを共通基準 | 全変更18.2%が新PR固有効果 |
| 最終強FullはPRより安い | [F]、[FFINAL]、[FROUND] | 同じ精度・成功条件、指定modular policy | 全PRに対する不可能性、global最適性 |

## 10. 新規性、未確定事項、今後の課題

### 10.1 Track Cとして何が残ったか

D内部BCH調整は、作用素誤差最小化とは異なる目的を置き、D内部・混合・全体の寄与を分け、RTE負時間を増やさずに基底位相と資源へ接続した成果候補である。共同isometryは、同一basis runの必要列を個別に保持し、Fock/controlled/X/Yの正しさとnative費用を別々に検証した追加改善である。対角融合は、既知controlを除いた利益の小ささとFullへの共通効果まで明らかにした点に意味がある。[D]、[J]、[F]

既知対称PF、Yoshida合成、2MN、QR/Givens、isometry、parity network、Hoeffding/RPE、cost-sensitive α、共役誤差評価は再利用した技術である。世界的新規性の確定、一般分子優位、全PF最適性、一般的資源下界は得ていない。厳密ground利用可能という仮定を保つ一方、係数の状態感度や他条件移送は未検証である。

### 10.2 残る技術・科学上の限界

保存Hと元の連続積分Hの無条件包含、実機noise、routing、状態準備、最終改良PRのT合成費は未評価。短回路correctness、条件付き区間保証、MC価格、経験的modelはそれぞれ別の証拠である。後期のmodular加算は具体的実装の価格を定義し、m17全回路を直接compileせずに会計できたが、最適whole回路やm17実DAG depthを確定していない。

独立holdoutは選択用データとの混同を避けた限定検証であり、16標本/層のMC SEを厳密な同時価格保証へ格上げしない。NumPy float64条件付き価格samplerの系統差を、そのままfresh finite-bit量子shotモデルの証明と同一視しない。新規性の網羅調査・別reviewerによる詳細科学レビュー・外部再現も本総括では行っていない。[GB]、[D]、[J]、[F]

### 10.3 次に検証する価値がある課題と担当境界

| 次の課題（未実行） | 動機・必要な比較 | 担当・判断 |
|---|---|---|
| RTE normalization・samplingとbasis価格の共同設計 | D/PF誤差のみの感度は小さく、残るRTE basisと半径が支配。新lawがshot減と回路増を両立できるかは未確定 | **ユーザー**がTrack Bとの接続・研究範囲を決定。**Track B担当／Codex**は別の明示指示後に実行 |
| sampling/return aggregationを変える場合の会計移送 | component分布・同一basis run統計・normalization・controlled相対位相・MC共分散を同時に更新する必要 | **Track B担当**と条件を合わせ、**Track C担当**は既存回路実装の利用範囲を確認 |
| Dのground-specific設計の頑健性 | exact ground利用可能の仮定内でも、未検証の系・状態・係数自由度への移送は別研究 | **ユーザー**が新条件・主要仮説を選択した場合のみ開始 |
| 共同completionのdepthと価格のtradeoff | 共同isometryのserial depth悪化は残った。RZとdepthの別目的を混同しない | 新研究を選ぶ場合に**Codex**が限定検証。現段階で開始しない |

今回の保存値から、局所対角RZの追加探索だけで強Fullを逆転すると予告する根拠はない。Track Bとの接続も勝利を保証する提案ではなく、未解決の支配要因に対応する次仮説である。既存sampling law・return aggregationは全研究を通じて保全した。

## 11. 出典、固定SOURCE、保存結果の案内

### 11.1 原報告と閲覧snapshot

以下の公開commitは**成果を読むためのsnapshot**であり、各科学計算の実行SOURCEと同一とは限らない。REPORTのSOURCE表、SOURCE.json、FINALの段階別完全SHAを併読する。C2-Bは本branchの祖先にない別branchの固定公開版を参照し、過去のファイルをこの総括のために上書き・mergeしていない。

| ID / 研究 | 固定公開snapshot | 主な保存結果・SOURCEの入口 |
|---|---|---|
| PILOT：C-N/C-S/C-P | [a78054dc4873][PILOT] | [SOURCE][PSOURCE]、[SUMMARY][PRESULT]、REPORT末尾のraw bundle |
| STATIC：旧compiler原因 | [05d9c5d2eb98][STATIC] | 保存trace・公式実装対応は原報告内 |
| ACOR：reference A | [f45e9b07bf73][ACOR] | 人工/全operator/測定の限定correctness |
| A18：18round局所保証 | [3258a89f225d][A18] | 128 context/1302 lowering・未資格項目 |
| GA/GB：保証とC-N価格 | [720416c5b88b][GROOT] | [A_REPORT][GA]・[B_REPORT][GB]、[artifact tree][GART] |
| A2：round依存PF | [ce84fa8e2d2f][A2] | [SUMMARY][A2RESULT]、SOURCE.json・signals/prices |
| B2P：構造合成pilot | [78004943ed5b][B2P] | 補正後verified_final、strict word bundle |
| B2：強baseline | [602ca4b67b10][B2] | [SUMMARY][B2RESULT]、COMPARISONS・ALLOCATIONS・下界 |
| M：混合公式 | [48209b088ca0][M] | [THEORY][MTHEORY]、[FULL_ROUNDS][MROUND]、[FINAL][MFINAL] |
| D：D内部BCH | [b9dacb639eab][D] | [FINAL][DFINAL]、[REGISTERED][DREG]、[NONRESONANT][DNR] |
| K：bottleneck/singleton | [10bf5a481355][K] | [FINAL][KFINAL]、[round分解][KROUND]、[感度][KSENS] |
| J：共同isometry | [2745d57bea5c][J] | [FINAL][JFINAL]、[CERTIFICATE][JCERT]、[run分析][JRUN] |
| F：対角融合 | [94b6499e4ce3][F] | [FINAL][FFINAL]、[OPPORTUNITY][FOPP]、[全round][FROUND] |

後期研究の原標本・中間・負結果・区間証拠は、各snapshotの **RAW_DATA.tar.gz / RAW_INDEX.json** に分離保管されている。[混合公式raw tree](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/artifacts/track_c_mixed_formula/2026-11-08)、[D内部raw tree](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08)、[singleton raw tree](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08)、[共同isometry raw tree](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08)、[対角融合raw tree](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08)から辿れる。原標本数と展開member数は別の量であり、本総括のために再生成していない。

### 11.2 主要科学SOURCEの完全SHA

初期pilotは基点 `7db4792d4cd0ed9a7abc255565f207dd31a437bd` 上のdebugを含むlocal SOURCEで、公開SOURCE.jsonに最終ファイルhashと初期INPUTSの区別がある。公開commitが固定CIでそのまま実行されたという主張には置き換えない。[PSOURCE]

後期の主要段階は次のとおり。各SHAの実装treeは `https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/<SHA>/src/trottertracks`、入力・raw・完全な段階対応は各FINALから辿れる。

| 研究 | 科学SOURCE段階 / 完全SHA |
|---|---|
| 混合公式 | BCH `ba62ea5556bab33eb2fc58b4417603ac5d15f6da`、全round `42d8c95b26d74629a455611aef0d8e587a501c6b`、価格 `ce2d440f4f8199d5b35cce3426bd93c60b5beaaa`、測定意味 `3971408636dc31a9063cee4c0dd13f490eb4f73e` |
| D内部 | BCH `ed3902a118b00e5da33e323b05b3e129c181d78a`、再配分 `dfc4d475f711e3885f1dbbef6b05ec5d48fa00a4`、信号 `f935da60aa89729d6bf84b0100e7b4a27285507b`、価格/強対照 `cb445a402749920d31d4bf76142c6f59a927bf9a`、初期保証 `91f2ebeeec521259535ded3d73d61028f39df2f4`、非共鳴/測定 `30aca97b237e49e6e47a45a90393c779a22d967a`、登録比較 `7287bdc86f52309c5c48fca7f9bca857ee7960b4` |
| singleton | partial QR `fbe141d1a8febfc37d9ee415a203c132d94a497f`、区間 `6bb9897529770ae42da554c7817bc8d27c6a1538`、共同選択 `e90988881ad3f049c02c2e14919b8a13079146b3`、holdout `0801c5c9666799946354f8762fdb31a1b0a02356`。その他の保存解析・always-partial等は[KFINAL] |
| 共同isometry | pilot `2f3ddd5566d62b99c38de79e85b77dd438e06068`、4layout `bce0168fcd9b4f7a25ef9854416e05c9106c4117`、修正/区間 `b1f59fd50ffbbdd297a3fbc1d72751a509f94e24`、資源 `2a5649cadffcb5cae54fff3551ce148d054a666f`、controlled/ablation `30aa3a8f56eaab816251549b264ef31d2d063523` |
| 対角融合 | pilot `62fe68b07cd638aad12ded90bc97922ba94a5589`、資源 `87c2f29f5526d66e0d1444ece271d639366bfd9a`、保存値集計 `55f8c0e22ab2725cea252878beb8221f29be7651` |

### 11.3 機械可読値と照合上の注意

補助[RESOURCE_VALUES.csv](RESOURCE_VALUES.csv)は、主要policyの保存JSON値を転記し、固定snapshot・artifact path・JSON pointer・比較条件を併記したものである。新しい科学解析・新しい価格推定ではない。全18roundの細目は元の[登録比較][DREG]、[K round CSV][KROUND]、[J round CSV][JROUND]、[F round CSV][FROUND]を正本とする。

本統合では保存JSONのshot/総資源と原報告の表示丸め、round合計、リンク先、固定Git objectを確認した。科学moduleのimport・runner・回路検証の再実行は行っていない。主な版差は、(i)初期保証Full6914対価格固定6912、(ii)C2-B旧.547%対強化後.514%、(iii)D登録対照対menu選択、(iv)独立holdoutごとの旧方式推定値、(v)共同isometryのlegacy pooled-index SEと採用FINALの独立cell covarianceである。各節で区別し、原記録を訂正・再計算していない。

物理Hamiltonianへの無条件包含、初期長q model系統誤差、C2-B強対照最適性gap、一般的新規性・他系移送は未解決として残る。これらを資料統合だけで解消したことにはしない。

**mandatory STOP：本統合資料の公開・remote SHA/tree/内容照合後に停止する。** 次の担当はユーザー（資料の利用・次の研究仮説と実行範囲を判断）。CodexはTrack B計算、別研究、追加compile/MC、詳細GPT科学レビューへ自動移行しない。

[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/docs/research/track_c_h4_pilots_20261106/REPORT.md
[PSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/SOURCE.json
[PRESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/SUMMARY.json
[STATIC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf/docs/research/track_c_compiler_static_investigation_20261106/REPORT.md
[ACOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f45e9b07bf730f7272933b28055914cfe76fdaab/docs/research/track_c_compiler_a_correctness_20261106/REPORT.md
[A18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3258a89f225d9530fba38e9d2343bc0165dd6d7d/docs/research/track_c_compiler_a_18round_qualification_20261106/REPORT.md
[GROOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/REPORT.md
[GA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/A_REPORT.md
[GB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/B_REPORT.md
[GART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/720416c5b88b227cf76ff7aa729d91a0a98ed58b/artifacts/track_c_guarantee_cn_price
[CNPAIR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/artifacts/track_c_guarantee_cn_price/2026-11-06/B_cn_price/analysis/PAIRED_CHECK.json
[A2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad/docs/research/track_c2_a_pilot_20261106/REPORT.md
[A2RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad/artifacts/track_c2_a_pilot/2026-11-06/SUMMARY.json
[B2P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78004943ed5b718c12841e80f55fddf1d1644364/docs/research/track_c2_b_pilot_20261106/REPORT.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/docs/research/track_c2_b_strong_baseline_20261106/REPORT.md
[B2RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/artifacts/track_c2_b_strong_baseline/2026-11-06/SUMMARY.json
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/docs/research/track_c_mixed_formula_20261108/REPORT.md
[MTHEORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/artifacts/track_c_mixed_formula/2026-11-08/THEORY_AND_LOCAL.json
[MROUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/artifacts/track_c_mixed_formula/2026-11-08/FULL_ROUNDS.csv
[MFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/artifacts/track_c_mixed_formula/2026-11-08/FINAL.json
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/docs/research/track_c_d_internal_20261108/REPORT.md
[DFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/FINAL.json
[DREG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/REGISTERED_COMPARISON.json
[DNR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/NONRESONANT.json
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/docs/research/track_c_bottleneck_20261108/REPORT.md
[KFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08/FINAL.json
[KROUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08/ROUND_COMPONENTS.csv
[KTEMPLATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08/TEMPLATE_COMPONENTS.csv
[KSENS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08/SENSITIVITY.json
[WP06A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json
[WP06B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json
[J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/docs/research/track_c_joint_isometry_20261108/REPORT.md
[JFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/FINAL.json
[JCERT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/CERTIFICATE.json
[JRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/RUN_ANALYSIS.json
[JREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/REPRESENTATIVES.json
[JROUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/ROUND_RESOURCES.csv
[F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_diagonal_fusion_20261108/REPORT.md
[FFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08/FINAL.json
[FOPP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08/OPPORTUNITY.json
[FROUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08/ROUND_RESOURCES.csv
