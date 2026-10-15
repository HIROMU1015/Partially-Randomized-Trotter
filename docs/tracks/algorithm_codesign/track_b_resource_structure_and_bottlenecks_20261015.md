# Track B：資源構造とボトルネックの固定証拠整理

2026-10-15 JST。[D＋Eの明示採択](inputs/track_b_de_direction_preparation_user_instruction_20261015.txt)に基づくread-only整理。現在のParent allocation探索を区切る記録は[別紙](track_b_parent_allocation_research_closure_20261015.md)。本書は研究系列の新しい採点、forecast、実験計画の採択を行わない。**mandatory STOP**。

## 1. 比較前に揃えるべき条件

| 系列 | target・アクセス | 費用・保証の単位 | 他系列へそのまま移せない点 |
|---|---|---|---|
| BF/BM | 分子DFのPF/PR設計、固定scheduleまたはBCH adapter | 解析係数・有限objective/proxy | RTE辞書のT価格・十分shot結果ではない |
| R0/R1/R1.5/RA-D0 | 一般involution-style finite P3、ordinary/PTSC-K0/A。CTSはPauli収集I1側 | native座標・bias込み十分N・証明付きLP計画 | 小型明示Vは実際にはPauli展開可能。I0 acquisition advantageは実証していない |
| SP-0.5/SP-1 | primitiveまたは2-qubit coherent wrapper | J、累積moment/rangeと十分N、加法的合成primitive価格 | 実DF/RTE、compiled wrapper、分子PRの結果ではない |
| G6–G8 | finite polynomial first operator moment、局所query/CQ provider | zero-fillを含む全trial予算・記号的provider価格、短い取得trace | provider取得・全normalizerを無料にできない |
| G9/G10 | 明示3-qubit providerの同次数finite target | strict native primitive価格、same-m比較、tail/hard cap | mの異なる行は同じ理想exp精度比較ではない |
| D1–D3 | 同じ物理時間\(\tau=5/7\)、r=1,2、finite P5近似 | Taylor/finite-bit/native biasを戻した十分N、加法＋D2境界簡約 | familywise規則は各12行ledger、D3四ledger全体で0.05ではない |
| Parent allocation S4 | 同development P5/r=1、ordinary/P5の配分とIS | 一つのcomplex signalのconfidence、h付き70 law | D系列の多row familywise policyと異なる。h=0やr=1からPR全体を断定しない |
| DF補正 E/T | 少数modeの三重交換子回転とPauli補正 | 局所native価格・方式別bias・weight、別CD系列のsignal予算 | A/Bは有限step平均自体も異なる。CD頻度×別E単価は実測PR総費用ではない |
| H4基底診断 | 保存8列、controlled RTE部分回路 | component別加法RZ/CX/sizeの構成比 | T構成比・RTE分布の期待構成比・PR総費用ではない |

情報アクセスのI0/I1やCQ記号は各契約内の定義へ戻す。同じ文字でも「明示Pauli取得可能」「queryだけ」「列挙oracleだけ」の違いを消さない。native1QはTを含み得るため、T＋native1Qのような重複加算をしない。古典setup、合成取得、online sampling、実量子shotは別の費用である。[G〜D統合ノート][G-D-NOTE]／[R0.5のアクセス監査][R05]。

## 2. 主要研究系列で確認されたこと

各項目は「問い／変更／数学的根拠」「対照・条件」「結果・資源」「限界・既知性・次の設計への含意」の順に整理した。数字は固定資料からの転記であり、新しい数値分析ではない。

### BF：高次scheduleを変える価値

問い・変更・根拠：固定5-stage第四次係数familyでscheduleの自由度を追加し、BCH/PF誤差proxyと資源を改善できるか。Fと制限Lを同情報・各32評価で比較した。

条件：linear H4、1.00 Å、STO-3G、8 system qubits、DF rank12、\(L_D=3\)（決定論generatorはone-body込み4）、T=0.8、q={1,2,4,8}、Rbud={5,10,20,40,80}、K=2（triggerで4）、ε=.01/α=.05。

結果・資源：F/Lは同じSuzuki5、q=1/R=10/K=2、proxy約20,709,937で一致し、共通reference nativeS2 q=2/R=5/K=2の約12,924,223を下回らなかった。固定family/評価予算でdecision-relevant extra gainが見られないという限定結果。原runの未完了INCONCLUSIVEと保存後回復BFAを別記した。

限界・示唆：compiled RZ/Tの総費用評価ではなく、全scheduleのno-goではない。既知Suzuki構成との差分を先に確認し、解析proxyの低下をnative改善と混同しない。[BF closure][BF]。

### BM：誤差評価を変えることとアルゴリズム差

問い・変更・根拠：BM adapterのcubic係数・aggregationで新しい設計差があるか。formal比較で既知compact BCHと一致し、BM floor＋KA/(4m²)の同じDF backend/aggregation/cache下で同scoreになることを確認した。

対照・資源・結果：既知compact BCHが直接対照。normを取る位置・解析policyを変えること自体は新しい時間発展アルゴリズム差ではなく、BM-1は新method実験として実行しなかった。

限界・示唆：数学的同値性による主張範囲の整理であり、全PF改善の性能negativeではない。新規計算前にoperatorと生成手順の同値性を閉じる。[BM監査][BM]。

### SP-0.5 → SP-1：安いprimitiveと累積weight

問い・変更・根拠：既知PAIで離散角を選び、合成T費用とweight二次momentを交換する。SP-0.5はkπ/4一catalogue、8 target、ε=1e-6、23 literal synthesis keys／16 primitive rows。deterministic baselineにもexact fast pathを共通適用した。

結果：π-rationalと±1/5 radを含む14 strict primitive witness、zero-cost control1、J=1 control1。controlled pairは独立primitiveのmoment積を含む。これは実primitiveの限定trade-offである。

SP-1：ancilla＋systemの2-qubit wrapper、A/B/C template、n={8,16,32,64}、NONE/D/R/DR、complex ε=.05、family α=.05、5% materiality。48行中42 eligible、6 shot-cap。A/BではD=DRが短いnで利益、n32で反転。CのD-only/R-onlyにmaterial gainはなく、DRが短いnで利益、n32で反転、n64はcap。加法的synthesized-primitive TとBernstein十分Nの結果である。

限界・既知性・示唆：既知PAIの実装検証で、D/Rの角度・個数が交絡するtoy。fusion候補0は確認したがcompiled wrapperの最適費用ではない。primitive J≪1を多gate・多stepの利益へ移すにはmoment、range、bias、N、準備費用を戻す必要がある。[SP05][SP05]／[SP1][SP1]。

### R0/R0.5 → R1/R1.5：normalizationとodd/native費用

問い・変更・根拠：同じfinite Taylor meanでidentity/odd coefficientの再配分によりnormalizationを下げる。Euler/common-angle等は既知だが、読んだPTSC/CTSの直接parameter選択と一致しないrestricted構成差を監査した。I0生成O(m)はPTSC-K0に対する漸近取得優位を意味しない。

条件：2-system-qubit synthetic finite P3、\(R=.75Q_0+.25Q_1\)、\(Q_0=Z_0,Q_1=V^\dagger Z_1V,V=e^{-i\pi XX/16}\)、controlled、x={1/8,1/4}、σ±、native精度1e-3/1e-4/1e-6、complex ε=.01、family α=.05。対照ordinary、同I0のPTSC-K0、Pauli側CTS。

結果・費用：AのB²改善はnative rotation、odd Q・relative phase、basis/conjugator費用を払う必要がある。x=1/4/precision1e-3で保存G_T比はordinary比約.79628、PTSC比約.82349だが、これはfixed precision。精度選択を全方式へ許したR1.5では単一座標・Paretoの関係が変わり、A全座標優越ではない。B²が減ってもnative bias増加でNが増える例がある。

限界・示唆：明示2-qubitなのでPauli展開取得可能、DF規模I0/I1境界の実証ではない。合成priceだけでなくcoherent biasと統計幅の接続が必要。σ±を独立再現と数えない。[R0][R0]／[R0.5][R05]／[R1][R1]／[R1.5][R15]。

### RA-D0：LP certificateの未完了と科学比較を分離

問い・変更・根拠：同有限P3・同resource constraintsで、既存representation全体の混合B2よりdegree-local再配分B3がcertifiedに改善するか。v3は上記R1保存表・x={1/8,1/4}、新しい量子回路取得なしのLP契約だった。

結果：最初のB2 T minimumがnumerical membership certificateを満たさず停止、B2/B3 certified comparisonは0。backend v4 pilot v2はSoPlex/GMP小型exact証明のprefixを取得したが、極小gap fixtureでERROR、B2/B3-shaped six LPは未実行。raw分類と後の監査分類PARTIALを区別した。

限界・示唆：技術的未完了からB3の優位・非優位を判断できない。小型LPの成功をproduction solver保証へ拡張しない。G系列の別のfixed-class証明と、未実行のD0全比較を同一視しない。[RA-D0実行][RA-D0]／[v4 backend pilot][RA-V4]。

### G5/G6：class証明とlocal return生成

問い・変更・根拠：G5はrestricted degree-local classとPauli CTSの関係、G6はreturn係数の局所queryとEuler pairingでgeneral odd-degree有限targetを生成する構成。隣接同一ラベルの消去だけを使い、無料のPauli全収集は前提にしない。

結果・条件：G5の2-qubit toyにおける固定B2 classの証明は、その列・budget・precision policy内のもの。G6は係数identity、short-step positivity、Green再帰型のlocal query、未知normalizer下のordinary-envelope rejection/zero-fillを接続した。

資源・限界・既知性：G6のsampling lawは理想式でacceptance=Bfull/Bord、二次moment=Bord Bfull、range=Bord。accepted平均へ無料で再正規化できず、全trial分母で保証する。Green再帰・Euler pairingは既知。任意algorithmへのlower boundや新しいnative利益の証明ではない。[G5][G5]／[G6][G6]／[先行研究対応][G6-ART]／[構成技術ノート][RA-THEORY]。

### G7/G8：provider費用とon-demand取得

問い・変更・根拠：局所生成をstrict phase Rz取得とcontrolled-Q providerへ接続し、情報を全列挙しない経路を確認する。G7/G8はP3 control p=(3/7,4/7),x=2/5 とP5 p=(1/5,3/10,1/2),x=5/7を用いた別の既知例。

結果・資源：G7はnative Rz＋CQ/preparationの記号係数を保持。G8は8行×cold128/warm128＝2048 interface trials、20 key live missを保存し、固定runtimeの取得/CPU記録を残した。frequencyからmoment/N/期待費用は推定していない。

限界・示唆：CQ物理providerの実費用を取得したわけではなく、大Lのonline母集団費用は`NOT_ESTABLISHED`。既知例の短いtraceはproduction scalabilityや独立validationではない。一方「Track Bでは古典時間を一切測定していない」とするのも誤り。アクセス・setup・miss・hit・実装queryを区別する。[G7][G7]／[G8][G8]。

### G9/G10：nativeへ戻したreturn集約の成立域

問い・変更・根拠：G9v2は明示3-qubit P5 providerでsix direct lawを同target比較。closed P5は少数群normalizerを利用、general local fullはzero-fillを残す。G10v3はm=3/5/7のsame-m比較17行と固定辞書lowerを取得した。

条件・結果：G9はp=(1/5,3/10,1/2),x=5/7、strict Rz ε=1e-6、finite target confidence family .05。closed P5は登録canonical direct表のT切片とprep係数が小さいが、CTSはCXが小さい。G10のgeneral fullは登録T会計でclosed対照に不利、m7のfixed-dictionary/policy lowerにも限定した不利がある。

限界・示唆：generic helperとdirectでworkspace/native費用が異なる。同じ理想full lawでもnormalizer取得が十分Nを変える。一般fullの性能探索は区切り、closed P5の構成を保持した。mの異なる行から同じexp精度の最良次数を選んだ結果ではない。資源停止・保存エラーを伴った旧v1/v2はv3成功と別記録。[G9v2][G9]／[G10v3][G10]／[既存endpoint][G10-END]。

### D1/D2/D3：表現・保証・価格の分離

条件：G9の既知3-system-qubit synthetic provider、p=(1/5,3/10,1/2)、物理総時間τ=5/7、r=1,2、各step finite P5、初期状態000。分子/geometry/basis/DF rank/\(L_D\) は非該当。六方式はordinary、partial-return、P3+tail、general local full、closed P5、matched CTS。native strict精度1e-6、各12行ledgerでcomplex ε=.01/family α=.05。D3四ledger全体の同時0.05保証ではない。

D1：1890event/36keysをnative価格へ戻した。P5対partialのT差はr1約−1.16%、r2約−5.84%。前者は主に十分N、後者は主に回路単価の差で、normalization低下だけを理由にできない。

D2：同じ保存IRに全六方式共通のliteral inverse境界相殺を適用、219,303 pair。law、価格、N、biasは固定。r2のP5対partial差は約−5.86%、二lawのCX費用h交点は約3.515→6.637。全回路compilerの新規最適化ではない。

D3：旧/改善Taylor保証×旧1e-6/新1e-7合成価格を分離した48行。全四ケースでnative誤差会計は旧1e-6に固定し、price-onlyでNを減らしていない。Hermitian contractionの\(|x|^6/720\)保証は旧上界より鋭く、保証変更はr1の十分Nを約29.5%、r2を約.93%下げ、最良rを変えた。全12構成のT最小はG0P0=r2 P5、G1P0=r1 P5、G0P1=r2 P3+tail、G1P1=r1 P3+tail。

G1P1のr1 P3+tail対P5差は保存値約598,967 T、約0.170%。代表root key価格はP5 ratio 2189372/2921709が68→80、P3 ratio310/413が70→78。既存key incidence分析が順位逆転の寄与を説明する。primary partial/P5ではP5のT利益を保持するが、全六方式のbestとは別である。

限界・既知性・示唆：Euler/Green/IS/CTSは既知原理、有限表現・保証・実合成価格の条件付き比較が取得内容。rを変えると有限polynomialは異なるので、同じ理想expへのTaylor biasを戻す。小さな価格差を一般的P3優位としない。D2保存深さ分布はあるが、深さごとの完全費用帰属やwhole-circuit費用は未取得。[D1][D1]／[D2][D2]／[D2最終分析][D2-FINAL]／[D3][D3]／[key機構分析][D3-MECH]／[先行研究対応][D-ART]。

### Parent allocation S4：角度辞書を変えた有限比較

問い・変更・根拠：parentのchild別配分で同じfinite P5 meanを保ち、commonで最小だった係数L1を増やしてnative価格を下げられるか。ordinary/P5にも同じcanonical・適用可能analytic IS・三regularized機会を与えた。

条件・結果：同development synthetic、r=1、strict1e-6、一つのcomplex signalのα=.05。50旧control＋20訂正KKT/halfway＝70 law。best-common対best-noncommonのh0T差約14%/19%、N/CX増加、価格とhに条件付く。旧raw-C proxy20は正式訂正集合から除外し、履歴を保持した。

限界・既知性・示唆：共通角L1最小性・任意配分mean保存は既知理論の直接帰結、KKTは固定price proxy。内部familyのnative最適性・新規性・多step利益は未確立。今回はD＋Eを明示採択して現在の探索を区切る。詳細は[新整理記録](track_b_parent_allocation_research_closure_20261015.md)、[正式Evidence Packet][PA-PACKET]、[完成版レビュー][PA-REVIEW]。

### DF sparse correction：local wholeとPauliの実装差

問い・変更・根拠：三次BCH補正の三重交換子を最大6軌道へ縮約し、whole回転Aを局所Pauli補正Bと比較。Aの条件付き補正weight1と、Bの軽い回転/weightの交換を調べた。NCC/HNCC等との重複は先に監査した。

条件：Eは3/6 modes、rank2×3、seed7101107、sign(+,-,+)、θ=1/32、κ=4、u=1/128、uncontrolledまたはMSB control。Qiskit1.3.0 rz/sx/x/cx opt0、seed20261011、native2e-7＋synthesis8e-7。分子geometry/basis/\(L_D\) は非該当。

結果・費用：22 compile、39T keysの保存記録。固定加法planでwhole Aのcompletion lowerは3 unctrl≥576対B112、3 ctrl≥692対B236、6 unctrl≥576対B116。6 ctrl Aはscalar-phase未対応でT未確定、B232。旧v7 rawの保存上限停止と、保存後会計完了の監査状態は分ける。

限界・示唆：A指数平均とB weighted線形平均は有限stepで異なり、方式別biasが要る。whole最適compilerのlowerではない。別CD（3mode3block .45/.35/.25、seed6121107、t1、δ={1/8,1/16,1/32}、γ1.25、s1、P5、ε1e-4/α.05）の頻度とE単価を掛けて実測PR総費用としない。generic whole主線の終了を維持するが、全whole法のno-goやPauli全PR優越ではない。[DF closure][DF]／[保存会計][DF-COMP]／[既存法対応][DF-ART]。

### Gaussian積圧縮：構成差の未確立

問い・変更・根拠：reflection productをactive spanに集め、number-conserving Gaussian基底変換でproduct生成費用を抑える候補。既知Gaussian/partial-basis構成と同じか、追加手順があるかを静的に監査した。

結果：数学的な部分構成を保持する一方、現手順を独立新アルゴリズムとして採択しない終了判断を記録した。期待資源の改善/非改善は`NOT_ESTABLISHED`で、性能negativeではない。

限界・示唆：古典O(d²)処理を不可避な量子lowerにしない。WとZを別々に実装する二重費用を最良対照へ強制しない。既知reflection/active-span compilerの別名を次候補として再提示しない。[Gaussian監査][GAUSS-AUDIT]／[終了記録][GAUSS]。

### Interstep correlation：平均を保つという条件

問い・変更・根拠：連続stepのラベル相関でbasis切替を減らせるか。既存監査はPRHS、qSHIFT/Adaptive sampling、cost-aware ISと対応付けた。周辺分布が同じでもordered-productのfirst momentは保存されない。

結果：full-path P/q重みで平均を保つ条件、weight二次moment・range・履歴依存biasの論点は整理済みだが、独立した具体的な方法差と性能利益は未取得。channel/observable精度の既知法とcontrolled coherent finite-RTE targetを無条件に同一視しない。

限界・示唆：古いIID cost式を新しい相関lawへそのまま移さない。単一shot±1のmeasurement noiseを落としてexact path平均だけの分散を最適化しない。既存監査のqSHIFT版差・表示式の未解消箇所は残す。[相関乱択の既存監査][CORR]。

## 3. 何に費用を払っているか

### 3.1 回路単価：basisと回転価格を分ける

H4にはbasis費用を直接分けた既存の小型診断がある。linear H4/1.00 Å/STO-3G/8 system qubits/DF12/\(L_D=3\)、δ=.02、paired K2、controlled RTE部分9qubits、保存WP06-b holdout列長3/6を各4列。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、結合制約なし、full-basis/shared-adjacent policyである。

| 既存診断の加法的分解 | basis / additive total | 保存構成比 |
|---|---:|---:|
| RZ count | 4109 / 4844 | 84.8266% |
| CX count | 1120 / 1624 | 68.9655% |
| 全native size | 8199 / 9568 | 85.6919% |

**全36eventはTaylor order0だった。** 稀なorder2や分布全体の期待費用を覆わない。component個別compileの加法比であり、一体最適化後のgateに由来を付けた分解でもない。whole compile側の保存RZ4465/CX1572/size8940とは分母が異なる。T比率は`NOT_MEASURED`。85%を削減可能量や全RTE母平均へ読み替えない。[診断報告][BASIS]／[保存表][BASIS-ROWS]。

原phase検証はstrict差でtechnical inconclusiveだったが、後の保存IR監査は両ancilla枝に共通global phaseというcaveat付きで費用帰属を整理した。statusは`SAVED_COST_ATTRIBUTION_COMPLETE_WITH_COMMON_GLOBAL_PHASE_CAVEAT`。今回そのoperator監査を再実行していない。control前のsystem global phaseや追加control時のrelative phaseを捨ててよいという根拠にはしない。[保存status][BASIS-STATUS]。

D3/S4はbasisが小さい3-qubit providerであり、root回転keyの価格が順位へ効く。H4のRZ構成比と同じT構成比ではない。RZ個数は角度ごとのT価格を含まず、Clifford端点とgeneric回転で単価が大きく異なる。controlled回転の分解、ancilla phase、inverse順序は費用と誤差の両方へ接続する。[D3-MECH][D3-MECH]／[PA-PACKET][PA-PACKET]。

D2のliteral inverse簡約と、DF/Gaussianのactive-span圧縮、whole-circuit compiler最適化は異なる操作。保存費用を単純加算する前に何が既に共有・相殺済みかを固定する。H4の古いhierarchical holdoutでも個別event和が一体compile費用とずれることが確認されているが、そのδ=.1/short-step .025/K条件・局所誤差は新しいS4やH6へ移送しない。[階層compiled-cost検証][RTE-COST]。

### 3.2 sampling：L1だけで十分Nは決まらない

係数L1、\(E[W^2]\)、range、biasを区別する。canonicalの一定絶対weightではL1²がmomentへ直結するが、IS、zero-fill、finite-bitでは実lawとweightで会計する。多step独立積ではmomentが積になり、稀な大weightはrange項を増やす。ISのleading proxy最適性はbias、ceil、zero T event、hを含む総費用最適性ではない。[RA-THEORY][RA-THEORY]／[PA-ART][PA-ART]。

G系列のlocal fullはnormalizerを知らず生成できても、真のacceptanceを無料利用してNを削減できない。closed P5はnormalizerを少数群式で取得できるため、同じ理想ensembleでもtrial分母と十分Nが変わる。P5のO(L²)は群のsetup算術scopeであり、現在のgroup selector全体がO(L)とは主張しない。[G9][G9]。

R1はnative bias、SP1はmoment/range、D3はTaylor保証、S4は価格とzero-costを含むproposalが効いた。それぞれN増減の理由が異なる。十分Nは同じconfidence関数の上界で、実測量子shot数でも必要最小Nでもない。[R15][R15]／[SP1][SP1]／[D3][D3]。

### 3.3 全実行：受理率・準備費用・古典費用

D系列の保存モデルは、資源jに対して

\[
G_{i,r,j}(h_j)=2N_{i,r}\left(r Z_i^r d_{i,j}-S_{i,r,j}+Z_i^r h_j\right).
\]

\(d\) はaccepted step単価、\(Z_i^r\) は全step受理率、\(S\) はD2の期待literal節約額。zero-fillは事前に棄却でき、hはaccepted full circuitに課す。S4はr1/all-acceptなので \(G_j=2N(E C_j+h_j)\)。hを何回課すかまで異なり、同じh記号だけで系列を足し合わせない。[G-D-NOTE][G-D-NOTE]／[D2-FINAL][D2-FINAL]／[PA-PACKET][PA-PACKET]。

主対象のPR/RPE/QPEでは、決定論部分との交互配置、roundごとの発展時間、RTE microstep、state preparation、readout、confidence、稀なTaylor次数まで戻す必要がある。元PR論文はその全体taskを扱うが、Track Bの固定toy表は全項目を同じ実用入力で揃えていない。**今回整理した資料だけから、実PRのどの要因が総費用の最大項かは`NOT_ESTABLISHED`。** H4診断の基底割合をその代用にしない。[既存RTE sourceのscope][RTE-SOURCE]／[資源モデルholdout][RTE-COST]。

| 古典・取得費用 | 取得済み | 未取得・移送不可 |
|---|---|---|
| setup/validation/compile/synthesis | 各runの固定runtime・CPU/RSS・call数、D3/S4 key取得記録 | 将来の量子shot反復へ一律に掛ける費用ではない |
| on-demand sampling/cache | G8の短いcold/warm trace・miss/hit・局所query経路 | 一般入力・大L・実量子N回でのonline平均/最悪費用は`NOT_ESTABLISHED` |
| S4/D系列online | quantum resource mapとlawは保存 | classical online costは`NOT_MEASURED` |
| 情報取得 | 小syntheticでの明示Q/Pauli収集、symbolic CQ係数 | 分子DF規模の列挙・normalizer・Pauli情報取得優位は`NOT_ESTABLISHED` |
| 最適化 | 一部literal boundary簡約、固定compiler結果 | whole-circuit最適化・provider最適化・FT scheduling全体は`NOT_MEASURED` |

## 4. 横断的に残る具体的な制約

| 証拠の型 | 固定証拠 | 新しい設計で確認すべき条件 |
|---|---|---|
| 数学的同値性／方法差未確立 | BM、Gaussian、Euler/IS既知部分 | operator、law、情報取得、native手順のどこが異なるかを特定する |
| 単価利益と統計費用の交換 | SP、R1、S4 | 単価×十分Nにbias/range/hを戻す。momentだけ・Tだけの全座標優越にしない |
| 価格・精度・保証への依存 | D1–D3、S4 | 誤差保証変更、price変更、law変更の効果を混ぜない |
| 実装構造による費用差 | D2、H4 basis、DF E | 加法的モデルと一体compile、部分回路と全taskの範囲を固定する |
| 情報アクセス／取得費用 | G6–G9、CTS対照、R0.5 | 非列挙生成やPauli収集を無料oracleにしない |
| 技術的未完了 | RA-D0、旧G10・DF保存停止、S3旧proxy | scientific negative・数学的同値性とは別に記録する |

いずれも新しい一般定理や科学的優先順位ではない。未解決の具体的な技術課題は、実taskでのbasis/T内訳とrare-order被覆、meanを保つpath生成、制御位相付きnative費用、非列挙の情報取得とbit費用、whole-circuit費用、online費用である。どれを研究RQにする価値があるかは[Decision Packet](track_b_next_algorithm_research_decision_packet_20261015.md)から、開始承認後のGPT判断へ渡す。

## 固定資料の参照先

参照IDのcommit・path・bytes・SHA256は[取得索引](../../../artifacts/track_b_de_direction_preparation/2026-10-15/v1/evidence_index.json)に記録した。リンク先はbranch先端ではなく固定commitである。

[BASIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/docs/diagnostics/rte_basis_cost_attribution_20261013.md
[BASIS-ROWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_v1/resource_rows.csv
[BASIS-STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_v1/summary.json
[BF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[BM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[D-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D2-FINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[D3-MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md
[DF-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md
[DF-COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md
[G-D-NOTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-14_track_b_return_aggregation_G_D_research_note.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[G10-END]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-11_track_b_return_aggregation_endpoint.md
[G5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G6-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[GAUSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md
[GAUSS-AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md
[PA-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
[PA-PACKET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[PA-REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[R0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md
[R05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md
[R15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
[RA-D0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/ra_d0_one_shot_result_validation_20261007.md
[RA-THEORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
[RA-V4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_gpt_handoff_20261009.md
[RTE-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/hierarchical_cost_validation.md
[RTE-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py
[SP05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md
[SP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
