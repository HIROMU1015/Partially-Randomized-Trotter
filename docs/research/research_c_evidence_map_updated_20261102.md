# 研究C：PR-aware DF 最新Evidence Map・成果統合

2026-11-02 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。Codex-Aによる既存証拠の整理。共通起点は`46f1461d213ad73f8e187ef98e1491ef8b522aaa`。

## 1. Executive Summary

**物理DF予算を保持してPR tailを軽くし、PR内部の条件付き総RZを削減できた。一方、同じ新J/Uの強決定論S2より主RZは高い。** この二つを同時に保持することが、現在の証拠の要点である。

| 確認できた成果 | 確認範囲・残る制約 | Claim |
|---|---|---|
| 同一初期値の通常CDF γ0とPR-tail γ.1で、λを2.20976184→1.44242989 Ha、約34.72%低下させた | H4・CDF8、各1 fit、両DF予算PASS。Optimizer successは一次収束を意味しない | C01 |
| Reduced経路で新たに5回のU更新を受理し、選定点λ=.56834985 Haを得た | 全受理点DF予算PASS。U一次未達の有限endpoint。5回の開始点は旧19-call受理点で、Stage A終点そのものではない | C03・C04 |
| 旧PRから新PRへ、ideal条件付き総RZが234117788135→207919150188、約11.19%低下 | s1/r128/K0。W低下とkernel価格上昇の交換。有限32件価格校正・理想誤差仮定 | C05〜C08 |
| 同じ新J/Uで強D総RZ105532782860、新PR207919150188、PR/D≈1.970 | 主RZではDが低費用。総CX depthはPRが小さく、全指標支配ではない | C09 |
| Support/rank/proxy/融合/round/lawの限定的な負結果と同値性を整理できる | 最新の技術判断は`TECHNICAL_NO_GO_CURRENT_EVIDENCE`。研究C全体の終了・PR一般の不可能性ではない | N01〜N10 |

[10月31日版Evidence Map][E0]のE01〜E23を歴史的な基盤とし、11月1〜2日の追加証拠を本書で接続する。旧版の値・判断・未資格事項は編集しない。旧版の中心結果であるs4/r16の改善も成立範囲を維持し、今回のs1/r128の改善率と交換しない。**独立新規性・論文化GO・研究終了は本書の判定対象に含めない。** Codex-Bは本書のClaim ID・条件・資格を用いて別途主張候補を照合できる。

## 2. 研究目的と対象範囲

中心的な問いは、**DF表現をPR向けに最適化し、物理Hamiltonianの近似精度を保ちながら量子資源を削減できるか**である。証拠は次の順に接続する。

`目的・J/U更新 → 独立DF誤差 → exact λ → PF／補正RTE／attenuation → RPE shots・W → native価格 → shell込み資源 → 同J/U強D`

前段のPASSだけで後段を認定しない。最適化器の正しさ、登録一次条件、DF予算、条件付き資源改善は別の判定である。

### 2.1 共通条件と比較単位

以下の条件コードは§8の各行にも適用する。条件差を省略するための記号ではない。

| コード | 固定された条件 |
|---|---|
| H4 | 直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36。8 system qubit/full256、ancilla込みfull512。通常number CDF、signed実対称J・実直交U、元g補正κ・核scalar一回・元spin/frame順 |
| A8 | H4・rank8、共通保存初期J/Uを修正real-Schur codecへ一度通した同じqualified x0、γ0/.1各1 fit。ffsim0.0.83・L-BFGS-B・options=None。tol1e-8はffsim初期化側で、SciPy停止tolの指定ではない |
| P128 | 指定候補の元順LD2・partial ordered symmetric S2、s1/r128/K0、δ=h=.1、m0..12、coherent回数2^m |
| P4/16 | A8最終2候補、LD2・partial S2、s4/r16/K0、δ=.1/h=.025、同13round。P128とは別条件 |
| D1 | 指定候補のκ＋8 frame全9因子ordered symmetric S2、s1、δ=h=.1、m0..12。PR分割LDやr/Kは非適用 |
| R13 | CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、候補ごとのfailure=.01、X/Y各round軸=.01/26。各DF自身のground-state oracle、状態準備費用除外 |
| P0/N32 | Qiskit1.3.0・opt1・seed17・rz/sx/x/cx・approximation degree1・qubits_initially_zero=false・MPFR160 declared phase・位相fitなし。PRは各候補32独立finite-bit/PCG64 trajectory、非paired cohort。Dは固定kernel1件 |
| 会計 | 既存head内部relative融合・tail厳密basis_key共有、D全sweep relative融合。Fenced serial反復＋interrogationごとのancilla head/X/Y shell。未観測のcross-boundary／全長追加compile削減を仮定しない |

旧版§2・6.3の条件・資格辞書を再利用する。H2/H6の通常CDF/XDF・codec検証は基盤資産であり、今回のPR-aware目的の分子間汎化試験ではない。候補別failureを全探索・選択後のfamilywise保証に読み替えない。科学callは登録logical unitで、内部iteration/nfevや量子shotsと同じ単位ではない。

### 2.2 候補identityを混同しないための対応

SHA256は保存metadataの引用で、今回配列をdecodeして同一性を再評価したものではない。

| 候補・入力 | SHA256・来歴 | 使用する比較 |
|---|---|---|
| 元P-H4 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` | 元入力来歴 |
| 共通h/g/scalar snapshot | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` | Stage A以降の保存物理入力 |
| Stage A γ0最終 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` | 初期paired比較、第三DF |
| Stage A γ.1最終＝旧PR | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` | M2/M4、旧round選択、新旧PR比較の旧側 |
| 初期RC2・17-call fit終点 | `4972174836a3fe33897a1bcdb922645596f45c8e28141ddafcdf1ba5a4a96ee6` | 初期L2対照。後の共通政策RC2とは別 |
| 共通収束政策RC2終点 | `d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798` | RC2/PR Stage B・shots・native限定比較 |
| 同政策PR終点 | `3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21` | 上とpaired fit由来だがnative cohortは非paired。旧PRや新selectedとは別 |
| 固定U LP後 | `eb064fcae6dae771fed3e07cc447530023cd53646726dbe30ddb56a4bc8279ca` | 同政策PRのtail係数再分解だけ |
| 最新selected update005 | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` | 統合PR性能と新強Dの共通J/U |

上記以外のP4/CDF7/歴史的CDF8/XDF8 identityは旧版§2.3・§11に保持される。λ=.95454のPR、λ=1.44243の旧PR、λ=.56835の新PRは別候補である。最新selectedの保存先は`artifacts/research_c_pr_integrated/2026-11-02/run01/selected.npz`。

## 3. 主要な肯定的成果

### 3.1 初期paired fitと下流接続

Stage Aの目的は`Fγ=f/f*+γ(Mε/M*)²`、`f=½‖g−g_DF‖²_F`。MεはLD2以後の非identity Z/ZZ係数に対する`Σ√(c²+ε²)`、ε=1e-8 Haである。元のf*=2.1839632053132894e-9 Ha²、M*=2.2093719370722074 Haを固定し、identity/scalarは位相として保持する。Exact λ=Σ|c|とsmooth目的を区別する。[SA]

| H4/A8終了値 | γ0 | γ.1 |
|---|---:|---:|
| Exact tail λ [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| Sector operator norm [Ha] | 7.018855038843621e-9 | 9.039161828383144e-9 |
| Norm＋engineering guard [Ha] | 1.0297636237465247e-8 | 1.229561498918692e-8 |
| DF予算 | PASS | PASS |

両方の初期値をそろえた各1 fitで、登録λ削減率34.72464491113348%を得た。γ.1の物理残差はγ0よりわずかに大きい。これは精度をさらに高めた結果ではなく、共通予算内でtailを軽くできた結果である。30/30 COMPLETE、元optimizerの相対目的減少successは一次収束を認定しない。

この2候補のP4/16では全13roundでA・補正RTEが改善し、ideal十分shots53136→19574、条件付き総RZ3194333274946→540290160109という既報成果がある（旧版E02〜E05）。83.08598%という削減率はこの比較に属し、P128の11.19%や同一新J/UのD比較へ移さない。[OLD-C1]

### 3.2 追加U更新と物理適格endpoint

最新の統合実験は旧19-call受理点を初期Uとする独立実験である。そこで新たに5回の更新を受理した。開始λ=.725819387316→update005の.568349854686 Ha、F=.0107924679369→.00661750979453。全受理点のJ一次・reduced数値資格と独立物理DF予算がPASSした。A＋B初期＋5受理点の7物理評価もすべて予算内だった。[I]

Selectedのtensor Frobenius=3.29090997756e-9、sector norm=5.23337351802e-9 Ha、guard=3.22739270291e-9 Ha、和=8.46076622093e-9 Ha。元の旧PR λ=1.4424298855756243に対する低下は約60.60%である。これは**Stage A通常CDFとの34.72%比較から独立した、追加最適化後の有限endpoint比較**である。

## 4. 最適化アルゴリズムの成立範囲

旧版時点で未取得だった「外側Uを数値資格付きで進められるか」には進展がある。ただし正しい勾配の厳密上界、J/U同時一次収束、最適RC2との公平な収束後比較は未達である。全経緯の再掲は避け、次の接続だけを残す。

| 検証 | 主要な到達点・停止理由 | 固定科学SOURCE／原本 |
|---|---|---|
| 保存点10-call | 旧勾配差2.37552e-7を再現。同じdecoded J経路で9.18462e-18、元allowance内。単一保存点の数値整合 | `565de95fc3ba4d19feaf25a17cbeecde69977a48`、[報告][G10]・[JSON][G10J] |
| Inner8-call | J∞=9.83543e-6で登録一次PASS。一方Newton-CG status2 precision loss、U₂≈1204.25、outer更新0 | `ab5d80bd64eeb82bab9dd8f11ef0a6c7c9a838bf`、[報告][G8]・[JSON][G8J] |
| Reduced18/54→guarded60-call | 初回trialのstability未資格を棄却。後続8trialでも未資格2・適格だがArmijo不成立6、更新0 | `b2e5ab4d5ff6986e0e991b7aa0d373ecdbf755c5`／`d03351ea00a9352b1d637e672c0dc679201b523a`、[18報告][G18]・[JSON][G18J]、[60報告][G60]・[JSON][G60J] |
| 局所49→小step19-call | 3つの正方向小stepでF低下を観測。その後の独立19-callで1更新を正式受理 | `e06913a234999d9f42df8a5aabd917f9510a2ac7`／`a426d62f27182526baa9479cd7cf8a953f92a9dc`、[49報告][G49]・[JSON][G49J]、[19報告][G19]・[JSON][G19J] |
| 統合543/1759-call | 新U10点・inner30（refinement20）・5受理。次方向の3登録stepが値尺度Armijoを満たさず`FINITE_STEP_GRID_NO_ACCEPT` | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`、[I]・[親JSON][IJ]・[STOP][IS] |

最新inner raw statusは0が28件、2が2件、登録J基準は30件MET。Raw precision loss、J一次条件、latest-pair stability、rank/nullspace、reduced-gradient admissionを独立に保持した。Jが一次閾値を満たすことだけでreduced数値資格を与えたわけではない。更新ごとにchartを作り直し、cold Jからsolve＋2 refinement、独立終端、物理評価へ進んだ。

選定点のJ∞=9.5932671593e-8に対し、物理U₂=.658935814277は閾値1e-5に未達。登録3stepの探索不成立は、より小さいstepの降下不存在・方式の数学的不成立ではない。`true_reduced_gradient_qualification`、`certified_descent`、`actual_CA_achieved`は親JSONでnullを維持する。[IJ]

RC2は別系列で、旧stable実行の全75点が数値admissionを通り64更新でF=.02919382875→.01742887277まで低下したが、U₂=.0047620784で未達。外側BFGSはiteration上限終了である。この終点の新DF物理誤差は親STOPにより未評価で、以前のDF PASSを移送しない。PR修正後の最新5更新と、RC2の最適化済み性能の勝敗を作らない。[RC-STABLE]

## 5. PR性能・総資源比較

以下は旧PRと新selectedを、それぞれ固有のHamiltonian・ground・law・PF/RTE・shots・nativeで評価した既報値。両者ともH4/P128/R13/P0、価格は各32件の非paired cohortである。[I]・[B]

| 指標 | 旧PR | 新PR | 解釈 |
|---|---:|---:|---|
| Exact tail λ [Ha] | 1.4424298855756243 | .5683498546862645 | 約60.60%低下 |
| m12 PF operator誤差 | .071614997914 | .094546804594 | 悪化。全13roundでも新側が大きい |
| m12補正RTE operator誤差 | .060945308518 | .012063601599 | 改善。全13roundでも新側が小さい |
| m12 attenuation A | .716844437765 | .949629430088 | 増加。Raw信号の絶対値そのものではない |
| Ideal総shots/X＋Y | 16056 | 14048 | 約12.5%減、実量子測定回数ではない |
| Kernel重みW | 16631888 | 10487392 | 約36.94%減、round別coherent回数で加重 |
| Kernel平均RZ | 14076.4375 | 19825.625 | 約40.84%増、母平均差の保証ではない |
| Shell込みideal総RZ | 234117788135 | 207919150188 | 約11.19%減、条件付き点推定 |

固定K0では`A_m=(1+(λh/r)²)^(−r·2^m/2)`なので、h/r/m固定でλ低下とA増加が結び付く。しかしPFは非可換積、native価格はJ/U・law・basis構造・compilerにも依存する。m12 raw signal errorは.09502511833464428→.10124037927499521と増えた一方、半径下界ρは.621819319431→.848389050813へ増加した。PF・補正RTE・A・raw信号誤差・phase roomを一つの「誤差改善」と呼ばない。[B §3][B]

資源会計は

\[
C_{total,k}=W\mu_{kernel,k}+S_{shell,k},\quad
W=\sum_{m,b}N_{m,b}2^m\quad(s=1),\quad b\in\{X,Y\}.
\]

Shellは各interrogationのancilla head/readout。決定論half-headはkernelに含む。r128はkernel内部event数でありWへ再乗算しない。32は古典校正数でありshots乗数ではない。各coherent occurrenceは独立trajectoryのlawを維持し、一つのsampleを繰り返して費用を割り引いていない。

同じ価格平均を全13roundへ使うため、既存の記述的校正SEはW×SE_kernelで伝播する。新RZ平均SE214.461468031、総SE2249141484.14は既報値であり、新しいCI・有意差・母平均判定ではない。13round分の独立校正として√13で縮小しない。観測範囲17668..22066を全law supportにしない。[I §4][I]・[B §5–6][B]

| 誤差lane | 最新selectedで取得した結果 | 証拠の資格 |
|---|---|---|
| evidence | shots=null、総資源=null | 実装law・全law合成誤差が未資格 |
| ideal_exact_native | shots14048、総RZ207919150188 | 理想合成・ideal finite-RTE IIDを仮定した反実モデル |
| assumed_raw_signal_budget | shots14104、総RZ208893540201.5 | synthesis/lawを各.001A以下と仮定する感度lane。整数丸め表示の元報告は208893540202 |

DF誤差[Ha]はenergy側で一回、PF/RTE/合成/lawは既存raw-signal接続へ一回ずつ計上する。単位の異なる誤差を根拠なく加算しない。Ideal/assumedのPASSは、未知誤差が実際に0または登録bound以下であるとの認定ではない。[CJSON]

## 6. 強決定論との公平比較

最新比較は**同じupdate005 J/U**の新D1と新PRである。共通CA/RPE政策・oracle・状態準備除外・P0/serial会計を保ち、方式に合法な融合を適用した。[D]

| 候補／方法 | Kernel RZ | 総shots | Shell込み条件付き総RZ |
|---|---:|---:|---:|
| 旧J/U D（M2） | 11026 | 13610 | 101825929455 |
| 旧J/U PR ideal | 14076.4375 | 16056 | 234117788135 |
| 新J/U D numerical | 11026 | 13728 | 105532782860 |
| 新J/U PR ideal | 19825.625 | 14048 | 207919150188 |

旧J/U内のPR/D主RZ比2.29919618105、新J/U内は1.97018542062。統合実行Iや寄与分析Bに残る**新PR／旧D≈2.04191**は異なるJ/Uの歴史的benchmarkであり、最新の同一J/U比へ置き換えて引用する場合は条件差を明記する。B当時の「新J/U D未取得」はDによって更新された。B原文を改訂したり、その時点の主張へ新結果を遡及させたりしない。

新Dの全13round、full512 raw/compiled・actual adjoint・X/Yは数値PASS。Compiled/reference Frobenius6.44328529886e-12等とhead/readout誤差・norm growthを全13roundへ伝播した**level B数値pilot**である。厳密丸めenclosureではない。D numericalと両方式idealで同じ整数shots/費用を得たため、今回の点差を誤差laneの資格差だけで説明できないが、両laneのphase bound自体は異なる。[D]・[DJ]

総CX depthは新D60069240312、新PR43549878473、PR/D=.724994660275。主RZではDが低費用でも、全資源指標でDが支配するとは言えない。Depthはserial scheduled sumで、hardware実時間・全長最小depthではない。

最新技術報告Nの許可済み代数照合では、新PR W/shell固定のkernel RZ損益分岐は10062.8195925164、現在の標本平均から約49.2434%低下に相当する。これは新J/UでW等が不変となる保証、実現可能性、新目的の閾値ではない。本書では保存値を引用し、新しい損益分岐分析を追加しない。[N §3][N]

## 7. 主要な負結果・NO-GOの読み方

§8のN01〜N10は同じ種類の結論ではない。P4・CDF7・surrogate・round選択は固定候補の限定負結果、shared-basisは条件付き数学的同値、sampling lawは調査集合での方法差未特定、native融合と最新NO-GOは技術的な追加実験選定である。技術STOPも科学的不適格・正常収束と区別する。

11月1日のRC2比較も一つにまとめない。初期寄与をそろえたL2の17-call fitはoptimizer successだが一次資格なし。別の共通収束政策の2 fitは両一次未達。後者の固定終点をStage B・shots・nativeへ接続した結果は存在するが、W×μ＋Sのinterrogation総費用はそのnative scopeでは未算定である。**本書も保存Wと価格を掛けて新しいRC2総費用・順位を作らない。**

## 8. Evidence Map

本書のClaim IDはC01〜C13、N01〜N10、G01。旧版E-IDとは別体系である。各行の条件コードは§2、証拠記号は§11の**固定収録ref・科学SOURCE・JSON/metadata**と結び付く。表と両辞書を一組として使用する。

### 8.1 肯定的成果・性能・optimizer

| Claim ID | 主張 | 科学条件 | 比較対象 | 証拠 | 数値・判定 | 証拠の資格 | 制約 | 研究上の意味 |
|---|---|---|---|---|---|---|---|---|
| C01 | Tail目的でDF予算を保ちmassを減らせた | H4/A8、fit各1 | γ0対γ.1、同x0 | SA・SAJ／旧E01 | λ2.20976→1.44243、34.72%減、両DF PASS | 独立sector＋guard数値pilot | 一次収束・最適γ・因果単独効果なし | 目的変更から物理精度保持への核となる接続 |
| C02 | 初期候補のPF/RTEから条件付き資源へ接続できた | H4/P4/16/R13/P0/N32 | A8最終2 arm | OLD-C1／旧E02〜E05 | shots53136→19574、総RZ83.08598%減 | 理想／仮定誤差lane、有限価格 | 新selectedの比較ではない | 初期PR内部改善の記録を維持 |
| C03 | Reduced経路で複数U更新を受理できた | H4/CDF8/γ.1、cold J、3登録step | 旧19-call受理Uからの新軌跡 | I・IJ・IS | 新5更新、全10点数値適格、543 COMPLETE | 工程的J一次・stability・rank/nullspace・独立終端 | U未収束、raw precision loss2件、真の勾配boundなし | 技術修正から実際の外側前進へ到達 |
| C04 | 選定endpointは物理予算を保持し旧PRより低λ | H4/CDF8/LD2、元scales | 旧PR対update005 | I・IJ.selected.physical | λ.56835、旧比60.60%減、norm＋guard8.46077e-9 Ha | Sector36の数値DF予算PASS | 追加5更新の開始λは.72582、全体の大域最適なし | 改善候補を下流へ渡すidentity・精度根拠 |
| C05 | 誤差・減衰は同方向に動かなかった | H4/P128/R13 | 旧PR対新PR | B §3・I・CJSON.rounds | 全13roundでPF増、補正RTE減、A増 | 候補固有operator referenceの数値pilot | Raw信号誤差は全改善ではなくCA未認定 | λ→Aの関係と他誤差を分離 |
| C06 | 十分shotsとcoherent重みが減った | H4/P128/R13、ideal | 旧PR対新PR | B §4・CJSON.lanes | N16056→14048、W16631888→10487392 | 反実モデル内Hoeffding十分数 | 実量子shots・普遍的下界ではない | 後半round反復を含む負担減を示す |
| C07 | 新cohortのkernel単価は上がった | H4/P128/P0/N32 | 非paired旧32対新32 | I §4・CJSON.calibration | 平均RZ14076.4375→19825.625 | Finite-bit標本平均・記述的SE | 母平均／全law support／原因未認定 | λ目的で扱えない価格の交換を示す |
| C08 | PR内部の総RZが改善した | H4/P128/R13/P0/N32 | 旧PR対新PR、shell込み | B §6・CJSON.lanes | 234117788135→207919150188、11.19%減 | Ideal条件付き点資源 | Evidence総費用null、causal ablationなし | W改善が価格増を一部相殺 |
| C09 | 同一新J/UではDが主RZで低費用 | H4、新D1対新P128、R13/P0 | 共通update005 | D・DJ.comparisons | PR/D RZ1.97018542、CX depth .72499466 | D数値pilot／ideal双方とPR idealの点比較 | 厳密CA・母平均差・全指標支配なし | 強baselineを含む現在の競争力の範囲 |
| C10 | 初期L2対照もDF予算を満たした | H4/A8、full uniform L2、初期penalty寄与一致 | RC2新1 fit対保存旧PR | RC-A | λRC2=2.20976187195、DF PASS、success true | 固定α=.1の有限fit・17 COMPLETE | 一次収束・最適L2強度・原RCDF全体との勝敗なし | norm正則化対照の最初の実装・数値基盤 |
| C11 | 共通停止政策でも両一次未達だった | H4/CDF8、同元x0、α=γ=.1 | 共通政策RC2/PR各fit | RC-COMMON・RC-REC | λ8.545687/.954543、両DF PASS、両一次未達 | 保存結果・JSON復旧、未収束終点 | Stage A／update005と別。出力修復は収束改善ではない | RC2比較の収束交絡を明示 |
| C12 | 未収束RC2/PRの下流数値は取得済み | 共通政策終点、P128/R13/P0/N32 | RC2対同政策PR | RC-B・RC-SHOTS・RC-NATIVE | m12 A≈8.4247e-6/.864344、ideal shots18200774092602/14504、kernel RZ20487/13246.375 | 潜在適格・仮定shots・有限価格、各独立範囲 | Interrogation総費用未算定、最適RCDF優位なし | attenuation由来の負担差を具体化、終点選択に留保 |
| C13 | RC2の外側更新能力と未収束は両立 | H4/CDF8、stable RC2軌跡 | 同一RC2初期→64受理 | RC-STABLE | F .02919→.01743、U₂ .004762>1e-5 | 全75点数値admission、iteration上限停止 | PR correctness STOPで新DF評価なし | PRの整合修正とRC2未収束を別問題として保持 |

### 8.2 構造・負結果・技術判断

| Claim ID | 主張 | 科学条件 | 比較対象 | 証拠 | 数値・判定 | 証拠の資格 | 制約 | 研究上の意味 |
|---|---|---|---|---|---|---|---|---|
| N01 | Tail-P4は登録主RZを改善しなかった | H4/CDF8/LD2/P128/R13/P0/N32 | P4対旧PR | P4／旧E10–11 | 位置216→144、λ1.51617、総RZ275422425481.5、約17.64%増 | DF PASS・固定cohort条件付き負結果 | 初期projection/J/U/lawも異なりsupport単独因果なし | 単純疎化を価格低下と同一視しない |
| N02 | CDF7はDF適格でも主RZ点改善なし | H4/rank7/LD2/P128/R13/P0/N32 | 通常CDF7対CDF8 γ0／旧PR | R7／旧E16–18 | DF PASS、kernel17754.9375、総638880821593.5 | 独自fit・32校正の限定負結果 | rank単独効果・低rank一般の否定なし | 少ないouter rankだけでは総費用を決められない |
| N03 | 複合surrogateの登録MAE条件は不成立 | 保存M4/P4各32、固定4-fold | C対mean/A-switch/B-overlap | SUR／旧E12 | M4 C227.798>A221.407、P4 C273.579>B261.538 | `SURROGATE_NOT_QUALIFIED`、記述予測 | 64trajectoryは2DF。η・新DF・因果への移送不可 | 相関の強さと最適化代理の資格を分離 |
| N04 | Switch-onlyの第三DF転移判定も不成立 | H4/CDF8 γ0/P128/R13/P0、新32holdout | 旧64学習固定A対B/mean | THIRD／旧E14 | MAE A369.787>B303.329、`SWITCH_TRANSFER_NOT_SUPPORTED` | 単一独立DF・再fitなし | Aはmeanより良い。汎化不存在ではない | 旧回帰係数を新selectedへ移さない |
| N05 | 一般J shared-basisは通常低rankへ帰着 | 厳密同一UをG群、一般対称J | 共有frame和対群内ΣJ | SHARED／旧E15 | 表現集合CDF outer rank≤G | 条件付き数学的同値・人工検証 | PF順/split/atom化・近似U共有は別 | 重複した新分解classを作らない |
| N06 | 試験した局所融合のwhole-P0改善根拠は不足 | 人工2/3軌道、既存P0 | Tail relative／中央fold前後 | NATIVE／旧E19 | Relative RZ748→780、1903→2039。中央局所減でもcomplete不変 | 人工68 PASS、技術優先度判断 | H4実性能の一般否定なし、head-tail境界未測定 | 既存融合を二重計上せず未測定を残す |
| N07 | 旧J/Uのround選択は混合利益なし | 旧PR共通J/U、D1/P128/R13/P0 | 各round主RZ選択対全D | ROUND／旧E20 | Ideal/assumed全13roundがD、`MIXED_ROUND_DOMINATED` | 固定点モデル・189 COMPLETE | 新selectedのround割当、全指標支配なし | 同じ保存条件の再探索を避ける |
| N08 | 調査したsampling lawに独立した新機構を未同定 | Euler/Parent/IS/finite return/相関 | 現K0平均operatorと既存枠組み | LAW／旧E21 | `NO_DISTINCT_FREEDOM`、人工13 PASS | 数学対応・限定技術調査 | 全sampler不存在・性能negativeではない | law変更時は平均積・weight・rangeを保持する必要 |
| N09 | 固定U・tail保存LPに小さいλ改善あり | 共通政策PR終点、H4/CDF8/LD2、U/head固定 | 元係数対LP後 | LP | .954543284658→.954432028127、約.0116555%減、保持gate PASS | `NUMERICAL_LP_IMPROVEMENT`・50 COMPLETE | 厳密LP証明・最新selected・native総資源改善なし | 係数冗長性の存在と費用改善を区別 |
| N10 | 現証拠では新しいcost-aware fit候補を選ばない | 最新A/Bと過去構造証拠の技術統合 | 候補指標とcomplete価格の接続 | N | `TECHNICAL_NO_GO_CURRENT_EVIDENCE` | 技術的な実装・実験準備選定 | 正式科学採択・独立新規性・研究終了ではない | 新式/runnerを増やさず不足する価格機構を明示 |

### 8.3 保証資産の範囲

| Claim ID | 主張 | 科学条件 | 比較対象 | 証拠 | 数値・判定 | 証拠の資格 | 制約 | 研究上の意味 |
|---|---|---|---|---|---|---|---|---|
| G01 | 固定回路のvalidated CAR接続は存在する | A8両arm各1保存trajectory、P4/16一h-step、full512全基底、κH/right-polar Q target | Raw/compiled対独立CAR、literal model bridge別計上 | 旧版E23・F2/F3-K1 | 各7 calls、登録operator上界PASS | Arb192による対象限定validated enclosure | 新selected/r128全law/全13round/physical input全誤差/CA保証へ移送不可 | 固定保証を消さず、全law未資格と両立して保存 |

## 9. 未資格・未解決事項

| 項目 | 保持する資格と不足 |
|---|---|
| 物理DF精度 | Sector norm＋engineering guardの予算PASS。積分入力・全Fock・丸めを含む厳密DF enclosureではない |
| Optimizer | Raw success、J一次、U一次、rank/nullspace、stabilityを別保存。最新PRとRC2の同時一次収束・大域最適は未達 |
| Reduced gradient | 同一配列整合・独立比較・latest-pairは工程的数値資格。真の勾配誤差上界・厳密降下保証はnull |
| Native正しさ | 新32回路は全full512数値比較、actual adjoint/X/Y PASS。F2/F3は別固定targetのvalidated保証。いずれも新r128 law全体の一様boundではない |
| Implemented law | Finite-bit/PCG64実装lawとideal IID lawの距離・それを信号へ伝える資格が不足 |
| 価格母平均・support | 平均/SE/min/maxは有限cohortの記述値。母平均保証・厳密support・選択後補正は未取得 |
| CA・shots | Ideal/assumedで同CA目標の十分式を使った条件付き比較。Actual CA、保証shots、evidence総資源は未資格/null |
| 機構・一般化 | 新旧J/Uと非paired cohortからJ・U・head・law別の価格因果効果を同定できない。H4一geometry・有限endpoint。別分子/初期値へのPR-aware汎化なし |
| 総資源の範囲 | 状態準備・hardware noise/runtime・未実装融合を除くserial interrogation会計。最適なPF family／全価格指標での一般優位ではない |
| 方法の位置付け | RCDF/SCDF/PR既知法との対応は既存資料を参照。新規性・独立貢献・論文化GOはCodex-B／研究判断側の別課題 |

旧版の「最終総費用未実施」という広い説明は、**現在はideal/assumedの条件付き点総費用は実施済み、実CAを保証した総資源比較は未実施**と読む。過去の未資格は、対応する新証拠が得られた項目だけを更新する。H2/H6の旧結果から現H4の新J/Uや収束資格を補わない。

## 10. 今後の研究判断に必要な情報・Codex-Bへの引渡し

| Bが組み立てる主張の対象 | 照合するClaim | 未解決事項／判断を変える情報 |
|---|---|---|
| PR-aware DFの限定的な資源改善 | C01–C08 | 共通x0の初期比較と追加未収束endpointを区別。既知のnorm正則化・λ最適化との目的/変数/task対応はB側で検討 |
| 強baselineを含む比較 | C09・N07 | 新selectedの同J/U比較を優先。主RZ不利とCX depth利点を両方記載。旧D benchmark比を公平比較の最終値にしない |
| 最適化アルゴリズムの到達範囲 | C03・C04・C10–C13 | 正しい評価点・小step規則により外側前進を確認。一方、同時一次収束・最適RC2との比較は未確立 |
| 設計上の知見・限定負結果 | N01–N10 | Support/rank/λ/一kernel価格だけでは総費用を決められない。数学的同値、予測基準不成立、技術NO-GOを混同しない |
| 保証付き主張 | G01・§9 | 固定targetのvalidated保証と全law/CA未資格を併記。全lawや母平均保証を主張する場合はその不足根拠を別判断 |
| 次の研究範囲 | N10 | 具体的な価格操作機構・独立条件での検証可能性・必要な保証水準のどれを研究課題にするか。自動的な新目的/fit再開はしない |

**Codex-Aの引渡しは本書の固定証拠体系で完了する。Codex-B**は独立に先行研究・貢献候補とこれらの範囲を照合する。**A/B統合担当**は両者完了後、必要な共有索引へのリンクを一度だけ追加する。本書でPROJECT_MAP/VALIDATION_STATUS/docs索引/研究概要/研究ノート/validation manifestは編集しない。**ユーザー／研究C**が正式な研究判断と、必要な場合の重要レビュー・追加科学実行を別々に選ぶ。本書は新科学契約・候補ランキング・科学実行許可を作成しない。

## 11. SOURCE・結果・参考資料と照合範囲

### 11.1 固定refの使い分け

| 記号 | 報告の固定収録commit | 固定科学SOURCE／技術公開commit | 結果・metadataの入口 |
|---|---|---|---|
| E0 | 共通起点46f1461d内の[10月31日版][E0] | 旧版の調査起点a776aa84a。数値実行SOURCEは旧版§11のE01–E23 | 既存23項目とその全raw/sourceリンクを再利用 |
| SA | 共通起点46f1461d内の[Stage A V2][SA] | `e1085c95f5c01b68566e17d543ace5832cb94061` | [result][SAJ] `sector/fit_diagnostics/exact_tail_mass_reduction`、同run SOURCE/STOP |
| OLD-C1 | 共通起点内の[C1][OLD-C1] | `b1c735da394d4b0a444cbc618ead9781ef4d9134`。B1/B2 SOURCEは旧E02–03 | C1-run01の[baseline ideal][C1BASEJ]／[proposed ideal][C1PROPJ]・STOP、旧版E02–05 |
| OLD-M4 | 共通起点内の[M4][OLD-M4]／[Native128][OLD-NATIVE] | PF/RTE `7dbabc7866cc62a5460bf1579ce84c59493d8e1e`／native `dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad` | [M4 result][M4J]、[native result][M4NJ]、それぞれSOURCE/STOP |
| I | `aced1647b81e8d28091058cdc7b6e3e1ee54896a`・[統合報告][I] | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` | [親result][IJ]、[C/result][CJSON]、[SOURCE manifest][IM]、[STOP][IS] |
| B | `2639221a59713d431bc7c157854628bbc006a1bd`・[寄与分析][B] | 新科学実行なし。旧M4/nativeとIのSOURCEを使用 | 別branch文書。§2/3/4/5/6と出典key/path。共通起点に収録されたと扱わない |
| D | `38b8d0df39b8943178ecef611194819cff1c2169`・[同J/U強D][D] | `5f27539ae0ab88181d9cfcd45badb54daf855c28` | [result][DJ] `saved_DF/numerical/ideal_exact_native/metrics/comparisons`、[STOP][DS] |
| N | `46f1461d213ad73f8e187ef98e1491ef8b522aaa`・[技術判断][N] | 新科学実行なし、同技術公開commit | §3の保存scalar代数、§4–6の既存負結果、§8のSOURCE/過去人工証拠 |
| RC-A | 共通起点内の[初期L2][RC-A] | `150f904f6975868abe09966dda8ab41722e2a7cc` | [result][RCAJ]、fit/optimizer.json・SOURCE/STOP |
| RC-COMMON／RC-REC | 共通起点内の[共通fit][RC-COMMON]／[JSON修復][RC-REC] | 元fit `e2f7dfd44357510d54ab9c6c9c5032e0072ab376`。修復実装929da067d8a29a1b36f54c8cf19642c5aa4e3789、独立SOURCE47/plan | [復旧result][RCRECJ]、旧fit optimizer/diagnosticと旧failed STOPを保存 |
| RC-B／RC-SHOTS／RC-NATIVE | 共通起点内の[Stage B][RC-B]／[shots][RC-SHOTS]／[native][RC-NATIVE] | `d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6`／`e88fa599af7724f34e11a0f2eab959f92135cf13`／`14843a28f729cfe5f238d72019b060ee281c9a45` | [Stage B JSON][RCBJ]、[shots JSON][RCSJ]、[native JSON][RCNJ]、各SOURCE/STOP。合算総資源JSONなし |
| RC-STABLE | 共通起点内の[stable部分STOP][RC-STABLE] | `12e0079a264dc0f5831791efc06c0aa159b88908` | [summary][RSTJ]、run01/rc2のstate/traceと親failed STOP。未使用1083unitsは転用不可 |
| P4 | 共通起点内の[構造PR性能][P4] | `4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff` | [result][P4J]、旧版E10–11のfit/source/STOP |
| R7 | 共通起点内の[CDF7 native][R7] | `173ba2fd83172f41f08900fcba7e2a977565f57f` | [result][R7J]、旧E16–18のfit/DF/PF/STOP |
| SUR／THIRD | 共通起点内の[surrogate][SUR]／[第三DF][THIRD] | `0301286493ebf2b95a1786c2bd78756015c1015d`／`30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c` | [SUR result][SURJ] `qualification/cohorts`／[THIRD result][THIRDJ] `transfer`、両run02と旧run01保全 |
| SHARED／NATIVE／LAW | 共通起点内の[共有basis][SHARED]／[native構造][NATIVE]／[law調査][LAW] | 技術公開 `0876449d7ce75e25277865834efbc3bafaed7b3e`／`0cb2e5ea986320283465ea51e5b273d0b3313b96`／`a776aa84aefc128cc317e33075415140916013a0` | 各technical evidence・旧版E15/E19/E21。科学runなし |
| ROUND | 共通起点内の[round選択][ROUND] | `79fa0107658c92c3b3169d62348498c7dc37fab0` | [result][ROUNDJ]、両仮定allocation JSON・SOURCE/STOP |
| LP | 共通起点内の[LP微小係数profile][LP] | `c40f8128197177fd5ec2388cd306d89d36c3bd67` | [result][LPJ] `LP/preservation/candidate`、LP_solver.json・SOURCE/STOP。旧LP未資格STOPも維持 |
| F2/F3-K1 | 共通起点内の旧版E23と[F3][F3] | `3742ec10ddc0df5faaaa2a6856a9cc60ea013eb9`／`9b0f4d2b40159b77f28caf3f215ad22d0d075879` | 旧版§11の固定CAR各run、Arb enclosure/model bridge・SOURCE/STOP |

「共通起点内」は本書が参照する固定treeの意味で、各報告が最初に公開されたcommitの主張ではない。科学SOURCE commit、実装基点、SOURCE manifestのbase_commit、結果追記commitを同一視しない。過去の論文対応・Track B資料は旧版§7・§11から辿れ、本書で新しい網羅的文献調査・新規性監査を実施していない。

### 11.2 再利用したコード・証拠資格

目的/係数mapは`c_pr_aware_df.py`、同一配列評価とinner/refinement/admissionは`c_pr_coefficient_point.py`・`c_separated_j_u_consistent.py`・`c_j_u_stable.py`、最新接続は`c_pr_integrated.py`・`c_pr_integrated_performance.py`、独立物理評価は一般K/`SectorCAR`、費用は`c1_stage3_accuracy.py`・`c1_stage3_connection.compose_serial_cost`、強D/nativeは`c_same_ju_strong_d.py`・`StrongBuilder`を参照する。今回再利用するのは静的定義・既存報告・JSON scalar・metadataであり、これらの科学entryを起動しない。

`VALIDATION_STATUS.md`と`artifacts/validation_manifest.json`の当該項目は保存されたlocal evidenceの状態を示す。公開済みGit objectは来歴の固定であり、immutable CI・外部独立再現を意味しない。一般的な古い禁止proseより個別の最新固定報告とrawのscopeを優先するが、未資格を勝手に資格昇格しない。

### 11.3 本作業の検証・保全・STOP

必須4報告を各指定refのGitHubから取得し、固定Git blob本文と照合する。旧版E01–E23、最新I/DのJSON scalar・SOURCE/STOP、別branch Bの出典・条件を使って転記と来歴を確認する。新しいNPZ/QPY decode、operator/PF/RTE/shots/native価格再計算、統計解析、ランキング、fit、科学runner、科学callはすべて0。人工試験や旧科学suiteの再実行も行わない。文書のリンク・Claim ID・引用scalar・Git差分・公開本文の確認を今回の技術検証とする。

独立worktree`/tmp/prt-c-evidence-map-a-20261102`、branch`codex/research-c-evidence-map-a-20261102`。正式共通運用指示2026-10-25版§8/8.1/8.4/10/10.1/10.3を適用し、変更は本Markdown一つに限定する。旧Evidence Map、共有索引、研究概要、研究ノート、manifest、科学SOURCE/raw/契約/grant/ledger/marker/STOP、Codex-Bの成果物は編集しない。承認証明・承認文保存・新grant・分析管理機構を追加しない。重い計算はなく、共有環境変更やCPU/RAM/workerの人工上限追加もない。

Commit/push後、remote SHA/tree・文書本文・引用元・旧成果物の不変を照合して完了する。**Mandatory STOP。Bの作業完了を待たずに引き渡し、研究判断・新契約・新科学計算へ自動進行しない。**

[E0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md
[SA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md
[OLD-C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_stage_c1_execution_20261027.md
[OLD-M4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md
[OLD-NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_M4_native128_execution_20261028.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2639221a59713d431bc7c157854628bbc006a1bd/docs/research/research_c_pr_resource_contribution_analysis_20261102.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_coefficient_point_execution_20261102.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_inner_point_execution_20261102.md
[G18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_reduced_limited_execution_20261102.md
[G60]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_guarded_step_execution_20261102.md
[G49]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_local_direction_execution_20261102.md
[G19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_small_step_execution_20261102.md
[RC-A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_regularization_comparator_execution_20261101.md
[RC-COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_common_convergence_execution_20261101.md
[RC-REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_common_convergence_export_repair_20261101.md
[RC-B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_rc2_pr_stage_b_execution_20261101.md
[RC-SHOTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_rc2_pr_conditional_shots_execution_20261101.md
[RC-NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_rc2_pr_native_execution_20261101.md
[RC-STABLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_j_u_stable_execution_20261102.md
[P4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_df_structural_pr_performance_20261028.md
[R7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_cdf7_native32_execution_20261030.md
[SUR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_df_basis_surrogate_repair_execution_20261029.md
[THIRD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_df_basis_third_repair_execution_20261029.md
[SHARED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_shared_basis_df_design_20261029.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_native_structure_optimization_design_20261030.md
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_sampling_law_freedom_survey_20261030.md
[ROUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_roundwise_det_pr_execution_20261030.md
[LP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_fixed_u_tail_lp_numerics_execution_20261101.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_F3_K1_execution_20261028.md
[SAJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df/2026-10-27/run01/result.json
[M4J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/result.json
[M4NJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/result.json
[IJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/result.json
[CJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/result.json
[IM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/technical/SOURCE.json
[IS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/STOP.json
[DJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/artifacts/research_c_same_ju_strong_d/2026-11-02/run01/result.json
[DS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/artifacts/research_c_same_ju_strong_d/2026-11-02/run01/STOP.json
[G10J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_coefficient_point/2026-11-02/run01/result.json
[G8J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_inner_point/2026-11-02/run01/result.json
[RCAJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_regularization_comparator/2026-11-01/run01/result.json
[RCRECJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/result.json
[RCBJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/result.json
[RCSJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/result.json
[RCNJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_native/2026-11-01/run01/result.json
[RSTJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_j_u_stability/2026-11-02/execution_summary.json
[P4J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_structural_pr/2026-10-28/run01/result.json
[R7J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_cdf7_native32/2026-10-30/run01/result.json
[SURJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_basis_surrogate/2026-10-29/run02/result.json
[THIRDJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_basis_third/2026-10-29/run02/result.json
[ROUNDJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_roundwise_det_pr/2026-10-30/run01/result.json
[LPJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/result.json
[G18J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_reduced_limited/2026-11-02/run01/result.json
[G60J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_guarded_step/2026-11-02/run01/result.json
[G49J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_local_step/2026-11-02/run01/result.json
[G19J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_small_step/2026-11-02/run01/result.json
[C1BASEJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/baseline_ideal_exact_native.json
[C1PROPJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/proposed_ideal_exact_native.json
