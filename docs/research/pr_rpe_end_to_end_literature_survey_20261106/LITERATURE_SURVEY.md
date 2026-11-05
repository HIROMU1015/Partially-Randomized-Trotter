# PR＋RPE総回路費用を目的とする文献差分調査

2026-11-06 JST。固定証拠基点：`3571477093441ac4852166fe829f8ccc5db5ec4f`。書誌、版、確認節、取得制限は [SOURCES.md](SOURCES.md)。本書の `L01` 等から同書の同IDへ追跡できる。新しい科学計算・保存値再解析・アルゴリズム設計・独立証明・詳細科学レビューは実施していない。

## 1. 固定する研究目的

元の物理Hamiltonianについて `H|E₀⟩=E₀|E₀⟩` を満たす厳密基底状態を利用でき、状態準備費を除く。baselineにも同じアクセスを認める。H_D、打切りH、再fitしたDF Hamiltonianの基底状態で代用しない。真のE₀、gap、正確なbias、最適splitを無料で知るという仮定は加えない。

最終対象は、指定するエネルギー精度と成功確率を満たす

\[
G_{\mathrm{total}}^{(a)}=
\sum_m\sum_{b\in\{X,Y\}}N_{m,b}
\mathbb E[C_{m,b}^{\mathrm{full,no\text{-}prep},(a)}]
\]

である。`a` はRZ数、CX数、T数、depth等の**個別**資源指標。必要shot数Nは測定手順・保証に依存し、原著のRMSE、query数、固定Tのobservable精度、エネルギー成功確率は別のtaskである。状態が厳密でも近似evolutionの位相bias・半径低下・他固有位相の寄与は残り得る。精度、失敗確率、分子、compiler、主資源指標の新しい数値は定めない。

## 2. 今回の差分

既存70研究単位、Phase 2AのRQ-01〜07、統合129記録を再利用した。129は実験数ではない。SPRINT、cost-aware sampling、basis共有等を初めて発見したという報告ではない。

今回追加した情報の中心は、UWCの全round目的とround別表現に関する具体的記述、SPRINTのspectral shift/減衰/合成の本文、連続時間PAI最新版のHadamard/TETRIS分離、SE-QPEの共通固有基底・真空reference条件、QSP信号設計のoracle/query条件、RPE訂正の明示である。これらは方法の採択や本projectでの性能実証ではない。

## 3. PRと全費用に直接関係する先行研究

### L01 PR原著 — 全費用への接続自体が既知

[Günther et al., v2](https://arxiv.org/pdf/2503.05647v2) は決定論blockとrandom tailを結合し、平均振幅をHadamard測定へ接続する。App.Eは厳密固有状態を仮定する資源推定で、今回の状態仮定に近い。原著の費用推定はRMSE条件と経験的RPE定数を含むため、任意の失敗確率保証へそのまま置換できない。

E21–E23は決定論回路、tail mass、normalization、測定回数、全roundを結ぶ。E24以降の係数丸めをrandom residualへ移す手順も、合成費とtail負担の交換として既知である。Toffoliと二量子bit gateを別々に評価し、分子・hydrogen-chain例の資源比較を行う。元のモデル資源推定と本repositoryのfull-wrapper compiled期待価格を同一視しない。

したがって「splitとshotを一緒に最適化」「角度を丸め残差を乱択」「厳密基底状態を使う」だけでは方法差にならない。今回の課題は、既存の統合機構を強対照にも与えた後に、未取得のsignal保証・全law期待価格まで含む増分が残るかである。

### L03 UWC — norm最小化を越えた目的も、round別表現の着想も既知

[Kanasugi et al., v2](https://arxiv.org/html/2603.22778v2#S3.SS3) は軌道最適化とBLISSによる表現変更をPR資源目的へ結び付ける。II.2.4のEqs.17–18はdeterministic項数とrandom tailの交換、III.3のEqs.33–34は全費用目的と最終roundによる近似を示す。直後にはroundごとの別表現最適化と、その古典最適化負担も明記される。

数値対象はFe–S cluster、P450等のactive-space modelで、STAR条件下の物理資源・時間へ接続するモデル推定がある。主たるrandom部は本文でqDRIFTと説明され、本repositoryのfinite paired-Taylor、全wrapper価格、N資格を自動的に同一視できない。最終round近似は本研究の全round合計そのものの検証ではない。

10月28日の[PRX Quantum受理情報](https://journals.aps.org/prxquantum/accepted/10.1103/863w-jxzp)を確認した。外部UWCの存在・出版更新と、来歴未結合のrepository旧UWC値は独立である。表現＋split＋RPE費用の結合を新規性の根拠にせず、同じ元H・状態・費用境界で何が追加できるかを問う。

### L02 SPRINT/GRADE — 構成要素の相互作用を使う直接的な対照

[Casares et al., v1](https://arxiv.org/html/2606.30741v1#S4) は表現、近可積分性、processing、symmetry protection、randomization、QROM/合成を統合する。§IVは誤差とstep価格を合わせて組合せを選ぶ手順であり、単にparameterが多いだけの提案ではない。GRADEの拡張空間では漏洩も扱う。

数値taskはbattery材料のXASで、Toffoli・logical qubitの資源評価である。App.Fはrandomized PFによるspectral shift、重み変化、減衰を分け、F.3は合成誤差から信号減衰への経路を示す。これを読めば、biasだけを小さくしても測定負担が増える可能性を無視できない。

ただしXASのline broadeningとRPEのbranch成功条件は違う。全技法を必須baselineとして一括導入せず、対象Hamiltonian・状態・入力情報に適合する処理だけを公平に移す。既存B-M/高次PF/FRの照合・STOPを再利用し、既知processingを新しい補正機構と呼ばない。

### L04/L05 importance sampling — 異なる目的関数を区別する

[Cugini–Atif–Subaşı, v1](https://arxiv.org/html/2603.13495v1#S2) のThm.1は、結果を事前に知らないworst-case二次momentを用いるnet costに対して `q*(ω) ∝ p(ω)/√c(ω)` を与える。Thm.2は同じ回路familyを再重み付けしても実装・近似biasが変わらないことを述べる。II.1には、独立成分かつ加法価格という条件で長い積の利益が縮む解析もある。error detectionではZeroFillとDiscardを区別する。

[Kiss–Grossi–Roggero, v2](https://arxiv.org/pdf/2212.05952v2) のcost-aware qDRIFTは `q_c(j) ∝ h_j/C_j` を使う。§4.1はsimulation内の費用削減と独立experiment数の増加を同時に示す。lattice nuclear modelの数値検証があり、bias/concentration条件も与える。

この二つのlawの違いは矛盾ではなく、推定対象・時間step・保証・目的の違いである。どちらも「安いeventを増やせばRPE全体が安い」の根拠にはならない。全path価格を得る古典費、有限confidenceでのweight range/分散、X/Y両軸、alias/branch失敗を接続する必要が残る。既知IS式の再発見や、停止済みsampling-freedom案の再命名は候補にしない。

## 4. bias・合成・測定を結ぶ方法

### L08/L09/L10 TE-PAI、連続時間PAI、TETRIS

[TE-PAI v3](https://arxiv.org/html/2410.16850v3) はchannelの期待値を確率的角度補間で再現し、離散化biasを測定負担へ交換する。Theorems 1–2は平均gate数と測定overheadを扱い、spin-model simulationやfault-tolerant synthesisを検討する。一定角度に限定した回路が短くても、normalizationの二乗による反復負担は残る。Loschmidt/statistical PEへの利用は原著にもあり、Hadamardへの応用という名称だけで新しくはない。

[連続時間PAI v3](https://arxiv.org/html/2604.02854v3) はPoisson event生成を用い、H₃⁺のenergy推定・SYKのOTOC・trapped-ion実験を扱う。重要なのは、§III.1/App.CがHadamardのevolutionに **TETRIS** を用いる点である。TE-PAIによるadiabatic state preparationの改善は、今回の状態準備除外という目的にはそのまま寄与しない。

[Granet–Dreyer](https://arxiv.org/abs/2308.03694v2) のoperator平均によるevolutionは既存比較でも既知で、L09 App.Cには具体的なdeterministic H_HFとrandom H_intのrecipeがある。原著の本調査での直接確認はabstractまでである。平均振幅・channel・符号weightを分け、同一元Hのenergy成功条件と全round費用へ結ぶことが未解決であり、「Trotter errorなし」だけを総費用改善としない。

### L13 phase2 — 誤差情報を取得する費用と量子実行費用を分ける

[Miller et al., v2](https://arxiv.org/html/2504.17881v2) はPauli rotation列のclassical simulatorと、ruthenium-ligand active-spaceにおけるPR/RPEの誤差解析を報告する。Methods/Data analysisと補足は、time signalからeffective energyを抽出し、step幅依存をfitする経路を示す。収束していない大きいactive-spaceのfitは区別されている。

解析上界より緩い経験的誤差が得られても、その係数は別の分子・DF・splitへ移せず厳密上界でもない。classical simulatorの速度と量子gate数の改善率も別物である。未知のE₀を実装algorithmへ無料で渡すことなく、どの誤差情報を取得でき、その取得費をどこに記録するかを考える対照になる。

## 5. controlと推定signalの変更

### L11 SE-QPE — 制御を外す代わりにreferenceとqubitを使う

[Rowe et al., v2](https://arxiv.org/pdf/2604.14921v2) はtarget/referenceの二registerとCSWAP干渉を使う。§IIはfactor U_A/U_Bが共通固有基底を持ち、既知reference位相を補正できる場合に、canonical/iterative QPEの位相register分布を保存する。分子の粒子数保存Hamiltonianでは真空referenceを利用する。FeMoco DF資源分析とethyleneのhardware例がある。

費用分析のEqs.18–19はevolution blockを中心とし、QFT・状態準備を除く。CSWAP、追加register、fan-out、routingが増え、短いphase powerでは逆効果もあり、原著が低power/高powerの混在を既に議論する。CX、T、depthを区別する。

本PRの異なるrandom pathが同じ固有基底を持つこと、各pathで真空がreferenceとして使えることは確認されていない。DF Hamiltonian全体の粒子数保存から個々のPauli/RTE pathの性質を推定しない。既存diagonal-only controlと比較した後に残る差、signal meanとreference補正、全Nの接続が問いとなる。

### L12 QSP-PE — 測定数とquery回数を同時に変える既知の代替推定器

[Jia–Liu–Dong, v1](https://arxiv.org/html/2604.01205v1) はconfidence interval内でsignalの感度を設計する。Theorem 1/S4は測定数・query数と成功確率を結び、S4.2はpost-processing、S6はcontrolled evolutionとspectrumのshift/rescale条件を述べる。数値ではRPEとの推定分散比較を行う。

これはRPEのshot scheduleを増やすだけの方法ではなく、oracleを複数回組み込む測定回路への変更である。query単価、ancilla rotation合成、signal設計の古典最適化、PF/RTEの蓄積誤差は本PRのcompiled費用へ未接続。unitary oracleを前提とする保証に非unitaryなfinite-RTE平均を代入できるとは確認していない。

RPEを研究対象とする今回の正式方式には採用しない。周辺baselineとして、signal側の既知改善にも同じsimulation/精度情報を与える必要があるかをユーザーが選ぶ材料とする。

## 6. 再利用する基礎対照・限定して確認した文献

| 論文 | 対象・主要操作 | 保証・資源・数値検証のscope | 全費用との関係・本研究での扱い |
|---|---|---|---|
| L06 Composite | HamiltonianをTrotter/qDRIFTへ割当 | diamond距離・simulation費用の保証。今回は出版社要約まで | split自体は既知。controlled first moment、RPE全roundへの保証移送は別 |
| L07 cost-aware circuit synthesis | cheapな可換fragment lexicon、部分・符号付き割当、greedy選択 | 回転合成の構成とHeisenberg/LiH/H₂のsimulation比較。主にnon-Clifford資源 | 辞書選択＋回路価格は既知。recipeの取得費を支払う。一般finite-RTE/RPE定理ではない |
| L14 RC-DF | compressed DFのregularized fit、normと近似誤差の交換 | 小分子測定資源とP450 qubitization。大系のenergy誤差はCCSD(T) heuristic | fit誤差、状態、回路価格を元Hへ戻す。norm削減をPR総費用削減率にしない |
| L15 BLISS | 指定symmetry blockを保ち他blockを変更 | Eq.4のblock保存と小分子LCU比較。今回は関連理論節を確認 | 対象sectorの指定と有限実装の漏洩が重要。全空間のground順位が保存されるとは限らない |
| L16 partitioning assessment | fermionic/qubit分解、対称性を用いる誤差評価 | Trotter誤差と単回回転数、App.Aのadaptive QPE費用。分子例あり | PF誤差を減らしてもstep単価が増える直接例。compiled RPE保証とは区別 |
| L17 unitary partitioning | 反可換Pauli和のunitary化と測定回路 | operator恒等式とVQE groupingの数値比較 | 正規化involution/pairは既知。混在辞書の独立新規性やRPE shot改善を意味しない |
| L18 RPE/Erratum | 幾何学的powerのphase復元、additive errorへのrobustness | gate calibrationの保証。訂正は旧branch失敗解析を修正し、Heisenberg scaling自体は維持 | π/(2k_j)の旧条件を無条件採用しない。repository既存decoderの検証済み範囲を今回再判定しない |
| L19 connectivity | 適合する電子構造表現のfermionic swap/Givens | linear-depth/connectivity構成。今回は要約・既存限定読解のみ | 既知routingをbaselineにも許す。任意DF-PRへの一般移送を主張しない |
| L20 compressed control | translation-invariant local Hamiltonianのcontrolled evolution圧縮 | depth/controlの主張とspin-modelのnoisy-emulator例。今回は要約のみ | 化学DFへの適用条件未確認。論文のdepth改善をPR総gate改善へ置換しない |
| L21 qDRIFT | 係数に応じたrandom term simulation | channel近似の基本法。今回は要約と既存比較を再利用 | 全Hの純RTE、DF二体tailの全ランダム化、paired finite RTEは区別 |
| L22 PRHS | 相関sliceとquasi-probabilityによる高次化 | abstractは分子simulationのquery/bias比較。本文・補足の条件は既存Track B照合を継承 | 高次biasの低減と符号/shot・生成費の交換。既知対応とSTOPを保持 |
| L23 qSHIFT | 履歴を使うadaptive sampling | query scalingと古典coefficient取得費を持つ。今回は現版metadata＋既存照合 | marginal保存からcoherent mean保存を導かない。旧式表示の不整合を修正済み扱いにしない |

## 7. 証拠への照合結果

固定統合の `MIX-F2` は登録単回資源の成功とN未資格を分離し、`C1-PAUSE` はlogical path完了とnative/full-wrapper未完了を分離する。`SUP-RPE-V3R1/ENERGY` は条件付き全round成功・energy接続を保持する一方、費用優越・実装全体保証は未決定である。`SUP-C-064/WORD` はown-DF ground条件、nominal価格、固定word correctnessに限定される。今回の元Hの厳密基底状態仮定へ、その結果を自動移送できない。

この照合から残るのは [RESEARCH_OPPORTUNITIES.md](RESEARCH_OPPORTUNITIES.md) の条件付きの問いである。「既存機構の実装を増やす作業」と「新しい構造・保証・適用条件が成立すれば知見になる問い」を分けた。優先順位、新規性の正式判定、研究GO/NO-GOは付けていない。
