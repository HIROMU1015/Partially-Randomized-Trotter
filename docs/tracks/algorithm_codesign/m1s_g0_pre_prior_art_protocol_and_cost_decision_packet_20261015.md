# M1-S G0-pre：先行研究・controlled protocol・費用比較条件

2026-10-15 JST。**`G0_PRE_COLLECTION_COMPLETE_M1S_ADOPTION_PENDING`**。本書は採択前の資料照合であり、M1-Sの正式採択、G0の新理論、数値検証は開始していない。**mandatory STOP**。

## 1. 作業根拠と到達点

ユーザー指定の[GPT独立科学レビュー原本][REVIEW]を、内容を変更せず先に公開した。入力固定commitは`41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03`、41,818 bytes、SHA256 `30baecc1c9f66b1adcc52805669fa03a3fb7849e970dbf8c1237e4ea85372253`。remote再取得でもbyte一致した。[受領記録][INTAKE]。ユーザーの本依頼をG0-preの読取り・調査・公開の指示として扱い、科学的採択・G0・実験の承認とは扱っていない。

運用は[2026-10-12共通指示][COMMON]による。P0/P0.5の資料は[既存報告][P05]と[索引][P05INDEX]を再利用し、Gu/Li/PR/H6の監査を繰り返していない。今回、新たに近接8論文の関連本文を読んだ。[文献locator][LOCATOR]に著者、版、該当ページ・式、取得PDFのbytes/SHA256を記録する。PDF全文はrepositoryへ複製しない。

今回明確になったことは次の通り。

- **回文型の対称PFは既知。** Childs–Ostrander–Su（COS）のEq6は、同じpermutationを前後で使う対称なwordを定義している。従って回文の外形だけはM1-Sの方法差にできない。
- **共有乱数を量子的な二枝へ載せる一般構成にも先行例がある。** MLMC-qDRIFT v2の§2.3はindex共有fine/coarse pathをcontrol二枝へ載せる。M1-Sのopposite-sign・同深さ・RPEとは対象が違うが、「量子枝で乱数を共有すること」全般を未踏と扱えない。
- **mirror/echoの名が近くても、測るものとlawを確認する必要がある。** Proctorのmirrorは逆演算による既知出力、SYK/TETRISのmirror-on-averageは独立なU/U′によるobservable信号である。M1-Sの同順sign reversalと、その非CPTPな平均写像を証明済みとする根拠にはならない。
- **取得した関連節では、M1-Sの特定law＋非可換D＋RPE＋同条件native費用を一体で与える結果は特定していない。** 全文献での不存在、新規性、既知結果からの非自明性を認定した意味ではない。組合せや直接系への還元可能性をGPTへ残す。

これらは科学的GO/STOPの判定ではない。元M1を独立新アルゴリズムとして採択しないという既存GPT判断、M1-Aを補助候補、M2を代替候補として保持する方針を変更しない。

## 2. 同値性を照合する単位

今回の「同じか」は、次の五層を分けて調べた。

1. **wordの構造**：gateの順序、鏡像、逆演算、符号。
2. **ensembleのlaw**：IIDかpermutationか、replacement、branch間の同時分布、angle、correlationの範囲。
3. **観測対象**：振幅のfirst moment、channel、survival probability、observable difference、energy/phase。
4. **保証**：対象時間、bias、減衰、signal floor、gap/overlap、RMSか高確率か。
5. **資源**：同じprimitive・precision・control・shots・準備／古典費用か。

第1層の共通部分だけで完全同値とせず、第2層以降が未照合だから独立新規とも言わない。P0/P0.5で既知とされたfull-qDRIFT、Composite、PRの独立±、FRのphase/radius整理はそのまま参照する。

## 3. 新たに確認した近接文献

ページはPDFの1始まり番号。表の「残る対応」は今回未実施の適用・還元証明である。本文の数学的保証を再証明したという意味ではない。

| 文献・固定版／確認箇所 | 実際の構成・乱数・観測対象 | M1-Sと共通する既知部分 | 同一視できない点／残る対応 |
|---|---|---|---|
| **COS**：Childs–Ostrander–Su, *Faster quantum simulation by randomization*, [1805.08385v2][COS]。pp3,5–8、Eq6/7、Lemma1、Eq15–17、Lemma2 | Hamiltonian項のpermutationをランダム化。Eq6は同じpermutationを前後で用いる対称PF。平均演算子の誤差をmixing lemma経由でchannel diamond誤差へ接続。first-orderのforward/reverse混合は別構成 | 回文型word、順序相関、平均演算子を利用する解析は既知 | M1-SはIID Pauli label列の再利用、COSは全項permutationと各項の係数付きhalf-step。一般にlawとangle規則が違う。再パラメータ化やconditioningによる還元可能性、controlled振幅/RPEへの移送は未証明 |
| **MIR**：Proctor et al., *Measuring the Capabilities of Quantum Computers*, [2008.11294v2][MIR]。p1、Appendix IV pp12–16、Eq20/21/29–31/40/41 | Cと逆順inverse／quasi-inverseを組み、Pauliやfiducialを入れて既知のbitstring出力を作るbenchmark。simple mirrorはC⁻¹C | wordの逆転、共有した回路情報、echo型検証は既知 | M1-Sの各gate同順sign reversalはfull-word inverseではない。benchmarkのreturn probabilityをenergy位相信号へ読み替えない。使用版は著者が最終出版版との差を注意するarXiv v2である |
| **SYK**：Granet et al., *Simulating sparse SYK model with a randomized algorithm on a trapped-ion quantum computer*, [2507.07530v1][SYK]。pp2–4,8–9、Eq31/32 | TETRISから**独立**U/U′をdrawし、control二枝へ実装。平均X⊗Oは正規化係数付きのevolution後observable、O=Iがmirror-on-average。echo verificationも別節 | stochastic回路の枝間干渉とnoise診断、平均でのみmirrorとなる構成は既知 | 同じgate labelを両枝で共有しない。目標はobservable/noiseで、M1-Sのopposite-sign palindromeとRPE減衰保証ではない。実験結果の数値をM1-S費用へ流用しない |
| **KGW**：Kivlichan–Granade–Wiebe, *Phase estimation with randomized Hamiltonians*, [1907.10070v1][KGW]。pp2–4、Fig1/Thm1/Eq4–5、pp13,17 | iterative phase estimationのcontrolled evolutionを、異なるランダムHamiltonian列へ置換。Thm1は列中のHamiltonianの最小ground gapと隣接Hamiltonian差、状態追従・失敗条件を使用。後段でposterior meanの安定性を扱う | random Hamiltonian、状態依存IS、phase/success保証は直接の近接先行 | 論文のgapは生成Hamiltonian列のもの。M1-Sの固定Hのv_kmやalias距離と同じでない。gate共有±のΦを定義していない。Thm1とそのproof／posterior解析の仮定をM1-Sへ移す証明はしていない |
| **KIM**：Kim–García-Pintos, *Randomized product formulas beyond optimal deterministic scaling*, [2608.07720v1][KIM]。pp2–3、Eq2–13/Thm1、SI pp10–11 | H=A+αBでBの挿入位置uを乱択し、uと1−uの組でsymmetrize。反復segment間はIID u。channel diamond誤差のα依存を改善。強いaccess modelのrandomized THRIFTは別方式 | D/Rに似た分割、相関する対称step、乱択でleading errorを消す構成は既知 | Pauli labelのbranch共有ではなく、既知A/B指数と挿入時間を利用。強いaccessではA+αB_lも使う。M1-Sの情報access・law・coherent RPE guaranteeと同じではない。これを新候補として採択しない |
| **MLMC**：Mohammadipour–Li, *MLMC-qDRIFT*, [2604.26865v2][MLMC]。pp5–11、Eq11/14/15/26–40、§2.2/2.3、Thm1 | fine/coarseでindexを共有してobservable差を推定。§2.3は**control二枝**へのcoherent配置も記し、固定normの差observableではshot noiseが残ることを説明。augmented difference-stateと非unitary block mapを別に導入 | shared randomnessによるcoupling、coherent枝構成、経路分散と測定shot noiseを区別する必要性は既知 | fine/coarseはstep数・角度・label subsequenceが異なる。M1-Sは同深さopposite-sign interference。augmented mapの実装費用や同論文の複雑度をM1-Sへ転記しない。共通乱数を使うだけでshot利益が保証されるとはしない |
| **SCU**：Peetz–Smart–Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, [2407.21095v2][SCU]。pp2,4、Fig1/Eq1–3/6–8 | ancillaでTr[O e^(iφ)V1ρV2†]のcross termを取得。CTSでは左右propagatorを独立sampleし、normalizationとshot overheadを戻す | 量子枝間cross termを確率的に得る一般枠組みと費用・分散交換は既知 | cross-term回路の存在はM1-Sの特定joint lawやΦの長時間保証を与えない。独立CTS平均を共有label平均へ代入しない。G/D系列の既存CTSとの照合は再実行しない |
| **CHEN**：Chen–Huang–Kueng–Tropp, *Concentration for random product formulas*, [2008.11751v4][CHEN]。pp2–5、Algorithm0.1/Thm1/2 | IID random unitaryの単一実現を、全状態または固定状態に対して制御。平均channel・平均演算子・単一trajectoryを区別 | random PFのconcentrationと固定入力での改善は既知 | reused labelを含む回文wordを独立2N gateの積とは扱えない。依存を保持した適用やcontrolled liftが可能かは別の数学課題。取得したv4は2026-03-25版で、2021年出版PDFと黙って同一視しない |

### 文献照合の技術的結論と限界

**既知として扱う共通部分**は、対称PFの回文、random Hamiltonian phase estimation、coherent cross term、共有index couplingである。M1-Sで残り得る差分は「それらが無かったこと」ではなく、**明示したjoint law・時間／control規約に対し、既知保証の直接適用以上に何が示せるか**である。その評価はGPTへ残す。

検索はmirror/echo、palindromic/symmetric randomized PF、shared randomnessとHadamard/phase estimationを中心とする16 queryと、指定文献・直接ヒットの関連節に限定した。無関係なbenchmarkや一般通信のshared randomnessは根拠にしていない。全citation networkの網羅や完全同値の証明はしていない。[検索範囲と取得状態][EVIDENCE]。

KGW、COS、MIR、SYK、KIM、MLMC、SCU、CHENの**本文は取得済み**。CHENの著者サイトはDNS取得失敗だったため、版を明示したarXiv v4を使用した。mirrorの出版最終版、Claude原本、関連研究すべての補遺を読了したとは主張しない。

## 4. Protocol仕様：既存レビューをordered circuitへ対応させる

本節は[GPTレビュー][REVIEW] §§4.1–4.6/5.2と、[P0.5][P05] §3の**転記・規約整理**である。新しいprotocolを採択せず、新first-moment定理を証明していない。完全乱択の参照だけを具体化し、D≠0の式を新たに作っていない。

### 4.1 順序・符号・位相

右端が先に作用する表記を使う。レビュー§4.2のA_j=exp(+iθP_{j})を参照記号とし、chronological label列ω=(j1,…,jN)を固定する。符号付きPauliはP_j²=I、p_j≥0、Σp_j=1、列内はIIDというレビューの限定である。これはλ、θ、rを新しい実行条件として選んだものではない。

| 方式 | B0／B1と順序 | 乱数の同時分布 | 観測・平均対象と参照 |
|---|---|---|---|
| ordinary | B0=I、B1=指定ordered word U_ω | 一つの列。shot間freshnessは別条件 | ⟨ψ¦E[U_ω]¦ψ⟩。[P0.5][P05]と旧ordinary wrapper |
| 独立± | B1=正時間law、B0=負時間law。literal adjointの条件を満たすにはgate inverseと**逆順**も指定 | ω+とω−独立。単に異なるseedを付けたことだけを独立性保証にしない | E[B0†B1]の因数分解条件はP0/P0.5／PR A46。物理半時間・D配置が合うことまで自動保証しない |
| **gate共有±（M1-S参照）** | B1=A_jN…A_j1、B0=A_jN†…A_j1†。両枝は同じchronological label順でsignだけ反転。B0をB1全体のliteral adjointとしない | 同一j_kを二枝で共有、異なるkはIID。別shotへのreuseは含めない | レビュー§4.2の回文wordとΦ^N(I)。ここは完全乱択参照の既存恒等式である |
| full-block共有± | B1=U_ω、B0=U_ω†。負枝はfull wordの逆順inverse | 一つのfull wordを共有 | E[U_ω²]型という既存レビューの指摘。同順gate共有と区別し、性能negativeとはしない |

M1-S参照の干渉wordはレビューの回文表示に合わせ、A_j1…A_jN A_jN…A_j1とする。その平均写像としてレビューが記した

`Φ(X)=E_j[A_j X A_j]`、`E[B0†B1]=Φ^N(I)`

を**既存の限定的な式として引用**する。A_j†を右側に持つ通常channelと違う。共分散式・v_kmの式・長時間一次展開はレビューに戻り、今回再導出しない。新しい証明、固有値評価、一様remainder、RPE保証を作成していない。

各実装でP_jの符号、scalar identity、global phaseのbranch依存、control polarity、X/YのS†規約を保持する必要がある。system上のprojective同値はcontrolled relative phaseの受理条件にならない。レビューの+時間表記と既存sourceのexp(−iHt)を黙って混ぜず、実行契約を作る場合に対応を固定する。

### 4.2 物理時間とD blockは未採択

- N、θ、λ_R、arctan angleとlinear angleのどちらを使うか、半時間とfull signal timeの対応は未固定。異なるensemble間で同じNやθを与えるだけでは同じ物理taskにならない。
- gateごとの共有を「一step内だけ」とするか、複数step全体に及ぼすかは別仕様。IIDの範囲、X/Y・round・shot間の独立／共有も別項目である。
- 非可換Dのforward/reverse half-step、branch別位置、scalar phaseを含めたordered circuitは未採択。完全乱択の固定Φ反復をそのまま当てはめない。
- [P0.5][P05]で追跡済みのH6 v5は決定論directionalとordinary random tailを扱う。M1-S samplerや二枝lawの実装済み証拠ではない。今回はsourceを変更・import・実行していない。

この欠落は新しい数値計算で埋めるものではなく、M1-Sを採択する場合に対象statementと共に定義する事項である。

## 5. 公平な実装費用比較に必要な仕様

ここでは**費用契約に必要な項目**だけを整理する。新しいgate数、T価格、shot数、誤差予算、計算上限を採択・計算していない。

### 5.1 共通primitiveと課金対象

レビュー§5.2の既存分解

`diag(I,exp(−i2θP)) = (I_c⊗exp(−iθP)) exp(+iθ Z_c⊗P)`

は、ordinaryとdirectionalを同じ「1 rotation」の価格と置けないことを示す参照である。gate共有側にもancilla込みPauli支持のbasis change・parity network・phaseを課金する。具体的なCX数やT数は本調査で出していない。

| 比較項目 | ordinary／独立±／gate共有±に共通して固定すべきもの |
|---|---|
| 物理task | 同じH・分解・状態情報・物理時間・最終観測量。complex信号、固有位相、RPE energyのいずれかを明示し、RMS／高確率の定義を共通にする |
| algorithmic bias | 各方式が実際に生成するmean／signalに対応するbias。qDRIFT angle歪み、Dのouter PF、correlationによるsignal変更、near-zero/abstentionを別項目にする |
| Pauli lowering | Pのsupport・basis mapping、Z_c⊗Pのparity network、uncontrolled補助回転、branch inverse、scalar/control phase。gatewise controlとfull-word generic controlの価格を区別する |
| 精度 | 同じ最大coefficient／finite-bit／synthesis予算、strict controlled phase、operator error規範と信号への戻し方。合成後に都合のよい方だけ精度やangleを選び直さない |
| reuseとcompile | 合成器／version／seed／cache key／phase identity、全方式共通のfusion・inverse cancellation・routing規則。別wordのwhole-wrapper価格を加法単価へ無条件に移さない |
| 統計 | lawからのmoment/range、物理的signal floor、初期overlap、round/axis confidence配分、十分shot予算。経路間共分散と量子測定shot noiseを区別する |
| 情報取得 | 分解・state/gap情報、normalizer、preprocessing、classical online生成、controllerのbranch同期。共有した乱数を無料の完全な情報取得と扱わない |
| 付帯費用 | state preparation、readout/reset、ancilla/workspace、measurement overhead。未測定online費用は0でなくNOT_MEASURED |

各方式のsample数・gate数が同じことと、同じ精度の最終taskを解くことは別である。PRの独立RTE ±、ordinary qDRIFT、M1-Sは必要に応じた強い既知対照の候補だが、どれを正式baselineにするかは今回採択しない。各対照へ許す精度・sampling・compiler最適化の機会も事前固定事項である。

### 5.2 価格座標を混ぜない

- **FTQC HWP＋catalyst**：commuting/equiangular集合、ordering、触媒準備・再利用、ancilla、Toffoli/rotationの課金が要る。IID wordを価格のために並べ替えることはlaw変更になり得る。PR Appendix Eの既知会計がM1-Sへ成立するかは未確認。
- **compiled RZ/CX/depth**：任意角RZの本数はT数でない。whole-circuit最適化後の価格と加法primitive価格は同じモデルでない。
- **Clifford+T**：角度ごとの実sequence・strict phase・precision・誤差配分を伴う。旧D1〜D3/S4の価格はM1-Sの登録価格ではなく、今回は合成keyの新規取得もしていない。
- **STAR**：logical non-Clifford rotationを単位とするモデル。富士通のC_gateやγをT/CXへ換算したり、M1-Sの未認証の追加減衰仮説と乗算したりしない。

M1に整合する長roundのD/R費用割合は[P0.5][P05] §4の**NOT_ESTABLISHED**のまま。旧screeningやlate-round割合から新しい割合・forecastを作っていない。

## 6. GPTが次に判断する論点

G0-preとして資料収集を終えた。次の科学的判断をCodexが代行しない。

| 判断事項 | 揃った技術材料 | 未解決／G0に進むなら必要なこと |
|---|---|---|
| 方法上の差 | COSの対称word、MLMCのcoherent共有、SCU/SYKのcross termを具体的に照合した | M1-Sのjoint law・target・accessに実質差が残るか。既知法の組合せ／直接系なら独立claimをどう限定するか |
| 対象定義 | 完全乱択のgate共有±をB0/B1・順序・位相に対応させた | ordinary／独立±と同物理taskを定義し、非可換Dの配置・時間・angle・randomness範囲を正式採択するか |
| 保証の価値 | 最新レビューはΦの恒等式と、長時間一次展開を区別している | Nθ²=O(1)での一様剰余、低周波重み・sector外support、gap/alias、overlap/signal floor、RPE成功保証を閉じる価値があるか。今回は証明していない |
| 総資源への接続 | 同一primitive、control phase、native／HWP／STAR、shot noise、付帯費用の必要項目を整理した | 保証が成立しても同条件costで意味があるか。費用座標と未取得情報をどこまで取得する必要があるか。新compile/forecastの採否は未判断 |
| 代替／停止 | 元M1のfactor-twoは既知、M1-AとM2の位置付けは保持 | M1-S採択、限定理論、M2、休止のいずれを選ぶか。今回は採択／科学的STOPを出していない |

「同じ結果が見つからなかったからG0へ進む」「文献に回文があるから全M1-Sを終了する」のどちらにも自動移行しない。必要なのは上の対応に基づく科学的な差分・価値の判断である。G0の新理論やV1の契約・実装は作成していない。

## 7. 未取得資料・公開と停止

- Claude原文`レビュー_inner(1).md`は今回のworkspace内の提供物として未取得。GPTレビューが記録したSHA256は受領記録に残したが、原本hashを確認済みとはしない。今回のprotocol参照は提供されたGPTレビューの明示式に限定した。Claude全文の独立再審査ではない。
- 8PDFの関連節を取得・読解し、版・pages・hashをlocatorへ保存した。論文の全証明の再現、M1-Sへの新しい還元証明、網羅的新規性調査は実施していない。
- **新証明、G0、新しいsimulation、行列評価、operator guard、回路生成、compile、synthesis、sampling、trajectory、LP、費用再集計・比率・shot/resource forecastは0。** 既存source/result/classification/auth/marker/STOPを変更しない。
- 共通指示に従い、新原本・報告・索引の取得性を確認し、親worktreeの既存HEAD/index/dirty fileを保護する。本書は研究採択の承認記録ではない。

**作業完了後はmandatory STOP。次のGPTによるM1-S採択判断は利用者の明示的な開始承認後。採択されてもG0・科学計算・実装の範囲は別途固定し、自動開始しない。**

[REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03/docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md
[INTAKE]: ../../../artifacts/track_b_m1s_g0_pre/2026-10-15/v1/review_intake.json
[COMMON]: ../../research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
[P05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[P05INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_reference_index_20261015.json
[LOCATOR]: ../../../artifacts/track_b_m1s_g0_pre/2026-10-15/v1/primary_literature_locator.csv
[EVIDENCE]: ../../../artifacts/track_b_m1s_g0_pre/2026-10-15/v1/evidence_index.json
[COS]: https://arxiv.org/pdf/1805.08385v2
[MIR]: https://arxiv.org/pdf/2008.11294v2
[SYK]: https://arxiv.org/pdf/2507.07530v1
[KGW]: https://arxiv.org/pdf/1907.10070v1
[KIM]: https://arxiv.org/pdf/2608.07720v1
[MLMC]: https://arxiv.org/pdf/2604.26865v2
[SCU]: https://arxiv.org/pdf/2407.21095v2
[CHEN]: https://arxiv.org/pdf/2008.11751v4
