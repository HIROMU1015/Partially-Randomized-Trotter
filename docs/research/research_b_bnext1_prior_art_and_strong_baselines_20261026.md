# 研究B B-Next 1：先行研究との差分と強いbaselineの限定静的調査

2026-10-26 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。
作業起点：`3cfbac1d4f7df751e972170e54ad21aa947ceff1`。
本書は既存証拠・原論文・SOURCEの静的調査であり、新規科学レビュー、数学的保証の独立認定、科学計算、保存値の数値再解析、実装、研究継続の正式採択ではない。

## 1. Executive Summary

**現時点で、研究Bの独立した新算法構造を支持する具体的差分は特定できなかった。** 高次PFとRTEの併用、支配部分だけの高次化、processing、signed stage別のRTE配分には明確な先行対応がある。現在の成果の技術的位置付けは、既知方式を固定DF入力へ接続した誤差診断・実装検証・条件付き資源評価である。[統合報告][INTEGRATED]、[PR原論文][PR]、[SPRINT][SPRINT]

次の判断を変え得る未取得証拠は、まず**単一DF因子のExact-Rとfull deterministic DF-PFを同条件の強い対照として加えたとき、現在の資源上の解釈が残るか**である。両者の回路primitiveは存在するが、現Bの評価器へ接続済み・検証済みという意味ではない。Exact-Rが安いとは仮定しない。

- **既知の部分**：D内部Yoshida4／Morales、SABA2、processor相殺、一般PFの各R stageをRTEで置換する構成、時間絶対値を使うstep配分。
- **研究Bで追加評価した部分**：D内部・outer・finite-RTEの対照診断、actual inverse／published列照合、固定full prepared wrapperとphase保持修正、S0条件付きshot会計と未知rare費用の感度。世界初の評価と認定したものではない。
- **不足する強い対照**：Exact-Rの実回路、全DFの決定論的PF、調整済みA2のn_D、既存compact PF。現S0費用比較から方式の最小資源順位は決まらない。
- **旧研究との関係**：H2/H4/H6ではすでに高次PRとshot加重費用を評価した。R2-CSは事前構成選択の独立研究として正式終了し、P1も未実行のまま実行予定から除外された。双方の履歴・STOPを維持する。

A：独立算法研究には具体的機構差が必要で、現段階では未提示。B：既知方式の適用・baseline不足を検証する限定研究には問いが残る。C：synthetic系列を一区切りにし追加計算を保留する選択も維持できる。A/B/Cの採択はユーザーへ返す。

## 2. 調査対象と固定証拠

主資料は[研究B統合報告][INTEGRATED]、[H2/H4/H6保存費用報告][H246]、[研究B初期調査][INITIAL]。指定GitHub commitから取得し、研究Bの8一次報告・実行SOURCE・最終状態を統合報告と対応させた。研究B一次報告は同じ会話で取得・読取り済みの固定版を利用し、今回数値表を再集計していない。

原論文はPR v2、SPRINT v1、Morales v3、CPF v4、Laskar–Robutel v1を取得した。PRはHTML版が404だったためPDFのAppendix A.2/A.3、D、Eを確認した。追加文献は直接関連するHagan–WiebeのComposite Quantum Simulations v3に限定する。全証明・全参考文献・全branchを網羅した調査ではない。

R2-CSはremote tracking参照のC0 `8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88`、closure `270078a87c1febe8944f829270e8c2a00dbed7de` を固定して公開本文を取得し、C0.6 `dee25ba13ee8ebcb23c7d294c659ea678d818277` の終了提案もgit原本で確認した。[C0][CS0]、[C0.6][CS06]、[正式終了][CSCLOSE]

背景は既読の研究概要・研究目的・validation status／manifestを用いた。統合作業commitでこれらの科学本文が変わっていないことを確認し、今回再生成や科学テストをしない。ユーザーが要約した2026-10-25 Level 1レビューは調査の動機として受け取ったが、**レビュー原本文書は未取得**であり、独立に読了・再レビューしたとは扱わない。

## 3. 現研究B：成立した評価と、その特殊性

モデルはsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0占有、L_D=2 weight-ranked prefix、Dはone-body＋因子0,1、残差は因子2。分子geometry／AO basis／分子DF rank探索は非該当。36行はδ=±0.09・q2/3・n_D1/2・r2K2、後続はT=±0.18・主にq2・n_D1、候補別r/Kとrefinementを使用した。[統合 §3][INTEGRATED]

| 既存評価 | 最新の到達点 | 未取得・適用外 |
|---|---|---|
| D内部36行 | A0現partial-S2、A1 D-S2、A2 Yoshida4、A3 Y8m10b、A4 YP8m8。入力・A4数学・published列確認後COMPLETE | A3/A4優劣は丸め水準でINCONCLUSIVE。Native費の比較なし |
| Outer16行 | S2/refined-S2/Yoshida4/SABA2とA2/A3。Corrected signal改善を固定窓で観測 | Raw改善・SABA2一般四次・同費用優位ではない |
| Outer/RTE22行 | r/Kと内部PFの相互作用を対照。全correctness後に完了 | 全行u_num未資格、K4微小差の識別限界 |
| A0高精度 | 8行の90/128桁、EMPIRICAL_CONSISTENCY | Intervalによるsignal包含、certified u_numではない |
| B0 native | 最終v5で全48代表／96 full wrapper完了 | 代表費用は期待費用・全support上界でない |
| B1 IID費用 | 3,072 trajectory／6,144 wrapper完了 | 全51,200 short-stepがorder0。INCONCLUSIVE_RARE_EVENT_COVERAGE |
| Rare感度 | 未知rare費用を残すCONDITIONAL_SENSITIVITY_ONLY | 母平均費用順位なし |
| Shot24条件 | 48軸別shot数・144資源行・24比較を完了 | u_num／γ／rare費用は未取得。INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS |

上表は[統合報告の既報値][INTEGRATED]を引用した。A0高精度時点の「B0未実行」、B0途中の「未解決」は履歴であり、現在のB0完了と混同しない。実行COMPLETEを科学的GOへ変換しない。

**特殊性の数学的説明**：N=1では、任意の基底でn_j n_k=0（j≠k）なので、DF因子の二乗はそのsectorで一体作用へ還元される。これはnumber-operator恒等式からの静的な説明で、新しい行列評価ではない。非可換性があっても、多電子の二体相関・長い非可換tailを検証したことにはならない。全Hの対角化を無料の量子oracleにはせず、以下のExact-Rは単一DF因子の既存物理primitiveを使う対照として扱う。

## 4. 原論文との対応と差分分類（表1）

分類は技術調査上の対応であり、新規性の正式判定ではない。`POTENTIAL_METHOD_DELTA`は異なる機構・構成を具体化できた場合に用いるが、本調査では該当する現実装を特定できなかった。資料不足は`UNRESOLVED_PRIOR_ART`として残し、既知ではないことの証明にしない。

| 比較対象 | 数学的構成・task・資源指標 | 明確に既知の部分 | 研究Bの追加評価／差分候補 | 分類・根拠 |
|---|---|---|---|---|
| PR原論文 | 一般PFのR指数をRTEへ置換しHadamard first momentを補正。Phase/energy taskのgate×samples | 高次PF、signed stage、stage別step、normalization積、有限次数打切りの可能性 | 固定有限K・DF-native・複素signalの詳細対照 | KNOWN_METHOD／KNOWN_METHOD_NEW_EVALUATION。[A32–A41、E][PR] |
| SPRINT | 支配groupの高次PF・弱いgroupの低次PF、full processing、順序乱択、GRADE/QROM。XAS・Toffoli/qubit | D側に精度を集中する機構、反復内部PF、残差へRTE等を使う接続案 | 現Bのsigned有限RTEとprepared X/Y native会計は別評価。SPRINTそのものを実装したわけではない | KNOWN_METHOD_NEW_EVALUATION。[Fig.1、III.2–3、IV][SPRINT] |
| Morales | S2 compositionとprocessed PΣP^-1、operator/eigenvalue誤差・列長 | Published Y8m10b/YP8m8とprocessor償却 | DFのD内部への適用、R障壁ごとの端点、full coherent signalでの識別 | KNOWN_METHOD_NEW_EVALUATION。[II.3、Tables1/2、IV][MORALES] |
| CPF | Correctorの共役／対称挿入、perturbed系、operator誤差・指数数 | Correctorで既存誤差項を除去、合成誤差、非exact partitionの議論 | 現A4はMoralesでありCPF実装ではない。Finite-RTE kernelへ保証移送は未確定 | KNOWN_METHOD／UNRESOLVED_PRIOR_ART。[§2、Prop.1、§5/8][CPF] |
| SABA | 弱摂動A+εB、positive splitting、modified Hamiltonian誤差 | SABA2係数と一重B項の選択相殺。一般四次ではない | 近似D・有限RTEを入れた固定signal／費用評価 | KNOWN_METHOD_NEW_EVALUATION。[§4、(14)–(19)、Table I][SABA] |
| Hagan–Wiebe | 高次TrotterとqDRIFTのcomposite channel、partition、query/error | Deterministic/randomを階層的に組み合わせる一般構造 | 現Bは位相を保つfirst momentと有限RTE。Channel保証をそのまま移送しない | KNOWN_METHOD。[§5、Def.7 (57)–(58)][COMPOSITE] |
| 旧H2/H4/H6 | Signed S2 compositionの高次PR、測定込みGの有限探索 | S2/Yoshida4/m5、stage別整数allocation、shot/nativeの競合 | Nested D内部を独立に変え、outer・finite tailの残差を診断した現Bの対照設計 | KNOWN_METHOD_NEW_EVALUATION。[旧結果][H246]、[旧SOURCE][OLDPLAN] |
| 現Bの新算法という主張 | 既知PF・DF primitive・RTE・wrapperの接続 | 新adapterやcompiler数値修正は実装上の追加 | 新しい生成原理・誤差除去則・安価な選択certificateは未提示 | NOT_SUPPORTED。性能・新規性の不存在を証明した意味ではない |

### 原論文で確認した、判断を左右する箇所

**PR v2**：Appendix A.3はRを一項とする一般PFを定義し、正負の各R時間δ_jとstep r_jを持たせる。全normalizationを積で合成し、絶対R時間を用いる整数配分を提示する。A.2は無限RTEの補正meanと有限打切りに言及する。従って「高次PR」「負stage対応」「stage配分」自体は独立差分にならない。特定native価格と有限Kを含む最適配分が解決済みとまでは読まない。[A23–A41][PR]

**SPRINT v1**：Fig.1、III.2 Eq.(28)は高次A／低次Bのnested構成、Eq.(31)はA側の内部反復を扱う。III.3 Eq.(32)はfull evolutionのprocessor相殺。Fig.1は小さい第三groupへのRTE等も挙げる。主な実証taskはXASで、今回の固定prepared signal会計とは異なる。本文のhalf-stepとAppendix E.3のsquare-root記法を、照合なしに同じ回路列へ置換しない。[SPRINT][SPRINT]

**Morales v3／CPF v4／SABA v1**：既知係数・処理・弱摂動の対応を確認した。Moralesはkernelだけのeigenvalue精度とprocessor込みoperator精度を区別する。CPFは主partitionのexact access、近似小partitionの条件、corrector合成を論じ、一般のnested近似への可能性も述べる。SABAのmodified HamiltonianでのO(ετ^4+ε²τ²)をunitaryの一般四次保証へ変えない。[Morales II.3/IV.3][MORALES]、[CPF §2/8][CPF]、[SABA §4][SABA]

## 5. 旧H2/H4/H6との具体的差分

[旧結果報告][H246]には、すでに「緩い精度ではS2、厳しい精度では高次PFが低費用になり得る」という有限探索結果がある。線形H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8、保存DF rank3/10/19、prefix1/3/5、T=.8、P=0、K4、ABS_TIME_PER_OUTER32、q1/2/4/8。ε=.0005の探索点最良はH2 Yoshida4/q2、H4/H6 m5_best/q1で、対S2削減率約47.12%／53.30%／63.12%が既報である。新しい計算や現在Bへの改善率転用はしていない。

| 比較軸 | 旧H2/H4/H6 | 現研究B | 直接比較の可否 |
|---|---|---|---|
| 入力・state | 保存分子DF・多電子状態hash固定、generation prefixを確認 | Synthetic2mode/N1/mode0、weight-ranked LD2 | 不可。同じ名前のprefixでも条件が異なる |
| PFを変える位置 | `S2_partial(w_j T/q)`のsigned composition。元D sweepもRも各pieceで反復 | Nested D単独A2/A3/A4とouterを別に変える | 構造差を説明可能。数値的優劣の転用不可 |
| Time/grid | T=.8、q1/2/4/8、K4・総step32/outer | T±.18、主にq2、n_D1、r2/4・K2/4 | 不可。qやrの同じ数字でも仕事量が違う |
| 精度・shots | 5精度.05/.01/.005/.001/.0005、軸headroom・経験的allowanceによる会計 | ε1e-8/1e-9、b+u_num・γ・統計を別予算、資格null | 同精度Gとして結合不可 |
| 単価・control | P=0、symmetric-directional、cos/sin paired、旧native系列 | Mode0準備込みordinary controlled、X/Y paired、native-v5 | 同じQiskit/basisでも価格同一でない |
| Correctness | 保存state-vector等の旧閾値・CT01条件を維持 | Full operator spectral atol1e-12／phase fitなし等 | 一方のPASSを他方のcertificateへ流用不可 |
| 統計・選択 | 探索cohortでqを選択、未観測層・選択後保証未取得 | n256固定点、全S0・unknown rare感度 | 母平均最適性・一般順位へ拡張不可 |

[旧plan][OLDPLAN]と`h46_m5_native.py`を静的確認した。旧signal経路にもexact-tail／corrected／rawとscalar-onceの分離がある。**Ideal-tailを分けて誤差を見ることも、shot×単価会計も、現Bで初めて導入した発想ではない。** 現Bが加えたのはD内部PFとouterを独立に変える対照、published processingの実現照合、現在のfull-wrapper／scalar区間会計等の具体的情報である。

旧planの`SYSTEMS`には初期H4 rank12が残る。上表の最終rank10/19は最終報告・入力bindingを優先した。[取得SOURCEのsignal adapter][OLDSIGNAL]はbindingのrankを使い、[cost adapter][OLDCOST]はそのcontextと既存native描画を再利用する。[探索runtime][OLDRUNTIME]も既存pilot経路を接続している。planのschedule再利用と最終入力identityを区別し、初期plan定数を現Bの入力へ転用しない。

旧報告はH4/H6のm5決定論部分集合とも比較したが、全強化S2/S4集合への優位は未確定、H2の同系列決定論結合はnullだった。これを「full deterministic対照は一般に不要」と読むことも、現Bで検証済みと読むこともできない。旧結果・採択・STOPは維持する。

## 6. Exact-R：最優先baselineの実装可能性

### SOURCEが示すRとscalar

[fixture原本][FIXTURE]の係数は`[0.6,-0.35,0.22]`、constantは0.17。既存B0の独立CAR参照は、F_j=Σ_pq g_j,pq a†_p a_qとして各fragmentをλ_j F_j²と定義し、`R=fragment[2]-extracted_identity*I`、`scalar=constant+extracted_identity`とする。R因子のλは+0.22であり、負係数−0.35はD側にある。これはSOURCE literalの確認で、固有値・norm・費用の再評価ではない。[b0_native.py:full_inputs][B0SRC]

単一DF因子はGaussian基底変換でFをΣ_j η_j n_jへ対角化できる。既存[二乗対角primitive][OPS]は、number operatorの二乗を

`(Σ_j η_j n_j)² = Σ_j η_j² n_j + 2 Σ_{j<k} η_j η_k n_j n_k`

としてI/Z/ZZへ表し、signed durationとλを回転角へ含める。数値的な全H対角化ではなく、DF因子固有の既存指数実装である。Fock全体でこの構造を使え、N1だけの省略に依存しない。

### 再利用する経路と必要な差分

| 要素 | 静的に確認できた既存実装 | 将来接続時の責務 |
|---|---|---|
| RのDF basis・対角値 | `build_df_blocks`、`DFDeterministicFragmentSpec`、basis registry | 現D-only specにない因子2のblock specを同じ入力から供給。旧D contextを書き換えない |
| Basisとactual inverse | `QiskitDFPartialS2CircuitBuilder._append_basis` | 既存のinverse→diagonal→forward規約を使う。別合成列をinverseとみなさない |
| Signed指数 | `append_block_evolution`→`df_squared_diagonal_primitives` | 各outer R時間をそのまま渡す。絶対値にするのは費用／normalization説明側だけ |
| Control・relative phase | `append_diagonal_primitives`はcontrolled RZ/RZZとancilla PhaseGate | Basis往復は非制御でもcontrol0で相殺。I成分のphaseを捨てない |
| Full wrapper・compile・metric | B0 `wrapper/compile/measure`、prepared Hadamard helper、stable_phase=True | 同じprep/axis/measurement/profileへ接続。Exact-R分岐・scope・独立参照は未実装 |

Centered Rを保つなら、その指数は非centered DF指数と`exp(+i c_ext t_R)`の組になる。既存の`exp[-i(c0+c_ext)T]`と二重計上しない。因子を戻したからといってscalarを旧値のまま無条件に流用できない。浮動係数の実signed時間和と公称Tも記録する。全Hの定義は不変とし、旧runを遡及変更しない。[B0 SOURCE][B0SRC]、[block builder][DFBUILD]、[tail identity][TAIL]

**適合性：primitiveを再利用した限定adapterで実装可能と判断する。** 現B0はD blockの描画とRTE eventを区別しており、Exact-Rの専用入口・比較・native correctnessはまだない。既存primitiveの人工／科学test本文にはcontrolled phase等の検査があるが、今回は実行せずExact-R PASSの代替にもしない。[tests/test_df_partial_s2.py][DFTEST]

Exact-RではRTE normalization・Taylor打切り・RTE samplingは使わないが、Gaussian往復・対角回転・制御・合成の費用と誤差は残る。Normalizationを1と置けることだけでshot数やG、RTEに対する費用優位は計算しない。旧analyticのexact-tail参照は古典診断で、Exact-R量子回路の費用証拠ではない。

## 7. Full deterministic DF-PF：最優先の全体対照

同じphysical Hのone-bodyと**全3 DF因子**をordered block列としてS2／Yoshida4等で近似する。Rを落とすtruncation／B0Iや、全Hの無料exact oracleとは異なる。現fixtureではone-body＋DF0/1/2を含むfull列と、Dだけを近似してExact-Rと挟むnested列も別のbaselineである。

現SOURCEには[build_df_trotter_circuit][DFCIRCUIT]と`iter_pf_steps`、[block単位の制御付き描画][DFBUILD]がある。前者は非制御のmixed-block PFで、そのままB0のcontrolled prepared wrapperにはならない。後者は任意signed block時間を供給できるため、**現在Bのordinary controlを保った全block列の接続**が最小の候補となる。

さらに公開済み[AX-2B native provider][NATIVEOLD]には`build_deterministic_native`があり、S2／Yoshida4、ordinary／symmetric-directional、scalar-once、Hadamard終端に対応する。指定旧H246結果commit自体にはproviderのpathが存在せず、履歴上の固定公開commit `31cdff47f3282d2898c153a1af22f90e500be4c6` から取得して確認した。**このproviderを旧H246 runtimeの全closureと同一だとは認定しない。** 再利用する場合は現在のprepared wrapper・v5 compiler・独立参照へ明示接続する。

全DFを決定論に置く別preparationを作る場合もphysical Hの入力identityと全factorは保ち、tail extractionが不要になる分のscalar政策を照合する。N=1 sectorだけで得た古典行列のexact指数を回路価格として計上しない。Full PFの必要q、bias、実native単価、shot加重費用は未評価である。

## 8. 調整済みYoshida4・compact PFの接続（表2）

36行ではA2のn_D1/2を比較したが、後続のouter／native／shotはn_D1固定だった。[DStage][OUTERSRC]は`tau`と`repetitions`を表し、n_D変更のdescriptorを作れる。一方、B0 `reference`／`analytic_mean`は現在n_D1のliteral列を直接展開する。**候補rowのn_Dだけ変更すると、描画と独立参照の対応が崩れる。** 接続するなら参照・inventory・science scope・input binding・費用schemaも限定更新が必要で、既存SOURCEへ後付けしない。

Compact／高次PFは固定PF repository `20d64c2fea1ed2d2188777df4f3d962ee94df486` の実関数を取得した。[PF SOURCE][PFSOURCE]

- Yoshida4は3 S2 composition、旧m5_bestは11 signed S2のnon-processed固定binary64列。m5はH3実回路探索由来の既存係数で、普遍的な「最良4次法」ではない。旧[source receipt][M5RECEIPT]は厳密代数certificateを主張せず、中央係数のbinary64演算順を固定している。
- Published Y8m10bは21 S2、YP8m8は17 S2 kernelと片側20 S2 processor。Legacy `morales_8th_list`とpublished版を混同しない。現repository直下の同名product_formula moduleには版差があり、現Bは固定抜粋`_baselines.py`／`_published.py`を用いる。
- P†は同じ実現Pの逆順・符号反転。隣接同block・同contextの融合は可能でもRを越えた相殺、別shot間の相殺はできない。D内部processedとfull列processedでは端点位置が異なる。[静的構造費][STRUCT]

| Baseline | 再利用SOURCE／実装済み範囲 | 未接続部分 | 同条件比較の可否 | 必要な新科学評価 |
|---|---|---|---|---|
| Exact-R | DF二乗primitive、signed block、B0 phase保持wrapper | 因子2 spec、centered identity、R分岐・独立参照 | 可能な設計。現run未検証 | R/controlled/full列correctness、signal、native単価 |
| Full DF-S2／Yoshida4 | Mixed-block PF、制御block、旧native deterministic builder | 全block contextと現在のprep/compilerへの接続 | 同じphysical H/state/Tで可能 | 全列correctness、qと精度の対応、費用 |
| A2 n_D調整 | `make_arm_plan`、`DStage`、36行n_D1/2、関連tests | B0独立参照・analytic mean・inventory等のn_D対応 | 後続r/K/outerを固定すれば対照可能 | 調整点のbias・full wrapper費。現結果へ外挿不可 |
| m5_best | 固定11 S2列、旧H246 signed schedule | 現D/outer whitelistとnative参照、係数来歴 | 既存列を固定して接続可能 | 同taskのbias／normalization／費用。再探索不要 |
| Published Y8m10b／YP8m8 | A3/A4のdescriptor・math test、固定PF抜粋 | A4 native接続、full PF側のendpoint描画 | Processor費込みなら設計可能 | 用途に必要なn/qと実費用。kernel価格だけでは不可 |
| Stage別r/K | 異種RTEConfig、signed moment、ordered N積 | 現固定candidate/launcherのstage mapping・scope | 同じ誤差targetと総仕事量を明示すれば可能 | 配分の効果。最適性・新規性は別問題 |

表2の「可能」は静的な接続上の見通しで、接続済みや性能PASSではない。すべて今回未実装・未実行である。

## 9. Stage別RTE配分：既知理論と現在の実装の境界

PR Appendix A.3のstage別r_jとN積は既知。旧H246の`h46_m5_plan.schedule`はPER_PIECEまたはABS_TIME_PER_OUTERを持ち、後者は各signed係数の絶対値に従う整数配分を使う。従って「長いstageに多くstepを配る」は現B固有の発想ではない。[PR][PR]、[旧plan][OLDPLAN]

現[make_rte_config][RTE]はsigned time・r・finite cutoffを受け、`compose_finite_rte_occurrences`はoccurrenceごとに異なるtail/time/r/K/Nを扱う。ただしこれはtail同士のordered積のhelperで、Dが交互に入るfull列を置換しない。[outer.py][OUTERSRC]はD/R interleavingと各N_jを扱うが、`make_outer_plan`に渡すr/Kはcase内で共通、22行候補ではcaseごとのr/Kを変えた。

任意stage別r_j/K_jは基礎configを利用できる一方、現固定grid・launcherがすでに最適化しているわけではない。Finite Kの打切りbias、signed law、normalization積、実回路費、精度予算を合わせる配分は新しい評価が必要になる。原論文が示す理想RTEの配分式を、有限bit・native費込みの最適配分保証へ拡張しない。

「異種config対応」はKNOWN_METHOD／実装能力、「費用・誤差を踏まえた配分の実益」は未取得評価。「stage配分という新算法」はNOT_SUPPORTEDで、より具体的な新しい機構の先行対応は未調査ならUNRESOLVED_PRIOR_ARTとする。単なる最適化目的の追加で方法差を作らない。

## 10. R2-CS終了との関係

R2-CSの科学的問いは、全candidateのsignal／native費を取得する前に、情報と取得予算からPF次数・q・partition等を事前予測・選択できるか、だった。C0ではH4/H6 Full S2/Y4・q1/2/4/8・ε=.001の16 endpoint P1案を置いたが、**これは未実行のまま終了時に実行予定から除外**された。[C0][CS0]、[closure][CSCLOSE]

C0.6では安全除外、全path biasによる一方向適格性、同じR=qrのlaw同値classを検討した。有用な既存G6・finite-RTE・区間比較・cacheは保持したが、一般法を超える安価で有用な選択機構を特定できなかった。ClosureはR3、SPRINT、cost-aware sampling、multi-fidelityとの重複、oracle情報取得の不足、P1実行でも方法差を解消する見通しがないことを終了理由として採択した。[C0.6][CS06]、[正式終了][CSCLOSE]

最終状態は`R2_CS_INDEPENDENT_SELECTION_RESEARCH_CLOSED`、P1は`REMOVED_FROM_R2_CS_EXECUTION_PLAN_UNEXECUTED`。C0のHOLDや未採択記述は履歴として維持する。一般の構成選択が不可能という定理でも、他の限定研究の無価値判定でもない。

| B-Nextで考え得る作業 | 重複と境界 |
|---|---|
| q/n_D/r/K/partitionをproxyで事前選択、regret・照会削減を売りにする | R2-CS/R3と実質的に重なる。名称変更で再開しない |
| 同law／normalizationのcacheを共有する | 既知のsetup削減。signal・費用順位・安全除外の新保証ではない |
| 単一DF因子Exact-Rという欠けた対照を測る | 特定物理accessのbaseline検証で、事前選択研究とは別。新算法ではない |
| 同fixtureでfull DF-PFや調整済み内部PFと比較 | 比較の不足を埋める応用評価。低費用予測器の研究へ自動拡張しない |
| 新しい独立選択法を主張する | 既存法と異なる具体的命題・取得機構が必要。今回見つけたとはしない |

今回R2-CSの実行・test・proxy fit・P1再解析は行わず、STOPを維持する。

## 11. 公平な比較の成立条件

| 項目 | Baselineへ共通に戻す条件 | 現在の不足 |
|---|---|---|
| Hamiltonian | 同じ保存primitive、DF全factor・符号、scalarとidentity | Exact-R／full PFに対応した入力結合を新adapterで確認 |
| State/sector | N1/mode0、同じbit mapping。Full operator検査とsector signalを区別 | 旧分子状態・別sectorの結果は転用不可 |
| Time | 同じT=±.18、公称Tとsigned列の実時間を保持 | qを変えるならδも変わる。δ固定でTを変えない |
| Approximation/target | Physical complex signal直接誤差を主、operator・rawを別保存 | Energy bias／eigenvalue誤差・channel normと同一視不可 |
| Error target | εとdeterministic/implementation/statistical予算を揃える | u_num・γ未取得。観測bias点予算は保証でない |
| Normalization | RTE各N_jとrangeを保持。Exact-RはRTE補正を使わない | 旧shot整数を新baselineへ流用不可 |
| Resource | RZ/CX/count/depthを個別に、prepared X/Y full wrapper同士で比較 | S0条件付き単価対deterministic単価だけでは無条件G比較でない |
| Preparation | Mode0準備とmeasurementを含める現在のscope | 旧P=0価格へ混ぜない。Shot数が違えばprepは相殺しない |
| Compiler/control | 同じnative-v5、profile、ordinary control、phase fitなし | 旧symmetric-directionalや旧compilerとの数値混合不可 |
| Tuning | 各方式に合理的なq/n_D/r/K調整機会と同じ情報・探索budget | 現24条件は固定点。方法別最小Gや最適性は未評価 |
| Guarantees/statistics | 経験的bias・MCのSE・quantum confidence・certificateを分ける | Unknown rare費用、全条件同時confidence、選択後保証が残る |

固定q点比較は「その設定の差」を問う。各方式が精度を満たすよう調整した比較は「指定候補域・予算内の資源差」を問う。両者を分け、最小qやGを今回探索しない。既存S0比較のB4<B3、D1<R2、D1/R4未確定という限定結果を、強い対照を含む方式一般の順位へ拡張しない。

## 12. 今後の科学的問い候補（表3）

以下の「追加」は現在のB評価に対する追加であり、先行研究に存在しない世界初の問いという意味ではない。機構差のない問いを独立算法候補へ無理に分類しない。

| 問い | 反証可能な仮説・位置付け | 強いbaseline | 最小限に欲しい証拠 | 研究判断が変わる結果 |
|---|---|---|---|---|
| Q1：単一因子RにRTEを使う理由が残るか | 既存のRTE列の短さが、Exact-Rのbasis往復／対角費と競合し得る。応用評価候補 | 同一outer/D列のExact-R、full DF-S2/Y4 | 独立correctness後の同T signal・full wrapper単価。Shotは別scope | Exact-Rが同精度で費用を下げれば、toyの資源主張を限定し直す。RTEが有利でも単一因子の限定結果 |
| Q2：A3の長い内部列は調整済みA2／compact対照に必要か | n_Dを調整した既知低次数法で必要精度を満たせる可能性。新算法ではない | A2 n_D1/2、固定m5、published A3/A4、full DF-PF | 事前限定した設定のbias・構造・native費 | A3の優位が消えれば内部高次化の必要性を縮小。残っても方式全体最適性とはしない |
| Q3：Stageごとのfinite-RTE精度を変える実益があるか | Uniform r/Kの無駄を既知配分で減らせる可能性。既存PR配分の応用候補 | PR絶対時間配分、uniform、Exact-R、full PF | 同じ総仕事量／誤差予算でのmean・N・費用 | 固定toyの改善だけでは独立算法差を補えない。Q1でRTE不要なら優先理由は弱まる |
| 独立算法の継続 | 具体的新機構の存在は現在NOT_SUPPORTED | 上記全対照と近接先行法 | 先行法と異なる命題・構成・access・適用条件の先行整理 | その差が特定された時だけ再判断。データ追加だけで新規性を成立させない |

Q1は現在の対照不足を直接問える点で技術的に明確だが、現toyのbaseline確認そのものが独立研究として十分かは別の科学判断である。Q2/Q3を網羅的探索・分子展開へ膨らませることは提案しない。

## 13. A／B／Cの判断材料

**A：独立算法研究**には、PFの既知係数変更・内部反復・配分・proxy選択を超える具体的な生成機構が必要。現実装の新規性分類にPOTENTIAL_METHOD_DELTAを付ける根拠は得られていない。現在の静的調査は不存在証明ではなく、Aの正式NO-GOでもない。

**B：応用・資源評価研究**には、強い対照と公平な調整の下で何を実用的に判断できるかを先に限定する。Q1/Q2は欠けた対照を補う候補だが、旧H246にも高次PRの費用評価があるため「精度により方式を選ぶ」だけでは追加価値が薄い。単一Rのアクセス構造、nested Dとouterの対照をどこまで必要とするかが論点になる。

**C：一区切り／一時保留**は、現synthetic成果を実装・機構診断・条件付き会計として保持し、追加実行をしない選択。Independent noveltyや一般資源優位を主張しない範囲で、既存成果は利用できる。研究目的を変える判断をCodexは代行しない。

## 14. 必要なら行う最小検証のscope候補

**以下は将来案であり、実行契約・採択・実行許可ではない。** 本作業で新SOURCE、science scope、runner、call登録、grantを作らない。

1. **技術準備を選ぶ場合**：同じ保存fixtureの因子2にExact-R descriptorを接続し、既存D/outer plan・prepared wrapper・compilerを再利用する案を具体化する。R/identity phase、signed順、control0/1、数学参照を人工検証できる形にする。Full PF側も同じblock描画へ接続する。実物のbasis／回路を生成する検証はここに含めない。
2. **最初の限定科学候補**：現B3（outer-Yoshida4/A2）とB4（SABA2/A2）のn_D1/q2、T±.18を動かさず、各同列のExact-Rを比較する。旧22行のideal-tailは参照情報として保持し、新回路の独立参照と混同しない。Full DF-S2/Yoshida4のq2を固定点の対照に置く案もあるが、必要性をユーザーが選ぶ。
3. **検査・費用の順序**：同じH/state/sector・scalar/identityを確認し、R単独／controlled／full wrapperの独立operator・X/Y correctnessが通過した範囲だけ、同じv5でRZ/CX/depth/sizeを測る。物理complex signalとoperator、raw/correctedを分離する。許容差・precisionは将来契約で固定し、FAIL時に緩和しない。
4. **調整比較は別単位**：固定点が判断材料になる場合だけ、A2 n_D2や必要q、既存m5の限定設定を先に選ぶ。同精度最小費用を主張するなら、対照にも同じ調整機会を与える。自動grid拡張・係数探索は行わない。
5. **資源結論の境界**：最初はsignal＋deterministic wrapper単価までに絞る。旧S0単価・巨大shot整数を新baselineへ機械的に掛けない。新shot/G、MC、rare、A1、γは必要性を別判断し、今回の追加案へ束ねない。

ユーザーが対象・問いを選んだ後、Codexが最小実装・人工確認・新科学SOURCEと正確なcall scopeを準備する。その後、**新operator/signal評価・実basis生成・build/compileの固定実行について別途チャットで許可**を得る。既存R2-CS P1を再開せず、現科学runもretry/resumeしない。各新固定検証は終了・FAIL・識別不能でmandatory STOPする。

## 15. 判断に残る問題・未取得資料（表4）

| 問題 | 技術確認だけで解決できる範囲 | 新科学計算が必要な範囲 | 独立科学判断／ユーザー判断 |
|---|---|---|---|
| Exact-Rの接続 | DF primitive・phase規約・adapter差分は本書で整理済み | 実回路correctness・signal・単価 | その対照取得が継続判断に必要か |
| Full PFとn_D／compact | 既存列・control経路・未対応箇所は整理可能 | 同条件の精度達成・費用 | 調整域と有用な比較taskを選ぶ |
| Method delta | 原論文・SOURCEの対応を特定 | 数値優位があっても構造的新規性の証明にはならない | 新規性・独立研究価値の正式判断は今回しない |
| R2-CS | 正式終了・P1未実行を確認済み | 再実行しない | 類似scopeを再開するなら明確な差と別判断が必要 |
| u_num／γ／rare | 既存null・経験値の限界を確認 | 取得は別科学scope | 今回先行させる提案にはしない |
| 旧providerの再利用 | 公開v5の実装は取得済み | 現taskへの適合検証が必要 | 旧H246 runtime全closure一致は今回未認定 |
| レビュー・先行性の全体評価 | 指定5原論文＋直接関係1本の限定箇所 | 数値を追加しても網羅性は得られない | 2026-10-25レビュー本文未取得。詳細レビューの開始は別許可 |

取得制約として、PR HTMLは404でPDFを使用、CPF PDFの紙面画像はcache missだった。CPF v4 Eq.(9)の中央B/2という表示はHTML／PDF抽出に残る既存保留で、推測修正して実装仕様にしない。Moralesのabs改訂日とHTML本文日付にも差があるため、本書はarXiv v3という版を固定する。これらから論文全体の誤りや無効を新たに認定していない。

指定論文自体は取得できたが、全定理の再証明、SPRINT full pipelineの現B等価性、stage別finite-Kの全先行研究、m5の一般最適性は未確認。現在BとExact-R/full PFの実費用・精度は未測定であり、推測で結論を埋めない。

## 16. SOURCE・参考資料・作業保全

主要根拠と固定版を以下にまとめる。結果公開commitと実際の科学SOURCEは[統合報告 §16][INTEGRATED]を正とし、新しい実行SOURCE候補は固定しない。

| 根拠 | 固定commit／版 | 本調査で読んだ対象 |
|---|---|---|
| 現B | `3cfbac1d4f7df751e972170e54ad21aa947ceff1` | 統合・初期調査、関連input/outer/native/RTE/DF primitive/test本文 |
| 現Bの最終科学SOURCE | `9bebdc46ebd60039f315da9b6c6e57bd9c004b85` | 統合に記載のshot scope・未取得資格を維持 |
| 旧H246結果 | `8613b3ecda2030a677b1690fe7aa3da23dec4873` | 結果、input provenance metadata、m5 plan/receipt/native/new2 core |
| 旧H246取得SOURCE | H2 `4ec26921c6872ed6e8667f167074e1e405cf1a3f`、H4/H6 `5a448d94ea389ada6e2b2755de6a215ed2034679` | 旧報告・metadataの固定来歴。今回再実行なし |
| PF repository | `20d64c2fea1ed2d2188777df4f3d962ee94df486` | 実product_formula関数、現Bの固定抜粋・inverse規約 |
| AX-2B provider | `31cdff47f3282d2898c153a1af22f90e500be4c6` | 公開native_df_v5のcontrolled block／full PF接続 |
| R2-CS | C0 `8cb41c40…`、C0.6 `dee25ba1…`、closure `270078a8…` | 目的、prior-art対応、最終終了理由、P1取扱い |
| PR | `2503.05647v2` | A.2 A23–A31、A.3 A32–A41、Dの高次対照、Eのtask/費用 |
| SPRINT | `2606.30741v1` | Fig.1、III.2 (25)–(31)、III.3 (32)–(38)、III.5、IV、E.3 |
| Morales | `2210.15817v3`、QIC2025 | II.3 (27)、Tables1/2、IVの比較・eigenvalue/operator区別 |
| CPF | `2409.08265v4` | §2・Prop.1 (9)–(10)、§5合成、§8の近似partition議論 |
| SABA | `astro-ph/0005074v1` | §4 (14)–(19)、Table I、§7のMcLachlan先行対応 |
| Composite | `2206.06409v3` | §5とchannel/query・partitionの既知範囲 |

独立worktree `/tmp/prt-research-b-bnext1-static-baselines-20261026`、branch `codex/research-b-bnext1-static-baselines-20261026`で本書と最小索引だけを変更する。科学module import・runner起動・数値テスト・Hamiltonian/operator/signal生成・sampling・compile・保存値の新差分／比／順位／統計／shot/G評価は行わない。既存報告に掲載された数値だけを引用した。

文書検査は引用条件・固定path・参照link・表・差分を対象にし、科学的再検証や網羅的hash監査を行わない。原本・SOURCE・旧契約・grant/ledger/marker/STOP・R2-CS branch・共通指示・元dirty workspace・共有環境は変更しない。承認証明や新管理frameworkは作成しない。

本書と索引をcommit/pushし、remote SHAと公開内容を確認した後、**mandatory STOP**。次担当は**ユーザー**でA/B/Cと必要ならQ1等の限定技術準備を判断する。**Codex**はその指示と必要な科学実行許可の後にだけ進める。研究統括への追加送信・詳細GPT/Claudeレビュー・終了済み研究の再開を行わない。

[INTEGRATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/docs/research/research_b_integrated_outcomes_and_future_directions_20261025.md
[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/docs/research/research_b_time_evolution_initial_survey_20261024.md
[H246]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/docs/research/track_a_h246_saved_cost_results_20261021.md
[OLDPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/src/trotterlib/h46_m5_plan.py
[OLDSIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a448d94ea389ada6e2b2755de6a215ed2034679/scripts/h46_m5_signal_adapter.py
[OLDCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a448d94ea389ada6e2b2755de6a215ed2034679/scripts/h46_m5_cost_native.py
[OLDRUNTIME]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a448d94ea389ada6e2b2755de6a215ed2034679/scripts/h46_pr_exploration_runtime.py
[M5RECEIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/references/h46_m5_best_formula_v1.json
[CS0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88/docs/research/r2_configuration_selection_c0_20261024/REPORT.md
[CS06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dee25ba13ee8ebcb23c7d294c659ea678d818277/docs/research/r2_one_sided_c06_20261024/REPORT.md
[CSCLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/270078a87c1febe8944f829270e8c2a00dbed7de/docs/research/r2_cs_closure_20261024/REPORT.md
[FIXTURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/tests/test_df_partial_s2_repeated.py
[B0SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trottertracks/b2d_evaluator/b0_native.py
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trotterlib/df_trotter/ops.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trotterlib/df_partial_s2.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trotterlib/df_rte_tail.py
[DFTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/tests/test_df_partial_s2.py
[DFCIRCUIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trotterlib/df_trotter/circuit.py
[NATIVEOLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[OUTERSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trottertracks/b2d_evaluator/outer.py
[PFSOURCE]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[STRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/src/trotterlib/rte.py
[PR]: https://arxiv.org/pdf/2503.05647v2
[SPRINT]: https://arxiv.org/html/2606.30741v1
[MORALES]: https://arxiv.org/html/2210.15817v3
[CPF]: https://arxiv.org/html/2409.08265v4
[SABA]: https://arxiv.org/pdf/astro-ph/0005074v1
[COMPOSITE]: https://arxiv.org/html/2206.06409v3
