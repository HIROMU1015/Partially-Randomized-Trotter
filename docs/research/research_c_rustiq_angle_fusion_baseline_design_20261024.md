# 研究C・C4：Rustiq改善機構の切り分けと角度結合baseline設計

2026-10-24 JST。状態：**DESIGN_ONLY_IMPLEMENTATION_AND_SCIENCE_NOT_AUTHORIZED**。

推奨案は、既存のtyped rotation領域で**連続する同一Pauliのcontrolled rotationだけを角度結合するB2**を、旧B1と同じ中央emitterへ接続すること。最初の科学比較はB2だけを新規実行し、B0/B1/R1の保存証拠と独立referenceを再利用する。R2は初期scopeへ入れない。

今回はSOURCE・テスト・保存報告・入力metadata・固定Qiskit SOURCEの静的読取りと本設計だけを行った。B2/R2実装、テスト実行、QuantumCircuit/QPYの動的復元、build、Rustiq、transpile、Operator/Statevector評価、角度の数値結合、保存費用の再集計・再ランキングは**いずれも行っていない**。以下のcall数と分類は未承認の将来契約案であり、grantではない。

## 1. 固定した出発点と既存結果の引用

Repositoryは`HIROMU1015/Partially-Randomized-Trotter`、起点commitは`f95148e3f176efc6b3fd81a93d043ccd20255343`。本調査の独立branchは`codex/research-c-rustiq-angle-fusion-design-20261024`。既存SOURCE、raw、結果、承認・ledger・marker・STOP、研究概要・validation manifest等の共通文書は編集しない。

参照した固定資料：

- [直前の限定科学実行・適用範囲・資源値][RUN]（実際のscience SOURCEは`b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9`）。
- [adapter準備資料][PRE]、[旧限定契約][CONTRACT]（準備commit`5038427e034c5e2cca6674998d28715cd93ec551`）。
- [接口監査][AUDIT]（`1dd39865e6b5598d3dfac5f65e434d7db6777788`）。
- 正式共通指示`research_project_instructions_unified_2026-10-23.md`の§8.4／§10。SHA256は`7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。既存コード再利用、共有環境保護、将来の計算資源に人工的な低い上限を置かない方針を適用する。

保存F1は4 mode・2粒子・occupied `(0,1)` のdensity-square合成Hamiltonian、split `L_D=1`、残差2fragment。分子geometry／AO basis／分子DF rankは非該当。`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`の一点、Qiskit1.3.0・opt1・seed17・native `rz,sx,x,cx`。以下はこの条件の既存報告を引用しただけである。

| case（各axisでX/Y共通の保存値） | RZ：B0/B1→R1 | CX：B0/B1→R1 | total depth：B0/B1→R1 |
|---|---:|---:|---:|
| `SS_motif_2` | 139→135 | 66→62 | 172→164 |
| `SS_motif_3` | 115→113 | 51→49 | 136→132 |
| `SS_motif_4` | 87→83 | 48→44 | 108→100 |
| `PP_motif_0` | 59→59 | 36→36 | 78→78 |

[旧報告][RUN]では全登録gateがPASSし、設計した8 wrapperの無重み合計はRZ `800→780`、CX `402→382`、total depth `988→948`。これはlaw期待値、認証shot、全round資源、実分子への移送結果ではない。新しい合計・割合・順位は今回計算していない。B2の費用はすべて**未測定**。

## 2. B2が保持するoperatorと結合の定義

Hermitian Pauli `P²=I`と同じcontrol/layoutについて

\[
C_P(a)=|0\rangle\langle0|_a\otimes I+
       |1\rangle\langle1|_a\otimes e^{-iaP},\qquad
C_P(b)C_P(a)=C_P(a+b).
\]

これは実数演算での**完全なcontrolled operator**の等式であり、global phase同値ではない。既存little-endian／ancilla-lastでは`diag(I,e^{-iaP})`を保つ。固定basis Bの中央Z列なら、同じBを残したまま中央角を結合する。Gaussian basisの選び直しはしない。

### 2.1 結合対象と境界

1. `plan_sequence`が元のordered eventsから確定したrunとRegionを先に固定する。元のproductを含むrun長で決めた`singleton`、support/full、basis recipe/cacheはそのままにする。
2. `eligible=True`のrotation region内部を元順序で走査する。結合keyはcase/macro/run/region、kind、basis ID/hash/frame、元Pauli word、中央support、system/ancilla mapping。keyが完全一致する**最大連続区間**だけを対象とする。
3. atom ID・occurrence ID・係数signは全memberの来歴として残す。符号は既にsigned angleへ入っているので再度掛けない。atom IDが同じだけでは判定せず、逆に同一generatorでもsignの異なる角は結合できる。元word/supportが異なるものは、同supportに見えても結合しない。
4. `P,Q,P`は、PとQが可換でも3区間のまま。非可換項の移動、sort、set化、異basis/run/frame/macro、D境界、product、identityを越えた結合をしない。
5. 各区間を1回の既存`append_central(qc,support,ancilla,A)`へ渡す。productは旧literal emitter、identity/scalar/Taylor phase/product signは旧phase処理。eventとsampling record自体を削除・再生成しない。

singleton groupは元のangleをbit表現ごと通過させる。複数memberの`A`は元signed binary64角を順序付きで`math.fsum`へ渡す設計とする。倍率×代表角には置き換えない。特にSS M2の最初のregionにはorder2のchildとorder0のchildがあり、同一atomでも角が同じとは限らない。実装のfloat丸めを誤差0やbitwise operator一致と宣言せず、既存の絶対gateで確認する。

角の剰余化、Clifford近傍へのsnap、`isclose`による削除はしない。合計0でも最初の案では0角の既存emitterを残し、除去は共通compilerに任せる。非有限入力・overflowはFAILとする。`a+π`はsystemの符号を変え、control後はancillaの相対位相になるため、位相を捨てた簡約を許さない。

### 2.2 controlled展開の前に結合する理由

Q13へ渡す旧展開はPRの`a`ごとに`(P,a),(Z_aP,-a)`。同じPの次occurrenceがあっても、この展開列ではPとZₐPが交互になる。展開後のtripleへ「隣接する同一wordだけ」を適用すると、この対照の目的を外す。

B2は展開前のtyped **controlled occurrence**を結合し、旧literal中央回路へ戻す。既存emitterが既にcontrolを実装するので、追加の`.control()`は掛けない。全Pauliを集めて可換項を並べ替える方法でも、native gate列からGaussian由来のRZを推測する方法でもない。R1は旧Q13経路のまま保存比較し、B2へRustiqを混ぜない。

### 2.3 provenanceとbasis計画

元Occurrence/SequencePlan/ordered_events/phase記録は保持し、別のfusion ledgerへmember ordinal、atom ID、word/support、basis/run ID、元angleのhex、合計angleのhex、和の方針・版、元plan fingerprintを記録する。B2のroute/fusion policyとledgerを新fingerprint・compile contextへ結び付ける。これは来歴の圧縮ではなく、emit処理の局所置換である。

**圧縮後の1 rotationからrunを再構成しない。** 元が2 rotationのfull-basis runなら、結合後も元の`singleton=False`を保持する。1回に縮んだからsupport補完へ変更すると、Gaussian改善までB2へ混入する。

## 3. 既存簡約の静的確認と公平な比較

### 3.1 Qiskitに既にあるもの

固定Qiskit SOURCEは`896d1c8b72888a52a46081a6d1f4de8c56db8aea`。以下の公開SOURCEと、既存専用venv内の同じ4ファイルはGit blob hashが一致した。今回はimport・PassManager実行を行っていない。

| 静的に確認した既存機能 | B2設計との関係 |
|---|---|
| [opt1の既定optimization stage][QOPT]は`Optimize1qGatesDecomposition`と`InverseCancellation`を使用 | 既存profileから外さず全routeに共通で与える |
| [Optimize1qGatesDecomposition][Q1Q]は1量子bit gateの連続列を既存decomposerで簡約 | 通常のRZ列を独自compilerで再実装する必要はない。一方、PR controlled primitiveのRZ間にはCXがあり、typedな複数controlled rotationを直接受け取る接口ではない |
| [InverseCancellation][QINV]は指定された隣接inverseを除去 | 既存CX対の取消しを妨げない。任意のPauli回転のtyped angle foldそのものではない |
| [CommutativeCancellation][QCOMM]は可換性を使う別の既存pass | opt2等のSOURCEに存在するが、今回これを追加するとprofile・並替え能力まで変える。B2の代わりとして新たに挿入しない |

確認したblobは順に`85b0c05d5cae2c41fa1666e41f9aa89749a101cc`、`55232ac1be20955df6a99cabaa9b9b151227c797`、`508cfc66946a58361368953805df245b7842a69f`、`130ff0609354b59da55e41c8cf3c701b840f26b4`。

PRの調査対象SOURCEには、中央typed Pauli occurrenceに対する同種の隣接角結合helperは見付からなかった。既存のD境界時間融合とbasis/run共有は既に共通baselineにあり、残す。したがって必要な追加は小さいtyped foldであり、Pauli行列・可換判定・合成器を作る必要はない。

これは「Qiskitが本caseで結合できない」との実測結論ではない。pipelineには展開・繰返し簡約もあり、B2の前処理が最終compiled費用へ反映されない可能性がある。pass別traceは旧packetに保存されていないので、どのpassがどの結合を実行したかは未取得である。

### 3.2 比較route

| route | 本比較での扱い |
|---|---|
| B0 | 強いcommon-run baselineの保存値・既存PASSを引用。再build/compileしない |
| B1 | 同じtyped partition・旧中央emitterの保存値。B2の直接対照 |
| B2 | 元partition/basis planを固定し、隣接同一Pauli角結合だけを加えて新規実行する将来候補 |
| R1 | 前段のQ13保存値。B2と同じcase/axisのfull-wrapper値で比較 |
| R2 | 初期scopeから除外。後述の識別不能が残ったときだけ別採択・別承認 |

全routeに旧basis共有、元singleton support/full、D境界融合、scalar/identity/product/Taylor phaseを同条件で与える。新B2でもfull Hadamard X/Y・occupied(0,1)・ancilla4・terminal measureを含む。同一compilerの公式definition展開、native basis、opt1、seed17、approximation_degree1、qubits_initially_zero=false、backend/target/coupling/layout/routing/scheduling指定なしを維持し、非identity layoutは拒否する。

旧SOURCEとB2追加後のSOURCEのcommit SHAは当然異なる。再利用可否はSHAの同一性を装うのでなく、shared builder/compiler/reference/wrapperのbyte hash、物理spec/input/law、元plan、register、profileが同じで、変更がB2 opt-inと保存target接続に限られることをdiffで確認する。旧B0/B1/R1 branchの動作と既定routeを変えない。

## 4. SOURCE・test・保存証拠の再利用地図

パスはすべて[f95148e3固定tree][TREE]のもの。以下は将来の差分案で、今回は実装していない。

| 資産 | そのまま使う部分 | 最小差分／既存簡約 | 再利用test・新検証 |
|---|---|---|---|
| `rustiq_adapter.py` | Occurrence/Run/Region、`plan_sequence`、basis選択、B0委譲、D/scalar組立 | 新版だけにB2 routeとrotation branchのfoldを追加。元planは変更しない。既存R1は既にQ13を呼ぶ | order/sign/fence/singleton/cache/default B0の既存mockを継続。新たにgroup membership、signed sum、singleton passthrough、B2 cache分離を検証 |
| `circuits.py` | `BasisProvider`、`append_central`、`append_deterministic`、result、actual inverse、full wrapper | 編集不要。basis/run/Dの合法最適化は共通 | 元run長とbasis呼出しがB1/B2で同じこと、phaseとwrapper scopeを新B2で検査 |
| `rustiq_execution.py` | Journal、`_matrix`、`_full`、`_adjoint`、`_wrapper`、`_compile`、stage barrier、保存 | `_local`のB2 branch、saved-target preparation、route/request集合と集計を固定B2 scopeへ切替。旧`_prepare`は必ずreferenceを再生成するのでそのまま呼ばない | 既存のforeign-approval拒否・fresh output・failure保全mock。新B2 scopeが旧承認を拒否することを追加 |
| `gates.py` | 絶対max-entry/Frobenius/spectral、unitarity、X/Y、official-definition lowering、full-QPY cache | 編集不要。既存Qiskit passを共通profileで再利用 | global-phase無視やphase fitに置換しない。実B2のraw/compiled gateは新たに必要 |
| `reference.py`／保存reference | CAR/exterior由来の旧90/128結果、`precision_context`、`controlled_operator`、`to_numpy` | 新しいexpm/law/referenceは作らず、保存128桁targetをdecodeする小さいreaderが必要 | 旧paired precision PASSをhash付きで再利用。文字列decodeの技術確認は未実施 |
| `f1_saved_execution.py` | 保存binary64 lawのrestore、`SavedF1.resolve_case/macros` | law/イベントのsampling・再設計をしない。元4case recordと一致を確認 | 同一wordでもTaylor orderで角が異なるcaseを保持 |
| `scripts/run_research_c_rustiq_limited.py` | 引数処理、verify→executeの順序 | 新版に明示B2 scope選択を追加する案。旧入口をそのまま起動すると旧3route全件を実行するため不可 | 既定の旧scope維持と新B2 opt-inをmockで検査 |
| `test_research_c_rustiq_adapter.py`／`test_research_c_rustiq_execution.py` | 前段24件のunit/mockとreference/hash管理の設計 | 新しい科学testを無条件に通常suiteへ追加しない | product/identity、±角、同support異word、`P,Q,P`、異frame/macro、0角、phase wrap禁止を人工recordで追加する将来案。今回は旧testも実行していない |
| `f2_execution.circuit_metrics` | 前段と同じ8指標 | 再実装しない。F2 law/mean/統計/grantは使わない | 新B2 compiled8件とraw診断のみを登録する |

新しいbuilder/compiler/Pauli algebraや管理基盤は不要。想定差分は既存adapter、既存実行module、既存CLIのscope、既存mockへの局所追加で足りる。関数名や行数は将来の実装判断に委ねるが、sampling/planを圧縮する方式へ変更しない。

### 4.1 保存証拠の固定identityと不足

旧packetを`P=artifacts/research_c_rustiq_limited/2026-10-24/attempt01/`と略す。

| 証拠 | SHA256／再利用 |
|---|---|
| `P/OUTPUT_HASH_MANIFEST.json` | `5d563764454bef76e97f653d71fe8a3eeb7b9f479e59bef44c39d61c30a42069`。個々の保存target/QPY/NPY/inputのbyte identityを固定 |
| `P/provenance.json` | `8ceec3a459774155732f19d358a32610e56a4f81c92b31ac6b88b3fdfb60e0ea`。science SOURCE、40 source hashの由来、50package lock、API/native binary hashを確認 |
| `P/preparation_contract.json` | `e948a2e7fdf47e20b8af8cd7debae0631dabbd5e6ad77f419a9a9e2c668319a1`。spec/compilerと7 input hashをそのまま参照 |
| `P/saved_inputs/handoff/f1_bundle.json` | `9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0` |
| `P/metrics.csv` | `fcd48e821b425514926adb200698d36c38e518b4bd80896b3a463196acb83f9a`。case/route/axisの既存8指標を直接対応付ける |
| `P/reports/<case>_input.json` | 元events/atoms/plans、angle hex、singleton、90/128 agreementを保持 |
| `P/references/` | full4＋local10の128桁bundleと旧90桁対が存在。新reference生成は不要な設計 |

再利用するファイル名はfullが`P/references/<case>_128.json.gz`、localが`P/references/<case>_macro<m>_run<r>_region<k>_128.json.gz`。結合前literalは`P/operators/<case>_macro<m>_run<r>_region<k>_B0_B1_literal.npy`であり、将来の承認後にだけ数値loadする。

保存referenceの値は`0.0`や`(real - imagj)`等の`ctx.nstr`文字列。現行runnerにその逆readerはない。将来は制限した十進／指数表記grammarを解析し、real/imagを別々に128桁contextへ渡す。高精度復元前にPython float/complexを経由して桁を落とさない。復元後の最終binary64変換には旧`to_numpy`／gateと同じ処理を使う。`eval`と候補行列をreferenceへ代入するfallbackは禁止。schema、digits、shape16/32、有限性と全hashを確認し、不明形式はSTOP。今回は文字列表現を読んだだけで、数値decode・round-tripは未検証。

不足しているものは、B2 SOURCE/実測値、readerの動作確認、旧compilerのpass別変換trace、同時期・同一worker/load条件での反復古典timingである。旧raw wrapper QPYは保存されているが、raw費用の新測定は今回しない。未取得の値を予測で埋めない。

## 5. 元4case・同じ10領域を使う将来契約案

仮称`C4_B2_SAVED_COMPARISON_V1`、**未承認・未実装・未実行**。SOURCE baseはf95148e3。実装commit・新SOURCE hash・新grant/run IDはまだ存在せず、実装後かつ科学開始前に別途固定する必要がある。

### 5.1 対象領域とB2の構造

元contractと保存planを引用した次の領域だけを許す。`m0/m1`はmacro番号、`o`はそのmacro内の元occurrence ordinal。下表の結合は設計上のmember対応であって、角の和やcompiled費用の計算結果ではない。

| case | 対象region／元member | 固定するbasis条件 |
|---|---|---|
| `SS_motif_2` | m0 `run2:region0` o2,o3、m1 `run0:region0` o0,o1。各pairは同じ`S:F0:IIZI` | 両runとも元のsingleton=False。m0先行productと異frame fenceを残す |
| `SS_motif_3` | m0 `run1:region0` o3はpass-through、m1 `run0:region0` o0,o1は同じ`S:F1:IIIZ` | m0はsingleton=True、m1はFalse。identity/product/phaseを残す |
| `SS_motif_4` | 各macroの`run0:region0` o0,o1は同じ`S:F0:IIZI` | 各runの元singleton=Falseを保つ |
| `PP_motif_0` | 各macroの`run0:region0` o0と`run1:region0` o1はそれぞれsingletonでpass-through | Z3とX2X3を別basisのまま保持。結合なしの負対照 |

正確なcase record/product→child列は旧`preparation_contract.cases`を変更せず使う。SSのframe間commutatorを新たに評価しない。PPの非可換順序はfenceで保持し、非可換network合成の検証へ拡張しない。

### 5.2 保存値との比較可能性preflight

将来の新しい承認はB2実装と科学実行のどちらを許すか明示し、固定SOURCE/本案の採択版/入力/旧packet hash/fresh outputへbindする。旧grantと技術的retry許可を流用しない。新SOURCEはB2 branchとsaved-target接続以外の科学的差分がないことを確認する。

7 input bytes、元ordered events/angle hex、元plan/basis/phase、same register、同じprofile/official lowering、50package lockおよびQiskit API/native binary identityを照合する。変更済み環境や欠測があれば旧値との直接比較を保留し、値の補完や未登録B0/B1/R1再実行へ進まない。共有環境を変更して一致させない。

science sourceが異なるためnew cache contextへB2/fusion policy/member ledger/input/親証拠identityを含める。旧R1/B1のcompile cacheは使わない。cold8 miss、同一IR warm8 hitを別に記録し、warm追加transpileは0とする。B0/B1/R1の保存費用を新cache価格として登録しない。

### 5.3 gate順序・許容差・reference

1. 保存128桁targetを14 bundle（local10＋full4）だけdecode。fullではU、signal、wrapper X/Yを使う。旧90/128 agreementのPASSとhashを再利用し、新しいCAR/exterior/law/expm/reference評価やprecision再比較はしない。
2. B2局所10回をbuildし、各actual matrixを一度取得。同じmatrixを、保存独立controlled referenceと保存結合前literal NPYの**両方**へ比較する。32次元全体、control0=I、unitarityを確認する。保存literalを再buildしない。
3. B2 full evolution4回をbuildし、保存Uから既存helperで得る`diag(I,U)`と比較する。scalar/Taylor/product/identity位相を含み、sectorだけの比較ではない。
4. 完成B2 circuitのactual inverse4回を作り、独立targetのadjointへ比較する。
5. full measured wrapper8回をbuildしraw X/Y8評価・4 complex pair gate。XまたはY一方で代用しない。
6. 同じ8 wrapperをcold8回compileし、warm8 requestのcache hitを確認。compiled full-wrapper Operator8評価とX/Y8評価・4 pair gateを行う。Operator比較で外すのはterminal measureだけで、prep/readoutとglobal_phaseは残す。
7. 全正しさgateを通ったcaseだけcompiled資源へ進む。B2対旧B1/R1を同じcase/axisで並べ、旧routeの新規合計・再ランキングを目的にしない。

max-entryの絶対tolはlocal/full/adjoint/compiled/unitarityすべて**1e-10**、complex X/Yは**2e-10**、rtol0。Frobenius/spectralも保存する。global-phase同値だけの`equiv`、phase fitting、事後緩和でPASSにしない。旧独立reference資格の1e-60を新たなrigorous boundへ格上げしない。

### 5.4 予定call数：新規はB2のみ

fully passing runの論理request数。内部のgate append/QC copy/QPY serializationを独立caseとして増やさない一方、新しい数値評価をそこへ隠さない。

| 内容 | B2予定数 |
|---|---:|
| 元motifのrestore（保存law、samplingなし） | 4 request／対象2arm |
| 保存128桁reference bundle decode | 14（local10＋full4） |
| decodeで読むtarget matrix／signal | 22 matrix（local10＋U4＋wrapper8）／4 signal |
| 結合前literal NPY load | 10 |
| B2局所build／full evolution build | 10／4 |
| actual circuit adjoint／full wrapper build | 4／8 |
| Operator materialization | 26＝local10＋full4＋adjoint4＋compiled8 |
| 絶対operator比較／unitarity検査 | 36（local20＋full4＋adjoint4＋compiled8）／14 |
| raw／compiled Statevector signal評価 | 8／8 |
| X/Y pair gate | 8＝raw4＋compiled4 |
| compiler cold／warm request | 8／8 |
| 実transpile／warm追加transpile | 8／0 |
| correctness後のcompiled resource metric | 8 |
| correctness後のraw IR診断（下記） | saved B1 QPY decode8、raw metric16＝B1旧8＋B2新8 |
| 新しい独立reference生成／precision再比較 | 0／0 |
| B0/B1/R1再build・再compile・Operator/Statevector再評価 | 各0 |
| Rustiq合成／R2実行／sampling・law生成／mean・認証shot・law加重総資源 | 各0 |

保存targetのread/数値変換やUからcontrol/adjoint targetを作る既存helperは、新規reference生成0の中身として省略せず上記に区別した。実装によって追加referenceや旧回路評価が必要と判明した場合、この上限を勝手に拡張しない。

### 5.5 compilerの既存最適化と古典費用の診断

旧B1の`P/circuits/<case>_B1_<X|Y>_raw.qpy`を8件読み、B2の既にbuildしたraw wrapper8件と同じ既存metric関数で比較する将来案。旧回路のbuild/compile/simulationはしない。これは新しいraw IR測定であり、**今回のread-only作業では行っていない**。Gaussian compositeを含むraw表現の診断値を、native compiled資源値と混ぜない。

B2 ledgerでemit結合が行われ、raw表現にも差があるのにcompiled B2=B1なら、このprofileがB2前処理の差を費用上吸収したことを支持する。compiled値の一致だけでは「Qiskitが同じ内部角度結合を実行した」とは断定しない。どのpassで起きたかは旧traceがなく、必要なら別の観測設計へ戻す。

adapter計画、角度fold、literal emitter、whole builder、cold lowering/transpile、warm cache lookupのwall/CPUとRSSを分ける。foldはbuilderの部分時間であり再加算しない。B2のRustiq合成時間は「呼出しなし」とし、R1の未測定同時条件の時間差を0や推定で埋めない。

旧R1は単発・24 worker capacityの測定で、新B2と同じload/worker/起動状態の反復対照ではない。歴史的timerは区別して引用するだけにし、今回案の一回測定だけで古典費用の速度優位を認定しない。matched timingが採否に必要なら別scope/別承認。最初から旧routeを再実行しない。

### 5.6 資源・停止・出力契約

RZ/CX count・RZ/CX depth・total depth・quantum-only depth・full-wrapper size・measurement countを旧定義で保存する。measurementは1を保ち、省略して改善扱いしない。T countへ換算しない。

実行時にCPU affinity/load、空き物理RAM/pressure、利用可能backendを確認し、独立local/full/wrapperをstage内で並列化する。gate間はbarrierとし、workerとBLAS/OpenMP/Numba/Qiskit/Rayonを調整する。旧24-worker値を流用せず、論理task数とその時の資源から決める。CPU/RAM/GPU/worker/wall-timeの人工上限なし。共有venv/設定/cache/他jobは変更しない。

hash/profile/phase/plan drift、decoder不明・非有限、gate FAIL、unexpected layout/control、cache/call不一致、実際のresource pressureでSTOP。新grantが定めるfresh attemptだけを許し、自動retryやcase置換・reference再生成を既定にしない。科学的FAILは隠さない。前回の技術的retry承認は本案への実行承認ではない。

予定出力は`artifacts/research_c_rustiq_angle_fusion/<execution-date>/<fresh-run-id>/`。元証拠を上書きせず、承認/SOURCE/input/旧packet/profileのhash、元plan＋fusion ledger、old/new source差分、QPY/NPY/targets、gate report、case/axis別8 compiled metric、raw診断、cold/warm/古典timer、attempt ledger、summary/STOPを保存する。旧と同じ4096行journal上限を計画し、行列出力はsystem16／controlled32に限定する。これはログ・科学call scopeの制約で、計算資源の上限ではない。

## 6. 実行前に定めるGO／STOP判断

比較は同じcase/axisの整数vectorで行う。主vectorはRZ/CX count、RZ/CX depth、total/quantum-only depth、sizeの7指標、measurementは同一性条件。1 gate/depth差を許容差で0扱いせず、都合のよい新しい重みや跨armランキングを作らない。下表の資源に基づく判断は全4case・両axisの正しさとprovenance確認が揃うことを前提とし、PASSした一部だけを選んでGOにしない。

| 将来のB2結果 | このscopeでの解釈と提案 |
|---|---|
| 全8 wrapperでB2=R1、全指標一致 | 既知の隣接角結合で観測資源差を再現できたという十分性の証拠。Rustiq内部の一意な機構同定ではない。実装改善として保持し、C4の独立研究継続を原則いったん区切る案をユーザー／統括へ返す |
| R1がB2をPareto支配 | 全対応record/7指標でR1が悪化せず、少なくとも1 recordのRZまたはCXが整数で小さい場合を明確なcount追加差とする。角結合だけでは説明し切れない差があるので、構造診断の設計へ進む価値を審議。一般的優位や自動追加実行は認めない |
| B2がR1をPareto支配 | 同じ条件を逆に満たすなら、単純結合を実装候補として保持しRustiqを必須にしない。C4の独立研究継続を再判定 |
| B2=B1でR1とも同値 | compiled資源の改善はなく、raw診断からcompiler吸収の有無を区別。追加最適化としての実益を主張せず区切る案 |
| B2=B1だがR1が小さい | 本B2は旧改善を再現できなかった。単に「Qiskitが全部実施済み」とせず、raw診断と構造の違いを確認して次の設計価値を判断 |
| trade-off、depthのみ差、axis/case依存、わずかな差 | 一意な勝者を作らず混在結果として報告。整数差の有無と、研究継続に十分な実益かを分ける。古典費用の不確かさと追加情報の費用を踏まえ、C4をいったん区切る選択肢を残す |
| 正しさgate FAIL／provenance不一致 | 資源順位を認定しない。技術原因か科学条件の問題かを記録し、無断で救済条件やcaseを追加しない |

いずれのGOも次の**設計を検討する提案**であり、独立研究の新規性認定・科学実行許可ではない。少数fixtureでの同点を一般的no-goへ、差を一般的最適性へ広げない。

## 7. R2を必要とする条件と追加scope

まずB2と保存R1で費用再現の可否を判定する。再現しただけならR2は不要。R1/B2間の差・trade-offが、network合成の残差なのか、角の事前結合による表現/float/同じcompilerの相互作用なのかを判別できず、その区別が継続判断を変える場合だけR2案を審議する。

R2は**B2と同じ結合済みledger/角**へQ13を適用するcounterfactualで、B2と同じ入力からnetworkの追加効果を比較する。Q13 optionsとactual末尾inverseは旧R1と同じ。新しいPauli探索器は不要。R2 route/cache識別とB2 fold→既存binding→Q13の接続だけを将来追加する案である。

別承認する場合の候補上限は同じ4case・10 local、full4、adjoint4、wrapper8、Operator26、Statevector16、cold/warm各8・transpile8、Q13合成20（local10＋full内10）、独立reference新規0。旧B2/B1/R1の証拠を再利用し、新caseは加えない。これは**初期B2契約に含まれない未承認の追加call**であり、必要性が実際に残った後にSOURCE/input/対象を再固定する。

## 8. 現時点で不足する情報と統括への判断事項

現状では、同一Pauliの重複が存在するというmetadata上の候補説明があるだけで、B2の角を実際に足したり費用を求めたりしていない。Rustiqの追加価値がある／ないの採択はまだできない。通常Qiskitのどのpassが旧変換を行ったかも不明であり、単発古典timerから経済性を確定できない。

ユーザー・研究C・統括へ渡す判断は次の三点。

1. 本B2の限定定義（元run/basis固定、隣接同一controlled Pauliのみ）を採用し、最小差分実装と非科学的mockをCodexへ別途依頼するか。
2. 保存独立reference/旧routeの再利用、B2のみの上記call、raw診断8 QPY loadを含む科学契約を実装後に承認するか。今回の文書はその承認ではない。
3. 再現なら既知角度結合として保持しC4を区切る提案、非再現/混在なら情報価値に応じて構造設計または終了を選ぶ基準を採用するか。R2は必要時の別承認に留めるか。

Codexはこのread-only設計をGitHubへcommit/pushし、remote SHAと公開本文を照合して**mandatory STOP**する。共通研究文書・旧SOURCE/結果/承認/STOPの採択状態を変更しない。B2/R2実装、build/compile、Operator/signal、保存数値の新規再集計、C1/終了済みGaussian singleton等へ自動進行しない。

[RUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f95148e3f176efc6b3fd81a93d043ccd20255343/docs/research/research_c_rustiq_execution_20261024.md
[PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5038427e034c5e2cca6674998d28715cd93ec551/docs/research/research_c_rustiq_adapter_preparation_20261024.md
[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5038427e034c5e2cca6674998d28715cd93ec551/docs/research/research_c_rustiq_limited_contract_20261024.json
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1dd39865e6b5598d3dfac5f65e434d7db6777788/docs/research/research_c_rustiq_interface_audit_20261024.md
[TREE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f95148e3f176efc6b3fd81a93d043ccd20255343
[QOPT]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L565
[Q1Q]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py
[QINV]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/optimization/inverse_cancellation.py
[QCOMM]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/optimization/commutative_cancellation.py
