# 研究C・C4：B2角度結合baselineの実装・技術検証・科学実行準備

2026-10-24 JST。状態：**B2_IMPLEMENTATION_PREPARED_SCIENCE_NOT_RUN**。

[比較設計正本][DESIGN]に沿って、B2のopt-in実装、非科学的unit/mock、保存証拠を使う実行接続、[未承認契約案](research_c_angle_fusion_b2_contract_20261024.json)を準備した。科学的同値性、回路資源改善、Rustiqの追加価値は未検証である。新しい実行grantは発行していない。

## 1. 版と承認の境界

- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- Branch：`codex/research-c-angle-fusion-b2-preparation-20261024`
- 起点：`b7b70f6a5e3ee4ae4a55b215b49de0390e5732e5`
- 実装SOURCE commit：`c7b6bc17f73d7b1dd2c9221272b4a5d858b5c472`
- SOURCE集合identity（41ファイルのhash辞書の既存digest）：`8db2c2b844b324507a1efd7a9fcd8420ef6454d9f10f94783595fbd719a0cff6`
- 契約SHA256：`f03839377158a44477a898fdc35af7d6a3b96bf953e86275915bf97324df035f`
- Input identity：`b53b89174549c721f082e1e444eaa9f88a6132df0d58e20d14e439fb70a8f539`

SOURCE commitを先に固定し、その子commitへ本書と契約を追加する。契約はSOURCE/test 41ファイルのbyte hashを保持し、自分自身・本書・自分を含むcommit SHAをhash集合に含めない。実行・テスト再現の対象は**契約を含む最終公開commit**であり、SOURCE commit単独は準備完了版ではない。将来の別承認は、その公開commitの正確なHEAD、本契約SHA256、fresh outputへbindする。この構造に循環参照はない。

今回の許可はSOURCE実装・人工unit/mock・技術preflight・契約準備・公開まで。実QuantumCircuit生成、回路build/compile、Rustiq、Operator/Statevector、保存F1の実角度の結合、実保存targetの高精度復元・数値比較、保存資源の再集計・再ランキングは実施していない。旧承認とretry許可を新B2へ流用しない。

## 2. B2の実装

元の`plan_sequence`でrun、region、basis、whole-run singletonを固定した後、適格rotation region内の最大連続同一generatorだけを結合する。Pauli word、中央support、kind、basis ID/hash、frame、run、case/macro/region、system/ancilla mappingを保持し、`P,Q,P`を移動させない。atom IDだけでは結合を決めない。

保持する等式はcontrolを含む完全なoperatorについての

\[
C_P(b)C_P(a)=C_P(a+b),\qquad
C_P(a)=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes e^{-iaP}.
\]

結合はQ13の`(P,a),(Z_aP,-a)`展開**前**。`fusion_groups`が元signed binary64角を受け、複数memberには`math.fsum`、singletonには元floatをそのまま使う。符号を二重に掛けず、Taylor orderによる異角を保持する。modulo、snap、近似的削除をしない。合計0も既存`append_central`へ渡し、非有限・overflowを拒否する。emitterは既にcontrolを実装するため、追加controlは掛けない。

元event/occurrence/phase/SequencePlanは変更しない。融合後の1 rotationを見てsingletonへ切り替えない。product/identity、Taylor/product sign/scalar、D境界融合、Gaussian actual inverse、basis共有、actual adjoint、full measured X/Yは旧経路を再利用する。

B2 ledgerには元member ordinal/atom ID/event/application、符号、angle hex、合計hex、word/support、basis/frame/run/region、元singleton、case/macro、mapping、元plan fingerprint、方針`C4_B2_adjacent_controlled_fsum_v1`を保存する。局所reportとfull build traceの双方に残し、full evolution fingerprintから既存compile identityへ結び付ける。旧B0/B1/R1のidentityにはfusion項を加えない。

## 3. 既存SOURCEの再利用と最小差分

| 対象 | 追加・再利用 |
|---|---|
| [rustiq_adapter.py][ADAPTER] | B2 route、typed fold、ledger、case/macro scopeとB2 fingerprint。元planner・run・basis policy・B0委譲・R1 bindingは共通 |
| [rustiq_execution.py][EXEC] | B2-only verification、保存target preparation、局所before/after比較、raw診断とcase/axis別費用併記。既存Journal、local/full/adjoint/wrapper/compile、stage barrierを再利用 |
| [reference.py][REF] | strict保存文字列readerだけ追加。既存reference関数は全てAST body一致を確認。新しいreference生成器を作らない |
| [runner][CLI] | `--scope C4_B2_SAVED_COMPARISON_V1`で明示opt-in。scope省略時は旧`C4_Q13_LIMITED_V1`。B2技術照合だけの`--preflight-only`を追加 |
| [adapter test][TAD]／[execution test][TEX] | 人工record、protocol token、mock worker、人工十進文字列、人工費用表を使用。実回路・実F1角度を使わない |
| `circuits.py`／`gates.py` | byte変更なし。basis provider、controlled emitter、D/result、actual inverse、Hadamard wrapper、絶対gate、official loweringとcacheをそのまま使用 |
| `SavedF1`／`f2_execution.circuit_metrics` | byte変更なし。保存law/motif復元と旧8指標だけを将来再利用。sampling・reference生成・F2実行許可は流用しない |

新しいbuilder/compiler/Pauli algebra/実行基盤は作っていない。旧40 SOURCE pathのうち35はbyte不変、5は今回の新版へ変更し、従来hash集合に未登録だったadapter testも加えた計41pathを契約へbindした。旧SOURCE commitと旧契約は保存したままにしている。

旧adapter testの「旧契約SOURCE hash一致」は、変更後の作業ファイルを旧hashと比較する方式から、旧固定commit `5038427e...`の原本を照合する方式へ変更した。新契約では現在の41 SOURCE hashを別testで照合する。旧契約のhash自体は変更しない。

## 4. 技術検証

**41 unit/mock test PASS**（従来24件と追加17件）。専用worktree・CPython3.11.9のローカル実施であり、CI・外部独立再現ではない。

検査対象は最大連続区間、同support異word、`P,Q,P`、±角、異Taylor orderに対応する異角、負の0のsingleton pass-through、0和のemitter、非有限/overflow、product/identity/frame/basis/macro fence、元run singleton、phase保持、ledger/member、route/fingerprint/cache、旧B0/B1/R1 mock互換、旧grant拒否、fresh output、reader grammar/精度文字列、1回のmock Operatorの二重reference比較、B2-only stage dispatchと失敗後の後続stage停止、人工費用表のcase/axis対応である。

初回に既存Q13 mockの`sys.modules`復元と`importlib.metadata`のmodule寿命の問題を検出した。test側でmetadataを明示importして修正し、再実行でPASSした。実Q13呼出し・科学的FAILではない。

最終確認ではmeta-path guardで**実`numpy/qiskit/mpmath/trotterlib` importを禁止**し、人工module tokenだけを許して、41件と技術preflightがPASSした。controller testで表示されるstage PASSは全てmock workerの結果であり、実科学stageのPASSではない。readerの技術テストは人工文字列とDecimalを使うmock contextだけである。

通常の再現command（科学実行ではない）：

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/pr-phase2df1pre-private-20261023/venv/bin/python -m unittest discover -s tests/management -p 'test_research_c_rustiq_*.py' -q
```

安価なstandard-libraryテストであり、実行時約0.13秒の逐次依存の小規模suiteへprocess poolを作る必要はなかった。観測CPU affinity32、MemAvailable約55 GB。これは資源capや科学処理の速度優位性の主張ではない。

## 5. 保存証拠・環境の技術preflight

旧packet：`artifacts/research_c_rustiq_limited/2026-10-24/attempt01/`。元manifest SHA256は`5d563764454bef76e97f653d71fe8a3eeb7b9f479e59bef44c39d61c30a42069`。379個のmanifest登録ファイルを**byte hashだけ**で照合し、旧summaryのPASS資格、元4case・10region、model/compiler/tolerance、reference report metadataを確認した。7入力・lockと41 SOURCE hashも一致した。

保存128桁bundle14件ではgzip/JSON、digits、keys、shape、ASCII十進/指数と`(real ± imagj)`文字列grammarだけを確認した。NPYはbyte hashだけ、QPYはbyte hashだけであり、`np.load`、QPY decode、mpmath復元、行列・signal比較はしていない。現存保存値がreaderの受理grammarに入ることと、数値復元の正しさは別の確認である。

既存専用venvのPython3.11.9 binary、50 package version、Qiskit1.3.0 API SOURCEとnative binary hashが旧provenanceと一致した。packageの実importは不要でmetadataとbytesだけを使用し、venv/package/共有cacheは変更していない。

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/pr-phase2df1pre-private-20261023/venv/bin/python scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1 --preflight-only --inputs artifacts/research_c_rustiq_limited/2026-10-24/attempt01/saved_inputs
```

結果は`B2_TECHNICAL_PREFLIGHT_PASS_NO_SCIENCE`。この照合はgrantや科学実行前preflightの代替ではなく、**次回の明示承認後に必ず再実施**する。

## 6. 確定した未承認の科学実行契約案

`C4_B2_SAVED_COMPARISON_V1`はSOURCE/input/旧packetを固定した**未承認・未実行**の案。`scientific_execution_authorized=false`。使用可能なapproval.jsonは今回作成していない。

固定scopeは保存F1の4-mode・2粒子・occupied(0,1)、density-square合成Hamiltonian、`L_D=1`・残差2fragment、`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`の一点。分子geometry/AO basis/分子DF rankは非該当。元SS2/SS3/SS4/PP0、10領域と両axis8wrapperだけを使う。native `rz,sx,x,cx`、Qiskit1.3.0 opt1・seed17・approximation1・qubits_initially_zero=false、backend/target/coupling/layout/routing/scheduling指定なしを維持する。

| 将来の科学request | 予定数 |
|---|---:|
| 保存motif restore／対象law arm | 4／2 |
| 保存128桁bundle復元 | 14 |
| target matrix変換／signal変換 | 22／4 |
| 保存結合前literal NPY load | 10 |
| B2 local／full evolution build | 10／4 |
| actual adjoint／full measured wrapper | 4／8 |
| Operator materialization | 26 |
| 絶対行列比較／unitarity | 36／14 |
| raw／compiled Statevector signal | 8／8 |
| X/Y complex pair gate | 8 |
| cold／warm request・実transpile | 8／8・8 |
| warm追加transpile | 0 |
| correctness後compiled metric | 8 |
| 保存B1 raw QPY decode／raw metric | 8／16 |
| 新独立reference・precision再比較・B0/B1/R1再実行・Rustiq/R2 | 全て0 |

保存targetのreaderはreal/imagの十進文字列を128桁contextへ直接渡し、その後だけ既存`to_numpy`／signal変換を使う。`eval`や不明文字列の推測補完はしない。保存target bytesを新outputへコピーし、原本を上書きしない。旧90/128 PASSとhashを再利用し、新しい独立referenceは生成しない。

順序は元plan/angle/input照合→局所の独立targetおよび結合前literalへの絶対比較→full `diag(I,U)`→actual adjoint→raw X/Y→compiled operator＋X/Y→全4case両axis correctness PASS後の資源比較。tolは行列/unitarity **1e-10**、complex X/Y **2e-10**、rtol0。位相fit・global-phaseだけの同値判定・事後緩和は禁止する。

B2だけを新規実行する。保存B0/B1/R1の8指標は新B2と同case/axisで併記する実装を用意したが、今回は実データへ適用していない。law加重合計、shot、全round、T count、実分子への拡張は行わない。R2は契約外であり必要時の別採択・別承認。

cacheはB2 route/policy/ledger-bound evolution/source/inputと既存full-QPY keyで分離し、cold8 miss・warm8 hitを記録する。旧cacheを流用しない。fold/emitterの部分時間、plan、whole builder、cold/warm、RSSを保存し、部分時間を重複加算しない。Rustiqは呼出しなしと記録する。旧R1単発timerはmatched反復対照ではなく、古典的速度優位は未判定。

将来は空きCPU/RAM/load/pressure/backendを再確認し、独立taskをstage内で並列化、gate間はbarrierとする。最大fanout10というtask構造と利用可能CPUからworkerを選び、BLAS/OpenMP/Numba/Qiskit/Rayonを調整する。CPU/GPU/RAM/worker/wall-timeの人工capなし。共有環境・他者jobに影響する変更はしない。

SOURCE/input/profile/保存証拠/plan/phase drift、decode不能、非有限、operator/adjoint/signal FAIL、cache/call不整合、実資源不足・共有干渉でSTOPし、partial evidenceを残す。新grantの範囲にない追加call、retry/resume、case置換・許容差変更を自動で行わない。4096行journalと16/32次元の出力範囲を保持する。

## 7. 未検証事項と引渡し

未検証は実B2 local/full/adjoint同値性、実保存referenceのmpmath復元と最終cast、raw/compiled X/Y、実compiler/cache、B2対R1の資源差、matched古典timingである。mock PASSをこれらのPASSへ置き換えない。旧pass別traceとmatched反復timingは元packetにもなく、不足を推定で補完しない。

GO/STOPは[設計正本§6][DESIGN]を維持する。全4case両axisの正しさとprovenanceが揃ってから同case/axis整数vectorを判断する。B2=R1なら単純結合の十分性を支持するがRustiq内部機構の一意同定ではない。R1追加差、B2優位、混在・小差を区別し、C4を区切る選択肢も残す。今回新規性や継続価値は判定していない。

保全対象は元workspace、旧SOURCE commit、raw・結果・grant・ledger・marker・STOP、正式共通指示、PROJECT_MAP、研究概要・VALIDATION_STATUS・validation manifest等。専用worktreeの差分は本実装6ファイルと新規本書・契約だけであり、共通研究文書は変更しない。

次の担当は**ユーザー・研究C・統括**：公開実装と契約案を確認し、科学実行の採否を決める。**Codex**は別途の明示科学承認後だけ、公開HEADと契約SHAへfresh approvalをbindし、preflightを再実施してB2限定runへ進む。現時点では公開・remote SHA/本文/SOURCE照合後に**mandatory STOP**し、C1/R2/R425/QuCLEAR/Gaussian singleton等へ自動進行しない。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7b70f6a5e3ee4ae4a55b215b49de0390e5732e5/docs/research/research_c_rustiq_angle_fusion_baseline_design_20261024.md
[ADAPTER]: ../../src/trottertracks/pr_dictionary_validation/rustiq_adapter.py
[EXEC]: ../../src/trottertracks/pr_dictionary_validation/rustiq_execution.py
[REF]: ../../src/trottertracks/pr_dictionary_validation/reference.py
[CLI]: ../../scripts/run_research_c_rustiq_limited.py
[TAD]: ../../tests/management/test_research_c_rustiq_adapter.py
[TEX]: ../../tests/management/test_research_c_rustiq_execution.py
