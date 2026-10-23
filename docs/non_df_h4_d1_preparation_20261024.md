# H4費用逆転：限定診断D1の実行計画・非実行契約

2026-10-24 JST。**D1の9対象、descriptor再構成方法、phase分離、旧費用再現性、非加法性、GO／STOPを固定した。今回の契約準備は完了したが、実行可能なproduction adapter・新SOURCE／runtime seal・grantは未作成である。** 実H4での再構成／fingerprint一致は未検証で、次回の実行前提条件として残す。今回の成果を「実回路再現済み」「直ちに科学runを開始可能」とは表示しない。

[非実行契約](../artifacts/non_df_h4_d1_preparation/2026-10-24/contract_DRAFT.json)、
[固定入力・代表回路・検証結果](../artifacts/non_df_h4_d1_preparation/2026-10-24/preparation.json)、
[静的／人工checker](../artifacts/non_df_h4_d1_preparation/2026-10-24/check_preparation.py)。
実H4 build／compile／新科学値集計／sampling／RNG replay／N/Gは0、新grantなし。
旧結果・科学SOURCE・ledger・marker・STOPは変更していない。

## 1. 目的と固定条件

PFではDF、PRではPauliが安くなる保存compiled RZ単価の逆転に対して、
保持部・event部・全体のどこに価格が残り、接続／compilerの非加法性がどの程度あるかを、
固定代表4構成で調べる。
一般的な表現の優劣、母平均、256 trajectoryの寄与率、厳密精度の認定を目的にしない。

固定条件：Linear H4、隣接1.00 Å、STO-3G、system n8・ancilla8、共通保存stateの来歴、S2／T=.8／q8／r4／K4。
Pauli184項・PR prefix27、DF tol-only1e-8／rank10・PR prefix1＋one-body。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend／coupling／layout／routing指定なし。
cosineのfull Hadamard wrapperを測定し、状態準備を除く。
DFは元のraw_concatenation、occurrence内の隣接equal-basis reuseを維持する。

基点branch `codex/non-df-h4-cost-reversal-structure-20261024`、
commit `6cc5bded8e0ea2a4f0f1ad7e990f909eb1878d13`。
元H4科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、
C `e80d7510880f8db993b777c627bb8e95b05a070d`、
L `eb195bf2dedd0aed9d80ad7f84948b59ea53d854`、
C/L `0880f4af98b125d62599e970db89fc33634d1b61`、
条件付きadapter `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8`、
結果 `5600f0e4b28c140a6a50b8aba47425860972a932` を保全する。

基点commit、報告／旧案、event監査・compiler・C exporterの固定SOURCE、選択した4price_joinをGitHubから実取得した。
[取得9file](../artifacts/non_df_h4_d1_preparation/2026-10-24/github_verification.json)をローカルGit blobと固定refに照合。
前回の[構造報告](non_df_h4_cost_reversal_structure_20261024.md)と
[条件付き結果](non_df_h4_conditional_cng_results_20261024.md)の科学的限界は変更しない。

## 2. 選択した入力と9回の有限予算

以下は保存price_joinからの**直接引用**。PR256本平均から代表単価を推定したものではない。
全4caseのtrajectory_indexは0。新しい代表選択・価格順の探索・event再集計は行っていない。

| case | 選択 | trajectory ID | 旧cosine RZ |
|---|---|---|---:|
| Pauli PF | deterministic singleton | d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d | 255500 |
| Pauli PR | 保存index0 | 93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97 | 9652 |
| DF PF | deterministic singleton | 71567586d92a908244f28a7ed1339c7d93907f4d7e7f4912bf492421dea1b83f | 135355 |
| DF PR | 保存index0 | ee4b33ef3f4ee363ae4a3d4c1d8b468dbbc0fe20a269361b22d68f1799e274cf | 23021 |

各入力のpath、SHA、event hash、trajectory hash、cosine input fingerprint、旧compilerは
preparation.jsonの `selected` に固定した。
PR平均9,321.10／23,678.44をこの4回の再現性判定へ流用しない。

| id／出力名 | 元case | 内容 | 上限 |
|---|---|---|---:|
| pauli_jw_PF_full | Pauli PF singleton | 元full | 1 |
| pauli_jw_PR_full | Pauli PR index0 | 元full | 1 |
| df_native_PF_full | DF PF singleton | 元full | 1 |
| df_native_PR_full | DF PR index0 | 元full | 1 |
| pauli_jw_PR_retained | 同PR | 保持部＋deterministic phases | 1 |
| pauli_jw_PR_event | 同PR | 保存event列＋event phases | 1 |
| df_native_PR_retained | 同PR | 保持部＋deterministic phases | 1 |
| df_native_PR_event | 同PR | 保存event列＋event phases | 1 |
| empty | 共通9qubit | H(ancilla)→H→measure、phaseなし | 1 |

**最大9回はtop-levelの費用transpile呼出し数**で、CPU／worker制限ではない。
元generic controlの内部 `_unroll_gate`／BasisTranslator等は元builderに必要な処理として別記録し、
追加の価格測定transpileや別診断回路を隠して増やさない。
新しいprivate cacheまたは既存compiler直接呼出しを用い、旧価格cacheを新compile済み証拠へ読み替えない。
失敗したcallも消費済みとして保存し、retry／refund／代表差替えはしない。

次回は4fullの入力対応・fingerprintを先に確認し、4fullだけをcompileする。
全4件が旧整数RZと一致してから、残る4ablation＋emptyへ進む。
独立baselineや後半5対象は並列化可能。依存関係だけを固定し、恣意的なworker上限は置かない。

## 3. 既存実装をどこまで使えるか

| 既存資産 | D1で再利用する範囲 | そのまま実行できない理由／最小差分 |
|---|---|---|
| pauli_signal_adapter | append_pauli_evolution／append_pauli_event | trajectory()はsamplingする。保存eventを供給する薄いschedule接続が必要 |
| rte_event_audit | verify_join・event／trajectory hash規則 | 今回は純関数のみAST抽出して利用。監査adapterのtrajectoryは呼ばない |
| df_partial_s2 | blockのbasis inverse→primitive→forward、phase、forward/reverse順 | preparationを作り直すと数値再準備になる。保存primitiveを入力にする小さいemitter接続が必要 |
| df_partial_s2_repeated | rawのstep接続規則 | make_df_partial_s2_repeated_requestはseed展開・samplingするため禁止。保存outer/inner列から接続する |
| df_rte_qiskit／df_rte_circuit | 既存sequence requestとbuild_sequence、basis reuse、phase処理 | samplerや_make_eventを通さず保存event objects・specs・registryを復元する |
| df_rte_tail | DFBasisRegistryの型／gate・basis hash検証 | gate descriptor復元が必要。DF生成／固有分解／Gaussian分解は呼ばない |
| df_trotter.ops | append_diagonal_primitives | one_body_diagonal_primitives／df_squared_diagonal_primitivesの再計算を避け、C保存listを渡す |
| fixed_signal_evaluation | 元Hadamard wrapper・compiler設定 | evaluate経路はstate／reference／signal等も計算するので使わない |
| h4_audited_evaluation | 旧price／fingerprintの保存仕様を参照 | evaluate_h4_audited全体は新signal・sampling等を含むため呼ばない |
| rte_compiled_cost | 既存transpile_and_measure_cost・histogram・depth・recursive hash | 新しいcompiler wrapperを作らず1対象1call。callbackは既存APIにない |
| df_partial_s2_cost／repeated_cost | additive部品・matched differenceの定義を設計参考にする | lawの列挙／MC・複数compileを含むためD1 runnerとして流用しない |
| native_df_gate_model・既存tests | 人工のphase／inverse／basis reuse／wrapper検証 | 宣言的loweringモデルはQiskit実測価格ではない。実H4へ数値適用しない |
| DF structural counters | SOURCE上のliteral構造・予算検証方法を参照 | 今回実H4の新しい命令数／価格を計算しない |

既存関数の適合部分を使い、samplerやpreparation入口だけを避ける。
新規に必要なのは**JSON→既存gate/event型の復元と、保存primitiveを元順序で接続するadapter**である。
既存SOURCEを改変して別意味の関数に差し替える設計にはしない。
本段階ではそのproduction adapterやscientific runnerを実装／起動していない。

### Pauliの復元手順

1. 元pauli.jsonからlabel／係数／constantをそのまま読み、C modelのblocks／components、旧result.modelのprefix／tailへtoken一致で結ぶ。
2. 保存application_sequenceを含むRTEEvent等のnested型を復元する。order・component・sign・angle・phase・outer/innerを保存値から取り、seed・distributionから再抽選／再計算しない。
3. 元順序で各stepのforward retained half、保存r event、reverse half、元constant phase更新を行う。元append関数を呼び、global phaseの足し方・順序を維持する。
4. body全体を元wrapperのgeneric controlへ渡す。D2のcentral-only controlに差し替えない。

### DFの復元手順

1. C model内の各blockのbasis operations、RZ/RZZ/global_phase primitive、native_scalar_phasesを使う。PR保持2blockは保存PF先頭2blockとtoken一致する。
2. RZ／XXPlusYY descriptorのparameter文字列を元binary64へ復元し、同じgate class、qubit順、明示zeroを維持する。元gate.inverse()を逆順に使う。保存matrixをUnitaryGateとして置換しない。
3. eventのbasis hash／ID／support／sign／absolute coefficient／descriptionをC保存componentへ結ぶ。現在このdescriptor joinはPASSしているが、runtime gateの復元と局所matrix hash一致は次回の未実行gateである。
4. 既存DFBasisRegistryとDFRTEEventSequenceCircuitRequestを直接作り、q stepごとの保存r eventsをbuild_sequenceへ渡す。隣接equal-basis reuse／identity flush／末尾closeを元通り使う。tail ID/hashが保存元と結合できない場合は、APIが許す両方Noneをdiagnostic provenanceとして明記し、旧識別子を捏造しない。event自身のID/hashは変えない。
5. 各stepの元scalar phases、forward blocks、occurrence、reverse blocksをrawで接続する。primitive内は元append_diagonal_primitivesのRZZ→RZ→phase順。q境界最適化・別basis planは導入しない。
6. 元wrapperのalready_controlled=Trueを使い、basisを再度generic controlしない。

使用するC exporterのSOURCEは、同じ原preparationからbasis／primitiveを保存した経路である。
Cのfull operator／state／固有値をD1へ持ち込んで再計算しない。
次回に許す数値操作は明示承認されたgate復元・小さい局所gateのhash検証・制御構成・費用compile／診断量に限り、
SCF／FCI／DF再生成、full Hamiltonian／state evolutionは必要ない設計とする。

## 4. 部分回路・phase・wrapperの固定規則

| 回路 | 保持する部分 | phaseの所有者 |
|---|---|---|
| Full | 元順序の保持halfとevent occurrenceをq回 | 元の全phaseと浮動小数点更新順をそのまま維持 |
| Retained-only | forward/reverse保持部のみをq回。eventを削除 | constant、DFの抽出identity、deterministic primitive phase |
| Event-only | 保存event occurrenceをqグループ順に残す | order phase、negative product sign、event identity rotation |
| Empty | 共通H→H→measurementのみ | なし |

System bodyのglobal phaseも、controlled bodyではancilla相対phaseとして測定に効くので捨てない。
price_joinのscalar_phase.angleは監査用の全体要約で、元の各step更新をその値1個に置換しない。
Fullを、隔離したRetainedとEventの集約phaseを後から足して再構成しない。
浮動小数点の加算順・modulo処理まで含めて元fullの経路を保つ。

Event-onlyでも各DF occurrenceの最後にbasisを閉じ、phaseを出す。
保持部を削除したため隣り合った別stepのevent間を、新しいbuilder reuseで融合しない。
Retained-onlyでも新しい手動mergeはしない。ただし同じ元compilerが削除後の隣接gateを最適化することはD1の測定対象である。

これらはunitaryの変わる**診断ablation**で、元PF/PRの有効なシミュレーションではない。
同じq／event列から作ることは、隔離回路が元fullと同じ誤差を持つという意味ではない。

## 5. 再現性のGO／STOP：input hashは存在するが未照合

4price_joinすべてにcosineの `wrapper_fingerprints` があり、旧input hashを新たに仮定する必要はない。
event hash／trajectory hash／IDは既存verify_joinで確認し、各旧resultのsingleton／index0 rowと直接一致した。

次回のfull4件の採用には、SOURCE／使用input bytes／実runtime closure／compiler／ID・event列／gate descriptor・control・phase・順序の対応に加え、
**旧recursive input fingerprintの完全一致と旧整数RZの完全一致を必要条件**とする。
4fullの入力hashをcompile前に確認し、1件でも不一致なら費用compileへ進まない。
full価格が1件でも不一致ならablationをcompileせずSTOPする。
出力や失敗receiptは保存するが、その結果を旧費用の原因分解として採用しない。

### Pauliの自動gate名に関する未解決リスク

旧fingerprint SOURCEは再帰的なoperation.name／型／parameter／definition／wire／global_phaseを含む。
Pauli wrapperのto_gate()は元QuantumCircuitの自動名をgate名に使い、そのcontrolled gate／base_gateにも名前が残る。
旧nameそのものやhashのpreimageは、今回調べたprice_join／result／既知ログに別項目として記録されていない。
従って、同じ意味の回路を正しく復元しても自動名だけで旧hashと異なる可能性がある。

既存hash関数をmock operationへ適用した人工テストで、同じparameter・配線でも名前の差がhashを変えることを確認した。
これは実H4で不一致が起こったという観測ではない。
次回はこの不一致を成功へ読み替えず、名前の総当たり・hashに合わせるrename・無視するfieldの追加は行わない。
後からnormalized descriptor hashを保存しても、それを旧input hashの代替証拠にしない。
**不足が疑われるのは旧program／自動名preimage側の情報で、保存eventやbasis／primitiveを補う新科学計算ではない。**

同じRZだけの一致は不十分。
input hashと価格がすべて一致しても、それは固定runtimeでの入力対応と価格再現であり、
失われた旧compiled programとのbitwise同一性、旧native PASS、元監視終了の回復を意味しない。
この境界を新しいD1のnative終了証拠で置換しない。

## 6. 測定と非加法性

保存単位はvariant idごとのinput／output QPY、file SHA、recursive fingerprints、
metrics JSON、source-sidecar tagと基底counterとする。
metricsにはRZ／CX整数、total depth、既存APIのRZ/CX depth、pre/post histogramを記録する。
全体summaryではbaseline gate・variantの完了／未完了と診断式を対応させる。
全件保存に必要なfile出力が失敗したら完了PASSにしない。

Pauliはwrapper前のbody、control後のtranspile入力、post出力を対応させる。
DFには「whole-body control前」という共通経路がないため、
元scheduleの非制御primitive記述とcentral制御化後の命令をtagし、
**pre-controlの仮想histogram**として実在するPauli bodyのhistogramと区別する。
そのためだけに追加のDF全body回路や新たなcompileを作らない。

Tagはsidecarのみ。gate name／label、barrier、命令列を書き換えない。
role（保持／event／scalar／wrapper）とkind（basis／Pauli／diagonal／control）はnestedなので、
basisを保持部と別の加算価格にする二重計上をしない。
最終RZ全てにcomponent所有者を強制的に付けない。

固定compiler関数はpre/post metricを返すがcallback引数を公開しないため、
D1は**pre/post比較を必須、実pass traceはnull＋理由**とする。
traceのための別passmanager、compiler wrapper変更、追加compile、optimization変更を行わない。
固定runtimeのpass SOURCEは可能な機構の説明であり実行traceとしない。

Count指標m∈{RZ,CX}ごとに、次を事前固定する。

```text
retained_increment = C_retained - C_empty
event_increment    = C_event - C_empty
isolated_prediction = C_retained + C_event - C_empty
Delta_interaction  = C_full - C_retained - C_event + C_empty
contextual_event_increment    = C_full - C_retained
contextual_retained_increment = C_full - C_event
```

共有wrapperを重複補正した比較である。
phase配分も固定しているので、それを変えて望むDeltaを選ばない。
負のincrement／Deltaもそのまま報告し、0へのclipや加算的な寄与率への配賦をしない。
Deltaには制御構成の展開・隣接境界・再合成・phase・wrapperの相互作用が混ざり、
一意の「compilerだけの費用」にはならない。
Depthは各回路の値を報告する。並列性を持つdepthを加算予測へ流用しない。

## 7. H1〜H5とD2／D3の判断

| 問い | D1で分かる範囲 | 残る識別不能要因 |
|---|---|---|
| H1 Pauli保持部が安い | 両PRの隔離保持部とcontextual増分、Z/ZZ対DF保持basisの対応 | Z/ZZ選択だけの因果効果、保持operator量の違い、最大寄与率 |
| H2 Pauli eventが安い | 同じ固定index0のevent-stream全体価格／tag比較 | law／角度／実event内容／controlが違い、event母平均や1event公平単価を推定しない |
| H3 DF両部にbasis費用が残る | 保持・event双方のbasis tagと元builder counterを記録 | basisそのもののcompiled価格、全RZへの因果配賦 |
| H4 compiler相互作用 | Delta≠0ならその代表の非加法性を検出、符号と大きさを表示 | 接続・control・wrapper・compilerを単独に分離しない。「大きい」の恣意的な事後閾値を設けない |
| H5 control実装差 | source経路とcontrol前後の構造変化を観測 | 同じunitaryでcontrol位置だけ変えた費用差はD2が必要 |

このD1から256本平均の逆転原因が統計的に分解できるわけではない。
代表4構成の再現性と局所機構の診断として位置付ける。

**D2**はD1 baselineを採用でき、Pauliのgeneric controlが未解決の研究上の問いとして残る場合だけ提案する。
同一unitary・両control branch・constant/sign/order phaseを保つ人工検証と別承認を要し、最大追加2compile。
**D3**はD1のDF-PRで実際のcancelled equal-basis pairが正の場合だけ候補にする。
0なら新しい代表を探さずskipする。別承認で最大追加1compile。
今回はD2/D3のadapter実装・実行準備・数値実行へ進んでいない。

## 8. 今回の検証・保全と次回の起動条件

人工14件PASS：新しい契約検査7件と、既存native gate人工tests7件。
phase所有、9対象固定、hash欠測／不一致のSTOP、同じRZだけでは通らないこと、
自動gate名のhash依存、signed非加法性、
basis inverse／reuse／identity barrier、負符号と相対phase、wrapperを検証した。
宣言的gateモデルによる小さい人工入力だけで、Qiskitの実H4再現や費用最適化をテストしたものではない。
主agent自身による確認で、独立した科学レビューではない。

既存611fileのSHA mapを再利用し、そのmapを複製せずpath／SHAで参照した。
関連source、今回実際に使う選択12receipt/resultとCモデル等のSHAをpreparation.jsonに固定した。
Qiskit runtimeの参照source10件は既存C legacy-runtime sealと照合した。
これは**参照runtimeの静的照合**で、新しいproduction runtime sealの完成とは区別する。

契約flagsはexecution_authorized=false、dispatch_enabled=false、grant_valid=false、
grant／science_SOURCE／production_adapter／launch_commandはnull。
flagsを反転するだけでは実行可能にならない。
今回のcheckerには科学dispatchもrunnerもなく、科学grantを発行・消費していない。

次回は、ユーザーの明示承認でD1のdescriptor回路生成・最大9費用compile・診断量計算を許可した後、
Codexが最小adapterを人工検証し、新SOURCEをcommit、新runtime／入力SHAとschemaをsealする。
承認参照・新SOURCE・runtime・input manifestを持つ別active契約と新one-shot grantを作り、
exclusive claim／call ledgerを先に保存してから起動する。
元grantは使わず、使用済みcall・失敗を巻き戻さない。

Native supervisorはexit code、worker identityと回収、出力永続化、最終STOPを記録する。
正常終了／異常終了をいずれも保存し、未確認値をPASSにしない。
一時的なCPU／PSI競合、無関係なworktreeの更新だけで正常workerを中断しない。
実OOM・継続不能・自分のprocess異常・使用input破損への保護は維持する。
資源方針は§8.4、再利用は§10に従い、共有設定・他ユーザーを変更しない。
所在を確認した2026-10-23版には§10.3がないため、添付が明記する「研究目的に必要十分な技術準備」を適用した。

専用branch：`codex/non-df-h4-d1-contract-preparation-20261024`。
[STOP・公開記録](../artifacts/non_df_h4_d1_preparation/2026-10-24/completion.json)は今回の契約準備だけの記録で、
旧N/G null、C/L BLOCKED、compiler MISSING_EVIDENCE、元native UNKNOWNを維持する。

**次の担当：ユーザー**が、この契約のD1限定build／compile／診断量計算と、
新adapter実装・seal後のone-shot実行を明示承認するか判断する。
**Codex**は承認後だけその実装・固定・新grant発行とD1を行い、fingerprint／旧RZ不一致ならSTOPする。
旧hash一致が自動名の理由で閉じない場合も迂回せず、その不足をユーザーへ返す。
今回は契約準備・技術検証・保存・commit／pushまでで **mandatory STOP**。
