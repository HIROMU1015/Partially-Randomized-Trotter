# H4 D1：fingerprint不一致調査とSOURCE再現性基準案

2026-10-24 JST。**Pauli-PRの旧hash不一致の排他的な原因はUNKNOWN。
一方、元SOURCEの構築規則・保存入力と、保存D1回路の高水準命令の対応は確認できた。
旧hash一致とは別のSOURCE対応基準を、限定された費用構造診断に使う案を作成した。**
基準は未採用、実行未承認であり、今回の成果物で科学実行を開始することはできない。

実H4回路生成、H4 unitary/state/signal演算、費用compile、sampling、N/G、
新科学grant、Stage B/Cはすべて0。旧Pauli-PR hash FAIL・消費済みgrant・STOPを保持する。
人工検査17件は主agent自身の検査であり、独立レビューや実H4 certificateではない。

[機械可読調査](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspection.json)、
[12辺のSOURCE対応](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/correspondence.json)、
[非実行契約](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/contract_DRAFT.json)。

## 固定対象と来歴

Linear H4、隣接1.00 Å、STO-3G、8 system qubit＋ancilla8。
Pauli184非identity項・PR prefix27、DF tol-only1e-8／rank10・PR prefix1＋one-body。
S2、T=.8、q8、r4、K4、cosine、元trajectory_index0。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
元full Hadamard wrapper、状態準備費用除外、DF raw_concatenationを固定する。
別系列H2/S2/Yoshida4の研究とは統合しない。

| 役割 | 固定commit |
|---|---|
| 今回の基点 | `a5f77bbb99e23f15a2a465e0ab485777e7a6f40d` |
| 保存D1科学SOURCE | `48d4d27a05a529e05b11599f5f3c671302826617` |
| 元H4科学SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C descriptor export | `e80d7510880f8db993b777c627bb8e95b05a070d` |

GitHubから基点commitとD1報告全文を取得し、報告のblob IDを固定commitのローカルGit objectと照合した。
[取得記録](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/github_verification.json)。
D1封印code35件をSOURCE commit／保存SHAと照合し、入力46件、元611-file保全map、
固定runtime6,999件もSHA照合した。元H4の主要7 moduleは0c329b30とbyte一致。
C exportの既存`circuit_correspondence_proved`はencoded operator/law用であり、
旧literal/compiler certificateへ格上げしていない。

## 不一致の原因：確認事実と仮説

元[`h4_pilot_science.event_gates`](../src/trotterlib/h4_pilot_science.py:87)は、
PRではtrajectory生成前にforced-order検証用QuantumCircuitを生成する。
続いて[`AuditedPauliSignalAdapter.trajectory`](../src/trotterlib/rte_event_audit.py:91)が
保存対象の本番eventを一度sampleし、observer保存後にbodyを作る。
元[`evaluate_audited_signal`](../src/trotterlib/h4_audited_evaluation.py:63)は、
最初の3 trajectoryについて生成後・wrapper前に`Operator(circuit)`検証を呼ぶ。
この検証はphase alignmentなしで、科学的な運算変更の指示ではない。

対して[`h4_d1_adapter.build`](../src/trotterlib/h4_d1_adapter.py:92)は保存eventから直接bodyを作り、
これらの科学検証を再実行しない。構築履歴が同一ではないことはSOURCEから確認できる。
ただし、forced-order生成個数から旧名を逆算したり、旧counterを総当たりしたりしていない。

固定runtimeの`QuantumCircuit._name_update`はclass instance counterと、条件によりprocess IDを使う。
`to_gate()`は`circuit.name`をoperation名へ継承する。
`.control(1)`はgeneric bodyをunrollし、controlのdefinitionとbase_gateを作る。
[`_operation_payload`](../src/trotterlib/rte_compiled_cost.py:287)は次をhashに含む。

- operationのtype/name、qubit/clbit数、数値parameter、classical condition。
- control数、ctrl_state、base_gate、control-flow block。
- definitionの再帰命令列・bit配置・global phase。

Circuit.name単独は含まないが、to_gateでoperation.nameとなると含まれる。
従って**命名履歴だけでもhashを変え得る機構**はSOURCEと人工例で確認できる。
元の`Operator`検証による遅延definitionの生成時期も、再帰構造の履歴差の候補になる。
これらは今回の不一致を説明し得るが、**今回の差が名前だけ、またはこの二機構だけという証明ではない**。

元run01直下、4 case直下、選択index0の4 trajectory直下に限定し、
71既存file（うちtext70）を確認した。旧QPY・recursive payload・`circuit-N`名は見つからなかった。
[探索範囲と全SHA](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/bounded_recovery.json)。
未確認の別保存先まで不存在とは主張しない。広域探索、agent履歴探索、名前総当たりは0。

| 問題 | 結果 |
|---|---|
| hashが名前に感応する機構 | 確認済み |
| 元PRとreplayの構築・検証履歴の差 | SOURCEで確認済み |
| 今回のbody/base名`circuit-162`、control名`ccircuit-162` | 保存記録とQPYで確認済み |
| 旧名、旧payload、旧definitionとの差分 | 未回収・比較不能 |
| 名前だけが今回の差の原因か | **UNKNOWN** |
| 旧実行終了監査の完全性 | 元native UNKNOWNを維持 |

## 保存QPYと元SOURCEの接続

[読取検査script](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspect_saved_evidence.py)は
元QPYを読み、**新しい回路を組み立てず**保存された命令・parameterを照合する。
PauliEvolutionGateの`_define`、新control/to_gate、Operator、transpile、D1 build/compile呼出しは遮断した。
浮動小数点の係数乗算、角度生成、三角関数、phase総和、matrix/state演算を行っていない。
半stepの`.05`は固定C SOURCEの`.1/2`が表すparameter tokenとして照合した。

元Pauli JSONのconstantと184 ordered termはC descriptorと一致。
PF184／PR prefix27のordered label/coefficientとtailのlabel/sign/absolute係数tokenも一致した。
4つの元price_joinは既存`verify_join`でhash/ID/orderを確認し、events.jsonとの同一列も確認した。
C case.metadataは元result.modelに直接照合した。descriptor内自己一致だけではない。

保存Pauli bodyのPF2,944命令／PR464命令は、元SOURCEのforward/event/reverse規則と保存値から
列挙した高水準命令tokenの列と一致した。保持部のlabel/係数/time、eventのcomponent/sign/angle、
各productのPauli軸とwire、命令順序を含む。新H4 gate生成はしていない。
Pauli fullのgeneric control base内の命令列も同じ列で、control数1・state1、qargs `[8,0,...,7]`。
全4 QPYのwrapper境界はH(8)、body、H(8)、measure(8→0)、9q/1c。
DF2本は既存の旧recursive hash完全一致を継承し、DF基底・角度の再生成を行わなかった。

位相は数値再総和せず、元と同じ順序の**更新式とoperandの対応**を記録した。
Negative productのπ、event order phase、各step constant phase、DF identity/primitive phaseの所有を保つ。
同じappend関数・operand・更新順序・runtimeであることがSOURCE対応の根拠となる。
今回body/base/sidecarの保存global phase値は一致するが、これだけを旧bodyの数値位相証明にはしない。
固定`circuit_to_gate`はglobal phaseを引継ぎ、`add_control`はunrollのphaseをcontrolのP gateへ転送する。
従ってsystem-global phaseを捨てればancilla-relative phaseが変わる。人工両branch検査もこれを検出する。

**QPY上の重要な限界**：QiskitはPauliEvolutionGateをoperator/time/synthesisで保存し、
読み取り時のそのlazy `_definition`は未生成である（PF2,944／PR464箇所）。
現在のgeneric ControlledGate自体のdefinitionは保存されている。
今回、元のrecursive hash関数を読み取り後の実H4へ再適用していない。
それは内部でlazy definitionを生成し得るためである。
保存D1 hashは元stage_A/input.jsonから継承し、QPY bytesはSHAで照合した。
QPYの現在のbody/base一致は現在の保存整合性であり、失われた旧payloadとの一致証明ではない。
[保存構造](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/saved_QPY_structure.json)。

## 旧SOURCEとreplayの12辺

詳しいsource関数・根拠・限界は[correspondence.json](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/correspondence.json)、
file SHAと行位置は[source_locations.json](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/source_locations.json)に保存した。
いずれも**SOURCE規則対応を支持**する評価であり、旧program identityのPASSではない。

| 辺 | 元SOURCE → replay | 保存側の確認 |
|---|---|---|
| Hamiltonian/係数/順序 | PauliHamiltonian → C export → D1 | 原JSONと184 ordered term一致 |
| prefix27/tail | adapter.retained/tail → blocks/components | ordered label/係数/sign token一致 |
| trajectory | production-returned events → grouped_events | 元4 join hash/ID、events列一致 |
| event order/component/sign | RTEEvent → event_from_record → 同一append | saved applicationに沿うQPY命令列 |
| angle | saved rotation_angle → 同一PauliEvolutionGate | 保存operator sign/time token一致 |
| qubit順序 | range/reversed(label) → 同一append | QPY qargs、control qargs一致 |
| forward/reverse | 元trajectory loop → D1 loop | 全Pauli body ordered token一致 |
| constant/identity/order phase | 同一appendとper-step scalar式 | ordered SOURCE phase ledger、DF既存exact hash |
| global/relative phase | to_gate → add_controlの位相転送 | runtime SOURCE、現在body/base phase、人工両branch |
| whole-body control | 同一wrapper `.control(1)` | Pauli単一generic control/base/wires/state |
| Hadamard wrapper | 元同一関数 | 4 QPY境界命令/measurement、準備費用なし |
| runtime/compiler | 固定runtimeとdefault_compiler | 6,999 SHA、元profileと固定profile一致 |

## 判定A/BとClass1/2/3

**A：Historical fingerprint**を変更しない。Pauli-PRはFAILのまま保存する。

**B：SOURCE対応**は、上の12辺を満たすSOURCE定義の選択回路として扱う研究基準案。
元実行が保存されたSOURCE/入力に従ったという既存来歴と、固定runtimeの実装意味を根拠とする。
今回の候補statusは`SOURCE_SEMANTIC_CORRESPONDENCE_CANDIDATE`、`criterion_adopted=false`。
実H4全回路の新unitary検算・旧compiled program同一性・形式的certificateを意味しない。
旧payloadの未回収を自動的な科学差とみなす必要はないが、未知の重要なSOURCE接続を無視してはならない。

| full代表 | A：旧hash | B：SOURCE対応案 | 旧RZ参照 | 新RZ |
|---|---|---|---:|---|
| Pauli-PF singleton | PASS | 支持・未採用候補 | 255500 | 未測定 |
| Pauli-PR index0 | **FAIL** | 支持・未採用候補 | 9652 | 未測定 |
| DF-PF singleton | PASS | 支持・未採用候補 | 135355 | 未測定 |
| DF-PR index0 | PASS | 支持・未採用候補 | 23021 | 未測定 |

Pauli-PR旧hash `8a8adeded83edfddead525494906ccbd2a3796b11ed038f1d53f113b61468986`、
D1 hash `b677002ac37cd37a3f8a0ec84320963f7e44d3dcd29c271c369b1e6ab49fcd21`。
他3本も含め全hashはinspection.jsonに転記した。

- **Class1**：label/係数/event/angle/order/wire/control/phase/definitionの意味を変える差。解消できなければSTOP。
- **Class2**：SOURCEと再帰bodyの根拠を持つgeneric compositeの表示名差の候補。原始gate名を一括無視しない。
- **Class3**：科学的に重要な対応が未確定。理由とfieldを保存してSTOP。

今回の実差全体は旧payload欠測で分類確定できない。
特に「実差はClass2だけ」という認定はしていない。
別途SOURCEの全12辺を支持する証拠と、歴史的payloadの未確定を分離した。
今後SOURCE接続自体のphase/control/definitionが未確定と判明すればBは不成立となる。

固定runtimeのHighLevelSynthesisはoperation.nameでsupport/plugin/equivalenceを選ぶ。
generic自動名を外して同一hashにするだけの手続ではBを認定できず、compiler名非依存も仮定できない。
同じRZ数にもsemantic proofの役割を持たせない。
新しい小さな[判定module](../src/trotterlib/h4_d1_reproducibility.py)は
明示されたgeneric名pathだけをClass2候補にし、残るfield差はClass1として保守的に扱う。
不足SOURCE辺と未解決fieldはClass3状態で止める。科学dispatch機能は持たない。
このpayload比較は人工テストでのみ使い、旧H4 preimageの代用品にしていない。

## 人工検証と再利用

[test](../tests/test_h4_d1_reproducibility.py)は元adapter test8件と、新規9件の計17件PASS。
自動名だけの差、phase、event順序、ctrl_state、同名異definition、保存角度と発行角度の差を検出する。
原始gate名のwaiver拒否、SOURCE辺欠測・未知phaseのSTOP、QPY自己一致だけではSOURCE辺を埋められないことも確認した。
名前だけの人工例は旧hashが変化してもunitaryが一致し、phase変更例はcontrol=1側が変わる。
この小系の結果をH4全回路の同等性証明には用いない。
[17件結果](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/artificial/result.json)。

旧`h4_d1_adapter`、Pauli append、RTE event型、Hadamard wrapper、旧recursive fingerprint、
既存人工fixturesを再利用し、旧SOURCEは変更していない。
新規実装は差の分類と承認されない候補statusを扱うstdlib module、読取検査、追加人工testに限定した。
既存controllerはall4旧hash gateを持つため、そのままでは再開できない。
**契約案の作成は、実行controllerの新基準への実装・採用を済ませたという意味ではない。**

## 再実行契約案と最小の次段階

旧9対象とphase所有・signed非加法性式はJSONで原契約からそのまま継承した。
4つの保存full QPYをbyte再利用する案なので、追加full構築は0、
retained/event/empty構築は最大5、費用top-level compileは最大9。
実行の都合で4fullを再生成する権限は含めない。

1. **Stage A**：新基準採用と有限実行の別承認後、最小controller接続を実装・人工検証し、
   新SOURCE/入力/runtimeを固定する。別namespaceの新one-shot grantだけを使用する。
   全4入力のA/Bを別記録し、全てでAまたは採用済みBが成立すること、重要fieldにClass1/3がないことを確認。
   読取後の現在recursive hashと保存D1 hashとの照合は自己整合性gateで、旧hashの置換ではない。
   読取時のlazy定義と実入力の対応に新しい不一致が出た場合もSTOPし、無断再構築しない。
2. **Stage B**：full4を元固定順序で各一度compile。各callを事前予約し、
   旧RZ 255500/9652/135355/23021とexact一致しない最初の時点でSTOP。
   失敗call返還・残り対象への自動継続・profile変更はしない。
3. **Stage C**：full4の対応と価格の全PASS後だけ、残り5対象を各一度build/compile。
   RZ/CX/depth、pre/post histogram、保持/event/empty費用を保存する。
   独立taskは依存gateの後で並列化できる。

Count metricには元の `C_R-C_0`, `C_E-C_0`, `C_R+C_E-C_0`,
`Delta=C_F-C_R-C_E+C_0`, `C_F-C_R`, `C_F-C_E` を符号付きで使用する。
Depthを加算的に分解しない。constant/identity/deterministic primitive phaseはretained、
order/negative-product/event identity phaseはevent、fullは元の更新順序、emptyはH/H/measureだけ。
source sidecarに留め、barrier/label等をcompile入力へ追加しない。

この限定比較は、**固定SOURCE規則に対応する4代表の費用構造**を調べる用途では妥当なGO候補である。
旧hash preimage回収を必須目的とせずに進める合理的な代替案になる。
ただし、名称差だけと断定する、旧program identityを回復したと主張する、
full価格が違っても旧費用差の分解を続ける、という運用は認めない。
Stage Bの同RZだけで旧programの同一性を主張せず、256本の母平均・優位性・一意の因果寄与率へ外挿しない。
新しい比較結果・価格・非加法性は今回すべて未測定。

## 保全・終了・担当

旧611件にD1封印code/inputと固定D1 subtreeを加えた725件のbefore/after SHAは同一。
旧SOURCE・raw・grant・CONSUMED・ledger・marker・STOPは改変していない。
旧grant `b4a61a9064c687d00df10f49bbc24e90d62f47074dbc655727a6ab33e4e17ba8` は
CONSUMED＋`STAGE_A_FINGERPRINT_STOP_NO_RETRY`のまま。
今回の調査STOPは旧runの終了証拠を補うものではない。
旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、元native UNKNOWNも維持する。

共有設定・他ユーザー・別worktreeを変更せず、CPU/RAM/workerに人工的な低い上限を設けなかった。
最新版運用指示§8.4/§10を適用した。所在確認済み版は§10.2までなので、添付の§10.3相当の
「研究目的に必要十分な技術検証」は今回の直接指示として適用した。
旧記録の資源数値を新規作業の恒常上限にしていない。

専用branch `codex/non-df-h4-d1-reproducibility-design-20261024`。
commitとpush/remote照合の結果は[公開記録](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/publication_status.json)。
調査・設計・人工検証・保存・commit/push後 **mandatory STOP**。

次の担当は**ユーザー**：Bの限定主張を採用するか、保存full4＋隔離5・最大9 compileと
元count式による診断を一回承認するか判断する。
**Codex**：その明示承認後だけ、新controller接続・人工検証・新SOURCE固定・新grantを準備し、
Stage A/B/Cのgateに従って有限D1を実行する。今回その次段階は開始しない。
