# Track B：fingerprint診断source・次回診断範囲の限定レビュー

- 作成日：2026-10-12 JST
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 固定診断準備source：`b5fd12370eae8fc5064039355ba218d342125d06`
- 最終公開commit：`47bb7e8ef67690846ef4d4769a1f3057e0922061`
- 対象branch：`track-b-df-correction-fingerprint-diagnostics-preparation-20261012`
- 直前の原因監査：`86e5c43d17f8f4ba2e232776b5e4beb7e5db3a98`
- 旧比較source S'：`41349b73157c7781e0b534cf0fe61e06bf439954`
- 旧authorization A：`a1f537b68f0f22419dff0202c553b7511653caa3`
- 位置付け：診断adapterの整合性、登録外のhash差の解釈、次の実行範囲の判断。A/Bの数学・新規性・資源優位の再レビューではない。

## 0. 結論

**新しい診断adapterは、現在対象としているno-backendの費用計測経路について、採用したfingerprintの定義と主要な計測意味論を保ち、前回不足した失敗時の観測を追加するものとして支持する。今回、新たなadapter修正を必須とする阻害要因は確認しなかった。**

ただし、本branchは`RUN_READY=false`である。既存launcherの消費済みmarker・authorization・旧manifestを付け替えて動かしてよいという判断ではない。

**次の実行対象は、全A/B比較ではなく、保存済みE-r3／uncontrolled入力のidentity-only診断へ絞ることを推奨する。** 入力frame、Q、T、U、recursive payloadとhashを取得して終える。CD再計算、E-r6、局所Pauli対照の費用評価、native `transpile()`、Clifford+T合成は含めない。definition評価が内部で回路分解を行うことは別の実作業として明示する。

この限定診断は、欠落した歴史的実行を再現したと主張するためではなく、**現在の固定環境・保存入力で不一致がどの層にあるかを識別し、次の修正を根拠付きで選ぶため**に行う。

研究判断ラベルの提案：`DIAGNOSTIC_ADAPTER_SUPPORTED / PREPARE_E_R3_IDENTITY_ONLY_DIAGNOSTIC / EXECUTION_NOT_AUTHORIZED`。
これは本レビューの結論であり、リポジトリの保存statusを変更したものではない。

## 1. 確認した資料と取得範囲

以下は固定sourceに対するGitHub取得資料である。コード・報告・保存結果を区別して使用した。

| 資料 | 使用目的 |
|---|---|
| [診断準備handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/docs/tracks/algorithm_codesign/df_correction_fingerprint_diagnostics_preparation_and_gpt_handoff_20261012.md) | 認可scope、実装・試験履歴、未解決事項 |
| [fingerprint_diagnostics.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/src/trottertracks/algorithm_codesign/df_correction_comparison/fingerprint_diagnostics.py) | 観測adapter本体 |
| [native_evaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py) | whole回路構成、adapter呼出し、既存guardの順序 |
| [worker.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py) | 保存callback・provenance・旧launchとの接続 |
| [共通rte_compiled_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/src/trotterlib/rte_compiled_cost.py) | payload、canonical encoding、None fallback、no-backend費用計測の照合 |
| [新focused tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/tests/tracks/algorithm_codesign/test_df_correction_fingerprint_diagnostics.py) | テストのassertionと限界 |
| [最終tests receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/focused_tests_v3.json) | 104件PASS、実行環境、試験scope |
| [保存payload差分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/off_domain_payload_difference_audit_v1.json) | 行列bytesと名前だけの差の保存結果 |
| [失敗時metadata監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fd12370eae8fc5064039355ba218d342125d06/artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/saved_input_and_failure_metadata_v1.json) | 原実行で保存されていないものの特定 |
| [公開receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47bb7e8ef67690846ef4d4769a1f3057e0922061/artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/publication_receipt_v1.json) | 固定sourceと後続の証拠公開の区別 |

branchのremote HEADは`47bb7e8e…`と確認した。`b5fd1237…`から最終公開commitまでには、失敗・最終testのraw logと証拠索引を追加するcommit `33224f0b…`、その後の公開receipt commitがある。したがって**最終HEADを固定sourceの直接のreceipt-only子と表現しない**。比較diffでは、この期間の変更は証拠公開であり、診断adapterの改変ではない。

本レビューで全57ファイルを独立再取得・byte照合したわけではない。その件数はCodexの公開記録に依拠する。上記の主要source・報告・metadataについて実際の取得を行った。不可欠なレビュー資料の取得不足は見つからなかった。

## 2. 診断adapterと共通helperの対応

### 2.1 対象とする呼出しを限定する

共通`transpile_and_measure_cost`はbackendや外部fingerprint引数なども受けるが、新adapterは**今回の固定no-backend・外部fingerprintを指定しないwhole呼出し**を対象とする。この範囲について比較する。全APIの一般的な置換として認定しない。

両者が行う主要処理は次のように対応する。

| 処理 | 共通helper | 診断adapter | 判断 |
|---|---|---|---|
| compiler/runtimeの文脈確認 | `_validate_transpile_context` | 同じhelper | 対応 |
| input payload | `_circuit_payload`経由 | 同じhelperを一回 | 対応 |
| JSON hash | sorted keys、compact separators、default ensure_ascii | 同じencoding | 対応 |
| 取得失敗 | TypeError/ValueErrorならactual=None | 同じfallback、例外を保存 | 対応 |
| その他の例外 | 伝播 | 記録後に再送出 | fail-closedを保持 |
| compile設定 | basis、opt、seed、任意のcoupling/layout/routing | 同じ値を渡す | 固定no-backendで対応 |
| cost | counts/depth/size/phase・dataclass | 同じhelper群・field | 対応 |
| historical expectedとの比較 | native callerで判定 | 同じcaller、同じguard | 変更なし |

保存を追加した以上、wall/CPU/outputと処理タイミングが全く同じとは言えない。支持するのは**比較の意味論とhash policyの保持**であり、全副作用がゼロという認定ではない。

### 2.2 失敗が記録される位置

`BEFORE_FINGERPRINT`でframe・Q・T・raw U・input targetの情報を同期保存する。payload取得後は、その同じ一回のpayload、actual、definition indexを保存する。TypeError/ValueErrorの場合は例外を保存しactual=Noneを維持する。その他の例外も保存して再送出する。

`_definition_index`は、すでに取得されたdict/listだけを辿る。Qiskit gateの`definition`を再評価しない。この点は、自動生成名が変化し得る対象の観測として重要である。

compile呼出し前のattempt、返却後のactual/expectedの一致状況が保存される。既存のinstruction数／fingerprint guardは、その後で判定する。不一致を検出した回路の資料を保存できる位置になっている。

`save()`に失敗した場合は例外を伝播して停止する。記録できなくても後続を続けるfallbackはない。

### 2.3 配列bytesは何を復元できるか

`array_record`はdtype、shape、strides、flags、C-order bytesのhexとSHA256を記録する。最大サイズは128×128 complex128相当の262,144 bytes／配列である。値のC-order bytesは復元でき、元operandは変更しない。

ただし、C-order bytesとstrides metadataは、失われたhistorical memory allocation、address、BLAS内部状態まで保存・復元するものではない。この違いを原因断定に使わない。単一配列上限は全recordやrecursive payloadの総量上限でもない。全出力・資源は既存supervisorで別に監視される。

### 2.4 worker側の接続

whole controlled maskごとに専用logical artifactを作り、source/authorization/contract/markerのidentityを追加する。同じ呼出し内の段階更新は、既存lossless storageを通じて置換する。旧結果や別実行のpathへ上書きする設計ではない。

旧launch・manifest・消費済みmarkerを維持しているため、現在の準備sourceはそれだけでは実行できない。これは報告の`RUN_READY=false`と整合する。

### 2.5 今回の限界

入力frameの生成・Pauli table作成・UnitaryGateコンストラクタなど、adapterへ入る前に起きた失敗まで、このadapterだけで全て保存するわけではない。今回観測対象となったfingerprint guard停止の前後について評価した。

共有cost組立の小さな重複は存在する。無限の一般API互換性は要求しないが、将来共有helperを変える場合はこの固定版との対応を再確認する。今回の104件だけで全登録入力への一致が証明されたとはしない。

## 3. 登録外の観測：何が分かり、何が分からないか

保存された二つの2-mode rank-one controlでは、Q、T、U、input targetのbytes一致が報告されている。uncontrolledのhashは一致し、controlledのhashは不一致だった。

controlledのpayload差は、いずれも以下の4か所の`operation.name`だけである。

- `/instructions/0/operation/definition/instructions/0/operation/name`
- `/instructions/0/operation/definition/instructions/7/operation/name`
- `/instructions/0/operation/definition/instructions/16/operation/name`
- `/instructions/0/operation/definition/instructions/24/operation/name`

case 0では`circuit-309/312/315/318`と`circuit-379/382/385/388`、case 1では`circuit-457/460/463/466`と`circuit-527/530/533/536`の差である。名前以外の保存payload差は記録されていない。strict operator residualの0は保存された浮動小数診断値であり、独立の誤差証明に昇格しない。

### 3.1 hashは表現の同一性を検査している

本policyは、おおむね

`F(C) = SHA256(canonical_JSON(type, name, params, phase, wires, recursive_definition, ...))`

である。行列parameterの全raw bytes hashだけではない。

保存例は、**同じ数値operatorを与えても、捕捉されたrecursive nameが違えばFが違う**ことを示す。これはhash衝突ではなく、hash入力が異なるための正当な結果である。しかし、そのFを「同じoperatorなら必ず一致する」意味論的な正規形として用いるのは不適切である。

`canonical`という関数名／docstringは、入力表現のJSON encodingを一意化することを意味しても、unitary同値類の正規化を証明しない。

今回のold→new controlは別の回路インスタンスを順に作るため、生成履歴に依存する名前を含むhashだけからold/newの数理実装差を帰属できない。保存された例の結論は、**現hash policyの非意味論的な名前依存**である。

### 3.2 controlledというラベルだけでE-r3との関連を切らない

ここには本レビューによる追加のsource解釈がある。

| ケース | system mode | 比較するinput target | UnitaryGateの大きさ |
|---|---:|---|---:|
| 登録外uncontrolled | 2 | U | 4×4／2 qubits |
| 登録外controlled | 2 | diag(I4,U) | 8×8／3 qubits |
| 停止した登録E-r3 uncontrolled | 3 | U | 8×8／3 qubits |

登録外controlled例は`UnitaryGate(U).control()`ではなく、block diagonal行列を通常の`UnitaryGate`へ渡している。停止したE-r3も8×8を通常の`UnitaryGate`へ渡している。

外部の一次sourceとしてQiskit **1.3.0固定tag**の`UnitaryGate._define`を確認した。1-qubit、2-qubitを分け、3以上では`qs_decomposition`へ進む。`qsd.py`では回路を作り、その`to_instruction()/to_gate()`を組み込む。公式APIは、名前未指定時の自動名と、回路名が変換後のgate/instruction名になることを説明する。

従って両者は、同じ3-qubit definition生成経路に入るという具体的な関連がある。ただし、matrix構造や実際のdefinition内容は異なり得る。**この関連はE-r3で名前依存を優先的に調べる根拠であって、原失敗の原因の証明ではない。**

外部資料：
- [Qiskit 1.3.0 UnitaryGate source](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/generalized_gates/unitary.py)
- [Qiskit 1.3.0 qsd source](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/synthesis/unitary/qsd.py)
- [公式QuantumCircuit v1.3 API](https://docs.quantum.ibm.com/api/qiskit/1.3/qiskit.circuit.QuantumCircuit)

### 3.3 原失敗は未確定のままである

原実行ではactual fingerprint、fingerprint生成例外、入力Uのbytes、失敗したnative IRが未保存である。actualが非nullでexpectedと異なったのか、TypeError/ValueErrorによってNoneだったのかも区別できない。

今から新しい診断を行っても、過去に保存されなかった値を観測したことにはならない。再実行で同じメッセージが出ても、同じ直接原因だったと一意に決められるとは限らない。逆に新実行で一致したとしても、原実行の失敗を取り消せない。

旧classification `TECHNICAL_INCONCLUSIVE`、消費済みmarker、retry0は維持する。

## 4. 104件PASSとharness修正の評価

最終receiptは104 PASS、0 failure/error/skip。固定runtimeはNumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0。登録入力生成、CD/E評価、T合成は0と記録されている。

新testsには、C/F/slice配列のbytes復元、unsupported配列拒否、None fallback、その他例外、compile失敗、保存失敗、共有helperとのreal小回路対照、recursive hash、分割保存、guard前保存、old/newの登録外構成経路が含まれる。

初期97 PASS/7 FAILと次の103 PASS/1 FAILは、marker-absent仮定、外側RLIMIT_FSIZEと内側guardの関係、metadata childのPYTHONPATHの問題として報告されている。旧markerを消してtestを通したのではなく、before/after不変性を確認するよう変更している。failure log・当時のharness/test sourceも別に残っている。

これらはoff-domain testの反復であり、旧科学one-shotのretryとして数えるものではない。ただし、本レビューで初期失敗を全て同環境再現したわけではない。失敗理由の細部は保存報告とtest sourceに依拠する。

real共通helper対照は1-qubit RZ回路を用いるため、全登録UnitaryGateのrecursive自動名が不変であることまで保証しない。この限界を、104 PASSから拡張しない。

## 5. 本レビューの独立metadata/mock確認

### 5.1 source identity

GitHub connectorから取得した`fingerprint_diagnostics.py`をレビュー環境へbyte-identicalに複写し、Git blobを独立に照合した。

- bytes：7,223
- Git blob：`ccbef930ce58c944f80809d57553035fb50f4a66`
- SHA256：`eb228f8c41deac3de35c2a72bd6020b5f791abac83e6ba1189257eddc66ae3af`

connector返却blobと、`SHA1("blob "+length+NUL+bytes)`が一致した。実験repositoryへ書き込んだわけではない。

### 5.2 確認内容

実際のQiskit／shared helperは起動せず、小さいmock moduleを注入してadapterの分岐を検査した。**15件PASS、0 failure**。

- canonical encodingと名前field変更への応答
- C/F/sliceの値bytesとlayout metadata
- object array／サイズ上限の拒否
- payload一回・mock compile一回・保存順序
- TypeError／ValueErrorのNone fallback
- 想定外例外の非握り潰し
- fingerprint前／後の保存失敗時の停止
- mock compile／metric失敗時の診断保持
- 最終保存失敗を成功返却しないこと
- 捕捉済みpayloadだけを使うdefinition索引
- bounded exception metadata
- expected値を置換せず、不一致を記録すること

環境：Python3.13.5、NumPy2.3.5。本番固定runtimeではない。配列はmetadata用の小さいliteralであり、Hamiltonianや登録fixtureではない。

### 5.3 明示的に行っていないこと

- Codexの104件全再実行
- Qiskitのimportやreal compile、Clifford+T合成
- 登録E-r3/r6/CD frameの生成・行列再構成
- 古い実行の復元、当時のcauseの証明
- 新authorization・markerの作成
- GitHub repositoryのwrite／commit／push
- 全57公開資料の独立hash再照合

この15件はadapterの制御フローの補助証拠であり、native結果や104件へ加算する科学的evidenceではない。

## 6. 今後の選択肢の比較

| 選択肢 | 判断 | 理由 |
|---|---|---|
| 現diagnostic adapterで全CD/E/Tを再実行 | 現時点では採用しない | CDやE-r6/Tを再実行する必要がなく、旧guardで再停止し得る |
| observed値でexpectedを更新 | 採用しない | 比較条件の結果後変更。原因の区別にもならない |
| 名前fieldを一括削除してguardを通す | 採用しない | 任意のoperation名の意味を同一視できない。policy変更を隠すことになる |
| compiler seed、precision、capを変更 | 採用しない | ここまでの資料に解決策として支持する根拠がない |
| 同じoff-domain準備だけをさらに繰り返す | 優先しない | 名前依存の存在はすでに分かっている。登録E-r3の分岐を識別する情報が不足 |
| **保存E-r3のidentity-only診断** | **次の最小scopeとして推奨** | originalの失敗点がtranspile前のinput hashであり、その情報だけ取得できる |
| 過去の根本原因を完全確定するまで研究を永久停止 | 採用しない | 未保存の過去は完全復元できるとは限らない。将来の妥当なidentity設計を判断する情報が目的 |

現adapterは妥当でも、その関数自体は必ず`transpile()`へ進む。**次のidentity-only診断で、そのfull関数をそのまま呼ぶことは推奨しない。** payloadを一回取得して保存する部分を再利用する、診断専用の薄い入口を用意する。

## 7. 次に準備する診断のscope

これは本レビューで選択した次の技術的課題であり、実行認可ではない。

### 7.1 対象

保存済み`E-inputs.json`／対応jobに含まれる**E-r3、rank2×3、uncontrolled**だけを対象とする。保存hexと元結果commitのidentityを検証し、その値を使う。新しいRNG、seed変更、別fixture、入力の選び直しは行わない。

符号(+1,−1,+1)、nested commutator、既定の`expm(1j*.03125*T/4)`は維持する。

### 7.2 比較する二経路

1. 失敗時のsourceに対応するcurrent numerical pathを、同じ保存frameから再構成する。
2. 旧helperの数値式を、**同じ保存frameと現在の固定runtime**に適用したreference pathを再構成する。

二番目は歴史的旧runの原本ではない。歴史的frame・runtimeが完全保存されていないため、`legacy-source reconstruction under current runtime`として明記する。

各経路のframe/Q/T/U/input targetのdtype・shape・bytes hashと復元可能なbytes、layout metadataを記録する。最初に差が現れる段階と、最終targetの一致・不一致を分ける。

### 7.3 hash観測

各経路でinput circuitを一つ作り、recursive payloadを一回取得して保存する案を基本とする。二つの経路の評価順は結果前に固定し、生成名やcounterを都合よくresetしてexpectedへ合わせない。

actualがNoneか、例外のtype/message、historical expectedとの一致、二経路のpayload field差分を保存する。名前だけの差、matrix parameter bytes差、phase/wires/definition数値差を区別する。名前fieldを隠したsecondary diffを作る場合でも、それは**診断用の比較表示**であり受理guardを変更しない。

現在の実行順序・circuit生成contextも記録する。fresh processを使っても歴史的contextの復元と呼ばない。

### 7.4 実行しないもの

- `transpile()`によるnative compile／gate数比較
- CDの6 taskの再計算
- E-r6、registered controlled側への拡張
- local Pauli対照の費用・moment探索
- Clifford+T合成、T-key inventory生成
- quantum shots、trajectory、分子、LP、GPU
- 旧runner／marker／authorizationの再利用

ただし、`operation.definition`の評価はQiskit内部で`qs_decomposition`等を呼び得る。従って「native compile API呼出し0」と「行列計算・回路分解0」は同じではない。definition生成を診断作業としてカウントし、CPU/RSS/output管理に含める。

### 7.5 保存・資源・実行管理

既存のlossless chunks、failure-before-guardの記録、process supervisionを使う。新しい診断専用result namespace／source manifest／launch contractを準備し、旧runを上書きしない。

resource上限と呼出し数は、**この縮小scopeに必要な範囲として結果前にCodexが具体化**する。今回、旧科学contractの上限拡大やprecision変更は認めない。二経路各一回の観測を基本とし、失敗を見てcall追加・再試行しない。

一致しなくても診断資料が保存できれば、診断としての完了と、不一致という観測結果を分離できる。これは本番比較に不一致を受理することではない。`MATCH/MISMATCH/EXCEPTION`を観測値として残し、各caseの完了とtechnical failureも分ける。

### 7.6 診断で答える問い

- 現current pathのactualは有効digestか、Noneか、その他例外か。
- 同じ保存frameで旧/newのどの段階から値／bytesが異なるか。
- 同じ最終行列でも名前だけでrecursive payloadが異なるか。
- 変更した診断観測のcontextで、historical expectedと一致するか。

この問いに答えるために全A/Bを再実行する必要はない。

## 8. 診断後の判断規則

| 観測 | 言えること | 次の判断 |
|---|---|---|
| matrix bytes同一、payload差が自動名だけ | current contextで表現名依存が確認された | 将来のidentity policyを分層化する変更案を検討。旧guardを遡及無効化しない |
| frame同一だがQ/T/Uに差 | 数値構成経路の違いをlocalizeできた | 操作順序と同じidealへの距離を評価。単にold bytesへ合わせる修正を先に採らない |
| fingerprint取得で例外 | currentのNone／例外分岐を識別 | 例外を限定修正。物理的誤りや資源非優位とはしない |
| current hashがhistorical expectedと一致 | currentの試行では一致 | 原失敗の取消ではない。context依存・当時の欠測は残る |
| caps／保存／認証により観測不足 | technical inconclusive | 部分結果から勝敗やcauseを確定しない。範囲拡張は別判断 |

**過去のcauseを一意確定することと、今後の妥当な比較を可能にすることは別の目的である。** 診断結果が将来のidentity修正判断に十分なら、失われたhistorical payloadを永久に求め続けない。一方、現時点で名前だけが原因だと断定してexpectedを更新することもしない。

## 9. 今後のfingerprint policyについての考え方

今後は以下の層を区別するのが合理的である。ただし今回、これを実行契約として採択・実装認可したわけではない。

1. 入力値identity：frame・generator・unitaryのbytes、dtype、shape、wire/control convention。
2. 意味論の確認：同じideal targetへのstrict operator error、特にcontrolled relative phase。
3. 表現／来歴identity：自動名を含むrecursive payloadを原本として保存。
4. 実装identity：compiler/runtime/settingsと実際のnative IR・数値価格の記録。

unitaryとして近いことだけでは、同じnative価格の実装とは限らない。名前fieldを全消去すれば安全ということでもない。将来のcanonicalizationを採るなら、除外対象を自動生成された表示名などへ限定し、phase・params・wires・definitionを保持する検証が必要である。

一方、source/counter履歴に依存する古いopaque hashを唯一の科学的正当性判定とするのも妥当ではない。本レビューではこの設計上の弱点を認めるが、A/Bのguardをその場で置換はしない。

## 10. 担当と次の手順

**直近の担当はCodex。** 現adapterを再々監査するのではなく、E-r3 identity-onlyに閉じた薄い診断runner、new namespace・manifest・scope contract、off-domain failure-path testsを一つの作業として準備する。

必要な処理分離やcallbackの実現方法はCodexの裁量でよい。アルゴリズム、登録fixture、expected hash、cap、precisionを変更する提案が生じた場合だけ、研究・契約変更としてGPTへ戻す。

準備後の確認は、本レビューで決めたscopeを超えていないこととsource/manifest/launchの結合へ限定する。同じ研究方針や診断adapter全体の詳細レビューを何度も追加しない。

登録inputを用いる診断実行は、固定準備sourceと認可scopeが明確になり、利用者が別途明示的に承認した後に行う。本レビューの開始承認をその実行承認と同一視しない。

旧CD保存結果、未完了E/T、TECHNICAL_INCONCLUSIVE、旧source・authorization・markerは不変。新しい診断で得たものを、旧runが取得したactualとして書き戻さない。

## 11. 最終整理

- 診断sourceの主要変更は妥当と評価する。
- 同一operatorでhashが変わる例は、名前込みの表現identityの問題として支持される。
- 8×8/3-qubitの共通definition経路を確認したため、controlled例はE-r3に無関係ではない。
- ただし、原E-r3の直接原因は未確定で、再試行で歴史的欠測が埋まるとは約束できない。
- 次は全A/B比較ではなく、保存E-r3のidentity-only診断準備。
- 本番compile/T合成は不要。definition生成自体の計算は正しく記録する。
- 実行は別途の明示承認までSTOP。
- この技術的進捗から、A/Bの科学的優劣・新規性・Track Bの研究終了は判定しない。
