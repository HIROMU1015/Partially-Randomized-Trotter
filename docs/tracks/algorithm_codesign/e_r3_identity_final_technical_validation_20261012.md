# E-r3 identity-only診断：最終技術検証

2026-10-12 JST。判定：**BLOCKED_NEEDS_TECHNICAL_REPAIR**。

固定sourceの来歴・入力・実行保護は照合できた。ただしobserverの設置／復元失敗を
通常のfingerprint例外として完了扱いにする経路を、登録外mockで再現した。
この一点を限定修正するまで登録診断の実行承認待ちにはしない。
研究方針・数式・expected hashの変更を求める問題ではなく、現時点でGPTによる追加の詳細研究レビューは不要。
今回は固定sourceを修正せず、新authorizationも作成せず、証拠を公開してmandatory STOPする。

## 対象・指示・identity

| 役割 | 固定値 |
| --- | --- |
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 診断source S | `8377c1d12439cabab3b97bcc1f57ac030a479785` |
| 公開確認 P | `0cf991d4371250dc6a2d8b346ed4ed7414f4a198` |
| 元準備branch | `track-b-e-r3-identity-diagnostic-preparation-20261012` |
| 今回の監査branch | `track-b-e-r3-identity-final-technical-audit-20261012` |
| 旧science source | `41349b73157c7781e0b534cf0fe61e06bf439954` |
| 旧authorization | `a1f537b68f0f22419dff0202c553b7511653caa3` |
| 旧result公開基点 | `ce0e286430a2a23869cb9417c82113c98064c72c` |
| 旧marker SHA256 | `dd6b9d5ec96e9d34d7bea0277eadb03f031992e1010cfdb2c940a393b2242701` |

PはSのreceipt-only直接子であり、実行authorizationではない。今回の監査commitも実行HEADにしない。
ユーザーの「最終技術検証と実行承認準備」指示に従い、登録診断は実行しない。

最新版[共通指示原本](../../research/research_project_instructions_gpt_codex_review_approval_2026-10-10.md)を
追加後に全文確認し、無加工で本branchへ受領した。37,287 bytes、
SHA256 `de8641791d1b42aa2fc1de6a5550c896fa573bf9b15f8a7041fa48c70ec1d0b7`。
[受領記録](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/common_instructions_intake_v1.json)。
§2/3/5/14に従い、技術事項をCodexで完結し、科学的判断を伴わない修正のためにGPT詳細レビューを反復しない。
この資料の受領は実行承認ではない。

前回[GPTレビュー](../../research/track_b_df_correction_fingerprint_diagnostics_review_20261012.md)と
[source準備報告](e_r3_identity_diagnostic_source_review_and_gpt_handoff_20261012.md)は変更していない。
その時点のsource-review待ち記述は履歴として保持し、本報告が今回の技術検証判定を追加する。

## A：GitHub・source・保存入力の照合

[機械可読監査](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/identity_and_remote_audit_v1.json)：
source manifestの**260ファイル**について、ローカルbytes/SHA256、SのGit object、GitHub raw取得を照合。
authorization、manifest、34 PASS記録、静的監査、旧handoff、Pのpublication receiptを加えた
**266ファイルすべて**のremote取得bytes/SHA256が一致した。
元準備branchのremote SHAはPと一致。取得失敗・欠落・ref不一致は0。
監査scriptはHTTP 404/ref-or-path、アクセス／サービス障害、接続障害を区別し、取得失敗を未pushと即断しない。

| 保存資料 | bytes | SHA256 |
| --- | ---: | --- |
| `E-r3.job.json` | 2,130 | `545ee96c6929e88779bc48fd06fe54f0950fd4cdc42424d9f1b5ec319ff8dc3a` |
| `E-inputs.json` | 5,500 | `71494102a896c53f3666d6a58ce4ec76a9b1e882e7d6dd5ca81ea03d00f010aa` |
| `phase_E_checks.json` | 6,354 | `1751a7fd410410291d1f13e48e628fc00311021827be32e20cd1199500108b47` |
| legacy helper source | 14,429 | `5b13247a292d704b65efcb68b3385bfda36302281da92c826b2664813aa32ee7` |

jobと保存入力のr3 hex、3×rank2の形状metadata、jobの旧marker参照を確認した。
これはJSON/bytes確認のみで、登録frameをarrayへ変換していない。
旧Phase Eのr3/uncontrolled expectedは
`171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`のまま。
旧result inventory **28ファイル**も不変。旧science classificationは`TECHNICAL_INCONCLUSIVE`のまま。

新authorizationのcontract/manifest SHA256は実ファイルと一致し、status pending、
diagnostic/science authorized=false、source_commit=null、runs=0、retry=0。
新result namespace `artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1/` は存在しない。
監査branchに追加したtests・報告はSのmanifestへ追加していない。
固定source Sの260保護ファイルは検証前後とも不変。

## B：診断内容

静的source確認と保存済み34 PASS、追加28件で次を照合した。

- 保存r3・3 mode・rank2×3・uncontrolledのみ。current→legacy各一回。
- signs `(+1,-1,+1)`、`comm(Q0,comm(Q1,Q2))`、`expm(1j*.03125*T/4)`。
- legacyは旧helperの三関数をAST抽出し、同じ保存frameと現在の固定runtimeへ適用する。
  historical runtime／当時の配列layout／旧actual hashを完全再現する主張はない。
- frame、Q0/Q1/Q2、T、U、input target、実gate parameterについてdtype/shape/layout/C-order全bytes/SHA256を保存する。
- recursive payloadは経路ごとに一回。元のoperation名・phase・params・wiresを含むpolicyを維持し、
  名前を除去した受理hashや新expectedを採用しない。
- actual非null MISMATCHと、TypeError/ValueErrorのactual=Noneを別々に保存できる。
  正常なfingerprint生成例外は観測値でありA/Bの勝敗ではない。
- `matrix_stage_bytes_equal`と`first_different_stage`は厳密bytes比較。
  自動名だけのpayload差は表示分類で、matrix差とは独立に取得する。

**表示上の注意**：`automatic_names_only`はpayload差だけの属性であり、全行列の同一性を意味しない。
追加controlではQ0 bytes差とpayloadの自動名差を同時に与え、`first_different_stage=Q_0`と
`automatic_names_only=true`の両方を保存した。
[独立fieldの証拠](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/additional_focused_run_01/matrix_and_name_independent_fields.json)。
この二fieldを合わせて読むことで、行列差を自動名差として取り違えずに診断できる。

## C：scope・資源

入口は新diagnostic runnerのみ。旧比較runner／full native adapterを呼ばない。
CD、E-r6 array化、controlled、native transpile body、T合成、Pauli費用、分子/NPZ/DF/GPU、
sampling/trajectory/LPは対象外。共通payloadに伴うUnitaryGate `_define`／内部QSDは許可された診断処理。
保存済み34件には8×8 synthetic controlで `_define` 1 / QSD Python entries 5 / native body 0の証拠がある。
今回はその実QSD controlを再計算せず、保存記録と対応sourceを確認した。
追加native entry probeはbody前に拒否し、観測を保存した後technical STOPへ伝播した。

| 固定実行条件 | 値 |
| --- | --- |
| workers / numeric threads | 1 / 1 |
| total CPU / worker CPU | 120 s / 90 s |
| aggregate RSS / worker RSS | 768 MiB / 512 MiB |
| output / log | 8 MiB / 1 MiB |
| terminal reserve | 256 KiB |
| review file / lossless chunk | 512 KiB / 256 KiB |
| total/phase wall cap | なし |
| path / payload / expm / input circuit | 各2、経路ごと一回 |
| retry | 0 |

上限・guard・契約を変更していない。Linux process-group監視は50 ms周期でCPU/RSS/output/logを検出し、
RLIMIT_CPU/FSIZEを併用する。RSS瞬間上限の無overshootは保証しない。
内部definition処理は実CPU/RSS会計へ含むが、C/Rustの個別呼出し数まで観測したとはしない。
available CPU32、MemAvailable54,360,036 kBをread-only確認。
name/counter順序を守るため1 worker/threads1を維持し、追加の並列化・speedup claim・GPU queryは行っていない。

## D：one-shot保護・失敗保存

既存34件のsource-bound保存証拠とlaunch sourceを確認：pending拒否、直接子／許可diff、
source/contract/manifest改変拒否、remote/branch/clean、旧marker不変、新namespace再利用拒否、
markerのsource/manifest/branch/remote改変拒否、exclusive child-start、terminal後拒否。
実際のregistered runnerを拒否probe目的でも起動していない。

追加mock runnerはtmp専用path、fake launch/runtime/supervisorを使用し、子workerを起動しない。
cap、KeyboardInterrupt、出力欠落、partial status、provenance不一致でtechnical inconclusive、
observations=null、retry0、mandatory STOPが保存された。
complete mockでもSTOPした。これはregistered成功を主張するものではない。

保存ストレージ自体が故障したmockではterminalの保存はできないが、exclusive markerは残る。
namespace存在によって再起動を拒否するため、terminal欠落を完了と解釈しない。
外側process statusと保存できたreceiptをread-only監査する必要がある。
実OS強制終了・OOMなど、processが一行も書けない障害でもterminal JSONを必ず保存できるとは主張しない。

## E：追加focused validationと発見したblocking issue

保存済み34 PASSはsource/hash bindingを確認し、再実行していない。
[今回の追加tests](../../../tests/tracks/algorithm_codesign/test_e_r3_identity_final_technical.py)のみ
**28 PASS / 0 FAIL / 0 ERROR / 0 SKIP**。
[実行監査](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/additional_focused_tests_v1.json)：
wall1.973270343 s、CPU(parent/reaped)1.885084 s、sampled aggregate peak RSS405.4609375 MiB。
全件一回、full suiteなし。pytest log/XMLとsource identitiesを保存した。
固定Python executable、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0の
code/shared-binary tree hashesは契約と一致した。
追加tests用guardはCPU60 s、aggregate512 MiB、child448 MiB、output8 MiB、log1 MiB、wall capなし。

| 追加検証 | 件数 | 結果 |
| --- | ---: | --- |
| 非null mismatch / None fallback | 3 | expected保持・一回attemptを確認 |
| matrix差 / 自動名差 / 両方 | 3 | 独立field確認 |
| UTF-8 lossless復元 / corruption / 保存失敗 | 8 | 復元一致、破損拒否、旧完全generation保持、失敗伝播 |
| 禁止native entry | 1 | body前拒否・technical STOP |
| observer lifecycle failure再現 | 2 | **不具合を確認した再現testとしてPASS** |
| fixed CPU/RSS/output/log cap breach mock | 4 | CapFailureを確認 |
| runner cap/interrupt/missing/partial/identity/complete mock | 6 | terminal/STOP分類確認 |
| terminal保存不能 | 1 | marker保持・完了と扱わない |

### ISSUE E3-ID-TECH-01：observer lifecycle failureを通常観測へ吸収

固定Sの[evaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8377c1d12439cabab3b97bcc1f57ac030a479785/src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/evaluation.py#L34)：
`DefinitionCalls.__enter__`の`sys.setprofile(self.trace)`と`__exit__`の復元は、
`observe_path`のpayload/hash用`except Exception`（lines92–99）に一緒に入る。
保存後のtechnical STOP判定（lines110–113）は「previousが非null」「native entry」の二条件だけ。
previous=Noneでhook設置／復元だけが失敗すると、この二条件には該当しない。

登録外2-mode/rank1、mock payloadとmock profile APIで次を確認した。

| 注入した故障 | payload実呼出し | pathの返り値 | profile状態 | 必要な扱い |
| --- | ---: | --- | --- | --- |
| 設置失敗 | 0 | actual=None、EXCEPTION、IDENTITY_OBSERVATION_SAVED | 設置なし | technical inconclusive |
| 復元失敗 | 1 | actual=None、EXCEPTION、IDENTITY_OBSERVATION_SAVED | mock observer残留 | technical inconclusive |

設置失敗ではpayloadを観測できていない。復元失敗では取得済みpayloadが通常例外処理でNoneへ置換される。
両者を「fingerprint生成自体の例外を完全保存した正常観測」と扱うのは不適切。
最後のlegacy経路でこのfailureが起きれば、次経路のpreexisting-profiler検出にも依存できない。
workerはobserve_pathが返ればpath完了として扱うので、現sourceには技術失敗を正常完了へ混入させる経路が残る。

[設置失敗の保存証拠](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/additional_focused_run_01/known_bug_profile_install.json)／
[復元失敗の保存証拠](../../../artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12/additional_focused_run_01/known_bug_profile_restore.json)。
**synthetic fault injectionであり、現在のruntimeや旧実行で実際に発生したと主張しない。**
通常条件の34 PASSや今回のその他PASSを取り消さないが、28 PASSだけでREADYとは判定できない。

最小修正案：新準備sourceで、observer設置・復元状態と例外をpayload生成例外から分離し、
観測prefixを保存したうえでobserver failureをworker失敗へ伝播する。
profile復元を検証し、payload attempt数も実際のentryと区別する。
正常なTypeError/ValueError等のfingerprint観測、hash policy、expected、行列式、path順序、capsは維持する。
今回この修正は採択・実装していない。固定Sのコード／contract／manifest／authorizationは不変。

## 未検証事項・次の担当

| 未検証事項 | 重要度と扱い |
| --- | --- |
| 登録E-r3のQ/T/U/recursive payloadと実actual | 今回禁止。原因判断の本証拠は別認可された診断後にのみ得る |
| 登録r3でCPU/RSS/output cap内に完了するか | off-domain testから保証しない。失敗時はretry0・technical inconclusive |
| 実kernel kill/RLIMIT/OOM時の完全terminal保存 | 静的guard＋mockの確認。保存不能でもmarker/namespaceを保護し外側statusを監査する |
| observer設置／復元失敗のproduction発生頻度 | 未評価。再現はmockであり発生頻度や旧停止原因を推定しない |
| A/Bの科学的優劣・新規性・旧不一致の確定原因 | 対象外。旧結果からは判断しない |

**次の担当はCodexの限定技術修正**。研究方針やfingerprint policyの採択をGPTへ毎回戻す必要はない。
ただし固定Sは保持し、修正する場合は別の準備sourceとして管理し、manifest・focused evidence・publicationを更新する。
科学契約・expected・比較条件・caps・受理guardの変更が必要になった場合は採択せずGPT判断へSTOPする。
修正後の技術検証が通れば`READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION`としてユーザーの明示承認へ進める。
今回の指示はその登録実行を承認せず、新Aも作らない。

本監査の完了後はmandatory STOP。新source修正・登録診断・科学的比較へ自動移行しない。
診断が別途承認され完了した後、実測証拠による原因／修正方針に科学判断が必要ならGPT側へ戻す。
