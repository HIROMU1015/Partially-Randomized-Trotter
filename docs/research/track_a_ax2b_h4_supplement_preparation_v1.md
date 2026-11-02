# Track A AX-2B：H4補完準備 v1

2026-10-10 JST。実装・synthetic検証・結果前計画の固定と公開。
`H4_SUPPLEMENT_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`。
**分子補完は未実行。CPU未割当・新grantなし・execution_plan_sealed=false。mandatory STOPを維持する。**

## 採用した判断と証拠の位置づけ

ユーザーが提示した[GPT独立科学レビュー](track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md)
§§10–14,19を受け、同一MP演算の再利用、S4 2 cellとexplicit 4群の独立補完、進捗保存を実装した。
レビュー原文はexact bytesで保存し、[対応記録](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/review_adoption_v1.json)にSHA-256を記載する。
レビューの科学的判断と、今回のCodexによる実装・準備を区別する。レビュー自体から実行認可は生じない。

[旧実行報告](track_a_ax2b_h4_limited_execution_v2.md)と
[一次保存監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json)を保持する。
旧runは`H4_LIMITED_STOP / PHASE_WALL_CAP:correctness`、6 correctness・12 MP・explicit 0群。
旧runのterminalをCOMPLETEへ変更せず、完了6 cellの全面再実行も予定しない。

## 固定identity・補完coverage

linear H4 / 1.00 Å / STO-3G / legacy DF rank12 / generation prefix 0,6,12。
8 system qubits、Nα=Nβ=2、sector dimension36、T=.8、指定saved binary64 H_DFと数学的に正規化した指定saved state。
q=1,4、δ=.8,.2。B2/B3はq=4、R=8、r=2、K=2/4/6を維持する。
energy estimationの精度、chemical accuracy、stateのground-state証明を今回の完成条件へ追加しない。

| 実行単位 | 新たに満たす義務 | その単位の完了条件 |
|---|---|---|
| `S4_MP` | H4_B1_S4_q1/q4 | correctness 2件、MP80/120計4件、全stage・信号・参照・precision比較 |
| `EVENT_CONTROL` | H4_B2_K2/H4_B3_K6 × Taylor order0/2 | explicit 4群、ordinary/directional両branch・X/Y測定・signed algebraic event比較 |

両単位はそれぞれ、同じ入力確認、36列の独立occupation照合、旧全8 cellのcoverage一致、
179 primitive/time組×3 probe=537作用を先に実施する。
この安価な前提検査の重複は意図的であり、旧6 cellのcorrectness/MP再実行ではない。
EVENT_CONTROLはS4 MP完了に依存しない。両単位に別のmanifest・grant・exclusive outputが必要で、順序は独立。
現在の[義務台帳](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/coverage_obligations_v1.json)は補完全件を`NOT_EXECUTED`とする。
将来の統合では旧6 cell＋新2 cellという複数runの出典を各record/hashに結び、一回8/8の実行と記載しない。

## Sourceと意味論の維持

新source commit：`67aa6bb54dd5385eb3c56def1b6052da12643447`。
[source freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/source_freeze_v1.json)は
実行closure180件・validation2件をlocal/Git blob SHA-256で照合し、旧173 science sourceのbytes一致を確認した。
旧science commit `b228f2307f5fea77f066ee13e11ba6d2d8b9bea7`、seal `d2b1511cd87ee51e2a08d0f7c5025e55e920745d`、
旧grant `a699a74dbf7a10a4e43c32ed6b6d3cb9d402a708`、結果 `a87cf25548a3b93262027ce780317872d7c4e883`は不変。
旧grantは消費済みで、新runnerの別schemaは受理しない。

| Source | 役割 |
|---|---|
| [MP oracle v3](../../src/trottertracks/resource_applicability/ax2b_stage_validation_v3.py) | 旧v2のindependent occupation/forward Taylor/整数比lift・stage列を維持 |
| [MP cache](../../src/trottertracks/resource_applicability/ax2b_mp_cache_v1.py) | 各cell・各dps内部だけの演算子再利用 |
| [supplement port](../../src/trottertracks/resource_applicability/ax2b_h4_supplement_port_v1.py) | 旧v3 setup/native作用を継承、対象とphaseを分離 |
| [launch gate](../../src/trottertracks/resource_applicability/ax2b_supplement_launch_v1.py) | 新grant/source/input/env/CPU/output/全coverageを結果前に結合 |
| [records](../../src/trottertracks/resource_applicability/ax2b_supplement_records_v1.py) | 原子的exclusive JSON、bounded進捗 |
| [watchdog](../../src/trottertracks/resource_applicability/ax2b_supplement_watchdog_v1.py) | 独立phase/total/log/output上限、強制STOP時の最後の保存snapshot |
| [runner](../../scripts/resource_applicability/run_track_a_ax2b_h4_supplement_v1.py) | default metadata only、別grantの前にnumerical importsをしない |
| [tests](../../tests/tracks/resource_applicability/test_ax2b_h4_supplement_v1.py) | 合成MP一致・cache分離・選択・grant・記録・STOP |

Cache identityは保存係数/state、sector/basis順序、cell/formula、exact binary64時刻・符号、λ/identity/K、dps/backendを含む。
固定sourceはlaunch freezeで拘束する。MP working precisionもget時に確認し、別precisionの値を流用しない。
primitive identityとtime.hex()でdeterministic演算を識別する。rawにはcorrected演算子のcopyを渡してbで割り、cache値を変更しない。
native prepared diagonalizationをMP側へ流用せず、occupation構成をcell・precisionごとに独立に生成する。
初期state以外の正規化、負時間、scalar phase、stage順序、全state記録、raw/corrected/exact-tailの役割を維持する。

Cacheは最大128 entry / accounted heap64MiB / exp生成64 / polynomial生成4 / lookup4096、MP stage記録4096を各cell・precisionに適用する。
終了時にcell-local cacheへの参照を失い、次cell/runへ共有しない。stored matrixと取得値はcopyする。
heap会計はPython objectの実装予算でありRSSの厳密上界ではない。factory workspace・一時copyはworkerのAS上限で制限する。
logical stage作用、生成attempted/completed、hit/miss、成功したpolynomial matrix productsは別指標とする。
Frobenius normを数値誤差の実際の増幅率と認定しない。

## 検証結果と限界

[local synthetic監査](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/synthetic_test_audit_v1.json)：
**122 passed**（新73、既存coverage metadata49）。
人工2-orbital/1-particle/dimension2のfixtureで、S2/S4 q1/q4、B0、B2 K2/K4、B3 K6、
正負T、MP80/120の28比較を実施した。
追加診断`oracle_work`を除き、全decimal信号・誤差分解・norm・全stage state/timeが旧v2と完全一致した。
指数生成回数が減り、作用列は維持された。input/precision/backend/符号・mutation・cap・failure記録も検査した。
local testsをimmutable CIまたは外部独立再現と呼ばない。

[静的source監査](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/static_source_audit_v1.json)では、
copied primitive/correctness/control/event各bodyが、進捗・budget・routingの指定差分を除いて旧sourceとAST一致した。
これは分子実行や全回路の同値検証ではない。
[静的budget監査](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/static_budget_audit_v1.json)では、
S4各cell・各precisionのPF指数73/292論理作用に対し、unique生成26＋reference1となる。
これはschedule由来の演算数であり、wall speedupや完走時間の測定ではない。
旧H4 36次元でのcache前後フル再実行は行っていない。

## 結果前の古典計算予算と保存先

| 予算 | S4_MP | EVENT_CONTROL |
|---|---:|---:|
| input_reference phase | 300秒 | 300秒 |
| primitive_validation phase | 300秒 | 300秒 |
| validation phase | 1800秒 | 300秒 |
| total（起動/import含む） | 2400秒 | 900秒 |
| worker / BLAS threads | 1 / 1 | 1 / 1 |
| AS / output / log | 8GiB / 128MiB / 64KiB | 同左 |
| reference matvec / primitive action | 36 / 537 | 36 / 537 |
| control state action | 0 | 100 |
| progress / diagnostics | 1024 / 1536 | 同左 |
| trajectory / occurrence sampling / compile | 0 / 0 / 0 | 0 / 0 / 0 |

以上は提案された固定上限であり、使用CPUの割当や実行承認ではない。
量子回路の資源C、量子shot N、総資源Gとは別の、評価のための古典計算上限である。
EVENT_CONTROLの代表回路build/state actionは将来の別認可後だけ行う。transpile/compileしない。
N/G=null、accuracy_eligibility=UNDETERMINED、numerical_allowance_certified=falseを維持する。

未来の保存先は`artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/`配下の
`s4_mp_launch_v1/`と`event_control_launch_v1/`。**両科学outputは未作成**で、旧launch_v2と分離する。
[S4 prepared manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/prepared_s4_mp_v1.json)と
[event prepared manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/prepared_event_control_v1.json)にrepository pathとhashを固定した。
実行時はCPU/resource確認後のmetadata sealと、各unitへの別の明示的ユーザーgrant・manifest digest・grant SHA-256が必要。
現在はunsealedであり、旧CPU3や消費済みgrantを自動継承しない。retry/resumeなし。

## 進捗・STOP・未解決事項

JSONは一時ファイル完成後のhard-linkでexclusiveに公開し、既存ファイルを置換しない。
進捗は演算開始/完了、MP outer完了、record保存等の境界で記録する。高価な一つのexpm内部では定周期の更新を保証しない。
最後のcell/path/dps/stage、last_completed_record、各attempted/completed、cache統計、elapsed、観測peak RSSを保存する。
primitive actionのcounterは各probeの開始/完了を更新し、snapshotはprimitive/time組の境界で保存する。
強制終了後は最後のsnapshot時点の値のみを引用し、その後のcounterや未観測lifetime peakを補わない。
診断capもSTOP条件であり、成功させるためのstage/probe削減・精度低下・上限の結果後変更を行わない。

数値不一致・precision不安定・target/PF/event/independence変更・必要coverageを保てない場合は、GPTへ早期に戻す。
合成一致は経験的検証基盤であり、総u certificate、full molecular event mean、fresh-shot IID証明、rare-event期待cost、
PR資源winner、RQ-P2の達成を認定しない。RQ-R主軸・RQ-P1補助・RQ-P2未達を維持する。

H6は[既存準備契約](track_a_ax2b_h6_pilot_preparation_contract_v2.md)のtol-only / 7 cell / 36 wrapper方針を維持する。
最新の[backend v3](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)には
H6 snapshot loader、400-sector reference、4096-vector native action、sampling/cost経路が存在する。
旧準備文書やmodule docstringの「backend未実装」は当時の記録であり、現在の機能一覧へそのまま引用しない。
未解決は別入力生成のscope/budget/grant、実入力・actual rank/sector/DF provenance、actual probe/instruction bounds、
source/env/CPUとH6 launch grantの固定、実分子での性能・正しさである。今回これらを生成・検証・認可していない。
別系列server run10のsource・branch・job・比較条件は変更せず、本補完の証拠へ統合しない。

次は明示指示後にCPUと各unitのgrantを固定し、H4補完を一回実行・保存監査してSTOPする。
その後のH6入力・技術pilotも別指示を要する。主要GPTレビューはH4補完＋認可されたH6技術結果で本検証GO/STOPを判断する節目。
現段階は準備完了として停止し、科学launchを自動開始しない。
