# Track A AX-2B：H4メモリ処理修正・合成検証・実行前固定v4

公開時のリンク補修（2026-10-10）：未公開sourceへの参照は[公開依存関係](track_a_ax2b_gpt_review_index_v1.md)。原文と補修一覧を保存し、科学的主張・数値は変更していない。

2026-10-09 JST。**`AX2B_H4_MEMORY_V4_PREPARED_SCIENCE_NOT_AUTHORIZED`**。
利用者の継続指示に基づき、前回の[H4部分結果・MemoryError停止](track_a_ax2b_h4_pilot_result_v1.md)に対する
実装修正と合成検証、v4の準備manifest作成までを行った。
分子signal評価・trajectory sampling・回路compile・H4再実行・H6/H8/GPU実行は今回0件。
前回pilotの停止原因を特定した結果でも、分子MemoryErrorの解消を実証した結果でもない。

## 1. 保存方針と実装の置き場所

旧source collectorは`src/trotterlib`と`src/trottertracks`のPython一覧を含む。
新moduleの追加でも旧manifestとの照合が変わるため、別worktreeへ既存状態をbyte一致でコピーし、
新しいv4 sourceだけを追加した。旧native・science・contract・runner・artifactは編集していない。

- 作業branch：`track-a-ax2b-h4-memory-v4-20261009`
- 作業worktree：`.worktrees/track-a-ax2b-h4-memory-v4-20261009`
- 旧v3と部分結果の正本：`.worktrees/track-a-ax2a-preparation-20261009`
- base HEAD：`b2e1bf65e21893b6c617223b42313623d3186f12`。今回commit/pushなし。

旧v3のsource集合とinventoryの照合は**旧worktree上**で行う。
v4側にコピーした旧inventoryは当時の履歴であり、現在のv4 source集合への照合記録ではない。
元dirty worktree、利用者のroot review、M1〜PM-2/AX-1b、原稿、Track Bも保持する。

| 新しいファイル | 役割 |
|---|---|
| [ax2b_stream_fingerprint_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | 旧canonical JSONと同じbytesを順次hashへ渡す |
| [ax2b_native_df_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | 旧native loweringのbodyを保ち、fingerprintのimport先だけを変更 |
| [ax2b_diagnostics_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | 段階・current RSS/VmSize・累積peak RSS・bounded traceback |
| [ax2b_h4_science_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | 1 cell-replicaずつ処理し回路を解放する将来pilot |
| [ax2b_h4_contract_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | v4 schema/source/環境/入力へ束縛するstdlib契約 |
| [run_track_a_ax2b_h4_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | defaultはmetadataのみ。科学importは認可・resource enforcement後 |
| [verify_track_a_ax2b_h4_pilot_v4.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) | 将来の保存JSON/hash、partial/complete、診断記録をscience再計算なしで照合 |

## 2. メモリ処理の変更

旧cost処理は完了した6 cell-replicaの両control circuitを保持したまま、最後の四次q4を準備した。
v4は登録順を保ち、1 cell-replicaのordinary/directional × cosine/sineを処理してから次へ進む。
保持するnative evolutionは現在groupの両controlだけとし、wrapper/transpiled circuitは各task終了後に解放する。
random requestとlegacy replay oracleもcontrol比較後に解放する。
groupから返すのは小さなtask/整数metricsで、group間に回路をcacheしない。
これはPython参照の寿命の管理であり、RSSが即座に減るという保証ではない。

fingerprintは旧`qiskit_recursive_numeric_circuit_v2`のJSON内容・sort順・escape・numeric payloadを保つ。
recursive payload全体、JSON文字列全体、encoded bytes全体を作らず、最大64 KiBのhash bufferを使う。
sub-definition hashへの置換、scalar/global phaseの省略、parameter/control stateの丸めを行っていない。
旧private helperのnumeric leafとcondition規則を再利用し、symbolic/layout/calibration/recursive definition等の拒否も維持する。
旧方式が拒否する古典register条件を、この修正で対応済みに変更していない。

**64 KiBはhash bufferの上限で、process memoryの上限ではない。**
Qiskitがdefinitionを構築・コピーするメモリ、native回路、leaf array payload、compilerのメモリは別に必要である。
固定200 instructionの合成回路では、Python追跡対象allocationのpeakが旧方式の1/4未満になるgateを通過した。
これは当該toyの`tracemalloc`比較で、H4のpeak RSSやH6/H8の必要メモリへ外挿しない。

## 3. 停止時の診断

native build、numeric fingerprint、legacy replay build、control probe、Hadamard wrapper build、
transpile/metricsをbegin/endで記録する。登録3 phaseのwatchdog timerは変更せず、groupごとに延長しない。
各記録にcell/replica/control/axis/task、elapsed、`VmRSS/VmSize/VmHWM`、累積peak RSSを保存する。
診断は最大1,024 records、aggregate output capの内側でwriteする。

exceptionでは最も深いstage contextを保持する。
1 MiBの緊急用reserveを解放し、最大32 traceback framesのfile/function/lineだけを保存する。
localsとsource textは保存せず、unwound frameをclearして失敗した回路への参照を減らす。
診断write自体が失敗しても、元のexception type/messageを別exceptionで置換しない。
parentは終端が欠けてもSTOPに分類する。
snapshotは診断時点の値であり、失敗allocationの正確なpeakを保証しない。

## 4. 科学条件を変更しない

v3とv4で、schemaとimplementation metadata以外のplan fieldsが一致することを検査した。
対象はlinear H4、1.00 Å、STO-3G、legacy DF rank12、generation-prefix、T=0.8、
既存normalized state、N=4/Nα=Nβ=2の36-dimensional sector。
q1/q4（δ0.8/0.2）、B0 prefix6、B1 prefix12、B2 prefix6/R8/K2,K4、B3 prefix0/R8/K6、
global四次B1 prefix12/q1,q4の登録8 correctness cellを保つ。
prefixはDF block数であり、prefix0でもone-bodyは決定論的に残す。

28 wrapper、4 random trajectory/16 occurrence、seed/event共有、ordinary/directional、二軸、
compiler `rz,sx,x,cx`/optimization1/seed17/指定backend・couplingなしを保つ。
測定付きfull wrapperはstate preparationを除く。sampling自由度や探索gridを追加しない。
CPU1/thread1、address space8 GiB、phase900秒/total2700秒、output512 MiB、
instruction/matvec/validation/compile countersの上限も同じ。
S4 q4を外すことでstrong baselineの欠落を隠す変更はしていない。

setup、correctness、trajectory等15 function/methodのASTが旧scienceと一致し、
native loweringもfingerprint import以外のbodyが一致することを静的に照合した。
総数値allowanceは未認定のまま、accuracyはUNDETERMINED。
shot・total cost・winner・chemical energy accuracy・H6へのGOを主張しない。

## 5. 合成検証と証拠

**155 local tests passed、fail/error/skip0**（154件の一組＋後追加のstdlib実OOM検査1件）。
旧toy検証をv4 importへ接続した回帰検査を含む。immutable CIや独立再現の証拠ではない。

- [stream tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：canonical bytes/hash完全一致、nested definition/control state、phase/parameter/register変化、旧拒否条件、toy memory。
- [native tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：独立Fock参照、global二次/四次、正負時間、scalar/ancilla branch、明示event replay、wrapper測定。
- [runner tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：28-task plumbing、weakrefで回路解放、最後のq4で失敗するpartial shape、診断失敗・上限、旧authorization拒否。
- [saved tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：fabricated JSONだけでpartial/complete、改変・欠落・診断記録の整合性を検査。

実OOM検査はstdlib processのRLIMIT_AS64 MiBに対して128 MiBのbytearrayを要求する固定fixture。
科学libraryをimportせず、MemoryErrorのstage/tracebackとSTOPを保存した。
実分子のnative/compile allocationや8 GiB cap下での完走を検証したものではない。
compile/samplingのplumbingはダミーで、分子snapshotの数値配列はloadしていない。

[artifact入口](../../artifacts/resource_applicability/track_a_ax2b_h4_memory_preparation/2026-10-09/)に
JUnit/stdout、metadata-only manifest、static audit、source freeze、未認可draft、保護照合を収録する。
[別inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_memory_preparation/2026-10-09/memory_preparation_inventory_v4.json)
へ追加し、旧科学manifestと準備/pilot inventoriesは編集しない。

## 6. 固定値と次の一段階

science sourceは153 byte hashesで固定した。manifest digestは
`de1a9624a683f4de49d5eac83755c4676040ed74b25f64d867680a5ec38e3006`。
科学sourceのcommit固定ではなく、隔離branchの未commit byte固定である。
`science_authorized=false`、`launch_allowed=false`、`assigned_cpu=null`。
旧v3 authorizationは新schema/source/manifestと一致せず、v4に使い回せない。

次はこのv4 sourceで同じH4範囲を一回実行するかの判断である。
実行する場合はv4 schemaの新しい認可、実CPU割当、hashを束縛した新規exclusive outputを用い、
input/reference→8 correctness→28 wrapperを再度一組として扱う。
旧partial outputのresume、欠落4件だけの補完、cap拡大、H6/H8への移行はこの準備に含めない。
完了/停止どちらでも保存監査後にSTOPする。今回の細かな実装修正に形式的GPT reviewは要求しない。
分子でのmemory改善と全28件完了は未検証であり、数値allowance・研究判断も別途残る。
