# Track A AX-2B：H4限定検証・H6 backend接続の準備 v2

2026-10-10 JST。[独立科学レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §21と利用者の継続指示に基づくsource/synthetic準備。
[前回追補 v1](track_a_ax2b_post_independent_review_amendment_v1.md)と[H6準備契約 v2](track_a_ax2b_h6_pilot_preparation_contract_v2.md)の未接続portを追加する。旧文書は履歴として保持。
`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。入力生成・科学launchの認可は作成しない。

## 接続したsource

| source | 接続と制限 |
|---|---|
| [stage validation](../../src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py) | native callback/既存Hornerのraw・corrected各stageを記録。独立occupation構成と80/120桁のTaylor前進和・MP expm・inner product。MPはsector≤64のH4用 |
| [molecular ports](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py) | 保存H4読込と専用H6 variable-rank NPZ読込、独立column、reference、全実時間probe、7 cell/36 wrapper・4 trajectory/8 occurrence、群解放。入力生成は含まない |
| [bound launch](../../src/trottertracks/resource_applicability/ax2b_bound_launch_v2.py) | metadata案、固定scope、source commit/blob/local SHA、環境、入力政策、CPU、actual coverage、別grant、exclusive outputを数値import前に照合 |
| [watchdog](../../src/trottertracks/resource_applicability/ax2b_launch_watchdog_v2.py) | 一workerのphase/total wall、log/output、process group停止、欠測terminal・部分完了のSTOP。完了済phaseのtimestamp超過も検査 |
| [runner](../../scripts/resource_applicability/run_track_a_ax2b_bound_v2.py) | defaultはmetadata JSONのみ。`--execute`は別のpinned authorizationとsealed manifestが必要。今回のdraftでは起動できない |
| [tests](../../tests/tracks/resource_applicability/test_ax2b_bound_ports_v2.py) | 小型合成occupation/MP、mock backend/compile/request、dummy process、metadata/source gate。実分子I/Oとsampling・回路構築・compileを拒否 |

共有trotterlib、旧H4 v5 source/freeze/results、前回の172項目preparation freezeを変更しない。
H4 controllerのfull-space dense setupをH6へ転用していない。旧primitive certificate/basis bridgeなどは関数単位で再利用する。

## H4-N/A/E/Mの実行経路

対象はlinear H4 1.00 Å/STO-3G/legacy DF rank12/generation-prefix、保存指定vectorを数学的に正規化したstate、36-dimensional spin sector、T=.8、旧8 cell。
source接続のみであり、この入力を今回loadして新signalを求めてはいない。

H4-N：既存sector matvecから単一sector referenceを作り、独立occupation ladderの全columnと照合する。
MPは保存binary64係数と実行binary64時刻を整数比でliftする。MPの状態は**元の保存vector**から正規化し、binary64側の正規化丸めへ置換しない。
MP内のexpm、Taylor和、全stage、inner productをdecimal stringsとして保存。80/120桁差・binary64 reference差を別記録する。
target HermiticityがMP検査を満たさない場合は無断HermitizationせずSTOPする。

H4-A：merged global PF、unmerged directional、negative S4、half/undoの実時間集合を展開する。
全unique primitive timeに対しsaved stateとfirst/last sector columnの3 probesを登録し、structural sector証明と併記する。
全column×全time検査とは主張しない。actual coverageがcapを超える場合は間引かずSTOP。
native/Hornerと独立MP前進Taylorを各物理stage endpointで対応させ、state差・norm・raw/corrected/logBを保存する。
Horner内部matvecは別stage。rawはbによる除算後のendpointとMP meanを比較する。有限平均の中間stateを正規化しない。
MPのB0 discard/PF、random finite/outer PFのsigned分解は別recordで、既存科学結果を更新しない。

H4-E：既存toy完全列挙の検証に加え、future分子経路ではB2 K2/B3 K6からgeneration順のfirst nonidentity componentを選び、order0/2各1 explicit eventを検査する。
samplingせず、1 microstepの代表eventを明示構成する。negative event phase、signed involution、prepared basis hash、little-endian registerを、局所matrixの代数作用とnative controlled作用で照合する。
E専用stepのδ/rに伴うprimitive half/undo時間もactual coverageへ加え、登録8 cellのPF時刻だけでcoverageを済ませない。
ordinary/directionalの両branch、cosine/sineの実wrapper stateも検査する。compileは0。
これは分子event母集団の完全列挙や独立orbital decompositionの証明ではない。
future `FreshShotRequests`は各shotでwhole-trajectory factoryを呼び直す。cost用2 trajectoryの固定再利用を測定平均へ持ち込まない。seedの相違だけから統計的独立性を証明しない。
今回quantum shotは0、実N/Gはnull。

H4-M：前回のsynthetic u-aware会計を再検査する。今回の経験的差をu_boundへ昇格せず、総allowance未認定・accuracy UNDETERMINEDを維持する。

H4の**未割当proposal**：reference900秒、stage1800秒、explicit estimator300秒、total3000秒、AS8GiB、CPU/BLAS/worker1、output512MiB/log64KiB、diagnostics1024、primitive2000/control200。
前回2700秒案へE専用300秒を明示追加した。資源割当・実行許可ではない。旧v5上限も変更しない。
tail896/cell、deterministic100000/cell、stage record4096/path-function、instruction1e6、compile/sampling0。

## H6の専用backendと入力契約

H6はlinear 1.00 Å/STO-3G、12 system qubits、Nα=Nβ=3/sector400、T=.8、actual L≥2、p=(L+1)//2。
7 cell/36 measured wrappers、4 cost trajectories/8 outer occurrences、n=2工程用途、symmetric_directional primary/ordinary paired sensitivityは[前回契約](track_a_ax2b_h6_pilot_preparation_contract_v2.md)を維持。
H6全露出はdevelopment。H8/GPU、本比較、winner/期待cost母平均認定は対象外。

H6 NPZは旧H4の8 keysに対応する専用variable-rank layoutを持つ。
layoutのrank上限144は12 spin orbitalの係数配列に対する実装上限であり、rank探索やrank固定政策ではない。
zip keys/expanded bytes/NPY header・shape・dtypeをarray load前に検査し、内部Hamiltonian/state/sector hashesとadapterのpost-Hermitization bytesを照合する。
top metadataは`model=linear_H6, geometry_angstrom=1.0, basis=sto-3g`とし、`input_generation`に別生成source commitとauthorization SHAを要求する。
`hamiltonian_metadata`は[既存tol-only adapter](../../src/trottertracks/resource_applicability/ax2b_h6_input.py)のactual kwargs/L/correction/Hermitization記録を持つ。
`final_rank`、config fallback、追加cutoffは拒否。入力生成の予算・integral取得・state solve・保存receiptは別scopeであり、今回入力は作成しない。
[bounded solver](../../src/trottertracks/resource_applicability/ax2b_h6_controller.py)は既存準備機能。pilot runnerはstateをsolveせず保存stateを使用する。

referenceは一枚400×400 sector matrixをbounded matvecで構築し、独立occupation columnと比較後signalを求めて解放する。
native Gaussian中間は4096 full-vectorで作用させ、complete primitive後だけleakage検査とsector投影を行う。
独立binary64 primitive oracleは一枚を置換し、random cellのsector tail matrixと併用する。full4096×4096 fragment行列・固有vector cacheは存在しない。
400×400 complex128一枚2,560,000 bytes、4096-vector65,536 bytesという配列算術と、回路/DAG/fingerprint/RAM/wallの実負荷は別である。
H6分子のmemory・wall・正しさを今回実証してはいない。oracleの反復expmもphase wall capの対象。

actual rankから全unique時間のprobe数と、native block instruction bound・最大Taylor orderに基づくtail structural upper boundsをsourceへ接続した。
sampling/build前にcap検査し、runtime actual boundsがsealed coverageと異なればSTOPする。
instruction数は展開native gate数・compiler RAM・物理資源の上界とは呼ばない。
H6 capsは前回の7200秒、AS8GiB、compile36、primitive2000/control256、reference20000等の未割当proposalを維持。

## 固定・検査・残る事項

local synthetic/mock testsのreceiptとsource freezeは[別inventory](../../artifacts/resource_applicability/track_a_ax2b_bound_ports_preparation/2026-10-10/preparation_inventory_v2.json)へ収録する。
mockの400-dimensional reference、7 cell作用、36 wrapper・partial compile failure、H4形式8 cell×80/120桁、phase/time/logのSTOPは工程検証で、分子科学結果やCIではない。
途中でH6 `target.T`→backend渡し忘れを検出し、同じT=.8を明示接続した。testのphase markerをcell recordへ数えたglobも修正した。

**source固定と科学実行planのsealは別**。今回source/環境receiptをcommitへ結合するが、actual input/coverage、CPU/RAM割当、別authorizationは未固定nullである。
H6入力生成budgetはnullのまま。H4の保存input identityは既知だが、実入力に基づくnative準備とactual instruction receiptの固定は未実施。
runnerはこの未seal draftを拒否する。source完成をlaunch-ready・GO認定へ読み替えない。
次は対象・予算を明示した指示でH4の入力/coverage/preflightを固定し、別launch認可で限定検証する。H6は別入力生成・receipt固定後にpilot planをsealする。
通常のsource修正はCodex範囲。研究意味論の変更・正しさの未解決矛盾はGPTへ戻す。
今回は新分子計算/NPZ array load/signal/trajectory/circuit build/transpile/compile0。mandatory STOP。
