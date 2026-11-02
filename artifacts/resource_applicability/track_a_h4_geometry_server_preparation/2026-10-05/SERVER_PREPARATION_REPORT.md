# Track A H4 geometry server preparation / contract draft v0

`SERVER_NATIVE_ENV_REQUIRES_SOURCE_PORT`

2026-10-06 JST完了（2026-10-05 handoff準備directoryを継続使用）。handoff `48f9ac3b756bcf572aeb7c594a0c88791a458725` から独立branch/worktreeを作成し、環境記録・静的監査・純synthetic benchmark・契約/schema/zero-compute草案を作成した。準備statusは `SERVER_PREPARATION_COMPLETE_AWAITING_CONTRACT_REVIEW`。science_execution_authorized=false、geometry_frozen=false、sealed=false。準備完了時点では本計算・science source実装・commit/pushを行わず、mandatory STOPした。その後の利用者の明示依頼は、この準備bundleのみの専用branchへのcommit/non-force pushに限り、本計算・source portは引き続き未着手。

## 既存worktreeと共有環境

作業root：`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-server-prep-20261005`。branch：`track-a-h4-geometry-resource-server-prep-20261005`。最初のstatusはclean。
Git tree内にNPZが4件あるため、`git worktree add --no-checkout`後にworktree専用sparse patternsとcommand-scoped `-c core.sparseCheckout=true -c core.sparseCheckoutCone=false`でcheckoutした。NPZ/NPY/pickle/runtime/checkpointは除外し、内容・分子hashへアクセスしていない。共有Git configは変更していない。Git操作はfetch、新branch/worktree登録のみ。

既存mainと既存未追跡 `docs/gpu_execution_environment.md` は保持。venv、package、OS、CUDA/driver、shell/Qiskit共有設定、他ユーザーのprocess・priority・affinityは変更0。専用一時directory内のQiskit設定を自分のprocessにだけ渡し、自分のdriver/workerをnice19にした。

## 環境と比較scope

Python：`/home/AbeHiromu/venvs/trotter-common/bin/python`、3.12.3。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、rustworkx0.17.1、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。CPU：AMD EPYC 7742 64-Core Processor、2 sockets、128 physical/allowed logical cores、NUMA 8。
RAM total 1007.66 GiB、available 980.19 GiB、swap 15.98 GiB。開始load1/5/15は [0.248046875, 1.35986328125, 3.48974609375]。自身のcgroup CPU/memory quotaは観測時max、scheduler環境のjob割当指定はなし。これは観測値であり共有資源の予約ではない。詳細は[environment inventory](environment_inventory_v0.json)。

metadataを優先して依存version・installer・WHEEL・RECORD digestを保存した。元wheel archiveや全installed binaryの独立照合は未実施で、wheel/archive identityはnull。GPU packageはimportしていない。NumPy/QiskitだけCPU importし、BLAS実装・全transpile defaults・plugin entrypointsを記録した。

旧PM-1契約のPython3.11.0rc1 guardと現況3.12.3は一致しない。旧guardを削除せず、次段で新Track A source/契約に分離する。247 sourceをAST parseして構文failure0。legacy Almost_optimal_grouping.pyにはPython3.12のinvalid-escape SyntaxWarningが5件あるが、旧fileを修正していない。構文検査はPySCF/DFのruntimeや数学的同値性の検証ではない。

Qiskitは旧記録と同じversion1.3.0だが、完全なcompiler/dependency/source identityの同一性は未確立。新campaign compiler案はbasis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing/targetなし、approximation_degree=1.0、default synthesis、custom pass/pluginなし、num_processes=1。全default/plugin情報とprivate config digestを[plan](zero_compute_plan_draft_v0.json)へ結合した。
Qiskitのmultiprocessing条件とnum_processesは[公式1.3 compiler文書](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/compiler)およびinstalled utils/parallel.py・user_config.pyから確認した。BLAS/OMP/MKL/NumExpr各1、QISKIT_PARALLEL=false、QISKIT_NUM_PROCS=1、RAYON_NUM_THREADS=1。process限定の設定で共有環境は変更しない。

## 保存証拠identity

指定保存4 JSONの内容hashとbase/handoff blob、候補inventory/input identity二件のbyte identityを照合し、全PASS。[static audit](static_audit_v0.json)にexact path/hashを保存した。218 template集合は保存inventoryと完全一致し、B0=20、B1=4、B2=145、B3=49。r64二件は予め含み、旧selectorを再実行しない。旧candidate fingerprintは新geometryへコピーしない。

## 科学データを使わないCPU benchmark

研究DF係数・candidate/trajectory seedを使わない9 qubit/1 classical-bit fixture。ancillaは8、systemは0..7。固定synthetic seedからcontrolled XX+YY Givens network、対角回転、inverse networkを作り、cosine/sine Hadamard wrapperでcompileした。小/中/大は16/64/128 layers、各10 task。全worker条件で同じ30 task、結果を見た規模変更・選抜なし。

| workers | wall s | tasks/min | speedup vs 1 | efficiency | peak own pool RSS MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 150.15 | 11.99 | 1.00 | 1.000 | 242.9 |
| 6 | 29.61 | 60.78 | 5.07 | 0.845 | 1287.4 |
| 12 | 16.73 | 107.58 | 8.97 | 0.748 | 2334.8 |
| 16 | 15.62 | 115.25 | 9.61 | 0.601 | 2776.0 |

固定compiled sizeの実測範囲：small 23814–23815、medium 95238–95239、large 190470–190471。旧約12k/45k/90kは参考で、規模を合わせる追加transpileはしていない。build/transpile wallはtask別に別記録し、出力gate数・depth・worker RSS/thread数を[benchmark result](synthetic_fixture/benchmark_result_v0.json)、scale別集計を[summary](preparation_summary_v0.json)へ保存した。

benchmark120＋小型意味論検査4＋別に事前定義した全operator照合4＝128 actual synthetic transpile、synthetic wrapper record 128、science wrapper0。benchmark全wall 212.14秒、failure/OOM0、全worker条件でgate metrics一致。小型3-qubit wrapperは非零global phase・diag(I,U)・bit0→+1/bit1→−1・cosine/sine・compile前後operator/axis意味論を4件確認し、最大axis残差は1e-12未満。追加全operator照合4件は[別定義](synthetic_fixture/operator_check_definition_v0.json)を実行前に保存し、初期fixture cap124を守りながら利用者総cap128内で実行した。compile前後の全operator最大差は1.49e-14、global phase込みで一致した。既存科学tests・全repository testsは実行0。

自分のpool以外の並列を抑制し、1 process AS4 GiB/CPU600秒、総wall30分、共有load32/RAM64 GiB/CPU-pressure5%の停止guardを置いた。guard変更やfailureなら自分のworkerだけ停止しretryなし。実測はサーバー内部scalingで、同じfixtureをローカルでは実行していないためhost間速度倍率や実H4 ETAは示さない。

推奨workerは **12**。最大throughputの90%以上を満たす最小の測定worker数という共有CPU節約の運用提案であり、科学的受理gateではない。productionのworker capはnullのままで、自動認可しない。science memory profileも未測定のため、syntheticの4 GiB上限を科学計算へそのまま流用しない。future launch直前に共有負荷と利用者の許容資源を再確認する。

fixture/task fingerprint：`2b3fa5ab1b67093be38ebb907f89830ef0f4dd096830a54521bcc189af2fc8d7`。source SHA-256は[task definitions](synthetic_fixture/task_definitions_v0.json)に保存。再利用可能source/config/definitions/resultsを新準備directoryだけへ保存した。

## 結果前契約の草案

H4 linear、STO-3G、8 system qubits、requested/actual DF rank12、T=0.8、二次DF-prefix PF、canonical finite-RTE。L_DはB0=3/4/5/6/9、B1=12、B2=3/6/9、B3=0。q=1/2/4/8、delta=0.8/0.4/0.2/0.1、B2/B3 r=1/2/4/8/16/32・K=2/4に固定r64二件。各geometryで同じ218 template。

新規6距離0.70/0.80/0.90/1.10/1.40/1.60 Aは未承認案。geometry_frozen=falseを保つ。1 geometry random194×32×2=12,416＋baseline24×2=48＝12,464 wrapper。6点74,784、8点99,712は上限案であり正式capではない。保存1.30 Aは固定5構成で完全gridに含めない。8点、1.30 A全候補化、追加96、高次PF、H6/H8/H12、energy/RPE、Track Bは自動追加しない。

新snapshotはgeometry座標・単位、SCF収束、rank実数、fragment order/ties、物理sector、phase規約、残差・正規化、生成source/dependencyを新provenanceで固定する。参照はそのDF Hamiltonianの同じsector内の最低固有状態で、exact untruncated chemistry ground energyとは区別する。rank不足/不収束をpadding・別rank・別geometry/状態で救済しない。旧snapshot hash再現を仮定せず、旧状態loadによる環境比較も行わない。

seed/master seed、生成source、solver/threshold/DF ordering detailsは未固定。master seed・新snapshot/source hashはnull。checkpoint keyはgeometry/H/DF/state/candidate/axis/seed/index/compiler/environment/source/wrapper semanticsを全て結合する。logical wrapper ledgerと実transpile invocation/cache hitを分け、atomic complete recordだけ再利用する。同じcandidate・axisの厳密numerical full-wrapper一致だけcache reuseを許し、geometry/cell間reuse・角度差無視は不可。未解決予約は停止し、残予算を確認せず自動retry/resumeしない。

precisionは保存bias/normalization/軸別costだけを用い、PM-2同じ302表示grid epsilon0.005–0.1・alpha_axis0.025。軸headroom=epsilon/sqrt(2)−bias_axis、N_axis=ceil(2 B^2/headroom^2 log(2/alpha_axis))。不適格はshots/work=null、欠測を0にしない。primaryはN_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]、secondary6指標と共通P>=0。paired-axis covarianceから±2SE engineering intervalを保存し、formal CI・独立表示実験・一般的winnerは主張しない。表示materiality/uncertainty・境界規則は新source実装前にreviewする。

## 旧証拠layerとanchor

旧1.00 A/1.30 Aは旧source/compiler/dependency layerで保存し、新6点同士は一つの新identityで比較する。完全identity差がある旧costを同条件のmap点へ混ぜない。新環境1.00 A anchorは新6点間比較には必須でなく、旧1.00 Aとの直接geometry比較が必要なら別予算で提案する。218候補anchorは12,464 wrapper追加、6新点＋anchor87,248。今回は未認可・未実行。1.30 A全候補化も別認可。

## source実装前にreviewすべき未固定条件

1. 利用者が確定する距離list・formal wrapper/worker/memory cap・固定output root。
2. snapshot生成規約：SCF/DF algorithm・rank policy・fragment order/ties・sector/phase・solver/残差threshold。
3. 新campaign master seedとcanonical distance/seed/cache key規則。
4. サーバー3.12.3への新source環境bindingとsemantic gate、全default/pluginを含むcompiler identity。
5. precision表示/materiality/uncertaintyと旧証拠layer、anchorを追加するか。
6. 曖昧reservationの停止、resume/retryの別review、atomic ledgerと実transpile予算管理。

順序は契約review→新science module/runner/synthetic tests→actual source commit→source-bound sealed plan→別authorization→最終review→利用者の明示launch。現状は準備草案であり、production source/authorizationは未作成。

成功terminal案GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW、failure IMPLEMENTATION_GATE_FAILED、どちらもmandatory STOP、next-stage=false、research_decision=null。準備もここでSTOPする。

## 新規資料と差分

[zero-compute plan](zero_compute_plan_draft_v0.json)、[exact draft schema](zero_compute_plan_draft_schema_v0.json)、[future checkpoint schema draft](future_checkpoint_schema_draft_v0.json)、[static audit](static_audit_v0.json)、[environment inventory](environment_inventory_v0.json)、[summary](preparation_summary_v0.json)、[synthetic source](synthetic_fixture/synthetic_fixture.py)を新規追加した。旧source/test/result/manifest/authorization/原稿/図/研究概要/研究ノートは変更しない。研究判断・科学結果の変更ではなくserver準備記録として別directoryに保存する。

明示science access countersは[plan](zero_compute_plan_draft_v0.json)を参照。準備完了時点の分子resolve/stat/hash/load、runtime/checkpoint、科学runner実行、GPU query/allocation/kernel、共有環境変更、他job変更、commit/pushは0。JSON内のcounterはこの時点の履歴として保持し、後続の明示依頼による準備bundle公開と区別する。本計算へ自動移行しない。

静的/契約検査21件はPASS（fail/skip0）。最終監査ではcompile前後の全operator照合・合計128-call予算・45件のCPU dependency metadata closure（不足/制約不一致0）・旧247 sourceと保存6 JSONの不変性を再確認した。詳細は[最終監査](final_audit_v0.json)と[CPU依存closure](dependency_metadata_closure_v0.json)。
