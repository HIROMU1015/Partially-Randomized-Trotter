# GPUサーバー側Codexへの依頼 H4資源mapの実行準備とサーバー環境固定

## 今回の依頼と停止位置

Track Aの原稿作成を保留し、H4のgeometryと要求精度による全候補resource mapを次の計算候補として準備します。
**新しい実行環境は、GPUサーバー側の既存環境をできるだけそのまま採用してください。**
旧ローカルPython3.11.0rc1やpackage versionへの完全一致を理由に、直ちにdowngradeや比較不能判定をしないでください。

今回進めるのは、clean worktree、環境記録、科学データを使わないCPU benchmark、source互換性の静的監査、
新契約・schema・zero-compute planの草案までです。
**新geometryの分子計算・参照状態solve・signal評価・実candidate sampling/build/compileは起動しないでください。**
science module/runnerの実装、source commit固定、別authorization、最終review、明示launchは次段階です。

計算場所としてGPUサーバーを想定していますが、主な処理はCPUのQiskit compileです。
今回GPU query/allocation/kernel、CuPy import、`nvidia-smi`は0にしてください。
環境と設計を報告した時点でSTOPし、長時間の本計算へ自動移行しないでください。

## repositoryと引継ぎidentity

- repository：`HIROMU1015/Partially-Randomized-Trotter`
- 証拠の起点commit：`4c23453c541700c6a41ba71fc5ec9323b53858d6`
- その証拠branch：`pr2-v4-s2-parallelization-20260928`
- PM-2 result commit：`5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`
- 新branch推奨名：`track-a-h4-geometry-resource-server-prep-20261005`
- 本依頼は新しいuser handoffです。旧M1/M2/PM-1の実行authorizationを追加geometryへ適用しないでください。

新設計案と本promptは作成時点ではローカル未commitです。
この全文を渡された場合は、上記の既に公開済みcommitをcode/evidence起点にし、本handoffの準備指示を別に保存してください。
このprompt自体がGit tree内にあると仮定しないでください。
後で新しいhandoff bundle commitが明示された場合は、その40文字SHAと上記起点からの系譜を確認してください。
追従するbranch tipを勝手に起点としないでください。

`origin`のownerが`HIROMU1015`であることを先に確認し、`git fetch origin --prune`後、
指定commitから独立clean worktreeと新branchを作成してください。同名があれば上書きせず連番にします。
既存worktreeのreset/clean/stash、mainへのmerge、force-pushは禁止です。
今回commit/pushは依頼していません。新しい準備資料はlocalに残してdiffとpathを報告してください。
quration等の他repositoryにはcommit/pushしないでください。

最初にそのcheckoutの`AGENTS.md`、`PROJECT_MAP.md`、`docs/research/研究概要・現状.md`を読み、
保存結果とdated history、旧promptを区別してください。
旧文書の原稿収束方針は当時の判断として保存し、このhandoffは原稿保留と新設計準備だけを更新します。

次の保存JSONだけは、内容・hash・指定commit blobを照合してよいです。分子snapshotは開きません。

| evidence | repository相対path | SHA-256 |
|---|---|---|
| M1-A signal | `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json` | `1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086` |
| M1-B1 compile | `artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json` | `71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4` |
| PM-1 nearby discard | `artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json` | `9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b` |
| M2 fixed transfer | `artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json` | `f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931` |

候補の正本は
`artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/candidate_inventory_v1.json`、
入力の正本は同directoryの`input_identity_v1.json`です。
各fileを起点commitのblobとbyte-identicalであることまで確認し、不一致を代替artifactで救済しないでください。

## サーバー既存環境を優先する方針

既存の利用可能Python/venvから、必要なCPU科学packageが最も揃ったものを選んでください。
既知の候補`/home/AbeHiromu/venvs/trotter-common/bin/python`が存在するかは確認してよいですが、存在を仮定しません。
最終的に選ぶPythonはabsolute pathにし、既存venvのactivateやshell設定変更を避けます。

読み取り専用で次を記録してください。

- OS/kernel、CPU model、socket/core/logical CPU、NUMA、利用可能core、共有job負荷。
- RAM/available memory/swap、filesystem/free space、cgroupやschedulerの制限。
- Python executable/version、NumPy/SciPy/Qiskit/rustworkx/OpenFermion/OpenFermion-PySCF/PySCFのversion。
- BLAS実装とthread設定、必要なQiskit APIの存在、package provenanceまたは取得可能なwheel/source identity。
- 新しい作業root、output候補、userが利用できるresource制約。

依存version取得はmetadataを優先します。Qiskit/NumPyのCPU-only importや純synthetic操作は許可します。
GPU関連packageを存在確認のためにimportしないでください。実分子を構築して依存確認しないでください。

既報のPython3.12.3/Qiskit1.3.0等は参考値であって、今回の現況ではありません。
Python/NumPy/SciPy/OpenFermion/PySCFはサーバーの既存versionを優先して新campaignの基準にします。
Qiskitも、既存環境に1.3.0があれば優先候補ですが、別versionしかない場合は即停止せず、
そのversionでのAPI互換性・compiler条件・比較scopeを草案へ明記してください。

**実装を新しい環境へ適合させることと、数学的な意味論を変えることは別です。**
canonical finite-RTE、二次PF、global phase、normalization、軸別shot式、wrapperの測定意味論は維持します。
互換化で旧source-bound module/testを書き換えず、後続で新しいTrack A sourceへ分離する設計にしてください。
version guardを削除した旧runnerを起動することは禁止です。

不足依存がある場合、install/upgrade/downgradeを勝手に行わず、別の既存venvが使えるか確認します。
それでも不足なら必要最小限の専用環境案・変更対象・理由を報告し、利用者の判断へ戻してください。
今回system Python、CUDA、driver、他jobの環境、global shell configは変更しません。

## compilerの差と旧証拠の扱い

新campaignのbasisは`rz,sx,x,cx`、optimization level1、transpiler seed17、
backend/coupling/layout/routingなしを草案の第一案とします。
実際のversion、追加default options、custom pass/plugin、approximation設定までcompiler identityへ記録してください。
version差を隠してQiskit1.3.0と同一compilerと呼ばないでください。

新geometry同士は同じsource・compiler・dependency identityで比較します。
旧1.00 ÅやM2保存costはimmutableな旧条件の証拠として保持します。
compiler policy/versionが異なる場合、旧costを新mapへ同条件の点として直接混ぜず、別layerで表示する案を提示してください。
直接接続するための新環境1.00 Å anchor再評価が必要と考える場合も、必要性と追加予算を提案するだけにします。
218候補のanchorは12,464 wrapper追加で、6追加geometry＋anchorなら87,248 wrapperです。自動追加は禁止です。

旧snapshot hashの再現を新環境で保証できると仮定せず、新snapshotは新しい生成provenanceを持たせます。
旧状態を読み込んで環境差を確認することも今回は許可していません。
意味論の照合は純synthetic小型operator/wrapper testsで設計し、分子snapshot比較の必要性は次段階の契約へ送ってください。

## 提案するH4全候補map

モデル第一案はH4 linear、STO-3G、8 system qubits、explicit/actual DF rank12、
geometry別の同じphysical sector内の最低DF固有状態、T=0.8です。
新geometryにはSCF・積分・DF・reference solveが必要ですが、今回実行しません。
requested/actual rank、fragment順序、状態位相、残差のgateを新契約へ明示します。
rank不足、不収束を別rank/geometry/状態で救済する設計にしないでください。

追加距離の**未承認草案**は隣接H–H距離0.70、0.80、0.90、1.10、1.40、1.60 Åの6点。
利用者がまだ確定していないため、planの`geometry_frozen=false`、`science_execution_authorized=false`を維持してください。
8点や別距離をサーバー側で自動選択しません。
過去の別研究解析で使用したgeometryを含み、fresh blindとは呼びません。
1.00 Åは既存218候補、1.30 Åは使用済みM2固定5構成だけです。後者を完全gridと数えないでください。

各geometryには次の**template集合**を固定する案です。

| method | template | count |
|---|---|---:|
| B0 discard | L_D=3/4/5/6/9、q=1/2/4/8、r=K=0 | 20 |
| B1 deterministic | L_D=12、q=1/2/4/8、r=K=0 | 4 |
| B2 partial | L_D=3/6/9、q=1/2/4/8、r=1/2/4/8/16/32、K=2/4 | 144 |
| B3 random-dominant | L_D=0、同じq/r/K grid | 48 |
| 既登録r64二件 | B2-rank3-q1-r64-K2、B3-rank0-q8-r64-K2 | 2 |
| 合計 | random194＋deterministic/discard24 | 218 |

保存candidate inventoryの218 templateと一致することをzero-computeで確認してください。
旧snapshot固有のfingerprintをコピーせず、新geometryのH/DF/state/source/compiler identityへ結合する設計にします。
旧r64選抜を各geometryで再実行せず、上記二件を最初から含めます。旧16-cell selectorも使いません。

32 classical trajectory/cell、cosine/sineで同一trajectoryを共有し、axis別keyを持つ案です。
random194×32×2=12,416、baseline24×2=48、**1 geometry計12,464 wrapper records**。
追加6点で74,784、8点なら99,712ですが、距離数・capはまだ正式固定していません。
accuracy-ineligible候補も登録を維持し、完全map案ではcompile監査に残しますが、matched-accuracy workはnullです。
point Pareto/winnerに不適格候補を入れず、0 costとして扱わないでください。

precision sweepは保存bias/normalization/軸別costだけを使います。
第一案はPM-2と同じε=0.005〜0.1の表示grid、α_axis=0.025。
primaryは`N_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]`、secondaryは6 compiled metricsと共通P>=0感度です。
εごとに新signal/trajectory/compileを行わず、表示点を独立実験と呼ばないでください。
±2SEはengineering intervalであってformal CIではありません。

## 科学データを使わないCPU benchmark

環境inventoryと静的互換性確認後、科学データを読まずにbenchmarkを実行してよいです。
repository外の新規一時directoryを使い、次を守ってください。

1. 9 qubit・1 classical bit、固定synthetic seed、研究DF係数を使わない高level controlled Gaussian/Givens類の回路を用いる。
2. 小/中/大の三規模を、transpile結果を見てtaskを選抜せず事前固定する。実baselineの約12k/45k/90k compiled sizeは規模の参考に過ぎず、完全一致を要求しない。
3. 固定task setを1/6/12/16 workerで比較する。共有負荷・memory制約に危険な条件は実行せず理由を記録する。
4. 全条件合計128 transpile以下、総wall30分以下。予算timeoutは自分のbenchmarkだけに適用する。
5. build wallとtranspile wallを分け、tasks/min、scaling、parallel efficiency、peak RSS、failure/OOM、出力gate数を記録する。
6. fixture sourceとtask定義のfingerprintを保存し、同じfixtureをローカルで後から実行できるようにする。
7. 外側process pool以外の並列が重ならないよう制御する。各workerのBLASは1、Qiskit内部process/thread設定も記録する。
8. 小型wrapperのglobal phase・測定axis・operator一致をsynthetic testsで確認する。実H4 signal/costを計算して性能probeにしない。

固定process environmentの第一案は次です。サーバーの選択済みPythonをabsolute pathで使ってください。

```text
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Qiskit/Rust内部並列制御は実際のversionに適合する設定を確認し、採用値を保存してください。
参照：[Qiskit1.3 compiler documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/compiler)。
既存環境が別versionならその公式文書を確認し、1.3の設定を無条件に仮定しません。
他userのprocess停止・priority/affinity変更はしないでください。自分のjobを低priorityにする場合は値を記録します。

今回はローカル側を遠隔起動しません。ローカルとの同じfixture実測がない場合、
サーバーのworker scalingだけを報告し、「ローカルより何倍速い」を断定しないでください。
旧M1-B1の6worker wall21,021.267秒は参考です。新geometryのETAへ単純変換しないでください。
worker上限はbenchmarkと共有資源から提案し、旧最大6を流用したり、16を自動認可したりしません。

## 新しい契約とplanの草案

環境inventory・benchmarkを踏まえ、Track A専用の新しい設計資料とmachine-readable草案を保存してください。
既存証拠を移動・上書きせず、新しい準備directoryを使ってください。

次を草案へ含めます。

- 選んだserver Python/venv・全依存・compiler identity、利用可能CPU/memory、推奨worker/thread数。
- 未確定geometryと218 template、32 trajectory、axis共有、予算、seed/key生成規則。
- 新snapshot生成・rank/sector/fragment/state gateと、同じgeometry内のsignal/cost fingerprint照合。
- checkpointのgeometry/H/DF/state/candidate/axis/seed/index/compiler/source/wrapper identity。
- atomic completion ledger、実transpile数とcache reuseの区別、曖昧な予約の停止と無断retry禁止。
- 旧compilerと新compilerが異なる場合の比較layer、anchor必要性と追加予算。
- 保存値precision mapの規則、accuracy-ineligible=null、missing/uncertaintyの表示。
- terminal案`GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW`または`IMPLEMENTATION_GATE_FAILED`、
  全status mandatory STOP、next-stage=false、research decisionの自動分類なし。
- 次段階のsource→sealed plan→別authorization→review→利用者launch順序。

まだ新snapshotやproduction sourceが存在しないため、それらのhashを想像して埋めないでください。
source/geometry/environment/seedが未固定のplanを正式sealed planとは呼びません。
今回の草案statusは`SERVER_PREPARATION_COMPLETE_AWAITING_CONTRACT_REVIEW`、
`science_execution_authorized=false`、`geometry_frozen=false`です。

## 禁止事項

- 新旧のproduction science runnerのrun/resume、既存M1/M2/PM-1/PM-2の再実行。
- 分子NPZ/NPY/pickleや旧runtime/checkpointのresolve/stat/hash/load、copy/move/rsync/scp。
- 新geometryのSCF/DF/reference solve、actual候補のsignal/sampling/build/compile。
- 全repository testsの実行。許可するtestsは新しい純synthetic fixtureと静的/契約検査だけ。
- 旧source/module/test/result/manifest/authorizationや原稿・図の変更。
- 依存のinstall/upgrade/downgrade、system/CUDA/driver設定変更。
- GPU操作、H6/H8/H12、追加96、別geometry追加、高次PF、strong synthesis、energy/RPE、Track B統合。
- 実行前source/authorizationの先取り、capを超えたbenchmark、本計算の自動起動。
- 既存job操作、worktree破棄、main merge、force-push、今回のcommit/push。

## 最終報告とmandatory STOP

先頭に次のいずれかを示してください。

- `SERVER_NATIVE_ENV_READY_FOR_CONTRACT_REVIEW`
- `SERVER_NATIVE_ENV_REQUIRES_SOURCE_PORT`
- `BLOCKED_DEPENDENCY_OR_RESOURCE`
- `HANDOFF_IDENTITY_MISMATCH`

その後、repository commit/worktree、保存4 JSON identity、選択Python・依存・compiler、
旧環境との差、CPU/RAM/NUMA/load、synthetic fingerprint・実測・resource使用を報告してください。
synthetic wrapper数とscience wrapper0を分け、分子・runtime・GPU access0を明示してください。

新規6点案/218 template/74,784 wrapperの設計、未固定距離、推奨worker、
旧証拠と比較可能な範囲、anchor追加の要否、次に必要な最小修正・資料pathを列挙してください。
科学module/runnerの実装前に固定すべき条件が残る場合は具体的に示してください。

報告後にSTOP。**このpromptだけでは本計算を開始しません。**
研究結論や投稿可能性を推測せず、結果前契約のreviewへ戻してください。
