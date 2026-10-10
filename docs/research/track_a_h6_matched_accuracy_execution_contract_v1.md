# Track A H6 精度一致・測定込み資源比較：結果前実行契約 v1

2026-10-11。利用者が採用した[GPT独立レビュー](track_a_h6_technical_pilot_post_review_2026-10-11.md)と[準備範囲の契約](track_a_h6_matched_accuracy_preparation_contract_v1.md)を、有限候補・実装・別実行identityへ具体化する。今回は準備・synthetic検証・公開だけ。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP、`next_stage_authorized=false`。レビュー完了はlaunch認可ではない。

## 研究対象と保全する証拠

linear H6、等間隔1.00 Å、STO-3G、12 modes、α3/β3、sector400、tol-only `1e-8`で得たsigned generation-order DF rank19、cutoff0、T=0.8、同じ採用weighted Hermitian projectionと保存stateを使用する。新しい分解・SCF・solver・state正規化は行わない。有限時間複素coherent signalのRQ-Rが主、費用予測RQ-P1が補助。energy推定やchemical accuracyの主張には置き換えない。

入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、保存入力commit `554fc52add39c2c1b45b765a3135df76fda6f15a`、[入力監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json) commit `f52787a22542b31bd39fd004a8d3d71325bc56b0`。snapshot SHA-256 `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。[saved snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz)とreceipt全33ファイルを元commit/local bytesで照合する。

旧pilotは[結果索引](track_a_h6_technical_pilot_result_v2.md)、固定commit `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2`、correctness7/7、wrapper32/36、`PHASE_WALL_CAP:wrapper_cost` STOPのまま。旧欠測7、N/Gnull、u/ground-state未認定は変更しない。旧B3残り4 wrapperの補完、全面再実行、H8は今回の前提・対象ではない。旧random費用標本は新版のfresh meanへ混ぜない。

## 全候補と探索自由度

| family | prefix | q | r、R、K | formula | signal cell数 |
|---|---|---|---|---|---:|
| B0 discard | 5/10/15 | 1/2/4/8 | tailなし | S2 | 12 |
| B1 full deterministic | 19 | 1/2/4/8 | tailなし | S2およびglobal Yoshida S4 | 8 |
| B2 partial random | 5/10/15 | 1/2/4/8 | r=1/2/4、R=qr、K=2/4 | symmetric partial S2 | 72 |

計92 cell。要求精度ε_sig=0.05/0.01/0.005/0.001を全候補へ共通適用する。delta=T/qは0.8/0.4/0.2/0.1、tail micro-timeはdelta/r。生成prefixを使い、結果後のrank削除・並べ替え・fallback・候補追加はしない。B1 S2/S4・B0・B2は各自の最良qを選び、同qだけで優劣を決めない。この有限集合内の比較であり、連続パラメータの全最適性ではない。

B3はprefix0のlambdaを用いる有限式のnormalization診断のみ：q=1/2/4/8、r=1/2/4/8/16/32/64、K=2/4/6、168 scalar組。新B3 signal/sampling/compileは0。family全体を科学的に排除したと主張しない。

機械的な正本は[contract source](../../src/trottertracks/resource_applicability/h6_matched_contract_v1.py)の`plan()`と[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/sealed_manifest_v1.json)。planとsource digestをlaunch前に厳密照合し、手入力候補・CLIによる上書きを認めない。

## signal・数値幅・normalization・shots

参照は保存DFの400次元sectorをmatrix-free作用から構築し、全400 occupation columnの独立構成、sector expm/eighを比較する。保存stateの振幅は変更しない。各候補のnative primitive/Hornerと独立occupation/spectral/forward Taylor経路を比較し、raw/corrected/exact-tail signal、signed biasのPF/discard/finite-RTE分解、作用回数、最大中間ノルム、reference・経路・normalization差を保存する。S4の負時間を含む全登録physical primitive/timeで、保存stateとsector先頭・末尾columnの3 probeを検査する。

コンパクト経路は旧`traced_signal`と同じHorner・scalar・演算順で、巨大な各stageベクトル列を省きノルムとcountを保存する。toyで旧経路の最終ベクトル・signal・normalizationとbytes一致を検査済み。全primitiveのsector固有分解と、prefix別独立tail/truncated referenceをcacheする。cacheは400次元以下、primitive全20個以内であり、全4096次元fragment行列は作らない。expmとの全登録時間probe、候補ごとのnative/forward比較をcacheによって省略しない。

**数値allowanceは経験的であり、誤差上界・精度保証ではない。** 設定はbinary64、floor=1e-12、safety factor=10。`rounding_proxy=64*machine_epsilon*(1+native/oracle作用回数)*max(1,中間ノルム)`。combined axis uは `10*max(floor,rounding_proxy,reference差+最大経路差+raw/B closure差)`。log_B marginは `10*max(floor,rounding_proxy,normalization log差)`。両軸へ同じuを適用する。primitive probe最大差を普遍的uへ読み替えない。高精度400次元検証やground-state証明の代用ではなく、このbinary64独立経路でのengineering allowanceとして保存する。u・log_B marginをさらに10倍した感度表も固定して出す。

軸headroomは `h_a=ε_sig/sqrt(2)-abs(bias_a)-u_a`、`log_B_upper=log_B+declared log margin`。各軸 `N_a=ceil(2*exp(2*log_B_upper)*log(2/0.025)/h_a^2)`、N_total=N_real+N_imag。各candidate/要求精度のfresh whole-trajectory Hadamard estimatorを想定したconditional Hoeffding予算で、量子shots自体はsamplingしない。軸失敗確率0.025を使用する。探索全体のfamilywise winner保証ではない。

strict headroom>0、整数shotsをbinary64で表現でき、各軸u/h≤0.01の場合だけcost適格とする。数値的境界・log-domain-onlyは未解決、明らかなbias超過は不適格と記録する。経験的uを受け入れた条件下の精度会計であり、formal accuracyは未認定。guard超過・technical STOPを科学的不適格へ置き換えない。

## 費用の探索と独立確認

signal先行後、4精度のいずれかでcost適格となる**全候補**を費用取得へ登録する。解析費用proxyによる上位だけの非対称shortlistは使わない。不適格／未解決候補もcoverage表に残す。B0/B1は各構成のdeterministic measured-wrapper費用を1回、B2は各構成8 fresh whole-cost-trajectoryを取得する。二軸は同じevent列を共有し、独立標本数を16と数えない。paired mean/SD/SEを記録する。

探索RZのshot込みpoint scoreから精度ごとB2上位2構成を選ぶ。tieはcell ID辞書順。union最大8構成を固定してから、探索と別namespace seedで各32 fresh trajectoryを取得する。確認結果へ探索標本をpoolせず、deterministic費用とfresh確認標本を比較する。top2選択と有限候補coverageを明示し、未確認B2を列挙する。確認後に勝つまで標本やgridを増やさない。

主指標はno-prep measured Hadamard wrapperのRZ。CX/depth/sizeを副指標とする。`G=sum_a N_a*mu_a`。cosine/sine双方を実compileし、過去のcount一致から片軸を代用しない。共通state-prep Pはexcluded P=0を基本とし、`G(P)=G(0)+N_total*P`とpairwise非負crossingを保存する。RZをT countや物理実行時間へ変換しない。

cost primaryは`symmetric_directional`。ordinary sensitivityは、適格なら `B1_p19_q1_4th` と `B2_p10_q2_2nd_r1_K2` の探索replica0を同じeventでpaired取得する。各構成の探索replica0で保存stateのcontrol両branch、X/Yを検査し、deterministicの場合はfull-wrapper zと対応signal recordを照合する。random単一trajectory zをensembleに等置しない。他replicaは凍結lowering、primitive coverage、event digest、compiler fingerprintに結ぶ。未測定のprobe coverageを全面検証と呼ばない。

order確率、観測order count、全order0 trajectory確率、未観測orderを含むtrajectoryの確率、cost標本にnonzero orderが一つもない確率を保存する。component列support全体の網羅やµのtail boundではない。n=8/32、sample SD/SE、descriptive 2SE分離は母平均精度・同時confidence・正式winnerの保証ではない。主要比較はconditional point mapと確認結果、ばらつき、未取得／未観測massをGPTへ渡し、population winnerはUNDETERMINEDを許す。instruction guardを分布全体のcompiled-cost上界へ流用しない。

## 並列化・予算・実行上限

CPU IDs [0,2,5,6]。signal workerはNumba/OMP4、BLAS1。費用はdisjoint CPU sets [0,2] / [5,6] の2 disposable process、各Numba/OMP/BLAS/Qiskit/Rayon1、Qiskit parallel false。seedはtask ID/phase/replicaのSHA-256に結び、dispatch順に依存しない。各taskでprimary evolutionを保持し、ordinary sensitivityへ進む前に解放する。process終了でallocator/cacheを解放する。既存sector/matrix-free/native/atomic I/Oを再利用し、旧workerと凍結sourceは変更しない。既存generic executorは確認済みだが、新grant・one-shot・taskごとのprocess終了と全run guardのため専用stdlib supervisorを追加した。GPU対応statevector経路は今回の既存native portにはなく、GPUによる未検証の置換は行わない。速度向上率は実測していない。

| 上限 | 固定値 |
|---|---:|
| 新H6 signal cell | 92 |
| 探索cost group | 最大596（20 deterministic+72×8） |
| 確認cost group | 最大256（8×32） |
| cost-only random trajectory | 最大832 |
| occurrence | 最大6656（各trajectory q≤8） |
| primary measured wrapper | 最大1704 |
| ordinary sensitivity wrapper | 最大4、総最大1708（compile guard1712） |
| control probe guard | 1024 |
| primitive guard | 10000または登録必要数の大きい方 |
| 参照matvec guard | 20000、各action20000 |
| 各cell deterministic action | 100000 |
| B2各cell corrected+raw/forward tail action | 512 |
| untranspiled / transpiled instructions | 1000000 / 5000000 |
| signal / cost各worker AS | 各8 GiB |
| coordinator AS / live aggregate RSS | 2 GiB / 18 GiB |
| total output / logs | 2 GiB / 1 MiB |
| phase / total wall cap | **null / null** |

ASはRSSや予約RAMとは異なる。signalとcost phaseは同時に動かさず、cost2 worker+coordinatorのAS合計18 GiB。seal時ホストのavailabilityを記録するが予約とは主張しない。cost writer32 MiB/task、signal writer64 MiB。supervisorが子孫RSS/全output/logを監視し、workerはAS/FSIZE/core dump制限と個別call guardを持つ。全taskの最大countはdispatch前の有限planで制限し、完了taskの実countも集計する。途中で失われたattempt数は推定で埋めない。

既存pilot時間から精密な所要時間は予測できない。最大数は取得義務ではなく、signal適格数に従う有限budget。許可後は長時間runになる可能性があり、progressと経過時間は保存するがelapsedによる停止はしない。`launch_v1/STOP_REQUEST`を作れば全子process groupを終了する。メモリ/output/log/call/instruction超過、非有限値、整合性gate失敗、source/input/env変更はSTOP。失敗・欠測を保全し、retry/resumeは行わない。

## 固定source・実行入口・監査

[runner](../../scripts/resource_applicability/run_track_a_h6_matched_v1.py)、[metadata sealer](../../scripts/resource_applicability/prepare_track_a_h6_matched_v1.py)、[static auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v1.py)、[accounting](../../src/trottertracks/resource_applicability/h6_matched_accounting_v1.py)、[numerical port](../../src/trottertracks/resource_applicability/h6_matched_port_v1.py)、[no-wall supervisor](../../src/trottertracks/resource_applicability/h6_matched_execution_v1.py)、[synthetic tests](../../tests/tracks/resource_applicability/test_h6_matched_v1.py)。

source commit、全library/tracks Pythonと3 runnerのsource closure、入力commit/hash、環境、plan、assigned resources、exclusive outputはsealed manifestに保存する。prepareはstdlibでGit blobs/ZIP/NPY headers・metadataを読むだけで、ndarray decodeやscience port importはしない。default runnerもmetadata-only。old pilot grant・自由なCLI候補・未固定sourceはlaunchを拒否する。

launch用grantは**今回発行しない**。別途利用者が、sealed manifestへ結ぶ **「固定H6精度一致・測定込み資源比較 v1を一回実行」** を明示認可した後だけ、`h6_matched_authorization_v1`の新one-shot grantを作る。source commit/manifest digest、CPU IDs、exclusive output、retry=false、resume=falseを必須とする。grant bytes SHA-256をCLIでpinする。出力namespaceは `artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1`、未作成でseal。入力生成許可やH8許可には拡張しない。

入力/reference、primitive、92 signal、precision rows、選抜前後task、event、wrapper fingerprint/費用、探索・確認resource map、order diagnostic、coverage、call/resource/time、worker terminalと失敗、execution inventory/terminalをexclusive保存する。source/input/environmentを実行前・task前後・全run後で照合する。結果公開時は保存bytesのhash・coverageをstatic auditし、GitHub remoteから再取得して照合する。数値を再実行して監査結果を作り直さない。

## 完成条件と次の科学判断

準備完成は、契約・source・synthetic tests・入力identity・metadata seal・GitHub再取得照合が揃い、launch未認可を保つこと。real H6の新PASSではない。過去のsource/input/results/STOP、dirty/untracked、Track Bを保全する。

実行認可後の工程完成は、92 signal取得と全登録cost/確認taskの完了、source/input/envとcoverage audit、資源map・order/標本限界の明示、公開。科学的な精度保証・最適法認定とは分ける。候補が全て未適格でも結果を保全し、勝つまで追加しない。欠測なら工程STOPのまま判断を返す。

次の主要GPTレビューはH6 map/確認結果が得られた後、H8の独立検証設計前。H6はdevelopment、H8は未exposure・未認可を維持する。数値矛盾やtarget/estimator/比較条件の実質変更が必要なら途中でもGPTへ戻す。Codexはscientific GO/STOP、DF表現/Hermitization変更、H8の認可を代行しない。
