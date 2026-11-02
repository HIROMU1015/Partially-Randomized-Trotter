# H6 native B0+I pilot：6軸compileの並列化準備

2026-10-15。利用者の「十分に並列化高速化してほしい」を受けて、既存の直列pilotから独立worktreeで実装した。**実装・限定mock検証PASS、実科学実行は未承認、mandatory STOP**。

- Branch: `track-a-h6-native-b0i-parallel-readiness-20261015`
- SOURCE: `7300b39df4c53d854d5bc8934430af19fbdb0085`
- 起点REVIEW_BUNDLE: `ec6bad241d0ab26dd5628c33d65c94b1536f7e9f`、旧SOURCE `b4c12d7a71dec57751ce21ebd434d797577ab450`。
- [契約・検証・独立差分レビューの入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_parallel_readiness_v1/2026-10-15/README.md)。
- [既存pilotの数学・入力・compiler契約](track_a_h6_native_b0i_pilot_readiness_20261015.md)、採択根拠は固定review `3d910ee1de4597b8895e8506500553dbad5a5085`。

## 並列化する処理

Linear H6 1Å/STO-3G、保存tol-only DF19/H/state、T=.8、二次S2、symmetric_directional、固定native Gaussian/compilerを保持する。対象はp5q1・p10q2・p15q8のcosine/sine計6軸のみ。

全3構成のPhase A（scalar/signal/reference/u/headroom）とPhase B（native/control/X-Y/fingerprint）を親driverで完了してから、Phase Cの6独立compileを最大6processへ割り当てる。[新worker module](../../scripts/h6_b0i_parallel_compile_v1.py)と[runnerのparallel hook](../../scripts/h6_b0i_pilot_runner_v1.py)が担当する。既定の直列経路も保持した。

検査済みwrapperをLinux forkで継承する。snapshot再load、native再build、回路のQPY/pickle保存・再読込、別compilerへの変換は増やさない。各子は固定axisを一度だけcompileし、返すのは最大16KiBのJSON metrics/identity/resource情報。入力・回路そのものをIPCで渡さない。各processの内部thread/Qiskit num_processesは1、compiler optionはbasis rz/sx/x/cx・opt1・seed17のまま。

| 操作 | 将来pilotの最大件数 |
|---|---:|
| snapshot load | 1 |
| overlay / scalar receipts | 各3 |
| native evolution build | 3 |
| measured wrappers / unmeasured probe copies | 各6 |
| control/X-Y statevector probes | 12 |
| 新compile | **合計6** |
| matched B0 / 新SCF・DF・state・signal指数作用・trajectory・hardware shots | 0 |

compile件数を並列数倍に増やす実装ではない。A/Bのどれかが不合格ならfork/compile予約0。予約はfork前にdurable journalへ記録し、失敗・未実行・不明でも返却しない。科学的な誤差政策、full-H reference不回転、signed full-Fock scalar、empirical allowance/certified=false、matched delta nullを保持する。

## 起動・監視・終了

1. 認可・SOURCE/input/profile/compiler/carry・候補・新CPU/resource bindingを検証後、専用outputのone-shotを消費する。
2. 全A/B合格後、最大6軸を同時に開始。worker数2〜6にも対応するが、最終契約へ明示固定し、資源不足時に無断で変更しない。
3. 親はcompileでブロックせず、各子のidentity、RSS/AS、thread数、affinity、CPU、pipe、native waitを確認。既存のhost/cgroup PSI/OOM・filesystem・5秒staleness gateも維持する。
4. 子は専用CPU1個へbindしてからcompileする。親がfork直後のbootstrapを観測する場合、CPU設定確認は最大5秒、設定後の逸脱はSTOP。全てown approved CPUの範囲内で動く。子ごとにprivate temp/MPL/cacheをoutput内へ作り、`/tmp` fallbackを防ぐ。
5. 子は最初に `PR_SET_PDEATHSIG(SIGKILL)` と親PID前後照合を行い、driver退出後のcompile継続を防ぐ。forkは親の単一thread確認後、停止signalを一時blockしてregistryへ登録する。成功PIDをidentity IOより先に保存する。
6. failureで未開始queueを止め、全own子へTERM、共通deadline5秒のwait後、残りをKILL・5秒wait。pidfd前後のidentityを確認し、ESRCHだけを既退出扱いする。他errorでも後続子のcleanupを続ける。FDを閉じ、最初のSTOPとcleanup errorを分離保存する。
7. native exit0、pipe EOF、PASS payload、identity/name、fresh fingerprint/compiler、6個の整数metricsが揃った軸だけ受理。partial結果とbounded raw IPCを保全し、全6結果・全子reap・最終source/profile/input照合が揃った場合だけCOMPLETEにする。親driverのnative waitは従来launcher/同一親recorderで保存する。

新しい汎用pool/observer/quota campaignは追加していない。launcher＋数値driver＋最大6compile子の常駐最大8processを見込む。同期Git metadata processやworker退出後のrecorderは別に扱う。

## 未承認の資源案

| 項目 | 案 |
|---|---:|
| compile workers / 内部thread | 最大6 / 各1 |
| 数値driver | 1 |
| 各数値process AS / RSS | 32 GiB / 9 GiB |
| metadata launcher AS / RSS / CPU | 512 MiB / 256 MiB / 120 s |
| 数値process aggregate AS上限案 | 224 GiB |
| launcher込みaggregate AS / RSS上限案 | 224.5 GiB / 63.25 GiB |
| fresh effective memory admission | **240.5 GiB以上**（AS合計＋headroom16GiB） |
| stage wall | 4時間 |
| driver＋全compile子の累積CPU | 合計4時間 |
| output / log / journal | 512 / 32 / 64 MiB |
| output inode / fresh FS・inode | 10,000 / 2 GiB・20,000 |

資源は実測予測ではなく未承認STOP上限。CPUを6倍使っても総CPU予算を6倍へ変更しない。親のCPUと、実行中の/proc CPUおよびwait4で確定した子CPUを合算し、CPU上限超過をSTOPする。per-process RLIMITと親のRSS/AS監視も保つ。

最終CPUは数値driver1個＋compile子に重複しない最大6個。metadata launcherはdriver CPUを共有する。`allowed_cpus=[]`、driver CPU未定、worker CPUs未定のまま。共有環境・他jobのaffinity/設定は変更しない。空きCPU/memory/pressure/OOM/fs/block/inode/quotaを直前に読取り、不足なら起動しない。quota有効でeffective余裕へ接続できなければguardを緩めずBLOCKED。

storageとcarryは旧契約を継承する。temp512MiBはadmission予約であり独立したdirectory hard capではない。過去のknown/held/failed/unknownを保持し、reset/refundをしない。子のpayloadにはcompile呼出し入口到達を記録するが、欠測を未実行と推定せずheld/unknownを維持する。

## 限定人工検証・差分レビュー

[検証計画](../../artifacts/resource_applicability/track_a_h6_native_b0i_parallel_readiness_v1/2026-10-15/artificial_test_plan_v1.json)内の単一test process/thread1・AS512MiB/CPU120秒/wall180秒で、**新24件＋影響する旧author23件＋旧独立23件＝70件PASS**。wall約1.25秒、peak RSS31MiB。旧63/75結果は保全し、無関係なbridge/benchmark全campaignを再実行していない。

新24件は最大6in-flight/2workerの3wave、queue failure、A/B barrier、重複禁止、直列との人工metrics/shot費用一致、未認可・synthetic実backend拒否、CPU/memory binding拒否、IPC・native exit・PDEATHSIG・fork失敗・所有identity・ESRCH/非ESRCH・cleanup継続・first STOPをmockで検査する。harness自体が実forkを拒否する。科学import、実child、実affinity変更、実build/probe/compileは0。

[軽い独立差分レビュー](../../artifacts/resource_applicability/track_a_h6_native_b0i_parallel_readiness_v1/2026-10-15/independent_review_v1.md)で、fork直後の未登録child、cleanup例外によるfirst STOP上書き、helperからsynthetic実forkへ届く経路の3点を修正した。レビュー済bytesを[actual SOURCEへ照合](../../artifacts/resource_applicability/track_a_h6_native_b0i_parallel_readiness_v1/2026-10-15/independent_review_source_binding_v1.json)した。科学的reviewは追加実施していない。

実fork・PDEATHSIGの当host実動作・実COW/RSS・compiler並列時の出力/速度・実H6 correctness/costは未実証。mockの同時割当ては高速化率の実測ではない。6compile部分の直列待ちを除く実装だが、A/Bは直列で、重い軸への偏りもあるため全体6倍速は保証しない。共有サーバー全体の空き状況に依存する。

Python timerは長いC内部の即時停止を保証しないが、親driverは子のC処理と独立に観測できる。OS/観測IO停止やpoll間のcap超過、D-state、SIGKILL時のterminal/native wait欠測は残る。PDEATHSIGもnative回収receiptの代用ではなく、欠測・回収未確認をCOMPLETE扱いしない。

## 固定・次の実行条件

runtime closureは10ファイル。新parallel moduleとrunner hook以外のruntime8ファイル、科学provider230ファイル、既存private profile34,273ファイルの契約値を変更していない。入力5件は旧bytes/SHA receiptを継承し、今回はNPZや実signal数値を読み直していない。科学SOURCE・旧grant/ledger/checkpoint・venv・旧77worktreeは保全した。

次は利用者による、このSOURCE・最大6worker/6compileと資源案の明示的な実行承認。その後、SOURCE/input/profile/carry、7 CPUとown affinity、未使用output/control/temp/one-shot、memory/PSI/OOM/fs/inode/quotaをfresh bindingし、独立最終binding確認とsealを経て一度だけ実行する。入口は従来のshell/launcherと同じで、final contractにparallel設定とCPUを結合する。absolute commandは現在nullのまま。

`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`execution_plan_sealed=false`、`automatic_retry=false`、`next_stage_authorized=false`を維持する。新auth/grantは発行していない。H6/H8の全面map・matched B0へ自動進行しない。

今回の変更は高速化実装と人工検証・資料整備だけ。実H6科学計算・新実データ再会計・実compileは0。SOURCE/testsと軽量契約/監査資料を別commitにし、non-force pushの成否を報告してmandatory STOPする。
