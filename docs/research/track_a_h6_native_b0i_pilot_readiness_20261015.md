# Track A H6 native B0+I 6軸pilot：最終実装・実行準備

2026-10-15。状態は **実装・限定人工検証PASS、実科学計算は未承認、mandatory STOP**。

- Branch: `track-a-h6-native-b0i-pilot-readiness-20261015`
- 最終SOURCE: `b4c12d7a71dec57751ce21ebd434d797577ab450`。最初の実装SOURCE `25c45704d8e5dc1cca53b76faabbd53168b10699` も履歴に保持。
- [契約案・全証跡の入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/README.md)
- 採択根拠: [固定GPT総合科学review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)。新しい科学判断・原稿改訂は実施しない。
- 基準bridge/guard SOURCE `20f89bc32b161ee22e4a1ed709f9e3414c19633a`、独立監査bundle `40a4f55464dba80ab2192d777f7d69d2f551676f`、科学native provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、旧H6独立確認結果 `321cd8c3066194fe98741cbebbcdba1515d1deaa`。

## 対象と科学契約

Linear H6、1.00 Å、STO-3G、既存tol-only DF actual rank19、保存H/DF/state、T=.8、二次S2、symmetric_directional、既存native Gaussian lowering/compilerを保持する。

| prefix | q | native evolution | 実compile予定 |
|---:|---:|---:|---|
| 5 | 1 | 1 | cosine / sine |
| 10 | 2 | 1 | cosine / sine |
| 15 | 8 | 1 | cosine / sine |

Full-Fock identity `C_tail = Σ_tail λ/4 [(Tr G)^2 + Tr(G²)]` をsigned λ、Hermitian G、元generation orderで計算する設計。元prepared blocksとnuclear/one-body/retained scalarを保ち、最終ancillaに `P(-.8 C_tail)` を一度追加する。元B0を置換しない。

保存corrected/exact-truncated signalと保存済みの独立forward corrected oracleを位相回転する。full-H referenceを回転しない。新biasは回転後signalと元full-H referenceの差。独立保存signalはfull-vector/full-operatorの新しい証明ではない。欠測・非finite・不一致はcompile前STOP。

既存のsignal/reference誤差分離、scalar/phase/libm roundoff、normalization/shot規則を変更しない。旧combined uしかない場合の保守的splitも固定bridgeの明示政策に従う。`64ε`等はEMPIRICAL、`certified=false`。保存binary64係数を対象とした入力係数誤差0を化学/DF誤差0と解釈しない。PR勝利をGO条件にしない。

## 実装と実行順序

[runner](../../scripts/h6_b0i_pilot_runner_v1.py)、[adapter](../../scripts/h6_b0i_pilot_adapter_v1.py)、[metadata launcher](../../scripts/h6_b0i_pilot_launcher_v1.py)、[shell入口](../../scripts/launch_h6_b0i_pilot_once_v1.sh)を独立moduleとして追加した。旧科学source・bridge・guard・shot数学は不変。

認可gateを入力mapping・科学importより先に通す。SOURCE/disk/Git、科学review、profile、入力、候補、compiler、CPU/carry/outputのdigestを照合する。未承認draftをCLIへ渡しても科学loaderに到達しない。

1. A: 全3caseのsnapshot/DF/state binding、元prepared blueprint、tail scalar、overlay、reference/u/headroomを検査し、caseごとのreceiptと10精度条件の会計を保存。
2. B: 全3caseでnative evolution各1＋測定wrapper各2を構築。control低枝identity、高枝leakage/norm/保存signal、X/Y符号、numeric fingerprintを確認。全A/Bが合格するまでcompile予約0。
3. C: 最大6 compile。fresh fingerprintと返却compiler hashを確認し、RZ/CX/RZ depth/CX depth/total depth/size、wallを取得。必要shotを両軸別に掛ける。不適格はINELIGIBLE、欠測/invalid metricを0に置換しない。
4. D: 入力/runtime profile・230科学sourceのdisk/Git bytesを再照合し、結果SHA・資源履歴・first STOP・terminalを保存。親launcherがwaitした後、同じ親からrecorderを起動してnative exitを別保存する。

Correctnessの主gateは固定native helperの `1e-9 + 1e-10 × max(1,scale)`、このadapterのscale1では約1.1e-9。追加の1e-8 ceilingより先に主gateが働く。許容誤差を緩めていない。

| 将来pilotで予約する操作 | 最大件数 |
|---|---:|
| snapshot load | 1 |
| analytic overlays / scalar receipts | 各3 |
| native evolution build | 3 |
| measured wrappers | 6 |
| unmeasured probe copies | 6 |
| statevector control/X-Y probes | 12 |
| B0+I compile | 6 |
| matched B0 compile / 新signal指数作用 / SCF/DF/state/trajectory / hardware shots | 0 |

高コスト操作は呼出前にdurable journalへ予約し、失敗しても返却しない。同一case再build・再retry・次stageを拒否する。旧B0 wrapper CSVはmetadata参考値だけに使い、完全matched条件未取得なので全deltaはnull。必要時のmatched B0最大6軸は別scope・別承認。

## 固定と限定人工検査

[closure](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/source_closure_v1.json)にruntime9ファイルのGit blob/bytes/SHAを記録した。新4 runtime、旧5 runtimeは不変。科学provider230 source filesと既存private profile34,273 filesを読取SHA照合し、差分0。profile原本8,926,400 bytes、SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、basis rz/sx/x/cx、opt1、seed17。既存venvは変更せず、旧compilerとの全output bit同一性は未検証のまま。

[人工検証結果](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/regression_result_v1.json): **63 distinct PASS** = author23＋独立担当23＋影響する旧bridge17。単一test process/thread1、AS512MiB/CPU120秒/wall180秒の事前計画内、科学imports・実child起動・実compile0。旧人工75件の固定結果は保持し、全面再実行しなかった。初回fixtureのcomplex JSON表現と拒否理由期待値を修正した失敗履歴も保存。追加temp差分はauthor23だけ再検査した。

[独立技術監査](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/independent_reviewer_report_v1.md)は実装authorとは別担当。schema接続時のB0I cell欠落、終了時provider再照合、pilot固有seal/review gate、launcher TERM/早期例外時cleanup、journal cap、tempfile fallbackを修正した。[検証matrix](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/correctness_validation_matrix_v1.json)は人工PASSと実H6未検証を区別する。

## 入力・資源・運用条件

[入力receipt](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/input_receipt_v1.json)のsnapshot receipt、DF receipt、NPZ、signal summary、旧wrapper CSVの5件、計2,049,991 bytesをbytes/SHA照合した。NPZ arrayと実signalの数値内容は読まず、新scalar/signal/bias/u再会計を行っていない。公開物はreceiptのみ。実入力とprivate mappingはGit管理外homeに保持。

| 未承認STOP上限候補 | 値 |
|---|---:|
| 科学worker / 内部thread | 1 / 1 |
| worker AS / RSS | 32 GiB / 9 GiB |
| stage wall / worker CPU | 4 h / 4 h |
| output / log / journal | 512 MiB / 32 MiB / 64 MiB |
| output inode / terminal reserve | 10,000 / 4 inode・1 MiB |
| launcher soft AS / RSS / CPU | 512 MiB / 256 MiB / 120 s |
| fresh effective memory | 48.5 GiB以上（32＋.5＋headroom16） |
| fresh filesystem / inode | 2 GiB以上 / 20,000以上 |
| monitor staleness | 5 s |

これは停止上限案であり必要資源の測定予測ではない。常駐は科学worker＋metadata launcher、worker終了後だけrecorder。source照合の同期Git subprocessもあるため総process最大2とは主張しない。科学pool・追加observerはない。

2GiB容量案にはfinal512MiB、pending reserve512MiB、temp reserve512MiB、log32MiB、journal64MiB、terminal1MiBとblock paddingを含む。journalはoutput capにも算入するため保守的重複を含む。temp512MiBはadmission予約であり独立したdirectory hard capではない。実行前にtemp/cacheをexclusive output内へbindし、認可後にmkdirしてprocess限定TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME/tempfile.tempdirを固定する。実inputの新規コピー0。監視はeffective filesystem余裕も継続確認する。

現在の読取観測では64 CPU可視、effective空きmemory約468GiB、filesystem空き約309.8GiB、inode約2.195億、host/cgroup full PSI 0。host/祖先cgroupの過去OOM count1はbaselineとして保存し、0へ書き換えない。mount option上quotaなし。これはfresh launch合格の代用ではない。quota有効ならuser/group/projectのeffective残量に接続できない現guardはSTOP/BLOCKEDとし、guardを緩めない。

CPUは未選定、`allowed_cpus=[]`。他jobと衝突しないCPUを直前読取確認し、own processだけaffinityを適用する。carryのknown/held/failed/unknownは元ledgerへfresh bindingし、nullを0へ補完しない。reset/refund0。旧grant/ledger/checkpointは一切更新しない。

Python timer/checkpointsは長いC処理中の5秒以内の即時観測を保証しない。復帰時stalenessでfail-closed、launcherは1秒pollでwall/log等を監視しTERM5秒→KILL5秒のowned waitを行う。poll間のcap超過、回収不能なD-state、launcher自体のSIGKILL、terminal/native exit欠測はあり得る。欠測を成功扱いせずcleanup未確認として保全する。過去exit時刻は推測しない。

## 承認後の具体的な手順

今回発行したのはdraftと[将来の認可schema](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot_readiness_v1/2026-10-15/future_authorization_schema_v1.json)のみ。authorization receipt・実grantは発行していない。

1. 利用者が最終SOURCE、3case/6軸、追加12probes/6copies、worker1、compiler、CPUと資源案、one-shot/no retry/mandatory STOPを明示承認する。
2. 既存ledger/historyをresetせずcarryへ結合。SOURCE/input/profile/compilerを再照合し、未使用output/control/temp namespaceとone-shot未使用を確認する。controlはown exclusive directory、outputはまだ未作成。
3. CPU/memory/pressure/OOM増分/fs block/inode/quotaとhard AS/CPU継承をfreshに確認。quota接続不能・資源不足なら起動しない。
4. 承認証拠SHA、final contract raw-byte SHA、field digests、独立最終binding review commitを新authに結合。`approved/runtime_authorization=true`・CPU/commandを埋めてsealする。この準備では実施しない。
5. 以下の**未実行テンプレート**をfinal absolute commandとして結合後、一度だけ起動する。環境変数・affinityはown processだけに作用する。

```bash
# 未実行テンプレート。承認・fresh binding後の値が必要。
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 QISKIT_NUM_PROCS=1 QISKIT_PARALLEL=FALSE \
  taskset -c "<approved_cpu>" bash "<SOURCE_ROOT>/scripts/launch_h6_b0i_pilot_once_v1.sh" \
  "<SOURCE_ROOT>" "/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python" \
  "<sealed_contract>" "<new_user_authorization>" "<private_input_mapping>" "<unused_output>"
```

6. 完了またはfirst STOPで終了し、native exit/owned cleanup/SHA/metricsと保存shot会計を監査・公開してmandatory STOP。失敗したgrant/outputは再使用せず、自動retry・matched B0・全面mapへ進まない。

現時点は `approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`、`execution_plan_sealed=false`、`automatic_retry=false`、`next_stage_authorized=false`。科学的な再判断を要する新blockerは独立監査で見つかっていない。残る作業は明示的な実行承認と上記fresh binding。ただし実保存independent reference、prepared digest、native物理correctness、u/headroom適格性、実compiled費用・所要時間は将来の認可pilot内で初めて確かめる。

## 公開範囲と次の担当

SOURCE/tests、未承認契約、metadata receipts、限定tests/監査資料と索引だけをcommitする。NPZ・実runtime/checkpoint/cache・credential・内部SSH情報は含めない。GitHub認証失敗時は設定を変更せず、手動pushコマンドと未公開状態を報告する。

次は利用者の実行承認、その後サーバーCodexがfresh bindingと固定one-shot実行。結果取得後にGPTへ科学的解釈を戻す。実H6 science/build/probe/compile、新実データ再会計は今回すべて0。新原稿・新科学claim・他job/共有venv設定変更0。準備終了後mandatory STOP。
