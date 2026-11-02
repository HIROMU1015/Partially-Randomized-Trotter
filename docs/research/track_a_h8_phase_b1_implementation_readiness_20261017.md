# H8 Phase B1：費用モデル・holdout検証の実装準備

**実装・人工検証まで完了。実H8のfit・残差計算・費用再会計は未実行。**
利用者はB0の8軸取得と独立したB1準備だけを認可した。実B1の数値実行はB0取得完了後の別承認を待つ。

- branch：`track-a-h8-phase-b1-implementation-20261017`
- 起点：`847266323ea3d7bd29ac42322b7819b0b2a6c43f`（origin一致確認）
- 既存B1契約：`b09f408873cad94a371261386f7f3bf9cc0baa91`（origin一致・起点の祖先）
- 実装SOURCE：`1e25d5691b2a5c4738ca8b67c398ae06b70c4e0f`。source/testsを先に固定し、準備資料を別commitへ分離した。
- [契約・無効grant・人工検証・source closure](../../artifacts/resource_applicability/track_a_h8_phase_b1_implementation/2026-10-17/README.md)

新しい独立worktreeで作業した。B0のbranch/worktree/SOURCE/grant/ledger/lock/CPU割当を変更せず、既存の別B1作業worktreeにも書き込んでいない。
当作業でB0ジョブの起動・停止・再開・affinity変更・状態監視を行っていない。現在のB0取得の成功や完了を本書では認定しない。
サブエージェント0。主担当による実装と限定技術確認であり、別担当の独立科学レビューではない。

## 固定した範囲

linear H8、1 Å、STO-3G、16 spin modes、Nα=Nβ=4、sector4900。
保存DF rank28、tol=1e-8、cutoff0、signed generation order、weighted Hermitian projection、T=.8。
S2の `C_RZ=a+b*q+d*prefix*q` をmethod×axisの4層で独立にfitする。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなし、symmetric-directional。
完全profile SHAは `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
既存full-H reference・経験的allowance factor1/10・科学条件は変えない。

|層|fitする入力|検証する入力|検証の位置付け|
|---|---|---|---|
|B0/cosine|p14/q1、p21/q1、p14/q8 の3軸|TRAIN3軸＋p21/q8 HOLDOUT1軸|holdoutはfitへ入れない|
|B0/sine|同じ3構成のsineのみ|同じ4構成のsineのみ|cosineとは独立|
|B0+I/cosine|p14/21×q1/2/4/8の既存8軸|同じ保存8軸すべて|既存契約のfit残差診断。独立holdoutとは称しない|
|B0+I/sine|同じ8構成のsineのみ|同じ保存8軸すべて|cosineとは独立|

B0+Iは固定済み16軸だけを再利用する。[既存台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b09f408873cad94a371261386f7f3bf9cc0baa91/artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/anchor_registry_v1.json)のSHA-256 `b58b236d6b767a06142a87d6e482aa356a334f81b15ba506b98bdf035ca82f83` をpinした。
未解決の旧B0の16行はfit・検証へ渡さない。B0+IをB0へpoolせず、H6係数・理論A/B係数で代用しない。

S4の `a+3*b*q+3*d*prefix*q` は既存仮説の文字列として保持する。
今回のrunnerはS4をfit・実測・検証せず、B2のcompileやCの再会計を呼ぶ経路も持たない。

## 実装

- [model](../../src/trotterlib/h8_phase_b1_model.py)：標準ライブラリのFractionによる固定3列の有理数最小二乗。singularならSTOPし、solver・model・層を切り替えない。
- [receipt/認可gateと準備CLI](../../scripts/h8_phase_b1_validation.py)：source/input/profile/preparation・旧契約のbytesを照合し、future用configと**無効grant**だけを作る。
- [数値runner](../../scripts/run_h8_phase_b1_validation.py)：新B1 grant、live own observer、fresh gate、未使用output/lockが成立した場合だけ数値経路へ入る。既定実行はplan表示だけ。
- [launcher](../../scripts/launch_h8_phase_b1_validation.py)：2物理core・CPU quota・idle・memory/PSI/OOM・filesystem/inode/quotaをfresh確認し、既存のowned supervisorへ接続する。今回この実動作は起動していない。

全24軸の入力schema・来歴を確認してから最初のfitへ入る。B0のsourceは新しい取得SOURCEへ固定し、B0+Iは元実測SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630` を保持。
各軸で `abs(predicted-measured)/measured >= 1/50` ならその場でSTOPする。measuredが分母で、2%ちょうどは不合格。
軸平均・最大値の丸め・refit・holdout再割当・追加測定による救済を行わない。
検査順はB0 cosine→B0 sine→B0+I cosine→B0+I sine。
途中までPASSしても全4層の完了モデルとして公開せず、最初のSTOP理由・partial event・消費済みfit数を残す。

新しいone-shot lockとlaunch-attempt markerを使い、同じ出力への二重実行を拒否する。
4fit、B0 TRAIN検査6、B0 HOLDOUT検査2、B0+I検査16が固定上限。
compile/signal/再会計/sampling/refit/追加測定の呼出上限はいずれも0。

## B0の受領gate

futureの受領manifestで次の7 JSONをpath/bytes/streaming SHAへ固定する。

1. B0取得config
2. その取得grant
3. B0取得terminal（全8軸）
4. B0 launcher terminal（exit0、STOP理由なし、全reap、cleanup errorなし、監視gap≤5秒）
5. B0 acquisition final gate（PASS）
6. B0 final fresh gate（SOURCE/config/grant/profile/input/observer/保全へ結合）
7. 既存B0+I固定台帳

B0の全8軸・TRAIN/HOLDOUT割当、SOURCE、compiler/profile、prefix preparation、RZ正整数、compile前後fingerprint、control error、native exit/reap、既存capsの完了会計を検査する。
取得configの34,273 runtime/265 provider mapping digestと固定科学入力receiptも既存契約へ照合する。
これはB0取得時の固定証拠を受け取る設計であり、今回既存venvの全file inventoryを再走査したという意味ではない。
B1自体は標準ライブラリのみで、実行interpreterのbyte SHAと17 source closureを独立に固定する。

**8測定軸が保存されていても、取得または監視がSTOPなら受理しない。**
不完全な結果の採用、過去のSTOP理由の消去、既存grantの拡張や後付け認可を行わない。
欠測・SOURCE/profile/preparation/入力の不一致は実fit前にSTOP。
今回B0の完了証拠をB1用に受領・sealする処理は実行していない。実取得の成否はこの実装準備と別である。

## 人工検証

[28件のunittest](../../tests/test_h8_phase_b1_model.py) がPASS。
4層の既知係数、有理数OLSの残差直交、rank不足、holdout非混入、行順不変、2%境界、measured分母、片軸/B0+I不合格の即STOP、全来歴項目の欠落、重複・余分軸、旧B0拒否を検査した。
receiptのbyte改変・重複JSON key・非finite JSON、取得STOP/reap失敗、grant/source/scope/capsの変更、既定CLI、first STOP・二重実行拒否も確認した。

人工driver検査は新しいfixture directory内で実際にpartial/terminalを保存した。SOURCE/環境/資源観測・affinityはmockし、実H8入力は一切渡していない。
OS observer/CPU affinity/実プロセスcleanupの新たな統合検査、real receipt受領、実fitは未実施。
再利用するsupervisorのsourceは変更0で、過去campaignの反復や実worker stressも行っていない。

テストは単一test process＋順次plan-only CLI child、thread1、AS512MiB/CPU120秒/wall120秒/output4MiB、temp/logはhome配下。
最終28件は約0.33秒。途中で人工driver fixtureをprivate-path制約外に置いた検査が失敗したため、fixtureをhome内の指定evidence領域へ直した。実行制約は緩めず、全試行logのhashを保存した。

## 実行契約・資源案・未承認事項

[契約案](../../artifacts/resource_applicability/track_a_h8_phase_b1_implementation/2026-10-17/execution_contract_UNAPPROVED_v1.json) と[無効grant案](../../artifacts/resource_applicability/track_a_h8_phase_b1_implementation/2026-10-17/grant_DRAFT_INVALID_v1.json)。
approved=false、runtime_authorization=false、valid=false、allowed_cpus=[]、command=null。
SOURCEだけ固定済みで、実B0 receipt・新config digest・CPU・output/control/lock・approvalは未結合、実行planは未seal。
新しい有効grantは発行していない。

既存B1案のdriver AS512MiB/RSS256MiB、CPU240秒、wall600秒、output64MiBを維持。
独立observer案はAS512MiB/RSS256MiB、CPU120秒、総RSS512MiB＋headroom512MiBでadmission1GiB。
1計算process＋observer1、内部thread1、worker0、2つのidle物理core。
1秒監視/最大gap5秒、host full PSI≥1%・cgroup full PSI>0・OOM増分・資源欠測でfail-closed。
block128MiB/log各16MiB/inode1000の保存上限、fresh disk2GiB/inode40,000以上、quota未知ならSTOP。
wall600秒は参照した既存B1契約の明示上限を保ったもので、新規実行への時間上限方針を独断で再定義していない。
これらは実行前の資源案であり、現在のCPU割当・既存jobの制限を変更していない。

B0完了後、immutable receiptsとB1 SOURCE/Python/入力へbindingし、fresh gateと新grantを作るには**B1数値fitの別の実行承認**が必要。
B1の準備完了または将来のPASSからB2・Cを自動起動しない。S4妥当性・全grid保証・費用優劣の新判断は未実施。

## 保全と公開

旧rootのdirty/untrackedと6 worktreeのHEAD/statusを前後照合した。B0関連4 worktreeと既存B1 readinessは不変。
別B1 run worktreeは、こちらで書き込んでいない間に `b19c337f0b16513d68b59eb3366dc6eff175c313` から `6fb8a0a71986fda571d82b5658686f5e74a32f8b` へ進み、未追跡資料も追加された。並行更新として別記録し、数値成果の読込・採用・resetは行っていない。
この報告の「実fit等0」は今回の準備作業の実施件数であり、別作業の実施状況を判定した主張ではない。
当worktreeは指定commitから分岐し、他branchをmerge/cherry-pickしていない。既存grant/ledger/lock・科学SOURCE・結果・venv・他jobへ書き込む操作は0。
新SCF/DF/state/signal/reference/trajectory/native build/compile、実H8 fit/残差/再会計、原稿改訂、サブエージェント0。
source/tests・軽量資料を明示stageし、HIROMU1015配下へnon-force pushを試す。
認証失敗時は設定を変更せずローカルcommitを保全し、remote未公開の事実と手動push commandを報告する。
公開またはpush待ちの最終報告でmandatory STOPする。
