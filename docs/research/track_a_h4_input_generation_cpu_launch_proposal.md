# H4入力生成 CPU候補・own-run起動条件・最終承認案

2026-10-06 JST。`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`。
**利用者承認未取得。CPU使用許可、独立最終review、明示launchは未実施。今回は準備だけでSTOPする。**

提案は **CPU [3,5,6,7,8,9]、requested workers6**。
OS topologyのpackage0/core3,5,6,7,8,9という異なる6 physical coresを選び、全てNUMA node0。
観測上の各SMT sibling集合はsingletonで、同じcoreからlogical CPUを重複選択していない。
science sourceとplan v2は変更せず、auth草案のallowed_cpusだけをこの候補に変更する。
reviewはauthorization_digestを再計算し、approved=falseのまま。

入口は[proposal bundle](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)と
[一括review/承認判断](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/FINAL_APPROVAL_REQUEST_v1.md)。

## 選定根拠と観測範囲

remote branchは起点 `a1b0ba5e1ae14c2fd7c345a4d979b85b1eff2538` と一致した。
SOURCE `9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a`、旧manifest42 members、source19 paths、
plan/source auditのbytes・hash・fingerprint、依存45/installed source11/compiler metadataを照合した。
既存science checkoutはそのまま使用し、proposal worktreeへ移さない。

CPU/load観測は2026-10-06 **16:05:51.914–16:05:54.915 JST** の約3.001秒。
online/process CPU集合0–255と、可視cpuset有効集合0–127の共通部分から選んだ。
所属leafとuser sliceには親でcpuset controllerが有効化されておらず、上位user.slice/rootの有効cpusetを読み取った。
cpusetが有効なのに必須interfaceが欠測ならSTOPする。OS上の利用可能性を許可・専有予約とみなさない。

全online CPUのpackage/core/SMT sibling/NUMAを読み、physical coreごとに代表logical CPUを一つ選ぶ。
負荷は/proc/statの二時点差から得る受動サンプルで、benchmark・worker起動ではない。
guest/guest_niceの二重計数を避けて最初の8 counterを使用し、idle/iowaitをbusyから除く。
coreは全online SMT siblingの最大busy・合計busyで保守的に比較し、同一NUMA内の6 coreを選んだ。
候補の全6 coreのbusyは当該sampleで0%。CPU0–5を固定採用したものではない。
50%未満を低負荷候補の運用heuristicとして使ったが、科学条件やproduction resource guardには追加していない。
短時間の0%は将来の空き・専有・性能保証ではない。sample/raw counters、全CPU topologyと比較NUMA候補をJSONへ保存した。

## resource/context観測・未解決容量

既存observerで初期cgroup namespace `cgroup:[4026531835]`、全非root祖先可視性、memory/pressure/OOMをread-only確認した。
実効availableは約981.887GiB、PSI full avg10=0、各非root OOM counter0。
準備時の成功であり、6 workerのfresh launch admissionやCPU使用許可ではない。
現在のprocess CPU集合は候補6個より広い。将来、新規own-runだけにtaskset maskを適用して既存subset guardを満たす必要がある。

固定outputはまだアクセス・作成していない。未作成の直近親directoryを除き、既存の上位`artifacts/`からfilesystemを照合した。
mountは `/home`、ext4、観測時のavailableは **3,991,597,056 bytes ≈3.717GiB**。
総output上限10GiBより少ないため、全上限分を確保する保守的launch容量条件は未達。
10GiBはgeneration＋将来map全体の上限で、6入力の実生成量ではない。今回実出力量を測定していない。
容量・quotaのfresh確認と条件解決を独立reviewへ送る。領域削除・output移設・共有設定変更は行わない。

観測attempt01は未作成親directoryへのstatvfsで失敗した。
attempt02は全output cap分の空き要求で停止した。
attempt03では容量値と未解決条件を保存した。失敗logを上書きせず全attemptを保持する。
production source/guardの修正は一切していない。

## 未実行launch案・承認対象

既存 `/usr/bin/taskset --cpu-list 3,5,6,7,8,9` で**自分の新規runだけ**を起動する文面を用意した。
既存PIDを指定するtaskset --pidは使わない。科学runnerのspawned workersは同じCPU maskを継承する提案で、今回は実起動で検証していない。
taskset、production runner、worker、affinity/priority/cgroup変更は今回実行していない。
利用Pythonは固定absolute interpreter、内部thread/process各1、package/environment変更なし。
初期namespaceと祖先可視性、process_cpus<=allowed_cpus、fresh memory/pressure/OOM/admissionは凍結sourceのまま検査する。

planのrequested workers6を維持し、actual wは既存admissionで決定され6以下。
6 workerに必要なavailable72GiB、driver/worker AS8GiB/RSS別/headroom16GiB、wall72h/output10GiBを維持する。
candidate/auth/source/guardを結果によって変更せず、failure/interruptionはSTOP、no retry/resume。
固定Python・science checkout・plan/auth/review絶対pathを[将来command](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/FUTURE_LAUNCH_COMMAND_v1.md)に記録する。

利用者に判断してもらう具体的事項：

> CPU [3,5,6,7,8,9]、6 worker、提示したown-run限定起動条件で、H4の6入力生成を一度行い、freeze後STOPすることを承認するか。

この文面は承認依頼案で、承認の記録ではない。
CPU/起動条件承認だけで独立reviewや明示launchを代行しない。
容量・fresh resource条件の解決、独立最終review、適切なreview文書更新、利用者の明示launchが揃うまで実行しない。

## 検査・不変性・STOP

12 metadata gate tests PASS、fail/error/skip0。
保存review=falseの拒否、CPU候補のtopology/NUMA、plan bytes/source19不変、auth/review digest、
旧auth/review混在・CPU不正・proposal rootへsource_root変更・別stage/permission/no launchの拒否を検査した。
合格authorize＋metadata-only checkoutはメモリ内review承認だけ。承認済みreviewとして保存しない。
production launch/科学データ/output/registry/worker境界はmock禁止、guardカウンタとprotected attemptsは0。

追加source/helper/test moduleをproduction repositoryへ作らず、一時read-only手順・testsを文書へ記録する。
commit対象は文書/軽量JSON/logと索引追記だけ。新science source commitは作成しない。
H4 linear/neutral singlet/STO-3G、6距離、4 spatial/8 system、DF fragments12、SCF/DF/order/solver/gates/master seed、218 templatesを不変に保つ。
入力生成6件freeze後STOP、inputs/freeze digestはnull、signal/compileと将来74,784 wrappersは未認可。
分子アクセス/生成、signal/seed/trajectory/build/compile/transpile、GPU、本番起動、共有環境・他job変更0。
追加transpile0、旧累積28/64、旧bundle/証拠/原稿/Track B/既存worktreeは不変。
このlocal metadata資料はCPU許可・予約・実性能・CI/独立外部再現・実行準備完了を示さない。
公開後も `H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL` でSTOPする。
