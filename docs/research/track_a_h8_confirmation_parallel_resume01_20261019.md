# Track A H8：受入修正・control/両軸並列化・resume01

2026-10-19T00:05:19.254965+09:00 の起動snapshot。**RUNNING、科学結果は未完了**。
SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8`。元dataset/seed SOURCEは `1549b6f891a15ff48379996c34b344118ba2606c` を保持する。
元run01と旧STOPを保全した独立branchで修正し、利用者の「修正と並列化を行って再実行」の指示を個別grantへ結合した。

## 修正理由と保全

元p8は代表1軌道の両軸compileとcontrolが完了しworker exit0・reap済みだったが、
JSONのidentity listと生handleのtupleを比較した`NATIVE_START_IDENTITY`でdriverがSTOPした。
4要素が全てintであることを検査してからtupleへ揃え、値・PID・UID・taskの一致を要求するよう修正した。
所有不一致・異常exit・未回収・欠測を成功扱いする変更ではない。
元のSTOPを遡及的にPASSへ書き換えず、保存p8の完全なnative証跡を別受入契約で再利用する。

元p7は代表軌道の途中だったため、所有driverをPID/starttime/UID/command/pidfdで照合しSIGTERMした。
旧witnessのnative wait、worker全回収・2回の同一identity残存なしを確認。旧STOP・未完了予約・イベント記録を保持した。
新p7代表は元seedとevent digestを要求し、抽選し直した別の標本へ置き換えない。

## 並列化

同一trajectoryでevolutionを一度構築し、forkのcopy-on-writeで共有する。QPY輸送・巨大circuitのpickleは使わない。
low/high branch・cosine/sine検査・event oracleの5処理を所有processで並列計算し、最後に元と同じ1e-8 toleranceで照合する。
control4probe＋oracle1という科学call数を維持し、必要な検査は省略しない。
cosine/sine full-wrapper compileは別の2processで同時実行し、元event列・compiler・fingerprint・gate metricsを検査する。

CPUを2coreずつtrajectory groupへ割り当て、計算を行うaxis childは1group最大2つ。待機するtrajectory親もnative identityで管理する。
最初のcontrolだけ最大5coreを使い、合格したscopeから残りを投入する。2候補のgrant/output/ledger/STOPは独立。
全ての階層でnative wait4・exit0・result SHAを受入条件とし、異常時はown descendantsのみ停止・回収する。

| 固定scope | 元48の完全受入 | 新trajectory | 新compile | CPU slots | trajectory groups |
|---|---:|---:|---:|---:|---:|
| H8_CONFIRM_P8_Q2 |1|47|94|8|4|
| H8_CONFIRM_P7_Q16 |0|48|96|50|25|

元p8の2wrapper＋新190wrapperで固定96軌道・192wrapperの完了を目指す。
旧p7には未完了の2wrapper予約があり、これを返却せず累積task-level予約上限は194となる。
親のadmission予約とtaskのbefore-call予約は同じ呼出しの二重chargeとして加算しない。
旧制御の未完了予約も保全し、carry reset・失敗費用返却は0。以後の自動retry・標本追加・候補変更は行わない。

## 科学条件と限界

linear H8/1Å/STO-3G/tol-only1e-8 DF28/16spin modes/sector4900/T=.8/P=0。
p8/q2/r4/K4、ε=.05、固定PF B0I p11/q1 S2と、p7/q16/r4/K4、ε=.0005、固定PF B0I p15/q4 S4。
元DF順序・S2・有限RTE law・full-Fock scalar一回適用・reference不回転・ordinary ancilla・symmetric-directional controlを保持。
compiler opt1/seed17/rz,sx,x,cx、元preparationと保存shotを再利用し、SCF/DF/state/signal/fit/新PF compileは0。
旧profile原file byte一致と旧compiler output完全同一性は未検証という限界を保持し、現在のruntime/providerはbyte固定した。

新しいseed導出は0。元の独立確認96masterと派生seedを固定し、旧探索n8・旧確認n32・条件付きpilotとは混合しない。
48本全ての受入後だけ、通常signed bias/倍率1を主評価、同じεの経験的非相殺を記述的副評価として保存shot込み会計する。
p8の固定PFは非相殺では不適格のため副評価のPF比較をnullとする。対照の自動差替え・新ランキング・shot再最適化はしない。
点費用再現と母平均優位未確定を分離し、未観測希少費用を0補完しない。

人工22検査PASS（旧18含む）。生tuple/JSON list、無効identity、owned control/axes fork、native wait4、失敗cleanup、元partial event固定を検査した。
人工compile0、作者の技術検査であり独立科学reviewではない。
起動前のSOURCE11closure・input/provider/runtime・CPU/cgroup memory/OOM/pressure・block/inode・未使用output/lockのgateはPASS。
mem admissionは待機親＋axis child2つのRSS見積りとcontext/headroomから再見積りし、強制AS/RSS/wall capは追加しない。
quotaはsiteで非公開、無制限と断定せずENOSPC/EDQUOTをSTOPとして保全する。PSI/一時的競合/無関係worktree更新で強制停止しない。
共有環境・既存venv・他job・他研究・GPUは変更していない。

[closure・検査・起動snapshot](../../artifacts/resource_applicability/track_a_h8_confirmation_parallel_resume01/2026-10-19/README.md)。
[並列native処理](../../scripts/h8_pr_confirmation_parallel.py)／[受入・継続sealer](../../scripts/prepare_h8_pr_confirmation_resume.py)／[実行driver](../../scripts/h8_pr_confirmation.py)／[人工tests](../../tests/test_h8_pr_confirmation.py)。
private運用原本、seed値、raw入力、有効grant、実ledger・runtimeはcommitしない。

次の担当：Codexの固定runnerが完了またはfail-closed STOPまで継続し、限定結果を保存してmandatory STOPする。
利用者の追加操作は現時点では不要。
