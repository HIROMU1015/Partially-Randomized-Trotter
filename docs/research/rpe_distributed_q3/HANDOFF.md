# 分散Q3：開始・停止・再開と統合検収

2026-11-17 JST。公開branchは `codex/rpe-distributed-q3-coordinator`。
固定基点は `5829a7288d28395d2b3e89bfb676c23c690b9ddb`、
**計算SOURCE（controller/worker）は `4ae47cf4844638647c83ea898ee53588675b0d75`**。
**local0 relay・配置helperの補正SOURCEは
`25575ce0f3b43396144c768e9f7dab7dc193beab`**。計算SOURCEを変更せず、
transport補正だけをZIPで配布する。
文書更新後もworkerとcontrollerは計算SOURCE4aeの専用checkoutを使う。
計算SOURCEとlocal0 transportの補正SOURCEを区別して記録する。
local0では初回relayが大量terminalを開いたためSTOP fileで停止済み。
補正版でserver1専用checkoutを準備してから再開し、Windows実機での抑止を確認する。
旧716の待機runはQ3起動0件のまま停止して保全し、4aeの専用runへ分離した。
[REPORT](REPORT.md)／[公開人工検収記録](checks/PUBLIC_ACCEPTANCE.json)。

実装・同一OS上の模擬1/2/3ホスト検収は完了した。
**実機2台はサーバー2のcontrollerを起動して参加待ち。実機2台・3台検収は未完了。**
実H4の新信号・較正・compile・科学Q3は0件で、稼働中の実H4探索へ途中導入していない。
今回の人工fixtureと次run用SOURCEを、既存科学runの条件・承認・STOPへ流用しない。

## 接続方向：local0がqg2とqg1を中継する

controllerはサーバー2で通常 `scripts/run_rpe_schedule_search.py` から1つ起動し、
候補選択、方式別最安、科学予算、結果採用を所有する。
HTTPは `127.0.0.1:8763` にbindする。方式・L_D・幅をホストに固定しない。

利用者が確認した既存経路は **local0 → `ssh qg2`（サーバー2）と
local0 → `ssh qg1`（サーバー1）**。サーバー1→サーバー2の直接SSHは使わない。
local0は計算workerとしての「ローカル」と別ホストであり、今回のrelayだけで
計算用ローカルの接続が確立した扱いにはしない。

`run_rpe_q3_relay.py` はlocal0でqg2へのSSH `-L` を維持し、qg1へ
`--mailbox status/submit/ack` を送る。サーバー1の空きslot・保存状態を取得し、
サーバー2の共通queueから得たassignmentだけを配送する。
サーバー1が完了した成果物は同じ経路でサーバー2へ回収する。
共有filesystem、サーバー間直接接続、qg1へのreverse forwardingを仮定しない。
初版のreverse-forward案はこのmailbox方式で置き換えた。
SSH設定・鍵・共有サービスは変更しない。

## 現在のサーバー2専用人工run

| 項目 | 固定値・記載時の状態 |
|---|---|
| SOURCE checkout | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48` |
| 人工runの親directory | `/home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host` |
| controller PID | `97835`（起動済み。操作前に同じcommandか確認する） |
| controller endpoint | `http://127.0.0.1:8763` |
| slot | server2 local=1、共通total=2 |
| 参加barrier | `required_hosts=["server2","server1"]` |
| 人工負荷 | exact有理数MSE、sleep=1.0秒、2方式・L_D=1,2・複数幅 |
| 記載時のqueue | 未割当QUEUED 4件。server1参加待ち、実機成果物未採用 |

controllerは既に起動している。二重起動せず、まずPIDとlogを確認する。
開始と再開は同じコマンドで、再開時も既存run・DB・spoolを保持する。

```bash
# サーバー2：状態確認
ps -p 97835 -o pid=,args=
tail -n 15 /home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host/run/PROGRESS.jsonl

# サーバー2：controllerが終了した後の再開。初回も同じ入口を使用済み。
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --controller /home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host

# 回収後の集計
/usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --check /home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host
```

人工wrapperは既存の人工producerを渡し、通常schedule CLIの `main()`、
候補順・shot探索・共通executorを使う。人工producer以外の探索入口を作り直していない。
実H4の次runでは固定したproblem/roundを通常の
`run_rpe_schedule_search.py --problem ... --round ... --run-dir ...` に渡す。
その科学実行は今回行っていない。

通常はrelayの新規配送を止め、受信済みjobの完了・回収を待つ。
この専用人工controller自体を中断する必要がある場合だけ、上のcommand確認後に
`kill -TERM 97835` を使う。これはcontrollerを止める操作で、受信済みQ3の停止ではない。
local polling daemon・detached supervisor・spoolを削除せず、同じコマンドでcontrollerを再開する。
固定SOURCEの再開は保存local daemonも照合する。無関係なprocessへ `pkill` 等を使わない。

## サーバー1：固定checkoutとprofileの準備

サーバー1担当が自身の許可済み環境で行う。既存root checkoutのbranchを切り替えない。
サーバー1担当の `d8ee4efe4` は旧89 SOURCEでmailbox statusと単体人工jobを
確認した準備記録で、統合4aeの実機2台検収ではない。**参加前に下記の専用checkoutと
profileを4ae47cf48へ更新すること。** 旧89のspool・結果・準備記録は保全する。
次のworktree作成・profile生成は初回だけで、再開時は同じものを使う。

```bash
# サーバー1：共通branchのobjectsと運用fixtureを取得
cd /home/AbeHiromu/projects/partially-randomized-trotter
git fetch origin codex/rpe-distributed-q3-coordinator

# 未作成の場合に一度だけ実行
 git worktree add --detach \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-worker-4ae47cf48 \
  4ae47cf4844638647c83ea898ee53588675b0d75
mkdir -p /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3
git show origin/codex/rpe-distributed-q3-coordinator:docs/research/rpe_distributed_q3/fixtures/server1.4ae47cf48.json \
  > /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/PROFILE_server1_4ae47cf48.json

# HEADと、人工専用profileの準備状態を確認
 git -C /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-worker-4ae47cf48 rev-parse HEAD
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-worker-4ae47cf48/scripts/run_rpe_q3_worker.py \
  --profile /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/PROFILE_server1_4ae47cf48.json \
  --mailbox status </dev/null
```

[server1 profile](fixtures/server1.4ae47cf48.json) はslot=1で、上記root/interpreter、
専用 `integration-4ae47cf48/spool` と `prefix-cache` を指定する。
人工routeなので `route_views={}` でよい。server1上でHTTP polling daemonを別途重複起動しない。
mailboxはrelayの要求ごとに起動し、`submit` が1回限りのsupervisorをdetachする。
`status` は保存assignmentの照合・未開始assignmentの開始・完了成果物の回収準備を行う。
新しい候補を自分で選んだり、新しいjob IDを作ったりしない。

## local0（Windows PowerShell/cmd）：補正ZIP・server1準備・再開

初回の大量terminal発生後、利用者がSTOP fileで旧relayの `stopped=true` を確認済み。
受信済みQ3は0件で、controller/workerの計算SOURCE4aeと専用runはそのまま保持する。
**同じhome・同じrelay stateへ補正ZIPを展開し、server1配置helperの成功後だけ再開する。**
cmdの場合は `powershell` を起動し、以下を実行する。

補正ZIPはrelay/helper SOURCE25575ce0f、controller/worker SOURCE4aeの組合せ。
`SOURCE.txt` と `COMPUTATION_SOURCE.txt` に両者を記載した。
同梱物はrelay本体、`setup_rpe_q3_mailbox.py`、coordinator helper、
`relay.json`、`server1.profile.json`、操作説明。
[ZIPと検査の記録](checks/LOCAL0_ZIP.json) を参照する。
新しいSSH windowを作らない分岐は人工検査済みで、Windows nativeでの抑止確認は再開待ち。

以下はlocal0の既存Python 3 launcher `py -3`、OpenSSH `ssh` / `scp` と
alias `qg1` / `qg2` を使う。このサーバー2環境からlocal0への実行経路はない。
WSLやBashを追加せず、鍵・認証情報も複製しない。

```powershell
# local0：旧relayはSTOP済み。同じhomeへ補正ZIPを取得・展開する。
$RpeRelayHome = Join-Path $env:USERPROFILE 'rpe-q3-relay-4ae47cf48'
New-Item -ItemType Directory -Force -Path $RpeRelayHome | Out-Null
$RpeRelayZip = Join-Path $RpeRelayHome 'relay.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-coordinator/docs/research/rpe_distributed_q3/LOCAL0_RELAY.zip' $RpeRelayZip
if ($LASTEXITCODE -ne 0) { throw 'Relay ZIP download failed' }
Expand-Archive -LiteralPath $RpeRelayZip -DestinationPath $RpeRelayHome -Force
$RpeRelayScript = Join-Path $RpeRelayHome 'scripts\run_rpe_q3_relay.py'
$RpeRelaySetup = Join-Path $RpeRelayHome 'scripts\setup_rpe_q3_mailbox.py'
$RpeRelayConfig = Join-Path $RpeRelayHome 'relay.json'
$RpeServer1Profile = Join-Path $RpeRelayHome 'server1.profile.json'

# 明示した初回配置：qg1でfetch、専用4ae worktree/profile配置、mailbox status確認。
# helperはok=trueのときだけ終了code0を返す。失敗時はrelayを起動しない。
py -3 -B $RpeRelaySetup --config $RpeRelayConfig --profile $RpeServer1Profile --source 4ae47cf4844638647c83ea898ee53588675b0d75
if ($LASTEXITCODE -eq 0) {
    py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
    Start-Sleep -Seconds 2
    py -3 -B $RpeRelayScript --config $RpeRelayConfig --status
} else {
    throw 'Server1 mailbox setup failed; relay was not restarted'
}
```

helperはqg1の既存正本repositoryをfetchし、専用detached worker viewとprofileを
作成または同一内容で再利用する。既存正本のbranchを切り替えず、他SOURCEや
変更済みview/profileを上書きしない。初回の専用spoolが空である状態でmailbox statusを
確認し、科学的な新候補を選ばず、relayも自動では起動しない。
setupの `ok=true` / `source=4ae...` を確認してから上の条件分岐で再開する。

以降の4操作は同じ設定・stateを使う。

```powershell
# 開始・再接続：保存STOPを消し、qg2 tunnelとqg1配送を開始
py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
# 状態：stopped=false、初期heartbeatとhost別error/最終成功を確認
py -3 -B $RpeRelayScript --config $RpeRelayConfig --status
# 停止：relay/tunnelだけを止める。受信済みQ3は継続する。
py -3 -B $RpeRelayScript --config $RpeRelayConfig --stop
# --statusでstopped=trueを確認し、同じ設定・stateで再接続
py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
```

relay設定はqg2の8763をlocal0の18763へ転送し、qg1上の固定4ae workerとprofileを使う。
state directoryは同梱 `relay.json` を参照する。既存 `STATUS.json`、`EVENTS.jsonl`、
配送intent・STOP以外の保存状態・tunnel logを保持し、job/attemptを作り直さない。
切断中に完了したserver1の結果は、再開時の `status` → controller受理 → `ack` で回収する。
補正版が `stopped=false` を先に保存するため、起動直後の古い停止表示を避ける。
Windows nativeの余分なterminal抑止、server1配置、実SSH配送は利用者の再開後に受入する。

## 計算用ローカルRomuTail：2台受入後に参加

local担当の `codex/rpe-distributed-q3-local`／`4058b48d5935b949929f093381447336ac86714d`
は、利用者確認の **local0 → RomuTail** を記載している。
その実際のSSH接続計測は当セッションから行えておらず、3台での採用検収も未実施。
local0のqg1/qg2 aliasを計算用ローカルへ引き継ぐ必要はなく、local0からmailbox relayする。

RomuTailの既知Linux配置は
`/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local`、
Pythonは `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`。
local担当の現在の固定worker viewは旧89なので、参加前に計算SOURCE4aeの専用viewへ更新する。
今回、local branchのhost DP上書き拡張は統合していない。
workerのDP数はcontrollerの `execution_profile.dp_workers` を継承する。

固定4aeの専用worker viewと `host_id=local` のprofile/spool/cacheをlocal担当が準備し、
新しい3台用人工runでlocal0のrelay設定にRomuTail mailboxを加える。
local0の開始・状態・停止・再接続は同じ4操作を使い、RomuTail自身からqg2へ接続しない。
新規3台用設定のremote root/profile絶対pathは、local担当が4aeへの更新後に確定する。

```bash
# RomuTail：固定viewとprofileを配置した後の状態確認。
# 各変数はlocal担当がそのLinux実機で確認した絶対path。
"$RPE_LOCAL_PYTHON" -B "$RPE_LOCAL_ROOT/scripts/run_rpe_q3_worker.py" \
  --profile "$RPE_LOCAL_PROFILE" --mailbox status </dev/null
```

停止はlocal0のrelayを `--stop` し、受信済みjobはRomuTail内で完了・保存させる。
再開は同じhost ID/profile/spool/cacheとrelay stateで `--start` し、未転送結果を回収する。
3台目用の新しい専用人工runではtotal slot・required_hostsをその構成に設定する。
待機中の2台runのSOURCE・round・科学予算を書き換えない。
HTTP直結workerの `--drain` 等は既存CLIとして保持するが、今回の実機接続はmailboxを使う。

## 固定入力・ID・永続状態

各ホストで固定4ae47cf48のHEADとsource manifestを照合する。実routeを使う次runでは
各 `view_commit` のdetached view、入力locatorが参照するGit objects、必要な大入力を
既存の許可済み転送経路で配置する。workerは固定locator blobのSHA-256、route HEAD、
既存adapterの評価契約・metadata gateを検査する。選択されたconfigurationのinput hashを
controller内でcacheし、無関係な全候補のblobを各jobで再読込しない。
科学SOURCE/input hashと、利用者の承認の証明を混同しない。

[worker profile例](fixtures/worker.example.json) の `route_views` は
commit→ホスト別絶対path、`interpreter` と `prefix_cache_directory` もホスト別。
REQUEST原本は保持し、path解決は `RESOLVED_REQUEST.json` へ分離する。
DP並列度は既存 `execution_profile.dp_workers`、prefix checkpointも既存設定を使う。
DP cacheはホストごとに保持し、同じprefixの局所性は先頭候補の優先順位を変えず、
空きslot・短い待ち時間を考慮して割当先を選ぶ。

科学job IDは既存の方式/configuration/shot vector/route kind/評価契約のkey。
実行host IDとattempt IDは別欄で、hostが変わったことを新科学評価の根拠にしない。
controllerの `run/distributed/QUEUE.sqlite3` が割当・予算予約・採用を永続化し、
同時claimをtransactionで排他する。queue、`SOURCE.json`、transfer stage、
各jobのREQUEST/LAUNCH/OUTCOME/PROCESS_RETURN、workerのASSIGNMENT/WORKER_STATEを保持する。
未割当QUEUEDのキャンセルと、割当済みattemptの放置・再投入を区別する。
`STARTING` / `UNKNOWN` は曖昧な起動境界であり、自動executeし直さない。

## HTTPとmailboxのprotocol v1

- `POST /v1/poll`: `host_id, slots, cached_prefixes, source_sha`。空きslotを申告し、
  `assignments` を得る。各assignmentは `job_id, method, attempt_id, request,
  request_sha256, source_manifest, input_hashes, prefix_key`。
  required_hosts未登録時は `waiting_for_hosts` を返す。
- `POST /v1/state`: 同じhost/job/attemptと `state, detail`。
  `ASSIGNED`（未起動）、`STARTING`、`RUNNING`、`UNKNOWN`、
  `COMPLETED_PENDING_TRANSFER`（完了・未転送）を区別する。
- `POST /v1/complete`: 同じhost/job/attempt、`request_sha256`、
  relative path→base64の `files`。controllerは親子processの終了code/signalと
  元のidentity/契約gateを確認し、成果物を永続化して `ok, duplicate` を返す。
  完了通知の再送でも予算・結果採用は1回。
- `POST /v1/transfer-metric`: 同じownerと `timing`。HTTP workerのupload/ACK往復、
  relayのSSH status転送時間を観測する。検証保存時間とnetwork往復を分ける。
- mailbox `status` はhost/source/空きslot/cacheと保存attemptを返し、完了時は成果物も返す。
  `submit` は同じassignmentを受理して一度だけ実行し、`ack` は回収確認を保存する。
  relayだけがこれをcontroller APIへ橋渡しする。

HTTP timeoutやSSH切断によるlease expiryはない。応答消失だけで別ホストへ再投入しない。
再接続では同じhost・spool・job・attemptを照合する。
adapterは既存 `run_rpe_optimizer_q3_job.py` の `dry-check / execute / collect` を使い、
回収だけの経路から新しい科学評価を呼ばない。

## 人工検収の再現と実機受入の残作業

固定SOURCE checkoutでfresh directoryを指定する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --acceptance /tmp/rpe-distributed-q3-artificial-NEW
```

同一OS上の別spool/HTTP clientによる1/2/3模擬ホスト、legacy単一参照、
MSE上下界/判定/方式別最安一致、途中READY、通常再開、切断中完了、
controller kill/restart、同attempt回収、重複完了受理を検査する。
人工sleepの速度倍率を実H4の性能として報告しない。

実機は次の順で受入する。現在の待機runを実機PASSとして扱わない。

1. local0 relayを開始し、server2+server1の双方で異なる人工Q3が完了する。
2. サーバー2の同じ方式別最安へ両ホストの結果を採用し、次候補がserver1へ再配送される。
3. relay切断中も受信済みQ3が継続し、再開時に同じattemptの結果を回収する。
4. 新しい専用runに計算用ローカルを加え、同じ参照一致と再開0件を検査する。
5. host別の起動/完了/結果再利用、queue待ち、転送、全体実時間、実際の接続構成を追記する。

担当ごとの実機記録は `hosts/<host_id>/HANDOFF.md` へ残し、共通branchへforce-pushしない。
SOURCE/生結果/STOPを保持し、新科学runの許可をこの人工検収から拡張しない。
