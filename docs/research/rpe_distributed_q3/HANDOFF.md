# 分散Q3：開始・停止・再開と統合検収

2026-11-17 JST。公開branchは `codex/rpe-distributed-q3-coordinator`。
固定基点は `5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
**次回公開版の固定SHAは `28d976b99cdf8329adcc1b79327e329e947ef774`**。
完了した実2台は計算SOURCE4ae／relay25575、完了した実3台は計算SOURCE1ff。
実機fault runは計算SOURCE `1ffdf610a682a23e2a0a388b570d33d0f353fedc` を保持し、
transport/観測toolsに28dを使う。完了runへ途中導入せず、SOURCE・結果・attemptを保全する。
[REPORT](REPORT.md)／[実2台要約](checks/REAL_TWO_HOST_PUBLIC_SUMMARY.json)／
[実3台要約](checks/REAL_THREE_HOST_PUBLIC_SUMMARY.json)。

**通常実2台・3台は全18件参照exact一致・全host採用・完了再開0件でPASS。**
**専用実機faultも、通信断中の継続・同attempt回収・controller再開・重複通知を含めPASS。**
[実機fault受入](checks/REAL_FAULT_ACCEPTANCE.json) に根拠を保存した。
fault relayは最後に停止済みで、この検収のための追加利用者操作は不要。
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

## 完了した実2台・3台構成

実2台はserver2+server1、local0のWindows SSH mailbox relay経由で完了した。
[受入記録](checks/REAL_TWO_HOST_ACCEPTANCE.json) と
[timeline](checks/REAL_TWO_HOST_TIMELINE.jsonl) を保持する。
server1は起動/完了/採用5件、server2は13件。全18件のMSE上下界・判定・方式別最安は
単一参照とexact一致し、完了runの再開追加起動0件を確認した。
server1への後続割当も5回あり、方式をhostに固定していない。

| 項目 | 完了した2台run |
|---|---|
| 計算SOURCE | `4ae47cf4844638647c83ea898ee53588675b0d75` |
| SOURCE checkout | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48` |
| runの親directory | `/home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host` |
| 初回controller PID | `97835`（履歴上の値。完了後の停止操作には使わない） |
| slot | server2=1、server1=1、total=2 |
| 開始済みjobの区間 | 最初のLAUNCH〜最後のEXIT 21.262秒 |
| controller開始からの区間 | 1107.508秒。local0準備と参加barrier待ちを含む |
| 切断検収 | この通常runには含めない。別SOURCE固定の専用faultでPASS |

この実機runから速度比は算出していない。ACK往復は17/18件の通知だけ取得し、
最後の1件が欠測。詳細とqueue待ちの定義はREPORTに記載した。

実3台runはSOURCE `1ffdf610a682a23e2a0a388b570d33d0f353fedc` で完了した。
[受入記録](checks/REAL_THREE_HOST_ACCEPTANCE.json) と
[timeline](checks/REAL_THREE_HOST_TIMELINE.jsonl) を保持する。
server2=11件、server1=2件、RomuTail=5件を起動・完了・採用した。

| 項目 | 完了した3台run |
|---|---|
| SOURCE checkout | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-1ffdf610a` |
| runの親directory | `/home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/three_host` |
| 初回controller / supervisor PID | `142163` / `142162`（履歴値） |
| endpoint / local0転送 | `127.0.0.1:8763` / `127.0.0.1:18764` |
| required_hosts | `server2`, `server1`, `local` |
| Q3 slots | server2=1、server1=1、RomuTail=2、total=4 |
| host DP workers | server2=1、server1=1、RomuTail=2 |

1ffは巨大なexact整数の保存・再読込と、local4058のhost別DP設定を統合した。
元REQUEST/hash/科学job IDを保ち、実行profileでhost別DP数を解決する。
完了した4aeのSOURCE、round、予算、spoolを書き換えて1ffへ移行しない。

2台runの集計・追加0確認を再現する場合は固定4aeと元runを使う。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48
/usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --check /home/AbeHiromu/projects/rpe-distributed-q3-integration-4ae47cf48/two_host
```

人工wrapperは既存の人工producerを渡し、通常schedule CLIの `main()`、
候補順・shot探索・共通executorを使う。
実H4の次runで使う通常入口は `run_rpe_schedule_search.py --problem ... --round ... --run-dir ...`。
今回の人工実機検収を、新規H4科学実行の許可へ広げない。

## 完了した専用実機fault run

[準備記録](checks/FAULT_PREPARATION.json) のscopeは人工Q3計2件、各sleep12秒。
準備時のPENDINGは履歴として保ち、[最終受入](checks/REAL_FAULT_ACCEPTANCE.json) でPASSを確認した。
実3台の通常runとは別のrun・host ID・spoolで切断を検査する。

| 項目 | fault run |
|---|---|
| 計算SOURCE | `1ffdf610a682a23e2a0a388b570d33d0f353fedc` |
| relay/inspector SOURCE | `28d976b99cdf8329adcc1b79327e329e947ef774` |
| runの親directory | `/home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/fault_two_host` |
| 初回controller / supervisor PID | `164874` / `164873`（中断前の履歴値） |
| endpoint / local0転送 | `127.0.0.1:8763` / `127.0.0.1:18765` |
| required_hosts | `server2`, `server1-fault` |
| 予算 | 方式ごと1件、計2件の人工Q3 |
| 最終状態 | 人工2件完了、同attempt回収、重複通知受理、完了再開追加0件 |

**完了3台用relayは停止してからfault relayへ切り替える。**
required_hostsは登録barrierであり、同じSOURCEの旧pollerへqueueを制限するものではない。
旧3台のstate/spoolを保持し、faultは `server1-fault` の専用profile/spoolを使う。
[LOCAL0_RELAY_FAULT.zip](LOCAL0_RELAY_FAULT.zip) に固定28dのrelay・setup・
読み取り専用 `inspect_rpe_q3_mailbox.py` とfault profileを同梱した。
local0のfault転送portは18765、stateは `~/rpe-q3-relay-state-1ffdf610a-fault`。

28dは接続不可aliasへのretry backoff、完了後の保存log維持、読み取り専用inspectorを追加する。
inspectorはmailbox `status` を呼ばず、ASSIGNED仕事も開始しない。
切断中のRUNNING→完了未転送を観測し、relay再開後の同attempt回収・二重採用0を確認する。
faultは通常2台・3台から独立に実施してPASSした。
server1-faultはRUNNING→COMPLETED_PENDING_TRANSFER→TRANSFERREDを観測し、
同じjob/attemptのexecuteは1回、転送前後とcontrollerのOUTCOME byteは一致した。
controllerの意図的SIGKILL後も同SOURCE/runから再開し、終了code0を確認した。
同じ完了payloadの再送はok=true/duplicate=trueで、予算・ACCEPT・LAUNCHは不変だった。
直接ADOPTイベントはserver2の1件。remoteはPRIOR_ROUND_JOB/RESULT_REUSE経由で
PR_DFの方式別bestへ取り込まれ、同じ直接ADOPTイベントには数えていない。


以下は実施済みfaultの2段階操作の記録。現在relayは停止済みで、再実行は不要。
最初の段階ではrelay停止中の保存状態を取得し、
その記録を見てcontroller中断・再開へ進む。sleepだけでRUNNING/完了を認定しない。

```powershell
# local0：先に完了3台用relayを停止する。既存spool/stateを削除しない。
$RpeThreeHome = Join-Path $env:USERPROFILE 'rpe-q3-relay-1ffdf610a-three'
py -3 -B (Join-Path $RpeThreeHome 'scripts\run_rpe_q3_relay.py') --config (Join-Path $RpeThreeHome 'relay.json') --stop
Start-Sleep -Seconds 2
py -3 -B (Join-Path $RpeThreeHome 'scripts\run_rpe_q3_relay.py') --config (Join-Path $RpeThreeHome 'relay.json') --status

$RpeFaultHome = Join-Path $env:USERPROFILE 'rpe-q3-relay-1ffdf610a-fault'
New-Item -ItemType Directory -Force -Path $RpeFaultHome | Out-Null
$RpeFaultZip = Join-Path $RpeFaultHome 'relay-fault.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-coordinator/docs/research/rpe_distributed_q3/LOCAL0_RELAY_FAULT.zip' $RpeFaultZip
if ($LASTEXITCODE -ne 0) { throw 'Fault ZIP download failed' }
Expand-Archive -LiteralPath $RpeFaultZip -DestinationPath $RpeFaultHome -Force
$RpeFaultRelay = Join-Path $RpeFaultHome 'scripts\run_rpe_q3_relay.py'
$RpeFaultSetup = Join-Path $RpeFaultHome 'scripts\setup_rpe_q3_mailbox.py'
$RpeFaultInspect = Join-Path $RpeFaultHome 'scripts\inspect_rpe_q3_mailbox.py'
$RpeFaultConfig = Join-Path $RpeFaultHome 'relay.json'
$RpeFaultProfile = Join-Path $RpeFaultHome 'server1.profile.json'

py -3 -B $RpeFaultSetup --config $RpeFaultConfig --profile $RpeFaultProfile --source 1ffdf610a682a23e2a0a388b570d33d0f353fedc
if ($LASTEXITCODE -eq 0) {
    py -3 -B $RpeFaultRelay --config $RpeFaultConfig --start
    Start-Sleep -Seconds 8
    py -3 -B $RpeFaultRelay --config $RpeFaultConfig --stop
    Start-Sleep -Seconds 2
    py -3 -B $RpeFaultRelay --config $RpeFaultConfig --status
    py -3 -B $RpeFaultInspect --config $RpeFaultConfig --profile $RpeFaultProfile | Out-File -Encoding utf8 (Join-Path $RpeFaultHome 'INSPECT_AFTER_STOP.json')
    Start-Sleep -Seconds 15
    py -3 -B $RpeFaultInspect --config $RpeFaultConfig --profile $RpeFaultProfile | Out-File -Encoding utf8 (Join-Path $RpeFaultHome 'INSPECT_AFTER_WAIT.json')
} else {
    throw 'Fault mailbox setup failed; relay was not started'
}

# snapshotを確認してサーバー2担当へ回収する。ここではrelayをまだ再開しない。
Get-Content -Raw (Join-Path $RpeFaultHome 'INSPECT_AFTER_STOP.json')
Get-Content -Raw (Join-Path $RpeFaultHome 'INSPECT_AFTER_WAIT.json')
ssh qg2 'mkdir -p /home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/fault_two_host/local0'
scp (Join-Path $RpeFaultHome 'INSPECT_AFTER_STOP.json') (Join-Path $RpeFaultHome 'INSPECT_AFTER_WAIT.json') 'qg2:/home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/fault_two_host/local0/'
```

inspectorのsnapshotで対象のjob/attempt、RUNNINGまたは起動済みの観測、
完了未転送への遷移、execute count=1と終了codeを確認する。
空のattemptや未知の状態をPASSとせず、受信したassignmentを手動executeし直さない。
その後サーバー2担当が専用controllerの中断・同SOURCE/runからの再開を記録する。
再開後は次の同じrelay状態で回収し、same attempt・新規起動0・採用1回を検査する。

```powershell
# サーバー2担当がcontroller再開を確認した後に実行
py -3 -B $RpeFaultRelay --config $RpeFaultConfig --start
Start-Sleep -Seconds 2
py -3 -B $RpeFaultRelay --config $RpeFaultConfig --status
```

## 実2台で使用したhost配置と操作の記録

完了した実2台は計算SOURCE4ae、relay/helper SOURCE25575を使った。
server1の固定workerは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-worker-4ae47cf48`、
profileは `.server-preparation/rpe-distributed-q3/PROFILE_server1_4ae47cf48.json`、
Pythonは `/home/AbeHiromu/venvs/trotter-common/bin/python`。
local0の配置helperでfetch・専用view/profile設置・mailbox statusを確認してからrelayを開始した。
server1→server2直接SSHは使わず、local0からqg2/qg1へ接続した。

初版relayのterminal多重発生をSTOP fileで停止し、SSH子processの
`CREATE_NO_WINDOW` と起動直後statusを補正した。
4ae runのSOURCEを保って25575のrelay ZIPへ差し替えた後、通常2台探索は完了した。
初版停止時は新Q3起動0件だった。旧89、716、4aeのview・spool・記録は保全する。

当時のlocal0 homeは `%USERPROFILE%\rpe-q3-relay-4ae47cf48`。
開始/状態/停止/再開は `run_rpe_q3_relay.py --config relay.json` の
`--start / --status / --stop / --start` を使う。
停止はrelay/tunnelだけで、受信済みsupervisorをkillしない。
現在の配布ZIPは新3台用へ更新するため、完了4aeの再開には既存25575配置を保持する。
新ZIPを旧runのSOURCE・profileと混ぜて再起動しない。

## 実3台で使用したlocal0配置手順（履歴）

次は実3台で通った手順の履歴。ZIPは `LOCAL0_RELAY_THREE.zip`。SOURCE.txtは1ff、profileとstateは
完了4aeの2台runと分離する。local0はWindows PowerShellを使う。
cmdの場合は `powershell` を起動してから実行する。
[3台用配布記録](checks/THREE_HOST_PREPARATION.json) にZIP hashと構成を記録した。

```powershell
# 完了した2台用relayだけを停止。旧配置・spoolは削除しない。
$RpeOldHome = Join-Path $env:USERPROFILE 'rpe-q3-relay-4ae47cf48'
$RpeOldScript = Join-Path $RpeOldHome 'scripts\run_rpe_q3_relay.py'
if (Test-Path $RpeOldScript) {
    py -3 -B $RpeOldScript --config (Join-Path $RpeOldHome 'relay.json') --stop
}

# 新3台用は別home・別state。既存のqg2/qg1/RomuTail aliasを使う。
$RpeRelayHome = Join-Path $env:USERPROFILE 'rpe-q3-relay-1ffdf610a-three'
New-Item -ItemType Directory -Force -Path $RpeRelayHome | Out-Null
$RpeRelayZip = Join-Path $RpeRelayHome 'relay-three.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-coordinator/docs/research/rpe_distributed_q3/LOCAL0_RELAY_THREE.zip' $RpeRelayZip
if ($LASTEXITCODE -ne 0) { throw 'Three-host relay ZIP download failed' }
Expand-Archive -LiteralPath $RpeRelayZip -DestinationPath $RpeRelayHome -Force
$RpeRelaySetup = Join-Path $RpeRelayHome 'scripts\setup_rpe_q3_mailbox.py'
$RpeRelayScript = Join-Path $RpeRelayHome 'scripts\run_rpe_q3_relay.py'
$RpeRelayConfig = Join-Path $RpeRelayHome 'relay.json'
$RpeServer1Profile = Join-Path $RpeRelayHome 'server1.profile.json'
$RpeLocalProfile = Join-Path $RpeRelayHome 'local.profile.json'

# helperの既定SOURCE4aeを使わず、両hostへ新1ffを明示する。
py -3 -B $RpeRelaySetup --config $RpeRelayConfig --profile $RpeServer1Profile --source 1ffdf610a682a23e2a0a388b570d33d0f353fedc
if ($LASTEXITCODE -eq 0) {
    py -3 -B $RpeRelaySetup --config $RpeRelayConfig --profile $RpeLocalProfile --repository '/home/abe/Project/Partially Randomized Trotter' --source 1ffdf610a682a23e2a0a388b570d33d0f353fedc
    if ($LASTEXITCODE -eq 0) {
        py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
        Start-Sleep -Seconds 2
        py -3 -B $RpeRelayScript --config $RpeRelayConfig --status
    } else {
        throw 'RomuTail mailbox setup failed; three-host relay was not started'
    }
} else {
    throw 'Server1 mailbox setup failed; three-host relay was not started'
}
```

helperは両hostの既存repositoryをfetchし、新1ffの専用detached view・専用profileを
設置してmailbox statusを確認する。既存view/profileを異なる内容で上書きしない。
両helperの `ok=true` / `source=1ff...` を確認してからrelayを開始する。
新stateは `~/rpe-q3-relay-state-1ffdf610a-three`、local0転送portは18764。
server1 profileは `PROFILE_server1_1ffdf610a_three.json`、worker rootは
`.server-preparation/worktrees/rpe-distributed-q3-worker-1ffdf610a`。
RomuTail profileと固定viewは次節の配置を使う。

普段の開始・状態・停止・再接続は、同じPowerShell変数を使う。

```powershell
py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
py -3 -B $RpeRelayScript --config $RpeRelayConfig --status
py -3 -B $RpeRelayScript --config $RpeRelayConfig --stop
# --statusでstopped=trueを確認してから再接続
py -3 -B $RpeRelayScript --config $RpeRelayConfig --start
```

停止はrelay/tunnelだけで、受信済みQ3は各host内で完了し未転送状態を保存する。
再開時に同じjob/attemptを回収する。実機切断・復帰は通常3台完了後の
別の専用fault runで実施してPASSした。通常完了とfaultの記録は分離している。

サーバー2の通常開始・再開コマンドも同じ新runを使う。このcontrollerは通常完了済みで、
同じrunの追加起動0件も確認済み。新しいfault runと混ぜない。

```bash
# server2：完了runの再開を再現する場合だけ。同じSOURCEとrunを保持
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-1ffdf610a
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --controller /home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/three_host
```

## 計算用ローカルRomuTail：実3台で使用した配置

local担当の `codex/rpe-distributed-q3-local`／`4058b48d5935b949929f093381447336ac86714d`
は、利用者確認の **local0 → RomuTail** を記載している。
local0から実際に接続し、1ff固定viewで人工5件の完了・採用と再開追加0件を確認した。
local0のqg1/qg2 aliasを計算用ローカルへ引き継ぐ必要はなく、local0からmailbox relayする。

RomuTailの既知Linux配置は
`/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local`、
Pythonは `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`。
local担当の旧89固定viewは保持し、実3台では計算SOURCE1ffの専用viewを使った。
1ffではlocal branchのhost DP上書きを統合した。
workerは元REQUESTを保ち、host profileから実DP数を解決する。

固定1ffの専用worker viewと `host_id=local` のprofile/spool/cacheを配置し、
完了した3台用人工runでlocal0のrelay設定にRomuTail mailboxを加えた。
local0の開始・状態・停止・再接続は同じ4操作を使い、RomuTail自身からqg2へ接続しない。
3台用fixed worker viewは上記checkout内 `.local-q3/common-1ffdf610a`、
profileは `.local-q3/worker_three_1ffdf610a.json`。新ZIPの配置helperがそれぞれ準備する。

```bash
# RomuTail：固定viewとprofileを配置した後の状態確認。
# 各変数はlocal担当がそのLinux実機で確認した絶対path。
"$RPE_LOCAL_PYTHON" -B "$RPE_LOCAL_ROOT/scripts/run_rpe_q3_worker.py" \
  --profile "$RPE_LOCAL_PROFILE" --mailbox status </dev/null
```

停止はlocal0のrelayを `--stop` し、受信済みjobはRomuTail内で完了・保存させる。
再開は同じhost ID/profile/spool/cacheとrelay stateで `--start` し、未転送結果を回収する。
3台目用の新しい専用人工runではtotal slot・required_hostsをその構成に設定する。
完了した2台runのSOURCE・round・科学予算を書き換えない。
HTTP直結workerの `--drain` 等は既存CLIとして保持するが、今回の実機接続はmailboxを使う。

## 固定入力・ID・永続状態

各hostはそのrunに固定したHEADとsource manifestを照合する
（完了実2台は4ae、完了実3台は1ff、次回公開版は28d）。実routeを使う次runでは
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

## 人工検収の再現と次のrun

固定SOURCE checkoutでfresh directoryを指定する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-1ffdf610a
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B scripts/check_rpe_distributed_q3.py \
  --acceptance /tmp/rpe-distributed-q3-artificial-NEW
```

同一OS上の別spool/HTTP clientによる1/2/3模擬ホスト、legacy単一参照、
MSE上下界/判定/方式別最安一致、途中READY、通常再開、切断中完了、
controller kill/restart、同attempt回収、重複完了受理を検査する。
人工sleepの速度倍率を実H4の性能として報告しない。

通常実2台・3台では全host採用・参照exact一致・完了再開0件を確認済み。
実機faultではrelay停止中の継続と完了未転送、controllerの意図的中断後の同SOURCE再開、
同attempt回収、duplicate通知の冪等性、二重実行0件を別に確認した。
最終fault relayの停止も確認済みで、受入の追加利用者操作はない。

次の科学runでは固定28dの新checkout・固定problem/roundを使い、必要な科学実行許可と
入力・評価契約を別に維持する。今回の人工2件faultの予算を通常18件探索や実H4へ広げない。

担当ごとの実機記録は `hosts/<host_id>/HANDOFF.md` へ残し、共通branchへforce-pushしない。
SOURCE/生結果/STOPを保持し、新科学runの許可をこの人工検収から拡張しない。
