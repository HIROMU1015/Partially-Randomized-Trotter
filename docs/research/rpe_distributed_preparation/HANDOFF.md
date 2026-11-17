# 信号・compile・監査とQ3の共通分散実行

基点は科学Q3の分散実装349558b74、配置検収91fe799dd。
固定実装は`f956f5197a5acfe113fbb2ed3bfa78564e7f4eec`、branchは`codex/rpe-distributed-preparation`。
検査結果と工程×host表は[REPORT](REPORT.md)、機械可読の根拠は[checks](checks/SOURCE.json)。
この版はserver2だけが候補を選び、準備結果・価格registry・探索結果を採用する。
server1・server2・RomuTail・local0-wslのどれにも工程別profileを持てる。方式ごとの専任hostは作らない。
今回の新規実H4信号・回路compile・Q3は0件。既存の科学run・SOURCE・原結果・UNKNOWN・STOPを変更しない。

## 通常入口と再開

別途採択した**新しい**科学roundに、配置packのprofile/settingsを使う。

```bash
python3 -B scripts/run_rpe_schedule_search.py \
 --problem /ABS/PROBLEM.json --round /ABS/ADOPTED_NEW_ROUND.json --run-dir /ABS/NEW_RUN
```

再開も同じコマンドと同じrun directory。hostやPythonを候補ごとに指定しない。
稼働中の旧runへ途中導入せず、旧runのSOURCE・予算・queue・STOPは維持する。
controllerはGit publicationが可能な専用の可変worktreeで起動し、workerの不変transportからは起動しない。
PR準備を使う採択roundでは`candidate_completion.source`を上の固定実装SHAへ明示する。
省略時のcontroller HEAD（資料公開commit等）と、工程profileの固定SOURCEを混同しない。

設定は既存の`executor.scheduler=concurrent`と`executor.distributed`へ追加する。

```json
{
  "candidate_completion": {
    "source": "f956f5197a5acfe113fbb2ed3bfa78564e7f4eec"
  },
  "executor": {
    "scheduler": "concurrent",
    "max_concurrent_jobs": "auto",
    "distributed": {
      "preparation_profiles": {
        "server2": "/ABS/PROFILE_server2.json",
        "server1": "/ABS/PROFILE_server1.json",
        "local": "/ABS/PROFILE_local.json",
        "local0-wsl": "/ABS/PROFILE_local0-wsl.json"
      },
      "preparation": {
        "budgets": {"signal": 0, "compile": 0, "audit": 0}
      }
    }
  }
}
```

上の0は設定例の安全な初期値。準備を実行する新roundでは、採択された準備範囲からstageごとのtask上限を設定する。
上限を省略してUNLIMITEDに広げない。各provider内のsignal数・compile cell数・holdout・SOURCE・one-shot契約も別に保持する。
Q3の`max_new_q3_per_method`は従来どおりで、準備taskをQ3件数に加えない。
有限roundの候補・配分・科学予算は最適化器の既存規則を使う。

## 準備と探索の接続

[工程と既存provider](LEAF_PATHS.md)に通常経路をまとめた。
`Completion.call()`の重い葉は`DistributedPreparationBridge`で同じ`DistributedExecutor`へ送る。
Full PF/B0Iも既存Dの選択・metadata plan・preflightを使い、native compileの後からCの監査を別taskに分ける。
PRのsignalと価格枝は独立して進み、compile成果の受領後にauditを開始する。
compile・audit・Q3は同じJobStore、worker spool、空き枠報告を使い、同じ枠を重ねて予約しない。

幅探索ではREADYの不変snapshotを取り込み、その候補を既存のshot探索・Q3へ戻す。
別候補の準備完了を待つ必要はない。metadata・候補選択・registry採用は中央に残る。
明示的な人工/既存planner用graphは`distributed_preparation.tasks`で宣言でき、
`candidate_dependencies`で同じtaskを複数候補に共有する。科学的な候補選択をgraphで置き換えない。

## host照合・枠・Pause

元のQ3の16 PASSは準備工程へ転用しない。
新しい準備REQUESTは、stage/provider/SOURCEに対応するprofileからPython・入力・compiler/toolを解決する。
保存証拠が同じ完全REQUESTと現在fingerprintに一致すれば再利用し、新しいREQUESTはworkerのmetadata challengeで自動確認する。
PASSが得られるまではattemptや科学予算を予約しない。不足・変更は理由付きHOLDに残す。
完了済みや割当済みのattemptは元のhostで照合・回収する。

優先groupはserver2・server1・local、その次がlocal0-wsl。
優先3台が今すぐ処理できない独立taskは、参加中のWSLの空き枠へ渡す。
WSLはrequired_hostsへ加えず、明示Pause・未接続を参加待ちに数えない。
枠の減少時も実行中taskはdrainし、停止・再起動しない。

## SOURCEと成果物の移送

Windows local0の既存SSH中継を使う。server間直接SSHや共有filesystemは不要。
controller・worker・relayを同じ新固定transportへ更新し、科学checkoutや元のQ3 adapterのSOURCEとは分ける。
新stage adapter/helperのmanifestには`rpe_preparation_worker/artifacts/paths`とstage scriptを含める。

入力・回路・receiptはhashで識別し、中央とhostのCASに存在しないものだけを転送する。
後続hostはreceiptと必要な回路artifactを取得して照合できる。大きな回路を毎回のstatusへ埋め込まない。
新しいGit locatorはbundleで移送し、元commitを保つ。固定viewは編集せず、task所有cloneへ入力を置く。
bundle適用後も科学SOURCEの固定file hashを再確認する。
元JSONやreceiptの絶対pathは書き換えず、実際のファイルアクセス先だけをprofileと保存path mapで解決する。

同一条件のsignal・較正は完全な科学条件と既存cell/group keyで共有する。
切断後も完了済みtaskを別hostへ再投入せず、同じjob/attemptから未転送artifactを回収する。
保存結果・転送・採用はそれぞれidempotentで、科学的失敗とUNKNOWNのSTOPを自動的に開かない。

## 検収と実入力検査の区別

`check_rpe_distributed_preparation.py`は通常schedule入口と実際の人工subprocessを使う。
4つのhost名は1 OS上の模擬hostであり、実4台の科学実行や速度倍率ではない。
元の実Q3の16 PASSは保存記録として保持する。

実入力の検査packには二つの用途を分けている。
保存した実入力・固定SOURCE・Python・compilerが読めるかを調べるinspection例は、
`dry_check_only=true`であり、実行やqueue投入を禁止する。
`SAVED_INPUT_RUNTIME_INSPECTION_PASS`を、その候補の科学REQUEST全体の実行PASSと読み替えない。
新しい実タスクの完全REQUESTは、通常入口が組み立て、実行前の工程別照合と元providerのpreflight/gateを通す。

local0向けの一括配置・検査は、固定SOURCEのdeployment pack内のPowerShell手順を使う。
新しい専用transport/profile/検査directoryを使用し、旧relay・spool・runを再開しない。
Python環境のinstall、WSLのPause解除、workerや科学計算の起動は行わない。
検査結果はpackに指定したqg2の`remote_results`へ返送する。

固定版の[配置pack](deployment_f956f5197/PACKAGE.json)と
[一括手順](deployment_f956f5197/RUN_PREPARATION_DEPLOYMENT.ps1)を使用する。
local0のPowerShellで次を実行する。既存環境へのpackage installはしない。

```powershell
$P = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-f956f5197a5a'
New-Item -ItemType Directory -Force -Path $P | Out-Null
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-preparation/docs/research/rpe_distributed_preparation/deployment_f956f5197/RUN_PREPARATION_DEPLOYMENT.ps1' (Join-Path $P 'RUN.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Script download failed' }
& (Join-Path $P 'RUN.ps1')
```

返送先はqg2の本worktree内`docs/research/rpe_distributed_preparation/remote_results`。
返送だけに失敗した場合は、同じdeployment runnerの`--return-only`を使い、検査や科学処理を再実行しない。
通常runのrelay設定はpackの`RELAY_TEMPLATE.json`を新しいrunのendpointに結び付ける。
旧人工runのrelay設定をそのまま再起動しない。worker開始・中継停止の具体的な手順はpackのHANDOFFを参照する。

## 計時

共通queueの既存clock/eventにstageを付け、body開始/終了、待ち、転送、採用を残す。
Q3の既存集計はそのまま使う。

```bash
python3 -B scripts/summarize_rpe_q3_timing.py /ABS/RUN_DIR --output /ABS/Q3_TIMING.json
```

準備bodyの`JOB_STAGES.jsonl`と、共通queueの`PROGRESS.jsonl`・JobStore eventsも同じrunに保存する。
異なるhost時計の絶対時刻差で片道転送時間を作らない。
方式別最安、identity照合済みQ3結果/時、準備stage結果/時、判断待ち、CPU/RSS、slot変更、
artifact cache再利用、DP cache再利用を、次に採択された科学runで記録する。
