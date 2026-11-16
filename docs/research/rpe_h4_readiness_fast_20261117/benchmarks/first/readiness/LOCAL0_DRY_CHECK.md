# local0：実入力dry-checkの引継ぎ

この資料は今回の計画専用。前回の人工run・relay・spoolを再開しない。
実行するのは、固定SOURCE/入力hashの照合と既存adapterの明示的な
`--mode dry-check` だけである。controller、HTTP、queue、mailbox status/submit、
Q3、信号生成、価格較正、compileは起動しない。

サーバー2からqg1/RomuTailへの直接接続は使わない。以下は既存の
`qg2`・`qg1`・`RomuTail` aliasがあるlocal0のWindows PowerShellで行う。
実行前の遠隔結果はすべて「今回は実機未確認」であり、人工検収や過去の
PF/Pauli metadata PASSを今回選んだ候補のPASSとして扱わない。

## 配布物と専用配置

`LOCAL0_DRY_CHECK.zip` には `scripts/check_rpe_q3_host_readiness.py`、
`assignments/*.json`、`profiles/server1.json`、`profiles/local.json` を収録する。
assignmentは今回の実候補REQUEST・固定SOURCE manifest・入力blob hashであり、
queueへsubmitするために使用しない。ZIPのprofileのrootと下記配置を一致させる。

| host | repository | 今回の専用directory | 固定SOURCE view |
|---|---|---|---|
| qg1 | `/home/AbeHiromu/projects/partially-randomized-trotter` | repository内 `.server-preparation/rpe-distributed-h4-plan-20261117` | repository内 `.server-preparation/worktrees/rpe-distributed-q3-source-28d976b99` |
| RomuTail | `/home/abe/Project/Partially Randomized Trotter` | repository内 `.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117` | repository内 `.worktrees/rpe-distributed-q3-local/.local-q3/common-28d976b99` |

固定SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。
既知のPythonはqg1 `/home/AbeHiromu/venvs/trotter-common/bin/python`、
RomuTail `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`。
前者は過去観測3.12.3、後者は3.11.0rc1であり、今回の実機結果で改めて記録する。
新しいpackage installや共有設定変更は行わない。

## local0で取得する

```powershell
$H4PlanHome = Join-Path $env:USERPROFILE 'rpe-distributed-h4-plan-20261117'
New-Item -ItemType Directory -Force -Path $H4PlanHome | Out-Null
$H4PlanZip = Join-Path $H4PlanHome 'LOCAL0_DRY_CHECK.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/LOCAL0_DRY_CHECK.zip' $H4PlanZip
if ($LASTEXITCODE -ne 0) { throw 'dry-check bundle download failed' }
Expand-Archive -LiteralPath $H4PlanZip -DestinationPath $H4PlanHome -Force
```

展開先の上書きだけが `-Force` の対象。遠隔ではこのZIPを新しい所有directoryへ
展開し、既存の異なる内容や過去の結果directoryを上書きしない。

```powershell
ssh qg1 'mkdir /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117'
if ($LASTEXITCODE -ne 0) { throw 'server1 plan directory exists or is unavailable; inspect before proceeding' }
ssh RomuTail 'mkdir "/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117"'
if ($LASTEXITCODE -ne 0) { throw 'RomuTail plan directory exists or is unavailable; inspect before proceeding' }
scp $H4PlanZip 'qg1:/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/plan.zip'
if ($LASTEXITCODE -ne 0) { throw 'server1 bundle upload failed' }
scp $H4PlanZip 'RomuTail:/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117/plan.zip'
if ($LASTEXITCODE -ne 0) { throw 'RomuTail bundle upload failed' }
$H4Unzip = @'
import pathlib, sys, zipfile
plan = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(plan / 'plan.zip') as bundle:
    bundle.extractall(plan)
print(plan)
'@
$H4Unzip | ssh qg1 '/home/AbeHiromu/venvs/trotter-common/bin/python -B - /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117'
if ($LASTEXITCODE -ne 0) { throw 'server1 bundle extraction failed' }
$H4Unzip | ssh RomuTail '"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B - "/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117"'
if ($LASTEXITCODE -ne 0) { throw 'RomuTail bundle extraction failed' }
```

上記 `mkdir` が既存directoryを検出した場合は、既存ZIPと今回ZIPのhash、profile、
assignmentを照合してから配置再開する。過去の結果を消して初回として扱わない。

## 遠隔配置の準備をlocal0のCodexが行う

1. qg1/RomuTailで専用directoryを作り、ZIPをscpする。repositoryは既存のものを使う。
2. 既存originから公開coordinator branchと、assignmentの入力locatorにある不足commitを
   fetchする。SSH設定・Git credential helper・元repositoryの作業branchを変更しない。
3. profileの `root` に固定28dのdetached worktreeを新設する。
   既存directoryの場合はHEADと必要な8ファイルのhashを照合し、異なれば上書きせず不足を報告する。
4. profileの `route_views` にある各commitを所有するdetached viewへ配置する。
   RomuTailの旧viewは `.local-q3/views/<full SHA>` にある。
   既存viewを流用する場合もHEAD・routeのSOURCE guardを確認する。
5. 新SOURCE viewの `src/` と `scripts/`、route viewの必要なmoduleと入力を配置する。
   必要な入力blobはassignmentの `input_hashes` が固定している。
   不足が分かったら、科学生成で補充せずlocator/pathを結果へ記録する。

最小Git操作の例（下記はqg1。RomuTailでは上表のrepository/pathを用いる）：

```powershell
ssh qg1 'git -C /home/AbeHiromu/projects/partially-randomized-trotter fetch origin codex/rpe-distributed-q3-coordinator'
if ($LASTEXITCODE -ne 0) { throw 'SOURCE fetch failed' }
# 次のaddは固定viewが未存在の場合だけ。既存viewをforceで置換しない。
ssh qg1 'git -C /home/AbeHiromu/projects/partially-randomized-trotter worktree add --detach --no-checkout /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-source-28d976b99 28d976b99cdf8329adcc1b79327e329e947ef774'
ssh qg1 'git -C /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-source-28d976b99 sparse-checkout set --cone scripts src configs schemas'
ssh qg1 'git -C /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-source-28d976b99 checkout --detach 28d976b99cdf8329adcc1b79327e329e947ef774'
```

元の科学runのSOURCE/round/input/queue/budget、過去のworker spoolは変更しない。
固定routeのscript/AST guardを変更してPASSにしない。

## 各候補を乾式検査し、結果を回収する

配置後、下記で新しい結果directoryを作る。ファイル名はZIP内のassignment名を使う。
PowerShellの単一引用符を含むSSH command文字列内では、空白を含むLinux pathを
二重引用符で囲む。local0のPythonや既存relayは起動しない。

```powershell
$H4Methods = Get-ChildItem (Join-Path $H4PlanHome 'assignments') -Filter '*.json' | ForEach-Object { $_.BaseName }
foreach ($H4Method in $H4Methods) {
    $H4Server1Command = '/home/AbeHiromu/venvs/trotter-common/bin/python -B /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/scripts/check_rpe_q3_host_readiness.py --assignment /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/assignments/' + $H4Method + '.json --profile /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/profiles/server1.json --output /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/dry/' + $H4Method
    ssh qg1 $H4Server1Command
    # exit=2は不足を記録した結果。READINESS.jsonとADAPTER.logを残し、次候補へ進める。
    $H4LocalBase = '/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117'
    $H4LocalCommand = '"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B "' + $H4LocalBase + '/scripts/check_rpe_q3_host_readiness.py" --assignment "' + $H4LocalBase + '/assignments/' + $H4Method + '.json" --profile "' + $H4LocalBase + '/profiles/local.json" --output "' + $H4LocalBase + '/dry/' + $H4Method + '"'
    ssh RomuTail $H4LocalCommand
}
New-Item -ItemType Directory -Force -Path (Join-Path $H4PlanHome 'results') | Out-Null
scp -r 'qg1:/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-h4-plan-20261117/dry' (Join-Path $H4PlanHome 'results/server1')
scp -r 'RomuTail:/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local/.local-q3/h4-plan-20261117/dry' (Join-Path $H4PlanHome 'results/local')
```

同じoutputへの再実行は拒否する。再検査が必要なら `dry/<method>-check02` のように
新しいdirectoryへ保存し、前回の不足記録を保持する。中継は起動していないため停止操作は不要。

結果をサーバー2へ返す場合は、今回の専用directory内に新しい回収先を作る。

```powershell
scp -r (Join-Path $H4PlanHome 'results') 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/remote_results'
```

## 不足の切り分けと報告

- `VALIDATE_FIXED_SOURCE_INPUTS_AND_HOST_PATHS` の失敗：HEAD/hash不一致、入力Git blob欠損、
  route view未配置、profile path不備を具体的な例外とlocatorで記録する。
- `EXISTING_ADAPTER_DRY_CHECK` の失敗：`ADAPTER.log` と `adapter/DRY_CHECK.json` を読む。
  count scope不足、import/runtime不足、元metadata/評価契約不一致を区別する。
- `REAL_HOST_DRY_CHECK_PASS` は今回の候補・配分・入力・runtimeでの乾式PASS。
  未評価MSEを適合とせず、Q3性能やDP cache HITを認定しない。

特にPR-DFの固定route `23160cc876807c0300be3de811f767b5810640a7` は、
旧caller `e0b3cad2a024070389f757160f3a20d7da02c5ff` を読み、内側で
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115`
（SHA `27e70efe533862f38f4080aabb9cdd3ad1749fe9`）を参照する。
外側route viewを移すだけではこの絶対pathを解決できない。
さらに `runtime_check` はPython3.11.0rc1 exactを要求する。
qg1の既知3.12.3をそのまま使った場合や、RomuTailの内側path欠損は不足として記録する。
今回、固定SOURCE/AST guardの書換、サーバーpathを偽装するsymlink、別runtimeへの契約変更は行わない。
過去のRomuTail記録は `4058b48d5935b949929f093381447336ac86714d` の
`docs/research/rpe_distributed_q3/hosts/local/READINESS.md` にあり、今回の新しい実機確認とは分ける。

最終的に各「candidate×host」を、実機PASS／実機で不足確認／今回は実機未確認の
3区分で中央REPORTへ反映する。乾式検査後も本番queueへ投入しない。
