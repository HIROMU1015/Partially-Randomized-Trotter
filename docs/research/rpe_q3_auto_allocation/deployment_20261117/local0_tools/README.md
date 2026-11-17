# 固定349 transport配置とremote 12セル実入力dry-check

local0の既存`qg1`・`RomuTail`・`qg2`と`wsl.exe -d Ubuntu`を使う。Windows Python標準libraryのみ。
[RUN_AUTO_DEPLOY_349558B74.ps1](RUN_AUTO_DEPLOY_349558B74.ps1)をPowerShellで実行する。

3計算ホストを並列処理し、各ホストでは元の4 REQUESTを順に実入力dry-checkする。
科学SOURCEは元28d、Pauli routeは33e、専用Pythonと内側SOURCE mapは公開済みprofileをそのまま使う。
新しいtransport/worker/timed adapterは`349558b748a82f384a4589434a63ef9b5572032c`に固定する。
既存repositoryに不足する公開commitだけfetchし、新しいdetached sparse worktreeとprofileを配置する。
既存で異なるSOURCE・profileを見つけた場合は上書きしない。

- worker・controller・relay・科学Q3を起動せず、spoolも作成・開放しない。
- 科学checkout、元REQUEST、route view、既存run、UNKNOWN、STOP、既存spoolを変更しない。
- Python環境のinstall・更新・探索・buildはしない。profileの既知interpreterを自動選択する。
- WSLのWindows参加fileをPAUSEDと読取確認し、前後のbytes一致を確かめる。参加状態は変更しない。
- 通常の固定SOURCEのcheckerとadapterをdry-checkで使用する。新しいmetadata collectorはSOURCE外のpayloadに置く。
- SELECTED_PROFILE・実runtime・公式READINESS・adapter_source・route map・CURRENT_HOST_FINGERPRINTとraw DRY_CACHEを保存し、全file hashをqg2の返送先で照合する。

返送先：`docs/research/rpe_q3_auto_allocation/deployment_20261117/remote_results/`。
サーバー2へ接続できず返送だけ失敗した場合は、乾式検査を繰り返さず次を使う。

```powershell
$RpeAutoDeployHome = Join-Path $env:USERPROFILE 'rpe-q3-auto-deploy-349558b74-20261117'
py -3 -B (Join-Path $RpeAutoDeployHome 'run_auto_deployment.py') --return-only (Join-Path $RpeAutoDeployHome 'results')
```

旧12セルのPASSを新しいadapterへ転用しない。新しい公式READINESS・実fingerprintが一致したセルだけを中央で割当可能にする。
実ホストのdry-checkはlocal0での操作後に成立する。5つの人工unitは通信command・bundle・Git配置・受信hashの実装検査であり、実機PASSを意味しない。
