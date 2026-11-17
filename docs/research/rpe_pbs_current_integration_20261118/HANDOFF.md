# 通常運用へのPBS統合HANDOFF（2026-11-18 JST）

新規探索のCPU配分と採用済み設定は [CPU上限のHANDOFF](cpu_limits/HANDOFF.md) を参照。

採用current：`romutail-pbs-current-95a74e448`。controller/PBS transport SOURCE：
`95a74e4486131a97bd5235221bf2d8ab9f99f591`。
[REPORT](REPORT.md)、[採用記録](checks/ADOPTION.json)、[通常入口の実機検収](entry/FINAL_ACCEPTANCE.json)。

中央と状態の正本はRomuTailの
`/home/abe/Project/Partially Randomized Trotter/.rpe-central/state`。
共有価格storeは同baseの `price-validation/models.sqlite3`。
新SERVICEは `deployment/romutail-pbs-current-95a74e448/SERVICE.json`、
公開manifestは [fleet/RELEASE_COMPONENTS.json](fleet/RELEASE_COMPONENTS.json)。
旧e917・925 SERVICE、完了run、原結果・STOPは上書きしていない。

## 普段の開始・状態確認・停止・再開

Windowsローカル0の既存repo/CLIENTを使う。今回の人工用CLIENTは普段の探索に使わない。
次のコマンドは、次回利用者が対象・範囲を明示し、Codexが§10.5の重複照合と必要な配置を終えた後に使う。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Prepare -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId RUN_ID
```

新規runはcurrentを解決する。明示する場合は `-Release romutail-pbs-current-95a74e448`。
既存runのStatus/Resumeはrunに固定されたrelease・SOURCE・入力・runtimeを使用する。
別hostから同じ指示を送っても中央の同じrunを照会・再利用し、完了分を再投入しない。
L_D、要求幅、絶対κ両端込み区間、shot auto、個別の `scope.preparation_budgets` は従来どおり。
compile予算はstage task数であり、Q3予算からcompile許可を推定しない。

RomuTailの共通入口例：

```bash
root='/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/95a74e448-pbs-current'
client='/home/abe/Project/Partially Randomized Trotter/.rpe-central/operation-gateway/CLIENT.local.json'
python3 -B "$root/scripts/run_rpe_search.py" --config "$client" --action Preview --request-file SCOPE.json
python3 -B "$root/scripts/run_rpe_search.py" --config "$client" --action Start --request-file SCOPE.json
python3 -B "$root/scripts/run_rpe_search.py" --config "$client" --action Status --run-id RUN_ID
python3 -B "$root/scripts/run_rpe_search.py" --config "$client" --action Stop --run-id RUN_ID
python3 -B "$root/scripts/run_rpe_search.py" --config "$client" --action Resume --run-id RUN_ID
```

server2/WSLのrootとhost layoutは [配置HANDOFF](fleet/HANDOFF.md)、
既存の共通CLIENTとSSH経路は [共通入口の既存HANDOFF](../rpe_price_release_integration_20261118/common_entry/HANDOFF.md)。
server2は既存qg2の逆向きprivate socket、WSLは既存Windows Git SSHのRomuTail aliasを使う。
資格情報をコピーしていない。Codex sandboxから本人processの `/proc` が見えない場合は、
既存SSHまたは許可された実hostでStatusを実行し、確認不能を新規起動・retryで解消しない。

操作シェルやチャットを閉じても、受領済みcontroller/workerは独立して動く。
Start/Resume直後の同期、実行中10分間隔、終了時最終同期、公開待ち表示と実マップの場所は従来どおり。
操作転送daemonは既存e917のまま保持したが、95と該当3ファイルがbyte一致し、
毎要求fresh子processで選択したSOURCE95を読む（[互換性](entry/OPERATION_GATEWAY_COMPATIBILITY.json)）。

## PBS設定・回収・資源

RomuTail既存 `ssh prism`（akeno経由）を使う。
cluster baseは `/home/AbeHiromu/projects/rpe-pbs-20261118`。

- controller-local設定：`.rpe-central/deployment/romutail-pbs-current-95a74e448/{PBS_SETTINGS.json,PBS_PROFILE.json}`。
- PBS固定code：`releases/95a74e448-current`。
- PBS manifest/layout/profile：`fleet/metadata/romutail-pbs-current-95a74e448/{RELEASE_COMPONENTS.json,HOST.json,PROFILE.json}`。
- 科学Q3/view/providerは従来の `readiness/views/<SOURCE>`、Python/依存は既存専用環境を再利用。
- **gateway_pythonは `runtime/venv311rc1/bin/python` を明示**。通常exchange/collectと終了後harvestの両方で使用。

workerはPBSが割り当てた計算nodeで動き、実割当CPU/memory、DP worker/library thread、
残りwalltimeを同じ枠poolで会計する。login nodeで科学workerを実行しない。
qsub応答欠落・切断・再開では同じrun/token/job/attemptを照合し、新tokenで重複投入しない。
spoolは共有homeのrun別directoryに残り、stdout返送を待たずtask単位で回収する。
walltime受付停止・drain、UNKNOWN/one-shot/retry/STOPは既存契約に従う。

本番templateは元の科学false・Q3/準備予算0、PBS `task_seconds={}` を保持する。
一件のFull PF実績480秒を全方式へ広げていない。科学taskは適用可能な時間見積りが未設定ならPBS受付を保留する。
次回の採択scopeでCodexがPBS資源・時間見積り・host適格性を固定し、設定変更が必要なら
旧SERVICEを編集せず既存の新配置/release手順を使う。新統合版の通常Startからの実H4はその次回探索で確認する。

保存exact OUTCOMEの検査は [harvest/README](harvest/README.md) を参照する。
7168/7174/7741/7747桁の全整数とOUTCOME bytesを通常配送・終了後harvestで保持した。
元925 attemptを再計算/再採用していない。回収トラブルをfloat化・切詰めや科学retryで回避しない。

## 配置・最新版の更新・server1復帰

一つの既存registryが検証済みreleaseだけを選ぶ。branch HEADを自動採用しない。
新しいcode/settingsは新rootへ配置し、公開manifestとhost layoutを作り、保存入力/人工検収を行う。
`prepare_rpe_release_fleet.py`、`manage_rpe_release_deployment.py`、
`prepare_romutail_controller_deployment.py`、`prepare_rpe_pbs_release.py` を再利用する。
登録済みSERVICE/manifest/layoutは不変とし、環境固有pathはprivate配置設定で解決する。

```bash
PYTHONPATH="$NEW_CODE_ROOT/src" python3 -m trottertracks.rpe_release_registry \
  --registry '/home/abe/Project/Partially Randomized Trotter/.rpe-central/RELEASES.json' \
  --release NEW_RELEASE_ID --service /ABS/NEW/SERVICE.json --evidence /ABS/REPORT.md
```

明示releaseの通常Windows Preview後、中央担当が同じ引数に
`--activate --expected-current OLD_CURRENT_ID` を付けて採用する。
直前にcurrent/controller/relay所有を確認し、他担当の更新を上書きしない。
その後、release省略Preview・旧run pinを確認してcommit/push/remote照合する。

RomuTail/server2/PBSへ必要codeを配置済み。WSLは必要optimizerだけを更新し通常受付OFFを維持した。
server1は停止中のため未接続。復帰用packと具体的手順は [fleet/HANDOFF](fleet/HANDOFF.md)。
復帰後、Codexが同じmanifestで不足分配置→保存入力metadata→人工参加・切断回収を検査してから受付を有効にする。

## 今回の終了状態とSTOP

人工PBSは `267198.CPUcluster-PRISM`、workq・1node/2CPU/4GB/4分、node `CPUcluster031`。
人工8task（PBSでcompile1/Q3 4、RomuTailでsignal2/audit1）を一度ずつ採用。
job F/exit0・worker EXITED、完了Resume/Start再送の追加起動・二重採用・PBS再投入は各0件。
人工state/store/cache/出力/同期は検収専用で、通常の科学比較表・価格store・マップに混ぜていない。

実H4検収の既存証拠は925のFull PF/M5一件だけ（[元記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840525ed6a2dc3e018dd568898b9bc9e06349578/docs/research/rpe_pbs_h4_q3_portability_20261118/HANDOFF.md)）。
95の実H4検収へ付け替えない。今回の新規実信号・実compile・実Q3・価格fit・本番探索は各0件。
公開完了後mandatory STOPを維持し、次の本番探索へ自動進行しない。
