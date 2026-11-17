# PBS統合版の通常操作入口

検査担当：Codex sub-agent `common_entry`。実装者側の技術検査であり、独立科学レビューではない。

既存の `tools/local0-search/Invoke-RpeSearch.ps1`、`scripts/run_rpe_search.py`、中央bootstrap、
release registry、Gatewayを再利用する。別のcontroller・queue・認証機構は追加しない。
新規の実H4信号・実compile・実Q3・本番探索・価格再fitは対象外。

## 検査前の状態

[INITIAL_READ_ONLY.json](INITIAL_READ_ONLY.json) に着手時のcurrentと専用gatewayの所有情報を保存した。
currentは `romutail-price-fleet-e917165ba`。既存gatewayは固定SOURCE e917、PID597348と所有コマンドを照合し、
central UNIX socketの接続を確認した。server1転送はDISABLED。現行daemon・既存client・currentは変更していない。

人工検査は次の専用領域へ分離する。

```
/home/abe/Project/Partially Randomized Trotter/.rpe-central/acceptance/pbs-current-entry-20261118/
  SCOPE.json
  CLIENT.windows.pending.json
  CLIENT.local.pending.json
  gateway-v2/SERVICE.json
  fixture-v2/run/
```

Windowsには `C:/Users/romuk/rpe-pbs-current-entry-20261118/` へ新規client/requestを置き、
既存通常clientは上書きしない。人工clientの `result_sync_enabled=false` により、
人工runを通常C同期・利用者の実閲覧マップへ混入させない。

通常Windows呼出しは既存コマンドをそのまま使い、`-Config` に専用clientを指定する。

```powershell
$entry = 'C:/Users/romuk/Desktop/code/project/Partially-Randomized-Trotter/tools/local0-search/Invoke-RpeSearch.ps1'
$client = 'C:/Users/romuk/rpe-pbs-current-entry-20261118/CLIENT.json'
$scope = 'C:/Users/romuk/rpe-pbs-current-entry-20261118/SCOPE.json'
& $entry -Config $client -Action Preview -RequestFile $scope
& $entry -Config $client -Action PlanOnly -RequestFile $scope
& $entry -Config $client -Action Start -RequestFile $scope
& $entry -Config $client -Action Status -RunId <人工run_id>
& $entry -Config $client -Action Resume -RunId <完了した同じ人工run_id>
```

固定SOURCEで実施した結果は、後段の通常Start・完了Resumeのreceiptを正本とする。


## 固定SOURCE95の通常Preview・PlanOnly

統合SOURCEは `95a74e4486131a97bd5235221bf2d8ab9f99f591`、検査用registryのreleaseは
`romutail-pbs-current-95a74e448`。中央の通常current採用前に、専用人工SERVICEで検査する。
[Windows Preview](windows_PREVIEW.json)、[Windows PlanOnly](windows_PLANONLY.json)、
[共通CLI Preview](common_PREVIEW.json) は同じ `search-c9615bd24b6d44434a99` を返した。

[PRE_START_NORMAL_ENTRY](PRE_START_NORMAL_ENTRY.json) はGatewayが保存した実際のPlanOnlyコマンドを記録する。
固定版の既存 `check_rpe_romutail_distributed.py --controller .../fixture-v2 --control-run .../search-c9615bd24b6d44434a99 --plan-only`
から通常schedule入口へ接続した。returncodeは0で、この時点のcontroller LAUNCH・PBS_STATEは存在しない。
人工scopeのL_D=1、幅binary64 `.1`、shot auto、工程予算を通常Gatewayが適用した。
PBS Start・採用・完了Resumeの実施結果は次節を参照する。


## 通常Startから実PBSまでの完了検収

[通常Windows Startの接続全体](NORMAL_WINDOWS_START_CHAIN.json)。Windowsの既存 `Invoke-RpeSearch.ps1`
から人工runを1回だけStartし、Gatewayの通常launcherと `--execute-run` を通した。
`check_rpe_pbs.py` の移植専用executeを別途起動して代替したものではない。
実際の通常schedule入口、共通queue、PBS node、中央採用までを通した。
PBS割当・nodeの観測とqueue構造比較は親担当 `/root` が実施し、entry担当の検査と区別する。

[Windows Start](windows_START.json) は `STARTED`、[稼働中Status](windows_RUNNING_STATUS.json) は
`RUNNING`、[完了Status](windows_COMPLETION_STATUS.json) は `COMPLETE`・returncode0だった。
同じinvocation `b74ae6e865b644d9a7200b2201b65177` で8taskが採用された。
内訳はPBSでcompile1とQ3四つ、RomuTailでsignal2とaudit1。すべて明示した人工leafで、科学的compile/Q3ではない。

[Windows完了Resume](windows_RESUME.json) と [共通CLIのStart再送](common_REPEATED_START.json) は
ともに `REUSED_COMPLETED`。[最終共通Status](common_FINAL_STATUS.json) でも同じinvocation・ACCEPTED8・adopted8を維持する。
親担当の前後照合で追加invocation・task・割当・採用は各0、PBS token/jobは1つのままだった。
PBS job267198はF/exit0、worker EXITED。

通常Windows clientは [保全確認](WINDOWS_CLIENT_PRESERVATION.json) で着手前と同じhashを保つ。
人工clientだけを追加し、結果同期OFFを確認した。server1接続・WSL通常受付の変更はない。
人工store・state・spool・publicationを通常系と分け、人工成果を実閲覧mapへ送っていない。

共通CLIの稼働中Statusは実ホストのprocess可視性で確認した。
[制限sandbox内の参考Status](SANDBOX_RESTRICTED_STATUS.json) は `/proc` が見えずaliveをfalseとしたため、
[実ホストのStatus](common_RUNNING_STATUS.json) を稼働状態の確認に使った。再起動・再投入は行っていない。

操作gatewayは旧e917配置のまま稼働し、要求ごとに中央が選択した固定版をfresh childで読む。
[互換性](OPERATION_GATEWAY_COMPATIBILITY.json) では共通CLI・転送module・bootstrapの3ファイルが95とbyte-identicalだった。
そのため既存daemon・callbacksを不要に切り替えず再利用する。


## current採用後の通常運用

実際の採用currentは `romutail-pbs-current-95a74e448`。
[最終確認](FINAL_ACCEPTANCE.json)、[通常Windowsの明示版Preview](windows_PRODUCTION_EXPLICIT_PREVIEW.json)、
[release省略Preview](windows_PRODUCTION_DEFAULT_PREVIEW.json)。両Previewは同じ
`search-6d91ba8a68b54e2377be`・同じ適用設定・SOURCE95・科学SOURCE28dを返した。
保存実入力のPR-Pauli L_D=2..3、絶対κの両端込み[1/10,1]、shot autoを保ち、
Q3・signal・compile・audit予算はすべて0。実Startは行っていない。

[旧r4 Status](windows_OLD_R4_PIN_STATUS.json) では、global current更新後も既存run
`search-97f9dc344a64b0661d2d` がr4/6da/科学28dを返す。旧実runのStart/Resumeは実行していない。
旧完了人工runの保存releaseによるResumeの検査は親担当の上位記録に分ける。

通常操作では既存clientを使い、人工検査用 `-Config` を付けない。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\scope.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\scope.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\scope.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId <run_id>
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId <run_id>
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId <同じrun_id>
```

各hostの共通入口は固定95の `scripts/run_rpe_search.py --config <既存HOST/CLIENT.json>` に
同じAction/RequestFile/RunIdを指定する。host別optimizer/runtimeの配置先は [fleet](../fleet/HANDOFF.md) を参照する。
新規scopeは採用currentを解決して固定し、既存runのResumeはそのrunに保存したreleaseを使う。

現在のmandatory STOPを維持する。次回は利用者が対象・範囲を指定し、担当Codexが§10.5の重複照合後に、
その明示された科学実行範囲だけを通常Start/Resumeで実施する。
