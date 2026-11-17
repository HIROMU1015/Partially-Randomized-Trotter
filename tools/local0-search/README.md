# ローカル0 → RomuTailの通常操作

Windowsの既存 `tools/local0-dry-check/Common.ps1` と結果同期を使う。
`CLIENT.json` に固定 `code_root` や別のPlanOnly実装を置かず、RomuTailの
`.rpe-central/SERVICE.json` → `RELEASES.json` から検証済み版を解決する。

```powershell
.\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\scope.json
.\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\scope.json
.\Invoke-RpeSearch.ps1 -Action Prepare -RequestFile .\scope.json
.\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\scope.json
.\Invoke-RpeSearch.ps1 -Action Status
.\Invoke-RpeSearch.ps1 -Action Stop
.\Invoke-RpeSearch.ps1 -Action Resume
```

特定runは `-RunId search-...`、新規探索で特定登録版は `-Release <release-id>` を使う。
既存runのResumeはそのrunの `CONTROLLER_CONFIG.json`・`RELEASE.json`・固定入力を使い、
新規用のcurrent版切替で科学SOURCEや探索規則を置き換えない。
停止はRomuTailの候補投入を止め、同じrunのnormal driverだけへSIGINTを送り、
既に受領した遠隔workerは完了・保存する。Resumeまでspoolを保持する。

Start/Resumeの確認後は既存 `local0-result-sync/Invoke-H4ResultSync.ps1` を呼ぶ。
即時同期、実行中10分ごと、終了・停止時の最終同期は既存同期実装が担当し、
同じrunでmonitorを二重起動しない。探索していないときは既存の手動同期を使う。

## 統合版を一手順で登録・切替する

RomuTailで、固定checkout・SERVICEの配置と人工/保存入力検査を完了してから実行する。
版登録は科学実行許可ではなく、新規runが解決する配置の選択である。

```bash
PYTHONPATH=src python3 -B -m trottertracks.rpe_release_registry \
  --registry '/home/abe/Project/Partially Randomized Trotter/.rpe-central/RELEASES.json' \
  --release '<validated-release-id>' --service '/ABS/FROZEN_SERVICE.json' \
  --activate --evidence '/ABS/ACCEPTANCE.json'
```

登録済みの同じIDの内容は上書きしない。新しい版は新IDで登録し、`--activate` で
currentを切り替える。Preview/PlanOnly/Startは同じ選択を使い、run開始時に版と入力を固定する。
