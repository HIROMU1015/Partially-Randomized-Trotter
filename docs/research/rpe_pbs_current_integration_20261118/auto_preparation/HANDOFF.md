# 自動準備の通常操作

中央はRomuTailの `.rpe-central/state`。選択されたreleaseと工程ごとのruntimeは公開component manifestと各hostの配置設定から解決する。

Windowsの通常入口は既存の `tools/local0-search/Invoke-RpeSearch.ps1`。範囲・精度・指定する上限を `SCOPE.json` の `scope` に入れる。工程予算の省略時は、中央が保存成果にない必要な準備を既存契約の範囲で投入する。明示0でも保存成果は再利用できる。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId RUN_ID
```

Startは明示された範囲の通常準備を含む実行指示に基づく。既存結果の照合は§10.5に従う。Resumeは開始時に返ったrun IDを指定する。旧い異常終了runの復旧は親の固定SOURCE・残枠・STOPを維持した継続経路を使い、通常の新規Startで代用しない。

server1は停止中。復帰後は、既存の95a74e448配置を確認し、RomuTailの `.rpe-central/fleet` にある `romutail-auto-preparation-8313cea12` のworker差分、次に `romutail-auto-preparation-f03f2fcf4` のcontroller差分とserver1用manifest/profileを配置する。各差分は `CONTROLLER_DELTA.pack` と `packages/server1.json` の対応を照合する。保存view・runtime・提出物の指紋を確認し、人工参加・回収を検収してから受付を有効化する。[既存の配置手順](../fleet/HANDOFF.md)
