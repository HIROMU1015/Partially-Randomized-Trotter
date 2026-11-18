# 承認済みH4探索の操作と保存先

状態の正本はRomuTailの `.rpe-central/state`。操作は従来のWindowsクライアントまたは `scripts/run_rpe_search.py` を使う。run IDを渡すと、そのrunに固定されたreleaseを解決する。新しいscopeで起動する場合は、利用者が指定した範囲を含むREQUESTを使う。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId RUN_ID
```

`.rpe-central/state/runs/RUN_ID/ROUND.requested.json` が適用された科学条件と上限を、`RELEASE.json` が固定版を、`DUPLICATE_PREFLIGHT.json` が§10.5の照合結果を保持する。実作業の開始と結果採用は `distributed/QUEUE.sqlite3` と `PROGRESS.jsonl` を確認する。準備の `ACCEPTED` には回収済みの失敗も含まれるため、`OUTCOME.json` とworker側の工程ログで成否と科学実行数を確認する。

| 方式 | 今回の継続run | Q3上限 | 承認範囲 |
|---|---|---:|---|
| PR-Pauli | `search-9263482821b08a34a00a` | 元runと累計32 | LD4～7、絶対κ [2,64]、幅h/2h/4h/8h、shot auto |
| B0I | `search-8c5b56e7d2e6d5d599c7` | 元計画と累計16 | p6/7/8、2nd／4th／4th(m5_best)、幅h/2h/4h/8h、登録済み36構成、shot auto |
| PR-DF | `search-0071392407117fd5e51d` | 元計画と累計16 | LD7～9、幅h/2h/4h/8h、保存計画のκ規則とshot条件 |
| PR-DF LD7継続 | `search-4bd3a237a18ed0429847` | 上の16を共用 | LD7・幅h、元の論理routeから保存Q3を照合 |
| PR-DF LD9の価格準備 | `search-541575a96257b73ce1ac` | 上の16を共用 | LD9・幅hの保存構成、較正のSTOPは保持 |

新規探索のcurrentは `romutail-auto-preparation-a1d4b45aa`。通常入口は `.rpe-central/RELEASES.json` を参照する。固定版を追加する際は、候補releaseのSERVICEと工程別manifestを別ディレクトリへ配置し、稼働hostの配置結果と明示Previewを確認する。既存のrelease registryで `--expected-current` を指定して採用し、Windowsと共通入口の版指定なしPreviewを確認する。

保存比較点は [固定locatorと保存値](AUTHORIZED_SAVED_COMPARISONS.json) に記録した。Full PF M5・16hとB0I p7・Yoshida4・32h/5は保存参照。PR-DFのR06は元job、REQUESTとOUTCOMEのハッシュに遡れる。

server1が復帰したら、既存の95a74e448配置と8313cea12のworker差分を確認し、fleetの `romutail-auto-preparation-a1d4b45aa` にある `CONTROLLER_DELTA.pack`、`packages/server1.json`、host profileを使う。保存入力・固定view・工程別runtimeを照合し、人工参加と切断回収を通して受付を有効にする。[従来の配置手順](HANDOFF.md)

ローカル0の閲覧先は `C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html`。中央の `PUBLICATION_STATUS.json` からCの公開成果へ接続し、Windowsの通常同期は `LOCAL_SYNC.json` に取込みを記録する。中央の結果公開とC取込みに時間差がある場合は、既存の最終公開待ち表示を使う。

RomuTailの通常入口から状態を確認する場合は、リポジトリのルートで次を実行する。

```bash
.venv311/bin/python -B .worktrees/rpe-romutail-controller-migration/scripts/run_rpe_search.py \
  --config .rpe-central/CLIENT.json --action Status --run-id RUN_ID --no-install
```
