# 通常運用HANDOFF（2026-11-18）

採用版は `romutail-price-fleet-e917165ba`、controller SOURCEは
`e917165ba1f36b7627614cd25e88b5108d9b20b9`。
中央はRomuTail。正本は `/home/abe/Project/Partially Randomized Trotter/.rpe-central/state`、
共有価格storeは同baseの `price-validation/models.sqlite3`。
[最終採用記録](checks/ADOPTION.json)・[検収報告](REPORT.md) を参照。

## 利用者の通常操作

Windowsローカル0の既存repoで、従来の入口を使う。
`SCOPE.json` は利用者が対象・範囲を明示した要求であり、コマンド例自体は科学実行の指示ではない。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Prepare -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId RUN_ID
```

新規要求はcurrentを解決する。必要な場合だけ `-Release romutail-price-fleet-e917165ba` を明示する。
既存runのResumeではreleaseを指定し直さず、runの固定版を使用する。
同一要求の再送は既存runを再利用する。
PR-Pauli L_D、絶対κ両端込み区間、要求幅、shot autoを従来のscopeで指定できる。
準備工程の予算は `scope.preparation_budgets` の `signal`・`compile`・`audit` で個別に指定する。
compileはstage task数。Q3予算から準備の許可を推定しない。
今回配置した通常テンプレートは科学予算0・science_approved=falseのまま。

RomuTail/server2/WSLからは同一のPython入口を使う。
環境別CLIENT.json、Python、配置場所は [共通入口HANDOFF](common_entry/HANDOFF.md) に記載した。

```bash
python -B /ABS/SELECTED_OPTIMIZER/scripts/run_rpe_search.py --config /ABS/CLIENT.json --action preview --request-file SCOPE.json
python -B /ABS/SELECTED_OPTIMIZER/scripts/run_rpe_search.py --config /ABS/CLIENT.json --action start --request-file SCOPE.json
python -B /ABS/SELECTED_OPTIMIZER/scripts/run_rpe_search.py --config /ABS/CLIENT.json --action status --run-id RUN_ID
python -B /ABS/SELECTED_OPTIMIZER/scripts/run_rpe_search.py --config /ABS/CLIENT.json --action stop --run-id RUN_ID
python -B /ABS/SELECTED_OPTIMIZER/scripts/run_rpe_search.py --config /ABS/CLIENT.json --action resume --run-id RUN_ID
```

PlanOnly/Prepareも `--action planonly` / `--action prepare`。
どの入口も中央へ送信し、client側controllerを起動しない。
Start後に操作シェルやチャットを閉じても中央runと受領済みworkerは継続する。
切断時のjobは元hostに保存され、復帰後に同じID/attemptで回収される。

既存Windows SSH設定を使用する。Start/Resume確認直後に同期し、実行中は10分ごと、
終了・停止時は最終同期する。未公開の最終結果は「最終結果は公開待ち」と表示する。
利用者の閲覧ファイルは
`C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html`。
探索していない間の同期は既存 `Invoke-H4ResultSync.ps1 -Action Sync` を使用する。
人工結果は通常C・マップへ公開しない。

## 検証済み版の登録・切替

branch HEADを自動採用しない。技術検収を通したSOURCEを固定して、次の一手順で更新する。
詳細コマンドとhost別配置は [FLEET_HANDOFF](FLEET_HANDOFF.md) にある。

1. 既存current/並行作業を確認し、SOURCEを固定する。
2. `prepare_rpe_release_fleet.py` で公開manifestとhost layout/profileを新しい配置へ生成する。
   `manage_rpe_release_deployment.py --install` で不足コードだけを取得し、同じSOURCE/input/runtimeを照合する。
   old checkoutが不完全でも上書きせず新しい専用配置へ補う。依存追加は専用環境だけに行う。
3. `prepare_romutail_controller_deployment.py` で新しいSERVICEを作る。
   共有store・科学固定版・予算0・受付OFF・host layout/manifestとfixed_settingsを確認する。
4. 保存入力/人工検収後、既存registryの `python -m trottertracks.rpe_release_registry` から新releaseを登録する。
   explicit release Windows Previewを確認し、同じSERVICEを `--activate --expected-current OLD_ID` で採用する。
   採用後、release指定なしPreviewが同じ版を返すことを確認する。
5. 公開manifest・REPORT・HANDOFFと採用証拠をcommit/pushし、remote固定SHAを確認する。

登録済みSERVICE・manifest・host layout/runtime観測は不変として扱う。
新しい結果metadataのcommitでは科学実装/設定を変更しない。
既存runや明示版は保持し、current更新でruntime/inputを置き換えない。
中央のqueueは一つで、同じrunを複数controllerから再投入しない。

## server1復帰・worker参加

server1は停止扱いのまま。復帰後は同じ公開manifest/専用packを配置し、
不足分install→保存入力metadata検査→人工参加・切断回収を行う。
実機PASS後に当該hostの受付設定を変更する。今回の完了は復帰を待たない。
WSLは配置PASSと通常受付OFFを区別し、既存のPause設定を維持する。
新しいworker参加時は要求manifest/工程SOURCEを照合し、不一致をHOLDして他hostを続行する。

## STOP

今回の新規実H4信号・実compile・実Q3・本番探索・refitはすべて0件。
配置・技術検収・current採用・公開の後はmandatory STOP。
次回、利用者が対象・範囲を明示した際にCodexが§10.5の既存run/job/保存結果を照合し、
重複を避けて通常Startまたは固定run Resumeを行う。
