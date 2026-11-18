# H4 RPE中央管理：RomuTail

2026-11-18。実装と配置の検収。新規の実H4信号・compile・実Q3・本番探索は起動していない。

正本はRomuTailの `/home/abe/Project/Partially Randomized Trotter/.rpe-central`。
`SERVICE.json` → `RELEASES.json` が新規runの検証済み配置を選び、`state` にGateway索引とrunごとの固定設定を保存する。
現在の選択は `romutail-6da8554e0-r4`、配置は `.rpe-central/deployment/6da8554e0-r4`。
各runの通常探索directoryにcheckpoint・JobStore・STOP・採用記録がある。
科学SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`、Q3 timed adapterは `349558b748a82f384a4589434a63ef9b5572032c`。
制御コードは `6da8554e0040f98578667973e8802c680f87982e` の専用checkout
`.rpe-central/releases/6da8554e0`。準備wrapperとworkerの固定SOURCEは `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`。
Windows frontendは `9613580a6a62836f863f0bf235a0b372e9328d15`。
常駐relayは互換な89b3版を継続使用し、6daとの通信コードのbyte一致を記録した。

旧server2の `rpe-local0-search/CURRENT_SERVICE.json` は移行済み表示へ切替済み。
元のpointerは `CURRENT_SERVICE.before-romutail-20261118.json`、旧stateには `MIGRATED_TO.json` を保存した。
旧科学run・STOP・原成果は変更せず、履歴の索引をRomuTailへ引き継いだ。
[引継ぎ・実機検収](REPORT.md)、[固定版の対応表](RELEASE_COMPATIBILITY.json)、[転送内訳](transfer/STATE_TRANSFER.json)、[切替確認](transfer/CUTOVER.json)。

## ローカル0の通常コマンド

Windows PowerShellで次のdirectoryを使う。SSH実行ファイルは、実機で反復検証した
`C:/Program Files/Git/usr/bin/ssh.exe` を `CLIENT.json` と結果同期設定の `ssh_executable` に明示する。
既存alias・鍵・known_hostsと通常のhost認証を使い、全体のSSH設定は変更しない。
`CLIENT.json` はRomuTailのbootstrapを参照し、固定code_rootを持たない。

```powershell
Set-Location 'C:\Users\romuk\Desktop\code\project\Partially-Randomized-Trotter'
$search = '.\tools\local0-search\Invoke-RpeSearch.ps1'
& $search -Action Preview -RequestFile .\scope.json
& $search -Action PlanOnly -RequestFile .\scope.json
& $search -Action Prepare -RequestFile .\scope.json
& $search -Action Start -RequestFile .\scope.json
& $search -Action Status -RunId search-REPLACE_WITH_RETURNED_ID
& $search -Action Stop -RunId search-REPLACE_WITH_RETURNED_ID
& $search -Action Resume -RunId search-REPLACE_WITH_RETURNED_ID
```

`Start` は新しい科学実行の許可を得た範囲にだけ使う。現在配置した実入力テンプレートは
元の `science_approved=false` とQ3/準備の0予算を保持する。今回の人工Startは別SERVICE・別保存先を使う。
`Preview`/`PlanOnly`/`Prepare` は範囲・版・準備状況の確認に使える。
`Status` の `runtime` はcontrollerの生存、queue/採用数、終了code、方式ごとの停止理由を返す。
要求JSONの既存形式は [metadata専用の例](checks/SCOPE.metadata-only.json) を参照。
この例は `max_new_q3_per_method=0`。科学実行用の予算・入力は、許可された新規scopeに合わせて別に確定する。
PR-PauliのL_D、絶対κの両端込み範囲、要求幅とshot配分autoは、通常optimizer入力まで保存される。

同じ範囲の再送は同じrunを照会・再利用する。明示runは `-RunId`、新規runの版指定は `-Release RELEASE_ID`。
既存runのResumeはそのrunの `CONTROLLER_CONFIG.json`、科学SOURCE、入力、探索規則を使い、current切替には追従しない。
開始後のcontrollerはRomuTailで切り離して動き、Windowsシェル・チャットを閉じても続行する。
Stopは新規受付を止め、未割当queueを取り消す。受領済みjobの所有host/attemptを維持し、同じIDで回収する。
科学上のSTOPをResumeで解除しない。

## hostと通常relay

RomuTail自身が中央管理とlocal workerを担当する。server2へは `qg2`、Windowsへは既存 `Romulaptop` SSHを使い、
Windows上の `wsl.exe -d Ubuntu` からWSL mailbox workerを起動・照会・回収する。
鍵・認証設定はGit成果物に含めない。host別Python/SOURCE/profile/work_dirは配置overlayで解決し、server2の絶対pathを模倣しない。
relayの配置とコマンドは [配置HANDOFF](deployment/HANDOFF.md) を参照。
通常relayは `.rpe-central/RELAY.json` を使い、人工検収relayとはport・spool・状態を分ける。

Q3と準備工程は同じJobStore/予約枠を共有する。既存auto規則が資源、明示上限、受付Pause、再開時の所有を反映する。
server2・RomuTail・復帰したserver1の空き枠を優先し、埋まっていれば参加中のWSLへ待機なしで割り当てる。
server1停止やWSL切断で他hostを待たせない。遠隔で受領したjobは元hostへ保存され、再接続で回収する。

local0-wslの本来の受付はOFFのまま：
`/root/projects/Partially-Randomized-Trotter-local0-wsl/.local0-wsl/SETTINGS.json`。
人工検収用の受付設定とは別。普段のPause/復帰はこの既存設定とauto上限で行う。

## 検証済み版の登録と切替

新しいcontroller checkout・host profile・固定入力を配置し、保存入力inspectionと人工検収を終えてから、
新しいSERVICEを新しいdirectoryへ生成する。既登録SERVICEや既存runの固定設定は上書きしない。

```bash
cd '/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/6da8554e0'
PYTHONPATH=src '/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' \
  -m trottertracks.rpe_release_registry \
  --registry '/home/abe/Project/Partially Randomized Trotter/.rpe-central/RELEASES.json' \
  --release NEW_RELEASE_ID --service /ABS/NEW_DEPLOYMENT/SERVICE.json \
  --evidence /ABS/VALIDATION_RECORD.json --activate
```

この1操作で登録と新規runのcurrent切替を行う。版は科学実行の許可を表さない。
新規Preview/PlanOnly/Startは同じregistryを読み、実行開始時に解決版と入力hashをrunへ固定する。
版の明示は登録済みの版だけ、既存runには固定済み版だけを使う。

## 結果と探索マップ

実結果の公開は既存 `publish_h4_search_run.py` とCの通常集計経路を使う。
RomuTailの `results-publication` checkoutから専用結果branchへ保存metadataを公開し、Cが通常通り発見する。
人工runはこの公開処理から除外する。原科学成果は元のhost/pathとhashを記録し、必要なものだけ取得する。

ローカル0の既存 `tools/local0-result-sync/CONFIG.json` をRomuTailのC checkout
`.rpe-central/publication` とRomuTailのcontroller状態へ接続した。
開始・再開時の即時同期、実行中10分間隔、終了時の最終同期、同runの重複monitor抑制を引き継ぐ。
手動同期は既存 `tools/local0-result-sync/Invoke-H4ResultSync.ps1 -Action Sync`。
利用者の入口は次のファイル：

`C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html`

## server1復帰時

1. 利用者がserver1を起動後、CodexがRomuTailから既存SSH経路と資源・既存予約を確認する。
2. 固定SOURCE/transport/profileを専用directoryへ配置し、[保存入力の補充手順](preparation/HANDOFF.md)でPR-DF入力をhash照合する。
3. 既存12準備セルとQ3固定routeのdry-checkを新しい検査先へ保存する。科学計算は別の許可範囲で行う。
4. 人工mailboxで受領後切断・復帰回収と同jobの二重採用防止を確認する。
5. 共有予約と受付設定を照合してから、新しい配置版のserver1を有効にし登録・切替する。

今回server1へ接続しておらず、その復帰は完了条件に含めていない。

検収の詳細は [REPORT](REPORT.md)。3ホスト人工13件、逐次参照7件、最終制御版の停止・再開と
Windowsの実行中同期monitorを確認した。新規実科学計算は0。
次段階は、利用者がserver1を復帰した後にCodexが上記配置・保存入力検査・人工参加検収を行う。
新規の実探索は利用者の別の実行指示で範囲と科学予算を確定してから開始する。

## Romulaptopからの通常Startと初回起動の短縮

2026-11-18 JST。currentは `romutail-startup-e97575eb0`、固定controller SOURCEは
`e97575eb0916ae6e75333e750fd7305f98145a7b`。前版の容量対策、worker通信SOURCE、各科学SOURCE、
PBS・WSLの受付と資源設定を引き継いだ。RomuTail、Romulaptop WSL、server2、PBSゲートウェイの専用配置を照合した。

RomulaptopのCodexは、利用者の探索範囲を通常のscopeファイルに保存し、開始指示を受けたら
`Invoke-RpeSearch.ps1 -Action Start -RequestFile .\scope.json` を1回送る。RomuTailが§10.5の照合、
候補準備と適格workerへの割当を行う。受け取ったrun IDで `-Action Status -RunId <run_id>` を使う。
個別のPreviewやPrepareの送信は、利用者が確認を求めた場合に使う。

保存run `search-be59f46fceee86257816` では中央起動から初回Q3本体開始まで56.5秒、
JobStoreの初回RUNNING記録まで118.9秒だった。新しいStatusは、中央の割当とjob ID・attempt・要求SHAが
一致するRomuTailのローカルworkerについて、転送前でも本体開始記録があれば実行状況に反映する。
重複照合の全明細は `DUPLICATE_PREFLIGHT.json` に保存し、通常応答には照合時間、件数、明細の所在と抜粋を返す。
保存runの照合部分で約1.15 MBから約11 kBになった。

同じ保存H4 DF10入力で、幅ごとの初期計画と初回候補照合を科学処理なしで比較した。
既存mapperの起動を方式単位にまとめ、固定カタログを同じ起動内で再利用した結果、
計画は19.18秒から10.15秒、初回候補照合は8.07秒から2.90秒になった。
28構成群の固定値・保留理由・計画と66候補のID・準備可否・価格が一致した。
この比較は保存入力のメタデータ処理であり、実Q3本体までの所要時間は次の実行時に観測する。

人工回帰83件を通過した。Windows通常入口とRomuTailの共通入口で明示版とcurrentのPreviewが一致し、
既存runは元の固定版と状態を返した。既存設定等35ファイルのハッシュも一致した。
配置・比較・current採用の記録は中央の `.rpe-central/deployment/romutail-startup-e97575eb0/`。
実信号・実compile・実Q3・本番探索の新規起動は各0件。
