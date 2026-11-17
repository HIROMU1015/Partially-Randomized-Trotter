# 4環境共通の操作入口

実装は `scripts/run_rpe_search.py` と `trottertracks.rpe_operation_client`。
既存の `CLIENT_GATEWAY.py`、release registry、Gatewayと同じrun/stateを使う。
この入口は別の探索器・科学queue・モデルstoreを持たない。

```bash
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Preview --request-file /ABS/scope.json
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action PlanOnly --request-file /ABS/scope.json
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Prepare --request-file /ABS/scope.json
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Start --request-file /ABS/scope.json
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Status --run-id EXISTING_RUN_ID
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Stop --run-id EXISTING_RUN_ID
python3 -B /ABS/OPTIMIZER/scripts/run_rpe_search.py --config /ABS/HOST/CLIENT.json --action Resume --run-id EXISTING_RUN_ID
```

Start/Resumeの科学実行は利用者が次に対象と範囲を指示し、既存の重複照合を通した後に使う。
実入力の通常入口は科学予算0のPreviewで検査し、Status/Stop/Resume/再送Startは専用人工stateで確認した。新規実信号・実compile・実Q3・本番探索は0件。
実際の最終固定版・配置先と採用currentは上位REPORT/HANDOFFを参照する。

各操作は中央で版を解決する。新規scopeは解決したreleaseを固定してから同じ要求を送るため、
解決と送信の間のcurrent変更で別版に切り替わらない。中央が返す当該releaseのhost layoutを優先し、
不足するoptimizerだけを既存の配置helperで取得する。Status/Stopと既存runのResumeは、ローカルの
新版配置待ちで中央操作を止めず、runの固定版へ送る。経路不通時にローカルcontrollerを立てない。

## 経路

- RomuTail：`transport=local`。中央の既存bootstrapを子processで呼ぶ。
- server2：`transport=unix`。`/home/AbeHiromu/projects/rpe-common-operation/control.sock`。
  server2側にはRomuTail用SSH alias・鍵・agent・直接到達経路がないため、RomuTailからの既存
  `ssh qg2` に逆向きUNIX socket forwardingを付けて使う。新しい認証情報は作成・コピーしない。
- local0-wsl：`transport=windows-ssh`。既存 `powershell.exe` のWSL相互実行から
  `C:/Program Files/Git/usr/bin/ssh.exe` とWindowsの既存 `RomuTail` aliasを使う。
  HOMEの指定はその子processだけに適用し、SSH設定や鍵を変更しない。
- server1：停止中で未接続。復帰後は同じ配置処理と逆向きsocket forwardingを有効にし、
  保存入力・人工参加検収を通す。停止中は他環境を待たせない。

Windowsの既存 `Invoke-RpeSearch.ps1` と結果同期の動作は維持する。
共通CLIからのStart/Resumeは既存bootstrapへ `invoke_result_sync` を渡し、中央に設定された
`run_rpe_windows_sync_hook.py` が既存 `OnConfirmedRun` を使う。新しいmonitorは実装していない。
hook失敗で既に切り離したcontrollerや受領済みjobを止めない。

## 中央操作Gatewayの起動・停止・再接続

中央Gatewayは操作転送だけを行い、各要求で既存bootstrapを新しい子processから呼ぶ。
選択版のPython moduleを常駐processにcacheしない。

```bash
python3 -B /ABS/PINNED/scripts/run_rpe_search.py --gateway-config /ABS/GATEWAY.json --start-gateway
python3 -B /ABS/PINNED/scripts/run_rpe_search.py --gateway-config /ABS/GATEWAY.json --status-gateway
python3 -B /ABS/PINNED/scripts/run_rpe_search.py --gateway-config /ABS/GATEWAY.json --stop-gateway
```

再開は同じ `--start-gateway`。固定entry/configを変更するときは既存Gatewayの停止を確認する。
同じ設定の重複起動は既存relayと同じ所有lockで防ぐ。操作Gatewayの停止は探索のStopとは別で、
すでに受領した科学jobを停止しない。

中央とremoteの操作directoryは所有者だけが入れる0700、socketは0600。
切断後に残った本人所有の待受けなしsocketだけを回収する。稼働中のsocketや通常ファイルは変更しない。
各SSH転送の再接続は独立して進め、停止・不通のhostを他hostの操作待ちにしない。

## 検査の範囲

[PROVENANCE](PROVENANCE.json) と [TESTS](TESTS.log)。著者Codex sub-agent `common_entry` の技術検査。
独立科学レビューではない。

人工8件で、同じ中央Gatewayへの再送の追加起動0、current切替後の旧run pin、新規scopeの版固定、
不足codeのHOLD、経路不通時にローカル起動しないこと、稼働socket保全、既存同期hookの呼び出しを確認した。

固定版の配置前の経路検収は [RomuTail](LOCAL_PREVIEW.json)、[server2](SERVER2_PREVIEW.json)、
[WSL](WSL_PREVIEW.json)。すべて同じ中央r4・同じrun IDを返した。
[再接続](RECONNECT.json) では所有SSH転送だけを切断し、同じ要求が復帰した。
これらの事前検収を、最終統合版の不変CIや実科学性能と呼ばない。最終固定版での配置確認は上位記録へ分ける。


## 最終固定版の実配置と検収

採用currentは `romutail-price-fleet-e917165ba`、controller SOURCEは
`e917165ba1f36b7627614cd25e88b5108d9b20b9`。科学SOURCEは `28d976b99` の契約を保持する。
中央stateは `/home/abe/Project/Partially Randomized Trotter/.rpe-central/state`。

| 環境 | OPTIMIZER | CLIENT |
|---|---|---|
| RomuTail | `/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/e917165ba-price-fleet` | `/home/abe/Project/Partially Randomized Trotter/.rpe-central/CLIENT.json` |
| server2 | `/home/AbeHiromu/projects/rpe-release-fleet/releases/e917165ba1f3` | `/home/AbeHiromu/projects/rpe-release-fleet/CLIENT.json` |
| local0-wsl | `/root/projects/Partially-Randomized-Trotter-local0-wsl/.rpe-release-fleet/releases/e917165ba1f3` | `/root/projects/Partially-Randomized-Trotter-local0-wsl/.rpe-release-fleet/CLIENT.json` |
| server1 | 同じserver用配置規則。停止中で配置・実機検査は保留 | fleetの `clients/server1.pending.json` を復帰後に配置 |

上記の各ROOTとCLIENTを冒頭の共通コマンドへ指定する。Windows利用者は既存の
`C:/Users/romuk/Desktop/code/project/Partially-Randomized-Trotter/tools/local0-search/Invoke-RpeSearch.ps1`
を使える。`Preview -RequestFile scope.json` と `-Release romutail-price-fleet-e917165ba` を付けたPreviewは、
同じ `search-66adcb5cf75e3f621ced` と固定版を返した。

中央操作gatewayは、上表のRomuTail ROOTにある `scripts/run_rpe_search.py` と
`/home/abe/Project/Partially Randomized Trotter/.rpe-central/operation-gateway/GATEWAY.json` で起動・照会・停止する。
固定版daemonをチャットから独立して起動済み。

```bash
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B "/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/e917165ba-price-fleet/scripts/run_rpe_search.py" --gateway-config "/home/abe/Project/Partially Randomized Trotter/.rpe-central/operation-gateway/GATEWAY.json" --start-gateway
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B "/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/e917165ba-price-fleet/scripts/run_rpe_search.py" --gateway-config "/home/abe/Project/Partially Randomized Trotter/.rpe-central/operation-gateway/GATEWAY.json" --status-gateway
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B "/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/e917165ba-price-fleet/scripts/run_rpe_search.py" --gateway-config "/home/abe/Project/Partially Randomized Trotter/.rpe-central/operation-gateway/GATEWAY.json" --stop-gateway
```

中央の通常bootstrapには既存結果同期のhookを接続し、
旧run用relay起動は既存lockが稼働中なら成功として再利用する。
通常の旧global relayは停止状態を保ち、新版runの所有relayへ干渉しない。

[FINAL_ACCEPTANCE](FINAL_ACCEPTANCE.json)、[配置](FINAL_PLACEMENT.json)、
[Windows明示版](WINDOWS_EXPLICIT_PREVIEW.json)、[Windows既定版](WINDOWS_DEFAULT_PREVIEW.json)、
[RomuTail](FINAL_LOCAL_PREVIEW.json)、[server2](FINAL_SERVER2_PREVIEW.json)、[WSL](FINAL_WSL_PREVIEW.json)。
3環境の共通CLIも同じrelease・SOURCE・run IDを返し、各optimizer配置の固定SHAを照合した。
[旧r4のStatus](OLD_R4_PIN_STATUS.json) はcurrent更新後も科学実行なしで6da/r4を返す。

[共通人工操作](ARTIFICIAL_COMMON_OPERATIONS.json) は専用stateの完了runに対して、RomuTailのStatus、
server2のStop、WSLのResume、server2のStatus、RomuTailのStart再送、WSLのStatusを順に行った。
ResumeとStartは `REUSED_COMPLETED`、親担当の追加照合でも追加LAUNCH・採用・attempt変更0だった。
人工用gatewayは停止済みで、通常gatewayだけを稼働させている。

[通常C同期](NORMAL_SYNC.json) と [実閲覧map](REAL_MAP_CHECK.json) は、Cの既存公開commit
`ca5905e79d7654bdfe42a679b7b5e74aef80b1be` を通常経路で確認した。
実際の `C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html` に13モデル、
LD2のm5(q=32,R=33)、m6(64,66)、m9(512,526)と固定元receipt URLが保たれる。
保存値の読取りを新しい独立検証と数えず、人工結果はこのmapへ公開していない。
同期済み同一blobのため今回の追加転送は0ファイルだった。

server1復帰後はfleetの同一配置手順で不足分だけを取得し、pending CLIENTを配置、専用0700の操作directoryを作る。
中央 `GATEWAY.json` のserver1 forwardを有効にしてgatewayを停止・再起動した後、保存入力・人工参加検収を行う。
これは配置と参加許可を分ける。今回server1への接続・WSL通常受付の変更は行っていない。

次回は利用者が対象・範囲を指定し、Codexが§10.5の重複照合後に上の通常Start/Resumeを使う。
現在のmandatory STOPを維持する。
