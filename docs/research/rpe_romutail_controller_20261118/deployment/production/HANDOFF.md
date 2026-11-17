# RomuTail 通常中継と準備profile（2026-11-18）

登録済み統合版は `romutail-6da8554e0-r4`。正本サービスは
`/home/abe/Project/Partially Randomized Trotter/.rpe-central/deployment/6da8554e0-r4/SERVICE.json`。
controller SOURCE は `6da8554e0040f98578667973e8802c680f87982e`、SERVICE の SHA256 は
`1dad63f9d6e89fb931144e01c887265355cf3a0f34a0e4d0c8d7a02c3812022e`。
準備wrapper/worker transportはd887、Q3科学SOURCEは28d976、Q3 adapterは349558のまま。
旧 `deployment/d887abdca` と登録済みr2/r3は変更していない。登録後の SERVICE/ROUND/profile も変更しない。
通常の探索操作は上位 [HANDOFF](../../HANDOFF.md) を使う。

中継は RomuTail 常駐で、通常 DistributedExecutor の `127.0.0.1:8763` に接続する。
RomuTail の計算workerは executor が管理し、中継は qg2 と Windows SSH→Ubuntu WSL の
mailbox を個別に照会・配送・回収する。qg1 は無効化してあり接続しない。
各ホストに配送済みのjobは、中継停止中も元ホストに残る。中継の停止は探索停止操作とは別。

## 日常コマンド（RomuTail）

```bash
RPE_ROOT='/home/abe/Project/Partially Randomized Trotter'
RPE_PY="$RPE_ROOT/.venv311/bin/python"
RPE_RELAY="$RPE_ROOT/.rpe-central/releases/89b3da81a/scripts/run_rpe_q3_relay.py"
RPE_CONFIG="$RPE_ROOT/.rpe-central/RELAY.json"

# 参加・再接続。シェル／チャットを閉じても動作する。
"$RPE_PY" -B "$RPE_RELAY" --config "$RPE_CONFIG" --start
# 状態確認。heartbeat_age_seconds と stopped も確認する。
"$RPE_PY" -B "$RPE_RELAY" --config "$RPE_CONFIG" --status
# 中継を止める。受領済みjobは続行・保存する。
"$RPE_PY" -B "$RPE_RELAY" --config "$RPE_CONFIG" --stop
```

再開は `--status` の `stopped: true` を確認してから `--start` を使う。
稼働中の二重 `--start` は既存lockで拒否する。
ログ・状態は `.rpe-central/relay-state/{STATUS.json,EVENTS.jsonl,RELAY.log}`。
実機で起動・状態確認・停止・再開を通し、89b3固定版へ切替後の常駐PIDは `516949`
（[切替実機記録](RELAY_R3_UPGRADE.json)、[現在の状態記録](IDLE_RELAY_R3.json)）。
本番探索を今回起動していないため endpoint は未待受で、現在の
`Connection refused` と再試行は待機中の状態を表す。次の通常controllerが同じendpointを
開くと配送を始める。本番queueへの投入は今回行っていない。

## 配置と設定

- 中継設定: `.rpe-central/RELAY.json`（同内容の保存版 [RELAY.json](RELAY.json)）。
- qg2 profile: `/home/AbeHiromu/projects/rpe-romutail-worker/server2/profiles/d887abdca-r2.json`。
  spool/cache は同 `server2/{spool,artifact-cache}`。
- WSL profile: `/root/projects/Partially-Randomized-Trotter-local0-wsl/.romutail-controller/worker/profiles/d887abdca-r2.json`。
  spool/cache は同 `worker/{spool,artifact-cache}`。
- 実際のWSL参加設定: `/root/projects/Partially-Randomized-Trotter-local0-wsl/.local0-wsl/SETTINGS.json`。
  `accepting_new_jobs:false` を維持した。中継はOFF中も既存成果を回収する。
- ローカルprofile: `.rpe-central/deployment/6da8554e0-r4/profiles/local.json`。
- 接続alias、既存の鍵ファイルの場所だけを設定に記載し、認証情報本体は保存していない。

各profileは既存auto枠調整を使う。観測時は未計測の初期枠でserver2が1、WSLはPauseで0。
明示max枠は追加していない。ホスト資源は非排他的な観測値であり専有予約ではない。
Q3と準備工程は同じworker spool・executor予約を共有する。auto割当は server2/local/server1
を優先し、WSLが参加中でそれらに空きがない場合の優先待機時間は0秒。

## SOURCE対応と実機で確認した範囲

通常候補の準備REQUESTは統合wrapper SOURCE `d887…` を固定するため、旧a18d wrapper用の
11 stage/provider規則を対応するd887規則として追加した。旧a18d/f956規則は残した。
adapterのHEADと全manifest file hashをd887に固定し、native provider SOURCE、入力、予算、
STOPは維持した。Q3科学SOURCEは28d976、Q3 adapterは349558のまま。

r3ではcontroller/中継だけを89b3へ更新し、r4では中央のStop受付処理を修正した
controller 6daへ更新した。4ホストprofileはr2/r3とbyte一致、
準備wrapper14ファイルは6da内でもd887 hashと一致を確認した。
`candidate_completion_source=d887…` と旧d887の `worker_script` をlayoutで明示している。
[互換性記録](SOURCE_COMPATIBILITY.json) により同一の9 PASS metadataを再利用し、
科学入力の再評価や遠隔SOURCEの再配置は行っていない。
主通し人工検収の実行版は89b3で、その記録を互換性確認して再利用する。
6daで主通し検収全体を再実施したとは扱わない。Stop修正の確認は主担当の別記録を参照する。
中継は89b3のPID516949を継続使用する。中継本体とimport先3ファイルが6daとbyte一致するため、
再起動・遠隔再検査を繰り返していない
（[中継互換性とPIDの実機確認](RELAY_SOURCE_COMPATIBILITY_R4.json)）。

WSLの旧a18d profileが指していた未配置viewは、同じSHAを持つ既設f956 view
（36ad/c1/fefd/d878）へ対応付けた。a18d/13ab/326は専用
`.romutail-controller/preparation-sources/<SHA>` に既存Gitオブジェクトから配置した。
専用object cacheとpack hashは [CONFIGURATION.json](CONFIGURATION.json) に記録した。
サーバー絶対pathを模倣するsymlinkは作っていない。
配置した3版は既存workerの `_clone`（shared clone・sparse checkout）も通し、
HEAD一致と scripts/src/configs/schemas/references の差分0を確認した
（[WSL旧workspace複製記録](WSL_LEGACY_CLONE_METADATA.json)）。
d887自体のWSL `_clone` は通信担当が別途確認している
（[通信担当の記録](../../direct_transport/WSL_CLONE_METADATA_CHECK.json)）。

| 検査 | RomuTail | server2 | local0-wsl | server1 |
|---|---:|---:|---:|---:|
| profile内SOURCE HEAD・adapter全file hash | 39規則PASS | 29規則PASS | 39規則PASS | 停止中・未検査 |
| d887 wrapperで保存PR-DF信号入力のmetadata | PASS | PASS | PASS | 保留 |
| d887 wrapperで保存PR-DF compile入力のmetadata | PASS | PASS | PASS | 保留 |
| d887 wrapperで保存PR-DF監査入力のmetadata | PASS | PASS | PASS | 保留 |
| 専用通常mailboxの照会 | executor所有 | 空、0 attempts | 空、0 attempts、受付OFF | 無効 |

[9 PASSの集約](PRDF_INTEGRATED_METADATA_CHECKS.json) は各元READINESSのhashも保存した。
検査は `dry_check_only` の保存入力・runtime・SOURCE照合で、科学bodyやcompileを実行していない。
将来生成される正確なREQUESTはその都度既存のmetadata admissionを通す。
人工通し検収の結果と科学実行許可は別であり、今回の通常テンプレートの各予算0と
`science_approved:false` は維持した。

新規設定を作る場合は登録済みファイルを上書きせず、新しい出力directoryで
既存 `prepare_romutail_controller_deployment.py` と明示layoutを使い、固定SOURCE aliasと
ホスト実パスを確認してから一度だけ登録・current切替する。今回の実配置layoutは
[LAYOUT.r4.json](LAYOUT.r4.json)。登録・切替の通常手順は上位HANDOFFを参照する。

server1復帰時は、そのホスト専用d887 transport checkoutと保存済みnative provider view、
Pythonを配置し、12セルの保存入力検査と人工mailbox切断・回収検収を行う。
その後、新revisionでserver1 profileを登録し、HOST_ADMISSIONとRELAYの無効指定を解除する。
既存runへの参加はそのrunの固定科学SOURCEと予約情報に合わせる。今回は復帰を待たず、
server1へのSSH・計算起動は行っていない。
