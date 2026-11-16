# 分散Q3：サーバー1・ローカル担当への共通仕様

公開branch: `codex/rpe-distributed-q3-coordinator`。固定基点:
`5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
初版は実装前の接続仕様。実装SHA・実測構成は後続追記する。
今回の対象は新しい人工runと次run用の実装。稼働中の実H4探索へ導入しない。

## 接続方向

controllerはサーバー2の通常 `scripts/run_rpe_schedule_search.py` が1つ所有する。
HTTPは `127.0.0.1:8763` にだけbindし、workerが空きslotを申告してpullする。
公開ネットワークへHTTPを開放せず、既存許可済みSSH port-forwardで接続する。
共有filesystemは使わず、REQUESTと成果物をHTTPで転送する。

利用者確認: **ローカル0**からサーバー2は `ssh qg2`、サーバー1は `ssh qg1`。
ローカル0は計算workerとして指示された「ローカル」と別ホスト。
サーバー1担当の直接接続確認結果はまだ未取得。次のrelayは接続案であり実測済みでない。

```bash
# ローカル0: terminal A。qg2のcontrollerをローカル0へ転送
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18763:127.0.0.1:8763 qg2
# ローカル0: terminal B。qg1のloopbackを上記転送へ接続
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:18763:127.0.0.1:18763 qg1
# サーバー1 workerの接続先は http://127.0.0.1:18763
```

worker側からサーバー2へ接続できる場合は、worker上の
`ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18763:127.0.0.1:8763 <確認済みserver2-alias>`
でもよい。逆方向しか使えない場合はサーバー2からworkerへの `-R` を使う。
SSH設定・鍵・共有サービスは変更しない。トンネル停止はその `ssh -N` processのCtrl-C。

## 固定入力とホスト別path

各担当はこのbranchをfetchし、後続指定の実装SHAを専用checkoutに展開する。
実routeは `view_commit` ごとの固定detached viewと、その入力locatorのGit objectsを配置する。
既存のSOURCE検査に必要な祖先objectとtracked sourceも保持する。
大きい入力は担当間で既存の許可済み転送手段を使い、commit:pathとSHA-256で照合する。
controllerのREQUEST原本と科学job IDは変更しない。
worker profileの `route_views` でcommit→ホスト別絶対pathを解決し、
`interpreter` と `prefix_cache_directory` もホスト別に指定する。
解決結果は `RESOLVED_REQUEST.json` に分離する。DP cacheは各ホスト内に保持する。

予定profile形式（pathは各担当の所有場所へ変更）:

```json
{
  "host_id": "server1",
  "root": "/OWNED/rpe-distributed-q3-coordinator",
  "work_dir": "/OWNED/rpe-distributed-fixture-worker",
  "interpreter": "/OWNED/venv/bin/python",
  "slots": 2,
  "route_views": {},
  "prefix_cache_directory": "/OWNED/rpe-distributed-prefix-cache"
}
```

## 取得・完了・再接続のプロトコル v1

- `POST /v1/poll`: `host_id, slots`（現在の空き数）, `cached_prefixes, source_sha`。
  controllerが `assignments` を返す。各assignmentは `job_id, method, attempt_id,
  request, request_sha256, source_manifest, input_hashes, prefix_key`。
- `POST /v1/state`: `host_id, job_id, attempt_id, state, detail`。
  状態は `ASSIGNED`（未起動）, `STARTING`（起動境界）, `RUNNING`,
  `UNKNOWN`, `COMPLETED_PENDING_TRANSFER`。観測が途切れても割当を他ホストへ移さない。
- `POST /v1/complete`: 同じhost/job/attemptと `request_sha256`、
  `files`（relative path→base64）。成果物と親子processの終了記録を保存してから通知する。
  controllerが `ok, duplicate` を返す。応答消失時は同じ完了通知を再送する。

科学job IDは既存の方式/configuration/shot vector/route kind/評価契約のkey。
host IDとattempt IDは別欄。永続化した割当と科学予算予約はcontrollerが一元管理し、
二重取得・二重予算計上・二重完了採用を防ぐ。応答timeoutに自動lease expiryを設けない。
worker再開は保存spoolとPID/成果物を照合し、起動済みか不明なものを自動executeしない。
固定SOURCE、入力hash、routeと評価契約をworkerで検査し、既存adapterの
`dry-check / execute / collect` と元の終了gateを使用する。

候補・方式別最安・科学予算・採用順はcontrollerの既存探索規則が所有する。
方式/L_D/幅をホストに固定しない。prefixの局所性は既存優先順位を保つ範囲で、
空きslotと待ち時間を考慮して利用する。

## 人工検収への参加

人工fixtureはこのdirectoryの `fixtures/` と後続の
`scripts/check_rpe_distributed_q3.py` で配布する。
workerが動くcommitで、次のコマンドと実装SHAを確定して追記する。

```bash
# 全ホスト: 固定checkout内で。新規の人工専用spoolを使用
python3 -B scripts/run_rpe_q3_worker.py --profile OWNED_PROFILE.json \
  --controller http://127.0.0.1:18763
```

検収順は server2+server1 → server2+server1+local。
同じ人工探索を通常入口から実行し、方式別最安への複数ホスト結果の反映、
後続候補選択、単一ホスト参照とのMSE上下界/判定一致、切断後回収、
controller再開時の完了job二重起動0件を確認する。
host別起動/完了/再利用数、queue待ち/転送時間、全体wall timeを保存する。
同一マシン上の複数spool検査は模擬ホスト検査と明記し、実3ホスト検収と区別する。
人工負荷の速度倍率は実H4性能を示さない。

担当側で接続方向・profile・検収結果を公開する際は、当branchを基点として
担当専用branchの `docs/research/rpe_distributed_q3/hosts/<host_id>/HANDOFF.md`
に記載すると統合時に追跡できる。共通branchをforce-pushしない。
