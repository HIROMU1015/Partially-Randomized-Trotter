# RomuTail：計算用ローカルQ3の参加・離脱・復帰

2026-11-17。成果branch: `codex/rpe-distributed-q3-local`。
固定基点: `5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
[共通HANDOFF](../../HANDOFF.md)を直接確認し、公開された共通実装
`89f51a4fef47ecd2364d22c4c5cc2e225df3f2ce` を取り込んだ。

利用者確認：**ローカル0→qg2、ローカル0→qg1、ローカル0→RomuTail** が使える。
この実行環境が `RomuTail` のLinux側。サーバー1/2にRomuTailのaliasは不要。
採用経路はローカル0のSSH mailbox relay。候補選択・予算・結果採用はサーバー2だけが行う。
実際のローカル0→RomuTail接続はこのセッションから再測定できていない。

**準備・模擬検収済み。実2台検収後の実3台参加は未実施。新規H4は起動していない。**

## RomuTailに用意した配置

- 専用checkout: `/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local`
- 実worker用固定view: 上記の `.local-q3/common-89f51a4`（HEAD=89f51a4…、tracked変更なし）
- Python: `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`（既存3.11.0rc1）
- 設定: `.local-q3/LOCAL_SETTINGS.json`、参加profile: `.local-q3/worker.json`
- spool: `.local-q3/spool`、cache: `.local-q3/prefix-cache`。復帰まで保持する。
- 固定route: `.local-q3/views/<full SHA>`。対応表は `.local-q3/route_views.json`。

workerはcontrollerとHEADが完全一致する必要がある。**ローカル固有branchをそのままworker rootにしない。**
現在のcloneは共有Git objectsを使うので元repositoryを残す。別PCへ持ち出す場合はobjectsも取得する。
Windows本体のWSL一覧は未取得だが、この既存Linuxを使う経路に新規WSL導入は不要。

```bash
# RomuTailの専用checkout内。設定変更だけなら --smoke-output は不要。
Q3_PYTHON='/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python'
"$Q3_PYTHON" -B scripts/prepare_rpe_q3_local.py \
  --state-dir .local-q3 --slots 2 --dp-workers 2 --interpreter "$Q3_PYTHON" \
  --worker-root .local-q3/common-89f51a4 --route-views .local-q3/route_views.json
cat .local-q3/READINESS.json
```

`slots` は同時Q3数。`dp_workers` は1件内のDP数で、単独人工smokeでは設定を適用済み。
共通89f51a4の実workerはREQUESTのDP数を継承するため、現在のworker.jsonにはDP上書きを入れていない。
このbranchにはhost profileの `dp_workers` を `RESOLVED_REQUEST` だけに適用する最小拡張と8テストを収録した。
controllerとworker双方の固定共通SOURCEへ拡張を取り込んだ後に、準備コマンドへ
`--host-dp-override` を追加すればローカルDP数も独立に変更できる。元REQUEST/hash/job IDは変更しない。
未統合のまま89f51a4でローカルDP数が反映された扱いにはしない。

## ローカル0の普段の4操作

ローカル0の **qg2/RomuTailが使えるLinux/WSL** で行う。共通relayにはLinuxのfcntl依存がある。
ローカル0のこの環境・Python・alias解決は、このセッションからは未測定。
まず最初の2台検収（qg2/qg1の両人工結果採用→次候補をqg1へ配送）を完了する。
その後、サーバー2担当の人工専用runのport/SHAを合わせて参加する。

[relay.local0.example.json](relay.local0.example.json) をローカル0の所有場所へコピーし、
`state_directory` をローカル0の絶対pathへ変更する。`remote_port=8765` は人工fixtureの例であり、
サーバー2担当の指定値へ合わせる。local_port=18764は既存server1 relayとの衝突を避ける例。
worker側の絶対pathは上記実配置を記載済み。鍵・パスワードは含まない。

```bash
# ローカル0。このbranchのrelayを使う（再開時の古い停止表示の競合を修正済み）。
Q3_CONTROL_CHECKOUT=/absolute/path/to/local0-checkout
Q3_RELAY_CONFIG=/absolute/path/to/relay.local0.json

# 参加：専用relayをチャットと独立して起動
python3 -B "$Q3_CONTROL_CHECKOUT/scripts/run_rpe_q3_relay.py" --config "$Q3_RELAY_CONFIG" --start
# 状態確認：workerごとの状態・heartbeat・通信エラー
python3 -B "$Q3_CONTROL_CHECKOUT/scripts/run_rpe_q3_relay.py" --config "$Q3_RELAY_CONFIG" --status
# 新規受付停止：受領済みjobはRomuTail内で完了して保存される
python3 -B "$Q3_CONTROL_CHECKOUT/scripts/run_rpe_q3_relay.py" --config "$Q3_RELAY_CONFIG" --stop
# 再接続：同じ設定・host ID・spool。未転送結果を回収してから空き枠へ次jobを取得
python3 -B "$Q3_CONTROL_CHECKOUT/scripts/run_rpe_q3_relay.py" --config "$Q3_RELAY_CONFIG" --start
```

停止要求後は `--status` の `stopped: true` を確認する。停止中の結果は次の再接続まで未転送。
PCを落とす前に、RomuTailで次を実行し、各attemptが `COMPLETED_PENDING_TRANSFER` / `TRANSFERRED`
になったことを確認する。`UNKNOWN` は再実行せず状態を調べる。

```bash
# RomuTailの専用checkout内。relay停止後のworker状態確認。
"$Q3_PYTHON" -B .local-q3/common-89f51a4/scripts/run_rpe_q3_worker.py \
  --profile .local-q3/worker.json --mailbox status < /dev/null
```

## 確認範囲と残作業

[詳細](READINESS.md) / [固定89f51a4の途中参加・復帰検査](checks/LOCAL_JOIN_CHECKS.json)。
同一Linuxの模擬3workerで途中参加・切断中の他job続行・同一attemptの結果回収・重複完了の冪等性がPASS。
人工参照18件とMSE上下界/判定、方式別最安が一致し、ローカル役は7件採用された。
人工adapterはDPを使わないため、このrunのcache HITは0。別途、既存人工DP cacheの10テストはPASS。
共通worker7テスト、ローカル拡張8テスト、修正後relay2テストもPASS。
PF/Pauliの実route metadataは4経路PASS。実PR-DF routeは内側のサーバー絶対path不足で保留。

次の担当：サーバー2担当が実2台検収・接続port/固定SHAを確定し、必要ならDP拡張を統合する。
利用者がローカル0で上記relayを起動する（このセッションからローカル0のプログラムは起動できない）。
Codexは既許可の人工専用runで実3台の途中参加・離脱・回収・二重採用0・DP cache再利用を確認する。
新規H4計算の許可には広げない。
