# サーバー1：分散Q3 worker 実機準備・検収

2026-11-17 JST。対象 `HIROMU1015/Partially-Randomized-Trotter`。
固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
公開branch `codex/rpe-distributed-q3-server1`。

## 先行確認後の追補

先行文書はGitHubアプリ経由で `4b1225e59997780d1c5261e44e8b168d623394a9` に公開した。
通常HTTPS pushは `could not read Username` で失敗したが、アプリで同一treeを公開し、
fetch後にローカルbranchも同じcommitへ同期した。

共通branchの初版HANDOFF `b9a39567a51bd27ca89e751d4f42e170af7e6c71` を取得した。
これは実装前の仕様で、実装固定SHAは未発行。
ローカル0には `qg1` / `qg2` がある。サーバー1で実際に `ssh ... qg2 hostname` を試すと
`Could not resolve hostname qg2`。`127.0.0.1:18763` のTCP実接続もConnection refused。
したがって現時点はローカル0のSSHリレー起動待ちで、両機間接続は未成立。

既存adapterの `--mode dry-check` を13round・各軸2shotという**metadata専用入力**で実行した。
PR-Pauli LD3 / LD5は `DRY_CHECK_PASS`。分子信号再計算・実Q3は0。
ただしPauli dry-checkはAST constructorを検査しないため、Python 3.12で数値実行可能とは認定しない。

既存 `tests/test_rpe_dp_prefix_reuse.py` を固定DP viewで実行すると、Python 3.12では
10検査中1 PASS・8 ERROR・1 FAILとなった。原因は元guardの
`Reference total DP SOURCE changed; do not silently adapt`。元ソースを変更していなくても
Python版によるAST形式差で拒否される。失敗ログを保存し、guardや期待hashを変更しない。
既存PR-DF routeは明示的にPython **3.11.0rc1** を要求する。
公式archiveを今回専用領域へ取得し、同runtimeを構築中。

PR-DFはさらに、固定caller内部の絶対pathに別の固定kernel viewを要求した。
`27e70efe533862f38f4080aabb9cdd3ad1749fe9` を今回専用の疎worktreeへ展開し、
未存在だった既定path `.../partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115`
から同viewへ互換symlinkを作成した。元SOURCEや既存directoryを上書きしていない。
この依存はtop-levelの `route_views` mappingだけでは解決しない。[再現・不足](NEEDS_COORDINATOR.md)。

実測・固定入力所在は [ENVIRONMENT.json](checks/ENVIRONMENT.json)、
[ROUTE_INPUTS.json](checks/ROUTE_INPUTS.json)、失敗を含むrawログは [checks/](checks/) に保存した。
以下の「先行公開」は着手時点の履歴であり、追補を優先する。

## 先行公開：接続と環境の確認

状態：`PREPARING / LOCAL0_RELAY_IMPLEMENTATION_PENDING / TWO_HOST_ACCEPTANCE_PENDING`。
この時点では2台検収は未実施。新規H4科学計算は0件。

- サーバー1の実測hostname：`GPUA100-gleap-01`。実IP `192.168.5.23`。
- ログインユーザー `AbeHiromu`、uid/gid 30038。
- `nproc=128`。RAM約1.0 TiB、available約969 GiB。ローカルext4空き約456 GiB。
- 現sessionのcgroupはCPU `max 100000`、memory `max`。専有割当を意味しない。
- ユーザー `.ssh` は `known_hosts` 1件のみ。`config`・秘密鍵なし、`SSH_AUTH_SOCK`なし。
  システムSSH設定のInclude先も空。SSH接続元 `192.168.5.2` はdefault gatewayと同一であり、
  これをサーバー2とは認定しない。サーバー2のhost/user/portは未特定で、推測接続はしていない。
- `findmnt -rn -t nfs,nfs4,cifs,ceph,lustre,glusterfs` は該当mountなし。
  両機間の共有directoryは未確認。逆方向SSHも未検収であり、不可とは断定しない。
- `git ls-remote origin refs/heads/codex/rpe-distributed-q3-coordinator` は先行確認時点で出力なし。
  共通実装の固定SHA・worker起動方法はHANDOFF公開後に取得する。

## 隔離した作業領域

`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-server1`

元mainには既存の未commit変更があるため変更せず、上記専用worktreeを固定基点から作成した。
人工jobと固定route viewは今回専用の領域に置く。既存H4本番runを起動・再開・変更しない。

## Pythonと既存route

`/home/AbeHiromu/venvs/trotter-common/bin/python` はPython 3.12.3。
numpy 1.26.4 / scipy 1.14.1 / mpmath 1.3.0 / numba 0.60.0 / qiskit 1.3.0 /
openfermion 1.6.1 / pyscf 2.7.0 / pytest 9.0.3 が既存。gmpy2は未導入。
`/usr/bin/python3` は同3.12.3だが上記数値依存なし。

固定problem `configs/rpe_method_optimizer/h4_df10_problem.json` のサーバー2固有絶対path
`/home/AbeHiromu/projects/h6-environments/...` と `.../partially-randomized-trotter-worktrees/...`
はサーバー1にはない。固定routeのcommit objectは取得済み：

|方式|既存route|固定SHA|先行状態|
|---|---|---|---|
|PR-Pauli LD3|pauli_b_ld3|a844ac25f88c5cdf39e9e1837d85b6d4341774ff|ローカルview作成・dry-check待ち|
|PR-Pauli LD5|pauli_b_ld5|b6d4dea49c39f364048af422bc2928ee7c169261|同上|
|PR-DF|prdf_d2_split|23160cc876807c0300be3de811f767b5810640a7|同上。Python/runtime gateの確認が必要|
|人工DP既存参照|B固定DP|bd42dd1e95eb1d4152216d62792ad3421b803f7c|人工tableのみで再利用予定|

同時Q3件数と1 Q3内DP worker数は別設定。初回2台人工検収はサーバー1を **1 Q3 × 2 DP workers**、
その後の保守的初期候補は **2 Q3 × 4 DP workers** とする。128論理CPUを使い切る設定は採用しない。
人工DPで時間・RSS・逐次一致を確認後に更新する。人工小規模RSSから本番容量を外挿しない。

## 残る検収

1. 既存dry-checkと人工DPを実行しrawログ・固定入力・結果を追記。
2. 共通HANDOFFから固定SHA、サーバー2実接続先、同一runと人工jobの起動契約を取得。
3. 両機が異なる人工Q3を同時取得し、サーバー2の中央採用記録を照合。
4. 専用人工jobの計算完了直後の転送中断→再接続→再計算0での回収を照合。

ジョブID・中央採用記録・共通workerの起動/状態/停止/再開コマンドは未発行。
未確認値を補って検収済みとはしない。[HANDOFF](HANDOFF.md)を参照。

## 接続訂正を受けた追記（2026-11-17 JST）

上記は初回公開時の観測履歴。利用者の訂正を採用し、**ローカル0が既存alias `qg1`・`qg2`で両サーバーを中継する**。
サーバー1→2直接接続は使用しない。ここをローカル0と誤認しない。
この端末の `ssh -G qg2` は `hostname qg2` のままで、BatchMode接続も名前解決失敗。
ローカル0→両機の実接続・転送はローカル0側で検収する必要がある。
サーバー1のmount一覧にはネットワーク共有mountがない。両機で同一内容を読み書きできる共有領域は未検収。

共通branchで現時点に取得した固定SHAは `b9a39567a51bd27ca89e751d4f42e170af7e6c71`。
このcommitは仕様・人工REQUEST・profile例のみで、worker実行コードとローカル0中継コードは含まない。
HANDOFFのSSHトンネル案は訂正後の要求を実測した記録ではなく、正式中継プログラムの公開を待つ。

先行検査ログを [checks/](checks/) に保存した。

|検査|実測結果|範囲・不足|
|---|---|---|
|PR-Pauli LD3 dry-check|PASS|Python 3.12.3、固定view a844ac25f、metadata/countのみ|
|PR-Pauli LD5 dry-check|PASS|Python 3.12.3、固定view b6d4dea49、metadata/countのみ|
|PR-DF dry-check|STOP|固定caller内にサーバー2専用の絶対viewパス。FileNotFoundErrorでruntime gateより前に停止|
|既存人工prefix回帰|FAIL (1 failure / 8 errors / 1 pass)|Python 3.12.3で原DP AST hash guardが拒否。guard変更なし。指定3.11.0rc1で再確認中|

[ENVIRONMENT.json](checks/ENVIRONMENT.json) が更新時の資源snapshot。
CPU affinity 128・RAM約1 TiBは専有予約ではない。共有hostのload averageがCPU数を大幅に超えていたため、
初回の小さい人工検収だけ **同時Q3=1、DP workers/Q3=2、BLAS/OMP=1、nice=10** とする。
先行案の2 Q3×4 workersは承認済み容量ではなく、現負荷では採用しない。
本番Q3の最大RSS・所要時間は未計測で、人工DPのRSSから外挿しない。
