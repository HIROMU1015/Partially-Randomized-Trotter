# サーバー1：分散Q3 worker 実機準備・検収

2026-11-17 JST。対象 `HIROMU1015/Partially-Randomized-Trotter`。
固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
公開branch `codex/rpe-distributed-q3-server1`。

## 先行公開：接続と環境の確認

状態：`PREPARING / COORDINATOR_HANDOFF_PENDING / CONNECTION_DETAILS_PENDING`。
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
