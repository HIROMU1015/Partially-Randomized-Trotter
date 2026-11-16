# サーバー1からサーバー2担当への先行引渡し

固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
branch `codex/rpe-distributed-q3-server1`。[REPORT](REPORT.md)。

共通実装は `codex/rpe-distributed-q3-coordinator` の
`docs/research/rpe_distributed_q3/HANDOFF.md` から固定SHAと起動手順を取得する。
サーバー1側で候補選択器・分散queueを独立実装しない。

## 訂正後の接続方式

サーバー1は `AbeHiromu@GPUA100-gleap-01` / `192.168.5.23`。
SSH認証設定・転送agent・共有mountは現環境で見つからない。
利用者の訂正：ローカル0から `ssh qg1`・`ssh qg2` が利用可能。サーバー1→2直接接続は使わない。
ローカル0の中継プログラムが空き枠照会、割当済みREQUESTの配送、保存結果回収を行う。
サーバー1は受領済みjobを中継から独立して実行し、同一job IDのspoolを保持する。
中央候補選択・予算・採用はサーバー2のみが所有する。
このsessionはサーバー1内にあり、ローカル0のalias・認証はここにはない。
新しい認証情報を公開branchへ記載しない。

## サーバー1の作業場所

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-server1
git rev-parse HEAD
git ls-remote origin refs/heads/codex/rpe-distributed-q3-coordinator
```

数値Python候補 `/home/AbeHiromu/venvs/trotter-common/bin/python` (3.12.3)。
初回profile：同時Q3=1、DP workers/Q3=2、BLAS/OMP threads=1。
サーバー2固有の絶対view/interpreter pathを、そのままサーバー1へ持ち込まない。
科学SOURCEとguardを保持し、ローカルpath対応表は後続検査結果に記載する。

## 実機検収の記録欄

|項目|状態|
|---|---|
|共通worker固定SHA|公開待ち|
|worker起動/状態/停止/再開コマンド|共通CLI待ち|
|同一run ID|未発行|
|サーバー1取得job ID|未取得|
|サーバー2取得job ID|未確認|
|両機の中央採用記録|未確認|
|計算後転送中断の専用job ID|未発行|
|再接続時の再計算0の記録|未検収|

ここまでの成果は環境整備の先行公開。2台実機検収完了ではない。

## 先行検査の更新

共通branch `b9a39567a51bd27ca89e751d4f42e170af7e6c71` は仕様公開のみ。
起動可能なworker、中継、人工runを指す固定SHAの追記待ち。
PR-Pauli LD3/LD5の既存dry-checkはPASS、PR-DFは固定caller内部のホスト絶対パスでSTOP。
人工prefix回帰はPython3.12で固定AST guardが拒否。詳細・rawログは [REPORT](REPORT.md) と [checks](checks/)。
ローカル0での操作はこの端末から実行できたと扱わず、実job IDと中央採用receiptが揃ってから検収済みへ変更する。
