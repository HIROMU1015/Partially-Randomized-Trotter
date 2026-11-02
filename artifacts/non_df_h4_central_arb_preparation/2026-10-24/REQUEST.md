# 利用者指示と今回の承認境界

2026-10-24：H4 Pauli central-control Arb包含評価の準備。
基点 `codex/non-df-h4-central-r1-sine-20261024`、
`36e993e698953f24415f4e90c4cac199a693ee16`。

利用者の明示範囲：

> 今回承認するのは技術実装、人工検証、実行準備までです。

- 既存C/R0/R1・保存QPY・state/ray・C-targetを確認し、適合部分を再利用する。
- native rz/sx/x/cx、phase/control/wire、cosine/sine、raw normを保持するArb bridgeを準備する。
- 人工2〜3qubitで包含性を検証する。
- PF cosine 1件を最小実H4候補にし、精度/call/STOPを定めた**非実行契約**を作成する。
- 実H4 Arb評価、新compile/sampling/Hamiltonian/state生成、N/G/R2/正式受理は禁止。
- 新科学grant発行・消費は禁止、契約は無効のまま、旧SOURCE/結果/grant/ledger/marker/STOPを保持する。
- 専用branchにcommit/pushしremote SHAを確認、認証失敗なら保全して報告する。
- 準備完了後mandatory STOP。

CPU/RAM/worker数の任意上限なし。共有環境・他ユーザーの設定やプロセスに変更なし。
検査は主agent本人が行い、別reviewerによる独立reviewとは呼ばない。
