# 2026-10-16 H7の限定pilot準備

H5完了後、利用者の「同様の計算をH7でも行えるか　まだ実行しなくてよい」「その方針で進めて」に従い、H7専用worktreeで準備した。中性ROHF・Na4Nb3/sector1225・T.8/q1の既定6構成、実rank未知/p=ceil(L/4)を維持する。

H5の100/400次元制限を外すだけで済ませず、CSR/referenceとbounded solver adapterを追加。H5 native・α/β別primitive・pidfd monitorを再利用し、通常4並列・thread1・hash/preparation/evolution reuseを接続した。SOURCEf93ad35b/54人工mock PASS、旧source/resultsを保全。実H7のscientific call/grant0、起動は別承認とfresh gateが必要。

詳細は[準備資料](../track_a_h7_neutral_open_shell_readiness_20261016.md)。H7でのspin純度・実rank・費用順位・実資源は未確認で、H5の点結果から結論を移さない。公開後mandatory STOP。
