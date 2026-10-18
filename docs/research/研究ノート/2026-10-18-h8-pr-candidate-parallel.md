# 2026-10-18 H8 A_COST：局所gateと引継ぎの準備

利用者の並列実行準備依頼に従い、[候補単位の投入方式](../track_a_h8_pr_candidate_parallel_readiness_20261018.md)をSOURCE `87f1f0d528478ad212ba98159db06eec3c78433e` へ固定した。旧runの全16代表待ちを候補ごとのnative/control gateへ変え、元seed registryと科学kernelを保持する。元run/source/grantや他jobには書き込まず、独立cloneで準備した。

引継ぎは元process退役とnative parent receiptを要求する。exit0/reap/SHA・paired/event/compiler/input/controlが揃ったtaskだけ受け入れ、native証拠未確定分は再計算対象として旧予約/消費を保持する。科学的STOPを意図的handoff STOPへ置換しない。人工24検査・既存承認境界18検査・未承認CLI2拒否PASS、追加の実科学起動0。

ユーザーが切替を指示した後にCodexがfresh確認・所有停止/回収・最終seal・新grant・残task継続へ進む。旧runが完了した場合は追加compileを行わない。今回は準備のみで停止し、既存H8は継続する。
