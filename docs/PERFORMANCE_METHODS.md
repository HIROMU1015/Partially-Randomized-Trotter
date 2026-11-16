# 性能対策の再利用索引

既存の大規模計算の手法集は固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を参照する。本C branchへその科学runnerを取り込んだ意味ではない。

## 保存RPE結果の読取

`src/trottertracks/saved_rpe_cost_catalog.py:Reader.document` は既存Git blob readerを使用し、固定commit/pathをprocess内でcacheする。同じSUMMARY/INPUTS/価格表が複数HANDOFF・配分から参照されてもJSONの再読取を省く。seed/計算順/価格値は変更せず、適用対象は保存JSONのmetadata処理だけ。

既存B2の保存summarizer同様、Python process内の整数文字列桁上限を解除し、exact MSEの長い整数を損失なく読み書きする。共有環境設定や科学runtimeを変更しない。

[再利用・検査・実行範囲](research/h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)と[検収](research/h4_saved_rpe_cost_catalog_20261116/TESTS.txt)を参照。今回の固定10入力の実処理時間はCATALOG.jsonに保存。人工＋公開値の読取で確認したが、速度倍率や将来の大入力性能は測定していない。

## RPE探索の共通非同期scheduler（D2、2026-11-16）

実装SOURCE `c3bb2ad2c409c48ab73708ca61aeab5b9323b317` の `src/trottertracks/rpe_optimizer_concurrent.py` と既存round CLIの `executor.scheduler="concurrent"`。元の構成内採否順を保ち、4方式のready候補を共通queueへ投入する。有限予約・先読み計数・重複共有・保存/live再開を含む。過去の失敗（blocking prefetched launchが回答取込を塞ぐ、再開live workerをcapacityに数えない、未採用先読みがnew_callsから漏れる）を人工検査した。

固定B2完了logでは18 slot利用率99.955%であり、現行実Q3の速度改善は未確認。人工4方式は3反復の対比較中央値3.52倍、単一方式の飽和例は短縮なし。DP worker profileの変更は既存accelerationパラメータだけで、AST/kernel/精度条件は維持。人工fork負荷の最適worker数は実DPへ移さない。実装者検査15＋既存関連17 PASS、保存C catalog再生・固定Pauliの元AST constructor確認、実Q3 0。適用条件・メモリの標本性・次runと再開commandは [REPORT](research/d2_rpe_optimizer_throughput_20261116/REPORT.md)／[HANDOFF](research/d2_rpe_optimizer_throughput_20261116/HANDOFF.md) を参照。
