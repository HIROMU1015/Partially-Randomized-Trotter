# 2026-10-11 Track A H6補完準備 v3

利用者がGPTの[MemoryError後独立レビュー](../track_a_h6_matched_v2_memory_stop_independent_review_2026-10-11.md)を採用し、補完準備を指示した。
同じ研究方針の再レビューや新しいH6実行は行わない。
[新版契約](../track_a_h6_matched_supplement_execution_contract_v3.md)を追加し、旧runのretry/resumeへ変更せず、旧完了証拠をimmutable入力とする。

静的に全92 signal、B3全84組、原登録592 group、449完了・902 wrapper、未完了143 groupを確認。
旧447の正確なallocation位置は未保存。旧source・grant・STOP・欠測は保持する。
既存disposable worker、streaming fingerprint、ResourceTraceを再利用し、single-owner wrapperと入力circuitの短い寿命を別sourceに実装。
原447を最初の一回取得とし、成功時のみ残142と原独立確認へ進む設計。
全探索集合gate、old/new sourceとresult hashの来歴、paired seed、no-pooling、最大798新wrapperを固定する。
synthetic bridgeは実H6 q8成功の証明ではない。AS8 GiB・compiler・科学gridを変更していない。

科学計算/signal/分子trajectory/H6 circuit build/compile0、新approved grant0。
local synthetic/process検査とsource/入力/保存rawの静的照合だけを行う。
source、sealed manifest、証拠registry、検証・保全記録をcommit/push/remote照合後、実行認可待ちでmandatory STOP。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false、H8・追加探索未認可を維持。
