# 2026-11-04 Controlled RTE・Hadamard局所probe：selector不在STOP

固定SOURCE `76d25ac69c22da6efeec673dc36c5fe04b71533f`の実行許可後、専用worktreeでrun01を1回起動した。
H4 linear1Å/STO-3G/legacy DF12、Partial L_D3/δ=.02/K2、保存physical-state ray/sector70、
extract_identity_phase/cutoff1e-12、r1,2,4,8,16,32×q_probe1/2の予定12request。
H/state load、DF preparation、support登録の各1後、固定component不在でexit1／STOPPED_WITH_ERROR。
config/θ replay、参照、回路build、state-action、Hadamard平均、sampling/compileは0。

停止後の保存exportの文字列照合で、固定ID df-fragment-10:Z0／df-fragment-11:Z0は両方存在しないことを確認。
準備時に旧B cutoff0のIDを固定cutoff1e-12 payloadと照合しなかった実装上の不足。
人工60PASSを実selector可用性の確認として使えなかった。科学的不一致の観測ではない。

[報告・原RESULT](../rpe_controlled_hadamard_local_probes_execution_20261104/REPORT.md)に原本・call数・次の提案を保存。
今回の対象を黙って差し替えず、source変更・retry・新selector選定へ移らない。
次は別指示後のCodexによるselector/profileの静的再設計・別SOURCE準備を提案し、
その科学再実行はユーザーの別途明示許可を要する。
全体UNDETERMINED/false、旧条件付き判定・raw・grant/ledger/marker/STOPを保持し、公開照合後mandatory STOP。
