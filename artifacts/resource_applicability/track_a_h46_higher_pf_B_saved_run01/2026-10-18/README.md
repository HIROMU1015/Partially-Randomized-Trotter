# H4/H6 高次PF：B_SAVED_ONLY run01

[結果報告](../../../../docs/research/track_a_h46_higher_pf_B_saved_run01_results_20261018.md)。

状態：保存値解析一回がnative exit0、終了gate PASS、mandatory STOP。24条件中18適格・6不適格、適格な費用欠測12条件。6層ともfull-grid winnerはnull。新signal/sampling/compile/係数最適化0。

- `result.json`：封印SOURCE runnerの未加工出力。24評価行、6選択層、6点費用差。
- `candidate_summary.csv`：上記出力の表示用転記。欠測は空欄（0ではない）。
- `summary.json`：実行範囲・不足12条件・未確立事項。
- `completion_audit.json`：保存6群のshot・paired費用・平均/SE再現、source/input不変、native回収、消費marker1。
- `approved_plan.json` / `grant.json` / `consumed_grant_marker.json`：利用者承認に結合した過去一回の実行証拠。grantは消費済みで再利用不可。元の無効draftを上書きしない。
- `fresh_gate.json`：remote SHA/公開3file・source/input/runtime・利用可能資源の起動前証拠。
- `native_exit.json`：native wait4・exit0、wall/RSS。
- `followup_scope_proposal.json`：適格な不足12条件。n未決定、未承認、未起動。

SOURCE `ed14d951a4761838f25aeb16a653b55aba499c34`、実行前公開commit `8464c06a333686d2b503ffb26b2014112bc973b2`。
元の1Å/STO-3G、H4 rank12/p3・H6 rank19/p5、T=.8/K4/ε=.001、α_axis=.025、経験的allowance1e-9、P0を維持。n8を探索/確認間で二重計上しない。厳密bias・母平均winner・最適q・RPE全round成功保証は未確立。

これは再実行手順ではない。消費grantで再実行しない。新native取得・独立確認は別承認。
