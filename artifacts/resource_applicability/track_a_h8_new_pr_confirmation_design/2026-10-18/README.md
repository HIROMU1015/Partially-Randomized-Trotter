# H8新PR2候補：独立確認の設計のみ

[資料入口](../../../../docs/research/track_a_h8_new_pr_confirmation_design_20261018.md)。
起点結果64c85f7a…、通常ε=.05のp8/q2/r4/K4とε=.0005のp7/q16/r4/K4。
H8/1Å/STO-3G/DF28/T=.8/P=0、元S2/finite-RTE。H6高次PFとは別契約。

- `plan_PROPOSED.json`：各48 trajectory・合計192 compileの案。実行未承認・未seal。
- `sample_size_scenarios.json`：既存SEからのnormal近似と、保存lawでのrare可視性の計画算術。
- `grants_INVALID.json`：無効な2 scope案。future SOURCE/input/runtime/seedは未固定。
- `verification.json`：metadata16件と人工11検査、新sampling/compile/native build0。
- `files.json`：今回の軽量成果のbytes/SHA、自己参照除外。

normal nは実power/coverage保証ではない。n48は点費用の独立再現を調べる案で、
未観測Taylor層を費用0とせず、母平均優位は未確定に保持する。
新しいexecution entry、有効grant、seed割当、実科学計算はない。mandatory STOP。
