# 2026-10-16 M1-S Gate 0：数値精度の式候補を準備

基点4019d045の9未決欄に対し、input L1尺度、γ_k、reference/Pauliの演算構造から8件の条件付きformula候補を具体化した。
eps_state_residualの真のoracle quality係数ηはroundoffだけでは決まらず、nullを維持する。
W_floorはrho resolutionから導出し、任意fixed floorを置かない。
v3はnew namespaceでscalar/allowlisted formulaを選択できる。観測残差/W/v/ρをresolver入力にしない。
式が既存resolutionを満たせなければ未評価/precision unresolved。閾値調整、scalar fallback、研究GO/STOPを自動化しない。

[設計](../../tracks/algorithm_codesign/m1s_gate0_preregistered_error_formula_design_20261016.md)、
[最終契約v3候補](../../tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md)、
[監査資料](../../../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1)を参照。
共通運用2026-10-16のユーザー指定原本をbytes不変で取り込み、旧版は履歴として保全。
数値correctness/BLASモデルのcoverageは未検証。旧source・結果・scope・auth・marker・STOPは不変。
新科学計算/Pauli表/数値correctness test/formula数値評価/authority/markerはいずれも0。
68静的checks PASSはAST/schema/guardに限るlocal技術証拠。immutable CI/外部再現/科学的結論ではない。
次はユーザーの最終契約採択、その後に別明示Gate 0実行承認。mandatory STOP。
