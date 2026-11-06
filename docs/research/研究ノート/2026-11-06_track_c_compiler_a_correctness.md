# Track C compiler A限定実行（2026-11-06）

基準05d9c5d、ユーザーの明示実行許可後に実施。[結果](../track_c_compiler_a_correctness_20261106/REPORT.md)。

定義ベースloweringと固定gateの正確な逆相殺に限定し、微小CRZの消失を解消した。人工adjointのstrict global phase差は初期失敗記録を保全し、control0側だけで共通joint phaseを校正する物理契約で評価した。制御後回路に限定し、出力への新controlは未資格。

人工44ケース、H4 LD3 S2/LD12 Y4・h=.02・q1のX/Y全512次元operator、同profile強baseline、保存Hamiltonian targetへの負half検証は数値PASS。q1の差を全18round保証に外挿せず、native RPEは未資格とした。価格・T合成・C-S/C-P追加なし。旧run01とSTOPを変更せず、今回もmandatory STOP。次はユーザーが全round資格の範囲と新価格cohortを判断・許可し、その許可後にCodexが実施する。
