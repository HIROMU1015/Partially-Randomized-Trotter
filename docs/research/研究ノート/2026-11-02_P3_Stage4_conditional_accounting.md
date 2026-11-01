# 2026-11-02 P3 Stage 4条件付き会計

ユーザー許可のH4/案A/M_B・M_C/5精度について、科学SOURCE`2f0d31f75ab1ecacc73d9ee5c34ddfc997396102`を人工7PASS後に公開・固定し、保存8JSONだけからscalar解析1batchを実行した。
[報告](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md)。10行/20軸すべて条件付き適格、DFは必要shot数が少なく、保存run04の各32価格を使うRZ総費用はPauliが小さい。G_P/G_DFは全5精度で約.45273。
nominal-reference端点からbias上界候補を直接作り、数値不確かさを一回計上。B上界もengineering仮定を明記し、scalar ceil20軸照合/増分0。cos/sin共分散と独立方式差の価格SEを保持した。
finite-bit gap=0/compiled ρ=0は仮定。母平均tail・実装精度・全RPE総費用は保証しない。新sampling/compile/Gaussian/reference/信号/finite bias高精度化0、exit0、wall約.19秒。
旧結果と全STOPを保全し、公開remote照合後mandatory STOP。次の担当はユーザー：結果の採用範囲と、finite-bit/compiled/tailの追加検証の要否・優先順位を判断する。
