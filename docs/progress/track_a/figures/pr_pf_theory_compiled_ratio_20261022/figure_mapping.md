# 図・元record対応

|図|panel|PR|PF|保存総費用record（stage3_shot_weighted_totals.json）|保存pair record（stage3_pf_pr_comparisons.json）|
|---|---|---|---|---|---|
|参考A/B|PR q1|B2_p3_q1_2nd_r4_K2|B1_p12_q1_2nd|/124 と /4|/53|
|参考A/B|PR q8|B2_p3_q8_2nd_r2_K4|B1_p12_q1_2nd|/184 と /4|/173|

全てRESULT 6e2b7eb6、H4、ε=.05、factor1、model NATIVE_MIXED、metric RZ。
保存shotはpreserved_shots.jsonの/20、/30、PF/0。候補は新しく選ばず両保存pairをそのまま表示。

figure_data.jsonのreferencesに候補別SOURCE・Hamiltonian/state/input/DF hash・n・cohortとexact fractionがある。
figure_data.csvは実際に描画した4つの候補表示枠（共通PFの再表示を含む）。
display_coordinates.jsonは2図×2panel×2系列の8表示値。Aは保存比と照合した除算値、Bは保存Gからの表示用削減率。

現行の20条件はcoverage_matrix.csvとfigure_data.jsonのcurrentを参照。
方式別80条件、旧S2候補25条件、拒否したIDだけの結合はcoverage_audit.jsonにある。
