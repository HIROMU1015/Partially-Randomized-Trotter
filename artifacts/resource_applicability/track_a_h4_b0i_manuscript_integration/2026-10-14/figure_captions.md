# v0.4の新しい実測図：caption・対応データ

共通条件：linear H4/STO-3G/固定DF12、6 geometry、prefix3/4/5/6/9、q1/2/4/8、δ=.1〜.8、T=.8、二次PF/ordinary control/structured Gaussian/fixed compiler/P=0。
実行SOURCE924b779c…、結果commit98cdb35c…。εはcomplex signal。302精度は同じ保存標本の再会計。全240軸のB0+I compiled metricsを使用し、旧未測定感度の図4を転用しない。

|図|対応CSV|Caption・解釈|
|---|---|---|
|[図1 method地図](figures/figure_1_measured_method_map.png)|`cost_factors_1812.csv`, `method_grid_regions.csv`|元登録集合＋B0+Iの費用点最小397/1415。離散geometry row、log εの描画cellは中点。連続geometry/精度の物理的補間・厳密境界ではない|
|[図2 費用比](figures/figure_2_measured_cost_ratio.png)|`cost_factors_1812.csv`|G_B0I/G_B2。双方family minimaを各εで再選択。黒点ε=.005/.05/.1、橙表示は保存点で比<1。線は表示用で連続精度の証明ではない|
|[図3 費用因子](figures/figure_3_shot_circuit_factors.png)|`selected_factors18.csv`|総shot比×有効RZ/shot比=総費用比。比1を基準に棒を描画、log軸。構成再選択を含み、同一回路の物理応答へ帰属させない|
|[図4 paired SE差](figures/figure_4_paired_sample_comparison.png)|`uncertainty_comparison_1812.csv`|100(G_B2−G_B0I)/G_B2と±2SEの記述的帯。normalizerは固定保存点推定。symlog軸。397点中374点はB2点推定−2SEも下回るが、正式confidence interval、多重選択補正、母平均保証ではない|
|[補助S1 matched bias/shot](figures/figure_S1_matched_phase_bias_shots.png)|`matched_B0_B0I_120.csv`, `selected_factors18.csv`|ε=.05でB0+Iが選んだ6構成のprefix/qを固定し旧B0と比較。上段biasとε/√2、下段必要shotのlog軸。片軸headroom非正なら全構成不適格として両軸shot欠測を表示。phase保持の算術効果であり、signalが一般に改善する定理ではない|

各図は同basenameのSVG/PDFを収録。5図を目視確認し、図3の比<1とS1の不適格を欠落・費用0に見せないよう表示した。
旧v0.3の4図は元path/bytesのまま保全。これらと今回図の番号は別draftの独立番号である。
