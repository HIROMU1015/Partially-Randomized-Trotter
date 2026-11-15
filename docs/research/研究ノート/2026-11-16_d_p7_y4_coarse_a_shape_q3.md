# 粗幅p7／標準Yoshida4：保存A_SHAPE配分のQ3適合

保存H4 linear1Å/STO-3G/tol-only DF10・B0I p7/標準Y4/PF要求幅32h/5/h_RPE=.1・P0除外。ユーザー承認に従い、保存粗幅13round信号と現幅Y4の保存A_SHAPE整数列 `[37,34,32,29,27,24,22,19,17,14,11,9,6]` を再利用し、専用互換Python/元PF workerでQ3を1件実行した。

科学実行SOURCE1cf462e3a、raw442a386bc、C結合8855f420a。数値・保存gateは正常、MSE区間は表示で `[2.074377165792602390524467e−6,2.074377165792685953356238e−6] Ha²`。元exact τ以下で「Q3適合・価格待ち」とした。幅16h/3・32h/5×B_PRICE・A_SHAPEの4点は[REPORT](../d_p7_y4_coarse_a_shape_q3_20261116/REPORT.md)／[SUMMARY](../d_p7_y4_coarse_a_shape_q3_20261116/SUMMARY.json)。他3点は再Q3せず固定保存結果を引用した。

前回の粗幅B_PRICE不適合はその配分に限定して維持し、旧UNKNOWNとSTOPも保持。粗幅総費用はnullで、最小N・費用順位・G_qualified・winnerは未判定。新信号/reference/build/compile/価格fit/別配分Q3は0。統合担当へ適合点1件追加・価格待ちの差分を渡し、公開・照合後mandatory STOP。
