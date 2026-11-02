# H4/H6/H8主表6条件とH2追加2条件の照合

文書上の並列表。既存主表の置換・cohort統合・系サイズ効果の解析ではない。全行linear 1 Å/STO-3G、T=.8/P=0、通常signed/倍率1、tol-only1e-8、full Hadamard wrapperのcompiled RZ×保存analytic shots。exact費用・SEは保存値の転記。

|区分・系/DF|ε|PR / PF|L_D / δ|n|保存G_PR|paired SE|保存G_PF|保存方向判定|低い点費用|
|---|---|---|---|---|---|---|---|---|---|
|既存主表 H4 DF10|0.05|B2_p3_q1_2nd_r4_K2 / B0I_p5_q1_2nd|3 / 0.8|1108|39405103093/554|174917.73703479147|70108214|NOT_REPRODUCED|PF|
|既存主表 H4 DF10|0.0005|B2_p2_q8_2nd_r2_K4 / B0_p7_q8_2nd|2 / 0.1|78|193598605843204/39|14140828222.905321|7856558002614|REPRODUCED|PR|
|既存主表 H6 DF19|0.05|B2_p4_q2_2nd_r4_K2 / B0I_p7_q1_2nd|4 / 0.4|48|10144814685/16|4054009.1059934883|356097424|REPRODUCED|PF|
|既存主表 H6 DF19|0.0005|B2_p4_q8_2nd_r4_K2 / B0I_p11_q4_4th|4 / 0.1|104|1709532575849751/52|65539320043.303085|36384720355183|REPRODUCED|PR|
|既存主表 H8 DF28|0.05|B2_p8_q2_2nd_r4_K4 / B0I_p11_q1_2nd|8 / 0.4|48|9840910801/8|3175066.353534034|1267321444|POINT_REPRODUCED|PR|
|既存主表 H8 DF28|0.0005|B2_p7_q16_2nd_r4_K4 / B0I_p15_q4_4th|7 / 0.05|48|546493203592571/6|144211830223.71762|94193016652860|POINT_REPRODUCED|PR|
|追加系列・本文未採否 H2 DF3|0.05|B2_p1_q1_2nd_r2_K4 / B1_p3_q1_2nd|1 / 0.8|80|384135011/80|15992.018778605378|5533984|REPRODUCED|PR|
|追加系列・本文未採否 H2 DF3|0.0005|B2_p1_q4_2nd_r2_K4 / B1_p3_q4_2nd|1 / 0.2|12|998482376201/4|276760628.92856276|305695076994|REPRODUCED|PR|

H4粗精度のNOT_REPRODUCEDとH6粗精度のPF低費用を保持する。REPRODUCEDはPR低費用の同義語ではない。H8の状態はREPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTSのまま。H8主表2条件はFigure2B/Mechanism H8_CONFIRM_1/3と同じ観測で、追加cohortではない。

H2は旧v1対象外。Fはdense eigh/sector4で、H4 dense sector36、H6/H8の記録上eigsh/sector400/4900と生成・正規化・残差gateの完全同一性を認定しない。H4 legacy DF12と新DF10も分離する。各行のSOURCE・結果・公開監査状態はCSVとstage_evidence.jsonおよび固定H468 packetに結合する。

有限標本点、保存SE、経験的bias allowanceに限定する。母平均優位、実power、希少層の費用/分散、厳密bias、全域最適、FT T資源、実量子shots、全energy/RPE費用は保証しない。新しい比・削減率・CI・統計量は作成していない。H2の元Pに比/削減率は保存されていないため空欄を保持する。
