# Table 1 | Fixed independent-confirmation comparisons

Linear H4/H6/H8, adjacent separation 1 Å, STO-3G, tol-only 1e-8 DF10/19/28, T=.8, P=0, usual signed bias and allowance factor 1. Costs are saved analytic shots × measured compiled RZ; SE is the saved paired-trajectory cost standard error. Fixed PF controls are directly measured for these endpoints.

|ID|ε|L_D / δ|n|PR mean G (exact)|paired SE (saved)|PF G (exact)|saved direction test|lower point cost|
|---|---|---|---|---|---|---|---|---|
|H4_DF10_P_1|0.05|3 / 0.8|1108|39405103093/554|174917.73703479147|70108214|NOT_REPRODUCED|PF|
|H4_DF10_P_2|0.0005|2 / 0.1|78|193598605843204/39|14140828222.905321|7856558002614|REPRODUCED|PR|
|H6_DF19_P_1|0.05|4 / 0.4|48|10144814685/16|4054009.1059934883|356097424|REPRODUCED|PF|
|H6_DF19_P_2|0.0005|4 / 0.1|104|1709532575849751/52|65539320043.303085|36384720355183|REPRODUCED|PR|
|H8_DF28_P_1|0.05|8 / 0.4|48|9840910801/8|3175066.353534034|1267321444|POINT_REPRODUCED|PR|
|H8_DF28_P_2|0.0005|7 / 0.05|48|546493203592571/6|144211830223.71762|94193016652860|POINT_REPRODUCED|PR|

|ID|PR candidate|fixed PF control|q / r / K|
|---|---|---|---|
|H4_DF10_P_1|B2_p3_q1_2nd_r4_K2|B0I_p5_q1_2nd|1 / 4 / 2|
|H4_DF10_P_2|B2_p2_q8_2nd_r2_K4|B0_p7_q8_2nd|8 / 2 / 4|
|H6_DF19_P_1|B2_p4_q2_2nd_r4_K2|B0I_p7_q1_2nd|2 / 4 / 2|
|H6_DF19_P_2|B2_p4_q8_2nd_r4_K2|B0I_p11_q4_4th|8 / 4 / 2|
|H8_DF28_P_1|B2_p8_q2_2nd_r4_K4|B0I_p11_q1_2nd|2 / 4 / 4|
|H8_DF28_P_2|B2_p7_q16_2nd_r4_K4|B0I_p15_q4_4th|16 / 4 / 4|

H4_DF10 and the legacy H4 DF12 series in Figure 2A are distinct. H4 coarse precision is NOT_REPRODUCED and PF-favouring. H6 coarse precision is REPRODUCED because its fixed expected direction was PF-favouring. H8 POINT_REPRODUCED refers to a saved PR-favouring finite-sample point result. H8 rows are the same two n48 observations as Figure 2B, not extra independent evidence. No exploratory n8, saved FP n32, or old confirmation sample is pooled here.

Exact tokens are retained without new decimal conversion, ratios, CI, or statistical analysis. Unobserved rare-event cost/variance is unknown, not zero; no population-mean ordering is certified. Detailed H8 independent-PR completion receipts are not publicly re-verified. Candidate/source/cohort/CEM/ledger mappings are given in main_table.csv and evidence_linkage.csv.
