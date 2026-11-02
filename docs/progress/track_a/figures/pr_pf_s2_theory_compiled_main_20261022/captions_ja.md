# 図caption

全図は現行探索A n8のS2 PRと、登録Full/B0/B0+Iの最良適格S2 PFの保存compiled総RZ費用比。
Theory値は未取得なのでcompiled系列だけを表示し、理論／compiled比較が完成した主図とは扱わない。
横軸H2/H4/H6/H8、破線1未満はPR低費用、1超はPF低費用。線はguide、数字は表示丸め。
linear 1 Å/STO-3G/tol-only DF3/10/19/28、T=.8、P=0、signed、allowance1、cos/sin両軸、
symmetric-directional、元scalar+tail identity1回、保存Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし。
H6/H8のALL_SAVED458価格政策のprofile/runtime未証明等の条件は保持する。
比の母平均保証や独立確認の結論を意味せず、仮説A/Bやcompiler単独の因果効果を判定しない。


## ε=0.05

[PNG](A_compiled_only_eps_0p05.png) / [SVG](A_compiled_only_eps_0p05.svg) / [PDF](A_compiled_only_eps_0p05.pdf)

PF採用方式（H2/H4/H6/H8順）：full PF B1_p3_q1_2nd / B0+I B0I_p5_q1_2nd / B0+I B0I_p7_q1_2nd / B0+I B0I_p11_q1_2nd。

PR採用候補：B2_p1_q1_2nd_r2_K4 / B2_p3_q1_2nd_r4_K2 / B2_p4_q2_2nd_r4_K2 / B2_p8_q2_2nd_r4_K4。


## ε=0.01

[PNG](A_compiled_only_eps_0p01.png) / [SVG](A_compiled_only_eps_0p01.svg) / [PDF](A_compiled_only_eps_0p01.pdf)

PF採用方式（H2/H4/H6/H8順）：full PF B1_p3_q1_2nd / B0 B0_p7_q2_2nd / B0 B0_p11_q2_2nd / B0 B0_p15_q4_2nd。

PR採用候補：B2_p1_q1_2nd_r2_K4 / B2_p3_q2_2nd_r4_K2 / B2_p4_q4_2nd_r4_K2 / B2_p8_q4_2nd_r4_K2。


## ε=0.005

[PNG](A_compiled_only_eps_0p005.png) / [SVG](A_compiled_only_eps_0p005.svg) / [PDF](A_compiled_only_eps_0p005.pdf)

PF採用方式（H2/H4/H6/H8順）：full PF B1_p3_q2_2nd / B0 B0_p7_q4_2nd / B0 B0_p11_q4_2nd / B0 B0_p15_q4_2nd。

PR採用候補：B2_p1_q1_2nd_r2_K4 / B2_p2_q4_2nd_r4_K2 / B2_p4_q4_2nd_r4_K2 / B2_p8_q4_2nd_r4_K2。


## ε=0.001

[PNG](A_compiled_only_eps_0p001.png) / [SVG](A_compiled_only_eps_0p001.svg) / [PDF](A_compiled_only_eps_0p001.pdf)

PF採用方式（H2/H4/H6/H8順）：full PF B1_p3_q4_2nd / B0 B0_p7_q8_2nd / B0 B0_p11_q8_2nd / B0 B0_p15_q8_2nd。

PR採用候補：B2_p1_q4_2nd_r2_K4 / B2_p2_q8_2nd_r2_K4 / B2_p4_q8_2nd_r4_K2 / B2_p6_q8_2nd_r4_K4。


## ε=0.0005

[PNG](A_compiled_only_eps_0p0005.png) / [SVG](A_compiled_only_eps_0p0005.svg) / [PDF](A_compiled_only_eps_0p0005.pdf)

PF採用方式（H2/H4/H6/H8順）：full PF B1_p3_q4_2nd / B0 B0_p7_q8_2nd / B0 B0_p11_q8_2nd / B0 B0_p15_q8_2nd。

PR採用候補：B2_p1_q4_2nd_r2_K4 / B2_p2_q8_2nd_r2_K4 / B2_p4_q8_2nd_r4_K2 / B2_p7_q16_2nd_r4_K4。
