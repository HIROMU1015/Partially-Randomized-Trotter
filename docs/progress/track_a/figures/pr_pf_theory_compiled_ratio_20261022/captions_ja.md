# 図caption

## 参考A：理論／compiledのPR/PF総RZ費用比

[PNG](reference_A_H4_legacy_DF12_eps_0p05.png) / [SVG](reference_A_H4_legacy_DF12_eps_0p05.svg) / [PDF](reference_A_H4_legacy_DF12_eps_0p05.pdf)

直鎖H4・隣接1 Å・STO-3G・legacy固定DF12・T=.8・P=0・ε=.05・通常signed bias・数値allowance倍率1。
左右はそれぞれ固定PR p3/q1/δ=.8/r4/K2/S2とp3/q8/δ=.1/r2/K4/S2。
両panelの対照は同じFull PF p12/q1/δ=.8/S2で、B0/B0+Iを含む最良PFを表す図ではない。
紫の白抜き菱形はNATIVE_MIXED native分解の理論期待総費用比、青緑の丸は保存compiled総費用比。
両値とも候補を固定し、各候補の理論側とcompiled側で同じ保存cos/sin shot数を継承したGを使う。
理論lower/upperは保持し、この2組ではexact値が一致する。中点や新しいCIは作らない。
破線1未満はPR低費用、1超はPF低費用。PRは旧anchor n32、PFは固定回路n1。
symmetric-directional control、状態準備なしfull Hadamard wrapper、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし。
理論法則期待値対compiled有限標本平均の条件付き比較であり、同一event列比較、compiler単独の因果効果、母平均順位の認定ではない。
出典は6e2b7eb6のstage3保存総費用・preserved_shots・pf_pr_comparisons。
現行H4 DF10・新探索Aとは別の参考図。

## 参考B：理論→compiled総RZ費用の削減率

[PNG](reference_B_H4_legacy_DF12_eps_0p05.png) / [SVG](reference_B_H4_legacy_DF12_eps_0p05.svg) / [PDF](reference_B_H4_legacy_DF12_eps_0p05.pdf)

条件・候補対・shot・cohort・理論モデル・compilerは参考Aと同じ。
PR S2と固定Full PF S2について100×(1−保存compiled G / 保存theory G)を表示する。
0線が一致、正はcompiledが低費用、負はcompiledが高費用。この保存2組では両方式とも正。
図中の53.9%のPF棒は、同じPF固定1候補を両panelで再表示したもので、独立な2件ではない。
保存PRは42.5%／46.5%、PFは53.9%で、削減率はPFの方が大きい。
現行最良PFを選んだ比較でも、新しい科学的順位判定でもない。
