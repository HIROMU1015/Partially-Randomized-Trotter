# 2026-11-13：Full PF H_effの承認済み限定実行

ユーザーが前段[有限案](../h4_full_pf_heff_preparation_20261113/SCIENCE_PLAN.md)を承認したため、SOURCE `3a26b9ba70f7c5022a7e07356fe1df8b620545e8`で保存H4 DF10/sector(2,2)/Full PF S2/Δ.1を一回実行した。[REPORT](../h4_full_pf_heff_results_20261113/REPORT.md)。

step1、Schur1、元full-H Hermitian対角化1、m0/m12の直接PFと全spectral signal各2件。principalのminimum shiftは+0.00013371206319456874 Ha、effective最低groupへの保存state weight0.999999924308577。元最低groupから最大trace overlapを持つeffective groupも0だった。二時点のsignal差は2.3276653796347474e−15／8.691709382164289e−12で全数値gate通過した。

個々のRPE biasや厳密上界を認定せず、alias/ground/numerical certificationはfalse、十分N/global RMSE/Gはnull。新reference・旧mean再解析・normalizer・MC/C/G・幅探索・retry0。元input3hashは実行後も一致。caller4人工test PASS、既存evaluator19testは前段の記録を維持した。

次はユーザーがBへの接続又は追加有限検証を指定する。raw由来riskへ同じPF δEを重ねて加算しない。成果commit/push・remote照合後mandatory STOP。
