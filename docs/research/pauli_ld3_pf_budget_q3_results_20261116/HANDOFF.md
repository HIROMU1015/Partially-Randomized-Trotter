# C・統合担当へのHANDOFF：LD3 PF予算の有限配分

FULL_PF__B_PRICEが適合。条件付き参照費用は541936295.9549784 RZ（exact 625936421828/1155）。今回生成した4列内の最安適合点。 p7基準との差T−395,962,678は145973617.9549784 RZ、Full PF基準との差T−559,457,386は-17521090.04502165 RZ。

固定4列の表・X/Y・exact予算残額・MSE上下界は[REPORT](REPORT.md)と[SUMMARY](SUMMARY.json)。新規Q3 4件、終了理由`FIRST_FEASIBLE_IN_COST_ORDER`、安い保留列[]。未評価列は[]。

## 機械可読引渡し

- `C_COMPARISON_JOIN.json::records`：各configuration（元dd5b881d…/total6757998a…）・binding・allocation/source・全26軸N/C_ref/積・元/total candidate/law・Q3結果SOURCE・exact総額。各recordの`result_source`は公開snapshot `8afe9d59dcba9fc639c1968c1794d694ad24e3e0`内の原APIへ解決可能。未評価はnullと理由を保持する。
- `INPUTS.json`：科学実行SOURCE `a844ac25f88c5cdf39e9e1837d85b6d4341774ff`で全候補・評価順・価格参照・具体配分・exact費用を実行前固定。
- `run01/SEARCH_HISTORY.json`：実際の順序と判定・再利用/新規・停止。`run01/*/*/API_RETURN.json`、partials、RESOURCE_FINAL、PERSISTED、SUPERVISION、PROCESS_EXITSを保全。
- 固定価格入力：f25a05f8a92aab43cfbe30fc80b974054156aac1 / docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json。

価格は全26軸がモデル参照値。`E_C/CI/G/G_qualified/sufficient_N`はnull、価格qualified_for_candidate=false。Cの正式G guardを緩めず、条件付き参照費用とriskを別の資格で受け入れること。価格cell receiptをround receiptへ付け替えない。

## ロードマップへの差分

LD3に、現在PF達成費用を予算としたA_SHAPE/B_PRICEの有限4列のQ3・条件付き費用証拠が追加された。結論は今回の生成集合・採択Q3モデル・A参照価格の範囲。新結果が適合でも正式十分N、物理的精度保証、E[C]、G、方式全体のwinnerは認定しない。旧VPやLD2の既評価配分を再取得する必要はない。価格domainの追加検証や別配分・別候補探索が必要なら別指示。ロードマップ本体は統合担当が更新する。

新signal/sampling/MC/compile/fit/retry各0件。mandatory STOP。
