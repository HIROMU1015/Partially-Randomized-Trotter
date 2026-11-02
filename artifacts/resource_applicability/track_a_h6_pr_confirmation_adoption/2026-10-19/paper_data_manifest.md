# 論文manifest：H6独立確認の正式採用を反映した更新版

[CSV](paper_data_manifest.csv)は共通プロトコルv1固定commit `9036d8a87ad1c38e83b9ff654b412dde7041e9d3` の論文準備manifest 1,568行をbyte prefixとして保持し、H6の確認PR／固定PFを各2行追加した1,572行。旧manifest・旧主表転記は当時のsnapshotとして保持する。今後のH6固定P証拠の入力には本版を使用し、旧版を再連結しない。完成図表・最終freezeの採否は今回行わない。

schemaは元manifestと同じ62列。追加 `record_type=H6_FIXED_CONFIRMATION` のキーは `H6P_Q2_CONFIRMATION_PR`、`H6P_Q2_FIXED_PF`、`H6P_Q8_CONFIRMATION_PR`、`H6P_Q8_FIXED_PF`。原1,568行のH4/H8/旧H6値とstatusは変更していない。

新規行は結果commit `3cccfcb2051dacf29e91473e2e1152acef1574bd` のresult.jsonから原文字列を転記した。`source_record=/scopes/0` または `/scopes/1`、`source_field` はその中の相対JSON pointer。`source_file` は各行の `result_commit` におけるrepository相対pathで、現worktreeへの同名ファイル存在を意味しない。[catalog](evidence_catalog.json)の固定Git objectとSHAで解決する。新たな比・平均・SE・順位・shotは計算していない。

`science_SOURCE` は確認の費用会計SOURCE、`acquisition_SOURCE` はPRの新cohort取得か、保存済みPF単価の再利用かを区別する。PFをこのSOURCEで新compileしたという意味ではない。PFの保存費用は元manifestのB126/B174とも一致を確認する。boundaryとmetadata_sourceは元探索domainの注記を継承し、確認時に新しく探索・境界判定したとはしない。

PRの `saved_cost_status=REPRODUCED` は[採用status](completion_status.json)の期待方向一致を示す。PF行はその比較に使った固定controlであり、確率的なPF再確認標本ではない。PR_CHEAPERはstatusで別項目として管理する。ε=.05のPR不利も正式採用する。

α_total=.05／α_axis=.025は保存shot会計のmetadata。費用差のCI・有意水準・n設計のfamily αと混同しない。PRのnはtrajectory数でcos/sinを2nとして扱わない。paired SEは同一trajectoryの両軸総費用のMCSE。

`TWO_FIXED_CELLS_ONLY`は今回の固定比較範囲であり、全登録136候補の再探索・直接実測完了を意味しない。図表IDは後続統合への入力用途だけで、最終レイアウトを採用した意味ではない。母平均、希少support、経験的bias/allowanceの限定は保持する。
