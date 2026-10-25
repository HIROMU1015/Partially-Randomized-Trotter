# H4 Pauli-PRパラメータ探索：固定H4/1.00Å/STO-3G

[結果報告](../../../docs/non_df_h4_pauli_pr_search_20261025.md)。141候補、仮想ρ=0・有限cohort価格の条件付き比較。科学実行完了・mandatory STOP。

- [主結果5精度CSV](best_by_precision.csv)、[exact結果JSON](result_summary.json)
- [725候補/精度/cohort行](candidate_precision_table.csv)、[candidate catalog](candidate_catalog.json)
- [全2,320価格標本](trajectory_prices.csv)：cosine/sineは同じtrajectoryでpaired
- [事前凍結した確認候補](confirmation_nominations.json)、[確認停止判定](confirmation_stop_decision.json)
- [固定active科学範囲](active_scope.json)、[入力参照](input_references.json)、[最終解析入力SHA](final_analysis_inputs.json)
- [native/保存照合](verification.json)、[mandatory STOP](STUDY_STOP.json)
- [raw/段階別解析archiveとmember SHA](data_archives.json)

batch archiveは各stageのplan、SOURCE、law/targetまたはevent/価格、native/STOP、代表QPYを元bytesのまま含む。analysis archiveには探索roundごとの全結果、入力SHA、最小候補・提案を含む。復元は空の別directoryで行い、既存SOURCE・run・STOPへ上書きしない。QPY gzipは各候補/cohort index0のinput/outputを保存。全indexのcompiled bytes SHAはtrajectory.jsonに残す。

共通policy固定p27再会計と歴史的PR256/旧N/Gは別系列。探索と確認はpoolしない。主表は探索nomineeの独立確認で、確認標本内の事後最小は別panel。全lawρ、実N/G、母平均tail、一意な最適K、一般化は未保証。正式certificate受理なし。
