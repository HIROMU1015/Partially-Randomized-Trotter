# H4独立確認Pの論文統合

固定結果`8a18b277663bf2f8998f79e9b8c0a9e61d2b184c`、科学SOURCE`73ec1964501f14390d037a378c3149110e273635`の公開結果・監査・実行契約とSOURCE6 fileを読み取り照合した。
H4 linear/隣接1 Å/STO-3G/**legacy DF12**/T=.8/P=0、通常signed/factor1、固定9候補の独立確認。
別のtol-only DF10系列、H6独立確認、共通プロトコルへ接続していない。

保存結果は4,185軌道／8,370 compile完了、事前登録した主比較10/10・境界診断4/4で点費用方向を再現。
S4は全候補で未観測。固定n設計や点順位再現は母平均優位・多重比較補正済み有意差・厳密bias保証ではない。
H4 Dは登録104候補、保存適格な71候補の費用あり（新55＋保存16）、残33は保存不適格。
最良PF代表10/10不変のD証拠を継承する。

原科学resultの記録SHAは`4f79637089d2c61f5e85edbc887cfafd28d8df0624df3b04c9ecad58f6504043`。
Gitのconfirmation_result.json公開projectionのbyte SHAは`4645cdc6ff2c8fdd656f85664bd0155d9240c62bf6287c636647d9bf4daa25a5`。
後者は固定artifact_byte_manifestと一致し、その内部の原科学result SHAはcompletion_auditと一致する。
両SHAは役割が異なる。原private result全byte・全runtimeを今回独立再監査したとは扱わない。

[H4 P別表14行](data/H4_P_confirmation_separate.csv)へ保存exact G・保存paired SE・n・PF・statusを転記した。
mean/SE/共分散・順位を再計算していない。同一候補の複数εは同じ固定標本を使うため、14比較を独立14標本と数えない。
新Pの値を旧探索n8／fixed-anchor n32へ混ぜていない。

主表32行の元列は全て保持し、H4の10行へH4_latest_provenance、P commit、対応する保存endpointの索引・statusを追加した。
元H4_new_provenance=D_COMPLETE_P_PENDINGとH4_stage=H4_PRE_D_P_VALIDATIONは当時の履歴のまま残す。
新provenanceはD_AND_P_COMPLETE_LEGACY_DF12。Figure 2のH4数値は引き続き元n8/n32のcoreを使用する。
P標本を使う新費用比・単価平均・機構分解は未計算であり、必要なら別の解析契約と承認を要する。
