# H6 PR pilot：cutoff修正と軽量高速化

修正SOURCE：`e7595e8ff543a0d0f5e1ea91dec14283b61f9c32`。資料は技術修正・人工検証・保存metadata/byte監査であり、実H6の再取得結果ではない。

- [報告書](../../../../docs/research/track_a_h6_pr_pilot_cutoff_repair_20261020.md)
- [SOURCE closure](repair_source_closure.json)
- [保存結果監査](saved_metadata_audit.json)：74 pair/444 archive/4404片。旧STOPは不変。
- [人工検査・回帰](tests.json)、[人工性能測定](artificial_benchmark.json)
- [無効draftの実CLI拒否](draft_rejection.json)
- [保全](preservation.json)、[構造照合](structure_validation.json)
- [再開提案DRAFT](continuation_proposal_DRAFT.json)：74本の受入と元固定322登録/最大644軸の取得は未承認。継続専用adapterも未完成。

登録seed一覧・private原本・QPY/NPZ・認証情報は含めない。既存stageのexit1/STOPを保持し、統計・n_E・shot・会計・AD03へ進まない。新有効grant0、mandatory STOP、local commitのみ。
