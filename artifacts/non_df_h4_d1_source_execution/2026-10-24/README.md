# H4 D1：採用SOURCE基準・9/9計測完了

[固定条件・全表・研究上の限界・終了監査](../../../docs/non_df_h4_d1_source_execution_20261024.md)。
SOURCE **9843b803878db6c3aee7ab5e7c6bb4818572c944**。
保存full4の旧RZ価格が全一致し、retained2/event2/empty1も完了した。
RZ非加法性はPauli+18、DF+3765。旧Pauli-PR hashFAIL/原因UNKNOWNは保持し、旧program identityを主張しない。

- [結果summary](result_summary.json)・[9構成CSV](metrics_table.csv)・[相互作用CSV](interaction_table.csv)
- [検証](verification.json)・[旧740file保全map](preservation_before.json)
- [承認原文・文脈](approval.json)・[seal](seal.json)・[active contract](active_contract.json)
- [Stage A](run01/stage_A.json)・[Stage B](run01/stage_B.json)・[Stage C](run01/stage_C.json)
- [grant](run01/grant.json)・[CONSUMED](run01/CONSUMED.json)・[ledger](run01/call_ledger.json)
- [9 input/output QPYとmetrics](run01/variants)・[run177file SHA](run01/output_inventory.json)
- [STOP](run01/STOP.json)・[engine native receipt](run01/native/worker_native_exit.json)・[task native receipts](run01/native_tasks)
- [新人工10件](artificial/result.json)＋[既存17件](artificial/reused_17.json)・[author技術確認](author_check.json)
- [commit/push/remote](publication_status.json)

消費：full QPY確認4、full再生成0、隔離/empty構築5、費用compile9。
engineと18taskは全exit0/native回収。旧740file・code42/input71・runtime6999を保全した。
旧native UNKNOWN、C/L BLOCKED、compiler certificate欠測、旧N/G nullは不変。
人工検査は主agent自身の確認であり、独立レビュー/実H4形式certificateではない。

この9回は固定4代表の費用構造診断。PR256本平均・母平均・一意の因果寄与率・全RPE総費用には外挿しない。
正式certificate受理、sampling、分子再生成、state/signal、N/G、D2/D3は0。
新grantも消費済みで再利用しない。結果と終了監査をcommit/push後 **mandatory STOP**。
次はユーザーがD2の要否を判断し、Codexは別指示後だけ準備する。
