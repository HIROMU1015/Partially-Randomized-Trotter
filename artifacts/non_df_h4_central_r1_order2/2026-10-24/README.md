# H4 Pauli central R1 order2 cosine（2026-10-24）

PR index1の1論理診断・初回3状態作用、成功3／総試行3。今回native exit0/reap、mandatory STOP。
経験的d=4.3771358202651746e-16、保存cosineρ予算2.593611368819805e-6。
`EMPIRICAL_ONLY_NO_RHO_UPPER`、rho_upper/numerical_error_upper=null、LOCAL_BUDGET_MET=false。

- [報告書](../../../docs/non_df_h4_central_r1_order2_20261024.md)
- [承認原文](approval.txt)、[GitHub来歴](github_verification.json)、[固定入力](inputs.json)、[旧証拠保全](preservation_before.json)
- [科学SOURCE/input/runtime seal](seal.json)、[実行契約](active_contract.json)、[schema](output_schema.json)
- [新one-shot grant](run01/grant.json)、[消費](run01/CONSUMED.json)、[call ledger](run01/call_ledger.json)
- [実QPY/order2 SOURCE照合](run01/preflight.json)、[再利用raw norm](run01/state_identity.json)
- [input作用](run01/PR1_input.json)、[compiled作用](run01/PR1_output.json)、[個別C-target作用](run01/PR1_C_target.json)
- [三信号とδ/η/d](run01/PR1_result.json)、[R0保存値との限定比較](run01/R0_R1_comparison.json)、[転記CSV](results.csv)
- [実行集計](run01/result_summary.json)、[native exit](run01/native/worker_native_exit.json)、[STOP](run01/STOP.json)
- [保存検証](verification.json)、[保存値専用verifier](verify_saved.py)、[run出力SHA](output_inventory.json)、[完了監査](completion_audit.json)
- [公開状態](publication_status.json)

各状態作用receiptはcall予約claim・UTC・伝播後振幅・signalを保存する。
SOURCE/旧raw・R0結果・grant・ledger・marker・STOPを変更しない。
後検証は新状態作用・QPY decode・保存vector再縮約を行わない。
PF/PR0再計算、sine、Arb、R2、新compile、sampling、N/G再解析、正式certificate受理は今回0。
次の候補はユーザー明示承認後の別SOURCE/未使用grantでのみ実施する。今回runを再開しない。
