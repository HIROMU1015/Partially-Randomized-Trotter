# H4 R1 sine経験的診断（2026-10-24）

PF・PR0・PR1 sine3論理、初回6状態作用、成功6／総試行6。
今回native exit0/reap・mandatory STOP。新C-target・cosine再評価・Arb・compile・sampling0。
全dは保存sine予算より小さい経験値、rho_upper/numerical_error_upper=null、LOCAL_BUDGET_MET=false。

- [報告書](../../../docs/non_df_h4_central_r1_sine_20261024.md)
- [承認](approval.txt)、[GitHub照合](github_verification.json)、[固定入力](inputs.json)、[旧証拠保全](preservation_before.json)
- [SOURCE seal](seal.json)、[実行契約](active_contract.json)、[schema](output_schema.json)、[新grant](run01/grant.json)、[消費](run01/CONSUMED.json)、[ledger](run01/call_ledger.json)
- [実QPY・測定軸・cache照合](run01/preflight.json)、[同一raw norm](run01/state_identity.json)、[C-cache参照](cache_binding.json)
- [PF C-target再利用](run01/PF_C_target_reused.json)、[PR0](run01/PR0_C_target_reused.json)、[PR1](run01/PR1_C_target_reused.json)
- [三信号・残差・sine予算](run01/results.json)、[cosine保存引用との比較](run01/axes_comparison.json)、[6局所行転記CSV](results.csv)
- [実行集計](run01/result_summary.json)、[native exit](run01/native/worker_native_exit.json)、[STOP](run01/STOP.json)
- [保存検証](verification.json)、[保存専用verifier](verify_saved.py)、[run出力SHA](output_inventory.json)、[完了監査](completion_audit.json)
- [SOURCE前metadata修正](pre_SOURCE_recovery.json)、[公開状態](publication_status.json)

run01のPF/PR0/PR1 input/output JSONは、call claimと伝播後振幅・sector和・signalを保持する。
C_target_reusedは既存値の転記・PFのexact scalar区間変換であり新状態作用ではない。
保存検証は新状態作用・QPY decode・保存vectorからの再縮約を行わない。
元SOURCE/raw/結果/grant/ledger/marker/STOPを変更しない。Arb/R2/N_G等へ自動移行しない。
