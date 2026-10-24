# H4 central-control R0経験的診断（2026-10-24）

PF/PR index0のcosine2診断、初回5状態作用の成功5／総試行5を保存。
両者のdは引用した保存ρ予算より小さい経験値。`EMPIRICAL_ONLY_NO_RHO_UPPER`、局所上界null、正式certificate未受理。
今回native exit0/reap済み、mandatory STOP。

- [報告書](../../../docs/non_df_h4_central_r0_empirical_20261024.md)
- [SOURCE・runtime/input seal](seal.json)、[承認](approval.txt)、[GitHub照合](github_verification.json)
- [実行契約](active_contract.json)、[新one-shot grant](run01/grant.json)、[消費](run01/CONSUMED.json)、[call ledger](run01/call_ledger.json)
- [実QPY構造照合](run01/preflight.json)、[raw norm](run01/state_identity.json)
- [PF保存C-target](run01/PF_C_target_saved.json)、[PR0選択C-target](run01/PR0_C_target.json)
- [2対象の信号・残差・予算比較](run01/results.json)、[転記CSV](results.csv)
- [結果集計](run01/result_summary.json)、[今回native exit](run01/native/worker_native_exit.json)、[STOP](run01/STOP.json)
- [保存検証](verification.json)、[読取・scalar恒等式専用verifier](verify_saved.py)、[出力SHA](output_inventory.json)、[完了監査](completion_audit.json)
- [SOURCE前metadata修正](pre_SOURCE_recovery.json)、[公開状態](publication_status.json)

PF_input/output、PR0_input/output/C_targetの各JSONは、call予約claim・UTC・伝播後振幅・signalを保持する。
`run01/`のgrant・ledger・native・STOPを変更せず、追加実行しない。
verifierは新しい状態作用・QPY decode・vectorからの信号再解析を行わない。
Arb・R1/R2・sine・新compile・sampling・N/G再解析は今回未承認。
次段階は報告書の候補A〜Dをユーザーが選択・明示承認した後に別SOURCE/grantで行う。
