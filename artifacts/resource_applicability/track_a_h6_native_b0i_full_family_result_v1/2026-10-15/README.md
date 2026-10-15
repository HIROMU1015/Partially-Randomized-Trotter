# H6 native B0+I 全12構成：結果入口

[結果の説明・条件・限界](../../../../docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)。
新規12軸＋原SOURCEで旧6軸再利用、全条件不適格の6軸は未測定。P=0の登録集合内点費用比較であり、母平均winnerやenergy/QPE/RPE最終費用の保証ではない。

- [summary・SOURCE・scope・旧D0固定参照](result_summary_v1.json)
- [18取得済み軸のRZ/CX/depth等](axis_compiled_metrics_v1.csv)
- [全120 precision行](precision_coverage_v1.csv)、[詳細shot/headroom/費用](b0i_accounting_v1.json)
- [主RZ比較10行](primary_RZ_P0_v1.csv)、[全6指標60比較](comparison_P0_v1.csv)
- [全family minimaと仮想Pの60lower envelope](comparison_and_P_sensitivity_v1.json)
- [新6構成のnative正しさ](new_case_correctness_v1.json)
- [native exit/cleanup/資源/元証拠SHA](completion_audit_v1.json)
- [保存表の限定再照合script](verify_public_tables_v1.py)

repository rootでPython -P -Bからverify_public_tables_v1.pyを実行できる。固定D0 commitのGit objectが必要で、ネットワーク取得・科学import・NPZ読込・新compileは行わない。
公開用scalar/CSV/projectionのみを収録し、実runtime/ledger/checkpoint/NPZ/venv/credentialは含めない。原証拠はprivate保存先に保全しbytes/SHAで対応する。
新計算・新grant・共有環境/他job変更・原稿改訂・subagentsは今回0。次stageなし、mandatory STOP。
