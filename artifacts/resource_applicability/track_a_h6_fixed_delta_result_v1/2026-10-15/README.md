# H6 fixed-delta T-dependence: completed result bundle

[結果・条件・限界・終了監査](../../../../docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)。
実行SOURCE `669842c85e2c04a90dcdcfb774b323a2737cb8ba`。run02は144/144 compile・64新規PR trajectory完了、約52分、native exit0。T=.8独立32は旧結果の再利用。

- [主表: PR / best eligible deterministic, P=0/RZ](primary_RZ_P0_v1.csv)
- [図: T×精度とbias/B/回路費用](T_precision_cost_and_causes_v1.svg)
- [全6費用指標・allowance factor1/10](all_metric_comparisons_v1.csv)
- [signal/bias分解/B/数値allowance](signal_summary_v1.json)
- [軸別shot・paired費用平均/SE](accounting_v1.json) / [全比較JSON](comparisons_v1.json)
- [回路6metrics平均](per_circuit_metrics_v1.csv) / [全216軸の出所・event/compiler/fingerprint](wrapper_projection_v1.json)
- [希少次数coverage](rare_orders_v1.json)
- [終了/会計/監視監査](completion_audit_v1.json) / [新144軸のwall/RSS](execution_resources_v1.csv)
- [証拠fileのbytes/SHA](input_receipts_v1.json) / [source/profile/carry出所](result_provenance_v1.json)
- [7人工算術検査](unit_tests_v1.txt) / [公開のみで900 paired metric再計算](public_arithmetic_verification_v1.json)

固定5構成の有限標本点比較。未観測Taylor費用・母平均上界は未保証、empirical allowance、旧host compiler outputのbit同一性は未確認。実runtime/cache/NPZは含まない。
集計sourceはscripts/summarize_h6_fixed_delta_v1.py、図はscripts/plot_h6_fixed_delta_result_v1.py。新科学計算0、mandatory STOP。
