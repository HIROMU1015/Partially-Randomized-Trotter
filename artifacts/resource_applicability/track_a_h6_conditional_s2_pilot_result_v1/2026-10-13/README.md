# H6 s2限定pilot完了：結果入口

**8 trajectory・16 full-wrapper compile完了、coordinator exit0、全worker cleanup/reap・残存0、mandatory STOP。** SOURCE843265b624eac8fbb9acea47ef21cab1207e1880、B2_p5_q8_2nd_r2_K2のwhole-trajectory s2。

- [結果・固定科学条件・解釈・残る限界](../../../../docs/research/track_a_h6_conditional_s2_pilot_result_v1.md)
- [s2条件付き平均・4精度の対象q8比較](cost_contribution_analysis_v1.json)、[16 wrapper CSV](pilot_wrapper_metrics_v1.csv)、[8新seed/order/両軸metrics/原bytes来歴](pilot_metadata_evidence_v1.json)
- [metadata audit・carry/source/input/grant](result_audit_v1.json)、[native terminal](execution_terminal.json)、[native exit0](native_coordinator_exit_v1.json)、[全8退出/監視/cleanup](pilot_resource_summary.json)、[post-exit残存0・容量](post_exit_cleanup_storage_v1.json)
- [累積消費と旧予約](aggregate_calls.json)、[8/16 completion receipt](pilot_complete_receipt.json)、[元driverの条件付き統計](conditional_resource_statistics.json)
- [保存metadata集計script](analyze_pilot.py)、[公開CSV/JSONだけの別算術](verify_public_result.py)、[40検査結果](public_arithmetic_verification_v1.json)
- [新grant/直前gate/初回prelaunch拒否証跡](../../track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/launch_authorization_v1/README.md)

s2平均RZ96,285.25（SE324.0004）vs既存s0確認32の95,863.78125。保存pi2=1.66969%を使う層別平均の増加は0.00734086%。ε=.001の対象q8/K2 PR/PF比0.3870055113→0.3870339209、点推定RZ削減61.2966079%。他3精度の旧点winner・事前K4 winnerは変更しない。

全8本はM=1だった。8標本では複数order2/希少component列を覆えず母平均上界/正式winnerは未認定。探索/確認/pilotをpoolせず、旧compiler bit同一性未検証も維持。

native wall698.894秒、aggregate peak RSS0.956207GiB、runtime logical7.91MB。known compile1716/trajectory842＋旧予約20/10＝1736/852、reset/返却0・unknown actual=null。runtime grantは消費済み。追加science/H8/retry/resume/次stageなし。H4/共有/他job/venv/GPU操作0。旧592探索/256確認・科学224/runtime251/profile/input/sourceを保全。


[結果commitと28 fileのremote byte取得確認](publication_receipt_v1.json)。公開receiptは結果を変更しない別commitで固定する。
