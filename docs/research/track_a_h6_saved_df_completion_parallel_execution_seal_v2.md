# Track A H6：保存DF受理・state/snapshot 並列一回実行seal v2

2026-10-10 JST。ユーザー指示「並列化機能を使ったうえで計算を開始して」を保存DF入力完成一回へ結合した。
[独立レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md)の工学政策と[旧v1準備](track_a_h6_saved_df_completion_preparation_v1.md)を維持する。
この段階は実行前固定であり、H6の受理/stateの科学結果はまだない。H6 pilotは未認可。

## 実行sourceと資源

source commit `f37005f01b2be38c5993d6e82df91abe9c643d21`。既存のNumba並列matrix-freeを4論理CPUで使う。
CPU IDs `[0, 2, 5, 6]`（観測時に異なる物理core 4個）、worker1、Numba4/OMP4、BLAS1、chunk1、GPUなし、host専有保証なし。
旧資料のCPU2はCPU ID2への一論理CPU affinityを表しており、2コア割当ではなかった。
plan変更はnum_threads 1→4と新schema/CPU list/namespace/実行receipt。科学的Hamiltonian・rank・budget・solver精度を変更しない。
親/旧sourceは書き換えず、新v2 contract/port/watchdog/runner/byte auditorとtestを追加した。

入口：[runner](../../scripts/resource_applicability/run_track_a_h6_saved_completion_v2.py)、[contract](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_contract_v2.py)、[port](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_port_v2.py)、[watchdog](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_watchdog_v2.py)、[auditor](../../scripts/resource_applicability/audit_track_a_h6_saved_completion_v2.py)、[test](../../tests/tracks/resource_applicability/test_ax2b_h6_saved_completion_v2.py)。
[source freeze](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/source_freeze_v2.json)：science 198、validation4。
[environment/CPU観測](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/environment_resource_observation_v2.json)、[input identity](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/input_identity_v2.json)。
[synthetic 92 passed](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/synthetic_test_audit_v2.json)、[JUnit](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/synthetic_tests_v2.xml)：新51/旧41。
toy 8 modes・sector36の3 vectorで1/4 thread matvec exact bytes一致。H6性能・誤差認証ではない。

## 固定入力・条件・認可

linear H6/1.00 Å/STO-3G、12 modes、Nalpha=Nbeta=3、sector400、全19 fragments・signed lambda/order保持。
保存integrals SHA `edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d`、診断raw SHA `2943295c2131ce9896fcad83e3dee78763efce492389ce49a6a66b87b57beebb`。
親診断source `ff24de4bc410234472a416186b773fc7875ae373`、結果 `8a3189e69dd461724fa9e2c01ea08562c1c35f8d`。
tol-only1e-8/cutoff0、final_rank/rank fallback/fragment削除なし。新分解を行わず保存rawを読む。
[政策](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/fixed_policy_v1_unchanged.json)はv1 exact bytes保持。projection追加予算1e-10 Ha、decision9.9e-11 Ha、summary/独立係数照合を維持。
PASS_ENGINEERING、representation_error_certified=false。provider truncation/DF残差/projection/PF-RTE/roundoff/measurementを別記録。

[sealed manifest](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/sealed_preparation_v2.json) digest `8f0abaca8acfee4a3bc61f7e85426615c85e45f263df4b0f00f19d2de159e2c9`。
[新grant](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/authorization_v2.json) SHA `88f40be5643d0f2ea09ecc0398193ffefa2f6197e1f71f85daefd00124d84153`、[launch target](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/launch_target_v2.json)。
manifest自体のscience_authorized/launch_allowed=falseは「manifest単独では認可にならない」ための固定flagであり、別grantでこの一回だけを認可する。
phase60/120/900秒、total1080秒（18分上限、ETAではない）、AS8GiB、output32MiB、log64KiB、matvec10000（residual/rmatvec込み）。
受理1回・solver1回、SCF/integral生成/DF再分解/signal/trajectory/compile0。retry/resume禁止。
output `artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1`。旧v1未実行outputや旧失敗runを再利用しない。

実行後にraw・projection/DF/state/snapshot/resource receipt・progress/terminal・stdlib保存bytes監査・source/認可来歴を公開しremote照合してmandatory STOP。
失敗・欠測を保存し、閾値/rank/sourceの救済変更や再試行を行わない。
H6 pilot（7 cell/36 wrapper）、本検証、H8や科学GO/STOPは認可しない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINEDを維持する。
