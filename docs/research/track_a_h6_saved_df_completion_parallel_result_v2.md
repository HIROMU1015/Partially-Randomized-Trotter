# Track A H6：保存DF受理・state/snapshot 並列一回実行結果 v2

2026-10-10 JST。ユーザーの並列計算開始指示を[実行前seal](track_a_h6_saved_df_completion_parallel_execution_seal_v2.md)へ結合して一回実行した。
**H6_SAVED_DF_COMPLETION_COMPLETE / 保存bytes監査SAVED_COMPLETION_BYTES_PASS。保存DFの工学的受理とstate/snapshot保存まで完了。**
科学的優位性、total u、ground-state認証、H6 pilotのGOを意味しない。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。

## 条件・immutable identity

linear H6/1.00 Å/STO-3G、12 modes、Nalpha=Nbeta=3/sector400、actual DF rank19。
全signed lambda・generation order保持、tol-only1e-8/cutoff0、final_rank/rank fallback/fragment削除なし。
source `f37005f01b2be38c5993d6e82df91abe9c643d21`、seal/認可commit `f70416ec4f3a8fe1ab7ba4c02ccde7260a361d0e`。
[source freeze](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/source_freeze_v2.json)、[入力identity](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_preparation_v2/2026-10-10/input_identity_v2.json)、[認可](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/authorization_v2.json)。
親診断source `ff24de4bc410234472a416186b773fc7875ae373`、親結果 `8a3189e69dd461724fa9e2c01ea08562c1c35f8d`。
integrals SHA `edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d`、raw SHA `2943295c2131ce9896fcad83e3dee78763efce492389ce49a6a66b87b57beebb`。
manifest digest `8f0abaca8acfee4a3bc61f7e85426615c85e45f263df4b0f00f19d2de159e2c9`、grant SHA `88f40be5643d0f2ea09ecc0398193ffefa2f6197e1f71f85daefd00124d84153`。
旧失敗runの未保存rawとの同一性は主張せず、旧STOP/政策/sourceは保全した。

## 記録された結果

[DF receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/df_receipt.json)で構造・独立係数再構成・保存summary照合を通過し、PASS_ENGINEERING。
N12 projection三角和の工学評価は `9.90253426943740866153729179392732588696648965838539201917712E-13` Ha、N6は `2.49198493502476651902725100525045491606292158595859473192924E-13` Ha。
判定上限9.9e-11 Ha（追加予算1e-10 Ha）を維持した。これはbinary64 norm由来の評価で、representation_error_certified=false。
旧無重みgate違反index15–18は保存されたままで、新policyが旧STOPを成功へ書き換えたわけではない。
provider truncation `2.693610667847679e-09` とraw→integrals係数上界式の有限精度評価 `1.075523811863775e-08` は別量である。
後者は厳密u認証でも、provider truncationと同じ閾値のPASS/FAILでもない。PF/RTE・roundoff certificate・measurementは未評価。

[state receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/state_receipt.json)：採用targetのsolver energy `-3.2360662799087647` Ha、residual `6.1060388500553216e-15`、固定gate `1.3236066279908765e-09`。
solver1回、matvec41回（rmatvec/residual込み）。これは数値eigenpair残差検査で、ground_state_certified=false。
[snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz)、[snapshot receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/snapshot_receipt.json)を保存し、新policy-bound loaderのroundtripを通過。
snapshot SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。
H6_input_accepted=trueはこの新versionの工学的入力完成を表し、pilot/精度/資源比較の承認ではない。

## 古典計算資源と実行監査

[parallel resource receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/parallel_resource_receipt.json)：CPU IDs [0, 2, 5, 6]、Numba4/OMP4/BLAS1、worker1、chunk1。4論理CPU affinityを実観測。
parent wall `3.8784506930969656` 秒、worker peak RSS `411328512` bytes。
上限total1080秒/AS8GiB/output32MiB/matvec10000内で完了。serial H6対照runは実施しておらず、速度向上率は主張しない。
保存raw 33件・1584749 bytes。scratchのfont cache/Numba JIT cacheも保存監査対象として公開するが、科学結果/sourceの代用ではない。
量子回路build/transpile/compile・signal・trajectory・新SCF/integral生成・DF再分解は0。Numba JITは古典matrix-free backendの実行に伴うもの。
[terminal](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/terminal_status.json)、[実行監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_audit_v2.json)、[保存bytes監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/saved_run_audit_v2.json)、[証拠inventory](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json)、[結果要約](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/result_summary_v2.json)。
[実行前remote照合](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/prelaunch_remote_verification_v2.json)でsource198/親raw25/旧raw20/旧v1/保護対象4127パスを確認した。
[保全監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/preservation_audit_v2.json)。一回grant消費済み、retry/resumeなし。

## 未確定事項と停止点

representation_error_certified=false、ground_state_certified=false、numerical_allowance_certified=false、N/Gnull、accuracy_eligibility=UNDETERMINED。
H6の7 cell/36 wrapper pilot、H6本検証/H8、科学GO/STOPを今回認可・実行していない。
独立レビューの前提から政策/表現を変更せず、今後のpilot接続・実行は別の固定・認可対象。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP、next_stage_authorized=false。
