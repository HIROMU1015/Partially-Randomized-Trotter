# Track A H6技術pilot：一回実行認可・seal v1

2026-10-10 JST。ユーザー「次の作業に進んで」を、直前に説明した[固定H6 pilot](track_a_h6_technical_pilot_preparation_v1.md)一回の実行指示として記録した。
[独立科学レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md) §15に従い、完了または失敗・欠測を保存/監査/公開/remote照合した後GPTへ戻す。
H6本検証・H8・科学GO/STOP・表現政策変更・入力再生成・再試行は認可しない。

source `a643220cde1e24b7e3d637f4bc5d1b0342ce2e86`、準備公開 `16bba9dfc40ffb12b3614120523ea7b318c9b735`、manifest digest `f05a6c26ae2d1a19a15808a0dd45b416848cadf872f84b9396c526b77bb8aba5`を変更しない。
[seal](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/sealed_preparation_v1.json)、[source freeze](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/source_freeze_v1.json)、[入力identity](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/input_identity_v1.json)。
[専用grant](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/authorization_v1.json) SHA `64d3322bb6e383e411189580229da6b12a07dd75ba86377d58b740aff3d24228`、[launch対象](../../artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v1/2026-10-10/launch_target_v1.json)。
元template/manifestのscience_authorized=false・launch_allowed=falseは、それ自体が認可ではないことを示す固定flagであり、今回の別grantがこの一回だけを認可する。
旧入力完成grantは消費済みで流用しない。新output `artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1`、retry/resumeなし。

linear H6/1.00 Å/STO-3G、12 modes、α3/β3、sector400、全19 signed fragments生成順、tol-only1e-8/cutoff0。
新weighted policyを維持し、T0.8、prefix10/full19/0、q1/q2、partial R4/r2/K2またはK6。
7 correctness cells通過後だけ4 cost trajectories・8 occurrences・36 ordinary/directional×cosine/sine measured wrappersを評価する。
actual prepared representation/instruction boundsを先に保存/検査、native全primitive作用後sector leakage、independent occupation/forward recurrence/sector expm-eighを照合。
B0 discard/PFとB2/B3 outer-PF/finite-RTEのsigned complex差を分離保存。N/Gnull、u/ground-state未認定、UNDETERMINED。
CPU IDs [0,2,5,6]、worker1/Numba4/OMP4/BLAS1/chunk1。phase1800/1800/3600秒、total7200秒（上限、ETAでない）、AS8GiB/output512MiB/log64KiB。
新SCF/DF再分解/state solver0。対象条件、seed、gate、caps、sourceを救済変更しない。旧source/freezes/results/STOP・dirty/untracked・Track Bを保全。

この段階は実行前固定でH6 pilotの科学結果はまだない。GitHubからsource/manifest/input/grant exact bytesを再取得して照合した後、一回だけrunnerを起動する。
完了またはSTOP/欠測後、原statusと最後の保存boundary countersを保持し、保存bytes監査と結果inventoryを公開する。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATIONは後続科学段階への未認可状態として維持し、mandatory STOP/next_stage_authorized=false。
