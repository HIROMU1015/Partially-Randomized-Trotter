# Track A AX-2B：H6入力生成・一回実行のseal v1

2026-10-10 JST。ユーザーの継続指示を、直前に説明したH6入力生成一回の実行認可として記録する。
H6技術pilot・H6本検証・H8の実行認可ではない。結果後はmandatory STOP。

## 固定範囲と出典

[準備契約](track_a_ax2b_h6_input_preparation_v1.md)のsource・plan・予算・tol-only政策を変更しない。
[GPT独立レビュー](track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md) §§16,19の順序を維持。
linear H6 / 1.00 Å / STO-3G、12 spin orbitals、Nα=Nβ=3 / sector400。
DFへ`truncation_threshold=1e-8`のみを渡し、actual rank・切断値・補正・Hermitizationを保存する。
入力生成ではPF prefix・delta windowを使わず、finite-time signal・shot・資源winnerを評価しない。
HF初期occupationからbounded sector eigshの指定stateを保存する。小residualはground-state証明ではない。

source commit：`67312f3195aede26e8ba4f5727d89c236772f82e`。
準備公開HEAD：`58dfb6d730a91d60cb5ff7c748b6b4d36e9bcd17`。
[execution source freeze](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/execution_source_freeze_v1.json)は既存183 science / 3 validationのbytesを再照合。
元source・source freeze・H4結果・旧STOPのbytesは変更しない。

## 認可・古典資源・一回性

正確な今回指示を[input-generation専用grant](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/input_generation_authorization_v1.json)へ記録した。
[sealed manifest](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/sealed_input_generation_v1.json)とdigestで結合。
original grant SHA-256：`7535f34c321c5c70f5b5f4a11b103c97b8986e339c8b3a269368b87366203152`。
CPU2、worker1、BLAS1、GPUなし。割当は単一logical CPUであり、host専有を保証しない。
[resource観測](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/resource_observation_v1.json)はhostの時点観測で、実worker RSSや完走時間の測定ではない。
integrals900秒・DF300秒・state/snapshot900秒、総2100秒、AS8GiB、output128MiB、worker log64KiB。
integral1・DF1・共通solver matvec/rmatvec/residual10000、trajectory/occurrence/compile0。
失敗時にtol/rank/solver/capを変えて救済しない。retry/resumeなし。

manifestの`science_authorized=false` / `launch_allowed=false` / NOT_AUTHORIZEDラベルは既存gate仕様として保持する。
実行はこれらの内部ラベルを変更するのではなく、別schemaの今回grantでのみ許可される。
`H6_NOT_AUTHORIZED`はH6技術pilot以降を未認可とする。`DRAFT_NOT_AUTHORIZATION`も維持。

## 起動と確認先

[metadata preflight](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/seal_preflight_v1.json)はgate/source/env/output/schemaをnumerical importなしで確認した。
87 preparation testsは既存の合成検証であり、この段階で実H6の検証証拠へ昇格しない。
[launch command](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/launch_command_v1.json)をseal前に固定。
GitHubへcommit/pushし、remote bytesを確認した後に一回だけ既存runnerを起動する。
output予定：`artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/`。seal時点では未作成。
生成後はintegral/DF/state/NPZ receipt・parent/worker terminal・partial/progress/logと保存bytes監査を保全・公開しSTOP。
背景起動後にチャットを終了しても親watchdogが監視する。次回は保存terminalと監査から再開し、runnerを再起動しない。

[seal inventory](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/execution_inventory_seal_v1.json)は認可済み・未起動というこの段階のimmutable記録。
実行状態は起動receiptとterminal、結果報告で別に記録する。
H6入力が揃った後のactual-rank依存7 cell/36 wrapper固定・pilot gate接続・別grantは次段。
N/Gnull、UNDETERMINED、u未認定、ground-state未認定。H6本検証の科学GO/STOPはGPTに戻す。
Track Bと別系列server run10のjob/sourceには触れない。
