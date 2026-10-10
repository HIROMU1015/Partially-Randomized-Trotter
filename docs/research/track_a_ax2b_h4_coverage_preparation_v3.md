# Track A AX-2B：H4 coverage serialization修正・準備v3

2026-10-10 JST。利用者の「次のGPTレビューが必要だと思われるところまで作業を進めて」を、直前の修正・合成検証・新固定・別認可付きH4一回実行の範囲へ適用する。
旧[H4限定STOP](track_a_ax2b_h4_limited_execution_v1.md)のsource・manifest・認可・8 raw fileを保持し、凍結v2は編集しない。
この文書の段階はsource実装と合成検証。分子の正しさ・u・科学GO/STOPの新しい結論ではない。

## 保全的修正と検証

[新coverage helper](../../src/trottertracks/resource_applicability/ax2b_coverage_binding_v3.py)はtuple/listだけを正規化し、有限binary64値（signed zero含む）・scalar型・配列順序・重複・key集合を厳密に照合する。
`bool/int`、`int/float`、one-ULP時間変更も不一致とする。数値toleranceや丸めを導入しない。
各canonical coverageは4MiB/200,000 node/depth64を上限とし、差分は最大24 row/各path256文字。
新[port](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)はboundsのcap確認後、actual coverageと比較receiptを既存の排他的・総output/diagnostic上限付きwriterへ保存し、不一致を拒否する。数値reference/actionは照合通過後のみ。
cap違反や不正JSON値ではcoverage比較まで達しない場合があり、全失敗でactual保存を保証するものではない。

新[grant gate](../../src/trottertracks/resource_applicability/ax2b_bound_launch_v3.py)と[runner](../../scripts/resource_applicability/run_track_a_ax2b_bound_v3.py)はH4限定・v3 schema・別認可・manifest-intended outputへの結合を強制する。
旧v2認可を再利用しない。limits/one-shot claim/watchdogと数値backend algorithmは旧経路を保つ。
[静的変更監査](../../artifacts/resource_applicability/track_a_ax2b_h4_coverage_preparation_v3/2026-10-10/static_change_audit_v1.json)でport/runnerの許可した局所patch以外と、gateのinput/scope/environment/source関数ASTの一致を確認した。

[専用tests](../../tests/tracks/resource_applicability/test_ax2b_coverage_binding_v3.py)は49 passed。[保存test記録](../../artifacts/resource_applicability/track_a_ax2b_h4_coverage_preparation_v3/2026-10-10/tests_v1.json)はlocal metadata/mock engineering evidence。
登録8 cellの実schedule JSON往復、値・順序・時間・件数・上界の変異拒否、実setup methodの照合位置、bounded failure保存、認可/schema/output/H6拒否を検査した。
wrapper上界は保存値を使い、新native準備で再計算していない。numpy/scipy/mpmath/qiskit/openfermion/pyscf importとNPZ payload accessをguardし、分子load・prepare・signal・sampling・build・compileを実行しない。
setup検査のload/prepare/referenceはmockであり分子PASSではない。

## 新固定から一回実行への範囲

[metadata-only reseal](../../scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v2.py)は旧manifest c56ebc4・STOP79858db・旧source61091c2・freeze189と新sourceを結び付ける。
source commit後に新manifest/freeze/auditを専用 `artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10/` へ排他的に保存・公開する。
保存snapshot/header/Unicode/byte監査と純粋scheduleのみ。実行時の全actual boundsが一致することは、今回sealでも未確定である。
`execution_plan_sealed=true`は認可ではなく、manifestのscience_authorized/launch_allowedはfalseのまま。

科学scopeはlinear H4 1.00 Å/STO-3G/legacy DF rank12/generation-prefix L_D=0/6/12、8qubit、Nα=Nβ=2・sector36、T=0.8。
保存binary64 H_DFと指定保存stateの数学的正規化をtargetにし、旧8 cell・q1/q4（δ0.8/0.2）・B2/B3 R8/r2/K2/4/6、B1二次/四次を保持する。
179 primitive-time組/537作用、MP80/120、4 explicit代表event群を登録。control上限200（計画100）、trajectory/compile/occurrenceは0。
CPU3/worker1/BLAS1、phase900/1800/300秒・total3000秒、AS8GiB・output512MiB・log64KiB・diagnostics1024の旧capsを変更しない。
ASはRSS上限ではない。新Hamiltonian/state生成・候補探索・再fit・H6/H8/GPUは対象外。

新固定をGitHubから照合後、今回利用者指示を根拠とする別v3認可を新manifest/source/CPU/outputへ結合して公開する。
出力は新 `artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2`。旧grant・launch_v1のretry/resumeはしない。
一回run後は成功・失敗・欠測を保存・公開してmandatory STOP。追加run・科学修正を自動実施しない。
state/target/metric/PF/control/独立性変更が必要、または正しさを左右する矛盾が判明した場合は[独立レビュー§21](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)に従い、その時点でGPTへ戻す。

H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATIONを維持。N/G=null、accuracy UNDETERMINED、数値allowance未認定。
新source/工程の来歴は[準備inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_coverage_preparation_v3/2026-10-10/preparation_inventory_v1.json)へ登録し、旧validation_manifestや凍結結果を改変しない。
