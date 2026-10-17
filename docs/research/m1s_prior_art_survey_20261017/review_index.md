# GPT独立科学レビュー用の資料索引

2026-10-17 JST。レビューは**未開始**。これは取得可能な固定資料と判断課題の引渡しである。
科学資料の固定ref：`81d6103ec89d485cb7336c3a90bd95dad66992aa`、repository：`HIROMU1015/Partially-Randomized-Trotter`。
本調査の提出refは公開receiptで指定する。このbranchより後の並行shot・cost成果は取り込んでいない。

## 最短の読解順

1. [調査報告](README.md)と[技術比較表](comparison.md)。
2. T1b原本の§§2–10とCheckpoint 2レビュー（Jensen floor・共鳴を含む）。
3. finite-step完了／科学レビュー、γ V1完了／科学レビュー、π/3限定gate報告。
4. [一次文献証拠](primary_evidence.md)のE01–E06/E08/E10–E11。
5. [文献一覧](bibliography.md)の固定一次版。必要時に他の高次化・費用・平均化文献へ進む。

## 固定GitHub資料

ファイルidentityは[repository_evidence.json](repository_evidence.json)。rawは閲覧・identityのみで、今回再計算していない。

| 役割 | 固定commitの資料 |
|---|---|
| repository入口 | [PROJECT_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/PROJECT_MAP.md) |
| 現状・証拠水準 | [docs/research/研究概要・現状.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/研究概要・現状.md) |
| 研究目的 | [docs/research/研究目的・研究課題.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/研究目的・研究課題.md) |
| evidence status | [VALIDATION_STATUS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/VALIDATION_STATUS.md) |
| 既存machine manifest | [artifacts/validation_manifest.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/validation_manifest.json) |
| 2026-10-16共通指示 | [docs/research/research_project_instructions_unified_2026-10-16.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/research_project_instructions_unified_2026-10-16.md) |
| T1a全証明 | [docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md) |
| T1b全証明 | [docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md) |
| 正式理論scope | [docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md) |
| finite-step完了 | [docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md) |
| γ依存V1完了 | [docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md) |
| π/3限定gate完了 | [docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md) |
| RPE条件・費用の記号整理 | [docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md) |
| 既存G0-pre文献監査 | [docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md) |
| 既存Checkpoint2後文献監査 | [docs/tracks/algorithm_codesign/m1s_checkpoint2_prior_art_delta_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_checkpoint2_prior_art_delta_20261016.md) |
| T1a独立受理範囲 | [docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md) |
| T1b・共鳴floor受理範囲 | [docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) |
| Gate0の全scope限界 | [docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md) |
| finite-step設計レビュー | [docs/research/m1s_finite_step_pilot_gpt_independent_review_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/m1s_finite_step_pilot_gpt_independent_review_20261016.md) |
| finite-step結果レビュー | [docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md) |
| Task A/B・RPE設計レビュー | [docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md) |
| V1科学条件レビュー | [docs/research/m1s_v1_scientific_conditions_independent_review_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/m1s_v1_scientific_conditions_independent_review_20261017.md) |
| γ依存V1結果レビュー | [docs/research/m1s_gamma_v1_completed_results_scientific_review_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/m1s_gamma_v1_completed_results_scientific_review_20261017.md) |
| finite-step raw | [artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json) |
| finite-step信号表 | [artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/signals.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/signals.csv) |
| finite-step対照表 | [artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/contrasts.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/contrasts.csv) |
| finite-step保存監査 | [artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/saved_result_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/saved_result_audit.json) |
| γ V1 raw | [artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json) |
| γ V1比較表 | [artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/gamma_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/gamma_comparisons.csv) |
| γ V1 method bounds | [artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/method_bounds.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/method_bounds.csv) |
| γ V1保存監査 | [artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/saved_completion_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/saved_completion_audit.json) |
| γ V1採択契約 | [artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/execution_contract_adopted.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/execution_contract_adopted.json) |
| π/3 gate raw | [artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01/result.json) |
| π/3 gate監査 | [artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/saved_result_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/saved_result_audit.json) |

## SOURCE・契約・結果の来歴

| 系列 | SOURCE | 契約 | 実施authority | 結果固定資料 |
|---|---|---|---|---|
| T1a | `4e7b18dd2aa14c127bce9e248c63996c62a055cc`に全証明 | S0 `373b8e4b9831f4d00bdc00ffc741ac61e03f0841` | 原本の範囲 | 全証明とCheckpoint1 |
| T1b | `54336028edd7ef759f8a0c73bd0a96ade13f47f9`に全証明 | 同S0 | 原本の範囲 | 全証明とCheckpoint2 |
| finite-step完了02 | `2eac35ff33f100fa96e296b737752addde5af5de` | `5afd57ed3feb900b8f97317c38cc62f52959981b` | `3dfd4470f8755937be1818c24580e8cbb76d3171` | `025c5a90888b02b73d2815fbb10d6bd1d2823602` |
| γ V1完了02 | `677dd6957e0e1913b70ca5f999458835ae8a2e23` | `2cb5160d9051a6568b8acc0cfb29fc169bdfca09` | `161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c` | `b34572770e7c2105e16da003af050d789f490662` |
| π/3限定gate | `2f800f179c7b29589b2f9a7e532408b16b8fe6d2` | `bf97fcb2607904f8b96a625c5b64440a78bbbfdc` | `a1dd87a1f81467bea0ce27cacf9d21e49cff7908` | 今回基点`81d6103e…` |

SOURCE directoryは`src/trottertracks/algorithm_codesign/`配下の`m1s_finite_signal_v2`、
`m1s_saved_signal_gamma_v2`、`m1s_gamma_pi3_v1`。各ファイルの固定URL/hashはmachine索引に含む。
旧authorityを新しい実行許可として流用しない。失敗試行・消費済markerも基点のまま保持する。

## 新規性判断に影響する未解決事項

| ID | GPTが判断する具体的論点 | 必要な照合／既存証拠の限界 |
|---|---|---|
| U1 | 同順sign reversalという特定joint lawに独立した方法差があるか | MLMCのcoherent共有、COSの対称word、SCUの一般cross termとの相関・周辺分布・角度・情報accessを揃える。外形や名称の一致だけでは結論しない |
| U2 | T1a/T1bの全保証は既知mean-operator／averaging定理の直接系か | MF Eq.(35)、Chen、COS、古典平均化へHS／ancilla liftして何が残るか。今回は還元定理未作成。Davies全文未取得 |
| U3 | 具体的係数・共鳴floor・定数のうちpriorityを支える部分はどれか | 既知分散恒等式・pinching・正のspectral measure/Jensenと、明示lawの新しいstatementを分ける。共鳴floorの原本はGPT2 |
| U4 | 有効mode／phase安定性はGu/YCからどの程度直接従うか | norm、pair-sum対energy-difference、非normal性、gap量化、三次係数、exact Dの違い。YC出版SIは今回関連箇所確認できた |
| U5 | finite-step visibility／phase tradeoffに独立した科学的価値があるか | 開発H4の限定観測、G9未解像、oracleとmodel drift、T1bとは極限が違う。一般性や新規性を本調査は認定しない |
| U6 | γ境界は新理論か、既知統計設計のinstance図か | Hoeffding＋円板幾何と既存レビューの閉形式境界、十分上界と最小sample complexity、ceilとconfidence scopeを区別 |
| U7 | RPE資源に結び付く実用差が残るか | Task A/B、KLY訂正、PR/RTEとWBC等の強い対照、whole-wrapper・state準備・normalizer・古典費用。進行中feasibilityの結果は別途確定後に読む |
| U8 | 理論ノート／方法論／応用資源研究のどのclaimが支えられるか | 新規性と研究価値・論文化の最終判断。今回は候補を採択しない |

これらは追加計算の実施指示ではない。新しい理論証明・simulation・native compile・保存値再解析は
ユーザーの別途指示があるまで行わない。既知の数学的対応の詳細判断自体も後続GPTレビューの範囲とする。

## レビュー開始条件と停止

今回ユーザーが委ねた最終新規性・研究価値の評価は、共通運用指示上のLevel 1に相当し、
GPT-6 Pro推奨。ただしモデル変更やレビュー開始を自動実行しない。
必須資料のGitHub可用性と公開identityは本調査のreceipt／repository evidenceで確認する。
未取得のDavies全文、全論文の出版proof、未完の還元証明は明示した検討上の限界である。
既存結果の数値再生成を始める条件にはならない。

**今回のCodex担当はここでmandatory STOP。GPTレビュー・新科学実施は未開始。**
