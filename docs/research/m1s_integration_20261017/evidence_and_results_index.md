# M1-S証拠・結果索引：主張から固定原本へ

2026-10-17 JST。[統合サマリー](integrated_research_summary.md)／[再開ハンドオフ](codex_restart_handoff.md)。
**既存証拠の案内。科学的数値再解析・独立再実験は実施していない。**

基点はStage S最終 `f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
先行研究調査 `4301c0f87a16be45b620304c3fa935d5aa2d721a` は別履歴。
本文の固定URLは、統合branchの将来HEADに依存しない。
[evidence_inventory.json](evidence_inventory.json)に全対象path/ref/bytes/SHA256/Git blobを収めた。
この独立索引は科学的validation manifestの代替・status変更ではない。既存manifest/ledgerは保全した。

## 1. 科学的主張と原本

| ID | 主張／状態 | 原本・根拠 | 限界 |
|---|---|---|---|
| T1A | 固定有限model・有限減衰でC-secへの一様bound | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md) | 固定r/次元/gap依存。受理はCheckpoint 1 |
| T1B | 全Hのpair-sumへ部分乱択移送、W/v/rho・位相・共鳴 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md) | exact HD、fixed r、小step。原本のpendingは提出当時 |
| CP1 | T1a限定数学受理 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md) | 新規性/資源認定ではない |
| CP2 | T1b限定受理、Jensen floor、rho単独STOP不採用 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) | 非縮退対象の参照floor。旧Claudeについては二次記録 |
| ADD | 後続Jensen結果を原本非改変で整理 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_g0t1_post_checkpoint2_theory_addendum_draft_20261016.md) | DRAFT/NOT ADOPTED。新証明ではない |
| G0-V4 | 精度事前gateでTECHNICAL_STOP、metrics null | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md) | 技術停止で科学NO-GOではない |
| G0-V5 | 3分割W/v/rho・17項目適用・1239到達gate PASS | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md) | 条件付きoracle診断のみ |
| LOCAL-BOUND | alpha_R=gamma_(713+r)*S_Rの条件付き導出 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md) | platform certificateではない。証明固定b5a185cc |
| IND | 独立整数/有理数reference、available PASS／overall UNDETERMINED | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md) | 旧full-H/HD行列未保存、同じ入力での事後照合 |
| FINITE | 108/108、G3/G6各16visibility higher、G9未解像 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md) | 条件付き、phase bias増加あり |
| FINITE-REVIEW | visibilityとbias・T1b極限差・費用を評価 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md) | 限定継続の推奨は採択そのものではない |
| V1 | G3/G6各16γ境界、G9形式順位と絶対未解像 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md) | 連続十分設計。絶対shots未計算の時点 |
| PI3 | G3 16/G6 T4 8成立、G6 T1 8不成立、G9 16未照合 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md) | RPE成功ではない |
| S | G3 T4 8局所整数witness、主対象他16cap同数 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md) | 同数は全γ不可能性ではない。4cell×二対照 |
| RPE-DESIGN | Task A/B・phase budget・全費用の非数値設計 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md) | 後続GPTの中心化/費用gate修正も読む |
| COST-DESIGN | shot・全wrapper/HD/準備/Fの記号会計 | [原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_shot_cost_feasibility_decision_packet_20261017.md) | 未採択の次期全費用案。Stage Cを開始していない |

数値の共通scope：linear H4 1.00 Å/STO-3G/8 spin orbitals/保存DF rank12、G3/G6/G9、保存oracle。
有限step以降はδ1/8,1/16 Ha^-1、R2,4、T1,4 Ha^-1。Gate 0はδ windowなし。
同じ保存信号の後続解析を独立再現と数えない。条件付き数値coverageを統計confidenceへ混ぜない。

## 2. SOURCE・契約・別実施承認の固定順序

S/C/Aはそれぞれ異なる役割。Cの採択はAを代行せず、過去Aは新実行許可ではない。
以下のS→C→A→Stage S基点へのancestor関係を今回Git metadataで照合した。

| 系列 | SOURCE S | 契約 C | 実施A |
|---|---|---|---|
| Gate 0 v4 | [`606036db52da2f080cf846e48824e5e418430762`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/606036db52da2f080cf846e48824e5e418430762) | [`5e57955baa79e42f37aa233ac011c60539e431fd`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5e57955baa79e42f37aa233ac011c60539e431fd) | [`b70abf111e6459c45c24331a2c7d99c05e7dc228`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b70abf111e6459c45c24331a2c7d99c05e7dc228) |
| Gate 0 v5 | [`69de022dc8142f703423b1ce0bd42f05e123954e`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/69de022dc8142f703423b1ce0bd42f05e123954e) | [`50b3669b9253eb889fa4daa87e793646df2d5138`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/50b3669b9253eb889fa4daa87e793646df2d5138) | [`8da63d672f779c947dc295eb2c5d524d927ac66b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8da63d672f779c947dc295eb2c5d524d927ac66b) |
| Independent Gate 0 | [`4787cf90b053d251bf6392988af35516bdbbe663`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4787cf90b053d251bf6392988af35516bdbbe663) | [`3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8) | [`2a55ba0103c68a074b7be7bc43b90b383bb63233`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2a55ba0103c68a074b7be7bc43b90b383bb63233) |
| Finite attempt02 | [`2eac35ff33f100fa96e296b737752addde5af5de`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2eac35ff33f100fa96e296b737752addde5af5de) | [`5afd57ed3feb900b8f97317c38cc62f52959981b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5afd57ed3feb900b8f97317c38cc62f52959981b) | [`3dfd4470f8755937be1818c24580e8cbb76d3171`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3dfd4470f8755937be1818c24580e8cbb76d3171) |
| Gamma attempt02 | [`677dd6957e0e1913b70ca5f999458835ae8a2e23`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/677dd6957e0e1913b70ca5f999458835ae8a2e23) | [`2cb5160d9051a6568b8acc0cfb29fc169bdfca09`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2cb5160d9051a6568b8acc0cfb29fc169bdfca09) | [`161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c) |
| Pi3 attempt01 | [`2f800f179c7b29589b2f9a7e532408b16b8fe6d2`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2f800f179c7b29589b2f9a7e532408b16b8fe6d2) | [`bf97fcb2607904f8b96a625c5b64440a78bbbfdc`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/bf97fcb2607904f8b96a625c5b64440a78bbbfdc) | [`a1dd87a1f81467bea0ce27cacf9d21e49cff7908`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a1dd87a1f81467bea0ce27cacf9d21e49cff7908) |
| Stage S attempt01 | [`7a338c4eba778c4ce1f189cfa8c8f050afb5df08`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/7a338c4eba778c4ce1f189cfa8c8f050afb5df08) | [`b4b4c12848d0cc65025e1ce9eb95902aec7d97fe`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b4b4c12848d0cc65025e1ce9eb95902aec7d97fe) | [`437e6bbc4cc2c550268d8644b8fc2bc0ec56e791`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/437e6bbc4cc2c550268d8644b8fc2bc0ec56e791) |

理論S0：`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`。
T1a証明：`4e7b18dd2aa14c127bce9e248c63996c62a055cc`。
T1b証明：`54336028edd7ef759f8a0c73bd0a96ade13f47f9`。
Gate 0局所係数証明：`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`。
元raw DF入力は8652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。

[保存入力NPZ（今回配列decodeなし）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)。

## 3. 各結果からSOURCE・machine契約・raw・監査へ

### Gate 0 v5

固定結果commit：`cdcb7f6f151df217960611f502eb1bac8be63f58`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/cdcb7f6f151df217960611f502eb1bac8be63f58/src/trottertracks/algorithm_codesign/m1s_gate0_v5)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/execution_contract_v5_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json)：440470 bytes、SHA256 `8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/publication_receipt.json)。
- [saved_identity_status_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/saved_identity_status_audit.json)。

### 独立Gate 0

固定結果commit：`cc2f451324b7292ae50cef94ff794d46f225d892`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/cc2f451324b7292ae50cef94ff794d46f225d892/src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/independent_validation_contract_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot/result.json)：541714 bytes、SHA256 `32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/publication_receipt.json)。
- [saved_only_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/saved_only_audit.json)。

### 有限step完了02

固定結果commit：`025c5a90888b02b73d2815fbb10d6bd1d2823602`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/025c5a90888b02b73d2815fbb10d6bd1d2823602/src/trottertracks/algorithm_codesign/m1s_finite_signal_v2)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_02/execution_contract_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json)：454422 bytes、SHA256 `ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/publication_receipt.json)。
- [saved_result_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/saved_result_audit.json)。
- [preservation_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/preservation_audit.json)。
- [保存表示 contrasts.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/contrasts.csv)。
- [保存表示 signals.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/signals.csv)。

### γ V1完了02

固定結果commit：`b34572770e7c2105e16da003af050d789f490662`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b34572770e7c2105e16da003af050d789f490662/src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/execution_contract_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json)：146764 bytes、SHA256 `bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/publication_receipt.json)。
- [saved_completion_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/saved_completion_audit.json)。
- [preservation_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/preservation_audit.json)。
- [保存表示 gamma_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/gamma_comparisons.csv)。
- [保存表示 method_bounds.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/method_bounds.csv)。

### π/3 gate

固定結果commit：`81d6103ec89d485cb7336c3a90bd95dad66992aa`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/81d6103ec89d485cb7336c3a90bd95dad66992aa/src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/execution_contract_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01/result.json)：37054 bytes、SHA256 `693ce941bf7793a6104af28443746f43e05d4903118c0ebbb6f09729f389fadf`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/publication_receipt.json)。
- [saved_result_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/saved_result_audit.json)。
- [preservation_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/preservation_audit.json)。
- [保存表示 pi3_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/pi3_comparisons.csv)。

### Stage S

固定結果commit：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。

- [SOURCE directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/src/trottertracks/algorithm_codesign/m1s_stage_s_v1)／[machine契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/execution_contract_adopted.json)。
- [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_result/2026-10-17/attempt_01/result.json)：67770 bytes、SHA256 `9a11df3fcc60609cfe63ebcd392c6b78585054983487a730dbdf603f8682ab0b`。今回hashのみ、科学値の再集計なし。
- [evidence_index.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/evidence_index.json)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/publication_receipt.json)。
- [saved_result_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/saved_result_audit.json)。
- [preservation_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/preservation_audit.json)。
- [保存表示 stage_s_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/stage_s_comparisons.csv)。

SOURCEを実行する入口は権限確認後にだけ使用する。実装済み・テスト済みという履歴を実行承認にしない。
全関連runner/checker/tests、旧namespaceもmachine inventoryから同じ固定refで取得できる。

## 4. 失敗・旧試行・markerの索引

| 履歴 | 保全対象 | 解釈 |
|---|---|---|
| Gate 0 v4 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md) | global precisionでSTOP、W/v/rho null、one-shot消費済み |
| finite起動前 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_pre_execution_configuration_stop_20261016.md) | Fraction精度文字列parse不具合、科学起動0・未消費 |
| finite試行01 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_01/result.json) | 16行まで、precision STOP。消費済み。02と独立した出力 |
| finite修正承認 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/repair_and_completion_authority.json) | 後続のユーザー完了指示。閾値を緩和せず別S/C/A |
| finite01 marker | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_01/exclusive_launch_marker.json) | 消去/再利用しない |
| γ試行01 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_v1_gamma_one_shot_technical_stop_20261017.md) | Python optional APIで入力decode前STOP。marker消費済み |
| γ01 raw | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_saved_signal_gamma_v1_result/2026-10-17/v1_one_shot/result.json) | 未評価placeholderを保全 |
| γ01 marker | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_saved_signal_gamma_v1_result/2026-10-17/v1_one_shot/exclusive_launch_marker.json) | 消費済み、02は新authority |
| γ修正来歴 | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_v1_gamma_runtime_repair_and_completion_contract_20261017.md) | 後続の明示修正/完了承認。科学条件不変 |
| Stage S marker | [固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_result/2026-10-17/attempt_01/one_shot_consumed.json) | 初回完了、grant0、marker1、retry0の既存記録 |

残る全authorization/grant/ledger/markerのpathとhashはinventoryのrole欄でも区別した。
歴史的authorityを新grantとして再発行しない。各系列の旧失敗後にユーザーが許可した別試行を、
無承認自動retryとも独立した科学再現とも記載しない。

## 5. レビュー原本・受理・推奨の時系列

| レビュー | 取得原本 | 引継ぐ判断 |
|---|---|---|
| 2026-10-15 Claude完成版に対するGPT | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md) | 条件付き理論候補、freshness/長時間/移送/費用の留保 |
| G0-T1 Claude草案に対するGPT | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md) | A′方針支持、局所高次誤差・dressing相対誤差を修正 |
| S0正式採択レビュー | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md) | C-sec主target、fixed r joint scalingの範囲 |
| Checkpoint 1 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md) | T1a限定数学受理、新規性は未認定 |
| Checkpoint 2 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) | T1b/Jensen受理、rho総資源STOP不採用 |
| Gate 0 v3 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md) | 残差qualityを絶対1e-8 Haへ改訂する根拠 |
| v5数学 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_gate0_v5_gpt_independent_mathematical_review_20261016.md) | 局所誤差モデルの条件付き受理、platform未認証 |
| 独立Gate 0結果 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md) | available PASS、overall UNDETERMINED保持 |
| finite設計 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_finite_step_pilot_gpt_independent_review_20261016.md) | Hamiltonianモデル差と丸めの分離 |
| finite結果 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md) | visibilityとbiasの代償、限定継続推奨 |
| RPE設計 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md) | Task A先行/Task Bの区別とraw route |
| V1科学条件 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_v1_scientific_conditions_independent_review_20261017.md) | 単一γからγ依存へ。未採択数値条件の扱い |
| γ完了 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_gamma_v1_completed_results_scientific_review_20261017.md) | 境界の限定価値、π/3との交差を推奨 |
| Stage S設計 | [GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/research/m1s_stage_s_scientific_conditions_review_20261017.md) | cap単調性/floor+1/no-witness非対称性、当時未採択 |
| 最新Claude新規性 | [受領原本](../m1s_prior_art_novelty_and_research_value_claude_review_20261017.md) | 新primitiveの新規性弱い、ノート優先の推奨。Stage S未完了の時点 |
| 最新GPT総合再評価 | [受領原本](../m1s_integrated_reassessment_claude_prior_reviews_stage_s_20261017.md) | centered分散/partial構成/費用gateを修正、Stage S更新、全定理priority未決、推奨未採択 |

旧Claude inner 3原本は今回未回収。旧GPT原本に引用された判断・SHAは二次記録としてのみ使用する。
最新Claude原本54,861 bytes/SHA256 `51dcfcd3a94300a4637a79d4fa3d453a07a3a16c87a000fcf5bfbce9bda0e50e`、
最新GPT原本50,382 bytes/SHA256 `52a5f73616ebae98b1fc51657c2d591622a9f2b1c63ab24c9f069e738aff05ff`。
共通指示2026-10-17は38,191 bytes/SHA256 `d2ebdac4b0e44e871c65f240f317e6b5d3c3460f27d92f4d798992da90bf422f`。
全3原本はユーザーの追加後に全文読解し、byte copyした。過去の添付/zipまで新たに取得したとは主張しない。

## 6. 別履歴の先行研究調査

- [README.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/README.md)。
- [bibliography.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/bibliography.md)。
- [primary_evidence.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/primary_evidence.md)。
- [comparison.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/comparison.md)。
- [review_index.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/review_index.md)。
- [literature_retrieval.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/literature_retrieval.json)。
- [repository_evidence.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/repository_evidence.json)。
- [audit.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/audit.md)。
- [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/publication_receipt.json)。

調査自体はNOVELTY_NOT_ADJUDICATEDで停止。後続Claude/GPTの科学判断とは別の成果である。
今回、論文本文を新たに検索・取得・再評価していない。既存の版/頁/式/取得失敗を調査原本からたどる。
Davies全文未取得、YC出版SIのbyte hash未取得などの限界を維持する。

## 7. 今回の文書確認と公開証拠

[文書検査](document_checks.json)、[保全監査](preservation_audit.json)、[公開receipt](publication_receipt.json)は
この記録整備のmetadata証拠。研究のnumerical correctnessや独立科学受理ではない。
receiptはpayload公開後の子commitで固定し、receipt自身を自己参照hashで認証しない。
最終remote SHAとreceipt公開bytesは公開後に別途確認する。

**新科学計算0、新証明0、新レビュー0、SOURCE/科学契約/旧authorityの変更0。公開・保全確認後mandatory STOP。**
