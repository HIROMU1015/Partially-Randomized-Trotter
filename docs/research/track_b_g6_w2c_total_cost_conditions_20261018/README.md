# G6/G7 W2c-A：全trial費用の比較条件

2026-10-18。技術状態：**`W2C_CONDITIONAL_WITH_EVIDENCE_GAPS`**。

Fullの古典費用が対照以下となる必要十分条件は、同じ費用単位・実装層で、削減trial分の費用が追加setup・一trial処理・保存費用を補うことである。量子の各座標では、認証全trial数と**無条件digital event価格の期待値**の積を比較する。条件式は閉じたが、一般入力のCPU秒・native費用順位は未確定である。

既存情報だけでも、Ordinaryと同じorder/label lawを使う条件では、Fullの期待accepted実行数とchild/angle非依存の論理的IR呼出数の非劣を確認できる。これは実測結果やnative総資源の比較ではない。Partial/Closed P3について同じ包含関係はなく、trial数・event価格・追加古典費用を個別に比較する必要がある。

## 対象・固定条件

基点はW2b最終 `ed26d1c85cf7ad80625f7abbc662b70abd690a97`。独立checkoutで専用branch `track-b-g6-w2c-total-cost-conditions-20261018` を使用した。既存SOURCE、旧資料、研究状態、親worktreeは変更しない。

4方式は `ordinary`、`partial_return_tail`、`closed_P3_tail`、`full_return`。有限演算子は

\[
M=P_m(-i\sigma xR),\qquad R=\sum_i p_iQ_i,\quad Q_i^2=I.
\]

正の明示有理p（同じ順序、和1）、有理0<x≤1、奇数m≥5、同じfinite P_m、σ±、η/ρ/H/K_root、外向きroot、angle/provider error、target/state/readout、ε_axis/confidence/familywise、全trial zero-fillを共有する。分子・geometry・basis・DF rank・split・δ窓の新評価はない。一般G6の形式入力に関する整理である。

W2bの認証Nをそのまま渡す。Nは**各axisの十分全trial数**でありaccepted数ではない。固定G7のα_axis=1/320、2 target×4方式×X/Y=16 axesのfamilywise≤1/20を保つ。任意target集合への同時保証や任意confidence APIを作らない。W2bのceil/moment/range条件、同一SOURCE/input/KでのN_F≤N_O、Partial/Closed P3との条件付き比較を再証明しない。

## 成果物

| 資料 | 内容 |
|---|---|
| [古典全trial費用](classical_total_cost_comparison.md) | F2標準bitモデル、R/C/A経路、canonical law再構築、F3追加費用、保存・memory |
| [量子event費用](quantum_event_cost_comparison.md) | digital law、literal IR、Ordinaryとの構造比較、native境界 |
| [break-even条件](break_even_conditions.md) | 必要十分な代数条件、上下界による認証、strict/tie/Pareto、比較不能 |
| [不足証拠・取得計画](missing_cost_evidence_and_execution_plan.md) | B1/B2/B3/C1/C2の11項目、依存関係、別途承認 |
| [引継ぎ](handoff.md) | 問いへの直接回答、歴史的結果と現在の不足、次の担当 |
| [証拠索引](evidence_index.json) / [SOURCE索引](source_inventory.json) | 固定commit/path/bytes/SHA256、読解範囲、静的AST位置 |
| [条件索引](cost_conditions.json) | 非実行metadata。入力値・実行許可・新budgetは含まない |
| [管理検査](verification.json) / [公開照合](publication_receipt.json) | 保全・リンク・静的対応・GitHub bytes検査。数学/数値検証ではない |

## 固定根拠と到達範囲

[W2b][W2B]、[F2][F2]、[F3][F3]、[G7 generator][G]、[ReturnKernel][K]、[provider][CP]、[E2E費用契約][E2E]を使用した。既存[P_U反証][PU]は維持し、救済や新normalizer定理を試みていない。

過去[G7結果報告][G7R]には、登録済みP5 development入力における条件付き期待Tの優位という既存結論がある。これはliteral helper IR、exact abstract CQ、取得済み有限RZの費用会計に限定した**過去結果の引用**である。新ランキング・F3付き全CPU・一般native優位へ拡張しない。[G8報告][G8R]の有限cold/warm/cache記録も利用可能だが、G7と異なるpolicy、同backend/seed、有限traceの範囲を保つ。

P5Closedはm=5の強い別対照である。G9のbias/confidence/log/axesがG7と異なるため、共通budgetなしにN/費用を混合しない。形式P5＋tailを実装済みsamplerとしない。IS/CTS/合法なstreaming/cache等にも公平な機会を与えるが、今回固定4方式以外の最適化・最適性は評価していない。

## 実施と未実施

実施：既存資料・SOURCEの読解、通常の代数的合成、静的AST/text/link/hash検査、文書・管理metadataの公開。現行共通指示2026-10-17の原本と固定公開bytesの一致を確認し、§8.4を適用した。軽量な独立I/Oは並列取得し、共有設定・venv/cache・他jobには触れない。

**0件**：新規科学計算、保存値再解析、新N/受理率/event期待値、列挙/Monte Carlo、科学SOURCE変更/import/実行、test、新非自明証明、回路build/compile/synthesis/backend、科学契約・authorization/grant/marker/claim発行、独立科学レビュー。管理検査PASSは科学的・数値的correctnessやプラットフォーム認証ではない。

**mandatory STOP。** W2c-B/C、SOURCE実装、DF/PR/RTE接続、追加理論、重要科学レビュー、論文執筆へ進まない。次はユーザーが既存の限定結論で十分か、追加取得を承認するか判断する。追加取得には対象・固定条件を明示した別途承認が必要である。

<!-- FIXED_REFERENCES -->

[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[E2E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/implementation_contract.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/README.md
