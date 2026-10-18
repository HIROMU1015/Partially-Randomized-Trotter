# Track B G6/G7 W2b：認証された十分全trial数の比較

2026-10-18 JST。**W2B_CERTIFIED_CONDITIONS_CLOSED**。固定G7 budgetのmoment・range・共通bias・remaining・confidence・外向きroot・ceilを残し、同一条件のFull RA対Ordinary/Partial/Closed P3の比較条件を直接代数で確定した。具体入力への数値代入、SOURCE import/実行、新証明・科学計算は行っていない。

**結論**：同じ固定G7/K SOURCEとK_rootなら、Ordinaryとの実上端の対応を静的に追えるため`N_F≤N_O`。Partial/Closed P3については、`o=B_o^+`,`u=U^+`,`b=B_a^+`,`c=2s/[3(1+ρ)]`と置き、ceil前の非劣条件は`o(u+c)≤b(b+c)`。整数Nの正確な条件は別に`T_F≤ceil(T_a)`であり、strict条件は`T_F≤ceil(T_a)−1`。Partial/Closed P3に対する全域優位は認定しない。

「条件が閉じた」は個々の入力の勝敗取得や一般資源優位を意味しない。条件には同じ正のremaining、既存finite-bit law/測定の成立が必要。[F3]はFullのpreflightだけを実装しており、canonical非zeroを実行可能としない。

| 成果物 | 内容 |
|---|---|
| [certified_budget_comparison.md](certified_budget_comparison.md) | **主資料**。4方式のG7認証N、SOURCE対応、同条件・confidence・precision |
| [trial_count_inequality_conditions.md](trial_count_inequality_conditions.md) | moment/rangeの直接差分、Ordinaryの実上端、ceilの必要十分条件、外側区間 |
| [comparator_certification_matrix.md](comparator_certification_matrix.md) | 4方式とm=5 P5Closedの認証・実装・比較可能性。G9は別契約 |
| [w2b_conclusions_and_limits.md](w2b_conclusions_and_limits.md) | P_U反証を含む結論と不足、一般主張との境界 |
| [handoff.md](handoff.md) | 次の担当・別途承認・W2cへ渡す情報のみ |
| [comparison_conditions.json](comparison_conditions.json) | 非実行の比較条件・status索引 |
| [source_inventory.json](source_inventory.json) / [evidence_index.json](evidence_index.json) | 固定commit/path/関数位置/SHA256・静的取得範囲 |
| [verification.json](verification.json) / [publication_receipt.json](publication_receipt.json) | 管理/公開bytes検査。数学機械検証ではない |

基点`1aea9b8c228c5b940f2fcba0f94493b3ffdec46e`、branch`track-b-g6-w2b-certified-trial-budget-20261018`。変更は本専用directoryのMarkdown/JSONだけ。独立clone/checkoutを使用し、他branchをmerge/rebaseしない。現在の親PROJECT_MAP/研究概要/共通指示は既読原本とbytes一致を確認し、共通§8.4を再読した。SOURCE・test・旧成果・STOPを保持する。

新規科学計算/保存値再解析/SOURCE変更・import・実行/新tests/新非自明証明/新実行契約・authorization/grant/marker/claim/回路・合成は**0件**。W2a/P_Uを再証明・再反証せず、[W2]のW2bだけを扱った。**公開・保全確認後mandatory STOP。W2cへ自動移行しない。**

固定仕様は[F12]/[F1]/[F2]、正式共通運用指示は[COMMON]。既存の入力数値・科学状態を変更しない。

<!-- FIXED_REFERENCES -->

[W2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/de709339e2539fa4d9806a84e738040d52c97be1/docs/research/track_b_g6_theory_deepening_triage_20261018/minimal_next_work_proposals.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/README.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/README.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md
