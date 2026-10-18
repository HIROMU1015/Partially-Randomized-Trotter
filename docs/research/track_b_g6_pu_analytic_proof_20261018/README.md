# G6/G7 P_U：限定的な解析的反証

2026-10-18 JST。**結果status：`DISPROVED`。** 指定された命題P_Uは、許可域内の単一入力

\[
L=2,\quad p_1=p_2=\tfrac12,\quad x=1,\quad m=7
\]

で破れる。本記録の厳密な有理数評価により

\[
U_7-B^{\mathrm{form}}_{5\oplus o,7}>\frac{13}{1024}>0
\]

が成立する。これはP_{U,5}の反例なので、minを用いたP_U全体が否定される。反例確定後、P_{U,3}の追加検討や有利な領域の探索は行わない。

## 固定対象・出典

基点commitは `43d6533970e0eb4c430fe1cff2e798d681b707d4`、専用branch/worktreeは `track-b-g6-pu-analytic-proof-20261018`。命題とnormalizerの正本は[固定W2aの問い][S02]と[定義表][S03]。全有限L≥2、正の明示有理p、Σp=1、有理0<x≤1、奇数m≥7、同じ有限Taylor operatorを対象とする。σ±やQ間の追加関係はこれらscalar式を変えない。

既存定義は[G6数学監査][M]、[G7契約][GC]、[U導出][U]、[K]、[G]、[P5 SOURCE][P5]と[既存P5導出][PC]から静的確認した。W1は指定固定commit `8870ed37855ca8da97f8370e0a295d21cf5414f6` の[README][W1]だけを読み、理想scalar不等式に技術的未完成事項を追加していない。資料14件の来歴は[source_inventory.json](source_inventory.json)。

## 成果物

| 文書 | 内容 |
|---|---|
| **[analytic_proof_attempt.md](analytic_proof_attempt.md)** | 対象の分離、既存式、今回の有限word計数・係数・反証の全導出 |
| **[proof_or_counterexample_certificate.md](proof_or_counterexample_certificate.md)** | 単独で追える反例と厳密な不等号の証明 |
| [implications_and_limits.md](implications_and_limits.md) | 理想Full/U/形式P5、W2b/W2cへの限定的な意味 |
| [handoff.md](handoff.md) | 結果・適用範囲・追加理論作業の要否・担当・STOP |
| [evidence_index.json](evidence_index.json) | 既存定義、新しい紙上導出、未検討事項の区別 |
| [verification.json](verification.json) / [publication_receipt.json](publication_receipt.json) | 文書管理・公開・保全の検査。数学的証明の機械検証ではない |

ユーザーがこの1命題について承認した新しい解析的理論作業である。平方根のノルム不等式と有理数の紙上評価を用い、Python等の数値計算、数式ソフト、grid、科学SOURCE/testのimport・実行・変更、fixture、保存値再解析、回路・合成、W2b/W2cの新しい資源結果は0件。管理用コードはGit・Markdown・JSON・リンク・hashだけを扱い、反例の計算や証明確認に転用しない。

共通運用指示2026-10-17は既読と同一bytesを確認し§8.4を再確認。独立した公開text/hash取得14件を並列化した。計算資源の人工上限・共有環境変更・科学計算grantの発行はない。変更は専用研究記録9ファイルだけで、旧SOURCE/結果/certificate/契約/grant/ledger/marker/claim/STOPを変更しない。他worktreeの内容・index・lock・process・出力にはアクセスせず、並行F1/F2の独立した変更を監視・停止しない。

**公開・remote HEAD・全公開bytes/SHA256・保全確認後mandatory STOP。** P_Uの決着に追加の理論作業は不要。別不等式、U改良、sampler、数値探索、W2b/W2c、重要科学レビュー、論文執筆へ自動移行しない。

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/general_order_comparison_question.md
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/baseline_normalizer_definition_matrix.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
[W1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8870ed37855ca8da97f8370e0a295d21cf5414f6/docs/research/track_b_g6_precision_api_cost_w1_20261018/README.md
