# 反証が意味することと範囲

2026-10-18 JST。指定命題P_Uは`DISPROVED`。反例はL=2、p=(1/2,1/2)、x=1、m=7の1件で、`U_7>B^form_(5⊕o,7)`を厳密に示した。[certificate](proof_or_counterexample_certificate.md)

| 対象 | 今回確定した意味 | 今回は結論しないこと |
|---|---|---|
| 既存認証上界U | 指定全入力で強いclosed式の小さい方以下、という一様比較は成立しない。最小の許可次数と許可境界x=1で反例がある | 他の入力でUが有効でない、全域で同じ向き、Uが無効な上界、という結論 |
| 理想Full RA | `B_F,m≤U_m`という既存の上界保証は保持する。今回の反例はUの比較を否定する | `B_F,7>B^form_(5⊕o,7)`。UはB_Fの真値ではなく、この推論はできない |
| Closed P3対照 | P_Uは二つの比較の連言なのでP5側の反例で決着する | P_{U,3}の一般的な真偽、追加の反例、成立域・等号条件。これらは未検討 |
| Closed P5対照 | m=5の同じ理想分解・normalizerという既存対応を使い、m=7の形式ordinary tailを定義通り加えた | P5＋tailの統合sampler、digital保証、native費用が完成済みという認定 |
| G6/G7構成成果 | 正値分解・局所生成・有限bit保証の既存証拠とは別に、P_Uという一つの比較主張の限界を固定できた | 独立新規性、論文化GO/NO-GO、研究終了、一般的な資源優位の直接認定 |

反証には追加oracle、全parent和、真の受理率、観測variance、物理provider価格を使っていない。原SOURCEを静的に読み、その定義に有限wordの紙上計数と有理不等式を適用した。新しい科学SOURCEやfixtureは作らず、旧数学監査・certificate・研究方針へ結果を上書きしていない。

W1の固定README[W1]は既存lemmaと実装条件の区別を説明するが、この理想scalar反証の仮定ではない。W1の未完成処理のために命題が偽になった、という解釈はしない。

## W2b/W2cへ渡す結論

今後の作業が別途承認された場合、**P_Uによる一様なclosed後の上界優位を前提に置けない**という結論を渡せる。これだけでN、期待量子費用、全trial古典費用の大小は決まらない。

十分trial数にはfinite-bit precision、bias、range、confidence、ceil、Uの外向き評価が必要。実費には量子前zero、digital受理率、event依存価格、拒否試行の古典費用、setup、prep/readoutが別途必要である。未知Zを無料代入せず、既知ISやclosed/CTS等の合法access機会も維持する。

今回、新しいN、moment上界、break-even、費用順位を導出していない。W2b/W2cへの進行、別の不等式・U改良・入力条件の救済を正式採択していない。

**P_Uの決着に追加理論作業は不要。** 研究状態への正式反映や別問題の採否はユーザー/GPTの判断であり、必要な新レビュー・証明・科学実行は別の明示承認後。今回の公開・保全完了でmandatory STOPする。

<!-- FIXED_REFERENCES -->

[W1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8870ed37855ca8da97f8370e0a295d21cf5414f6/docs/research/track_b_g6_precision_api_cost_w1_20261018/README.md
