# 既存図と将来の図候補

## HIST-expectation_vs_saved

旧検証SVG：[図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18/expectation_vs_saved.svg)／[元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18/expectation_vs_saved.json)。S4互換性修正前の84行を描画。旧図の値・生成履歴を保全し、修正版の図と呼ばない。

## HIST-same_event_comparisons

旧検証SVG：[図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18/same_event_comparisons.svg)／[元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18/same_event_comparisons.json)。S4互換性修正前の84行を描画。旧図の値・生成履歴を保全し、修正版の図と呼ばない。

## FIG-01

同一eventの理論/native/compiled差。横軸：固定候補・軸（H4/H6は別panel）。縦軸：RZ又はCX count／保存rho。系列：A/B/mixed; native; compiled。

支持する事実：同じeventでもdeclared loweringとcompiledは異なる。 注意：1 eventをn8/n32平均と混同しない。 [元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1/artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/same_event_comparisons.json)。既存値で描画可能だが今回は生成せず、正式採用もしない。

## FIG-02

理論期待費用と有限標本平均。横軸：固定候補・軸。縦軸：保存theory lower/upperとcompiled_saved_mean。系列：A/B/mixed; PR n32又はn8; PF n1。

支持する事実：理論期待値と保存平均の条件付き差。 注意：widthはCIではなく、same-eventとの比較単位が異なる。 [元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1/artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/expectation_vs_saved.json)。既存値で描画可能だが今回は生成せず、正式採用もしない。

## FIG-03

固定pairのshot込み比較。横軸：5つのepsilon。縦軸：保存theory_PR_PF範囲とcompiled_PR_PF（基準1）。系列：全6固定pairを分離; mixed/RZ/factor1。

支持する事実：H4指定条件ではPR/PF方向が異なる。 注意：不適格を0扱いせず欠測表示。点を滑らかに補間しない。6相関行を独立実験としない。 [元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1/artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/pf_pr_comparisons.json)。既存値で描画可能だが今回は生成せず、正式採用もしない。

## FIG-04

S4修正前後の技術差。横軸：H4/H6・axis・metric。縦軸：保存old/new/native count。系列：A/B/mixed; compiled unchanged。

支持する事実：互換性修正と残るcompiled差を分離。 注意：微小な修正を総資源削減や新しい量子アルゴリズム効果と誇張しない。 [元データ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1/artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/old_new_differences.json)。既存値で描画可能だが今回は生成せず、正式採用もしない。
