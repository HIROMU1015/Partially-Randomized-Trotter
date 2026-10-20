# 今回未対応の主要既存図

優先6図は簡潔版を作成済み。それ以外は以下の固定済み主要図群のSVG本文とMatplotlibコメント内の文字列を調査し、原図を保全した。新規科学レビュー・結果採否・再解析は含まない。リポジトリ全履歴・全worktree・private rawを横断した網羅調査ではない。

通常の軸・凡例・短いラベルは削除候補としない。文章段落が見つかった次の21図は今後の整理候補であり、今回の追加修正・旧図差替えは行わない。H8 cost breakdownはFigure 2系として調べた。独立確認8条件の旧補助表示は今回のDに対応済み。

[調査記録・元SVG hash・英文例](verification/existing_figure_inventory.json)を保存。

## H-chain absolute cost v2

固定commit：`0566c3bb6c9c1a2d634519db6aabc5d2ef412798`。

- [absolute_rz_eps_0p0005.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/absolute_rz_eps_0p0005.svg)
- [absolute_rz_eps_0p001.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/absolute_rz_eps_0p001.svg)
- [absolute_rz_eps_0p005.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/absolute_rz_eps_0p005.svg)
- [absolute_rz_eps_0p01.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/absolute_rz_eps_0p01.svg)
- [absolute_rz_eps_0p05.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/absolute_rz_eps_0p05.svg)

見つかった文章：SE≠CI、H8 receipt/profile、入力生成差、離散線・母平均の限定。両端の現行用途は新A1/A2で対応したが、旧版と中間3精度には注釈が残る。

## H8 cost breakdown / Figure 2

固定commit：`0566c3bb6c9c1a2d634519db6aabc5d2ef412798`。

- [figure2A.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/figure2A.svg)
- [figure2B.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0566c3bb6c9c1a2d634519db6aabc5d2ef412798/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/figure2B.svg)

見つかった文章：cohort区分、高次event未観測、母平均優位未認定、旧時点の説明欠測等。将来整理時も元図の時点履歴を消さず日本語へ移す。

## Theory / compiled totals

固定commit：`f15752625613956d087bab221e48f1ce7dd59fef`。

- [h4_absolute_rz_factor_1.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h4_absolute_rz_factor_1.svg)
- [h4_absolute_rz_factor_10.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h4_absolute_rz_factor_10.svg)
- [h4_ratio_rz_factor_1.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h4_ratio_rz_factor_1.svg)
- [h4_ratio_rz_factor_10.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h4_ratio_rz_factor_10.svg)
- [h6_absolute_rz_factor_1.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h6_absolute_rz_factor_1.svg)
- [h6_absolute_rz_factor_10.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h6_absolute_rz_factor_10.svg)
- [h6_ratio_rz_factor_1.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h6_ratio_rz_factor_1.svg)
- [h6_ratio_rz_factor_10.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/figures/h6_ratio_rz_factor_10.svg)

見つかった文章：期待model費用と有限標本compiled費用、端点≠CI、欠測・不適格、RZ範囲とFTQC除外。

## H4 PR variability

固定commit：`295b93d50662a0f8af3ec469df43099a3f04af60`。

- [pr_mean_cost_vs_n.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_mean_cost_vs_n.svg)
- [pr_observed_strata_cost.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_observed_strata_cost.svg)
- [pr_saved_S2_groups.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_saved_S2_groups.svg)
- [pr_standard_error_vs_n.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_standard_error_vs_n.svg)

見つかった文章：条件付き分位帯、保存SE参照列、層別min-max≠SE/CI、未観測層、疎なS2群と非因果解釈。C1/C2の主/補助図は今回対応したが、この旧4図は原図保全。

## q-dependent unit-price sensitivity

固定commit：`3983fe4f1ede003953735e757d2c098c12b4644f`。

- [PF_cost_change100.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3983fe4f1ede003953735e757d2c098c12b4644f/artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PF_cost_change100.svg)
- [PR_PF_ratios100.svg](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3983fe4f1ede003953735e757d2c098c12b4644f/artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PR_PF_ratios100.svg)

見つかった文章：profile/runtime差、S4仮説、初回最終gate欠測、旧70/新30、PR n32、scope重複。Bの本編要約は今回対応済み、旧2詳細図は補足版として保全。

## 今回の扱い

将来候補の一覧化で今回の作業を終了する。追加整理は自動実行しない。新たな必要条件や独立レビューを要求するものではない。次の作業なし。
