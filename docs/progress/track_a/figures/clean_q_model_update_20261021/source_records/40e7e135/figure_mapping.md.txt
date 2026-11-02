# 図セットの対応表

既存6図は`55b719c8e0881bda848a8ed721b0cd452267b801`、追加4図の元は`7430a386bcd4dc545107e413cd022ab7cb261968`。`*`はPNG/SVG/PDFの3形式を意味する。

|ID|入力・旧図|新版セットでの参照・保存先|変更|
|---|---|---|---|
|A1|旧簡潔版`A1_absolute_rz_eps_0p05_clean.*`|`../2026-10-20/figures/A1_absolute_rz_eps_0p05_clean.*`|画像なし。notesでEの範囲を区別。|
|A2|旧簡潔版`A2_absolute_rz_eps_0p0005_clean.*`|`../2026-10-20/figures/A2_absolute_rz_eps_0p0005_clean.*`|画像なし。|
|B|旧簡潔版`B_price_replacement_summary100_clean.*`|`../2026-10-20/figures/B_price_replacement_summary100_clean.*`|画像なし。新版B節に「450軸は別図E」を追記。|
|C1|旧簡潔版`C1_pr_mean_cost_vs_n_clean.*`|`../2026-10-20/figures/C1_pr_mean_cost_vs_n_clean.*`|なし。|
|C2|旧簡潔版`C2_pr_standard_error_vs_n_clean.*`|`../2026-10-20/figures/C2_pr_standard_error_vs_n_clean.*`|なし。|
|D|旧簡潔版`D_independent_confirmation8_clean.*`|`../2026-10-20/figures/D_independent_confirmation8_clean.*`|なし。正式主表は不変。|
|E|診断`display/predicted_vs_measured.*`|`figures/E_predicted_vs_measured_clean.*`|短いタイトル、B0I→B0+I、filled/openキー。長文footerをnotesへ。|
|E2|診断`display/relative_error_distribution.*`|`figures/E2_relative_error_distribution_clean.*`|タイトル・軸ラベル短縮、B0+I表記、footer/S4仮説文をnotesへ。保存bin/輪郭は不変。|
|E3|診断`display/error_vs_q.*`|`figures/E3_error_vs_q_clean.*`|タイトル・B0+I表記、footer/記述平均の説明をnotesへ。保存平均線は不変。|
|E4|診断`display/error_vs_prefix.*`|`figures/E4_error_vs_prefix_clean.*`|E3と同方針。保存点・平均線は不変。|

診断の共通prefixは`artifacts/resource_applicability/track_a_pf_q_model_diagnostics/2026-10-20/`。追加4図は元SVGの非文字構造（点・path・histogram・平均線・一致線・帯・軸・基準・clipping）を完全保持。B0+Iを元の凡例枠に収めるため補足図の該当文字だけを2.5 SVG単位左へ移した。図中の英語・記号以外の説明は[日本語notes](figure_notes_ja.md)へ分離した。

[全図入口](README.md)、[caption](captions_ja.md)、[出典](provenance.json)、[SVG照合](verification/svg_equivalence.json)を参照。
