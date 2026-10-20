# 旧図 → 簡潔版図の対応

旧図の共通prefixは`artifacts/resource_applicability/track_a_presentation_figure_updates/2026-10-20/`、固定commitは`0566c3bb6c9c1a2d634519db6aabc5d2ef412798`。新図は本ディレクトリの`figures/`に保存。各stemに`.png`・`.svg`・`.pdf`がある。

|ID|旧stem（共通prefixから）|新stem（本ディレクトリから）|図外へ移した説明|
|---|---|---|---|
|A1|`A_absolute/figures/absolute_rz_eps_0p05`|`figures/A1_absolute_rz_eps_0p05_clean`|shot/compiled定義、SE≠CI、H8 receipt/profile、入力差、離散線・母平均の限定、PF実測/model、P/A cohortとH2追加説明。|
|A2|`A_absolute/figures/absolute_rz_eps_0p0005`|`figures/A2_absolute_rz_eps_0p0005_clean`|A1と同じ長文・脚注。H2 n12等の数値ラベルは維持。|
|B|`figures/B_price_replacement_summary100`|`figures/B_price_replacement_summary100_clean`|同じ方向の解釈、旧70/新30・40条件の詳細、モデル誤差と置換感度の違い、H8条件の詳細、profile/runtime・S4・最終gate欠測等。|
|C1|`figures/C_pr_mean_cost_vs_n_presentation`|`figures/C1_pr_mean_cost_vs_n_clean`|分位帯は母平均CIでない、保存resamples、入れ子系列、平均と累積費用の違い、rare event・FTQC等。|
|C2|`figures/C_pr_standard_error_vs_n_supplement`|`figures/C2_pr_standard_error_vs_n_clean`|保存参照SE列・新計算なし、入れ子系列、費用定義、rare event・母平均保証の限定。|
|D|`figures/D_independent_confirmation8_auxiliary`|`figures/D_independent_confirmation8_clean`|正式主表未差替え、H2追加、再現判定の意味、H8公開証拠、有限標本・条件説明。|

図中には軸・単位・目盛り・凡例・系列・短いパネル名・要求精度・n/N・候補ID・保存判定を残した。AのH8†、DのH2*/H8†は脚注本文とともに記号を外した。Bの`PF remains lower-cost`は保存カテゴリ名`PF lower`へ短縮し、元の意味は[日本語説明](figure_notes_ja.md)へ移した。

文章・ページ余白以外の描画データは一致。[caption](captions_ja.md)、[日本語notes](figure_notes_ja.md)、[原文と短縮ラベルの全記録](verification/text_moves.json)、[保持ラベル](verification/retained_labels.json)、[provenance](provenance.json)を参照。
