# PR資源評価：PF総費用変化量改訂版 Codex参照メモ

作成：2026-10-21 JST。[今回の構成案](PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_PF総費用変化量改訂版.md)。本編は14枚7／圧縮12枚6だけを改訂し、B.19と対応表・図参照・保存記録を同期した。その他の本編1–6・8–14は本文・図・ノートをそのまま保つ。

## 1. 原本・範囲・変更の理由

原本は commit 0ddd7c70f48542e9c536ff1b1211ef73350957ba、docs/progress/track_a/PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_図・qモデル検証更新版.md。原本SHA256：22643b43b4f032e4b2460762a9987fc55a055ad6e1b88d691b1f365ad73a13f5。通常作業場所の同名MDと固定Git objectは一致する。原本・原画像・旧メモ／検証記録は保全し、別名MDを作成する。

以前の主図Bは変更件数を示していた。今回の主図は最良PF総費用の保存変化率を全100scopeで示し、H8候補変更2scopeの前後の最小総費用・方向判定を具体化する。新しい費用・変化率・候補選択・ランキング・統計量を生成しない。

専用branch：codex/pr-slides-pf-cost-change-20261021。作業用worktree：/tmp/pr-slides-pf-cost-change-20261021。通常作業場所にも新版を保存する。今回の指示は図の描画・文書更新・技術検証・local commitまで。以前の版に対するpush指示を今回の新成果物の公開承認へ拡張しない。

## 2. 固定SOURCE・保存field

<a id="saved-cost-change"></a>

B_PRICES結果commit：3983fe4f1ede003953735e757d2c098c12b4644f。

科学SOURCE：c0d8ca8ecd3fc29528ca8424f8b8d953d06eff91。

root：artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/

確認・コピー対象：combined_selection100.json/csv、figure_data.json、summary.json、PF_cost_change100.png/svg/pdf、PR_PF_ratios100.png/svg/pdf、README.md。原bytesを専用図ディレクトリsource/へ保存した。READMEは相対リンクを移設先の実用リンクと誤認しないようREADME.md.txtで保存した。コピーのcommit/path/SHA/bytesは[source_provenance.json](figures/pf_total_cost_change_20261021/source_provenance.json)。

元の報告は同commitの docs/research/track_a_h6_noncancellation_stage_b_rerun_20261020.md。後続の条件付き受入は80b7a35c4dbf683f952293eeb55944bc10b9efc8、docs/research/track_a_pf_q_model_evidence_integration_20261020.md。前回の[参照メモ](2026-10-21_PR資源評価_図・qモデル検証更新版_Codex参照メモ.md)のB_PRICES/B_INTEGRATIONを継承する。

|表示・説明|原field|
|---|---|
|縦軸の費用変化率|figure_data.json / points[i] / total_change_percent を無加工のfloat値で描画|
|横軸のscope index|元points配列の0始まりindex＋1。表示用番号のみ、ソートなし|
|系・bias・scope・精度・allowance・来歴|各pointの system / bias_mode / scope / epsilon / factor / origin|
|政策間の最小PF費用の表示丸め|各pointの baseline_G / replacement_G を.6e表示。費用から変化率を再計算しない|
|不変／非ゼロ・候補変更の記号|combined_selection100.json / rows の winner_total_changed / best_candidate_changed。保存フラグをそのまま使用|
|H8のexact総費用|同rowsの baseline.G_exact / replacement.G_exact を文字列のまま転記|
|H8の保存PR/PF比|同rowsの PR_over_PF_baseline / PR_over_PF_replacement。差分や変化率を計算しない|
|保存判定・件数|同rowsのPR_PF_relation_*とsummary.previous70 / new30 / all100|

[display_points.json](figures/pf_total_cost_change_20261021/display_points.json)に元100pointの全field、index、元候補・exact費用・保存フラグを対応づけた。描画に使わない比なども元pointを丸ごと保持する。SVGとPNG/PDFの作図から科学データを逆算せず、原JSONを数値正本とする。JSON再serializationの写しは元bytesコピーではなくfield値の抽出記録である。

## 3. 100scopeと主図の表示

主図：[PF_total_cost_change100_clean.png](figures/pf_total_cost_change_20261021/PF_total_cost_change100_clean.png)。同名SVG/PDFも保存する。従来図Bと混同しないよう、本文では「PF総費用変化率図」と呼ぶ。

1–30：旧70のH6 signed、31–70：旧70のH8両bias、71–100：後続新30のH6経験的非相殺。青丸・橙四角・緑三角で来歴と系を区別し、0は白抜き丸。H8のindex68と70は候補変更の星を重ねる。重ねた2つの星を新しい条件2件に数えない。100scopeは重複除外40endpoint条件で、独立100実験ではない。

縦軸は正がPF費用増、負が減、0は不変。PR削減率やPR/PF比の変化率ではない。保存100pointすべてを一回ずつ主系列へ置き、50非ゼロ・50不変という既存集計と対応させる。座標の重なりを避けるため保存順のindexを使い、jitter、候補再ランキング、費用順ソート、histogram/bin、分布集約を加えない。

本編の−0.221699%はH6_DF19|signed_signal|0.05|1.0|fine_grid_S2、＋0.327685%はH6_DF19|signed_signal|0.0005|1.0|fine_grid_S2_plus_truncated_S4 の保存値の表示丸め。特定の点の例として使い、新しく集計した最小・最大とは呼ばない。＋0.327685%という点があるので「全体が0.3%以内」と一般化しない。グラフのylimは表示範囲だけで科学的な閾値や保証ではない。

元の図の二次元表示は費用の近い点が重なりやすいため、100点の変化率の配置を明示する図へ変更した。分布を新しく推定した図ではない。全データを再会計したりPR/PFを再比較して方向を決め直したりしない。

## 4. H8の2scope：exact値と解釈

条件ID：

- H8_DF28|EMPIRICAL_NONCANCELLATION|0.0005|1.0|strengthened184（保存index68）
- H8_DF28|EMPIRICAL_NONCANCELLATION|0.0005|10.0|strengthened184（保存index70）

両者はVERBATIM_PREVIOUS70。候補はB0I_p16_q4_4th→B0I_p15_q4_4th、同q4/S4。元の保存rowsの写しは[H8_candidate_change_saved_rows.json](figures/pf_total_cost_change_20261021/H8_candidate_change_saved_rows.json)。

|倍率|baseline.G_exact|replacement.G_exact|保存total_change_percentの表示|baseline PR/PF|replacement PR/PF|
|---:|---|---|---:|---:|---:|
|1|770296671316160/7|110353805016120|＋0.283003%|1.290910|1.287267|
|10|770305822651840/7|110355077430428|＋0.282968%|1.290921|1.287278|

Gの分数・整数は文字列のままで、Fraction評価・除算・会計を新しく行わない。本編のGは保存済みfigure_dataの表示丸め。変化率はtotal_change_percentの表示丸め。PR/PF比は既存表示fieldから転記する。保存exact変化率fieldは winner_total_change_percent_exact にあるが、今回それを評価して変化率を作り直さない。

同じ政策内でp15とp16を比べたregretではなく、政策が変わった前後の最小費用比較である。候補切替だけの改善量を求めない。allowance倍率1/10は数値誤差余裕の設定で、q・回路費用の10倍指定ではない。同じ物理条件の2設定であり、分子実験2件ではない。保存方向は両者ともPF_LOWER→PF_LOWER、全100scopeの逆転0はsummaryの既存値を転記する。

物理条件：直鎖H8・1 Å・STO-3G・tol-only DF28・T=.8・P=0。PFはq4/δ=.2、PRは旧H8 union6 n32のp7/q8/S2/r4/K4/δ=.1。現行のH8独立n48・両端Pと混ぜない。旧H6はunion8 n32。

## 5. 補足・現在番号・保全

|配置|更新後の内容|
|---|---|
|14枚7／12枚6|PF総費用変化率図・H8の丸め費用表・短い口頭説明|
|B.19|変化率の定義、保存順の読み方、exact費用と保存比、旧件数図B、従来件数表と条件・限界、元PF_cost_change100／PR_PF_ratios100全形式|
|B.23|450軸の回路単価診断を維持。図Bが本編主図だった旧参照だけを訂正|
|その他の本編|1–6・8–14は本文・図・ノートを変更しない|

14枚2＋3統合と9補足移動の圧縮規則は不変。旧図Bは元のfigures/clean_q_model_update_20261021/B_price_replacement_summary100_clean.pngをB.19から参照し、ファイルを動かさない。旧図と元2図は新しい独立証拠ではない。論文化残作業B.21、旧履歴B.22、主表・他画像も保持する。

旧70は保存費用・比・候補・順位を再計算せず、そのまま転記。新30は別SOURCEの完了baselineで、一部は旧全domain winnerがnullだった履歴を持つ。旧winnerの再現と呼ばない。モデル・既存実測混在→保存直接単価という政策間の感度で、元実測を保持する分岐もある。モデル固有誤差だけを分離した反実仮想ではない。

profile/runtime差、S4仮説、初回最終gate欠測、旧profile48軸照合待ち、未観測rare event、母平均・大域最適性保証なしを維持。450軸の単価誤差率から100scopeの優劣逆転0を導かず、同じ比較集合とも扱わない。

## 6. 技術検証・性能・終了

描画プログラムは[render_saved_cost_change.py](figures/pf_total_cost_change_20261021/render_saved_cost_change.py)、座標と出力SHAは[render_receipt.json](figures/pf_total_cost_change_20261021/render_receipt.json)。新規科学コードではなく、保存JSONの値をMatplotlibへ渡す表示専用の処理。

通常の既存.venv311環境（Python3.11.0rc1、Matplotlib3.9.2）を使用。実行前に32CPU・物理62GiB／available51GiBを確認。100点の小さな描画であり、worker制限や共有BLAS設定の変更は行わない。元図のMatplotlib3.11.2とは表示環境が異なるが、科学値・軸座標は保存値に固定する。Matplotlibは書込不可の共通設定パスを使わず自動的に/tmpの描画キャッシュを作った。共有環境・依存パッケージ・他worktreeを変更しない。

検証：[文書検証JSON](2026-10-21_PR資源評価_PF総費用変化量_文書検証.json)。主系列100点と星2点の座標・条件ID・保存percent・exact文字列・旧70／新30、PNG/SVG/PDFの形式と表示、MD画像／anchor、14枚／12枚の対応、他スライド・数式・従来表・元成果物SHAを確認。PNGは目視し、SVG文字とscatter座標を原fieldへ照合する。統計量や費用を再評価するテストは実行しない。

変更は新版MD・新しい図とその描画／対応／来歴・本メモ・技術検証JSON・進捗索引に限定する。新規sampling・compile・signal・shot/資源再会計・統計再解析・再ランキング・fit・bootstrapを行わない。既存SOURCE・結果・原画像・grant・ledger・marker・STOP・主表は変更しない。著者の技術自己確認であり、詳細科学レビュー・最終evidence freezeではない。

**MANDATORY STOP。** 新しい専用branchへのlocal commitまでで終了し、今回のpushは行わない。次担当ユーザーが公開範囲を判断し、新たな明示指示がある場合に限り次担当Codexが公開する。
