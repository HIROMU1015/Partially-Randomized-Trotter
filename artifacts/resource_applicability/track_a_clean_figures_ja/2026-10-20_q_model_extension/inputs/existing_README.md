# PR資源評価：簡潔版図と日本語説明（2026-10-20）

固定commit `0566c3bb6c9c1a2d634519db6aabc5d2ef412798`の優先6図から文章説明・注意段落・脚注を外し、日本語説明を同じディレクトリへ分離した。元図は上書きしていない。変更は文章・短い見出し・余白だけで、数値・点・線・誤差棒・帯・棒・軸範囲・軸スケール・凡例の意味は維持している。

## 成果物

各PNGと同名のSVG・PDFも保存。PNGは300 dpi、SVGは文字を保持、PDFはフォント埋込。論文組版にはPDF/SVG、スライド・GitHubプレビューにはPNGを推奨する。

|ID|簡潔版PNG|用途|
|---|---|---|
|A1|[絶対RZ費用 ε=0.05](figures/A1_absolute_rz_eps_0p05_clean.png)|独立Pと探索Aを分けた本編候補。H2追加Pを含む。|
|A2|[絶対RZ費用 ε=0.0005](figures/A2_absolute_rz_eps_0p0005_clean.png)|A1のもう一方の要求精度。|
|B|[100 scope単価置換感度](figures/B_price_replacement_summary100_clean.png)|変更件数・保存点方向・H8 prefix変更の本編要約。|
|C1|[H4平均PR費用](figures/C1_pr_mean_cost_vs_n_clean.png)|MC標本数依存性の主図候補。|
|C2|[H4標準誤差](figures/C2_pr_standard_error_vs_n_clean.png)|C1に対応する補助・補足図。|
|D|[独立確認8条件](figures/D_independent_confirmation8_clean.png)|発表・論文候補の補助表。正式主表の置換ではない。|

- [captions_ja.md](captions_ja.md)：各図の短い日本語キャプション。
- [figure_notes_ja.md](figure_notes_ja.md)：結果・比較・読み方・条件・制約・SOURCE・変更点。
- [figure_mapping.md](figure_mapping.md)：旧図から新図への対応、残す短いラベルと移した説明。
- [provenance.json](provenance.json)：固定入力、元図SHA256、科学SOURCE一覧、使用データ。
- [remaining_candidates_ja.md](remaining_candidates_ja.md)：今回未対応の主要既存図。
- [verification.json](verification.json)、[表示確認](visual_review.json)、[STOP.json](STOP.json)：検証・終了記録。

図だけでは図外へ移した制約は伝わらないため、掲載時は対応するcaptionとnotesの条件・限界を併用する。図中は英語・記号のみで、日本語はMarkdownに限定する。

## 移した内容と保持した内容

H8の公開receipt/profile未再検証、入力生成の違い、離散chain比較と母平均主張の限界、100 scopeの重複と単価置換の解釈、profile/runtime差・S4仮説・初回最終gate欠測、bootstrap帯の条件付き解釈、未観測rare event、費用に含まれない状態準備/FTQC、H2と既存主表の関係を日本語notesへ移した。

軸・単位・目盛り・PR/PF凡例・n/N・精度・候補ID・短い条件表示・保存判定は維持。長いタイトルは名詞句へ短縮し、脚注本文を移したH8†およびDのH2*は記号を外した。Bの`PF remains lower-cost`はカテゴリ表示`PF lower`へ短縮。ページを低くして空いた脚注領域を詰めたがデータ範囲は変更していない。

## 実装と検証

新規scriptは`scripts/render_clean_pr_figures_ja_20261020.py`。元の表示scriptを固定バイト列と照合して読み、表示関数のみを同じ引数で実行した。旧`main`・`prepare`・科学moduleは実行しない。保存データ9ファイルを[data/](data/)へバイト一致で写した。Dの元CSVのCRLFも保持する。

同じ描画オブジェクトを削除前後で記録し、線/点の座標・マーカー・線種・色、誤差棒、分位帯、棒、表の値・保存判定、軸範囲/scale/目盛り、凡例を一致照合した。脚注参照の`*`/`†`だけは照合時に除いた。ページ座標・文章・タイトルは表示変更として別記録にした。

[変更前オブジェクト](verification/scenes_before.json)と[変更後](verification/scenes_after.json)は同一。[外した英文・短縮ラベル](verification/text_moves.json)と[残した全ラベル](verification/retained_labels.json)も保存した。PNGは6枚をCodex本人が目視確認し、SVGはXML/text、PDFはheader/EOFを検証。これは技術・表示確認であり独立科学レビューではない。GitHub画面のブラウザ撮影を行ったとの主張はしない。

```bash
<既存のmatplotlib環境のpython3> scripts/render_clean_pr_figures_ja_20261020.py
# PNGを確認してvisual_review.jsonを記録後、ハッシュと検証記録を固定する。
python3 scripts/render_clean_pr_figures_ja_20261020.py --finalize
python3 scripts/render_clean_pr_figures_ja_20261020.py --check
```

既存のプロジェクト専用`.plot-venv`（matplotlib 3.9.4）を利用し、共有環境を変更していない。新規sampling・compile・bootstrap・fit・shot/費用再会計・PR/PF再比較・再ランキングは0。既存SOURCE・raw結果・主表・Figure 2・grant・ledger・marker・STOPを保全した。詳細科学レビュー・最終evidence freezeは開始していない。

## 公開範囲

新しい図・日本語説明・描画データの写し・出典・技術検証・scriptだけが今回の公開対象。親commitは公開済み`0566c3bb...`、専用branchは`track-a-clean-figure-notes-ja-20261020`。旧公開図の差替え、実験状態の改訂、主表の正式変更、未公開raw/grant/私的receiptの追加は含まない。

今回の作業は終了。未対応候補は一覧化だけを行い、自動で追加修正しない。次の作業なし。MANDATORY STOP。
