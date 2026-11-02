# PR資源評価図セット：450軸qモデル診断の追加

2026-10-20。現在の簡潔版6図（`55b719c8e0881bda848a8ed721b0cd452267b801`）を保全し、固定された450軸診断（`7430a386bcd4dc545107e413cd022ab7cb261968`）の図E・補足E2〜E4を別版として追加した。既存6図を含む10図への入口を本書にまとめる。**ローカル専用。今回の追加packetはpushしていない。**

## 図一覧

リンクはPNG。同名のSVG・PDFも保存。既存A1〜Dは旧ディレクトリのファイルを参照し、再生成・上書きしていない。

|ID|図|用途・指標|
|---|---|---|
|A1|[H-chain ε=0.05](../2026-10-20/figures/A1_absolute_rz_eps_0p05_clean.png)|現行のshot込み絶対G_RZ。独立Pと探索A。|
|A2|[H-chain ε=0.0005](../2026-10-20/figures/A2_absolute_rz_eps_0p0005_clean.png)|A1の高精度側。|
|B|[100 scope単価置換感度](../2026-10-20/figures/B_price_replacement_summary100_clean.png)|既存の費用・順位・方向の保存集計。450軸とは別比較。|
|C1|[H4平均費用とMC標本数](../2026-10-20/figures/C1_pr_mean_cost_vs_n_clean.png)|H4固定候補の主図。|
|C2|[H4保存標準誤差](../2026-10-20/figures/C2_pr_standard_error_vs_n_clean.png)|C1の補助・補足。|
|D|[独立確認8条件](../2026-10-20/figures/D_independent_confirmation8_clean.png)|既存主表6条件を置換しない補助表。|
|E|[予測RZ対直接compile RZ](figures/E_predicted_vs_measured_clean.png)|新しい主要図。H6/H8、S2/S4、B0/B0+I、一致線、±2%帯。shotを掛けない単価比較。|
|E2|[符号付き相対誤差分布](figures/E2_relative_error_distribution_clean.png)|補足。保存済み20 binと件数・±2%線を保持。|
|E3|[q別の誤差](figures/E3_error_vs_q_clean.png)|補足。保存全点と保存記述平均線。|
|E4|[prefix別の誤差](figures/E4_error_vs_prefix_clean.png)|補足。保存全点と保存記述平均線。|

[日本語caption](captions_ja.md)・[日本語説明/制約](figure_notes_ja.md)・[旧新対応表](figure_mapping.md)・[出典とhash](provenance.json)・[スライド7〜9更新案](slides_7_9_update_proposal_ja.md)・[公開範囲](publication_scope.md)を併用する。

## 保存結果の要点と範囲

|対象|比較軸|保存平均絶対相対誤差|保存最大絶対相対誤差|
|---|---:|---:|---:|
|H6 DF19|192|0.353456%|0.805394%|
|H8 DF28|258|0.316507%|1.488276%|
|全体|450|0.332272%|1.488276%|

上表は固定`summary.json`の転記（小数6桁の表示のみ）。450/450軸で絶対相対誤差2%未満という保存診断であり、正式な全適用域の精度保証ではない。450軸は225 cos/sin候補対に対応し、450独立実験ではない。取得済み788軸のうち比較したのは450軸、残り338軸は対象外。H6 profile差、runtime同一性未証明、S4合成仮説、初回最終gate欠測を保持する。

Eはfull Hadamard-wrapperの**RZ/軸、shotなし・状態準備なし**。Bは保存analytic shotsを含む総費用政策の感度。100 scope（重複除外40条件）へ450軸を加算せず、PR/PF優劣逆転0というBの結果も変更しない。旧PR/PF候補・費用・SE・順位・正式主表を更新していない。

## 表示編集と検証

既存`display/*.svg`の文字部分だけを編集した。全ての点・path・histogram輪郭・平均線・軸・clipping・一致線・±2%帯は元SVGと完全一致する。元の描画scriptは実行せず、bin・平均・誤差・モデル予測を計算していない。CairoSVGで300 dpi PNGとフォント埋込PDFへ変換した。B0I表記を同じ方式のB0+Iへ直し、長文脚注・S4仮説の文章は日本語notesへ移した。Eには短いfilled/openの記号キーを追加した。

[保存値照合](verification/saved_value_check.json)、[SVG幾何同一性/ラベル移動](verification/svg_equivalence.json)、[総合検証](verification.json)、[表示確認](visual_review.json)、[出力hash](manifest.json)を保存。技術・表示確認はCodex本人によるもので、独立科学レビューではない。

```bash
<CairoSVG 2.8.2環境のpython3> scripts/extend_q_model_figures_20261020.py
# 表示確認・日本語文書を整えた後
python3 scripts/extend_q_model_figures_20261020.py --finalize
python3 scripts/extend_q_model_figures_20261020.py --check
```

[renderer依存一覧](requirements-render.txt)と[環境・資源観測](verification/render_environment.json)も同梱。描画専用の一時venvだけに依存を用意し、共有環境・既存venv・OS・他ユーザーのプロセスは変更していない。CPU/RAM/worker数への独自上限は設けていない。

既存スライドMDは変更せず、ユーザーが追加した[原本snapshot](inputs/slide_plan_original.md)を参照して14枚基本・12枚圧縮を維持する2配置案を別保存した。詳細科学レビュー・最終evidence freezeは未開始。[STOP](STOP.json)。公開判断が必要な場合の担当はユーザー。追加作業を自動開始しない。
