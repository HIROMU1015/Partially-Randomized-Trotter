# PR資源評価：発表・論文候補図（2026-10-20）

固定済み保存結果の転記・描画だけでA〜Dの6図を別版として作成した。新規科学計算、shot再計算、PF再選択、PR/PF再比較、再ランキング、bootstrap、fitは行っていない。既存図、主表6条件、Figure 2、証拠台帳、SOURCE、結果、grant、ledger、marker、STOPは変更していない。

## 図と推奨用途

各リンクのPNGと同名のSVG・PDFを保存。PNGは300 dpi、SVGは編集可能な文字、PDFは埋込フォントを使用する。

|図|PNG|推奨用途・既存版との差分|
|---|---|---|
|A1|[ε=0.05](A_absolute/figures/absolute_rz_eps_0p05.png)|スライド・論文候補の絶対費用図。左PにH2 n=80を追加、左の既存6条件と右の探索8条件の保存値は不変。|
|A2|[ε=0.0005](A_absolute/figures/absolute_rz_eps_0p0005.png)|同上。左PにH2 n=12を追加。|
|B|[100 scope置換感度要約](figures/B_price_replacement_summary100.png)|本編用。保存集計件数を横棒、PR/PF方向を数字、H8 prefix変更を小パネルに再構成。旧詳細図は補足用に保全。|
|C1|[H4平均費用とMC標本数](figures/C_pr_mean_cost_vs_n_presentation.png)|主図候補。両精度を横並びにし、保存bootstrap帯・平均・累積平均・全保存平均の意味を区別。|
|C2|[H4標準誤差とMC標本数](figures/C_pr_standard_error_vs_n_supplement.png)|補助・補足図候補。保存済みSE列の表示のみ。|
|D|[独立確認8条件](figures/D_independent_confirmation8_auxiliary.png)|発表・論文候補の補助表。H2追加2条件と既存主表系列6条件を区別。正式主表の差替えではない。|

[caption草案](captions.md)、[入力SOURCE一覧](input_sources.json)、[公開範囲](publication_scope.md)、[構造・保存値検証](verification.json)、[表示検証](visual_review.json)、[出力ハッシュ](manifest.json)、[STOP記録](STOP.json)を同梱。

## 入力と対応データ

元のCSV/JSON/文書を`git show <固定commit>:<path>`で読み、バイト一致の写しを`inputs/`へ保存した。各出典のcommit・元path・SHA256・bytesは[input_sources.json](input_sources.json)に記録。入力コミットをmergeしたり実験を再実行したりしない。

|対象|固定commit|主な保存入力・描画データ|
|---|---|---|
|A既存v2|`c5ad79457aea87cc36b38022a2821ebb42eb276d`|[既存比較](inputs/A_comparisons.csv)、[既存点](inputs/A_plot_points.csv)、[採否](inputs/A_selection_decisions.csv)、[旧renderer](inputs/A_renderer.py)|
|A/D H2 P|`f2d1a86c431f835db700f348ce618d8604e7ee23`|[confirmation_result](inputs/H2_P_confirmation.json)|
|A/D統合|`8fbe7d77556c2353be837dbd9ed0fddc02ee81c4`|[H2採用点](inputs/H2_adopted_points.csv)、[8条件](inputs/H2468_comparison_separate.csv)|
|D既存主表|`1923fce8f7461513d1bc47b5061e87cd70917e87`|[既存主表6条件の写し](inputs/H468_main_table.csv)|
|B|`3983fe4f1ede003953735e757d2c098c12b4644f`|[保存summary](inputs/B_summary.json)、[figure_data](inputs/B_figure_data.json)、[100 scope](inputs/B_combined_selection100.csv)|
|C|`295b93d50662a0f8af3ec469df43099a3f04af60`|[bootstrap保存summary](inputs/C_bootstrap_summary.csv)、[全保存参照](inputs/C_full_sample_reference.csv)|

- A：[16比較](data/A_comparisons_with_H2_P.csv)、[32描画点](data/A_plot_points_with_H2_P.csv)。有理数tokenを保持し、float化・3桁注記だけを表示用に行う。H2のG・SE・n・候補ID・保存判定をP結果と照合。H4/H6/H8 Pの6条件、探索Aの8条件は元レコードおよび描画点と全フィールド一致。
- B：[6種の変更件数](data/B_change_counts.csv)は保存summaryの`previous70`、`new30`、`all100`から転記。[H8変更例2件](data/B_H8_prefix_examples.csv)は保存済み`best_candidate_changed=true`行の抽出。費用の再比較や新しい順位付けは行わない。
- C：[描画座標と元token](data/C_plot_coordinates.csv)。18個の保存n行の7列と全保存参照2点を記録。`q025/q975`、`bootstrap_mean`、`cumulative_mean`、`bootstrap_SE`、`conditional_empirical_SE`、`cumulative_SE`をそのまま使う。参照SE曲線も保存列であり新計算ではない。
- D：[8条件表](data/D_independent_confirmation8.csv)は統合CSVのバイト一致の写し。低費用側も保存済み`lower_saved_point_cost`列を使用。H4/H6/H8部分が既存主表CSVと完全一致することを検証。

## 採用範囲と解釈

Aはε=0.05/0.0005だけを更新した。左は独立PのH2追加2条件＋H4/H6/H8既存6条件、右は探索AのH2/H4/H6/H8各n=8。元v2の中間3精度は今回の対象外とし、旧版のまま保存する。H2 Pの候補・cohortは統合CSVの登録値を使い、旧探索cohortとpoolしない。H2の正式主表入りは決定していない。H8の公開PR実行receipt未再検証・元profile bytes未検証を維持する。離散的なchain比較でありサイズスケーリングの主張はしない。

Bは旧70件の保存結果の転記＋新30件の保存結果。重複を除くと40個のsystem/bias/epsilon/allowance条件であり、100独立実験ではない。最良PF候補2、q 0、PF次数0、最良PF総費用50、候補順位50、PR/PF優劣逆転0件を転記し、両政策とも保存点のPR低費用90・PF低費用10件と示す。2件のH8 prefix16→15はともに旧70件由来、q=4・4次PF・PF低費用は維持。混在単価政策から直接実測単価への置換感度であり、モデル固有誤差だけを分離した検証ではない。profile/runtime差、S4仮説、初回最終gate欠測、old-profile軸未完を解消していない。

Cは固定H4候補のMC平均推定値の揺れの表示であり、横軸nはpaired trajectory標本数。必要shot数ではなく、n実行の集計費用やサーバーCPU時間でもない。保存済みB=2000/nのbootstrap分位帯は保存経験分布に条件付きで、母平均CIではない。累積系列は1本の入れ子の保存系列で、全保存平均も母平均の真値ではない。未観測rare eventや必要nの保証を与えない。

全図のRZ費用は今回定義のshot込み費用。状態準備・FTQC全体費用を含まない。独立確認は固定された候補・有限標本・登録した対照の範囲に限定され、一般的なPR優位やglobal optimumを主張しない。詳細科学レビュー・最終evidence freezeは開始していない。

## 再描画と検証

新規scriptは`scripts/render_pr_presentation_updates_20261020.py`。Aは固定v2 scriptの`render`関数だけを取り出し、H2追加の表示文言を変更して流用した。Cは固定rendererの表示構成を踏襲し、保存値のみを渡す。科学moduleや旧scriptの`main`は実行しない。

作図は既存プロジェクト専用`.plot-venv`のmatplotlib 3.9.4で実行。共有パッケージや環境設定は変更していない。

```bash
<matplotlibを含むpython3> scripts/render_pr_presentation_updates_20261020.py
python3 scripts/render_pr_presentation_updates_20261020.py --check
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check c5ad79457aea87cc36b38022a2821ebb42eb276d
```

`--check`は固定入力とのバイト照合、既存14比較の保持、全描画token・形式・サイズ・ハッシュ・リンク・変更pathを検証する読取専用処理。保存監査はこのworktreeのbaseとの差分だけを対象とし、他worktreeや他ユーザーの変更は制限しない。表示検証はCodex本人による6枚のPNG確認であり、独立した科学レビューではない。

一部の固定入力CSVはCRLF改行を含むため、そのバイト列を写し・D表で保持した。空白検査はコマンド単位の`cr-at-eol`指定で行い、保存入力やGit設定の改変は行わない。

次の作業なし。MANDATORY STOP。
