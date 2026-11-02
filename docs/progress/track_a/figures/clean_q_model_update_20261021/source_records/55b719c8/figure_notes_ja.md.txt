# 図の日本語説明・条件・注意点

元図内の文章を図外へ移した説明であり、新しい科学的結論・採否・数値再解析を追加しない。[対応表](figure_mapping.md)、[固定入力・元図hash](provenance.json)、[短いcaption](captions_ja.md)を併用する。

## 共通条件と来歴

直接の入力commitは`0566c3bb6c9c1a2d634519db6aabc5d2ef412798`。費用は保存された解析shot数×compiled RZに基づく今回定義のRZ費用で、状態準備・FTQC全体コストを含まない。Linear H-chain、隣接1 Å、STO-3G、T=0.8、P=0。A/C/Dは通常signed bias・allowance factor 1。Bは複数bias・精度・allowance scopeの保存集計で、A/C/Dと同じ標本集合ではない。

A/Dの独立P条件を下表に示す。候補ID・cohort・exact G・保存SE・判定・SOURCEは[8条件CSV](data/D_independent_confirmation8.csv)、探索Aとの区分は[A比較CSV](data/A_comparisons_with_H2_P.csv)に記録。入力生成規則・solver・正規化の同一性を追加認定しない。

|系・DF rank|ε|PRのL_D|δ|独立Pのn|
|---|---|---|---|---|
|H2 / DF3|0.05|1|0.8|80|
|H2 / DF3|0.0005|1|0.2|12|
|H4 / tol-only DF10|0.05|3|0.8|1108|
|H4 / tol-only DF10|0.0005|2|0.1|78|
|H6 / DF19|0.05|4|0.4|48|
|H6 / DF19|0.0005|4|0.1|104|
|H8 / DF28|0.05|8|0.4|48|
|H8 / DF28|0.0005|7|0.05|48|

## A1

**タイトル：H-chainの絶対RZ費用、ε=0.05。**

- 結果・比較：横軸H2/H4/H6/H8、縦軸shot込み絶対G_RZの対数軸。左の独立Pと右の探索Aは別cohort。左H2は追加P n80、右は各選択候補n8。元の保存点・誤差棒・線を使用する。
- 読み方：青丸はPR保存平均、棒は保存paired ±1 SE、橙四角/破線はdeterministic PF。独立PではH4/H6がPF低費用、H2/H8がPR低費用という保存点方向を維持。
- 解釈上の注意：±1 SEはCIではない。線は離散chain間の視認性のためであり、厳密なサイズスケーリング則ではない。有限標本の点結果で、PR母平均・未観測高費用event・global optimumを保証しない。
- 制約：H8のfull public PR execution receiptと元profile bytesは未再検証。入力生成手順は系間で異なる。独立Pの固定PF対照は直接実測区分、探索H6/H8の登録PF候補領域にはmodel依存が残る。H2は正式主表未追加、旧探索cohortとpoolしない。
- 入力SOURCE：直接`0566c3bb...`のA比較/点CSV。上流v2 `c5ad79457aea87cc36b38022a2821ebb42eb276d`、H2 P `f2d1a86c431f835db700f348ce618d8604e7ee23`、統合`8fbe7d77556c2353be837dbd9ed0fddc02ee81c4`。科学SOURCEの完全path/hashはprovenanceに継承。
- 元図との差：費用定義、SE、H8監査、入力差、線/母平均の限定、PF model、別cohortの長文を本節へ移した。条件を短縮しH8†を外した。数値・軸範囲は不変。

## A2

**タイトル：H-chainの絶対RZ費用、ε=0.0005。**

- 結果・比較：A1と同じ配置・定義の高精度側。H2追加Pはn12、H4/H6/H8は既存独立P、右は別探索A n8。L_D・δ・nは共通表の該当行。
- 読み方：保存点では独立Pの4系ともPR低費用。青丸/棒、橙四角/破線の意味はA1と同じ。
- 解釈上の注意：固定候補・登録対照・有限標本の点方向であり、全精度や一般的なPR優位性へ外挿しない。SEはCIではなく、線からサイズスケーリングを主張しない。
- 制約：H8公開receipt/profile未再検証、入力生成差、探索H6/H8 PF領域のmodel依存、H2主表未追加・cohort非poolingを維持。母平均・未観測rare event・global optimumの保証なし。
- 入力SOURCE：直接`0566c3bb...`のA比較/点CSV。上流`c5ad794...`、`f2d1a86...`、`8fbe7d7...`の同じ固定資料。完全候補・cohortはCSVを参照。
- 元図との差：A1と同種の長文・脚注を本節/A1へ移し、短い軸・凡例・条件・nだけを図内に残した。数値・誤差棒・軸範囲/scaleは不変。

## B

**タイトル：混在単価政策から直接実測単価への置換感度。**

- 結果・比較：保存summaryでは最良PF候補変更2/100、q変更0/100、PF次数変更0/100、最良PF総費用変更50/100、候補順位変更50/100、PR/PF優劣逆転0/100。両政策ともPR低費用90件、PF低費用10件。再比較・再ランキングではなく保存集計の転記。
- 読み方：棒の青は旧70件の保存結果、緑は新30件の保存Stage B結果。100 scopeは重複除外40個のsystem/bias/epsilon/allowance条件で、100独立実験ではない。保存PRはH6 union8/H8 union6のn32に限定。
- H8例：旧70件のnoncancellation、ε=0.0005、strengthened184、allowance factor 1/10の2 scopeでprefix16→15。q=4・4次PF・PF低費用は維持。図中の`PF lower`はこの保存カテゴリ。
- 解釈上の注意：直接実測単価への置換で保存点方向は変わらず、一部の候補順位・最良PF候補は変わった。混在単価→直接実測単価の置換感度であり、モデル固有誤差だけを分離した検証ではない。
- 制約：profile/runtime差、S4仮説、初回最終gate欠測、old-profile軸未完を解消しない。固定候補領域と有限標本の診断で、母平均・未観測event・global optimumの保証なし。
- 入力SOURCE：直接`0566c3bb...`の[B変更件数](data/B_change_counts.csv)、[H8例](data/B_H8_prefix_examples.csv)、B_summary.json。上流`3983fe4f1ede003953735e757d2c098c12b4644f`のsummary/figure_data/combined_selection100等。
- 元図との差：長いタイトル、同じ方向との説明、H8条件段落、下部の解釈・来歴・制約を本節へ移した。q=4/PF4等の短い条件、90/10・件数・prefix数値は保持。棒の内訳・長さ・軸範囲は不変。

## C1

**タイトル：H4 PR平均費用推定値とMC標本数。**

- 結果・比較：ε=0.05/0.0005の保存bootstrap分位帯・平均、累積MC平均、全保存平均を別パネルに表示。固定H4/tol-only DF10、L_D/δは共通表。候補はB2_p3_q1_2nd_r4_K2（全保存N1108）、B2_p2_q8_2nd_r2_K4（N78）。
- 読み方：青帯は保存bootstrap平均の2.5–97.5分位、青丸は保存bootstrap平均の平均、緑四角は保存累積MC平均、灰点線は全保存PR平均。nはpaired trajectory標本数。費用はfull deterministic prefix＋stochastic RTE tail＋Hadamard wrapperを含む。
- 解釈上の注意：帯は保存経験分布に条件付きで、母平均CIではない。保存B=2000 resamples/nを描いただけで新規bootstrap・fit・必要n計算はない。全保存平均も母平均の真値ではない。
- 制約：累積系列は1本の入れ子の保存系列。未観測rare event費用は不明。n回実行の累積費用・CPU時間を示さず、状態準備・FTQC全体費用を含まない。母平均や必要nの保証なし。
- 入力SOURCE：直接`0566c3bb...`のC_bootstrap_summary/C_full_sample_reference、上流`295b93d50662a0f8af3ec469df43099a3f04af60`。元tokenは[C描画座標](data/C_plot_coordinates.csv)にも保存。
- 元図との差：帯の条件付き解釈、保存resample数、入れ子系列、費用定義、rare event/FTQCの長文を本節へ移した。短いタイトル・条件とし、全n点・帯・参照線・軸範囲/scale・凡例は保持。

## C2

**タイトル：H4 PR平均費用の保存標準誤差。**

- 結果・比較：C1と同じ2固定候補で、保存bootstrap平均のSD、保存empirical SD/√n列、保存累積paired SEを比較。条件・L_D・δ・N・候補/cohortはC1と同じ。
- 読み方：青丸実線は保存bootstrap平均のSD、灰破線は保存列の参照曲線、緑四角点線は保存累積paired SE。n軸とSE対数軸を維持。
- 解釈上の注意：参照線を今回√n計算で生成したわけではない。新しいSE計算・bootstrap・fit・必要n推定はない。平均費用推定の揺れであり、CPU時間やn回実行の集計費用ではない。
- 制約：保存経験分布への条件付き、1本の入れ子系列、未観測rare event不明、母平均保証なし、状態準備/FTQC除外はC1と同じ。
- 入力SOURCE：直接`0566c3bb...`、上流`295b93d...`のC1と同じ保存summary/reference。
- 元図との差：参照SEは保存列、新計算なし、入れ子系列と費用/rare eventの注意を本節へ移した。SE値・軸・凡例は不変。

## D

**タイトル：独立確認8条件の補助表示。**

- 結果・比較：H2/H4/H6/H8を行、ε=0.05/0.0005を列にし、保存低費用側・判定・nを表示。共通条件表と[8条件CSV](data/D_independent_confirmation8.csv)を参照。色はPR低費用/PF低費用のカテゴリ。
- 読み方：H2の2条件は追加独立P、H4/H6/H8は既存主表系列6条件。REPRODUCEDは事前指定方向の再現で、常にPR低費用を意味しない。H6 ε=0.05はPF低費用・REPRODUCED、H4 ε=0.05はPF低費用・NOT_REPRODUCED、H8はPOINT_REPRODUCED。
- 解釈上の注意：発表・論文候補の補助表示。正式主表6条件を8条件へ差し替えておらず、H2正式主表入りの決定を含まない。
- 制約：H8のfull public PR execution receipt/profileは未再検証。固定有限標本点で、母平均・未観測event・全PF候補への優位を保証しない。入力生成差、状態準備/FTQC除外も維持。
- 入力SOURCE：直接`0566c3bb...`の8条件CSV。上流統合`8fbe7d77556c2353be837dbd9ed0fddc02ee81c4`、既存主表`1923fce8f7461513d1bc47b5061e87cd70917e87`、H2 P `f2d1a86...`。元成果物の主表照合を継承し、今回は同じCSVをバイト一致で転記。
- 元図との差：正式主表未差替えの副題、H2追加・判定解釈・H8証拠・有限標本・条件の脚注を本節へ移した。脚注記号*と†は外し、8セルの内容・色は保持。

## 除去英文の対応記録

削除・短縮した原文、図ID、残した英語ラベル、本書の対応節を[text_moves.json](verification/text_moves.json)に記録。図内に残った文字列は[retained_labels.json](verification/retained_labels.json)に保存。科学的制約を撤回したのではなく、日本語説明として図外へ移した。
