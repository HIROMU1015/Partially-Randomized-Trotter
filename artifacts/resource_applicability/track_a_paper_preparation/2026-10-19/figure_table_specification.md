# Figure / Table Specification

2026-10-19 JST。完成した論文図はまだ生成していない。全図でH4/H6/H8を同じ階層の対象に含める。

## Figure 1 — 探索domainと資源比較

3列をH4/DF12、H6/DF19、H8/DF28とし、横軸は主要5ε（log表示、数値順は表示規則であって費用順位ではない）、縦軸は**保存された**PR削減率。0の水平線。旧n32と新A探索n8を別記号／別系列、弱い決定論対照と強化対照をfacetまたは線種で区別する。新旧から低い方を選んだ曲線は作らない。

PF選択点は直接実測を塗りつぶし、modelを中抜き、cohortは形状で表す。色だけへ依存しない。各点のcandidate ID、元行、DF政策、比較scope、p/q/r/K、外側次数、prefix/K/q-r schedule端を補足表またはcaption索引で追跡できるようにする。singletonは内部と描かない。

H4は直接測定subset16であって登録96の最良ではないとpanel内に表示。H6は中間点PFにmodel残存。H8 n48の固定2点は独立確認insetへ置き、n8の点を置換しない。固定2対照の比較を184全実測の確認と呼ばない。

入力：manifestのNEW_A_TOTAL_COST行、該当する旧対照scopeと強化scope、H8_FIXED_CONFIRMATION primary行。削減率exact fractionの数値は保存済み。現在の文書表は元報告の丸め表示を転記する。将来のfraction→plot座標変換・表示丸めはレンダリングの処理として明示し、比そのものを再計算しない。

状態：`SPEC_READY_SAVED_VALUE_TRANSCRIPTION`。図描画は未実施。新しい費用比・union winnerは不要。

## Figure 2 — 回路単価とshot負担

3列の系別panel。各系の主要5ε×旧PR／新Aを補足表に全保全する。主図を抜粋する場合の規則は、各系のε=.05/.005/.0005をあらかじめ固定する。結果の大きさで代表点を選ばない。H4も同じ構成とする。

上段は軸別R_C,a、中央は軸別R_N,a、下段はR_Gと正確なPF費用重み分解ΣwR_NR_C。R_N×R_Ceffの別表示ではCeffのshot依存を明記。各軸・model・cohortを保持し、単純平均回路比の積は使わない。保存bias/headroom/normalizationが揃う場合に限って補足の軸別説明panelを加える。

正式入力はMechanism Contractの将来mechanism_axes.csv／mechanism_totals.csv。レビュー概算、旧設計の試算、別系列の数値は流用しない。欠測は穴／状態注記にし、0の棒を描かない。母平均CIや因果寄与の図にしない。

状態：**PENDING_NUMERICAL_AUTHORIZATION**。レイアウトとschemaだけで、定量データ・図は未生成。

## Figure 3 — bias会計への依存性

H4/H6/H8の3列。通常signedと経験的非相殺の保存比較をpaired facetで示す。n32、新A n8、H8 n48は別系列。allowance factor1を主、factor10は補足し、保存shot欠測を明示する。

H6はfull registered domainをUNRESOLVED表示とし、旧Dの有限subset値は別facet・AVAILABLE_SUBSET_ONLY注記で引用する。旧Dのunresolved=[]で欠測を覆い隠さない。H8 n48 p8副評価は固定PF不適格なので比較null。他PFを選び直さない。

二つの会計の差をCIの帯・誤差棒にしない。値が両方保存されていても「差」「倍率」の新列は今回計算しない。

状態：`SPEC_READY_SAVED_VALUE_TRANSCRIPTION`、描画未実施。

## Main Table / Supplement

[main_table_transcription.csv](main_table_transcription.csv)は主表の原表。系・ε・新A/旧PR/PF、候補・費用exact・保存SE・保存削減率・標本役割・直接/model・欠測・出典を別行にした。H8 n48は独立確認block。H4は補助へ格下げしない。3系間のDF政策・比較集合差を脚注で保持する。

S1：全新A960 endpoint。S2：旧Dの120比較とH6 subset限定。S3：全採用点の境界属性と登録集合。S4：契約承認後の機構表全条件。S5：H8固定確認の主副4条件。H6高次PF確認は別系列の掲載候補として所在だけを記し、今回の主表へ数値を混ぜない。

## 作成可能範囲と承認境界

| 作業 | 今回 |
|---|---|
| 既存G/削減率/SE/候補ID/statusの転記 | 実施済み |
| 図のcaption、panel、記号、入力／出力仕様 | 実施済み |
| 論文用の完成figure描画 | 未実施 |
| 回路単価比・shot比・G比・normalization/headroom分解 | 別の保存値解析承認が必要 |
| 新G平均／SEの再構成・保存値との算術一致 | 別の保存値解析承認が必要 |
| winner再選択・順位・cohort pooling・sampling/compile | 今回禁止、上記機構解析契約の承認にも含めない |
