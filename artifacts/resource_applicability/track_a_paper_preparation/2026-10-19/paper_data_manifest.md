# Paper Data Manifest — H4 / H6 / H8

2026-10-19 JST。状態：`TRANSCRIPTION_COMPLETE_NUMERICAL_ANALYSIS_NOT_AUTHORIZED`。

[CSV](paper_data_manifest.csv)は保存値の文字列転記とmetadata結合で作成した。新しい比・平均・SE・最小費用・順位は算出していない。`source_file`はrepository相対path、`source_record`はCSVのheaderを除く1-based record番号、JSONでは明示した配列pointer。`source_field`と公開元fileのSHA256を付す。各行のmetadataにも固定監査の元recordを付す。

| record_type | 用途 | 内容 |
|---|---|---|
| NEW_A_TOTAL_COST | 主表／補足 | 保存120比較の新PR・旧PR・PF、360行 |
| EXISTING_D_ACCOUNTING | 履歴／会計差 | 保存D120比較の旧PR・PF、240行。要約にないG/SEは未保存欄として保持 |
| NEW_A_REGISTERED_ENDPOINT | 補足データ保全 | 全新A960 endpoint、保存statusをそのまま保持 |
| H8_FIXED_CONFIRMATION | 独立固定候補確認 | 主／副評価4条件のPR・固定PF、8行 |

合計1,568索引行。これはmetadataの件数であり科学値の新規集計ではない。同じ候補が複数scope／会計に現れることは重複標本を意味しない。[主表転記](main_table_transcription.csv)は主系列45行＋H8確認8行。主系列では新旧PRを別行に置き、両者から新しいunion winnerを選ばない。

## 主panelの保存表示値

以下は結果報告64c85f7a…の表示値を直接転記した。CSVのexact fractionを今回小数化した値ではない。

| 系／cohort | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---:|---:|---:|---:|---:|
| H4 新A探索n8 |16.0220%|25.6958%|50.6900%|46.5074%|54.9402%|
| H4 旧PR n32 fixed anchor |19.3462%|48.0918%|50.0985%|43.7395%|52.6334%|
| H6 新A探索n8 |−76.8195%|15.4763%|29.2796%|30.8980%|9.9766%|
| H6 旧PR確認union n32 |−13.2901%|30.8905%|34.9653%|31.1677%|10.3257%|
| H8 新A探索n8 |3.1084%|31.8387%|29.9926%|17.9519%|3.1178%|
| H8 旧PR確認union n32 |−8.0230%|33.2373%|31.4549%|23.1282%|−50.8125%|

通常signed／倍率1／P=0。H4はold_plus_direct_subset16、H6はstrengthened_S2_S4 136、H8はstrengthened184。正はPR点費用が低い。DFはH4 legacy12、H6 tol-only19、H8 tol-only28。3系を同じ探索・DF政策と扱わず、3点からscaling lawを認定しない。

H8の別n48確認はe6eea538…の固定2候補で、保存報告の主評価はε=.05で2.935924%、ε=.0005で3.302597%。探索n8・旧n32とは合算しない。候補ID・費用・SE・固定PFはCSVと[h8_confirmation_status.md](h8_confirmation_status.md)に対応する。

## schemaと状態

`record_id`が索引の一意キー。科学上のjoin keyはsystem、DF系列、epsilon、bias、allowance、候補ID、cohort、比較scope、result commitの組である。candidate IDだけでは結合しない。

- `saved_cost_exact`／`saved_reduction_percent_exact`／`saved_SE`は原表の数値文字列。単位はcompiled RZ×解析shot、削減率欄はpercent。SEはpaired trajectory費用のMCSE。
- `null`は未保存・未確定、`NOT_APPLICABLE`は対象外。0に置換しない。空の元科学値には数値を補完しない。
- `eligibility`は保存statusの意味付けまたはNOT_REEVALUATED。H6の旧eligible:nullをFalseへ変更しない。候補の有限点とdomain完全性は別列。
- `full_domain_state`／`unresolved_ids`は保存完全性の記録。H6非相殺の旧DはAVAILABLE_SUBSET_ONLYと明示し、保存unresolved=[]を完全性の証明としない。
- `cost_class`は直接実測／validated modelを保持。PRは直接compiled標本の有限平均であり母平均実測ではない。D要約だけにGがない場合は別の費用を推定しない。
- 境界属性は固定監査の座標slice／登録q-r scheduleから転記。未注記の新A非winnerには登録表参照とNOT_PREVIOUSLY_ANNOTATEDを付す。新しい境界winnerの判定はしない。
- H8確認の境界属性は元の探索domainでの選択来歴であり、確認時の新たな探索domainではない。
- `science_SOURCE`はその結果の会計SOURCE、`acquisition_SOURCE`は新A取得または既存測定来歴の区分。旧n32やPFの原取得SOURCEが公開rowにない場合は未結合／内部証拠参照を明記し、新A取得SOURCEで埋めない。
- `alpha_total=.05, alpha_axis=.025`は主系列の保存shot会計。費用標本の有意水準・SE・H8標本数設計のα_Cとは別。追加α感度は別結果として所在のみ保全する。

公開projectionには公開済み科学表とrepo相対参照のみを使用した。privateの保存shot／運用path／grant／seed台帳／詳細hashは内部記録へ分離。private入力をpublic原本と呼ばない。
