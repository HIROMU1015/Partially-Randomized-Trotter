# GPT科学レビュー：H4 B0+I全面実測後のv0.4原稿

2026-10-14。Codex側の保存値統合を完了したhandoff。科学レビューそのものは未実施、追加計算のgrantを含まない。

## 一つの入口と確認順

1. [通し原稿v0.4 draft](track_a_resource_study_geometry_native_v0_4.md)。
2. [主張候補と証拠の対応](track_a_h4_b0i_v0_4_claim_evidence.md)、[v0.3からの変更](track_a_h4_b0i_v0_4_changes.md)。
3. [技術証拠・検証入口](../research/track_a_h4_b0i_manuscript_integration_20261014.md)、[図・CSV・provenance bundle](../../artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/README.md)。
4. [H4/H6/H8の条件対応](../research/track_a_resource_comparison_conditions_20261014_v0_4.md)、bundleの固定参照。

## 事実として整備したもの

結果branch `track-a-h4-b0i-map-parallel-run02-20261014` のorigin HEADは指定結果commit `98cdb35cd08469714f20d6cc3b982d9c80908f46` と一致。
実行SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、data identity SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
科学scopeはH4 linear/STO-3G/DF12、6距離、T=.8、prefix3/4/5/6/9、q1/2/4/8、δ=.1〜.8、S2/ordinary/structured/fixed compiler/P=0。

- 120構成・240 wrappers COMPLETE、234 new compile＋6 provenance/numeric照合再利用。
- 全240軸のRZ/CX数、各depth、sizeの旧B0差0。今回は実測であり、旧Δ=0仮定の転用ではない。
- 元B0/B1/B2/B3集合だけなら全1,812点B2点最小。B0+I追加後は397点B0+I、1,415点B2、ties0。
- ε=.005では全6距離B2が低い。ε=.05では.70/.80/1.40 ÅでB0+I、他3距離B2。ε=.1では.70/1.40 ÅでB0+I。
- method領域12、点最良template領域19。0.80/0.90 Åは中間精度だけB0+Iが低い。隣接grid bracketを収録し、連続境界を認証しない。
- 397点中374点は保存B2点推定−2SEも下回り、23点はその記述的幅内。B2側も1,392/23に分かれる。
- 1,812 factor行、matched120構成、固定display216行でbias/headroom/shot/軸別RZ/有効RZを分離。
- .70/1.40 Å、ε=.05のB0+IはB2よりshotが多くても回路が軽く低費用。.80 Åはshot減少が重い回路費用を補う。同一構成のB0→I位相効果と別family再選択の会計を分けた。
- 実行完了/cleanup/owned14残存0は固定native receiptの証拠を再利用。今回live runを操作・再監視していない。

## 既存の決定

直前レビューはB0+Iを正式な追加評価対象に採用し、全120構成/240軸を全面地図の標準完成scopeとした。
旧B0の定義をバグと認定せず保持する、truthにfitしたscalarへ変更しない、追加H4 samplingを一律必須にしない、H8を変更しない方針を継承する。
今回の240軸取得は利用者の別実行承認後に完了しており、6件grantを自動拡張したものではない。
本taskは結果・原稿の整備のみで、旧grant再利用・新launch・新baseline追加は0。

## 今回まとめて判断を求める科学的論点

|判断事項|既存事実|未決定事項・原稿の候補|
|---|---|---|
|主要claimの更新|元集合の全面B2点最小と、拡張集合397/1415が両立|v0.4 Abstract/第1節/Conclusionの限定表現を採用するか。「全点でPR優位」は拡張集合に使えない|
|PRが低い領域の表現|1,415保存点、ε=.005の全6距離、.80/.90の再入領域|grid点の記述と物理的な適用条件をどこまで結び付けるか。連続精度・外挿を追加主張しない|
|強い決定論的対照の意味|known scalarだけを保持、全metrics差0、shot会計に影響|資源利益の帰属を切り分けたbaseline-sensitive case studyを中心にするか|
|統計的不確かさ|同じ32本で選抜と推定、374点の記述的2SE条件|この有限標本scopeで論文を閉じるか、境界に限定した独立確認を必要とするか|
|希少次数の追加確認|H4履歴NOT_SAVED、172候補法則未解決、母平均寄与未保証|確認の要否と最小scope。Codexは新統計保証や追加samplingを決めない|
|H4/H6/H8の統合|異なるDF/control/lowering/u/selection、H8 core最終結果は未収録|独立事例としてまとめるか。共通scaling・削減率の直接比較は支持されない|
|論文化の着地点|原稿・4主図＋1補助図・再現CSVを用意した|独立paper/technical note/否定的診断、題名、主張の強さ、未取得H8の必要性、追加検証の要否|

## 未取得・未検証

B2内の細かなr/K順位の独立確認、H4の新しい独立cost標本、Taylor次数/component観測履歴、172丸め候補の正確な法則確率、希少event母平均寄与の保証はない。
1,812点は独立標本でなく、εはcomplex signal、P=0。±2SEは正式confidence intervalではない。
他compiler、旧host output完全同一性、H6/H8への移送、state preparation、noise、energy/QPE/RPE全体、fault-tolerant総資源への保証はない。
H6は既存固定参照の独立確認と局所pilotとして継承した。H8 core最終shot結果、対照scalarの最新処理、live状態を今回取得・判定していない。
今回prior-artの新しい網羅的検索や投稿先適合性reviewはしていない。

## STOPと次段

本資料で新規性・研究価値・主要claim・追加計算の必要性をCodexが確定しない。
科学レビューは上記論点をまとめて判断し、採用表現と論文のscopeを示してほしい。
追加計算が必要と判断された場合も、目的・候補・統計設計・budget・source・新grant・利用者承認を別に定め、今回の完了を自動認可へ使わない。
原稿/図表/公開監査の整備後はmandatory STOP。H8や共有環境への変更は0。
