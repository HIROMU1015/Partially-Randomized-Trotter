# H4〜H8：比較可能性の技術監査 v1

2026-10-17。新しい科学採否ではなく、固定証拠から比較範囲を分類した。`MATCHED_CONDITIONS`は指定した単価置換比較にだけ使う。異なる系の物理入力が同一だという意味ではない。

|ID|比較|分類|一致する条件|相違・未確認|許容する記述・出典|
|---|---|---|---|---|---|
|C01|H4 geometry内の距離・精度|LIMITED_COMPARISON|固定rank12/S2/ordinary/structured/compiler政策・同じ保存標本によるshot規則|距離でHamiltonian/stateが変化。精度点は同じ標本を再利用。B0I追加前後のdomainも別|距離・精度依存の保存点比較。1812独立実験、連続境界や普遍的勝率にしない（E01;E02;E03;E28）|
|C02|H4 geometry対1Å native anchor|DIFFERENT_SCIENTIFIC_CONTRACT|H4/STO-3G/legacyrank12/T.8/P0/RZ|geometryの6距離に1Åなし。ordinary対symmetric_directional、structured reuse対native raw、S2対fullS4、候補218対16、u無し対経験的u|別caseとして併記。費用単価・削減率を移送しない（E01;E03;E04;E05）|
|C03|H4固定anchor対H6/H8 native|LIMITED_COMPARISON|1Å/STO-3G/T.8/native symmetric-directional/full measured wrapper/P0/Qiskit設定ラベル|modes/sector/state、H4fixedrank12対tol-only19/28、prefix、候補panel、探索確認選抜、u/profile来歴が異なる|native実装による個別benchmarkの記述比較。統制されたsize scalingではない（E03;E04;E07;E16）|
|C04|H5対H7中性開殻pilot|LIMITED_COMPARISON|ROHF共通real MO/DF sector state/tol-only政策/T.8q1r2/K2,4/固定6/各PR8/P0/compiler SHA|分子/軌道/DF rank14対23/sector100対1225/prefix4対6/適格精度域が異なる。native gate列のbit同一性は未確認|各固定6構成内の点比較・feasibility。奇数鎖全体の最適性やn32独立確認とは呼ばない（E06;E15;E27;E32）|
|C05|奇数pilot対H6/H8独立確認|DIFFERENT_SCIENTIFIC_CONTRACT|finite signal/経験的allowance/paired full-wrapper/P0/RZ|q1固定6・各8対多q探索92・事前union各32。軌道/電子sector/DFと適格域も相違|限定pilotと独立確認を別系列で記述。n8/n32 poolingや同じ探索自由度との表示を禁止（E06;E07;E15;E16）|
|C06|H6強化136対H8強化184|LIMITED_COMPARISON|1Å/STO-3G/tol-only政策/T.8/q1,2,4,8/5精度/S2,S4/B0,B0I/full/経験的u/P0|DF19対28、prefix domain136対184、PRunion8対6/r範囲、モデル係数・実測範囲、非相殺欠測が違う|各保存domain内の符号・対照依存の記述比較。係数移送・共通最適化・サイズ則は不可（E09;E10;E21;E22）|
|C07|H6固定δT依存対H8単一T|DIFFERENT_SCIENTIFIC_CONTRACT|δ=.2/T=.8の一部条件、native/P0/RZ|H6は固定5構成でT.4,.8,1.6/q2,4,8。H8はT.8で候補を選択。分子/DF/r/Kも相違|T応答の機構記述に留める。H8のT依存・連続T則を推論しない（E12;E21）|
|C08|H8通常対経験的非相殺|DIFFERENT_SCIENTIFIC_CONTRACT|同じDF/state/T/compiler/184候補/PRunion6、単価優先規則、精度・α|bias定義がsigned total絶対値対成分絶対値和。headroom/shot/適格性/対照winnerが変化|保存済み感度比較として別namespaceで表示。厳密bias保証への格上げ不可（E19;E21;E22）|
|C09|H8旧32対強化184|LIMITED_COMPARISON|物理入力/PR標本/精度/shot政策は各bias namespace内で固定|決定論domainと最小値の分母が変化。旧32にも一部model単価が含まれる|対照強化の保存された影響を記述。旧32の削減率を184の結果と呼ばない（E18;E19;E21;E22）|
|C10|H8主要6軸単価更新前後|MATCHED_CONDITIONS|各bias namespace内で同じ候補・保存shot/適格性・PR平均/SE・旧32、固定model・選択規則|意図した変更は3候補6単価のmodelから実測への置換。実行SOURCE/result版は別。全候補実測化ではない|同じ会計問題での保存済み単価更新比較に限定してMATCHED。新しい再会計は実施しない（E18;E19;E20;E21;E22;E23）|
|C11|H6 rare補正のH8への移送|INSUFFICIENT_EVIDENCE|条件付き費用標本という方法上の参照のみ|DF/tail/component/basis/r/K/occurrence/whole-wrapper最適化が異なる。H8 pilot未完了成果を未収録|H6の0.00734086%をH8へ移さない。H8母平均上界は現固定証拠では未確立（E14;E16;E19）|
|C12|H6 epsilon.05のp7限定pair更新|MATCHED_CONDITIONS|同じp7/q1/S2保存shot/PR点費用・経験的u/P0|一候補両軸のmodel単価を実測へ置換。全136候補再ランキングや他4精度の更新ではない|保存された限定pairの逆転確認に限定してMATCHED（E09;E10）|

E番号は[source catalog](source_catalog_v1.json)の固定commit/path/blob/SHA256を参照。H4〜H8をpoolした平均・新しい順位・scaling fitは作成していない。
