# 不足証拠と引継ぎ v1

2026-10-17。以下は未着手の技術候補であり、追加実験の採択・承認ではない。既存のSTOP・V2 CLOSED_NO_GOを解除しない。

## O01：残る4精度の選択対照の直接費用と全136格子の誤差保証

- **affected_studies**：H6_STRENGTHENED136;H6_P7_MEASURED_PAIR
- **existing_partial_evidence**：S4登録4軸PASS、epsilon.05のp7両軸実測
- **what_is_already_verified**：p7限定pair実測費用356097424; PR403423200; 保存削減率負
- **what_is_not_verified**：他4精度のmodel推定を実測とは認定できない
- **possible_codex_task**：既存実測receiptの有無の限定調査。未取得ならユーザー判断後に必要候補だけ別compile計画
- **new_scientific_computation_required**：IF_NEW_COMPILE: YES
- **user_authorization_required**：metadata照合は不要、compile/再会計は別明示承認
- **related_SOURCE**：2261a3ad1d6c23521474b31204572a6c3a51258b;fcf37e6794e2e51fc95e958a75fc8f75fb049423;e078a7e6a8b53226f16994f405732785917a0779
- **related_result_commit**：f89c40af73289af9f4c2b70ed0b3966d95d47b06
- **evidence_ids**：E09;E10;E11
- **action_status**：NOT_STARTED; proposal only

## O02：旧B0I12候補のsigned bias成分

- **affected_studies**：H6_STRENGTHENED136
- **existing_partial_evidence**：他追加候補とPRの成分、通常会計は保存
- **what_is_already_verified**：非相殺全domainはUNRESOLVED_FULL_DOMAIN/null
- **what_is_not_verified**：欠測候補を除外した最良や厳密保守的優劣
- **possible_codex_task**：既存component receiptの存在だけ確認可能。無ければ信号分解の取得要否をユーザーへ戻す
- **new_scientific_computation_required**：IF_COMPONENT_RECOMPUTATION: YES
- **user_authorization_required**：新分解・非相殺再会計は別明示承認
- **related_SOURCE**：2261a3ad1d6c23521474b31204572a6c3a51258b;fcf37e6794e2e51fc95e958a75fc8f75fb049423
- **related_result_commit**：f89c40af73289af9f4c2b70ed0b3966d95d47b06
- **evidence_ids**：E09
- **action_status**：NOT_STARTED; proposal only

## O03：未実測単価のmodel誤差・全184実測最良

- **affected_studies**：H8_CURRENT_SIGNED;H8_CURRENT_NONCANCELLATION
- **existing_partial_evidence**：B1 S2 anchors24軸、B2 S4登録4軸、主要6軸
- **what_is_already_verified**：最新通常4精度/非相殺1精度winner単価は主要実測に置換
- **what_is_not_verified**：全格子で2%未満、未実測候補より真に最小、全系列実測比較
- **possible_codex_task**：保存receipt一覧と未測定候補の来歴を提示し、追加実測の必要性はユーザーが判断
- **new_scientific_computation_required**：IF_NEW_COMPILE: YES
- **user_authorization_required**：追加compile/fit/再会計は別明示承認
- **related_SOURCE**：aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E20;E21;E22;E25;E26
- **action_status**：NOT_STARTED; proposal only

## O04：PR母平均費用のtail上界、未観測次数・component列

- **affected_studies**：H6_CONFIRMATION;H6_CONDITIONAL_S2;H8_ORIGINAL_MAP;H8_CURRENT_SIGNED
- **existing_partial_evidence**：n32独立確認とH6局所条件付き8本
- **what_is_already_verified**：H6局所補正小、H8保存確認order4=0という観測
- **what_is_not_verified**：未観測を費用0、観測最大を全support上界、H6補正をH8へ移送
- **possible_codex_task**：他チャットH8 pilotの正式完了報告後、別指示で固定証拠を照合。本監査は途中結果・私有領域に入らない
- **new_scientific_computation_required**：追加取得ならYES。本作業では0
- **user_authorization_required**：新sampling/compile/層別総費用更新は別明示承認
- **related_SOURCE**：5a903aea2492c2d211c1e15e7737df0d50694c12;87b6b9f46b7e8ba9abe132ee21478c114b6f46f7;73018cbcc39803890974c7ee51401063e6c8daf6;843265b624eac8fbb9acea47ef21cab1207e1880;cb8a56c6b2a772827275f9200b02638030f33f72;c31ef424175c02379eafebe8fc6c7548c1d9aabe;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：321cd8c3066194fe98741cbebbcdba1515d1deaa;0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41;681ec23c93fdd1090d325908f4b155a79115afdf;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E07;E13;E14;E16;E19;E30
- **action_status**：NOT_STARTED; proposal only

## O05：保存event次数histogramと一部normalization原記録

- **affected_studies**：H4_GEOMETRY;H4_GEOMETRY_B0I
- **existing_partial_evidence**：finite law・元32費用平均・旧解析に172丸めnormalizationの限定
- **what_is_already_verified**：保存点最小1812と実測240軸の結果
- **what_is_not_verified**：法則から実際の過去イベント観測を断定できない
- **possible_codex_task**：既存receiptがある場合のみに限定回収。seedからの再生成を過去観測証拠と混同しない
- **new_scientific_computation_required**：receipt回収NO; 再生成ならYES
- **user_authorization_required**：新sampling/統計解析は別明示承認
- **related_SOURCE**：d0726e946f89f8c30a08d383fb8ff48b363629ec;5aa6c3685afb9c3afcfbdc36f80244da0a7fe525;924b779c2bc6f6e844d6283c355f91c73aed6445
- **related_result_commit**：7c4483378047b57f514c47dad98905e3f2a9a756;98cdb35cd08469714f20d6cc3b982d9c80908f46
- **evidence_ids**：E01;E02;E03
- **action_status**：NOT_STARTED; proposal only

## O06：親driverのnative exit数値

- **affected_studies**：H4_GEOMETRY_B0I
- **existing_partial_evidence**：worker12 exit0・terminal/cleanup・identity absent
- **what_is_already_verified**：MAP240_COMPLETE_STOPという保存終了状態
- **what_is_not_verified**：親native exit0は資料でnullのまま
- **possible_codex_task**：独立native wait receiptが既存の場合だけ読取確認。無ければUNKNOWNを維持
- **new_scientific_computation_required**：NO
- **user_authorization_required**：読取専用証拠回収の範囲で対応可能
- **related_SOURCE**：924b779c2bc6f6e844d6283c355f91c73aed6445;5aa6c3685afb9c3afcfbdc36f80244da0a7fe525
- **related_result_commit**：98cdb35cd08469714f20d6cc3b982d9c80908f46
- **evidence_ids**：E02
- **action_status**：NOT_STARTED; proposal only

## O07：旧host/compiler出力のbit同値、旧B0・fullPF一部profile、prefix19元runtime SHA

- **affected_studies**：H4_NATIVE_ANCHOR;H6_CONFIRMATION;H6_NATIVE_B0I;H8_CURRENT_SIGNED
- **existing_partial_evidence**：compiler SHA1239304c…、新profile、個別native数値probeと受入receipt
- **what_is_already_verified**：現在の登録実測/保全・個別SOURCE分離
- **what_is_not_verified**：同じversion/hashラベルだけで別preparationや全runtimeを同一認定不可
- **possible_codex_task**：既存profile/receiptの限定照合。原記録欠測はunknown、STOPは保全
- **new_scientific_computation_required**：metadata NO; 独立再compileならYES
- **user_authorization_required**：新compileは別明示承認
- **related_SOURCE**：217c971dd537b1cc7fbc13742bc0f6c263ff7f45;5a903aea2492c2d211c1e15e7737df0d50694c12;87b6b9f46b7e8ba9abe132ee21478c114b6f46f7;73018cbcc39803890974c7ee51401063e6c8daf6;ac6d28e35cf01398f169ce59577182a38da9b959;2e733d8a683d415792ce92a56c7d149b78f42716;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41;321cd8c3066194fe98741cbebbcdba1515d1deaa;34efc914b2f485ec694c7d6016025d5c76829be6;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E03;E07;E08;E19;E25
- **action_status**：NOT_STARTED; proposal only

## O08：厳密numerical allowance/DF・state・bias certificate

- **affected_studies**：H4_GEOMETRY;H5_NEUTRAL_PILOT;H6_STRENGTHENED136;H7_NEUTRAL_PILOT;H8_CURRENT_SIGNED
- **existing_partial_evidence**：経験的uとx10感度、native numerical correctness検査
- **what_is_already_verified**：保存binary64モデル内の宣言会計と局所数値一致
- **what_is_not_verified**：u/headroom gateやx10が厳密証明を与えるわけではない
- **possible_codex_task**：既存誤差予算の説明資料を作成可能。厳密証明を研究要件にするかはユーザー判断
- **new_scientific_computation_required**：証明設計次第; 新数値検証ならYES
- **user_authorization_required**：新科学検証は別明示承認
- **related_SOURCE**：d0726e946f89f8c30a08d383fb8ff48b363629ec;5aa6c3685afb9c3afcfbdc36f80244da0a7fe525;d86a4673d6ada701797d3e035f745d0d4faf5150;2261a3ad1d6c23521474b31204572a6c3a51258b;fcf37e6794e2e51fc95e958a75fc8f75fb049423;f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2;d9fd36fc630a67fab220752154bfaa9c4b8d60c6;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：7c4483378047b57f514c47dad98905e3f2a9a756;bde58ba23f1fecdf77ee950795cb80c199c0f86b;f89c40af73289af9f4c2b70ed0b3966d95d47b06;162c256ea39b4ab3dadde65a94b3eec73cea4720;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E01;E06;E09;E15;E21;E29
- **action_status**：NOT_STARTED; proposal only

## O09：q/prefix/r/Kを広げた資源map・独立確認

- **affected_studies**：H5_NEUTRAL_PILOT;H7_NEUTRAL_PILOT
- **existing_partial_evidence**：固定6・q1・各PR8の完了pilot
- **what_is_already_verified**：中性ROHF/DF-sector/native取得完了
- **what_is_not_verified**：全候補最良・奇数鎖全域・n32独立確認ではない
- **possible_codex_task**：限定panelで言える範囲を保持。拡張実行の採否は今回決めない
- **new_scientific_computation_required**：拡張ならYES
- **user_authorization_required**：別実行契約と明示承認
- **related_SOURCE**：d86a4673d6ada701797d3e035f745d0d4faf5150;f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2;d9fd36fc630a67fab220752154bfaa9c4b8d60c6
- **related_result_commit**：bde58ba23f1fecdf77ee950795cb80c199c0f86b;162c256ea39b4ab3dadde65a94b3eec73cea4720
- **evidence_ids**：E06;E15;E29
- **action_status**：NOT_STARTED; proposal only

## O10：DF/軌道/候補/標本政策を統制したsize comparison

- **affected_studies**：H4_NATIVE_ANCHOR;H6_STRENGTHENED136;H8_CURRENT_SIGNED
- **existing_partial_evidence**：個別native fixed modelの条件表
- **what_is_already_verified**：H4 legacyrank12とH6/8 tol-only19/28を区別
- **what_is_not_verified**：H4-H8全体のscaling則・単調なPR利益
- **possible_codex_task**：既存比較を記述的LIMITEDに留める。既存2026-10-15レビューではH4全面再統一を必須にしていない
- **new_scientific_computation_required**：統制した追加研究ならYES
- **user_authorization_required**：新設計・新科学計算は別判断/承認
- **related_SOURCE**：217c971dd537b1cc7fbc13742bc0f6c263ff7f45;2261a3ad1d6c23521474b31204572a6c3a51258b;fcf37e6794e2e51fc95e958a75fc8f75fb049423;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41;f89c40af73289af9f4c2b70ed0b3966d95d47b06;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E03;E29;E31
- **action_status**：NOT_STARTED; proposal only

## O11：選択補正済みCI・母平均winner・独立追試

- **affected_studies**：H4_NATIVE_ANCHOR;H6_CONFIRMATION;H8_CURRENT_SIGNED
- **existing_partial_evidence**：探索からunion固定後の独立32、paired SE
- **what_is_already_verified**：n8/n32の分離、paired n32でありn64でない
- **what_is_not_verified**：確認後点最小・K2/K4小差を母平均順位へ格上げ不可
- **possible_codex_task**：選抜/評価namespaceの説明と既存seed来歴照合は可能。新統計推定をしない
- **new_scientific_computation_required**：新統計解析/追試ならYES
- **user_authorization_required**：別明示承認
- **related_SOURCE**：217c971dd537b1cc7fbc13742bc0f6c263ff7f45;5a903aea2492c2d211c1e15e7737df0d50694c12;87b6b9f46b7e8ba9abe132ee21478c114b6f46f7;73018cbcc39803890974c7ee51401063e6c8daf6;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41;321cd8c3066194fe98741cbebbcdba1515d1deaa;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E04;E07;E16;E21
- **action_status**：NOT_STARTED; proposal only

## O12：実state preparation費用・最終energy/QPE/RPE/FT資源

- **affected_studies**：H6_NATIVE_B0I;H6_FIXED_DELTA_T;H8_D0;H8_CURRENT_SIGNED
- **existing_partial_evidence**：保存仮想P感度/固定pair交点/一部envelope
- **what_is_already_verified**：主結果はP0 finite-time complex signalの解析的shot予算
- **what_is_not_verified**：epsilon_sigをHaへ読み替え、実機runtime、end-to-end費用
- **possible_codex_task**：既存P感度を明示して範囲を説明。新energy実験やP再最適化を開始しない
- **new_scientific_computation_required**：追加評価ならYES
- **user_authorization_required**：新会計・測定/compileは別明示承認
- **related_SOURCE**：ac6d28e35cf01398f169ce59577182a38da9b959;2e733d8a683d415792ce92a56c7d149b78f42716;87b6b9f46b7e8ba9abe132ee21478c114b6f46f7;669842c85e2c04a90dcdcfb774b323a2737cb8ba;5a903aea2492c2d211c1e15e7737df0d50694c12;20f89bc32b161ee22e4a1ed709f9e3414c19633a;fa75c7997e135da7b7a0c4e23949e47252bd7a2e;aa984d98a7b438dc953a829a24fe3d7edc9d0a6b
- **related_result_commit**：34efc914b2f485ec694c7d6016025d5c76829be6;043f58ac9336d662a3181caff508526568f1e57a;3d910ee1de4597b8895e8506500553dbad5a5085;96e69a01d1fae7a402aac2c4efa570a59f8cc7a0
- **evidence_ids**：E08;E12;E17;E21;E29
- **action_status**：NOT_STARTED; proposal only

## O13：一部charge/orbital policyの原記録、完全環境digestの直接join

- **affected_studies**：H4_GEOMETRY;H4_NATIVE_ANCHOR;H6_CONFIRMATION;H8_ORIGINAL_MAP
- **existing_partial_evidence**：電子sector/modes/DF rank・固定scopeとcompilerラベル
- **what_is_already_verified**：確認済み値だけ表に収録
- **what_is_not_verified**：閉殻電子数からRHF/ROHFの使用を推測しない。UNKNOWNは不存在を意味しない
- **possible_codex_task**：既存公開input/profileの限定metadata照合候補。本作業では未照合欄をNOT_VERIFIEDのまま保存
- **new_scientific_computation_required**：NO for existing metadata
- **user_authorization_required**：新科学計算不要; 実行は今回STOP後の指示に従う
- **related_SOURCE**：d0726e946f89f8c30a08d383fb8ff48b363629ec;5aa6c3685afb9c3afcfbdc36f80244da0a7fe525;217c971dd537b1cc7fbc13742bc0f6c263ff7f45;5a903aea2492c2d211c1e15e7737df0d50694c12;87b6b9f46b7e8ba9abe132ee21478c114b6f46f7;73018cbcc39803890974c7ee51401063e6c8daf6;cb8a56c6b2a772827275f9200b02638030f33f72;c31ef424175c02379eafebe8fc6c7548c1d9aabe
- **related_result_commit**：7c4483378047b57f514c47dad98905e3f2a9a756;0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41;321cd8c3066194fe98741cbebbcdba1515d1deaa;681ec23c93fdd1090d325908f4b155a79115afdf
- **evidence_ids**：E01;E03;E04;E07;E16
- **action_status**：NOT_STARTED; proposal only

## 既存科学レビューの判断履歴（今回の新規採否ではない）

|既存判断日|固定出典|記録されている判断|今の取扱い|
|---|---|---|---|
|2026-10-13|E30、pilot SOURCE843265b624eac8fbb9acea47ef21cab1207e1880、科学SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12|H6条件付きpilotを留保付きbenchmarkへ統合、追加H6 samplingを必須工程にしない|当時の判断として保持。H8への費用移送や今回の実行認可ではない|
|2026-10-14|E31、結果98cdb35cd08469714f20d6cc3b982d9c80908f46、SOURCE924b779c2bc6f6e844d6283c355f91c73aed6445|H4限定実装資源研究として採用、追加全H4 sampling/S4拡張を必須にしない|元レビュー原本は未変更。397/1415は独立実験勝率ではない|
|2026-10-15|E29、旧H8結果681ec23c93fdd1090d325908f4b155a79115afdf、科学SOURCEcb8a56c6b2a772827275f9200b02638030f33f72|強化nativeB0I、H5/H7中性pilot、H6固定δT比較、epsilon.0005を採択。H4 DF全面再統一・全support証明をその時点の必須にしない|後続完了は各結果E06-E22により更新。レビュー当時の未実施を現在の状態として転記しない。追加実行の包括認可でもない|

レビュー本文の算術・付録コード・外部文献監査は再実行していない。新規性や主要claimは今回採択しない。E番号は[source catalog](source_catalog_v1.json)。
