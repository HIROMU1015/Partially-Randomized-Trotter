# 判断：既存interfaceは接続可能、productionの追加binding・証拠が必要

2026-11-05 JST / LOCAL。新H4科学実行0、既存確率/energy/mean/certificate再評価0。
linear H4/1Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02/K2/cutoff1e-12、保存ray/N4 sector70、
全18round・36軸の保存設定を維持した技術準備。

| 項目 | 判定 |
|---|---|
| 18round測定interface | **接続可能**。既存load/collect/decodeを再利用。production全round caller/callback glueは未接続 |
| Trajectory生成callback | **意味論の契約確定**。q*r events、signal τ、controlled/phase、per-step run_le_1、freshness。具体provider/戻り型bindingはbackend選択後 |
| 測定callback | **意味論・型契約確定**。Re/+Im、bit0/+1、bit1/−1、組込みint±1、reset/同じray。実backend適合性は未検証 |
| 保存uniform53との対応 | **条件付き対応**。同じCDF規則＋独立uniform53という仮定。PCG64のIID性は未保証 |
| Integer-sum・cmath経路 | **条件確定**。int和→単一除算→complex→cmath→既存scalar decoder。cmath/decoderの精度保証とは区別 |
| Production backend | **追加証拠必要**。既存候補の接続準備は可能だが未採択。standard wrapperはq1/2/4 |
| Float64 decoder一様保証 | **未保証・旧STOP維持** |
| RPE実装全体の成功保証 | **未保証**。既存のideal IID＋数学的decoderの条件付き結果を保持 |

[実経路とSOURCE](INTERFACE_SOURCE_MAP.md)、[callback契約](CALLBACK_CONTRACT.md)、
[backend比較・gap分類・次の最小計画](VALIDATION_GAPS_AND_NEXT_PLAN.md)、
[人工検証記録](TECHNICAL_VALIDATION.json)。

実装済みinterfaceを置換する根本的不一致は見つからなかった。
具体的な不足は、external providerのbinding、run_le_1等の明示接続、strict outcome型、
backend state/bit/numeric意味論、全round caller、G全域でのFloat64数値接続である。
これを埋めるために新scientific runnerやbackendを今回無理に追加しない。
追加コードは人工fixture/テストだけ。旧interface、sampler、decoder、SOURCE/raw/STOPは不変更。

次の担当は**ユーザー**：候補backendと、次に行う技術stage Tの対象を選択する。
**Codex**は別指示後、そのbackendと既存callbackの最小bindingを人工fixtureで固定する。
新しいH4数値対照が必要ならSOURCE/対象/精度/call範囲を先に公開し、別途明示科学実行許可後だけ実行する。
詳細科学レビュー、Float64 kernel・PCG64 IID研究、compiled/noisy・費用研究は自動開始しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。
**専用branchへのcommit/push・remote SHA/公開成果物照合後mandatory STOP。**
