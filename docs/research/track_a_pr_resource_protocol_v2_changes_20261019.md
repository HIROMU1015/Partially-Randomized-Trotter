# 共通実験プロトコル v1 → v2改訂案の変更点

2026-10-19。**改訂案であり、正式採用前。** [v2本文](track_a_pr_resource_evaluation_protocol_v2_DRAFT.md)を参照。
v1 `9036d8a8`を文書基準として保持し、掲載文書整備 `1923fce8`を実装親にする。既存v1や固定科学契約の条件を変更していない。

|変更ID|v1時点|v2案での整理|判断・証拠の性質|
|---|---|---|---|
|V2C01|対象はH4/H6/H8|**H2を追加対象として提案**|新しい対象範囲。ユーザー正式採用待ち|
|V2C02|H2は対象外|新tol-only rank3/sector4、最新829d697aでF/FP/D/A66/S_STATS完了、P未seal。初回7b202a1aのA未完了は履歴|公開結果/契約/監査の転記。旧H2 rank3/sector6と分離|
|V2C03|H4 tol-only主比較は未完了|DF10 D190/A50/S_STATS/P完了。DF12 legacyは別系列|H4粗精度NOT_REPRODUCED・PF低費用を保持|
|V2C04|H6独立PはPhase0段階|S_STATS後のn48/104独立P完了・採用|粗精度REPRODUCEDはPF優位方向の再現|
|V2C05|H8 control一致/profile未確認|選択PFの公開receiptと、PR詳細実行監査の公開不足を分離|SOURCE規則・保存結果・実receipt・申告を同一視しない|
|V2C06|A82はfield回収・説明未検証|後続1ff6ac0bで元32比較/64軸の説明完了|原図/注記を保全し、現行captionで履歴を説明|
|V2C07|図表の個別採否判断待ち|主表6独立条件、Figure2A legacy/探索、2B H8n48の表示固定を反映|H2を既存主表/Methods/CEMへ混合しない|
|V2C08|全入力生成同一性は未確認|H2/H4 dense、H6/H8記録eigsh、正規化・残差gate差を明記|未取得条件はnull/理由。系サイズの因果関係を主張しない|
|V2C09|v1文書採択は実行承認ではない|v2も新規実行・解析・有効grantを認可しない|権限の拡張なし。正式採用も別判断|
|V2C10|有限標本・希少event・母平均未保証|全4系に同じ限界を保持。H2 n1診断を独立確認と呼ばない|新しいCI、比、必要n、tail上界を計算しない|

機械可読版：[changes.csv](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/changes.csv)。
保存SOURCE/結果・候補登録・現行論文資料との対応は[bundle](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/README.md)にある。

時点差：指示時点/初回確認ではA未完了だった。ユーザーの追加指示で最新829d697aへ更新した。旧7b202a1aのprojectionと旧null契約は履歴として保全。H2 Aの528軌道/1056軸は公開完了監査の範囲で実施済み。探索n8であり独立Pではない。
H2 S_STATSのn80/12と92軌道/184軸/256 eventは保存済みの未採用提案で、新しい算出をしていない。P最終n/SOURCE/seed/grantは未固定。
H6/H8の全登録PFが直接実測になったわけではなく、入力全手順やH8原profile/全PR実行監査が確認済みになったわけでもない。
Methodsの既存H2除外と、本案のH2追加は論文scopeとプロトコルscopeの違いで整理する。既存CEM/台帳を上書きしない。

次は**ユーザー**がv2案の採否と追加対象範囲を判断する。H2の提案n採否、P最終seal、P実行承認は別個に判断する。Codexは文書公開確認後mandatory STOP。
