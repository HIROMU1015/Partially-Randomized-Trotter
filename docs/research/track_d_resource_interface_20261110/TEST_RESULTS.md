# 人工・静的・保存読取テスト

[REPORT](REPORT.md)／[test SOURCE](../../../tests/check_track_d_resource_interface.py)／[log](../../../artifacts/track_d_resource_interface/2026-11-10/technical/)。分類は技術整合性確認であり、新しいH4誤差・成功確率・総費用の数値検証ではない。

## 最終結果

| 検査 | 件数 | 内容・結果 |
|---|---:|---|
| 新Synthetic test method | 6 | 非一様round N、異なるX/Y単価、両method、6資源、peak幅、既存集計へ委譲、missing/duplicate、scope・target・資格・不正値等。PASS |
| 新SavedReadOnly test method | 6 | A/F/S×両methodの保存field照合、q1/cut/直接・外挿の資格、未知PR/target/law/state、control profile、actual未資格、copy保全。PASS |
| 旧check_track_d_bc_total人工回帰 | 12 | 合成scalarのmodel/集計/covariance/不適格input等。PASS。実分子の価格や総費用を再評価していない |

最終は**重複なし24 test method PASS**。同じ12件の修正後再実行を別test件数へ加算しない。6つのmethod/scenario viewについて6資源＝36保存scalarの選択が一致した。これは保存値をそのまま読む照合であり、新しい計算結果36件という意味ではない。

## H4再計算を呼ばない確認

保存readerの構築・選択中は、既存accounting.aggregate_totals、bc.aggregate_registered、shot/provider読込を行うaccounting._legacy、bc.read_b4をmockで遮断し、誤って呼べば数値処理へ入る前にFAILさせた。読取経路はGit固定blob→schema/binding→保存column選択のみ。

保存14,738 shotsはB4のstored fieldを比較し、round Nの新算定も合計の再計算もしない。旧A/F/S総値はresult.jsonの該当columnと比較するだけで、real H4のΣNμを呼ばない。q1 directと長q proxyのラベル、全体CI=null、actual_success_guaranteed=falseを維持する。

CLIも固定Reference/Aを一回読取確認した。[保存出典付きview](../../../artifacts/track_d_resource_interface/2026-11-10/SAVED_REFERENCE_A.json)は既存結果の引用・copyであり、total_originとshot_originに再計算していない旨を含む。raw event/QPY/NPZは読み出し・追加公開していない。

## 初回failureと修正

初回SOURCE `0c4950691bcf54685ab1b62a577740f1cd00ad69` は新12件中2件の人工集計testがFAIL、保存read-only6件は全PASS。旧人工12件もPASSした。原因は、新価格型から旧集計器が要求するlong_round_coverageへ、全horizon価格資格を橋渡ししていなかったこと。旧集計器は正しく未資格としてtotals=Noneを返した。

初回[log](../../../artifacts/track_d_resource_interface/2026-11-10/technical/check_track_d_resource_interface.py.txt)とRESULT.jsonをそのまま保全した。修正SOURCE `46b7fbbfcc3a0117c77294bf0d7e8446f56baa70` でMethodPricesに明示的horizon資格を加え、無資格を拒否した上で旧metadataへ投影した。旧価格や数式、閾値を変えて通したのではない。修正後12件PASS。

最終SOURCE `afc3bc5106773e7b36f4cb189e697b140c20c3a1` で保存SOURCE/decoder/STOP照合と読取CLIを加え、12件を再確認してPASS。[最終log](../../../artifacts/track_d_resource_interface/2026-11-10/technical/interface_final.txt)。実H4の科学処理retry/resumeは0、旧STOPは再開していない。

## 未確認事項

新しい実H4 shot算定、全13round再集計、元H ground solve、PF/RTE誤差、sampling、回路生成/compile/simulation、q≥64価格、A/F/S fit、別PRの測定同値性・総資源優位は検査していない。旧有限検査器のq4/q8 correctness・全56資源PASSも今回再実行していない。

このadapterのtest PASSは、未知PR設定の科学的資格を認定しない。上流から渡すaccuracyやhorizon資格は、その根拠の正しさまで自動的に証明されるものではない。
