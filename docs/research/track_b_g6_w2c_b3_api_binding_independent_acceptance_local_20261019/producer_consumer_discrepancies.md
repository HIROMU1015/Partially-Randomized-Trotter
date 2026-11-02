# 不一致・不足と修正境界

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

固定人工4case、公開port22 mutations、固定Bundle43/35、new130と既存845の範囲で、**Producer側の新しい実装不具合は確認されなかった**。これを未測定の実科学意味論が正しいという主張へ拡張しない。

|ID|内容|処理・証拠|
|---|---|---|
|D01|科学8moduleのreturn文字列表現に3.10/3.12差|独立SOURCE/segment/hash・AST比較へ分離。PEP701の2表現の再parse未確立を保存|
|D02|初回独立checkerがprofile≥16のhash retentionをfull-documentとして読んだ|SOURCEのretention境界に合わせ、人工再構成event/evidence/angleの3hash照合へ修正。最終130PASS|
|D03|初回checker用固定treeにinput10件が未materialize|指定HEADと科学固定commitからbytes取得し、全779公開照合へ追加。資料不足をProducer不適用と扱わない|
|D04|既存165のfresh fixture root指定不足|旧SOURCE未変更、正しいfresh_cases下を指定し165PASS。初回10 ERRORをdevelopment_diagnosticsへ保全|
|D05|初回Producer73 launchでPYTHONPATH不足|元test未変更。疎runtimeのPYTHONPATH=srcで73PASS。初回importエラーlogを保全|
|D06|Bundle結果JSONのFraction serialization|独立集計helperで明示exact_fractionを使用。固定rawを変更しない|

新consumerはProducer self-checkerより、測定commit、report STOP/閉鎖flag、failure caseのcaller witness、人工event再構成等を明示的に照合する。追加の厳しさだけをProducer不具合と数えない。独立mutationの最小入力は[35保存差分](independent_mutation_overlays/)とnew test source（40 component/18 contract/8 static negatives）に保存。

未接続codec/real bits/authority/Global Bundle injectionは設計上のpending。資料不足や実行未確認は `NOT_ESTABLISHED/PENDING` であり、技術的不可能 `NOT_APPLICABLE` ではない。実APIを実行しない限り解消しない事項を今回の範囲外として保持する。
