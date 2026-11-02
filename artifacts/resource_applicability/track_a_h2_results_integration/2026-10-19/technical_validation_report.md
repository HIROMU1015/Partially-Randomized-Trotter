# 文書構造・証拠保全の検証

2026-10-19。主agentによる文書検証。別担当による独立科学レビューではない。

`structural_validation.json` の結果はPASS。GitHubの固定commit15件・公開stageファイル105件を取得したreceiptを使用し、Git blob ID/bytes、実行bindingのSOURCE hash117件、公開artifact manifest、結果hash、保存native exit0・同一親wait4/reapを全6段階で照合した。P公開projectionとprivate原resultの異なるhashを区別する。private原結果の再集計は行っていない。

比較表の先頭6行が既存主表と全fieldで一致し、主表・Methodsのsnapshotが固定公開bytesと一致する。H2のmean/SE/PF/n/statusは保存JSON pointerのliteralと一致し、新しい統計値・比・削減率を計算していない。paper data追加4行、CEM追加2行、台帳追加10行のschemaとID非衝突・JSON pointerを照合した。

v1本文・採択資料と原図34ファイルの固定bytes、H2 private8493ファイルの初回snapshot、既存validation manifest160項目、6入口文書の旧本文を保全照合した。既存科学SOURCE・結果・grant・ledger・marker・STOPを編集していない。公開対象は `publication_scope.json` の文書・公開値・metadataの範囲に限定する。

H4非再現・H6粗精度PF低費用・H8公開詳細PR監査不足はそのまま。H2は旧v1対象外・本文未採否。H2 v2追記案とpaper/CEM/台帳追加案は未採択／正本未適用。母平均優位、希少event費用/分散、全入力手順同一性、実powerを保証しない。

新規科学call・統計解析・sampling・compile・grant・seed生成0。詳細科学レビューと最終evidence freezeへ進まずmandatory STOP。
