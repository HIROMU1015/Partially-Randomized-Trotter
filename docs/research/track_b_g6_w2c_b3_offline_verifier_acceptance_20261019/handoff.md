# Q1–Q8と次の担当

**Q1：実装できたか。** はい。独立pair streaming verifier、別cross-pair iterator、serial global adapterとCLIを実装。入力read-only、別の新規出力directory、探索・overwriteなし。

**Q2：何を検証するか。** 元payload文字列のchain SHA、実file SHA/bytes/rows、trial ID/order/R-C-A/N、summaryとreceiptの結合、F01 committed/candidate/inflight/partial write/flush/file hash失敗。全N未達・receipt欠落・失敗prefix・schema未対応・破損を区別する。

**Q3：full traceの限界。** 全文のあるcorrectnessとprofile先頭16/axisのevent bytes/連結SHAは独立再計算。compact以後は保存event hash同士をstreaming比較し、full trace summary値を照合する。元eventや連結SHAを個々のhashから復元・独立再計算できたとはしない。

**Q4：費用会計は。** CPU/wall、setup/online/R-C-A/angle/evidence/output/memory observer、outer/residual、RSS/HWM/bytes/rowsを別座標で実装。coarse/fine、親子wall、共有X/Ysetup、prepare/constructor、二passの二重計上を防ぎ、unknownはnull。rankingや純observer overhead認証はない。

**Q5：PASS件数。** 既存275＋新85＝**360 tests PASS、FAIL/ERROR/SKIP0**。受入例の385 PASSは受入check行数で別。13項目pending。

**Q6：高速化基盤はどこまで。** canonical bytes/Unicode/surrogate/DEL/key collision、source/contract、F01/Admission/Host/STOP、observer変更・費用の人工受入APIと回帰を実装。既存baseline上で実行済み。fused SOURCEの統合・今回の候補再実行・正式採択はしていない。parallel具体adapterはPENDING_SERVER_CONTRACT。

**Q7：server契約公開後は。** 固定commit/schemaの取得後、global ID、worker failure/inflight/exit、started/completed/failed/unknown、global call/output/log budget、phase barrier、atomic claim、STOP通知後新開始0・既開始inflight、複数失敗、CPU/wall配分、global/pair summaryとreceiptを新adapterで検査する。

**Q8：禁止範囲への移行は。** 実G7/F3 science import・sampling・科学計算・実データ解析・実authorization/grant/marker/claim発行消費は0。実claim ledgerを新設・使用・変更していない。実行したSQLite claim/STOP回帰はprivate人工namespaceだけ。科学SOURCE/Driver/Admission/Host/Readiness/R1契約・結果・既存STOPは保全。

次の担当は**ユーザー**：server側の並列最終成果が公開された後、その固定commitと契約変更案を指定し、adapter実装・人工統合検証を依頼するか決める。
依頼後の**Codex**：新schema adapterとglobal accounting/STOP/observer検査を追加し、候補SOURCE/pinの人工受入結果を公開する。これは後続提案で今回未着手。
正式高速化採択、最終science release/digest、実authority/claim、科学実行・実結果解析・科学レビューは別途明示承認が必要。
今回の実装・人工検証・commit/push・公開bytes/remote HEAD・保全確認後 **mandatory STOP**。
