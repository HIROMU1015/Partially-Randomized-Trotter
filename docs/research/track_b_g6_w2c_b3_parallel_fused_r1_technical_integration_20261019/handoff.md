# Q1–Q9・最終handoff

非採択技術候補。最終測定/回帰SOURCE `860fef4a3a8ebb5a46628f4e0364e5934b5af416`。
1. **Q1 接続**：新v3 Host→独立fake承認→既存atomic ledger engine→改善P1→global summary/receipt→outer STOPを実装・人工実行した。旧Host v2は変更しない。
2. **Q2 fused**：固定helper bytesを再利用。exact canonical全bytes/chain/file SHA、Unicode/collision、F01/partial write/flushを人工検証。実時計runの非決定metadataは正規化して比較した。
3. **Q3 承認**：未承認host、候補のbool編集、実module loader/factory/bit/endpoint直接呼出しを拒否。global one-shot、commit後spent、retry0/resumeなしを維持。今SOURCEは正式科学releaseではなく、実scienceは閉じている。
4. **Q4 tests**：481 PASS（既存保護356＋統合125）、fail/error/skip0。同じ最終SOURCEで60性能＋35障害観測も完了。過去件数の単純合算ではない。
5. **Q5 性能**：P→PFのCPU削減light/mixed/heavy 5.6/4.0/1.8%、wall削減4.0/3.9/0.7%。新Host全費用を含む5反復中央値。heavy追加利益は小さく、実G7へ外挿しない。
6. **Q6 STOP**：P1同期例は開始8のうち失敗1＋完了7、同時失敗8を全保存。flush物理行と論理prefixを分け、unknown/未ACK creditを返却せず、保存不能を不完全STOPとして報告。未知CPUはnull。
7. **Q7 adapter**：SOURCE inventory、7 schema＋protocol、runtime manifest、正常12raw、障害35raw/観測とmachine reportを公開。ローカル側がこの公開情報だけでadapter実装へ進める。offline branchは未変更。
8. **Q8 判断**：重要事項はparallel CPU/observerの正式estimand、cooperative STOP/unknown保全の採否。その後、新たな正式SOURCE releaseと独立した実行scope承認が必要。6個の細かな実装gateにはしない。
9. **Q9 科学境界**：実G7/F3 import/sampling、新科学計算、実保存科学値再解析、実authorization/grant/claim、実correctness256/full-Nは全0。正式採択・科学レビューなし。

**次の作業（未着手）**：**ローカルCodex**が別scope/worktreeでoffline parallel adapterと受入試験を追加。**ユーザー**がその後に上記重要事項と正式採択/実行scopeを判断する。
**サーバーCodex**はcommit/publication/remote bytes/保全照合後mandatory STOPし、科学実行・正式releaseへ自動移行しない。
