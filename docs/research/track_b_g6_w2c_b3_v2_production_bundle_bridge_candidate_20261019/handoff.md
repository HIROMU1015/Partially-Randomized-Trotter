# Q1–Q10と引継ぎ

**人工技術候補は完成。新規107 PASS、補正版既存を含むunique828 PASS。科学入口は閉鎖、正式採択せずmandatory STOP。**

1. **Q1 固定照合**：Producer 596781daeba253418ba35f4b7964312a54caf233、独立受入 77126767c00585d0967558c0756907df2d69d1ffを固定参照し101公開ファイルをGit bytesへ照合。45bf測定SOURCEは公開bundleをfresh bare Gitへ復元し15測定SOURCE一致、後追加2helperと区別。旧108=正常2/技術失敗40/不完全66を維持。
2. **Q2 技術契約**：`b3_storage_v2_global_bundle_candidate_v1`／`GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1`へRAM/physical、trusted prefix、UNKNOWN非返却、cooperative STOP、全証拠joinを統合。新global receiptはstorage close後、markerはreceipt close後、final/delivery/caller観測も必要。旧V2履歴は変更0。
3. **Q3 legacy保守**：原版SHA・test・失敗traceを保全。co_qualname suffix期待値1個だけの別versionをcompile。原版622 PASS/1 FAIL、補正版623 PASS。全observer rowsとexclusive partition比較を維持。
4. **Q4 global結合**：単一fake claim、coordinator、RAM/statefile、call counter、trace、result/log budget、STOPでS→P1を結合。二つの独立実行ではない。S失敗でP1未開始、P1失敗でS証拠保持、spent/retry/resume拒否を人工確認。
5. **Q5 算術**：16cell・3phase・二条件、各256+4096+4096=8448、global16896をcontract/plan/manifest/receipt/consumerで一致。人工n256のreserved=started=completed=16896。
6. **Q6 prefix/F01**：固定科学SOURCE/input/precision/seed参照bytesを維持。modeはtrial IDへ追加しseed domainへ追加しない。人工S/P1/A/B/pass/correctness prefixとcanonical/F01を確認。実科学law/bit/angleの実行同値証明は未取得。
7. **Q7 障害**：atomic32競合、commit ACK不明/spent、STOP、in-flight drain、worker/parent crash、UNKNOWN、予約/entry/complete/state/partial trace/ACK、summary/diagnostic/receipt/STOP/close、marker偽成功、caller post-close観測を人工検証。
8. **Q8 件数**：最終43人工=global正常2/条件正常2/技術失敗13/証拠不完全26。新規107 PASSに43matrixと35mutationを含む。旧補正623＋既存storage41＋既存独立57＋新107=unique828 PASS。原版622/1、初期開発・歴史108/680/820は加算しない。予期しない最終test FAIL/ERROR/SKIP=0、未解決新規整合不一致0（実施範囲内）。
9. **Q9 残件**：[正式採択前の判断](pending_and_required_decisions.md)。V2/global/順序・費用・予算候補の正式採択、別production SOURCE・実API/authority/監視bindingと追加技術確認、科学実行の別明示認可が必要。実G7費用/coverage/保存量、独立環境再現、実store耐久性は未取得。
10. **Q10 禁止処理0**：実G7/F3 sampling/import、新科学計算、科学SOURCE採択、実authorization/grant/claim発行消費、native quantum build/compile/synthesisはすべて0。旧STOP解除/旧claim再使用0。サブエージェント0。作者とchecker実装者は同じprimary Codex。

今回の公開測定SOURCE：[e099dcc4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e099dcc461247e81eff981867dc28c3f3b16df5e)。最終contentとremote HEAD/各公開SHA256は[publication_receipt.json](publication_receipt.json)で照合する。runtime/source inventory・raw復元・保全記録を同梱した。

**次の担当：ユーザーとGPT**。本候補のSOURCE/技術契約と未採択条件をレビューし、正式採択の可否と次に必要な範囲を判断する。**Codexは公開照合後mandatory STOP**し、pilot/Full-N/科学実行へ進まない。
