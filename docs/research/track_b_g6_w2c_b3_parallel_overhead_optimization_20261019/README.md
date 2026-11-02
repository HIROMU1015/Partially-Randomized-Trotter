# B3 並列管理費用の改善・最終人工検証

**ARTIFICIAL_COMPLETE / NOT_ADOPTED / RUN_READY_FALSE**。独立branch `track-b-g6-w2c-b3-parallel-overhead-optimization-20261019`、直接基点7b6f526b。正式R1・本番Host/Admission/Driver・科学SOURCEは変更していない。

最終SOURCE **`77f8667fc0898a3615a4efa834d971b3c339df71`** で、既存311＋追加45＝**356人工tests PASS**、**99性能測定＋27診断＋17障害ケース**を完了。追加45件には旧36ケースの改善候補への再適用と9件の新しい境界検証を含む。skip0、主agent自身のchecksであり独立reviewerの検査ではない。

| 負荷 | S0 wall / CPU秒 | 旧P1 wall / CPU秒 | 改善P1 wall / CPU秒 | CPU削減 | wall削減 |
|---|---:|---:|---:|---:|---:|
| light | 13.8067 / 11.9682 | 6.0168 / 18.4206 | 2.9703 / 6.1490 | 66.6% | 50.6% |
| mixed | 7.0756 / 6.6512 | 3.3021 / 10.2070 | 1.8227 / 3.7741 | 63.0% | 44.8% |
| heavy | 6.5647 / 6.4287 | 2.9307 / 8.3151 | 2.0818 / 4.9703 | 40.2% | 29.0% |

同じ人工task集合・全3phase・各3反復の中央値。CPUは親＋全worker、wallはhost外側STOPまで。旧P1は追加Counterのない `before_exact8`。共有hostの負荷が測定中に増加したため、min/max・環境観測も保存し、専有機性能や実G7の予測とは扱わない。

主要改善は、全call/return・function対象を保持する**code/label/row cache付きobserver**と、開始gateを残して完了/write ACKを同梱する**coalesced IPC**。状態pwrite・全体byte事前予約・global one-shot・F01を維持する。親がcommit/bytesを観測するmessage単位は変わり、未ACKで喪失したcreditを返却しない。

- [原因の定量分解](overhead_cause_analysis.md)、[実装と意味論](implementation_and_semantics.md)
- [全設定・ばらつき・memory・障害結果](performance_and_regression_results.md)、[CSV](measurements/benchmark/summary.csv)
- [方法・再現](methodology_and_reproduction.md)、[契約差分・重要判断2点](contract_delta.md)、[Q1–Q8・次の担当](handoff.md)
- [最終SOURCE inventory](source_inventory.json)、[保護SOURCE](protected_source_inventory.json)、[SOURCE履歴bundle](measurement_source_history.bundle)
- [356 tests](tests/summary.json)、[全測定再読監査](artifact_verification.json)、[保全検証](verification.json)、[公開照合receipt](publication_receipt.json)

第1候補45c29d98の90性能/27診断/17障害・354 testsとrawはattempts/v1_*へbytes不変で保全。probeの追加案を採用後、最終SOURCEで全性能・回帰を再実施した。以前のP1測定b1dba4b6や第1候補の値を最終値へ流用していない。

実G7/F3 import/sampling、実科学データの数値再解析、新科学計算、実authorization/grant/claim発行消費、correctness256/31,710,848 calls本番はいずれも0。ローカルfused canonical候補ad3117d2、冪漸化式、GMP/nativeを統合していない。共有設定/他ユーザーprocess・file/他worktreeを変更していない。公開・保全照合後mandatory STOP。

公開照合：完全content commit `35617bdecb4fe38f0a348e7855db75132c071afd` の185ファイル・29,454,942 bytesをremoteから再取得し、全SHA256とbranch HEADを照合した。receipt自身を含む最終commitは別途照合する。
