# 登録caseの状態

全11caseが1attempt／COMPLETED。失敗・未実行0。表は実attempt/completedと保存manifestから転記した。既存DAGを変更していない。

| Case | attempts | 結果 | 保存出力（private outputからのrelative path） |
|---|---:|---|---|
| B.INPUT.physical | 1 | COMPLETED | `inputs/physical.json` |
| B.INPUT.catalog | 1 | COMPLETED | `inputs/catalog.json` |
| B.ARMS.SS | 1 | COMPLETED | `arms/SS.json` |
| B.ARMS.SP | 1 | COMPLETED | `arms/SP.json` |
| B.ARMS.PS | 1 | COMPLETED | `arms/PS.json` |
| B.ARMS.PP | 1 | COMPLETED | `arms/PP.json` |
| B.CONTEXTS.SS | 1 | COMPLETED | `laws/SS.json` |
| B.CONTEXTS.SP | 1 | COMPLETED | `laws/SP.json` |
| B.CONTEXTS.PS | 1 | COMPLETED | `laws/PS.json` |
| B.CONTEXTS.PP | 1 | COMPLETED | `laws/PP.json` |
| B.CONTEXTS.catalog | 1 | COMPLETED | `laws/catalog.json` |

内部関数呼出しの対応は[CALL_ACCOUNTING.json](CALL_ACCOUNTING.json)。これは完了した固定SOURCE pathと登録台帳からの対応で、runtime profilerを追加した観測値ではない。RTEComponent構築は保存Lによる153件、make_rte_config5、finite_rte_distribution10、paired weight40、residual-bound25、normalization20等。科学的importとしての単独rte.py loadは最初のCONTEXTS内1回。preflightのNumPy技術importはこのcase数に含めない。

各caseはsource hash照合→immutable attempt記録→science1回→payload/hash→completedの固定実装を使用した。科学SOURCEは変更していない。post-run監査はread/hash/既存record比較だけで、scienceの再実行ではない。
