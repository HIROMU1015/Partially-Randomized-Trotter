# N1公開・独立GitHub再取得

専用branch `n1-decode-recovery-execution-20261011` をHIROMU1015へpushし、新しい別bare repositoryへGitHubからdepth5で取得した。local repositoryのobjects/alternatesは使用していない。固定result commit `8816f9340de383d8a7efea4f083788584dc2e1af`、固定science source `3fe779f88ab6367631355cf5e5152bf32e897a4d`。receiptはこの取得確認の後に別commitで追加するので、下の一次結果URLは取得済みparent commitを指す。

- [固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3fe779f88ab6367631355cf5e5152bf32e897a4d/src/trottertracks/representation_exploration/n1_decode_recovery_v1.py)
- [source-bound plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/docs/research/n1_decode_recovery_v1_execution_plan_20261011.json)
- [結果とGPT判断事項](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/docs/research/n1_decode_recovery_v1_results_20261011.md)
- [一次結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/result.json)
- [独立保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/independent_audit.json)
- [実行監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/run_audit.json)
- [STOP記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/MANDATORY_STOP.json)
- [公開file inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/file_inventory.json)

646 changed published blobs、232 protected prior blobs、source/input bindings 162を含むunique 1003 blobs / 287826538 bytesを取得し、local byte/hashと照合PASS。science module・runner・synthetic tests・saved checkerの4 source blobsはsource commitとresult commitでもbyte同一。

remote exportから保存checkerを実行し、26,575 checks PASS。metadata/preservation checkerは1,644 checks PASS、旧source/結果220 blobs、準備を含む232 blobs、旧manifest89 entries保持、個別new entry schema、source/input/公開inventory hashを確認。元rootの1346 files・HEAD・dirty statusも保持した。global manifestの旧description欠落FAILは保持し、global PASSとは報告しない。

これは再取得した科学一次証拠の保存算術/native replayであり、新しいscientific batch、compile、ground solve、LP/SDP solveは0。immutable CI・別研究環境での外部科学再現ではない。現在のscience attempt1・retry0、mandatory STOP、次段認可false、中心テーマnull。

[機械receipt](../../artifacts/n1_decode_recovery_v1/2026-10-11/publication_receipt.json)、[remote保存監査](../../artifacts/n1_decode_recovery_v1/2026-10-11/remote_saved_audit.json)、[remote metadata監査](../../artifacts/n1_decode_recovery_v1/2026-10-11/remote_publication_metadata_audit.json)に全file hashと検査結果を記録。

全594 native IR、1424 rows、3 contextのresult/selection/checkpointを取得し、全列作用と独立Pauli交換子会計を再検査した。science numerical kernel48定義同一、source/input固定、実保存numeric scalar5428 ratio checksのpreflightを保持。parent+2×child peakのRSS値は主計算processの見積りで、helper processを含む同時peakの実測ではない。
