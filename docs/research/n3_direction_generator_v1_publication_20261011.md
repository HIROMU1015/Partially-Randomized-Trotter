# N3公開・独立GitHub再取得

専用branch `n3-generator-transfer-execution-20261011` をHIROMU1015へpushし、新しい別bare repositoryへGitHubからdepth5で取得した。local repositoryのobjects/alternatesは使用していない。固定result commit `b3e41b76fbb21c238c25572620e967d131571458`、固定science source `178d69a31dc7f2f1162d76294ba35a94b67d7fee`。receiptはこの取得確認の後に別commitで追加するので、下の一次結果URLは取得済みparent commitを指す。

- [固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/178d69a31dc7f2f1162d76294ba35a94b67d7fee/src/trottertracks/representation_exploration/n3_direction_generator_v1.py)
- [source-bound plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/docs/research/n3_direction_generator_v1_execution_plan_20261011.json)
- [結果とGPT判断事項](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/docs/research/n3_direction_generator_v1_results_20261011.md)
- [一次結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/result.json)
- [独立保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/independent_audit.json)
- [実行監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/run_audit.json)
- [STOP記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/MANDATORY_STOP.json)
- [公開file inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/file_inventory.json)

37 changed published blobs、232 protected prior blobs、source/input bindings 11を含むunique 269 blobs / 259890366 bytesを取得し、local byte/hashと照合PASS。science module・runner・synthetic tests・saved checkerの4 source blobsはsource commitとresult commitでもbyte同一。

remote exportから保存checkerを実行し、49,016 checks PASS。metadata/preservation checkerは290 checks PASS、旧source/結果220 blobs、準備を含む232 blobs、旧manifest89 entries保持、個別new entry schema、source/input/公開inventory hashを確認。元rootの1346 files・HEAD・dirty statusも保持した。global manifestの旧description欠落FAILは保持し、global PASSとは報告しない。

これは再取得した科学一次証拠の保存算術/native replayであり、新しいscientific batch、compile、ground solve、LP/SDP solveは0。immutable CI・別研究環境での外部科学再現ではない。現在のscience attempt1・retry0、mandatory STOP、次段認可false、中心テーマnull。

[機械receipt](../../artifacts/n3_direction_generator_v1/2026-10-11/publication_receipt.json)、[remote保存監査](../../artifacts/n3_direction_generator_v1/2026-10-11/remote_saved_audit.json)、[remote metadata監査](../../artifacts/n3_direction_generator_v1/2026-10-11/remote_publication_metadata_audit.json)に全file hashと検査結果を記録。

[全72証拠ledger](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/certificate_ledger.json)、[ledger seal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/ledger_seal.json)、[truth診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/diagnostics.json)も同じ固定commitから取得した。
