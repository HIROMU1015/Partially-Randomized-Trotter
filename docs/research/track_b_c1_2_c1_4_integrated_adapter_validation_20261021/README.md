# Track B C1-2〜C1-4 — C1-3入力照合でFAILED_STOP

**C1-2人工検証PASS、C1-3は1回実行・retry 0でFAILED_STOP、C1-4未着手。** H4のoperator・有限bit・X/Y比較には到達していない。後続のC1-5/native取得/Full-N/Pilotは未実行。

| 段階 | 状態・証拠 | 結論の範囲 |
|---|---|---|
| C1-2 | [artificial_checks.json](artificial_checks.json)、[SOURCE_AUDIT.md](SOURCE_AUDIT.md) | 2 system qubit/3 label人工入力。10 controlled event比較、正負sigma、長さ0/2/4、scalar/actual adjoint/X/Y、raw/coin zeroとtechnical failureの区別、全trial分母、63 eventのfirst moment。主agent自身の確認 |
| C1-3 | [failure.json](result/failure.json)、[terminal.json](result/terminal.json)、[raw stdout](result/raw_stdout.log)、[raw stderr](result/raw_stderr.log) | `R1_RESTORE`の`STATE_BYTES_MISMATCH`で停止。新しいH4 correctness PASSはない |
| C1-4 | NOT_STARTED | 前段失敗のため誤差・confidence・native/baseline契約は作成していない |

人工ideal finite Taylor first momentのspectral残差は `4.787479370177279e-16`、digital meanの残差は `4.768462069300654e-16`。既存G10の数値比較基準1e-10内である。digital meanはroot近似を含み、有限Taylorと厳密に同一という主張ではない。これらは人工入力の浮動小数点診断であり、H4保証・実用的資源優位・独立新規性・投稿GOではない。

新adapterは`g6_df_sidecar_v1`で既存G6/F3のeventとangle Evidenceを保持し、保存DF basisのactual inverse・signed Z/ZZ・controlled rotation・parent位相・逆時間順・scalar相対位相を接続する。既存DF builderのbasis appendとHadamard wrapperのregister作成部を再利用し、RTEEventやRPE resultへ型偽装しない。samplerと既存科学SOURCEは変更していない。native合成/費用certificateはない。

固定H4予定条件はlinear H4 1.00 Å/STO-3G/DF12、R1の新partition、8 qubit/13 basis/324 label、m=5、T=4/5、q=1、r=4、outer symmetric second order PF。旧PR-2と同じ科学入力とは扱わない。事前登録した6 eventと固定H=2048/K=4096は[execution_contract.json](execution_contract.json)にあり、**このH4に対するF1/F3の十分性確認自体が未実行**である。

失敗原因は[FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。R1のstate hashがdtype/shapeを含むことを今回のrunnerが取り違えた。NPZファイルの固定SHAは一致しており、物理的なモデル不一致やRAの不利な性能結果とは分類しない。SOURCEを修正して再実行していない。

- 固定SOURCE commit: `f3a6aa1108bbc8359e585c470e23033458c57466`
- 新authorization commit: `c7d444d5c59acfc1d57d6b4ed348a86451522b07`（SOURCEの直後、receiptだけのcommit）
- 基点R1: `2937e6b655ba3cd77f982d6901d766355a8b92c1`
- branch: `track-b-c1-2-c1-4-integrated-adapter-validation-20261021`

[handoff.md](handoff.md)に未実施範囲と次の承認、[verification.json](verification.json)に保全確認、[publication_receipt.json](publication_receipt.json)に公開照合を記録する。旧SOURCE/raw/grant/claim/ledger/marker/STOP、R1原成果、非公開レビュー原本は変更しない。

**MANDATORY STOP。今回の実行承認は消費済み。**
