# 固定SOURCEとの差分と実装方式

基点 `564409487a35e15d73f1d0f0c2611824b400ab02` のF01/R1/Admission統合版から独立worktreeを作成。
サーバーbundleはSHA `d8b6a7da4be2947d4b2d1e2f648e96148eb793917c219ed6f5fd16e632735270`、25,046 bytesを確認。
`git bundle verify`後、prerequisite 07f65831...をread-only alternateとして専用/tmp bare Gitで復元した。
73451938.../23712610...のSOURCEを実際に取得し、後者のPython encoder/endpoint recurrenceを参考にした。
SOURCE再利用の適応実装であり、未取得なのに再利用したという記録ではない。scheduler/GMP/orjson/pool/bufferは取り込んでいない。

| SOURCE | 変更 |
|---|---|
| 新exact_serialization | driver.exact＋audit._packを融合。通常treeは中間正規化objectと二重sortを削減 |
| 技術候補driver | identity、Streamのpayload、trial event trace、child failure文書のcanonical呼出だけを置換。新依存pinを必須化 |
| 新angle_recurrence | endpoint乗算漸化式＋artificial-only enclosure。fixed precision/root/Evidence APIを使い、F3 globals/sessionは変更しない |
| 旧36/69 tests | 新codec dependencyを人工SOURCE pinへ追加するだけ |
| 旧F01 26 tests | serialization故障注入先を新encoder入口へ移すだけ。失敗assertions/正常8referenceは維持 |
| 新tests/benchmark | canonical36、angle28、4構成×2operand widths×5反復、serialization micro |

G7、ReturnKernel、F3 contract/precision/generator、cost helper、Admission、Host、Readinessはbytes不変。
Meter/FunctionObserver/Boundary/LegacyEndpoint/seed_for/load_modulesとF01 bookkeepingは変更していない。
F01 current/last completed、candidate/committed、emit完了、write/hash/flush、primary/cleanup、claim/STOPを省略していない。
[zero-context差分](source_diff.patch)と全SOURCE SHAはinventoryを参照。

新codec moduleはdriver candidateの必須pin。Admissionの既存完全SOURCE検査/role集合を通し、science/dependency roleへ含める。
新しい正式契約や実authorityは発行していない。旧候補を編集して実行可能にするものではない。
