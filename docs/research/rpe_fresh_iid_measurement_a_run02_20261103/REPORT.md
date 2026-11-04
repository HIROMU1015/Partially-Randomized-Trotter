# RPE候補A run02：fresh trajectory・仮想Hadamard測定の限定診断

2026-11-03 JST。**処理は `COMPLETED_MANDATORY_STOP`、4軸・12比較はすべて
`NO_DISCREPANCY_DETECTED`、科学判定は `UNDETERMINED`。**
係数閾値を修正したSOURCEで、計画した1,928 trajectory・条件付きsignal・仮想±1 shotを生成した。
事前固定した診断閾値を超える不一致は検出されなかった。
有限Nでの診断であり、周辺平均の同等性、sampler全law、全18round成功率の保証ではない。

## SOURCE・条件・修正範囲

科学実行SOURCE：`d0b1da1b04c266a1f7abf8e43966ca6e289a8695`。
[修正内容と実行計画](../rpe_fresh_iid_measurement_a_threshold_fix_20261103/SOURCE_FIX_AND_EXECUTION_PLAN.md)、
[実行runner](../rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py)、
[adapter](../rpe_fresh_iid_measurement_a_preparation_20261103/fresh_iid_adapter.py)を変更せず実行した。
SOURCE branchとは別の `codex/rpe-fresh-iid-measurement-a-run02-20261103` で結果を記録する。

| 項目 | 今回の固定条件 |
|---|---|
| Model/geometry/basis/rank | linear H4、隣接1.00 Å、STO-3G、保存legacy DF rank12 |
| State/sector | 候補Bの保存physical ground state、8 system qubit、4電子sector dimension70 |
| PF/RTE | Partial L_D=3、単一δ=0.02、r=1、K=2、extract_identity_phase |
| Coefficient threshold | coefficient_atol=1e-12、保存signalの条件を明示継承 |
| Round/axis/N | m0/q1はcosine/sine各510、m3/q8は各454 |
| Mean source | V2 commit78c49d0b5f1d71e63b5959e78813237923bbc648の保存attenuated event-mean |
| Sampler/source carrier | 4591b8d75a2a6a55c74a51225f0c61648023c8ee、既存sample_rte_events等を再利用 |
| State source | B result commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41 |

先行[run01](../rpe_fresh_iid_measurement_a_execution_20261103/REPORT.md)は旧runnerの閾値省略でsampling前に停止した。
そのSOURCE `09237b48ea22ad10d860e67e4a2c1465aad65e41`と原RESULTは変更していない。
今回の明示指示は、修正版の実行と、意図しない技術的中断時の修正・再実行を許可するものとして適用した。
科学条件・seed・N・診断規則を結果に合わせて変更する許可には読み替えない。
今回は中断なく1回で完了し、技術的retry・resume・追加samplingは0。

保存N/α/β/εは変更していない。αはq1=2.8897922699779505e-8、q8=2.2180178685403194e-7、
β=(.014050012886294481,9.899584668737594e-8,.38594988811785885)、
coordinate εはq1=.266146579872968、q8=.26589319587097715。
公開namespace `rpe_fresh_iid_measurement_a_20261103_v1_public`、PCG64、既存のkey/role別seed派生を保持した。
各shotでfresh trajectoryを生成し、別の測定streamから`Pr(+1|ω)=(1+μ_axis(ω))/2`で仮想測定した。
異なるseedから数学的独立性を証明したとは主張しない。
未補正の条件付き測定平均を使用し、normalization/scalar位相を二重適用しない既存規約を保持した。

## 完了した出力と診断

原記録：[RESULT.json](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run02/RESULT.json)、
[shots.jsonl](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run02/shots.jsonl)。
以下はRESULTからの転記で、新しい標本や追加の診断を作っていない。精密値は原JSONを参照。

| q / axis | N | 保存target μ | trajectory条件付き平均の標本平均 | 仮想測定の標本平均 | 診断半幅h |
|---|---:|---:|---:|---:|---:|
| 1 / cosine | 510 | .998925571 | .998993206 | 1.000000000 | .155598605 |
| 1 / sine | 510 | .043308194 | .043043066 | .011764706 | .155598605 |
| 8 / cosine | 454 | .939502685 | .937639314 | .955947137 | .164916026 |
| 8 / sine | 454 | .339352237 | .340251195 | .277533040 | .164916026 |

各軸で「trajectory平均対保存mean」「測定標本平均対保存mean」
「測定標本平均対trajectory条件付き平均」の3比較を行い、全12比較が固定閾値内だった。
診断family η=.05、`h=sqrt(2 log(2×12/.05)/N)`。
前2比較には計画した数値許容差1e-12を加える。これは既存RPEのα/β配分を置き換えない。
roundoffによる平均座標の調整は全軸0件。

**標本平均は理論上の周辺平均でも厳密な測定期待値でもない。**
診断半幅は約.156/.165と広く、RESULTが事前定義に従って記録した高確率検出の十分な差は
約.311/.330超である。小さな系統差や統計的同等性を認定する検出力はない。
このため、不一致なしでも科学statusを`UNDETERMINED`に保つ。
これは技術的な中断ではなく、実行前に固定した正常完了時の科学判定である。

## 実行・数値identity・資源

```bash
python docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py --run --output-dir artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run02
```

exit0、記録wall time約4.318秒。数値形式はfloat64/complex128、
Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0。
32 available CPUに対して32 worker、spawn、job内BLAS/OpenMP等を各1 threadとした。
開始前空きRAMは約52 GiB。GPU utilityはあるがdriverと通信できず、準備済みCPU経路を使用した。
workerあたりcontext arrayは17,249,120 bytes。これは実際のprocess RSSの測定値ではない。
sector operatorを一度構成し、basis cacheは11件、最大basis sector leakageは数値0.0だった。
serialとの速度比較は実施していない。

| 処理 | 完了数 |
|---|---:|
| DF preparation / finite-RTE config | 各1 |
| retained signed sector operator | 218 |
| fresh trajectory / 条件付きsignal / 仮想shot | 各1,928 |
| event位置 | 8,284 |
| 軸 / 診断比較 | 4 / 12 |
| 新ground-state solve / controlled wrapper build / compile / 価格評価 | 各0 |

保存H・B RESULT/stateのbyte照合、既存12 sourceファイル、公開role付き3,856 seed key hashを実行前に確認した。
実行時のpreparation hashは保存値
`89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`と一致し、218 componentを再現した。
state fingerprintは`a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。
物理H基底状態をBから再利用し、新しいstate solveはしていない。
旧SOURCE内のDF preparationはGaussian gate定義を構成するが、controlled evolution/Hadamard回路の新build/compileは行っていない。

出力SHA256：

- RESULT.json（8,793 bytes）：`4ea0e361ff8c63f875e31618fc78105efef0eae4a08230f8107e0527978fafde`
- shots.jsonl（2,308,336 bytes）：`b47605ad97ed3de8eb6f013954ce804d410f7e25ade06322e6dbad935c881c54`

raw行の並列完了順は保存のまま保持する。key別のshot indexが再現性の単位である。
出力確認は件数・keyの完全性/重複なし・±1形式・event位置数・JSON/byte identityを対象とし、
保存標本から別の科学診断や分布再評価を追加しない。
SOURCEは準備時の人工回帰3＋既存18 tests PASSから変更していない。
人工PASSと今回の有限標本診断を、独立CI・外部再現・重要科学レビューと同一視しない。

## 保全と次段

旧A SOURCE/run01、B SOURCE/RESULT/state、V1/V2/V3/R1、他研究のworktree/jobを保全した。
V2の`CONDITIONAL_SUCCESS`、R1費用順位`UNDETERMINED`、Fullの半径・欠測fingerprint、
M06-F/R2/R3のSTOPを変更していない。
今回の結果は、このH4/legacy DF12/L_D3/δ=.02/q1,q8/r1,K2/保存physical stateに限る。
全law・稀eventの一様保証、Full、全18round、state preparation費用、compiled correctness、総費用は未評価。

計画した処理は完了したので追加実行は行わず、結果・公開内容の照合後に停止する。
**次段案**：ユーザー/GPTが次に必要な許容mean差と対象roundを指定し、
Codexがその科学的問いに対応する検証計画を具体化する。
検出力不足を理由に今の標本を増やすことや、全18round検証へ自動移行することは今回の範囲に含めない。
