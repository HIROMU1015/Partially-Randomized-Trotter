# RPE候補A：1回の実行はpreparation検査で停止

2026-11-03 JST。**`INTERRUPTED_OR_INVALID_MANDATORY_STOP` / 科学判定 `UNDETERMINED`**。
承認された固定SOURCEを1回実行したが、DF preparationの照合で例外終了した。
fresh trajectory・条件付きsignal・仮想±1 shotは生成していない。
保存meanとの整合／不整合を判定する標本はなく、科学的な不一致を検出した結果ではない。
再実行・resume・別namespaceへの移行は行わず、mandatory STOPを維持する。

## SOURCE・入力・実行範囲

- 科学実行SOURCE：`09237b48ea22ad10d860e67e4a2c1465aad65e41`。
- [固定計画](../rpe_fresh_iid_measurement_a_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)、
  [runner](../rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a.py)、
  [adapter](../rpe_fresh_iid_measurement_a_preparation_20261103/fresh_iid_adapter.py)を変更せず使用した。
- linear H4、隣接1.00 Å、STO-3G、保存legacy DF rank12、8 qubit、4電子sector dimension70。
  Bの保存physical ground state、Partial `L_D=3`、単一時間幅 `δ=0.02`、`r=1,K=2`。
- 計画はm0/q1でcosine/sine各510、m3/q8で各454、計1,928 trajectory・仮想shot。
  保存N/α/β/ε・保存attenuated event-mean・公開seed namespace・診断閾値は変更していない。
- Bの結果・保存stateはcommit `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`、
  meanはV2 commit `78c49d0b5f1d71e63b5959e78813237923bbc648`の固定4座標。
  sampler等はcarrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の既存実装を再利用するSOURCEである。

実行専用branchは `codex/rpe-fresh-iid-measurement-a-execution-20261103`。
新しい出力directoryに、次のコマンドを1回だけ実行した（Python 3.11環境）。

```bash
python docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a.py --run --output-dir artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01
```

実行前のbyte照合でH snapshot、B RESULT/state、既存12 sourceファイル、準備時のcode hashを確認した。
環境はPython 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0。
実行前は使用可能CPU32、物理RAM約62 GiB・空き約51 GiB。job内だけBLAS/OpenMP等を1 threadとする既存設定を使用した。
SOURCEはsector operatorの再利用・固定index chunkの独立shot並列処理を準備しているが、
今回はworker作成より前で停止したため、実H4での並列性能・速度向上は未検証。

## 保存された停止結果と到達点

原記録：[run01/RESULT.json](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01/RESULT.json)。
SHA256：`1ea128b2bc09da2fda0dafb1686fa1139c05bbddfc67a582d338dc87896953f4`。
process exit codeは1、記録されたwall timeは約1.381秒。
例外は `ValueError: Saved preparation/component inventory mismatch`。

| 処理 | 今回の到達状況 |
|---|---|
| H・B入力のbyte照合、H hash/sector/state fingerprint・norm検査 | 通過 |
| 保存H/stateの数値読込、`L_D=3` DF preparation | 実施。preparationから戻った後の照合で停止 |
| finite-RTE config、sector operator/half operatorの構成 | 未到達 |
| worker作成、RTE event sampling、trajectory state-action | 未到達、0件 |
| 仮想±1 shot、4軸の統計診断 | 未到達、0件 |
| 新ground-state solve、controlled wrapper build、compile、価格評価 | 各0（RESULTにも記録） |

run01の生成物は`RESULT.json`だけで、`shots.jsonl`は存在しない。
sampling等の0件は、例外位置とSOURCEの制御フロー・出力一覧から確認した。
中断recordにはcompleted/harvested shot counter自体がなく、欠測を数値counterの記録と混同しない。
同様に`context`はreturn前の例外のため記録されず、実行時のpreparation hash・component数・block数は保存されていない。
複合OR条件のどの項が実際に不一致だったかを、保存record単独から特定することはできない。

## 静的調査で確認したSOURCEの不整合

停止後はSOURCEと保存metadataの読取りだけで調査した。数値preparationの再呼出しはしていない。

1. 保存signal grid
   [`wp04_ld3_full_schedule_signal_grid_v1.json`](../../../artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json)
   の`request.coefficient_atol`は **`1e-12`**。
   同じ記録のpreparation hashは `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
   retained component数は218である。Aの計画・定数はこのhashと218を要求していた。
2. A runnerの`prepare_context()`は`prepare_df_partial_s2(...)`の呼出しで
   **`coefficient_atol`を指定していない**。
   [`df_partial_s2.py`](../../../src/trotterlib/df_partial_s2.py)の既定値は **`0.0`**である。
3. 同関数は`coefficient_atol`をpreparation hashのpayloadに含め、tailの係数除去にも使用する。
   保存signal側の条件とrunnerの要求は整合していない。
   [`finite_rte_signal_validation.py`](../../../src/trotterlib/finite_rte_signal_validation.py)の既存signal生成経路は
   `coefficient_atol=1e-12`を受け渡すため、Aのadapterでもこの引数の継承が必要だった。

これは準備実装での引数指定漏れである。準備時の人工18 tests PASSは、
実H4 preparationの入力条件の一致を検証しておらず、この漏れを検出できなかった。
一方、今回の停止はH snapshotやB stateの破損を示さず、fresh-IID lawや保存meanの科学的反証でもない。
閾値指定だけの修正ですべての照合が通るか、修正後の数値的整合性が成立するかは未検証である。
実行時の未保存値を得るための再計算、gateの無効化、hash/component条件の緩和は行っていない。

Bの固定request runnerも引数省略であり、その局所条件付き回路照合の結果は原条件のまま保持する。
Bの数値整合を、保存signalと同じ全分布での周辺平均検証へ読み替えない。

## 保全・次段

原A SOURCE・準備記録、B RESULT/state、V1/V2/V3/R1のSOURCE/判定、M06-F/R2/R3のSTOPは変更していない。
Fullの半径・fingerprint等の未保証事項も残る。V2は`CONDITIONAL_SUCCESS`、R1費用順位は`UNDETERMINED`のまま。
今回、全18round成功率、厳密半径、compiled correctness、総費用を評価していない。
旧SOURCE・rawを変更せず、新しい停止結果と最小限の索引/statusだけを追加する。
本記録は実行者の技術調査であり、独立した重要科学レビューを実施したものではない。

保存後の技術確認で、原RESULTのSHA256、固定code hash、保護対象の差分なし、
旧manifest全項目の保持、JSON読込、報告内リンク、公開情報を確認した。`git diff --check`も通過。
この確認はbyte/metadata/AST読取りのみで、H4数値処理やRNGを呼び出していない。

固定計画は「成功した実行・不一致・未判定の全てでmandatory STOP」、
「得られた部分rawは残すが部分推論・自動再開を行わない」と定めている。
今回はこの1回を消費した。標本が0件でも同じSOURCEのretryとして扱い直さない。

**提案する次段**：Codexが保存signalの`coefficient_atol=1e-12`を明示継承する修正版SOURCEと、
実数値を呼ばない引数受渡しの回帰テスト・新しい1回の実行計画を準備する。
その後ユーザーが新SOURCEでの別の1回実行を明示許可する。修正版の科学実行は今回開始しない。
