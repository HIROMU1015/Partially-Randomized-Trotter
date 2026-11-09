# 保存q8_sample00：全28資源の予測・保存P0・残差

[科学条件・SOURCE](REPORT.md)。中央cut4、C1 OFF、元H厳密ground仮定・準備費除外。sample ID昇順先頭を価格非依存に選定。予測commit0cd6de18fea06c858592f60d3b310e67eb759ce8を固定してから保存P0を参照した。残差＝予測−保存P0。

| 方式 | 軸 | 資源 | Rawからの予測 | 保存P0 | 残差 |
|---|---|---|---:|---:|---:|
| Strong | X | RZ count | 42316 | 42316 | 0 |
| Strong | X | CX count | 16204 | 16204 | 0 |
| Strong | X | RZ depth | 10479 | 10479 | 0 |
| Strong | X | CX depth | 7996 | 7996 | 0 |
| Strong | X | Total depth | 24370 | 24370 | 0 |
| Strong | X | Circuit size | 86571 | 86571 | 0 |
| Strong | X | Logical qubits | 9 | 9 | 0 |
| Strong | Y | RZ count | 42316 | 42316 | 0 |
| Strong | Y | CX count | 16204 | 16204 | 0 |
| Strong | Y | RZ depth | 10479 | 10479 | 0 |
| Strong | Y | CX depth | 7996 | 7996 | 0 |
| Strong | Y | Total depth | 24370 | 24370 | 0 |
| Strong | Y | Circuit size | 86571 | 86571 | 0 |
| Strong | Y | Logical qubits | 9 | 9 | 0 |
| Reference | X | RZ count | 42294 | 42294 | 0 |
| Reference | X | CX count | 14620 | 14620 | 0 |
| Reference | X | RZ depth | 5068 | 5068 | 0 |
| Reference | X | CX depth | 3108 | 3108 | 0 |
| Reference | X | Total depth | 11235 | 11235 | 0 |
| Reference | X | Circuit size | 85157 | 85157 | 0 |
| Reference | X | Logical qubits | 17 | 17 | 0 |
| Reference | Y | RZ count | 42294 | 42294 | 0 |
| Reference | Y | CX count | 14620 | 14620 | 0 |
| Reference | Y | RZ depth | 5068 | 5068 | 0 |
| Reference | Y | CX depth | 3108 | 3108 | 0 |
| Reference | Y | Total depth | 11235 | 11235 | 0 |
| Reference | Y | Circuit size | 85157 | 85157 | 0 |
| Reference | Y | Logical qubits | 17 | 17 | 0 |

4 wrapperとも7/7、完全一致28/28、部分一致／資源不一致0。native lowering未対応は遭遇せず、保存compiledのT0再計数もpricesと一致。X/Yを別pathへ倍増して標本統計を作っていない。

Reference−StrongはX/YともRZ −22、CX −1584、RZ depth −5411、CX depth −4888、total depth −13135、size −1414、qubits +8。追加register／二CSWAP cascade／χ／readout／cut融合損失を含むfull wrapperの比較。節約の二重控除、RZ→T、topology-free depth→実時間の換算は0。

これは固定一pathでの有限P0対応。全q／全PR law／母平均／全RPE費用や新しいH4作用保証ではなく、一般P0_UNQUALIFIEDを維持する。価格合わせのrule追加／補正定数／閾値緩和／fixture変更は0。

[固定予測](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/PREDICTIONS.json)／[全比較](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/COMPARISON.json)。
