# H4 q4_sample00：予測・保存P0・残差

[科学条件・SOURCE・資格](REPORT.md)。Strong symmetric-directional／whole-path Referenceの各X/Y、計4 wrapper。予測commit6049a71bddbf6cd24fdac5c78073d54588ccef51を保存P0参照前に固定した。下表の残差は予測−保存P0。28値すべて完全一致だが、有限fixtureの一致であり一般P0は未資格。

| 方式 | 軸 | 資源 | Rawからの予測 | 保存P0 | 残差 |
|---|---|---|---:|---:|---:|
| Strong | X | RZ count | 21020 | 21020 | 0 |
| Strong | X | CX count | 8024 | 8024 | 0 |
| Strong | X | RZ depth | 5013 | 5013 | 0 |
| Strong | X | CX depth | 3874 | 3874 | 0 |
| Strong | X | Total depth | 11671 | 11671 | 0 |
| Strong | X | Circuit size | 42839 | 42839 | 0 |
| Strong | X | Logical qubits | 9 | 9 | 0 |
| Strong | Y | RZ count | 21020 | 21020 | 0 |
| Strong | Y | CX count | 8024 | 8024 | 0 |
| Strong | Y | RZ depth | 5013 | 5013 | 0 |
| Strong | Y | CX depth | 3874 | 3874 | 0 |
| Strong | Y | Total depth | 11671 | 11671 | 0 |
| Strong | Y | Circuit size | 42839 | 42839 | 0 |
| Strong | Y | Logical qubits | 9 | 9 | 0 |
| Reference | X | RZ count | 21239 | 21239 | 0 |
| Reference | X | CX count | 7336 | 7336 | 0 |
| Reference | X | RZ depth | 2571 | 2571 | 0 |
| Reference | X | CX depth | 1590 | 1590 | 0 |
| Reference | X | Total depth | 5686 | 5686 | 0 |
| Reference | X | Circuit size | 42562 | 42562 | 0 |
| Reference | X | Logical qubits | 17 | 17 | 0 |
| Reference | Y | RZ count | 21239 | 21239 | 0 |
| Reference | Y | CX count | 7336 | 7336 | 0 |
| Reference | Y | RZ depth | 2571 | 2571 | 0 |
| Reference | Y | CX depth | 1590 | 1590 | 0 |
| Reference | Y | Total depth | 5686 | 5686 | 0 |
| Reference | Y | Circuit size | 42562 | 42562 | 0 |
| Reference | Y | Logical qubits | 17 | 17 | 0 |

4 wrapperとも全7指標一致、部分一致0、不一致0、native lowering未対応0。保存compiledの厳格reader／T0による再計数も公開pricesと一致した。X/Yは個別に検査したが、独立した別path標本とは扱わない。

Reference−StrongはX/YともRZ +219、CX −688、RZ depth −2442、CX depth −2284、total depth −5985、size −277、logical qubits +8。これらは一つの固定full wrapper比較で、全law平均や全RPE費用ではない。追加register・CSWAP・χ・cutによる影響を含む列を使い、局所節約の加算やRZ→T換算は行わなかった。

最大RZ角差はStrong5.3290705182007514e−14、Reference1.9539925233402755e−13 rad。wireごとのopcode／wire順は全4で一致。完成wrapperのglobal phase差mod2πはそれぞれ4.4945593913325865e−9／4.561506727895903e−9 rad。これは価格残差ではなく、相対χの不一致とも判定しない。新しい全入力correctnessは未実施で、新PASS／FAILを付けていない。

[機械結果](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/COMPARISON.json)／[固定予測](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/PREDICTIONS.json)／[位相・順序診断](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/DIAGNOSTICS.json)。予測値・元残差を保持し、アルゴリズム／科学条件／閾値の変更は0。
