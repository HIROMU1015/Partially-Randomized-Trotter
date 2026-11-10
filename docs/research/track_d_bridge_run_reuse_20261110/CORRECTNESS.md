# 正確性：固定旧bridgeと保存P0の再現

[条件・SOURCE](REPORT.md)。H4のcorrectness専用16 fresh processは26.956秒で全PASS。traceのための各round全native hash計算を行い、その追加走査を性能計測へ混ぜていない。

全8対で最終nativeのopcode／wire／clbit／順序／parameterを保存するcanonical hash、各roundのword hash、input/output size／置換数／CX相殺数、最終binary64 phaseのfloat.hex、各round phase追加値のbit、MPFR160 ledger全recordが一致。signed zeroを丸める比較や許容誤差の緩和はしない。最終資源は前回保存P0とも一致した。[個別照合](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/correctness/RESULT.json)。

| Fixture・方式（X/Y各同値） | RZ | CX | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4 Strong | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| q4 Reference | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| q8 Strong | 42316 | 16204 | 10479 | 7996 | 24370 | 86571 | 9 |
| q8 Reference | 42294 | 14620 | 5068 | 3108 | 11235 | 85157 | 17 |

上表のX/Yを個別に保存した[全56値CSV](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/RESOURCE_COMPARISON.csv)で、旧新差と保存P0差は全て0。X/Yは独立path標本ではない。P0値を予測補正に使用していない。

## 人工・回帰

旧63 test methodを再実行してPASS、新7もPASSし合計70。新7には5種opcodeの長さ0〜4の全781 word、固定seedの100技術opcode列、CX連鎖がrun結合を露出する例、disjoint命令／逆向きwireのflush、classical measure blocker、signed zero／特殊角、全chunk cut、不正入力、内部走査省略の確認を含む。これはPR lawの新samplingではない。下位ケース数を70 methodに重複加算しない。[ログ](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/technical/)。

固定54保存人工raw QPYも旧pipelineを薄く再利用して照合し、word／phase／round／全7資源・保存P0が一致。rawは読取専用で新しいP0は0。[結果](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/technical/saved54.json)。人工回帰とH4の同一性検証に未解決FAILはない。価格規則や閾値の変更、結果に基づくfixture変更もない。

公開前の静的一覧checkでは、新規tracked fileを旧保全対象へ誤って含め、Unicode pathのGit quoteも考慮しない最初のassertが失敗した。基点treeとの共通部分をNUL区切りで比較するよう修正してPASSした。[初回の原因記録](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/technical/publication_check_correction.json)。engine SOURCE／科学結果は不変、科学実行のretryは0。

## 保証の限界

このcorrectnessは既存の部分bridgeの再現であり、全P0互換の証明ではない。H4の新しい全入力作用／CAR／測定演算子／reference leakage simulationは実施していない。相対χ・actual adjointは固定native列とphase会計の不変性として保全した。価格一致だけで新しい量子作用を認定しない。q1024〜4096や全sampling lawへの移送は未資格。
