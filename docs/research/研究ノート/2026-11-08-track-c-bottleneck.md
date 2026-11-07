# Track C：資源ボトルネック検証

固定H4のD内部改善後の差を分解し、D/RTEのRZ比43/57%、m16/m17約91%を確認。always-partial QRは共有を壊して悪化したため棄却し、singletonのみ部分QRを採用した。既存step/resolutionとαの共同選択後、独立価格holdoutでPR RZ1.79060e12/CX3.64850e11。出発点PR比46.52%/49.44%減だが、同じα最適化後の強Fullに対しRZ1.96倍/CX1.27倍で、優越を主張しない。

RTE basisとnormalizationの両方が重要。D/R境界whole確認のcount削減は0。RTE law/return aggregationを変更せず、全負結果・source/native区間・固定policy独立holdoutを保存した。共同isometryやTrack Bとの担当調整は未実行の次仮説。元exact ground利用可能を維持。詳細GPTレビューなし、mandatory STOP。

[REPORT](../track_c_bottleneck_20261108/REPORT.md)、[FINAL](../../../artifacts/track_c_bottleneck/2026-11-08/FINAL.json)。
