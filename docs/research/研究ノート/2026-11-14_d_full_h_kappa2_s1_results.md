# 2026-11-14：Full-H κ2・K2 S-1（13 mean、承認済み科学実行）

利用者の承認に従い、Dの固定caller（`04bead115`）・C数値SOURCE（`5cc7c07e`）・R448 reference readerを無変更で使い、S-1を1回だけ実行した。

- 実行tree：新branch `codex/d-full-h-kappa2-s1-results-20261114`（`5ccc1dfc3`から分岐。byteは同一）。
- 環境：R448と同じvenv（Py3.12.3/NumPy1.26.4/SciPy1.14.1）。
- 件数：context 1、normalizer 13、mean 13。
- 負荷：wall 4.1 s、最大RSS 593 MB。

**数値gate**：全項目PASS。
- state norm、Hermiticity・再構成、reference pair（最大9.3e−12）、closure（最大3.5e−14）、raw範囲、b>1、scalarの一回会計、ID・q/R/s。
- logB包絡：13/13。1 ULPのallowanceがなくても、すべて包絡内（最小余裕2.3e−9）。
- logBは0.476（m0）から0.49999（m≥5）。
- |corrected−reference|は6.5e−8以下。

**B packet**：`run01/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json`（13 means、UNQUALIFIED）。

**R448との関係**：referenceは共通。R・s・normalizer・raw・logBは別物なので流用していない。

**未評価**：N/MSE/price/G。P-1・S-2は未実行。

[REPORT](../d_full_h_kappa2_s1_results_20261114/REPORT.md)。mandatory STOP。
