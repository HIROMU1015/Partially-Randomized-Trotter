# 2026-11-14 D：保存4件・新規32件への技術改訂

利用者指示により旧Stage 0 c848bbef/1fd47847を改訂した。既存D branchを再利用し、
旧34件SOURCE・資料・最大承認範囲・STOPは保持した。
Cの公開受入branchは確認時2fa88065で、p7保存receiptの新受入・新版CELL_PLANは未公開。
したがってC新版bindingは保留し、Dの36行提案表はp7 M5/m4 X/Yのactionだけを変更した。
新規CALIBRATION20＋HOLDOUT12、保存4（Full2受入済み、p7 2待ち）。

改訂SOURCE2b9983590。select/execute_stage/run_cellsを再利用し、保存bindingをroute別に拡張。
未公開C成果を仮置きせずscience child前に停止する。追加の保存価格・trial・実行中jobの重複照合も追加。
12 group、m4/m5較正・m6holdout、X/Y別、出力保全と凍結model gateを維持。
回路・価格処理と既存stage gateの主要関数19件は旧Stage 0とAST同一。人工試験はStage17＋子3、Full親6（子2）、p7親3（子5）、C guard14がPASS。
p7正例は人工fixtureだけで、実receiptの受入や実fitではない。

指定の11/14共通運用指示の所在は未確認で照会し、本文に再掲された§10.5条件を先行実装した。
比較policy・許容差も未採択。Stage1は開始不可。
新実H4 prepare/sampling/build/compile・保存値再集計・実fit/判定・N/G・Pauli B0再実行0。

[REPORT](../d_pf_price_saved4_revision_20261114/REPORT.md)／
[C HANDOFF](../d_pf_price_saved4_revision_20261114/HANDOFF.md)／
[次段条件](../d_pf_price_saved4_revision_20261114/NEXT_EXECUTION.md)。
Cの受入公開とユーザーによる条件確定を待ち、技術成果の公開・remote照合後mandatory STOP。

## 同日追補：共通指示取得とC保存4件の正式binding

ユーザーから追加連絡を受け11/14共通運用指示を全文確認した。Cの新受入e50524f4/43c15bfa/2b945397も公開され、D SOURCE22ccf3972へ固定接続した。初版の保留状態は上記の履歴として残す。保存4件は元SOURCE/cohort/axisのrecordと一致し、新規20＋12・全36行・12モデルを維持。旧IDの物理schedule一致の検出を加え、18 metadata/人工test PASS。新しい実H4や実fit/判定/N/Gは0。policy/許容差は未採択なのでscience STOPを維持する。

[現在のREPORT/HANDOFF追補](../d_pf_price_saved4_revision_20261114/ADMISSION_UPDATE.md)。
