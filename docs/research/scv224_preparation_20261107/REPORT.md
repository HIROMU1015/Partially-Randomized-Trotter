# SCV最小実装・224-shot準備：結果

2026-11-07 JST。**準備完了（人工検証限定）・科学実行未開始・mandatory STOP。**

rotation fragment群indicator、exact条件付き平均/二次moment、h1だけのintercept OLS/rank2/range/freeze、他4層λ=0、全5層加重合成、B1-OL/SCVの同じfresh評価dataによるpaired比較を実装した。
旧コードを変更せず、conditional sampler/ControlBank/FrozenCV/production_meanと既存worker・価格・journal・manager/reconcileへの注入型collectorを再利用した。

新規人工34件、再利用人工72件、**合計106件PASS**。実H4の科学的PASS・効果の証拠ではない。
[実装・再利用・テスト・制約](SCV_IMPLEMENTATION_REPORT.md)、[224-shot契約草案](SCV_224_SHOT_EXECUTION_DRAFT.md)、
[SOURCE/環境/参照hash](../../../artifacts/scv224_preparation/2026-11-07/SOURCE.json)、
[保全照合](../../../artifacts/scv224_preparation/2026-11-07/PRESERVATION.json)、
[STOP](../../../artifacts/scv224_preparation/2026-11-07/STOP.json)を保存した。

## 次回の準備状況

提案はh1 train32/軸、全5層eval(4,64,4,4,4)/軸、計224 physical shots。
評価160件を共通使用し、baseline評価費とSCV学習込み費を区別する。実支出を二重課金しない。
群選択はactual Qのν1を[1/8,7/8]に制限し、1/2に最も近い非空proper fragment群、canonical整数mask順同点処理を実装済。

**未決定**：224-shot条件の正式採択、campaign候補O2-SCV224-H4-m0-q1-20261107-v1、seed候補20261109/12-domain24子stream、range guard4、最終activation SOURCE。
実群S・ν/Σ・係数・分散・新RZ・改善はUNKNOWN。実H4群選択/平均計算も今回行っていない。

次の担当はSERVER1 Codex。次回は明示科学許可後、既存service/stream/qualified managerを薄くactivationし、SOURCE/環境/inputを公開固定して一campaignだけ実行できる設計。
実bundle/長時間exact engineでの新SCV完走、資源非干渉、全体古典timer接続は実運用時の確認事項。
数学的な精度・期待費用保証を準備済と呼ばない。科学的GO/NO-GOも行っていない。

## 実行境界と保全

今回の新しい実H4群選択・平均/moment・保存結果数値再解析・科学RNG・Born・実gate作用・実build/compile・測定は**全て0**。
N1/N2/engine再計算0、旧752-shot追加0、重要科学レビュー0。
旧397/B1339/752のdata・係数・stream・cacheを新pilotへ移していない。
元main/既存dirty/untracked、旧SOURCE/結果/STOP/grant/ledger/marker、C1凍結を保全。
独立worktree `/tmp/o2-scv-224-preparation-20261107`、専用branch `codex/o2-scv-224-preparation-20261107`。

本commitのコード・人工log・草案を公開し、remote SHA/本文/blob/treeを照合して停止する。
次回の科学計算は別途明示許可を受けるまで開始しない。承認証明書やgrantの発行は要求しない。
