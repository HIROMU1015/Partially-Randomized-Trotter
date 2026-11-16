# 性能対策の再利用索引

既存の大規模計算の手法集は固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を参照する。本C branchへその科学runnerを取り込んだ意味ではない。

## 保存RPE結果の読取

`src/trottertracks/saved_rpe_cost_catalog.py:Reader.document` は既存Git blob readerを使用し、固定commit/pathをprocess内でcacheする。同じSUMMARY/INPUTS/価格表が複数HANDOFF・配分から参照されてもJSONの再読取を省く。seed/計算順/価格値は変更せず、適用対象は保存JSONのmetadata処理だけ。

既存B2の保存summarizer同様、Python process内の整数文字列桁上限を解除し、exact MSEの長い整数を損失なく読み書きする。共有環境設定や科学runtimeを変更しない。

[再利用・検査・実行範囲](research/h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)と[検収](research/h4_saved_rpe_cost_catalog_20261116/TESTS.txt)を参照。今回の固定10入力の実処理時間はCATALOG.jsonに保存。人工＋公開値の読取で確認したが、速度倍率や将来の大入力性能は測定していない。

## H4公開成果の差分同期（C、2026-11-16）

[sync_saved_rpe_publications.py](../scripts/sync_saved_rpe_publications.py)はGitの公開ref/path/blobを用い、同じ入力docのReader cacheを再利用する。ref・入力manifest・consumer実装が不変なら集計を省略し、時刻やHEADだけの差分を作らない。raw/private directoryを走査しない。固定成果からの初回同期は本環境で約13秒（当時の公開16branch、51保存点）、科学計算の速度向上ではない。人工/保存入力で順序独立・冪等性を検査。[運用・検証](research/h4_optimizer_saved_results_intake_20261116/HANDOFF.md)。

[round03の修正](research/h4_round03_public_sync_20261116/REPORT.md)：公開selected REQUEST→EVALUATIONS hash→exact OUTCOME/CASE_INPUTS→保存価格coverageを再利用する。最終索引がなくても公開best_pointsから接続できる。configuration一致だけの偽INTEGRATEDを廃止し、同じ配分の再検出を重複登録しない。新価格予測・Q3は起動しない。
