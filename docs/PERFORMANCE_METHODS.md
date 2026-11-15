# 性能対策の再利用索引

既存の大規模計算の手法集は固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を参照する。本C branchへその科学runnerを取り込んだ意味ではない。

## 保存RPE結果の読取

`src/trottertracks/saved_rpe_cost_catalog.py:Reader.document` は既存Git blob readerを使用し、固定commit/pathをprocess内でcacheする。同じSUMMARY/INPUTS/価格表が複数HANDOFF・配分から参照されてもJSONの再読取を省く。seed/計算順/価格値は変更せず、適用対象は保存JSONのmetadata処理だけ。

既存B2の保存summarizer同様、Python process内の整数文字列桁上限を解除し、exact MSEの長い整数を損失なく読み書きする。共有環境設定や科学runtimeを変更しない。

[再利用・検査・実行範囲](research/h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)と[検収](research/h4_saved_rpe_cost_catalog_20261116/TESTS.txt)を参照。今回の固定10入力の実処理時間はCATALOG.jsonに保存。人工＋公開値の読取で確認したが、速度倍率や将来の大入力性能は測定していない。
