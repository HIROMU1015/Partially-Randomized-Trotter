# H2 m5の位相保持・段階6までの実行

旧初期化STOPと費用取得STOPは不変。利用者の段階1〜6包括承認に基づく別SOURCE/契約とする。固定Qiskit1.3.0のparameter付き翻訳でnumeric scalarが加算されない経路を人工再現し、値0のsymbolic scalarを残すprocess私有処理を採用する。gate/angle/wire stream一致を毎翻訳passで検査し、raw1e-7・他1e-9を維持する。信号101はSHA/生成関数AST同一性で再利用、PRは新cohort、旧結果とpoolしない。人工24件PASS。実行結果は後続receiptへ記録する。H4/H6 pilotの再実行、段階7、AD03は別判断。
