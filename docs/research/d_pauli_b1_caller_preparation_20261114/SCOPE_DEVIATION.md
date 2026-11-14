# 範囲逸脱：保存H4 P0演算子の数値再検査1回

今回許可されたのは技術実装・人工/metadata回帰であり、保存科学値の新しい数値再解析は禁じられていた。B0親suiteをunfilteredで選択した誤りにより、`Tests.test_saved_P0_bit_order_for_A` を1回実行した。これは許可範囲外の数値再検査である。

入力はA P0 `4220f3006d0d8d413e195ed948e250578513b557` の `docs/research/df_head_pauli_tail_p0_results_20261114/run01/` にある `COMMON_OPERATORS.npz`、`LD1_OPERATORS.npz`、`LD1_DICTIONARY.json`。`p0_arrays()` から保存演算子を読み、`pauli_tail_price_connection.bit_order_checks` でfull-256次元のbit順変換・演算子和・normを計算した。

実行時は `00f74f86d16cc6878e1d4c5977eb9abd9ce9c076` を基点とする技術作業中のdirty treeであり、科学実行SOURCEを事前固定していない。数値関数自体は元blobとAST一致で、prefix1の入力選択も同じであった。これを固定SOURCE科学実行と称さない。

[元ログ](OUT_OF_SCOPE_PARENT_TEST_LOG.txt) は親9テストPASS・12.042秒であり、そのうち1件が当該検査、8件が許可範囲の技術検査だった。保存されたのはassertion PASSのみで、残差の具体的数値は保存されていない。数値を回収するための再実行は行わない。結果を新たな科学的証拠・B1資格・P0の再認定に採用しない。

新しいH4 preparation、sampling、event/path生成、回路build/compile、H4 statevector、価格再集計、実fit、E[C]/N/Gは行っていない。元input/raw/結果を変更していない。範囲逸脱を直ちにチャットでユーザーへ開示した。

以後のB0回帰は当該caseを除いた安全subsetとした。default、`--metadata-only`、通常unittest discoveryでも実行しない。元検査は削除せず、将来別途許可がある場合の明示 `--include-saved-numerical` 選択として残した。B1の新テストはtoy演算子とfake compilerだけを使う。

機械可読の記録は [SCOPE_DEVIATION.json](SCOPE_DEVIATION.json)。今回の技術成果を公開後、mandatory STOPとし、追加数値検査やB1科学実行へ進まない。
