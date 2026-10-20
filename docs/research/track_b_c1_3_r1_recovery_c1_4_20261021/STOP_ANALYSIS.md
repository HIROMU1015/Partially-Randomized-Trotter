# C1-3-R1停止地点と未確定事項

分類は`C1_3_TECHNICAL_INCONCLUSIVE_STOP`。固定SOURCE `9a3503375c07037ab3e0b9428a7c0254b66c31f6`、新authorization `8c892129f0d16211ee2fd2f3eafafeba4f181bd7`で新規1回・retry 0。旧失敗runのresume/retryではない。失敗後に固定SOURCE・精度・event・許容差・runtime・入力・scheduleを変更していない。

`run_c13_r1.py:98`でselected labelに対応するreference basisを構成中、既存`reference.py::saved_basis`から`REFERENCE_LOCAL_BYTES_MISMATCH`。呼出しは6 eventのloopより前なので、accepted event query・controlled operator比較・X/Y評価は0件。`result/summary.json`は生成されていない。原tracebackは[failure.json](result/failure.json)と[stderr](result/raw_stderr.log)にある。

前段の保存state正式hashはactual=expected、state normとschedule確認は通過した。既存F1/F3の全pに対するP0〜P5がPASSし、指定精度での実law bindingはready=trueになった。ただしP0のfresh uniform independent bitsはcaller仮定、controlled-Qは条件付きexact query modelで、physical_provider_certified=false。H4 eventの角enclosure/weight/operatorを全て通したこととは異なる。

今回の新helperはstate hashの正式規約を修正した。停止した局所行列hashは**別経路**であり、今回変更していない旧[reference.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e72867b87e12d93d4fc1361f45f105febc6a94c6/docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/reference.py#L9)の処理である。同関数は保存real_hex/imag_hexから実配列を作り、`real + 1j*imag`という複素算術で行列を組み立ててraw bytes hashを比較する。

静的に確認できることとして、R1の保存basisには負のzeroのhex表現がある（例：固定R1 [basis_002.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/basis/basis_002.json)）。複素算術による組立てがsigned-zeroのbyte表現を変える可能性は、次に人工fixtureで調べる候補である。**原因を実行確認した結論ではない。** 最初に不一致となったbasis/operation、actual local hash、どのbyteが異なるかは現failureに保存されておらず、特定しない。

NPZ全体・R1各保存file・既存adapter/referenceとruntimeのbytesは固定manifestに一致した。保存入力の破損、physical operator不一致、RAの性能negativeを示す結果とは扱わない。一方、失敗を単なる位相差として無視したり、hashチェックを削除してPASSにしたりしない。

Phase Aの人工operator suiteはgate constructor由来の人工basisを使う。R1保存hexからの局所行列byte復元経路はこのsuiteの対象でなく、今回の停止で未確認部分が明らかになった。新しい人工hex fixture・診断追加・復元方法修正や科学再実行は行っていない。

失敗後は既存SOURCE/JSON/tracebackの静的読解と保存・公開確認のみ。NPZの再開封、reference行列の再構成、追加数値診断、追加event、合成/benchmarkは0。C1-4に進まず、修正範囲と新規one-shotの要否はユーザー・GPTへ返す。
