# 最終SOURCEと旧回帰

Producerの旧107PASSはSOURCE `8d5135ee77e6cd06d8b2ed8aa0b9ee50a41a0e36`。
最終 `b7ad344401b10395c49e4cfa806909282859bab5`との差分は7path/144insertion/4deletionで、Canonical Seal/angle contextとtestsが中心。旧4test file/291旧SOURCE-test bytesは不変。

今回、利用者のdriver/helper人工許可後、**最終SOURCEで旧107件を全再実行：107 PASS、FAIL/ERROR/SKIP0**。
Producer新unit102/integration30も最終SOURCEで再実行し全PASS。これらを新独立224checkへ加算しない。
Canonical補完はCodec/Result/Seal/Angle独立197probe、全保存Layer B raw、fresh人工Bundleで重点検証した。
Bits/seed/STOP/Storageも今回のprobe・107回帰・22保存case・fresh失敗caseで確認し、重要契約を未実行のまま省略していない。

一時runtimeのfixture不足により初回startup失敗、次に102PASS/5ERRORを記録した。
不足3fixture/schemaと原contractを固定Git blobから補い、Producer/source/expectedを変更せず107PASSへ到達した。
中間失敗はdevelopment_recordとfinal_regression_retry1へ保持し、別の追加成功件数として合算しない。
[差分](final_source_delta.patch)、[最終結果](final_regression_retry2/results.json)、[集計](regression_summary.json)。
