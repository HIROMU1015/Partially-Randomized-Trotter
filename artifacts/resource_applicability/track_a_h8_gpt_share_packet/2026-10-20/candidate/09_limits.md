# 不足証拠・匿名化による制限

`09_missing_evidence.csv`は「原本で確認済み／公開資料のみ確認／未取得／確認不能」を区別する。確認済みはローカル原本照合の範囲を意味し、受領者が全項目を独立再計算できるとの意味ではない。

- 原compiler profileの現物は未取得。継承identifierと現在のcompiler設定・runtimeの一致から、原profile bytesや旧出力の完全同一性を補わない。
- 保存PR wrapper/final circuitは未取得。event digest・準備hash・SOURCE・metricの来歴は残るが、元回路からのfingerprint再導出やfinal角度/T費用の検証はできない。新規回路再生成・compileは今回の範囲外。
- native identityをaliasにし、grantの内部path・CPU割当・承認本文を除いた。親子graph、exit/reap、technical digestとscopeの一致は確認できる。実process identityと元承認本文・署名の真正性は再認証できない。
- seed値、raw event列、および公開コードからseed復元を可能にする導出用digestを共有しない。旧除外4,136、A128を含む統合除外5,000、新96登録、旧探索592/確認192行のcoverageの原本監査結果と、その検証条件は保持する。具体的なcollision検査・seed階層・event digest計算を受領者が独立に再実行することはできない。
- input/provider/runtimeの原bytesを共有しない。expected hash・役割・件数・前回検査条件を保持する。原bytes照合や科学arrayの意味的正当性は受領者がこのpacketだけで再検証できない。
- 元raw結果全体を共有しない。公開済みscalar192軸と保存比較表、必要な監査field、原SHAのlinkを保持する。平均/SD/SE/費用は再集計せず保存文字列を転記した。
- JSONとSHAの自己整合性は、独立した署名・第三者認証・immutable CIの代わりにならない。SOURCE中のguardと実行receiptを区別する。公開PF receiptは選択済みPFに限定し、PR workerの証拠に置換しない。

完全な原本再検証を求める場合、対象file・受領者・共有先を固定した追加承認が必要。seed原本・private raw・内部pathを含む原本索引・未加工grant/receiptは添付しない。詳細科学レビュー、最終evidence freeze、追加科学計算は未開始。
