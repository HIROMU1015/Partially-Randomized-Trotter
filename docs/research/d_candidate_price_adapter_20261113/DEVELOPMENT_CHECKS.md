# 開発中の技術確認

初回は人工fixtureがBの名前付きpaper mapperを別callbackで置換しようとして15件が同じ理由で失敗した。Bの契約を変えず、人工専用の明示mappingとして修正した。原記録：`/tmp/candidate_adapter_initial_tests.txt`。

次の回は14 PASS、旧C1との比較だけがcontextのfingerprint_context属性不足で失敗した。既存use_fingerprint_reuseを接続して修正した。原記録：`/tmp/candidate_adapter_tests_02.txt`。

以後、元C1のhead生成bodyを直接使う照合、Aの実metadata helper、軸単独・逆順、signed half-timeと高次数の人工event、passive import等を追加した。33 PASS、追加後35 PASSを経て最終記録はTESTS.txtへ保存する。過去の科学runやSTOPを変更・再実行していない。
