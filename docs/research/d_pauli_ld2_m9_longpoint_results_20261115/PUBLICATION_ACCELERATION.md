# m9追加高速化：稼働科学SOURCEを保持した並列公開照合

利用者の追加指示を受け、段階別実測と既存手法索引を確認した。
科学SOURCE650dでは独立2 pathを別processで並行処理し、event_matrixの完全key memoと
同一byte fingerprint writerを適用済み。各pathのsamplingは約24秒、full-space oracleは約36〜39秒。
一方、構築約378〜383秒、X/Y compileは各約436〜463秒、状態vector検査は各約183〜267秒を要する。
固定core内のX/Yとvector検査は逐次であり、最大限の並列化が済んだとはいえない。

既存Full-Hの`codex/d-full-h-p2-caller-prep-20261114:scripts/publish_full_h_d7_holdout.py`
と`docs/PERFORMANCE_METHODS.md`を読んだ。X/Y先行並列compileやvector backend差替えは、
稼働中の封印済みcore・phase・元compiled-vector gate経路への変更になるため、その場で注入しない。
現在の正常な2 workerを停止・再実行せず、sampling/compile上限と出力を保持する。

今回追加適用するのは、同じ既存process-pool方式による**保存raw2 pathの並列公開照合**。
`scripts/publish_pauli_ld2_m9.py` はraw SHAからgzipを読み、bytes/hash/indexを確認し、
元13abのreceipt checkerと650dのaxis/SOURCE/seed/metrics/compiler照合を各workerで実行する。
大きなevent配列を親へ転送せず、受理済みprices/scalarと検証metadataだけを返す。
index順へ戻した後の平均・price_records・sidecar/evidence生成は元の式・経路を維持。
元650dの`--admit`でcanonical guardと2行exact比較を行う。実行SOURCEと公開後処理SOURCEを区別する。
この入口は科学jobを起動・停止・再開せず、sampling/compile/fitは0。

人工2 test PASS：[PUBLICATION_TESTS.txt](PUBLICATION_TESTS.txt)。実2 processのbarrier同時実行、
path順復元、2軸のexact平均/比較を確認。raw bytes/hash/index、軸/SOURCE/seed不一致とreceipt欠落は拒否。
人工入力だけであり、実H4の速度比や価格精度を認定しない。実対象の公開実測は最終REPORTに記す。

主な残余は元Qiskit Statevector.evolveの逐次処理と、path内X/Y逐次compileである。
今回完了した同一m9 cohortを速度測定だけのために再sampling/compileしない。
