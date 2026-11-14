# 並列化・高速化の再利用

## PR-DF P2/m3/m4：context共有、path並列、同一eventからS/C

保存H4 DF10・PR-DFの新ZZ経路では、[compiled_price_fast](../src/trottertracks/compiled_price_fast.py) の構造数え上げ、composeの回転再利用、fingerprintを使う。元実装SOURCEは `39ddd59edf54a3a901e5302207b2765ddbd03fbb`。contextは親で一回準備し、[m3 worker](../scripts/run_pr_df_ld1_m3.py) を4 pathにforkする。独立した4 pathが今回の並列単位であり、各compiler内部は元P2同様1 processとして重複poolを避ける。X/Yはそれぞれのwrapperをcompileする。

m3で実対象確認済み：[REPORT](research/d_pr_df_ld1_m3_results_20261115/REPORT.md)。m4は[専用caller](../scripts/run_pr_df_ld1_m4.py) が同じworkerを使い、保存seed表を元samplerの実行globalへ注入する。sampler本体・PCG64/CDF・event順序を変更しない。人工試験で実sampler bodyからのseed参照、fake compilerの同一event S/Cを確認した。m4の実測範囲は[専用結果](research/d_pr_df_ld1_m4_holdout_results_20261115/REPORT.md) に追記する。正常pathごとのstage進捗・所要時間・RSS、60秒周期stackを保存する。SOURCE固定後の書換えやscience retryはしない。

調査時には共通catalog `0dc1eeaee3d9d6465f18edc33d4e17ed312bd6f4:docs/PERFORMANCE_METHODS.md`（`codex/d-full-h-p2-caller-prep-20261114`）を参照した。Full-H固有のaxis workerやresume方式をPR-DFへ持ち込まず、既存P2の適合する経路を再利用した。未使用helperの存在を高速化の実行証拠としない。元SOURCE、cache identity、seed/order、lowering、wrapper/compilerを確認し、cache出所不明時のfallback compileは行わない。
