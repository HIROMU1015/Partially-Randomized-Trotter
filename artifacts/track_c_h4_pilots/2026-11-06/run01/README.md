# Track C H4 run01：STOP

[研究報告](../../../../docs/research/track_c_h4_pilots_20261106/REPORT.md)。native compiler資格不一致により比較を停止した。全RPE総G・勝者・native成功確率は未資格。

- `SUMMARY.json` / `NATIVE_QUALIFICATION.json`：現在の科学的資格。
- `INPUTS.json` / `fixed_df_input.npz` / `common_numerical_arrays.tar.xz` 内の `common_arrays.npz`：既存H4入力のidentityと再利用データ。元H groundは基準commitのH4照合run02を再利用し、再生成していない。
- `signals/` / `BRANCH_DIAGNOSTICS.json` / `CS_DIAGNOSTICS.json` / `CP_PROCESSOR_DIAGNOSTICS.json`：20設定×18roundのSOURCE mean診断。native全回路の精度・成功保証ではない。
- `COMPILER_MINIMAL_REPRO_FINAL.json`：人工CRZ(1e-5)がCommutativeCancellationで空回路になる再現。`native_correctness_measured_profile/` は実価格profileのH4 q1 X検査と、`native_compiled_QASM.tar.xz` 内のnative QASM。
- `PARTIAL_PRICE_TABLE.tsv`：取得済み価格の既存meanの転記。資源指標は分記し、Gや候補順位を作らない。
- `RAW_BUNDLES.json`：以下のlossless tar bundleの件数・size・SHA256。展開すると従来どおりの個別JSONを得る。
- `raw_Q_prices.tar.gz`：完了63セル。`raw_T_prices.tar.xz`：完了24セル。予定gridは各306/108であり、未完了を0補完しない。
- `raw_strict_synthesis.tar.gz`：取得されたkey記録。completed Tセルで使用されたkeyと、未完了trace・prefetch・人工debugのkeyを混同しない。`ACQUISITION_SUMMARY.json` が件数と古典取得時間の範囲を記す。
- `raw_strong_D_runs.tar.xz` 内の `strong_D_runs/`：強加法baselineのD/D融合補正。Fullにはactual full価格を与える実装。
- `SOURCE.json` / `RUN_ENVIRONMENT.json`：固定由来・最終SOURCE・実行環境。初期adapter hashとdebug後の版を区別する。
- `PRICE_WORKERS_STOPPED.json` / `DIAGNOSTIC_WORKERS_STOPPED.json` / `STOP.json`：本runの科学的停止。古い実験のSTOPとは別。

runtime pickle・venv・lock・途中tempは公開対象外。元の個別raw・45MBの中間gzipもローカルに保全し、公開T rawは長いdictionaryを使うxzへlossless圧縮した。
各bundleは標準tar/gzip/xzで展開できる。展開・閲覧は科学計算の再実行ではない。
runnerはSTOP後のrun01再開を拒否する。新compiler条件や再実行は新しい明示許可・実行範囲が必要。
