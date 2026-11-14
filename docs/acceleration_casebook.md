# 高速化・並列化の実装事例集

過去に「何が遅く、何を変え、どこまで効果を確認したか」を探すための一覧。似た実装の参考・再利用先として使う。各事例の詳しい説明や測定記録は元REPORTを参照する。

## 一覧

| 日付・事例 | 遅かった処理 | 実装した工夫 | 記録された効果と測定範囲 | 再利用先・詳細 |
|---|---|---|---|---|
| 2026-11-14：total RPE DP | 状態×countの多倍長整数演算、繰返しの約分 | 同じ出力状態への質量を先に合算／count batchのprocess並列化／有理数の約分を遅延 | 人工2,191,510遷移で80.936秒→3.782秒、約21.4倍。全output一致 | [`rpe_total_dp_parallel.py`](../src/trottertracks/rpe_total_dp_parallel.py)／[REPORT](research/rpe_total_dp_acceleration_20261114/REPORT.md)／[API](research/rpe_total_dp_acceleration_20261114/HANDOFF.md) |
| 2026-11-14：candidate-ID parser cache | 同じPython SOURCEを呼出しごとにparse・exec | SOURCE完全bytes＋絶対pathに束縛したparse済みcallableをcache | B経路の人工microbenchmark中央値6.770 ms→0.084 ms/回、約80.2倍 | [`pf_pr_rpe_pipeline.py`](../src/trottertracks/pf_pr_rpe_pipeline.py)／[固定REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3a27f99feef2073341dfd0b980ae6b6a23906e9f/docs/research/pr_resource_technical_fixes_20261114/REPORT.md) |
| 2026-11-13：R3 decoder | 各trial・roundで全q branchを列挙し、scalarを繰返し呼出し | 最近傍と次点を定数個の候補から求める／batch hookで既存callerへ接続 | 人工13round・512trialのB経路16.7995秒→0.025279秒、約665倍 | [`rpe_decoder_fast.py`](../src/trottertracks/rpe_decoder_fast.py)／[固定REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7f1f9053dfbbb81646ac9e483a0354d01cbec535/docs/research/rpe_decoder_r3_20261113/REPORT.md) |
| 2026-11-10：命令列stream bridge | 大きな中間wordのlist化、同じ1Q runの再合成、不要な再計数 | 1Q出力をCX処理へ直接stream／exact run cache／不要な中間Counter削減 | 保存q8のnative処理区間で約2.09〜2.16倍、peak RSS約53.2〜54.2%減。最終word・phase・資源一致 | [`track_d_stream_bridge_fast.py`](../src/trottertracks/track_d_stream_bridge_fast.py)／[固定REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6bd6c5dedf27601b0c3b50877653ad43d04cd17/docs/research/track_d_stream_bridge_fast_20261110/REPORT.md) |

上の値は各資料の保存記録を転記したもの。異なる問題の倍率を直接比較したり、別処理の所要時間へ外挿したりしない。全過去研究の網羅一覧ではなく、確認できた事例から随時追記する。

## 各事例の再利用メモ

### total RPE DP：集計順を変えて算術量を減らす

- **入口**：`Accelerator(workers=..., outcome_batch=...)`。既存Q3 callerへは`installed(...)`を使う。
- **守ったこと**：同じbinary64 decoder、signed zero、前状態の選択順、finite scope、残余の幾何上界。質量・重みは整数／有理数のままで集計する。
- **効果の内訳**：伝播だけの改善では約1.3倍だった。profilingで`math.gcd`の負荷を確認し、残余・終端モーメントの約分も遅延して上表の改善を得た。workerを増やすだけでは頭打ちになった。
- **検証・SOURCE**：[`check_rpe_total_dp_parallel.py`](../tests/check_rpe_total_dp_parallel.py)、[benchmark](research/rpe_total_dp_acceleration_20261114/BENCHMARK.json)。実装`9e294778266c9c867a8b1b98e040c2e94b1cb635`、元kernel `710aa69b663ef3313439762d25c80a652a03948f`。
- **実適用の記録**：[Pauli固定2診断](research/pauli_total_q3_accelerated_results_20261114/REPORT.md)は各8 worker・2構成並列で約235秒。ただし旧版は途中停止したので、実H4の完走同士の速度比ではない。停止済みSOURCEの履歴と、許可された切替を保全した。

### candidate-ID parser：結果ではなく変わらない定義をcacheする

- **入口**：`candidate_id`内の`_candidate_id_parser`、固定private module向けの`cache_legacy_candidate_parser`。
- **守ったこと**：毎回SOURCEを読み、完全bytesと絶対pathでcacheを区別する。candidateごとの結果や科学的資格はcacheせず、各入力に元関数を実行する。同じsize/mtimeでもSOURCEが変われば再parseする。
- **検証・SOURCE**：[`check_candidate_id_cache.py`](../tests/check_candidate_id_cache.py)。主実装`df71b4857d6cebee8688ab5c8cad5e2cfcd00787`、報告`3a27f99feef2073341dfd0b980ae6b6a23906e9f`。上表は同一process・200回×5測定の中央値であり、全pipelineの倍率ではない。

### R3 decoder：全探索を定数候補にし、callerごとbatch化する

- **入口**：`reconstruct_batch`、`decode_sampled_counts`と既存`batch_decoder` hook。
- **守ったこと**：scalarのtie選択、`math.atan2`、countのexact整数和→binary64変換、zero/tie status、callerの乱数消費順を保持した。
- **注意したこと**：旧vectorizedの完全tie規則はscalarと異なる。互換APIのexact-tie行には旧全q処理のfallbackがあり、全入力で最悪O(1)とは主張しない。検証domainはM=0〜12。
- **検証・SOURCE**：[`check_rpe_decoder_fast.py`](../tests/check_rpe_decoder_fast.py)、[固定HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7f1f9053dfbbb81646ac9e483a0354d01cbec535/docs/research/rpe_decoder_r3_20261113/HANDOFF.md)。現在は決定論的riskが主経路で、旧MCを使ったこの事例はdecoder/caller高速化の履歴として参照する。

### stream bridge：中間データと重複処理を減らす

- **入口**：`optimize_native_stream`、[`track_d_stream_resources.py`](../src/trottertracks/track_d_stream_resources.py)。
- **守ったこと**：wire/opcode/float64全bitが一致するrunだけ再利用し、1Q flush順・phase加算順を保持する。cache missだけでなく、全logical occurrenceのphaseを会計する。
- **測定の範囲**：保存H4・LD2・q4/q8のStrong/Reference X/Y、計8 wrapper。上表の倍率はnative入力以降の処理であり、raw読込・lowering・full P0を含む全体倍率ではない。
- **検証・SOURCE**：[実装と条件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6bd6c5dedf27601b0c3b50877653ad43d04cd17/docs/research/track_d_stream_bridge_fast_20261110/IMPLEMENTATION.md)、[性能記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6bd6c5dedf27601b0c3b50877653ad43d04cd17/docs/research/track_d_stream_bridge_fast_20261110/PERFORMANCE.md)。最適化実装`53f5d69e5c52159b55e6785b7e26256354fac874`、paired測定`97bf92583d6b03ace5c9173e8e2044a2469aa381`。

## 今後の追記

高速化・並列化の実装を終えたら、一覧に1行と、必要なら短い再利用メモを追記する。既存REPORTやテスト・benchmarkへのリンクを使い、同じ説明を別の長い報告書へ作り直す必要はない。

```text
日付・事例名：
遅かった処理 → 実装した工夫：
効果と測定範囲：（未測定なら「未測定」）
再利用先：module/API、実装SOURCE
詳細・同値性確認：既存REPORT、test、benchmarkへのリンク
適用時に注意した条件：
```

事例集を埋めるための追加benchmarkや科学計算は行わない。過去のworker数・速度・適用条件は当時の記録であり、新しい処理にそのまま固定しない。別branchの成果は固定commitへのリンクを使い、現在のcheckoutに取込まれていることと区別する。
