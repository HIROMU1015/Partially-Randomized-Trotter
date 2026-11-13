# B → A/C/D：S3のR3 decoder切替

2026-11-13 JST。[REPORT](REPORT.md)／[人工性能記録](BENCHMARK.json)。基点B=`bf515a976f1987ad6a19c61d14689b27b0894542`、加速A=`7f1f9053dfbbb81646ac9e483a0354d01cbec535`。この資料を含む公開commitを新しい技術実行SOURCEとして固定する。過去の科学SOURCEとSTOPはそのまま。

## 選択API（次の承認済み科学実行時用）

```python
from trottertracks import rpe_decoder_statistics as S
from trottertracks.rpe_decoder_fast import decode_sampled_counts

# 既定／比較用：batch_decoder=None（省略可能）
# 高速用：batch_decoder=decode_sampled_counts
# prepared・target・law・loss・反復条件の科学採択はこのAPI選択と別。
simulation = S.simulate_prepared(
    prepared,
    repetitions=approved_repetitions,
    seed=approved_seed,
    policy=approved_iid_policy,
    allow_conditional_unqualified=approved_conditional_scope,
    batch_decoder=decode_sampled_counts,
)
summary = S.summarize(simulation['trials'], **approved_loss_and_CI)
```

今回この例を実H4へ実行していない。`simulate(rounds, *, M,h,prior,repetitions,seed,policy,batch_decoder=None)`も同じhookを受け取る。`exact_counts`・`summarize`・`trial_budget`は無変更。既存S3 metadata CLIに科学実行optionは追加していない。

hook signatureはA提供のまま：

```python
decode_sampled_counts(samples, rounds, *, M, h, prior, count_mean, decoder_source)
```

samplesはBがround順/X→Y順に生成済みの整数array。戻り値はBの全trial dict list。RNG、shot数、count変換、prior、loss/CIを別実装しない。`decoder_acceleration_source`は各trialに追加され、元`decoder_source`も保持される。実行記録では公開SOURCE commit、hook名、seed/PCG64、反復数、policyも保持する。

## 互換性・境界

- Noneの既定経路は元scalar出力と一致。batchは加速SOURCE欄を除く全trial・summaryが一致。
- scalarと旧vectorizedは完全tieで選択branchが異なり得る。S3はscalar規則。Aのvector互換APIをS3代替として選ばない。
- exact整数和→Fraction→binary64、math.atan2、zero優先、tie時None、−phase/h、prior診断のみを保持。
- `prepare`のmissing/UNQUALIFIED gateはhookより前。条件付き実行の指定は科学資格の付与を意味しない。
- 人工M0〜12の検証であり、実H4のRMSE/十分Nを認定しない。Numpy binomialのint64超Nは既存どおり丸めず拒否。

## 保存S3と未決定事項

正本は以前の[STATISTICS_INPUTS](../rpe_s3_readiness_20261113/metadata01/STATISTICS_INPUTS.json)・[S3判断表](../rpe_s3_readiness_20261113/S3_DECISIONS.md)。全13保存mean・経験的nominal（64,60,56,52,47,43,38,34,30,25,21,17,12）、m3/m4 norm欠測、物理誤差契約・SOURCEは変更していない。

target proxy未採択、未解決trial loss未決定、ideal-law IID未資格、数値不確かさ未解決、MSE精度/信頼水準/有限反復数未決定。scientific_execution_ready=false、S3_executed=false、十分N/global bound/G=nullを維持。A/C/Dへの新しい科学計算依頼・作業変更は今回行わない。

## 技術再現

project専用Python環境を使用する。decoder系とmetadata系は別processで実行する（後者には受動importの検査がある）。

```bash
PYTHONPATH=src:tests python -B -m unittest -v check_rpe_decoder_fast check_rpe_decoder_s3_batch check_rpe_decoder_statistics check_pr_rpe_rmse.RMSETests.test_existing_decoder_sign_and_zero_boundary_without_sampling
PYTHONPATH=src:tests python -B -m unittest -v check_rpe_s3_intake check_compiled_residual
python -B scripts/bench_rpe_s3_batch_artificial.py --out /tmp/rpe-r3-batch-fresh.json
```

decoder34＋metadata26＝60技術test PASS。A固定fixtureはGitから旧B関数を読みtemporary treeへ提案patchを適用するだけで、実分子artifactは読まない。性能scriptは実B APIを使い、人工13round/512trial、fresh subprocess・各3回の中央値とpeak RSSを比較する。

次は**ユーザー**によるS3科学条件と固定SOURCE・有限実行範囲の別承認。**Codex B**は承認後に科学実行へ進む。今回は公開整合確認後mandatory STOP。
