# A → B：R3高速decoderとS3 batch hook

2026-11-13 JST。監査 `a8f32abbd925276e126588b3648e3fb0deb014ea`、B decoder統計 `170047ef6e7a72aef5f4c6ff9c73f942097a694e`、A物理契約 `bba150834ca30bdb186fbfd58c55a8d51177b855`を基点とする。**人工技術検証のみ。実H4/S3・保存13mean/N/RMSE再解析は0。** Bのworktree/branchは変更していない。

作業中に公開されたB `bf515a976f1987ad6a19c61d14689b27b0894542`も確認した。対象`rpe_decoder_statistics.py`のblobは両commitで`e4d5aba93a7a91c0061babaccc86f69e81dfa697`と同一で、同じpatchを適用できる。Bの他の新成果の科学的評価は行っていない。

## 最小の採用差分

1. 今回公開commitの `src/trottertracks/rpe_decoder_fast.py`を取り込む。旧scalar/vectorized、A RMSE wrapperは参照実装としてそのまま残す。
2. [B_S3_PATCH.diff](B_S3_PATCH.diff)をB170047eと照合し、`git apply --unidiff-zero <B_S3_PATCH.diffのpath>`で適用する（zero-context形式）。`simulate`と`simulate_prepared`に任意の`batch_decoder=None`を追加し、binomial全count生成後だけhookへ渡す。既定Noneは元trial loop。新controller・planner・sampler・loss/CI実装は追加しない。
3. 科学実行SOURCEを新commitに固定する際、hookの利用とacceleration SOURCEを実行記録へ含める。過去SOURCEやdecoder科学契約を付け替えない。

```python
from trottertracks.rpe_decoder_fast import decode_sampled_counts

# 科学実行時だけ。実H4について今回この呼出しは行っていない。
simulation = statistics.simulate_prepared(
    prepared, repetitions=approved_repetitions, seed=approved_seed,
    policy=approved_iid_policy,
    allow_conditional_unqualified=approved_conditional_scope,
    batch_decoder=decode_sampled_counts,
)
summary = statistics.summarize(simulation['trials'], **approved_loss_and_CI)
```

`prepare`の欠測・未資格gate、`simulate`のPCG64/binomial順・seed・shot整数検査は元のまま。`exact_counts`・`summarize`・CI・trial_budgetは変更しない。以前のS3計画で必要なtarget proxy、loss、数値不確かさ、統計精度・信頼度と科学実行承認も残る。

## API

| API | 用途・互換性 |
|---|---|
| `reconstruct_rpe_phase_fast(phases,q_values)` | 元scalarと同じround dict・final_phase。厳密同距離では最小branch ID |
| `vectorized_reconstruction_fast(phases,q_values)` | 元vectorizedと同じ `(estimate, branches, ambiguous)`。旧argsortのexact-tie順も保存 |
| `reconstruct_batch(phases,q_values,tie_rule='scalar')` | 全roundの位相・branch・最良/次点距離、ambiguousをarrayで取得。shapeは任意の共通batch shape |
| `decode_supplied_coordinates_fast(coordinates,h=...)` | 元A wrapperと同じstatus/energy/decoded、scalar入口 |
| `decode_coordinates_batch(xy,h=...)` | `(trials, rounds, 2)`の供給X/Yを元scalar規則で一括decode。未解決のenergyはNone |
| `decode_sampled_counts(samples,rounds,M,h,prior,count_mean,decoder_source)` | Bの生成済みcount arraysをbatch変換して元trial dictを返すhook |

有効なfinite binary64位相、consecutive dyadic grid、M0〜12を人工検証。実装はbranch整数がbinary64で正確なq≤2^52まで受け付けるが、その全域の科学的適格性を主張しない。非有限・shape不一致・不正gridを拒否。scalar wrapperは元と同様Python int/float座標を要求する。batch APIはfloat64座標arrayを受ける。

## 保持した規則と差異

- wrapの加算・剰余・減算順、candidate構成の演算順、`abs(angle(exp(i*(candidate−estimate))))`を保持。最良・次点距離は元の値と比較する。
- scalarはargminの最小branch ID。旧vectorizedは不安定なargsortであり、**完全tieでは異なるbranchを選ぶ場合がある**。例：q32/principal0/previous2.650718801466388ではscalar13/vector14。両参照を混同せず、vector互換APIは完全tieの行だけ元argsortを再現する。
- B向けscalar規則は常に定数個のbranchでO(1)/round。vector互換のexact-tie例外はO(q)、全域の最悪O(1)とは呼ばない。1e−12以内というだけのnear-tieは近傍比較で同じambiguousを返す。
- 座標→位相は元の`math.atan2`を使用。NumPyのarctan2に置換して1 ULP差を導入しない。zero座標が1roundでもあれば元と同じUNRESOLVED_ZERO_COORDINATESがbranch-tieより優先する。
- countは`np.unique`で同値をまとめ、**B自身のcount_mean**を呼ぶ。`2*k/N−1`を新たに計算しない。2^53を超えるNも丸めない。count-arrayは整数・全round・同shapeを検査する。
- priorは診断のみ。clipping、alias補正、energy shift、loss fallbackを追加しない。−phase/h、未解決None、全RMSE未資格を保持。
- 元trial dictに`decoder_acceleration_source`だけを追加。比較時にそのprovenance欄を除くと旧trial全項目と一致する。countsを再び各trialに保持しない。

## SOURCEとテストの使い方

```bash
python -B tests/check_rpe_decoder_fast.py -v
python -B scripts/bench_rpe_decoder_fast_artificial.py --out <fresh-benchmark.json>
```

`tests/rpe_decoder_r3_fixture.py`はGitの固定Bから元関数を読み、temporary directoryにだけpatchを適用してテストする。A branchに存在しないB controllerからは元exact変換関数のみを読み、統計アルゴリズムを複製していない。実分子のartifactは読み込まない。

Bが別commitへ進んだ場合は、まずsimulate/simulate_prepared/count_mean/SOURCEの差分を確認し、この人工テストで新旧のRNG・trial・loss/CI同値性を確認する。監査R3に記載されたCI方式の変更は今回のdecoder高速化へ含めず、B既存方式を保持した。

次：**B**が公開固定moduleと任意hookを取り込み、既存統計処理へ技術接続する。**ユーザー**が固定SOURCE・loss/target・統計精度・有限試行scopeを別途承認した後だけ、**B**が実S3へ進む。Aは公開照合後mandatory STOP。
