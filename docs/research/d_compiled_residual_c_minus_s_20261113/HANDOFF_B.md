# Codex B向け：E[S]＋E[C−S]の価格recordと受入条件

参照：[REPORT](REPORT.md)、[B_RECORD_EXAMPLE_ARTIFICIAL](B_RECORD_EXAMPLE_ARTIFICIAL.json)（mock値。価格ではない）。

module：`src/trottertracks/compiled_residual.py`（branch `codex/d-compiled-residual-c-minus-s-20261113`、B`3216d27a`基点）。

## 1. 流れ

```python
from trottertracks import compiled_residual as R
from trottertracks import rpe_diagnostic_intake as K

accepted = K.accept_structural_packet(config, k1_packet)    # 既存B：E[S]行を受入（価格recordは作らない）
combined = R.combine(structural_row, residual_record, config=config)   # 1 round・1軸
record = R.b_price_record(combined, config=config, cohort=..., source=...)
B.assess(config, n_packet, [record, ...], cohorts=...)      # 未資格なら拒否される
```

## 2. 残差record（E[D]）のenvelope

| field | 内容 |
|---|---|
| `quantity` | `FULL_WRAPPER_MEAN_C_MINUS_S` |
| `target` | `C_MINUS_S` |
| `mode` | `PAIRED_DIRECT_FINITE_COHORT`、`C1_AFFINE`、`LOCAL_C_MINUS_S`のいずれか1つ |
| `coverage` | `FULL_WRAPPER`以外ならE[C]はnull |
| `m`/`axis` | E[S]行と同じround・軸 |
| `identity`/`law_binding`/`physical_candidate_id`/`profile` | canonical configurationの行と完全一致 |
| `law_context.law` | E[S]と同じlawの宣言。不一致は理由として残す |
| `mean_D` | exact rationalまたは数値。未取得ならnull |
| `expectation`/`n_paths`/`role` | paired directでは`FINITE_COHORT_ONLY`・path数・DEVELOPMENT/CALIBRATION/HOLDOUT |
| `source`/`uncertainty` | 取得SOURCE、記述的分散など。未知はnull |

## 3. Bに渡るrecord

- `quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`、`classification=MODEL_PREDICTION`、`price_status=UNQUALIFIED_MODEL`、`expectation=SUPPLIED_MODEL_EXPECTATION`。
- `model.qualified_for_candidate=false`。`components`にE[S]とE[D]の出典を残す。
- 欠測時は`UNAVAILABLE`/`MISSING`、`rz_count=null`。
- 人工例でB `assess`は`eligible=false`、`cost=null`を返した。理由は次の3つを含む。
  - `C_UNQUALIFIED_MODEL`
  - `UNQUALIFIED_C_EXPECTATION_OR_FORCED_COHORT`
  - `PRICE_TARGET_LAW_DISTANCE_UNQUALIFIED`
- 構造S envelopeを価格として渡した場合は、`C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE`で拒否される。

## 4. 資格に必要なもの（今回は未取得）

1. E[S]：K0 PASS、K1 law期待値、数値区間。
2. E[D]：選んだ1 modelの校正とfresh holdout、order2 stratum、law一致、長q/R systematic。
3. 目標精度ε_Cを満たす不確かさの合成。round間で共有する係数の共分散を含む。
4. 以上を満たすまで、Bはこのrecordを`QUALIFIED_EXPECTATION`へ昇格させない。D adapterはその昇格機能を持たない。
