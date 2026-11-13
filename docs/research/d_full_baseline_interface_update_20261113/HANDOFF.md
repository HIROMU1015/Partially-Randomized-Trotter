# 標準Yoshida4・Full-H明示layoutのrequest：A/B/Cへの引渡し

**更新（2026-11-13）**：B f0c4bb55への最終接続と保存価格17件のreceipt照合は[d_full_pf_full_h_v2_final_receipts_20261113](../d_full_pf_full_h_v2_final_receipts_20261113/REPORT.md)。

[REPORT](REPORT.md)。adapterは`src/trottertracks/full_baseline_v2_requests.py`（branch `codex/d-full-pf-full-h-rte-v2-paths-20261113`）。このファイルは前段の[HANDOFF](../d_full_pf_full_h_rte_v2_paths_20261113/HANDOFF.md)を置き換える。前段との主な違いは2点：

- Full-H RTEはκを受け付けない。
- 標準4thは物理requestまで作る。

## 1. 使い方（metadata）

```python
from trottertracks import full_baseline_v2_requests as D

pf = D.bind(D.full_pf_spec(base, formula='4th', delta_pf=user_delta, M=user_M, rank=10))
fh = D.bind(D.full_h_rte_spec(base, M=user_M, finite_K=user_K, preparation=D.saved_full_h_preparation('H4')),
            layout_by_round={m: dict(q=..., R_outer=...) for m in user_rounds})
out = D.requests(pf)        # A 1件＋C X/Y 2件 / round。値はすべてnull
```

- `bind()`の戻り値：`physical`（物理schedule）、`v2_configuration`（Bが生成できる場合だけ）、`v2_status`、`identity_map`、`b_interface`、`invented_v2_ids=false`。
- `D.generate()`は公開済みBで生成できる場合だけconfigurationを返す。それ以外はpending理由を例外にする。
- `base`は比較対象の既存V2 spec（binding/h/rmse_policy/profileを共有）。`formula`・`delta_pf`・`M`・`K`・layoutはユーザーが決める値である。

## 2. Codex B向け：最終結合に必要なinterface

1. **公式hook**：
   - `formula='4th'`を受け付ける。
   - 構造は`signed_weights=h46_m5_plan.weights('4th')`（3 piece、中央が負）、`formula_id=YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64`、event_count 0、law deterministic、`scalar_identity_once_at_total_T`。
   - qの定義は既存PFと同じ`ceil(T/δ)`。
   - D specでのblockerは公式の検査だけである（test 12）。
2. **κなしの明示layout**：
   - `event_mapping.kind=D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`、`kappa=None`、`kappa_mode=NOT_DEFINED_EXPLICIT_LAYOUT`を受け付ける。
   - round mappingは`D.layout_mapping(spec, {q, R_outer})`の形：`kind, source, kappa=None, R_outer=None, event_count=qR, physical_layout{…}`。
   - layoutのないroundは未解決のまま残す。
   - D specでのblockerはκの検査だけである。
3. **physical candidate hook**：`identity_map`の`physical_candidate`をV2 roundへ添付する。
   - Full PF：`p.candidate_id({system, method: FULL, formula, prefix: rank, T, q, R_outer: None, K: None})`
   - Full-H：`FULL_H_`＋`p.candidate_id({system, method: PR, formula: 2nd, prefix: 0, T, q, R_outer, K})`

Bが公開したら、Dは`bind()`の分岐（`FORMULAS['4th']['b_v2_generate']`とFull-H経路）をそのcommitへ追従させ、`_check_against_b`で全roundを照合する。

## 3. Codex A向け：N/RMSE request

`requests()['rounds'][m]['A']`の内容：
- 物理round：T、q、実幅、公式とsigned weights、またはlayout・K・τ・λ・preparation hash
- binding、`rmse_policy`、`physics_source`、`signal_functions`
- `saved_asset_status`、`saved_reference_time`
- `missing`。`N_X`、`N_Y`、`rmse_bound_ha`はnull。

| 方式 | missing |
|---|---|
| Full PF | `FULL_H_ACTION_MEAN_OR_SAVED_SIGNAL_AT_T`、`RMSE_SUFFICIENT_N`、`GLOBAL_RMSE_CONTRACT_FOR_FULL_PF`、（保存外Tでは）`NEW_REFERENCE_AT_T`、（4thでは）`PENDING_B_FORMULA_INTERFACE` |
| Full-H RTE | `FULL_H_FINITE_MEAN_AND_NORMALIZATION`、`RMSE_SUFFICIENT_N`、`GLOBAL_RMSE_DECODER_CONTRACT_FOR_FULL_H_RTE`、`FINITE_K_AND_SAMPLER_LAW_QUALIFICATION`、`PENDING_B_KAPPA_FREE_EXPLICIT_LAYOUT_INTERFACE`、（layoutのないroundでは）`LAYOUT_NOT_SUPPLIED` |

## 4. Codex C向け：price request

`requests()['rounds'][m]['C']`（X/Y）の内容：
- 同じ物理候補、profile、測定軸・符号・単位
- `construction`（`h46_long_time_native.compose/cost_trial`）
- `price_law`（PFは決定論、Full-Hはsamplerの距離が未資格）、`expectation_needed`
- `saved_price_source`、`saved_price_match_requires`
- `mean.rz_count=null`、`price_status=NOT_EVALUATED`

保存価格の一致条件：
- Full PF：公式、T、q、FULL prefix、profileが一致すること。
- Full-H：layout q/R/K、T、preparation hash、law（またはcohortの宣言）が一致すること。

event数だけで価格を同じとみなさない。

## 5. 範囲外・注意

- 8次PFは今回の範囲外（catalog上は`OUT_OF_SCOPE`）。既存のSOURCE・結果はそのまま保全している。
- Pauli Full-H RTEは表現が異なるため接続していない。
- 科学的な実行（A/C）は、ユーザーが対象・SOURCE・上限を承認した後だけ行う。今回の公開でmandatory STOPとする。
