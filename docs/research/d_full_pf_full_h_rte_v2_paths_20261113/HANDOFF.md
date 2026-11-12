# Full PF・DF Full-H RTE request：A/B/Cへの引渡し

> 後続（2026-11-13）：標準Yoshida4・κなし明示layout・physical対応表を[d_full_baseline_interface_update_20261113](../d_full_baseline_interface_update_20261113/HANDOFF.md)で更新した。Full-H RTEのκ必須という本文の記述は、そちらで置き換えられている。

[REPORT](REPORT.md)。adapterは`src/trottertracks/full_baseline_v2_requests.py`（branch `codex/d-full-pf-full-h-rte-v2-paths-20261113`。基点B `af284d85`）。出力はmetadataだけで、科学実行・N/C/Gは含まない。

## 1. 使い方（metadata）

```python
from trottertracks import full_baseline_v2_requests as D

base = existing_V2_spec          # 比較対象の既存spec（例：PR config['spec']）。binding/h/rmse_policy/profileを共有
pf = D.generate(D.full_pf_spec(base, formula='2nd', delta_pf=user_delta, M=user_M, rank=10))
prep = D.saved_full_h_preparation('H4')        # 公開値の転記。prepare_full_h由来
fh = D.generate(D.full_h_rte_spec(base, M=user_M, finite_K=user_K, kappa=user_kappa, preparation=prep),
                layout_by_round={m: dict(q=..., R_outer=...) for m in user_rounds})
packet = D.requests(pf)          # rounds[m] = {A: {...}, C: [X, Y]}
```

- `formula`、`delta_pf`、`M`、`K`、`kappa`、`layout_by_round`はユーザーが決める値である。adapterは選ばず、既定値も持たない。
- `kappa`はB V2の`FULL_H_RTE`で必須。D側で作らない。
- 出力CLI：`python -B scripts/report_full_baseline_v2_requests.py --output-dir <fresh>`。SOURCE_CATALOGと人工例を出力する。

## 2. Codex B向け：V2 candidate/identity

- **Full PF**：`method=PF`、`deterministic_scope=FULL_H`、`L_D`=全rank（H4 10、H6 19）。`retained_hamiltonian={construction: FULL_H, prefix: rank, target_hamiltonian: binding.hamiltonian}`。`kappa`・`finite_K`・event mappingはなし。公式は2nd/m5_best。
- **Full-H RTE**：
  - split：`randomized_scope=FULL_H_INCLUDING_ONE_BODY`、`one_body_policy=RANDOMIZED`、`original_PF_method=NONE_FULL_H_PREPARATION`（B3ではない）。
  - splitには`preparation`（転記したdescriptor）、`lambda_R`（全H値）、`lambda_source`を入れる。
  - `event_mapping.kind=D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`。
  - round mappingの`physical_layout`は、S2のq×R、allocation、τ定義、K、符号、scalar一回、compose SOURCE、`priced_equivalence_claim=false`を持つ。
- **物理候補ID**（V2 identityとは別）：
  - Full PF：`p.candidate_id({system, method: FULL, formula, prefix: rank, T, q, R_outer: None, K: None})`
  - Full-H RTE：`FULL_H_`＋`p.candidate_id({system, method: PR, formula: 2nd, prefix: 0, T, q, R_outer, K})`。公開light studyのIDと一致する。
  - 対応は`requests()`の`physical_candidate_id`と`v2_candidate_id`で持つ。
- **Bでの拡張を推奨するもの**（Dでは実装していない）：
  1. 標準Yoshida4 `4th`をgenerate/structureに追加する。係数は`h46_m5_plan.weights('4th')`。m5_bestとは別のformula_idにする。
  2. 明示layoutの`FULL_H_RTE`で、κを不要にする種別を許すかを決める。現状はκ定義（kind/definition_source/value）の供給が必要。
  3. native生成roundに`physical_candidate`を付ける公式のhookを設ける。現状はA fixed-grid経由だけ。
- **比較の共通条件**：B.selectは、binding・h・rmse_policy・profileの同一性を要求する。D specは基底specからこれらをcopyする。Mと幅は方式ごとに異なってよい。現在のrmse_policyはPR用（A `PR_RPE_PAPER_V2_EMPIRICAL_RMSE_V1`）であり、Full PF/Full-H RTEへの適用はAが確認する必要がある。

## 3. Codex A向け：N/RMSE request

`requests(config)['rounds'][m]['A']`のfield：
- `identity`、`law_binding`、`v2_candidate_id`、`physical_candidate(_id)`、`physical_schedule_mapping`
- `physics_source`、`signal_functions`、`preparation`、`normalization`
- `saved_asset_status`、`saved_reference_time`
- `missing`。`N_X`、`N_Y`、`rmse_bound_ha`はnull。

| 方式 | 再利用できるSOURCE | 不足証拠（missing） |
|---|---|---|
| Full PF | `h46_long_time_signal.context/evolve_batch(method=FULL)`。保存grid（T∈{0.8,1.2,1.6,2.0}、q∈Q[T]、2nd/4th/m5_best）と一致するroundはcarrierを確認する | 保存外のTでのmeanとreference（`NEW_REFERENCE_AT_T`）。十分N。Full PF用の全体RMSE契約 |
| Full-H RTE | `h46_full_h_pure_rte.context`（prepare_full_h、binding・reference検査付き）＋`evolve_batch(prefix=0)`。保存referenceはT=0.8（light）とT=2.0（t20） | 全H λでのfinite meanとnormalization。finite-Kとsampler lawの資格。十分N。全体RMSE契約。layout未指定のroundは物理候補自体がnull |

Full-H RTEにPRのλ_1（2.8630、tail-only）やB3のλを使わない。全H λ（H4 21.0279、H6 41.2279）は公開summaryの転記で、Aが科学実行時にpreparation hashと照合する。

## 4. Codex C向け：price request

`requests(config)['rounds'][m]['C']`（X/Yの2件）のfield：
- 同じ物理候補・identity・law_binding
- 測定軸、符号、単位、profile
- `construction`（`h46_long_time_native.compose/cost_trial`）、`construction_mode`
- `price_law`（PFは決定論、Full-Hはsamplerとideal lawの距離が未資格）
- `saved_price_source`
- `mean.rz_count=null`、`price_status=NOT_EVALUATED`、`quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`

方式別：
- **Full PF**：決定論のsigned S2 piece列（全block）＋scalar一回。cohortはなく、価格は(公式, q, T)ごとに決定論である。既存のPF price route（Cの`bv2_mean_price`経由）が適合するかは、Cがdomainを照合する。
- **Full-H RTE**：S2 outer stepごとのcontrolled RTE列（prepare_full_hのprep）＋scalar一回。cohort価格である。
  - 同じevent数でも、groupingが違えば価格が同じとは仮定しない。
  - 公開light studyのT=0.8・2候補（n=9、`FULL_H_LIGHT_EXPLORATION_20261109`）は、layout・K・lawが一致する場合だけfinite-cohort参照になる。
  - C1のPR価格は、Full-Hへ移さない。

## 5. 未解決・注意

- Pauli Full-H RTEは、表現・state・lawがDFと異なるため接続していない（`pauli_reference()`）。数値をDFの結果と同一視しない。
- 8次PFは、現DF10/19のcontrolled full-wrapper経路がないため、V2 requestを作らない。
- 物理候補IDとV2 IDの対応はD requestの側にある。B identityへの組込みは、上記の拡張③。
- 科学的な実行（A/C）は、ユーザーが対象候補・SOURCE・call上限を承認した後だけ行う。今回の公開はmandatory STOPで、自動では進まない。
