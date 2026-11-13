# Codex A向け：Full PF・Full-H RTEのV2物理requestと、保存signal・referenceの状況

参照：[REPORT](REPORT.md)、[COVERAGE_V2.csv](COVERAGE_V2.csv)（`signal_*`、`reference_*`、`independent_pair`、`numerical_endpoints`の列）、[REQUEST_EXAMPLE_ARTIFICIAL](REQUEST_EXAMPLE_ARTIFICIAL.json)。

## 1. V2物理request（B `f0c4bb55`に接続済み）

```python
from trottertracks import full_baseline_v2_requests as D
packet = D.bind(D.full_pf_spec(base, formula='4th', delta_pf=width, M=M, rank=rank))      # 2nd / 4th / 4th(m5_best)
packet = D.bind(D.full_h_rte_spec(base, M=M, finite_K=K, preparation=prep), layout_by_round={m: {'q': q, 'R_outer': R}})
A_and_C = D.requests(packet)          # B V2 configurationを渡してもよい（再生成してIDを照合する）
```

- **IDの出所**：V2 configuration ID・candidate IDはB.generateが生成したものだけである。標準4thのformula IDは`PF_STANDARD_YOSHIDA4_BINARY64_V1`。
- **物理IDとの対応**：物理candidate ID（`cid`、Full-Hは`FULL_H_`＋cid）は`identity_map`で対応付けてある。
- **A requestの内容**：
  - 共通：`configuration_id`、`v2_candidate_id`、`identity`、`law_binding`、`physical_candidate_id`、`physical_round`、`physical_schedule_mapping`、`binding`、`preparation`、`rmse_policy`。
  - 値：`N_X`/`N_Y`/`rmse_bound_ha`はnull。
  - 不足項目：`missing`。
- **C requestとの一致**：A requestとC request（X/Y）は同じidentityを参照する（`validate_requests`）。
- **Full-H RTE**：κなし（`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`）。
  - layoutを与えないroundは`NOT_EVALUATED`のまま残す。
  - κを供給した場合は従来kindとなり、κなしとは別のparameterizationになる。

## 2. 既存signal/reference carrierとの対応（Full PF 136候補）

| 情報 | 状況 |
|---|---|
| signal | **EXACT 134**：長時間研究の`evolve_batch` 110件、pf_common T=0.8の再利用24件。**CONDITIONAL 2**：H6 T=1.6 q8の2nd/4th。fixed-delta由来で、raw signalとpropagation evidenceがない |
| reference | 136件すべてに値がある（元DF Hを保存sector stateで評価した値）。独立reference pairはT=2.0の38件だけ |
| 数値gate | 旧契約（経験的allowance、α_axis 0.025）のELIGIBLE/INELIGIBLE。RMSEのPASSではない |
| 物理ID | Dの`bind()`の物理IDと同じ。`full_pf_saved_inventory.for_packet()`で、V2 ID付きのまま結合できる |

結合例（人工、h=0.1、δ=0.1、M=4、[V2_SAVED_JOIN_EXAMPLE](V2_SAVED_JOIN_EXAMPLE.json)）：
- 保存gridに入るのはm3（T0.8 q8）とm4（T1.6 q16）だけ。
- m3は3公式ともsignal EXACT。
- m0–m2は保存gridの外。

## 3. Full PFのRMSE契約に必要な不足証拠

1. **誤差契約**：Full PF用の全RPE RMSE誤差契約がない。元H targetに対するPF bias、統計誤差、decoder/aliasの合成が必要。現在のV2 policyはPR用である。
2. **十分N**：全roundのRMSE型の十分Nがない。保存endpointのshot数は旧型のN。
3. **RPE条件**：prior・decoder・alias/branchの検証（全round）がない。
4. **独立reference pair**：T=2.0以外にはない。必要なTを決めてから取得する。
5. **H6 fixed-delta 2件**：Full carrier水準へ上げるかどうか。

## 4. 保存grid外の未取得mean/reference

- 保存されているのはT∈{0.8,1.2,1.6,2.0}だけ。RMSE scheduleのT＝h·2^mは、m3・m4以外の多くのroundでsignal・referenceがない。
- 新しい計算を提案するときは、保存signalと完全一致するセル（formula、T、q、FULL prefix、binding）を除外する。そのうえで、残りのセルと上限を明示する。
- Full-H RTEは、保存light candidate（H4 q16/R16/K4、H6 q32/R32/K6、T=0.8）以外の全セルについて、finite-K mean・normalizationが未取得である。

## 5. 未採択の状態

- Full PFの公式・幅（δ）・M・比較menuは、ユーザーが正式採択していない。
- Full-H RTEのlayout menuとK、κとの等価性も採択されていない。
- 新しいmean・reference・N/RMSEの評価は、ユーザーの明示承認があるまで行わない。
