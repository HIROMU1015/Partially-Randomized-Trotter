# C・統合担当・A・B・D2へ：PR-Pauliのκ区間探索と、Cの共通入口の接続（round04）

正本は [REPORT](REPORT.md)。B2（Claude Code）が作成し、確認は作成者自身による。

## 集計に使える固定commit・path

- 固定commit：このHANDOFFを含むbranch `codex/b2-rpe-split-preparation-20261116` のcommit（push済み）。
- 基準path：`docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/`。

| 内容 | path | 値 |
|---|---|---|
| LD5 κ=7（κ=2から上側で最初に異なるschedule） | `best_points/LD5_kappa7_32ecd1ff57d0b2e6a8c6/` | 518,960,285.57 RZ（構成 `a620a355`、MSE上界/τ=0.999993） |
| LD5 κ=2の先読み適合点（規則は採択していない） | `best_points/LD5_kappa2_lookahead_not_adopted_9d10cc9f815edeeefdd7/` | 500,962,225.57 RZ（規則上の最安501,233,190.43より0.054%安い。小改善の閾値未満） |
| 全点の表 | `KAPPA_SCHEDULE_RESULTS.json`・`.csv` | 区間、変わるround、価格判定、信号の新規round、最安、MSE |
| 全評価 | `run01/EVALUATIONS.csv` | 284件（先読みを含む）。exact値はjobのOUTCOME |
| 幅Δ_PF≠hのメタデータ | `width_requests/WIDTH_REQUESTS.json`、`BATCH.json`、`BATCH.CALIBRATION_0*.json` | 較正cellは4件（未実行） |

- PR-Pauli全体の保存最安は、変わらず410,199,237.00 RZ（`kappa_abs_round03/best_points/`）。
- 信号：branch `codex/b2-pauli-kappa-signals-20261116` の `fa9c4cf17`。構成と経路は `run01/ROUTE_REGISTRY.json` にある。

## Cの共通入口（02e7cc6a1）の利用結果

- **κ探索の新しい7点**：すべて `REUSE_SUPPLIED_PRICES_NO_CALIBRATION`。新しい較正は0件。
- **幅Δ_PF≠h**（LD1・2・3・5のκ=2）

| 幅 | 判定 | 必要なもの |
|---|---|---|
| 厳密な1/5 | `NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION` | m2〜m12の実幅がbinary64で表せない。driverの規則か実装の拡張が必要 |
| binary64の0.2 | `CALIBRATION_REQUEST_READY_NOT_EXECUTED` | 次の4 cellの実行承認 |

  - binary64 0.2の4 cell：LD1 Q8R1792、LD2 Q8R692、LD3 Q8R109、LD5 Q8R1。
  - 4 cellとも、q=8、T=1.6、実幅は0.2（binary64）。
- **B2のdriver側の処理**：判定が再利用以外のときは、その点を保留にし、round単位の処置・cellのrequest id・不足を `KAPPA_STATE.json` の `price_entry` と `calibration_cells` に残す。

## 停止と再開

- **停止**：2026-11-16 21:37:56 JST、`NEIGHBOURHOOD_RULE_EXHAUSTED_NO_PROMISING_CANDIDATE`。
  - 同じround fileで再実行しても、同じ判断で止まる。
- **規則を変える場合**（例：LD2・3を大きいκへ延長する）
  - 新しいround fileとrun dirを作る。
  - `prior_run_dirs` に、round01〜04の各 `run01` を指定する。

## 判断が必要な事項

| 事項 | 内容 |
|---|---|
| LD2・LD3の大きいκ | shot上限でのMSE下界/τが、κとともに下がっている（LD3：κ=3で1.95、LD2：κ=3で3.29）。今の近傍規則は、適合点のない系列を延長しない |
| Δ_PF≠h | 上の4 cellを較正するかどうか（科学実行の承認が必要）。厳密な1/5の扱い |
| 引継ぎ | LD4・6のBの取込み、LD7・8・9のP0、LD1の価格のD2本体への登録 |

## ロードマップへの差分（本体は統合担当が更新）

- **κそのもの=2から最初に異なるscheduleまで評価した**
  - LD5：上側のκ=7（m12のR=2）は518,960,285.57 RZで、κ=2より高い。範囲[1/2, 8]の中にあるLD5のscheduleは、この2つだけ。
  - LD1・2・3：上下の最初の異なるschedule（6点）は、shot上限内に適合配分がない。
- **PR-Pauliの保存最安は変わらない**：410,199,237.00 RZ。Full PFの0.733倍、p7の1.103倍。
- **Δ_PF≠h**：メタデータのみで、科学評価はまだない。厳密な1/5とbinary64の0.2は、別の候補として扱う。
