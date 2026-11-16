# C・統合担当・A・B・D2へ：PR-Pauli κ=2近傍と、未準備L_Dの準備

正本は [REPORT](REPORT.md)。B2（Claude Code）が作成し、確認は作成者自身による。

## Cが集計に取り込める固定commit・path

固定commitは、このHANDOFFを含むbranch `codex/b2-rpe-split-preparation-20261116` のcommit（push済み）。path は `docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/` から。

| 内容 | path | 値 |
|---|---|---|
| PR-Pauli全体の保存最安 | `best_points/LD5_kappa_x2_edd63e0befd3af591809/`（REQUEST/OUTCOME/CASE_INPUTS/worker成果物） | 410,199,237.00 RZ（LD5、κ≈48.61、構成 `cb1aa748`、MSE上界/τ=0.999912） |
| κ=2の最安 | `best_points/LD5_kappa2_e3cb2cfda719117d6222/` | 501,233,190.43 RZ（LD5、κ=2、構成 `ff6b21ae`） |
| 全評価・各点の表 | `run01/EVALUATIONS.csv`、`KAPPA_ABS_RESULTS.csv` | MSEは外向き30桁区間。exact値はOUTCOME |

- 構成・信号packetの所在は `run01/ROUTE_REGISTRY.json` にある（信号branch `codex/b2-pauli-kappa-signals-20261116`）。
- 価格はD2 PriceRules MODEL（c1f95e080）で求めた。LD1だけは、B2が拡張した登録表を使った。

## 停止と再開

- **停止**：2026-11-16 19:09 JST、`NEIGHBOURHOOD_RULE_EXHAUSTED_NO_PROMISING_CANDIDATE`。
  - 近傍規則（κ∈[1/2, 8]、r=3/2、中点比9/8、端の拡張には1%の改善が必要）の追加候補が尽きた。
  - LD5のκ点では、0.1%未満の改善と、後回し候補だけが残っている。
- **再開**：規則を変えずに同じコマンドを再実行しても、同じ判断で止まる。
  - 規則を変える場合は、新しいround file（新しいround_id）とrun dirを作る。そのうえで、`prior_run_dirs` に round01・round02・round03 の各 `run01` を指定すると、評価がすべて再利用される。
  - 連鎖の先読み（`a58a0f31f`）は、次の実行から効く。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-split-preparation-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <Evaluation venv python> -B scripts/run_rpe_kappa_neighbourhood.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json --round <新round file> --run-dir <新run dir> --batch 256
```

## 保留した候補と必要な作業

| L_D | 止まった所 | 必要なこと |
|---|---|---|
| 4・6 | 信号 | P0とP1の構成はある。Bの取込みが、6e4b6bd6a以降の信号入口（`--tail-descriptor`）で組み立てたpacketを受け付ける必要がある（Bの担当） |
| 7・8・9 | P0の数値gate（Pauli_tail_full_fro が 1.1e-7〜1.3e-6） | descriptorか、tail分解の数値精度の見直し（Aの担当）。規則上、再試行していない |
| 1・2・3（κ=4/3・2・3） | shot上限内に適合配分がない | より大きなκ、または上限の見直しが必要。上限の見直しは今回の範囲外 |

## 新L_Dの価格（D2へ）

- LD1の較正価格は、B2が拡張した登録表に入っている（`codex/b2-pauli-calibration-20261116` の `54b4353b9`、`PRICE_RULES_REGISTRY_EXTENDED.json`）。
  - 変更したのはPAULI_LD1の項目だけで、他の項目の価格は変わらない。
  - 根拠：raw `1b049cbe4`、audit `86fa4ce9e`、要求 `ccf0793de`。
- D2の正本の登録表へは、まだ入っていない。D2が取り込む場合は、この項目が元データになる。

## ロードマップへの差分（本体は統合担当が更新）

- PR-Pauliの保存最安：410,470,823.86 → 410,199,237.00 RZ（LD5、κ≈48.61）。
  - Full PFの0.733倍。p7（372,035,410）の1.103倍で、p7は下回っていない。
- κ=2近傍（κ≈1.3〜3）の評価
  - LD5は501.2Mで平坦。
  - LD1・2・3は、shot上限内に適合配分がない。
  - したがって、PR-Pauliの最安はκ≈16〜49側にある見込み。
- 未準備L_Dの評価
  - LD4・6：Bの取込みの版の問題で、Q3まで進めていない。
  - LD7・8・9：P0の数値gateで停止した。
