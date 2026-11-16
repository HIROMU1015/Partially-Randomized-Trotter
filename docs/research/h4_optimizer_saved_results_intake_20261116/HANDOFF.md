# B2・各担当向け：C比較表と現在地の定期統合

最新費用・配分・MSE・物理候補・価格根拠の正本：

- `docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json`
- 同`REPORT.md`、`INPUTS.json`、`CHECKS.json`
- 現在地：`docs/research/RESEARCH_ROADMAP.md`
- 取込み待ち：`docs/research/h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md`

公開branch：`codex/h4-saved-rpe-cost-catalog-20261116`。B2が比較基準を固定する際は、次で得たSHAと上記pathを保存して読む。将来の更新で過去の予算・判定SOURCEを付け替えない。

```bash
git ls-remote origin refs/heads/codex/h4-saved-rpe-cost-catalog-20261116
git show <上記で解決したSHA>:docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json
```

今回の固定受入ではFull PF M5要求16hが480258064、PR-Pauli LD5元κ×2が2873295767/7、p7は保存372035410 RZ。Full PF S2・559457386 RZ等の既存点も保持。Pauliは途中終了であり、26単価未保存のため保存総額＋26MODEL coverageの受入と明記した。κ絶対値/倍率は別field。exact OUTCOMEは`5304fc149ac3e522f806457b36c03d91a170593e`の[B2_BEST_OUTCOME.json](B2_BEST_OUTCOME.json)、入口は同[B2_SAVED_TOTAL_INDEX.json](B2_SAVED_TOTAL_INDEX.json)。元hashを保持する。

## 再集計

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/c-h4-saved-cost-catalog-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
scripts/collect_saved_rpe_costs.py \
  --inputs docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json \
  --output docs/research/h4_saved_rpe_cost_catalog_20261116/run01
```

同じ固定入力の再実行はbyte安定。所要時間はstdoutのみで、公開JSONに時刻/HEAD差分を足さない。

## 定期同期

`scripts/run_h4_catalog_sync.sh`が利用者自身のcronから10分ごとに起動する。公開branchの後続成果を検出し、既存adapterで照合・比較・文書更新・commit/push・remote照合まで行う。利用者のSkill起動やcommit/path転送は不要。

- 監視範囲・比較契約：`SYNC_CONFIG.json` / `INPUTS.json`。担当系列のbranch patternを利用する。
- lock/log：`artifacts/local_h4_catalog_sync/`（Git対象外）。dirty checkoutはその周期をskipし、手元の変更を保全する。並列jobや他checkoutには介入しない。
- SOURCE・MSE契約・physical/law・価格bindingが解決しない成果はSYNC_STATUSへ保留し、他成果は反映する。新schemaはCがadapterを更新する。未公開rawを自動探索・開封しない。
- ロードマップは依存段階の変化だけを反映し、費用の正本は比較表を参照する。研究方針や新科学実行を同期処理で採択しない。

確認・解除（利用者自身の専用マーカーだけを操作）：

```bash
python3 scripts/install_h4_catalog_sync.py --show
python3 scripts/install_h4_catalog_sync.py --remove
```

科学実行はSTOPを維持する。今回の定期統合は別途利用者が明示承認した保存成果の後処理であり、新signal/Q3/compile/fit/探索を起動しない。
