# D registryのC通常同期

branch：`codex/h4-saved-rpe-cost-catalog-20261116`。

固定D SOURCE：`352539e2749989b4ddc3ab74c31f486982bfe265:docs/research/d_pf_on_demand_evidence_20261116/C_RESULT_JOIN.json`。

通常の利用者自身の10分cronへschema readerを追加した。監視対象D branchとC_RESULT_JOIN検出は既存のものを使用し、利用者によるcommit/path転送・Skill起動は不要。

```bash
/bin/bash scripts/run_h4_catalog_sync.sh --python \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
```

正本は`docs/research/h4_saved_rpe_cost_catalog_20261116/run01/{CATALOG.json,REPORT.md}`。新Full PFはconfiguration `9e96ed75…`で照会する。statusはQ3_WAITING、`signal_rounds_available=13`、`price_axes`26件、`q3_template_source`は未評価templateの出所、`allocation_proposal_status=TEMPLATE_ONLY_NOT_EVALUATED`。allocation・N・MSE・Q3判定・conditional_costはnull。Dの配分容器や提案だけでは最安適合点に選ばない。

既存p7 `dd8bfc248…`のregistry根拠は`signal_only_history`に記録する。既存Q3行・各費用・予算・SOURCEを保全し、Q3待ちの別配分を追加しない。新しいp7価格や旧費用の再集計はしていない。

原13信号と26軸価格のphysical ID・identity/law・compiler・scalar等の出所はD readerが照合した範囲と、軸別`price_evidence`からたどる。新条件の科学資格は付与しない。欠測や不一致は既存SYNC_STATUSへ理由付きで残す。

ロードマップへの差分は「Full PF・S2・2hの信号/条件付き価格入力が準備済み、Q3待ち」。費用はnullであるため数値改善として扱わず、方針・Q3/価格資格・原STOPを変更しない。
