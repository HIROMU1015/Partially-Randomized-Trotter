# H4/H6 高次PF PR：準備のみ

[技術報告](../../../../docs/research/track_a_h46_higher_pf_readiness_20261018.md)。SOURCE `ed14d951a4761838f25aeb16a653b55aba499c34`。

- `evidence_inventory.json`：固定PF repository8file、PR SOURCE closure8/11、imported provider89file、元入力4file、24signal/6費用/18欠測のmetadata。実値の費用・適格性・順位計算なし。
- `formula_registry.json`：元3公式と未採用2案を区別。保存moment値は引用、係数最適化0。
- `source_closure.json`：新SOURCEの6file（5新規code/test＋frozen shot rule）のGit/checkout SHA一致。
- `plan_*.json` と `invalid_grant_*.json`：A、B保存値のみ、B不足native取得、C独立確認。全て未承認/無効。B取得とCのnはnull。
- `artificial_tests.json/.log`：32人工/mock PASS。実H4/H6再解析・compile・sampling0。
- `draft_refusal.json`：公開B_saved draftの実CLI拒否（入力読込前）。
- `preservation.json`：対象Git blob不変・原本への書込なし。無関係worktreeを拘束しない。

metadata監査を再現する場合のみ（科学計算ではない）：

```bash
python3 -B scripts/audit_h46_higher_pf_readiness.py \
  --pf-repo /path/to/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae \
  --output /unused/path/metadata.json
python3 -B -m unittest discover -s tests -p test_h46_higher_pf_readiness.py -v
python3 -B scripts/run_h46_higher_pf_comparison.py
```

本bundleに有効grantや科学実行の承認はない。manual approvalを得る前にstatusを変更して実データを起動しない。mandatory STOP。
