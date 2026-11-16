# C比較表：B2 round03・以後の公開selected point

branch：`codex/h4-saved-rpe-cost-catalog-20261116`。

通常の利用者自身の`*/10`cronを維持する。設定・担当branchへの追加変更は不要。実装・原因・元証拠は[REPORT](REPORT.md)。B2や利用者が新commit/pathをCへ転送する手順は不要である。

```bash
scripts/run_h4_catalog_sync.sh --python \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
```

これはcronと同じ入口であり、公開成果の読取・受入・commit/pushだけを行う。dirty checkout時はその周期をskipし、他担当のjob・private rawへ介入しない。

出力正本は`docs/research/h4_saved_rpe_cost_catalog_20261116/run01/{CATALOG.json,REPORT.md}`。

- 方式全体：`method_minima.PR_PAULI`、410199237/1 RZ。
- 絶対κ=2限定：`restricted_minima.PR_PAULI_KAPPA_ABSOLUTE_2`、3508632333/7 RZ。
- 各pointから`configuration_source`、`q3_source`、`public_evaluation_source`、`outcome_sha256`、`price_evidence`、`model_coverage_axes`、κ絶対値/倍率へ戻れる。
- 不足証拠は`SYNC_STATUS.md`／`PUBLICATIONS.json`のPENDING理由へ表示。configuration一致だけで取込み完了としない。途中ROUND_RESULTと後続最終選択は別の保存記録として扱う。

次roundは既存監視branch pattern上へ、既存joinか、公開`best_points/<label>/REQUEST.json`・`OUTCOME.json`・`CASE_INPUTS.json`と同一roundの`EVALUATIONS.json`・`ROUND_RESULT.json`・`NEEDS.json`、参照されるSOURCEを公開すれば同じ入口で照合する。finalの`KAPPA_ABS_RESULTS.json`も対応する。公開元hash・価格binding等がなければ必要な証拠を表示し、自動的な科学再実行はしない。

ロードマップ差分：PR-Pauliの新しい保存適合配分と探索停止理由を接続。費用の数値は比較表だけを正本とし、ロードマップ本文へ重複管理しない。未資格MODEL、全law E[C]・CI・正式Gの未認定、旧予算・STOPは変更しない。
