# S0a run01失敗・技術修正HANDOFF

[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[run01 STOP](run01/STOP.json)。科学起動1回、新S0/8・保存S比較0。すべて未評価、RESULTS.jsonなし。

原因はS0a exec scopeへastを渡し忘れたこと。保存input内容検査・preparation・raw/law/digest・Accountingより前に停止した。元科学SOURCE11cb91a65とrun01 PLAN/STOP/logは保全し、retryはしていない。

修正版SOURCE`9516351e6`は`ast=ast`の1行修正と、固定contextの必要global名をmock execで照合する追加test。修正後の人工14件PASS。物理contextの成功やparity PASSとは区別する。

[RUN02_PROPOSED_PLAN](RUN02_PROPOSED_PLAN.json)は準備済み・未実行。元4候補、normal index0、元CALIBRATION/VALIDATION、XYとevent digestは不変。旧Sを対照にし、S8件だけを新たに計数する。内容hash/preparation/実law/digestの確認は実行時に必要。

## ユーザーが修正版でのrun02を明示許可した場合だけ

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-h6-provider-coverage-20261110
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B scripts/run_s0a_saved_structure.py execute \
  --plan docs/research/s0a_saved_structure_results_20261113/RUN02_PROPOSED_PLAN.json \
  --output docs/research/s0a_saved_structure_results_20261113/run02 --execute
```

このコマンドは今回実行していない。run01 STOPを解除・書換えずfresh run02を用いる。科学scopeは元保存metadata再構成・law/digest確認・同4path×XY8 S照合だけ。sampling・旧S再計数・QC build/compile・fit・平均価格・N/G・追加pathは含めない。

Bの共通API/provider/controllerを変更していない。BはS0a parityをmissingのまま扱い、C1/平均modelのfalse/nullを保つ。ユーザーが再試行を判断、Cが承認範囲だけ実行、Bが最終統合を担当する。現在mandatory STOP。
