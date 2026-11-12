# S0a実行準備HANDOFF

[REPORT](REPORT.md)／[EXECUTION_PLAN](EXECUTION_PLAN.json)／[SCIENCE_PLAN](SCIENCE_PLAN.json)。SOURCE`11cb91a65413bd33f5c02fc43fda40e583d91bd8`、新13＋旧9＝22技術test PASS。実pathの新S・保存S比較は未実行。旧C1/C2/mean成果・A/B作業領域は不変。

## 技術経路

- `SavedEvents`は保存JSONをevent/tail単位で読み、元canonical digestを逐次検査する。法的tail内でphaseを再初期化せず、旧`stream_structure`へ順に渡す。
- `Context`は固定C1の保存input/prepareとsigned headのSOURCEを再利用するS専用leaf。元backendのcompile/sample/composeは呼ばない。元H/state/snapshot/preparation/basis/normalization/timeの照合を維持する。
- 保存4pathのnormal index0だけ。元CALIBRATION/VALIDATION役割を保持し、XYは`cosine/sine`。保存Sが対照で、旧Sの新再計数は0。
- 先にdigest/lawを検査してから計数。source/入力/法則/axis/phase/terminal不一致・欠測はSTOP、価格fallbackなし。結果は一pathの構造RZ parityであり、平均compiled価格recordではない。

## 実行可能性

reader・runnerの人工接続は準備済み。実raw4本と必要snapshot/receiptは存在する。実内容hash、event digest、preparation再構成と実runtimeの成功は未確認であり、実行時の検査対象。S0a科学実行は今回未承認。

初回人工testの広いpackage importでQiskit passmanagerを通る副作用があったため、純metadata fixtureと明示submodule namespaceへ修正した。実行contextにもQC生成/transpile拒否を設けた。旧SOURCEを改変せず、guardに触れた場合は失敗STOPして再検討する。詳細はREPORT。

## 別途科学実行承認後だけ使う入口

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-h6-provider-coverage-20261110
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B scripts/run_s0a_saved_structure.py execute \
  --plan docs/research/s0a_saved_structure_preparation_20261113/EXECUTION_PLAN.json \
  --output docs/research/s0a_saved_structure_results_20261113/run01 --execute
```

上記は今回実行していない。fresh outputだけを受け入れ、既存結果/STOPを上書きしない。input SOURCEが合わない場合、SOURCE lockを無効化しない。技術修正と新計画の必要性を報告し、黙って科学retryしない。

許可対象は保存4path×XYの**新S8件**、そのための元保存input→構造metadata再構成とlaw/digest検査、保存Sとの一致照合。sampling、旧S再計数、QC build/compile、価格fit/残差再校正/平均予測、N/G、C2、他pathは含めない。

## Bとの境界

B固定最終技術統合`f09af82482aad11460c4bb73d3e13d7ada40be6f`を参照しただけで、共通API変更や取込み要求はない。Bは最終統合を継続管理する。S0a parityは平均law価格、閉状態、compiler残差、model資格の検証ではない。MeanPriceProvider/MeanKernel/connected-cluster/C1係数・false/nullは不変。

次はユーザーが科学scopeを判断、Cが許可された場合だけ8件を実施してSTOP、Bはparityと平均価格資格を区別する。現在は技術公開・remote照合後mandatory STOP。
