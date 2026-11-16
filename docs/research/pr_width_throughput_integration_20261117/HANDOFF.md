# 統合通常入口：B2・Skill・探索順の検証担当へ

branch：`codex/pr-width-throughput-integration-20261117`。[REPORT](REPORT.md)／[固定入力・再利用SOURCE](SOURCE.json)／[CHECKS](CHECKS.json)。実H4科学実行は0件。人工結果やp9の取得優先順位を科学的最安と扱わない。

## PRで両機能を使う最小差分

所有する新roundの既存problem・routes・保存入力・精度/科学範囲を保持し、次を追加する。値は設定例であり、科学実行やprofileの最適性を採択したものではない。

```json
{
  "methods": ["PR_PAULI", "PR_DF"],
  "science_approved": false,
  "executor": {
    "scheduler": "concurrent",
    "max_concurrent_jobs": 32,
    "dp_workers": 8,
    "wait_for_running_jobs": true,
    "prefix_cache_directory": "prefix_cache",
    "prefix_checkpoint_rounds": [5, 11]
  },
  "search_overrides": {
    "requested_widths": {
      "PR_PAULI": [0.1, 0.2, "1/5"],
      "PR_DF": [0.1, 0.05, 0.2, "1/5"]
    },
    "adaptive_hybrid": false,
    "allocation_exchange": null
  },
  "overlap_ready_batch": false
}
```

JSON数値と厳密文字列を区別する。h・T_mは元入力のまま。mapper非対応や価格不足はその幅を保留し、READY候補を進める。部分価格が届くと同じCompletionで再判定・installし、同じqueue/累積予算へ戻る。要求IDと評価IDを別欄で保持する。

幅宣言を省略した元round05は元WideDriver、`executor.scheduler`を省略又はlegacyとした複数幅roundは旧executorを使う。適応先読み/振替/準備待ち先行Q3は個別設定のままで、既定を有効化していない。幅版も明示的な`overlap_ready_batch=true`なら、準備前のREADY入力で先行Q3を実行し、準備完了後も同じqueue/予算を使う。実H4での効果は未測定。

## 同じ開始・再開コマンド

所有integration checkout内に新runを置く。以下の1コマンドを再開時にも使う。実H4の今回の上限はplan-onlyである。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/pr-width-throughput-integration-20261117
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
scripts/run_rpe_schedule_search.py --problem PROBLEM.json --round OWNED_ROUND.json \
  --run-dir docs/research/pr_width_throughput_integration_20261117/local_plan/NEW_RUN \
  --max-cycles 1 --plan-only
```

PRは同じ入力でplan-only再開すると新候補metadataを追加する場合がある。完了科学処理の再起動と区別する。実行済みのUNKNOWN・stale・科学STOPは自動retryしない。未観測の別配分をprefixから評価する場合も1 Q3予算であり、完了jobの再利用だけが0件。

## B0Iを同じ入口で使う

`methods`をFull PF/B0Iの方式にし、公開5efaの`PROBLEM.json`・`SAVED_PLAN_ROUND.json`と同様にp/formula/幅の**範囲**・精度・予算・固定D providerを渡す。`pf_completion.selected`と次のshot列は記入しない。同じ`run_rpe_schedule_search.py --problem ... --round ... --run-dir ... --plan-only`が既存method CLIへdispatchし、未評価pを選ぶ。B0IにPR専用WideDriverや新optimizerは使わない。

D providerは固定`36ad751d4ebf0ec160532615da6425b268a0c18a`を所有viewに展開する。今回の検収viewは`pr-width-b0i-plan-source-36ad-20261117`。Dの`output_root/registry_file`はそのview内の所有directoryを指定する。Dの相対出力guardを緩和せず、他担当viewへ出力しない。

固定保存入力の期待選択はp9/Yoshida4/32h/5だが、規則は汎用のまま。別入力にp9を強制しない。p10はFull PF alias、未評価prefixは各p自身の信号・価格・元Full DF-H targetへ接続する。

## 人工検収の再現と記録

```bash
# 実Completion + scoped監査/receipt + 共通queue、途中READY、再開
PR_WIDTH_QUEUE_EVIDENCE=/tmp/NEW_QUEUE_OUTPUT PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests \
  python3 -B tests/test_pr_width_throughput_integration.py

# 通常CLI/queue + 元guard付き人工DP。逐次参照・HIT・identity MISS
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests \
  /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
  scripts/check_pr_width_prefix_integration.py --output /tmp/NEW_PREFIX_OUTPUT

# 固定H4 metadata plan-only + B0I人工補完/shot/Q3（legacy/concurrent）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests \
  python3 -B tests/test_b0i_unified_entry.py --output /tmp/NEW_B0I_OUTPUT
```

出力先はfresh directoryを指定する。保存した記録：

- `checks/queue/PROGRESS.jsonl`：4 ENQUEUE/LAUNCH、1 invocation、2方式。`INVOCATIONS.jsonl`・`completion/SEARCH_CYCLES.json`：共通予算・後続READY・再開。
- `checks/prefix/{QUEUE_EVENTS,DP_PREFIX_EVENTS}.jsonl`：donor完了後の4 HIT・各12round。`CLI_SUMMARY.json`：逐次とMSE上下界/判定のexact一致。`IDENTITY_SUMMARY.json`：別configuration/信号はMISS。
- `checks/b0i/{saved,artificial_legacy,artificial_concurrent}/CHECKS.json`：p無指定の選択、通常shot seed、provider/Q3追加0。
- `checks/INTEGRATED_REGRESSION.log`、`PREFIX_REGRESSION.log`：既存入力・精度/queue/price互換性の回帰。
- `checks/OVERLAP_INTEGRATION.log`：明示先行Q3・既定無効・fatal時STOPの3検査。
- `checks/plan/COMMANDS.json`：実行したH4 plan-onlyの開始/再開コマンド。`PLAN_ACCEPTANCE.json`は追加metadataと完了stage再起動0を区別する。全metadataは所有`local_plan/`に保全し、公開は要約に限定。

## 探索順を比較する次担当へ

この統合版を固定し、同じproblem・physical SOURCE・候補/幅scope・価格規則・Q3契約・予算・executor/DP profileを使い、別の所有round/runで選択規則の設定だけを変更する。元SOURCEや既存runを書き換えない。開始・再開の入口は上記と同じ。queue投入順と規則の採用順、全完了結果と採用結果を分けて評価する。今回、候補選択規則の優劣や実H4の最安幅を評価していない。

人工成果を科学catalogへ取り込まず、MODEL価格・law距離・CI・正式Gの資格を引き上げない。新H4信号/較正/compile/実Q3は後続の別承認範囲。本統合の公開・remote照合でmandatory STOP。
