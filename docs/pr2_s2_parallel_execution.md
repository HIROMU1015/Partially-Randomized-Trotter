# PR-2 S2並列実行層

## 位置づけ

この実装は、固定済みPR-2 V4/S2検証の科学条件を変更せず、S2のfull-wrapper
transpileをcell単位で並列実行するための運用層である。新しい科学的検証、追加sampling、
held-out評価、S3許可ではない。

元のserial runnerと実行中のprocessは変更しない。並列runnerによる実H4 S2の再実行も、
別途明示的に開始するまでは行わない。

## 不変条件

- `B0/B1`、`B2/B3`、rank 3/9 controlの候補集合を変更しない。
- `q=8`、`delta=0.1`、`r`、`K`、compiler設定、transpiler seedを変更しない。
- cell seedと各cell内のtrajectory生成・Welford集計順を変更しない。
- cosine/sineのpaired full wrapperを同じ既存関数で構築・transpileする。
- `initial32`全cellの完了後にだけ固定拡張規則を適用する。
- 選ばれたcellだけを独立`extension96` streamで128 samplesへpoolする。
- extension完了後に固定decisionを行い、その後にrank 3/9 controlを実行する。
- held-out NPZをloadせず、S3を許可せず、quantum shotを実行しない。
- worker完了順にかかわらず、入力ordinal順へ戻してから候補を集約する。

## 実装

- `src/trotterlib/pr2_v4_s2_parallel_execution.py`
  - `spawn` process poolを使用する。
  - 既定4 workers、上限8 workersとする。
  - phaseごとにbarrierを置き、段階間の依存関係を維持する。
  - 既存`_compile_cost_batch`だけをworker内で呼び、科学ロジックを複製しない。
  - 任意のSQLite persistent compiled-cost cacheをworker間で共有する。
  - phase開始・各cell完了・phase完了を時刻、worker PID、cell identity付きJSONで出力する。
  - worker例外にはordinal、method、rank、q、r、K、sample count、seedを付けて再送出する。
- `scripts/run_pr2_s2_development_parallel.py`
  - BLAS/OpenMP thread数をworkerごとに1へ固定する。
  - source freeze、V4 PASS、dedicated test log、非上書きを要求する。
  - serial成果物とは異なる既定output名を使う。
  - Python例外時は完全なtracebackを非上書きfailure JSONへatomic writeする。
- `tests/test_pr2_v4_s2_parallel_execution.py`
  - toy Hamiltonian上でserial/parallel出力の完全一致を検査する。
  - worker完了順から独立したcanonical orderを検査する。
  - persistent cacheを再利用した実行の完全一致を検査する。

## 実行例

実H4 S2を開始するには、並列化専用testのJUnit XMLと、既存のdeviation-free V4
artifactを指定する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
.venv311/bin/python scripts/run_pr2_s2_development_parallel.py \
  --workers 4 \
  --test-log /tmp/pr2_v4_s2_parallel_tests.xml \
  --v4-artifact artifacts/pr2_v4_s2_development/2026-09-28/pr2_v4_correctness_result_v1.json \
  --failure-report artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_parallel_failure.json
```

SQLite cacheは既定で
`artifacts/pr2_v4_s2_development/2026-09-28/cache/`以下に置かれ、git管理対象外である。
中断後に同じcacheを指定すると、既にtranspile済みの実回路costを再利用できる。ただし最終JSONは
非上書きであり、既存成果物を置換しない。

長時間runではstdout/stderrと終了コードを、runnerとは別の監視shellから永続logへ保存する。
Python例外はfailure JSONにも残る。SIGKILLなどPythonが捕捉できない終了でも、監視shellが残れば
終了コードをlogへ記録できる。

## 現在確認済みの範囲

toy Hamiltonianの2 cellについて、1 worker serialと2 worker parallelの返却dictは完全一致した。
同一SQLite cacheを使う再実行も完全一致した。これは実装回帰であり、実H4 S2の並列再実行結果や
速度倍率の証拠ではない。実H4 benchmarkは、現在のserial検証と資源競合するため実施していない。
