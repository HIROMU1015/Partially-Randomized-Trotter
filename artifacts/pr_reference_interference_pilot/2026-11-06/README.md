# 新A限定pilot：保存出力

科学SOURCEは `602ca4b67b1037e221e910a2733444630c5add62`。現在の結論と条件は [REPORT](../../../docs/research/pr_reference_interference_pilot_20261106/REPORT.md)。

- `A1_REFERENCE_PHASE.json` と `A1_SIGN_IDENTITY_NEGATIVE_TIME.json`：vacuum family、符号・identity・負時間。
- `A2_SYNTHETIC.json`、`A2_H4_rare*.json`、`A2_FUSION_*.json`：source/native・X/Y・全operatorまたはstatevectorの数値correctness。
- `A3_PRICES.json`、`PRICE_COMPARISON.tsv`：両軸・強baseline・MC平均とpaired差。`raw_prices.tar.xz` は48 JSON、各sampleのseed・ordered event記録を保持。
- `native_circuits.tar.xz`：q1/q8、両stratum、3armのX軸native代表回路12 QPY。reference resetを含む。個別QPY hashはraw sampleに記録。
- `A4_RPE_ACCOUNTING.json`、`A4_ROUNDS.tsv`：保存shot scheduleと観測3roundのsubtotal。未取得価格・新native diskはnullで、全18round改善は未確定。
- `common.pkl.xz`：実際に使った保存preparation・Hamiltonian・stateのsnapshot。元の公表済み入力との対応は `INPUTS.json`。展開した `common.pkl` はruntime cacheとしてgitignoreする。
- `ENVIRONMENT.json`、`SOURCE.json`：再利用元、入力・実行SOURCE・archive hash、実行stage、並列度。`EXECUTED_SOURCE_A12.py` と `EXECUTED_SOURCE_A23.py` は実行時版の保全。
- `STOP.json`：科学作業完了によるmandatory STOP。T合成、長round価格、別候補の追加は実施していない。

SOURCE moduleの入口は `src/trottertracks/pr_reference_interference_pilot.py`、runnerは `scripts/run_pr_reference_interference_pilot.py`。実行環境はENVIRONMENT記載の既存venvを読取専用で使用し、`PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`、BLAS 1 thread/workerでprocess並列化した。新規SCF/DF fit/ground solveはない。現在の出力はSTOP済みであり、runnerを再実行しない。再実行・追加検証は別のユーザー判断・必要な実行許可による。
