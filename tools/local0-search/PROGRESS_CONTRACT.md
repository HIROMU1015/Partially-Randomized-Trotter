# 中央との応答契約

Windowsクライアントは既存のSSH bootstrapに要求ファイルをバイト列で渡す。方式・物理範囲・数値の表現・予算の省略を保持する。開始・再開は対象run IDまたは新規範囲を一つ指定する。中央が継続runを作成した場合、`result.continuation.parent_run_id`を元run IDとし、`result.run_id`で実際の継続先を返す。

既存の応答フィールドはそのまま使う。中央が次の追加フィールドを返すと、クライアントは候補別の状態と実行数を表示する。中央が未返却の件数はunknownと表示する。

```json
{
  "progress": {
    "schema": "RPE_LOCAL0_PROGRESS_V1",
    "effective": {
      "preparation_budgets": {"signal": 0, "compile": 0, "audit": 8},
      "max_new_q3_per_method": {"PR_PAULI": 32},
      "resources": {"max_concurrent_jobs": "auto"}
    },
    "candidates": [
      {"candidate_id": "example-a", "method": "PR_PAULI", "state": "PREPARATION_WAITING", "reason": "SIGNAL_PACKET_MISSING"},
      {"candidate_id": "example-b", "method": "PR_PAULI", "state": "EVALUATION_RUNNING"}
    ],
    "stages": [
      {"stage": "compile", "queue_accepted": 8, "execution_started": 0, "execution_completed": 0, "execution_unit": "compile"}
    ]
  }
}
```

`queue_accepted`は中央の受理済み段階タスク数。`execution_started`と`execution_completed`は実行receiptに基づく。工程ごとの数え方は`execution_unit`に記す。旧runの消費済み枠と新runの残枠を返す場合は`effective`に両方を含める。

| 候補のstate | 表示 |
|---|---|
| `REUSED_SAVED_RESULT` | 保存成果を再利用 |
| `PREPARATION_WAITING` / `PREPARATION_RUNNING` | 準備待ち / 準備実行中 |
| `EVALUATION_WAITING` / `EVALUATION_RUNNING` / `EVALUATION_COMPLETED` | 評価待ち / 評価実行中 / 評価完了 |
| `BUDGET_HOLD` | 予算不足 |
| `UNSUPPORTED_GENERATION` | 未対応の生成規則 |
| `MISSING_FIXED_INPUT` | 固定入力不足 |
| `TECHNICAL_FAILURE` | 技術的な失敗 |

`runtime.phase=FAILED`や`RECONCILIATION_REQUIRED`は技術的な失敗として表示する。旧中央が途中結果に対して`REUSED_COMPLETED`を返しても、停止した実行を完了扱いせず、元応答をそのまま保存する。STOPの解除、既存runの予算更新、継続runへの引継ぎは中央の操作で行う。

既存の公開同期は実際に中央が確認したrun IDに紐付ける。接続に失敗した場合はSSHエラーを返す。
