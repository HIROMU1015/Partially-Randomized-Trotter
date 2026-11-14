# Cへの引渡し：D7 CALIBRATIONは実行前STOP

一次資料：[REPORT.md](REPORT.md)、[PREFLIGHT.json](PREFLIGHT.json)。固定実装 `6069605d50270111d1510478b08e11b8e5a18d86`、開始時HEAD `a43050c786c741d267ddf322232d888780efc03c`。D7は今回採択・較正最大18 compile許可済み。

- 予定18／実行0／PASS 0／FAIL 0／UNKNOWN・未取得18。新価格record、compiled receipt、implementation evidenceは0件。fit入力は渡していない。
- A `7a0c9468b` の `check_plan` でD7 rank3・共通domain・較正/holdout分離を確認した。
- pairedは固定callerが同seedから再sampling後に親比較する実装で、今回の「親を再samplingしない」を満たさない。toy spyで確認した。
- 較正5 cellの `IMPLEMENTATION_EVIDENCE.json` は固定callerでは `{}`。既存の証拠出力分岐はm5/m6だけである。
- C `c628e9e8` のcanonical round guardは、非configuration較正cellへ偽適用しない。新しいround identityは作成していない。

必要な後続は、固定SOURCE変更の許可→保存親events直接連結・較正実装証拠接続の最小修正→同一D7条件で取得再開→公開後の較正record受入確認である。Cのfit/係数凍結・HOLDOUT・科学的資格付けには今回進まない。

finite cohortは `DIRECT_COMPILED_FINITE_SAMPLE` / `FINITE_COHORT_ONLY` の予定であり、law E[C]ではない。価格精度2%、CI、rare寄与、長時間外挿、N/Gは未資格。mandatory STOPを維持する。
