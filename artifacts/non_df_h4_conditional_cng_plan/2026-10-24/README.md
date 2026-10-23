# H4条件付きC/N/G比較：計画のみ

2026-10-24。ユーザーが条件付き固定4構成C/N/G比較を正式採択。今回は実行計画と最小の人工検証だけであり、新しいH4 N/G・集計・順位・感度・sampling/compile・科学grantは0。Mandatory STOP。

- [実行計画・再利用・残差・MC/tail・承認範囲](../../../docs/non_df_h4_conditional_cng_plan_20261024.md)
- [非実行契約](execution_contract_DRAFT.json)：20条件関数・100仮定scenario・30比較領域、次回の明示承認が必要。実解析runnerは未実装。
- [40軸の入力pointer・不足条件](axis_conditions.json)、[550保存入力と514価格joinのSHA](inputs.json)
- [GitHub27資料の取得・照合](github_verification.json)、[正式採択とpolicy受領](policy_receipt.json)、[ユーザー指示](authorization.md)
- [構造・人工8検査](verification.json)、[既存関数を人工値だけで呼ぶ検査script](check_plan.py)、[環境と初期失敗の保全](test_runtime.json)
- [完了・保全・STOP](completion.json)、[公開状態](publication_status.json)

残差の既定値はnull。rho=0は仮想下限例で、未知実装誤差を消す仮定にはしない。C bias_distanceを直接再利用し、C totalから減算・L上界の自動加算をしない。未観測費用は未知変数のまま。次の担当はユーザーが契約を承認するか判断、Codexが明示承認後のみ再解析する。
