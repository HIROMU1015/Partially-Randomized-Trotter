# 2026-10-18：必要範囲を選んだPR検証と並列実行の準備

利用者は、総当たりではなく必要な変数範囲を選び、検証間・検証内を並列化する方針を承認した。現在並行チャットはない。AI subagentは使わず、主agentで調査・人工検証を行った。

完了済みRARE_SENSITIVITY/SELECTION_SAVEDの再実行を除外し、PR境界16構成、希少層4群n8、新ε6点、直接対照3構成、H8 T2点、H6/H8各1.40Åを提案した。n8はpilotであり統計的十分性とはしない。距離の新入力生成→封印STOP→信号/compileの別承認を維持する。重い全探索や追加独立holdoutは自動採用しない。

計画SOURCE `daf701d8e4fb25a22e5c383202f535f9db7a6e25`。metadata DAG/CLIはscientific dispatchを持たず、全15scopeは未承認・adapter未seal。18固定fileのGit blob/SHA/ASTを監査し、人工21＋既存135=156件PASS。初回人工テストでsignal/cost候補リストの共有が範囲差検出を妨げる問題を検出し、リストを独立copyにして修正後全PASSした。これは本番実行・科学的なretryではない。

[計画報告](../track_a_targeted_parallel_validation_plan_20261018.md)に、再利用先、未完成adapter、軸/trajectory上限、STOP、承認単位を記載。既存完了cloneをcleanに保ち、科学実行・数値再解析・native compile・新入力生成は0。V2 NO_GO、V1、H5/H7、旧SOURCE・grant・ledger・markerは変更しない。次はCodexがC/B/Dの具体的な実行準備を完成し、その後ユーザーが実行する単位を別途承認する。今回mandatory STOP。

## 後続の実行承認とCの完了

利用者から「実行準備ができたらそのまま実行してよい」の追加指示を受けた。C専用adapter・固定manifest・人工8件を追加し、SOURCE3156b036…で一回実行。保存anchor1,520行再現、新2,280行/12比較、native exit0/wait完了、終了gate PASS。通常biasのε=.00075は+12.769532%、.000625は−0.497174%。実測/モデル単価、有限n32の点比較を分離する。全人工164件PASS。他系列は準備未完了で科学起動していない。各単位のmandatory STOPを守り、この結果を保存してSTOPする。残る範囲への条件付き実行承認は保持し、承認だけを再要求しない。
