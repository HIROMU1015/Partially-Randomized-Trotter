# 引継ぎ：原因・充足性監査完了後mandatory STOP

基点結果commit86b8b0027、C科学SOURCEe80d75108、旧科学SOURCE0c329b30。今回の調査branchは `codex/non-df-h4-certificate-gate-audit-20261023`。旧科学コード・旧run・grant・ledger・marker・STOPは変更していない。

DF range falseはsourceの表現別proof留保で、数値範囲逸脱を示さない。literal二値測定±B_savedとencoded DF行列のnormを別scopeへ分ける。既存数値包含を最大限再利用し、40軸の再計算を既定にしない。

次の担当と承認境界：

1. **ユーザー**：`minimal_next_contract_DRAFT.json` の技術段階T、必要なら局所数値段階L、compiler段階K、受理段階Aのどこまで進めるか判断する。compilerの対象が失われた旧programか、新たに保存するprogramかも明示する。
2. **Codex**：承認後に別SOURCEでscope-aware proof schema/checker・人工テストを実装し、技術sealでSTOP。Lの実H4局所行列診断は別の具体的実行承認・新grantが必要。
3. **Codex**：正式certificate受理とN/G再会計を自動接続しない。無条件のPCG64独立性証明を新たな必須条件に追加せず、既存のIID仮定を明記する。

現在のDRAFTは実行不可、execution_authorized/grant_valid/dispatch_enabled false、grant/approval null。旧消費済みgrantの再使用、旧run retry/resume、実H4のreference/law/bias再計算、sampling/RNG replay/compile、N/G会計、詳細科学レビュー・最終freezeは今回の範囲にない。調査branchのpushは更新共通指示により承認済み。

再現時は `audit_saved_evidence.py` がoutputを排他的に作る点に注意する。保存済み監査ファイルを消去・上書きせず、未使用の別技術outputを用意するか、既存記録を読取照合する。人工testにも実H4 modelを入力しない。

報告書・matrix・SOURCE・保存監査をcommitし、承認済みのpushを実施した後mandatory STOP。次の科学作業は承認待ち。
