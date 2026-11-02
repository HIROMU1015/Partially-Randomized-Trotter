# Track B：G1限定source準備

2026-10-09 JST。基点`6c91d1b17a70f773080be9315fabd10e926dae44`。
利用者の「その方針で進めてください」に従い、直前に提示した限定script/controller実装・off-domain tests・source公開を行った。
[source review資料](../../tracks/algorithm_codesign/g1_source_review_20261009.md)にscope、実装、未検証点と次の境界を記録した。

stdlib exact algebraとrestricted ASTによる独立数式監査を実装したが、本modelに対しては未実行。
source-bound one-shot controllerは全8 echo後に8 solveを行い、初回失敗STOP・exclusive marker・retry=0を守る。
旧guard/verifier/compiled binaryは変更せず利用する。取得失敗と不正証明を別分類にし、verifierのexit1もverdictから照合する。

focused off-domain testsは59 PASS / 0 FAIL / 0 ERROR。
本audit()呼出しを禁止し、mock backendと人工2変数のFraction証明だけを使った。
本監査・実backend・固定8 inputs solve・実one-shot marker・新science authorization・compile・synthesisは0。
old source/contract/authorization/result/marker/STOPは保持し、現状要約と入口のdocsにだけ追記した。

判定は`G1_SOURCE_PREPARED_FOCUSED_TESTS_PASS_AWAITING_SEPARATE_EXECUTION_INSTRUCTION`。
実行可能sourceは公開commitのSHAで識別する。別のsource-bound明示指示が来るまでpacketを開始しない。
全outcomeでGPT G1へ戻すという既存方針を維持し、production・登録B2/B3へ進まない。
