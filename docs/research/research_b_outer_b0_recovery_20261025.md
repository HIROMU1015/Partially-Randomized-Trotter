# 研究B B0：技術復旧と新attempt

旧SOURCE `8d0cb96df9d1f6229f207550634c7e11aff2851c` と[旧失敗原本・STOP](../../artifacts/research_b_outer_b0/20261025/native-pilot-once/)は保全する。今回の修正・同条件再実行はユーザーが明示的に指示した範囲。承認証明・grantは作らない。

## 将来attemptの限定変更

科学fixture／sector／state、6候補・両T・4代表、全48 trajectory・96 compile要求、complex128、operator 2-normのabsolute atol1e-12／rtol0／phase fitなし、compiler profileは[既存契約](research_b_outer_a0_b0_execution_preparation_20261025.md)と同じ。旧attemptを上書き・resumeせず、各attemptはfixed clean SOURCEと排他的fresh outputで実行しSTOPする。今回許可された技術復旧に限り、原因と修正理由を確認して新attemptを開始できる。原因不明の反復・科学的条件の変更には拡張しない。

まず `--diagnose-first` で旧失敗identity B3／T=+0.18／代表0／Xだけを検査する。入力5境界、config6、event／event operator各6、RTE sequence12、D block18、独立exponential9、system／controlled／reference各1、operator gate4、expectation2、compile1。費用、全mean、他caseは起動しない。Metadata scopeに登録した全境界を照合する。

FAIL時には数値差・actual／reference matrix、raw／compiled回路をgzip+base64 QPYでpartial_results内に保存する。成功したgate・meanの数値をeventsへ保存し、wrapper rowをcompiled gateより前にpartialへ登録する。既存EvidenceのJSON／総出力枠は維持する。保存失敗もSTOP。共通認証・汎用frameworkを追加しない。

診断で明確な実装問題を確認した場合だけ最小修正・関連テストを行い、新科学SOURCEをcommit固定し、同じ科学条件の全B0をfresh outputで実行する。全gateを費用96要求に先行させ、数値許容差を緩和しない。技術的に解決できないcorrectness FAIL・識別力不足は保存してSTOPする。

## 実績

この節は各新attemptのSOURCE・結果・原因・修正・人工検証・公開内容を追記する。旧契約や旧結果を遡及変更しない。A0再実行、A1包含、MC、shots、新PF・grid、全RPE評価は対象外。
