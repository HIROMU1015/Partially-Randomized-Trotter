# Phase A引継ぎ：次は科学条件・契約採択、その後に別実行承認

**状態：`PHASE_A_IMPLEMENTED_ARTIFICIAL_PASS_AWAITING_CONTRACT_ADOPTION`。Mandatory STOP。** [技術報告](technical_report.md)／[契約草案](execution_contract_draft.md)／[machine草案](execution_contract_draft.json)／[入力候補](input_candidate_and_provenance.md)／[SOURCE固定](source_freeze.json)／[公開照合](publication_receipt.json)。

完了した技術作業は、保存NPZ→exact raw adapter、正直な固定DF scopeでの旧SPEC-01〜07接続、one-shot/authorization/保存runner、exact bit/trace高速化、人工36件＋旧98件の回帰、SOURCE/入力byte identity固定と公開。134 methods・6,777 subtests PASS、静的検査・再利用監査PASS。既知の未解決test failureはない。実入力への適合性・runtimeは未検証である。

## ユーザーが採択する必要がある条件

- 保存H4/1.00 Å/STO-3G/rank12の**固定DFモデル**を対象とするか。元電子Hamiltonianの厳密保証は対象外。
- stored modes 0〜7、全12 blocks、total N=4、onebody→保存block順というproposalを使うか。Nα/Nβ・S²は今回の保証に追加しない。
- U専用trial。候補はindices 0,1,2,3のdeterminantであり、HF/groundとは主張しない。
- exact d/α、sqrt_bits、root_width/max_bisections、record/log byte上限、canonical実行repository/ledger。
- guard・取得費・条件付きexact-ground access、強いcentered/構造対照、failure/未解像/one-shot消費・retry禁止・mandatory STOPを含む正式契約。

未決fieldはmachine草案で`UNSET`。人工test値や慣例値を科学条件へ代入していない。model_idはpinned raw＋mappingから承認後にexactに生成・独立検証するpolicyなので、採択前の科学decodeを要求しない。別のmodelへ変換してHermiticityを修復するpolicyは含まない。

## 次の担当

| 担当 | 作業 | 必要な承認 |
|---|---|---|
| ユーザー | 候補と未決条件を確認し、値・範囲を選んで契約採択を判断 | 未採択。今回のPhase A指示では代行しない |
| Codex | ユーザーが採択した内容を正式契約Cと記録に固定・公開 | 採択の明示指示後。ここでは実行しない |
| ユーザー | 固定S/C/inputを指定してPhase B一回実行を判断 | 契約採択とは別の明示実行承認 |
| Codex | 承認後だけ専用authorizationを固定し、初回実入力gate・全証拠生成/独立検査・比較/費用/partial保存、公開してSTOP | 現在未承認 |

input gateが失敗すれば消費済みclaim・原input・partialを保全しSTOPする。成功時でも認定できるのは、選んだ証拠/保証での十分stepとその取得費まで。実QPE/native/state準備や総資源改善、A2採択へ進まない。

新しい重大な科学選択が生じる場合の詳細レビューは別途開始承認を要する。今回新数学保証・新重要レビューは行わず、自動提案/開始しない。A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行、他研究の停止/保留状態を維持する。

**Phase Aの完了をPhase B承認と解釈しない。科学係数decode・実certificate・science authorization/grant/markerは全て0件。**
