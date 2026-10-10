# G10最終source review受領と実行境界

状態：**G10_FINAL_SOURCE_REVIEW_ACCEPTED_EXPLICIT_ONE_SHOT_INSTRUCTION_PENDING**。
利用者が2026-10-10に「こんな感じで進める」として
[GPT最終レビュー](../../research/track_b_G10_source_final_review_20261010.md)を採用した。
これはdocs-onlyの受領記録で、固定source Sへの必須修正なしという判定を保存する。

## 固定sourceと受領commitの区別

- scientific source S：`05c5ef23fce775a822ab5686f5da2f0d77675864`。
- [固定machine contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json) SHA256：`71ae2310a08f9d111faab76f27e6de277189ab132370d8996a0e2420c9510818`。
- receipt branch：`track-b-g10-final-review-intake-20261010`。
- reviewの参照元：`/home/abe/Project/Partially Randomized Trotter/track_b_G10_source_final_review_20261010.md`。35,316 bytes、SHA256 `20659bd1168c4cf53816fdecdd286c13a3b5ea603ffd4480d057736dbd8e0a67`。

本branchはレビュー・索引追記を含むため、authorization-only childではない。実行HEADには使わない。
今後のAは必ずSの直接の子とし、本受領commitから分岐してAを作らない。
source branch/worktree、code、tests、contract、pending authorization、過去result/marker/STOPは保持する。

## 採用したレビューの意味

レビュー§1/14の判定は、指定Sと固定scopeについて別途明示one-shot承認へ進んでよい、必須source修正なし。
完走・key合成成功・P7改善・新規性・論文十分性を保証しない。
同p=(1/5,3/10,1/2)、x=5/7、同3-system-qubit synthetic providerでm3/5/7を比較し、
17 direct rows/34 axes、m5は保存anchorの共通policy再会計のみ。各m内で同じ有限full first operator momentを比較する。
異なるm間を同じexponential精度の費用としてランキングしない。geometry/basis/DF rank/split/PF windowは適用外。
known/development contextで、CI/独立外部再現ではない。

保存任意proposal下界とcanonical lawの費用を分け、未分離を非改善としない。
不完了prefixの勝敗を科学的判定に使わない。general fullの優位／非優位だけから有限mean恒等式を採否しない。
全outcomeでSTOP、m9・新p/x/provider・precision/seed/backend・分子/DF/QPEへ自動進行しない。
研究上の解釈・採否・次scopeはGPT/利用者へ戻す。

## 次のexecution gate

レビュー§14は明示one-shot実行承認を待つよう指定している。今回の採用メッセージをその承認として代用しない。
認可後、Sから別execution branch/worktreeを用意してauthorization-only直接子Aを作る。
Aの変更許可は新G10 authorization JSONと任意receiptだけ。source/contract/instructionをbindし、
Aをpush、remote SHA=A、clean、critical/protected/runtime一致、fresh markerを確認する。
固定runnerを一回、runs=1/retries=0、全結果mandatory STOP。

現在authorizationはpending/null/false、A未作成、runner呼出し/登録P3/P7評価/backend/marker作成0。

## 読み取り専用の照合と証拠限界

本受領時に原source worktreeでstdlib preparation verifierを確認：1,378 checks、113 critical/1,241 protected paths PASS。
source branch remote SHA=S、clean、contract identityを照合。登録angle/予算/matrix/samplerは取得しない。
レビューのg10_launch blob `42f99b314f098d3029bacf613ac33692c44de048`と3,708 bytesもsourceに一致する。
41 focused testsは既存準備証拠で、今回再実行していない。GPTの49 selfchecksとは合算しない。

レビュー§16の`g10_source_review_support_20261010/`は今回のローカル受領元に存在しない。
49 checksはGPTの報告として保持し、添付検算scriptを取得・再実行・独立再現したとは表現しない。
レビュー§11のmarker後import/OS/serialization/disk障害の残余リスクも維持する。
発生時は消費済markerを残し、欠落と原因を報告してSTOP。marker削除や再実行で回復しない。

[受領manifest](../../../artifacts/track_b_g10_final_review_intake/2026-10-10/intake_manifest_v1.json) / [provenance](../../../artifacts/track_b_g10_final_review_intake/2026-10-10/provenance_audit_v1.json) /
[read-only source照合記録](../../../artifacts/track_b_g10_final_review_intake/2026-10-10/source_preparation_receipt_v1.json)。
