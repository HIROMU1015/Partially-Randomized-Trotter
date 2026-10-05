# Track B — Algorithm Co-design

新たな利用者指示は[BF1-R0契約](bf1_read_only_recovery_contract_v1.md)。別branch `track-b-bf1-read-only-recovery`で
既存1269 cellだけの一回のdeterministic replayを準備し、全outcome後STOPする。science rerunは未認可。
module: `src/trottertracks/algorithm_codesign/recovery.py`、runner: `scripts/tracks/algorithm_codesign/run_bf1_read_only_recovery.py`、
test: `tests/tracks/algorithm_codesign/test_bf1_recovery.py`、contract: `artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/`。

B worktree: `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`。
branch: `track-b-algorithm-codesign`。Aの最新worktreeとは分離する。
このcheckoutのA文書はM2 result commitのsnapshotであり、並行して進むAの現在状態を更新する場所ではない。

現在はBF-1一回実行後の`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE`で停止。
source `e59344a`、authorization-only child `cc971e4`を固定して実行したが、NumPy整数のJSON保存例外で中断した。
入口は[BF-1結果照合](bf1_one_shot_result_validation_20261005.md)。retry、BF-2は未認可。
利用者指示により、研究方針全体の修正はGPT側、細かな検証はCodex側で扱う。
別branch `track-b-bf1-serialization-repair`の[JSON保存修正・限定検証](bf1_serialization_repair_review_20261005.md)は
実装上の修正のみで、旧source/authorizationによる科学再実行を認可しない。
`296ec7e`の[旧最終source review](bf1_final_source_review_296ec7e.md)と以下の契約・準備packetは結果前履歴。
[BF-1 preregistration v1](bf1_preregistration_v1.md)、[execution gate amendment v2](bf1_execution_gate_revision_v2.md)、
[assembly guard amendment v3](bf1_assembly_guard_revision_v3.md)、
`artifacts/track_b_bf1_preparation/2026-10-05/v3/`を一組として読む。
v1/v2準備packetと[旧実装review packet](bf1_implementation_and_review_20261005.md)は履歴として保持する。
one-shot markerはconsumedのまま保持する。全outcomeでmandatory STOP後、研究Bの方針を全面再評価する。

BF-0のreviewされた履歴は[数学契約](bf0_mathematical_contract.md)、[claim監査](bf0_prior_art_claim_matrix.md)、
[pilot proposal](bf1_minimal_pilot_proposal.md)、[review request](bf0_external_review_request_20261004.md)。
BF-1の具体化は新preregistrationへ記録し、review対象の本文を無言で書き換えない。

code: `src/trottertracks/algorithm_codesign/`、runner: `scripts/tracks/algorithm_codesign/`、
test: `tests/tracks/algorithm_codesign/`。`src/trotterlib`の共有API・Aのartifact/status/contractは変更しない。
repair reports/audit: `artifacts/track_b_bf1_serialization_repair/2026-10-05/`。
