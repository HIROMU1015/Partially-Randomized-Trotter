# Track B — Algorithm Co-design

最新は[BM-0.5同値性監査・GPT packet](bm05_review_packet_20261005.md)。
compact再帰=BM floor/internal三次係数。同backend／同集約のscoreは同じ、共通項reuseも強い対照に可能。
現adapterのnew-method gateは不通過。BM-1は実行せず、application／別deltaの採否はGPTへ戻してSTOP。
[BM-1 v2 amendment](bm1_pilot_scope_amendment_v2.md)はleading heuristic／I2、primary count／secondary人工costを記録する。
独立branch/worktree `track-b-bm05-equivalence-20261005`。以下のBM-0原文・manifestは固定commitの履歴として保持する。

最新は[BF-A後GPT review・BM-0設計packet](bm0_review_packet_20261005.md)。
[B-F現行仮説を限定negativeとして閉じる](bf_current_hypothesis_closure_20261005.md)review判断を記録する。
B-Mはnative列・DF評価・小型pilotの**未実証の候補**。新scienceは0、BM-1実行は未認可、mandatory STOP。
文書作業は独立branch/worktree `track-b-bm0-design-20261005`。以下は過去のhandoff・実行履歴として保持する。

研究Bの全面再設計をGPTへ渡す入口は[固定資料handoff index](research_redesign_handoff_20261005.md)。
最新R0・過去STOP・構造結果・最新Aの別commit・既存提案入力を辿る。docs-only公開、研究方針の採否はGPT側、mandatory STOP。

最新は[BF1-R0復元結果・GPT handoff](bf1_read_only_recovery_result_validation_20261005.md)。
`BF1_READ_ONLY_RECOVERY_COMPLETE`、事後のpreregistered primary復元はBF-A。F/L有限最小値は同一。
新science data/rerunは0、bridgeは不足cellを記録し、mandatory STOP。原resultはINCONCLUSIVEのまま保持する。
result/audit: `artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/`。以下のR0準備記述は履歴。

新たな利用者指示は[BF1-R0契約](bf1_read_only_recovery_contract_v1.md)。別branch `track-b-bf1-read-only-recovery`で
既存1269 cellだけの一回のdeterministic replayを準備し、全outcome後STOPする。science rerunは未認可。
module: `src/trottertracks/algorithm_codesign/recovery.py`、runner: `scripts/tracks/algorithm_codesign/run_bf1_read_only_recovery.py`、
test: `tests/tracks/algorithm_codesign/test_bf1_recovery.py`、contract: `artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/`。

B worktree: `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`。
branch: `track-b-algorithm-codesign`。Aの最新worktreeとは分離する。
このcheckoutのA文書はM2 result commitのsnapshotであり、並行して進むAの現在状態を更新する場所ではない。

原BF-1は一回実行後の`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE`を保持する。
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
