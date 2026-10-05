# Track B — Algorithm Co-design

B worktree: `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`。
branch: `track-b-algorithm-codesign`。Aの最新worktreeとは分離する。
このcheckoutのA文書はM2 result commitのsnapshotであり、並行して進むAの現在状態を更新する場所ではない。

現在は`2d0ddaf`への利用者review `PROCEED_BF1_AFTER_MINIMAL_EXECUTION_GATE_REVISION`を反映する段階。
[BF-1 preregistration v1](bf1_preregistration_v1.md)、[execution gate amendment v2](bf1_execution_gate_revision_v2.md)、
`artifacts/track_b_bf1_preparation/2026-10-05/v2/`を一組として読む。
v1準備packetと[旧実装review packet](bf1_implementation_and_review_20261005.md)は履歴として保持する。
科学実行は未認可。全outcomeでmandatory STOP後、研究Bの方針を全面再評価する。

BF-0のreviewされた履歴は[数学契約](bf0_mathematical_contract.md)、[claim監査](bf0_prior_art_claim_matrix.md)、
[pilot proposal](bf1_minimal_pilot_proposal.md)、[review request](bf0_external_review_request_20261004.md)。
BF-1の具体化は新preregistrationへ記録し、review対象の本文を無言で書き換えない。

code: `src/trottertracks/algorithm_codesign/`、runner: `scripts/tracks/algorithm_codesign/`、
test: `tests/tracks/algorithm_codesign/`。`src/trotterlib`の共有API・Aのartifact/status/contractは変更しない。
