# Track B — Algorithm Co-design

B worktree: `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`。
branch: `track-b-algorithm-codesign`。Aの最新worktreeとは分離する。
このcheckoutのA文書はM2 result commitのsnapshotであり、並行して進むAの現在状態を更新する場所ではない。

現在はBF-0 review `PASS_FOR_BF1_PREREGISTRATION`受領後の結果前準備段階。
[BF-1 preregistration](bf1_preregistration_v1.md)、[実装・review packet](bf1_implementation_and_review_20261005.md)、
`artifacts/track_b_bf1_preparation/2026-10-05/v1/`を一組として読む。
科学実行は未認可。全outcomeでmandatory STOP後、研究Bの方針を全面再評価する。

BF-0のreviewされた履歴は[数学契約](bf0_mathematical_contract.md)、[claim監査](bf0_prior_art_claim_matrix.md)、
[pilot proposal](bf1_minimal_pilot_proposal.md)、[review request](bf0_external_review_request_20261004.md)。
BF-1の具体化は新preregistrationへ記録し、review対象の本文を無言で書き換えない。

code: `src/trottertracks/algorithm_codesign/`、runner: `scripts/tracks/algorithm_codesign/`、
test: `tests/tracks/algorithm_codesign/`。`src/trotterlib`の共有API・Aのartifact/status/contractは変更しない。
