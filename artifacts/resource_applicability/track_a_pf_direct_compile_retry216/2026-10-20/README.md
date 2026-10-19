# PF directly compiled retry: remaining 216 axes

See [report](../../../../docs/research/track_a_pf_direct_compile_retry216_20261020.md).

- `retry_axes.csv`: exact 216 authorized axes; disjoint from 572 reused acceptances.
- `scope_and_calls.json`, `user_approval_receipt.json`: scope and explicit approval.
- `technical_checks.json`: 13 new artificial checks and 13 existing boundary checks.
- Sealed contract, fresh gates, independent native exit receipts and `results01/` are added after launch/completion. Preparation alone is not evidence of scientific completion.
- Original interrupted-run global final gate remains missing. Reuse does not retroactively certify that run.

Actual source: `0c5fcc83ef55045c2c64746aadfaf3128667b058`. See `launch_evidence.json`, `sealed_contract_and_grant.json.gz`, `running_observation01.json` and `RUNNING_HANDOFF.json`. This packet currently records a running attempt with task-level control STOPs, not full success. The sealed source must not be edited while its workers run.
