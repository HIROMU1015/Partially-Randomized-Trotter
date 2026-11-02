# H6 native B0+I pilot6 — independent final binding check

Date: 2026-10-15 JST. Reviewer: `/root/h6_pilot_independent_audit`.

**PASS for the inspected launch binding.** The user's recorded instruction is 「計算を実行して」 in response to the maximum-six-worker proposal. This check validates that binding; it does not issue or replace user approval, reopen the source/scientific review, or report a successful computation.

- Runner SOURCE: `7300b39df4c53d854d5bc8934430af19fbdb0085`; preparation review: `6baf246cb46cf69536db6c16872286e5ed12e85a`. All ten runtime source files match their contract hashes and exact fixed Git bytes in the launch checkout.
- Final contract SHA-256: `756a95d7fbfe2f6176a21036f72896c090226a7b7bb14e485a778ebccb4a8da2`; approval raw SHA-256: `ab829bc6d5a828949093997f334ed480300c70bea62128198e1ed89b6b5eb1fb`; launch wrapper SHA-256: `2bb7f953a95ae31fc2dd062e737bd378cd7e5761b7ecd085fb2e6d4895952b2d`. These match the inspected binding receipt.
- Scope is unchanged: p5/q1, p10/q2, p15/q8, S2/T=0.8, cosine/sine; snapshot 1, evolution builds 3, control probes 12, measured wrappers 6, probe copies 6, compile reservations at most 6. Matched B0, new SCF/DF/state/trajectory/signal exponential actions and hardware shots remain outside this execution.
- Driver CPU 2 and worker CPUs `[4,5,8,9,10,11]` are disjoint and match allowed CPUs and the command. Thread variables are one and Qiskit parallelism is disabled. Per numerical role AS 32 GiB/RSS 9 GiB, numerical stage wall/aggregate CPU 14,400 seconds, and all-role admission 240.5 GiB agree with the proposal; the metadata launcher retains its separately bounded budget.
- Initial resource evidence shows approximately 468.7 GiB effective memory, PSI zero, approximately 308 GiB available filesystem space, ample inodes and quota disabled. Historical OOM counters are recorded as the baseline. These observations are initial evidence, not a substitute for the parent's immediately-before-launch fresh gate.
- The byte/profile receipt records five inputs, 230 fixed-provider files and 34,273 runtime-profile files passing, with no scientific input decode/import. Carry receipt and all eight historical exit-receipt byte hashes were independently checked. Known actual/historical unknown totals remain null/UNKNOWN, never zero; these exit receipts do not claim to be a complete compile ledger. No reset/refund is made and this is a separate maximum-six-reservation one-shot.
- At this check, `run01`, its `.oneshot` marker, launch stdout log and launcher wait receipt were absent. The owned control metadata temp/mpl/cache directories exist. Input mapping has exactly the five contract logical names, all under home; the output/control paths and exact launch wrapper agree with the contract.

The operative contract fields are approved/runtime-authorized/sealed true, with no automatic retry or next stage. Its descriptive `resource_proposals` string still says “unapproved”; this is stale wording, not the operative approval gate. If any contract byte is changed, all bound contract hashes/authorization fields and this binding reference must be updated before launch.

The parent must bind the actual commit containing this review to `authorization.review_commit`, bind the exact reviewed contract/approval digests, and perform its final fresh source/input/profile/carry/resource/CPU/unused-namespace checks. Then the already authorized single launch may proceed; a failed gate or STOP does not authorize retry. Existing long-C/polling, SIGKILL, missing-terminal/cleanup-UNKNOWN and runtime native-correctness limitations remain.

No source re-review, new test, scientific import, real build/probe/compile, or launch was performed by this reviewer. The prior 70-test evidence was not rerun. **Independent reviewer STOP.**
