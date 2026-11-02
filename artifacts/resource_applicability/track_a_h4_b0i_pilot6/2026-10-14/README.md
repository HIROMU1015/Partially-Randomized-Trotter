# H4 B0+I representative six wrappers — approved one-shot

Client date: 2026-10-14. Fixed SOURCE: `8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`.
Preparation REVIEW: `34ad305d55a797b57d00aebed99fc03a49a3ed2f`.

The user accepted the immediately preceding concrete execution proposal with
「次の作業に進んで大丈夫です」. The exact approval scope and evidence hash are
in `user_approval_evidence.json`, `authorization.json` and `binding.json`.
The prior scientific review is preserved verbatim in `scientific_review.md`.
B0+I is now an additional deterministic evaluation target; the original B0 and
historical comparison remain intact. The fixed implementation's validation
method/namespace labels remain unchanged.

Exactly six new full-wrapper compile calls: 0.70 Å p5/q1, 0.80 Å p5/q1,
1.40 Å p3/q1, each cosine and sine. Linear H4/STO-3G/DF rank12, eight
system modes plus one ancilla, T=0.8, ordinary control, structured Gaussian
lowering and PF S2; compiler options and frozen inputs are unchanged.
One worker on CPU0, driver CPU1, observer CPU2; internal threads and Qiskit
processes one. Driver/worker AS and RSS caps 8 GiB, observer AS256 MiB/RSS64
MiB, admission120.25 GiB, stage3600 s/task600 s, output256 MiB. These are
approved caps, not predictions of needed RAM or wall time.

`preflight_verification.json` records the byte-only source/profile/input
checks and passive CPU/memory/pressure/OOM/filesystem/inode/quota check.
The runner repeats its fresh gate immediately before claiming the one-shot
lock. Existing 44 artificial tests and the scientific review are retained;
no artificial compile or repeated scientific review was run.

`plan.json`, `authorization.json`, `review.json`, and `binding.json` are the
fixed launch artifacts. The review is a Codex 1 technical binding check;
no new independent reproduction or CI certification is claimed. The
absolute command is in the plan. Private output/control paths are unique,
under the user's home, and excluded from Git. No private venv/shared
configuration, H8 job, GPU, SCF/DF/state/signal or trajectory regeneration
is part of this execution.

Historical carry remains 73,023 consumed/reserved calls, 13,615,088,338
charged bytes and 268,318.34715263546 recorded wall seconds. At most six
new calls gives 73,029, below the unchanged 74,805 actual cap. Failed calls
are not refunded. Cumulative charge cap remains17 GiB.

Completion or fail-closed STOP is terminal. The six-call grant does not
allow retries/resume, 10/240 calls or a next stage. The result audit will
record measured metrics, old-axis RZ differences, saved-shot cost accounting,
native termination/cleanup, sampled memory and observed phase timing. A
phase's wall time includes circuit preparation/hash/compile/metrics and
must not be described as exact transpile-only time. Observer sampling is
not a certified instantaneous peak-RSS measurement.
