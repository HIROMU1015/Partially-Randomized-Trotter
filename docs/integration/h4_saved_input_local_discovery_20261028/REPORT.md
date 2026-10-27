# H4 saved input: LOCAL discovery follow-up (2026-10-28)

Status: **AWAITING_INDEPENDENT_SERVER_REPORTS**.
The target NPZ and complete array/freeze manifest were not found in the added
LOCAL search scope. Scientific numerical calls: **0**. No files were transferred
or copied as scientific input. No SERVER_A/B connection or remote command occurred.

This follows [the completed P1/P2 validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2917843dd6d03c8a94029023a8dac5a1cf86b25d/docs/integration/p1_p2_validation_and_h4_equivalence_20261028/REPORT.md).
The prior 105/147 PASS technical results are reused; those tests were not rerun.
The user's latest supplement assigns server searches to independent Codex sessions
and prohibits LOCAL-side SSH/SFTP/scp/transfers, overriding the older cross-server
execution method. This record therefore describes LOCAL discovery only.

| Environment | Current evidence | Status |
|---|---|---|
| LOCAL | No target payload/full freeze found within the added targeted scope | NPZ_NOT_FOUND_IN_LOCAL_SEARCH_SCOPE; MANIFEST_INCOMPLETE |
| SERVER_A | Independent report not provided to LOCAL in this task | NOT_ASSESSED_AWAITING_USER_SUPPLIED_REPORT |
| SERVER_B non-B3 | Independent report not provided to LOCAL in this task | NOT_ASSESSED_AWAITING_USER_SUPPLIED_REPORT |

No claim of absence on a server or outside the searched LOCAL scope is made.
LOCAL is not kept running to wait for replies.

## New useful findings

The fixed H4 [historical resume02 plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/plan_authorized_v1.json) contains the exact target
`input-0.90.npz` digest, input/state identities and a historical input-root
locator. That locator is non-LOCAL historical metadata; its actual current host,
existence and A/B assignment were not inferred or probed. Exact internal paths
are retained only in the private handoff leads.

The fixed input reader expects **generation-freeze.json**, with file SHA-256
`75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69`.
It requires per-distance input records plus `identities` carrying full array
identity metadata. The expected filename/hash/schema is not the actual manifest.

The reader checks generation SOURCE `049e69919af16ad29a67a217dc7a407d6b1754a6`.
The plan's dataset/seed source `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`, run05 source
`6d365257770e99022b91d6a38dbee49ee0077503`, later execution source
`d0726e946f89f8c30a08d383fb8ff48b363629ec` and donor `821cc0f6ced57168293ea166644024867b064ffb` remain
separate roles. No source version was silently substituted or combined into a
new scientific SOURCE. The metadata chain is more explicit, but cannot qualify
the input without the actual freeze and payload.

The existing public H4 result-bundle description expressly omits NPZ/raw runtime.
It remains a saved lightweight projection, not a private-input backup. No saved
results were reaggregated, ranked, interpreted anew or adopted.

See [discovery scope](INPUT_DISCOVERY.md), [identity status](INPUT_IDENTITY_VALIDATION.md),
[ten comparison items](SCIENTIFIC_VALIDATION_RESULTS.csv) and
[preservation/next steps](PRESERVATION_AND_NEXT_STEPS.md).

**Mandatory STOP after LOCAL report publication/readback.**
