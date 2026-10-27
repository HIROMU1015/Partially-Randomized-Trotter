# INPUT_IDENTITY_VALIDATION

Scope: one target only — linear H4, 0.90 Angstrom, STO-3G, DF rank12.
`input-0.90.npz` expected SHA-256:
`06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d`.
The public target identities remain comparison metadata; no scientific payload
was loaded, authenticated by name alone, regenerated or substituted.

| Evidence | Observation | Qualification |
|---|---|---|
| Target NPZ | No payload identified in added LOCAL scope | NPZ_NOT_FOUND; actual SHA NOT_VERIFIED |
| input/state IDs | Fixed historical plan matches the requested public IDs exactly | METADATA_MATCH_ONLY |
| generation-freeze.json | Expected name/hash and reader/schema identified | MANIFEST_INCOMPLETE; file not obtained |
| Full array keys/shapes/dtypes/order | Expected to be bound in the actual freeze identities | NOT_VERIFIED_ON_PAYLOAD |
| SOURCE chain | Generation, dataset/seed, run05, execution and donor roles separated | SOURCE_CHAIN_UNRESOLVED_ON_ACTUAL_FREEZE |
| Scalar/one correction, fragments, sector/state/bit order/norm | Prior source descriptions preserved | NOT_VERIFIED_ON_PAYLOAD |
| L_D/partition/r/K/time/law/cohort | Historical registered families do not select this new validation case | STATE_OR_PARTITION_UNQUALIFIED |
| Independent saved comparison and tolerance/call scope | Not fixed for an actual qualified input | REFERENCE_UNAVAILABLE; SCIENCE_SCOPE_NOT_FIXED |

The fixed reader's expected freeze SHA is
`75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69`.
It checks fingerprint `0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7`
for the historical freeze, original `INPUTS_FROZEN_STOP`, mandatory STOP, matching
input records and generation SOURCE `049e69919af16ad29a67a217dc7a407d6b1754a6`.
These are original lineage conditions, not new approval-proof machinery.
No old gate was invoked and no grant/receipt was issued or consumed.

Fixed references:

- [Plan and target input record](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/plan_authorized_v1.json).
- [Frozen-receipt reader at dataset/seed SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5aa6c3685afb9c3afcfbdc36f80244da0a7fe525/src/trottertracks/resource_applicability/h4_geometry/launch_binding.py).
- [Generation storage/schema at that fixed SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5aa6c3685afb9c3afcfbdc36f80244da0a7fe525/src/trottertracks/resource_applicability/h4_geometry/execution.py).
- [Input array-identity encoding](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5aa6c3685afb9c3afcfbdc36f80244da0a7fe525/src/trottertracks/resource_applicability/h4_geometry/inputs.py).

Static Git text shows the input module at `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525` matches the donor's
input module. This is code identity, not authenticity or numerical equivalence
of a saved scientific input. Source-role differences are retained rather than
treated as an error or silently unified.

NPZ numeric load, per-array digest verification, numerical state/partition checks,
science comparison and new tolerances were not performed. Scientific calls: **0**.
All ten scientific items remain SOURCE_OR_INPUT_UNQUALIFIED, with pending state
AWAITING_INDEPENDENT_SERVER_REPORTS. No numeric MATCH/MISMATCH is asserted.
