# RTE reallocation R1: one conditional implementation proposal for GPT review

Date: 2026-10-06 JST. `DRAFT_FOR_REVIEW`, `RUN_READY=false`,
`science_execution_authorized=false`, `implementation_authorized=false`.
No target operator, circuit or synthesis key described below was constructed or executed.
This is one bounded possible follow-up to [R0](rte_reallocation_r0_review_packet_20261006.md),
not a decision that it is needed. GPT first decides novelty, value and scope.

## Question and synthetic domain

Compare ordinary pairing and A's analytic endpoint at the **same degree-three
finite mean**, keeping canonical sampling and known cost-aware IS equally available.
Does reduced weight second moment survive phase-preserving native implementation
and its extra odd-event action cost? No eta search, B compression optimization,
C allocation optimization, molecule, split or PF search.

Proposed fixed involution-access contexts, each with probabilities (3/4,1/4):

| Context | Two involutions | Purpose |
|---|---|---|
| Common Pauli, commuting | Z0, Z1 | Full Pauli word simplification common to both arms |
| Common Pauli, noncommuting | Z0, X0 X1 | Noncommutation and phase-sensitive Pauli word control |
| Distinct basis | Z0, V† Z1 V; V=exp(-i pi X0 X1/16) | Cost-bearing basis conversion, not a molecular DF input |

For each, propose x in {1/8,1/4}, both time signs, eta in {0,1}, K=2.
There are 24 finite-mean representation settings. Ordinary and controlled contexts
are separate records, never pooled as equivalent tasks. These x values are synthetic
small-step controls: the elementary exponential remainder bound is exp(x)x^4/24.
They are not claimed to come from an acquired physical Hamiltonian or an H4 optimum.
If GPT requires a physically derived x policy, this proposed domain must be revised
**before** any source freeze or observation, not expanded after seeing an outcome.

## Fixed lowering proposal and call bound

Use the common-angle complement rewrite for odd events, retaining its extra Q
and (-i sigma)^(k+1) phase. Shared Pauli reduction and algebraically exact same-basis
run cancellation apply to every arm. Conjugators of controlled involutions are
implemented as V† Ctrl(Z1) V, with their cost/error retained. This is a proposed
construction, not an existing verified controlled-DF adapter.

For each x, the only non-Clifford event rotation magnitudes in exp(-i phi Q) are
atan(x), atan(x/3), atan(rho). Native RZ angles are ±2phi in ordinary context and
±phi in the controlled-pair lowering. Two x values therefore give at most 24 event
synthesis angle keys. Include ±pi/8 for the basis conjugators: 26 angle keys.
At native operator precision {1e-3,1e-4,1e-6}, the proposed ceiling is **78 unique
synthesis calls** to the same pinned pygridsynth tool as SP-0.5, not a new catalogue.
Source freeze must enumerate/deduplicate those keys mechanically and confirm this
bound. Literal complementary-angle synthesis is excluded from this one proposal.

All exact Clifford paths are shared. Actual synthesizer sequences, T/T† counts,
sequence hashes, phase convention and interval error guards must be saved. No
whole-wrapper compiler or molecule-scale resource claim is proposed.

## Accounting and numerical safeguards to fix before execution

Enumerate the small degree-three word support; do not sample trajectories.
For each setting/context, report expected additive native T count, weight second
moment, maximum corrected weight, worst-case coefficient-weighted implementation
bias, conjugator action count, workspace and classical acquisition cost. Unit-cost
or unknown basis cost cannot silently substitute for acquired native costs.

The same IS policy must be specified for both arms. Its positive-cost theorem is
not a complete rule for zero-cost events or range-sensitive sufficient shots.
Freeze the treatment of C=0 and probability floors before source freeze; do not
choose a policy to favor an arm. Fix a common one-block coherent-signal task, state
preparation accounting and Re/Im confidence allocation if a sufficient-shot metric
is to be primary. Point/exact signal may not lower shot counts.

Proposed guards: outward coefficient/angle/probability intervals, no division by
rho at x=0, zero-support omission, explicit signed/controlled phase, full error
propagation for native rotations and V/V†, normalization uncertainty returned to
bias and moments, and identical guard tolerances in all arms. Use one fixed interval
precision (proposed 100 decimal digits) without adaptive retries; widths and any
unresolved comparator are saved as inconclusive rather than counted as a win.

Primary proposal is a resource-vector comparison, with any trade-off shown as
such. A scalar materiality threshold, task accuracy, confidence, shot cap, treatment
of preparation cost, interval implementation and the required CTS/PTSC or other
structure-aware arm are **not adopted yet**. R0's math does not set these choices.
Do not inherit SP-1's 5% threshold or claim DF-native improvement from this toy domain.

## Budget and execution boundary

Proposed caps: one process, one run, retry 0, 20 minutes wall time, 512 MiB RSS,
16 MiB new output, at most 78 unique syntheses, no GPU, no trajectory, no molecule,
no wrapper pilot, no q/r/K or geometry expansion. The 24 settings and all planned
keys must be fixed before any synthesis; failures consume the proposed run.

Before this proposal could become executable, GPT must decide the comparison
claim and scope, then approve a complete preregistration, source-bound focused
semantic checks and separate one-shot authorization. Tool/wheel identity, input,
targets, keys, precision, guards, baseline, thresholds and caps all need a new freeze.
All outcomes must STOP and return to GPT. This document authorizes none of those steps.
