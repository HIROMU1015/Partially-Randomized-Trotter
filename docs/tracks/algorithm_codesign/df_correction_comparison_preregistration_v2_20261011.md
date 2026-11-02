# DF whole / local Pauli comparison contract v2

2026-10-11 JST. **SOURCE_PREPARATION / EXECUTION_NOT_AUTHORIZED**.
The [GPT review original](../../research/track_b_df_correction_comparison_preregistration_review_20261011.md)
is byte-preserved; input commit `7dbbc0b098231446aebd4c97302dd62a5ab87a70`.
The [v1 contract](df_correction_comparison_preregistration_20261011.md) remains history, at
`73b2102a305f018d22348418adc7939cae63f0b2`.
Normative machine specification: [contract_v2.json](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json).

## Fixed scope and inputs

CD is a conditional signal-accuracy envelope and symbolic additive resource comparison.
E is a separate local native-implementation screening. Neither measures full PR T cost,
nor excludes other whole-commutator implementations. No automatic scientific winner.
E unit prices/angles must not be transplanted to CD, even at the same mode count.

CD: three modes/blocks; coefficients 9/20, 7/20, 1/4; ranks/signs (1,+),(2,-),(1,-);
NumPy1.26.4 PCG64 seed6121107 with three ordered random_unitary(3) draws;
t=1; delta=1/8,1/16,1/32; steps=8,16,32; gamma=5/4; kappa=4; s=1; paired cutoff4/P5.
The six-basis initial state and X/Y family alpha allocations from v1 are adopted by GPT:
indices1,2,4,3,5,6 carry amplitudes1,i,1,1,-i,1 divided by sqrt6 (little-endian).
This is not a molecular ground state. A/B times three steps gives six tasks; each axis/task
has alpha1/240, each X/Y family1/40, total1/20. Axis epsilon1/20000 is conservative
for complex epsilon1/10000. Exact signal is diagnostic, never an oracle shot reduction.

E: seed7101107; consume the original random_unitary(4), then three rank2 r3 frames,
then three rank2 r6 frames, signs(+,-,+). theta1/32, kappa4, u1/128; system/control masks;
control is the MSB. Active frame identity only; embedding/preparation/readout unknown.
E has no registered PR outer rate. It reports accepted-correction conditional quantities.

## M1–M7 synchronization

| ID | v2 source/contract behavior |
|---|---|
| M1 | Separate CD conditional envelope/symbolic break-even, E local tables, and unknown PR total |
| M2 | Exact executed complete-law probabilities feed expected price, weight moment, range, and weighted native-error ledger |
| M3 | Exact ideal frame definition; assembly balls; exact rational computational Pauli proxy; certified proxy distance and zero status |
| M4 | Closed pi/atan/raw-rad adapter; strict W/phase guard; signed/control/period tests; coverage ledger |
| M5 | Multiplicity-weighted fixed-plan completion lower; B completes before A synthesis; incomplete A remains T_INCONCLUSIVE |
| M6 | Arbitrary-angle vs total RZ and label-specific 1Q; additive cost vs compiled wrapper; fixed common gate precision |
| M7 | External sampled resource containment, per-job exclusive starts, no retry, provenance, focused off-domain tests |

## Ideal frame and numerical approximation contract

Random draws and float QR recipe are unchanged. The ideal input is **ordered exact
Gram-Schmidt of the exact binary64 recipe columns**, with positive real square roots.
No pivot, truncation, regenerated frame, alternative seed, or numerical-rank drop.
At 100dps, directed interval Gram-Schmidt proves positive column norms and encloses
each column's distance. An uncertain positive-norm test stops technically.

Ideal CAR primitives are Hermitian involutions, number conserving, with norm1.
Their nested commutator has norm<=4. `certification.Ball` propagates frame perturbation
and standard complex BLAS forward error through the original float assembly operations.
The ideal-frame theorem, conditional IEEE/BLAS guards, and bare floating residuals are
distinct output fields. Native Operator verification additionally assumes faithful libm
within4ulp and local gate arithmetic. These are not physical DF error certificates.

**Result-prior computational-table amendment:** apply the exact rational Pauli transform
to the Hermitian projection of the exact dyadic float T, with proven trace/parity projection.
The reconstructed computational generator T_hat has a certified distance r_T to ideal T.
Its real rational beta table defines the implemented Pauli law/angles/weights.
For every ideal coefficient, |beta_ideal-beta_hat|<=r_T. Ideal trace-zero and odd-XY
parity zeros have algebraic proofs; all other exact computational zeros are recorded as
ideal-unresolved and **charged to r_T**, not declared ideal zero or dropped by threshold.
The same ideal target remains in the comparison; the B mean difference u*r_T goes into
the numerical ledger. This amendment does not claim exact proxy/ideal equality.

Interval or matrix-radius upper serialization uses a fixed outward256-bit dyadic grid,
without changing the100dps calculation. Matrices are immutable and Frobenius bounds cached.
Large exact product moments retain their rational factorization and outward interval;
no lossy float moment/cost replaces the saved executed law.

## Corrector and finite-bit law

A uses outer coin eta, tuple q=c/A_mass, exp(iuT), accepted weight1.
B uses the same coin and tuple; conditional Pauli probability |beta_hat|/ell_hat,
alpha=atan(u ell_hat), and accepted weight sqrt(1+(u ell_hat)^2).
Identity/zero branch is weight1 and zero correction cost; PR base cost is still paid.
Ideal B weighted mean is I+i delta^3 K3/2. A's finite mean is different.
The B proxy mean differs by the explicitly charged assembly perturbation.

Outer eta is exact dyadic. Tuple and conditional Pauli nodes use exact48-bit LRM,
lexicographic rational remainder ties, no threshold drop. Complete event probability is
the product of those node masses and the parent coin. Complete-law L1 includes the
parent-weighted conditional distances. RTE ideal probability intervals are mapped to a
normalized exact rational midpoint computational law before exact48-bit LRM; both
midpoint and LRM distances are charged. This avoids uncertain ideal remainder ordering.
No result-driven precision or tie retry.

Original RTE per-step normalization and B weights are fixed80-bit dyadic points.
Their ideal interval distances are separate numerical errors. Every executed cost is
sum p_exec*C_event; moment is sum p_exec*W_exec^2; range excludes zero-mass events.
Signed paired-order phases and actual binary64 angles are saved. Native signal bias is
2 sum p_exec*|W_exec|*sum_event_error, retaining correlation.
No replacement of expected cost by E[W^2*C].

## Finite-step and confidence budgets

Keep the v1 scalar formulas for b_sym, e_R, e_A, e_B and distinct A/B propagation.
In particular b_B=b_sym+2e_B+e_B^2+e_R(1+e_B)^2; B is not a contraction.
The global bound is (1+b_arm)^n-1 evaluated upward.
Axis maxima: algorithmic1/40000, digital1/200000, native1/200000, numeric1/1000000.
Only the certified algorithmic bound replaces its maximum in statistical-width allocation.
The other full reserved maxima remain subtracted. Failure means technical STOP, not a
change of epsilon, gamma, denominator, tolerance, fixture or synthesis policy.

Primary moment keeps the ideal CD ell cap32 and ell^2 cap1024. For the computational
generator, ell_hat<=32+8*r_T and implemented-weight rounding is returned explicitly.
This encloses executed weights; it does not silently inflate the ideal theorem or tighten
the primary with observed signals. Detailed executed moments are separate diagnostics.
Original Gamma=B5(delta/4)^n is retained and Gamma_exec is separately saved.
The Bernstein sufficient-shot formula uses upward executed moment/range enclosures,
fixed alpha, and the reserved statistical width. No quantum shots are generated.

CD native/preparation/readout implementation is absent; accuracy remains conditional
on their global error allowance (or ideal preparation). Stored floating signal points are
diagnostics, not a certified exact signal or operational full PR implementation.
CD costs h_x, C_base_x, C_event_x stay symbolic:

`G_i,x = 2 N_i (h_x + C_base_x + 2n sum p_exec,event C_i,event,x)`.

A break-even is the corresponding inequality G_A,x<G_B,x, not a numerical winner.
Depth is not a proven optimal additive depth.

## Native and T scope

Native all-to-all Qiskit1.3.0, rz/sx/x/cx, optimization0, seed20261011 is unchanged.
A has4 fixed generic compilations, and must match previous E fingerprints.
B uses label-specific fixed native parity IR for every label; at most18 representative
compiles (first nonzero label per support weight and control mask), total<=22.
Basis-change total RZ/1Q are included; arbitrary-angle RZ1/2 are separate. A generic
reference is not a lower bound for arbitrary whole compilers.

T selection remains A48 fixed positions, B8 angles, phase reserve8, <=64 slots;
no refilling dedup/Clifford free slots. Whole inventory freezes before any synthesis.
All B keys complete, then required phases, then registered A positions;2 workers per batch.
pygridsynth2.0.0/mpmath1.3.0; explicit backend and mp/iv dps100; up_to_phase=False;
gate epsilon1/40000000000; request epsilon/4; fixed seed SHA256 policy, retry0.
The explicit backend dps prevents its automatic epsilon-derived precision choice.
Strict4pi RZ period, signs, half angles, W and controlled relative phase are retained.
RZ(pi/4) is not a phase-free one-T fast path.

Full local accuracy1e-6 requires all keys and phase covered, RZ<=32000, native numeric
<=2e-7, and strict synthesis sum<=8e-7. Missing coverage is T_INCONCLUSIVE.
Uncovered arbitrary global phase is unsupported, rather than implemented by an invented
ancilla/gate. Clifford global phases are exact symbolic W with their rounding distance.
No sample-average T extrapolation. A fixed additive completion lower sums the **occurrence
multiplicity** of valid obtained sequences plus exact known Clifford T. If greater than
the certified complete B upper, a fixed-plan local separation can be recorded while
A_TOTAL_T_UNKNOWN remains true. It is not an optimal-synthesis/general whole lower bound,
nor a CD/PR separation. The saved sequences are not shortest sequences.

## Authorization and stopping

Preparation runs0; science_execution_authorized=false; no execution marker.
No registered CD six tasks, E compilations, or T synthesis during source preparation.
Focused tests use independent small controls only; no molecule, GPU, LP, trajectory,
G10/RA-D0 restart, parameter search or fixture replacement.

After source S is published, source/contract consistency review and separate explicit
one-shot approval are required. Execution HEAD must be S's direct authorization-only
child A on a dedicated execution branch; remote A/clean/source/runtime identity required.
Parent marker and each child-start receipt are exclusive, with retries0. All terminal
results STOP. Technical/partial results have no prefix positive/negative classification.
Research/new-method/adoption decisions return to GPT.
