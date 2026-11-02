# RTE reallocation R0: claim-level prior-art audit v1

Date: 2026-10-06 JST. Scope: the [new user/GPT design](inputs/track_b_all_evidence_algorithm_redesign_20261006_user_input.md)
§§5–7 and §13. This is a focused primary-text audit, not a proof of priority.
No failure to find a matching expression is treated as evidence of novelty.
The [independent proof](rte_reallocation_r0_independent_proof_v1.md) establishes
mathematics under explicit hypotheses; this document addresses known overlap.

## Primary passages read and compared

| Primary source and locator | Known content | Same-convention comparison / remaining difference |
|---|---|---|
| [Wan–Berta–Campbell, 2110.12071v2](https://arxiv.org/pdf/2110.12071v2), Appendix C, C4–C5 and following coefficient-norm expression; Appendix F.2 truncation | Adjacent even/odd Taylor pairing; one Pauli rotation plus an even product; canonical unitary sampling and product normalization | Under time-sign conversion and degree cutoff, this gives A's ordinary endpoint. Their displayed rotation is on the right of the word, ours on the left; IID first means agree, individual noncommuting branches need not. Finite-tail handling must retain degrees through K+1. The read passages do not give A's full overlapping nonnegative family and attaining recurrence. |
| [Phase estimation with partially randomized time evolution, 2503.05647v2](https://arxiv.org/pdf/2503.05647v2), Appendix A.2, A18–A29 | Normalized Hamiltonian, paired RTE events, expectation attenuation and independent repeated steps | Direct baseline for the finite paired implementation. Adopt the independently derived signed rule exp(-i sigma phi Q), phi=atan[x/(n+1)], rather than infer signs from isolated extracted notation. The input plan's A3–A5 locator refers to earlier material, not the paired-event derivation. A broad claim to introduce finite paired RTE is excluded. |
| [Zhao–Yuan, 2103.07988v2](https://arxiv.org/pdf/2103.07988v2), §4.2, Eqs.24–29; §4.3, Eqs.34–40 | Return/cancellation terms from squared and higher Hamiltonian powers are absorbed into identity and lower-degree coefficients; modified Taylor LCU uses structure | B's identity-return mechanism and coefficient absorption are known in principle. The scoped degree-three formula and direct conditional law are correct specializations, not an established new principle. Further returns and anticommutation can improve on B, so B is not the strongest possible dictionary. |
| [Zeng et al., 2212.04566v2](https://arxiv.org/pdf/2212.04566v2), §IV.A, Eqs.42–50; §IV.B, Eqs.73–75 | Taylor compensation pairs identity with anti-Hermitian terms; identity is distributed between leading groups using a common rotation parameter | **Coefficient redistribution and common-angle pairing are already known.** Their displayed construction concerns Trotter remainder groups and Pauli reductions. A's exact all-odd finite nonnegative adjacent family and its explicit attainment need a narrower equivalence audit; broad reallocation/common-angle novelty is unavailable. |
| [Peetz–Smart–Narang, 2407.21095](https://arxiv.org/pdf/2407.21095), Theorem 1, Eq.6; Methods IV.B; Supplementary Notes 3 and 5, S13–S15 | Convex Taylor sampling separates real/imaginary Pauli coefficients, pairs the imaginary part with identity at a common angle, and also permits layered sampling without full expansion | Strong comparator missing from the original plan list. After Pauli collection, the displayed normalization is L_c+sqrt(1+L_s^2), not A's E/O class norm. Different target decomposition and available word cancellations must be compared, including classical cost. It is incorrect to exclude this method merely by assuming exhaustive word enumeration is mandatory. |

PDF passages above were read in context, including definitions and neighboring derivation.
No full paper was copied into the repository. Links identify primary sources.
The first four PDFs were accessed by explicit version. For SCU/CTS, the versionless
16-page PDF is dated July 8, 2026; the [arXiv metadata](https://arxiv.org/abs/2407.21095)
identifies v2 (submitted July 6, 2026). An explicit-v2 PDF fetch failed during this
audit, so the receipt distinguishes metadata identity from byte-level verification.
This audit has no archived PDF hash and makes no immutable-paper-byte claim.

The PR PDF's paired derivation is A.2, whereas the user plan lists A3–A5.
Only the review locator is corrected here; the byte-exact input is preserved.
Potential sign/index ambiguity in text extraction is not treated as a published
erratum. Algebra, current source's signed angle, and branch order are specified
explicitly in the proof and static receipt.

## Claim scope and exclusions

| Claim | Status | What can be retained | What cannot be claimed |
|---|---|---|---|
| Ordinary endpoint equals known adjacent paired Taylor mean | KNOWN_EQUIVALENT | A has a direct Wan/PR baseline | New paired RTE or a new first-moment estimator |
| General A finite mean and explicit class-bound attainment | PROVED mathematically; UNRESOLVED for priority | Nonnegative adjacent-degree construction and its proof | First-ever redistribution, unrestricted LCU optimum, superiority over Pauli-collected CTS/PTSC |
| Changing coefficients/angles/support rather than only probabilities | PROVED distinction within the defined arms | A differs from reweighting the fixed ordinary ensemble | Reweighting alone is the strongest baseline |
| Shared-angle redistribution as an upper-level idea | KNOWN_EQUIVALENT | Known building block | A broad new-method claim based only on that idea |
| B's returned identity compression | KNOWN_EQUIVALENT principle; PROVED formula | Exact finite specialization, direct conditional law | New Taylor-cancellation principle or validated finite-bit sampler |
| C contractivity and continuous KKT rule | PROVED; standard mathematical tools | Common bias/accounting tools for all arms | New precision-allocation principle, free second-moment reduction, reversal of FR STOP |
| A resource advantage after native/controlled implementation | UNRESOLVED | Explicit implementation hypothesis | Actual compiled gain, DF-native gain, molecular transfer, independent validation |

The earlier [BS-0.5 finding](bs05_method_target_design_audit_v1.md) remains:
its specific candidate had the same task, supplied dictionary and acquisition
procedure as a generic sparse/operator LCU baseline. That does not imply every
LCU-representable construction lacks value. A closed-form generator, a class
theorem, acquisition savings, or a concrete native implementation difference can
be meaningful; each needs its own known-method comparison.

## Fixed-ensemble sampling and previous STOPs

The [previous ordinary baseline](bs05_ordinary_finite_rte_baseline_v1.md) already
records [Resource-Optimal Importance Sampling, 2603.13495v1](https://arxiv.org/html/2603.13495v1),
§II Theorem 1: for its fixed-ensemble positive-cost setting, probabilities vary
with coefficient and cost. This R0 does not re-audit the entire paper or claim
an implementation of that optimizer. A changes the represented events themselves;
ordinary and A arms must both receive the same admissible IS, zero-cost, estimator
range and bias policy before a resource comparison. Known Pauli simplification,
return compression and run-level basis policy must also be shared where applicable.

- B-F remains the limited negative result at the registered family/task; its PF
  coefficient search is not resumed by changing the RTE representation.
- B-M remains equivalent to same-backend compact BCH for its audited adapter.
- P-A's effective run-level basis reuse stays a baseline; interval DP is not revived.
- P-D/R3's comparison-fairness problems are avoided only if both arms have the same
  sampling, precision and simplification access. No stopped q/r/K search is resumed.
- FR's stopped standalone optimization claim is unchanged by C's basic identities.
- SP-0.5/SP-1 findings and their consumed markers stay fixed. A is not an SP replay,
  a placement claim, or a license for a new dictionary/catalogue sweep.

## Questions returned to GPT

1. Does the **specific adjacent finite family / all-odd attaining formula / limited
   optimum** remain distinct after CTS and PTSC are mapped to the same target and
   involution-access model? The broad common-angle/reallocation idea is known.
2. Is that scoped mathematical result valuable as a theory/technical note even if
   the small-step normalization benefit is small? Codex does not choose the paper claim.
3. Is an implementation mechanism comparison worth authorizing, and which strong
   Pauli-collected/structure-aware comparator is necessary for its eventual claim?

The [single conditional R1 proposal](rte_reallocation_r1_proposal_for_review_v1.md)
is review material only. None of these questions has been answered by a new
science run. Research mode, novelty and further scope remain GPT-owned.
