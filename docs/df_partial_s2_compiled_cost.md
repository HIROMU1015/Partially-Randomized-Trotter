# DF partial-S2 one-step circuit and compiled cost

## Exact step definition

For deterministic blocks in their adopted ranked-prefix order
`D_0, ..., D_(L-1)`, one second-order partial step of duration `delta` is:

```text
D_0(delta/2), D_1(delta/2), ..., D_(L-1)(delta/2)
one finite-RTE occurrence for tail time delta
D_(L-1)(delta/2), ..., D_1(delta/2), D_0(delta/2)
```

The implementation appends these half sweeps explicitly. It does not build a
complete internal S2 circuit for all of `H_D` and place that complete circuit
on both sides of the tail. Only second order (`pf_label="2nd"`) is supported;
other product-formula labels are rejected.

## Partition and preparation

`L_D` is a strict Python/NumPy integer counting two-body DF fragments. The
ranked prefix is deterministic and the ranked suffix is randomized. Neither
side is silently reordered by original fragment index. Constant and one-body
corrections always remain deterministic and are never part of the randomized
tail.

`prepare_df_partial_s2` checks complete, disjoint fragment coverage and stores
Hamiltonian, partition, and preparation hashes; original fragment indices;
the deterministic block order; basis IDs/hashes and runtime operations;
identity/threshold policies; and tail extraction metadata. Both deterministic
and randomized blocks use the recorded `diagonal_sort` and
`fermionic_gaussian_jw` basis-construction policy.

Two quantities deliberately remain distinct:

- `ranking_proxy_lambda_r` is the suffix sum of fragment-ranking weights and
  is the explicit name for legacy `DFFragmentPartition.lambda_r`.
- `exact_rte_lambda_r` is the coefficient one-norm after symbolic I/Z/ZZ
  expansion, thresholding, and identity policy. Only this value enters RTE
  config, event probabilities, Taylor selection, and attenuation.

An empty randomized suffix (`L_D` equal to the fragment count) is a valid
deterministic-only preparation and requires no RTE config or occurrence.

## Deterministic blocks and phases

One-body and squared-fragment central evolutions are converted to the same
shared primitive representation: global phase, RZ operations, and RZZ
operations. Existing uncontrolled `apply_D_one_body` and `apply_D_squared`
also consume that representation, preventing angle drift between paths.

Each deterministic block uses:

```text
inverse basis-operation sequence
-> central diagonal primitives
-> forward basis-operation sequence
```

In a controlled step, basis operations remain uncontrolled. Only each central
RZ/RZZ and its diagonal global phase are controlled. Together with controlled
tail events this produces the complete `diag(I_system, U_partial-S2)` matrix,
including ancilla-relative phase.

Three phase sources are separate in metadata and implementation:

1. full-Hamiltonian constant correction, applied once for `delta`;
2. identity coefficient extracted from the randomized tail, applied once for
   `delta` only under `extract_identity_phase`;
3. Taylor, sign, rotation, and faithful-identity phases inside the reused RTE
   event builder.

With `faithful_identity_in_tail`, no additional extracted-identity phase is
added. Threshold counts, dropped coefficient L1, and operator-error bound are
retained even though the constructed circuit and compiled count are exact for
the thresholded representation. RTE normalization and attenuation are stored
as metadata, not gates.

## Compiled expectations

`estimate_exact_compiled_partial_s2_cost` precomputes the number of event
sequences. For `r` RTE events, each sequence has probability equal to the
product of its event probabilities. It verifies the total mass and computes:

```text
sum(sequence_probability * compiled_full_step_metric)
```

`maximum_event_sequences` prevents exponential enumeration, and
`maximum_untranspiled_circuit_size` bounds each constructed step.

`estimate_monte_carlo_compiled_partial_s2_cost` instead samples `r` events per
step directly from the finite distribution and reports the unweighted mean,
unbiased sample variance, standard error, range, unique full-step count, and
cache hits. Sampled event probability is never multiplied a second time. This
is classical circuit selection, not quantum-shot sampling.

For every exact or sampled sequence, the APIs compile matching costs for:

```text
forward deterministic half (including the once-per-step scalar phases)
+ RTE occurrence
+ reverse deterministic half
= additive cost

full step compiled once
- additive cost
= partial-S2 nonadditive difference
```

The difference can contain deterministic/random-boundary cancellation, phase
consolidation, compiler optimization, and routing effects. It is distinct from
the Level-4 nonadditivity measured only inside a short RTE occurrence.

The compiler-independent full-step fingerprint includes Hamiltonian and
partition hashes, `L_D`, deterministic order, step time, tail hash, exact RTE
lambda, finite Taylor order, ordered event-sequence fingerprint, identity and
threshold policy, control/ancilla condition, and basis-reuse policy. The
transpile cache additionally includes compiler settings and Qiskit version.

## Fidelity and exclusions

A transpiled full-step expectation is Level 5. It includes one deterministic
forward half, one finite-RTE occurrence, and one reverse half. The unchanged
one-step result is also the `repetition_count=1` limit of the short repeated
Level-5-R API documented in `df_partial_s2_repeated_compiled_cost.md`. Neither
API includes long `2^m` RPE repetition, Hadamard-test X/Y controls,
attenuation-driven shot counts, state preparation, quantum shots, noise,
backend jobs, or an RPE total.
