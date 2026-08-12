# Finite-RTE RPE resource accounting

`trotterlib.rpe_resource_accounting` is the first resource-accounting layer
above the existing finite-RTE and Level-5-R implementations. It is restricted
to directly constructible, second-order DF partial-S2 rounds with fixed DF
data, fixed `L_D`, fixed `delta_time`, one randomized-tail occurrence per
partial-S2 step, and the ordinary controlled `diag(I,U)` circuit model.

It does **not** build a full RPE circuit. In particular, the direct Level-5-R
provider excludes state preparation, Hadamard-test H/X/Y gates, measurements,
noise, and backend execution. Its scope is recorded as
`compiled_time_evolution_subcircuit`.

## Round calculation

`RPERoundSpecification(m, delta_time)` fixes

```text
q_m = 2**m
t_m = q_m * delta_time
```

For a randomized DF tail, `evaluate_rpe_round_candidate` obtains
`lambda_R` only from `DFPartialS2Preparation.exact_rte_lambda_r`. It never uses
`ranking_proxy_lambda_r` as an RTE norm. For candidate `(r_m, K_m)`, it reuses:

- `finite_rte_distribution(lambda_R*delta_time/r_m, K_m)` for the one-step
  residual and finite normalization `B_K`;
- `compose_truncation_residual_bounds` for
  `(1+epsilon_step)**(q_m*r_m)-1` in log space;
- `make_rte_config` and `finite_rte_attenuation` for
  `A_att=B_K**(-q_m*r_m)`.

The unit-radius reference lower bound is

```text
rho_obs_lb = A_att * (1 - epsilon_Z).
```

An empty randomized tail has no RTE config or distribution and uses the exact
conventions `epsilon_Z=0`, `B_K=1`, `A_att=1`, and `rho_obs_lb=1`.
Its unique canonical candidate is `r_m=0, K_m=0`; nonzero values are rejected
rather than silently ignored.

## Phase budgets and shots

`RPEPFErrorModel` supplies, rather than derives, the coefficient in

```text
epsilon_PF = coefficient * delta_time**2
beta_PF = t_m * epsilon_PF
beta_RTE = asin(epsilon_Z).
```

The existing DF phase-bias fit is a state-specific screening surrogate (its
fit state is the selected deterministic Hamiltonian's ground state), not a
rigorous full-target RPE input or the paper's strict `C_gs` bound. It must be
passed with `is_rigorous_bound=False`.

`RPEErrorAllocation` stores separate PF, RTE, and statistical phase budgets
and separate cosine/sine failure probabilities. A candidate is infeasible if
its finite-RTE error is at least one, its radius lower bound or residual phase
budget is nonpositive, its allocated phase budgets exceed `beta_RPE`, or an
actual PF/RTE phase bound exceeds its allocation. Infeasible candidates are
not sent to a compiled-cost provider.

With a common coordinate tolerance,

```text
epsilon_coord = rho_obs_lb * sin(beta_stat_budget) / sqrt(2)
N_m,b = ceil(2 / epsilon_coord**2 * log(2 / alpha_m,b)).
```

`build_rpe_resource_summary` checks the explicit all-round union bound
`sum_m(alpha_m,c+alpha_m,s) <= alpha_total`. A summary must contain exactly
one candidate for every round `m=0,...,M`. It also rejects mixed DF
preparations, `L_D`, `delta_time`, `beta_RPE`, PF models, accounting versions,
RTE seeds, compiled-cost scopes, provider versions, compiler settings, or explicit
compiled-cost model fingerprints. The direct provider's fingerprint covers
its exact/Monte Carlo mode, seed and sample count, boundary policy, ordinary
control convention, compiler settings, and canonical backend target when one
is available. An uncanonical backend context is recorded explicitly, leaves
the fingerprint unverified, and therefore cannot be combined into a
multi-round summary. Generic providers must likewise supply a stable model
fingerprint before their results can be combined across rounds.

## Compiled cost

`DFLevel5RCompiledCostProvider` adapts the existing exact or classical Monte
Carlo repeated-partial-S2 evaluator. It directly builds only short rounds,
retains the existing trajectory and workload guards, and by default rejects
`q_m>4` through an explicit configurable `maximum_repetition_count` guard. It
exposes all six `CircuitCost` metrics:

```text
rz_count, cx_count, rz_depth, cx_depth, total_depth, circuit_size
```

For selected metric `g`, accounting uses

```text
G_m = N_m,c * E[g_c] + N_m,s * E[g_s]
G_total = sum_m G_m.
```

The direct provider returns the same evolution-subcircuit expectation for the
two measurement axes because the excluded X/Y measurement circuitry is the
only axis-dependent circuit part. Its Monte Carlo `sample_count` (`S_MC`) is
stored as classical estimator metadata and is never multiplied into `G_m` or
the quantum shot counts. Per-axis Monte Carlo standard errors are retained,
but this layer does not report a total-cost standard error: the direct
provider reuses the same evolution estimate for both axes, so treating those
two fields as independent would be incorrect.

## Guarantee status

An empirical state-specific phase-bias coefficient always produces
`empirical_screening`, even when all numerical constraints pass. `certified`
is available only when the caller explicitly marks an externally supplied PF
coefficient as a rigorous bound, the candidate is feasible, the recorded DF
threshold operator-error bound is exactly zero, and the all-round union bound
passes. A rigorous input whose numerical or union-bound conditions fail, or
whose nonzero DF threshold error has not been allocated a phase budget, is
`not_certified`; it is not relabeled as empirical. These statuses are
conditional on the recorded exact-ground-state, unit-radius, and alias-free
phase-window assumptions. They are not claims that a full RPE circuit or an
end-to-end noisy execution has been certified.
The PF input's independent `provenance_status` records
`rigorous_upper_bound` versus `empirical_surrogate`; it is kept separate from
whether a candidate or all-round budget actually earns `certified`.

`RPERoundSpecification` has a numerical safety limit of `m<=62`. The direct
Level-5-R provider applies the much tighter default construction guard
`q_m<=4`; raising that guard requires an explicit direct-construction workload
decision. Long rounds otherwise require a separately validated provider or
proxy and are not extrapolated by this implementation.

The simplified analytic PR cost prefactors are not used by this layer. Their
explicit, versioned definitions are documented in `docs/rte_source_versions.md`.
