# Finite RTE truncation budgets

This module accounts only for finite Taylor truncation. It does not combine
that operator error with product-formula error, signal attenuation, shot
growth, coefficient threshold error, or classical event-cost sampling error.

## Three levels of bound

For dimensionless short-step time $\tau$ and even cutoff $K$, the directly
summed scalar remainder is

$$
\epsilon_{\mathrm{step}}=R_K(\tau)
\le \sum_{j=K+2}^{\infty}\frac{|\tau|^j}{j!}.
$$

Repeating the same approximate short step $r$ times gives the safe bound

$$
\epsilon_{\mathrm{occ}}
\le (1+\epsilon_{\mathrm{step}})^r-1.
$$

If occurrence kind $i$ uses $r_i$ short steps and appears $c_i$ times in one
RPE round,

$$
\epsilon_{\mathrm{round}}
\le \prod_i(1+\epsilon_{\mathrm{step},i})^{r_ic_i}-1.
$$

`occurrence_truncation_residual_bound` and
`compose_truncation_residual_bounds` evaluate these expressions in log space.
Exact deterministic unitaries between occurrences have norm one and require no
additional factor.

`step_taylor_truncation_residual_bound` is the explicit one-step function.
The older `taylor_truncation_residual_bound` function and
`truncation_residual_bound` constructor fields remain available for
compatibility; `step_truncation_residual_bound` is their explicit one-step
property name.

## Types

- `RTEOccurrenceParameters`: dense-free time, L1, step, and round-count input.
- `RTEOccurrenceTruncation`: selected cutoff plus step, occurrence, and round
  contribution bounds.
- `RPERoundTruncationBudget`: target and allocation policy.
- `RPETruncationSummary`: heterogeneous composed bound and budget result.

When an `RPERound` is supplied, the total occurrence count and total short-step
count must match its `tail_evolutions` and `rte_total_steps`. Occurrences are
not assumed to share time, L1, step count, or cutoff.

## Baseline cutoff allocation

`select_rpe_round_taylor_orders` uses
`equal_log_budget_per_short_step`. For target $\epsilon_\star$ and total short
step count $S$, it assigns

$$
\ell_{\mathrm{step}}=\frac{\log(1+\epsilon_\star)}{S},
\qquad
\epsilon_{\mathrm{step,alloc}}=e^{\ell_{\mathrm{step}}}-1.
$$

For each actual $\tau_i=\lambda_i t_i/r_i$, it directly computes finite
remainders and chooses the smallest non-negative even $K_i$ meeting the local
allocation. It then recomputes the full heterogeneous round bound. This is a
transparent baseline, not a Taylor-order circuit-cost optimization.

## Error separation

- DF coefficient threshold: Hamiltonian approximation, recorded by
  `DFTailExtractionMetadata`.
- Finite Taylor cutoff: operator approximation, recorded here.
- Deterministic product formula: not newly implemented in this milestone.
- RTE normalization: signal attenuation, not bias.
- Attenuation-driven shots: future sampling-cost layer.
- Classical event sampling: expected-cost estimation uncertainty, not an
  operator error.

The event circuit builder, transpilation, RPE round circuits, and total cost
remain unimplemented.
