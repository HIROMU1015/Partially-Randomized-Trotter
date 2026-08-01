# Finite RTE conventions

## Scope and primary source

The primary source is Günther et al., *Phase estimation with partially
randomized time evolution*, arXiv:2503.05647v2 (2026-07-10). The RTE formulas
used here are Appendix A, Eqs. (A18)–(A31); partial randomization uses
Eqs. (A32)–(A40). The repository PDF is v1 and is legacy reference material,
not the implementation source. Exact source-file metadata is recorded in
`docs/rte_source_versions.md`.

This is the Randomized Taylor Expansion (RTE), not qDRIFT. One event contains
one involution rotation and an even-length product of additional involutions.
Every randomized component is validated as a Hermitian involution $P_l^2=I$.
A complete DF fragment is generally not one such component.

## Tail decomposition and DF provenance

The randomized tail is normalized as

$$
H_R=\lambda_R\sum_l p_lP_l,\qquad
\lambda_R=\sum_l |h_l|,\qquad p_l=|h_l|/\lambda_R.
$$

The sign of $h_l$ is absorbed into $P_l$. Each component and event application
still records the original sign, absolute coefficient, identity flag, DF
fragment ID, basis ID and hash, diagonal Pauli support, basis-change
operations, and circuit application index. This is needed when a future
circuit builder uses an unsigned Z/ZZ support:

- a product occurrence of $-P$ contributes a scalar phase $-1$;
- a rotation $\exp[-i\phi(-P)]$ uses the unsigned support with angle $-\phi$.

These are distinct operations and are exposed as `product_sign_phase` and
`unsigned_rotation_angle`.

Identity components may remain in the faithful RTE tail. On an uncontrolled
system they are global phase; under control they are relative ancilla phase.
The DF extractor also supports moving their exact sum to a deterministic phase
baseline.

The DF ranking value `abs(lambda_l) * ||G_l||_F^2` is only a selection proxy.
The RTE $\lambda_R$ is recomputed from the exact signed I/Z/ZZ expansion and is
stored separately.

`SymbolicRTETail` is the normal DF event-generation input. It contains the
normalized `RTEComponent` tuple but no dense operator tuple. The guarded
`NormalizedRTETail` remains a small-system oracle. Both satisfy the lightweight
interface consumed by `make_rte_config`; event generation needs only tail ID,
tail hash, $\lambda_R$, and normalized components.

## Exact threshold policy and tail hash

`normalize_involutory_tail(..., atol=0.0)` does not discard any nonzero
coefficient. A nonzero threshold is explicit and records:

- coefficient threshold and normalization policy;
- input, retained, and dropped component counts;
- dropped coefficient L1 and the corresponding operator-norm error bound.

The tail hash covers the canonical input before thresholding plus the policy.
Two inputs that retain the same terms but drop different terms therefore have
different hashes.

## Dimensionless time and finite distribution

For physical tail time $t_R$ and integer RTE step count $r$,

$$
\Delta t=t_R/r,\qquad \tau=\lambda_R\Delta t.
$$

`rte_steps` counts independently sampled events, not Taylor order. Distribution
weights use $|\tau|$, while the rotation angle retains the sign of $\tau$.

All count-like arguments use an integer-only validator based on
`operator.index`. Python and NumPy integers are accepted and normalized to a
Python `int`. Booleans, floats such as `2.0` or `2.5`, strings, and negative
counts are rejected; zero is accepted only by APIs where an empty count is
meaningful. Finite Taylor cutoffs additionally must be even. There is no silent
`int(...)` truncation in step, occurrence, round, sampling, or cutoff counts.

Appendix A pairs Taylor degrees $n$ and $n+1$ for even
$n=0,2,4,\ldots$. Define

$$
a_n(\tau)=\frac{|\tau|^n}{n!}
\sqrt{1+\frac{\tau^2}{(n+1)^2}}.
$$

For finite even cutoff $K$,

$$
B_K(\tau)=\sum_{\substack{0\le n\le K\\n\ \mathrm{even}}}a_n(\tau),
\qquad q_K(n)=a_n(\tau)/B_K(\tau).
$$

`exact_finite_distribution` is the directly summed $B_K$. The separately
named `paper_upper_bound` stores $B_\infty\le e^{\tau^2}$. They are not
interchangeable.

## Event coefficient, order, and angle sign

Given even order $n$, sample $(l,l_1,\ldots,l_n)$ independently from $p$.
The positive unnormalized event coefficient is

$$
c_m=a_n(\tau)p_l p_{l_1}\cdots p_{l_n},
$$

with probability $c_m/B_K$, phase $(-1)^{n/2}$, and unitary

$$
U_m=(-1)^{n/2}V_l(\phi_n)P_{l_n}\cdots P_{l_1},\qquad
V_l(\phi)=e^{-i\phi P_l}.
$$

Circuit application order is
$(P_{l_1},\ldots,P_{l_n},V_l(\phi_n))$. Baseline code preserves this order.

Direct algebra from v2 Eqs. (A18)–(A23) gives

$$
I-\frac{i\tau}{n+1}P_l
=\sqrt{1+\frac{\tau^2}{(n+1)^2}}
V_l\!\left(+\arctan\frac{\tau}{n+1}\right).
$$

The implementation therefore uses
$\phi_n=+\arctan(\tau/(n+1))$. This sign is confirmed by dense reconstruction
of the finite Taylor polynomial. The displayed negative definition of
$\phi_n$ in v2 is documented as a notation inconsistency only after checking
the v2 text itself, its definition $V_l(\phi)=e^{-i\phi P_l}$, the preceding
algebra, and the dense result.

## Exact enumeration versus Monte Carlo

`exact_enumerated_event_mean_operator` evaluates
$\sum_m q_mU_m$ and requires the supplied event probabilities to sum to one.
Duplicate records contribute their stated probability mass.

`sample_event_mean_operator` accepts already sampled events and computes the
unweighted mean $M^{-1}\sum_{j=1}^M U_j$. It never multiplies event
probabilities again. Its result includes sample count, entrywise standard
error, and a Frobenius standard-error summary. The old
`finite_event_mean_operator` is a deprecated exact-enumeration wrapper.

## Corrected operator and attenuation

The finite event mean and the normalization-corrected Taylor operator are
different quantities:

$$
B_K\,\mathbb E_K[U]
=\sum_{j=0}^{K+1}\frac{(-i\tau H_R/\lambda_R)^j}{j!}.
$$

`finite_rte_corrected_operator` returns the right-hand polynomial across all
integer steps. `finite_rte_operator_moments` returns both that operator and the
attenuated event mean, with the normalization product stored explicitly.

For multiple tail occurrences, `compose_finite_rte_occurrences` multiplies the
individual $B_{K_i}^{r_i}$ values. Occurrences may have different signed
times, step counts, cutoffs, and tails; no common $B$ is assumed.

Normalization attenuation is not bias. It is stored separately from every
Taylor residual and will later affect shot requirements rather than the
operator-error budget.

## Step, occurrence, and RPE-round Taylor truncation

The omitted one-short-step coefficient 1-norm is bounded by

$$
R_K(\tau)\le
\sum_{j=K+2}^{\infty}\frac{|\tau|^j}{j!}.
$$

This is `step_truncation_residual_bound`, not an occurrence or round bound. For
$r$ integer RTE steps,

$$
\epsilon_{\mathrm{occ}}
\le (1+\epsilon_{\mathrm{step}})^r-1.
$$

For heterogeneous occurrence kinds $i$, repeated $c_i$ times in one round,

$$
\epsilon_{\mathrm{round}}
\le
\prod_i(1+\epsilon_{\mathrm{step},i})^{r_ic_i}-1.
$$

The implementation evaluates this with `log1p` and `expm1`. Interleaved exact
deterministic unitaries add no norm amplification. `RTEOccurrenceTruncation`,
`RPERoundTruncationBudget`, and `RPETruncationSummary` keep the three levels
explicit and can validate `RPERound.tail_evolutions` and `rte_total_steps`.

`select_rpe_round_taylor_orders` is a baseline allocator. It divides
$\log(1+\epsilon_{\mathrm{round}})$ equally among all short RTE steps, then
chooses the smallest finite even cutoff whose directly evaluated scalar
residual meets that local allocation. It finally recomputes the heterogeneous
round product and verifies the requested budget. This is not yet a
circuit-cost-optimal allocation.

Taylor truncation remains separate from coefficient-threshold Hamiltonian
error, deterministic product-formula error, normalization attenuation,
attenuation-driven shot growth, and classical event-cost sampling error. See
`docs/rte_truncation_budget.md`.

## Output taxonomy

- `paper_upper_bound`: analytic inequality from the paper.
- `exact_finite_distribution`: directly computed finite normalization.
- `empirical_compiled_estimate`: transpiled event/circuit estimate with
  compiler and sampling metadata.
- `legacy_analytic_proxy`: pre-existing `G_rand`, `B(kappa)`, anchor-Cgs, or
  fragment-summed model; not a finite-RPE total.
