# Finite RTE conventions

## Scope and primary source

The primary source is the repository copy of *Phase estimation with partially
randomized time evolution*, especially Sec. IV, Eq. (10)--(11), Sec. V,
Eq. (13)--(14), and Appendix A.2--A.3, Eq. (A3)--(A8).

This implementation is the Randomized Taylor Expansion (RTE) of Appendix A.2.
It is not qDRIFT and never replaces an event by “sample one Hamiltonian term
and exponentiate it.”  One RTE event contains one Pauli/involution rotation and
an even-length product of additional involutions.

The proof in the paper assumes that every randomized component is a Hermitian
involution, (P_l^2=I).  `trotterlib.rte` enforces that assumption.  Sec. VII B
gives the DF conversion to use: expand a diagonal DF fragment into diagonal
Paulis (P_m), then use

\[
\widetilde P_m=(U^{(l)})^\dagger P_mU^{(l)}.
\]

Each (\widetilde P_m) is still an involution and carries its source
`df_fragment_id` and shared `basis_id`.  The whole generic DF fragment
(U_l^\dagger D_lU_l), whose diagonal has more than two eigenvalues, is not one
RTE component.  Treating it as if it were (P_l) would not be the cited RTE
algorithm.

## Tail decomposition

The randomized tail is normalized as

\[
H_R=\lambda_R\sum_l p_lP_l,\qquad
\lambda_R=\sum_l |h_l|,\qquad p_l=|h_l|/\lambda_R.
\]

The sign of (h_l) is absorbed into (P_l).  Identity terms are allowed and
are kept in the same decomposition.  Their phase is a global phase on an
uncontrolled system circuit, but a relative ancilla phase in a controlled
Hadamard-test circuit.

The current DF ranking proxy
`abs(lambda_l) * ||G_l||_F^2` is not automatically this RTE
(\lambda_R).  The next stage must expand each selected diagonal DF tail
fragment, combine its identity/Z/ZZ coefficients with a fixed convention, and
hash the resulting (\widetilde P_m) LCU before computing the RTE norm.

## Dimensionless step time and integer steps

For physical tail evolution time (t_R) and integer RTE step count (r),

\[
\Delta t=t_R/r,\qquad \tau=\lambda_R\Delta t.
\]

`rte_steps` is (r).  It counts repetitions of independently sampled RTE
events.  It is not a Taylor order.  Negative evolution time is represented by
signed (\tau); distribution weights depend on (|\tau|), while rotation
angles retain its sign.

## Taylor-order distribution

Appendix A.2 pairs ordinary Taylor degrees (n) and (n+1) for even
(n=0,2,4,\ldots).  Define

\[
a_n(\tau)=\frac{|\tau|^n}{n!}
\sqrt{1+\frac{\tau^2}{(n+1)^2}}.
\]

For a finite, even cutoff (K), the implemented distribution is

\[
B_K(\tau)=\sum_{\substack{0\le n\le K\\n\ \mathrm{even}}}a_n(\tau),
\qquad q_K(n)=a_n(\tau)/B_K(\tau).
\]

`exact_finite_distribution` is the directly summed (B_K).  The separate
`paper_upper_bound` is the one-step specialization of Eq. (A5),
(B_\infty(\tau)\le e^{\tau^2}).  These fields must not be interchanged.

The PDF's Eq. (A7) prints
(B_{i_p}\le\exp(-\delta_{i_p}^2\lambda_R^2/r_{i_p})).  The minus sign cannot
hold because (B_{i_p}) is a sum of positive LCU coefficients and is at least
one.  Eq. (A5), Lemma A.2, and Eq. (A8) all use the positive exponent.  This
implementation uses the directly summed finite (B_K\ge1) and retains the
positive-exponent expression only as the paper upper bound.

## Conditional event distribution, coefficient, and phase

Given order (n), sample (l,l_1,\ldots,l_n) independently from (p).  The
event has unnormalized positive coefficient

\[
c_m=a_n(\tau)p_l p_{l_1}\cdots p_{l_n},
\]

finite event probability (c_m/B_K), phase
((-1)^{n/2}), and unitary

\[
U_m=(-1)^{n/2}V_l(\phi_n)P_{l_n}\cdots P_{l_1},\qquad
V_l(\phi)=e^{-i\phi P_l}.
\]

Circuit application order is therefore
(P_{l_1},\ldots,P_{l_n},V_l(\phi_n)).  `selected_component_ids` and basis reuse
intervals use this circuit order.  Baseline code must not reorder it.

Pairing the two Taylor degrees gives

\[
I-\frac{i\tau}{n+1}P_l
=\sqrt{1+\frac{\tau^2}{(n+1)^2}}
V_l\!\left(+\arctan\frac{\tau}{n+1}\right).
\]

Accordingly, this implementation uses
(\phi_n=+\arctan(\tau/(n+1))).  The Appendix A PDF line following Eq. (A3)
prints a minus sign, which conflicts with its own definition
(V_l(\phi)=e^{-i\phi P_l}), the immediately preceding
(I-i\tau P_l/(n+1)), and Lemma A.2.  The positive sign is fixed by direct
dense reconstruction of (e^{-i\tau H}).  This source inconsistency remains a
documented theoretical/editorial ambiguity to confirm with the authors or the
source paper cited as Ref. [13].

## Finite implementation and truncation bias

Only the finite set (n=0,2,\ldots,K) is sampled.  It exactly represents the
ordinary Taylor polynomial through degree (K+1):

\[
B_K\,\mathbb E_K[U]
=\sum_{j=0}^{K+1}\frac{(-i\tau H_R/\lambda_R)^j}{j!}.
\]

The omitted LCU coefficient 1-norm is bounded by the scalar exponential tail

\[
R_K(\tau)\le
\sum_{j=K+2}^{\infty}\frac{|\tau|^j}{j!}.
\]

`truncation_residual_bound` evaluates this tail directly.  Automatic cutoff
selection chooses the smallest even (K) with this value no larger than
`truncation_tolerance`.  This is a one-step bound.  Multi-step and RPE-round
bias must be propagated separately rather than relabeling it as the paper
normalization bound.

## RPE attenuation

For infinite RTE, Lemma A.2 gives

\[
B^r\,\mathbb E[Z]=
\langle\psi|e^{-it_RH_R}|\psi\rangle.
\]

Thus the normalization-only attenuation of one tail evolution is (B^{-r}).
For a partial product formula with tail occurrences (i) and `partial-S2`
repetition count (s), Eq. (A8) multiplies the corresponding normalizations;
the round attenuation is their reciprocal.  Finite code stores the directly
computed product from the selected integer parameters.  The paper's
(\exp(O(\lambda_R^2\delta^2s/r))) expression is retained only as
`paper_upper_bound` metadata.

The finite-(K) operator is not exactly unbiased for the infinite exponential.
Its truncation bias and normalization attenuation are reported separately.

## Identity and controlled circuits

Identity components participate in event probabilities and coefficients.
Dense references include their phase exactly.  In the next circuit milestone,
uncontrolled system global phase may be represented by circuit global phase;
controlled events must convert it to an ancilla-relative phase.  Basis changes
may remain uncontrolled when the central diagonal operation is controlled.

## Output taxonomy

- `paper_upper_bound`: an analytic inequality from the paper.
- `exact_finite_distribution`: a value computed from the selected integer
  cutoff and normalized finite event distribution.
- `empirical_compiled_estimate`: a transpiled circuit/event estimate with
  compiler and sampling metadata.
- `legacy_analytic_proxy`: the pre-existing `G_rand`, `B(kappa)`, anchor-Cgs,
  or fragment-summed cost model.  It is Level 0 and is not a finite-RPE total.
