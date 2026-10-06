# RTE reallocation R0: independent mathematical and semantic audit v1

Date: 2026-10-06 JST. This is a technical verification of the user/GPT
[plan](inputs/track_b_all_evidence_algorithm_redesign_20261006_user_input.md),
§13, on base `5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d`.
`PROVED` below means the stated mathematical claim under its explicit hypotheses.
It does not establish publication priority, a compiled implementation, or a resource improvement.
See the [review packet](rte_reallocation_r0_review_packet_20261006.md) for decisions.

## A1. Target, event order, and exact finite mean — PROVED

Let \(\widehat R=\sum_\ell p_\ell Q_\ell\), with \(p_\ell\ge0\),
\(\sum p_\ell=1\), and Hermitian involutions \(Q_\ell^2=I\).
No mutual commutation or Pauli closure is assumed. Absorb coefficient signs into
the involutions; retain any scalar identity evolution with its controlled phase.
For absolute dimensionless microstep \(x>0\), sign \(\sigma\in\{-1,+1\}\),
odd degree \(m=2d+1\), and \(t_n=x^n/n!\), the fixed target is

\[
M=P_m(-i\sigma x\widehat R)=\sum_{n=0}^{m}(-i\sigma)^n t_n\widehat R^n.
\]

The repository's even paired cutoff is \(K=m-1=2d\). Finite means degree
\(K+1\), not an infinite exponential or a normalized channel.
Choose \(a_k,b_k\ge0\), \(0\le k\le m\), with

\[
b_{-1}=0,\quad a_k+b_{k-1}=t_k,\quad b_m=0.
\]

Set \(c_k=\sqrt{a_k^2+b_k^2}\) and \(\phi_k=\operatorname{atan2}(b_k,a_k)\).
Omit zero coefficient events before calculating an angle or probability.
For independently drawn \(Q_0,Q_1,\ldots,Q_k\) from the same distribution, define

\[
U_k=(-i\sigma)^k e^{-i\sigma\phi_kQ_0}Q_k\cdots Q_1.
\]

Products act first in circuit time, rotation last. In particular, moving the
rotation through a noncommuting word is not a branch identity. Involution algebra gives

\[
c_k\mathbb E U_k=(-i\sigma)^k(a_kI-i\sigma b_k\widehat R)\widehat R^k.
\]

Summing and collecting consecutive degrees proves \(\sum_k c_k\mathbb E U_k=M\),
including negative time. The terminal condition prevents a spurious degree \(m+1\).
At \(x=0\), use the identity event directly; formulas dividing by \(\rho\) below
are not evaluated. These identities preserve the first operator moment, not
\(\mathbb E[U\varrho U^\dagger]\), event-wise circuit identity, or variance.
Independent samples at distinct occurrences preserve the product of finite means;
reusing one sample across occurrences has no such general guarantee.

Let \(B=\sum c_k\), draw order \(k\) with probability \(c_k/B\), then the IID
word indices. The corrected variable \(X=BU_k\) has \(\mathbb E X=M\) and
\(\mathbb E|B|^2=B^2\). Thus \(B^2\) is the canonical weight second moment;
it is not the exact state-dependent measurement variance. Two independent
forward/backward estimators require the corresponding product of moments.

## A2. Class lower bound and its all-odd-degree attainment — PROVED

This theorem concerns only the nonnegative adjacent-degree family just defined,
with the same IID involution distribution and one rotation per event. It is not
an optimum over arbitrary dictionaries, signed coefficients, merged words,
anticommutation-aware ensembles, coherent LCU, or different finite targets.
Write

\[
E_j=\sum_{\ell=0}^{j}t_{2\ell},\quad O_j=\sum_{\ell=0}^{j}t_{2\ell+1},
\quad O_{-1}=0,\quad \rho=O_d/E_d>0.
\]

Associate vector \((a_k,b_k)\) to even \(k\), and \((b_k,a_k)\) to odd \(k\).
The coefficient constraints make their sum exactly \((E_d,O_d)\).
The Euclidean triangle inequality therefore gives

\[
B\ge B_\star=\sqrt{E_d^2+O_d^2}.
\]

An explicit attaining solution, derived by alternating the equality slopes, is

\[
\begin{aligned}
a_{2j}&=E_j-O_{j-1}/\rho,& b_{2j}&=\rho E_j-O_{j-1},\\
a_{2j+1}&=O_j-\rho E_j,& b_{2j+1}&=O_j/\rho-E_j,
\end{aligned}\qquad 0\le j\le d.
\]

Here is a general feasibility proof, separate from the finite fixtures.
\(O_j/E_j\) is the weighted average of the decreasing sequence
\(x/(2\ell+1)\), with positive weights \(t_{2\ell}\).
Its prefixes decrease toward \(\sinh x/\cosh x=\tanh x\).
Likewise \(O_{j-1}/E_j\) is the weighted average of the increasing sequence
\(2\ell/x\), including its zero initial value, with the same weights.
Its prefixes increase toward \(\tanh x\). Therefore, for every \(j\le d\),

\[
\frac{O_{j-1}}{E_j}<\tanh x<\rho\le\frac{O_j}{E_j}.
\]

The strict middle inequality holds for finite \(d\) and \(x>0\).
This proves all four coefficients are nonnegative. Direct substitution proves
\(a_k+b_{k-1}=t_k\). At the terminal odd index \(m\), both coefficients are zero.
Every nonzero vector has slope \(\rho\), so equality in the lower bound is attained.
For \(d=0\), this is exactly the ordinary pair. For \(d\ge1\), \(x>0\),
ordinary even pairs have unequal positive slopes \(x/(2j+1)\) and hence
\(B_\star<B_{\rm pair}\), where

\[
B_{\rm pair}=\sum_{j=0}^{d}t_{2j}\sqrt{1+\left(\frac{x}{2j+1}\right)^2}.
\]

All coefficients depend only on \((x,m)\); constructing them costs \(O(m)\)
arithmetic operations, before precision, event generation, and application costs.

## A3. K=2, interpolation, and asymptotic size — PROVED

For degree three, \(\rho=(x+x^3/6)/(1+x^2/2)\), and the coefficients are

| k | \(a_k\) | \(b_k\) |
|---|---|---|
| 0 | 1 | \(\rho\) |
| 1 | \(x-\rho=2x^3/[3(x^2+2)]\) | \((x-\rho)/\rho=2x^2/(x^2+6)\) |
| 2 | \(x^3/(6\rho)=x^2(x^2+2)/[2(x^2+6)]\) | \(x^3/6\) |
| 3 | 0 | 0 |

The rational forms avoid subtracting nearly equal \(x\) and \(\rho\).
\(B_\star=\sqrt{(1+x^2/2)^2+(x+x^3/6)^2}\). In this case

\[
B_{\rm pair}^2-B_\star^2=x^2\left[\sqrt{(1+x^2)(1+x^2/9)}-1-x^2/3\right]>0,
\]

because the difference of the squares inside the comparison is \(4x^2/9>0\).
Formal expansion gives

\[
B_{\rm pair}-B_\star=x^4/9-13x^6/81+O(x^8).
\]

Consequently, at fixed total dimensionless time \(\tau\) split into \(s\) equal
steps, \(s\log(B_{\rm pair}/B_\star)=\tau^4/(9s^3)+O(\tau^6/s^5)\).
The canonical second-moment log ratio is twice this value. This is an asymptotic
formula; no H4 resource estimate or existing result was evaluated with it.

For \(0\le\eta\le1\), interpolate the ordinary and attaining coefficient arrays.
Linearity preserves the finite mean and nonnegativity, while convexity gives

\[
B_\eta\le(1-\eta)B_{\rm pair}+\eta B_\star\le B_{\rm pair}.
\]

This does not prove a convex synthesized cost, a cost-optimal \(\eta\), or an
improved finite-confidence task resource. Those require an explicit fair cost
and bias contract. No \(\eta\) search was performed.

## A4. Odd phase and implementation semantics — PROVED / UNRESOLVED

Even events carry \((-1)^{k/2}\); odd events carry \(\pm i\), with time-sign dependence.
For the optimum, even rotations use \(\theta=\arctan\rho\) and odd rotations use
\(\pi/2-\theta\). The exact complement identity is

\[
e^{-i\sigma(\pi/2-\theta)Q}=(-i\sigma)Qe^{+i\sigma\theta Q}.
\]

An odd event rewritten at the common small angle therefore has phase
\((-i\sigma)^{k+1}\), an additional \(Q_0\), and the opposite rotation sign.
Neither the extra involution nor its controlled relative phase is free.
Dropping a system global phase changes a controlled unitary's ancilla phase.

Static source inspection at the base found that `RTEEvent` explicitly rejects
odd orders and accepts only the even sign phase. `_make_event` uses the algebraic
signed angle \(\arctan(\tau/(n+1))\) and products-then-rotation application order.
No existing DF controlled wrapper verification is transferred to odd events.
See [source receipt](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/audit_manifest_v1.json).
The symbolic phase identity is proved; builder semantics, synthesis error, and
controlled gate cost remain `UNRESOLVED` and were not implemented.

Pauli words can collapse into one Pauli with a retained phase. The same reduction
must be available to every comparison arm. General DF components
\(V_\ell^\dagger P_\ell V_\ell\) in different bases do not close as Pauli words.
Consecutive same-basis runs can be simplified algebraically, but long-word,
phase, and basis-change cost must be counted. P-A's effective run-level policy
is a common baseline; this proposal does not revive its stopped interval-DP claim.

## A5. Counterexample to an excluded global claim — COUNTEREXAMPLE

For a single involution and \(x=1\),
\(P_3(-iQ)=\tfrac12I-i\tfrac56Q\) is one scaled rotation with normalization squared
\(17/18\). The adjacent nonnegative class optimum has normalization squared
\(65/18\). Thus the class optimum is not an arbitrary-LCU optimum. This refutes
an overclaim, not the plan's explicitly limited class theorem.

## B. Identity-return compression and direct probabilities — PROVED / KNOWN_EQUIVALENT

Let \(s_2=\sum p_i^2\), and
\(D=\widehat R^2-s_2I=\sum_{i\ne j}p_ip_jQ_iQ_j\). Without assuming any further cancellation,

\[
P_3(-i\sigma x\widehat R)=
(1-s_2x^2/2)I-i\sigma(x-s_2x^3/6)\widehat R
-\frac{x^2}{2}(I-i\sigma x\widehat R/3)D.
\]

The returned event has \(a=1-s_2x^2/2\), \(b=x-s_2x^3/6\), normalization
\(\sqrt{a^2+b^2}\), and angle \(\operatorname{atan2}(b,a)\).
Outside a small-step domain these may have different signs; a first-quadrant
\(\arctan(b/a)\) shortcut is not valid. The residual keeps the minus phase and
has normalization \((1-s_2)x^2\sqrt{1+x^2/9}/2\).
Writing \(v_0=(1,x)\), \(v_2=(x^2/2,x^3/6)\) proves

\[
B_{\rm ret}=\|v_0-s_2v_2\|+(1-s_2)\|v_2\|
\le\|v_0\|+\|v_2\|=B_{\rm pair}.
\]

For small \(x\), \(B_{\rm ret}=1+(1-s_2)x^2+O(x^4)\).
This is not an optimum over all algebraic returns, anticommutation, or joint A/B designs.
The compression principle is directly known from Zhao–Yuan; see the prior-art table.

When \(s_2<1\), the residual ordered-pair law is

\[
q(i,j)=\frac{p_ip_j}{1-s_2}\ (i\ne j),\quad
q(i)=\frac{p_i(1-p_i)}{1-s_2},\quad q(j\mid i)=\frac{p_j}{1-p_i}.
\]

Skip zero first-index probabilities, and when \(s_2=1\) omit the residual entirely.
With an \(O(L)\) prefix CDF for \(p\), draw \(u\) on \([0,1-p_i)\), then skip the
removed interval: \(v=u\) below \(F_{i-1}\), and \(v=u+p_i\) otherwise; invert
the original CDF. Together with the first-index CDF, this needs two draws and
\(O(\log L)\) searches, without a rejection loop. This is a distribution identity,
not a tested finite-bit RNG or sampling implementation. Near concentration,
rounding of \(1-s_2\), CDF endpoints, bias and acquisition cost need a separate contract.

## C. Mean stability and precision allocation — PROVED / KNOWN_EQUIVALENT

For real \(y\),
\(|P_3(-iy)|^2=1-y^4/12+y^6/36\le1\) exactly when \(|y|\le\sqrt3\).
Since \(\|\widehat R\|\le1\), degree-three mean operators are contractions for
\(|x|\le\sqrt3\). This is not a theorem for every cutoff or every step size.

If ideal factors satisfy \(\|M_j\|\le1\), approximate factors satisfy
\(\|\widetilde M_j-M_j\|\le e_j\), and errors include all local synthesis,
phase and coefficient approximations, then telescoping gives

\[
\|\prod_j\widetilde M_j-\prod_jM_j\|\le\prod_j(1+e_j)-1
\le e^{\sum_j e_j}-1.
\]

A local coefficient-weighted bound is \(e_j\le\sum_i|\alpha_{ji}|\delta_{ji}\).
Mean contraction can improve bias propagation but does not remove canonical
weight second moments, estimator range, or shot costs. It is not a finite-time
resource certificate by itself.

For positive \(d_i,\kappa_i,w_i\), the continuous surrogate
\(\min\sum_i d_i\kappa_i\log(1/\eta_i)\) subject to \(\sum_iw_i\eta_i\le\delta\)
has the interior KKT solution

\[
\eta_i=\frac{\delta d_i\kappa_i}{w_i\sum_jd_j\kappa_j}.
\]

Bounds on precision, non-logarithmic and discrete synthesizer counts, and
context-dependent coefficients require additional active-set or discrete work.
Rare probability alone does not justify coarse precision if \(d_i/w_i\) is
unchanged. This is standard constrained allocation, not a revived FR result.

## Fixed exact checks and stopping boundary

[Checker](../../../scripts/tracks/algorithm_codesign/check_rte_reallocation_symbolic.py)
uses Python standard-library `Fraction` arithmetic and reduction only by
\(Q_iQ_i=I\). It imports no research library and constructs no matrices or circuits.
The [pre-run protocol](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/symbolic_fixture_protocol_v1.json)
fixes all inputs; [results](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/exact_symbolic_checks_v1.json)
contain 80 A mean fixtures, 9 B fixtures, four deliberately incorrect mutations
and the excluded-global-optimum counterexample. Phase, negative time, endpoint,
concentrated/zero support, and the degree-six formal expansion passed.
This one local technical run is neither immutable CI nor external replication.
There was no trajectory, matrix solver, synthesis, compilation, Hamiltonian,
molecular NPZ access, or GPU operation. `RUN_READY=false`; mandatory STOP.
