# Locally Generated Return-Aggregated Taylor Ensembles: Finite-Precision Construction and Resource Trade-offs

**Standalone manuscript draft v1.2 — 2026-10-22 JST.** The title and central contribution are candidates for user/GPT assessment. This draft consolidates the existing G6 construction and its evidence; publication does not certify independent originality or authorize submission. Equations (1)–(37) retain the numbered mathematical content of [V11]. No new scientific calculation, numerical reanalysis, native synthesis, nontrivial proof or independent scientific review was performed for this edition. The [editorial and evidence record](../research/track_b_g6_v1_2_draft_review_integration_20261022/README.md) distinguishes existing results from the two restricted interpretations supplied in the 2026-10-22 review.

## Abstract

We describe a construction of weighted random-unitary ensembles for a fixed, odd-degree, short-step Taylor polynomial of a positive rational combination of Hermitian involutions. Formal return aggregation combines Taylor words related by adjacent equal-label cancellation while preserving the finite first operator moment. An existing insertion bound establishes positive aggregate coefficients in the specified short-step domain. Known free-product Green recurrences provide local coefficient queries, and even-parent/single-child Euler pairing converts these coefficients into unitary events. One generator, General Full, uses an ordinary Taylor envelope and pre-quantum rejection without first acquiring the full aggregated normalizer. We state its finite-bit proposal correction, coefficient and implementation bias, supplied-precision preflight conditions, and classical cost obligations. A closed degree-five construction is a specialization of the same aggregated representation and shares General Full's ideal acceptance-conditioned event law. Existing fixed synthetic native-resource examples show improvements over Ordinary while exposing stronger low-degree and CTS comparators. An exploratory double-factorized H4 application shows small logical circuit differences and substantial implementation-dependent classical processing. These observations delimit a construction and its resource trade-offs; they establish neither general quantum speedup nor equal-accuracy molecular total-resource superiority or inferiority.

## 1. Introduction

Randomized Taylor representations can replace a long coherent linear combination by repeated executions of relatively short unitary circuits and classical weights. The circuit for one event, its probability and weight, the number of trials required for a confidence guarantee, and the cost of obtaining the representation all matter. A reduction in coefficient mass alone does not settle the total-resource question.

This manuscript concerns **return aggregation (RA)**: a representation of a fixed finite Taylor operator in which all formal words related by adjacent cancellations of equal involution labels are collected. It distinguishes this representation from **General Full**, one generator of its events using local Green coefficient queries and an ordinary envelope. RA's mathematical validity, the generator's information requirements, its classical implementation cost, and its quantum resource consequences are separate questions. A useful special case is **Closed P5**, a grouped canonical construction developed in the G9 series for degree five. Closed P5 belongs to the RA family; its performance against General Full does not by itself invalidate that family. [E1][E2][E3][G9]

The constructive content is the connection between the same finite target, positive return coefficients, local coefficient acquisition, unitary event generation without first acquiring the full RA normalizer, and finite-precision/error/cost conditions. Green-function theory, Euler pairing, rejection sampling and importance weighting are established ingredients. Ordinary Taylor sampling and CTS layer sampling also have nonenumerative implementations. We therefore do not present nonenumeration alone, or the absence of identical prior code, as evidence of independent novelty. [WBC][AK][PSN][CAS][PRIOR]

The evidence contains both positive resource examples and important limits. G7 gives a conditional positive result under an exact controlled-query/helper model. G9 and G10 introduce stronger degree-five and matched-native comparisons, where Closed P5 and CTS constrain the additional benefit of General Full. The later H4 study provides an application and a limited logical diagnostic, rather than a common-accuracy native-cost experiment. These studies retain their original targets, providers, confidence policies and accounting units; we do not combine them into a new resource ranking. [G7][G9][E4][C1SUM]

Sections 2–7 give a self-contained construction and its implementation conditions. Section 8 presents the existing synthetic evidence and counterexamples. Section 9 integrates the molecular application. Section 10 states the actual open claims. Appendices retain the supplied-precision conditions, fixed-dictionary resource limitation, implementation evidence and source provenance.

## 2. Problem setting and access model

Let p be an explicit table of L≥1 positive rational numbers with sum one. Let x be rational with 0<x≤1, m≥1 an odd integer, and σ∈{−1,+1}. On one Hilbert space, each Q_i is a Hermitian involution. Define

\[
R=\sum_{i=1}^{L}p_iQ_i,\qquad
M_m=P_m(-i\sigma xR)=\sum_{n=0}^{m}\frac{(-i\sigma xR)^n}{n!}
\tag{1}
\]

Zero-mass labels can be removed before entering this domain. The case x=0 is a separate exact identity branch. The finite polynomial M_m is the target of the main construction. Replacing it by the exponential, assigning a Hamiltonian/model-binding error, or estimating an energy requires additional task-specific accounting.

For a word u=(i₁,…,i_l), let Q(u)=Q_i₁⋯Q_i_l and p(u)=∏_t p_i_t. The reduced word red(u) is obtained by deleting adjacent ii pairs. **Formal return** refers only to this relation. Distinct reduced words may map to equal physical operators when the Q_i have additional relations; the construction does not claim to collect every such equality. It does not reorder noncommuting factors. Q(u) is written in operator order from left to right, so the corresponding circuit-time word is reversed. A general odd word Q(u) need not be an involution. [E1]

Write t_n=xⁿ/n!, χ=Σ_i p_i² and μ_k=Σ_i p_i^k. A subscript on s_u denotes a child coefficient sum; the scalar s in trial-budget formulas denotes the remaining statistical error width. We use B_RA=B_full for the full aggregated coefficient mass and B_o=B_Ordinary for the ordinary envelope. N is the number of **all classical trials per axis**; K is the expected accepted quantum-call count for two axes in the resource tables. Root/probability precision parameters H_root and H_prob are unrelated to a Taylor cutoff that may be called K elsewhere. In the retained equation (13), ρ denotes the density matrix ρ_sys; in the finite-bit equations (16) onward, ρ denotes the relative coefficient-error parameter.

The coherent target for a density matrix ρ_sys is Re Tr(ρ_sys M_m) or Im Tr(ρ_sys M_m). Equality of the first operator moment E[V] does not imply equality of E[Vρ_sys V†]. Global phases of system unitaries become relative phases in controlled implementations and must be retained.

Required access consists of the explicit p table, controlled Q_i, signed controlled exp(−iσφQ_i), actual adjoints, and evidence for strict implementation error, workspace and native cost. System-only oracle access does not supply controlled access for free. Local scalar arithmetic does not certify the physical provider, the input state, preparation/readout, a random-bit source, or a complete estimator. [E2]

## 3. Related work and the scope of the contribution

Wan, Berta and Campbell use adjacent Taylor-order pairing and efficient order/IID-label sampling in randomized statistical phase estimation. We use the known Euler identity after formal return collection and retain the changed parent-dependent coefficients and event law. One involution rotation or nonenumeration, considered alone, is not a new principle. [WBC]

Aomoto and Kato study Green multipliers on free products of cyclic groups. The Z₂ specialization supplies the recurrences used here. Their Green theory is established; the present construction concerns its finite-Taylor coefficient interpretation and connection to positive event coefficients, rejection, finite-bit weights and costs. The correspondence to their conventions is given in §4.4. [AK][E1]

Zhao and Yuan exploit anticommutation, modified Taylor expansions and absorption of higher-order contributions. Zeng, Sun, Jiang and Zhao use Taylor/LCU compensation and order pairing. These are relevant precedents for coefficient absorption and pairing. Additional algebraic information, a modified target and the same fixed finite polynomial must be distinguished when comparing full algorithms. We do not infer equality or inequality of complete event laws from a shared Euler step. [ZY][ZSJZ][PRIORPIPE]

Peetz, Smart and Narang's stochastic combination of unitaries includes CTS and layer sampling that can avoid a full high-order expansion. The explicit matched CTS used in §8 is one finite-target specialization in a setting where Pauli data are available. A result against that implementation is not a result against every CTS construction. [PSN][G9]

Cugini, Atif and Subaşı connect importance sampling, circuit cost and statistical burden and distinguish ZeroFill from Discard. Correcting by the actual proposal is known importance weighting. Our pre-quantum classical rejection should not be assigned the physical cost of a different post-detection rejection protocol. The fixed-dictionary accounting in Appendix B applies the known cost–moment principle to a restricted comparison class. [CAS]

The partially randomized time-evolution literature provides the motivating split-Hamiltonian/RTE setting and includes involution-based Taylor constructions. This context does not establish that the present finite-target representation reduces the total molecular phase-estimation cost. [PR]

The existing full-generation prior-art audit maps these correspondences without proving priority, complete reduction to a prior algorithm, or a general non-reducibility theorem. The remaining contribution candidate is the complete specification under the stated input/access conditions, rather than independent novelty of each component. Existing audits of partial-sum queries and exact coefficient-type canonical generation also retain standard sampling/counting components and unresolved standalone originality. No new sampler from those audits is developed here. [PRIOR][PRIORPIPE][PRIORCAND][ASOURCE][BSOURCE]

## 4. Return aggregation and mathematical construction

The statements and proof arguments in this section restate the existing G6 audit and v1.1; they are not new proofs prepared for this edition. [E1][V11]

### 4.1. Collecting formal returns preserves the finite operator

Let P_n(u) be the probability that n IID labels drawn from p reduce to u. Parity is preserved by pair deletion, and therefore

\[
a_u=\sum_{\substack{|u|\le n\le m\\n-|u|\ {\rm even}}}
(-1)^{(n-|u|)/2}t_nP_n(u),\qquad
M_m=\sum_{u}(-i\sigma)^{|u|}a_uQ(u).
\tag{2}
\]

This is a finite regrouping of Taylor words. The sign follows from (−iσ)ⁿ=(−iσ)^|u|(−1)^((n−|u|)/2). It requires neither commuting the Q_i nor assuming any further relations. The identity remains valid after mapping formal words to physical operators with additional relations.

### 4.2. Short-step positivity

The insertion-cover argument gives

\[
P_{n+2}(u)\le(n+1)\chi P_n(u),\qquad
\frac{t_{n+2}P_{n+2}(u)}{t_nP_n(u)}\le\frac{\chi x^2}{n+2}\le\frac12.
\tag{3}
\]

To see the origin of the first inequality, delete a deterministically chosen adjacent pair from a length-(n+2) raw word that reduces to u. Conversely, insert ii into every gap of each length-n word reducing to u. This multiset covers the desired words and can overcount them; summing the insertion weights yields an upper bound, not an equality. If P_n(u)=0, the same cover implies P_(n+2)(u)=0, so the ratio need not be formed.

For l=|u| the successive magnitudes of the finite alternating sum decrease. Its first two terms and first term give

\[
t_lp(u)\left(1-\frac{\chi x^2}{l+2}\right)\le a_u\le t_lp(u),\qquad a_u>0.
\tag{4}
\]

The bound also covers a terminal coefficient with only one contribution. Positivity is conditional on the specified short-step domain; no general x>1 or signed-p extension is asserted.

### 4.3. Even-parent and single-child pairing

Use an even-length reduced word u as a parent. For the empty parent, C(u) contains all labels; otherwise C(u) excludes first(u). Each odd reduced word iu has exactly one even suffix u. Define

\[
s_u=\sum_{i\in C(u)}a_{iu},\quad d_u=\sqrt{a_u^2+s_u^2},\quad
\phi_u=\arctan(s_u/a_u),\quad q(i\mid u)=a_{iu}/s_u,
\tag{5}
\]

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_uQ_i}Q(u).
\tag{6}
\]

The known involution Euler identity gives

\[
d_u\sum_iq(i\mid u)V_{u,i}
=(-i\sigma)^{|u|}a_uQ(u)+\sum_i(-i\sigma)^{|u|+1}a_{iu}Q_iQ(u).
\tag{7}
\]

Thus a parent and all its children are counted once. The rotation generator is the single involution Q_i, not the possibly non-involutory odd word Q(iu). If s_u=0, use the pure-parent convention. The parent phase (−iσ)^|u|=(−1)^(|u|/2) is retained in a controlled circuit. Pairing is known; the RA-specific inputs to it are the collected coefficients and the resulting parent/child law. [E1][WBC]

### 4.4. Local Green coefficient queries

Formal reduced words form the Cayley tree of the free product of L copies of Z₂. First-passage series F_i and the root-return series G satisfy

\[
F_i(z)=\frac{zp_i}{1-z\sum_{j\ne i}p_jF_j(z)},\quad F_i(0)=0,\qquad
G(z)=\frac1{1-z\sum_ip_iF_i(z)},
\tag{8}
\]

\[
\sum_{n\ge0}P_n(u)z^n=G(z)\prod_{t=1}^{|u|}F_{i_t}(z).
\tag{9}
\]

The path to a reduced word is unique. First-passage factors account for this path and G for the return contribution. Scalar products of these series commute; the physical Q(u) retains its order. The formal solution at z=0 is sufficient, including the recurrent L=1 case F₁=z, G=(1−z²)⁻¹. No numerical resolvent branch or infinite-length sampler is required.

These formulas specialize the known Aomoto–Kato multipliers. The existing audit uses their walk weights p_i/2 and spectral variable ζ=1/z, dividing the resolvent by z to match G. This is a correspondence established in the G6 audit, not a claim that the original Green-function paper contains the present Taylor-event generator. [AK][E1]

Let f_i[n]=[zⁿ]F_i and C[n]=Σ_i p_i f_i[n]. The truncated recurrences are

\[
f_i[n]=p_i\mathbf1_{n=1}+\sum_{b=1}^{n-2}f_i[b]\big(C[n-1-b]-p_if_i[n-1-b]\big),
\tag{10}
\]

\[
G[0]=1,\qquad G[n]=\sum_{a=1}^{n-1}C[a]G[n-1-a].
\tag{11}
\]

After sharing C[n], preprocessing takes O(Lm²) rational operations and O(Lm) coefficients. Querying a specified parent and all children takes O(m³+Lm²) rational operations using a cached parent series. This avoids a global reduced-word dictionary. It is an arithmetic count, not a whole online bit-cost or runtime bound. [E2][F2]

## 5. Local generation without acquiring the full normalizer

### 5.1. Ordinary envelope and General Full

For even l≤m−1, put b_l=√(t_l²+t_(l+1)²) and B_o=Σ_l b_l. The existing positive-coefficient bounds imply

\[
d_u\le b_lp(u),\qquad
B_{\rm full}:=\sum_{u\ {\rm even,reduced}}d_u\le B_{\rm ord},\qquad
A_u:=\frac{d_u}{b_lp(u)}\ge\frac1{2\sqrt2}>\frac14.
\tag{12}
\]

The following ideal **General Full** generator realizes the RA representation.

1. Draw l with probability b_l/B_o and a raw length-l IID word from p.
2. If the raw word is not already reduced, output a zero trial. Do not move it to its reduced parent.
3. Otherwise query its coefficients locally and accept with probability A_u=d_u/(b_l p(u)). Rejection is also a zero trial.
4. On acceptance draw i with probability a_iu/s_u and execute V_(u,i), retaining controlled relative phase.
5. For the selected coherent axis, use a Hadamard outcome Y∈{−1,+1}. Output X=B_oY on acceptance and X=0 otherwise; average over all trials.

The first-moment and second-moment statements are

\[
Z=\frac{B_{\rm full}}{B_{\rm ord}},\quad
E[X]=\operatorname{Re}\text{ or }\operatorname{Im}\operatorname{Tr}(\rho M_m),\quad
E[X^2]=B_{\rm ord}B_{\rm full},\quad |X|\le B_{\rm ord}.
\tag{13}
\]

The full RA normalizer need not be acquired for this generator; the ordinary order normalizer and local coefficient work remain necessary. The empty-parent contribution gives Z≥(1−χx²/2)e⁻ˣ≥1/(2e). The conditional A_u bound in (12) is distinct from the all-trial acceptance Z. For m≥3 and x>0, the existing strict envelope argument gives B_RA<B_o, without implying strict improvement in native or total resources. [E1]

Formal return contributions have already been included in the coefficient query. Rejecting a raw nonreduced proposal does not discard those contributions from the target. Reassigning that proposal to its reduced parent would double-count the representation. Likewise, an accepted-only average changes normalization and is not (13).

### 5.2. A usable upper bound and all-trial budgets

For statistical remaining width s>0 and axis failure probability α, a certified acceptance upper bound Z_+ permits the existing sufficient budget

\[
N\ge\left\lceil\log(2/\alpha)
\left(\frac{2B_{\rm ord}^{2}Z_+}{s^2}+\frac{4B_{\rm ord}}{3s}\right)\right\rceil.
\tag{14}
\]

Absent additional information one can use Z_+=1. The later G6/G7 construction obtains a normalizer upper bound without acquiring B_RA:

\[
B_{\rm full}\le U=d_\emptyset+\sum_{l=2,4,\ldots,m-1}b_lr_l\le B_{\rm ord}.
\tag{15}
\]

Here r_l is the IID mass of raw words that are **already reduced**, not a distribution of reduced lengths. It is obtained from v_i(1)=p_i, v_i(l)=p_i(Σ_j v_j(l−1)−v_i(l−1)), and r_l=Σ_i v_i(l). The DP takes O(Lm) arithmetic work, in addition to the empty-parent coefficient/root query. Digital rounding factors and outward endpoints must be restored before using U in an implemented budget. The order of exact ideal quantities does not automatically order arbitrary rounded endpoints. [MATRIX][GEN][W2B]

Unknown B_RA or Z, an exact signal, and a reference enumeration are not free information for reducing N. N is a sufficient all-trial count under the measurement/IID and bias assumptions; it is not a necessary minimum. Expected accepted calls, a high-probability resource cap and a hard number of attempts are different quantities.

### 5.3. Classical work is part of the construction

Raw rejection still consumes order/label sampling and checking. A coin rejection also consumes the local coefficient packet, root bounds and any child-law construction already performed. The existing implementation constructs its child law before the coin and contains repeated child sums, law validation, sorting, finite-bit guards, hashing/serialization and angle-time requery. These costs are not removed by calling the method local or nonenumerative. [F2][F3]

The physical event provider, native-angle acquisition and preparation/readout add separate costs. A coefficient generator that does not enumerate a dictionary need not obtain all native prices or strict error certificates nonenumeratively. This distinction becomes concrete in the H4 application.

## 6. Finite-precision implementation and error accounting

### 6.1. Actual proposals and corrected weights

The ideal probabilities and d_u are generally irrational. In the existing finite-bit construction, order, label, acceptance and child masses are full-support dyadic approximations. A rational approximation d̃_u defines

\[
\pi_{u,i}=h_l\prod_{j\in u}h_j\,h_Ah_{i\mid u},\quad
\widetilde\alpha_{u,i}=\widetilde d_u\frac{a_{iu}}{s_u},\quad
\widetilde W_{u,i}=\frac{\widetilde\alpha_{u,i}}{\pi_{u,i}}.
\tag{16}
\]

The equality π_(u,i)W̃_(u,i)=α̃_(u,i) is exact rational importance correction. With ideal event unitaries it gives M̃=Σ_e α̃_eV_e, generally distinct from M_m. Correcting probability rounding does not remove coefficient approximation or angle/native error. [E2]

Assume each proposal factor lies within the stated relative η tolerance and |d̃_u−d_u|≤ρd_u, with 0<η,ρ<1 as in v1.1. Then the retained bounds are

\[
\pi_{u,i}\ge(1-\eta)^{l+3}\alpha_{u,i}/B_{\rm ord},\quad
|\widetilde W|\le\frac{(1+\rho)B_{\rm ord}}{(1-\eta)^{m+2}},\quad
\|\widetilde M-M_m\|\le\rho B_{\rm full}\le\rho B_{\rm ord}.
\tag{17}
\]

The pure-parent branch may use the same conservative envelope without a child factor. An implemented second moment is bounded rather than identified with its ideal value:

\[
E[\widetilde X^2]\le\kappa B_{\rm ord}U,\qquad
\kappa=\frac{(1+\rho)^2}{(1-\eta)^{m+2}}
\tag{18}
\]

If U is not acquired, substitute B_o as the conservative upper bound. Since B_o≤eˣ<3, range and coefficient bias can also be bounded without B_RA. Native event approximations V̂_e with strict operator error δ_e satisfy

\[
\|\sum_e\widetilde\alpha_e\widehat V_e-M_m\|
\le\rho B_{\rm ord}+\sum_e\widetilde\alpha_e\delta_e
\tag{19}
\]

Angle error, native error, input/model binding, Taylor truncation and statistical error remain distinct layers. A coherent measurement task must supply its own bias convention. A projective error bound that discards a controlled relative phase does not suffice.

### 6.2. Supplied-precision preflight and angle evidence

The existing support lower bounds include order mass ≥t_(m−1)/3, label mass ≥p_min, valid-parent acceptance >1/4, existing child mass ≥p_min/2, and d_u≥t_(m−1)p_min^(m−1)/2. Their inverse logarithms are bounded in terms of input bit length and numerical m. Thus suitable finite precision exists; this does not say that one fixed bit setting works for every input. [E2]

F1 specifies sufficient predicates P0–P5 for **provided** H_prob and H_root, including domain, order, label/child, coefficient, acceptance and law/input identity. Appendix A states the actual inequalities. They do not select minimal precision. F3 connects Full's preflight to the existing generator, provides an identity branch and distinguishes valid zero, invalid input and terminal failure. Its recorded 63 artificial tests are technical implementation evidence, not a new test run or a physical-provider certificate. The original F3 canonical nonidentity restriction and later specialized audit paths must not be conflated. [F1][F3]

The existing angle construction uses r=s_u/a_u≤2x/(l+1)≤2 and φ=2 atan[r/(1+√(1+r²))]. The inner argument has absolute value below 2/3. Outward root bounds and a finite alternating atan series enclose the angle, with its next-term error. A strict involution rotation changes by at most the angle error. F3's angle interval is not a T/CX certificate, provider implementation, physical RNG or completed signal estimator. [E2][F3]

Only legitimate raw-nonreduced and coin-reject outcomes enter the all-trial average as zero. Precision failure, source/law mismatch, invalid input, memory/storage failure and incomplete execution are terminal conditions. Retain the prefix and cause; do not assign the completed-N confidence guarantee to an incomplete prefix or condition it on runs that happened to finish.

### 6.3. Bit complexity and full classical cost

With total p-table bit length H_p, x bit length H_x and D=lcm den(p_i), log D≤H_p. The n-th probability coefficient has a denominator dividing Dⁿ; aggregate coefficients have denominators dividing D^m den(x)^m m!. The retained coefficient bit-length bound is O(m(H_p+H_x+log m)). This addresses numerical m, not polynomial complexity in log m. [E2]

F2 charges normalized rational arithmetic conservatively by Q(b)=O((b+1)³), and root enclosure by R(b,H_root)=O((b+H_root+1)³) in a standard bit model. A linear selector also validates/scans its distribution, dyadic construction sorts remainders, actual π can involve up to m+2 dyadic factors, and repeated child sums can add an O(L²) arithmetic term outside the coefficient-query count. Fixed-source F3 hashing, serialization, guards and event requery are additional work. No absent alias table or cache is assumed. [F2]

For method a, use the existing decomposition C_cl,a=S_a+N_a μ_a+Ω_a, where μ_a averages the executed raw/coin/accepted prefixes and Ω_a includes unaccounted output/finalization. In one common classical unit, Full's noninferiority condition is

\[
(N_a-N_F)\mu_a\ge N_F(\mu_F-\mu_a)+(S_F-S_a)+(\Omega_F-\Omega_a).
\tag{C1}
\]

This is the existing W2c bookkeeping identity, not a measured speedup or a new complexity theorem. Native resources instead use C_q,a,d=N_aΣ_eπ_a,e c_a,d(e), with π measured before conditioning on acceptance. Preparations can be charged only to accepted events if they can actually be delayed until after pre-quantum rejection. Bit operations, CPU seconds, bytes, T/CX/1Q and depth remain separate coordinates. [W2C]

## 7. Closed-P5 specialization and resource trade-offs

### 7.1. Closed coefficients and a small number of groups

For m=5, even parents have lengths 0,2,4. The existing G9 construction is a specialization **within RA**, using the following coefficients. With t_n=xⁿ/n!, the root is

\[
a_0=1-\chi t_2+(2\chi^2-\mu_4)t_4,
\tag{20}
\]

\[
a_i=p_i\{t_1-(2\chi-p_i^2)t_3+
(5\chi^2-4\chi p_i^2+2p_i^4-2\mu_4)t_5\},\quad s_0=\sum_i a_i.
\tag{21}
\]

For an ordered pair u=(j,k), j≠k, define

\[
A_{jk}=t_2-(3\chi-p_j^2-p_k^2)t_4,
\tag{22}
\]

\[
b_i^{jk}=\begin{cases}
p_i\{t_3-(4\chi-p_i^2-p_j^2-p_k^2)t_5\},&i\ne j,\\0,&i=j,
\end{cases}
\tag{23}
\]

\[
S_{jk}=(1-p_j)t_3-
\{(1-p_j)(4\chi-p_j^2-p_k^2)-(\mu_3-p_j^3)\}t_5
=\sum_i b_i^{jk}
\tag{24}
\]

Then a_jk=p_jp_kA_jk, a_ijk=p_jp_kb_i^jk, d_jk=p_jp_k√(A_jk²+S_jk²), and tan φ_jk=S_jk/A_jk. For length-four reduced parents whose first label is j,

\[
a_u=t_4p(u),\quad a_{iu}=t_5p_ip(u)\ (i\ne j),\quad
d_u=p(u)\sqrt{t_4^2+(1-p_j)^2t_5^2},\quad
\tan\phi_u=x(1-p_j)/5.
\tag{25}
\]

Let h_j(0)=1 and h_j(r)=Σ_(k≠j)p_kh_k(r−1). The length-four IID mass beginning with j is r_j^(4)=p_jh_j(3). Consequently

\[
B_5=d_0+\sum_{j\ne k}p_jp_k\sqrt{A_{jk}^2+S_{jk}^2}
+\sum_jr_j^{(4)}\sqrt{t_4^2+(1-p_j)^2t_5^2}.
\tag{26}
\]

There are one root group, L(L−1) ordered-pair groups and L first-label groups: at most L²+1 groups, omitting nonroot groups for L=1. Positivity and nonzero conditional laws follow from the existing short-step proof.

For completeness, the retained one-return identity for a reduced base word is

\[
P_{l+2}(u)=p(u)\{(l+1)\chi-\sum_{t=1}^{l}p_{i_t}^{2}\}.
\tag{27}
\]

This equality corrects duplicate insertions in a triple run; it is different from the general cover inequality (3). The existing walk recurrence gives P₂(empty)=χ, P₃(i)=p_i(2χ−p_i²), P₄(empty)=2χ²−μ₄ and P₅(i)=p_i(5χ²−4χp_i²+2p_i⁴−2μ₄). Substitution into (2) gives (20)–(25). These are the G9 derivation and frozen source formulas, restated rather than newly proved or recomputed. [E3][P5SRC]

### 7.2. Canonical generation and its implementation costs

Draw a group according to its mass in (26). The root and pair groups identify a parent directly. For a first-label j group, begin with j and complete the length-four word by drawing a next label k≠previous with probability p_kh_k(r−1)/h_previous(r), where r is the number of positions still to fill. The resulting parent law within the group is p(u)/r_j^(4). Draw the root/pair child by normalizing (21)/(23), or for length four use p_i/(1−p_j), i≠j. Execute the same V_(u,i) with ideal weight B₅. [E3]

The grouped formula avoids a full child table. Group construction uses O(L²) rational operations, with shared moments allowing constant-work pair scalar formulas; local conditional construction is O(L). These statements do not make the full sampler O(L): the frozen dyadic group selector can scan O(L²) groups, and root precision, bit lengths and output/provider costs remain. The later H4 cumulative integer selector/cache is an implementation choice, not a new sampling principle. [F2][G9][C1COST]

Finite-bit Closed P5 encloses **group masses**, uses rational representatives, and divides the approximate coefficient by the actual group/word/child proposal. A relative group error ρ yields a global coefficient bias bound ρB₅ by the existing L1 argument. The ideal Full/P5 ensemble is shared, but their rational coefficient approximations, proposal factors, guards and realized trajectories need not be byte-identical. A successful Full preflight does not certify every canonical law. [V11][F1]

### 7.3. What the ideal Full/P5 equality does and does not say

For the same p, x, σ, m=5, pairing convention, event labels and positive coefficient α_e=d_u a_iu/s_u, put B=Σ_eα_e=B₅ and O=B_o. The existing ideal Full law and Closed P5 canonical law give

\[
P_F(e)=\alpha_e/O,\qquad Z=B/O,\qquad
P_F(e\mid\mathrm{accepted})=\alpha_e/B=P_C(e).
\tag{R1}
\]

Under the same provider, compilation/precision convention and event-cost function, their accepted event circuit distribution is therefore the same. Event identity still does not make physical/native implementation free. The principal distinction is how the generator obtains the event and normalizer and pays for setup, local queries, rejection and certification. [E1][G9]

The 2026-10-22 review supplied the following restricted interpretation of these existing laws. For ideal ±1 measurements, Full outputs OY or zero, whereas canonical P5 outputs BY. With the same accepted-event mean quantum price c̄,

\[
M_{2,F}=OB,\quad M_{2,C}=B^2,\qquad
M_{2,F}(Z\bar c)=M_{2,C}\bar c=B^2\bar c.
\tag{R2}
\]

This second-moment metric balances pre-quantum rejection against the larger Full weight. The same accepted-law reasoning applies to independently generated microsteps under a common accepted-path cost convention, including common boundary cancellation. We include it as the supplied interpretation, not a new theorem or evaluated task-cost result. Actual variance, finite-bit approximation, range/ceiling terms, conservative budget bounds, hard caps, failure and classical processing are outside this equality. In particular it does not imply exact equality of finite-bit end-to-end costs. [E1][G9][V11]

### 7.4. Small-step interpretation and the empty parent

The supplied review also organized the existing normalizer formulas at fixed p and fixed odd m≥3 as x→0:

\[
B_o=1+x^2+O(x^4),\qquad
B_{\rm RA}=1+(1-\chi)x^2+O(x^4),\qquad \chi=\sum_i p_i^2.
\tag{R3}
\]

This is an interpretation of the existing coefficient/normalizer formulas, including the root and length-two parent masses; it is not a newly established general theorem in this edition. No uniform remainder constant in p or m is asserted. Neither a numerical χ for H4 nor a new normalization, moment, N or resource improvement is evaluated here. A label count alone cannot justify replacing χ by 1/L for a nonuniform input. [V11][GEN]

The empty parent still has the child rotation exp(−iσφ_empty Q_i). Its coefficient a_empty contains higher-order return contributions, explicitly visible in (20), and its child sum contains contributions through t₅ in (21). Thus an overwhelmingly degree-zero sample can offer few opportunities to shorten the **nonempty-parent logical word**, while aggregation still affects coefficients, angles, normalizer and statistical weights. “Degree zero” neither means a free identity circuit nor proves that the aggregation effect is zero.

## 8. Synthetic native-resource results and strong comparators

All numerical tables in this section quote fixed saved displays. Exact rational data remain at the linked source; no new ratios, budgets, common-confidence conversion or ordering were computed for this draft. Molecular geometry, orbital basis, DF rank and L_D do not apply to these synthetic examples.

### 8.1. G7: a conditional positive result

G7 uses exact controlled-Q access with label-dependent nonnegative provider prices. A helper compute–rotate–actual-adjoint uncompute implementation adds CQ2/H4/Rz2/CX2 for a child rotation, retains outer phase, and uses two extra qubits. The fixed P5 input is p=(1/5,3/10,1/2), x=5/7, m=5, H_prob=160, H_root=256, η=ρ=10⁻¹² and strict Rz primitive error 10⁻⁶. Each axis has error 1/200; the original family comprises 16 axes with total failure allowance ≤1/20. These conditions differ from G9/G10's direct provider and confidence families. [G7]

| G7 P5 arm | Saved sufficient all trials N/axis | Expected accepted calls, two axes | Saved Rz T intercept, two axes |
|---|---:|---:|---:|
| Ordinary | 1,174,526 | 2,349,052 | 321,659,736.670 |
| Partial Return + tail | 894,664 | 1,789,328 | 238,505,280.446 |
| Closed P3 + tail | 880,312 | 1,760,624 | 246,677,716.207 |
| General Full | 1,014,322 | 1,740,102.129 | 237,260,558.660 |

The saved affine Full-minus-baseline intercepts, label-provider coefficients and preparation coefficients are negative against these three registered P5 comparators. This establishes the original conditional expected-T result for all nonnegative prices in that additive model, without treating provider/preparation as free. Closed P5 was not a G7 comparator. The separate P3-control input has a different outcome against its closed comparator. Full's saved P5 hard call cap 2,028,644 exceeds the Closed-P3 comparator's 1,760,624, illustrating the distinction from expected calls. Actual quantum shots were not executed. [G7]

### 8.2. G9 and G10: matched native implementation

G9/G10 use p=(1/5,3/10,1/2), x=5/7, σ=+1 and an explicit three-system-qubit provider:

\[
Q_0=Z_0,\quad V_1=R_{XX_{01}}(\pi/4),\quad
V_2=R_{XX_{12}}(\pi/4)R_{ZZ_{01}}(\pi/4),\quad Q_i=V_i^\dagger Z_iV_i,
\tag{28}
\]

Here R_P(θ)=exp(−iθP/2), V₀=I, and operator products act from the right. Pauli expansions are explicitly accessible in this synthetic I1 context. We infer no inaccessible-Pauli or molecular input-acquisition advantage from it.

All arms may use direct V†–controlled Pauli rotation–V, actual adjoints, signed rotations, controlled parent phase, CZ→H CX H lowering and adjacent inverse cancellation. Literal T/T†, CX and native 1Q counts are retained. Provider π/4 sequences are exact in this Clifford+T model, with δ_provider=0; this is a model condition. Direct construction uses one extra qubit. It is not a globally optimized whole-circuit compiler or the best possible synthesis result. [G9][E4]

G9's original P5 comparison has 22 axes with α_axis=49/22000 and 11 resource rows with β=1/11000. Its saved native T intercepts include Ordinary 348,759,216.423, General Full 257,506,695.137, Closed P5 255,860,636.626 and matched CTS 364,311,583.986. Its recorded accepted-event T price is 139.489993570 for both ideal-law-matched Full and P5. These are G9 values under its original policy, not substitutes for G10's table. [G9]

G10 separately uses 17 rows/34 axes, α_axis=49/34000 and β_row=1/17000, with total failure 1/20. The six m=5 anchors reuse immutable G9 bindings/sequences under G10's already executed common policy; this draft does not rebudget them. New m=3/7 rows are comparisons **within each degree's finite target**, not an equal-exponential-accuracy ranking between degrees. [E4]

### 8.3. G10 error and cost definitions

The fixed parameters are H_prob=160, H_root=256, η=ρ=10⁻¹² and strict Rz primitive error ε_Rz=10⁻⁶, with `up_to_phase=false`. Each finite-target axis has error 1/200. The existing common bias and statistical remaining width are

\[
b_{\rm coh}=2\{8\rho+8(1+\rho)2\epsilon_{Rz}\},\qquad
s=1/200-b_{\rm coh}>0.
\tag{29}
\]

The factor-two convention is part of this fixed task. The four layers are the finite polynomial target, coefficient approximation bias, strict controlled native error, and statistical error. Taylor error to the exponential is outside this comparison. Matrix tolerances used for phase/order diagnostics do not replace the confidence bias bound.

For saved certified moment/range bounds,

\[
N=\left\lceil\ell_+\left(2M_{2,+}/s^2+4W_+/(3s)\right)\right\rceil,
\quad\ell_+\ge\log(2/\alpha_{\rm axis}),\quad\alpha_{\rm axis}=49/34000.
\tag{30}
\]

Full uses outward B_o/U-based bounds with its rounding factor; Closed P5 uses its grouped normalizer and CTS its collected I1 norm. Reference exact signals/normalizers are not inserted to shrink Full's budget. The 34 axis allowances sum to 49/1000 and the 17 resource allowances sum to 1/1000.

With actual saved proposal masses π_e and native prices C_e,j,

\[
Z=\sum_e\pi_e,\quad K=2NZ,\quad
C_{0,j}=2N\sum_e\pi_eC_{e,j},\quad
G_j(h_j)=C_{0,j}+Kh_j\quad(j=T,CX,1Q).
\tag{31}
\]

The nonnegative preparation/readout prices h_j remain independent variables. T is included in native 1Q and must not be double-counted by adding those coordinates. Common outer 1Q fields are distinct from the tabulated native-event intercepts. The registered accepted-call cap, for M=2N and t_resource=10, is

\[
C_{\rm acc}=\min\{M,\lceil MZ_++\sqrt{2MZ_+t_{\rm resource}}+2t_{\rm resource}/3\rceil\}.
\tag{32}
\]

Outward roots and Z_+ are used; if Z_+=1 the cap is M. The saved policy charges exp(−10)<1/17000 as resource failure. Expected C₀, an accepted-tail bound C_acc max_e C_e and a deterministic hard-attempt bound 2N max_e C_e are not interchangeable. These are resource accounts, not executed quantum-shot totals. [V11][E4]

### 8.4. G10 saved P5 comparison and degree-specific limits

The following values are literal display strings from [G10CSV]. Exact fractions, sufficient N, digital moment/range data, resource caps and all 17 rows are preserved in [G10EXACT]. “Intercept” is the expected two-axis native cost before adding K h_j.

| G10 m=5 arm | Saved native T intercept | Saved CX intercept | Saved native 1Q intercept | Saved K slope |
|---|---:|---:|---:|---:|
| Ordinary | 371085419.8 | 21904736.27 | 996749432.8 | 2651626 |
| Partial Return + tail | 275437398.9 | 16998102.83 | 731603228.6 | 2019804 |
| Closed P3 + tail | 284749629.9 | 17095836.74 | 734643183.8 | 1987402 |
| General Full | 273991325.7 | 16756376.23 | 692820724.4 | 1964236.421 |
| Closed P5 | 272239830.7 | 16649260.76 | 688391853.9 | 1951680 |
| Matched CTS | 387633441.2 | 10391732.24 | 968944321.5 | 3396808 |

This fixed comparison retains a positive T-intercept result for the RA arms relative to Ordinary. General Full does not attain a lower T intercept than Closed P5. CTS has a lower CX intercept in the same table; a T advantage does not establish dominance of the resource vector or of the CTS family. Preparation/readout are not assigned an observed price of zero.

The existing within-degree m=3/5/7 results retain positive registered Full-minus-closed T and call-slope differences against Closed P3, Closed P5 and Closed P5+tail respectively. At m=3/5 the ideal aggregated ensemble is shared but envelope/certified-budget and digital implementations differ. At m=7 Full and P5+ordinary tail are different ensembles. Existing small-angle changes can also change native synthesis prices: the saved m=7 root-child T counts are [136,138,140] for P5+tail and [140,142,144] for Full. This is a fixed price effect, not a unique causal attribution or a general law. [V11][E4]

The fixed-dictionary sampling-only bound and its restricted m=7 separation are retained in Appendix B. Changing proposal probabilities alone within that dictionary/price/error/confidence class cannot be credited with a benefit contrary to the saved bound. A new dictionary, synthesis sequence, precision or estimator lies outside the comparison class.

### 8.5. Counterexamples and exponential-task accounting

The existing W2a proposition P_U asks a universally quantified comparison between a certified Full upper bound U_m and low-degree-plus-tail benchmarks. Its fixed analytical certificate uses L=2, p=(1/2,1/2), x=1 and m=7 to show U₇−B_form,5⊕o,7>13/1024>0. P_U is therefore `DISPROVED`. U is an upper bound, not the true B_RA. This result neither proves that true aggregation is always inferior nor makes all RA resource benefits impossible. No new counterexample, rescue inequality or proof was sought here. [PU]

W2b retains N_F≤N_O under the same fixed source/root/precision path, while comparisons with Partial/Closed P3 require their stated moment, range, bias, residual-width, confidence and ceiling conditions. An ideal m=5 Full/P5 normalizer identity does not make two digital budgets equal. W2c supplies classical and quantum cost conditions, including (C1), rather than a general CPU/native victory. [W2B][W2C]

D1–D3 address a different, common **exponential** task. They use the explicit three-qubit synthetic provider, p=(1/5,3/10,1/2), total time τ=5/7, σ=+1, state 000, finite P5(−iτR/r)^r with r=1,2, six arms, H160/root256, η=ρ=10⁻¹² and primitive error 10⁻⁶. D1 includes Taylor/native/coefficient product error and sufficient trials; D2 adds common saved boundary cancellation without changing laws/budgets; D3 separates guarantee G0/G1 and price P0/P1 ledgers. Each ledger retains its own confidence family. [D1][D2][D3]

For example, D3's saved G1_P1 r=1 P3+tail and P5 entries have sufficient N of 1,103,489 and 1,083,653, but T intercepts 352,454,511.977 and 353,053,479.316 respectively. The smaller certified moment/N does not guarantee the lower T intercept at that fixed price. This is an existing example; no new common-policy aggregation or cross-study winner is inferred. Molecular geometry, DF parameters and all-round energy estimation are outside D1–D3's evidence scope.

## 9. Molecular H4 application

### 9.1. Fixed model, time and measured layer

The C1 series connects the RA generators to a reconstructed double-factorized model of **linear H4, separation 1.00 Å, STO-3G, DF rank 12, R1 generation-prefix L_D=3, thirteen basis definitions and 324 signed labels, eight system qubits**. The frozen experiment uses T=4/5, q_outer=1, r_RTE=4, m=5, σ=+1 and four tail microsteps of duration 1/5. There is no delta/time sweep. Its coefficient input is the adopted R1 dyadic model, not an unverified reuse of the historical PR-2 coefficients. [C1SUM][R1]

The precision settings are H_prob=2048, root=4096, η=ρ=angle τ=10⁻¹², angle root=256 with 64 series terms, and diagnostic seed 20261021 with separate method keys. These describe the fixed implementation; they do not supply an independently certified sufficient all-task N or a physical random-bit source. Each method generated 1024 whole paths, or 4096 microsteps. The measured circuit scope is `TAIL_ONLY_PLUS_SCALAR`. Deterministic halves and physical state preparation/readout are excluded, not priced at zero. [C1SUM][C1HAND]

### 9.2. Implementation evidence and model-binding limits

The R1 packet treats reconstructed binary64 coefficients as exact dyadic inputs and preserves exact positivity/normalization identities. It has different provenance from PR-2, including 135 differing coefficient hex values. A floating reconstruction comparison with the saved DF operator passed the existing numerical tolerance, while rigorous operator-norm/model binding remains `MODEL_BINDING_UNCLOSED`. Exactness of the adopted numbers is not exact physical-Hamiltonian certification. [C1SUM][R1BIND]

The adapter retains label signs, actual adjoints, reversed circuit-time order and controlled relative phase. Saved local matrix bytes were recovered for all thirteen bases (412 local operations). Separate checks covered six preselected controlled events and X/Y diagnostics. Recovery of those bytes and numerical checks of selected events do not certify every label/word/path or a uniform strict native provider. The whole-path interface separately preserves product order, weight products, pre-quantum zero and the all-trial denominator; a complete physical coherent signal estimator was not executed in this application. [C1CORR][C1SUM]

### 9.3. Exploratory path observations

Ordinary and General Full completed their substages in the old C1-5A execution. The old P5 worker failed after setup; its root cause remains unresolved. P5R is a separate source/authorization/marker execution that subsequently completed 1024 paths. Its completion does not rewrite the old failure as success or identify the old worker's exit cause. [C1OLDHAND][C1HAND]

The table quotes the stored summary fields at their saved precision. Each column is a 1024-path fixed diagnostic, not a certified common-accuracy trial budget. [H4ORD][H4FULL][H4P5][C1QUOTES]

| Saved H4 metric | Ordinary | General Full | Closed P5 (P5R) |
|---|---:|---:|---:|
| Accepted paths | 1024 | 1024 | 1024 |
| Valid zero paths | 0 | 0 | 0 |
| Logical instructions/path | 285.125 | 284.6494140625 | 284.7666015625 |
| Logical depth/path | 76.673828125 | 76.619140625 | 76.6455078125 |
| Empirical weight-square mean | 1.1141049662602474 | 1.1141049662602474 | 1.1105279232736538 |

Full and P5 each recorded 4073 degree-zero parents and 23 degree-two parents out of 4096 microsteps. Ordinary recorded 4070 and 26 raw degree-zero/two entries. Degree is the even-parent degree; all three methods retain four child rotations per path. The logical differences are small and no zero-path quantum-call avoidance was observed in this sample. A zero rejection count does not establish a zero population rejection probability. Different method PRNG keys do not provide a paired comparison. [C1CMP][C1QUOTES]

These observations support a narrow conclusion: nonempty parents are rare, limiting opportunities to shorten this logical word layer. They do not prove zero aggregation effect. The empty parent's coefficient, child angle and normalizer still incorporate formal returns (§7.4). Logical instructions may count an opaque controlled rotation as one instruction, so similar logical lengths do not certify equal native T/CX/1Q costs.

The empirical weight-square means in the table describe the saved paths. They are not certified population moments/ranges or sufficient-N estimates and are not evaluated here into new variances or costs. For an ideal ±1 measurement, Y²=1 links a **specified probability law's** weighted second moment to its weight-square expectation; it does not turn this finite sample into a certificate. No quantum shots were performed. [C1SUM]

### 9.4. Classical processing and incomparable implementation conditions

The following saved CPU-second displays retain their original stage boundaries. [C1COST]

| Saved CPU seconds | Ordinary, old run | General Full, old run | Closed P5, P5R |
|---|---:|---:|---:|
| Normalizer/completion setup | 0.026659721 | 0.203877525 | 283.993160344 |
| Sampling/checks/event evidence | 21.573830977 | 4410.007131644 | 400.634031349 |
| Angle query/evidence | 7.104218642 | 67.692002726 | 23.539041366 |
| Logical build | 22.452788567 | 21.202090197 | 20.856494351 |
| Whole measured hierarchy | 71.138041 | 4523.42039 | 998.71687 |

Full paid for 4096 local parent packets; its sampling stage includes finite-bit checking and evidence, not just Green arithmetic. Closed P5 paid for 104,977 normalizer/completion groups and a cumulative integer selector. The old runs used 21 fork workers per method with three concurrent method coordinators and worker-local caches. P5R used a single producer/global cache followed by 32 fresh-exec logical providers, job-local BLAS threads=1, additional monitoring, fsync and evidence rereading. Input restoration and timed hierarchy boundaries differ. [C1COST][C1CMP]

These values record substantial implementation burden; they are not optimized CPU rankings or a fundamental complexity separation. Comparable legal cache, selector, basis-reuse and preprocessing opportunities must be available to the baselines. No new amortization threshold, speedup ratio or matched-load benchmark is computed in this draft.

### 9.5. What this application does not establish

The 1024-path records do not provide a common-error/confidence sufficient N; uniform strict native T/CX/1Q; deterministic-half costs; physical preparation/readout; a completed coherent signal estimator; rigorous DF/exponential binding; or all-round RPE/QPE costs. H4 canonical P5 remains `UNCERTIFIED_EXPLORATORY`. Guarded event checks and Full preflight do not fill these gaps. The experiment is an auxiliary molecular implementation and cost-limit case, not the main demonstration of a general quantum advantage or an equal-accuracy total-resource defeat for RA. [C1SUM][C1HAND]

The adopted C1 pause is retained. This edition does not reopen additional molecular runs, native certificate closure or large resource comparisons in order to complete the manuscript.

## 10. Limitations and open questions

The ideal construction has a clear domain: explicit positive normalized rational inputs, odd finite degree, 0<x≤1 and Hermitian involutions, collecting only formal adjacent cancellations. General x>1 positivity, signed inputs and infinite-series guarantees are outside the retained statements. Finite first-moment equality is distinct from channel equality and from exact exponential dynamics.

General Full avoids acquiring the full aggregated normalizer **before generation**. It still needs the ordinary envelope, local coefficients, precision checks and controlled providers. This does not establish the fastest classical generator, a general input-access advantage, few distinct angles or inexpensive online synthesis. Closed P5 supplies a strong canonical specialization. Their ideal accepted-law equality does not imply digital or complete cost equality.

The synthetic positive results are limited to their registered providers, targets, precision, confidence and native-price models. CTS is strong in other resource coordinates, low-degree canonical baselines constrain Full, and P_U limits one upper-bound comparison. The fixed-dictionary lower does not bound every quantum estimator or algorithm. D1–D3's exponential examples retain different assumptions from the finite-target tables. No experiment here establishes a general best-to-best resource winner.

The H4 records establish selected adapter correctness and exploratory logical behavior for one exact adopted model/split/time/degree. They do not decide total native superiority or inferiority at common accuracy, general CPU performance, other molecules, or final energy estimation. Statistical PF/RPE/QPE error and deterministic product-formula coefficients are not combined into a new guarantee.

Finally, the existing prior-art audit does not settle independent originality, complete reduction, non-reducibility, priority or standalone article sufficiency. The local construction, its finite-precision specification and its cost trade-offs are the proposed unit of contribution. Assessing that unit and the article type is distinct from publishing this draft. The [open-claims record](../research/track_b_g6_v1_2_draft_review_integration_20261022/OPEN_CLAIMS_AND_LIMITATIONS.md) marks these questions `NOT_ESTABLISHED` or awaiting judgment. It does not create a new experiment/proof program or make additional runs a condition for the present document.

## 11. Conclusion

Return aggregation connects formal returns in a short-step finite Taylor operator to positive parent/child coefficients and weighted random-unitary events. The existing General Full generator realizes this representation through local Green queries and an ordinary envelope without first acquiring the full RA normalizer. Finite-bit correction, explicit bias/range conditions, supplied-precision preflight and classical accounting specify the limits of that implementation claim. Closed P5 is a low-degree specialization within the same family.

The preserved synthetic examples show conditional improvements over Ordinary and limits from stronger canonical/CTS comparisons. H4 demonstrates an application with small logical differences and non-negligible implementation costs. Together these support a bounded constructive account of methods and trade-offs. They establish neither general quantum speedup nor a final molecular total-cost conclusion. Independent originality, formal scientific adoption and submission remain for user/GPT assessment.

## 12. References

The following are the existing primary literature used by v1.1 and the prior-art audit. This edition verifies the seven specified bibliographic landing pages and reuses the existing relevant-section correspondences; it does not claim a fresh exhaustive literature search or complete proof/edition audit.

1. Kianna Wan, Mario Berta and Earl T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, Phys. Rev. Lett. 129, 030503 (2022), [arXiv:2110.12071v2][WBC]. Existing correspondence: Appendix C and Algorithm 2, order pairing and IID-label sampling.
2. K. Aomoto and Y. Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988), [primary article/PDF][AK]. Existing correspondence: §1, Eq. (1.6), Lemma 1.1; the Z₂ convention mapping is part of [E1].
3. Qi Zhao and Xiao Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021), [arXiv:2103.07988v2][ZY]. Existing correspondence: §4.2, modified Taylor and absorption.
4. Pei Zeng, Jinzhao Sun, Liang Jiang and Qi Zhao, *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, [arXiv:2212.04566v2][ZSJZ]. Existing correspondence: IV.A and Taylor/LCU pairing; not a claim that all G6 generation stages are the same.
5. Joseph Peetz, Scott E. Smart and Prineha Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, [arXiv:2407.21095v2][PSN]. Existing correspondence: Methods and Supplementary Notes 3/5, including layer sampling and the finite CTS comparator.
6. Davide Cugini, Touheed Anwar Atif and Yigit Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495v1][CAS]. Existing correspondence: Theorem 1, cost–moment accounting, ZeroFill/Discard. The bibliography retains the cited preprint version.
7. Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl and Aram Harrow, *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647v2][PR]. Existing RTE/involution context; no new detailed theorem correspondence is asserted here.

Fixed repository evidence is listed in Appendix D and the accompanying source inventory. Historical local checks remain source-bound evidence, rather than immutable CI or externally reproduced science.

## Appendix A. Existing preflight conditions and implementation accounting

This appendix retains F1's actual sufficient conditions rather than replacing them with a generic statement that enough bits exist. Let n_o=(m+1)/2, t_min=x^(m−1)/(m−1)!, p_min=min_i p_i, d_min=t_min p_min^(m−1)/2, Δ=2^(−H_root), and h=2^(−H_prob). F1 denotes H_root by K_root. Its normalized-interval predicate is

\[
0\le\zeta<1,\qquad
\frac h{q_{\min}}\le\frac{1-\zeta}{1+\zeta}-(1-\eta)(1+\zeta),\qquad
\frac h{q_{\min}}\le(1+\eta)(1-\zeta)-\frac{1+\zeta}{1-\zeta}.
\tag{A1}
\]

| F1 condition | Existing all-reachable-Full-parent sufficient specification |
|---|---|
| P0 domain | Exact rational inputs, positive normalized p, odd m, short step, legal helper calls, supplied positive integer precisions, stated η/ρ and fresh independent uniform-bit assumptions |
| P1 order | e=n_oΔ<1, a=Δ/t_min, ζ_o=(a+e)/(1−e), and predicate (A1) at (ζ_o,t_min/3) |
| P2 label/child | h≤ηp_min and h≤ηp_min/2; no child law is required for a branch without a child |
| P3 coefficient | Δ<d_min and Δ/2≤ρ(d_min−Δ) |
| P4 acceptance | Δ<1/4 and (2−η)Δ+h≤η/4 |
| P5 law/identity | Same source/input/policy and actual order/label/child laws, fixed bit mapping, identity handled separately |

These are the existing F1 inequalities with the root-precision symbol renamed, not new constants or a selector. They certify conditional mathematical guard sufficiency across unvisited as well as visited valid Full parents; they do not certify unrelated canonical groups, physical controlled access, RNG hardware or an entire failure-free run. F3 supplies a specific technical implementation and angle enclosure under these conditions. [F1][F3]

The fixed-source cost obligations in F2 include O(Lm²) coefficient preprocessing, order/label root and dyadic-law construction, preflight/budget work, every raw draw and check, coin-reject packet work, child-law sorting, actual-proposal weight construction, repeated child sums, angle work/requery, evidence and all-N output. Coefficient counts and peak bit storage are distinct; storing all outputs adds a separate N-dependent term. Those obligations are not replaced by measured Python seconds or by an implementation that has not been written. [F2]

## Appendix B. Existing fixed-dictionary sampling-only limitation

Fix positive coefficients a_e, unitaries V_e, native prices T_e≥0, the same synthesis sequences/precision, remaining width s and failure α. For any full-support proposal q_e>0 with Σ_eq_e≤1, let the missing mass be pre-quantum zero and use weight a_e/q_e. The existing formulas are

\[
M_2(q)=\sum_ea_e^2/q_e,\qquad C(q;h)=\sum_eq_e(T_e+h),\quad h\ge0.
\tag{33}
\]

\[
M_2(q)C(q;h)\ge\left(\sum_ea_e\sqrt{T_e+h}\right)^2.
\tag{34}
\]

\[
\sqrt{(T_i+h)(T_j+h)}\ge\sqrt{T_iT_j}+h,
\tag{35}
\]

\[
G(q;h)\ge\frac{4\log(2/\alpha)}{s^2}
\left[\left(\sum_ea_e\sqrt{T_e}\right)^2+h\left(\sum_ea_e\right)^2\right].
\tag{36}
\]

The Cauchy–Schwarz argument and cost-aware importance principle are established; this is the retained fixed-dictionary specialization. Formula (36) drops positive range and ceiling contributions from the same sufficient Bernstein policy and uses two axes. Zero-T events remain in coefficient mass and sampling. It is not an achievable optimal proposal, a minimum physical-shot count, or a lower bound for arbitrary estimators, dictionaries or native compilations. [CAS][E5][V11]

For the saved m=7 Full dictionary and the registered P5+tail comparator, the existing displayed difference is

\[
L_F(h)-G_{P5+tail}(h)\simeq5,415,782.443240-5,537.20852461h.
\tag{37}
\]

The saved exact difference is positive for 0≤h≤970. The previously recorded lower-bound endpoint near 978.0708852063 is not an actual winner crossover and does not show an improving proposal beyond that endpoint. The registered canonical Full comparison separately retains its positive T-intercept and call-slope differences. The saved m=5/7 literal CTS-dictionary lower comparisons also retain their restricted class, rather than becoming a statement about the entire CTS family. No root/log certificate, lower bound, optimum or crossover was regenerated for this edition. [V11][E4][E5]

## Appendix C. Existing implementation evidence and preservation boundaries

The G6 mathematical arguments, finite fixtures, implementation checks, native operator diagnostics, saved-output audits and external scientific reproduction are different evidence types. For example, the existing G6 sixteen fixtures are finite checks; F3's 63 artificial tests document implementation behavior. G9/G10's phase/order diagnostics and source-bound saved audits do not independently reproduce native synthesis or prove the general construction. This edition reruns none of them. [E1][F3][G9][E4]

The earlier 16,896-trial fixed P5 pilot connected G7/F3 event generation, checks and retention. It used p=(1/5,3/10,1/2), x=5/7,m=5,σ=+1,H160/root256,η=ρ=10⁻¹², ε_axis=1/200, primitive error 10⁻⁶, α_axis=1/320, familywise 1/20, angle τ=10⁻⁶ and K_phi=J=32 in a math-only configuration. Its 16,896 scheduled trials were classical pipeline records, not quantum measurement shots. Its original `science_correctness_certified=false`, finite-prefix limits, diagnostic-bitstream assumptions and CPU/cache/order confounding remain. Full-N, general physical-provider success and final molecular energy cost do not follow from that pilot. The prior full cost/retention tables remain at their fixed source rather than being recomputed or expanded here. [PILOT][V11]

BF/BM/SP, parent allocation, DF correction and M1-S retain their separate targets and existing closures. This G6 draft does not combine their results into a new total improvement or revise their adoption/STOP records. No old failed run, consumed grant, marker or raw result is rewritten by this manuscript.

## Appendix D. Fixed evidence and data access

| Evidence | Fixed record and locator | Role |
|---|---|---|
| Prior manuscript | [V11], commit 243eea8e840f3c472753f7935e510391626fd325 | Definitions and unchanged numbered equations (1)–(37), original guarantees and limits |
| G6 ideal / finite-bit audit | [E1], [E2] | Existing proofs, access assumptions and finite-bit bounds |
| F1/F2/F3 | [F1], [F2], [F3] | Supplied-precision conditions, bit cost and implementation scope |
| Construction/prior-art correspondence | [MATRIX], [PRIORPIPE], [PRIORCAND] | Claim/proof/source correspondence and unresolved independence |
| Closed P5 | [E3], [P5SRC], [G9] | Closed coefficients, groups, canonical law and original matched-native result |
| G7 / G10 | [G7], [E4], [G10CSV], [G10EXACT] | Original policies; displayed and exact saved resource values |
| D1–D3 / P_U / W2 | [D1], [D2], [D3], [PU], [W2B], [W2C] | Separate exponential-task evidence, negative result and total-cost conditions |
| H4 original comparison | [C1CMP], commit bb9059bacb697e1105e63c1b3826b0c34886ad15; [H4ORD], [H4FULL] at b022f2ea421649015a4c21ea3c904fe619d0c198; [H4P5] at bb9059bacb697e1105e63c1b3826b0c34886ad15 | Literal summary values; raw trajectories were not reanalysed |
| C1 synthesis | [C1SUM], [C1QUOTES], commit 185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e | Previously completed synthesis and quotation provenance; left unchanged |
| Model / correctness / CPU boundaries | [R1BIND], [C1CORR], [C1COST] | Numerical binding status, selected correctness and incomparable implementation boundaries |

All repository links resolve to full fixed commits. The accompanying [claim matrix](../research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md) maps important draft claims to formulas and saved evidence. The source inventory and public-byte verification document the actual retrieval scope; they do not claim that unavailable old-server private files were inspected. Private review originals are not included. No raw trajectory archive was opened, no scientific module was imported/executed, and no new scientific quantities were calculated to prepare this draft.

<!-- Fixed references; private review originals are not public dependencies. -->

[V11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[E1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[E2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[E3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[WBC]: https://arxiv.org/abs/2110.12071v2
[AK]: https://www.numdam.org/item/AIF_1988__38_1_59_0/
[PSN]: https://arxiv.org/abs/2407.21095v2
[CAS]: https://arxiv.org/abs/2603.13495v1
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/README.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[E4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[C1SUM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[ZY]: https://arxiv.org/abs/2103.07988v2
[ZSJZ]: https://arxiv.org/abs/2212.04566v2
[PRIORPIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/generation_pipeline_alignment.md
[PR]: https://arxiv.org/abs/2503.05647v2
[PRIORCAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/remaining_independent_claim_candidates.md
[ASOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df90ed8c8e847f8ea1861d9761b9515e7a895916/docs/research/track_b_partial_sum_query_sampler_audit_20261020/README.md
[BSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f823e909cb7049a612f1bf2c247f3bbe15fc184d/docs/research/track_b_coefficient_type_canonical_audit_20261020/README.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[MATRIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/handoff.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/handoff.md
[P5SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g9_p5.py
[C1COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/RESOURCE_REPORT.md
[G10CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/artifacts/track_b_g10_degree_result/2026-10-10/v3/resource_rows_display_v3.csv
[G10EXACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/README.md
[C1HAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/handoff.md
[R1BIND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/model_binding.json
[C1CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md
[C1OLDHAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/handoff.md
[H4ORD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/ordinary/summary.json
[H4FULL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/full_return/summary.json
[H4P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json
[C1QUOTES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/quoted_resource_values.json
[C1CMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.md
[E5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_saved.py
[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
