# Exact DF-tail extraction

`trotterlib.df_rte_tail` converts each selected native squared DF fragment into
the Hermitian involutions required by finite RTE. It does not treat an entire
multi-eigenvalue DF block as one involution.

For $n_k=(I-Z_k)/2$,

$$
\lambda\left(\sum_k\eta_kn_k\right)^2
=c_I I+\sum_k c_kZ_k+\sum_{k<j}c_{kj}Z_kZ_j,
$$

where

$$
c_I=\frac{\lambda}{2}\left(\sum_k\eta_k^2+
\sum_{k<j}\eta_k\eta_j\right),
$$

$$
c_k=-\frac{\lambda}{2}\eta_k\sum_j\eta_j,
\qquad
c_{kj}=\frac{\lambda}{2}\eta_k\eta_j.
$$

Components are ordered by fragment, support size, and support indices.
Identical support is aggregated only within the same fragment/basis. Equal
Z/ZZ support from distinct orbital bases remains distinct.

Two identity policies are available:

- `faithful_identity_in_tail`: identity coefficients participate in RTE L1 and
  event probabilities;
- `extract_identity_phase`: identity coefficients are summed into an exact
  deterministic phase baseline and excluded from RTE L1.

Both reconstruct the same retained Hamiltonian. In a controlled evolution the
extracted phase must act relatively on the ancilla; it cannot be discarded as
a system global phase.

`rte_lambda_r` is the actual retained coefficient L1. The existing DF fragment
ranking sum is stored separately as `ranking_proxy_lambda_r` and is never used
as the RTE normalization.

The threshold default is zero. A nonzero threshold records dropped count,
dropped L1, operator error bound, and policy. Hashing covers canonical
pre-threshold components plus identity and threshold policy.
