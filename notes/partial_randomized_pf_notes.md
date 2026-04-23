# Partial Randomized PF Notes

## Current approximations

- `A_p(L_D)` is approximated by the number of Pauli evolutions in one deterministic PF step on `H_D`.
- `B` is kept as a single user-supplied constant.
- `C_gs^(p)(L_D)` is fit on `H_D` only, using the exact eigenstate assumption.
- The perturbative fit uses the same notebook-style time windows as the existing deterministic study.
- The Hamiltonian is split at the Pauli-term level; grouping and UWC are intentionally outside the main path.

## Not yet addressed

- Hardware resource estimation.
- A rigorous randomized-side prefactor beyond the constant `B`.
- Reusing precomputed grouped artifacts for `L_D`-dependent `C_gs`.
- Larger-system performance tuning for dense `L_D` scans.
