"""Local M2 JW collection without small-coefficient SymbolicOperator sums.

Only unit-coefficient monomials enter OpenFermion. Their quartic JW weights
are exact binary fractions >= 1/16; physical coefficients never enter its
thresholded arithmetic. Final collection uses existing PauliHamiltonian.fsum
and exact-zero removal. No shared-library or global tolerance is modified.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def _unit_jw(term):
    from openfermion import FermionOperator, jordan_wigner
    return tuple(jordan_wigner(FermionOperator(term, 1.)).terms.items())


def map_integrals(packet, *, model_id, metadata=None):
    """Map an already validated real spatial-integral packet, including I.

    Reuses the source convention in c1_stage1: interleaved spin,
    eri.transpose(0,2,3,1), and 1/2 on the normal-ordered quartic tensor.
    This preserves nonzero binary64 contributions, not symbolic exactness.
    """
    from openfermion import InteractionOperator
    from .c1_stage1 import spin_lift_unthresholded
    from .o1_m2_residual_adapter import _pauli_module

    one, two = spin_lift_unthresholded(packet.h, packet.g)
    interaction = InteractionOperator(packet.nuclear_constant, one, two / 2)
    n = len(one)
    contributions = []
    count = 0
    for term in interaction:
        coefficient = interaction[term]
        if coefficient == 0:
            continue
        count += 1
        for word, unit_weight in _unit_jw(term):
            label = ["I"] * n
            for q, axis in word:
                label[n - 1 - q] = axis
            contributions.append(("".join(label), coefficient * unit_weight))
    md = dict(metadata or {})
    md.update(mapping="unit_monomial_JW_then_binary64_fsum", coefficient_cutoff=0.,
              nonzero_input_monomials=count, spin_order="2*p+alpha(0)/beta(1)",
              two_body_prefactor=.5, input_eri="chemist_(pq|rs)",
              nuclear_repulsion=float(packet.nuclear_constant))
    return _pauli_module().PauliHamiltonian.from_terms(
        contributions, num_qubits=n, model_id=model_id, metadata=md)
