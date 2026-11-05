"""Dormant input dispatch for the integrated H4 1.15 Angstrom recipe test.

Import/specification are metadata only. Calling generate() with real chemistry
is scientific work and requires separate permission for the integrated plan.
No old H4 arrays, PF values, or fitted J/U are used here.
"""


def specification():
    return dict(atoms=4, norb=4, modes=8, Nalpha=2, Nbeta=2,
                distance_angstrom=1.15, basis='sto-3g', charge=0, spin=0,
                sector_dimension=36, full_dimension=256, controlled_dimension=512,
                tensor_convention='chemist_pqrs_original_h_g',
                orbital_frame='canonical_RHF_MO', kappa='original_g_corrected',
                scalar='nuclear_once', rank=8, LD=2)


def generate(call, *, factory=None):
    """Delegate to the existing RHF/ao2mo engine, without a second implementation.

    Two instrumented scientific units: scf and spatial_integrals. A synthetic
    factory may be injected for technical tests; it must account its own calls.
    """
    from .c1_molecule_extension import integrals_from_specification
    return integrals_from_specification(specification(), call, factory=factory)
