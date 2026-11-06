"""B-relative core omission bounds; supplied arrays only, no input I/O.

Exact Fraction coefficient sums implement the existing CAR monomial L1 bound.
No operator eigensolve, PF, state, circuit, shot or physical-H adoption occurs.
"""
from fractions import Fraction
import math

from . import o1_m2_hermitian_b as b
from .o1_m2_residual_adapter import SpatialIntegrals


def _adjoint(packet):
    return SpatialIntegrals(packet.h.T,packet.g.transpose(1,0,3,2),packet.nuclear_constant)


def _car_l1(terms):
    """Exact bound |c|+2 sum|h|+2 sum|g| for a packet linear combination.

    There are two spin copies of h, four spin copies of g with prefactor 1/2.
    Every CAR monomial has operator norm <=1, including non-Hermitian ones.
    Scalar cancellations and all small nonzero terms are retained exactly.
    """
    import numpy as np
    terms=tuple((Fraction(weight),b._packet(packet.h,packet.g,packet.nuclear_constant))
                for weight,packet in terms)
    if not terms: return Fraction()
    n=len(terms[0][1].h)
    if any(len(packet.h)!=n for _,packet in terms): raise ValueError('Packet dimensions differ')
    exact=lambda value:Fraction.from_float(float(value))
    total=abs(sum((w*exact(p.nuclear_constant) for w,p in terms),Fraction()))
    for name in ('h','g'):
        shape=getattr(terms[0][1],name).shape
        for index in np.ndindex(shape):
            total+=2*abs(sum((w*exact(getattr(p,name)[index]) for w,p in terms),Fraction()))
    return total


def _record(value):
    rounded=float(value)
    if not math.isfinite(rounded): raise OverflowError('Finite bound representation required')
    if Fraction.from_float(rounded)<value: rounded=math.nextafter(rounded,math.inf)
    if not math.isfinite(rounded): raise OverflowError('Finite upward bound required')
    return dict(exact_fraction=str(value),upper_bound=rounded,
        method='exact binary64 coefficient CAR monomial L1; binary64 rounded upward')


def core_only_bounds(raw, *, hh_omission_budget=None):
    """Bound dropping the complete B residual relative to B H_H, never H_phys.

    Exact P references preserve the mathematical B definition on saved bits.
    Existing B adapter outputs are kept separately, including projection and
    residual-formation errors; H_H=C_H+R_H is not silently made bitwise exact.
    """
    import numpy as np
    budget=None if hh_omission_budget is None else float(hh_omission_budget)
    if budget is not None and (not math.isfinite(budget) or budget<=0):
        raise ValueError('Positive finite explicit H_H budget required')
    candidate=b.derive_hermitian_b(raw)
    for packet in (candidate.physical,candidate.core,candidate.residual):
        if (not np.array_equal(packet.h,packet.h.T)
                or not np.array_equal(packet.g,packet.g.transpose(1,0,3,2))):
            raise ValueError('B output must have exact coefficient adjoint symmetry')
    half=Fraction(1,2)
    H_ref=((half,raw.physical),(half,_adjoint(raw.physical)))
    C_ref=((half,raw.core),(half,_adjoint(raw.core)))
    R_ref=H_ref+tuple((-w,p) for w,p in C_ref)
    reference_bound=_car_l1(R_ref)
    bounds=dict(
        reference_R_H=_record(reference_bound),
        formed_R_H=_record(_car_l1(((1,candidate.residual),))),
        formed_H_H_minus_C_H=_record(_car_l1(((1,candidate.physical),(-1,candidate.core)))),
        formed_H_H_minus_C_H_minus_R_H=_record(_car_l1(((1,candidate.physical),(-1,candidate.core),(-1,candidate.residual)))),
        formed_R_H_minus_reference_R_H=_record(_car_l1(((1,candidate.residual),)+tuple((-w,p) for w,p in R_ref))),
        formed_H_H_minus_reference_H_H=_record(_car_l1(((1,candidate.physical),)+tuple((-w,p) for w,p in H_ref))),
        formed_C_H_minus_reference_C_H=_record(_car_l1(((1,candidate.core),)+tuple((-w,p) for w,p in C_ref))),
        reference_H_H_minus_raw_H0=_record(_car_l1(H_ref+((-1,raw.physical),))))
    if hh_omission_budget is None:
        budget_status='DEFERRED'
    else:
        budget_status=('WITHIN_DECLARED_B_REFERENCE_BUDGET' if reference_bound<=Fraction.from_float(budget)
                       else 'BOUND_INCONCLUSIVE')
    return dict(scope='CORE_OMISSION_RELATIVE_TO_B_H_H_ONLY',
        reference='exact P on saved binary64 raw H0/C0 packet operators; not physical H_phys',
        bounds=bounds,energy_bound_reference_H_H_vs_C_H=bounds['reference_R_H'],
        energy_bound_formed_B_packets=bounds['formed_H_H_minus_C_H'],
        energy_statement='|E0_S(H_H)-E0_S(C_H)| <= bound for common Hermitian invariant sector S',
        full_fock_bound=True,sector=raw.sector,
        sector_condition='normal-ordered spatial spin lift conserves Nalpha/Nbeta algebraically; native circuit not qualified',
        hh_omission_budget=hh_omission_budget,hh_budget_status=budget_status,
        physical_target_status='DEFERRED',physical_model_status='UNASSESSED',
        physical_energy_bound=None,physical_model_tolerance=None,
        native_core_and_exact_JU_contraction_status='DEFERRED',
        raw_H0_change_is_physical_energy_bound=False,
        state_energy_simulation_and_RPE_status='DEFERRED',
        source_B_candidate_unchanged=True,formal_physical_redefinition=False)
