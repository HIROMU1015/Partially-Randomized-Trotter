"""Explicit convention-B implementation candidate; no adoption or input I/O.

Only caller-supplied arrays/operators are used. This module does not qualify
the saved H4 input, replace an old result, or build circuits/states/signals.
"""
from dataclasses import dataclass
import math

from .o1_m2_residual_adapter import SpatialIntegrals, WholeCoreResidual, _pauli_module


def _packet(h, g, scalar):
    import numpy as np
    if any(np.iscomplexobj(x) for x in (h,g,scalar)):
        raise ValueError("Finite real spatial packet required; no complex cast")
    if np.asarray(scalar).shape != ():
        raise ValueError("Scalar nuclear constant required")
    h, g = np.array(h, dtype=np.float64, copy=True), np.array(g, dtype=np.float64, copy=True)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or not len(h)
            or g.shape != (len(h),)*4 or not np.isfinite(h).all()
            or not np.isfinite(g).all() or not math.isfinite(float(scalar))):
        raise ValueError("Finite real spatial packet required")
    h.setflags(write=False); g.setflags(write=False)
    return SpatialIntegrals(h, g, float(scalar))


def hermitian_packet(packet):
    """P on the operator adjoint, not an arbitrary ERI permutation average."""
    import numpy as np
    if any(np.iscomplexobj(x) for x in (packet.h, packet.g, packet.nuclear_constant)):
        raise ValueError("This saved-real-integral convention does not accept complex spatial inputs")
    p = _packet(packet.h, packet.g, packet.nuclear_constant)
    return _packet((p.h+p.h.T)/2, (p.g+p.g.transpose(1,0,3,2))/2, p.nuclear_constant)


def packet_scale(packet):
    """CAR monomial absolute-coefficient bound; every monomial norm <= 1."""
    import numpy as np
    return abs(packet.nuclear_constant)+2*float(np.sum(abs(packet.h)))+2*float(np.sum(abs(packet.g)))


@dataclass(frozen=True)
class HermitianBCandidate:
    raw: WholeCoreResidual
    physical: SpatialIntegrals
    core: SpatialIntegrals
    residual: SpatialIntegrals
    projected_raw_residual: SpatialIntegrals
    convention: str = "B_IMPLEMENTATION_CANDIDATE_NOT_ADOPTED"

    @property
    def sector(self):
        return self.raw.sector

    def pauli_parts(self):
        return tuple(strict_pauli_map(p, model_id="M2_B_"+name)
                     for name,p in (("physical",self.physical),("core",self.core),("residual",self.residual)))


def derive_hermitian_b(raw):
    """Canonical order: P(H), P(C), then binary64 subtraction for R.

    P(raw R) is retained as a separate formation diagnostic. It is never
    substituted for the canonical R or claimed bitwise identical to it.
    The raw general-K model, original-g kappa and saved frame arrays are kept.
    """
    physical, core = hermitian_packet(raw.physical), hermitian_packet(raw.core)
    if physical.h.shape != core.h.shape or physical.nuclear_constant != core.nuclear_constant:
        raise ValueError("Shared physical/core scalar and shape required")
    residual = _packet(physical.h-core.h, physical.g-core.g, 0.)
    return HermitianBCandidate(raw, physical, core, residual, hermitian_packet(raw.residual))


def complex_pauli_coefficients(packet):
    """Reuse the unit-JW collector but retain all imaginary coefficients."""
    from openfermion import InteractionOperator
    from .c1_stage1 import spin_lift_unthresholded
    from .o1_m2_cutoff_free_pauli import _unit_jw
    packet = _packet(packet.h,packet.g,packet.nuclear_constant)
    one,two = spin_lift_unthresholded(packet.h,packet.g)
    interaction = InteractionOperator(packet.nuclear_constant,one,two/2)
    grouped = {}
    for term in interaction:
        coefficient = interaction[term]
        if coefficient == 0: continue
        for word,weight in _unit_jw(term):
            label = ["I"]*len(one)
            for q,axis in word: label[len(one)-1-q] = axis
            grouped.setdefault("".join(label),[]).append(complex(coefficient*weight))
    return {label:complex(math.fsum(v.real for v in values),math.fsum(v.imag for v in values))
            for label,values in grouped.items()}


def strict_pauli_map(packet, *, model_id):
    """No implicit 1e-12 imaginary projection after explicit convention B.

    Unit JW coefficients are powers of two. For exactly adjoint-paired real
    packet entries their imaginary contributions cancel exactly in fsum.
    Any nonzero imaginary remainder is rejected, not silently thresholded.
    """
    from .o1_m2_cutoff_free_pauli import map_integrals
    coefficients = complex_pauli_coefficients(packet)
    imaginary = [abs(c.imag) for c in coefficients.values()]
    if any(imaginary):
        raise ValueError("NONZERO_IMAGINARY_PAULI_COEFFICIENT: explicit Hermitian convention required")
    return map_integrals(packet,model_id=model_id,metadata=dict(
        convention="B_IMPLEMENTATION_CANDIDATE_NOT_ADOPTED",imaginary_policy="reject_any_nonzero",
        imaginary_coefficient_max=max(imaginary,default=0.),sector_projection=False))


@dataclass(frozen=True)
class RoundoffPlan:
    """Operation counts fixed by a future runner before molecular evaluation.

    Counts must cover its actual real arithmetic/contraction paths. This is
    an engineering allowance, not a certified interval enclosure. No count
    is inferred from an observed operator discrepancy.
    """
    tensor_contraction_operations: int
    kappa_operations: int
    core_one_body_operations: int
    mapper_operations: int
    reference_operations: int
    guard: float = 8.
    maximum_relative_mapper_allowance: float = .01

    def __post_init__(self):
        for name in ("tensor_contraction_operations","kappa_operations","core_one_body_operations",
                     "mapper_operations","reference_operations"):
            value = getattr(self,name)
            if type(value) is not int or value < 1 or value*2.**-53 >= 1:
                raise ValueError("Positive finite binary64 operation counts required")
        if (not math.isfinite(self.guard) or self.guard < 1
                or not 0 < self.maximum_relative_mapper_allowance < 1):
            raise ValueError("Invalid guard or residual resolution fraction")


def predeclared_budgets(candidate, plan):
    """Separate model projection, upstream formation and residual mapping.

    Positive absolute tensor products bound cancellation in the original
    J/U contraction. Input float values are treated as exact saved numbers;
    no SCF/integral-generation or half-ULP storage guarantee is invented.
    """
    import numpy as np
    from .c1_stage1 import gamma
    raw = candidate.raw
    J = np.asarray([f.spatial_J for f in raw.general_core.frames])
    U = np.asarray([f.spatial_orbital_rotation for f in raw.general_core.frames])
    absolute_fit = np.einsum('kpi,kqi,kij,krj,ksj->pqrs',abs(U),abs(U),abs(J),abs(U),abs(U),optimize=False)
    h_abs = abs(raw.physical.h)+.5*np.einsum('prrq->pq',abs(raw.physical.g))
    fit_l1 = float(np.sum(absolute_fit))
    kappa_l1 = float(np.sum(h_abs))
    core_h_scale = kappa_l1+.5*float(np.sum(np.einsum('prrq->pq',absolute_fit)))
    # Tensor error affects quartic (factor2) and contracted one-body (<=factor1).
    upstream = plan.guard*(3*gamma(plan.tensor_contraction_operations)*fit_l1
        +2*gamma(plan.kappa_operations)*kappa_l1
        +2*gamma(plan.core_one_body_operations)*core_h_scale)
    hp,cp,rp = (packet_scale(p) for p in (candidate.physical,candidate.core,candidate.residual))
    projection = plan.guard*gamma(2)*(packet_scale(raw.physical)+packet_scale(raw.core))
    subtraction = plan.guard*gamma(1)*(hp+cp)
    mapping = plan.guard*(gamma(plan.mapper_operations)+gamma(plan.reference_operations))*rp
    shifts = {}
    for name,before,after in (("physical",raw.physical,candidate.physical),
                              ("core",raw.core,candidate.core),
                              ("residual",raw.residual,candidate.projected_raw_residual)):
        shifts[name] = packet_scale(_packet(after.h-before.h,after.g-before.g,
                                            after.nuclear_constant-before.nuclear_constant))
    route_difference = packet_scale(_packet(candidate.residual.h-candidate.projected_raw_residual.h,
        candidate.residual.g-candidate.projected_raw_residual.g,0.))
    return dict(input_semantics="exact_saved_binary64_values; physical generation accuracy unassessed",
        projection_changes=shifts,projection_or_subtraction_allowance=projection+subtraction,
        upstream_core_formation_allowance=upstream,
        formation_allowance=upstream+projection+subtraction,
        canonical_vs_projected_raw_residual_bound=route_difference,
        residual_mapper_allowance=mapping,residual_scale=rp,
        max_relative_mapper_allowance=plan.maximum_relative_mapper_allowance,
        norm="operator max-entry <= CAR coefficient L1 bound",rigorous_enclosure=False)


def assess_residual_mapper(candidate, reference, mapped, plan):
    """Compare canonical R itself. Never add the upstream/full-H allowance."""
    import numpy as np
    reference,mapped = np.asarray(reference),np.asarray(mapped)
    d = 1 << (2*len(candidate.residual.h))
    if (reference.shape != (d,d) or mapped.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(mapped).all()):
        raise ValueError("Full-Fock finite canonical residual matrices required")
    budget = predeclared_budgets(candidate,plan)
    tolerance = budget['residual_mapper_allowance']
    size = float(np.max(abs(reference)))
    error = float(np.max(abs(mapped-reference)))
    ref_hermiticity = float(np.max(abs(reference-reference.conj().T)))
    map_hermiticity = float(np.max(abs(mapped-mapped.conj().T)))
    resolution = ((size > 0 and tolerance < plan.maximum_relative_mapper_allowance*size)
                  or (size == 0 and budget['residual_scale'] == 0))
    matches = error <= tolerance and max(ref_hermiticity,map_hermiticity) <= 2*tolerance
    status = "NOT_QUALIFIED" if not resolution else ("PASS" if matches else "FAIL")
    return dict(status=status,max_entry=error,reference_max_entry=size,tolerance=tolerance,
        reference_hermiticity_defect=ref_hermiticity,mapped_hermiticity_defect=map_hermiticity,
        zero_mapper_error=size,reverse_sign_error=2*size,resolution_qualified=resolution,
        upstream_allowance_used=False,physical_model_qualified=False,
        convention=candidate.convention,budgets=budget)


def assess_formation(candidate, plan, *, closure_error, input_projection_limits):
    """External independent closure error and explicit model-shift budgets.

    Model-shift limits are caller conditions, not inferred from observed shifts.
    Unknown input/model precision remains unqualified rather than a fake PASS.
    """
    if not math.isfinite(closure_error) or closure_error < 0:
        raise ValueError("Finite nonnegative independent closure error required")
    b = predeclared_budgets(candidate,plan)
    if input_projection_limits is None:
        model_status = "UNASSESSED"
    else:
        if set(input_projection_limits) != {'physical','core','residual'}:
            raise ValueError("Explicit physical/core/residual model-change limits required")
        if any(not math.isfinite(v) or v < 0 for v in input_projection_limits.values()):
            raise ValueError("Finite nonnegative model-change limits required")
        model_status = 'PASS' if all(b['projection_changes'][k] <= input_projection_limits[k]
                                  for k in input_projection_limits) else 'FAIL'
    return dict(formation_status='PASS' if closure_error <= b['formation_allowance'] else 'FAIL',
                model_change_status=model_status,closure_error=closure_error,budgets=b,
                physical_generation_accuracy_qualified=False)
