"""New integration version; retain the published Stage3 connection unchanged."""
from fractions import Fraction
import math
from . import c1_stage2_native as native
from .c1_stage3_connection import FullFrameProvider


def partial_kernel(plan, step_time, events, *, rte_config, builder=None, provider=None):
    """Existing native/event emission, validating the unsigned stored angle.

    RTEEvent.rotation_angle is atan(tau/(n+1)); its unsigned_rotation_angle
    property already applies the coefficient sign. The published bridge's
    sign-dependent validation incorrectly rejected negative atoms.
    """
    from qiskit import QuantumCircuit
    from .pr_dictionary_validation.circuits import build_event_sequence
    events=tuple(events)
    if not plan.atoms or rte_config is None:
        raise ValueError('A nonempty registered PR tail is required')
    if len(events)!=rte_config.rte_steps or not math.isclose(float(Fraction(step_time)),
            rte_config.evolution_time,rel_tol=1e-14,abs_tol=0):
        raise ValueError('Finite RTE occurrence/time mismatch')
    for event in events:
        expected=math.atan(rte_config.dimensionless_step_time/(event.taylor_order+1))
        if not math.isclose(event.rotation_angle,expected,rel_tol=1e-14,abs_tol=1e-15):
            raise ValueError('Finite RTE angle mismatch')
    builder=builder or native.NativeBuilder();provider=provider or FullFrameProvider(plan)
    half=tuple((i,Fraction(step_time)/2) for i in range(plan.ld+1))
    qc=QuantumCircuit(plan.model.modes+1)
    qc.compose(builder.sequence(plan.model,half,optimized=True,scalar_time=0),inplace=True)
    qc.compose(build_event_sequence(events,plan.atoms,provider,optimized=True),inplace=True)
    qc.compose(builder.sequence(plan.model,tuple(reversed(half)),optimized=True,scalar_time=0),inplace=True)
    qc.p(-(plan.model.constant+plan.extracted_identity)*float(Fraction(step_time)),plan.model.modes)
    return qc
