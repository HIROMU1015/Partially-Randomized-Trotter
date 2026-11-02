"""Inactive number-conserving JW Gaussian candidate; no dense generic8q gate.

The proposed circuit represents Gamma(U), with vacuum phase exactly one.
Changing synthesis changes resource metrics even when the unitary agrees.
No runtime module imports this candidate and no launch approval is emitted.
"""
import math
from .identity import require

POLICY=dict(schema_version='h4-gaussian-structure-proposal-v1',max_modes=8,
    maximum_two_mode_rotations=28,maximum_single_mode_phases=8,
    numerical_atol=1e-12,approximate_pruning=False,
    approved=False,runtime_authorization=False,production_wiring_present=False,
    cost_semantics_change=True,compiled_gate_metrics_equivalent_to_dense=False)


def givens_plan(rotation):
    """Adjacent complex QR elimination, inverse factors in circuit order.

    For column j eliminate rows bottom-up with SU(2) rotations on adjacent
    modes. If Gk ... G1 U = D, apply D then Gk^dagger ... G1^dagger.
    Exact zero is skipped; no small-angle/near-zero truncation is performed.
    """
    import numpy as np
    raw=np.asarray(rotation)
    require(raw.ndim==2 and raw.shape[0]==raw.shape[1] and 1<=raw.shape[0]<=8 and
            raw.dtype.kind in 'fciub' and np.isfinite(raw).all(),'finite square Gaussian rotation1..8')
    U=np.asarray(raw,dtype=np.complex128);n=len(U);atol=POLICY['numerical_atol']
    require(np.max(np.abs(U.conj().T@U-np.eye(n)))<=atol,'unitary orbital rotation')
    current=U.copy();eliminations=[]
    for column in range(n-1):
        for lower in range(n-1,column,-1):
            upper=lower-1;a,b=current[upper,column],current[lower,column]
            if b==0:continue
            magnitude=math.hypot(abs(a),abs(b));require(math.isfinite(magnitude) and magnitude>0,'Givens norm')
            G=np.array([[a.conjugate()/magnitude,b.conjugate()/magnitude],[-b/magnitude,a/magnitude]],dtype=complex)
            current[[upper,lower],:]=G@current[[upper,lower],:]
            inverse=G.conj().T.copy();inverse.setflags(write=False)
            eliminations.append((upper,lower,inverse))
    diagonal=np.diag(current)
    require(np.max(np.abs(current-np.diag(diagonal)))<=atol and
            np.max(np.abs(np.abs(diagonal)-1))<=atol,'Gaussian QR residual')
    phases=tuple(float(np.angle(value)) for value in diagonal)
    plan=dict(schema_version=POLICY['schema_version'],modes=n,phases=phases,
              rotations=tuple(reversed(eliminations)),vacuum_phase=1.0)
    require(np.max(np.abs(orbital_matrix(plan)-U))<=atol,'Gaussian reconstruction')
    return plan


def orbital_matrix(plan):
    import numpy as np
    value=np.diag(np.exp(1j*np.asarray(plan['phases'])))
    for upper,lower,G in plan['rotations']:value[[upper,lower],:]=G@value[[upper,lower],:]
    return value


def fermionic_pair_matrix(G):
    """Little-endian Fock order00,01,10,11; include two-particle determinant."""
    import numpy as np
    value=np.zeros((4,4),dtype=complex);value[0,0]=1
    value[1:3,1:3]=G;value[3,3]=np.linalg.det(G)
    return value


def build_basis(rotation):
    """Future proposal builder only. Qiskit is imported lazily, never compiled."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    plan=givens_plan(rotation);qc=QuantumCircuit(plan['modes'])
    for mode,phase in enumerate(plan['phases']):
        if phase!=0.0:qc.p(phase,mode)
    for upper,lower,G in plan['rotations']:
        qc.append(UnitaryGate(fermionic_pair_matrix(G)),[upper,lower])
    return qc
