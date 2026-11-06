"""Explicit sign/identity/negative-time checks required by A1/A2."""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .pr_reference_interference_pilot import OUT,common,save_json,vacuum_check,vacuum_phase
from trotterlib.rte import RTEComponent,_make_event,finite_rte_distribution
from trotterlib.df_rte_circuit import DFRTEIdentityCircuitSpec,DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_tail import DFBasisRegistry
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder


def check():
    prep=common()['preps'][3].rte_preparation
    components=prep.symbolic_tail.components
    pos=next(i for i,c in enumerate(components) if c.coefficient_sign==1)
    neg=next(i for i,c in enumerate(components) if c.coefficient_sign==-1)
    rows=[]
    for identity in (False,True):
        for sign in (1,-1):
            if identity:
                comps=(RTEComponent('I',1.,1.,sign,is_identity=True,diagonal_pauli_support=()),)
                specs=(DFRTEIdentityCircuitSpec('I',1.,sign,2),)
                registry=DFBasisRegistry();rot=0;n=2
            else:
                comps=components;specs=prep.component_specs;registry=prep.basis_registry
                rot=pos if sign==1 else neg;n=8
            for timesign in (1,-1):
                dist=finite_rte_distribution(timesign*.02,2)
                for oi,order in enumerate((0,2)):
                    indices=[rot]+(([0,0] if identity else [pos,neg]) if order else [])
                    event=_make_event(indices,comps,dist,oi)
                    req=DFRTEEventSequenceCircuitRequest(events=(event,),component_specs=specs,
                        controlled=False,ancilla_qubit=None,occurrence_rte_steps=1)
                    qc=QiskitDFRTEEventCircuitBuilder(basis_registry=registry).build_sequence(req).circuit
                    expected=math.pi*(order//2)+sum(math.pi for a in event.application_sequence[:-1] if a.coefficient_sign==-1)-event.unsigned_rotation_angle
                    v=vacuum_check(qc)
                    manual_error=float(abs(np.exp(1j*v['phase'])-np.exp(1j*expected)))
                    rows.append(dict(identity=identity,coefficient_sign=sign,time_sign=timesign,
                        order=order,expected_phase=expected,manual_phase_error=manual_error,**v))
    ok=max(max(r['manual_phase_error'],r['phase_action_error']) for r in rows)<1e-11
    result=dict(checks=rows,all_passed=bool(ok),interpretation='Artificial signed factors; no new H or sampling law; exact same builders',
        formula='pi*(even order/2)+pi*(number negative products)-signed rotation angle')
    save_json(OUT/'A1_SIGN_IDENTITY_NEGATIVE_TIME.json',result)
    if not ok:raise ArithmeticError('Signed vacuum phase')
    return result
