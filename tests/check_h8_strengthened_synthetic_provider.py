"""Tiny Qiskit/NumPy fixture for saved-operation restoration and phase policy.

Run only as a synthetic test. Does not open molecular input or compile gates.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h8_strengthened_contract as contract
import h8_strengthened_export as exported
import h8_strengthened_signal as adapter


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.provider/'src'))
    import numpy as np
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    from trotterlib.df_rte_tail import describe_basis_change_operations
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
    from trottertracks.resource_applicability.ax2b_native_df_v5 import make_native_block_action
    from trottertracks.resource_applicability import h6_matched_port_v1 as m
    import h6_native_b0i_bridge_v1 as overlay
    operations=((RZGate(.2),(0,)),(XXPlusYYGate(.4,-.3),(0,1)))
    described=describe_basis_change_operations(operations)
    metadata=json.loads(json.dumps([asdict(x) for x in described]))
    h=contract.digest(dict(num_system_qubits=2,operations=metadata,hash_policy='ordered_local_basis_operations_v1'))
    block=dict(block_id='df-one-body-correction',basis_id='df-basis-'+h,basis_hash=h,operations=metadata,
               diagonal_hex=[(.3).hex(),(-.2).hex()],lambda_hex=None)
    fragment=dict(block,block_id='df-fragment-0',lambda_hex=(-.7).hex())
    full=dict(blocks=[block,fragment],components=[],bases={},constant_hex=(.6).hex(),
              extracted_hex=0..hex(),lambda_r_hex=0..hex(),preparation_hash='synthetic')
    export={'faithful:p1':full}
    calls=[];permit=NS(require=lambda scope:calls.append(scope),take=lambda name:calls.append(name))
    def factory(name,params):return {'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params)
    def inspect(ops):
        d=describe_basis_change_operations(ops)
        return d,json.loads(json.dumps([asdict(x) for x in d]))
    restored=exported.restore_blocks(export,permit,gate_factory=factory,inspect_operations=inspect,
        one_type=DFDeterministicOneBodySpec,fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.linalg import expm
    from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
    gaussian=QuantumCircuit(2)
    for gate,qubits in operations:gaussian.append(gate,qubits)
    U=Operator(gaussian).data
    state=np.array([.5,.5j,-.5,.5],dtype=complex)
    matrices=[];actions=[]
    for b in restored:
        action=make_native_block_action(b,max_instructions=1000)
        actions.append(action)
        diagonal=b.diagonal_eigenvalues if isinstance(b,DFDeterministicOneBodySpec) else b.diagonal_eta
        energies=np.array([sum(x*((k>>q)&1) for q,x in enumerate(diagonal)) for k in range(4)])
        if isinstance(b,DFDeterministicFragmentSpec):energies=b.lam*energies**2
        H=U@np.diag(energies)@U.conj().T;matrices.append(H)
        for t in (.2,-.3):
            assert np.linalg.norm(action(state,t)-expm(-1j*t*H)@state)<1e-14
    # Compare merged production S4 with an independent unmerged three-S2
    # composition, including the negative middle time and one scalar per q.
    cell=dict(id='synthetic',method='B0',prefix=1,q=4,formula='4th',r=None,R=None,K=None)
    actual=m.compact_signal(state,actions,None,cell=cell,T=.8,scalar=.6,lambda_r=0.,
                            budget=ActionBudget(0,10000))
    w=1/(2-2**(1/3));expected=state.copy()
    for _ in range(4):
        for weight in (w,1-2*w,w):
            t=.8/4*weight
            expected=expm(-1j*t/2*matrices[0])@expected
            expected=expm(-1j*t*matrices[1])@expected
            expected=expm(-1j*t/2*matrices[0])@expected
        expected*=np.exp(-1j*.6*.8/4)
    assert np.linalg.norm(actual['vectors']['corrected']-expected)<1e-13
    # Distinct signal, exact truncation and full-H reference catch accidentally
    # rotating the reference or adding the tail scalar twice.
    base=dict(cell=dict(id='B0_p1_q1_4th',method='B0',prefix=1,q=1,formula='4th'),
        signals=dict(corrected=dict(real=.4,imag=-.3),exact_truncated=dict(real=.5,imag=-.2)),
        independent_forward_signals=dict(corrected=dict(real=.4,imag=-.3)),
        reference=dict(real=.2,imag=-.6),allowance=dict(real=1e-10,imag=1e-10,
        log_B_margin=1e-10,kind='EMPIRICAL',certified=False),log_B=0.)
    plan=dict(epsilons=[.05],numerical=dict(sensitivity_factors=[1.,10.],maximum_u_over_headroom=.01),
              shot_rule=dict(alpha_axis=.025))
    port=NS(reference=.2-.6j,plan=plan)
    result=adapter.overlay_record(base,(.17,0.,{}),port,permit,overlay,m)
    z=complex(result['signals']['corrected']['real'],result['signals']['corrected']['imag'])
    assert abs(z-(.4-.3j)*np.exp(-1j*.8*.17))<1e-14
    assert result['reference']==base['reference']
    parts=result['error_decomposition']['signed_parts'];total=result['error_decomposition']['total_signed']
    assert abs(sum(complex(v['real'],v['imag']) for v in parts.values())-complex(total['real'],total['imag']))<1e-14
    assert result['cell']['formula']=='4th' and result['allowance']['certified'] is False
    # Gate-description/matrix hashes must fail rather than silently regenerating.
    bad=json.loads(json.dumps(export));bad['faithful:p1']['blocks'][0]['operations'][0]['matrix_sha256']='0'*64
    try:
        exported.restore_blocks(bad,permit,gate_factory=factory,inspect_operations=inspect,
            one_type=DFDeterministicOneBodySpec,fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
    except ValueError:pass
    else:raise AssertionError('tampered basis accepted')
    print(json.dumps(dict(status='PASS_SYNTHETIC_ONLY',checks=8,real_molecular_inputs=0,
        new_compile=0,restored_qubits=2,scalar_once_and_reference_unrotated=True)))


if __name__=='__main__':main()
