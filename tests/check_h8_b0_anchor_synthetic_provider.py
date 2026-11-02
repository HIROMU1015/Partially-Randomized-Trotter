"""Two-qubit fixture using the pinned real native and compiler APIs, no molecule."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_b0_anchor_adapter as a
import h8_b0_anchor_contract as c


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.provider/'src'))
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    from trotterlib.df_rte_tail import describe_basis_change_operations
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trottertracks.resource_applicability import h6_matched_port_v1 as m
    import qiskit
    operations=((RZGate(.2),(0,)),(XXPlusYYGate(.4,-.3),(0,1)))
    def inspect(ops):
        d=describe_basis_change_operations(ops);return d,json.loads(json.dumps([asdict(x) for x in d]))
    _,metadata=inspect(operations);h=c.digest(dict(num_system_qubits=2,operations=metadata,hash_policy='ordered_local_basis_operations_v1'))
    one=dict(block_id='df-one-body-correction',basis_id='df-basis-'+h,basis_hash=h,operations=metadata,
             diagonal_hex=[(.3).hex(),(-.2).hex()],lambda_hex=None)
    frag=dict(one,block_id='df-fragment-0',lambda_hex=(-.7).hex())
    root=dict(blocks=[one,frag],preparation_hash='synthetic',constant_hex=(.6).hex())
    export={'faithful:p1':root,'discard:p1':deepcopy(root)}
    counts={};permit=NS(require=lambda s:None,take=lambda k:counts.update({k:counts.get(k,0)+1}))
    def factory(name,params):return {'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params)
    blocks=a.restore(export,permit,factory=factory,inspect=inspect,one_type=DFDeterministicOneBodySpec,
        fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
    expected=dict(saved_preparation_hash='synthetic',ordered_block_description_sha256=c.digest(root['blocks']),
        constant_hex=(.6).hex(),block_count=2,view_recipe_sha256='synthetic-view')
    assert a.observe_prefix(blocks,export,1,expected,inspect,n=2)==expected
    qc=QuantumCircuit(2)
    for gate,qs in operations:qc.append(gate,qs)
    U=Operator(qc).data;matrices=[]
    for i,b in enumerate(blocks):
        diagonal=b.diagonal_eigenvalues if i==0 else b.diagonal_eta
        E=np.array([sum(v*((k>>q)&1) for q,v in enumerate(diagonal)) for k in range(4)])
        if i:E=b.lam*E**2
        matrices.append(U@np.diag(E)@U.conj().T)
    state=np.array([.5,.5j,-.5,.5],complex);max_error=0.;compiles=0
    compiler=m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
    for q in (1,8):
        result=state.copy();dt=.8/q
        for _ in range(q):
            result=expm(-1j*dt/2*matrices[0])@result
            result=expm(-1j*dt*matrices[1])@result
            result=expm(-1j*dt/2*matrices[0])@result
        result*=np.exp(-1j*.6*.8);z=complex(np.vdot(state,result));pack=dict(real=z.real,imag=z.imag)
        ws,fps,error=a.build_case(dict(q=q),blocks,.6,state,dict(corrected=pack,independent_forward=pack),permit,native,m,n=2,cap=10000)
        max_error=max(error,max_error)
        for axis in ('cosine','sine'):
            row=a.compile_wrapper(ws[axis],fps[axis],m,compiler,cap=100000)
            assert row['measured_RZ']>0 and row['compiler_hash']==c.COMPILER;compiles+=1
    # Re-inspection must reject a saved matrix claim that differs from the Gate.
    bad=deepcopy(export);bad['faithful:p1']['blocks'][0]['operations'][0]['matrix_sha256']='0'*64
    try:a.restore(bad,permit,factory=factory,inspect=inspect,one_type=DFDeterministicOneBodySpec,
        fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
    except ValueError:pass
    else:raise AssertionError('tampered saved basis accepted')
    print(json.dumps(dict(status='PASS_SYNTHETIC_PROVIDER_ONLY',system_qubits=2,checks=8,
        synthetic_compile_axes=compiles,real_H8_compile_axes=0,max_control_error=max_error,
        saved_molecular_inputs_opened=0,new_fit=0,scalar_once=True)))


if __name__=='__main__':main()
