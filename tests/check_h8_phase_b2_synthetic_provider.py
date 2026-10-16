"""Two-qubit S4/B0/B0I correctness against independent dense evolution."""
from copy import deepcopy
from dataclasses import asdict
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_phase_b2_contract as c
import h8_phase_b2_adapter as a


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',type=Path,required=True);args=p.parse_args()
 sys.path.insert(0,str(args.provider/'src'))
 import numpy as np
 from scipy.linalg import expm
 from qiskit import QuantumCircuit
 from qiskit.quantum_info import Operator
 from qiskit.circuit.library import RZGate,XXPlusYYGate
 from trotterlib.df_rte_tail import describe_basis_change_operations
 from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
 from trottertracks.resource_applicability import ax2b_native_df_v5 as native,h6_matched_port_v1 as m
 import qiskit
 operations=((RZGate(.2),(0,)),(XXPlusYYGate(.4,-.3),(0,1)))
 def inspect(ops):
  described=describe_basis_change_operations(ops);return described,json.loads(json.dumps([asdict(x) for x in described]))
 _,metadata=inspect(operations);h=c.digest(dict(num_system_qubits=2,operations=metadata,hash_policy='ordered_local_basis_operations_v1'))
 one=dict(block_id='df-one-body-correction',basis_id='df-basis-'+h,basis_hash=h,operations=metadata,diagonal_hex=[(.3).hex(),(-.2).hex()],lambda_hex=None)
 fragment=dict(one,block_id='df-fragment-0',lambda_hex=(-.7).hex())
 full=dict(blocks=[one,fragment],preparation_hash='artificial',constant_hex=(.6).hex())
 export={'faithful:p1':full,'discard:p1':deepcopy(full)};calls={}
 def require(scope):
  assert scope==c.SCOPE
 def take(name):calls[name]=calls.get(name,0)+1
 permit=NS(require=require,take=take)
 def factory(name,params):return {'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params)
 blocks=a.restore(export,permit,factory=factory,inspect=inspect,one_type=DFDeterministicOneBodySpec,fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
 view=dict(policy='artificial',prefix=1)
 expected=dict(block_count=2,ordered_block_description_sha256=c.digest(full['blocks']),constant_hex=full['constant_hex'],full_preparation_hash='artificial',saved_root_preparation_hash='artificial',view_recipe=view,view_recipe_sha256=c.digest(view))
 assert a.observe_prefix(blocks,export,1,expected,inspect,n=2)==expected
 qc=QuantumCircuit(2)
 for gate,qs in operations:qc.append(gate,qs)
 U=Operator(qc).data;matrices=[]
 for i,block in enumerate(blocks):
  diagonal=block.diagonal_eigenvalues if i==0 else block.diagonal_eta
  E=np.array([sum(v*((k>>q)&1) for q,v in enumerate(diagonal)) for k in range(4)])
  if i:E=block.lam*E**2
  matrices.append(U@np.diag(E)@U.conj().T)
 state=np.array([.5,.5j,-.5,.5],complex);max_error=0.;compiles=0
 compiler=m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
 w=1/(2-2**(1/3));weights=(w,1-2*w,w);assert weights[1]<0
 for method,q in [('B0',1),('B0I',8)]:
  scalar_saved=None if method=='B0' else dict(policy='REUSED_SAVED_P21_FULL_FOCK_SCALAR',receipt=dict(C=.125,full_Fock_identity=True,new_scientific_certificate=False))
  scalar=a.scalar_for(method,(.6).hex(),dict(scalar_receipt=scalar_saved))
  vector=state.copy()
  for _ in range(q):
   for weight in weights:
    dt=.8*weight/q
    for H in matrices:vector=expm(-1j*dt/2*H)@vector
    for H in reversed(matrices):vector=expm(-1j*dt/2*H)@vector
  vector*=np.exp(-1j*.8*scalar);z=complex(np.vdot(state,vector));pack=dict(real=z.real,imag=z.imag)
  target=dict(corrected=pack,independent_forward=pack)
  built=a.build_case(dict(q=q),blocks,scalar,state,target,permit,native,m,n=2,cap=10000)
  max_error=max(max_error,built['max_control_error'])
  for axis in ('cosine','sine'):
   row=a.common.compile_wrapper(built['wrappers'][axis],built['fingerprints'][axis],m,compiler,cap=100000)
   assert row['measured_RZ']>0 and row['compiler_hash']==c.COMPILER;compiles+=1
  bad=deepcopy(target);bad['independent_forward']['real']+=.1
  try:a.build_case(dict(q=q),blocks,scalar,state,bad,permit,native,m,n=2,cap=10000)
  except ValueError as exc:assert str(exc)=='S4_CONTROL_ERROR_GE_1E8'
  else:raise AssertionError('wrong independent forward target admitted')
 mutated=deepcopy(expected);mutated['ordered_block_description_sha256']='bad'
 try:a.observe_prefix(blocks,export,1,mutated,inspect,n=2)
 except ValueError:pass
 else:raise AssertionError('changed prefix accepted')
 print(json.dumps(dict(status='PASS_TWO_QUBIT_ARTIFICIAL_S4_ONLY',checks=11,system_qubits=2,synthetic_compile_axes=compiles,
  saved_molecular_inputs_opened=0,H8_objects_restored=0,real_H8_compile=0,new_fit=0,max_control_error=max_error,
  signed_middle_stage=True,scalar_once_B0_B0I=True,independent_dense_reference=True)))


if __name__=='__main__':main()
