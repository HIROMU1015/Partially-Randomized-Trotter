"""Two-qubit only: parallel controls, QPY exactness and unchanged compiled metrics."""
from dataclasses import asdict
from copy import deepcopy
import argparse,json,os,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_parallel as p
import h8_phase_b2_adapter as old
from types import SimpleNamespace as NS


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--provider',type=Path,required=True);args=parser.parse_args();sys.path.insert(0,str(args.provider/'src'))
 import numpy as np
 from scipy.linalg import expm
 from qiskit import QuantumCircuit,qpy
 from qiskit.circuit.library import RZGate,XXPlusYYGate
 from qiskit.quantum_info import Operator
 from trotterlib.df_rte_tail import describe_basis_change_operations
 from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
 from trottertracks.resource_applicability import ax2b_native_df_v5 as native,h6_matched_port_v1 as m
 import qiskit
 operations=((RZGate(.2),(0,)),(XXPlusYYGate(.4,-.3),(0,1)))
 descriptions=json.loads(json.dumps([asdict(x) for x in describe_basis_change_operations(operations)]));h=c.digest(dict(num_system_qubits=2,operations=descriptions,hash_policy='ordered_local_basis_operations_v1'))
 one=dict(block_id='df-one-body-correction',basis_id='df-basis-'+h,basis_hash=h,operations=descriptions,diagonal_hex=[(.3).hex(),(-.2).hex()],lambda_hex=None);frag=dict(one,block_id='df-fragment-0',lambda_hex=(-.7).hex())
 full=dict(blocks=[one,frag],preparation_hash='artificial',constant_hex=(.6).hex());export={'faithful:p1':full,'discard:p1':deepcopy(full)}
 def inspect(ops):
  ds=describe_basis_change_operations(ops);return ds,json.loads(json.dumps([asdict(x) for x in ds]))
 fake=NS(require=lambda scope:None,take=lambda name:None)
 blocks=old.restore(export,fake,factory=lambda name,params:{'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params),inspect=inspect,one_type=DFDeterministicOneBodySpec,fragment_type=DFDeterministicFragmentSpec,n=2,rank=1)
 qc=QuantumCircuit(2)
 for gate,qs in operations:qc.append(gate,qs)
 U=Operator(qc).data;matrices=[]
 for i,block in enumerate(blocks):
  diag=block.diagonal_eigenvalues if i==0 else block.diagonal_eta;E=np.array([sum(v*((k>>q)&1) for q,v in enumerate(diag)) for k in range(4)])
  if i:E=block.lam*E**2
  matrices.append(U@np.diag(E)@U.conj().T)
 state=np.array([.5,.5j,-.5,.5],complex);q=2;scalar=.725;v=state.copy();w=1/(2-2**(1/3))
 for _ in range(q):
  for weight in (w,1-2*w,w):
   for H in matrices:v=expm(-1j*.8*weight/q/2*H)@v
   for H in reversed(matrices):v=expm(-1j*.8*weight/q/2*H)@v
 v*=np.exp(-1j*.8*scalar);z=complex(np.vdot(state,v));pack=dict(real=z.real,imag=z.imag);target=dict(corrected=pack,independent_forward=pack)
 old_built=old.build_case(dict(q=q),blocks,scalar,state,target,fake,native,m,n=2,cap=10000)
 compiler=m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__);cpus=sorted(os.sched_getaffinity(0))[:2];assert len(cpus)==2
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);g=dict(scope=c.SCOPE,runtime_authorization=True,resources=c.RESOURCES,call_caps=c.CAPS,worker_cpus=cpus)
  permit=r.Permit({},g,root);permit.child_CPU_seconds=0.
  start=time.monotonic();built=p.build_case(dict(q=q),blocks,scalar,state,target,permit,native,m,n=2,cap=10000);wall=time.monotonic()-start
  assert 'wrappers' not in built and built['fingerprints']==old_built['fingerprints']
  assert built['max_control_error']<1e-8
  reports=[r.read(root/name/'parallel_receipt.json') for name in ('directional_controls','axis_controls_and_fingerprints')]
  assert all(v['all_reaped'] and len(v['workers'])==2 and all(w['exit_code']==0 and w['reaped'] for w in v['workers']) for v in reports)
  measured=[]
  for axis in ('cosine','sine'):
   loaded=p.load_wrapper(built['spools'][axis],root,qpy,c.RESOURCES['qpy_file_bytes'])
   assert m.canonical_qiskit_circuit_fingerprint(loaded)==built['fingerprints'][axis]
   before=old.common.compile_wrapper(old_built['wrappers'][axis],old_built['fingerprints'][axis],m,compiler,cap=100000)
   after=old.common.compile_wrapper(loaded,built['fingerprints'][axis],m,compiler,cap=100000)
   for key in ('measured_RZ','CX','depth','size','RZ_depth','CX_depth','fingerprint_before','fingerprint_after','compiler_hash'):assert before[key]==after[key],key
   measured.append(dict(axis=axis,measured_RZ=after['measured_RZ'],fingerprint=after['fingerprint_before']))
  assert permit.actual()['control_probe']==4 and permit.actual()['qpy_dump']==2
  print(json.dumps(dict(status='PASS_TWO_QUBIT_PARALLEL_QPY_ONLY',parallel_worker_counts=[2,2],parallel_workers_exit0=4,QPY_roundtrip_fingerprints_equal=True,compiled_metrics_equal=True,synthetic_compile_axes=4,max_control_error=built['max_control_error'],parallel_preparation_wall_seconds=wall,real_H8_objects=0,real_H8_compile=0,molecular_inputs_opened=0,new_fit=0,axes=measured)))


if __name__=='__main__':main()
