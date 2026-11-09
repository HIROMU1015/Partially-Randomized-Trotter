"""Small H8 full-H RTE study; original saved science is read-only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import h8_long_time_adapter as old
from h46_m5_contract import core

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10'
PRIVATE=PUBLIC/'PRIVATE'
CONFIG=Path('/home/AbeHiromu/projects/h8-ab-validation/20261018/private/config_A_SEALED_PRIVATE.json')
H8_SCRIPTS=CONFIG.parents[1]/'repo/scripts'
COHORT='H8_FULL_H_PURE_RTE_LIGHT_EXPLORATION_20261110'
SIGNAL_FILE=ROOT/'artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_combined.json.gz'
CONTEXT=None
require=old.require


def read(path):
    p=Path(path);data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
    temp=p.with_name(p.name+'.tmp');temp.write_bytes(data);temp.replace(p)


def schedule(row):
    require(row['family']=='FULL_H_PURE_RTE' and row['system']=='H8' and row['T']==1.6
        and row['formula']=='2nd' and row['prefix']==0 and row['K'] in (0,2,4)
        and row['q']>=1 and row['R_outer']>=4,'NEW_PURE_H8_SCOPE')
    return core.schedule('2nd',row['q'],'ABS_TIME_PER_OUTER',row['R_outer'],T=1.6)


def candidate(R,K):
    row=dict(system='H8',T=1.6,epsilon=.0005,formula='2nd',method='PR',
        family='FULL_H_PURE_RTE',prefix=0,L_D=0,deterministic_block_count=0,
        q=32,R_outer=int(R),K=K,s=32*int(R))
    row['id']='H8FH_'+old.digest(row)[:20]
    schedule(row)
    return row


def tensor_action(vector,matrix,qubits):
    """Vectorize the existing <=4-qubit little-endian local tensor oracle."""
    qubits=tuple(qubits);n=len(vector).bit_length()-1;k=len(qubits)
    require(len(vector)==2**n and 1<=k<=4 and len(set(qubits))==k,'LOCAL_LAYOUT')
    front=[n-1-q for q in reversed(qubits)]
    perm=front+[i for i in range(n) if i not in front]
    data=np.asarray(vector).reshape((2,)*n).transpose(perm).reshape(2**k,-1)
    result=(np.asarray(matrix)@data).reshape((2,)*n)
    return result.transpose(np.argsort(perm)).reshape(-1)


class EventOracle:
    """Same saved-basis algebra as explicit_event_action, with cached matrices."""
    def __init__(self,registry,n):
        from qiskit.quantum_info import Operator
        self.definitions={};self.n=n
        self.bits=[1-2*((np.arange(2**n)>>q)&1) for q in range(n)]
        self.Operator=Operator;self.registry=registry

    def operations(self,basis_id,basis_hash):
        if basis_id not in self.definitions:
            d=self.registry.definition(basis_id)
            require(d.metadata.basis_hash==basis_hash,'SAVED_BASIS_HASH')
            self.definitions[basis_id]=(basis_hash,[(self.Operator(g).data,tuple(q)) for g,q in d.runtime_operations])
        require(self.definitions[basis_id][0]==basis_hash,'COMPONENT_BASIS_HASH')
        return self.definitions[basis_id][1]

    def rotate(self,v,operations,*,inverse=False):
        for matrix,qubits in reversed(operations) if inverse else operations:
            v=tensor_action(v,matrix.conj().T if inverse else matrix,qubits)
        return v

    def component(self,v,c):
        if getattr(c,'is_identity',False):return c.coefficient_sign*v
        operations=self.operations(c.basis_id,c.basis_hash)
        v=self.rotate(v,operations,inverse=True)
        for q in c.diagonal_pauli_support:v=self.bits[q]*v
        return c.coefficient_sign*self.rotate(v,operations)

    def event(self,v,e):
        for application in e.application_sequence:
            image=self.component(v,application)
            v=image if application.role=='product' else math.cos(e.rotation_angle)*v-1j*math.sin(e.rotation_angle)*image
        return e.phase*v

    def trajectory(self,ctx,row,requests):
        v=ctx.qstate.copy()
        for request in requests:
            for event in request.rte_occurrence.events:v=self.event(v,event)
        return np.exp(-1j*row['T']*ctx.scalar(row))*v


def install():
    config=read(CONFIG)
    for p in reversed([str(H8_SCRIPTS),str(Path(config['provider_root'])/'src')]):
        if p not in sys.path:sys.path.insert(1,p)
    import h46_full_h_t20_aer as verifier
    from qiskit_aer import AerSimulator
    verifier.AerSimulator=partial(AerSimulator,fusion_enable=True,fusion_threshold=0,fusion_max_qubit=4)
    from trotterlib.df_trotter import circuit
    circuit.simulate_statevector=verifier.simulate_statevector
    from h46_b3_fast_fingerprint import use_fast_fingerprint
    from trottertracks.resource_applicability import h6_hash_reuse_v1
    h6_hash_reuse_v1.use_fingerprint_reuse=use_fast_fingerprint
    import h46_long_time_native as native
    native.PRIVATE=PRIVATE
    native.trial_seed=lambda row,cohort,index,outer,piece:int(old.digest(
        ['H8_FULL_H_LIGHT_20261110',cohort,row['id'],index,outer,piece])[:16],16)
    # Only this new study process dispatches pure rows; old fixed files remain intact.
    old.schedule=schedule
    return native


def context(phase):
    global CONTEXT
    if CONTEXT is not None:return CONTEXT
    install()
    original=read(CONFIG);used=[]
    for item in original['inputs']:
        if item['role'] not in {'h8_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','h8_preparation_export'}:continue
        require(old.sha(item['path'])==item['sha256'],'SAVED_INPUT_HASH:'+item['role'])
        used.append({k:item[k] for k in ('role','sha256','bytes')})
    prior=read(SIGNAL_FILE);ref=prior['reference'];old.check_reference(ref,ref['binding'])
    b=ref['binding'];ob=original['binding']
    require(all(b[k]==ob[k] for k in ('hamiltonian','state','df','compiler'))
        and b['snapshot']==ob['input'] and b['fragment_order']==list(range(28)),'SAME_SAVED_H8_INPUT')
    import h8_long_time_runtime as runtime
    config=dict(import_paths=[str(H8_SCRIPTS),str(Path(original['provider_root'])/'src')],
        native_config=str(CONFIG),binding=b,preparation_identities={})
    folder=PRIVATE/'contexts'/phase;folder.mkdir(parents=True,exist_ok=False)
    ctx=runtime.Context(config,folder,[0],signal=False)
    ctx.qindices=np.asarray(ctx.qindices,dtype=int)
    import h46_full_h_pure_rte as full
    ctx.base=ctx.preps[0]
    ctx.preps[0]=full.prepare_full_h(ctx.ham,base=ctx.base)
    prep=ctx.preps[0]
    require(not prep.deterministic_blocks and prep.randomized_block_indices==tuple(range(28)),'FULL_H_RTE_ALL_FRAGMENTS')
    ctx.prepare_sparse()
    ctx.tails[0]=ctx.sparse.df_operator(-prep.extracted_identity_coefficient,ctx.one,
        ctx.ham.lambdas,ctx.blocks,ctx.counts)
    ctx.reference=complex(ref['signal']['real'],ref['signal']['imag'])
    image=ctx.full_op[0]@ctx.state
    ctx.energy=float(np.real(np.vdot(ctx.state,image)))
    ctx.residual=float(np.linalg.norm(image-ctx.energy*ctx.state))
    require(ctx.residual<1e-10,'SAVED_EIGENSTATE_RESIDUAL')
    ctx.reference_error=float(abs(np.exp(-1j*1.6*ctx.energy)-ctx.reference))
    require(ctx.reference_error<1e-10,'SAVED_T16_REFERENCE_EIGEN_PHASE')
    ctx.oracle=EventOracle(prep.rte_preparation.basis_registry,16)
    ctx.reference_record=ref
    save(folder/'full_h_context.json',dict(binding=b,T=1.6,lambda_R=prep.exact_rte_lambda_r,
        scalar=float(ctx.ham.constant+prep.extracted_identity_coefficient),energy=ctx.energy,
        residual=ctx.residual,reference=ref,reference_difference=ctx.reference_error,
        components=len(prep.tail_extraction.components),deterministic_blocks=0,
        saved_inputs=used,new_SCF_DF_state=0,new_target_reference=0,new_Gaussian_operations=0))
    CONTEXT=ctx
    return ctx


def grouped_action(ctx,vector,*,one_only=False):
    prep=ctx.preps[0];groups=defaultdict(list)
    for c in prep.tail_extraction.components:
        if not one_only or c.df_fragment_id=='full-h-one-body':groups[c.basis_id].append(c)
    out=np.zeros_like(vector)
    for bid,components in groups.items():
        operations=ctx.oracle.operations(bid,components[0].basis_hash)
        v=ctx.oracle.rotate(vector,operations,inverse=True)
        diagonal=np.zeros(len(v))
        for c in components:
            values=np.full(len(v),c.coefficient)
            for q in c.diagonal_pauli_support:values*=ctx.oracle.bits[q]
            diagonal+=values
        out+=ctx.oracle.rotate(diagonal*v,operations)
    return out


def validate(phase='validation'):
    ctx=context(phase);prep=ctx.preps[0]
    errors=[];one_errors=[]
    one_scalar=math.fsum(ctx.base.deterministic_blocks[0].diagonal_eigenvalues)/2
    for state in [ctx.state,np.eye(1,len(ctx.state),0,dtype=complex)[0],np.eye(1,len(ctx.state),len(ctx.state)-1,dtype=complex)[0]]:
        v=np.zeros_like(ctx.qstate);v[ctx.qindices]=state
        expected=np.zeros_like(v);expected[ctx.qindices]=ctx.full_op[0]@state
        actual=(ctx.ham.constant+prep.extracted_identity_coefficient)*v+grouped_action(ctx,v)
        errors.append(float(np.linalg.norm(actual-expected)))
        target=np.zeros_like(v);target[ctx.qindices]=ctx.one@state
        one_errors.append(float(np.linalg.norm(one_scalar*v+grouped_action(ctx,v,one_only=True)-target)))
    require(max(errors+one_errors)<1e-9,'SAVED_FULL_H_RECONSTRUCTION')
    components=prep.rte_preparation.symbolic_tail.components
    require({c.df_fragment_id for c in prep.tail_extraction.components if c.df_fragment_id!='full-h-one-body'}
        =={f'df-fragment-{i}' for i in range(28)},'ALL_28_DF_COMPONENTS')
    require(abs(math.fsum(c.probability for c in components)-1)<1e-12,'COMPONENT_LAW_NORMALIZED')
    require(abs(math.fsum(c.coefficient_abs for c in prep.tail_extraction.components)-prep.exact_rte_lambda_r)<1e-10,'LAMBDA_COMPONENT_SUM')
    from trotterlib.rte import finite_rte_distribution,_make_event
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder,DFRTEEventCircuitRequest
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import explicit_event_action
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    one=next(i for i,c in enumerate(components) if c.df_fragment_id=='full-h-one-body')
    two=max((i for i,c in enumerate(components) if c.df_fragment_id!='full-h-one-body'),key=lambda i:components[i].probability)
    checks=[]
    for sign in (1,-1):
        distribution=finite_rte_distribution(sign*prep.exact_rte_lambda_r*1.6/32768,2)
        for order in distribution.orders:
            event=_make_event([one]+[two if j%2==0 else one for j in range(order)],components,distribution,distribution.orders.index(order))
            qc=builder.build_event(DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=16)).circuit
            high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate];low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)]
            actual=simulate_statevector(qc,high);expected=ctx.oracle.event(ctx.qstate,event)
            values=[float(np.linalg.norm(actual[len(expected):]-expected)),float(np.linalg.norm(actual[:len(expected)])),
                float(np.linalg.norm(simulate_statevector(qc,low)-low)),float(np.linalg.norm(simulate_statevector(qc.inverse(),actual)-high))]
            # One real event against the original literal local-tensor implementation.
            if sign==1 and order==0:values.append(float(np.linalg.norm(expected-explicit_event_action(ctx.qstate,event,prep.rte_preparation.basis_registry))))
            require(max(values)<1e-8,'NATIVE_EVENT_SIGN_ADJOINT')
            checks.append(dict(sign=sign,order=order,errors=values,conditional=True,in_cost_samples=False))
    result=dict(status='PASS',hamiltonian_errors=errors,one_body_errors=one_errors,conditional_events=checks,
        deterministic_blocks=0,DF_fragments=28,one_body_components=sum(c.df_fragment_id=='full-h-one-body' for c in components),
        lambda_R=prep.exact_rte_lambda_r,scalar=float(ctx.ham.constant+prep.extracted_identity_coefficient),
        positive_negative_and_actual_adjoint=True,reference=ctx.reference_record,
        eigenstate_residual=ctx.residual,dense_full_matrix_allocated=False,phase_alignment=False)
    save(PUBLIC/'validation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate']);p.add_argument('--run-tag',default='validation');args=p.parse_args()
    require(Path(args.run_tag).name==args.run_tag,'LOCAL_RUN_TAG')
    try:print(json.dumps(validate(args.run_tag)))
    except BaseException as exc:
        save(PUBLIC/(args.command+'_'+args.run_tag+'_STOP.json'),dict(reason=str(exc),type=type(exc).__name__))
        raise
