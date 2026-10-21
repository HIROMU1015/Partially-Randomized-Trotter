"""One-shot, fixed-candidate native S2 comparison. No import-time science."""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22'
COHORT = 'FIXED_CANDIDATES_RECOMPUTED_NATIVE_20261022'
EPS = ['0.05', '0.01', '0.005', '0.001', '0.0005']
ENV = {'numpy':'1.26.4','scipy':'1.14.1','qiskit':'1.3.0',
       'qiskit-nature':'0.7.2','pyscf':'2.7.0','qiskit-aer':'0.15.1'}
_BOOTED = False
CTX = None
CANDIDATE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def require(ok,message):
    if not ok:raise ValueError(message)


def seed(*parts):
    return int.from_bytes(hashlib.sha256(('/'.join(map(str,(COHORT,)+parts))).encode()).digest()[:8],'big')


def bootstrap():
    global _BOOTED
    if _BOOTED:return
    require({k:importlib.metadata.version(k) for k in ENV}==ENV,'ENVIRONMENT_VERSION_MISMATCH')
    for name in ['trotterlib','trottertracks','trottertracks.resource_applicability']:
        mod=types.ModuleType(name)
        mod.__path__=[str(BASE/'fixed_provider/src'/name.replace('.','/'))]
        sys.modules[name]=mod
    _BOOTED=True


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def counter():
    return load_file('_recomputed_counter',ROOT/'src/trotterlib/current_s2_theory.py')


def unit_path(system,candidate,stage,epsilon=None):
    return BASE/'units'/system/candidate/(stage if epsilon is None else stage+'_'+epsilon)


@contextmanager
def unit(system,candidate,stage,inputs,conditions=None,epsilon=None):
    source=read(BASE/'SOURCE_v2.json')
    for path,h in source['files'].items():
        require(sha(ROOT/path)==h,'FROZEN_SOURCE_CHANGED:'+path)
    dest=unit_path(system,candidate,stage,epsilon)
    require(not dest.exists(),'ONE_SHOT_UNIT_ALREADY_EXISTS:'+str(dest))
    dest.mkdir(parents=True)
    plan=dict(cohort=COHORT,system=system,candidate=candidate,stage=stage,epsilon=epsilon,
              SOURCE=source['commit'],source_manifest_sha256=sha(BASE/'SOURCE_v2.json'),
              input_hashes=inputs,seed=seed(system,candidate,stage,epsilon),
              conditions=conditions or {},one_shot=True,automatic_retry=False)
    grant=dict(approved_by_user=True,authorization='user_authorization.md',
               authorization_sha256=sha(BASE/'user_authorization.md'),plan_sha256=digest(plan),
               grant_id=digest(plan),SOURCE=source['commit'],one_shot=True,automatic_retry=False)
    save(dest/'plan.json',plan);save(dest/'grant.json',grant)
    save(dest/'marker.json',dict(state='CONSUMED_NONREFUNDABLE',grant_id=grant['grant_id'],time=time.time()))
    started=time.monotonic()
    try:
        yield dest
    except BaseException as exc:
        save(dest/'STOP.json',dict(status='FAILED_NO_RETRY',reason=str(exc),traceback=traceback.format_exc(),
             grant_id=grant['grant_id'],SOURCE=source['commit'],elapsed_seconds=time.monotonic()-started))
        raise
    else:
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
        save(dest/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP_UNIT',grant_id=grant['grant_id'],
             SOURCE=source['commit'],elapsed_seconds=time.monotonic()-started,result_files=files))


class Budget:
    def __init__(self):self.used={}
    def take(self,name,count=1):self.used[name]=self.used.get(name,0)+count


def exact_identity(matrix,weight):
    diagonal=[F(float(matrix[i,i].real)) for i in range(len(matrix))]
    square=sum((x*x for x in diagonal),F())
    for i in range(len(matrix)):
        for j in range(i):
            z=complex(matrix[i,j]);square+=2*(F(z.real)**2+F(z.imag)**2)
    return F(float(weight))/4*(sum(diagonal,F())**2+square)


def generate_h8(dest):
    import numpy as np
    from openfermion import low_rank_two_body_decomposition
    from scipy.sparse.linalg import eigsh
    from trottertracks.resource_applicability.h8_preflight_input_port_v1 import obtain_integrals
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector,_hartree_fock_initial_vector
    from trotterlib.pr2_s0_s1_validation import _canonicalize_state_phase
    from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR,df_operator
    (dest/'scratch').mkdir()
    payload=obtain_integrals(dest,dict(conv_tol=1e-9,max_cycle=50))
    with (dest/'integrals.npz').open('xb') as f:
        np.savez(f,**{k:v for k,v in payload.items() if isinstance(v,np.ndarray)},constant=payload['constant'])
    w,g,correction,truncation=low_rank_two_body_decomposition(payload['two_body'].copy(),truncation_threshold=1e-8)
    w=np.asarray(w.real,dtype='<f8');raw=np.asarray(g,dtype='<c16');one=payload['one_body']+correction
    require(len(w)==28 and 0<=truncation<=1e-8,'H8_FIXED_DF_RANK_OR_TRUNCATION')
    gs=(raw+raw.conj().transpose(0,2,1))/2;corrected=(one+one.conj().T)/2
    terms=[abs(float(l))*np.linalg.norm(p-a)*(2*np.linalg.norm(a)+np.linalg.norm(p-a)) for l,a,p in zip(w,raw,gs)]
    projection=16*np.linalg.norm(corrected-one)+256*math.fsum(terms)
    require(projection<=9.9e-11,'H8_WEIGHTED_HERMITIAN_PROJECTION_BUDGET')
    metadata=dict(input_policy='WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1',df_tol_requested=1e-8,
        df_rank_actual=28,df_truncation_value=float(truncation),final_rank_supplied=False,
        coefficient_order='decomposer_generation_order',subsequent_coefficient_cutoff=0.,
        projection_eta=float(projection),representation_error_certified=False)
    ham=DFHamiltonian(float(payload['constant']),corrected,w,tuple(gs),metadata)
    sector=PhysicalSector.spin_sector(n_qubits=16,nelec_alpha=4,nelec_beta=4)
    topology=OccupationCSR(sector.basis_indices,16);budget=Budget()
    op,trace=df_operator(ham.constant,topology.one(ham.one_body),w,tuple(topology.one(x) for x in gs),budget)
    eigen,vectors=eigsh(op,k=1,which='SA',tol=1e-12,maxiter=1000,ncv=40,v0=_hartree_fock_initial_vector(sector))
    state=np.asarray(vectors[:,0],dtype='<c16');state/=np.linalg.norm(state)
    full=np.zeros(65536,dtype='<c16');full[sector.basis_indices]=state
    full,state=_canonicalize_state_phase(full,state)
    residual=float(np.linalg.norm(op@state-eigen[0]*state))
    require(residual<1e-9+1e-10*max(1,abs(eigen[0])),'H8_STATE_RESIDUAL')
    save(dest/'generation_receipt.json',dict(hamiltonian_metadata=metadata,energy=float(eigen[0]),
        residual=residual,calls=budget.used,SCF=1,integral_build=1,DF=1,state_solver=1,
        state_policy='HF initial/eigsh SA tol1e-12 maxiter1000 ncv40/canonical phase',ground_state_certified=False))
    with (dest/'raw_df.npz').open('xb') as f:np.savez(f,lambdas=w,g_matrices=raw,correction=correction)
    return ham,sector,full,state


def build_input(system):
    bootstrap()
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector,df_linear_operator
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _state_hash,_to_qiskit_state
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
    from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR,SpinPrimitive,df_operator,reference_pair,exponential_action
    inventory=read(BASE/'input_inventory.json')[system]
    hashes={p:sha(p) for p in inventory.get('files',[])}
    with unit(system,'SYSTEM','input_preparation',hashes,inventory) as dest:
        if system=='H8':ham,sector,full,state=generate_h8(dest)
        else:
            with np.load(inventory['snapshot'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
            r=read(inventory['receipt']);md=r.get('hamiltonian_metadata') or r['metadata']['hamiltonian_metadata']
            ham=DFHamiltonian(float(a['constant']),a['one_body'],a['lambdas'],tuple(a['g_matrices']),md)
            n=int(system[1:]);sector=PhysicalSector.spin_sector(n_qubits=2*n,nelec_alpha=n//2,nelec_beta=n//2)
            full,state=a['state_vector'],a['sector_state_vector']
        require(ham.n_blocks=={'H2':3,'H4':10,'H6':19,'H8':28}[system],'FIXED_RANK')
        require(abs(np.linalg.norm(state)-1)<1e-12 and np.array_equal(full[sector.basis_indices],state),'STATE_LIFT')
        rows=[c for c in read(BASE/'candidates.json') if c['system']==system]
        prefixes=sorted({c['prefix'] for c in rows})
        preps={p:_prepare(ham,p) for p in prefixes}
        topology=OccupationCSR(sector.basis_indices,ham.n_qubits)
        one=topology.one(ham.one_body);blocks=tuple(topology.one(g) for g in ham.g_matrices)
        budget=Budget();op,trace=df_operator(ham.constant,one,ham.lambdas,blocks,budget)
        reference,refcheck=reference_pair(op,trace,state,.8)
        production,_=df_linear_operator(ham,sector,backend='python')
        probes=[state,np.eye(1,len(state),0,dtype=complex)[0]]
        errors=[float(np.linalg.norm(op@v-production@v)) for v in probes]
        require(max(errors)<1e-9,'PRODUCTION_CSR_OPERATOR_AGREEMENT')
        spin=[SpinPrimitive(ham.one_body,sector)]+[SpinPrimitive(g,sector,weight=w) for g,w in zip(ham.g_matrices,ham.lambdas)]
        primitive_errors=[]
        for j in range(len(spin)):
            if j==0:primitive=one;pt=float(np.real(one.diagonal().sum()))
            else:
                primitive=ham.lambdas[j-1]*(blocks[j-1]@blocks[j-1]);pt=float(np.real(primitive.diagonal().sum()))
            expected=exponential_action(primitive,pt,state,.4)
            e=float(np.linalg.norm(spin[j](state,.4)-expected));primitive_errors.append(e)
        require(max(primitive_errors)<1e-9,'SPIN_CSR_PRIMITIVE_AGREEMENT')
        old=read(ROOT/'docs/progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022/figure_data.json')
        previous=next(x['PR_original_record'] for x in old['rows'] if x['system']==system)
        identity=dict(hamiltonian_hash=df_hamiltonian_hash(ham),state_hash=_state_hash(full,state),
            rank=ham.n_blocks,n_qubits=ham.n_qubits,preparation_hashes={str(p):v.preparation_hash for p,v in preps.items()},
            original_hamiltonian_equal=df_hamiltonian_hash(ham)==previous['hamiltonian_hash'],
            original_state_equal=_state_hash(full,state)==previous['state_hash'],
            original_compiled_reused=False,new_MC_cohort=True,
            reference=dict(real=reference.real,imag=reference.imag),reference_check=refcheck,
            production_CSR_errors=errors,spin_CSR_errors=primitive_errors,reference_calls=budget.used,
            hamiltonian_metadata=ham.metadata)
        save(dest/'identity.json',identity)
        with (dest/'snapshot.npz').open('xb') as f:
            np.savez(f,constant=ham.constant,one_body=ham.one_body,lambdas=ham.lambdas,
                g_matrices=np.asarray(ham.g_matrices),state_vector=full,sector_state_vector=state,
                sector_basis_indices=sector.basis_indices)
        context=dict(ham=ham,sector=sector,full=full,state=state,qstate=_to_qiskit_state(full,ham.n_qubits),
            preps=preps,one=one,blocks=blocks,spin=spin,reference=reference,
            reference_error=max(errors)+refcheck['signal_difference'],identity=identity)
        with (dest/'context.pkl').open('xb') as f:pickle.dump(context,f,protocol=5)
        save(dest/'result.json',dict(status='PASS',identity=identity,context_sha256=sha(dest/'context.pkl')))
    return identity


def context(system):
    bootstrap();p=unit_path(system,'SYSTEM','input_preparation')
    require(read(p/'STOP.json')['status']=='COMPLETE_MANDATORY_STOP_UNIT','INPUT_STAGE_NOT_COMPLETE')
    require(sha(p/'context.pkl')==read(p/'result.json')['context_sha256'],'CONTEXT_CHANGED')
    with (p/'context.pkl').open('rb') as f:return pickle.load(f)


def scalar(ctx,c):
    p=ctx['preps'][c['prefix']];base=float(ctx['ham'].constant)
    if c['method']=='B2':return base+float(p.extracted_identity_coefficient)
    if c['method']=='B0I':
        ham=ctx['ham'];return base+float(sum((exact_identity(g,w) for g,w in zip(ham.g_matrices[c['prefix']:],ham.lambdas[c['prefix']:])),F()))
    return base


def theory(ctx,c):
    from trotterlib.df_trotter import ops
    from trotterlib.rte import finite_rte_distribution
    p=ctx['preps'][c['prefix']];api=counter();m=api.m
    bases={}
    def basis(bid,bh,runtime):
        key=bid,bh
        if key not in bases:bases[key]=m.basis_from_runtime(bid,bh,runtime)
        return bases[key]
    bs=tuple(basis(b.basis_id,b.basis_hash,b.runtime_basis_operations) for b in p.deterministic_blocks)
    def primitive(i,t):
        b=p.deterministic_blocks[i]
        if b.original_fragment_index is None:return ops.one_body_diagonal_primitives(b.diagonal_eigenvalues,t)
        return ops.df_squared_diagonal_primitives(b.diagonal_eta,b.lam,t)
    kwargs=dict(candidate=c,bases=bs,primitive_factory=primitive,ancilla=ctx['ham'].n_qubits,
                constant=float(ctx['ham'].constant),T=.8)
    law=None
    if c['method']=='B2':
        components=[]
        for comp in p.rte_preparation.symbolic_tail.components:
            b=None if comp.is_identity else basis(comp.basis_id,comp.basis_hash,p.rte_preparation.basis_registry.operations(comp.basis_id))
            components.append(m.Component(comp.component_id,F(comp.probability),b,tuple(comp.diagonal_pauli_support or ()),comp.coefficient_sign))
        tau=p.exact_rte_lambda_r*((.8/c['q'])/c['r']);dist=finite_rte_distribution(tau,c['K'])
        kwargs.update(components=components,native_order_probabilities=dict(zip(dist.orders,dist.order_probabilities)),
                      tau=tau,extracted=p.extracted_identity_coefficient)
        orders=dict(zip(dist.orders,map(float,dist.order_probabilities)))
        N=c['q']*c['r'];mass=math.fsum(v for k,v in orders.items() if k>0)
        rare=-math.expm1(N*math.log1p(-mass))
        law=dict(tau=tau,lambda_R=p.exact_rte_lambda_r,order_probabilities=orders,events_per_trajectory=N,
            probability_any_positive_order=rare,probability_any_order4=-math.expm1(N*math.log1p(-orders.get(4,0.))),
            expected_positive_order_events=N*mass,components=len(components),
            identity_components=sum(not x.support for x in components),all_positive_order_mass_retained=True,
            compiled_exact_expectation_feasible=False,
            compiled_exact_expectation_reason='Qiskit whole-trajectory optimizer depends on complete ordered gate stream; native DP is not a compiled-cost oracle.')
    if c['method']=='B0I':kwargs['saved_B0I_scalar']=scalar(ctx,c)
    return api.native_units(**kwargs),law


def signal(ctx,c):
    import numpy as np
    from trotterlib.rte import finite_rte_distribution
    p=ctx['preps'][c['prefix']];q=c['q'];dt=.8/q
    actions=ctx['spin'][:c['prefix']+1];initial=ctx['state'];random=c['method']=='B2'
    maximum=1.;work=0
    def tail(x):
        value=-p.extracted_identity_coefficient*x
        for w,g in zip(ctx['ham'].lambdas[c['prefix']:],ctx['blocks'][c['prefix']:]):value+=w*(g@(g@x))
        return value
    def checked(x):
        nonlocal maximum,work
        work+=1;maximum=max(maximum,float(np.linalg.norm(x)));require(np.isfinite(x).all(),'NONFINITE_SIGNAL')
        return x
    results=[]
    log_B=0.;b=1.
    if random:
        tau=p.exact_rte_lambda_r*(dt/c['r']);dist=finite_rte_distribution(tau,c['K'])
        b=dist.exact_finite_distribution;log_B=q*c['r']*math.log(b)
    for mode in ['horner','forward','raw']:
        v=initial.copy()
        for _ in range(q):
            for action in actions:v=checked(action(v,dt/2))
            if random:
                for _ in range(c['r']):
                    if mode!='horner':
                        term=v.copy();out=term.copy()
                        for k in range(1,c['K']+2):term=checked((-1j*dt/c['r']/k)*tail(term));out=out+term
                        v=checked(out)
                    else:
                        out=v.copy()
                        for k in range(c['K']+1,0,-1):out=checked(v+(-1j*dt/c['r']/k)*tail(out))
                        v=out
                    if mode=='raw':v=checked(v/b)
            for action in reversed(actions):v=checked(action(v,dt/2))
        v*=np.exp(-1j*scalar(ctx,c)*.8)
        results.append(complex(np.vdot(initial,v)))
    discrepancy=abs(results[0]-results[1]);require(discrepancy<1e-9,'HORNER_FORWARD_SIGNAL_DISAGREEMENT')
    raw_closure=abs(results[2]-math.exp(-log_B)*results[0])
    require(raw_closure<1e-9,'RAW_CORRECTED_NORMALIZATION_CLOSURE')
    from trottertracks.resource_applicability.h6_matched_accounting_v1 import empirical_allowance
    settings=dict(roundoff_multiplier=64.,safety_factor=10.,floor=1e-12,
                  scope='Fixed provider numerical policy: binary64 CSR/spin and Horner/forward checks; empirical, not certified')
    allowance=empirical_allowance(reference_error=ctx['reference_error'],path_errors=[discrepancy],
        normalization_error=raw_closure,raw_closure=raw_closure,maximum_norm=maximum,work=work,settings=settings)
    bias=results[0]-ctx['reference']
    return dict(corrected=dict(real=results[0].real,imag=results[0].imag),reference=dict(real=ctx['reference'].real,imag=ctx['reference'].imag),
        signed_bias=dict(real=bias.real,imag=bias.imag),allowance=allowance,log_B=log_B,
        Horner_forward_error=discrepancy,raw_corrected_closure=raw_closure,
        raw=dict(real=results[2].real,imag=results[2].imag),maximum_norm=maximum,work=work,scalar=scalar(ctx,c),
        preparation_hash=p.preparation_hash,reference_rotated=False,normalization_certified=False)


def shots(sig,epsilon):
    from trottertracks.resource_applicability.ax2b_numerical_accounting import u_aware_shots
    result=u_aware_shots({a:abs(sig['signed_bias'][a]) for a in ['real','imag']},
        {a:sig['allowance'][a] for a in ['real','imag']},epsilon=float(epsilon),
        log_B_upper=sig['log_B']+sig['allowance']['log_B_margin'],alpha_axis=.025,
        evidence_kind='EMPIRICAL',evidence_ref='NEW_COHORT_SAME_INPUT_SIGNAL')
    for a in result['axes'].values():
        if a['status']=='ELIGIBLE' and (a['u_over_headroom'] is None or a['u_over_headroom']>.01):
            a['status']='UNDETERMINED';a['shots']=None
    result['usable']=all(a['status']=='ELIGIBLE' and a['shots'] is not None for a in result['axes'].values())
    result['N_cos']=result['axes']['real']['shots'] if result['usable'] else None
    result['N_sin']=result['axes']['imag']['shots'] if result['usable'] else None
    return result


def evolution(ctx,c,stage,index):
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    p=ctx['preps'][c['prefix']];n=ctx['ham'].n_qubits
    if c['method']!='B2':
        return native.build_deterministic_native(p.deterministic_blocks,num_system_qubits=n,T=.8,q=c['q'],
            formula='2nd',scalar=scalar(ctx,c),control_policy='symmetric_directional',max_instructions=2**60),[]
    from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    tolerance=_explicit_cutoff_tolerance(p.exact_rte_lambda_r*(.8/c['q']/c['r']),c['K'])
    requests=[make_df_partial_s2_step_request(p,step_time=.8/c['q'],rte_steps=c['r'],
        finite_taylor_order=c['K'],truncation_tolerance=tolerance,seed=seed(c['system'],c['id'],stage,index,outer),
        controlled=True,ancilla_qubit=n) for outer in range(c['q'])]
    orders=[event.taylor_order for request in requests for event in request.rte_occurrence.events]
    return native.partial_native_from_step_requests(requests,control_policy='symmetric_directional',max_instructions=2**60),orders


def compile_one(task):
    stage,index,validate=task
    import numpy as np
    from recomputed_s2_phase_guard import transpile as guarded_transpile
    def transpile(circuit,**settings):
        original_phase=float(circuit.global_phase)
        bound=circuit.copy();bound.global_phase=0.
        result=guarded_transpile(bound,**settings)
        result.global_phase+=original_phase
        result.metadata['input_global_phase_preserved_once']=original_phase
        return result
    from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
    started=time.monotonic();evo,orders=evolution(CTX,CANDIDATE,stage,index)
    costs=[];checks=[]
    for axis in ['cosine','sine']:
        wrapper=build_native_hadamard_wrapper(evo,axis=axis,include_measurement=True,max_instructions=2**60)
        compiled=transpile(wrapper,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
        costs.append(int(compiled.count_ops().get('rz',0)))
        if validate:
            from qiskit_aer import AerSimulator
            from qiskit import QuantumCircuit
            backend=AerSimulator(method='statevector',max_parallel_threads=1)
            initial=np.zeros(1<<(CTX['ham'].n_qubits+1),complex);initial[:len(CTX['qstate'])]=CTX['qstate']
            from trotterlib.df_trotter.circuit import simulate_statevector
            vectors=[simulate_statevector(wrapper.remove_final_measurements(inplace=False),initial)]
            circ=compiled.remove_final_measurements(inplace=False)
            qc=QuantumCircuit(circ.num_qubits)
            # Aer set_statevector overwrites an initial circuit global phase.
            # Consume the exact metadata once in the supplied initial vector;
            # no phase is inferred from either output state.
            qc.set_statevector(initial*np.exp(1j*float(circ.global_phase)))
            body=circ.copy();body.global_phase=0.;qc.compose(body,inplace=True);qc.save_statevector()
            vectors.append(np.asarray(backend.run(qc).result().get_statevector()))
            error=float(np.linalg.norm(vectors[0]-vectors[1]));checks.append(error)
            require(error<1e-7,'COMPILED_WRAPPER_STATEVECTOR_MISMATCH:'+str(error))
    return dict(index=index,seed=seed(CANDIDATE['system'],CANDIDATE['id'],stage,index),
        cosine=costs[0],sine=costs[1],orders=orders,validation_errors=checks,
        compiler_phase_guard=compiled.metadata['H2_phase_guard'],seconds=time.monotonic()-started)


def workers(tasks):
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    idle=max(1,len(os.sched_getaffinity(0))-math.ceil(os.getloadavg()[0]))
    return max(1,min(tasks,idle,available//(1024**3)))


def sample(ctx,c,stage,n,validate=True):
    global CTX,CANDIDATE
    CTX,CANDIDATE=ctx,c
    tasks=[(stage,i,validate and i==0) for i in range(n)]
    nw=workers(n)
    with mp.get_context('fork').Pool(nw) as pool:
        values=list(pool.imap_unordered(compile_one,tasks,chunksize=1))
    return sorted(values,key=lambda x:x['index']),nw


def stats(values,shot):
    import numpy as np
    from scipy.stats import t
    total=[shot['N_cos']*r['cosine']+shot['N_sin']*r['sine'] for r in values]
    n=len(total);mean=F(sum(total),n)
    variance=(sum((F(x)-mean)**2 for x in total)/(n-1)) if n>1 else F()
    se=math.sqrt(float(variance)/n);half=float(t.ppf(.975,n-1))*se if n>1 else 0.
    return dict(n=n,mean=float(mean),mean_exact=str(mean),variance=float(variance),SE=se,
        nominal_95_halfwidth=half,relative_halfwidth=half/float(mean),target_met=half<=.01*float(mean),
        interval_kind='NOMINAL_STUDENT_T_NOT_RARE_EVENT_GUARANTEE',population_guarantee=False,
        cosine_mean=float(F(sum(x['cosine'] for x in values),n)),sine_mean=float(F(sum(x['sine'] for x in values),n)))


def plan_n(values,shot_rows):
    from scipy.stats import t
    n=128;pilot_stats={}
    for eps,shot in shot_rows.items():
        if not shot['usable']:continue
        s=stats(values,shot);pilot_stats[eps]=s
        required=math.ceil((float(t.ppf(.975,127))*math.sqrt(s['variance'])/(.01*s['mean']))**2)
        n=max(n,required)
    return n,pilot_stats


def candidate_stages(system,cid):
    ctx=context(system);c=next(x for x in read(BASE/'candidates.json') if x['system']==system and x['id']==cid)
    inp=unit_path(system,'SYSTEM','input_preparation');binding={'context':sha(inp/'context.pkl'),'candidate':digest(c)}
    with unit(system,cid,'theory_counter',binding,c) as out:
        units,law=theory(ctx,c);save(out/'result.json',dict(units=units,law=law,preparation_hash=ctx['preps'][c['prefix']].preparation_hash,scalar=scalar(ctx,c)))
    with unit(system,cid,'signal',binding,c) as out:
        sig=signal(ctx,c);save(out/'result.json',sig)
    shot_rows={}
    for eps in c['epsilons']:
        with unit(system,cid,'shot_accounting',dict(signal=digest(sig)),epsilon=eps) as out:
            shot_rows[eps]=shots(sig,eps);save(out/'result.json',shot_rows[eps])
    if not any(s['usable'] for s in shot_rows.values()):return dict(status='NO_ELIGIBLE_PRECISION',system=system,candidate=cid)
    if c['method']!='B2':
        with unit(system,cid,'compiled_cost',binding,c) as out:
            global CTX,CANDIDATE
            CTX,CANDIDATE=ctx,c
            row=compile_one(('deterministic',0,True));save(out/'result.json',dict(values=[row],deterministic=True))
        return dict(status='COMPLETE',system=system,candidate=cid)
    with unit(system,cid,'MC_pilot',binding,dict(c,n=128,independent_of_main=True)) as out:
        values,nw=sample(ctx,c,'MC_pilot',128);save(out/'result.json',dict(values=values,workers=nw,n=128))
    n,pilot_stats=plan_n(values,shot_rows)
    main_plan=BASE/'main_plans'/system/(cid+'.json')
    save(main_plan,dict(n=n,pilot_statistics=pilot_stats,
        target_relative_nominal_95_halfwidth=.01,automatic_extension=False,pilot_in_main=False))
    with unit(system,cid,'MC_main',dict(binding,pilot=digest(values),main_plan=sha(main_plan)),
              dict(c,n=n,pilot_pooled=False,independent_main=True)) as out:
        values,nw=sample(ctx,c,'MC_main',n,validate=False)
        summaries={eps:stats(values,s) for eps,s in shot_rows.items() if s['usable']}
        law=read(unit_path(system,cid,'theory_counter')/'result.json')['law']
        observed={str(k):sum(k in r['orders'] for r in values) for k in [2,4]}
        strata={}
        for k in [2,4]:
            selected=[v for v in values if k in v['orders']]
            strata[str(k)]=dict(observed=len(selected),
                observed_unconditional_cosine_cost_contribution=sum(v['cosine'] for v in selected)/n,
                observed_conditional_cosine_mean=None if not selected else sum(v['cosine'] for v in selected)/len(selected),
                probability_absent_in_main=(1-law['probability_any_positive_order' if k==2 else 'probability_any_order4'])**n,
                unobserved_cost_contribution='UNKNOWN_NO_COMPILED_POPULATION_BOUND' if not selected else 'SAMPLE_ESTIMATE_ONLY')
        rare=dict(law=law,trajectories_with_order=observed,strata=strata,unobserved_positive_order_possible=True,
            ordinary_MC_interval_is_population_guarantee=False,
            warning='Unobserved orders retain positive probability. No rigorous compiled-cost bound inferred from the native counter or sample variance.')
        save(out/'result.json',dict(values=values,workers=nw,n=n,statistics=summaries,rare_event_diagnostics=rare,
            pilot_pooled=False,automatic_additional_sampling=False))
    return dict(status='COMPLETE',system=system,candidate=cid,n=n,target_met=all(v['target_met'] for v in summaries.values()))


def run_system(system):
    require(system!='H2','H2_FINISHED_AND_FAILED_UNITS_MUST_NOT_BE_RETRIED')
    try:
        build_input(system)
    except BaseException as exc:
        return dict(system=system,status='INPUT_FAILED_NO_RETRY',error=str(exc))
    rows=[]
    for c in sorted(read(BASE/'candidates.json'),key=lambda c:(c['role']!='PR',c['id'])):
        if c['system']!=system:continue
        try:
            r=candidate_stages(system,c['id']);rows.append(r);print(json.dumps(r),flush=True)
        except BaseException as exc:
            r=dict(system=system,candidate=c['id'],status='FAILED_NO_RETRY',error=str(exc));rows.append(r);print(json.dumps(r),flush=True)
    return dict(system=system,status='SYSTEM_FINISHED_MANDATORY_STOP',candidates=rows)
