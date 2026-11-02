"""H5 numerical/native adapter, called only on caller-supplied admitted data.

Reuses the frozen DF/PF/RTE law, compiler and native lowering unchanged. The
unequal-spin action is the new part. Artificial tests never call real inputs.
"""
import gc
import math
import time
import resource

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

from trotterlib.df_hamiltonian import df_linear_operator, _hartree_fock_initial_vector
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, make_df_partial_s2_step_request
from trotterlib.rte import CompilerSettings
from trotterlib.rte_compiled_cost import transpile_and_measure_cost, compiler_settings_hash
from trotterlib.df_trotter.circuit import simulate_statevector
from ..ax2b_h6_controller import bounded_solver, finite_scale_guard
from ..ax2b_h4_science_v5 import checked_basis_bridge, primitive_sector_certificate
from ..ax2b_native_df_v5 import build_deterministic_native, partial_native_from_step_requests, build_native_hadamard_wrapper, make_native_block_action
from ..h6_matched_port_v1 import compact_signal
from ..h8_sparse_reference_v1 import OccupationCSR, counted_operator, recurrence_paths
from ..h6_matched_accounting_v1 import empirical_allowance, precision_rows
from ..h6_matched_contract_v1 import plan as old_plan
from ..ax2a_state_action import ActionBudget
from ..h8_resource_map_optimization_v1 import hashing, validate_control
from ..ax2b_molecular_ports_v3 import hadamard_expectation
from ..ax2b_h6_contract import primitive_time_schedule
from .sector import sector_for, SpinPrimitive, validate_state, independent_df_matrix
from .contract import registry, digest, COMPILER_HASH


def complex_record(z):return dict(real=float(z.real),imag=float(z.imag))


def solve_state(ham,calls):
    sector=sector_for(5,3,2)
    if ham.n_qubits!=10:raise ValueError('H5_SYSTEM_SIZE')
    primitive_sector_certificate(ham,sector)
    op,_=df_linear_operator(ham,sector,backend='python')
    initial=_hartree_fock_initial_vector(sector)
    if initial is None:raise ValueError('INITIAL_DETERMINANT')
    calls.take('state_solve')
    (values,vectors),bounded,_=bounded_solver(lambda x:op@x,100,initial,calls,solver=eigsh)
    x=np.asarray(vectors[:,0],dtype=complex)
    if not np.isfinite(x).all() or not np.isfinite(values).all():raise ValueError('SOLVER_NONFINITE')
    x/=np.linalg.norm(x);pivot=int(np.argmax(np.abs(x)));x*=np.exp(-1j*np.angle(x[pivot]))
    # Includes a post-solve residual under the same reserved matvec budget.
    residual=float(np.linalg.norm(bounded@x-float(values[0])*x))
    if residual>1e-9+1e-10*max(1.,abs(float(values[0]))):raise ValueError('SOLVER_RESIDUAL')
    calls.take('reference_matvec',100);calls.take('reference_matrix')
    checks,matrix=validate_state(ham,sector,x)
    full=np.zeros(1024,dtype=complex);full[sector.basis_indices]=x
    return sector,x,full,checks,matrix


def preparations(ham):
    L=len(ham.lambdas);p=(L+3)//4
    # Only three preparations; B0/I and K2/K4 share immutable source data.
    return dict(discard=_prepare_discard(ham,p),partial=_prepare(ham,p),full=_prepare(ham,L))


def tail_identity(ham,prefix):
    # Full-Fock identity, not sector mean or absolute-lambda replacement.
    return float(sum(w*(np.trace(g).real**2+np.trace(g@g).real)/4
        for w,g in zip(ham.lambdas[prefix:],ham.g_matrices[prefix:],strict=True)))


def validate_primitives(ham,sector,full,state,prepared,calls):
    qindices,qstate=checked_basis_bridge(ham,sector,full,state,1e-12)
    qindices=np.asarray(qindices,dtype=int)
    rng=np.random.default_rng(593102);probes=[state]
    for _ in range(2):
        v=rng.normal(size=sector.dimension)+1j*rng.normal(size=sector.dimension)
        probes.append(v/np.linalg.norm(v))
    times=set()
    for c in registry(len(ham.lambdas)):
        if c['method']=='B0I':continue
        times.update((i,t) for i,t in primitive_time_schedule(c,T=.8)['unique_primitive_times'])
    worst=0.;rows=[];spectra={};native_actions={}
    for i,t in sorted(times):
        block=prepared['full'].deterministic_blocks[i]
        matrix=ham.one_body if i==0 else ham.g_matrices[i-1]
        if i not in spectra:
            spectra[i]=SpinPrimitive(matrix,sector,weight=None if i==0 else block.lam)
            native_actions[i]=make_native_block_action(block,max_instructions=4000000)
        action=spectra[i];native=native_actions[i]
        maximum=0.
        for probe in probes:
            calls.take('primitive');lift=np.zeros_like(qstate);lift[qindices]=probe
            evolved=native(lift,t);expected=np.zeros_like(qstate);expected[qindices]=action(probe,t)
            error=float(np.linalg.norm(evolved-expected));maximum=max(maximum,error)
            finite_scale_guard(log_B=0.,intermediate_norm=float(np.linalg.norm(evolved)),absolute_discrepancy=error)
        rows.append(dict(primitive=i,time=t,max_error=maximum));worst=max(worst,maximum)
    return dict(rows=rows,probe_count=3,max_error=worst,full_system_native=True,
        intermediate_sector_projection=False,formal_certificate=False)


def panel_signals(ham,sector,state,matrix,checks,prepared,calls):
    oracle=OccupationCSR(sector.basis_indices,ham.n_qubits);p=(len(ham.lambdas)+3)//4
    calls.take('reference_matrix')
    truncated=independent_df_matrix(ham.select_blocks(tuple(range(p))),sector)
    ref=checks['reference']  # Already validated at the same fixed T=.8.
    truncated_ref=complex(np.vdot(state,expm(-.8j*truncated)@state))
    actions={};tails={};records=[];c=tail_identity(ham,p);settings=old_plan()
    settings['epsilons']=[.05,.01,.005,.001,.0005]
    for cell in registry(len(ham.lambdas)):
        calls.take('signal')
        if cell['method']=='B0I':
            b0=next(r for r in records if r['cell']['method']=='B0')
            record=dict(b0);phase=np.exp(-.8j*c)
            record['cell']=cell;record['signals']={k:complex_record(complex(v['real'],v['imag'])*phase) for k,v in b0['signals'].items()}
            corrected=complex(record['signals']['corrected']['real'],record['signals']['corrected']['imag'])
            record['error_decomposition']=dict(total_signed=complex_record(corrected-ref))
            # Conservative rotation of the two inherited axis allowances.
            absolute_terms=sum(abs(w)*(abs(np.trace(g))**2+abs(np.trace(g@g)))/4
                for w,g in zip(ham.lambdas[p:],ham.g_matrices[p:],strict=True))
            u=dict(b0['allowance']);extra=10*np.finfo(float).eps*(1+absolute_terms*.8)
            u['real']=u['imag']=(abs(phase.real)+abs(phase.imag))*max(u['real'],u['imag'])+extra
            record['allowance']=u;record['precision_rows']=precision_rows(record,settings);records.append(record);continue
        key='discard' if cell['method']=='B0' else ('full' if cell['method']=='B1' else 'partial')
        prep=prepared[key]
        if key not in actions:
            actions[key]=[SpinPrimitive(ham.one_body if isinstance(b,DFDeterministicOneBodySpec) else ham.g_matrices[b.original_fragment_index],
                sector,weight=None if isinstance(b,DFDeterministicOneBodySpec) else b.lam) for b in prep.deterministic_blocks]
        tail=None;pair=None
        if cell['R']:
            if prep.exact_rte_lambda_r<=0:raise ValueError('EMPTY_PR_TAIL_NO_AUTOMATIC_FALLBACK')
            if key not in tails:
                dense=(matrix-truncated-prep.extracted_identity_coefficient*np.eye(sector.dimension))/prep.exact_rte_lambda_r
                tails[key]=dense
            dense=tails[key]
            tail=lambda v:dense@v
            operator,trace=counted_operator(sector.dimension,tail,float(np.trace(dense).real),budget=_ReferenceBudget(calls))
            pair=(operator,trace)
        a=compact_signal(state,actions[key],tail,cell=cell,T=.8,scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
            lambda_r=prep.exact_rte_lambda_r,budget=_TrackedBudget(calls))
        # Independent forward polynomial path, including exact-tail comparison.
        # The frozen forward path owns its local ActionBudget. Reserve its exact
        # statically known polynomial/deterministic counts before calling it.
        from ..ax2b_h6_contract import primitive_time_schedule
        schedule=primitive_time_schedule(cell,T=.8)['ordinary_one_outer_step']
        paths=3 if cell['R'] else 1
        calls.take('tail_matvec',2*cell['q']*cell['r']*(cell['K']+1) if cell['R'] else 0)
        calls.take('deterministic_action',paths*cell['q']*len(schedule))
        b=recurrence_paths(state,actions[key],pair,cell=cell,T=.8,scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
            lambda_r=prep.exact_rte_lambda_r,maximum_tail=500000)
        errors=[float(np.linalg.norm(a['vectors'][k]-b['vectors'][k])) for k in a['vectors']]
        errors.append(checks.get('primitive_max_error',0.))
        finite_scale_guard(log_B=a['log_B'],intermediate_norm=max(a['intermediate_norm_max'],b['intermediate_norm_max']),absolute_discrepancy=max(errors))
        closure=float(np.linalg.norm(a['vectors']['corrected']-math.exp(a['log_B'])*a['vectors'].get('raw',a['vectors']['corrected'])))
        u=empirical_allowance(reference_error=checks['action_difference']+checks['reference_split_difference'],path_errors=errors,
            normalization_error=abs(a['log_B']-b['log_B']),raw_closure=closure,
            maximum_norm=max(a['intermediate_norm_max'],b['intermediate_norm_max']),
            work=sum(a['counts'].values())+sum(b['counts'].values()),settings=settings['numerical'])
        signals={k:complex_record(v) for k,v in a['signals'].items()}
        signals.update({k:complex_record(v) for k,v in b['signals'].items() if k not in signals})
        if cell['method']=='B0':signals['exact_truncated']=complex_record(truncated_ref)
        record=dict(cell=cell,signals=signals,reference=complex_record(ref),log_B=a['log_B'],b=a['b'],allowance=u,
            error_decomposition=dict(total_signed=complex_record(a['signals']['corrected']-ref)),formal_certificate=False)
        record['precision_rows']=precision_rows(record,settings);records.append(record)
    return records


class _ReferenceBudget:
    def __init__(self,calls):self.calls=calls
    def take(self,_):self.calls.take('tail_matvec')


class _TrackedBudget(ActionBudget):
    def __init__(self,calls):super().__init__(500000,500000);self.calls=calls
    def tail(self):self.calls.take('tail_matvec');super().tail()
    def deterministic(self):self.calls.take('deterministic_action');super().deterministic()


def compiler():
    import qiskit
    c=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
    if compiler_settings_hash(c)!=COMPILER_HASH:raise ValueError('COMPILER_IDENTITY')
    return c


def build_group(ham,prepared,task,calls):
    c=task['cell'];p=prepared['discard' if c['method'] in ('B0','B0I') else ('full' if c['method']=='B1' else 'partial')]
    if c['R']:
        calls.take('trajectory');calls.take('sampling_call');calls.take('occurrence',c['R'])
        request=make_df_partial_s2_step_request(p,step_time=.8,rte_steps=2,finite_taylor_order=c['K'],
            seed=task['seed'],controlled=True,ancilla_qubit=ham.n_qubits)
        events=[e.to_dict() for e in request.rte_occurrence.events]
        evolution=partial_native_from_step_requests([request],control_policy='symmetric_directional',max_instructions=4000000)
    else:
        events=None;scalar=p.constant_coefficient+(tail_identity(ham,c['prefix']) if c['method']=='B0I' else 0.)
        evolution=build_deterministic_native(p.deterministic_blocks,num_system_qubits=ham.n_qubits,T=.8,q=1,
            formula=c['formula'],scalar=scalar,control_policy='symmetric_directional',max_instructions=4000000)
    return evolution,events


def cost_group(ham,sector,full,state,prepared,task,calls,*,expected_signal=None,compile_fn=transpile_and_measure_cost):
    started=time.monotonic();_,qstate=checked_basis_bridge(ham,sector,full,state,1e-12)
    with hashing() as hashes:
        evolution,events=build_group(ham,prepared,task,calls)
        validation=None
        if task['validate']:
            validation,_=validate_control(evolution,qstate,calls=calls,simulate=simulate_statevector,
                expectation=hadamard_expectation,expected_signal=expected_signal if not task['cell']['R'] else None)
            finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=validation['max_error'])
        settings=compiler();rows=[]
        for axis in ('cosine','sine'):
            wrapper=build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=4000000)
            calls.take('compile');cost=compile_fn(wrapper,settings)
            metrics={k:int(getattr(cost,v)) for k,v in dict(RZ='rz_count',CX='cx_count',RZ_depth='rz_depth',
                CX_depth='cx_depth',depth='total_depth',size='circuit_size').items()}
            if any(v<0 for v in metrics.values()) or cost.compiler_settings_hash!=COMPILER_HASH:raise ValueError('COST_IDENTITY')
            rows.append(dict(axis=axis,metrics=metrics,compiler_hash=cost.compiler_settings_hash,
                fingerprint=cost.actual_circuit_fingerprint,event_digest=digest(events)))
            del cost,wrapper;gc.collect()
        return dict(task=task,events=events,event_digest=digest(events),rows=rows,validation=validation,
            calls_attempted=calls.used,wall_seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            population_mean_certified=False)
