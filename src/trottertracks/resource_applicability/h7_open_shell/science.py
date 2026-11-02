"""Neutral H7 sparse numerical adapter, reusing the frozen H5 native pipeline."""
import math
import numpy as np
from scipy.sparse.linalg import eigsh
from trotterlib.df_hamiltonian import df_linear_operator, _hartree_fock_initial_vector
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
from ..h5_open_shell.science import (preparations, validate_primitives, cost_group,
    build_group, compiler, tail_identity, complex_record, _TrackedBudget)
from ..h5_open_shell.sector import sector_for, SpinPrimitive
from ..h6_matched_port_v1 import compact_signal
from ..h8_sparse_reference_v1 import recurrence_paths
from ..h6_matched_accounting_v1 import empirical_allowance, precision_rows
from ..h6_matched_contract_v1 import plan as old_plan
from ..ax2b_h6_controller import finite_scale_guard
from ..ax2b_h4_science_v5 import primitive_sector_certificate
from .reference import SparseReference, bounded_solver
from .contract import registry


def solve_state(ham,calls,*,operator_factory=None,solver=eigsh):
    sector=sector_for(7,4,3)
    if ham.n_qubits!=14:raise ValueError('H7_SYSTEM_SIZE')
    primitive_sector_certificate(ham,sector)
    # Frozen production action, JIT-compiled with a private cache and one thread.
    op=(df_linear_operator(ham,sector,backend='numba',num_threads=1,block_chunk_size=1)[0]
        if operator_factory is None else operator_factory(ham,sector))
    initial=_hartree_fock_initial_vector(sector)
    if initial is None:raise ValueError('H7_INITIAL_DETERMINANT')
    calls.take('state_solve')
    (values,vectors),bounded=bounded_solver(lambda v:op@v,1225,initial,calls,solver=solver)
    x=np.asarray(vectors[:,0],dtype=complex).copy();norm=float(np.linalg.norm(x))
    if not math.isfinite(norm) or norm<=0:raise ValueError('H7_SOLVER_NORM')
    x/=norm;pivot=int(np.argmax(np.abs(x)));x*=np.exp(-1j*np.angle(x[pivot]))
    residual=float(np.linalg.norm(bounded@x-float(values[0])*x))
    if residual>1e-9+1e-10*max(1.,abs(float(values[0]))):raise ValueError('H7_SOLVER_RESIDUAL')
    reference=SparseReference(ham,sector,calls);checks=reference.validate(x,op)
    checks['production_backend']='numba/thread1/chunk1' if operator_factory is None else 'injected artificial provider'
    full=np.zeros(1<<14,dtype=complex);full[sector.basis_indices]=x
    return sector,x,full,checks,reference


class _TailBudget:
    def __init__(self,calls):self.calls=calls
    def take(self,_):self.calls.take('tail_matvec')


def panel_signals(ham,sector,state,reference,checks,prepared,calls):
    p=(len(ham.lambdas)+3)//4
    ref=checks['reference']  # Already validated at the same fixed T=.8.
    truncated_ref,truncated_check=reference.pair(state,.8,prefix=p)
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
                tails[key]=reference.operator(tail_from=p,identity=prep.extracted_identity_coefficient,
                    scale=prep.exact_rte_lambda_r,budget=_TailBudget(calls))
            pair=tails[key]
            tail=lambda v:reference.raw_action(v,tail_from=p,identity=prep.extracted_identity_coefficient,
                scale=prep.exact_rte_lambda_r)
        a=compact_signal(state,actions[key],tail,cell=cell,T=.8,scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
            lambda_r=prep.exact_rte_lambda_r,budget=_TrackedBudget(calls))
        # Independent forward polynomial path, including exact-tail comparison.
        # The frozen forward path owns its local ActionBudget. Reserve its exact
        # deterministic counts before calling it; the CSR tail wrapper charges
        # every polynomial/exponential action as it executes.
        from ..ax2b_h6_contract import primitive_time_schedule
        schedule=primitive_time_schedule(cell,T=.8)['ordinary_one_outer_step']
        paths=3 if cell['R'] else 1
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
