"""Grant-only H6 port. Frozen native lowering is reused, not reimplemented.

No computation occurs at import. Do not import this module before launch limits.
Independent sector oracles are binary64 engineering checks, not error certificates.
"""
from __future__ import annotations
import math
import os
import resource
import statistics
import time
import numpy as np
from scipy.linalg import expm, eigh
from trotterlib.df_hamiltonian import DFHamiltonian, df_linear_operator
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
from .ax2b_molecular_ports_v3 import MolecularPort, actual_bounds, check_bounds, occupation_column, complex_record
from .ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_h6_controller import bounded_sector_matrix, finite_scale_guard
from .ax2b_independent_reference import occupation_df_matrix
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_stage_validation_v2 import canonical_cell, traced_signal
from .ax2a_state_action import ActionBudget
from .ax2b_h6_pilot_contract_v1 import verify_parent, safe_path, read_json, resources
from .ax2a_preparation import digest


def reference_pair(matrix, state, T):
    if matrix.shape != (len(state),len(state)) or len(state)>400 or not np.isfinite(matrix).all():
        raise ValueError('SECTOR_REFERENCE_LAYOUT')
    if np.linalg.norm(matrix-matrix.conj().T)>1e-10:raise ValueError('REFERENCE_HERMITICITY')
    a=expm(-1j*T*matrix)@state
    w,v=eigh(matrix);b=v@(np.exp(-1j*T*w)*(v.conj().T@state))
    difference=float(np.linalg.norm(a-b))
    if difference>1e-10:raise ValueError('REFERENCE_EXPM_EIGH')
    return complex(np.vdot(state,a)),{'state_difference':difference,'signal_difference':float(abs(np.vdot(state,a-b)))}


def recurrence_paths(state, actions, tail_matrix, *, cell, T, scalar, lambda_r, maximum_tail):
    """Forward Taylor recurrence and exact tail, independently of native Horner.

    One tail matrix and one tail expm per cell; primitive cache is replaceable.
    Full vectors are never renormalized. Separate before-call counters for this oracle.
    """
    psi=np.asarray(state,dtype=complex).copy();cell=canonical_cell(cell)
    if psi.ndim!=1 or len(psi)>400 or not np.isfinite(psi).all() or abs(np.linalg.norm(psi)-1)>1e-12:
        raise ValueError('ORACLE_SAVED_STATE')
    random=cell['method'] in ('B2','B3');s=primitive_time_schedule(cell,T=T)
    tau=lambda_r*T/cell['q']/cell['r'] if random else 0.
    if random and (not math.isfinite(tau) or tau<=0 or tail_matrix is None):raise ValueError('ORACLE_TAIL')
    # Independent closed normalization; native uses finite_rte_distribution.
    b=sum(abs(tau)**k/math.factorial(k)*math.hypot(1.,tau/(k+1)) for k in range(0,cell['K']+1,2)) if random else 1.
    log_B=cell['q']*cell['r']*math.log(b) if random else 0.
    finite_scale_guard(log_B=log_B,intermediate_norm=1.,absolute_discrepancy=0.)
    if random and (tail_matrix.shape!=(len(psi),len(psi)) or not np.isfinite(tail_matrix).all()
                   or np.linalg.norm(tail_matrix-tail_matrix.conj().T)>1e-10):raise ValueError('ORACLE_TAIL_MATRIX')
    exact=expm(-1j*tau*tail_matrix) if random else None
    budget=ActionBudget(maximum_tail,100000)
    norms=[];signals={};vectors={}
    def checked(v):
        if v.shape!=psi.shape or not np.isfinite(v).all():raise ValueError('NONFINITE_ORACLE_STAGE')
        norm=float(np.linalg.norm(v));norms.append(norm)
        finite_scale_guard(log_B=log_B,intermediate_norm=norm,absolute_discrepancy=0.)
        return v
    for path in ('corrected','raw','exact_tail') if random else ('corrected',):
        current=psi.copy()
        for _ in range(cell['q']):
            sequence=s['ordinary_one_outer_step']
            if random:
                for i,t in sequence[:len(actions)]:
                    budget.deterministic();current=checked(actions[i](current,t))
                for _ in range(cell['r']):
                    if path=='exact_tail':current=checked(exact@current)
                    else:
                        power=current.copy();total=current.copy()
                        for degree in range(1,cell['K']+2):
                            budget.tail();power=checked((-1j*tau/degree)*(tail_matrix@power));total=checked(total+power)
                        current=checked(total/b if path=='raw' else total)
                sequence=sequence[len(actions):]
            for i,t in sequence:
                budget.deterministic();current=checked(actions[i](current,t))
            current=checked(np.exp(-1j*scalar*T/cell['q'])*current)
        signals[path]=complex(np.vdot(psi,current));vectors[path]=current
    return {'signals':signals,'vectors':vectors,'log_B':log_B,'b':b,'intermediate_norm_max':max([1.]+norms),
            'counts':{'tail':budget.tail_matvecs,'deterministic':budget.deterministic_actions}}


def differences(signals, reference, method):
    """Signed complex decomposition; no absolute-value additivity is implied."""
    if method=='B0':
        parts={'discard':signals['exact_truncated']-reference,
               'PF':signals['corrected']-signals['exact_truncated']}
    elif method in ('B2','B3'):
        parts={'outer_PF':signals['exact_tail']-reference,
               'finite_RTE':signals['corrected']-signals['exact_tail']}
    else:parts={'PF':signals['corrected']-reference}
    total=signals['corrected']-reference
    return {'signed_parts':{k:complex_record(v) for k,v in parts.items()},
        'absolute_parts':{k:float(abs(v)) for k,v in parts.items()},'total_signed':complex_record(total),
        'total_absolute':float(abs(total)),'complex_closure_discrepancy':float(abs(sum(parts.values())-total)),
        'absolute_additivity_claim':False,'certified':False}


class H6PilotPort(MolecularPort):
    def __init__(self, root, manifest, writer, progress, *, started=None):
        super().__init__(root,manifest,writer,started=started)
        self.progress=progress;self.wrapper_records=[]

    def phase(self,name):
        self.progress.phase(name)

    def observe(self,name,value):
        if name.startswith('wrapper_'):
            self.wrapper_records.append(value)
        if not name.startswith(('progress_','phase_')):
            self.progress.update(last_completed_record=name,calls_attempted=dict(self.calls.used),
                correctness_completed=self.completed+(1 if name.endswith('_correctness.json') else 0),
                compiled_wrappers=self.compiled+(1 if name.startswith('wrapper_') else 0))

    def setup(self):
        self.phase('input_reference');self.progress.update(point='verify_saved_snapshot')
        identity=verify_parent(self.root)
        if identity!=self.manifest['input_identity']:raise ValueError('PILOT_PARENT_CHANGED')
        self.ham,self.sector,full,state,metadata,_=load_h6_snapshot(
            safe_path(self.root,identity['snapshot_path']),read_json(self.root/identity['snapshot_receipt_path']),
            read_json(self.root/identity['df_receipt_path']))
        certificate=primitive_sector_certificate(self.ham,self.sector)
        self.qindices,self.qstate=checked_basis_bridge(self.ham,self.sector,full,state,1e-12)
        # Preserve exactly the designated saved amplitudes; no second normalization.
        self.state=state.copy();self.saved_state=state.copy()
        self.progress.update(point='prepare_native_representation')
        self.preps={c['id']:(_prepare_discard if c['method']=='B0' else _prepare)(self.ham,c['prefix']) for c in self.cells}
        bounds=actual_bounds(self.cells,self.preps,T=self.plan['T']);check_bounds(bounds,self.caps)
        cov=self.plan['coverage']
        if bounds['primitive_actions']!=cov['primitive_actions'] or digest([{'cell_id':r['cell_id'],'schedule':r['schedule']} for r in bounds['cells']])!=digest(cov['cells']):
            raise ValueError('ACTUAL_PRIMITIVE_COVERAGE')
        records=[]
        for c in self.cells:
            p=self.preps[c['id']]
            if p.deterministic_fragment_indices!=tuple(range(c['prefix'])) or p.coefficient_atol!=0. or p.threshold_dropped_component_count!=0:
                raise ValueError('PREPARATION_ORDER_OR_CUTOFF')
            records.append({'cell_id':c['id'],'hamiltonian_hash':p.hamiltonian_hash,'preparation_hash':p.preparation_hash,
                'partition_hash':p.partition_hash,'deterministic_fragment_indices':p.deterministic_fragment_indices,
                'randomized_block_indices':p.randomized_block_indices,'exact_rte_lambda_r':p.exact_rte_lambda_r,
                'ranking_proxy_lambda_r':p.ranking_proxy_lambda_r,'extracted_identity':p.extracted_identity_coefficient,
                'constant':p.constant_coefficient,'coefficient_atol':p.coefficient_atol,
                'basis_hashes':[b.basis_hash for b in p.deterministic_blocks]})
        self.writer.write('actual_prepared_representation.json',{'preparations':records,'bounds':bounds,
                'created_before_any_action_sampling_or_circuit_build':True,'compiled_cost_claim':False})
        operator,_=df_linear_operator(self.ham,self.sector,**self.plan['matrix_free'])
        self.progress.update(point='bounded_sector_reference_columns')
        matrix,count=bounded_sector_matrix(lambda v:operator@v,self.sector.dimension,self.calls,
                         per_action_cap=self.caps['reference_matvec_per_action'])
        maximum=0.
        for i,index in enumerate(self.sector.basis_indices):
            maximum=max(maximum,float(np.linalg.norm(matrix[:,i]-occupation_column(self.ham,self.sector.basis_indices,int(index)))))
        finite_scale_guard(log_B=0.,intermediate_norm=float(np.linalg.norm(matrix)),absolute_discrepancy=maximum)
        self.reference,pair=reference_pair(matrix,self.state,self.plan['T']);del matrix,operator
        self.writer.write('input_reference.json',{'metadata':metadata,'sector_certificate':certificate,
            'independent_occupation_all_columns_error':maximum,'reference_matvecs':count,'reference':complex_record(self.reference),
            'expm_eigh_agreement':pair,'saved_state_norm_before':float(np.linalg.norm(state)),
            'state_renormalized':False,'reference_kind':'binary64 sector expm/eigh and independent occupation columns',
            'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED','ground_state_certified':False})
        import numba
        self.writer.write('parallel_resource_receipt.json',{'actual_affinity':sorted(os.sched_getaffinity(0)),
            'numba_actual_threads':numba.get_num_threads(),'assigned_resources':self.manifest['assigned_resources'],
            'matrix_free':self.plan['matrix_free'],'BLAS_threads':1,'gpu':False})
        if resources(sorted(os.sched_getaffinity(0)))!=self.manifest['assigned_resources'] or numba.get_num_threads()!=4:
            raise ValueError('PILOT_PARALLEL_RESOURCE')

    def cell_signal(self, raw, *, oracle=False):
        if oracle:raise ValueError('USE_INDEPENDENT_FORWARD_ORACLE')
        cell=canonical_cell(raw);p=self.preps[cell['id']];random=bool(cell['R'])
        lam=p.exact_rte_lambda_r if random else 0.;tail=None
        if random:
            from trotterlib.rte import finite_rte_distribution
            tau=lam*self.plan['T']/cell['q']/cell['r']
            b=finite_rte_distribution(tau,cell['K']).exact_finite_distribution
            finite_scale_guard(log_B=cell['q']*cell['r']*math.log(b),intermediate_norm=1.,absolute_discrepancy=0.)
            indices=p.randomized_block_indices
            h=DFHamiltonian(-p.extracted_identity_coefficient,np.zeros_like(self.ham.one_body),
                self.ham.lambdas[list(indices)].copy(),tuple(self.ham.g_matrices[i] for i in indices),{})
            op,_=df_linear_operator(h,self.sector,**self.plan['matrix_free']);tail=lambda v:(op@v)/lam
        budget=ActionBudget(self.caps['tail_matvecs_corrected_and_raw'].get(cell['method'],0),self.caps['deterministic_actions_per_cell'])
        value=traced_signal(self.state,self.actions(p),tail,cell=cell,T=self.plan['T'],
                scalar=p.constant_coefficient+p.extracted_identity_coefficient,lambda_r=lam,budget=budget)
        value['counts']={'tail':budget.tail_matvecs,'deterministic':budget.deterministic_actions}
        return value

    def oracle_signal(self,raw):
        c=canonical_cell(raw);p=self.preps[c['id']];random=bool(c['R']);tail=None
        if random:
            indices=p.randomized_block_indices
            tail=np.asarray(occupation_df_matrix(-p.extracted_identity_coefficient,np.zeros_like(self.ham.one_body),
                self.ham.lambdas[list(indices)],tuple(self.ham.g_matrices[i] for i in indices),self.sector.basis_indices),dtype=complex)/p.exact_rte_lambda_r
        result=recurrence_paths(self.state,self.actions(p,oracle=True),tail,cell=c,T=self.plan['T'],
            scalar=p.constant_coefficient+p.extracted_identity_coefficient,lambda_r=p.exact_rte_lambda_r if random else 0.,
            maximum_tail=self.caps['tail_matvecs_corrected_and_raw'].get(c['method'],0))
        self.oracle_cache=None
        if c['method']=='B0':
            truncated=np.asarray(occupation_df_matrix(self.ham.constant,self.ham.one_body,self.ham.lambdas[:c['prefix']],
                self.ham.g_matrices[:c['prefix']],self.sector.basis_indices),dtype=complex)
            result['signals']['exact_truncated'],result['truncated_expm_eigh']=reference_pair(truncated,self.state,self.plan['T'])
        return result

    def correctness(self):
        self.phase('correctness');self.progress.update(point='all_scheduled_primitive_probes');self.primitives()
        for raw in self.cells:
            start=time.monotonic();self.progress.update(point='cell_correctness',cell_id=raw['id'])
            actual=self.cell_signal(raw);oracle=self.oracle_signal(raw)
            errors={}
            for k,z in actual['signals'].items():
                error=abs(z-oracle['signals'][k]);errors[k]=float(error)
                finite_scale_guard(log_B=actual['log_B'],intermediate_norm=max(actual['intermediate_norm_max'],oracle['intermediate_norm_max']),absolute_discrepancy=error)
                finite_scale_guard(log_B=actual['log_B'],intermediate_norm=oracle['intermediate_norm_max'],
                    absolute_discrepancy=float(np.linalg.norm(actual['vectors'][k]-oracle['vectors'][k])))
            normalization_error=abs(actual['log_B']-oracle['log_B'])
            finite_scale_guard(log_B=actual['log_B'],intermediate_norm=1.,absolute_discrepancy=normalization_error)
            raw_error=None
            if 'raw' in actual['signals']:
                raw_error=float(abs(actual['signals']['raw']*math.exp(actual['log_B'])-actual['signals']['corrected']))
                finite_scale_guard(log_B=actual['log_B'],intermediate_norm=actual['intermediate_norm_max'],absolute_discrepancy=raw_error)
            signals=dict(oracle['signals']);signals.update(actual['signals'])
            self.writer.write(raw['id']+'_correctness.json',{'cell':raw,'signals':{k:complex_record(v) for k,v in signals.items()},
                'reference':complex_record(self.reference),'error_decomposition':differences(signals,self.reference,raw['method']),
                'trace':actual['trace'],'action_counts':actual['counts'],'oracle_action_counts':oracle['counts'],
                'independent_forward_signals':{k:complex_record(v) for k,v in oracle['signals'].items()},
                'oracle_signal_discrepancies':errors,'log_B':actual['log_B'],'b':actual['b'],
                'normalization_log_discrepancy':normalization_error,'B_raw_corrected_difference':raw_error,
                'truncated_expm_eigh_agreement':oracle.get('truncated_expm_eigh'),
                'intermediate_norm_max':actual['intermediate_norm_max'],'classical_cell_wall_seconds':time.monotonic()-start,
                'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
                'evidence_kind':'TECHNICAL_AGREEMENT','certified':False})
            self.completed+=1;self.oracle_cache=None

    def sampled_steps(self,cell,seed):
        self.progress.update(point='before_cost_trajectory',cell_id=cell['id'],seed=seed,calls_attempted=dict(self.calls.used))
        return super().sampled_steps(cell,seed)

    def costs(self):
        super().costs()
        if self.compiled!=36 or self.calls.used['trajectory']!=4 or self.calls.used['occurrence']!=8:
            raise ValueError('PILOT_COST_COUNTS')
        groups={}
        for r in self.wrapper_records:
            t=r['task'];groups.setdefault((t['cell_id'],t['control'],t['axis']),[]).append(r)
        summary=[]
        for (cell,control,axis),rows in sorted(groups.items()):
            metrics={}
            for name in rows[0]['metrics']:
                v=[r['metrics'][name] for r in rows]
                metrics[name]={'n':len(v),'min':min(v),'max':max(v),'mean':statistics.mean(v),
                    'sample_sd':statistics.stdev(v) if len(v)>1 else None}
            summary.append({'cell_id':cell,'control':control,'axis':axis,'metrics':metrics})
        self.writer.write('cost_summary.json',{'groups':summary,'primary':'symmetric_directional',
            'scope':'individual/pair engineering costs; random n=2 is not a precise population mean',
            'measurement_shots_sampled':False,'N':None,'G':None,'winner_claim':False})
