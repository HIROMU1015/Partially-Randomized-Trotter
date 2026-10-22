"""Numerical implementation, imported only by authorized, limited workers.

The saved Hamiltonian/state and frozen native lowering remain unchanged.
Compact stages and cached sector spectra replace repeated trace/expm work.
"""
from __future__ import annotations
import gc
import math
import resource
import time
import numpy as np
from scipy.linalg import eigh, expm
from trotterlib.df_hamiltonian import DFHamiltonian, df_linear_operator
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from trotterlib.rte import finite_rte_distribution, CompilerSettings
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from trotterlib.df_trotter.circuit import simulate_statevector
from .ax2b_h6_pilot_port_v1 import H6PilotPort, reference_pair, recurrence_paths, differences
from .ax2b_molecular_ports_v3 import actual_bounds, check_bounds, occupation_column, complex_record, hadamard_expectation
from .ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_h6_controller import bounded_sector_matrix, finite_scale_guard
from .ax2b_independent_reference import occupation_df_matrix
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_stage_validation_v2 import canonical_cell
from .ax2a_state_action import ActionBudget, finite_taylor_action
from .ax2b_native_df_v5 import build_deterministic_native, partial_native_from_step_requests, build_native_hadamard_wrapper
from .ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
from .ax2a_preparation import digest
from .h6_matched_contract_v1 import verify_parent, safe_path
from .h6_matched_accounting_v1 import empirical_allowance, precision_rows
import json


def compact_signal(state, actions, tail, *, cell, T, scalar, lambda_r, budget):
    """Same Horner, operation and scalar order as frozen traced_signal.

    Record norms/counts instead of serializing each 400-entry intermediate.
    No initial or intermediate renormalization is performed.
    """
    c = canonical_cell(cell)
    psi = np.asarray(state, dtype=np.complex128).copy()
    if psi.ndim != 1 or not np.isfinite(psi).all() or abs(np.linalg.norm(psi)-1)>1e-12:
        raise ValueError('SAVED_STATE_NORMALIZATION')
    random = c['method'] in ('B2','B3')
    schedule = primitive_time_schedule(c,T=T)['ordinary_one_outer_step']
    b, log_B, tau = 1., 0., 0.
    if random:
        if lambda_r <= 0 or tail is None:
            raise ValueError('EMPTY_REGISTERED_TAIL')
        tau = lambda_r*(T/c['q'])/c['r']
        b = finite_rte_distribution(tau,c['K']).exact_finite_distribution
        log_B = c['R']*math.log(b)
    finite_scale_guard(log_B=log_B,intermediate_norm=1.,absolute_discrepancy=0.)
    maximum, stages = 1., 0
    def checked(v):
        nonlocal maximum, stages
        if not np.isfinite(v).all():
            raise ValueError('NONFINITE_STAGE')
        maximum = max(maximum,float(np.linalg.norm(v))); stages += 1
        finite_scale_guard(log_B=log_B,intermediate_norm=maximum,absolute_discrepancy=0.)
        return v
    def deterministic(v,i,t):
        budget.deterministic()
        return checked(actions[i](v,t))
    def counted(v):
        return checked(tail(v))
    signals, vectors = {}, {}
    for path in ('corrected','raw') if random else ('corrected',):
        v = psi.copy()
        for _ in range(c['q']):
            if random:
                for i,t in schedule[:len(actions)]:
                    v = deterministic(v,i,t)
                for _ in range(c['r']):
                    v = checked(finite_taylor_action(counted,v,tau,c['K'],budget=budget))
                    if path=='raw':
                        v = checked(v/b)
                seq = schedule[len(actions):]
            else:
                seq = schedule
            for i,t in seq:
                v = deterministic(v,i,t)
            v = checked(np.exp(-1j*scalar*T/c['q'])*v)
        signals[path] = complex(np.vdot(psi,v)); vectors[path] = v
    return dict(signals=signals,vectors=vectors,log_B=log_B,b=b,
                intermediate_norm_max=maximum,stage_count=stages,
                counts=dict(tail=budget.tail_matvecs,deterministic=budget.deterministic_actions),
                normalization_policy='saved state unchanged; no stage rescaling')


class MatchedPort(H6PilotPort):
    def setup(self):
        self.phase('input_reference')
        identity = verify_parent(self.root)
        if identity != self.manifest['input_identity']:
            raise ValueError('MATCHED_PARENT_CHANGED')
        self.ham,self.sector,full,state,metadata,_ = load_h6_snapshot(
            safe_path(self.root,identity['snapshot_path']),
            json.loads((self.root/identity['snapshot_receipt_path']).read_text()),
            json.loads((self.root/identity['df_receipt_path']).read_text()))
        certificate = primitive_sector_certificate(self.ham,self.sector)
        self.qindices,self.qstate = checked_basis_bridge(self.ham,self.sector,full,state,1e-12)
        self.state = state.copy(); self.saved_state = state.copy()
        cache = {}
        for c in self.cells:
            key = c['method']=='B0',c['prefix']
            if key not in cache:
                cache[key] = (_prepare_discard if key[0] else _prepare)(self.ham,key[1])
            self.preps[c['id']] = cache[key]
        self.spectra = {}; self.native_actions = {}; self.tails = {}
        self.oracle_tails = {}; self.truncated_references = {}
        bounds = actual_bounds(self.cells,self.preps,T=self.plan['T'])
        self.writer.write('prepared_representation.json',dict(bounds=bounds,
            unique_preparations=len(cache), preparations=[dict(cell_id=c['id'],
                preparation_hash=self.preps[c['id']].preparation_hash,
                partition_hash=self.preps[c['id']].partition_hash,
                lambda_r=self.preps[c['id']].exact_rte_lambda_r,
                extracted_identity=self.preps[c['id']].extracted_identity_coefficient,
                constant=self.preps[c['id']].constant_coefficient) for c in self.cells]))
        if digest([dict(cell_id=r['cell_id'],schedule=r['schedule']) for r in bounds['cells']]) != digest(self.plan['coverage']['cells']):
            raise ValueError('MATCHED_PRIMITIVE_COVERAGE')
        check_bounds(bounds,self.caps)
        for c in self.cells:
            p = self.preps[c['id']]
            if p.deterministic_fragment_indices != tuple(range(c['prefix'])) or p.coefficient_atol!=0. or p.threshold_dropped_component_count!=0:
                raise ValueError('PREPARATION_ORDER_OR_CUTOFF')
        op,_ = df_linear_operator(self.ham,self.sector,**self.plan['matrix_free'])
        matrix,count = bounded_sector_matrix(lambda v:op@v,self.sector.dimension,self.calls,
                        per_action_cap=self.caps['reference_matvec_per_action'])
        error = max(float(np.linalg.norm(matrix[:,i]-occupation_column(self.ham,self.sector.basis_indices,int(index))))
                    for i,index in enumerate(self.sector.basis_indices))
        finite_scale_guard(log_B=0.,intermediate_norm=float(np.linalg.norm(matrix)),absolute_discrepancy=error)
        self.reference,pair = reference_pair(matrix,self.state,self.plan['T'])
        self.reference_error = error + pair['signal_difference']
        self.writer.write('input_reference.json',dict(metadata=metadata,sector_certificate=certificate,
            reference=complex_record(self.reference),reference_matvecs=count,
            independent_occupation_all_columns_error=error,expm_eigh_agreement=pair,
            state_renormalized=False,ground_state_certified=False,numerical_allowance_certified=False))
        del matrix,op
        self.primitives()
        self.b3_diagnostic()

    def actions(self, prep, *, oracle=False):
        if not oracle:
            key = prep.preparation_hash
            if key not in self.native_actions:
                self.native_actions[key] = super().actions(prep)
            return self.native_actions[key]
        result = []
        for block in prep.deterministic_blocks:
            key = 'one' if isinstance(block,DFDeterministicOneBodySpec) else block.original_fragment_index
            if key not in self.spectra:
                self.spectra[key] = eigh(self.primitive_matrix(key))
            w,v = self.spectra[key]
            result.append(lambda x,t,w=w,v=v:v@(np.exp(-1j*t*w)*(v.conj().T@x)))
        return result

    def primitives(self):
        self.phase('primitive_validation')
        visited = set(); errors = []; spectral_errors = []
        probes = [self.state,np.eye(1,len(self.state),0,dtype=complex)[0],
                  np.eye(1,len(self.state),len(self.state)-1,dtype=complex)[0]]
        for c in self.cells:
            prep = self.preps[c['id']]; native = self.actions(prep); oracle = self.actions(prep,oracle=True)
            for i,t in primitive_time_schedule(c,T=self.plan['T'])['unique_primitive_times']:
                block = prep.deterministic_blocks[i]
                key = 'one' if isinstance(block,DFDeterministicOneBodySpec) else block.original_fragment_index
                if (str(key),t) in visited:
                    continue
                visited.add((str(key),t))
                exact = expm(-1j*t*self.primitive_matrix(key))
                for x in probes:
                    self.calls.take('primitive')
                    e = float(np.linalg.norm(native[i](x,t)-exact@x)); errors.append(e)
                    se = float(np.linalg.norm(oracle[i](x,t)-exact@x)); spectral_errors.append(se)
                    finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(e,se))
        if len(visited)*3 != self.plan['coverage']['primitive_actions']:
            raise ValueError('MATCHED_EXECUTED_TIME_COVERAGE')
        self.writer.write('primitive_validation.json',dict(actual_time_count=len(visited),probe_count=3,
            max_native_expm_error=max(errors,default=0.),max_spectral_expm_error=max(spectral_errors,default=0.),
            all_registered_times=True,includes_negative_yoshida_times=True,certified=False))

    def cell_signal(self, raw, *, oracle=False):
        if oracle:
            raise ValueError('USE_INDEPENDENT_FORWARD_ORACLE')
        c = canonical_cell(raw); p = self.preps[c['id']]; tail = None; lam = 0.
        if c['method']=='B2':
            lam = p.exact_rte_lambda_r
            if c['prefix'] not in self.tails:
                ii = p.randomized_block_indices
                h = DFHamiltonian(-p.extracted_identity_coefficient,np.zeros_like(self.ham.one_body),
                    self.ham.lambdas[list(ii)].copy(),tuple(self.ham.g_matrices[i] for i in ii),{})
                self.tails[c['prefix']] = df_linear_operator(h,self.sector,**self.plan['matrix_free'])[0]
            op = self.tails[c['prefix']]; tail = lambda v:(op@v)/lam
        budget = ActionBudget(self.caps['tail_matvecs_corrected_and_raw'].get(c['method'],0),
                              self.caps['deterministic_actions_per_cell'])
        return compact_signal(self.state,self.actions(p),tail,cell=c,T=self.plan['T'],
            scalar=p.constant_coefficient+p.extracted_identity_coefficient,lambda_r=lam,budget=budget)

    def correctness(self):
        self.phase('signal')
        signals = []
        for c in self.cells:
            start = time.monotonic();self.progress.update(point='signal_candidate',cell_id=c['id'])
            a = self.cell_signal(c); o = self.oracle_signal(c)
            errors = [float(abs(z-o['signals'][k])) for k,z in a['signals'].items()]
            vector_errors = [float(np.linalg.norm(v-o['vectors'][k])) for k,v in a['vectors'].items()]
            maximum = max(a['intermediate_norm_max'],o['intermediate_norm_max'])
            for e in errors+vector_errors:
                finite_scale_guard(log_B=a['log_B'],intermediate_norm=maximum,absolute_discrepancy=e)
            log_error = abs(a['log_B']-o['log_B'])
            raw_error = float(abs(a['signals']['raw']*math.exp(a['log_B'])-a['signals']['corrected'])) if c['R'] else 0.
            finite_scale_guard(log_B=a['log_B'],intermediate_norm=maximum,absolute_discrepancy=max(log_error,raw_error))
            merged = dict(o['signals']);merged.update(a['signals'])
            work = sum(a['counts'].values()) + sum(o['counts'].values())
            allowance = empirical_allowance(reference_error=self.reference_error,
                path_errors=errors+vector_errors,normalization_error=log_error,raw_closure=raw_error,
                maximum_norm=maximum,work=work,settings=self.plan['numerical'])
            record = dict(cell=c,signals={k:complex_record(z) for k,z in merged.items()},
                reference=complex_record(self.reference),error_decomposition=differences(merged,self.reference,c['method']),
                log_B=a['log_B'],b=a['b'],allowance=allowance,
                independent_forward_signals={k:complex_record(z) for k,z in o['signals'].items()},
                signal_discrepancies=errors,state_discrepancies=vector_errors,normalization_log_discrepancy=log_error,
                B_raw_corrected_difference=raw_error,intermediate_norm_max=maximum,
                action_counts=a['counts'],oracle_action_counts=o['counts'],stage_count=a['stage_count'],
                classical_wall_seconds=time.monotonic()-start,
                worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                numerical_allowance_certified=False,ground_state_certified=False)
            record['precision_rows'] = precision_rows(record,self.plan)
            self.writer.write(c['id']+'_signal.json',record);signals.append(record);self.completed += 1
        self.writer.write('signal_summary.json',dict(signals=signals,calls_attempted=self.calls.used,
                                                   formal_accuracy_certified=False))

    def oracle_signal(self, raw):
        c = canonical_cell(raw); p = self.preps[c['id']]; tail = None
        if c['method']=='B2':
            if c['prefix'] not in self.oracle_tails:
                ii = p.randomized_block_indices
                self.oracle_tails[c['prefix']] = np.asarray(occupation_df_matrix(-p.extracted_identity_coefficient,
                    np.zeros_like(self.ham.one_body),self.ham.lambdas[list(ii)],
                    tuple(self.ham.g_matrices[i] for i in ii),self.sector.basis_indices),dtype=complex)/p.exact_rte_lambda_r
            tail = self.oracle_tails[c['prefix']]
        result = recurrence_paths(self.state,self.actions(p,oracle=True),tail,cell=c,T=self.plan['T'],
            scalar=p.constant_coefficient+p.extracted_identity_coefficient,
            lambda_r=p.exact_rte_lambda_r if c['R'] else 0.,
            maximum_tail=self.caps['tail_matvecs_corrected_and_raw'].get(c['method'],0))
        if c['method']=='B0':
            if c['prefix'] not in self.truncated_references:
                matrix = np.asarray(occupation_df_matrix(self.ham.constant,self.ham.one_body,
                    self.ham.lambdas[:c['prefix']],self.ham.g_matrices[:c['prefix']],self.sector.basis_indices),dtype=complex)
                self.truncated_references[c['prefix']] = reference_pair(matrix,self.state,self.plan['T'])
            result['signals']['exact_truncated'],result['truncated_expm_eigh'] = self.truncated_references[c['prefix']]
        return result

    def b3_diagnostic(self):
        p = _prepare(self.ham,0);lam = p.exact_rte_lambda_r;rows=[]
        d = self.plan['b3_diagnostic']
        for q in d['q']:
            for r in d['r']:
                for K in d['K']:
                    dist = finite_rte_distribution(lam*self.plan['T']/q/r,K)
                    rows.append(dict(q=q,r=r,R=q*r,K=K,lambda_r=lam,
                        log_B=q*r*math.log(dist.exact_finite_distribution),
                        orders=dist.orders,order_probabilities=dist.order_probabilities))
        self.writer.write('b3_normalization_diagnostic.json',dict(rows=rows,
            signal_evaluated=False,cost_evaluated=False,family_excluded=False))


def cost_task(root, manifest, task, writer, progress):
    """One task per disposable process, one evolution retained at a time."""
    port = MatchedPort(root,manifest,writer,progress)
    c = task['cell']; identity = manifest['input_identity']
    port.ham,port.sector,full,state,_,_ = load_h6_snapshot(safe_path(root,identity['snapshot_path']),
        json.loads((root/identity['snapshot_receipt_path']).read_text()),
        json.loads((root/identity['df_receipt_path']).read_text()))
    port.qindices,port.qstate = checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    p = (_prepare_discard if c['method']=='B0' else _prepare)(port.ham,c['prefix'])
    port.preps[c['id']] = p
    bounds = actual_bounds([c],port.preps,T=port.plan['T'])
    if any(v>port.caps['untranspiled_instructions'] for v in bounds['cells'][0]['wrapper_instruction_upper_bounds'].values()):
        raise RuntimeError('COST_STRUCTURAL_INSTRUCTION_CAP')
    steps = port.sampled_steps(c,task['seed']) if c['R'] else None
    events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None
    event_hash = digest(events)
    distribution = finite_rte_distribution(p.exact_rte_lambda_r*port.plan['T']/c['R'],c['K']) if steps else None
    writer.write('trajectory.json',dict(task=task,events=events,event_digest=event_hash,
        finite_distribution=distribution.to_dict() if distribution else None,
        observed_orders=[e.taylor_order for s in steps for e in s.rte_occurrence.events] if steps else [],
        quantum_shots_sampled=False,rare_events_fully_covered=False))
    import qiskit
    cp = port.plan['compiler_proposed']
    compiler = CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    branch_high = None;records=[]
    low = np.concatenate((port.qstate,np.zeros_like(port.qstate)))
    high = np.concatenate((np.zeros_like(port.qstate),port.qstate))
    for policy in ('symmetric_directional','ordinary') if task['ordinary'] else ('symmetric_directional',):
        evolution = (partial_native_from_step_requests(steps,control_policy=policy,max_instructions=port.caps['untranspiled_instructions'])
            if steps else build_deterministic_native(p.deterministic_blocks,num_system_qubits=port.ham.n_qubits,
                T=port.plan['T'],q=c['q'],formula=c['formula'],scalar=p.constant_coefficient,
                control_policy=policy,max_instructions=port.caps['untranspiled_instructions']))
        z = None;checks=[]
        if task['validate']:
            port.calls.take('control_probe',2)
            a,b = simulate_statevector(evolution.circuit,low),simulate_statevector(evolution.circuit,high)
            checks += [float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(port.qstate)]))]
            z = complex(np.vdot(port.qstate,b[len(port.qstate):]))
            if branch_high is None:
                branch_high = b
            else:
                checks.append(float(np.linalg.norm(b-branch_high)))
            if not steps:
                expected = task['expected_signal']
                checks.append(abs(z-complex(expected['real'],expected['imag'])))
            for axis,expected in (('cosine',z.real),('sine',z.imag)):
                probe = build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,
                                                     max_instructions=port.caps['untranspiled_instructions'])
                port.calls.take('control_probe')
                checks.append(abs(hadamard_expectation(simulate_statevector(probe,low),port.ham.n_qubits)-expected))
                del probe
            finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(checks))
            writer.write('validation_'+policy+'.json',dict(max_error=max(checks),signal=complex_record(z),
                probe='saved state branches and X/Y',random_signal_is_ensemble=False,certified=False))
        for axis in ('cosine','sine'):
            if digest([[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None)!=event_hash:
                raise ValueError('COST_PAIRED_EVENTS_CHANGED')
            start = time.monotonic()
            wrapper = build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                                                   max_instructions=port.caps['untranspiled_instructions'])
            port.calls.take('compile')
            cost = transpile_and_measure_cost(wrapper,compiler,
                actual_circuit_fingerprint=canonical_qiskit_circuit_fingerprint(wrapper))
            if len(cost.transpiled_circuit.data)>port.caps['transpiled_instructions']:
                raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')
            fields = dict(RZ='rz_count',CX='cx_count',RZ_depth='rz_depth',CX_depth='cx_depth',
                          depth='total_depth',size='circuit_size')
            row = dict(cell_id=c['id'],phase=task['phase'],replica=task['replica'],seed=task['seed'],
                axis=axis,control=policy,event_digest=event_hash,
                metrics={k:getattr(cost,v) for k,v in fields.items()},
                fingerprint=cost.actual_circuit_fingerprint,compiler_hash=cost.compiler_settings_hash,
                classical_wall_seconds=time.monotonic()-start,
                worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope='measured Hadamard wrapper; state preparation excluded')
            writer.write('wrapper_'+policy+'_'+axis+'.json',row);records.append(row)
            del wrapper,cost;gc.collect()
        del evolution;gc.collect()
    writer.write('cost_result.json',dict(task=task,rows=records,calls_attempted=port.calls.used,
                                       population_mean_certified=False))
