"""H4 supplemental units. Original v3 port and all completed results retained.

Full registered coverage gates precede either unit. Only S4 correctness is
re-evaluated; explicit groups use the same registered representative events.
"""
from __future__ import annotations
import time
import resource
from .ax2b_molecular_ports_v3 import *
from .ax2b_molecular_ports_v3 import _explicit_cutoff_tolerance
from .ax2b_stage_validation_v3 import mp_cell

S4_IDS = ('H4_B1_S4_q1', 'H4_B1_S4_q4')

class SupplementPort(MolecularPort):
    def __init__(self, root, manifest, writer, progress, *, started=None):
        if manifest['kind'] != 'H4_SUPPLEMENT' or manifest['unit'] not in ('S4_MP','EVENT_CONTROL'):
            raise ValueError('SUPPLEMENT_SCOPE')
        internal = dict(manifest, kind='H4_LIMITED')
        super().__init__(root, internal, writer, started=started)
        self.unit, self.progress = manifest['unit'], progress

    def execute_unit(self):
        self.setup()
        self.phase('primitive_validation')
        self.primitives()
        if self.unit == 'S4_MP':
            self.correctness()
        else:
            self.phase('validation')
            self.explicit_estimator_probes()

    def control_state(self, circuit, state):
        self.calls.take('control_probe')
        self.progress.update(control_attempted=self.progress.row['control_attempted']+1, point='control_started')
        result = simulate_statevector(circuit, state)
        self.progress.update(control_completed=self.progress.row['control_completed']+1, point='control_completed')
        return result


    def primitives(self):
        visited, errors = set(), []
        probes = [self.state]
        for i in (0,self.sector.dimension-1):
            p = np.zeros_like(self.state); p[i] = 1; probes.append(p)
        for raw in self.cells:
            cell,prep = canonical_cell(raw),self.preps[raw['id']]
            native = self.actions(prep)
            for i,t in validation_times(cell,T=self.plan['T']):
                block = prep.deterministic_blocks[i]
                key = 'one' if isinstance(block,DFDeterministicOneBodySpec) else block.original_fragment_index
                if (str(key),t) in visited:
                    continue
                visited.add((str(key),t))
                self.progress.update(point='primitive_started', primitive_key=[str(key),t])
                expected = expm(-1j*t*self.primitive_matrix(key))
                for probe in probes:
                    self.calls.take('primitive')
                    self.progress.row['primitive_attempted'] += 1
                    error = float(np.linalg.norm(native[i](probe,t)-expected @ probe))
                    self.progress.row['primitive_completed'] += 1
                    finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=error)
                    errors.append(error)
                self.progress.update(point='primitive_completed')
        self.oracle_cache = None
        self.writer.write('primitive_validation.json',{'actual_time_count':len(visited),'probe_count':3,
                    'max_error':max(errors,default=0.),'evidence_kind':'TECHNICAL_AGREEMENT','certified':False})

    def correctness(self):
        self.phase('validation')
        for raw in (c for c in self.cells if c['id'] in S4_IDS):
            self.progress.update(cell=raw['id'], dps=None, oracle_work=None,
                  correctness_attempted=self.progress.row['correctness_attempted']+1)
            cell_started = time.monotonic()
            cell,prep = canonical_cell(raw),self.preps[raw['id']]
            actual = self.cell_signal(raw)
            maximum = actual['intermediate_norm_max']
            record = {'cell':raw, 'signals':{k:complex_record(z) for k,z in actual['signals'].items()},
                      'trace':actual['trace'], 'action_counts':actual['counts'],'log_B':actual['log_B'],
                      'N':None,'G':None,'accuracy_eligibility':'UNDETERMINED','numerical_allowance_certified':False}
            if self.kind == 'H4_LIMITED':
                precise = []
                for dps in self.plan['dps']:
                    self.progress.update(dps=dps, oracle_work=None,
                            mp_attempted=self.progress.row['mp_attempted']+1)
                    mp = mp_cell(self.ham,self.sector.basis_indices,self.saved_state,cell,T=self.plan['T'],
                                 scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
                                 lambda_r=prep.exact_rte_lambda_r if cell['R'] else 0.,
                                 extracted_identity=prep.extracted_identity_coefficient,dps=dps,
                                 progress=lambda value: self.progress.update(**value))
                    for path,z in actual['signals'].items():
                        ref = mp['signals'][path]
                        error = abs(z-complex(float(ref['real']),float(ref['imag'])))
                        finite_scale_guard(log_B=actual['log_B'],intermediate_norm=maximum,absolute_discrepancy=error)
                    ref = mp['signals']['reference']
                    reference_difference = abs(self.reference-complex(float(ref['real']),float(ref['imag'])))
                    finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=reference_difference)
                    record['reference_difference_mp'+str(dps)] = reference_difference
                    record['stage_comparison_mp'+str(dps)] = compare_stages(actual,mp)
                    self.writer.write(cell['id']+'_mp'+str(dps)+'.json',mp)
                    precise.append(mp)
                record['precision_comparison'] = compare_mp_records(precise[0],precise[1])
                for value in record['precision_comparison']['signal_differences'].values():
                    finite_scale_guard(log_B=0.,intermediate_norm=maximum,absolute_discrepancy=float(value))
            else:
                oracle = self.cell_signal(raw,oracle=True)
                record['independent_binary64_oracle_signals'] = {k:complex_record(z) for k,z in oracle['signals'].items()}
                record['independent_trace'] = oracle['trace']
                for key,z in actual['signals'].items():
                    finite_scale_guard(log_B=actual['log_B'],intermediate_norm=max(maximum,oracle['intermediate_norm_max']),
                                       absolute_discrepancy=abs(z-oracle['signals'][key]))
            if 'raw' in actual['signals']:
                error = abs(actual['signals']['raw']*math.exp(actual['log_B'])-actual['signals']['corrected'])
                finite_scale_guard(log_B=actual['log_B'],intermediate_norm=maximum,absolute_discrepancy=error)
                record['B_raw_corrected_difference'] = error
            record['classical_cell_wall_seconds'] = time.monotonic()-cell_started
            record['worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            self.writer.write(cell['id']+'_correctness.json',record)
            self.completed += 1

    def control_and_measurement(self, evolutions):
        """Both branches, ordinary/directional, and actual X/Y wrapper states."""
        n = self.ham.n_qubits
        errors = []
        probes = [self.qstate]
        for index in (self.qindices[0],self.qindices[-1]):
            v = np.zeros_like(self.qstate); v[index] = 1; probes.append(v)
        for probe in probes:
            low,high = np.concatenate((probe,np.zeros_like(probe))),np.concatenate((np.zeros_like(probe),probe))
            results = {}
            for policy,e in evolutions.items():
                a,b = self.control_state(e.circuit,low),self.control_state(e.circuit,high)
                errors.extend([float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:2**n]))])
                results[policy] = b
            errors.append(float(np.linalg.norm(results['ordinary']-results['symmetric_directional'])))
            z = np.vdot(probe,results['ordinary'][2**n:])
            for policy,e in evolutions.items():
                for axis,expected in (('cosine',z.real),('sine',z.imag)):
                    wrapper = build_native_hadamard_wrapper(e,axis=axis,include_measurement=False,
                                      max_instructions=self.caps['untranspiled_instructions'])
                    state = self.control_state(wrapper,low)
                    errors.append(abs(hadamard_expectation(state,n)-expected))
        maximum = max(errors)
        finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=maximum)
        return maximum

    def explicit_estimator_probes(self):
        """Registered two representatives per random cost cell, no sampler/compile."""
        from trotterlib.rte import finite_rte_distribution, _make_event
        from trotterlib.df_partial_s2 import DFPartialS2StepRequest
        from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
        for id_ in ('H4_B2_K2','H4_B3_K6'):
            raw = next(c for c in self.cells if c['id'] == id_)
            cell,prep = canonical_cell(raw),self.preps[id_]
            delta = self.plan['T']/cell['q']/cell['r']; tau = prep.exact_rte_lambda_r*delta
            config,dist = make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=delta,rte_steps=1,
                         finite_taylor_order=cell['K'],truncation_tolerance=_explicit_cutoff_tolerance(tau,cell['K']),seed=0)
            components = prep.rte_preparation.symbolic_tail.components
            for order_index in (0,1):
                self.progress.update(cell=id_, dps=None, oracle_work=None, event_order=dist.orders[order_index],
                     event_attempted=self.progress.row['event_attempted']+1)
                selected = next(i for i,c in enumerate(components) if not c.is_identity)
                event = _make_event([selected]*(dist.orders[order_index]+1),components,dist,order_index)
                occurrence = DFRTEEventSequenceCircuitRequest(events=(event,),
                        component_specs=prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=self.ham.n_qubits,
                        tail_id=prep.rte_preparation.symbolic_tail.tail_id,tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,
                        occurrence_rte_steps=1)
                step = DFPartialS2StepRequest(prep,delta,config,dist,occurrence,controlled=True,
                                             ancilla_qubit=self.ham.n_qubits,seed=0)
                evolutions = {p:partial_native_from_step_requests((step,),control_policy=p,
                         max_instructions=self.caps['untranspiled_instructions']) for p in ('ordinary','symmetric_directional')}
                error = self.control_and_measurement(evolutions)
                # Compare a single negative-phase event against algebraic
                # signed Pauli action; ordinary and directional share lowering.
                from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
                builder = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
                circuit = builder.build_sequence(occurrence).circuit
                high = np.concatenate((np.zeros_like(self.qstate),self.qstate))
                actual = self.control_state(circuit,high)
                expected = explicit_event_action(self.qstate,event,prep.rte_preparation.basis_registry)
                event_error = float(np.linalg.norm(actual-np.concatenate((np.zeros_like(expected),expected))))
                finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=event_error)
                self.writer.write(id_+f'_explicit_order{event.taylor_order}.json',{'event':event.to_dict(),'error':error,
                         'algebraic_event_action_error':event_error,
                         'sampling_performed':False,'full_molecular_event_mean_enumerated':False,
                         'fresh_shot_semantics':'each shot requires fresh whole trajectory; pairing is cost-only'})
                evolutions.clear(); gc.collect()
