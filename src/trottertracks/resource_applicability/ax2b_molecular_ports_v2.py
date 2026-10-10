"""Future bound H4/H6 backend. Never run by import or preparation.

H6 has one sector reference and a single replaceable primitive oracle; no
full-space fragment matrices/eigenvector cache. Input generation is absent.
Actual molecular checks, sampling and compilation require a separate grant.
"""
from __future__ import annotations

import ast
import gc
import json
import math
from pathlib import Path
import resource
import time
import zipfile

import numpy as np
from scipy.linalg import expm

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from trotterlib.pr2_new_series_validation import _load_snapshot_once
from trotterlib.pr2_s0_s1_validation import _array_hash, _sector_hash, _state_hash
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard, _explicit_cutoff_tolerance
from trotterlib.df_partial_s2_repeated import make_df_partial_s2_repeated_request
from trotterlib.df_rte_qiskit import estimate_df_rte_structural_size_upper_bound
from trotterlib.rte import make_rte_config, CompilerSettings
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from trotterlib.df_trotter.circuit import simulate_statevector

from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_native_df_v5 import (make_native_block_action, block_instruction_bound,
        build_deterministic_native, partial_native_from_step_requests, build_native_hadamard_wrapper)
from .ax2a_state_action import ActionBudget, project_primitive_checked, df_tail_operator
from .ax2b_h6_controller import bounded_sector_matrix, finite_scale_guard
from .ax2b_limits import CallBudget
from .ax2b_bound_launch_v2 import verify_input, safe_path
from .ax2b_stage_validation_v2 import canonical_cell, traced_signal, mp_cell
from .ax2b_independent_reference import _one_body, occupation_df_matrix
from .ax2b_h6_contract import primitive_time_schedule
from .ax2a_preparation import digest
from .ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint


def occupation_column(ham, basis, index):
    """Independent full-occupation G squared then declared-sector restriction."""
    value = _one_body(ham.one_body, {index:1.})
    value[index] = value.get(index,0) + ham.constant
    for w,g in zip(ham.lambdas,ham.g_matrices,strict=True):
        for target, amplitude in _one_body(g,_one_body(g,{index:1.})).items():
            value[target] = value.get(target,0) + w*amplitude
    positions = {int(x):i for i,x in enumerate(basis)}
    if any(a != 0 and i not in positions for i,a in value.items()):
        raise ValueError('INDEPENDENT_COLUMN_LEAVES_SECTOR')
    return np.asarray([value.get(int(i),0) for i in basis],dtype=complex)


def load_h6_snapshot(path, metadata):
    """A dedicated variable-rank layout; no H4 loader/rank fallback."""
    rank = metadata['hamiltonian_metadata']['df_rank_actual']
    layout = {'constant':((),'<f8'), 'one_body':((12,12),'<c16'), 'lambdas':((rank,),'<f8'),
              'g_matrices':((rank,12,12),'<c16'), 'sector_basis_indices':((400,),'<i8'),
              'state_vector':((4096,),'<c16'), 'sector_state_vector':((400,),'<c16')}
    # Check uncompressed sizes and NPY shapes BEFORE np.load allocations.
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 8 or {i.filename for i in infos} != {k+'.npy' for k in (*layout,'metadata_json')}:
            raise ValueError('H6_NPZ_KEYS')
        if sum(i.file_size for i in infos) > 16*2**20:
            raise ValueError('H6_NPZ_EXPANDED_SIZE')
        for key,(shape,dtype) in layout.items():
            with archive.open(key+'.npy') as stream:
                if stream.read(6) != b'\x93NUMPY':
                    raise ValueError('H6_NPY_MAGIC')
                version = tuple(stream.read(2))
                if version not in ((1,0),(2,0),(3,0)):
                    raise ValueError('H6_NPY_VERSION')
                length = int.from_bytes(stream.read(2 if version == (1,0) else 4),'little')
                if not 0 < length <= 65536:
                    raise ValueError('H6_NPY_HEADER')
                header = ast.literal_eval(stream.read(length).decode())
                if header != {'descr':dtype,'fortran_order':False,'shape':shape}:
                    raise ValueError('H6_NPY_LAYOUT:'+key)
    with np.load(path,allow_pickle=False) as payload:
        arrays = {key:np.array(payload[key],copy=True) for key in layout}
        if json.loads(str(payload['metadata_json'].item())) != metadata:
            raise ValueError('H6_METADATA')
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('H6_INPUT_NONFINITE')
    from .ax2b_h6_input import _array_record
    hm = metadata['hamiltonian_metadata']
    changes = hm.get('hermitization')
    if not isinstance(changes,list) or len(changes) != rank+1:
        raise ValueError('H6_HERMITIZATION_RECEIPTS')
    for i,(label,array) in enumerate([('corrected_one_body',arrays['one_body'])]+
                                   [(f'fragment_{j}',g) for j,g in enumerate(arrays['g_matrices'])]):
        row = changes[i]
        difference = row.get('difference_frobenius')
        if (row.get('label') != label or row.get('after') != _array_record(array)
                or isinstance(difference,bool) or not isinstance(difference,(int,float))
                or not math.isfinite(difference) or not 0 <= difference <= 1e-10):
            raise ValueError('H6_HERMITIZATION_BYTES:'+label)
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    ham = DFHamiltonian(float(arrays['constant']), arrays['one_body'], arrays['lambdas'],
                        tuple(arrays['g_matrices']),metadata['hamiltonian_metadata'])
    sector = PhysicalSector.spin_sector(n_qubits=12,nelec_alpha=3,nelec_beta=3)
    if not np.array_equal(arrays['sector_basis_indices'],sector.basis_indices):
        raise ValueError('H6_SECTOR_ORDER')
    checks = {'hamiltonian_hash':df_hamiltonian_hash(ham), 'sector_hash':_sector_hash(sector),
              'state_vector_hash':_array_hash(arrays['state_vector']),
              'sector_state_vector_hash':_array_hash(arrays['sector_state_vector']),
              'state_hash':_state_hash(arrays['state_vector'],arrays['sector_state_vector'])}
    if any(metadata.get(k) != v for k,v in checks.items()):
        raise ValueError('H6_INTERNAL_IDENTITY')
    return ham,sector,arrays['state_vector'],arrays['sector_state_vector'],metadata,layout


def actual_bounds(cells, preparations, *, T):
    """Before native build/sampling: all-time probe counts and structural bounds."""
    probes, rows = set(), []
    for raw in cells:
        cell = canonical_cell(raw)
        prep = preparations[cell['id']]
        schedule = primitive_time_schedule(cell,T=T)
        times = validation_times(cell,T=T)
        schedule['registered_validation_times_v2'] = times
        for i,t in times:
            block = prep.deterministic_blocks[i]
            key = 'one' if isinstance(block,DFDeterministicOneBodySpec) else block.original_fragment_index
            probes.add((str(key),t))
        random = cell['method'] in ('B2','B3')
        tail_bound = estimate_df_rte_structural_size_upper_bound(prep.rte_preparation.component_specs,
                      maximum_taylor_order=cell['K'],event_count=cell['r'],controlled=True)*cell['q'] if random else 0
        pieces = 3 if cell['formula'] == '4th' else 1
        bounds = {}
        for policy,modes in (('ordinary',('ORDINARY','ORDINARY')),
                             ('symmetric_directional',('UNCONTROLLED','DIRECTIONAL'))):
            bounds[policy] = cell['q']*pieces*sum(block_instruction_bound(b,m)
                           for b in prep.deterministic_blocks for m in modes)+tail_bound+1+4
        rows.append({'cell_id':cell['id'],'wrapper_instruction_upper_bounds':bounds,
                     'schedule':schedule})
    return {'primitive_actions':3*len(probes), 'primitive_probe_count':3,
            'cells':rows, 'policy':'all actual unique times x saved state/first/last sector columns',
            'structural_bounds_are_not_compiled_costs':True}


def validation_times(cell,*,T):
    """Cell PF/control times plus H4-E's additional explicit microstep halves."""
    times = set(map(tuple,primitive_time_schedule(cell,T=T)['unique_primitive_times']))
    if cell['id'] in ('H4_B2_K2','H4_B3_K6'):
        half = T/cell['q']/cell['r']/2
        times.update((i,t) for i in range(cell['prefix']+1) for t in (half,-half))
    return sorted(times)


def check_bounds(bounds,caps):
    if bounds['primitive_actions'] > caps['primitive']:
        raise RuntimeError('ACTUAL_TIME_COVERAGE_EXCEEDS_CAP')
    if any(b > caps['untranspiled_instructions'] for row in bounds['cells']
           for b in row['wrapper_instruction_upper_bounds'].values()):
        raise RuntimeError('STRUCTURAL_INSTRUCTION_BOUND_CAP')


def complex_record(z):
    return {'real':float(z.real),'imag':float(z.imag)}


def hadamard_expectation(state, n):
    if state.shape != (2**(n+1),) or not np.isfinite(state).all():
        raise ValueError('HADAMARD_REGISTER')
    return float(np.sum(abs(state[:2**n])**2)-np.sum(abs(state[2**n:])**2))


def local_matrix_action(vector, matrix, qubits):
    """Explicit little-endian local tensor action, no Qiskit circuit simulator."""
    qubits = tuple(qubits)
    matrix = np.asarray(matrix,dtype=complex)
    n = len(vector).bit_length()-1
    if (len(vector) != 2**n or len(set(qubits)) != len(qubits) or not qubits
            or len(qubits) > 4 or any(type(q) is not int or q < 0 or q >= n for q in qubits)
            or matrix.shape != (2**len(qubits),)*2 or not np.isfinite(matrix).all()):
        raise ValueError('LOCAL_MATRIX_REGISTER')
    result = np.empty_like(vector)
    mask = sum(1<<q for q in qubits)
    offsets = [sum(((i>>j)&1)<<q for j,q in enumerate(qubits)) for i in range(2**len(qubits))]
    for base in range(len(vector)):
        if base & mask == 0:
            positions = [base+i for i in offsets]
            result[positions] = matrix @ vector[positions]
    return result


def explicit_event_action(vector,event,registry):
    """Algebraic signed-involution/rotation action, including relative phase.

    Uses prepared local basis matrices; independent of event circuit lowering,
    instruction reuse and control/Hadamard construction. Not an independent
    orbital decomposition oracle. No full-space unitary is allocated.
    """
    from qiskit.quantum_info import Operator
    current = vector.copy()
    def pauli(v,application):
        if application.is_identity:
            return application.coefficient_sign*v
        definition = registry.definition(application.basis_id)
        if definition.metadata.basis_hash != application.basis_hash:
            raise ValueError('EVENT_BASIS_HASH')
        operations = [(Operator(gate).data,tuple(q)) for gate,q in definition.runtime_operations]
        for matrix,qubits in reversed(operations):
            v = local_matrix_action(v,matrix.conj().T,qubits)
        signs = np.ones(len(v))
        indices = np.arange(len(v))
        for q in application.diagonal_pauli_support:
            signs *= 1-2*((indices>>q)&1)
        v = application.coefficient_sign*signs*v
        for matrix,qubits in operations:
            v = local_matrix_action(v,matrix,qubits)
        return v
    for application in event.application_sequence:
        image = pauli(current,application)
        if application.role == 'product':
            current = image
        else:
            current = math.cos(event.rotation_angle)*current-1j*math.sin(event.rotation_angle)*image
    return event.phase*current


class MolecularPort:
    def __init__(self, root, manifest, writer, *, started=None):
        self.root, self.manifest, self.writer = Path(root),manifest,writer
        self.kind, self.plan = manifest['kind'],manifest['plan']
        self.caps = self.plan['caps_proposed']
        self.cells = self.plan.get('cells')
        self.started = time.monotonic() if started is None else started
        self.calls = CallBudget(**{k:self.caps[k] for k in ('primitive','control_probe','compile','trajectory','occurrence','reference_matvec')})
        self.completed, self.compiled = 0,0
        self.preps = {}
        self.oracle_cache = None

    def phase(self, name):
        self.writer.write('phase_'+name+'.json',{'phase':name,'elapsed':time.monotonic()-self.started})

    def setup(self):
        self.phase('input_reference')
        verify_input(self.root,self.manifest['input_binding'],self.kind)
        path = safe_path(self.root,self.manifest['input_binding']['path'])
        if self.kind == 'H4_LIMITED':
            loaded = _load_snapshot_once(path)
        else:
            loaded = load_h6_snapshot(path,self.manifest['input_binding']['metadata'])
        self.ham,self.sector,full,state,metadata,_ = loaded
        if self.kind == 'H4_LIMITED' and (self.ham.n_qubits,self.ham.n_blocks,self.sector.dimension) != (8,12,36):
            raise ValueError('H4_SCOPE')
        if self.kind == 'H6_TECHNICAL' and (self.ham.n_qubits,self.ham.n_blocks,self.sector.dimension) != (12,self.plan['actual_rank'],400):
            raise ValueError('H6_SCOPE')
        certificate = primitive_sector_certificate(self.ham,self.sector)
        self.qindices,self.qstate = checked_basis_bridge(self.ham,self.sector,full,state,1e-12)
        self.saved_state = state.copy()
        self.state = state/np.linalg.norm(state)
        self.qstate /= np.linalg.norm(state)
        self.preps = {cell['id']:(_prepare_discard if cell['method'] == 'B0' else _prepare)(self.ham,cell['prefix'])
                      for cell in self.cells}
        bounds = actual_bounds(self.cells,self.preps,T=self.plan['T'])
        check_bounds(bounds,self.caps)
        if self.manifest['coverage_binding']['expected_bounds'] != bounds:
            raise ValueError('ACTUAL_COVERAGE_CHANGED')
        self.writer.write('actual_coverage.json',bounds)
        operator,_ = df_linear_operator(self.ham,self.sector,backend='python')
        matrix,count = bounded_sector_matrix(lambda v:operator @ v,self.sector.dimension,self.calls,
                                            per_action_cap=self.caps['reference_matvec_per_action'])
        if np.linalg.norm(matrix-matrix.conj().T) > 1e-10:
            raise ValueError('REFERENCE_HERMITICITY')
        difference = max(np.linalg.norm(matrix[:,i]-occupation_column(self.ham,self.sector.basis_indices,int(index)))
                         for i,index in enumerate(self.sector.basis_indices))
        finite_scale_guard(log_B=0., intermediate_norm=float(np.linalg.norm(matrix)),absolute_discrepancy=float(difference))
        self.reference = complex(np.vdot(self.state,expm(-1j*self.plan['T']*matrix) @ self.state))
        del matrix
        self.writer.write('input_reference.json',{'metadata':metadata,'sector_certificate':certificate,
               'independent_occupation_all_columns_error':float(difference),'reference_matvecs':count,
               'reference':complex_record(self.reference),'saved_state_norm_before':float(np.linalg.norm(state)),
               'reference_kind':'binary64 sector expm; independent occupation construction checked',
               'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED','ground_state_certified':False})

    def primitive_matrix(self, index):
        if self.oracle_cache is not None and self.oracle_cache[0] == index:
            return self.oracle_cache[1]
        self.oracle_cache = None
        zero = np.zeros_like(self.ham.one_body)
        one = self.ham.one_body if index == 'one' else zero
        weights, blocks = ((),()) if index == 'one' else ((self.ham.lambdas[index],),(self.ham.g_matrices[index],))
        matrix = np.asarray(occupation_df_matrix(0.,one,weights,blocks,self.sector.basis_indices),dtype=complex)
        if np.linalg.norm(matrix-matrix.conj().T) > 1e-10:
            raise ValueError('PRIMITIVE_ORACLE_HERMITICITY')
        self.oracle_cache = index,matrix
        return matrix

    def actions(self, prep, *, oracle=False):
        result = []
        for block in prep.deterministic_blocks:
            if oracle:
                key = 'one' if isinstance(block,DFDeterministicOneBodySpec) else block.original_fragment_index
                result.append(lambda v,t,k=key: expm(-1j*t*self.primitive_matrix(k)) @ v)
            else:
                action = make_native_block_action(block,max_instructions=self.caps['untranspiled_instructions'])
                result.append(lambda v,t,a=action:project_primitive_checked(lambda x:a(x,t),v,self.qindices,
                    2**self.ham.n_qubits,leakage_tolerance=1e-12))
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
                expected = expm(-1j*t*self.primitive_matrix(key))
                for probe in probes:
                    self.calls.take('primitive')
                    error = float(np.linalg.norm(native[i](probe,t)-expected @ probe))
                    finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=error)
                    errors.append(error)
        self.oracle_cache = None
        self.writer.write('primitive_validation.json',{'actual_time_count':len(visited),'probe_count':3,
                    'max_error':max(errors,default=0.),'evidence_kind':'TECHNICAL_AGREEMENT','certified':False})

    def cell_signal(self,raw,*,oracle=False):
        cell,prep = canonical_cell(raw),self.preps[raw['id']]
        random = cell['method'] in ('B2','B3')
        lam = prep.exact_rte_lambda_r if random else 0.
        identity = prep.extracted_identity_coefficient
        tail = None
        if random and oracle:
            # One sector tail matrix, discarded at cell end; no fragment cache.
            zero = np.zeros_like(self.ham.one_body)
            matrix = np.asarray(occupation_df_matrix(-identity,zero,self.ham.lambdas[cell['prefix']:],
                   self.ham.g_matrices[cell['prefix']:],self.sector.basis_indices),dtype=complex)/lam
            tail = lambda v:matrix @ v
        elif random:
            operator,_ = df_tail_operator(self.ham,self.sector,range(cell['prefix'],self.ham.n_blocks),
                       extracted_identity=identity,primitive_sector_certified=True)
            tail = lambda v:(operator @ v)/lam
        maximum = self.caps.get('tail_matvecs_per_cell', self.caps.get('tail_matvecs_corrected_and_raw',{}).get(cell['method'],0))
        budget = ActionBudget(maximum,self.caps['deterministic_actions_per_cell'])
        value = traced_signal(self.state,self.actions(prep,oracle=oracle),tail,cell=cell,T=self.plan['T'],
                  scalar=prep.constant_coefficient+identity,lambda_r=lam,budget=budget)
        value['counts'] = {'tail':budget.tail_matvecs,'deterministic':budget.deterministic_actions}
        self.oracle_cache = None
        return value

    def correctness(self):
        self.phase('correctness')
        self.primitives()
        for raw in self.cells:
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
                    mp = mp_cell(self.ham,self.sector.basis_indices,self.saved_state,cell,T=self.plan['T'],
                                 scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
                                 lambda_r=prep.exact_rte_lambda_r if cell['R'] else 0.,
                                 extracted_identity=prep.extracted_identity_coefficient,dps=dps)
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

    def sampled_steps(self,cell,seed):
        """Cost-only fresh request per group. Measurement shots are not sampled."""
        prep = self.preps[cell['id']]
        delta = self.plan['T']/cell['q']; tau = prep.exact_rte_lambda_r*delta/cell['r']
        config,distribution = make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=delta,
                  rte_steps=cell['r'],truncation_tolerance=_explicit_cutoff_tolerance(tau,cell['K']),
                  finite_taylor_order=cell['K'],seed=seed)
        self.calls.take('trajectory'); self.calls.take('occurrence',cell['q'])
        request = make_df_partial_s2_repeated_request(prep,step_time=delta,repetition_count=cell['q'],
                   rte_config=config,rte_distribution=distribution,seed=seed,controlled=True,
                   ancilla_qubit=self.ham.n_qubits,construction_policy='raw_concatenation')
        return tuple(request.iter_step_requests())

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
                self.calls.take('control_probe',2)
                a,b = simulate_statevector(e.circuit,low),simulate_statevector(e.circuit,high)
                errors.extend([float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:2**n]))])
                results[policy] = b
            errors.append(float(np.linalg.norm(results['ordinary']-results['symmetric_directional'])))
            z = np.vdot(probe,results['ordinary'][2**n:])
            for policy,e in evolutions.items():
                for axis,expected in (('cosine',z.real),('sine',z.imag)):
                    wrapper = build_native_hadamard_wrapper(e,axis=axis,include_measurement=False,
                                      max_instructions=self.caps['untranspiled_instructions'])
                    self.calls.take('control_probe')
                    state = simulate_statevector(wrapper,low)
                    errors.append(abs(hadamard_expectation(state,n)-expected))
        maximum = max(errors)
        finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=maximum)
        return maximum

    def costs(self):
        self.phase('wrapper_cost')
        if self.kind == 'H4_LIMITED':
            self.explicit_estimator_probes()
            return
        if self.completed != 7:
            raise ValueError('CORRECTNESS_REQUIRED')
        import qiskit
        cp = self.plan['compiler_proposed']
        compiler = CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
        for raw in self.cells:
            cell,prep = canonical_cell(raw),self.preps[raw['id']]
            for replica in range(cell['replicas']):
                group = [t for t in self.plan['wrapper_tasks'] if t['cell_id'] == cell['id'] and t['replica'] == replica]
                steps = self.sampled_steps(cell,group[0]['seed']) if cell['R'] else None
                events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None
                event_digest = digest(events)
                evolutions = {}
                try:
                    for policy in ('ordinary','symmetric_directional'):
                        evolutions[policy] = (partial_native_from_step_requests(steps,control_policy=policy,
                                      max_instructions=self.caps['untranspiled_instructions']) if steps else
                           build_deterministic_native(prep.deterministic_blocks,num_system_qubits=self.ham.n_qubits,
                               T=self.plan['T'],q=cell['q'],formula=cell['formula'],scalar=prep.constant_coefficient,
                               control_policy=policy,max_instructions=self.caps['untranspiled_instructions']))
                    maximum = self.control_and_measurement(evolutions)
                    self.writer.write(cell['id']+f'_rep{replica}_trajectory.json',{'events':events,'event_digest':event_digest,
                        'seed':group[0]['seed'],'control_measurement_max_error':maximum,
                        'sample_scope':'engineering cost; fresh IID quantum-shot semantics not a cost sample guarantee'})
                    for task in group:
                        if steps and digest([[e.to_dict() for e in s.rte_occurrence.events] for s in steps]) != event_digest:
                            raise ValueError('PREPARED_EVENTS_MUTATED')
                        compile_started = time.monotonic()
                        wrapper = build_native_hadamard_wrapper(evolutions[task['control']],axis=task['axis'],
                                  include_measurement=True,max_instructions=self.caps['untranspiled_instructions'])
                        self.calls.take('compile')
                        cost = transpile_and_measure_cost(wrapper,compiler,
                                  actual_circuit_fingerprint=canonical_qiskit_circuit_fingerprint(wrapper))
                        if len(cost.transpiled_circuit.data) > self.caps['transpiled_instructions']:
                            raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')
                        self.writer.write('wrapper_%02d.json'%self.compiled,{'task':task,'event_digest':event_digest,
                             'metrics':{k:getattr(cost,k) for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')},
                             'fingerprint':cost.actual_circuit_fingerprint,'compiler_hash':cost.compiler_settings_hash,
                             'classical_wrapper_hash_compile_wall_seconds':time.monotonic()-compile_started,
                             'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                             'N':None,'G':None,'cost_scope':'measured Hadamard wrapper; no state preparation',
                             'statistic':'individual engineering sample; no mean precision or winner claim'})
                        self.compiled += 1
                        wrapper = cost = None
                finally:
                    evolutions.clear(); steps = None; gc.collect()

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
                self.calls.take('control_probe')
                actual = simulate_statevector(circuit,high)
                expected = explicit_event_action(self.qstate,event,prep.rte_preparation.basis_registry)
                event_error = float(np.linalg.norm(actual-np.concatenate((np.zeros_like(expected),expected))))
                finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=event_error)
                self.writer.write(id_+f'_explicit_order{event.taylor_order}.json',{'event':event.to_dict(),'error':error,
                         'algebraic_event_action_error':event_error,
                         'sampling_performed':False,'full_molecular_event_mean_enumerated':False,
                         'fresh_shot_semantics':'each shot requires fresh whole trajectory; pairing is cost-only'})
                evolutions.clear(); gc.collect()


class FreshShotRequests:
    """Separate future measurement port; never cache a cost trajectory.

    Each call invokes the existing whole-trajectory factory with a distinct
    shot-domain seed. Its outer/microstep draw law remains the existing RTE
    implementation. This is a draw contract, not an independence certificate.
    """
    def __init__(self,factory,*,max_shots,master_seed):
        if type(max_shots) is not int or max_shots < 1 or type(master_seed) is not int or master_seed < 0:
            raise ValueError('FRESH_SHOT_CONFIGURATION')
        self.factory,self.master_seed = factory,master_seed
        self.calls = CallBudget(shots=max_shots)
        self.seeds = set()

    def draw(self,cell):
        shot = self.calls.used['shots']
        seed = int(digest({'domain':'measurement_fresh_whole_trajectory_v2',
                          'master_seed':self.master_seed,'cell_id':cell['id'],'shot':shot})[:16],16)
        if seed in self.seeds:
            raise ValueError('FRESH_SHOT_SEED_COLLISION')
        self.calls.take('shots')
        self.seeds.add(seed)
        return self.factory(cell,seed)


def compare_mp_records(a,b):
    import mpmath as mp
    with mp.workdps(120):
        differences = {}
        for key in a['signals']:
            z,w = a['signals'][key],b['signals'][key]
            value = abs(mp.mpc(z['real'],z['imag'])-mp.mpc(w['real'],w['imag']))
            differences[key] = mp.nstr(value,120)
        return {'signal_differences':differences,'evidence_kind':'EMPIRICAL','certified':False}


def compare_stages(actual,precise):
    """Same physical stage endpoint, MP vs native; Horner internals stay separate.

    Raw endpoints are compared after division, not before it. MP uses its
    independently evaluated normalization; discrepancies are not u certificates.
    """
    import mpmath as mp
    selected = {}
    for row in actual['trace']:
        label = row['stage']
        if 'horner_matvec' in label:
            continue
        if row['path'] == 'raw' and ':micro:' in label:
            if not label.endswith(':divide_b'):
                continue
            label = label[:-len(':divide_b')]
        selected[row['path'],label] = row
    values = []
    with mp.workdps(precise['dps']):
        for row in precise['trace']:
            if row['path'] == 'exact_tail':
                continue
            endpoint = selected.pop((row['path'],row['stage']))
            error = mp.sqrt(sum(abs(mp.mpc(a['real'],a['imag'])-mp.mpc(b['real'],b['imag']))**2
                         for a,b in zip(endpoint['state_after'],row['state_after'],strict=True)))
            finite_scale_guard(log_B=actual['log_B'],intermediate_norm=actual['intermediate_norm_max'],
                               absolute_discrepancy=float(error))
            values.append({'path':row['path'],'stage':row['stage'],'state_difference':mp.nstr(error,precise['dps'])})
    if selected:
        raise ValueError('UNPAIRED_STAGE_ENDPOINTS')
    return {'stages':values,'evidence_kind':'EMPIRICAL','certified':False}
