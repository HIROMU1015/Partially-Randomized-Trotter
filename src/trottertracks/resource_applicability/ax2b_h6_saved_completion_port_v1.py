"""Future one-shot saved DF acceptance/state port. Preparation never calls it."""
from __future__ import annotations
import json
import math
import numpy as np
from .ax2a_preparation import digest
from .ax2b_h6_df_diagnostic_contract_v1 import file_hash
from .ax2b_h6_saved_completion_contract_v1 import RAW
from .ax2b_h6_input_generation_audit_v1 import read_json, npz_bytes
from .ax2b_h6_input_generation_port_v1 import atomic_npz
from .ax2b_h6_weighted_projection_v1 import accept, canonical_antisym
from .ax2b_limits import CallBudget


def load_saved_payload(root,identity):
    """Read-only decode only after launch gate. The seal is checked again first."""
    def load(name,receipt,sha,cap):
        path=root/name
        if file_hash(path)!=sha:
            raise ValueError('SAVED_COMPLETION_BEFORE_DECODE')
        npz_bytes(path,receipt,cap=cap)
        with np.load(path,allow_pickle=False) as z:
            arrays={k:np.array(z[k],copy=True) for k in receipt['arrays']}
        for a in arrays.values():a.setflags(write=False)
        if file_hash(path)!=sha:
            raise ValueError('SAVED_COMPLETION_AFTER_DECODE')
        return arrays
    old=identity['integrals'];integral_path=old['input_path']
    integral_receipt=read_json(root/integral_path.replace('integrals.npz','integral_receipt.json'))
    arrays=load(integral_path,integral_receipt,old['input_sha256'],16*2**20)
    raw_receipt=read_json(root/RAW/'raw_decomposition_receipt.json')
    raw=load(identity['raw_path'],raw_receipt,identity['raw_sha256'],4*2**20)
    hyp_receipt=read_json(root/RAW/'hypothetical_receipt.json')
    hyp=load(RAW+'hypothetical_hermitization.npz',hyp_receipt,identity['hypothetical_sha256'],4*2**20)
    if file_hash(root/RAW/'diagnostic_summary.json')!=identity['summary_sha256']:
        raise ValueError('SAVED_COMPLETION_SUMMARY_BYTES')
    return {'constant':float(arrays['constant']),'one_body':arrays['one_body'],'two_body':arrays['two_body'],
        'lambdas':raw['lambdas_raw'],'g_matrices':raw['g_matrices_raw'],
        'correction':raw['one_body_correction_raw'],'truncation':raw['truncation_value_raw'],
        'summary':read_json(root/RAW/'diagnostic_summary.json'),
        'diagnostic_coefficients':(hyp['normal_order_one_body_raw_residual'],
            canonical_antisym(hyp['normal_order_two_body_raw']),hyp['normal_order_two_body_target_antisym']),
        'hypothetical_projection':(hyp['corrected_one_body_hypothetical_hermitian'],hyp['g_hypothetical_hermitian'])}


class SavedCompletionPort:
    def __init__(self,manifest,writer,progress,*,root,grant_sha256,payload=None,solver=None,operator_factory=None):
        self.manifest,self.writer,self.progress,self.root=manifest,writer,progress,root
        self.plan,self.caps=manifest['plan'],manifest['plan']['caps']
        self.grant_sha256,self.payload=grant_sha256,payload
        self.solver,self.operator_factory=solver,operator_factory
        self.calls=CallBudget(**{k:self.caps[k] for k in ('acceptance','state_solver','solver_matvec','integral_build','df_decomposition','signal','trajectory','occurrence','compile')})
        self.completed=dict.fromkeys(('acceptance','state_solver','solver_matvec'),0)

    def phase(self,name):
        self.progress.phase(name)

    def execute(self):
        self.phase('saved_input')
        payload=self.payload if self.payload is not None else load_saved_payload(self.root,self.manifest['input_identity'])
        payload=dict(payload);hyp=payload.pop('hypothetical_projection')
        self.phase('df_acceptance');self.calls.take('acceptance')
        if len(payload['lambdas'])!=19 or np.asarray(payload['one_body']).shape!=(12,12):
            raise ValueError('SAVED_COMPLETION_H6_SCOPE')
        arrays,projection,coefficients=accept(**payload)
        if not projection['independent_coefficients_checked'] or not projection['saved_summary_checked']:
            raise ValueError('SAVED_COMPLETION_REQUIRED_INDEPENDENCE')
        if not np.array_equal(arrays['one_body'],hyp[0]) or not np.array_equal(arrays['g_matrices'],hyp[1]):
            raise ValueError('SAVED_COMPLETION_HYPOTHETICAL_PROJECTION_MISMATCH')
        projection['parents']=self.manifest['input_identity']
        projection['source_commit']=self.manifest['source_commit']
        projection['authorization_sha256']=self.grant_sha256
        self.projection=projection
        self.completed['acceptance']+=1
        saved=atomic_npz(self.writer,'accepted_df.npz',arrays,expanded_cap=self.caps['snapshot_expanded_bytes'])
        self.writer.write('accepted_df_receipt.json',{**saved,'projection_receipt':projection})
        coeff=atomic_npz(self.writer,'independent_coefficients.npz',coefficients,expanded_cap=self.caps['snapshot_expanded_bytes'])
        self.writer.write('coefficient_receipt.json',coeff)
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _sector_hash
        from .ax2b_h4_science_v5 import primitive_sector_certificate
        metadata={'input_policy':projection['policy']['name'],'df_rank_actual':19,'df_tol_requested':1e-8,
            'final_rank_supplied':False,'subsequent_coefficient_cutoff':0.,
            'coefficient_order':'decomposer_generation_order','df_truncation_value':float(payload['truncation']),
            'representation_error_certified':False,'projection_receipt_digest':digest(projection),
            'parents':self.manifest['input_identity']}
        ham=DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),metadata)
        sector=PhysicalSector.spin_sector(n_qubits=12,nelec_alpha=3,nelec_beta=3)
        certificate=primitive_sector_certificate(ham,sector)
        self.df_receipt={'hamiltonian_metadata':metadata,'projection_receipt':projection,'actual_rank':19,
            'hamiltonian_hash':df_hamiltonian_hash(ham),'sector_certificate':certificate,'sector_hash':_sector_hash(sector)}
        self.writer.write('df_receipt.json',self.df_receipt)
        self.progress.update(point='df_accepted_saved',last_completed_record='df_receipt.json',calls_attempted=dict(self.calls.used),calls_completed=dict(self.completed))
        self.save_state(ham,sector,certificate)

    def save_state(self,ham,sector,certificate):
        from .ax2b_h6_controller import bounded_solver
        from trotterlib.df_hamiltonian import df_linear_operator, _hartree_fock_initial_vector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _array_hash, _sector_hash, _state_hash, _canonicalize_state_phase
        self.phase('state_snapshot')
        initial = _hartree_fock_initial_vector(sector)
        if initial is None:
            raise ValueError('FIXED_HF_INITIAL_VECTOR_MISSING')
        if self.operator_factory is None:
            operator, _counter = df_linear_operator(ham, sector, backend='numba', num_threads=1, block_chunk_size=1)
        else:
            operator = self.operator_factory(ham, sector)
        def action(vector):
            # bounded_solver increments the shared before-call budget first.
            number = self.calls.used['solver_matvec']
            if number == 1 or number % 100 == 0:
                self.progress.update(point='solver_matvec_started', calls_attempted=dict(self.calls.used),
                                     calls_completed=dict(self.completed))
            result = np.asarray(operator @ vector, dtype=np.complex128)
            if result.shape != (400,) or not np.isfinite(result).all():
                raise ValueError('STATE_MATVEC_RESULT')
            self.completed['solver_matvec'] += 1
            if number == 1 or number % 100 == 0:
                self.progress.update(point='solver_matvec_completed', calls_attempted=dict(self.calls.used),
                                     calls_completed=dict(self.completed))
            return result
        if self.solver is None:
            from scipy.sparse.linalg import eigsh
            self.solver = eigsh
        self.calls.take('state_solver')
        (values, vectors), bounded, counter = bounded_solver(action, 400, initial, self.calls, solver=self.solver)
        self.completed['state_solver'] += 1
        values, vectors = np.asarray(values), np.asarray(vectors)
        if (values.shape != (1,) or vectors.shape != (400,1) or not np.isfinite(values).all()
                or not np.isfinite(vectors).all() or np.any(np.imag(values) != 0)):
            raise ValueError('STATE_SOLVER_RESULT')
        energy = float(np.real(values[0]))
        state = np.array(vectors[:,0], dtype='<c16', copy=True)
        norm_before = float(np.linalg.norm(state))
        if not math.isfinite(norm_before) or norm_before <= 0:
            raise ValueError('STATE_SOLVER_NORM')
        state /= norm_before
        full = np.zeros(4096, dtype='<c16')
        full[sector.basis_indices] = state
        full, state = _canonicalize_state_phase(full, state)
        residual = float(np.linalg.norm(bounded @ state-energy*state))
        threshold = 1e-9+1e-10*max(1., abs(energy))
        if not math.isfinite(residual) or residual > threshold:
            raise ValueError('STATE_RESIDUAL_GATE')
        if abs(float(np.linalg.norm(state))-1.) > 1e-12 or abs(float(np.linalg.norm(full))-1.) > 1e-12:
            raise ValueError('SAVED_STATE_NORM')
        metadata = {'model':'linear_H6', 'geometry_angstrom':1., 'basis':'sto-3g',
                    'hamiltonian_metadata':ham.metadata,
                    'sector':{'n_qubits':12,'nelec_alpha':3,'nelec_beta':3,'dimension':400},
                    'sector_hash':_sector_hash(sector), 'hamiltonian_hash':df_hamiltonian_hash(ham),
                    'state_vector_hash':_array_hash(full),'sector_state_vector_hash':_array_hash(state),
                    'state_hash':_state_hash(full,state), 'sector_certificate':certificate,
                    'state_policy':self.plan['state_policy'], 'solver_energy':energy,
                    'global_phase_policy':'largest_sector_amplitude_real_positive_v1',
                    'solver_residual':residual, 'solver_residual_threshold':threshold,
                    'solver_matvec_calls_including_residual':counter.used['calls'],
                    'state_norm_before_final_normalization':norm_before,
                    'ground_state_certified':False, 'numerical_allowance_certified':False,
                    'accuracy_eligibility':'UNDETERMINED', 'N':None, 'G':None,
                    'saved_df_completion':{'source_commit':self.manifest['source_commit'],
                        'authorization_sha256':self.grant_sha256, 'retry':False, 'resume':False},
                    'H6_status':'H6_NOT_AUTHORIZED', 'contract_status':'DRAFT_NOT_AUTHORIZATION'}
        arrays = {'constant':np.array(ham.constant,dtype='<f8'),
                  'one_body':np.ascontiguousarray(ham.one_body,dtype='<c16'),
                  'lambdas':np.ascontiguousarray(ham.lambdas,dtype='<f8'),
                  'g_matrices':np.ascontiguousarray(ham.g_matrices,dtype='<c16'),
                  'sector_basis_indices':np.ascontiguousarray(sector.basis_indices,dtype='<i8'),
                  'state_vector':full,'sector_state_vector':state,
                  'metadata_json':np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))}
        saved = atomic_npz(self.writer,'h6_input_snapshot.npz',arrays,
                           expanded_cap=self.caps['snapshot_expanded_bytes'])
        # Schema/hash compatibility check only; no signal/circuit computation.
        from .ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        load_h6_snapshot(self.writer.output/'h6_input_snapshot.npz', dict(saved,metadata=metadata), self.df_receipt)
        self.writer.write('state_receipt.json', {'state_hash':metadata['state_hash'],
            'hamiltonian_hash':metadata['hamiltonian_hash'],'residual':residual,'threshold':threshold,
            'matvec_including_residual':counter.used['calls'],'ground_state_certified':False})
        self.writer.write('snapshot_receipt.json', {**saved,'metadata':metadata,
                          'loader_roundtrip_checked':True,'calls_attempted':dict(self.calls.used),
                          'calls_completed':dict(self.completed),'mandatory_stop':True,
                          'next_stage_authorized':False,'H6_status':'H6_NOT_AUTHORIZED'})
        self.progress.update(point='snapshot_saved', last_completed_record='snapshot_receipt.json',
                             calls_attempted=dict(self.calls.used), calls_completed=dict(self.completed))
