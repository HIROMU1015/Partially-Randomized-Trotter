"""Complete state from existing raw H8 inputs; no molecular/decomposer API."""
from pathlib import Path
import json,time,numpy as np
from .h8_preflight_input_port_v1 import InputGenerationPort,atomic_npz
from .h48_preflight_contract_v1 import bounded_h8_solver as bounded_solver
from .ax2b_h4_science_v5 import primitive_sector_certificate
from .ax2a_preparation import digest
from .h8_saved_projection_v1 import accept,norm,array_record,reconstruct,canonical_antisym
from .ax2b_h4_contract_v5 import file_hash
from trotterlib.df_hamiltonian import PhysicalSector,DFHamiltonian,df_linear_operator,_hartree_fock_initial_vector
from trotterlib.df_partial_s2 import df_hamiltonian_hash
from trotterlib.pr2_s0_s1_validation import _array_hash,_sector_hash,_state_hash,_canonicalize_state_phase

def antisym(t):
    return (t-t.swapaxes(0,1)-t.swapaxes(2,3)+t.swapaxes(0,1).swapaxes(2,3))/4

def independent_coefficients(corrected,w,g):
    return (corrected+np.einsum('i,ipq,iqr->pr',w,g,g,optimize=False),
            antisym(-np.einsum('i,ipr,iqs->pqrs',w,g,g,optimize=False)))

def scalar_summary(one,two,corrected,w,g,post):
    # Near-zero norm summaries use the frozen evaluation order. The separate
    # vectorized coefficient check still tests the entire arrays independently.
    raw_c,raw_t=reconstruct(corrected,w,g)
    post_c,post_t=reconstruct(corrected,w,post)
    return {'raw_normal_order_one_body_residual_frobenius':norm(raw_c-one),
        'raw_normal_order_antisym_two_body_residual_frobenius':norm(canonical_antisym(two)-raw_t),
        'hypothetical_projection_one_body_change_frobenius':norm(raw_c-post_c),
        'hypothetical_projection_antisym_two_body_change_frobenius':norm(raw_t-post_t)}

class SavedStatePort(InputGenerationPort):
    def execute(self):
        self.phase('saved_raw_projection')
        parent=Path(self.manifest['saved_parent']['directory'])
        for name,h in self.manifest['saved_parent']['file_hashes'].items():
            if file_hash(parent/name)!=h:raise ValueError('SAVED_RAW_PARENT_CHANGED')
        with np.load(parent/'integrals.npz',allow_pickle=False) as z:
            one,two,constant=z['one_body'].copy(),z['two_body'].copy(),float(z['constant'])
        with np.load(parent/'raw_df.npz',allow_pickle=False) as z:
            weights,g,correction,truncation=(z[k].copy() for k in ('weights','blocks','correction','truncation'))
        corrected=one+correction;post=(g+g.conj().transpose(0,2,1))/2
        raw_c,raw_t=independent_coefficients(corrected,weights,g)
        fragment_post_c,post_t=independent_coefficients(corrected,weights,post)
        summary={'actual_rank':len(weights),'kwargs':{'truncation_threshold':1e-8},
          'truncation_value':float(truncation),'corrected_one_body_hermitization_change_frobenius':norm((corrected+corrected.conj().T)/2-corrected),
          'fragments':[{'index':i,'lambda':float(lam),'raw':array_record(a),'hypothetical_post':array_record(b),
             'raw_frobenius_norm':norm(a),'hermitization_change_frobenius':norm(a-b)} for i,(lam,a,b) in enumerate(zip(weights,g,post))],
          'representation_diagnostics':scalar_summary(one,two,corrected,weights,g,post)}
        arrays,receipt,coeffs=accept(constant=constant,one_body=one,two_body=two,lambdas=weights,
           g_matrices=g,correction=correction,truncation=truncation,summary=summary,
           diagnostic_coefficients=(raw_c-one,raw_t,antisym(two)))
        receipt['parents']=self.manifest['saved_parent']
        hm={'input_policy':receipt['policy']['name'],'df_rank_actual':len(weights),'df_tol_requested':1e-8,
             'coefficient_order':'decomposer_generation_order','subsequent_coefficient_cutoff':0.,
             'projection_receipt_digest':digest(receipt),'parents':self.manifest['saved_parent']}
        ham=DFHamiltonian(constant,arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),hm)
        sector=PhysicalSector.spin_sector(n_qubits=16,nelec_alpha=4,nelec_beta=4)
        certificate=primitive_sector_certificate(ham,sector)
        self.writer.write('projection_receipt.json',receipt)
        self.writer.write('df_receipt.json',{'hamiltonian_metadata':hm,'actual_rank':len(weights),
          'hamiltonian_hash':df_hamiltonian_hash(ham),'projection_receipt':receipt,
          'new_integral_builds':0,'new_df_decompositions':0,'prior_integral_builds':1,'prior_df_decompositions':1})
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
            if result.shape != (4900,) or not np.isfinite(result).all():
                raise ValueError('STATE_MATVEC_RESULT')
            self.completed['solver_matvec'] += 1
            if number == 1 or number % 100 == 0:
                self.progress.update(point='solver_matvec_completed', calls_attempted=dict(self.calls.used),
                                     calls_completed=dict(self.completed))
            return result
        if self.solver is None:
            from scipy.sparse.linalg import eigsh
            self.solver = eigsh
        (values, vectors), bounded, counter = bounded_solver(action, 4900, initial, self.calls, solver=self.solver)
        values, vectors = np.asarray(values), np.asarray(vectors)
        if (values.shape != (1,) or vectors.shape != (4900,1) or not np.isfinite(values).all()
                or not np.isfinite(vectors).all() or np.any(np.imag(values) != 0)):
            raise ValueError('STATE_SOLVER_RESULT')
        energy = float(np.real(values[0]))
        state = np.array(vectors[:,0], dtype='<c16', copy=True)
        norm_before = float(np.linalg.norm(state))
        if not math.isfinite(norm_before) or norm_before <= 0:
            raise ValueError('STATE_SOLVER_NORM')
        state /= norm_before
        full = np.zeros(65536, dtype='<c16')
        full[sector.basis_indices] = state
        full, state = _canonicalize_state_phase(full, state)
        residual = float(np.linalg.norm(bounded @ state-energy*state))
        threshold = 1e-9+1e-10*max(1., abs(energy))
        if not math.isfinite(residual) or residual > threshold:
            raise ValueError('STATE_RESIDUAL_GATE')
        if abs(float(np.linalg.norm(state))-1.) > 1e-12 or abs(float(np.linalg.norm(full))-1.) > 1e-12:
            raise ValueError('SAVED_STATE_NORM')
        metadata = {'model':'linear_H8', 'geometry_angstrom':1., 'basis':'sto-3g',
                    'hamiltonian_metadata':ham.metadata,
                    'sector':{'n_qubits':16,'nelec_alpha':4,'nelec_beta':4,'dimension':4900},
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
                    'input_generation':{'source_commit':self.manifest['source_commit'],
                        'authorization_sha256':self.grant_sha256, 'retry':False, 'resume':False},
                    'H8_full_campaign_authorized':False, 'contract_status':'DRAFT_NOT_AUTHORIZATION'}
        arrays = {'constant':np.array(ham.constant,dtype='<f8'),
                  'one_body':np.ascontiguousarray(ham.one_body,dtype='<c16'),
                  'lambdas':np.ascontiguousarray(ham.lambdas,dtype='<f8'),
                  'g_matrices':np.ascontiguousarray(ham.g_matrices,dtype='<c16'),
                  'sector_basis_indices':np.ascontiguousarray(sector.basis_indices,dtype='<i8'),
                  'state_vector':full,'sector_state_vector':state,
                  'metadata_json':np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))}
        saved = atomic_npz(self.writer,'h8_input_snapshot.npz',arrays,
                           expanded_cap=self.caps['snapshot_expanded_bytes'])
        # Schema/hash compatibility check only; no signal/circuit computation.
        from .h48_preflight_contract_v1 import verify_h8_snapshot_roundtrip
        verify_h8_snapshot_roundtrip(self.writer.output/'h8_input_snapshot.npz', arrays)
        self.writer.write('snapshot_receipt.json', {**saved,'metadata':metadata,
                          'loader_roundtrip_checked':True,'calls_attempted':dict(self.calls.used),
                          'calls_completed':dict(self.completed),'mandatory_stop':True,
                          'next_stage_authorized':False,'H8_full_campaign_authorized':False})
        self.progress.update(point='snapshot_saved', last_completed_record='snapshot_receipt.json',
                             calls_attempted=dict(self.calls.used), calls_completed=dict(self.completed))
