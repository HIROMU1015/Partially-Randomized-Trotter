"""Artificial terms/tokens and mocks only; never reconstruct a molecular state."""
from dataclasses import replace
from itertools import islice, product
from types import SimpleNamespace
from unittest.mock import Mock

from .contexts import CodeVersion, CompilerProfile, PauliContext
from .adapters import FrozenPauliInput, PauliStudyAdapter, H6_CANDIDATES, evidence
from .pipeline import Providers
from . import core_stages as stages


def synthetic_context(scope='h4-hybrid', *, code=None, candidate_id=None):
    if candidate_id is not None and scope != 'h6-hybrid':
        from .contexts import PipelineError
        raise PipelineError('candidate ID selection is only connected for H6 Hybrid')
    if scope == 'h4-hybrid':
        molecule, n, count, method = 'H4', 8, 184, 'HybridPR'
        cid, sig = 'p036_q2_r8_K4', (36, 2, 8, 4)
    elif scope == 'h6-hybrid':
        molecule, n, count, method = 'H6', 12, 918, 'HybridPR'
        cid = candidate_id or 'PR_p0078_q8_r16_K2'
        if cid not in H6_CANDIDATES:
            from .contexts import PipelineError
            raise PipelineError('fixed four H6 candidates only')
        sig = H6_CANDIDATES[cid]
    elif scope == 'h6-rte':
        molecule, n, count, method = 'H6', 12, 918, 'PureRTE'
        cid, sig = 'H6_p000_q8_r64_K2', (0, 8, 64, 2)
    else:
        from .contexts import PipelineError
        raise PipelineError('no synthetic provider for this scope')
    source, record, stop, result = evidence(scope)
    prefix, q, r, K = sig
    return PauliContext(
        scope=scope, study_id='P2_ARTIFICIAL_'+scope, scientific_source=source,
        source_record=record, source_stop=stop, result_index=result,
        result_version='ARTIFICIAL_NOT_A_SAVED_RESULT',
        code=code or CodeVersion(None, 'UNVERSIONED_FIXTURE'),
        input_id='ARTIFICIAL_INPUT_'+scope, hamiltonian_id='ARTIFICIAL_HAMILTONIAN_'+scope,
        term_order_id='ARTIFICIAL_TERM_ORDER_'+scope, state_id='ARTIFICIAL_STATE_'+scope,
        reference_id='ARTIFICIAL_REFERENCE_'+scope, norm_id='ARTIFICIAL_NORM_'+scope,
        molecule=molecule, representation='pauli_jw', geometry='linear:1.00Angstrom', basis='STO-3G',
        num_qubits=n, term_count=count, constant=-0.125,
        bit_order='qiskit_little_endian', term_order_rule='descending_abs_then_label',
        method=method, candidate_id=cid, prefix=prefix, q=q, r=r, K=K, pf_order='S2',
        evolution_time=0.8, effective_law='finite_rte_even_paired',
        cdf_graph='numpy1.26.4_CDF_53bit_uniform_word',
        normalization_scope='B_saved_once_complete_trajectory_raw_norm_unchanged',
        axes=('cosine', 'sine:+Im:Sdg_before_H'), state_preparation='excluded_P0',
        control='central_rotation', phase_scope='constant_once_global_to_relative',
        precision_bits=128, epsilon_sig='.05',
        compiler=CompilerProfile(('rz', 'sx', 'x', 'cx'), None, None, 1, None, None, 17, '1.3.0'),
        estimand='fixed_complex_signal', metric='rz_count',
        cost_scope='full_hadamard_wrapper_rz_without_state_preparation',
        cohort_id='ARTIFICIAL_COHORT_'+scope, cohort_role='confirmation', trajectory_index=0,
        seed_id='ARTIFICIAL_SEED_TOKEN_'+scope, seed_value=123,
        input_kind='synthetic', stop_status='HISTORICAL_STOP_PRESERVED')


def synthetic_fixture(c=None):
    c = c or synthetic_context()
    PauliStudyAdapter(c.scope).validate(c)
    if c.input_kind != 'synthetic':
        from .contexts import ScientificExecutionDisabled
        raise ScientificExecutionDisabled('only artificial fixture input is available')
    # Distinct, already label-ordered nonidentity strings, arbitrary signed units.
    # No physical Hamiltonian/operator, RNG, state vector, norm or law is evaluated.
    labels = (''.join(x) for x in islice(product('IXYZ', repeat=c.num_qubits), 1, c.term_count+1))
    terms = tuple((label, -1.0 if i == 0 else 1.0) for i, label in enumerate(labels))
    raw, reference = object(), object()
    model = dict(n_qubits=c.num_qubits, state=raw,
                 hamiltonian=dict(constant=c.constant, terms=terms), Hamiltonian_hash=c.hamiltonian_id)
    data = FrozenPauliInput(c.binding, c.input_id, c.hamiltonian_id, c.term_order_id, c.state_id,
        c.reference_id, c.norm_id, c.bit_order, c.constant, c.num_qubits, terms, model, raw, reference)
    wrappers = {'cosine': 'ARTIFICIAL_COSINE_NO_CIRCUIT', 'sine': 'ARTIFICIAL_SINE_NO_CIRCUIT'}
    core = stages.CoreBindings(c.scope, 'synthetic', Mock(side_effect=lambda **k: SimpleNamespace(**k)),
        Mock(return_value='ARTIFICIAL_CONFIG'), Mock(return_value='ARTIFICIAL_LAW'),
        Mock(return_value=('ARTIFICIAL_BODY_NO_CIRCUIT', wrappers, {'seed': c.seed_value})),
        Mock(return_value='ARTIFICIAL_TARGET_NO_MEAN'),
        Mock(return_value=dict(N=None, N_total=None, G=None, SE=None)),
        Mock(side_effect=lambda **k: SimpleNamespace(**k)),
        Mock(return_value=SimpleNamespace(rz_count=None)), Mock(return_value=c.seed_value))
    return data, Providers('synthetic', core,
        lambda ctx, prev: stages.prepare(core, ctx, prev),
        lambda ctx, prev: stages.event(core, ctx, prev),
        lambda ctx, prev: stages.signal(core, ctx, prev),
        lambda ctx, prev: stages.account_price(core, ctx, prev))
