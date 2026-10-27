"""Narrow saved Pauli scopes. No snapshot is opened or science called here."""
from dataclasses import dataclass
from typing import Any
import json

from .contexts import Locator, PauliContext, PipelineError, StageResult

BASE = '00958ae1945c780718b2deef07d5dab1096a5cd8'
SUMMARY = 'b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b'
H6_CANDIDATES = {
    'PR_p0078_q8_r16_K2': (78, 8, 16, 2),
    'PR_p0078_q8_r32_K2': (78, 8, 32, 2),
    'PR_p0078_q8_r64_K2': (78, 8, 64, 2),
    'PR_p0684_q1_r16_K4': (684, 1, 16, 4),
}


def evidence(scope):
    if scope == 'h4-hybrid':
        directory = 'artifacts/pauli_hybrid_rte_comparison/2026-10-25/'
        source = 'b19ed83c6b4b64821965fa0b756cd2691e34eda8'
        result = Locator(SUMMARY, 'docs/research/pauli_jw_resource_evaluation_summary.md',
                         'SAVED_RESULT_INDEX_AS_OF_SUMMARY')
    elif scope == 'h6-hybrid':
        directory = 'artifacts/h6_hybrid_fairness_extension/2026-10-27/'
        source = '887b69e449cb3c378286f09d0d7626599850fb67'
        result = Locator(BASE, directory+'README.md', 'SAVED_RESULT_INDEX')
    elif scope == 'h6-rte':
        directory = 'artifacts/pauli_rte_event_extension/2026-10-26/'
        source = 'e755ecf581600192fd2745d7066e410cebd933d3'
        result = Locator('10cb3e5cca7834ebf75301bd91bd093ca51d4a6b', directory+'README.md', 'SAVED_RESULT_INDEX')
        return source, Locator(BASE, directory+'SOURCE_'+source+'.json', 'SOURCE_RECORD'), Locator(BASE, directory+'STUDY_STOP.json', 'HISTORICAL_STOP'), result
    else:
        raise PipelineError('unsupported connected study scope')
    return source, Locator(BASE, directory+'SOURCE.json', 'SOURCE_RECORD'), Locator(BASE, directory+'STUDY_STOP.json', 'HISTORICAL_STOP'), result


def h2_saved_index():
    directory = 'artifacts/h2_pauli_rte_validation/2026-10-26/'
    return dict(study_id='H2_SAVED_SOURCE_INDEX', scientific_source='bf5f2339f723f4edb7bf91a8ca88f33251bbef57',
                source_record=Locator(BASE, directory+'SOURCE.json', 'SOURCE_RECORD'),
                result_index=Locator('4aea054b3369709bc0d35ff081071e62b5f77cbe', directory+'README.md', 'SAVED_RESULT_INDEX'),
                source_stop=Locator(BASE, directory+'STUDY_STOP.json', 'HISTORICAL_STOP'),
                qualification='UNCONNECTED_SOURCE_INDEX',
                reason='H2 uses its saved study script/snapshot namespace; real loader/provider and full wiring are not migrated.')


@dataclass(frozen=True)
class FrozenPauliInput:
    binding: str
    input_id: str
    hamiltonian_id: str
    term_order_id: str
    state_id: str
    reference_id: str
    norm_id: str
    bit_order: str
    constant: float
    num_qubits: int
    terms: tuple
    model: Any
    raw_state: Any
    reference: Any


class PauliStudyAdapter:
    """Connected H4 Hybrid, fixed-four H6 Hybrid, or one saved H6 pure-RTE case.

    These restrictions describe a technical connection, never a launch allowlist.
    PF/B0 and other candidates stay unconnected rather than widening old contracts.
    """
    def __init__(self, scope):
        if scope not in ('h4-hybrid', 'h6-hybrid', 'h6-rte'):
            raise PipelineError('unsupported connected scope')
        self.scope = scope

    def api_paths(self, c):
        """Study-selected existing callbacks; merely declaring them imports nothing."""
        self.validate(c)
        if self.scope == 'h4-hybrid':
            module = 'trotterlib.h4_pauli_pr_search'
            return dict(config=(module, 'config'), law=(module, 'law_descriptor'),
                        trajectory=(module, 'central_trajectory'), account=(module, 'account'),
                        mean=('trotterlib.h4_pauli_pr_mean', '_mean_enclosure'),
                        seed=(module, 'trajectory_seed'))
        module = ('trotterlib.h6_hybrid_fairness_extension' if self.scope == 'h6-hybrid'
                  else 'trotterlib.pauli_rte_event_extension')
        return dict(config=(module, 'config'), law=(module, 'law_descriptor'),
                    trajectory=(module, 'trajectory'), account=(module, 'account_target'),
                    mean=(module, 'mean_target' if self.scope == 'h6-hybrid' else 'h6_mean'), seed=None)

    def bind_core(self, c):
        from .core_stages import bind_existing_core
        return bind_existing_core(self, c)

    def validate(self, c):
        if not isinstance(c, PauliContext) or c.scope != self.scope:
            raise PipelineError('study scope mismatch')
        source, record, stop, result = evidence(self.scope)
        if (c.scientific_source, c.source_record, c.source_stop, c.result_index) != (source, record, stop, result):
            raise PipelineError('fixed scientific SOURCE/result/STOP locator mismatch')
        n, terms = (8, 184) if self.scope == 'h4-hybrid' else (12, 918)
        if (c.molecule, c.num_qubits, c.term_count) != ('H4' if n == 8 else 'H6', n, terms):
            raise PipelineError('molecule/term dimension mismatch')
        if c.representation != 'pauli_jw':
            raise PipelineError('Pauli-JW representation required')
        if c.geometry != 'linear:1.00Angstrom' or c.basis != 'STO-3G':
            raise PipelineError('fixed geometry/basis')
        if c.bit_order != 'qiskit_little_endian' or c.term_order_rule != 'descending_abs_then_label':
            raise PipelineError('saved bit/label order required; no reordering fallback')
        if c.axes != ('cosine', 'sine:+Im:Sdg_before_H'):
            raise PipelineError('cosine / +Im sine convention')
        if (c.state_preparation, c.control, c.phase_scope, c.cost_scope) != (
                'excluded_P0', 'central_rotation', 'constant_once_global_to_relative',
                'full_hadamard_wrapper_rz_without_state_preparation'):
            raise PipelineError('state preparation/control/phase/cost scope')
        if c.estimand != 'fixed_complex_signal' or c.metric != 'rz_count':
            raise PipelineError('fixed-signal metric; no phase-RPE budget')
        if (c.pf_order, c.evolution_time, c.effective_law, c.cdf_graph, c.normalization_scope) != (
                'S2', 0.8, 'finite_rte_even_paired', 'numpy1.26.4_CDF_53bit_uniform_word',
                'B_saved_once_complete_trajectory_raw_norm_unchanged'):
            raise PipelineError('PF order / finite K / effective law / normalization mismatch')
        if c.precision_bits != 128 or c.epsilon_sig not in ('.05', '.01', '.005', '.001', '.0005'):
            raise PipelineError('explicit connected precision scope')
        p = c.compiler
        if (p.basis_gates, p.backend_name, p.coupling_map, p.optimization_level,
            p.layout_method, p.routing_method, p.transpiler_seed, p.qiskit_version) != (
                ('rz', 'sx', 'x', 'cx'), None, None, 1, None, None, 17, '1.3.0'):
            raise PipelineError('saved compiler profile mismatch; no default replacement')
        signature = (c.prefix, c.q, c.r, c.K)
        if self.scope == 'h4-hybrid':
            expected = (36, 2, 8, 4)
            if c.candidate_id != 'p036_q2_r8_K4' or signature != expected or c.method != 'HybridPR':
                raise PipelineError('one saved H4 Hybrid connection only')
        elif self.scope == 'h6-hybrid':
            if c.method != 'HybridPR' or H6_CANDIDATES.get(c.candidate_id) != signature:
                raise PipelineError('fixed four H6 Hybrid signatures only')
        elif c.method != 'PureRTE' or signature != (0, 8, 64, 2) or c.candidate_id != 'H6_p000_q8_r64_K2':
            raise PipelineError('one saved H6 pure-RTE connection only')

    def input(self, c, data, core):
        self.validate(c)
        if not isinstance(data, FrozenPauliInput) or data.binding != c.binding:
            raise PipelineError('input/SOURCE/cohort binding mismatch')
        for name in ('input_id', 'hamiltonian_id', 'term_order_id', 'state_id', 'reference_id',
                     'norm_id', 'bit_order', 'constant', 'num_qubits'):
            if getattr(data, name) != getattr(c, name):
                raise PipelineError(f'saved {name} mismatch; no repair or renormalization')
        if (not isinstance(data.terms, tuple) or len(data.terms) != c.term_count
                or data.model is None or data.raw_state is None or data.reference is None):
            raise PipelineError('explicit saved-field payload required; no generation fallback')
        # Pass supplied term order/signs/constant verbatim. The actual constructor's
        # canonicalization and numerical equivalence are not exercised by mocks.
        ham = core.hamiltonian(num_qubits=c.num_qubits, constant=data.constant, terms=data.terms,
                               metadata_json=json.dumps(dict(input_id=c.input_id, state_id=c.state_id,
                                                             term_order_id=c.term_order_id)))
        return StageResult('input', c.binding, 'SYNTHETIC', {}, payload=dict(data=data, ham=ham))
