"""Pauli-specific interface records, independent of DF rank/partition contexts."""
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from pathlib import PurePosixPath
import math
import re


class PipelineError(ValueError):
    pass


class ScientificExecutionDisabled(PipelineError):
    pass


def required(value, name):
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"explicit {name} required")


def full_sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise PipelineError("full commit SHA required")


@dataclass(frozen=True)
class Locator:
    commit: str
    path: str
    role: str

    def __post_init__(self):
        full_sha(self.commit)
        if (not self.path or PurePosixPath(self.path).is_absolute() or
                '..' in PurePosixPath(self.path).parts or ':' in self.path or '\\' in self.path):
            raise PipelineError("repository-relative evidence path required")
        required(self.role, 'locator role')


@dataclass(frozen=True)
class CodeVersion:
    commit: str | None
    state: str

    def __post_init__(self):
        if self.state not in ('CLEAN_CHECKOUT', 'DIRTY_WORKING_COPY', 'UNVERSIONED_FIXTURE'):
            raise PipelineError('explicit code state required')
        if self.state == 'UNVERSIONED_FIXTURE':
            if self.commit is not None:
                raise PipelineError('an unversioned fixture cannot claim a code commit')
        else:
            full_sha(self.commit)


@dataclass(frozen=True)
class CompilerProfile:
    basis_gates: tuple[str, ...]
    backend_name: str | None
    coupling_map: tuple[tuple[int, int], ...] | None
    optimization_level: int
    layout_method: str | None
    routing_method: str | None
    transpiler_seed: int
    qiskit_version: str

    def __post_init__(self):
        if not isinstance(self.basis_gates, tuple) or not self.basis_gates:
            raise PipelineError('explicit compiler basis required')
        required(self.qiskit_version, 'compiler version')
        if type(self.optimization_level) is not int or self.optimization_level not in range(4):
            raise PipelineError('compiler optimization level')
        if type(self.transpiler_seed) is not int or self.transpiler_seed < 0:
            raise PipelineError('compiler seed')


@dataclass(frozen=True)
class PauliContext:
    scope: str
    study_id: str
    scientific_source: str
    source_record: Locator
    source_stop: Locator
    result_index: Locator
    result_version: str
    code: CodeVersion
    input_id: str
    hamiltonian_id: str
    term_order_id: str
    state_id: str
    reference_id: str
    norm_id: str
    molecule: str
    representation: str
    geometry: str
    basis: str
    num_qubits: int
    term_count: int
    constant: float
    bit_order: str
    term_order_rule: str
    method: str
    candidate_id: str
    prefix: int
    q: int
    r: int
    K: int
    pf_order: str
    evolution_time: float
    effective_law: str
    cdf_graph: str
    normalization_scope: str
    axes: tuple[str, str]
    state_preparation: str
    control: str
    phase_scope: str
    precision_bits: int
    epsilon_sig: str
    compiler: CompilerProfile
    estimand: str
    metric: str
    cost_scope: str
    cohort_id: str
    cohort_role: str
    trajectory_index: int
    seed_id: str
    seed_value: int
    input_kind: str
    stop_status: str

    def __post_init__(self):
        for name in ('scope', 'study_id', 'result_version', 'input_id', 'hamiltonian_id',
                     'term_order_id', 'state_id', 'reference_id', 'norm_id', 'molecule',
                     'representation', 'geometry', 'basis', 'bit_order', 'term_order_rule', 'method', 'candidate_id',
                     'pf_order', 'effective_law', 'cdf_graph', 'normalization_scope',
                     'state_preparation', 'control', 'phase_scope', 'epsilon_sig', 'estimand',
                     'metric', 'cost_scope', 'cohort_id', 'seed_id'):
            required(getattr(self, name), name)
        full_sha(self.scientific_source)
        for name in ('source_record', 'source_stop', 'result_index'):
            if not isinstance(getattr(self, name), Locator):
                raise PipelineError(f'explicit {name} locator required')
        if not isinstance(self.code, CodeVersion) or not isinstance(self.compiler, CompilerProfile):
            raise PipelineError('explicit code version and compiler profile required')
        for name in ('num_qubits', 'term_count', 'prefix', 'q', 'r', 'K',
                     'precision_bits', 'trajectory_index', 'seed_value'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise PipelineError(f'{name} must be a nonnegative Python integer')
        if min(self.num_qubits, self.term_count, self.q, self.r, self.precision_bits) < 1:
            raise PipelineError('positive dimensions/q/r/precision required')
        if self.prefix > self.term_count or self.K % 2:
            raise PipelineError('prefix or finite Taylor K')
        if self.cohort_role not in ('exploration', 'confirmation'):
            raise PipelineError('explicit exploration/confirmation role required')
        if self.input_kind not in ('synthetic', 'frozen_scientific'):
            raise PipelineError('input kind')
        if self.stop_status not in ('HISTORICAL_STOP_PRESERVED', 'STOP'):
            raise PipelineError('old STOP cannot be cleared or resumed')
        for value in (self.constant, self.evolution_time):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PipelineError('explicit finite constant and time required')
        if self.evolution_time <= 0:
            raise PipelineError('positive time required')

    @property
    def binding(self):
        return repr(asdict(self))


@dataclass(frozen=True)
class StageResult:
    stage: str
    binding: str
    status: str
    values: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ''
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PipelineResult:
    mode: str
    context: PauliContext
    stages: tuple[StageResult, ...]
    qualification: str

    def to_dict(self):
        return dict(mode=self.mode, context=asdict(self.context),
                    stages=[dict(stage=s.stage, status=s.status, values=dict(s.values), detail=s.detail)
                            for s in self.stages], qualification=self.qualification,
                    science_executed=False, numerical_equivalence_verified=False,
                    reproduces_saved_results=False)
