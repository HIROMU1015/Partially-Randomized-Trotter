"""Small, dependency-injected DF pipeline. Only plan/mock execution is enabled.

The records describe one study, not a new schema imposed on existing SOURCEs.
No core API is imported or invoked by importing this module or planning a run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol
from pathlib import Path, PurePosixPath
import math
import re

STAGES = ("input", "pf_pr", "finite_rte_event", "accounting", "compiled_cost")
BASE_COMMIT = "f29eafbf4412193e8e8660d9f051889c2cba684b"


class PipelineError(ValueError):
    """A data/qualification boundary rejected a pipeline stage."""


class ScientificExecutionDisabled(PipelineError):
    """P1 does not provide a real-science launch path."""


def nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{name} must be explicitly supplied")


@dataclass(frozen=True)
class EvidenceLocator:
    commit: str
    path: str
    role: str

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise PipelineError("evidence requires a full commit SHA")
        p = PurePosixPath(self.path)
        if (not self.path or p.is_absolute() or ".." in p.parts or
                "\\" in self.path or ":" in self.path):
            raise PipelineError("evidence requires a repository-relative path")
        nonempty(self.role, "evidence role")


@dataclass(frozen=True)
class CompilerProfile:
    """All legacy CompilerSettings fields must be supplied, including None."""
    basis_gates: tuple[str, ...]
    backend_name: str | None
    coupling_map: tuple[tuple[int, int], ...] | None
    optimization_level: int
    layout_method: str | None
    routing_method: str | None
    transpiler_seed: int
    qiskit_version: str

    def __post_init__(self):
        if (not isinstance(self.basis_gates, tuple) or not self.basis_gates
                or any(not isinstance(x, str) or not x for x in self.basis_gates)):
            raise PipelineError("explicit compiler basis required")
        nonempty(self.qiskit_version, "qiskit_version")
        if type(self.optimization_level) is not int or self.optimization_level not in range(4):
            raise PipelineError("invalid compiler optimization level")
        if type(self.transpiler_seed) is not int or self.transpiler_seed < 0:
            raise PipelineError("invalid compiler seed")


@dataclass(frozen=True)
class ScientificContext:
    study_id: str
    cohort_id: str
    result_version: str
    source: EvidenceLocator
    evidence: tuple[EvidenceLocator, ...]
    representation: str
    geometry: str
    basis: str
    sector: str
    state_id: str
    input_id: str
    df_rank: int
    rank_policy: str
    ld: int
    deterministic_indices: tuple[int, ...]
    randomized_indices: tuple[int, ...]
    formula: str
    q: int
    stage_order: tuple[str, ...]
    evolution_time: float
    rte_steps: int
    finite_taylor_order: int
    rte_law: str
    normalization_policy: str
    seed: int
    precision: str
    applicable_domain: str
    estimand: str
    metric: str
    cost_scope: str
    compiler: CompilerProfile
    source_stop_status: str
    input_kind: str

    def __post_init__(self):
        for name in ("study_id", "cohort_id", "result_version", "geometry", "basis",
                     "sector", "state_id", "input_id", "rank_policy", "formula",
                     "rte_law", "normalization_policy", "precision", "applicable_domain",
                     "estimand", "metric", "cost_scope", "source_stop_status"):
            nonempty(getattr(self, name), name)
        if not isinstance(self.source, EvidenceLocator) or self.source.role != "SOURCE":
            raise PipelineError("a fixed SOURCE locator is required")
        if not isinstance(self.compiler, CompilerProfile):
            raise PipelineError("explicit compiler profile required; no default")
        for name in ("df_rank", "ld", "q", "rte_steps", "finite_taylor_order", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise PipelineError(f"{name} must be a nonnegative Python integer")
        if self.ld > self.df_rank or self.finite_taylor_order % 2:
            raise PipelineError("invalid split or finite Taylor order")
        if (self.df_rank == 0 or self.q < 1 or isinstance(self.evolution_time, bool)
                or not math.isfinite(self.evolution_time) or self.evolution_time <= 0):
            raise PipelineError("explicit positive q/time required")
        if any(not isinstance(a, tuple) or any(type(i) is not int or i < 0 for i in a)
               for a in (self.deterministic_indices, self.randomized_indices)):
            raise PipelineError("partition indices must be explicit integer tuples")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(x, EvidenceLocator) for x in self.evidence)):
            raise PipelineError("evidence must be an explicit tuple of locators")
        if set(self.deterministic_indices) & set(self.randomized_indices):
            raise PipelineError("DF partitions overlap")
        order = self.deterministic_indices + self.randomized_indices
        if len(order) != self.df_rank or set(order) != set(range(self.df_rank)):
            raise PipelineError("explicit partition must cover the fixed DF rank")
        if len(self.deterministic_indices) != self.ld:
            raise PipelineError("L_D and deterministic partition disagree")
        if self.input_kind not in ("synthetic", "frozen_scientific"):
            raise PipelineError("unknown input kind")

    @property
    def binding(self) -> tuple[str, ...]:
        """Identity of the complete supplied scope, not an approval certificate."""
        return (self.study_id, self.cohort_id, self.result_version, self.input_id,
                self.state_id, self.source.commit, self.source.path,
                repr(asdict(self)))


@dataclass(frozen=True)
class StageResult:
    stage: str
    binding: tuple[str, ...]
    status: str
    values: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PipelineResult:
    mode: str
    context: ScientificContext
    stages: tuple[StageResult, ...]
    qualification: str
    science_executed: bool = False
    numerical_equivalence_verified: bool = False

    def to_dict(self) -> dict:
        # Never serialize opaque Hamiltonian/state/circuit objects or private arrays.
        return {"mode": self.mode, "context": asdict(self.context),
                "stages": [{"stage": r.stage, "status": r.status,
                            "values": dict(r.values), "detail": r.detail} for r in self.stages],
                "qualification": self.qualification, "science_executed": False,
                "numerical_equivalence_verified": False}


class StudyAdapter(Protocol):
    def validate(self, context: ScientificContext) -> None: ...
    def input(self, context: ScientificContext, payload: Any, core: Any) -> StageResult: ...


@dataclass(frozen=True)
class StageProviders:
    """Explicit callbacks; P1 runner accepts synthetic callbacks only."""
    kind: str
    core: Any
    prepare: Callable
    finite_rte_event: Callable
    accounting: Callable
    compiled_cost: Callable


def compose(context: ScientificContext, adapter: StudyAdapter, *, mode: str = "plan",
            payload: Any = None, providers: StageProviders | None = None) -> PipelineResult:
    # Reject before adapter/input access, imports, output creation or provider calls.
    if mode not in ("plan", "synthetic"):
        raise ScientificExecutionDisabled("real scientific execution is disabled in P1")
    adapter.validate(context)
    if mode == "plan":
        return PipelineResult(mode, context, tuple(StageResult(
            stage, context.binding, "NOT_EXECUTED", {}, "fixed source/STOP retained")
            for stage in STAGES), "PLAN_ONLY")
    if context.input_kind != "synthetic" or not providers or providers.kind != "synthetic":
        raise ScientificExecutionDisabled("synthetic mode requires artificial input and mock providers")
    completed = []
    previous = None
    calls = (lambda: adapter.input(context, payload, providers.core),
             lambda: providers.prepare(context, previous),
             lambda: providers.finite_rte_event(context, previous),
             lambda: providers.accounting(context, previous),
             lambda: providers.compiled_cost(context, previous))
    for stage, callback in zip(STAGES, calls, strict=True):
        try:
            result = callback()
            if not isinstance(result, StageResult) or result.stage != stage:
                raise PipelineError("invalid stage output type/order")
            if result.binding != context.binding:
                raise PipelineError("different SOURCE/input/cohort/compiler/metric scope")
            if not isinstance(result.values, Mapping):
                raise PipelineError("stage values must be a mapping")
            for key in ("representation", "metric", "cost_scope", "precision", "estimand"):
                if key in result.values and result.values[key] != getattr(context, key):
                    raise PipelineError(f"stage changed {key}")
            for key in ("N", "N_cosine", "N_sine", "G", "rz_count"):
                value = result.values.get(key)
                if value is not None and (isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise PipelineError(f"invalid or nonfinite {key}; missing is None")
            if result.status not in ("SYNTHETIC", "UNQUALIFIED", "MISSING", "STOP", "FAILED"):
                raise PipelineError("unknown qualification; a mock cannot claim scientific PASS")
        except Exception as exc:
            # Retain partial stage history; do not retry or invent scientific values.
            completed.append(StageResult(stage, context.binding, "FAILED", {},
                                         f"{type(exc).__name__}: {exc}"))
            break
        completed.append(result)
        previous = result
        if result.status != "SYNTHETIC":
            break
    qualification = ("SYNTHETIC_END_TO_END_PASS" if len(completed) == len(STAGES)
                     and all(r.status == "SYNTHETIC" for r in completed)
                     else completed[-1].status)
    return PipelineResult(mode, context, tuple(completed), qualification)


def write_new_result_directory(result: PipelineResult, directory: str | Path) -> Path:
    """Reuse executor atomic publication only in a newly created output directory.

    This does not create task manifests, checkpoints, grants or STOP records and
    does not launch the executor or any scientific adapter.
    """
    from trotterlib.parallel_validation_executor import atomic_write_json
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    destination = path / "pipeline_result.json"
    atomic_write_json(destination, result.to_dict())
    return destination
