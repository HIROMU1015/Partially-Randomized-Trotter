"""Outer splitting preparation: descriptors and the existing memoized DAG.

Import/build only use stdlib metadata. No fixture, sampler, operator or circuit
is created. The fixed scientific entry point lives in outer_execution/launcher.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math

from trottertracks.b2d_adapter.adapter import (
    DContext, PhaseOperation, TailOccurrence, _expand_append_s2,
    _positive_integer, _rational,
)
from trottertracks.b2d_adapter._published import UPSTREAM_COMMIT, symmetric_s2_sequence
from ._baselines import yoshida_4th_list, morales_2025_y8m10b_list
from .evaluation import Graph, Node, evaluate

METHODS = ("outer-S2", "refined-S2", "outer-Yoshida4", "SABA2")
INNER = {"A2": yoshida_4th_list, "A3": morales_2025_y8m10b_list}


def mathematical_stages(method):
    """Return mathematical product order; keep adjacent D stages distinct."""
    if method in ("outer-S2", "refined-S2"):
        return (("D", Fraction(1, 2)), ("R", Fraction(1)), ("D", Fraction(1, 2)))
    if method == "outer-Yoshida4":
        # Preserve the published floating coefficients, without renormalizing.
        weights = map(Fraction, symmetric_s2_sequence(yoshida_4th_list()))
        return tuple(stage for w in weights for stage in
                     (("D", w / 2), ("R", w), ("D", w / 2)))
    if method == "SABA2":
        # Laskar--Robutel astro-ph/0005074v1, Eq.(14)--(17), Table I.
        c2 = Fraction(1 / math.sqrt(3))
        c1 = (1 - c2) / 2
        return (("D", c1), ("R", Fraction(1, 2)), ("D", c2),
                ("R", Fraction(1, 2)), ("D", c1))
    raise ValueError("Unsupported outer method")


@dataclass(frozen=True)
class OuterCase:
    method: str
    total_time: Fraction
    macro_steps: int
    inner_arm: str
    n_d: int
    refinement: int | None = None

    def __post_init__(self):
        if self.method not in METHODS or self.inner_arm not in INNER:
            raise ValueError("Only the four outer methods and inner A2/A3 are prepared")
        object.__setattr__(self, "total_time", _rational(self.total_time))
        if not self.total_time:
            raise ValueError("Nonzero physical time required")
        for name in ("macro_steps", "n_d"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name)))
        if self.method == "refined-S2":
            if self.refinement is None or _positive_integer(self.refinement) < 2:
                raise ValueError("Refinement must be chosen explicitly and be >=2")
        elif self.refinement is not None:
            raise ValueError("Refinement applies only to refined-S2")

    @property
    def step_count(self):
        return self.macro_steps * (self.refinement or 1)


@dataclass(frozen=True)
class RTEStage:
    duration: Fraction
    rte_steps: int
    cutoff: int
    truncation_tolerance: float
    seed: int

    def __post_init__(self):
        object.__setattr__(self, "duration", _rational(self.duration))
        _positive_integer(self.rte_steps)
        if not self.duration or type(self.cutoff) is not int or self.cutoff < 0 or self.cutoff % 2:
            raise ValueError("Nonzero signed time and nonnegative even cutoff required")
        if (not math.isfinite(self.truncation_tolerance) or self.truncation_tolerance <= 0
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("Finite positive RTE tolerance and nonnegative seed required")

    def make_config(self, symbolic_tail, factory):
        """Use the existing config factory, supplied explicitly (mock in tests)."""
        return factory(symbolic_tail, evolution_time=float(self.duration),
                       rte_steps=self.rte_steps, finite_taylor_order=self.cutoff,
                       truncation_tolerance=self.truncation_tolerance, seed=self.seed)


@dataclass(frozen=True)
class DStage:
    context: DContext
    inner_arm: str
    tau: Fraction
    repetitions: int

    def __post_init__(self):
        if not isinstance(self.context, DContext) or self.inner_arm not in INNER:
            raise ValueError("Bound context and inner A2/A3 required")
        object.__setattr__(self, "tau", _rational(self.tau))
        object.__setattr__(self, "repetitions", _positive_integer(self.repetitions))
        if not self.tau:
            raise ValueError("Nonzero D duration required")

    @property
    def duration(self):
        return self.tau * self.repetitions

    def append_operations(self):
        weights = tuple(map(Fraction, symmetric_s2_sequence(INNER[self.inner_arm]())))
        return _expand_append_s2(self.context, weights[::-1], self.tau) * self.repetitions


def fuse_d_stages(stages):
    """Concatenate identical kernels at fixed tau. Never combine durations."""
    result = []
    for item in stages:
        if not isinstance(item, (DStage, TailOccurrence)):
            raise TypeError("Only D stages and opaque R occurrences are supported")
        if result and isinstance(item, DStage) and isinstance(result[-1], DStage):
            previous = result[-1]
            if (previous.context, previous.inner_arm, previous.tau) == (
                    item.context, item.inner_arm, item.tau):
                result[-1] = replace(previous, repetitions=previous.repetitions + item.repetitions)
                continue
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class OuterPlan:
    case: OuterCase
    body: tuple
    scalar_phases: tuple[PhaseOperation, ...]
    independent_occurrences: bool

    @property
    def mathematical_operations(self):
        return self.body[::-1]

    @property
    def cost_metadata(self):
        ds = [x for x in self.body if isinstance(x, DStage)]
        rs = [x for x in self.body if isinstance(x, TailOccurrence)]
        return dict(D_runs=len(ds), R_occurrences=len(rs),
                    D_kernel_repetitions=sum(x.repetitions for x in ds),
                    D_block_exponentials=sum(len(x.append_operations()) for x in ds),
                    negative_R_stages=sum(x.evolution_time < 0 for x in rs),
                    R_absolute_time=sum((abs(x.evolution_time) for x in rs), Fraction()),
                    D_time=sum((x.duration for x in ds), Fraction()),
                    R_time=sum((x.evolution_time for x in rs), Fraction()),
                    price_status="STRUCTURE_ONLY_NO_NATIVE_OR_SHOT_COST")


def make_outer_plan(context, case, *, r_rte, cutoff, truncation_tolerance, seed,
                    independent_occurrences, fuse_boundaries=True):
    if not isinstance(context, DContext) or not isinstance(case, OuterCase):
        raise TypeError("DContext and OuterCase required")
    if independent_occurrences is not True:
        raise ValueError("Ordered product of means requires independent occurrences")
    if type(fuse_boundaries) is not bool:
        raise TypeError("Explicit bool fusion policy required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer metadata seed required")
    delta = case.total_time / case.step_count
    stages, occurrence = [], 0
    case_id = ":".join(str(v) for v in asdict(case).values())
    for macro in range(case.step_count):
        for kind, coefficient in reversed(mathematical_stages(case.method)):
            duration = delta * coefficient
            if kind == "D":
                stages.append(DStage(context, case.inner_arm, duration / case.n_d, case.n_d))
            else:
                spec = RTEStage(duration, r_rte, cutoff, truncation_tolerance, seed + occurrence)
                stages.append(TailOccurrence(context, f"{case_id}:R{occurrence}", duration, spec,
                    "retain_existing_event_phase_and_control_zero_branch",
                    "per_occurrence_raw_equals_corrected_divided_by_Nj"))
                occurrence += 1
    body = fuse_d_stages(stages) if fuse_boundaries else tuple(stages)
    phases = tuple(PhaseOperation(t, case.total_time, context.controlled, context.ancilla_qubit)
                   for t in context.scalar_terms)
    return OuterPlan(case, body, phases, independent_occurrences)


def normalization_product(moments):
    """Actual per-occurrence N_j, including repetitions; no common N**q."""
    values = tuple(float(m.normalization_product) for m in moments)
    if not values or any(not math.isfinite(n) or n <= 0 for n in values):
        raise ValueError("Finite positive normalization required for every occurrence")
    total = math.prod(values)
    if not math.isfinite(total) or not total:
        raise ValueError("INCONCLUSIVE_NORMALIZATION_RANGE")
    return dict(normalization_product=total, attenuation=1 / total,
                occurrence_normalizations=values)


def build_outer_graph(settings, cases, *, independent_occurrences):
    """Compile explicit cases into existing Node/Graph; evaluate() is reused.

    No default refinement/grid. This does not authorize or launch science.
    Settings.deltas/q_values describe base macro widths/counts, not all the
    internal scaled durations introduced by an outer composition.
    """
    settings.validate()
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Explicit unique outer cases required")
    nodes, lookup, rows = [], {}, []

    def add(kind, deps=(), params=()):
        node = Node(kind, tuple(deps), tuple(params))
        if node not in lookup:
            lookup[node] = len(nodes)
            nodes.append(node)
        return lookup[node]

    def exp(label, duration):
        return add("exponential", params=(label, duration))

    def product(ops):
        return add("product_append", ops)

    def phased(ops, time):
        return add("physical_scalar_phase", (product(ops),), (time,))

    def error(a, b):
        return add("operator_error", (a, b))

    correctness = dict(input_validation=add("validate_input"),
                       weak_perturbation_diagnostics=add("outer_diagnostics"))
    for row_id, case in enumerate(cases):
        if (case.macro_steps not in settings.q_values or case.n_d not in settings.n_d_values
                or case.total_time / case.macro_steps not in settings.deltas):
            raise ValueError("Outer case is outside the declared base time/grid conditions")
        plan = make_outer_plan(settings.context, case, r_rte=settings.r_rte,
            cutoff=settings.cutoff, truncation_tolerance=settings.truncation_tolerance,
            seed=settings.seed, independent_occurrences=independent_occurrences)
        exact, ideal, corrected, raw, moments = [], [], [], [], []
        for stage_id, stage in enumerate(plan.body):
            if isinstance(stage, DStage):
                d = product(exp(op.block_id, op.evolution_time) for op in stage.append_operations())
                correctness[f"D:{row_id}:{stage_id}"] = add("outer_d_check", (d,), (stage,))
                exact.append(exp("exact_D", stage.duration))
                ideal.append(d); corrected.append(d); raw.append(d)
            else:
                spec = stage.payload
                moment = add("outer_rte_moments", params=(spec,))
                moments.append(moment)
                tail = exp("exact_R", stage.evolution_time)
                exact.append(tail); ideal.append(tail)
                corrected.append(add("moment_field", (moment,), ("corrected_operator",)))
                raw.append(add("moment_field", (moment,), ("attenuated_event_mean_operator",)))
        outer_unphased = product(exact)
        # Unfused mathematical reference, constructed from the case rather than
        # by reversing the realized body. Exact adjacent D factors may compose;
        # the approximate D words themselves must not be replaced this way.
        reference_word = tuple(("exact_" + kind, case.total_time * coefficient / case.step_count)
            for kind, coefficient in mathematical_stages(case.method)) * case.step_count
        correctness[f"outer:{row_id}"] = add("outer_order_check", (outer_unphased,),
                                             (reference_word,))
        outer = add("physical_scalar_phase", (outer_unphased,), (case.total_time,))
        ideal, corrected, raw = (phased(ops, case.total_time) for ops in (ideal, corrected, raw))
        physical = exp("exact_physical", case.total_time)
        norm = add("outer_normalization", moments)
        rescaled = add("rescale_raw", (raw, norm))
        consistency = add("outer_rescale_check", (rescaled, corrected))
        correctness[f"normalization:{row_id}"] = consistency
        rows.append(dict(case=asdict(case), coefficient_source=UPSTREAM_COMMIT,
            outer_coefficient_source=("astro-ph/0005074v1:TableI:SABA2" if case.method == "SABA2"
                                      else UPSTREAM_COMMIT),
            input_fingerprint=settings.fingerprint, tail_identity=settings.context.tail_id,
            independent_occurrences=True, plan=asdict(plan), cost=plan.cost_metadata,
            outputs=dict(outer_operator_error=error(outer, physical),
                         ideal_tail_physical_error=error(ideal, physical),
                         corrected_physical_error=error(corrected, physical),
                         raw_physical_error=error(raw, physical),
                         D_approximation_error=error(ideal, outer),
                         finite_RTE_error=error(corrected, ideal),
                         normalization_consistency=consistency, normalization=norm,
                         signals=add("outer_signals", (physical, outer, ideal, corrected, raw)))))
    return Graph(settings, tuple(nodes), tuple(rows), correctness)


def evaluate_outer(graph, backend_factory, *, execution=None, scope=None, mock_only=True):
    """Reuse all-gates-first evaluation; no launcher or permission proof added."""
    result = evaluate(graph, backend_factory, execution=execution, scope=scope, mock_only=mock_only)
    return dict(result, schema="b2d_outer_splitting_v1")
