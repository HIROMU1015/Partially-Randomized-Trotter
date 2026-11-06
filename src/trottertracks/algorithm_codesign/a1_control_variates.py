"""A1 classical adapters; no sampler, quantum builder, science input or runner.

Only caller-supplied metadata/outcomes are consumed. Exact rational arithmetic
keeps centering and range bookkeeping separate from the floating pilot fit.
Disjoint stream/trial checks detect data reuse; they do not prove RNG independence.
See docs/research/o2_a1_preparation_20261106/A1_PREFLIGHT.md for unresolved
physical-input and finite-law binding conditions. No performance is certified.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from math import ceil, isfinite, prod

from .parent_allocation import integer, rational
from .return_aggregation import Interval


@dataclass(frozen=True)
class CategoricalLaw:
    labels: tuple
    masses: tuple[F, ...]
    source: str

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "masses", tuple(map(rational, self.masses)))
        if (not self.source or not self.labels or len(set(self.labels)) != len(self.labels)
                or len(self.labels) != len(self.masses) or min(self.masses) < 0
                or sum(self.masses) != 1):
            raise ValueError("complete normalized categorical law and source required")

    def mass(self, label):
        try:
            return self.masses[self.labels.index(label)]
        except ValueError as exc:
            raise ValueError("label outside declared support") from exc


def law_from_choice_cdf(labels, cdf, *, source):
    """Masses of searchsorted(side='right') on a uniform 53-bit double grid.

    Input is the *actual normalized binary64 CDF*, not ideal probabilities.
    NumPy 1.26.4 Generator.choice(replace=True,p=...) uses p.cumsum();
    cdf /= cdf[-1]. PCG64 next_double uses (next_uint64 >> 11) / 2**53.
    This inspects no PRNG and changes no sampler. IID uniform words remain a
    statistical assumption. A future caller must bind CDF/version to its source.
    """
    cdf = tuple(cdf)
    if (not cdf or any(type(x) is not float or not isfinite(x) for x in cdf)
            or any(a > b for a, b in zip((0.0,) + cdf, cdf))
            or cdf[-1] != 1.0):
        raise ValueError("finite monotone normalized binary64 CDF required")
    ends = (0,) + tuple(ceil(F.from_float(x) * 2**53) for x in cdf)
    return CategoricalLaw(tuple(labels), tuple(F(b-a, 2**53)
                          for a, b in zip(ends, ends[1:])), source)


@dataclass(frozen=True)
class EventMetadata:
    order: int
    selected_component_ids: tuple[str, ...]  # product labels, then rotation
    phase: complex

    def __post_init__(self):
        integer(self.order, "even Taylor order")
        object.__setattr__(self, "selected_component_ids", tuple(self.selected_component_ids))
        if (self.order % 2 or len(self.selected_component_ids) != self.order + 1
                or self.phase != (-1)**(self.order // 2)):
            raise ValueError("RTE order/application order/phase mismatch")

    @classmethod
    def from_rte_event(cls, event):
        selected = tuple(event.selected_component_ids)
        if (selected != tuple(event.product_component_ids) + (event.rotation_component_id,)):
            raise ValueError("selected labels must be products then rotation")
        return cls(event.taylor_order, selected, event.phase)


@dataclass(frozen=True)
class OccurrenceLaw:
    orders: CategoricalLaw
    labels: CategoricalLaw

    def __post_init__(self):
        for n in self.orders.labels:
            integer(n, "Taylor order")
            if n % 2:
                raise ValueError("even orders required")

    def validate(self, event):
        if self.orders.mass(event.order) <= 0:
            raise ValueError("zero-probability order")
        if any(self.labels.mass(x) <= 0 for x in event.selected_component_ids):
            raise ValueError("zero-probability label")


@dataclass(frozen=True)
class LocalControl:
    occurrence: int
    order: int
    positions: tuple[tuple[int, str], ...]  # empty = order-only comparison

    def __post_init__(self):
        integer(self.occurrence, "occurrence")
        integer(self.order, "order")
        object.__setattr__(self, "positions", tuple(tuple(x) for x in self.positions))
        if (self.order % 2 or len(self.positions) > 2
                or len({p for p, _ in self.positions}) != len(self.positions)):
            raise ValueError("local one/two-position indicator or order-only required")
        for p, _ in self.positions:
            integer(p, "application position")
            if p > self.order:
                raise ValueError("application position outside event")


@dataclass(frozen=True)
class ControlBank:
    law_id: str
    occurrences: tuple[OccurrenceLaw, ...]
    controls: tuple[LocalControl, ...]

    def __post_init__(self):
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        object.__setattr__(self, "controls", tuple(self.controls))
        if not self.law_id or not self.occurrences or len(set(self.controls)) != len(self.controls):
            raise ValueError("law identity, occurrences and unique controls required")
        for c in self.controls:
            if c.occurrence >= len(self.occurrences):
                raise ValueError("missing occurrence law")
            law = self.occurrences[c.occurrence]
            if law.orders.mass(c.order) <= 0:
                raise ValueError("unsupported control order")
            for _, label in c.positions:
                if law.labels.mass(label) <= 0:
                    raise ValueError("unsupported control label")

    def joint_mean(self, indices):
        """Known factorized indicator moment; contradictory conditions give zero."""
        conditions = {}
        for i in indices:
            c = self.controls[i]
            n, positions = conditions.setdefault(c.occurrence, (c.order, {}))
            if n != c.order:
                return F(0)
            for pos, label in c.positions:
                if pos in positions and positions[pos] != label:
                    return F(0)
                positions[pos] = label
        return prod((self.occurrences[o].orders.mass(n)
                     * prod((self.occurrences[o].labels.mass(x) for x in ps.values()), start=F(1))
                     for o, (n, ps) in conditions.items()), start=F(1))

    @property
    def means(self):
        return tuple(self.joint_mean((i,)) for i in range(len(self.controls)))

    def values(self, events):
        if len(events) != len(self.occurrences):
            raise ValueError("complete ordered interrogation path required")
        for law, e in zip(self.occurrences, events):
            law.validate(e)
        return tuple(int(events[c.occurrence].order == c.order and all(
            events[c.occurrence].selected_component_ids[p] == label for p, label in c.positions))
            for c in self.controls)

    def pattern_mass(self, pattern):
        """One supplied feature stratum, by inclusion-exclusion; no path table."""
        pattern = tuple(pattern)
        if len(pattern) != len(self.controls) or any(type(x) is not int or x not in (0, 1) for x in pattern):
            raise ValueError("one binary indicator per control required")
        ones = tuple(i for i, bit in enumerate(pattern) if bit)
        zeros = tuple(i for i, bit in enumerate(pattern) if not bit)
        return sum(((-1)**k * self.joint_mean(ones + subset)
                    for k in range(len(zeros) + 1) for subset in combinations(zeros, k)), F(0))


@dataclass(frozen=True)
class Scope:
    target_id: str  # H/state/finite-target/normalization/bias references externally
    law_id: str
    circuit_id: str  # full no-prep wrapper, scalar/control/readout source
    round_id: str
    axis: str
    weight_bound: F

    def __post_init__(self):
        object.__setattr__(self, "weight_bound", rational(self.weight_bound))
        if (not all((self.target_id, self.law_id, self.circuit_id, self.round_id))
                or self.axis not in ("X", "Y") or self.weight_bound < 0):
            raise ValueError("same-target full interrogation scope required")


@dataclass(frozen=True)
class Measurement:
    stream_id: str
    trial_id: str
    events: tuple[EventMetadata, ...]
    bit: int
    weight: F
    trajectory_id: str
    wrapper_id: str

    def __post_init__(self):
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "weight", rational(self.weight))
        if (not all((self.stream_id, self.trial_id, self.trajectory_id, self.wrapper_id))
                or type(self.bit) is not int or self.bit not in (0, 1)):
            raise ValueError("stream/trial/path/wrapper identity and actual bit required")

    @property
    def key(self):
        return self.stream_id, self.trial_id

    @property
    def raw(self):
        # Phase/sign is already in the controlled circuit. Never apply it twice.
        return self.weight * (1 - 2*self.bit)


def bind_wrapper_measurement(scope, wrapper, *, trajectory_id, events,
                             stream_id, trial_id, bit, weight):
    """Bind a bit and retained path to the existing wrapper's public metadata.

    events must come from the named retained trajectory, in occurrence order;
    this adapter cannot reconstruct that association from outcome bits alone.
    Target/round/source binding remains the caller's preflight responsibility.
    No Qiskit import or circuit operation is performed.
    """
    axis = {"X": ("cosine", "real"), "Y": ("sine", "imaginary")}[scope.axis]
    if (wrapper.wrapped_trajectory_fingerprint != trajectory_id
            or (wrapper.axis, wrapper.signal_component) != axis
            or wrapper.bit_value_mapping != ((0, 1), (1, -1))
            or wrapper.circuit_scope != FULL_SCOPE or not wrapper.include_measurement
            or wrapper.state_preparation_included or wrapper.additional_control_applied
            or not wrapper.wrapped_evolution_already_controlled):
        raise ValueError("path/axis/sign/full-wrapper binding mismatch")
    return Measurement(stream_id, trial_id, tuple(events), bit, weight,
                       trajectory_id, wrapper.wrapper_fingerprint)


@dataclass(frozen=True)
class Batch:
    scope: Scope
    stage: str
    records: tuple[Measurement, ...]

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))
        if (self.stage not in ("train", "cal", "prod") or not self.records
                or len({r.key for r in self.records}) != len(self.records)
                or any(abs(r.weight) > self.scope.weight_bound for r in self.records)):
            raise ValueError("nonempty stage, unique trials and bounded weights required")

    @property
    def streams(self):
        return frozenset(r.stream_id for r in self.records)


def require_disjoint(*batches):
    if len({b.scope for b in batches}) != 1:
        raise ValueError("target/law/round/axis/circuit mismatch")
    for a, b in combinations(batches, 2):
        if a.streams & b.streams:
            raise ValueError("stages/strata must use disjoint measurement streams")


def raw_mean(batch, bank):
    if batch.scope.law_id != bank.law_id:
        raise ValueError("law mismatch")
    for r in batch.records:
        bank.values(r.events)
    return sum((r.raw for r in batch.records), F(0)) / len(batch.records)


@dataclass(frozen=True)
class FrozenCV:
    bank: ControlBank
    training: Batch
    coefficients: tuple[F, ...]

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(map(rational, self.coefficients)))
        if self.training.stage != "train" or len(self.coefficients) != len(self.bank.controls):
            raise ValueError("training-only coefficients of the fixed control bank required")
        raw_mean(self.training, self.bank)

    @property
    def output_range(self):
        # Conservative box enclosure; dependencies between indicators may tighten it.
        lo = -self.training.scope.weight_bound
        hi = -lo
        for coefficient, nu in zip(self.coefficients, self.bank.means):
            a, b = coefficient*nu, -coefficient*(1-nu)
            lo += min(a, b)
            hi += max(a, b)
        return Interval(lo, hi)

    def centering_bias_bound(self, mean_error_bounds):
        errors = tuple(map(rational, mean_error_bounds))
        if len(errors) != len(self.coefficients) or any(e < 0 for e in errors):
            raise ValueError("one nonnegative centering error per control required")
        return sum((abs(c)*e for c, e in zip(self.coefficients, errors)), F(0))

    def apply(self, batch):
        if batch.stage not in ("cal", "prod"):
            raise ValueError("frozen coefficients only on fresh calibration/production")
        require_disjoint(self.training, batch)
        raw_mean(batch, self.bank)
        return tuple(r.raw - sum((c*(g-nu) for c, g, nu in zip(
            self.coefficients, self.bank.values(r.events), self.bank.means)), F(0))
            for r in batch.records)


def fit_and_freeze(bank, training):
    """Pilot-only OLS with intercept via existing NumPy least squares; no oracle.

    No feature search, regularizer search or performance test. Singular designs
    use NumPy's minimum-norm solution. The resulting binary64 slopes are frozen
    as exact rationals; all later range checks use those actual coefficients.
    """
    import numpy as np
    if training.stage != "train" or len(training.records) < 2:
        raise ValueError("at least two training outcomes required")
    raw_mean(training, bank)
    design = np.asarray([[1.] + [float(g-nu) for g, nu in zip(bank.values(r.events), bank.means)]
                         for r in training.records])
    values = np.asarray([float(r.raw) for r in training.records])
    if not np.isfinite(design).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite pilot regression input")
    solution = np.linalg.lstsq(design, values, rcond=None)[0]
    if not np.isfinite(solution).all():
        raise ValueError("nonfinite pilot coefficients")
    return FrozenCV(bank, training, tuple(F.from_float(float(x)) for x in solution[1:]))


@dataclass(frozen=True)
class BoundedSamples:
    scope: Scope
    stage: str
    values: tuple[F, ...]
    bounds: Interval
    centering_bias: F
    # No binary tails, variance, confidence radius or N is computed here.


def evaluation_inputs(cv, calibration, production, *, mean_error_bounds):
    if calibration.stage != "cal" or production.stage != "prod" or len(calibration.records) < 2:
        raise ValueError("independent variance calibration and production required")
    require_disjoint(cv.training, calibration, production)
    bias = cv.centering_bias_bound(mean_error_bounds)
    return tuple(BoundedSamples(b.scope, b.stage, cv.apply(b), cv.output_range, bias)
                 for b in (calibration, production))


def stratified_mean(bank, cells):
    """Fixed conditional strata supplied externally; not an IID poststratifier.

    cells = ((feature_pattern, fresh conditional production Batch), ...).
    Caller fixes positive allocations before these outcomes; this function
    neither samples nor allocates. Every positive stratum must be covered.
    """
    cells = tuple((tuple(key), batch) for key, batch in cells)
    if not cells or len({key for key, _ in cells}) != len(cells):
        raise ValueError("distinct exhaustive strata required")
    masses = tuple(bank.pattern_mass(key) for key, _ in cells)
    if min(masses) <= 0 or sum(masses) != 1:
        raise ValueError("all positive-probability strata must be covered")
    require_disjoint(*(b for _, b in cells))
    result = F(0)
    for (key, batch), mass in zip(cells, masses):
        if batch.stage != "prod" or any(bank.values(r.events) != key for r in batch.records):
            raise ValueError("fresh conditional production outcomes for the declared stratum required")
        result += mass * raw_mean(batch, bank)
    return result


QUANTUM_METRICS = ("T", "CX", "RZ", "1Q", "depth")
FULL_SCOPE = "single_hadamard_interrogation_without_state_preparation"


@dataclass(frozen=True)
class TrialCost:
    scope: Scope
    stream_id: str
    trial_id: str
    stage: str
    full_scope: str
    quantum: tuple[F | None, ...]  # QUANTUM_METRICS order; unavailable stays None
    classical_setup_seconds: F
    classical_online_seconds: F
    memory_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "quantum", tuple(self.quantum))


def account_costs(records):
    """One method's actual full-interrogation records; no free/duplicate pilot.

    Call separately for raw/stratified/CV deployment costs. Shared experimental
    acquisition may be reused across methods, but each deployment is charged
    for the data it needs. Classical-only setup/rejection costs belong in a
    separate external record; never fabricate a quantum trial for them.
    """
    records = tuple(records)
    seen, by_stage = set(), {s: [F(0)]*len(QUANTUM_METRICS) for s in ("train", "cal", "prod")}
    setup = online = F(0)
    memory = 0
    targets = {(r.scope.target_id, r.scope.law_id, r.scope.circuit_id, r.scope.round_id) for r in records}
    if len(targets) != 1:
        raise ValueError("one same-target round required")
    for r in records:
        key = (r.scope.axis, r.stream_id, r.trial_id)
        if (key in seen or not r.stream_id or not r.trial_id or r.stage not in by_stage
                or r.full_scope != FULL_SCOPE or len(r.quantum) != len(QUANTUM_METRICS)):
            raise ValueError("unique full no-prep trial costs required")
        seen.add(key)
        for i, value in enumerate(r.quantum):
            value = None if value is None else rational(value)
            if value is not None and value < 0:
                raise ValueError("nonnegative metric required")
            old = by_stage[r.stage][i]
            by_stage[r.stage][i] = None if old is None or value is None else old + value
        cs, co = rational(r.classical_setup_seconds), rational(r.classical_online_seconds)
        integer(r.memory_bytes, "memory bytes")
        if min(cs, co) < 0:
            raise ValueError("nonnegative classical costs required")
        setup += cs
        online += co
        memory = max(memory, r.memory_bytes)
    totals = tuple(None if any(v[i] is None for v in by_stage.values())
                   else sum(v[i] for v in by_stage.values()) for i in range(len(QUANTUM_METRICS)))
    return dict(quantum_by_stage=by_stage, quantum_total=totals,
                classical_setup_seconds=setup, classical_online_seconds=online,
                peak_memory_bytes=memory)
