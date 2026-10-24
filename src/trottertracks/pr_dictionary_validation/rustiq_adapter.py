"""Opt-in Qiskit 1.3 central-rotation adapter; new B2 science remains unrun.

Planning/binding uses only typed PR records and the standard library. Quantum
imports and synthesis occur only in explicit build calls after separate approval.
The frozen router, compiler, basis recipes and phase helpers are not modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time

from . import circuits as legacy
from .fixture import CompilerProfile, digest, fraction
from .prototype import atom_record

ADAPTER = 'C4_Q13_central_v1'
QISKIT_SOURCE = '896d1c8b72888a52a46081a6d1f4de8c56db8aea'
ROUTES = ('B0', 'B1', 'R1', 'B2')
FUSION_POLICY = 'C4_B2_adjacent_controlled_fsum_v1'


@dataclass(frozen=True)
class Options:
    optimize_count: bool = True
    preserve_order: bool = True
    upto_clifford: bool = False
    upto_phase: bool = False
    resynth_clifford_method: int = 0

    def validate(self):
        # Equality alone would accept 1/0 in place of bool and vice versa.
        if any(type(getattr(self, k)) is not type(v) or getattr(self, k) != v
               for k, v in asdict(Options()).items()):
            raise ValueError('Only the audited exact-phase Q13 options are allowed')


@dataclass(frozen=True)
class Occurrence:
    ordinal: int
    event_index: int
    application_index: int
    atom_id: str
    role: str
    kind: str
    word: tuple[str, ...]  # word[p] acts on physical qubit p, not a dense Qiskit label
    support: tuple[int, ...]
    signed_angle: float | None  # PR exp(-i a P), already signed exactly once
    coefficient_sign: int
    basis_id: str | None
    basis_hash: str | None  # saved frame identity, or local-basis recipe hash
    frame_id: str | None
    run_id: int | None = None


@dataclass(frozen=True)
class PhaseRecord:
    kind: str
    event_index: int
    occurrence: int | None
    angle: float


@dataclass(frozen=True)
class Region:
    region_id: str
    occurrences: tuple[Occurrence, ...]
    eligible: bool


@dataclass(frozen=True)
class Run:
    run_id: int
    basis_id: str
    basis_hash: str
    singleton: bool  # decided from the ORIGINAL whole run, including products
    regions: tuple[Region, ...]


@dataclass(frozen=True)
class SequencePlan:
    occurrences: tuple[Occurrence, ...]
    runs: tuple[Run, ...]
    phases: tuple[PhaseRecord, ...]
    phase: float
    system_qubits: tuple[int, ...]
    ancilla: int
    provenance: dict
    fingerprint: str


def route_identity(route, *, input_identity, source_identity, profile=None, options=Options()):
    if route not in ROUTES:
        raise ValueError('Unknown route')
    if not all(isinstance(x, str) and x for x in (input_identity, source_identity)):
        raise ValueError('Bound SOURCE/input identities required')
    options.validate()
    profile = CompilerProfile() if profile is None else profile
    profile.validate()
    identity = dict(adapter=ADAPTER, route=route, input=input_identity, source=source_identity,
                qiskit='1.3.0', qiskit_source=QISKIT_SOURCE, bundled_rustiq_core='0.0.10',
                options=asdict(options), compiler=asdict(profile),
                basis_policy='common_run_support_full_fused',
                mapping='word[p]=qubit_p;system=0..m-1;ancilla=m;no_second_control')
    if route == 'B2':
        identity['fusion_policy'] = FUSION_POLICY
    return identity


def _validate_atom(atom, modes):
    if len(atom.word) != modes or any(ch not in ('I', 'X', 'Y', 'Z') for ch in atom.word):
        raise ValueError('Invalid typed Pauli word')
    support = tuple(i for i, ch in enumerate(atom.word) if ch != 'I')
    if tuple(atom.support) != support or not atom.coefficient:
        raise ValueError('Invalid support or zero coefficient')
    if atom.kind == 'IDENTITY':
        if support or atom.basis_key is not None or atom.frame_id is not None:
            raise ValueError('Identity must retain its phase-only typed path')
        return None
    if not support:
        raise ValueError('Nonidentity atom has empty support')
    if atom.kind == 'FRAME_Z':
        if (any(ch not in ('I', 'Z') for ch in atom.word) or not atom.frame_id
                or not isinstance(atom.basis_key, str) or not atom.basis_key.startswith('frame:')):
            raise ValueError('Invalid FRAME_Z basis')
        value = atom.basis_key[6:]
        if len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value):
            raise ValueError('Saved frame hash required')
        return value
    if atom.kind == 'PHYSICAL_PAULI':
        expected = 'local:' + ''.join(ch if ch in ('X', 'Y') else 'Z' for ch in atom.word)
        if atom.basis_key != expected or atom.frame_id is not None:
            raise ValueError('Physical local-basis recipe mismatch')
        return digest(dict(local_basis=expected))
    raise ValueError('Unsupported typed atom kind')


def plan_sequence(events, atoms, modes, *, provenance):
    """Preserve original run sharing; only then partition products/rotations.

    No event/law generation, angle reconstruction, matrix calculation or circuit
    operation. Duplicate occurrences remain distinct. Identity closes a run.
    """
    if type(modes) is not int or modes < 1:
        raise ValueError('Positive system register size required')
    expected = route_identity(provenance.get('route'), input_identity=provenance.get('input'),
                              source_identity=provenance.get('source'))
    if provenance != expected:
        raise ValueError('Stale route/options/compiler provenance')
    atoms = tuple(atoms)
    by_id = {a.atom_id: a for a in atoms}
    if len(by_id) != len(atoms):
        raise ValueError('Duplicate atom definitions')
    hashes = {a.atom_id: _validate_atom(a, modes) for a in atoms}
    basis_types = {}
    for atom in atoms:
        if atom.kind == 'IDENTITY':
            continue
        value = (atom.kind, atom.frame_id)
        if basis_types.setdefault(atom.basis_key, value) != value:
            raise ValueError('Basis identity aliases different frames/kinds')
    occurrences = []
    phases = []
    phase = 0.0
    for ei, event in enumerate(events):
        apps = tuple(event.application_sequence)
        if tuple(x.component_id for x in apps) != tuple(event.selected_component_ids):
            raise ValueError('Event order metadata mismatch')
        if not apps or tuple(a.role for a in apps) != ('product',) * (len(apps)-1) + ('rotation',):
            raise ValueError('Products followed by exactly one terminal rotation required')
        if event.phase not in (1, -1):
            raise ValueError('Expected the unchanged paired-Taylor phase')
        angle = event.unsigned_rotation_angle  # despite its legacy name, SIGNED
        if type(angle) is not float or not math.isfinite(angle):
            raise ValueError('Finite saved binary64 signed angle required')
        theta = math.atan2(event.phase.imag, event.phase.real)
        phases.append(PhaseRecord('taylor', ei, None, theta))
        phase += theta
        for ai, app in enumerate(apps):
            atom = by_id[app.component_id]
            sign = 1 if atom.coefficient > 0 else -1
            if (type(app.coefficient_sign) is not int or app.coefficient_sign != sign
                    or type(app.is_identity) is not bool or app.is_identity != (atom.kind == 'IDENTITY')
                    or app.basis_id != atom.basis_key):
                raise ValueError('Stale event sign/identity/basis metadata')
            occurrence = Occurrence(len(occurrences), ei, ai, atom.atom_id, app.role,
                atom.kind, tuple(atom.word), tuple(atom.support),
                angle if app.role == 'rotation' else None, sign, atom.basis_key,
                hashes[atom.atom_id], atom.frame_id)
            occurrences.append(occurrence)
            if app.role == 'product' and sign < 0:
                phases.append(PhaseRecord('product_sign', ei, occurrence.ordinal, math.pi))
                phase += math.pi
    # Match the legacy floating addition order: all Taylor/product phases first,
    # then identity rotations encountered in the flattened sequence.
    runs = []
    pos = 0
    while pos < len(occurrences):
        first = occurrences[pos]
        if first.kind == 'IDENTITY':
            if first.signed_angle is not None:
                value = -first.signed_angle
                phases.append(PhaseRecord('identity_rotation', first.event_index, pos, value))
                phase += value
            pos += 1
            continue
        stop = pos + 1
        while (stop < len(occurrences) and occurrences[stop].kind != 'IDENTITY'
               and occurrences[stop].basis_id == first.basis_id):
            stop += 1
        rid = len(runs)
        occurrences[pos:stop] = [replace(o, run_id=rid) for o in occurrences[pos:stop]]
        regions = []
        start = pos
        while start < stop:
            end = start + 1
            eligible = occurrences[start].role == 'rotation'
            if eligible:
                while end < stop and occurrences[end].role == 'rotation':
                    end += 1
            regions.append(Region(f'run{rid}:region{len(regions)}',
                                  tuple(occurrences[start:end]), eligible))
            start = end
        runs.append(Run(rid, first.basis_id, first.basis_hash, stop-pos == 1, tuple(regions)))
        pos = stop
    data = dict(occurrences=[asdict(o) for o in occurrences], runs=[asdict(r) for r in runs],
                phases=[asdict(p) for p in phases], phase=phase, system_qubits=tuple(range(modes)),
                ancilla=modes, provenance=provenance)
    return SequencePlan(tuple(occurrences), tuple(runs), tuple(phases), phase,
                        tuple(range(modes)), modes, dict(provenance), digest(data))


def controlled_terms(word, signed_angle, *, ancilla):
    """Sparse Q13 pairs for exp(-i a P); never round angles or control again."""
    if (type(ancilla) is not int or ancilla != len(word) or ancilla < 1
            or any(ch not in ('I', 'X', 'Y', 'Z') for ch in word)
            or type(signed_angle) is not float or not math.isfinite(signed_angle)):
        raise ValueError('Invalid signed angle/word/ancilla-last mapping')
    indices = tuple(i for i, ch in enumerate(word) if ch != 'I')
    if not indices:
        raise ValueError('All-I is excluded from Q13; use the PR phase path')
    sparse = ''.join(word[i] for i in indices)
    return ((sparse, indices, signed_angle), (sparse+'Z', indices+(ancilla,), -signed_angle))


def region_network(region, *, ancilla):
    if not region.eligible or not region.occurrences:
        raise ValueError('Products/empty regions cannot enter Rustiq')
    boundaries = {(o.run_id, o.basis_id, o.basis_hash, o.frame_id, o.kind)
                  for o in region.occurrences}
    if len(boundaries) != 1 or region.occurrences[0].run_id is None:
        raise ValueError('A region cannot cross an existing basis/run boundary')
    terms = []
    for o in region.occurrences:
        if o.role != 'rotation' or o.kind not in ('FRAME_Z', 'PHYSICAL_PAULI'):
            raise ValueError('Only typed nonidentity central rotations allowed')
        central = tuple('Z' if p in o.support else 'I' for p in range(ancilla))
        terms.extend(controlled_terms(central, o.signed_angle, ancilla=ancilla))
    return tuple(terms)


@dataclass(frozen=True)
class FusionGroup:
    support: tuple[int, ...]
    signed_angle: float
    ledger: dict


def _signed_sum(members):
    angles = tuple(o.signed_angle for o in members)
    if not angles or any(type(a) is not float or not math.isfinite(a) for a in angles):
        raise ValueError('Finite binary64 rotation angles required')
    try:
        value = angles[0] if len(angles) == 1 else math.fsum(angles)
    except OverflowError as error:
        raise ValueError('B2 angle sum overflow') from error
    if not math.isfinite(value):
        raise ValueError('B2 angle sum is nonfinite')
    return value


def fusion_groups(plan, region, *, case_id, macro_index):
    """Pure typed fold; never re-plan runs, move events, or expand controls.

    Preparation tests use artificial records only. Calling on saved F1 angles
    belongs to the separately approved scientific execution.
    """
    data = asdict(plan)
    fingerprint = data.pop('fingerprint')
    if (digest(data) != fingerprint or plan.provenance != route_identity('B2',
            input_identity=plan.provenance.get('input'), source_identity=plan.provenance.get('source'))):
        raise ValueError('Stale original plan')
    if (not isinstance(case_id, str) or not case_id
            or type(macro_index) is not int or macro_index < 0
            or plan.system_qubits != tuple(range(plan.ancilla))):
        raise ValueError('Bound case/macro and original ancilla-last mapping required')
    matching = [(run, r) for run in plan.runs for r in run.regions if r == region]
    if len(matching) != 1 or not region.eligible or not region.occurrences:
        raise ValueError('One original eligible rotation region required')
    run = matching[0][0]
    def key(o):
        if (o.role != 'rotation' or o.kind not in ('FRAME_Z', 'PHYSICAL_PAULI')
                or o.run_id != run.run_id or o.basis_id != run.basis_id
                or o.basis_hash != run.basis_hash or not o.support
                or o.word != tuple(plan.occurrences[o.ordinal].word)
                or o != plan.occurrences[o.ordinal]):
            raise ValueError('Rotation/original occurrence or basis/run drift')
        return (o.kind, o.word, o.support, o.basis_id, o.basis_hash, o.frame_id, o.run_id)
    groups = []
    members = []
    previous = None
    def flush():
        first = members[0]
        angle = _signed_sum(members)
        ledger = dict(policy=FUSION_POLICY, case=case_id, macro=macro_index,
            run=run.run_id, region=region.region_id, singleton=run.singleton,
            basis_id=first.basis_id, basis_hash=first.basis_hash, frame=first.frame_id,
            kind=first.kind, word=first.word, support=first.support,
            system=plan.system_qubits, ancilla=plan.ancilla, generator='controlled_central_Z',
            original_plan=fingerprint, members=[dict(ordinal=o.ordinal, atom_id=o.atom_id,
                event=o.event_index, application=o.application_index,
                coefficient_sign=o.coefficient_sign, angle_hex=o.signed_angle.hex()) for o in members],
            angle_hex=angle.hex())
        groups.append(FusionGroup(first.support, angle, ledger))
    for occurrence in region.occurrences:
        current = key(occurrence)
        if members and current != previous:
            flush()
            members = []
        members.append(occurrence)
        previous = current
    flush()
    return tuple(groups)


def _q13(num_qubits, network, options):
    from importlib.metadata import version
    if version('qiskit') != '1.3.0':
        raise ValueError('The audited bundled Qiskit 1.3.0 is required')
    from qiskit.synthesis.evolution import synth_pauli_network_rustiq
    options.validate()
    return synth_pauli_network_rustiq(num_qubits,
        [(word, list(indices), angle) for word, indices, angle in network], **asdict(options))


def _new_circuit(qubits):
    from qiskit import QuantumCircuit
    return QuantumCircuit(qubits)


def emit_sequence(plan, atoms, provider, *, options=Options(), trace=None,
                  case_id=None, macro_index=None):
    """Explicit SCIENTIFIC build entry. Do not call during preparation."""
    options.validate()
    data = asdict(plan)
    fingerprint = data.pop('fingerprint')
    if digest(data) != fingerprint or plan.provenance != route_identity(
            plan.provenance.get('route'), input_identity=plan.provenance.get('input'),
            source_identity=plan.provenance.get('source'), options=options):
        raise ValueError('Stale plan or route/options provenance')
    if provider.modes != plan.ancilla:
        raise ValueError('Provider/register mismatch')
    if plan.provenance['route'] == 'B2' and (
            not isinstance(case_id, str) or not case_id
            or type(macro_index) is not int or macro_index < 0):
        raise ValueError('B2 requires bound case/macro scope before emission')
    by_id = {a.atom_id: a for a in atoms}
    qc = _new_circuit(plan.ancilla+1)
    for run in plan.runs:
        first = run.regions[0].occurrences[0]
        b = provider.basis(by_id[first.atom_id], singleton=run.singleton)
        qc.compose(b.inverse(), qubits=plan.system_qubits, inplace=True)
        for region in run.regions:
            for o in region.occurrences:
                if tuple(provider.reference_support(by_id[o.atom_id], run.singleton)) != o.support:
                    raise ValueError('Unreviewed provider changes the central support')
            if plan.provenance['route'] == 'R1' and region.eligible:
                network = region_network(region, ancilla=plan.ancilla)
                tick = time.perf_counter()
                cpu = time.process_time()
                central = _q13(plan.ancilla+1, network, options)
                seconds = time.perf_counter()-tick
                cpu_seconds = time.process_time()-cpu
                # Q13's exact-phase path includes its ACTUAL inverse tail.
                # The network already includes ancilla. Compose directly.
                qc.compose(central, inplace=True)
                if trace is not None:
                    trace.append(dict(kind='synthesis', plan=plan.fingerprint,
                                      region=region.region_id, seconds=seconds,
                                      cpu_seconds=cpu_seconds, terms=network))
            elif plan.provenance['route'] == 'B2' and region.eligible:
                tick, cpu = time.perf_counter(), time.process_time()
                groups = fusion_groups(plan, region, case_id=case_id, macro_index=macro_index)
                record = dict(kind='fusion', plan=plan.fingerprint,
                    region=region.region_id, seconds=time.perf_counter()-tick,
                    cpu_seconds=time.process_time()-cpu, ledger=[g.ledger for g in groups])
                tick, cpu = time.perf_counter(), time.process_time()
                for group in groups:
                    legacy.append_central(qc, group.support, plan.ancilla, group.signed_angle)
                if trace is not None:
                    record.update(emitter_seconds=time.perf_counter()-tick,
                                  emitter_cpu_seconds=time.process_time()-cpu)
                    trace.append(record)
            else:
                for o in region.occurrences:
                    legacy.append_central(qc, o.support, plan.ancilla, o.signed_angle)
        qc.compose(b, qubits=plan.system_qubits, inplace=True)
    if plan.phase:
        qc.p(plan.phase, plan.ancilla)
    return qc


def build_evolution(spec, context, macros, atoms, provider, *, input_identity,
                    source_identity, route='B0', options=Options(), trace=None, case_id=None):
    """Opt-in builder, preserving legacy D fusion, scalar and result/wrapper API.

    B0 delegates unchanged to the strong optimized baseline. B1 uses the same
    typed partition as R1 with the original central emitter. R1 alone calls Q13.
    B2 folds only original adjacent controlled rotations without re-planning.
    This orchestration is narrow duplication because the frozen builder has no
    emitter hook; its basis/central/D/result/compiler implementations are reused.
    """
    provenance = route_identity(route, input_identity=input_identity,
                                source_identity=source_identity, options=options)
    if route == 'B2' and (not isinstance(case_id, str) or not case_id):
        raise ValueError('B2 requires bound case scope')
    if route == 'B0':
        return legacy.build_evolution(spec, context, macros, atoms, provider,
            input_identity=input_identity, source_identity=source_identity, optimized=True)
    spec.validate()
    context.validate()
    if provider.modes != spec.modes:
        raise ValueError('Provider/spec register mismatch')
    macros = tuple(tuple(events) for events in macros)
    atoms = tuple(atoms)
    if len(macros) != context.q or any(len(x) != context.r for x in macros):
        raise ValueError('q/r event shape mismatch')
    tick = time.perf_counter()
    cpu = time.process_time()
    plans = tuple(plan_sequence(events, atoms, spec.modes, provenance=provenance) for events in macros)
    scalar = -float(fraction(spec.scalar))*float(fraction(context.total_time))
    if trace is not None:
        trace.append(dict(kind='adapter', seconds=time.perf_counter()-tick,
            cpu_seconds=time.process_time()-cpu,
            plans=[asdict(p) for p in plans],
            scalar_phase=dict(kind='scalar', angle=scalar, ancilla=spec.modes)))
    qc = _new_circuit(spec.modes+1)
    emit_trace = trace if trace is not None or route != 'B2' else []
    trace_start = len(emit_trace) if route == 'B2' else 0
    delta = float(context.delta)
    for j, plan in enumerate(plans):
        if j == 0:
            legacy.append_deterministic(qc, spec.deterministic, delta/2, spec.modes)
        scope = dict(case_id=case_id, macro_index=j) if route == 'B2' else {}
        qc.compose(emit_sequence(plan, atoms, provider, options=options, trace=emit_trace,
                                 **scope), inplace=True)
        legacy.append_deterministic(qc, spec.deterministic,
            delta if j < context.q-1 else delta/2, spec.modes)
    qc.p(scalar, spec.modes)
    records = tuple(tuple(e.to_dict() for e in macro) for macro in macros)
    fusion = dict(case=case_id, fusion_ledger=[t['ledger'] for t in emit_trace[trace_start:]
                 if t['kind'] == 'fusion']) if route == 'B2' else {}
    fp = digest(dict(provenance=provenance, context=vars(context), macros=records,
                     atoms=[atom_record(a) for a in atoms], plans=[p.fingerprint for p in plans],
                     scalar_phase=scalar, **fusion))
    return legacy.MixedEvolutionResult(qc, spec.modes, spec.modes, context, records,
        input_identity, source_identity, fp, ADAPTER+'_'+route)


def compile_identity(result, *, route, options=Options()):
    """Pass this context to EXISTING gates.compiled_wrapper/compile_cache_key.

    Their QPY serialization adds the complete IR, signed angles, global phase
    and register layout. No cache from a different route/input/source is shared.
    """
    identity = route_identity(route, input_identity=result.input_identity,
                              source_identity=result.source_identity, options=options)
    expected = 'common_run_support_full_fused' if route == 'B0' else ADAPTER+'_'+route
    if result.construction_policy != expected:
        raise ValueError('Compile route does not match the constructed evolution')
    identity.update(evolution=result.fingerprint, construction_policy=result.construction_policy,
                    context=vars(result.context))
    return identity
