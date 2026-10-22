"""Observe returned production events, without replay, RNG or global patches.

New opt-in adapters leave the frozen H2 APIs/source unchanged. Their trajectory
methods use the same production calls, arguments and builders as the originals.
The observer receives immutable event objects before circuit construction.
"""
from dataclasses import asdict
import math
from pathlib import Path

from .fixed_signal_evaluation import DFSignalAdapter
from .isolated_qiskit_execution import digest, read, save
from .pauli_signal_adapter import PauliSignalAdapter, append_pauli_evolution, append_pauli_event


class TrajectoryAudit:
    def __init__(self, case_id, binding, *, directory=None, validator=None, max_trajectories=None):
        self.case_id, self.binding = case_id, dict(binding)
        self.directory = None if directory is None else Path(directory)
        self.validator = validator
        self.max_trajectories = max_trajectories
        self.rows, self._seen = [], set()
        self.current = None

    def begin(self, seed, index):
        if self.max_trajectories is not None and (len(self.rows)>=self.max_trajectories or index!=len(self.rows)):
            raise ValueError("AUDIT_FIXED_TRAJECTORY_BUDGET_OR_ORDER")
        if self.current is not None:
            raise ValueError("AUDIT_PREVIOUS_TRAJECTORY_NOT_JOINED")
        tid = digest(dict(case_id=self.case_id, binding=self.binding, index=index, seed=seed))
        if tid in self._seen:
            raise ValueError("AUDIT_DUPLICATE_TRAJECTORY")
        self._seen.add(tid)
        self.current = dict(schema="production_rte_trajectory_v1", trajectory_id=tid,
            case_id=self.case_id, binding=self.binding, trajectory_index=index, seed=seed)
        self.objects = None

    def observe(self, adapter, sequences, step_seeds):
        if self.current is None or self.objects is not None:
            raise ValueError("AUDIT_MISSING_BEGIN_OR_DUPLICATE_EVENTS")
        self.objects = tuple(tuple(events) for events in sequences)
        records = []
        distribution = None if adapter.distribution is None else adapter.distribution.to_dict()
        distribution_id = digest(dict(representation=adapter.representation,
            hamiltonian=adapter.metadata()["hamiltonian_hash"], L_D=adapter.ld,
            law=distribution))
        for outer, events in enumerate(self.objects):
            for inner, event in enumerate(events):
                records.append(dict(event_position=len(records), outer_step_index=outer,
                    inner_RTE_position=inner, sampler_seed=step_seeds[outer],
                    distribution_id=distribution_id, **event.to_dict()))
        q = len(self.objects)
        expected = q*adapter.r if adapter.randomized else 0
        if len(records) != expected:
            raise ValueError("AUDIT_EVENT_COUNT")
        scalar = adapter.operators()[3]
        self.current.update(representation=adapter.representation, method=adapter.method,
            events=records, event_count=expected, q=q, r=adapter.r,
            step_seeds=list(step_seeds), distribution=distribution, distribution_id=distribution_id,
            normalization=dict(lambda_r=adapter.lambda_r, log_B=q*adapter.log_B_step),
            scalar_phase=dict(angle=-float(scalar)*adapter.step_time*q,
                policy="original_scalar_once_per_outer_step; event phases separate"),
            information_missing=[], production_returned_events=True, resampling_for_audit=False)
        self.current["event_record_hash"] = digest(self.current)
        if self.directory is not None:
            save(self.directory/self.current["trajectory_id"]/"events.json", self.current)

    def validate_evolution(self, adapter, config, circuit):
        if self.validator is not None:
            return self.validator(adapter, config, self.objects, circuit, self.current["trajectory_index"])

    def join(self, prices, compiler, fingerprints):
        if self.current is None or self.objects is None or set(prices) != {"cosine", "sine"}:
            raise ValueError("AUDIT_INCOMPLETE_JOIN")
        if any(type(n) is not int or n < 0 for n in prices.values()):
            raise ValueError("AUDIT_INVALID_PRICE")
        item = dict(self.current, compiled_RZ=dict(prices), compiler=asdict(compiler),
                    wrapper_fingerprints=dict(fingerprints), event_costs_additive=False)
        item["trajectory_record_hash"] = digest(item)
        if self.directory is not None:
            save(self.directory/item["trajectory_id"]/"price_join.json", item)
        self.rows.append(item)
        self.current = self.objects = None
        return item


class AuditedPauliSignalAdapter(PauliSignalAdapter):
    def __init__(self, *args, audit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = audit

    def trajectory(self, *, q, seed):
        from qiskit import QuantumCircuit
        from .pauli_signal_adapter import sample_rte_events
        # Exactly one production call, same q*r count and seed as frozen API.
        sampled = sample_rte_events(self.components, self.distribution,
            sample_count=q*self.r, seed=seed) if self.randomized else ()
        sequences = [sampled[i*self.r:(i+1)*self.r] for i in range(q)] if self.randomized else [()] * q
        if self.audit is not None:
            self.audit.observe(self, sequences, [seed if self.randomized else None]*q)
        circuit = QuantumCircuit(self.num_qubits)
        events = iter(sampled)
        for _ in range(q):
            for label, c in self.retained:
                append_pauli_evolution(circuit, label, c, self.step_time/2)
            if self.randomized:
                for _ in range(self.r):
                    append_pauli_event(circuit, next(events), self.labels)
            for label, c in reversed(self.retained):
                append_pauli_evolution(circuit, label, c, self.step_time/2)
            circuit.global_phase -= self.hamiltonian.constant * self.step_time
        return circuit


class AuditedDFSignalAdapter(DFSignalAdapter):
    def __init__(self, *args, audit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = audit

    def trajectory(self, *, q, seed, controlled=False):
        from .df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder, make_df_partial_s2_repeated_request
        # The returned production request holds the actual sampled occurrences.
        request = make_df_partial_s2_repeated_request(self.preparation, step_time=self.step_time,
            repetition_count=q, rte_config=self.config, rte_distribution=self.distribution,
            seed=seed, controlled=controlled, ancilla_qubit=self.num_qubits if controlled else None,
            construction_policy="raw_concatenation")
        if self.audit is not None:
            sequences = [occ.events for occ in request.rte_occurrences] if self.randomized else [()] * q
            self.audit.observe(self, sequences, request.step_seeds)
        return QiskitDFPartialS2RepeatedCircuitBuilder().build(request).circuit


def audited_adapter(hamiltonian, config, audit=None):
    cls = AuditedPauliSignalAdapter if config.hamiltonian_type == "pauli_jw" else AuditedDFSignalAdapter
    return cls(hamiltonian, config.method, config.L_D, config.T/config.q, config.r, config.K, audit=audit)


def verify_join(item, *, expected_case=None, expected_binding=None):
    """Reject corrupted, incomplete, repositioned or duplicated event records."""
    item = dict(item)
    expected_hash = item.pop("trajectory_record_hash")
    if digest(item) != expected_hash:
        raise ValueError("TRAJECTORY_JOIN_HASH")
    base = {k:v for k,v in item.items() if k not in
            {"compiled_RZ", "compiler", "wrapper_fingerprints", "event_costs_additive"}}
    event_hash = base.pop("event_record_hash")
    if digest(base) != event_hash:
        raise ValueError("EVENT_RECORD_HASH")
    if expected_case is not None and base["case_id"] != expected_case:
        raise ValueError("EVENT_CASE_BINDING")
    if expected_binding is not None and base["binding"] != expected_binding:
        raise ValueError("EVENT_INPUT_BINDING")
    tid = digest(dict(case_id=base["case_id"], binding=base["binding"],
        index=base["trajectory_index"], seed=base["seed"]))
    if tid != base["trajectory_id"]:
        raise ValueError("TRAJECTORY_ID_BINDING")
    events = base["events"]
    expected = base["q"]*base["r"] if base["method"] == "PR" else 0
    if len(events) != expected or base["event_count"] != expected:
        raise ValueError("EVENT_JOIN_COUNT")
    for i,e in enumerate(events):
        if (e["event_position"],e["outer_step_index"],e["inner_RTE_position"]) != (i,i//base["r"],i%base["r"]):
            raise ValueError("EVENT_JOIN_ORDER")
        if e["distribution_id"] != base["distribution_id"] or len(e["application_sequence"]) != e["taylor_order"]+1:
            raise ValueError("EVENT_LAW_OR_APPLICATION_ORDER")
    if set(item["compiled_RZ"]) != {"cosine", "sine"}:
        raise ValueError("EVENT_JOIN_PRICES")
    return True
