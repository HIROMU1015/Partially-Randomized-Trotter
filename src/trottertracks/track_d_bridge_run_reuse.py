"""Reuse unchanged 1Q runs, without changing the frozen 1Q/CX rules.

An internal run is an opaque, nonempty single-wire barrier for the original
CX feed. This reduces instruction visits, not the number of logical epochs.
It is not a general P0 compiler or a bounded-frontier streaming algorithm.
"""
import math
import time

from .track_d_native_resources import NativeCounter, count_native
from .track_d_stream_bridge_fast import OneQStream, ObservedCX, _consume_owned


class _Run:
    __slots__ = ('ops', 'qubits', 'memo')
    name = '_internal_1q_run'
    clbits = ()

    def __init__(self, ops, wire):
        self.ops = tuple(ops)
        self.qubits = (wire,)
        self.memo = None


def _weight(unit):
    return len(unit.ops) if isinstance(unit, _Run) else 1


def flatten(units):
    for unit in units:
        if isinstance(unit, _Run):
            yield from unit.ops
        else:
            yield unit


class _Epoch:
    def __init__(self, nq, previous):
        self.nq = nq
        self.resolver = OneQStream(nq, previous)
        self.phase = []
        self.replaced = self.input_size = self.unit_visits = 0
        self.native_classifications = self.memo_hits = self.memo_native_skipped = 0
        self.merged_runs = self.peak_pending_units = 0

    def run(self, units):
        pending = [[] for _ in range(self.nq)]
        pending_count = 0

        def flush(wire):
            nonlocal pending_count
            pieces = pending[wire]
            if not pieces:
                return None
            single = pieces[0] if len(pieces) == 1 and isinstance(pieces[0], _Run) else None
            if single is not None and single.memo is not None:
                output, addition, replaced = single.memo
                output = single if output is None else output
                self.memo_hits += 1
                self.memo_native_skipped += len(single.ops)
            else:
                # Reuse the exact previous bit cache and official run routine.
                # Only a new/changed/merged run is expanded into instructions.
                p0 = len(self.resolver.phase)
                r0 = self.resolver.replaced
                result = tuple(self.resolver.run(flatten(pieces)))
                replaced = self.resolver.replaced - r0
                addition = math.fsum(self.resolver.phase[p0:])
                if replaced:
                    output = _Run(result, wire) if result else None
                else:
                    output = single if single is not None else _Run(result, wire)
                    output.memo = (None, 0., 0)
                if single is not None and output is not None:
                    single.memo = (None if output is single else output, addition, replaced)
                if len(pieces) > 1:
                    self.merged_runs += 1
            self.replaced += replaced
            if replaced:
                self.phase.append(addition)
            pending_count -= len(pieces)
            pending[wire] = []
            return output

        for unit in units:
            self.unit_visits += 1
            self.input_size += _weight(unit)
            if isinstance(unit, _Run) or unit.name in ('rz', 'sx', 'x'):
                if not isinstance(unit, _Run):
                    self.native_classifications += 1
                pending[unit.qubits[0]].append(unit)
                pending_count += 1
                self.peak_pending_units = max(self.peak_pending_units, pending_count)
            else:
                self.native_classifications += 1
                for wire in unit.qubits:
                    output = flush(wire)
                    if output is not None:
                        yield output
                yield unit
        for wire in range(self.nq):
            output = flush(wire)
            if output is not None:
                yield output


def optimize_native_runs(operations, num_qubits, num_clbits=0, *, global_phase=0., trace=None):
    """Exact same epoch/flush schedule; optional trace is outside benchmarks.

    trace(round_index, iterator_of_native_ops, binary64_phase) consumes the
    iterator while ingestion is paused. Caller inputs are never modified.
    """
    import qiskit
    if qiskit.__version__ != '1.3.0':
        raise ValueError('fixed Qiskit1.3.0 required')
    initial = NativeCounter(num_qubits, num_clbits, global_phase=global_phase)

    def validated():
        for op in operations:
            initial.feed(op)
            yield op

    stream = validated()
    phase = float(global_phase)
    previous = {}
    rounds, additions, details = [], [], []
    peak_units = peak_units_bytes = peak_run_instructions = 0
    while True:
        start = time.perf_counter()
        epoch = _Epoch(num_qubits, previous)
        cx = ObservedCX(num_qubits, num_clbits, recount=False).extend(epoch.run(stream))
        addition = math.fsum(epoch.phase)
        additions.append(addition)
        phase = math.fsum((phase, addition))
        units = list(cx.operations())
        size = sum(map(_weight, units))
        peak_units = max(peak_units, len(units))
        peak_units_bytes = max(peak_units_bytes, units.__sizeof__())
        run_instructions = sum(len(u.ops) for u in units if isinstance(u, _Run))
        peak_run_instructions = max(peak_run_instructions, run_instructions)
        rounds.append(dict(input_size=epoch.input_size, output_size=size,
                           oneq_runs_replaced=epoch.replaced, cx_pairs_removed=cx.cancelled_pairs))
        details.append(dict(seconds=time.perf_counter()-start, cx=cx.statistics(),
            unit_visits=epoch.unit_visits, native_classifications=epoch.native_classifications,
            run_internal_visits=epoch.resolver.scanned, memo_hits=epoch.memo_hits,
            memo_native_skipped=epoch.memo_native_skipped, merged_runs=epoch.merged_runs,
            cache_hits=epoch.resolver.hits, synthesis_calls=epoch.resolver.calls,
            synthesis_seconds=epoch.resolver.synthesis_seconds,
            cache_slots=len(previous)+len(epoch.resolver.cache),
            peak_pending_units=epoch.peak_pending_units, unit_count=len(units),
            run_instruction_slots=run_instructions))
        if trace is not None:
            trace(len(rounds)-1, flatten(units), phase)
        previous = epoch.resolver.cache
        del cx, epoch
        if size == rounds[-1]['input_size']:
            break
        if size > rounds[-1]['input_size']:
            raise ValueError('same rule must strictly decrease size')
        stream = _consume_owned(units)
    word = list(flatten(units))
    output = count_native(iter(word), num_qubits, num_clbits, global_phase=phase)
    return dict(input=initial.result(), output=output, operations=word, global_phase=phase,
        rounds=rounds, phase_additions=additions,
        performance=dict(rounds=details, peak_unit_length=peak_units,
            peak_unit_list_bytes=peak_units_bytes, peak_run_instruction_slots=peak_run_instructions,
            full_native_rescan_after_first=False, all_unit_epochs_still_scanned=True),
        p0_qualification='P0_UNQUALIFIED')
