"""Opt-in C1 price acquisition acceleration; no sampling or scientific launcher.

The pinned backend still owns input/preparation checks and compose/wrapper/compiler.
Only the structural RZ projection and the canonical-byte writer are accelerated.
Historical backends, provider files, caches and execution contracts are untouched.
"""
from copy import deepcopy
from pathlib import Path
import time

from . import efficient_controlled_zz as E


class _RZCounter:
    """Projection of NativeCounter onto rz_count, retaining instruction validation."""
    def __init__(self, nq, nc):
        self.num_qubits, self.num_clbits, self.count = nq, nc, 0

    def feed(self, instruction):
        from .track_d_native_resources import NativeInstruction
        if not isinstance(instruction, NativeInstruction):
            raise TypeError('NativeInstruction required; no implicit lowering')
        if (any(q >= self.num_qubits for q in instruction.qubits)
                or any(c >= self.num_clbits for c in instruction.clbits)):
            raise ValueError('native wire outside declared register')
        self.count += instruction.name == 'rz'
        return self

    def result(self):
        return dict(metrics=dict(rz_count=self.count))


def price_accounting(base):
    """Keep all original expansion/folding logic; omit unused depth recurrences."""
    class RZAccounting(base):
        def __init__(self):
            super().__init__()
            self.native = _RZCounter(self.native.num_qubits, self.native.num_clbits)
            self.folded = _RZCounter(self.folded.num_qubits, self.folded.num_clbits)
    return RZAccounting


def structural_pair(Accounting, blocks, row, scalar, events):
    """Same C1 word and floating addition order, sharing only the common prefix.

    Snapshot BEFORE finish: pending basis/Z state crosses the wrapper boundary.
    Both suffixes are evaluated; equality of X/Y is never presumed.
    """
    if row['system'] != 'H4':
        raise ValueError('This acceleration is scoped to the H4 Accounting word')
    a = price_accounting(Accounting)()
    control, J = 8, len(blocks)
    a.gate('h', (control,), role='wrapper')
    for outer in range(row['q']):
        for j, bs in enumerate(blocks):
            for b in bs:
                a.block(b, False)
            a.tail(events[J * outer + j])
            for b in reversed(bs):
                a.block(b, True)
    if scalar * row['T'] != 0:
        a.gate('p', (control,), (-scalar * row['T'],), 'scalar_once')
    y = deepcopy(a)
    out = []
    for state, axis in ((a, 'cosine'), (y, 'sine')):
        if axis == 'sine':
            state.gate('sdg', (control,), role='wrapper')
        state.gate('h', (control,), role='wrapper')
        state.gate('measure', (control,), role='measurement')
        out.append(state.finish())
    return out


def make_backend_class(base, science, *, axes=('cosine', 'sine'), observer=None):
    """Extend the pinned C1 Backend explicitly; does not load any scientific input.

    `science` is the r_price_phase1_science module from p1_efficient_zz.load_c1_backend.
    Callers must activate efficient_controlled_zz before pricing. A future scientific
    use requires its own fixed SOURCE, inputs, finite scope and explicit permission.
    """
    axes = tuple(axes)
    if not axes or len(set(axes)) != len(axes) or any(a not in ('cosine', 'sine') for a in axes):
        raise ValueError('Explicit unique Hadamard axes required')
    class AcceleratedBackend(base):
        def price_requests(self, cell, index, directory, requests, seeds, laws, *, conditional):
            m, write, read = science.m, science.write, science.read
            c = m.row_for(self.plan, cell)['candidate']
            if c['system'] != 'H4':
                raise ValueError('This acceleration is scoped to H4')
            ctx = self.context(cell['system'])
            # The pinned context establishes the provider import path first.
            from trotterlib import df_rte_qiskit as dfq
            from h46_b3_fast_fingerprint import use_fast_fingerprint
            if getattr(dfq.QiskitDFRTEEventCircuitBuilder, 'lowering_id', None) != E.LOWERING_ID:
                raise ValueError('Efficient controlled ZZ must be active')
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            row = m.row_for(self.plan, cell)
            allocation, weights = row['structure']['allocation'], row['structure']['signed_weights']
            if len(requests) != c['q'] * len(allocation) or any(
                    len(r.rte_occurrence.events) != allocation[i % len(allocation)]
                    or r.step_time != c['T'] * weights[i % len(weights)] / c['q'] for i, r in enumerate(requests)):
                raise ValueError('SIGNED_EVENT_ALLOCATION_OR_TIME')
            stages = {}
            def timed(name, fn):
                if observer is not None:
                    observer(name, 'STARTED', 0., None)
                start = time.perf_counter()
                value = fn()
                stages[name] = time.perf_counter() - start
                if observer is not None:
                    observer(name, 'COMPLETE', stages[name], value)
                return value
            start = time.perf_counter()
            events = timed('event_dicts', lambda: [[e.to_dict() for e in r.rte_occurrence.events] for r in requests])
            timed('event_write', lambda: write(directory / 'events_PRIVATE.json', dict(
                events=events, seeds=seeds, laws=laws, conditional=conditional)))
            event_digest = timed('event_digest', lambda: m.digest(events))
            pair = timed('structural', lambda: structural_pair(
                ctx.Accounting, ctx.blocks, c, ctx.scalar, events))
            S = [pair[('cosine', 'sine').index(axis)]['structural_rz'] for axis in axes]
            write(directory / 'STRUCTURAL.json', dict(S=S, event_digest=event_digest, scientific_source=self.plan['source']))
            C, fingerprints, hits = [], [], 0
            with use_fast_fingerprint() as fp:
                # Reuse the exact carrier compose and provider wrapper; no gate emitter fork.
                evolution = timed('compose', lambda: ctx.h['compose'](c, ctx.prep, requests, ctx.scalar, fp.fingerprint))
                for axis in axes:
                    wrapper = timed(axis + '_wrapper', lambda: ctx.native.build_native_hadamard_wrapper(
                        evolution, axis=axis, include_measurement=True, max_instructions=len(evolution.circuit.data) + 4))
                    fingerprint = timed(axis + '_fingerprint', lambda: fp.fingerprint(wrapper))
                    old = Path(self.plan['runtime']['old_cache']) / (fingerprint + '.json')
                    write(directory / (axis + '.started.json'), dict(fingerprint=fingerprint, compiler=m.p.PROFILE['compiler']))
                    hit = old.exists()
                    if hit:
                        cache = read(old)
                        if cache['fingerprint'] != fingerprint or cache['compiler'] != m.p.PROFILE['compiler']:
                            raise ValueError('CACHE_BINDING_NO_FALLBACK')
                        metrics = cache['metrics']
                        hits += 1
                    else:
                        settings = ctx.CompilerSettings(('rz', 'sx', 'x', 'cx'), None, None, 1, None, None, 17, '1.3.0')
                        cost = timed(axis + '_compile', lambda: ctx.compile(
                            wrapper, settings, actual_circuit_fingerprint=fingerprint))
                        if cost.compiler_settings_hash != m.p.PROFILE['compiler']:
                            raise ValueError('COMPILER_HASH')
                        metrics = timed(axis + '_metrics_extract', lambda: {k: getattr(cost, k) for k in (
                            'rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')})
                        del cost
                    value = metrics['rz_count']
                    after = timed(axis + '_mutation_check', lambda: fp.fingerprint(wrapper))
                    if type(value) is not int or value < 0 or after != fingerprint:
                        raise ValueError('COMPILED_PRICE_OR_MUTATED_INPUT')
                    write(directory / (axis + '.json'), dict(metrics=metrics, fingerprint=fingerprint,
                          compiler=m.p.PROFILE['compiler'], cache_hit=hit))
                    C.append(value)
                    fingerprints.append(fingerprint)
                    del wrapper
                hash_stats = dict(calls=fp.calls, hits=fp.hits, misses=fp.misses)
            result = dict(cell_id=cell['id'], candidate_id=c['id'], index=index, cohort=cell['cohort'],
                conditional=conditional, C=C, S=S, source=dict(execution_source=self.plan['source'],
                private_result=str(directory / 'result.json')), event_digest=event_digest,
                compiler=m.p.PROFILE['compiler'], fingerprints=fingerprints, cache_hits=hits, new_compiles=len(axes) - hits,
                full_wrapper_statevector_validation=False, signal_N_G_computed=False)
            if axes != ('cosine', 'sine'):
                result['axes'] = list(axes)
            write(directory / 'result.json', result)
            write(directory / 'TECHNICAL_TIMING.json', dict(stages_seconds=stages,
                total_seconds=time.perf_counter() - start, fingerprint=hash_stats))
            return result
    return AcceleratedBackend
