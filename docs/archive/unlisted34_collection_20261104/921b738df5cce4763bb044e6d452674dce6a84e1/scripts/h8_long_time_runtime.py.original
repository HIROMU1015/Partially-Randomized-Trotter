"""Lazy scientific backend. Nothing in this module runs on import.

Prepared in a metadata/synthetic-only task; real H8 execution remains untested
and requires separate scientific permission. No historical launcher is called.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import importlib.metadata
import math
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import h8_long_time_adapter as a

CONTEXT = None


class Writer:
    def __init__(self, directory):
        self.directory = Path(directory)

    def write(self, name, value):
        a.require(Path(name).name == name, 'LOCAL_EVIDENCE_NAME')
        a.save(self.directory / name, value)


class Counts:
    def __init__(self):
        self.values = {}

    def take(self, name):
        self.values[name] = self.values.get(name, 0) + 1


class Context:
    def __init__(self, config, directory, prefixes, *, signal):
        # Keep this adapter's scripts first, then the original provider/library.
        for path in reversed(config['import_paths']):
            if path not in sys.path:
                sys.path.insert(1, path)
        import numpy as np
        import h8_ab_native as loader
        self.np = np
        old = a.read(config['native_config'])
        a.require({k: old['binding'][k] for k in ('hamiltonian', 'state', 'compiler')}
                  == {k: config['binding'][k] for k in ('hamiltonian', 'state', 'compiler')}, 'NATIVE_BINDING')
        original = loader.prefix_preparation
        captured = {}

        def view(ham, blocks, prefix, bridge):
            a.split_blocks(blocks, prefix)
            captured['blocks'] = blocks
            return original(ham, blocks, prefix, bridge)

        # Observe the saved Full blocks already restored by the H8 loader.
        # No original file is edited. Restore the process-local hook on failure.
        try:
            loader.prefix_preparation = view
            values = loader.load(old, Writer(directory), sorted(set(prefixes) - {28} or {7}))
        finally:
            loader.prefix_preparation = original
        (self.m, self.native, self.ham, self.sector, self.state, self.qindices,
         self.qstate, self.preps, self.compiler) = values
        export = a.read(old['mapping']['h8_preparation_export'])['faithful:p28']
        self.preps[28] = SimpleNamespace(deterministic_blocks=tuple(captured['blocks']),
            preparation_hash=export['preparation_hash'], num_system_qubits=16,
            constant_coefficient=self.ham.constant, extracted_identity_coefficient=0.,
            deterministic_fragment_indices=tuple(range(28)))
        for p, expected in config['preparation_identities'].items():
            if int(p) in self.preps:
                a.require(self.preps[int(p)].preparation_hash == expected, 'H8_PREPARATION_IDENTITY')
        self.scalars = {28: 0.}
        if any(p in (15, 16) for p in prefixes):
            from h6_native_b0i_bridge_v1 import exact_identity
            for p in set(prefixes) & {15, 16}:
                exact = sum((exact_identity(g, w) for g, w in zip(self.ham.g_matrices[p:], self.ham.lambdas[p:])), Fraction())
                self.scalars[p] = float(exact)
                if p == 16:
                    receipt = config['DBA_p16_scalar_receipt']
                    a.require(receipt['policy'] == 'EXACT_STORED_BINARY64_TAIL_IDENTITY'
                              and exact == Fraction(receipt['numerator']) / Fraction(receipt['denominator'])
                              and float(exact) == receipt['C'], 'DBA_P16_SCALAR_IDENTITY')
        if signal:
            self.prepare_sparse()

    def prepare_sparse(self):
        from trottertracks.resource_applicability import h8_sparse_reference_v1 as sparse
        self.sparse = sparse
        self.counts = Counts()
        topology = sparse.OccupationCSR(self.sector.basis_indices, 16)
        self.one = topology.one(self.ham.one_body)
        self.blocks = tuple(topology.one(g) for g in self.ham.g_matrices)
        self.actions = (sparse.SpinPrimitive(self.ham.one_body, self.sector),) + tuple(
            sparse.SpinPrimitive(g, self.sector, weight=w) for g, w in zip(self.ham.g_matrices, self.ham.lambdas))
        self.full_op = sparse.df_operator(self.ham.constant, self.one, self.ham.lambdas, self.blocks, self.counts)
        self.tails = {}
        for p in set(self.preps) & {7, 8}:
            prep = self.preps[p]
            self.tails[p] = sparse.df_operator(-prep.extracted_identity_coefficient, self.one * 0,
                self.ham.lambdas[p:], self.blocks[p:], self.counts)

    def scalar(self, row):
        p = row['prefix']
        tail = self.preps[p].extracted_identity_coefficient if row['method'] == 'PR' else self.scalars.get(p, 0.)
        return a.effective_scalar(row['method'], float(self.ham.constant), tail)

    def primitive_gate(self, rows):
        """New signed times, sparse-vs-spin; no reuse of historical time gates."""
        np = self.np
        times = set((i, s['signed_time'] / 2) for row in rows for s in a.schedule(row)
                    for i in range(row['prefix'] + 1))
        probes = [self.state, np.eye(1, len(self.state), 0, dtype=complex)[0],
                  np.eye(1, len(self.state), len(self.state) - 1, dtype=complex)[0]]
        maximum = 0.
        for i, t in sorted(times):
            op = (self.sparse.df_operator(0., self.one, (), (), self.counts) if i == 0 else
                  self.sparse.df_operator(0., self.one * 0, [self.ham.lambdas[i - 1]], [self.blocks[i - 1]], self.counts))
            self.actions[i].phases.clear()
            for v in probes:
                error = float(np.linalg.norm(self.actions[i](v, t) - self.sparse.exponential_action(*op, v, t)))
                a.require(math.isfinite(error) and error < 1e-9, 'NEW_SIGNED_PRIMITIVE_MISMATCH')
                maximum = max(maximum, error)
        return dict(status='PASS', block_times=len(times), probes_per_time=3, maximum_error=maximum,
                    phase_alignment=False, independent_of_orbital_eigensystem=True)

    def evolve(self, row, *, path, horner=False):
        np = self.np
        v = self.state.copy()
        maximum, log_B = 1., 0.
        p = row['prefix']
        for action in self.actions:
            action.phases.clear()  # retain eigensystems; avoid old per-run time-cache cap
        from trotterlib.rte import finite_rte_distribution
        for piece in a.schedule(row):
            t = piece['signed_time']
            for action in self.actions[:p + 1]:
                v = action(v, t / 2)
            if row['method'] == 'PR':
                op, trace = self.tails[p]
                count = piece['rte_steps']
                dt = t / count
                from h46_pr_pilot_repair import canonical_tau
                b = finite_rte_distribution(canonical_tau(self.preps[p], t, count), row['K']).exact_finite_distribution
                log_B += count * math.log(b)
                if path == 'exact_tail':
                    v = self.sparse.exponential_action(op, trace, v, t)
                else:
                    for _ in range(count):
                        v = a.finite_piece(v, lambda x: op @ x, dt, row['K'], b,
                                           raw=path == 'raw', horner=horner)
                        maximum = max(maximum, float(np.linalg.norm(v)))
            for action in reversed(self.actions[:p + 1]):
                v = action(v, t / 2)
            maximum = max(maximum, float(np.linalg.norm(v)))
            a.require(np.isfinite(v).all() and math.isfinite(maximum + log_B), 'FINITE_SIGNAL')
        v *= np.exp(-1j * row['T'] * self.scalar(row))
        return v, maximum, log_B


def complex_record(z):
    return dict(real=float(z.real), imag=float(z.imag))


def signal_job(row):
    ctx = CONTEXT
    np = ctx.np
    corrected, maximum, log_B = ctx.evolve(row, path='corrected')
    if row['method'] == 'PR':
        raw, rawmax, other_log = ctx.evolve(row, path='raw')
        independent, _, _ = ctx.evolve(row, path='corrected', horner=True)
        exact, _, _ = ctx.evolve(row, path='exact_tail')
        discrepancy = float(np.linalg.norm(corrected - independent))
        closure = float(np.linalg.norm(raw - math.exp(-log_B) * corrected))
    else:
        # PF has no normalization or finite recurrence. Reuse the same vector;
        # its independent action check is the sparse-vs-spin primitive gate.
        raw, rawmax, other_log, exact = corrected, maximum, log_B, corrected
        discrepancy = closure = 0.
    a.require(log_B == other_log and max(discrepancy, closure) < 1e-9, 'FINITE_MEAN_RECURRENCE_CLOSURE')
    a.require(abs(float(np.linalg.norm(exact)) - 1.) < 1e-9, 'EXACT_TAIL_PF_NORM')
    z, zr, ze = (complex(np.vdot(ctx.state, v)) for v in (corrected, raw, exact))
    reference = ctx.reference
    record = dict(candidate=row, old_u=max(ctx.reference_error, ctx.primitive_error, discrepancy), old_margin=0.,
        origin='NEW_H8_LONG_TIME_SPARSE', phase_alignment=False,
        signal=dict(reference=complex_record(reference), corrected=complex_record(z), raw=complex_record(zr),
            exact_tail=complex_record(ze), total_signed_bias=complex_record(z - reference),
            outer_PF_or_discard_bias=complex_record(ze - reference), finite_RTE_bias=complex_record(z - ze),
            log_B=log_B, allowance_evidence=dict(maximum_intermediate_norm=max(maximum, rawmax),
                raw_corrected_vector_closure=closure)), recurrence_difference=discrepancy)
    from run_pf_common_run import qualify_record
    return dict(record=record, endpoint=qualify_record(record, row, .0005), matvec_counts=ctx.counts.values)


def independent_trajectory(ctx, row, requests):
    """Algebraic events + saved block actions; arbitrary signed time, scalar once.

    Independent of event-circuit lowering, not of saved orbital decomposition.
    This avoids h6_dba_science.event_oracle's historical hard-coded T=.8.
    """
    np = ctx.np
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import explicit_event_action
    prep = ctx.preps[row['prefix']]
    actions = [ctx.native.make_native_block_action(b, max_instructions=ctx.native.block_instruction_bound(b, "UNCONTROLLED")) for b in prep.deterministic_blocks]
    v = ctx.qstate.copy()
    for index, piece in enumerate(a.schedule(row)):
        for action in actions:
            v = action(v, piece['signed_time'] / 2)
        if requests:
            for event in requests[index].rte_occurrence.events:
                v = explicit_event_action(v, event, prep.rte_preparation.basis_registry)
        for action in reversed(actions):
            v = action(v, piece['signed_time'] / 2)
    return v * np.exp(-1j * row['T'] * ctx.scalar(row))


def price_job(job):
    row, endpoint, cohort, index, directory = job
    ctx = CONTEXT
    np = ctx.np
    import h46_long_time_native as native
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib.df_trotter.circuit import simulate_statevector
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    prep = ctx.preps[row['prefix']]
    requests, seeds, laws = native.sample_requests(row, prep, cohort, index) if row['method'] == 'PR' else ([], [], [])
    events = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    a.save(directory / 'events_PRIVATE.json', dict(seeds=seeds, events=events, conditional=False, cohort=cohort))
    log_B = sum(len(r.rte_occurrence.events) * math.log(r.rte_distribution.exact_finite_distribution) for r in requests)
    a.require(abs(log_B - endpoint['log_B']) <= endpoint['log_B_margin'], 'SIGNAL_NATIVE_NORMALIZATION')
    prices, checks = {}, []

    def gate(name, error, tolerance):
        a.require(math.isfinite(error) and error < tolerance, name)
        checks.append(dict(name=name, error=float(error), tolerance=tolerance, phase_alignment=False))

    with use_fingerprint_reuse() as fp:
        evolution = native.compose(row, prep, requests, ctx.scalar(row), fp.fingerprint)
        low = np.r_[ctx.qstate, np.zeros_like(ctx.qstate)]
        if index == 0:
            expected = independent_trajectory(ctx, row, requests)
            gate('LOW_CONTROL', float(np.linalg.norm(simulate_statevector(evolution.circuit, low) - low)), 1e-8)
            actual = simulate_statevector(evolution.circuit, np.r_[np.zeros_like(ctx.qstate), ctx.qstate])
            gate('HIGH_CONTROL_LEAKAGE', float(np.linalg.norm(actual[:len(expected)])), 1e-8)
            gate('SAVED_BLOCK_ALGEBRAIC_EVENT_VECTOR', float(np.linalg.norm(actual[len(expected):] - expected)), 1e-8)
            signal = complex(np.vdot(ctx.qstate, expected))
        for axis in ('cosine', 'sine'):
            wrapper = ctx.native.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True,
                max_instructions=len(evolution.circuit.data) + 4)
            fingerprint = fp.fingerprint(wrapper)
            compiled, qc, hit = native.compile_cached(wrapper, fingerprint, index == 0)
            # A cached representative must still supply a circuit for validation.
            if index == 0 and qc is None:
                from trotterlib.rte_compiled_cost import transpile_and_measure_cost
                value = transpile_and_measure_cost(wrapper, ctx.compiler, actual_circuit_fingerprint=fingerprint)
                a.require(value.rz_count == compiled['metrics']['rz_count'], 'CACHED_RECOMPILE_PRICE')
                qc = value.transpiled_circuit
            a.require(fp.fingerprint(wrapper) == fingerprint and compiled['compiler'] == a.COMPILER, 'COMPILED_IDENTITY')
            prices[axis] = compiled['metrics']
            if index == 0:
                before = simulate_statevector(wrapper.remove_final_measurements(inplace=False), low)
                d = len(ctx.qstate)
                observed = float(np.vdot(before[:d], before[:d]).real - np.vdot(before[d:], before[d:]).real)
                gate('HADAMARD_' + axis, abs(observed - (signal.real if axis == 'cosine' else signal.imag)), 1e-8)
                after = simulate_statevector(qc.remove_final_measurements(inplace=False), low)
                gate('COMPILED_VECTOR_' + axis, float(np.linalg.norm(after - before)), 1e-7)
    a.require(a.digest(events) == a.digest([[e.to_dict() for e in r.rte_occurrence.events] for r in requests]), 'PAIRED_EVENTS')
    result = dict(status='PASS', candidate_id=row['id'], cohort=cohort, index=index, conditional=False,
                  endpoint_digest=a.digest(endpoint), prices=prices, checks=checks, event_digest=a.digest(events),
                  laws=laws, log_B=log_B, preparation_hash=prep.preparation_hash,
                  orders={str(k): sum(e['taylor_order'] == k for batch in events for e in batch)
                          for k in range(0, (row['K'] or 0) + 1, 2)})
    a.save(directory / 'result.json', result)
    return result


def map_jobs(fn, jobs, workers):
    if workers == 1:
        return [fn(job) for job in jobs]
    # Immutable sparse/spin data shared by COW. No host-wide configuration.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        return list(pool.map(fn, jobs))


def validate_scope(stage, scope, config):
    a.require(scope['stage'] == stage and scope['T'] == 1.6 and scope['epsilon'] == .0005
              and scope['state_preparation_cost'] == 0, 'FIXED_SCIENCE_SCOPE')
    a.require(scope['binding'] == config['binding'], 'SCOPE_INPUT_IDENTITY')
    a.require(type(scope['workers']) is int and scope['workers'] >= 1, 'OWN_WORKERS')
    a.require(scope['candidates'] and len({r['id'] for r in scope['candidates']}) == len(scope['candidates']), 'UNIQUE_CANDIDATES')
    for row in scope['candidates']:
        a.validate_candidate(row)
    diagnostic = [r for r in scope['candidates'] if r.get('diagnostic')]
    if diagnostic:
        selected = scope['diagnostic_S2_endpoint']
        a.validate_candidate(selected['candidate'])
        s2 = selected['candidate']
        a.require(len(diagnostic) == 1 and selected['status'] == 'ELIGIBLE' and s2['method'] == 'PR'
                  and s2['formula'] == '2nd' and s2['prefix'] == diagnostic[0]['prefix']
                  and s2['q'] == diagnostic[0]['q'], 'ONE_SELECTED_ELIGIBLE_S2_MATCHED_Q')
    a.require(scope['source_files'] == config['source_files'], 'FIXED_SCIENTIFIC_SOURCE')
    a.require(scope['SOURCE'] == config['SOURCE'] and len(scope['SOURCE']) == 40, 'FIXED_SOURCE_COMMIT')
    if stage in ('B', 'C'):
        a.require(type(scope['n']) is int and scope['n'] >= 2 and scope['cohort'] in ('EXPLORATION', 'CONFIRMATION'), 'FIXED_COHORT_N')
        a.require(scope['cohort'] == ('EXPLORATION' if stage == 'B' else 'CONFIRMATION'), 'STAGE_COHORT')


def run_stage(stage, config, scope, output):
    global CONTEXT
    validate_scope(stage, scope, config)
    output = Path(output).resolve()
    a.require(output.is_relative_to(Path(config['new_run_root']).resolve()), 'NEW_TASK_OUTPUT_ONLY')
    output.mkdir(parents=True, exist_ok=False)
    a.save(output / 'scope.json', scope)
    try:
        if any(r.get('diagnostic') for r in scope['candidates']):
            a.require(a.sha(scope['diagnostic_stage_A']) == scope['diagnostic_stage_A_sha256'], 'DIAGNOSTIC_SOURCE_ENDPOINT')
            selected = a.read(scope['diagnostic_stage_A'])
            a.require(selected['status'] == 'PASS' and selected['binding'] == config['binding']
                      and selected['source_files'] == config['source_files'], 'DIAGNOSTIC_SOURCE_INPUT')
            a.check_reference(selected['reference'], config['binding'])
            a.require(scope['diagnostic_S2_endpoint'] in [r['endpoint'] for r in selected['results']],
                      'EXACT_SELECTED_S2_ENDPOINT')
        if stage in ('B', 'C'):
            a.require(a.sha(scope['stage_A']) == scope['stage_A_sha256'], 'STAGE_A_BYTES')
            prior = a.read(scope['stage_A'])
            a.require(prior['status'] == 'PASS' and prior['binding'] == config['binding']
                      and prior['source_files'] == config.get('signal_source_files', config['source_files']), 'STAGE_A_SOURCE_INPUT')
            a.check_reference(prior['reference'], config['binding'])
            endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in prior['results']}
            a.require(all(r['id'] in endpoints and endpoints[r['id']]['status'] == 'ELIGIBLE'
                          for r in scope['candidates']), 'ELIGIBLE_SELECTED_ONLY')
        if stage == 'C':
            a.require(a.sha(scope['stage_B']) == scope['stage_B_sha256'], 'EXPLORATION_BYTES')
            exploration = a.read(scope['stage_B'])
            a.require(exploration['status'] == 'PASS' and exploration['binding'] == config['binding']
                      and exploration['source_files'] == config['source_files'], 'EXPLORATION_SOURCE_INPUT')
            exploration_scope = a.read(Path(scope['stage_B']).parent / 'scope.json')
            a.require(exploration_scope['stage_A_sha256'] == scope['stage_A_sha256'], 'SAME_FIXED_SHOT_ENDPOINTS')
            a.require(all(r['id'] in exploration['summaries'] for r in scope['candidates']), 'CONFIRM_SELECTED_FROM_EXPLORATION')
            a.require(scope['comparison_direction'] in ('PR_LOWER', 'PF_LOWER')
                      and math.isfinite(scope['absolute_SE_G_target'])
                      and scope['absolute_SE_G_target'] > 0, 'FIXED_CONFIRMATION_COMPARISON_PRECISION')
            by_id = {r['id']: r for r in scope['candidates']}
            ids = scope['comparison_ids']
            a.require(by_id[ids['PR']]['method'] == 'PR' and by_id[ids['PF']]['method'] != 'PR',
                      'FIXED_PR_PF_COMPARISON')
        CONTEXT = Context(config, output, [r['prefix'] for r in scope['candidates']], signal=stage == 'A')
        ctx = CONTEXT
        if stage == 'A':
            z, evidence = ctx.sparse.reference_pair(*ctx.full_op, ctx.state, 1.6)
            reference = dict(T=1.6, kind='NEW_FULL_H_SPARSE_REFERENCE', binding=config['binding'],
                signal=complex_record(z), certified=False, phase_alignment=False, evidence=evidence)
            a.check_reference(reference, config['binding'])
            a.save(output / 'reference.json', reference)
            gate = ctx.primitive_gate(scope['candidates'])
            a.save(output / 'primitive_gate.json', gate)
            ctx.reference, ctx.primitive_error = z, gate['maximum_error']
            ctx.reference_error = max(evidence['state_difference'], evidence['signal_difference'])
            results = map_jobs(signal_job, scope['candidates'], scope['workers'])
            value = dict(reference=reference, results=results)
        else:
            import h46_long_time_native as native
            native.PRIVATE = output  # confines reused cache to this new stage
            occupied = set(config['excluded_seed_values'])
            if any(r['method'] == 'PR' for r in scope['candidates']):
                from pr_cost_native import seed_hierarchy
                for old in config['historical_seed_tasks']:
                    occupied.update(seed_hierarchy(old['seed'], old['q']))
            if stage == 'C':
                previous_seeds = a.read(Path(scope['stage_B']).parent / 'seeds_PRIVATE.json')
                occupied.update(previous_seeds['values'])
            seed_map = {}
            for row in scope['candidates']:
                if row['method'] != 'PR':
                    continue
                for index in range(scope['n']):
                    for piece in a.schedule(row):
                        key = (row['id'], index, piece['outer'], piece['piece'])
                        seed = int(a.digest(['H8_LONG_TIME_V1', config['SOURCE'], config['source_files'], config['binding'],
                                            scope['cohort'], *key])[:16], 16)
                        a.require(seed not in occupied, 'OLD_OR_NEW_SEED_COLLISION')
                        occupied.add(seed)
                        seed_map[key] = seed
            a.save(output / 'seeds_PRIVATE.json', dict(cohort=scope['cohort'], values=list(seed_map.values())))
            native.trial_seed = lambda row, cohort, index, outer, piece: seed_map[(row['id'], index, outer, piece)]
            jobs = [(r, endpoints[r['id']], scope['cohort'], index,
                     str(output / 'prices' / r['id'] / str(index)))
                    for r in scope['candidates'] for index in range(scope['n'] if r['method'] == 'PR' else 1)]
            trials = map_jobs(price_job, jobs, scope['workers'])
            summaries = {}
            for row in scope['candidates']:
                selected = [r for r in trials if r['candidate_id'] == row['id']]
                stats = a.paired_G(selected, endpoints[row['id']], scope['cohort'])
                if row['method'] == 'PR':
                    observed = {str(k): sum(r['orders'].get(str(k), 0) for r in selected) for k in (0, 2, 4)}
                    stats['rarity'] = native.rarity(selected[0]['laws'], ctx.preps[row['prefix']], row['K'], len(selected), observed)
                summaries[row['id']] = stats
            value = dict(trials=trials, summaries=summaries, pooled=False)
            if stage == 'C':
                value['confirmation'] = a.confirmation_comparison(summaries, scope['comparison_ids'],
                    scope['comparison_direction'], scope['absolute_SE_G_target'])
        a.save(output / 'result.json', dict(status='PASS', stage=stage, binding=config['binding'],
            source_files=config['source_files'], SOURCE=config['SOURCE'],
            signal_SOURCE=config['SOURCE'] if stage == 'A' else prior['SOURCE'], scope_digest=a.digest(scope),
            environment={name: importlib.metadata.version(name) for name in ('qiskit', 'numpy', 'scipy', 'numba')}, **value))
    except BaseException as exc:
        a.save(output / 'HOLD.json', dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
