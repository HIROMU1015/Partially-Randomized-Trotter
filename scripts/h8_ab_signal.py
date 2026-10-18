"""Targeted H8 PR boundary signals, cached sparse oracles and fixed allowances."""
from copy import copy, deepcopy
import math
from pathlib import Path
import h8_ab_contract as c
import h8_ab_native as native
import h8_key_compile_parallel as pool
from h6_dba_science import checked_results
from h6_dba_signal import forward_finite, Progress
from h8_strengthened_signal import saved_reference


def saved_primitive_coverage(config):
    expected = {(str('one' if i == 0 else i - 1), t)
                for i in range(9) for t in (.2, -.2, .1, -.1, .05, -.05)}
    covered = set()
    for role in ('primitive_q2', 'primitive_q4', 'primitive_q8'):
        value = c.read(config['mapping'][role])
        c.require(value['all_registered_times'] and value['probe_count'] == 3, 'SAVED_PRIMITIVE_RECEIPT')
        for row in value['records']:
            key = (row['block'], row['time'])
            if key in expected:
                errors = row['native_spin_errors'] + row['spin_CSR_errors']
                c.require(len(errors) == 6 and all(math.isfinite(e) and 0 <= e < 1e-9 for e in errors),
                          'SAVED_PRIMITIVE_ERROR')
                covered.add(key)
    c.require(covered == expected, 'SAVED_PHYSICAL_TIME_COVERAGE')
    return dict(status='PASS', block_time_pairs=len(covered), saved_probe_results_recomputed=False,
                receipt_sha256={role: c.sha(config['mapping'][role])
                               for role in ('primitive_q2', 'primitive_q4', 'primitive_q8')})


def signal(config, grant, output):
    writer = c.Writer(output)
    m, n, ham, sector, state, qindices, qstate, views, compiler = native.load(config, writer, (6, 7, 8))
    from trottertracks.resource_applicability import h8_resource_map_contract_v1 as old
    from trottertracks.resource_applicability import h8_resource_map_port_v1 as core
    from trottertracks.resource_applicability import h8_sparse_reference_v1 as sparse
    from trottertracks.resource_applicability import ax2b_molecular_ports_v3 as molecular
    from trotterlib.df_hamiltonian import DFHamiltonian, df_linear_operator
    cells = c.plan()['A']['cells']
    p = old.plan()
    p.update(cells=cells, epsilons=c.plan()['A']['epsilons'], matrix_free=dict(p['matrix_free'], num_threads=1))
    p['numerical']['sensitivity_factors'] = [1.]
    p['caps_proposed'].update(compile=0, trajectory=0, occurrence=0, control_probe=0,
        primitive=48, reference_matvec=50_000_000)
    p['caps_proposed']['tail_matvecs_corrected_and_raw']['B2'] = max(2 * x['R'] * (x['K'] + 1) for x in cells)
    port = core.H8CoreSignalPort(Path(config['provider_root']), dict(kind='H8_AB_BOUNDARY', plan=p), writer, Progress())
    port.ham, port.sector, port.qindices, port.qstate = ham, sector, qindices, qstate
    port.state = state.copy(); port.saved_state = state.copy()
    port.reference, port.reference_error = saved_reference(c.read(config['mapping']['full_H_reference']), config['binding'])
    port.preps = {x['id']: views[x['prefix']] for x in cells}
    port.tails = {}; port.oracle_tails = {}; port.spin = {}; port.primitive_ops = {}; port.truncated_references = {}
    topology = sparse.OccupationCSR(sector.basis_indices, 16)
    port.one = topology.one(ham.one_body); port.blocks = tuple(topology.one(g) for g in ham.g_matrices)
    del topology
    for prefix, prep in views.items():
        ids = prep.randomized_block_indices; lam = prep.exact_rte_lambda_r
        port.tails[prefix] = df_linear_operator(DFHamiltonian(-prep.extracted_identity_coefficient,
            m.np.zeros_like(ham.one_body), ham.lambdas[list(ids)].copy(), tuple(ham.g_matrices[i] for i in ids), {}),
            sector, **p['matrix_free'])[0]
        port.oracle_tails[prefix] = sparse.df_operator(-prep.extracted_identity_coefficient / lam, port.one * 0,
            ham.lambdas[list(ids)] / lam, tuple(port.blocks[i] for i in ids), port.calls)
        port.actions(prep)  # Small spin eigensystems warmed once, inherited by COW.
    old_gate = saved_primitive_coverage(config)
    writer.write('saved_primitive_reuse.json', old_gate)
    fast = port.actions(views[7]); lowered = molecular.MolecularPort.actions(port, views[7])
    probes = [state, m.np.eye(1, len(state), 0, dtype=complex)[0],
              m.np.eye(1, len(state), len(state) - 1, dtype=complex)[0]]

    def batch(tasks, callback, name):
        return pool.map_owned(tasks, callback, cpus=grant['worker_cpus'],
            limits=dict(max_workers=len(grant['worker_cpus'])), directory=output / name,
            deadline=math.inf, font_seed=config['font_seed'], reserve=lambda task: writer.write(
                'reservation_' + task['id'] + '.json', dict(task=task, nonrefundable=True)))

    def primitive(task, directory):
        i = task['block']; op, trace = port.oracle_primitive('one' if i == 0 else i - 1)
        errors = []
        for t in (.025, -.025):
            for v in probes:
                a, f, o = lowered[i](v, t), fast[i](v, t), sparse.exponential_action(op, trace, v, t)
                errors += [float(m.np.linalg.norm(a - f)), float(m.np.linalg.norm(f - o))]
        error = max(errors)
        m.finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=error)
        c.require(math.isfinite(error) and error < 1e-9, 'NEW_TIME_PRIMITIVE_MISMATCH')
        return dict(block=i, times=[.025, -.025], probes_per_time=3, max_error=error, status='PASS')

    primitive_receipt = batch([dict(id='primitive_' + str(i), block=i) for i in range(8)], primitive,
                              'new_time_primitive_workers')
    values = checked_results(primitive_receipt)
    c.require(len(values) == 8, 'NEW_TIME_COVERAGE')
    writer.write('primitive_gate.json', dict(status='PASS', new_time=values, old_times=old_gate,
        primitive_probe_triples=48, new_Gaussian_operations=0))
    outer = {}
    for q in (4, 8):
        saved = c.read(config['mapping']['old_signal_q' + str(q)])['signals']
        row = next(x for x in saved if x['cell']['id'] == f'B2_p7_q{q}_2nd_r4_K4')
        reference = complex(row['reference']['real'], row['reference']['imag'])
        bridge = abs(reference - port.reference)
        c.require(bridge <= 1e-12, 'SAVED_FULL_REFERENCE_BRIDGE')
        outer[(7, q)] = dict(prefix=7, q=q, signal=row['signals']['exact_tail'],
            error=max(row['signal_discrepancies'] + row['state_discrepancies']) + bridge,
            kind='SAVED_ORIGIN_OUTER_DIAGNOSTIC', reference_bridge=bridge)

    def new_outer(task, directory):
        prefix, q = task['prefix'], task['q']; prep = views[prefix]
        tail_op, tail_trace = port.oracle_tails[prefix]
        scalar = prep.constant_coefficient + prep.extracted_identity_coefficient
        tau = prep.exact_rte_lambda_r * .8 / q
        vectors = []
        # Spin/direct-tail and CSR/split-tail paths validate each other.
        for oracle in (False, True):
            actions = port.actions(prep, oracle=oracle); v = state.copy()
            for _ in range(q):
                for action in actions: v = action(v, .8 / q / 2)
                if oracle:
                    v = sparse.exponential_action(tail_op, tail_trace,
                        sparse.exponential_action(tail_op, tail_trace, v, tau / 2), tau / 2)
                else:
                    v = sparse.exponential_action(tail_op, tail_trace, v, tau)
                for action in reversed(actions): v = action(v, .8 / q / 2)
                v *= m.np.exp(-1j * scalar * .8 / q)
            vectors.append(v)
        zs = [complex(m.np.vdot(state, v)) for v in vectors]
        discrepancy = float(m.np.linalg.norm(vectors[0] - vectors[1]))
        error = max(discrepancy, abs(zs[0] - zs[1]))
        m.finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=error)
        c.require(math.isfinite(error) and error < 1e-8, 'OUTER_PF_INDEPENDENT_PATH_MISMATCH')
        return dict(prefix=prefix, q=q, signal=m.complex_record(zs[1]), error=error,
                    kind='NEW_SPARSE_OUTER_PF_PAIR', full_H_reference_recomputed=False, certified=False)

    outer_tasks = [dict(id=f'outer_p{x["prefix"]}_q{x["q"]}', **x) for x in c.plan()['A']['new_outer']]
    outer_receipt = batch(outer_tasks, new_outer, 'outer_workers')
    for value in checked_results(outer_receipt): outer[(value['prefix'], value['q'])] = value
    writer.write('outer_gate.json', dict(status='PASS', new_pairs=7, saved_pairs=2,
        rows=list(outer.values()), new_full_H_reference=0))

    def candidate(task, directory):
        local = copy(port); local.cells = [task['cell']]; local.plan = dict(p, cells=local.cells)
        local.writer = c.Writer(directory); local.progress = Progress(); local.completed = 0
        # Each fork owns its copied counters/caches; operator closures reference that same budget.
        def oracle(cell):
            prep = views[cell['prefix']]
            value = forward_finite(state, local.actions(prep, oracle=True),
                local.oracle_tails[cell['prefix']][0], cell,
                prep.constant_coefficient + prep.extracted_identity_coefficient, prep.exact_rte_lambda_r, m.np)
            saved = outer[(cell['prefix'], cell['q'])]['signal']
            value['signals']['exact_tail'] = complex(saved['real'], saved['imag'])
            return value
        local.oracle_signal = oracle
        local.correctness()
        row = c.read(directory / 'signal_summary.json')['signals'][0]
        extra = outer[(task['cell']['prefix'], task['cell']['q'])]
        row['outer_diagnostic_binding'] = extra
        row['allowance']['real'] += extra['error']; row['allowance']['imag'] += extra['error']
        row['precision_rows'] = m.precision_rows(row, local.plan)
        envelope = deepcopy(row)
        envelope['error_decomposition']['total_signed'] = {
            axis: math.fsum(abs(value[axis]) for value in row['error_decomposition']['signed_parts'].values())
            for axis in ('real', 'imag')}
        row['empirical_noncancellation_precision_rows'] = m.precision_rows(envelope, local.plan)
        row['strict_bias_certificate'] = None
        local.writer.write('boundary_signal.json', row)
        return row

    signal_receipt = batch([dict(id=x['id'], cell=x) for x in cells], candidate, 'signal_workers')
    rows = checked_results(signal_receipt)
    c.require(len(rows) == 16 and {x['cell']['id'] for x in rows} == {x['id'] for x in cells}, 'EXACT_A_CELLS')
    writer.write('worker_receipts.json', dict(primitive=primitive_receipt, outer=outer_receipt, signal=signal_receipt))
    return dict(stage='A', status='COMPLETE_MANDATORY_STOP', signals=rows, signal_count=16,
        new_native_compile=0, new_sampling=0, new_full_H_reference=0, new_outer_diagnostic_pairs=7,
        saved_outer_diagnostic_pairs_reused=2, formal_accuracy_certified=False,
        global_optimum_claim=False, cost_acquisition_auto_started=False)
