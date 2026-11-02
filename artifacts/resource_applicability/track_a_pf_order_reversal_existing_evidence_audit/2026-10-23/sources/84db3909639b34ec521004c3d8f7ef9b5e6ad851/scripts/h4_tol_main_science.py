"""Grant-only rank10 adapter over the frozen H4 forward/native and A kernel.

No numerical modules or molecular inputs are accessed at import. Phase0 tests
replace these interfaces with artificial objects. The real entrypoint gates first.
"""
from copy import copy, deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import math
import sys

import h4_tol_main_contract as c


def setup(contract, cells, output):
    config = c.read(contract['native_config'])
    for directory in reversed(contract['import_paths']): sys.path.insert(0, directory)
    import h4_native_science as h
    from h4_tol_only_input import load
    m = h.provider()
    plan = h.accounting_plan(); plan.update(T=.8, actual_rank=10, cells=cells, epsilons=c.EPS,
        kind=c.SCOPE, df_policy=dict(kind='SAVED_TOL_ONLY_1E-8_RANK10', snapshot_sha256=c.BINDING['snapshot']),
        input_policy='EXISTING_F_RANK10_BYTES_NO_REGENERATION')
    plan['numerical']['sensitivity_factors'] = [1.]
    plan['matrix_free']['num_threads'] = 1
    caps = plan['caps_proposed']
    caps['tail_matvecs_corrected_and_raw']['B2'] = max([2*x['R']*(x['K']+1) for x in cells if x['R']] + [0])
    coverage = {('one' if i == 0 else str(i-1), t) for x in cells
                for i, t in m.primitive_time_schedule(x, T=.8)['unique_primitive_times']}
    count = 3*len(coverage)
    c.require(count <= c.budgets()[contract['stage']]['primitive_probe_maximum'], 'REGISTERED_PRIMITIVE_BUDGET')
    plan['coverage']['primitive_actions'] = caps['primitive'] = count
    # Fresh call counters must see the new grid, rather than the historical panel.
    caps['reference_matvec'] = 0
    caps['deterministic_actions_per_cell'] = max(x['q'] *
        len(m.primitive_time_schedule(x, T=.8)['ordinary_one_outer_step']) *
        (2 if x['R'] else 1) for x in cells)
    port = m.MatchedPort(Path(config['provider_root']), dict(kind=c.SCOPE, plan=plan), c.old_writer(output), h.Progress())
    port.ham, port.sector, full, state, metadata = load(config)
    c.require((port.ham.n_qubits, len(port.ham.lambdas), port.sector.dimension) == (8, 10, 36), 'RANK10_LAYOUT')
    c.require(metadata['hamiltonian_hash'] == c.BINDING['hamiltonian']
              and metadata['state_hash'] == c.BINDING['state'], 'RANK10_SAVED_STATE')
    port.qindices, port.qstate = m.checked_basis_bridge(port.ham, port.sector, full, state, 1e-12)
    port.state = state.copy(); port.saved_state = state.copy()
    ref = c.read(contract['saved_reference'])
    c.require(ref['metadata']['hamiltonian_hash'] == c.BINDING['hamiltonian']
              and ref['metadata']['state_hash'] == c.BINDING['state'] and ref['T'] == .8, 'SAVED_RANK10_REFERENCE')
    port.reference = complex(ref['reference']['real'], ref['reference']['imag'])
    port.reference_error = ref['error']
    port.spectra = {}; port.native_actions = {}; port.tails = {}; port.oracle_tails = {}; port.truncated_references = {}
    cache = {}
    for x in cells:
        key = (x['method'] == 'B0', x['prefix'])
        if key not in cache: cache[key] = (m._prepare_discard if key[0] else m._prepare)(port.ham, key[1])
        port.preps[x['id']] = cache[key]
    c.save(output/'preparations.json', dict(binding=c.BINDING, new_reference=0,
        SCF_DF_state_solver=0, reconstructions=len(cache),
        rows={cid: dict(preparation_hash=p.preparation_hash, hamiltonian_hash=p.hamiltonian_hash)
              for cid, p in port.preps.items()}))
    for prep in cache.values(): port.actions(prep); port.actions(prep, oracle=True)
    port.primitives()
    return port, config, h, m


def evaluate(port, task, directory):
    # Same H4 native evaluate path, with signed/factor1 accounting only.
    local = copy(port); local.plan = deepcopy(port.plan); local.cells = [task['cell']]
    local.plan['cells'] = local.cells; local.calls = deepcopy(port.calls)
    local.writer = c.old_writer(directory); local.completed = 0
    local.tails = {}; local.truncated_references = {}
    c.save(directory/'signal_reserved.json', dict(cell=task['cell'], nonrefundable=True))
    local.correctness()
    return c.read(directory/'signal_summary.json')['signals'][0]


def overlay(base, port, m):
    # Reuse the same exact tail-identity and conservative overlay primitives as
    # h4_native_science.overlay; omit its unrequested noncancellation accounting.
    import h6_native_b0i_bridge_v1 as bridge
    exact = sum((bridge.exact_identity(g, w) for g, w in zip(
        port.ham.g_matrices[base['cell']['prefix']:], port.ham.lambdas[base['cell']['prefix']:])), Fraction())
    C = float(exact); scalar = float(port.ham.constant+C)
    error = bridge.upper_float(abs(Fraction(C)-exact)+abs(Fraction(scalar)-Fraction(port.ham.constant)-Fraction(C)))
    phase_error = bridge.upper_float(abs(Fraction(-.8*C)+Fraction(.8)*Fraction(C)))
    effective = deepcopy(base); effective['error_components'] = bridge.conservative_saved_error_components(effective)
    out = deepcopy(base)
    out.update(bridge.saved_signal_overlay(effective, C=C, C_error=error,
        rotation_roundoff=max(1e-12, 64*m.np.finfo(float).eps*(1+abs(.8*C))), phase_roundoff=phase_error))
    out['cell'] = dict(base['cell'], method='B0I', id=base['cell']['id'].replace('B0_', 'B0I_', 1))
    values = {k: complex(v['real'], v['imag']) for k, v in out['signals'].items()}
    out['error_decomposition'] = m.differences(values, complex(out['reference']['real'], out['reference']['imag']), 'B0')
    out['scalar_receipt'] = dict(C=C, C_error=error, scalar=scalar,
        exact_numerator=str(exact.numerator), exact_denominator=str(exact.denominator),
        policy='EXACT_STORED_FULL_FOCK_TAIL_IDENTITY', certified_input_error=False)
    out['precision_rows'] = m.precision_rows(out, port.plan)
    return out


def batch(contract, tasks, callback, directory):
    if not tasks: return []
    import h4_native_science as h
    grant = dict(worker_cpus=contract['worker_cpus'], font_seed=contract['font_seed'])
    rows, receipt = h.batch(tasks, callback, grant, directory)
    c.save(directory/'accepted_native_receipt.json', receipt)
    return rows


def execute(contract, output):
    stage = contract['stage']
    c.require(stage in ('D', 'A'), 'P_AND_S_STATS_REQUIRE_NEW_FINAL_CONTRACT')
    saved = c.read(contract['FP_evidence'])
    saved_by_id = {r['cell']['id']: r for r in saved['signals']}
    registered = c.registrations()[stage]
    raw = [x for x in registered if x['method'] != 'B0I']
    port, config, h, m = setup(contract, raw, output)
    expected_preps = saved['preparation_identities']
    for cid, prep in port.preps.items():
        if cid in expected_preps:
            c.require(prep.preparation_hash == expected_preps[cid], 'SAVED_FP_PREPARATION_CHANGED')
    missing = [dict(id=x['id'], cell=x) for x in raw if x['id'] not in saved_by_id]
    limits = c.budgets()[stage]
    c.require(len(missing) == limits['new_base_signal'], 'EXACT_NEW_SIGNAL_SCOPE')
    acquired = batch(contract, missing, lambda t, d: evaluate(port, t, d), output/'signal_workers')
    records = {r['cell']['id']: r for r in acquired}
    records.update({x['id']: deepcopy(saved_by_id[x['id']]) for x in raw if x['id'] in saved_by_id})
    if stage == 'D':
        for x in registered:
            if x['method'] != 'B0I': continue
            if x['id'] in saved_by_id: records[x['id']] = deepcopy(saved_by_id[x['id']])
            else:
                c.save(output/(x['id']+'_overlay_reserved.json'), dict(cell=x, nonrefundable=True))
                records[x['id']] = overlay(records[x['id'].replace('B0I_', 'B0_', 1)], port, m)
    ordered = [records[x['id']] for x in registered]
    c.save(output/'signals.json', dict(signals=ordered, binding=c.BINDING,
        reused_records=[x['id'] for x in registered if x['id'] in saved_by_id]))
    acquisition = c.selected(ordered, registered)
    pairs = {}
    if stage == 'D':
        for row in saved['measurements']:
            if row['cell_id'] in c.FP_DETERMINISTIC:
                pairs[row['cell_id']] = [dict(axes={a['axis']: dict(a['metrics'], compiler_hash=a['compiler_hash'],
                    fingerprint=a['fingerprint']) for a in row['rows']})]
        tasks = [dict(id=x['id'], cell=x, replica=0, expected=records[x['id']])
                 for x in acquisition if x['id'] not in pairs]
        c.require(len(tasks) <= limits['native_evolution_maximum'], 'D_COMPILE_CALL_BUDGET')
        new = batch(contract, tasks, lambda t, d: h_compile(port, config, t, d), output/'compile_workers')
        for row in new:
            pairs[row['cell_id']] = [dict(axes={a['axis']: dict(a['metrics'], compiler_hash=a['compiler_hash'],
                fingerprint=a['fingerprint']) for a in row['rows']})]
        c.require(len(new) == len(tasks) and {r['cell_id'] for r in new}
            == {t['cell']['id'] for t in tasks}, 'COMPLETE_D_NATIVE_RESULTS')
        accounting = c.account_stage(ordered, pairs)
    else:
        import importlib
        import pr_cost_native as native
        from h4_dp_science import compiler
        context = SimpleNamespace(m=m, native=importlib.import_module(
            'trottertracks.resource_applicability.ax2b_native_df_v5'), qstate=port.qstate,
            preparations={x['prefix']: port.preps[x['id']] for x in raw},
            compiler=compiler(m, config), binding=c.BINDING)
        registry = c.read(contract['seed_registry'])
        eligible = {x['id'] for x in acquisition if x['id'] != c.FP_PR}
        tasks = []
        for task in registry['tasks']:
            if task['cell']['id'] not in eligible: continue
            cid = task['cell']['id']; record = records[cid]
            tasks.append(dict(task, preparation_hash=context.preparations[task['cell']['prefix']].preparation_hash,
                saved_log_B=record['log_B'], log_B_tolerance=record['allowance']['log_B_margin'], control_tolerance=1e-9))
        first = [t for t in tasks if t['replica'] == 0]
        rest = [t for t in tasks if t['replica'] != 0]
        c.require(len(tasks) == 8*len(eligible) <= limits['trajectory_maximum'], 'A_COMPILE_CALL_BUDGET')
        def compile_a(task, directory):
            row = native.compile_task(context, task, directory)
            row['stratum'] = c.old.event_stratum(c.read(directory/'trajectory.json')['events'])
            return row
        new = batch(contract, first, compile_a, output/'first_control_workers')
        new += batch(contract, rest, compile_a, output/'remaining_workers')
        c.require(len(new) == len(tasks), 'COMPLETE_A_NATIVE_RESULTS')
        for row in new: pairs.setdefault(row['cell_id'], []).append(row)
        accounting = c.account_stage(ordered, pairs, saved_FP_rows=saved['saved_rows'], d_result=contract['accepted_D'])
        accounting['new_cost_strata'] = {cid: {s: sum(row['stratum'] == s for row in values)
            for s in ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')} for cid, values in pairs.items()}
    c.save(output/'paired_measurements.json', dict(rows=new, old_FP_cohort_not_pooled=True))
    accounting.update(stage=stage, SOURCE=contract['SOURCE'], status='COMPLETE_MANDATORY_STOP',
        new_SCF_DF_state_reference=0, endpoint_accounting_pass=1, next_stage_authorized=False,
        actual_new_base_signals=len(missing), actual_new_compiled_trajectories=len(new),
        actual_new_wrapper_compiles=2*len(new), call_budget=limits)
    return accounting


def h_compile(port, config, task, directory):
    from h4_followup_science import compile_task
    return compile_task(port, config, task, directory)
