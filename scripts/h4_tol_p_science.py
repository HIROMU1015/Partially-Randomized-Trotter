"""Grant-only DF10 P adapter; unchanged sampler/control/paired compile kernels."""
from collections import Counter
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import h4_tol_p_contract as p
c = p.c


def setup(spec, output):
    for path in reversed(spec['import_paths']): sys.path.insert(0, path)
    from h4_tol_only_input import load
    from h4_dp_science import compiler
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    m = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    native = importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    frozen = importlib.import_module('pr_cost_native')
    config = c.read(spec['native_config'])
    ham, sector, full, state, md = load(config)
    c.require((ham.n_qubits, len(ham.lambdas), sector.dimension) == (8, 10, 36)
        and md['hamiltonian_hash'] == p.main.BINDING['hamiltonian']
        and md['state_hash'] == p.main.BINDING['state'], 'DF10_SAVED_CONTEXT')
    _, qstate = m.checked_basis_bridge(ham, sector, full, state, 1e-12)
    prefixes = sorted({ep['cell']['prefix'] for ep in c.read(spec['endpoints'])})
    c.require(prefixes == [2, 3], 'TWO_FIXED_PREPARATIONS')
    c.save(output/'preparation_reserved.json', dict(prefixes=prefixes, count=2, nonrefundable=True,
        SCF_DF_state_reference=0, signal=0, primitive_probe=0))
    preparations = {prefix: m._prepare(ham, prefix) for prefix in prefixes}
    saved = c.read(spec['saved_preparations'])['rows']
    for ep in c.read(spec['endpoints']):
        prep = preparations[ep['cell']['prefix']]
        c.require(prep.preparation_hash == saved[ep['cell']['id']]['preparation_hash']
            and prep.hamiltonian_hash == p.main.BINDING['hamiltonian'], 'SAVED_A_PREPARATION_IDENTITY')
    c.save(output/'preparations.json', dict(reconstructions=2, SCF_DF_state_reference=0,
        rows={str(k): dict(preparation_hash=v.preparation_hash, hamiltonian_hash=v.hamiltonian_hash)
              for k, v in preparations.items()}))
    return SimpleNamespace(m=m, native=native, frozen=frozen, qstate=qstate, preparations=preparations,
        compiler=compiler(m, config), binding=p.main.BINDING, fingerprint_reuse=use_fingerprint_reuse)


def validate_pairs(rows, tasks):
    by_task = {(t['cell']['id'], t['replica']): t for t in tasks}
    c.require(len(rows) == len(tasks) and len({(r['cell_id'], r['replica']) for r in rows}) == len(rows)
        and {(r['cell_id'], r['replica']) for r in rows} == set(by_task), 'EXACT_FIXED_N_NO_POOLING')
    calls = Counter()
    for row in rows:
        task = by_task[(row['cell_id'], row['replica'])]
        c.require(row['seed'] == task['seed'] and row['cohort'] == p.COHORT, 'ONLY_NEW_P_COHORT')
        c.check_pair(row, p.main.BINDING['compiler'])
        expected = dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
            native_evolution=1, wrapper_compile=2, control_probe=4 if task['replica'] == 0 else 0,
            independent_event_oracle=1 if task['replica'] == 0 else 0)
        c.require(row['calls'] == expected, 'EXACT_P_TASK_CALLS')
        calls.update(row['calls'])
    return dict(calls)


def account(rows, eps, tasks):
    calls = validate_pairs(rows, tasks)
    comparisons = []
    for ep in eps:
        selected = sorted([r for r in rows if r['cell_id'] == ep['cell']['id']], key=lambda x: x['replica'])
        c.require(len(selected) == ep['n'], 'COMPLETE_ENDPOINT_N')
        value = c.evaluate_pairs(selected, ep['saved_shots'], ep['PF_G_exact'], ep['expected_direction'])
        value['n'] = len(selected)
        strata = Counter(r['stratum'] for r in selected)
        comparisons.append(dict(ep, result=value, strata={s: strata[s] for s in
            ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')}, unseen_stratum_cost=None,
            unseen_stratum_variance=None, population_mean_certified=False))
    return dict(scope=p.SCOPE, stage='P', binding=p.main.BINDING, comparisons=comparisons,
        actual_worker_calls=calls, endpoint_accounting_pass=1, endpoint_comparisons=2,
        old_samples_pooled=False, population_mean_certified=False, actual_power_certified=False,
        candidate_reselected=False, shots_recalculated=False, unseen_strata_zero_filled=False,
        next_stage_authorized=False)


def execute(spec, output):
    from h4_dp_science import confirmation_task
    from h4_native_science import batch
    context = setup(spec, output)
    tasks = c.read(spec['seed_registry'])['tasks']
    def run(task, directory): return confirmation_task(context, task, directory)
    first = [t for t in tasks if t['replica'] == 0]; rest = [t for t in tasks if t['replica'] != 0]
    rows, receipt = batch(first, run, spec, output/'first_control_workers')
    c.save(output/'first_control_workers/accepted_native_receipt.json', receipt)
    # Complete control receipts must exist before launching any remaining replica.
    for task in first:
        control = c.read(output/'first_control_workers'/task['id']/'control.json')
        c.require(control['status'] == 'PASS' and control['control_probes'] == 4
            and control['independent_event_oracles'] == 1, 'FIRST_CONTROLS_BEFORE_REMAINING')
    more, receipt = batch(rest, run, spec, output/'remaining_workers')
    c.save(output/'remaining_workers/accepted_native_receipt.json', receipt)
    rows += more
    c.save(output/'paired_measurements.json', dict(rows=rows, old_cohorts_not_pooled=True))
    # Nonrefundable reservation immediately before the sole new-cohort accounting pass.
    c.save(output/'endpoint_accounting_reserved.json', dict(passes=1, comparisons=2, nonrefundable=True))
    return account(rows, c.read(spec['endpoints']), tasks)
