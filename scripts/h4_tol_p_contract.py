"""Independent DF10 P metadata, complete fixed-n binding and pre-import gate."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h4_tol_main_contract as main
import h4_tol_s_stats_contract as stats
c = main.old
ROOT = main.ROOT
SCOPE = 'H4_TOL10_INDEPENDENT_P_20261019'
STATS_COMMIT = 'd728d088c010089c6b672eef0521261791b2d209'
N = {'B2_p3_q1_2nd_r4_K2': 1108, 'B2_p2_q8_2nd_r2_K4': 78}
NAMESPACE = main.SCOPE+'_CONFIRMATION'
COHORT = 'INDEPENDENT_H4_CONFIRMATION'  # unchanged kernel label; scope/binding distinguish DF10
PENDING = ['SEPARATE_EXPLICIT_P_SCIENTIFIC_APPROVAL', 'FRESH_RESOURCE_ADMISSION',
           'REFRESH_KNOWN_SEED_INVENTORY_WITHOUT_REDRAW']


def plan():
    cells = {x['id']: x for x in main.registrations()['A']}
    return dict(scope=SCOPE, stage='P', binding=deepcopy(main.BINDING),
        fixed_D_commit=stats.D_COMMIT, fixed_A_commit=stats.A_COMMIT, fixed_S_STATS_commit=STATS_COMMIT,
        targets=[dict(epsilon=e, cell=deepcopy(cells[cid]), n=N[cid], PF_cell_id=pf)
                 for e, cid, pf in stats.TARGETS],
        n_fixed=deepcopy(N), cohort=COHORT, namespace=NAMESPACE,
        calls=dict(trajectory=1186, native_evolution=1186, wrapper_compile=2372, occurrence=1732,
            RTE_event=5680, control_probe=8, independent_event_oracle=2,
            preparation_reconstruction=2, endpoint_accounting_pass=1, endpoint_comparisons=2,
            seed_slots=1810, signal=0, PF_compile=0, SCF_DF_state_reference=0,
            bootstrap=0, shot_recalculation=0, old_sample_aggregation=0),
        decision='sign(saved PF_G - new paired PR mean) equals saved S_STATS direction; zero INCONCLUSIVE',
        family_alpha=.05, power_design=.8, SD_design_inflation=4, relative_gap_design=.02,
        n_purpose='finite-sample point-direction reproduction planning only',
        population_mean_certified=False, actual_power_certified=False, CI_or_p_values=False,
        rare_policy='Unobserved S2/S4 costs and variances remain unknown, never zero filled',
        law='original unconditional finite RTE, direct_master_to_step_seeds_v1, unmodified weights',
        no_pooling=True, no_reselection=True, no_optional_stopping=True,
        one_shot=True, automatic_retry=False, next_stage_authorized=False, mandatory_STOP=True,
        parallel=dict(agent_imposed_CPU_RAM_worker_wall_caps=None, worker_threads=1,
            allocation='fresh physical CPUs within existing affinity/quota and effective RAM; account for known jobs',
            first_controls_before_remaining=True, fork_COW=True, preparation_cache=True,
            paired_evolution_shared=True, fingerprint_reuse=True, PSI_diagnostic_only=True,
            unrelated_worktree_changes_are_gates=False))


def closure():
    pending = ['scripts/'+name+'.py' for name in ('h4_tol_p_contract', 'h4_tol_p_science',
        'run_h4_tol_p', 'seal_h4_tol_p', 'audit_h4_tol_p_completed', 'h6_dba_science')]
    pending += ['tests/test_h4_tol_p.py']
    out = {}
    while pending:
        name = pending.pop()
        if name in out: continue
        path = ROOT/name; out[name] = c.sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for mod in mods:
                local = 'scripts/'+mod.split('.')[0]+'.py'
                if (ROOT/local).is_file(): pending.append(local)
    return out


def endpoints(a, d, saved_stats):
    selected = stats.selected_summary(a, d)  # copies saved rows only; no SD/n/power/rank calculation
    c.require(saved_stats['recommended']['n_fixed'] == N, 'ACCEPTED_STATS_N')
    out = []
    for e in selected['endpoints']:
        cid = e['cell']['id']
        direction = next(x for x in saved_stats['reference_directions'] if x['cell_id'] == cid)
        c.require(direction['epsilon'] == e['epsilon'] and direction['PF_cell_id'] == e['PF']['cell_id']
            and direction['PF_G_exact'] == e['PF']['G_RZ_exact']
            and direction['expected_direction'] == e['saved_point_direction'], 'FIXED_SAVED_DIRECTION')
        out.append(dict(epsilon=e['epsilon'], cell=deepcopy(e['cell']), n=N[cid],
            saved_shots=deepcopy(e['PR']['shots']), PF_id=e['PF']['cell_id'],
            PF_G_exact=e['PF']['G_RZ_exact'], expected_direction=direction['expected_direction'],
            exploratory_n=8, exploratory_mean_exact=e['PR']['G_RZ_exact'],
            saved_rare_counts=deepcopy(e['saved_rare_counts']), unseen_cost=None, unseen_variance=None))
    return out


def registry(source, eps, preparations, signals):
    occupied = set(); tasks = []
    for ep in eps:
        cell = ep['cell']; cid = cell['id']; signal = signals[cid]
        for replica in range(N[cid]):
            seed = main.seed(NAMESPACE, source, c.digest(plan()), main.BINDING, cid, replica)
            hierarchy = c.reserve_seed(occupied, seed, cell['q'])
            tasks.append(dict(id=cid+'_'+str(replica), cell=deepcopy(cell), replica=replica, seed=seed,
                step_seeds=hierarchy if cell['q'] == 1 else hierarchy[1:],
                preparation_hash=preparations[cid]['preparation_hash'], saved_log_B=signal['log_B'],
                log_B_tolerance=signal['allowance']['log_B_margin'], cohort=COHORT))
    return dict(SOURCE=source, plan_digest=c.digest(plan()), namespace=NAMESPACE,
        tasks=tasks, values=sorted(occupied), redraw_on_collision=False, actual_RTE_event_sampling=0)


def check_registry(spec):
    reg = c.read(spec['seed_registry']); eps = c.read(spec['endpoints'])
    signals = c.read(spec['saved_signals']); preps = c.read(spec['saved_preparations'])['rows']
    expected = registry(spec['SOURCE'], eps, preps, {x['cell']['id']: x for x in signals['signals']})
    c.require(reg == expected and len(reg['tasks']) == 1186 and len(reg['values']) == 1810, 'EXACT_P_REGISTRY')
    old = c.read(spec['old_seed_exclusions']); receipt = c.read(spec['seed_inventory_receipt'])
    c.require(not set(reg['values']) & set(old['values']), 'P_SEED_COLLISION')
    c.require(receipt['status'] == 'COMPLETE_COLLISION_ZERO' and receipt['all_registered_scopes_covered']
        and receipt['new_seed_values_digest'] == c.digest(reg['values'])
        and receipt['old_explicit_exclusions_sha256'] == c.sha(spec['old_seed_exclusions'])
        and len(receipt['geometry_checks']) == 6
        and sum(x['old_seed_slots_checked'] for x in receipt['geometry_checks']) == 7737216
        and all(not x['collisions'] for x in receipt['geometry_checks']), 'FULL_SEED_INVENTORY')
    return reg


def verify_bytes(spec):
    c.require(spec['scope'] == SCOPE and spec['stage'] == 'P' and spec['plan'] == plan(), 'FIXED_P_PLAN')
    c.require(spec['source_hashes'] == closure(), 'COMPLETE_P_SOURCE_CLOSURE')
    for name, h in spec['source_hashes'].items():
        blob = subprocess.check_output(['git', 'show', spec['SOURCE']+':'+name], cwd=ROOT)
        c.require(hashlib.sha256(blob).hexdigest() == h, 'GIT_SOURCE_BYTES')
    for row in spec['inputs']:
        c.require(c.sha(row['path']) == row['sha256'], 'P_INPUT_BYTES:'+row['role'])
        if row.get('commit'):
            raw = subprocess.check_output(['git', 'show', row['commit']+':'+row['repo_path']], cwd=ROOT)
            c.require(hashlib.sha256(raw).hexdigest() == row['sha256'], 'ACCEPTED_COMMIT_BYTES')
    for group in ('runtime_hashes', 'provider_hashes'):
        for path, h in spec[group].items(): c.require(c.sha(path) == h, group+'_CHANGED')
    main.verify_provider_tree(spec)
    for row in c.read(spec['seed_origins']):
        # Owned cutoff snapshots are bound; live parallel registries may grow normally.
        c.require(c.sha(row['path']) == row['sha256'], 'OWNED_SEED_METADATA_SNAPSHOT_BYTES')
    c.require(c.sha(sys.executable) == spec['python_sha256'], 'P_PYTHON_BYTES')
    roles = {r['role']: r['path'] for r in spec['inputs']}
    fixed = endpoints(c.read(roles['A_result']), c.read(roles['D_result']), c.read(roles['S_STATS_result']))
    c.require(c.read(spec['endpoints']) == fixed, 'VERBATIM_FIXED_ENDPOINTS')
    bound = {r['path'] for r in spec['inputs']}
    c.require({spec[k] for k in ('native_config', 'endpoints', 'seed_registry', 'old_seed_exclusions',
        'seed_inventory_receipt', 'saved_signals', 'saved_preparations', 'seed_origins')} | {spec['font_seed']['path']} <= bound,
        'ALL_USED_METADATA_BOUND')
    config = c.read(spec['native_config'])
    c.require(config['binding'] == main.BINDING and set(config['mapping']) == {'tol_snapshot', 'tol_receipt'}
        and set(config['mapping'].values()) <= bound, 'RANK10_ONLY_CONFIG')
    check_registry(spec)
    return fixed


def invalid_grant(spec):
    return dict(scope=SCOPE, stage='P', SOURCE=spec['SOURCE'], contract_digest=c.digest(spec),
        approved_by_user=False, runtime_authorization=False, one_shot=True, automatic_retry=False,
        next_stage_authorized=False, approval_receipt=None, approval_receipt_sha256=None, launch_command=None)


def authorize(spec, grant):
    # Deny first: no molecular inputs, imports, output, reservation or subprocess.
    c.require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'SEPARATE_EXPLICIT_P_SCIENTIFIC_APPROVAL_REQUIRED')
    c.require(grant.get('scope') == SCOPE and grant.get('stage') == 'P', 'P_ONLY_GRANT')
    c.require(spec['status'] == 'SEALED_FOR_EXECUTION' and spec['pending'] == [], 'P_PENDING_ACTIVATION')
    c.require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
        and grant.get('next_stage_authorized') is False, 'P_ONE_SHOT_STOP')
    c.require(grant['SOURCE'] == spec['SOURCE'] and grant['contract_digest'] == c.digest(spec), 'P_GRANT_BINDING')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'P_APPROVAL_BYTES')
    approval = c.read(grant['approval_receipt'])
    c.require(approval['kind'] == 'P_SCIENTIFIC_EXECUTION' and approval['scope'] == SCOPE
        and approval['SOURCE'] == spec['SOURCE'] and approval['plan_digest'] == c.digest(plan())
        and approval['explicit_user_approval'] is True, 'SEAL_APPROVAL_IS_NOT_EXECUTION_APPROVAL')
    output, lock = Path(spec['output']), Path(spec['lock'])
    c.require(output.parent == lock.parent and not output.exists() and not lock.exists(), 'P_UNUSED_RUN')
    fresh = spec['fresh_resources']
    c.require(fresh['admitted'] is True and c.sha(fresh['receipt']) == fresh['sha256'], 'P_FRESH_RESOURCES')
    cpus = spec['worker_cpus']
    c.require(cpus and all(type(x) is int for x in cpus) and len(cpus) == len(set(cpus))
        and set(cpus) <= os.sched_getaffinity(0), 'P_WORKER_ASSIGNMENT')
    refresh = c.read(spec['seed_refresh_receipt'])
    c.require(c.sha(spec['seed_refresh_receipt']) == spec['seed_refresh_sha256']
        and refresh['status'] == 'COMPLETE_COLLISION_ZERO' and refresh['new_explicit_collisions'] == 0
        and not refresh['seed_redrawn'] and refresh['sealed_registry_sha256'] == c.sha(spec['seed_registry']),
        'P_ACTIVATION_SEED_REFRESH')
    return verify_bytes(spec)
