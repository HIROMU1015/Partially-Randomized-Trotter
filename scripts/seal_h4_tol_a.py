"""A50 metadata seal only. No science, valid grant, or launch entry point."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import h4_tol_main_contract as c
import h4_p_seed_inventory as seeds
import seal_h4_tol_d as d

BASE = '406f70570323366da185355550a3900388717fbb'
SOURCE = d.SCIENCE_SOURCE
ROOT = c.ROOT
PROJECTS = ROOT.parents[1]
PRIVATE = d.OLD_PRIVATE/'A_seal01_20261019'
PUBLIC = ROOT/'artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19'
D_PUBLIC = ROOT/'artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19'
P_PRIVATE = ROOT.parent/'h4-p-seal-20261019/.h4_p_seal_private'
H6 = PROJECTS/'h6-pr-confirmation-preparation/20261019/final-sealed02'
IMMUTABLE = tuple(x for x in d.IMMUTABLE if x != 'inputs')
PENDING = ['SEPARATE_EXPLICIT_A_SCIENTIFIC_APPROVAL', 'FRESH_RESOURCE_ADMISSION_AT_ACTIVATION',
           'REFRESH_NEWLY_REGISTERED_SEED_COLLISIONS_WITHOUT_REDRAW']


def check_scope(contract, draft):
    c.require(contract['stage'] == draft['stage'] == 'A', 'A_ONLY')
    c.require(contract['SOURCE'] == SOURCE and contract['plan'] == c.plan('A'), 'FIXED_A_SOURCE_PLAN')
    c.same_binding(contract['binding'])
    c.require(all(contract[k] == draft[k] for k in IMMUTABLE), 'UNCHANGED_PHASE0_SCIENCE')
    c.require(contract['inputs'][:len(draft['inputs'])] == draft['inputs'], 'ORIGINAL_INPUTS_RETAINED')
    c.require(all(contract.get(k) is False for k in ('approved_by_user', 'runtime_authorization',
        'automatic_retry', 'next_stage_authorized')) and contract['actual_scientific_calls'] == 0,
        'NO_SCIENTIFIC_AUTHORIZATION_OR_RETRY')
    c.require(contract['worker_cpus'] == [] and not contract['fresh_resources']['admitted'], 'NO_RESOURCE_ACTIVATION')
    c.require(not Path(contract['output']).parent.exists() and not Path(contract['lock']).exists(), 'UNUSED_A_NAMESPACE')


class Store:
    def __init__(self, private):
        self.directory = private/'seed_metadata_store'
        self.directory.mkdir()
        self.rows = []

    def copy(self, role, path, expected=None):
        path = Path(path).resolve(); h = c.sha(path)
        c.require(expected is None or h == expected, 'REGISTERED_METADATA_CHANGED:'+role)
        target = self.directory/h
        if not target.exists(): target.write_bytes(path.read_bytes())
        c.require(c.sha(target) == h, 'METADATA_SNAPSHOT_BYTES')
        self.rows.append(dict(role=role, origin=str(path), path=str(target), sha256=h))
        return target


def identifiers(value, mode):
    """Only identifier arithmetic, never the RTE distribution or scientific data."""
    if mode in ('values', 'all_master_and_step_values'):
        out = value[mode]
    elif mode == 'flat_registry':
        out = [s for task in value['tasks'].values() for s in task.values()]
    elif mode in ('tasks', 'fixed_delta', 'H6_CONDITIONAL', 'H8_CONDITIONAL'):
        out = list(value.get('excluded', []))
        tasks = value['tasks'] if mode != 'fixed_delta' else [x['task'] for x in value['plan']['registrations']]
        for task in tasks:
            master = task.get('seed')
            if master is None: continue
            cell = task.get('cell') or task['group']['cell']
            out.extend(c.old.seed_hierarchy(master, cell['q']))
            # Conservative additional exclusion of conditional order/component streams.
            if mode in ('H6_CONDITIONAL', 'H8_CONDITIONAL'):
                domain = 'H6_DBA' if mode == 'H6_CONDITIONAL' else 'H8_AB'
                out += [int(c.digest([domain+'_ORDER', master])[:16], 16)]
                out += [int(c.digest([domain+'_COMPONENT', master, i])[:16], 16) for i in range(cell['q'])]
            out.extend(task.get('step_seeds', []))
    else:
        raise ValueError('UNKNOWN_SEED_METADATA_MODE')
    c.require(all(type(x) is int and 0 <= x < 2**64 for x in out), 'UINT64_SEED_IDENTIFIERS')
    return set(out)


def inventory(store):
    excluded = set(); coverage = []

    def add(role, path, mode, expected=None):
        snapshot = store.copy(role, path, expected)
        values = identifiers(c.read(snapshot), mode)
        excluded.update(values)
        coverage.append(dict(scope=role, seed_values=len(values), sha256=c.sha(snapshot), status='COVERED'))

    old_inventory = c.read(store.copy('H4_P_inventory_inputs', P_PRIVATE/'inventory_inputs_PRIVATE.json'))
    # Retain provenance of the earlier exhaustive explicit/geometry inventory.
    for row in old_inventory['inputs']:
        if any(s in row['role'] for s in ('registry', 'exclusion', 'geometry', 'seed')):
            store.copy('H4_prior:'+row['role'], row['path'], row['sha256'])
            c.require(c.sha(row['origin']) == row['sha256'], 'H4_PRIOR_SEED_ORIGIN_CHANGED')
    add('H4_LEGACY_A_CONDITIONAL_T_FP_AND_GEOMETRY_MASTERS', P_PRIVATE/'old_seed_exclusions_PRIVATE.json', 'values')
    add('H4_RANK12_P_RESERVED_AND_USED', P_PRIVATE/'new_confirmation_seeds_PRIVATE.json', 'all_master_and_step_values')
    old_p = c.read(store.copy('H4_completed_P_contract', P_PRIVATE/'P_run01_20261019/spec_SEALED_PRIVATE.json'))
    for path in (P_PRIVATE/'old_seed_exclusions_PRIVATE.json', P_PRIVATE/'new_confirmation_seeds_PRIVATE.json',
                 P_PRIVATE/'geometry_seed_cases_PRIVATE.json'):
        # These are the exact seed metadata bound in the completed P provenance.
        if path.name != 'new_confirmation_seeds_PRIVATE.json':
            expected = next(x['sha256'] for x in old_p['inputs'] if x['path'] == str(path))
            c.require(c.sha(path) == expected, 'COMPLETED_P_INVENTORY_BINDING')
    p_values = identifiers(dict(tasks=old_p['tasks']), 'tasks')
    c.require(p_values <= excluded and len(old_p['tasks']) == 4185, 'ALL_COMPLETED_H4_P_SEEDS_EXCLUDED')
    old_a = c.read(store.copy('H4_prior_A', ROOT.parent/'h4-dp-phase0-20261019/.h4_dp_phase0_private/existing_A_seed_registry.json'))
    c.require(identifiers(old_a, 'tasks') <= excluded, 'ALL_PRIOR_H4_A_SEEDS_EXCLUDED')

    h6_inventory = c.read(store.copy('H6_campaign_inventory', H6/'seed_inventory_PRIVATE.json'))
    c.require(h6_inventory['schema'] == 'H6_FINAL_SEED_INVENTORY_V1'
        and h6_inventory['known_registered_campaigns_complete'] is True
        and set(h6_inventory['campaigns']) == {'H6_LEGACY_AND_A128', 'H6_FIXED_DELTA_RESERVATIONS',
            'H6_CONDITIONAL', 'HIGHER_PF_LEGACY_AND_ACQUISITION', 'H6_HIGHER_PF_CONFIRMATION_RESERVED',
            'H6_HIGHER_PF_CONFIRMATION_USED', 'H8_NEW_A_AND_CONFIRMATION',
            'H4_RECORDED_OLD_SEEDS', 'H4_P_CONFIRMATION'}, 'COMPLETE_KNOWN_PARALLEL_INVENTORY')
    for scope, rows in h6_inventory['campaigns'].items():
        for i, row in enumerate(rows):
            add('PARALLEL:'+scope+':'+str(i), row['path'], row['seed_mode'], row['sha256'])
    # Includes reserved and active confirmation identifiers; never observes raw costs.
    add('PARALLEL_H6_FINAL_CONFIRMATION_152', H6/'seed_registry_PRIVATE.json', 'tasks')
    add('PARALLEL_H6_CONDITIONAL_DERIVED', PROJECTS/'h6-dba-validation/20261018/private/B_attempt02/output/seed_registry.json', 'H6_CONDITIONAL')
    add('PARALLEL_H8_CONDITIONAL_DERIVED', PROJECTS/'h8-ab-validation/20261018/private/B_run01/output/seed_registry.json', 'H8_CONDITIONAL')
    batch = PROJECTS/'track-a-pr-cost-batch-seal/20261018/private/sealed03/contracts'
    for system in ('H6', 'H8'):
        seal = c.read(store.copy(system+'_original_A_seal', batch/(system+'_A_COST')/'technical_seal.json'))
        payload = c.read(store.copy(system+'_original_A_payload', seal['payload']))
        registry = payload['new_seed_registry']
        add(system+'_legacy_exclusions', registry['old_seed_exclusions'], 'values')
        values = identifiers(registry, 'tasks'); excluded.update(values)
        coverage.append(dict(scope=system+'_original_A128', seed_values=len(values), trajectories=len(registry['tasks']), status='COVERED'))
    cases_path = store.copy('H4_geometry_all_registered_cases', P_PRIVATE/'geometry_seed_cases_PRIVATE.json')
    by_role = {x['role']:x['path'] for x in old_inventory['inputs']}
    reconstructed = seeds.geometry_cases(c.read(by_role['geometry_frozen_plan']),
        c.read(by_role['geometry_fixed_identity_projection']))
    c.require(c.read(cases_path) == reconstructed, 'EXACT_REGISTERED_GEOMETRY_CASES')
    return excluded, coverage, c.read(cases_path), old_inventory['coverage']


def new_registry():
    plan = c.plan('A'); tasks = []; occupied = set()
    for cell in plan['cells']:
        if cell['id'] == c.FP_PR: continue
        for replica in range(8):
            master = c.seed(c.SCOPE+'_EXPLORATION', SOURCE, c.digest(plan), c.BINDING, cell['id'], replica)
            hierarchy = c.old.reserve_seed(occupied, master, cell['q'])
            tasks.append(dict(id=cell['id']+'_'+str(replica), cell=cell, replica=replica, seed=master,
                step_seeds=hierarchy if cell['q'] == 1 else hierarchy[1:], cohort='EXPLORATION_N8'))
    return dict(SOURCE=SOURCE, plan_digest=c.digest(plan), binding=c.BINDING, tasks=tasks,
        namespace=c.SCOPE+'_EXPLORATION', redraw_on_collision=False, actual_RTE_event_sampling=0), occupied


def preserve():
    old = c.read(d.OLD_PRIVATE/'preservation_manifest_PRIVATE.json')
    completed = c.read(d.PRIVATE/'completed_D_preservation_manifest_PRIVATE.json')
    for row in old:
        for path in {row['path'], row['copy']}:
            c.require(c.sha(path) == row['sha256'], 'PRESERVED_PRIOR_F_FP_D_P_RECORD')
    for row in completed:
        c.require(c.sha(row['path']) == row['sha256'], 'PRESERVED_COMPLETE_D_RECORD')
    manifest = c.read(D_PUBLIC/'artifact_manifest.json')
    for name, h in manifest['files'].items():
        c.require(c.sha(D_PUBLIC/name) == h, 'PRESERVED_PUBLISHED_D_BYTES')
    return dict(prior_records=len(old), completed_D_records=len(completed), published_D_files=len(manifest['files']))


def accepted_d(store):
    result_path = D_PUBLIC/'result.json'
    blob = subprocess.check_output(['git', 'show', BASE+':'+str(result_path.relative_to(ROOT))], cwd=ROOT)
    c.require(hashlib.sha256(blob).hexdigest() == c.sha(result_path), 'ACCEPTED_D_COMMIT_BYTES')
    result = c.read(result_path); audit = c.read(D_PUBLIC/'completion_audit.json')
    c.require(result['binding'] == c.BINDING and result['registered_set_complete']
        and audit['result_sha256'] == c.sha(result_path), 'ACCEPTED_COMPLETE_D')
    c.require(len(result['winners']) == 5 and [x['epsilon'] for x in result['winners']] == c.EPS,
        'FIVE_SAVED_D_WINNERS')
    snapshot = store.copy('accepted_D_result', result_path)
    store.copy('accepted_D_audit', D_PUBLIC/'completion_audit.json')
    return dict(path=str(snapshot), sha256=c.sha(snapshot), binding=c.BINDING, result_commit=BASE), deepcopy(result['winners'])


def refuse(contract_path, grant_path, private):
    guard = private/'refusal_guard'; guard.mkdir()
    (guard/'sitecustomize.py').write_text("import sys\ndef guard(event,args):\n if event=='import' and (args[0].startswith(('numpy','scipy','qiskit','trottertracks')) or args[0]=='h4_tol_main_science'): raise AssertionError('SCIENCE_IMPORT')\nsys.addaudithook(guard)\n")
    rows = []
    for witness in (False, True):
        command = [sys.executable, str(ROOT/'scripts/run_h4_tol_main.py'), '--stage', 'A',
            '--contract', str(contract_path), '--grant', str(grant_path)]+(['--witness'] if witness else [])
        result = subprocess.run(command, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=str(guard), PYTHONDONTWRITEBYTECODE='1'))
        c.require(result.returncode != 0 and 'SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED' in result.stderr
            and 'SCIENCE_IMPORT' not in result.stderr, 'REFUSAL_BEFORE_SCIENCE')
        path = private/('witness.stderr' if witness else 'direct.stderr'); path.write_text(result.stderr)
        rows.append(dict(witness=witness, exit_code=result.returncode, stderr_sha256=c.sha(path), refused_before_science=True))
    return rows


def audit(contract):
    draft = c.read(d.OLD_PRIVATE/'A_contract_DRAFT_PRIVATE.json')
    check_scope(contract, draft)
    inherited = d.verify_bytes(contract)
    counts = {k:inherited[k] for k in ('SOURCE_files','inputs','provider_files','runtime_files')}
    counts.update(saved_A_signal_records=1, saved_FP_configurations=1, saved_FP_compile_axes=64)
    c.validate_seed_registry(contract)
    bound = {r['path'] for r in contract['inputs']}
    c.require({contract['seed_registry'], contract['old_seed_exclusions'], contract['seed_inventory_receipt'],
        contract['accepted_D']['path']} <= bound, 'FINAL_A_D_SEEDS_BOUND')
    c.require(c.sha(contract['accepted_D']['path']) == contract['accepted_D']['sha256'], 'ACCEPTED_D_CHANGED')
    c.require(contract['pending'] == PENDING, 'EXPLICIT_APPROVAL_AND_ACTIVATION_REMAIN')
    fp = c.read(contract['FP_evidence'])
    anchor = [x for x in fp['measurements'] if x['cell_id'] == c.FP_PR]
    c.require(len(anchor) == 32 and {x['replica'] for x in anchor} == set(range(32)), 'SAVED_FP_N32')
    for row in anchor:
        c.reuse_measurement(row, c.BINDING, next(x for x in c.registrations()['A'] if x['id'] == c.FP_PR),
            accepted_result_sha256=fp['result_sha256'], source=fp['SOURCE'])
    return dict(status='PASS_METADATA_BYTES_AND_SEED_IDENTIFIERS_ONLY', counts=counts,
        preserved=preserve(), saved_FP_n=32, newly_assigned_trajectories=392,
        actual_science_calls=0, actual_numerical_analyses=0, scientific_execution_authorized=False,
        pending=PENDING, mandatory_STOP=True)


def parallel_and_completion_plan():
    return dict(agent_imposed_CPU_RAM_worker_wall_caps=None,
        resource_assignment='Fresh physical CPUs within existing affinity/quota and effective RAM; account for known active allocations; no fixed historical worker cap',
        phases=['one saved rank10 context; cache 3 preparations and primitive actions',
            '49 independent new base signals in owned native pool; saved FP signal reused',
            'fixed signed/factor1 eligibility only; no candidate reselection',
            'eligible candidates replica0 in parallel, all controls PASS before remaining replicas',
            'remaining eligible trajectories in parallel; paired cosine/sine share trajectory and compile in original order',
            'one new-cohort endpoint accounting pass; verbatim FP n32 and accepted D joined'],
        worker_threads=1, fork_COW=True, preparation_cache=True, fingerprint_reuse=True,
        PSI_diagnostic_only=True, unrelated_worktree_updates_are_gates=False,
        finish_requires=['driver native exit0 and same-parent wait', 'all signal/trajectory workers exit0 and reaped',
            '49 new signals plus saved FP signal; all five saved eligibility decisions explicit',
            'exact eligible subset of fixed 392 registry; eight paired trajectories per eligible new candidate',
            'both axes/compiler/fingerprints/event_digest/law/seed/normalization/preparation match',
            'replica0 control4 and independent event oracle PASS before dependent workers',
            'all nonrefundable reservations and measured receipts match fixed call maxima',
            '250 endpoint rows; saved FP statistics unmodified; no pooling or missing eligible costs',
            'SOURCE/input/provider/runtime/accepted-D/seed/preservation hashes unchanged',
            'save result/raw receipts/SHA/terminal then mandatory STOP'],
        scientific_accounting_repeated_by_auditor=False, automatic_retry=False, next_stage_authorized=False,
        adverse_PR_result_is_technical_failure=False, additional_samples_on_inconclusive=False,
        future_seed_policy='Later campaigns exclude this registry. Before launch refresh only newly registered identifier collisions, retain these seeds; collision requires STOP/new contract, never redraw.')


def seal(private=PRIVATE, public=PUBLIC):
    c.require(not private.exists() and not public.exists(), 'NEW_A_METADATA_SEAL_NAMESPACE')
    preserved = preserve(); private.mkdir(); store = Store(private)
    draft = c.read(d.OLD_PRIVATE/'A_contract_DRAFT_PRIVATE.json')
    counts = d.verify_bytes(draft)
    accepted, winners = accepted_d(store)
    excluded, coverage, cases, old_coverage = inventory(store)
    registry, fresh = new_registry()
    c.require(fresh.isdisjoint(excluded), 'EXPLICIT_SEED_COLLISION_NO_REDRAW')
    start = time.monotonic(); geometry, workers = seeds.check_geometry(cases, fresh)
    c.require(all(not x['collisions'] for x in geometry), 'GEOMETRY_SEED_COLLISION_NO_REDRAW')
    c.save(private/'seed_registry_PRIVATE.json', registry)
    c.save(private/'old_seed_exclusions_PRIVATE.json', dict(values=sorted(excluded)))
    receipt = dict(status='COMPLETE_COLLISION_ZERO', all_registered_scopes_covered=True,
        scope='KNOWN_REGISTERED_H4_AND_RECORDED_PARALLEL_H6_H8_CAMPAIGNS_AT_CUTOFF; NOT_UNKNOWN_OR_FUTURE_JOBS',
        cutoff_UTC=datetime.now(timezone.utc).isoformat(), coverage=coverage, inherited_H4_coverage=old_coverage,
        geometry_checks=geometry, old_explicit_exclusions_sha256=c.sha(private/'old_seed_exclusions_PRIVATE.json'),
        new_seed_values_digest=c.digest(sorted(fresh)), old_explicit_values=len(excluded), new_master_count=392,
        new_seed_slots=len(fresh), internal_collision_count=0, explicit_collision_count=0,
        integer_metadata_workers=workers, geometry_check_wall_seconds=time.monotonic()-start,
        scientific_RTE_sampling=0, mathematical_PRNG_independence_proved=False,
        registry_sha256=c.sha(private/'seed_registry_PRIVATE.json'))
    c.save(private/'seed_inventory_receipt_PRIVATE.json', receipt)
    c.save(private/'preparation_authorization_PRIVATE.json', dict(
        D_result_commit=BASE, D_formally_accepted=True, A_final_seal_authorized=True,
        user_instruction='D190の完了結果を正式に受け入れ、A50の最終実行契約の準備を承認します。',
        A_science_authorized=False, saved_numerical_analysis_authorized=False, S_STATS_P_authorized=False,
        commit_and_public_push_authorized=True, mandatory_STOP=True))
    c.save(private/'seed_origin_manifest_PRIVATE.json', store.rows)
    contract = deepcopy(draft)
    contract.update(status='TECHNICALLY_SEALED_PENDING_EXPLICIT_A_APPROVAL', pending=PENDING,
        accepted_D=accepted, seed_inventory_complete=True, seed_registry=str(private/'seed_registry_PRIVATE.json'),
        old_seed_exclusions=str(private/'old_seed_exclusions_PRIVATE.json'),
        seed_inventory_receipt=str(private/'seed_inventory_receipt_PRIVATE.json'),
        output=str(private/'run01/output'), lock=str(private/'run01/ONE_SHOT.json'),
        accepted_D_result_commit=BASE, scientific_conditions_changed=False)
    for row in store.rows:
        contract['inputs'].append({k:row[k] for k in ('role','path','sha256')})
    for name in ('seed_registry_PRIVATE.json','old_seed_exclusions_PRIVATE.json','seed_inventory_receipt_PRIVATE.json',
                 'preparation_authorization_PRIVATE.json','seed_origin_manifest_PRIVATE.json'):
        path = private/name; contract['inputs'].append(dict(role='A_final:'+name, path=str(path), sha256=c.sha(path)))
    path = private/'A_contract_SEALED_PENDING_APPROVAL_PRIVATE.json'; c.save(path, contract)
    grant = c.invalid_grant('A', contract); c.save(private/'A_grant_INVALID_PRIVATE.json', grant)
    refusal = refuse(path, private/'A_grant_INVALID_PRIVATE.json', private)
    resource = d.resource_snapshot(); c.save(private/'resource_observation_PRIVATE.json', resource)
    result = audit(contract)
    counts = result['counts']
    public.mkdir(parents=True)
    projection = dict(status=contract['status'], SOURCE=SOURCE, accepted_D_commit=BASE,
        binding=c.BINDING, compiler_options=contract['plan']['compiler_options'], plan_digest=c.digest(contract['plan']),
        candidate_set_digest=c.digest(contract['plan']['cells']), registered_candidates=50,
        cohorts=[dict(id='SAVED_FP_PANEL_N32_ANCHOR', cell_id=c.FP_PR, n=32, new_sampling=False, raw_reaggregation=False),
            dict(id='EXPLORATION_N8', candidates=49, n=8, maximum_trajectories=392, maximum_compile_axes=784)],
        calls=contract['plan']['calls'], contract_digest=c.digest(contract), source_hashes=contract['source_hashes'],
        input_count=len(contract['inputs']), input_binding_digest=c.digest(contract['inputs']),
        provider_binding_digest=c.digest(contract['provider_hashes']), runtime_binding_digest=c.digest(contract['runtime_hashes']),
        python_sha256=contract['python_sha256'], counts=counts,
        seed_registry_sha256=c.sha(contract['seed_registry']), seed_inventory_receipt_sha256=c.sha(contract['seed_inventory_receipt']),
        pending=PENDING, one_shot=True, approved_by_user=False, runtime_authorization=False,
        automatic_retry=False, next_stage_authorized=False, actual_science_calls=0, mandatory_STOP=True)
    c.save(public/'A_contract_SEALED_PENDING_APPROVAL.json', projection)
    c.save(public/'A_grant_INVALID.json', grant)
    c.save(public/'accepted_D_saved_winners.json', dict(result_commit=BASE, result_sha256=accepted['sha256'],
        binding=c.BINDING, winners=winners, verbatim_saved_projection=True, scientific_recalculation=False))
    c.save(public/'seed_inventory_projection.json', dict(receipt,
        geometry_checks=[{k:v for k,v in x.items() if k != 'collisions'} for x in geometry],
        seed_values_published=False, geometry_collision_count=0))
    c.save(public/'parallel_and_completion_plan.json', parallel_and_completion_plan())
    c.save(public/'approval_boundary.json', dict(preparation_authorized=True, D_formally_accepted=True,
        A_science_authorized=False, valid_grant_issued=False, launch_command=None,
        execution_approval_must_bind=['SOURCE', 'contract_digest', 'seed_registry_sha256', 'input/compiler/runtime',
            '392 trajectory/784 compile-axis maxima', 'fresh resources', 'one-shot output/lock', 'no retry; mandatory STOP'],
        activation='After explicit user approval create new execution contract, fresh receipt and valid grant; retain this pending seal and INVALID grant unchanged.',
        S_STATS_P_authorized=False, scientific_review_started=False))
    c.save(public/'resource_observation.json', dict(allowed_CPUs_observed=len(resource['affinity']),
        admitted_for_execution=False, actual_worker_assignment=None, agent_imposed_CPU_RAM_worker_wall_caps=None,
        private_observation_sha256=c.sha(private/'resource_observation_PRIVATE.json')))
    c.save(public/'refusal_report.json', dict(cases=2, rows=refusal, all_refused_before_science=True,
        native_driver_started=False, scientific_output=False, one_shot_marker=False))
    c.save(public/'completion_audit.json', dict(result, preserved=preserved, preparer_sha256=c.sha(__file__),
        seed_inventory_tool_sha256=c.sha(seeds.__file__), AI_subagents_used=False))
    c.save(public/'artifact_manifest.json', dict(files={p.name:c.sha(p) for p in sorted(public.glob('*.json'))
        if p.name != 'artifact_manifest.json'}, scientific_results_generated=False))
    # Private metadata is retained separately from all earlier D/F/FP/STOP records.
    c.save(private/'seal_preservation_manifest_PRIVATE.json', [dict(path=str(p), sha256=c.sha(p))
        for p in sorted(private.rglob('*')) if p.is_file()])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', help='Byte/identifier checks only; no dispatch')
    args = parser.parse_args()
    import json
    print(json.dumps(audit(c.read(PRIVATE/'A_contract_SEALED_PENDING_APPROVAL_PRIVATE.json')) if args.audit else seal()))
