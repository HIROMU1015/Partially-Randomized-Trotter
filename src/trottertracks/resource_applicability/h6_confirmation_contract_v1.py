"""Frozen top2-union independent32 confirmation; preparation issues no grant."""
from copy import deepcopy
import math
from pathlib import Path

from .ax2a_preparation import digest
from .h6_matched_accounting_v1 import confirmation_ids, resource_row, tasks, seed
from .h6_matched_reuse_v3 import read, committed_hashes, admit_group, complete_exploration, file_hash
from .h6_parallel_execution_v1 import admit_parallel_group
from .h6_parallel_contract_v1 import sources as parent_sources, proposed_resources, slots, roundoff_binding
from .h6_server_contract_v1 import safe_path, verify_bindings
from .h6_server_resources_v1 import observation_policy, fresh_resources

RESULT_COMMIT = 'df99d6ee28942df8d12aac3c4db7cd202df55d77'
PARENT_SOURCE = '78a39051d8b6d41c33d592ecfe172f8824adfaeb'
REPORT = 'artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/'
BASE = 'artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/sealed_manifest_server_v1.json'
PREPARATION = 'artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12/'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_independent_confirmation_v1/2026-10-12/'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_confirmation_v1.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h6_confirmation_v1.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_confirmation_v1.py'
LOCK = '.server-preparation/h6-independent-confirmation-v1/one-shot.lock'
KIND = 'H6_TOP2_UNION_INDEPENDENT32_CONFIRMATION_ONLY_V1'
SCHEMA = 'h6_confirmation_manifest_v1'


def private_path(value):
    path = Path(value)
    if (not path.is_absolute() or not path.is_relative_to('/home/AbeHiromu')
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('CONFIRMATION_PRIVATE_PATH')
    return path


def sources(root):
    return sorted(set(parent_sources(root)) | {RUNNER, PREPARER, AUDITOR,
        'src/trottertracks/resource_applicability/h6_confirmation_contract_v1.py',
        'src/trottertracks/resource_applicability/h6_confirmation_execution_v1.py'})


def select_and_register(signals, rows, plan):
    selected = confirmation_ids(signals, rows, plan)
    ranked = []
    for epsilon in plan['epsilons']:
        candidates = []
        for s in signals:
            if s['cell']['method'] != 'B2':
                continue
            r = resource_row(s, [x for x in rows if x['cell_id'] == s['cell']['id']], epsilon)
            if r is not None:
                candidates.append(r)
        candidates.sort(key=lambda r: (r['G_point'], r['cell_id']))
        ranked.append(dict(epsilon=epsilon, top2=[dict(cell_id=r['cell_id'], G_RZ_point=r['G_point'])
            for r in candidates[:plan['cost_rule']['confirmation_top_per_epsilon']]]))
    registered = tasks(signals, plan, 'confirmation', selected)
    if (len(selected) > 8 or len(registered) != 32*len(selected)
            or any(t['phase'] != 'confirmation' or t['cell']['method'] != 'B2'
                or t['ordinary'] is not False or t['validate'] is not False for t in registered)):
        raise ValueError('CONFIRMATION_REGISTERED_SCOPE')
    if len({t['seed'] for t in registered}) != len(registered):
        raise ValueError('CONFIRMATION_DUPLICATE_SEED')
    return dict(selected_ids=selected, rankings=ranked, registrations=[dict(index=i, task=t)
        for i, t in enumerate(registered)], rule='original B2 top2 G_RZ point per epsilon; lexical ID tie; union<=8; independent32',
        exploration_rows_digest=digest(rows), selected_before_confirmation=True,
        exploratory_samples_pooled=False, adaptive_samples_allowed=False)


def exploration_binding(root):
    """Admit all 592 saved groups before computing any ranking or selection."""
    root = Path(root)
    names = ['result_summary_v1.json', 'integrated_exploration_v1.json', 'summary_audit_v1.json']
    hashes = {REPORT+n: file_hash(root/REPORT/n) for n in names}
    committed_hashes(root, RESULT_COMMIT, hashes)
    scalar = read(root/REPORT/'result_summary_v1.json')
    provenance = read(root/REPORT/'integrated_exploration_v1.json')
    if (scalar['source_commit'] != PARENT_SOURCE or scalar['coverage']['exploration_groups'] != 592
            or scalar['coverage']['total_wrappers'] != 1188 or scalar['confirmation_started'] is not False
            or [x['index'] for x in provenance['groups']] != list(range(592))):
        raise ValueError('CONFIRMATION_EXPLORATION_REPORT')
    parents = {str(private_path(x['cost_result_path']).parent.parent) for x in provenance['groups']
        if x['origin'] == 'resume01_new135'}
    if len(parents) != 1:
        raise ValueError('CONFIRMATION_PARENT_RUN_SET')
    parent = private_path(next(iter(parents)))
    pm = read(parent/'frozen_manifest.json'); terminal = read(parent/'execution_terminal.json')
    completed = read(parent/'exploration_complete_receipt.json')
    if (pm['source_commit'] != PARENT_SOURCE or terminal['status'] != 'H6_PARALLEL_COMPLETE_MANDATORY_STOP'
            or terminal['reason'] is not None or not terminal['resource_summary']['all_dispatched_reaped']
            or terminal['resource_summary']['cleanup_errors'] or completed['group_count'] != 592
            or completed['wrapper_count'] != 1188 or completed['selection_not_performed'] is not True
            or file_hash(parent/'exploration_complete_receipt.json') != scalar['saved_completion_receipt_sha256']
            or file_hash(parent/'execution_terminal.json') != scalar['saved_execution_terminal_sha256']):
        raise ValueError('CONFIRMATION_PARENT_COMPLETION')
    base = read(root/BASE)
    if base != pm['base_manifest']:
        raise ValueError('CONFIRMATION_PARENT_BASE')
    registered = base['reuse_registry']['original_tasks']; groups = {}; inventory = {}
    for row in provenance['groups']:
        path = private_path(row['cost_result_path']); directory = path.parent; i = row['index']
        if file_hash(path) != row['cost_result_sha256'] or row['task_digest'] != digest(registered[i]):
            raise ValueError('CONFIRMATION_EXPLORATION_BYTES')
        result = (admit_group if row['origin'] == 'legacy449' else admit_parallel_group)(
            directory, registered[i], source=row['source_commit'])
        groups[i] = result
        for f in directory.glob('*.json'):
            if f.name.startswith(('diagnostic_', 'progress_', 'compile_attempt_')):
                continue
            inventory[str(f)] = dict(bytes=f.stat().st_size, sha256=file_hash(f))
    rows = complete_exploration(registered, groups)
    if len(rows) != 1188 or digest(rows) != provenance['exploration_rows_digest']:
        raise ValueError('CONFIRMATION_EXPLORATION_ROWS')
    old_run = root/base['reuse_registry']['original_raw_path']
    signal_path = old_run/'signal/signal_summary.json'
    expected = base['reuse_registry']['raw_hashes'][str(signal_path.relative_to(root))]
    if file_hash(signal_path) != expected:
        raise ValueError('CONFIRMATION_SAVED_SIGNAL')
    signals = read(signal_path)['signals']
    selection = select_and_register(signals, rows, base['plan'])
    old_seeds = {t['seed'] for t in registered}
    if old_seeds & {x['task']['seed'] for x in selection['registrations']}:
        raise ValueError('CONFIRMATION_SEED_OVERLAP')
    return dict(result_commit=RESULT_COMMIT, report_hashes=hashes, parent_run=str(parent),
        parent_manifest_sha256=file_hash(parent/'frozen_manifest.json'),
        parent_terminal_sha256=file_hash(parent/'execution_terminal.json'),
        parent_completion_sha256=file_hash(parent/'exploration_complete_receipt.json'),
        saved_group_metadata_inventory=inventory, selection=selection,
        carry_known_calls=deepcopy(completed['known_calls']), carry_reservations=deepcopy(completed['reservations']),
        historical_unknown_actual_totals=None, budget_reset=False, failed_calls_refunded=False,
        roundoff_preparation_binding=deepcopy(pm['roundoff_preparation_binding']))


def demand(rows):
    return dict(compile=2*len(rows), control_probe=0, trajectory=len(rows),
        occurrence=sum(x['task']['cell']['q'] for x in rows))


def accounting(binding, base):
    known = binding['carry_known_calls']; reserved = binding['carry_reservations']
    needed = demand(binding['selection']['registrations'])
    original = deepcopy(base['plan']['caps_proposed']); proposed = deepcopy(original)
    # Historical diagnostics and stopped tasks are not refunded. This is a
    # proposal bound into the future grant, not an approved cap modification.
    proposed['trajectory'] = max(original['trajectory'], known['trajectory'] + reserved['trajectory'] + needed['trajectory'])
    totals = {k: known[k]+reserved.get(k, 0)+v for k, v in needed.items()}
    if any(totals[k] > proposed[k] for k in totals):
        raise ValueError('CONFIRMATION_CUMULATIVE_CAP')
    return dict(known_calls=known, reservations=reserved, new_demand=needed, admission_totals=totals,
        original_caps=original, proposed_caps=proposed,
        amendments={k:dict(original=original[k], proposed=proposed[k]) for k in original if original[k] != proposed[k]},
        unknown_actual_totals=None, reservations_are_not_actual_totals=True, cap_amendment_approved=False,
        failed_calls_refunded=False, budget_reset=False)


def rare_order_plan(binding):
    """Finite-law diagnostics for fixed32, not an unobserved compiled-cost bound."""
    root = Path(binding['parent_run']).parents[4]
    saved = read(root/REPORT/'cost_order_diagnostics_v1.json')
    result = []
    for cell in binding['selection']['selected_ids']:
        row = next(x for x in saved['rows'] if x['cell_id'] == cell)
        law = next(iter(row['saved_distributions'].values()))
        p = math.fsum(v for k,v in zip(law['orders'],law['order_probabilities'],strict=True) if k != 0)
        unseen = row['unseen_single_draw_mass_stable']; R = row['R']
        result.append(dict(cell_id=cell, R=R, confirmation_trajectories=32,
            exploratory_observed_counts=row['observed_counts'], nonzero_mass_per_draw=p,
            probability_no_nonzero_order_in_confirmation32=math.exp(32*R*math.log1p(-p)) if p < 1 else 0.,
            probability_confirm32_contains_exploration_unseen_order=-math.expm1(32*R*math.log1p(-unseen)) if unseen < 1 else 1.,
            population_cost_tail_bound=None, observed_SE_is_tail_bound=False,
            adaptive_sampling=False, exploratory_and_confirmation_samples_pooled=False))
    return dict(rows=result, fixed32_even_if_no_nonzero_order=True,
        population_mean_decision='UNDETERMINED_ALLOWED', instruction_cap_is_population_bound=False)


def output_binding(root, m):
    relative = NAMESPACE+'launch_confirmation_v1'
    if (m['exclusive_output'] != dict(repository_path=relative, absolute_path=str(safe_path(root, relative)))
            or m['one_shot_lock'] != LOCK):
        raise ValueError('CONFIRMATION_OUTPUT_BINDING')


def validate_scope(root, m):
    expected_flags = dict(approved=False, runtime_authorization=False, allowed_cpus=[],
        confirmation_authorized=False, H8_authorized=False, retry=False, resume=False, mandatory_stop=True,
        next_stage_authorized=False)
    if m.get('schema') != SCHEMA or m.get('kind') != KIND or m.get('repository_root') != str(Path(root).resolve()):
        raise ValueError('CONFIRMATION_SCHEMA_OR_ROOT')
    if any(type(m.get(k)) is not type(v) or m[k] != v for k,v in expected_flags.items()):
        raise ValueError('CONFIRMATION_PREPARATION_FLAGS')
    if (m['proposed_resources'] != proposed_resources() or slots(m['proposed_resources']) != 4
            or m['observation_policy'] != observation_policy()
            or m['registrations'] != m['exploration_binding']['selection']['registrations']
            or m['candidate_selection_sealed'] is not True or m['execution_plan_sealed'] is not True
            or m['absolute_launch_command'] is not None):
        raise ValueError('CONFIRMATION_SCOPE_OR_RESOURCES')
    selected = m['exploration_binding']['selection']['selected_ids']
    registrations = m['registrations']; counts = {}
    if selected != sorted(set(selected)) or not 0 < len(selected) <= 8:
        raise ValueError('CONFIRMATION_SELECTED_SET')
    for i, row in enumerate(registrations):
        t = row['task']
        if (type(row['index']) is not int or row['index'] != i
                or t['phase'] != 'confirmation' or t['cell']['method'] != 'B2'
                or t['cell']['id'] != t['cell_id'] or t['cell_id'] not in selected
                or t['validate'] is not False or t['ordinary'] is not False
                or type(t['replica']) is not int or not 0 <= t['replica'] < 32
                or type(t['seed']) is not int or t['seed'] != seed(t['cell_id'], 'confirmation', t['replica'])):
            raise ValueError('CONFIRMATION_FIXED32_SCOPE')
        counts.setdefault(t['cell_id'], []).append(t['replica'])
    if (len(registrations) != 32*len(selected) or set(counts) != set(selected)
            or any(sorted(v) != list(range(32)) for v in counts.values())):
        raise ValueError('CONFIRMATION_FIXED32_SCOPE')
    output_binding(root, m)


def validate_plan(root, m, *, launch=False):
    root = Path(root); validate_scope(root, m)
    base = read(root/BASE)
    if m['base_manifest'] != base or m['base_sha256'] != file_hash(root/BASE):
        raise ValueError('CONFIRMATION_BASE')
    if set(m['source_hashes']) != set(sources(root)):
        raise ValueError('CONFIRMATION_SOURCE_CLOSURE')
    committed_hashes(root, m['source_commit'], m['source_hashes'])
    verify_bindings(root, base)
    binding = exploration_binding(root)
    if (m['exploration_binding'] != binding or m['roundoff_preparation_binding'] != binding['roundoff_preparation_binding']
            or m['cumulative_accounting'] != accounting(binding, base)
            or m['rare_order_plan'] != rare_order_plan(binding) or not roundoff_binding(root, m)):
        raise ValueError('CONFIRMATION_SELECTION_OR_ACCOUNTING_CHANGED')
    if launch:
        if safe_path(root, LOCK).exists() or safe_path(root, LOCK).is_symlink():
            raise ValueError('CONFIRMATION_ONE_SHOT_USED')


def grant_draft(m):
    return dict(schema='h6_confirmation_authorization_v1', kind=KIND, approved_by_user=False,
        server_usage_approved_by_user=False, trajectory_cap_amendment_approved=False,
        confirmation_authorized=False, manifest_digest=digest(m), source_commit=m['source_commit'],
        profile_sha256=m['base_manifest']['profile']['sha256'],
        exploration_binding_digest=digest(m['exploration_binding']), task_list_digest=digest(m['registrations']),
        candidate_selection_digest=digest(m['exploration_binding']['selection']),
        cumulative_accounting_digest=digest(m['cumulative_accounting']),
        proposed_cumulative_caps=deepcopy(m['cumulative_accounting']['proposed_caps']),
        resources=deepcopy(m['proposed_resources']), exclusive_output=m['exclusive_output']['absolute_path'],
        roundoff_preparation_binding_digest=digest(m['roundoff_preparation_binding']),
        one_shot=True, retry=False, resume=False, H8_authorized=False, next_stage_authorized=False, mandatory_stop=True)


def validate_grant(m, g):
    expected = grant_draft(m)
    expected.update(approved_by_user=True, server_usage_approved_by_user=True,
        trajectory_cap_amendment_approved=True, confirmation_authorized=True)
    if g != expected or any(type(g.get(k)) is not bool for k in (
            'approved_by_user', 'server_usage_approved_by_user', 'trajectory_cap_amendment_approved',
            'confirmation_authorized', 'one_shot', 'retry', 'resume', 'H8_authorized', 'mandatory_stop', 'next_stage_authorized')):
        raise ValueError('NEW_INDEPENDENT32_APPROVED_GRANT_REQUIRED')


def worker_scope(root, out, m):
    validate_scope(root, m)
    if str(Path(out).resolve()) != m['exclusive_output']['absolute_path']:
        raise ValueError('CONFIRMATION_WORKER_OUTPUT')


def launch_gate(root, m, g, out):
    validate_grant(m, g); validate_plan(root, m, launch=True)
    out = Path(out).resolve()
    if str(out) != m['exclusive_output']['absolute_path']:
        raise ValueError('CONFIRMATION_LAUNCH_OUTPUT')
    return fresh_resources(m['proposed_resources'], m['observation_policy'], out)
