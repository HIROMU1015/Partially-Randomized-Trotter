"""Final metadata binding. No molecular sampler, native provider or statistics."""
from pathlib import Path
import h6_pr_confirmation as h
c = h.c
FINAL_N = dict(H6_CONFIRM_P4_Q2=48, H6_CONFIRM_P4_Q8=104)
REQUIRED_CAMPAIGNS = (
    'H6_LEGACY_AND_A128', 'H6_FIXED_DELTA_RESERVATIONS', 'H6_CONDITIONAL',
    'HIGHER_PF_LEGACY_AND_ACQUISITION', 'H6_HIGHER_PF_CONFIRMATION_RESERVED',
    'H6_HIGHER_PF_CONFIRMATION_USED', 'H8_NEW_A_AND_CONFIRMATION',
    'H4_RECORDED_OLD_SEEDS', 'H4_P_CONFIRMATION')


def process_slots(cpu_slots):
    """Driver + trajectory parents + active axis/probe descendants (upper estimate)."""
    c.require(type(cpu_slots) is int and cpu_slots > 0, 'POSITIVE_CPU_SLOTS')
    return max(2 + min(5, cpu_slots), 1 + cpu_slots + (cpu_slots+1)//2)


def seed_values(value, mode):
    """Read explicit identifiers or reproduce only the frozen master-to-step API."""
    if mode == 'values':
        values = value['values']
    elif mode == 'all_master_and_step_values':
        values = value['all_master_and_step_values']
    elif mode == 'flat_registry':
        values = [s for task in value['tasks'].values() for s in task.values()]
    elif mode in ('tasks', 'fixed_delta'):
        tasks = value['tasks'] if mode == 'tasks' else [r['task'] for r in value['plan']['registrations']]
        values = list(value.get('excluded', []))
        for task in tasks:
            if task.get('seed') is not None:
                cell = task.get('cell') or task['group']['cell']
                values.extend(h.shared.hierarchy(task['seed'], cell['q']))
    else:
        raise ValueError('UNKNOWN_SEED_METADATA_MODE')
    c.require(all(type(s) is int and 0 <= s < 2**64 for s in values), 'UINT64_SEED_METADATA')
    return set(values)


def inventory_values(inventory, *, verify_bytes=True):
    c.require(inventory['schema'] == 'H6_FINAL_SEED_INVENTORY_V1'
              and inventory['known_registered_campaigns_complete'] is True
              and set(inventory['campaigns']) == set(REQUIRED_CAMPAIGNS), 'COMPLETE_CAMPAIGN_INVENTORY')
    values = set()
    for campaign in REQUIRED_CAMPAIGNS:
        rows = inventory['campaigns'][campaign]
        c.require(bool(rows), 'EMPTY_CAMPAIGN')
        for row in rows:
            if verify_bytes:
                c.verify_byte_rows([row])
            values.update(seed_values(c.read(row['path']), row['seed_mode']))
    return values


def verify_final_binding(spec, registry):
    covered = {(str(Path(r['path']).resolve()), r['sha256']) for r in spec['inputs']}
    for key in ('seed_inventory', 'seed_plan', 'final_selection'):
        c.require((str(Path(spec[key]).resolve()), c.sha(spec[key])) in covered, 'UNCOVERED_FINAL_METADATA')
    selection = c.read(spec['final_selection'])
    c.require(selection['schema'] == 'H6_FINAL_SELECTION_V1'
              and selection['n_by_scope'] == spec['n_by_scope'] == FINAL_N
              and selection['targets'] == c.read(h.ROOT / h.PLAN)['targets']
              and selection['scientific_execution_authorized'] is False, 'FINAL_SELECTION_BINDING')
    inventory = c.read(spec['seed_inventory'])
    excluded = inventory_values(inventory)
    c.require(registry['excluded'] == sorted(excluded)
              and registry['seed_inventory_sha256'] == c.sha(spec['seed_inventory']), 'EXACT_COMPLETE_EXCLUSIONS')
    plan = c.read(spec['seed_plan'])
    c.require(c.digest(plan) == spec['seed_plan_digest']
              and plan['SOURCE'] == spec['SOURCE'] and plan['n_by_scope'] == FINAL_N
              and plan['source_hashes'] == spec['source_hashes']
              and plan['design_digest'] == spec['design_digest']
              and plan['seed_inventory_sha256'] == c.sha(spec['seed_inventory'])
              and plan['final_selection_sha256'] == c.sha(spec['final_selection']), 'FINAL_SEED_PLAN_BINDING')
    c.require(all((str(Path(r['path']).resolve()), r['sha256']) in covered
                  for r in plan['immutable_inputs']), 'UNCOVERED_SEED_PLAN_INPUT')


def audit_completion(spec, native_exit):
    """Revalidate raw identities, exits and reservations, without rerunning statistics."""
    c.require(native_exit['exit_code'] == 0 and native_exit['reaped'], 'H6_DRIVER_NATIVE_EXIT0_REQUIRED')
    output = Path(spec['output'])
    c.require(not (output / 'STOP.json').exists(), 'NO_FAILURE_STOP_ON_SUCCESS')
    result, terminal = c.read(output / 'result.json'), c.read(output / 'terminal.json')
    c.require(terminal['status'] == result['status'] == 'COMPLETE_MANDATORY_STOP'
              and terminal['SOURCE'] == result['SOURCE'] == spec['SOURCE']
              and terminal['result_sha256'] == c.sha(output / 'result.json')
              and result['scope'] == spec['scope'] and result['calls'] == spec['calls']
              and result['old_samples_pooled'] is False and result['automatic_retry'] is False
              and result['next_stage_authorized'] is False, 'FINAL_COMPLETION_BINDING')
    tasks = [t for t in c.read(spec['registry'])['tasks'] if t['scope'] == spec['scope']]
    receipt = c.read(output / 'native_receipt.json')
    c.require(receipt['all_reaped'] and all(r['all_reaped'] and not r['cleanup_errors']
              for r in receipt['maps']), 'ALL_NATIVE_MAPS_REAPED')
    handles = [w for r in receipt['maps'] for w in r['workers']]
    by_id = {w['task_id']: w for w in handles}
    c.require(len(handles) == len(by_id) == spec['n'] and set(by_id) == {t['id'] for t in tasks},
              'EXACT_COMPLETED_TASK_SET')
    accepted = [h.accept(t, by_id[t['id']], spec)[0] for t in tasks]
    c.require(accepted == result['paired_costs'], 'SAVED_PAIRED_RESULT_IDENTITIES')
    for kind, count in spec['calls'].items():
        paths = list(output.glob(kind + '_*_reserved.json'))
        rows = [c.read(p) for p in paths]
        c.require(rows and sum(r['count'] for r in rows) == count
                  and max(r['total'] for r in rows) == count
                  and all(r['kind'] == kind and r['nonrefundable'] is True for r in rows),
                  'EXACT_NONREFUNDABLE_PARENT_LEDGER')
    return dict(status='COMPLETE_MANDATORY_STOP', raw_trajectories=spec['n'],
        compiled_axes=2*spec['n'], all_native_descendants_reaped=True,
        fresh_statistics_recomputed=False, result_sha256=c.sha(output / 'result.json'))
