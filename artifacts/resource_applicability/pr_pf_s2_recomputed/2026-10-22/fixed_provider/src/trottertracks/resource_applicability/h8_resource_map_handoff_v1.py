"""Technical-only SOURCE transition and conservative small-circuit admission.

Circuit size and observed RSS are engineering evidence, not a memory theorem.
The 9GiB STOP remains active after promotion; rare/large/unknown stays 24GiB.
"""
import json
import time
from pathlib import Path
from .ax2a_preparation import digest

PARENT_SOURCE = 'cb8a56c6b2a772827275f9200b02638030f33f72'
AUTHORIZATION = '8ワーカーにして引き継いて'
PREFIX = 'src/trottertracks/resource_applicability/'
NEW_SOURCE_PATH = PREFIX+'h8_resource_map_handoff_v1.py'
CHANGED_SOURCE_PATHS = {PREFIX+'h8_resource_map_contract_v1.py',
                        PREFIX+'h8_resource_map_execution_v1.py',
                        PREFIX+'h8_resource_map_optimization_v1.py'}
EXTRA_KEYS = ('compile', 'trajectory', 'occurrence', 'control_probe')
REPAIR_PARENT_SOURCE = 'c6692964c948071ed00363b5b711fbca8930ad61'
REPAIR_AUTHORIZATION = '修正して再開させて'
REPAIR_CHANGED_PATHS = {PREFIX+'h8_resource_map_contract_v1.py',
                       PREFIX+'h8_resource_map_execution_v1.py', NEW_SOURCE_PATH}


def compatibility(current, previous):
    """Accept only this fixed parent's scheduler/binding change, preserving seed."""
    for key in ('input_identity','profile','compiler_hash','plan','cumulative_carry',
                'proposed_resources','observation_policy','initial_registrations'):
        if digest(current[key]) != digest(previous[key]):
            raise ValueError('H8_RESUME_SCIENTIFIC_CONTRACT_CHANGED:'+key)
    if current['source_commit'] == previous['source_commit']:
        if current['source_hashes'] != previous['source_hashes']:
            raise ValueError('H8_RESUME_SOURCE_HASHES')
    elif previous['source_commit'] == REPAIR_PARENT_SOURCE:
        if (current.get('seed_source_commit') != PARENT_SOURCE
            or current.get('repair_authorization_basis') != REPAIR_AUTHORIZATION):
            raise ValueError('H8_RESUME_REPAIR_AUTHORIZATION')
        old,new=previous['source_hashes'],current['source_hashes']
        if set(old)!=set(new) or {p for p in old if old[p]!=new[p]}!=REPAIR_CHANGED_PATHS:
            raise ValueError('H8_RESUME_SCIENCE_SOURCE_CHANGED')
    else:
        if (previous['source_commit'] != PARENT_SOURCE
            or current.get('seed_source_commit') != PARENT_SOURCE
            or current.get('handoff_authorization_basis') != AUTHORIZATION):
            raise ValueError('H8_RESUME_SOURCE_TRANSITION')
        old, new = previous['source_hashes'], current['source_hashes']
        if set(new) != set(old)|{NEW_SOURCE_PATH}:
            raise ValueError('H8_RESUME_SOURCE_CLOSURE_TRANSITION')
        changed = {p for p in old if new[p] != old[p]}
        if changed != CHANGED_SOURCE_PATHS:
            raise ValueError('H8_RESUME_SCIENCE_SOURCE_CHANGED')
    ref = current.get('resume_parent', {})
    extra = ref.get('additional_held_reservations', {})
    expected = dict(previous['call_budget'])
    for k, v in extra.items():
        if k not in EXTRA_KEYS or type(v) is not int or v < 0:
            raise ValueError('H8_RESUME_EXTRA_BUDGET')
        expected[k] += v
    if current['call_budget'] != expected:
        raise ValueError('H8_RESUME_CALL_BUDGET_CHANGED')
    return dict(parent_source=previous['source_commit'],
                seed_source=current.get('seed_source_commit',current['source_commit']),
                scientific_source_changed=False, compiler_changed=False,
                input_changed=False, allowed_changes=sorted(CHANGED_SOURCE_PATHS),
                extra_reservations=extra)


def light_case_evidence(task, trajectory, diagnostic, memory):
    """Inspect actual sampled events and first constructed evolution, never sample."""
    if task.get('kind') != 'COST' or trajectory.get('task') != task:
        return None
    events = trajectory.get('events')
    if trajectory.get('event_digest') != digest(events):
        return None
    if task['cell']['R']:
        if (not isinstance(events,list) or len(events)!=task['cell']['q']
            or any(not isinstance(row,list) or len(row)!=task['cell']['r'] for row in events)):
            return None
        actual = [e.get('taylor_order') for row in events for e in row]
        if (trajectory.get('observed_orders') != actual
            or any(type(x) is not int or x != 0 for x in actual)):
            return None
    elif events is not None:
        return None
    context = diagnostic.get('context', {})
    count = context.get('instructions')
    cap = 100000 if task['ordinary'] else 200000
    if (context.get('stage') != 'numeric_fingerprint'
        or context.get('control') != 'symmetric_directional' or 'axis' in context
        or diagnostic.get('kind') not in ('begin','end')
        or type(count) is not int or not 0 < count <= cap
        or type(memory.get('VmHWM_bytes')) is not int
        or memory['VmHWM_bytes'] > 6*2**30):
        return None
    return dict(instructions=count, instruction_cap=cap,
                event_digest=trajectory['event_digest'],
                observed_peak_rss_bytes=memory['VmHWM_bytes'],
                rss_stop_bytes=9*2**30, hard_memory_bound_proved=False,
                policy='actual small native evolution; zero Taylor orders; 9GiB fail-closed')


def memory_request(task, trajectory, bounds, source):
    """Two-stage gate: prepare at 1GiB, wait for 9/24GiB before any native build."""
    events=trajectory.get('events');orders=trajectory.get('observed_orders')
    valid=(trajectory.get('task')==task and trajectory.get('event_digest')==digest(events))
    if task['cell']['R']:
        valid=valid and isinstance(events,list) and len(events)==task['cell']['q']
        valid=valid and all(isinstance(x,list) and len(x)==task['cell']['r'] for x in events or [])
        actual=[e.get('taylor_order') for row in events for e in row] if valid else None
        valid=valid and orders==actual and all(type(x) is int and x==0 for x in actual or [])
    else:valid=valid and events is None
    cap=100000 if task['ordinary'] else 200000
    values=[bounds.get(p) for p in (('symmetric_directional','ordinary') if task['ordinary'] else ('symmetric_directional',))]
    small=valid and all(type(x) is int and 0<x<=cap for x in values)
    return dict(source_commit=source,task_digest=digest(task),event_digest=trajectory['event_digest'],
        wrapper_instruction_upper_bounds=bounds,requested_rss_bytes=(9 if small else 24)*2**30,
        classification='zero-order structural small' if small else 'heavy/rare/unknown',
        memory_bound_proved=False,operation='native evolution/compile waits for coordinator grant')


def await_memory_admission(writer, progress, task, trajectory, bounds, source):
    request=memory_request(task,trajectory,bounds,source)
    writer.write('memory_admission_request.json',request)
    progress.update(point='waiting_memory_admission',requested_rss_bytes=request['requested_rss_bytes'])
    grant_path=Path(writer.output)/'memory_admission.json'
    while not grant_path.is_file():time.sleep(.05)
    grant=json.loads(grant_path.read_text())
    expected=dict(request_digest=digest(request),source_commit=source,
        task_digest=digest(task),rss_stop_bytes=request['requested_rss_bytes'])
    if grant!=expected:raise ValueError('WORKER_MEMORY_ADMISSION_BINDING')
    progress.update(point='memory_admission_granted',rss_stop_bytes=grant['rss_stop_bytes'])
