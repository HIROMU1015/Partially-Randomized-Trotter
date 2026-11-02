"""Closed resume01 -> resume02 receipts and runtime gates; no stop or launch.

The dataset/seed SOURCE remains run10. Imported run10 bytes and resume01
completions are verified separately. Actuals include ancestor reservations;
the journal and observer already contain cumulative charge/wall.
"""
import math
import os
from pathlib import Path
import re
import time

from . import continuation as original
from .continuation import read_json, private_path, path_ref, MAX_RECEIPT
from .identity import require, sha

PARENT_RUN = 'h4-newhost-signal-compile-20261011-resume01'
PARENT_SOURCE = '941749b8d4aedb43fd5f47b824116082493df393'
PARENT_REVIEW = 'd0b4e1734b063c9fef353fdb3a7b63b26c961e72'
PARENT_REVIEW_DIGEST = '1d4fc82f79216f5afe63f4e0ac23ae0fe233f47bee3b48d87dd0e423b0390491'
PARENT_BUNDLE = 'artifacts/resource_applicability/track_a_h4_production_resume01/2026-10-11/'
DATASET_SOURCE = original.PARENT_SOURCE
NEXT_RUN = 'h4-newhost-signal-compile-20261011-resume02'
SCOPE = 'resume01_inverse_instruction_resume'
SPEEDUP_SOURCE = 'def7d48f4af6c07f2d4e361fae868f3b01db8915'
STARTED_EXPECTED = {'bytes': 4968, 'sha256': 'e8f8efed65d697d4589d1d30b61c029732a9fa56266f6ba1bd04b4e6e57301d3'}
CLOCK_EXPECTED = {'bytes': 630, 'sha256': '8f12b48a1bca3d6a257149fe438a93b6c85ec0268ebc43717d71129e18cbb73e'}
CLOCK_PATH = '/home/AbeHiromu/projects/h4-handoff-evidence/20261011/h4-resume02-preparation-20261011/RESUME01_CLOCK_CONTEXT_v1.json'
CONTROL_CHANGES = frozenset(('continuation.py', 'continuation_binding.py', 'ledger.py', 'launch_binding.py',
                             'execution.py','resources.py','observer.py'))


def verify_time_source(root, source, path):
    """Only the reviewed time-control entry points may differ from resume01."""
    import ast
    from .gates import git_blob
    name=Path(path).name
    allowed={'execution.py':{'OwnedRun':{'__init__'}},'resources.py':{'WallBudget':{'__init__','consumed'}},
             'observer.py':{'ObservationState':{'__init__','evaluate'},'IndependentObserver':{'__init__'}}}
    old=ast.parse(git_blob(root,PARENT_SOURCE,path));new=ast.parse(git_blob(root,source,path))
    def bodies(tree):
        values={}
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                if name=='observer.py' and node.name=='observer_main':continue
                values[node.name]=ast.dump(node,include_attributes=False)
            elif isinstance(node,ast.ClassDef):
                for method in node.body:
                    if isinstance(method,ast.FunctionDef) and method.name not in allowed.get(name,{}).get(node.name,set()):
                        values[node.name+'.'+method.name]=ast.dump(method,include_attributes=False)
            elif isinstance(node,ast.Assign):
                values['assign:'+','.join(t.id for t in node.targets if isinstance(t,ast.Name))]=ast.dump(node,include_attributes=False)
        return values
    require(bodies(old)==bodies(new), 'time amendment changed scientific/resource/ownership source')


def parent_plan(root):
    from .gates import git_blob
    import json
    root = private_path(root);path = PARENT_BUNDLE+'plan_authorized_v1.json'
    raw = (root/path).read_bytes()
    require(raw==git_blob(root, PARENT_REVIEW, path), 'pinned resume01 plan blob')
    plan = json.loads(raw)
    require(plan['run_id']==PARENT_RUN and plan['source_commit']==PARENT_SOURCE and plan['sealed'] is True and
            plan['carry']==dict(actual_invocations=7878, charged_bytes=4320481456, wall_seconds=35474.12562095933) and
            plan['continuation']['identity_source_commit']==DATASET_SOURCE, 'resume01 lineage/carry')
    return plan


def ancestor_store(root, plan):
    store, data = original.load_receipt(root, plan['continuation']['receipt'])
    require(data['carry']==plan['carry'] and data['imported_wrappers']==plan['continuation']['imported_wrappers'],
            'resume01 fixed ancestor receipt/carry')
    return store, data


def native_identities(started):
    require(started['source']==PARENT_SOURCE and started['artifact_commit']==PARENT_REVIEW and
            started['run_id']==PARENT_RUN, 'resume01 started receipt lineage')
    rows = [{k:r[k] for k in original.IDENTITY_FIELDS} for r in started['roles']]
    require(len(rows)==6 and len({r['pid'] for r in rows})==6 and len(started['worker_pids'])==4 and
            {r['pid'] for r in rows}=={started['driver_pid'],started['observer_pid'],*started['worker_pids']} and
            all(r['uid']==os.getuid() for r in rows) and
            sorted(r['role'] for r in started['roles'])==['driver','observer','worker','worker','worker','worker'] and
            all(r['parent']==started['driver_pid'] for r in rows if r['pid']!=started['driver_pid']),
            'resume01 exact owned native6/parent graph')
    return rows


def capture_clock(started):
    """Copy the already pinned live context; never refresh cumulative origin."""
    from .observer import process_sample
    rows = native_identities(started)
    for row in rows:
        current = process_sample(row['pid'])
        require(all(current[k]==row[k] for k in original.IDENTITY_FIELDS), 'resume01 live clock identity')
    clock = read_json(CLOCK_PATH, expected=CLOCK_EXPECTED)
    require(clock['identities']==rows and clock['wall_origin_monotonic']==original.wall_origin(started) and
            clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'resume01 fixed live clock')
    return clock


def capture_stop_proof(started, clock, *, pause=time.sleep):
    require(clock['schema_version']=='h4-resume01-clock-v1' and clock['identities']==native_identities(started) and
            clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'resume01 clock domain')
    audits = []
    for index in range(2):
        if index:pause(2)
        original.audit_absence(clock['identities'])
        audits.append(dict(monotonic=time.monotonic(), identities=clock['identities'], same_owned_remaining=0))
    return dict(schema_version='h4-resume01-native-stop-v1', run_id=PARENT_RUN, source_commit=PARENT_SOURCE,
                artifact_commit=PARENT_REVIEW, clock=clock, native_audits=audits,
                historical_exit_code=None, exact_exit_time=None, exit_data_not_inferred=True)


def verify_stop(proof, started, clock):
    require(proof['schema_version']=='h4-resume01-native-stop-v1' and proof['run_id']==PARENT_RUN and
            proof['source_commit']==PARENT_SOURCE and proof['artifact_commit']==PARENT_REVIEW and
            proof['clock']==clock and clock['schema_version']=='h4-resume01-clock-v1' and
            clock['identities']==native_identities(started) and
            clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'resume01 native stop binding')
    require(clock['wall_origin_monotonic']==original.wall_origin(started), 'resume01 cumulative wall origin changed')
    require(proof['historical_exit_code'] is None and proof['exact_exit_time'] is None and
            proof['exit_data_not_inferred'] is True and clock['historical_exit_code'] is None and clock['exact_exit_time'] is None,
            'unrecorded resume01 exits not inferred')
    audits = proof['native_audits']
    require(len(audits)==2 and all(a['identities']==clock['identities'] and a['same_owned_remaining']==0 for a in audits) and
            all(type(a['monotonic']) in (float,int) and math.isfinite(a['monotonic']) for a in audits) and
            type(clock['captured_monotonic']) in (float,int) and math.isfinite(clock['captured_monotonic']) and
            clock['wall_origin_monotonic']<=clock['captured_monotonic']<=audits[0]['monotonic']<=audits[1]['monotonic']<=time.monotonic() and
            audits[1]['monotonic']-audits[0]['monotonic']>=2, 'resume01 two separated native zero audits')
    original.audit_absence(clock['identities'])
    return float(audits[-1]['monotonic']-clock['wall_origin_monotonic'])


def replay(plan, files, ancestor):
    return original.replay(plan, files, ancestor=ancestor, execution_source=PARENT_SOURCE,
                           prior_invocations=plan['carry']['actual_invocations'], prior_charge=plan['carry']['charged_bytes'])


def verify_marker(plan, files, started):
    require('control/one-shot.json' in files, 'resume01 one-shot missing')
    marker = read_json(Path(plan['control_root'])/'one-shot.json', limit=4096, expected=files['control/one-shot.json'])
    require(marker['run_id']==PARENT_RUN and marker['source']==PARENT_SOURCE and marker['review']==PARENT_REVIEW_DIGEST and
            marker['driver_pid']==started['driver_pid'] and marker['uid']==os.getuid(), 'resume01 marker identity')


def evidence_state(root, started_path, clock_path, proof_path):
    plan = parent_plan(root)
    started = read_json(started_path, expected=STARTED_EXPECTED)
    clock = read_json(clock_path, expected=CLOCK_EXPECTED)
    proof = read_json(proof_path)
    wall = verify_stop(proof, started, clock)  # Reject live parent before inventory/replay.
    ancestor, ancestor_data = ancestor_store(root, plan)
    files = original.inventory(plan);verify_marker(plan, files, started)
    state = replay(plan, files, ancestor)
    require(wall>=max(plan['carry']['wall_seconds'],state['observed_wall_floor']) and
            proof['native_audits'][0]['monotonic']>=state['observed_monotonic_floor'], 'resume01 proof precedes retained observer')
    original.audit_absence(clock['identities'])
    require(files==original.inventory(plan), 'resume01 inventory changed during receipt')
    store = original.ResumeStore(plan, files, state, '')
    original.validate_saved_signals(store)
    return plan, state, files, wall, ancestor_data


def make_receipt(root, started_path, clock_path, proof_path):
    plan, state, files, wall, ancestor = evidence_state(root, started_path, clock_path, proof_path)
    return _receipt(plan, state, files, wall, ancestor, started_path, clock_path, proof_path)


def _receipt(plan, state, files, wall, ancestor, started_path, clock_path, proof_path):
    completed = set(state['records'])
    orphan = sorted(Path(n).name[7:-5] for n in files if re.fullmatch(r'output/record-[0-9a-f]{64}\.json',n) and Path(n).name[7:-5] not in completed)
    return dict(schema_version='h4-resume01-continuation-receipt-v1', parent_run=PARENT_RUN,
        parent_source=PARENT_SOURCE, parent_review=PARENT_REVIEW, dataset_source=DATASET_SOURCE,
        ancestor_receipt=plan['continuation']['receipt'], output_root=plan['output_root'], control_root=plan['control_root'],
        started=path_ref(started_path), clock_context=path_ref(clock_path), native_stop_proof=path_ref(proof_path), files=files,
        ledger_head=state['ledger_head'], ledger_versions=state['ledger_versions'], imported_wrappers=len(completed),
        carry=dict(actual_invocations=state['total_actual_invocations'], charged_bytes=state['charged_bytes'], wall_seconds=wall),
        new_parent_actual_invocations=len(state['reservations']),
        unresolved_reservations=sorted(k for k,r in state['reservations'].items() if r['status']=='RESERVED'),
        unresolved_ancestor_reservations=ancestor['unresolved_reservations'], orphan_records=orphan,
        pending_files=sorted(n for n in files if n.endswith('.pending')), historical_exit_code=None, exact_exit_time=None,
        exit_data_not_inferred=True, discarded_work_refunded=False, automatic_retry=False)


def load_receipt(root, descriptor):
    require(set(descriptor)=={'path','bytes','sha256'}, 'resume01 receipt reference')
    data = read_json(descriptor['path'], expected={k:descriptor[k] for k in ('bytes','sha256')})
    require(data['schema_version']=='h4-resume01-continuation-receipt-v1' and data['parent_run']==PARENT_RUN and
            data['parent_source']==PARENT_SOURCE and data['parent_review']==PARENT_REVIEW and data['dataset_source']==DATASET_SOURCE,
            'resume01 receipt lineage')
    for name, pinned in (('started',STARTED_EXPECTED),('clock_context',CLOCK_EXPECTED),('native_stop_proof',None)):
        ref = data[name];require(set(ref)=={'path','bytes','sha256'}, 'resume01 proof reference')
        if pinned is not None:
            require({k:ref[k] for k in ('bytes','sha256')}==pinned, 'pinned resume01 identity/clock receipt')
        read_json(ref['path'], expected={k:ref[k] for k in ('bytes','sha256')})
    # Reconstruct the full byte/ledger/native receipt, including unresolved costs.
    paths = [data[k]['path'] for k in ('started','clock_context','native_stop_proof')]
    plan, state, files, wall, ancestor = evidence_state(root, *paths)
    rebuilt = _receipt(plan, state, files, wall, ancestor, *paths)
    require(data==rebuilt, 'resume01 receipt/cumulative carry mismatch')
    store = original.ResumeStore(plan, data['files'], state, descriptor['sha256'])
    original.audit_absence(read_json(data['clock_context']['path'],expected=CLOCK_EXPECTED)['identities'])
    return store, data


def verify_runtime(permit):
    """Immutable ancestry and explicit launch permission precede fresh gates."""
    from . import gates
    from .launch_binding import reference
    from .continuation_binding import authorize, verify_roots
    plan = permit.plan;authorize(plan, permit.authorization)
    require(plan['run_id']==NEXT_RUN and plan['source_commit']!=PARENT_SOURCE, 'fresh resume02 execution SOURCE')
    parent = parent_plan(plan['source_root']);root = Path(plan['source_root'])
    for field in ('inputs','input_root','generation_freeze_digest','templates','contract_plan_fingerprint',
                  'environment_fingerprint','compiler_fingerprint','environment_profile','compiler_profile',
                  'library_cache_profile','memory_pressure_profile','gaussian_synthesis_profile','requested_workers','cpu_proposal'):
        require(plan[field]==parent[field], 'resume02 changed science/compiler/resource binding: '+field)
    from .wall_time import settings as wall_settings
    _cap,trace_override,unlimited=wall_settings(plan,permit.authorization)
    require({k:v for k,v in plan['caps'].items() if k!='wall_seconds'}==
            {k:v for k,v in parent['caps'].items() if k!='wall_seconds'}, 'resume02 other resource caps unchanged')
    if not unlimited:require(plan['caps']==parent['caps'],'resume02 unchanged legacy caps')
    authority = reference(root, permit.authorization['continuation_amendment']['authority'])
    require(authority.get('schema_version')=='h4-user-resume01-continuation-authority-v1' and
            authority.get('resume_launch_approved') is True and authority.get('preparation_only') is False and
            authority.get('parent_source')==PARENT_SOURCE and authority.get('parent_review')==PARENT_REVIEW and
            authority.get('dataset_source')==DATASET_SOURCE and authority.get('validated_speedup_source')==SPEEDUP_SOURCE and
            authority.get('preserve_dataset_seeds') is True and authority.get('preserve_consumed_budget') is True and
            isinstance(authority.get('instruction'),str) and bool(authority['instruction'].strip()),
            'resume02 explicit launch authority (preparation alone is insufficient)')
    prefix = 'src/trottertracks/resource_applicability/h4_geometry/'
    inherited = {p for p in parent['source_hashes'] if p.startswith(prefix)}
    require({p for p in plan['source_hashes'] if p.startswith(prefix)}==
            inherited|{prefix+'instruction_encoding.py',prefix+'continuation_resume01.py',prefix+'wall_time.py'}, 'closed resume02 runtime source set')
    for path in inherited:
        digest = parent['source_hashes'][path]
        require(sha(gates.git_blob(root,PARENT_SOURCE,path))==digest, 'resume01 SOURCE blob')
        name = Path(path).name
        if name in CONTROL_CHANGES:
            if name in ('execution.py','resources.py','observer.py'):
                verify_time_source(root,plan['source_commit'],path)
            continue
        expected = sha(gates.git_blob(root,SPEEDUP_SOURCE,path)) if name in ('circuits.py','signal.py') else digest
        require(plan['source_hashes'][path]==expected, 'resume02 changed validated scientific/monitor source')
    require(plan['source_hashes'][prefix+'instruction_encoding.py']==sha(gates.git_blob(root,SPEEDUP_SOURCE,prefix+'instruction_encoding.py')),
            'validated instruction encoding unchanged')
    store, data = load_receipt(root, plan['continuation']['receipt'])
    require(plan['carry']==data['carry'] and plan['continuation']['imported_wrappers']==data['imported_wrappers'],
            'resume02 actual receipt/carry binding')
    grandparent = original.parent_plan(root)
    readonly = [data[k]['path'] for k in ('started','clock_context','native_stop_proof','ancestor_receipt')]
    ancestor_data = original.read_json(data['ancestor_receipt']['path'],expected={k:data['ancestor_receipt'][k] for k in ('bytes','sha256')})
    readonly += [ancestor_data[k]['path'] for k in ('started','clock_context','native_stop_proof')]
    verify_roots(plan,parent,extra_readonly=readonly);verify_roots(plan,grandparent,extra_readonly=readonly)
    storage = plan['storage'];fresh = grandparent['storage']
    from .observer import FRAME_CAP,TERMINAL_RESERVE
    trace_delta=0 if trace_override is None else 2*(trace_override-((72*3600+2)*FRAME_CAP+TERMINAL_RESERVE))
    require(storage['prior_charge_bytes']==plan['carry']['charged_bytes'] and
            storage['cumulative_charge_bound']>=fresh['cumulative_charge_bound']+plan['carry']['charged_bytes']+trace_delta and
            storage['required_bytes']>=fresh['required_bytes'] and storage['required_inodes']>=fresh['required_inodes'],
            'resume02 storage includes cumulative carry and one fresh bound')
    return store
