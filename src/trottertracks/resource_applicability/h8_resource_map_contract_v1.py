"""Frozen H8 core registry, independent selection, complete-only receipts.

The user's later launch instruction supersedes PREPARATION ONLY solely for
this adopted core. No q16, transfer, conditional rare pilot, or auto retry.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_h6_contract import primitive_time_schedule
from .h8_signal_preflight_contract_v1 import plan as preflight_plan, input_identity, INPUT_SHA
from .h48_preflight_contract_v1 import profile, observation_policy
from .h8_fast_retry_contract_v1 import source_paths as previous_sources
from .h6_matched_accounting_v1 import eligible, confirmation_ids

RUNNER = 'scripts/resource_applicability/run_track_a_h8_resource_map_v1.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h8_resource_map_v1.py'
LAUNCHER = 'scripts/resource_applicability/launch_track_a_h8_resource_map_v1.py'
MODULES = ('h8_resource_map_contract_v1', 'h8_resource_map_port_v1',
           'h8_resource_map_optimization_v1', 'h8_resource_map_execution_v1')
AUTHORIZATION = '並列化高速化してから計算を始めてください。計算を実際に始めるところまで進めて大丈夫です'
NAMESPACE = 'H8_CORE_RESOURCE_MAP_V1_20261013'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
CPUS = [20, 21, 24, 25, 26, 27, 28, 29]


def cells():
    out = []
    def add(method, p, q, formula='2nd', r=None, K=None):
        name = f'{method}_p{p}_q{q}_{formula}'+(f'_r{r}_K{K}' if r else '')
        out.append(dict(id=name, method=method, prefix=p, q=q, formula=formula,
                        r=r, R=q*r if r else None, K=K))
    for q in (1, 2, 4, 8):
        for p in (7, 14, 21): add('B0', p, q)
        for formula in ('2nd', '4th'): add('B1', 28, q, formula)
        for p in (7, 14, 21):
            for r in (1, 2, 4):
                for K in (2, 4): add('B2', p, q, r=r, K=K)
    return out


def signal_groups():
    return [dict(index=i, task=dict(kind='SIGNAL', q=q,
            cell_ids=[c['id'] for c in cells() if c['q'] == q]))
            for i, q in enumerate((1, 2, 4, 8))]


def coverage(cs):
    pairs = set(); rows = []
    for c in cs:
        s = primitive_time_schedule(c, T=.8)
        s['registered_validation_times_v2'] = s['unique_primitive_times']
        rows.append(dict(cell_id=c['id'], schedule=s))
        pairs.update(('one' if i == 0 else str(i-1), t) for i, t in s['unique_primitive_times'])
    return dict(cells=rows, physical_times=sorted(pairs),
                primitive_actions=3*len(pairs), primitive_probe_count=3)


def call_budget():
    cs = cells(); groups = [[c for c in cs if c['q'] == q] for q in (1, 2, 4, 8)]
    primitive = sum(coverage(g)['primitive_actions'] for g in groups)
    exact_tail_exponentials = sum(c['R'] for c in cs if c['R'])
    # Primitive oracle + ideal reference 3 actions per group, and three
    # truncated references per prefix/group; each expm action has its own cap.
    expm_actions = primitive + 12 + 36 + exact_tail_exponentials
    reference = expm_actions*100000 + 64 + 2*sum(c['R']*(c['K']+1) for c in cs if c['R'])
    return dict(signal=92, normalization=72, compile=1708, trajectory=832,
        occurrence=4208, control_probe=376, primitive=primitive,
        reference_matvec=reference, reference_exponential_actions=expm_actions,
        reference_matvec_per_action=100000, actual_history_is_known=False)


def plan():
    p = deepcopy(preflight_plan()); cs = cells(); budget = call_budget()
    p.update(schema='h8_full_core_resource_map_plan_v1', kind='H8_CORE_RESOURCE_MAP',
        cells=cs, coverage=coverage(cs), H8_full_campaign_authorized=True,
        full_comparison_contract_adopted=True, maximum_cost_groups=852,
        maximum_primary_wrappers=1704, maximum_wrappers=1708)
    p['cost_rule'].update(exploratory_random_trajectories=8,
        confirmation_random_trajectories=32, seed_namespace=NAMESPACE,
        confirmation_top_per_epsilon=2, confirmation_union_max=8,
        acquire='all eligible at any epsilon under unchanged empirical u',
        selection_tie_break='G_RZ point then lexical cell_id',
        validation='all exploration replica0, ordinary anchors; no validation disabled')
    p['sensitivity_cells'] = ['B1_p28_q1_4th', 'B2_p7_q2_2nd_r4_K2']
    p['caps_proposed'].update({k: budget[k] for k in ('compile','trajectory','occurrence',
        'control_probe','primitive','reference_matvec','reference_matvec_per_action')})
    p['caps_proposed'].update(untranspiled_instructions=4000000,
        transpiled_instructions=10000000, output_bytes=32*2**30, log_bytes=128*2**20,
        progress_records=1000, diagnostics=1000, total_wall_seconds=None,
        deterministic_actions_per_cell=100000)
    p['scope_exclusions'] = ['q16','absolute_prefix5_transfer','conditional_rare_pilot',
        'new_SCF_DF_state','automatic_retry','next_stage']
    p['validation_algorithm'] = 'checked joint branches, unsafe structure two-branch fallback'
    return p


def task_seed(source, cell, phase, replica):
    if phase not in ('exploration','confirmation') or type(replica) is not int or replica < 0:
        raise ValueError('H8_CORE_SEED_SCOPE')
    key = f'{NAMESPACE}|{source}|{INPUT_SHA}|{cell}|{phase}|{replica}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def cost_tasks(signals, source, phase, selected=None):
    if {s['cell']['id'] for s in signals} != {c['id'] for c in cells()}:
        raise ValueError('H8_CORE_ALL_SIGNALS_REQUIRED')
    by = {s['cell']['id']: s for s in signals}; out = []
    sensitivity=set(plan()['sensitivity_cells'])
    for c in cells():
        s = by[c['id']]
        if not eligible(s) or (phase == 'confirmation' and c['id'] not in (selected or [])):
            continue
        if phase == 'confirmation' and c['method'] != 'B2':
            raise ValueError('H8_CONFIRMATION_B2_ONLY')
        n = 32 if phase == 'confirmation' else 8 if c['R'] else 1
        for replica in range(n):
            out.append(dict(index=len(out), task=dict(kind='COST', cell=c, cell_id=c['id'],
                phase=phase, replica=replica, seed=task_seed(source,c['id'],phase,replica),
                validate=phase=='exploration' and replica==0,
                ordinary=phase=='exploration' and replica==0 and c['id'] in sensitivity,
                expected_signal=s['signals']['corrected'])))
    if len(out) > (596 if phase=='exploration' else 256):
        raise ValueError('H8_CORE_GROUP_CAP')
    return out


def selection(signals, rows, source):
    selected = confirmation_ids(signals, rows, plan())
    return dict(selected_ids=selected, registrations=cost_tasks(signals,source,'confirmation',selected),
        exploration_rows_digest=digest(rows), signal_digest=digest(signals),
        rule='top2 G_RZ per epsilon; lexical tie; union<=8; independent32',
        fixed_before_confirmation=True, exploration_confirmation_pooled=False)


def resources():
    return dict(assigned_cpus=sorted(CPUS+[22,23]), coordinator_cpu_set=[22,23],
        cost_cpu_sets=[[c] for c in CPUS], cost_workers=8, cost_threads=1,
        cost_address_space_bytes=32*2**30, worker_rss_stop_bytes=24*2**30,
        normal_worker_rss_bytes=9*2**30, heavy_worker_rss_bytes=24*2**30,
        coordinator_address_space_bytes=8*2**30, coordinator_hard_address_space_bytes=32*2**30,
        aggregate_rss_bytes=76*2**30, coordinator_rss_reserve_bytes=2*2**30,
        aggregate_margin_bytes=2*2**30, job_ram_budget_bytes=96*2**30,
        memory_is_reservation=False, gpu=False)


def observation():
    p = observation_policy()
    p.update(journal_bytes=2**30, journal_segment_bytes=8*2**20,
        minimum_storage_bytes=64*2**30, minimum_free_inodes=200000,
        output_inode_cap=100000, output_bytes=32*2**30,
        rotation='exclusive sealed segments; bounded total; never overwrite evidence')
    return p


def source_paths(root):
    return sorted(set(previous_sources(root)) | {RUNNER, PREPARER, LAUNCHER} |
        {'src/trottertracks/resource_applicability/'+x+'.py' for x in MODULES})


def carry():
    return dict(lineages=dict(H6=dict(known_compile=1716,reserved_compile=20,
        known_trajectory=842,reserved_trajectory=10), H4_supplement=dict(compile=532,trajectory=256),
        H8_feasibility=dict(compile=21,trajectory=4,control_probe=44,occurrence=15,
            old_failed_compile=1,old_failed_control=4), H8_signal_preflight=dict(
            primitive=1164,reference_matvec=63717,signal=8), H8_input=dict(
            integral_build=1,df_decomposition=1,successful_solver_matvec=61,
            failed_solver_matvec_actual=None,failed_solver_reservation=10000)),
        listed_known_plus_reserved_compile=2289,listed_known_plus_reserved_trajectory=1112,
        projected_compile=3997,projected_trajectory=1944, production_H4_included=False,
        historical_unknown_actual=None, refund=0, reset=False, feasibility_reuse_savings=0)


def make_manifest(root, source, input_dir, output):
    root=Path(root); output=Path(output)
    return dict(schema='h8_resource_map_manifest_v1', kind='H8_CORE_RESOURCE_MAP', source_commit=source,
        source_hashes={p:file_hash(root/p) for p in source_paths(root)}, plan=plan(),
        input_identity=input_identity(input_dir), profile=profile(), compiler_hash=COMPILER,
        cumulative_carry=carry(), call_budget=call_budget(), output=str(output),
        proposed_resources=resources(), observation_policy=observation(),
        technical_sealed=True, candidate_registry_sealed=True,
        scientific_conditions_changed=False, automatic_retry=False, next_stage_authorized=False,
        complete_only_resume_supported=True, resume_automatically=False,
        feasibility_binding=dict(path='artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/measured_wrapper_rows_v1.json',
            sha256=file_hash(root/'artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/measured_wrapper_rows_v1.json'),
            reuse_savings=0,scope='deterministic canonical/metric continuity; no trajectory reuse'),
        initial_registrations=signal_groups(), authorization_basis=AUTHORIZATION)


def grant(m, approved=False):
    return dict(schema='h8_resource_map_grant_v1', approved_by_user=approved,
        runtime_authorization=approved, authorization_basis=AUTHORIZATION,
        source_commit=m['source_commit'], manifest_digest=digest(m),
        output=m['output'], allowed_cpus=resources()['assigned_cpus'] if approved else [],
        one_shot=True, automatic_retry=False, next_stage_authorized=False,
        scope='92_SIGNAL_ELIGIBLE_EXPLORATION_TOP2_UNION_INDEPENDENT32_CORE_ONLY')


def verify(root, m, g, *, committed=True):
    root=Path(root)
    if g != grant(m,True) or m.get('authorization_basis') != AUTHORIZATION:
        raise ValueError('H8_CORE_EXPLICIT_GRANT')
    if (m['plan'] != plan() or m['proposed_resources'] != resources()
        or m['observation_policy'] != observation() or m['call_budget'] != call_budget()
        or m['cumulative_carry'] != carry() or m['profile'] != profile()
        or m['initial_registrations'] != signal_groups() or m['compiler_hash'] != COMPILER):
        raise ValueError('H8_CORE_FIXED_CONTRACT')
    if m['input_identity'] != input_identity(m['input_identity']['directory']):
        raise ValueError('H8_CORE_INPUT_CHANGED')
    f=m['feasibility_binding']
    if (f['path']!='artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/measured_wrapper_rows_v1.json'
        or file_hash(root/f['path'])!=f['sha256'] or f['reuse_savings']!=0):
        raise ValueError('H8_FEASIBILITY_BINDING')
    if 'resume_parent' in m:
        ref=m['resume_parent']
        for p,h in ref['file_hashes'].items():
            if file_hash(p)!=h:raise ValueError('H8_RESUME_PARENT_CHANGED')
        previous=json.loads(Path(ref['manifest_path']).read_text())
        if any(previous[k]!=m[k] for k in ('source_commit','source_hashes','input_identity','profile','compiler_hash','plan','call_budget')):
            raise ValueError('H8_RESUME_CONTRACT_CHANGED')
    out=Path(m['output'])
    if not out.is_absolute() or not out.is_relative_to('/home/AbeHiromu') or str(out.resolve())!=str(out):
        raise ValueError('H8_CORE_OUTPUT_PATH')
    if set(m['source_hashes']) != set(source_paths(root)):
        raise ValueError('H8_CORE_SOURCE_CLOSURE')
    for p,h in m['source_hashes'].items():
        if file_hash(root/p)!=h: raise ValueError('H8_CORE_SOURCE_CHANGED:'+p)
        if committed and hashlib.sha256(subprocess.check_output(['git','show',m['source_commit']+':'+p],cwd=root)).hexdigest()!=h:
            raise ValueError('H8_CORE_SOURCE_COMMIT:'+p)


def admission(directory, task, source, *, reaped):
    """A terminal alone is insufficient: kernel/native reap receipt is required."""
    if reaped is not True: raise ValueError('H8_ADMISSION_REAP_REQUIRED')
    d=Path(directory); result=json.loads((d/'cost_result.json').read_text())
    term=json.loads((d/'worker_terminal.json').read_text())
    tr=json.loads((d/'trajectory.json').read_text());claim=json.loads((d/'worker_claim.json').read_text())
    if (term['status']!='COMPLETE' or term['reason'] is not None or term['source_commit']!=source
        or result['task']!=task or tr['task']!=task or claim['task_digest']!=digest(task)
        or claim['source_commit']!=source or tr['event_digest']!=digest(tr['events'])):
        raise ValueError('H8_ADMISSION_TASK_SOURCE')
    if task['cell']['R']:
        if len(tr['events'])!=task['cell']['q'] or any(len(x)!=task['cell']['r'] for x in tr['events']):
            raise ValueError('H8_ADMISSION_EVENT_SHAPE')
        if tr['observed_orders']!=[e['taylor_order'] for x in tr['events'] for e in x]:
            raise ValueError('H8_ADMISSION_ORDER_RECEIPT')
    elif tr['events'] is not None:
        raise ValueError('H8_ADMISSION_DETERMINISTIC_EVENTS')
    expected={(p,a) for p in (('symmetric_directional','ordinary') if task['ordinary'] else ('symmetric_directional',)) for a in ('cosine','sine')}
    if len(result['rows'])!=len(expected) or {(x['control'],x['axis']) for x in result['rows']}!=expected:
        raise ValueError('H8_ADMISSION_PAIRED_AXES')
    for row in result['rows']:
        if (any(row[k]!=task[k] for k in ('cell_id','phase','replica','seed'))
            or row['event_digest']!=tr['event_digest'] or row['compiler_hash']!=COMPILER
            or len(row['fingerprint'])!=64 or set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'}
            or any(type(x) is not int or x<0 for x in row['metrics'].values())
            or json.loads((d/f"wrapper_{row['control']}_{row['axis']}.json").read_text())!=row):
            raise ValueError('H8_ADMISSION_METRIC_COMPILER')
    if task['validate']:
        for p in {x[0] for x in expected}:
            v=json.loads((d/f'validation_{p}.json').read_text())
            if not 0<=v['max_error']<=1e-9 or v['logical_control_probes']!=4:
                raise ValueError('H8_ADMISSION_CONTROL')
    if result['calls_attempted']['compile']!=len(expected):raise ValueError('H8_ADMISSION_COMPILE_COUNT')
    return result


def checkpoint(directory, task, source, native_receipt):
    if native_receipt.get('exit_code')!=0 or native_receipt.get('reaped') is not True:
        raise ValueError('H8_CHECKPOINT_TERMINAL')
    result=admission(directory,task,source,reaped=True)
    return dict(task=task,task_digest=digest(task),source_commit=source,
        files={p.name:file_hash(p) for p in sorted(Path(directory).glob('*.json'))},
        native_receipt=native_receipt,result_digest=digest(result),complete_only=True)


def resume_result(directory, task, source, receipt):
    if receipt['task']!=task or receipt['source_commit']!=source or not receipt['complete_only']:
        raise ValueError('H8_RESUME_SOURCE_TASK')
    if any(file_hash(Path(directory)/p)!=h for p,h in receipt['files'].items()):
        raise ValueError('H8_RESUME_BYTES')
    result=admission(directory,task,source,reaped=receipt['native_receipt'].get('reaped'))
    if digest(result)!=receipt['result_digest']:raise ValueError('H8_RESUME_RESULT')
    return result


def bind_resume_parent(m, directory, native_exit):
    """Manual new-grant preparation only. Never resets held failed reservations."""
    parent=Path(directory).resolve();native_exit=Path(native_exit).resolve()
    if not parent.is_relative_to('/home/AbeHiromu') or not native_exit.is_relative_to('/home/AbeHiromu'):
        raise ValueError('H8_RESUME_PRIVATE_PATH')
    old=json.loads((parent/'manifest.json').read_text())
    end=json.loads((parent/'execution_terminal.json').read_text());native=json.loads(native_exit.read_text())
    if native.get('wait_reaped') is not True or type(native.get('exit_code')) is not int:
        raise ValueError('H8_RESUME_NATIVE_EXIT')
    identity=json.loads((parent/'coordinator_identity.json').read_text())
    from .h6_server_resources_v1 import process_identity
    try:
        if process_identity(identity['pid'])['start_ticks']==identity['start_ticks']:
            raise ValueError('H8_RESUME_PARENT_ALIVE')
    except FileNotFoundError:pass
    if any(old[k]!=m[k] for k in ('source_commit','source_hashes','input_identity','profile','compiler_hash','plan','call_budget')):
        raise ValueError('H8_RESUME_SAME_SOURCE_REQUIRED')
    inventory={str(parent/'manifest.json'):file_hash(parent/'manifest.json'),
        str(parent/'execution_terminal.json'):file_hash(parent/'execution_terminal.json'),
        str(native_exit):file_hash(native_exit)}
    accepted={}
    for phase in ('signal','exploration','confirmation'):
        pd=parent/phase;terminal=pd/'execution_terminal.json'
        if not terminal.is_file():continue
        t=json.loads(terminal.read_text())
        if not t['all_dispatched_reaped'] or t['cleanup_errors']:raise ValueError('H8_RESUME_CLEANUP')
        inventory[str(terminal)]=file_hash(terminal);accepted[phase]={}
        registrations=json.loads((pd/'registered_tasks.json').read_text())['registrations']
        for row in registrations:
            d=pd/f"group_{row['index']:04d}";reap=d/'native_reap_receipt.json'
            if not reap.is_file():continue
            nr=json.loads(reap.read_text())
            if nr['exit_code']!=0 or not nr['reaped']:continue
            if phase=='signal':
                wt=json.loads((d/'worker_terminal.json').read_text());sr=json.loads((d/'signal_result.json').read_text())
                ss=json.loads((d/'signal_summary.json').read_text())
                if (wt['status']!='COMPLETE' or wt['source_commit']!=m['source_commit'] or wt['task']!=row['task']
                    or {s['cell']['id'] for s in ss['signals']}!=set(row['task']['cell_ids'])):
                    raise ValueError('H8_RESUME_SIGNAL_BINDING')
                result=dict(sr,signals=ss['signals'])
            else:
                cp=json.loads((pd/f"checkpoint_{row['index']:04d}.json").read_text())
                result=resume_result(d,row['task'],m['source_commit'],cp)
            for p in d.glob('*.json'):inventory[str(p)]=file_hash(p)
            accepted[phase][str(row['index'])]=dict(task=row['task'],result=result,directory=str(d))
    m['resume_parent']=dict(directory=str(parent),manifest_path=str(parent/'manifest.json'),
        file_hashes=inventory,accepted=accepted,parent_ledger=end['ledger'],
        native_exit_path=str(native_exit),old_grant_reused=False,automatic_resume=False)
    return m
