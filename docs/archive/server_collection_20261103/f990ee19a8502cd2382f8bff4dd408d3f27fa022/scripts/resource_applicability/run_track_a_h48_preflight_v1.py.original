#!/usr/bin/env python3
"""Bound H4 panel / H8 input-only launch. No full H8 or automatic next stage.

Each phase requires a separately pinned manifest and grant. All actual jobs
run under the existing own-run H6 supervisor and its first-STOP/cleanup logic.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h48_preflight_contract_v1 import verify_binding,verify_candidate_runtime
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.h6_matched_execution_v2 import Progress
from trottertracks.resource_applicability.h6_server_resources_v1 import (
    process_identity,install_parent_guard,fresh_resources)

def bounded_json(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>8*2**20:raise ValueError('BOUND_JSON')
    return json.loads(path.read_text())

def make_resource_trace(writer,started):
    from trottertracks.resource_applicability.h6_matched_execution_v3 import ResourceTraceV3
    return ResourceTraceV3(writer.write,lambda:time.monotonic()-started)

def execute_worker(m,grant,directory):
    from trottertracks.resource_applicability.ax2b_limits import install_worker_limits
    expected=bounded_json(directory/'task_binding.json')
    parent=bounded_json(Path(m['output'])/'parent_identity.json')
    install_parent_guard(parent)
    if expected['manifest_digest']!=digest(m) or expected['threads']!=1 or len(expected['cpus'])!=1:
        raise ValueError('WORKER_TASK_BINDING')
    if expected['cpus'] not in m['proposed_resources']['cost_cpu_sets']:raise ValueError('UNGRANTED_CPU')
    # The coordinator is pinned to its own CPUs; a child restores only its
    # explicitly granted CPU after checking the parent's identity and binding.
    os.sched_setaffinity(0,set(expected['cpus']))
    if sorted(os.sched_getaffinity(0))!=expected['cpus']:raise ValueError('WORKER_CPU_BINDING')
    install_worker_limits(expected['cpus'][0],expected['address_space_bytes'],32*2**20)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    writer=AtomicWriter(directory,byte_cap=32*2**20,diagnostics_cap=m['plan']['caps_proposed']['diagnostics'])
    writer.write('worker_claim.json',{'pid':os.getpid(),'identity':process_identity(os.getpid()),
        'scope':m['scope'],'source_commit':m['source_commit'],'actual_affinity':sorted(os.sched_getaffinity(0)),
        'manifest_sha256':digest(m),'approved_by_user':grant['approved_by_user']})
    progress=Progress(writer,cap=m['plan']['caps_proposed']['progress_records'])
    trace=make_resource_trace(writer,progress.start)
    status='COMPLETE';reason=None;port=None
    try:
        if m['scope']=='H8_STATE':
            from trottertracks.resource_applicability.ax2b_h6_input_generation_watchdog_v1 import InputProgress
            from trottertracks.resource_applicability.h8_saved_state_port_v1 import SavedStatePort
            ip=InputProgress(writer,cap=m['plan']['caps_proposed']['progress_records'])
            port=SavedStatePort(m,writer,ip,grant_sha256=digest(grant));port.execute()
        elif m['scope']=='H8_INPUT':
            from trottertracks.resource_applicability.ax2b_h6_input_generation_watchdog_v1 import InputProgress
            from trottertracks.resource_applicability.h8_preflight_input_port_v1 import InputGenerationPort
            (directory/'scratch').mkdir(exist_ok=False)
            ip=InputProgress(writer,cap=m['plan']['caps_proposed']['progress_records'])
            def save_raw_decomposition(two,**kwargs):
                from openfermion import low_rank_two_body_decomposition
                from trottertracks.resource_applicability.h8_preflight_input_port_v1 import atomic_npz
                import numpy as np
                values=low_rank_two_body_decomposition(two,**kwargs)
                weights,blocks,correction,truncation=values
                raw=atomic_npz(writer,'raw_df.npz',{'weights':np.asarray(weights),
                    'blocks':np.asarray(blocks),'correction':np.asarray(correction),
                    'truncation':np.array(truncation)},expanded_cap=16*2**20)
                writer.write('raw_df_receipt.json',raw)
                return values
            port=InputGenerationPort(m,writer,ip,grant_sha256=digest(grant),decomposer=save_raw_decomposition);port.execute()
        elif m['scope']=='H4_SIGNAL':
            from trottertracks.resource_applicability.h4_common_panel_port_v1 import signal_port
            with trace.bind():signal_port(ROOT,m,writer,progress)
        else:
            task=next(x['task'] for x in m['registrations'] if x['index']==expected['index'])
            if digest(task)!=digest(expected['task']):raise ValueError('COST_TASK_CHANGED')
            from trottertracks.resource_applicability.h4_common_panel_port_v1 import cost_port
            with trace.bind():cost_port(ROOT,m,task,writer,progress)
        verify_binding(ROOT,m,grant,worker=True)
    except Exception as e:
        status='STOP';reason=type(e).__name__+':'+str(e)[:512]
    writer.write('worker_terminal.json',{'status':status,'reason':reason,'scope':m['scope'],
        'source_commit':m['source_commit'],'retry':False,'H8_full_campaign_authorized':False,
        'calls_attempted':dict(port.calls.used) if port is not None else None,
        'calls_completed':dict(port.completed) if port is not None else None},terminal=True)
    return 0 if status=='COMPLETE' else 1

def validate_result(directory,task,manifest):
    terminal=bounded_json(directory/'worker_terminal.json')
    if terminal['status']!='COMPLETE' or terminal['scope']!=manifest['scope'] or terminal['source_commit']!=manifest['source_commit']:
        raise ValueError('INCOMPLETE_PHASE_RESULT')
    if manifest['scope']=='H4_COST':
        result=bounded_json(directory/'cost_result.json')
        if result['task']!=task:raise ValueError('TASK_RESULT_BINDING')
        policies={'symmetric_directional','ordinary'} if task['ordinary'] else {'symmetric_directional'}
        if len(result['rows'])!=2*len(policies) or {(x['axis'],x['control']) for x in result['rows']}!={(a,c) for a in ('cosine','sine') for c in policies}:
            raise ValueError('PAIRED_COST_RESULT')
        for row in result['rows']:
            if row['cell_id']!=task['cell']['id'] or row['replica']!=task['replica'] or row['seed']!=task['seed']:
                raise ValueError('ROW_IDENTITY')
            if row['compiler_hash']!='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b':
                raise ValueError('FIXED_COMPILER_HASH')
            if set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'} or any(type(v) is not int or v<0 for v in row['metrics'].values()):
                raise ValueError('ROW_METRICS')
        calls=result['calls_attempted']
        if calls['compile']!=2*len(policies) or calls['trajectory']!=int(bool(task['cell']['R'])):
            raise ValueError('ACTUAL_CALL_RESULT')
    elif manifest['scope']=='H4_SIGNAL':
        if bounded_json(directory/'signal_result.json')['signal_count']!=16:raise ValueError('SIGNAL_COVERAGE')
    else:
        r=bounded_json(directory/'snapshot_receipt.json')
        expected=0 if manifest['scope']=='H8_STATE' else 1
        if r['loader_roundtrip_checked'] is not True or r['calls_completed']['integral_build']!=expected or r['calls_completed']['df_decomposition']!=expected:
            raise ValueError('H8_INPUT_RECEIPT')
        if manifest['scope']=='H8_STATE':
            from decimal import Decimal
            projection=bounded_json(directory/'projection_receipt.json')
            if projection['status']!='PASS_ENGINEERING' or Decimal(projection['eta_N_hartree_decimal']['16'])>Decimal('9.9e-11'):
                raise ValueError('H8_WEIGHTED_PROJECTION_GATE')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--grant-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',type=Path)
    args=p.parse_args()
    from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
    if file_hash(args.manifest)!=args.manifest_sha256 or file_hash(args.grant)!=args.grant_sha256:
        raise ValueError('PINNED_PLAN_AUTH_BYTES')
    m=bounded_json(args.manifest);grant=bounded_json(args.grant);verify_binding(ROOT,m,grant,worker=bool(args.worker))
    if not args.execute:
        print('BOUND_METADATA_CHECK_ONLY');return 0
    if args.worker:
        return execute_worker(m,grant,args.worker.resolve())
    out=Path(m['output'])
    if out.exists():raise ValueError('ONE_SHOT_OUTPUT_USED')
    native_profile=verify_candidate_runtime(ROOT)
    fresh=fresh_resources(m['proposed_resources'],m['observation_policy'],out)
    lock=out.parent/(out.name+'.one_shot_lock')
    with lock.open('x') as f:f.write(json.dumps({'manifest_sha256':digest(m),'source':m['source_commit'],'retry':False})+'\n')
    out.mkdir(exist_ok=False)
    writer=AtomicWriter(out,byte_cap=m['base_manifest']['plan']['caps_proposed']['output_bytes'])
    writer.write('manifest.json',m);writer.write('grant.json',grant);writer.write('fresh_resources.json',fresh)
    writer.write('fresh_native_profile_subset.json',native_profile)
    writer.write('parent_identity.json',process_identity(os.getpid()))
    from trottertracks.resource_applicability.h6_parallel_execution_v1 import supervise
    def command(row,directory,slot):
        return [sys.executable,'-I','-P','-B',str(ROOT/'scripts/resource_applicability/run_track_a_h48_preflight_v1.py'),
          '--manifest',str(args.manifest.resolve()),'--grant',str(args.grant.resolve()),
          '--manifest-sha256',args.manifest_sha256,'--grant-sha256',args.grant_sha256,'--execute','--worker',str(directory)]
    # Keep all intended worker CPUs available until each child applies its own affinity.
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,32*2**30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,set(m['proposed_resources']['coordinator_cpu_set']))
    report=supervise(ROOT,out,m,command=command,admission=lambda d,t:validate_result(d,t,m))
    writer.write('execution_terminal.json',report,terminal=True)
    print(json.dumps({'scope':m['scope'],'output':str(out),'first_stop_reason':report['first_stop_reason'],
        'completed':len(report['completed_indices']),'dispatched':len(report['dispatched'])}))
    return 0 if report['first_stop_reason'] is None and report['all_dispatched_reaped'] else 1

if __name__=='__main__':raise SystemExit(main())
