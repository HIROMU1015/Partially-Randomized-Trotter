#!/usr/bin/env python3
"""One monitored run of the seven remaining cases with frozen old seeds."""
import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_handoff_contract_v1 import (
    RUNNER,verify_binding,read,COMPILER)
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.h6_matched_execution_v2 import Progress
from trottertracks.resource_applicability.h6_server_resources_v1 import (
    process_identity,process_memory,install_parent_guard,fresh_resources,THREAD_KEYS)
from trottertracks.resource_applicability.h48_preflight_contract_v1 import verify_candidate_runtime


def admit(directory,task,m):
    # Reuse the existing paired wrapper/metric/seed/call checks without running
    # its main/worker. Its byte-fixed source remains in the closure.
    f=runpy.run_path(str(ROOT/'scripts/resource_applicability/run_track_a_h8_signal_preflight_v1.py'))['validate_result']
    f(directory,task,m)
    r=read(directory/'cost_result.json')
    if (r.get('validation_algorithm')!='reused evolution branches and X/Y suffix'
        or r.get('full_evolution_simulations')!=(4 if task['ordinary'] else 2)
        or r.get('expensive_full_wrapper_simulations')!=0):
        raise ValueError('HANDOFF_VALIDATION_ALGORITHM')
    for policy in ('symmetric_directional','ordinary') if task['ordinary'] else ('symmetric_directional',):
        v=read(directory/f'validation_{policy}.json')
        if (v.get('validation_method_changed') is not True or v.get('full_evolution_simulations')!=2
            or v.get('reused_axis_states')!=2 or v.get('logical_control_probes')!=4):
            raise ValueError('HANDOFF_CONTROL_METHOD_RECEIPT')
    return r


def compiled_count_from_failure(error):
    """Inspect an already-created circuit before traceback locals are cleared.

    This records a scalar only; it never compiles, loads an array or persists
    the circuit. Missing context stays null, as for the old failed run.
    """
    tb=error.__traceback__
    while tb is not None:
        code=tb.tb_frame.f_code
        if (code.co_name=='compile_owned' and
            code.co_filename.endswith('/resource_applicability/h6_matched_memory_v3.py')):
            compiled=tb.tb_frame.f_locals.get('compiled')
            if compiled is not None:
                return len(compiled.data)
        tb=tb.tb_next
    return None


def worker(m,g,directory):
    from trottertracks.resource_applicability.ax2b_limits import install_worker_limits
    from trottertracks.resource_applicability.h6_matched_execution_v3 import ResourceTraceV3
    b=read(directory/'task_binding.json');r=m['proposed_resources'];slot=b['slot']
    if type(slot) is not int or not 0<=slot<3:
        raise ValueError('HANDOFF_WORKER_SLOT')
    expected=dict(manifest_digest=digest(m),index=b['index'],task=b['task'],slot=slot,
        cpus=r['cost_cpu_sets'][slot],threads=1,address_space_bytes=r['cost_address_space_bytes'])
    if (digest(b)!=digest(expected) or digest({'index':b['index'],'task':b['task']}) not in [digest(x) for x in m['registrations']]
        or directory.parent!=Path(m['output']) or directory.name!=f"cost_exploration_{b['index']:04d}"):
        raise ValueError('HANDOFF_WORKER_BINDING')
    install_parent_guard(read(Path(m['output'])/'parent_identity.json'))
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL')!='FALSE':
        raise ValueError('HANDOFF_WORKER_THREADS')
    os.sched_setaffinity(0,set(b['cpus']))
    install_worker_limits(b['cpus'][0],b['address_space_bytes'],32*2**20)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    w=AtomicWriter(directory,byte_cap=32*2**20,diagnostics_cap=5000)
    w.write('worker_claim.json',dict(source_commit=m['source_commit'],scope=m['scope'],
        identity=process_identity(os.getpid()),affinity=sorted(os.sched_getaffinity(0)),
        rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),threads=1,manifest_sha256=digest(m),
        original_seed_source=m['seed_source_commit']))
    p=Progress(w,cap=5000);trace=ResourceTraceV3(w.write,lambda:time.monotonic()-p.start)
    status='COMPLETE';reason=None
    try:
        from trottertracks.resource_applicability.h8_fast_cost_port_v1 import cost_port
        with trace.bind():cost_port(ROOT,m,b['task'],w,p,trace)
        verify_binding(ROOT,m,g,worker=True)
    except Exception as exc:
        status='STOP';reason=type(exc).__name__+':'+str(exc)[:512]
        if str(exc)=='TRANSPILED_INSTRUCTION_CAP':
            try:w.write('instruction_cap_failure.json',dict(
                transpiled_instruction_count=compiled_count_from_failure(exc),
                transpiled_instruction_cap=m['plan']['caps_proposed']['transpiled_instructions'],
                source_commit=m['source_commit'],task=b['task'],compiler_hash=COMPILER,
                metric_measurement_started=False,retry=False),terminal=True)
            except Exception:pass
        try:trace.failure(exc)
        except Exception:pass
    w.write('worker_terminal.json',dict(status=status,reason=reason,source_commit=m['source_commit'],
        scope=m['scope'],memory=process_memory(os.getpid()),mandatory_stop=True,retry=False,
        H8_full_campaign_authorized=False),terminal=True)
    return 0 if reason is None else 1


def combined_result(out,m,report,w):
    groups=[m['accepted_heavy']['result']];new=[]
    for reg in m['registrations']:
        result=admit(out/f"cost_exploration_{reg['index']:04d}",reg['task'],m)
        groups.append(result);new.append(result)
    calls={k:sum(r['calls_attempted'][k] for r in groups) for k in groups[0]['calls_attempted']}
    if any(calls[k]!=v for k,v in dict(compile=20,trajectory=4,occurrence=15,control_probe=40,primitive=0,reference_matvec=0).items()):
        raise ValueError('HANDOFF_COMBINED_CALLS')
    rows=[row for r in groups for row in r['rows']]
    keys={(x['cell_id'],x['control'],x['axis']) for x in rows}
    if len(groups)!=8 or len(rows)!=20 or len(keys)!=20:
        raise ValueError('HANDOFF_COMBINED_CASES')
    w.write('combined_feasibility_result.json',dict(status='H8_EIGHT_CASE_FAST_HANDOFF_COMPLETE_MANDATORY_STOP',
        source_commit=m['source_commit'],old_heavy_source=m['old_parent']['source_commit'],
        old_heavy_input_receipt=m['accepted_heavy'],signal_receipt=m['signal_binding'],
        total_groups=8,total_wrappers=20,new_groups=7,new_wrappers=18,
        calls_attempted=calls,new_calls_attempted={k:sum(r['calls_attempted'][k] for r in new) for k in calls},
        rows=rows,compiler_hash=COMPILER,input_identity=m['input_identity'],
        preserved_carry=m['cumulative_carry'],validation_methods={'old_heavy':'independent full X/Y simulation','remaining':m['validation_method']},
        source_and_method_provenance_distinguished=True,numerical_allowance_certified=False,
        population_mean_certified=False,resource_summary=report,retry=False,refund=0,reset=False,
        H8_full_campaign_authorized=False),terminal=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--grant-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',type=Path)
    a=p.parse_args()
    if file_hash(a.manifest)!=a.manifest_sha256 or file_hash(a.grant)!=a.grant_sha256:
        raise ValueError('HANDOFF_PINNED_FINAL_BYTES')
    m=read(a.manifest);g=read(a.grant);verify_binding(ROOT,m,g,worker=bool(a.worker))
    if not a.execute:print('BOUND_FAST_HANDOFF_METADATA_ONLY');return 0
    if a.worker:return worker(m,g,a.worker.resolve())
    out=Path(m['output'])
    if out.exists() or (out.parent/(out.name+'.one_shot_lock')).exists():
        raise ValueError('HANDOFF_ONE_SHOT_OUTPUT_USED')
    native=verify_candidate_runtime(ROOT)
    fresh=fresh_resources(m['proposed_resources'],m['observation_policy'],out)
    with (out.parent/(out.name+'.one_shot_lock')).open('x') as f:f.write(json.dumps(dict(manifest_sha256=digest(m),SOURCE=m['source_commit'],retry=False))+'\n')
    out.mkdir(exist_ok=False);w=AtomicWriter(out,byte_cap=2*2**30)
    for name,value in [('manifest.json',m),('grant.json',g),('fresh_resources.json',fresh),
                        ('fresh_native_profile_subset.json',native),('parent_identity.json',process_identity(os.getpid()))]:w.write(name,value)
    r=m['proposed_resources'];resource.setrlimit(resource.RLIMIT_AS,(r['coordinator_address_space_bytes'],r['coordinator_hard_address_space_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**30,2*2**30))
    os.sched_setaffinity(0,set(r['coordinator_cpu_set']))
    from trottertracks.resource_applicability.h6_parallel_execution_v1 import supervise
    def command(row,directory,slot):
        return [sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(a.manifest.resolve()),
            '--grant',str(a.grant.resolve()),'--manifest-sha256',a.manifest_sha256,
            '--grant-sha256',a.grant_sha256,'--execute','--worker',str(directory)]
    report=supervise(ROOT,out,m,command=command,admission=lambda d,t:admit(d,t,m))
    if report['first_stop_reason'] is None:
        try:
            verify_binding(ROOT,m,g)
            if report['completed_indices']!=[r['index'] for r in m['registrations']]:raise ValueError('FAST_SEVEN_INCOMPLETE')
            combined_result(out,m,report,w)
        except Exception as exc:report['first_stop_reason']='FINAL_ADMISSION_ERROR:'+type(exc).__name__+':'+str(exc)[:256]
    w.write('execution_terminal.json',report,terminal=True)
    print(json.dumps(dict(scope=m['scope'],SOURCE=m['source_commit'],output=str(out),
                         completed=report['completed_indices'],first_stop_reason=report['first_stop_reason'])))
    return 0 if report['first_stop_reason'] is None and report['all_dispatched_reaped'] else 1


if __name__=='__main__':raise SystemExit(main())
