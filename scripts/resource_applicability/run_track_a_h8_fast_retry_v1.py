#!/usr/bin/env python3
"""Approved fast S4/q8 retry, or seven following cases; never auto retry."""
import argparse,json,os,resource,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_retry_contract_v1 import verify_binding,RUNNER
from trottertracks.resource_applicability.h8_fast_handoff_contract_v1 import read,COMPILER
from trottertracks.resource_applicability.h4_common_panel_port_v1 import bound_function
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.h6_server_resources_v1 import fresh_resources,process_identity
from trottertracks.resource_applicability.h48_preflight_contract_v1 import verify_candidate_runtime


def legacy_metadata():
    return runpy.run_path(str(ROOT/'scripts/resource_applicability/run_track_a_h8_fast_handoff_v1.py'))


def combined(out,m,report,writer,admit):
    groups=[m['accepted_retry_heavy']['result']]
    for r in m['registrations']:groups.append(admit(out/f"cost_exploration_{r['index']:04d}",r['task'],m))
    new_calls={k:sum(g['calls_attempted'][k] for g in groups) for k in groups[0]['calls_attempted']}
    failed=m['preserved_failure']['known_calls']
    cumulative={k:new_calls[k]+failed[k] for k in new_calls}
    if new_calls!=dict(compile=20,control_probe=40,trajectory=4,occurrence=15,primitive=0,reference_matvec=0):
        raise ValueError('FAST_RETRY_COMPLETE_NEW_CALLS')
    if cumulative!=dict(compile=21,control_probe=44,trajectory=4,occurrence=15,primitive=0,reference_matvec=0):
        raise ValueError('FAST_RETRY_CUMULATIVE_CALLS')
    rows=[row for g in groups for row in g['rows']]
    if len(groups)!=8 or len(rows)!=20 or len({(r['cell_id'],r['axis'],r['control']) for r in rows})!=20:
        raise ValueError('FAST_RETRY_COMPLETE_WRAPPERS')
    writer.write('combined_feasibility_result.json',dict(status='H8_FAST_RETRY_EIGHT_CASES_COMPLETE_MANDATORY_STOP',
        SOURCE=m['source_commit'],total_groups=8,total_wrappers=20,new_successful_calls=new_calls,
        preserved_failed_calls=failed,cumulative_calls=cumulative,rows=rows,input_identity=m['input_identity'],
        compiler_hash=COMPILER,preserved_carry=m['cumulative_carry'],approved_caps=m['approved_caps'],
        retry_approval='one explicit retry of failed heavy; no automatic retry',refund=0,reset=False,
        H8_full_campaign_authorized=False,numerical_allowance_certified=False,population_mean_certified=False,
        resource_summary=report),terminal=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--grant-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',type=Path)
    a=p.parse_args()
    if file_hash(a.manifest)!=a.manifest_sha256 or file_hash(a.grant)!=a.grant_sha256:
        raise ValueError('FAST_RETRY_PINNED_BYTES')
    m,g=read(a.manifest),read(a.grant);verify_binding(ROOT,m,g,worker=bool(a.worker))
    if not a.execute:print('FAST_RETRY_BOUND_METADATA_ONLY');return 0
    legacy=legacy_metadata()
    if a.worker:
        # Preserve worker limits, telemetry, sampler, optimized port and native
        # exception cleanup; replace only its plan/auth validator in this
        # private function namespace. Old module source/globals are untouched.
        return bound_function(legacy['worker'],{'verify_binding':verify_binding})(m,g,a.worker.resolve())
    out=Path(m['output'])
    if out.exists() or (out.parent/(out.name+'.one_shot_lock')).exists():raise ValueError('FAST_RETRY_ONE_SHOT_USED')
    native=verify_candidate_runtime(ROOT);fresh=fresh_resources(m['proposed_resources'],m['observation_policy'],out)
    with (out.parent/(out.name+'.one_shot_lock')).open('x') as f:f.write(json.dumps(dict(SOURCE=m['source_commit'],manifest_sha256=digest(m),automatic_retry=False))+'\n')
    out.mkdir(exist_ok=False);w=AtomicWriter(out,byte_cap=2*2**30)
    for name,value in [('manifest.json',m),('grant.json',g),('fresh_resources.json',fresh),
                       ('fresh_native_profile_subset.json',native),('parent_identity.json',process_identity(os.getpid()))]:w.write(name,value)
    r=m['proposed_resources'];resource.setrlimit(resource.RLIMIT_AS,(r['coordinator_address_space_bytes'],r['coordinator_hard_address_space_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,set(r['coordinator_cpu_set']))
    from trottertracks.resource_applicability.h6_parallel_execution_v1 import supervise
    def command(row,directory,slot):
        return [sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(a.manifest.resolve()),
            '--grant',str(a.grant.resolve()),'--manifest-sha256',a.manifest_sha256,
            '--grant-sha256',a.grant_sha256,'--execute','--worker',str(directory)]
    admit=legacy['admit']
    report=supervise(ROOT,out,m,command=command,admission=lambda d,t:admit(d,t,m))
    if report['first_stop_reason'] is None:
        try:
            verify_binding(ROOT,m,g)
            if report['completed_indices']!=[r['index'] for r in m['registrations']]:raise ValueError('FAST_RETRY_SCOPE_INCOMPLETE')
            if m['scope']=='REMAINING_AFTER_FAST_RETRY':combined(out,m,report,w,admit)
        except Exception as e:report['first_stop_reason']='FINAL_BINDING_ERROR:'+type(e).__name__+':'+str(e)[:256]
    w.write('execution_terminal.json',report,terminal=True)
    print(json.dumps({'scope':m['scope'],'SOURCE':m['source_commit'],'output':str(out),
        'completed':report['completed_indices'],'first_stop_reason':report['first_stop_reason']}))
    return 0 if report['first_stop_reason'] is None and report['all_dispatched_reaped'] else 1


if __name__=='__main__':raise SystemExit(main())
