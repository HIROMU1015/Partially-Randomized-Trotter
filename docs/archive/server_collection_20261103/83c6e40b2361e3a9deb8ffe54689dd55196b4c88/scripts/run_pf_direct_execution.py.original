"""Two separately granted PF contracts; owned workers, wait4 and no retry."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a
import pf_direct_execution as e


def rss(pid):
    try:
        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
        return next(int(row.split()[1])*1024 for row in lines if row.startswith('VmRSS:'))
    except (OSError,StopIteration):return 0


def task_estimate(row):
    # Admission estimate only, from saved H8 p21/q8/S4 peak 26.252 GiB,
    # scaled by native structural site counts plus interpreter baseline.
    # No hard RSS/CPU/worker limit and no RZ/cost model evaluation.
    n,_=c.SYSTEM_LAYOUTS[row['system']]
    sites=(int(row['prefix'])+1)*int(row['q'])*(1 if row['PF']=='2nd' else 3)*n*n
    anchor=22*8*3*16*16
    return math.ceil((.75+2*26.252*sites/anchor)*2**30)


def run(seal,*,execution_api=e,worker_script='scripts/pf_direct_production_worker.py',estimator=task_estimate,priority=None):
    api=execution_api
    started=time.monotonic();contracts={};pending=[];active={};observed_peak=0
    for group in api.GROUPS:
        specpath=seal/(group+'_spec.json');grantpath=seal/(group+'_grant.json')
        spec,grant=c.read_json(specpath),c.read_json(grantpath)
        # Fail before creating output or one-shot marker if grant is invalid.
        _,selected=api.authorize(spec,grant)
        root=Path(spec['output_root']);c.require(not root.exists() and not root.is_symlink(),'ONE_SHOT_OUTPUT_USED')
        root.mkdir(parents=True,exist_ok=False)
        api.save(root/'marker.json',dict(contract_id=group,SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),one_shot=True))
        journal=root/'ledger.jsonl';journal.touch(exist_ok=False)
        state=dict(spec=spec,grant=grant,root=root,specpath=specpath,grantpath=grantpath,
            used={k:0 for k in spec['call_caps']},workers=[],rejected=[],admitted=0,start=time.monotonic())
        contracts[group]=state
        try:
            gate=api.gate(spec,grant);api.save(root/'fresh_gate.json',gate)
        except Exception as error:
            api.save(root/'STOP.json',dict(reason=type(error).__name__+':'+str(error),phase='PRE_DISPATCH',automatic_retry=False))
            state['rejected']=spec['selected_task_ids'];continue
        for (system,cid),pair in api.candidate_pairs(selected).items():
            task=dict(group=group,system=system,cid=cid,pair=pair,memory_estimate=estimator(pair[0]))
            if any(r['task_id'] in spec['missing_saved_oracles'] for r in pair):
                api.append(journal,dict(status='STOP_TASK',task_ids=[r['task_id'] for r in pair],reason='NO_SAVED_CONTROL_ORACLE',calls=0))
                state['rejected'] += [r['task_id'] for r in pair]
            else:pending.append(task)
    # Prioritize registered H6 epsilon .005; then cheap structural footprints.
    pending.sort(key=priority or (lambda t:(0 if (t['system'],t['cid'])==('H6_DF19','B0_p11_q4_2nd') else 1,t['memory_estimate'],t['system'],t['cid'])))
    api.PRIVATE.joinpath('run01').mkdir(exist_ok=True)
    resource_log=api.PRIVATE/'run01/resource_observations.jsonl'
    api.save(api.PRIVATE/'run01/admission_policy.json',getattr(api,'ADMISSION_POLICY',dict(policy='LIVE_PERMITTED_CPU_AND_MEMORY_RESERVATIONS',
        historical_anchor='H8 p21/q8 S4 saved peak 26.252 GiB',structural_scale='(prefix+1)*q*(1 or 3)*n_qubits^2',
        estimate='0.75 GiB + twice scaled historical peak; admission estimate, never process limit',
        CPU_RAM_worker_limits=None,transient_PSI='observation only',no_retry=True)))
    last_report=0.;last_observation=0.
    while pending or active:
        now=time.monotonic();fresh=api.resource_snapshot()
        current_rss=sum(rss(pid) for pid in active);observed_peak=max(observed_peak,current_rss)
        slots=[dict(estimate=t['memory_estimate'],rss=rss(pid)) for pid,t in active.items()]
        admit=api.choose_admissions(pending,slots,fresh)
        for task in admit:
            state=contracts[task['group']];spec=state['spec'];pair=task['pair']
            caps=c.call_caps(pair)
            c.require(all(state['used'][k]+v<=spec['call_caps'][k] for k,v in caps.items()),'CONTRACT_CALL_CAP')
            api.append(state['root']/'ledger.jsonl',dict(event='BEFORE_WORKER_RESERVED',task_ids=[r['task_id'] for r in pair],
                calls=caps,nonrefundable=True,monotonic=time.monotonic()))
            for key,value in caps.items():state['used'][key]+=value
            output=state['root']/(task['system']+'__'+task['cid'])
            log=(state['root']/(task['system']+'__'+task['cid']+'.log')).open('x')
            env=os.environ.copy()
            for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):env[key]='1'
            env['QISKIT_PARALLEL']='FALSE';env['PYTHONDONTWRITEBYTECODE']='1'
            cmd=[spec['systems'][task['system']]['python'],'-B',str(api.ROOT/worker_script),
                '--spec',str(state['specpath']),'--grant',str(state['grantpath']),
                '--system',task['system'],'--candidate',task['cid'],'--output',str(output)]
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=api.ROOT)
            stat=Path(f'/proc/{proc.pid}/stat').read_text()
            start_ticks=stat[stat.rfind(')')+2:].split()[19]
            api.append(state['root']/'ledger.jsonl',dict(event='WORKER_STARTED',pid=proc.pid,
                start_ticks=start_ticks,system=task['system'],candidate_id=task['cid'],command=cmd,monotonic=time.monotonic()))
            active[proc.pid]=dict(task,process=proc,log=log,output=output,start=time.monotonic(),start_ticks=start_ticks)
            state['admitted']+=1;pending.remove(task)
        for pid,task in list(active.items()):
            got,status,usage=os.wait4(pid,os.WNOHANG)
            if not got:continue
            code=os.waitstatus_to_exitcode(status);task['process'].returncode=code;task['log'].close()
            state=contracts[task['group']];output=task['output'];resultpath=output/'result.json'
            witness=dict(pid=pid,start_ticks=task['start_ticks'],system=task['system'],candidate_id=task['cid'],exit_code=code,
                reaped=True,cleanup_errors=[],wall_seconds=time.monotonic()-task['start'],
                user_CPU_seconds=usage.ru_utime,system_CPU_seconds=usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,
                result_sha256=api.sha(resultpath) if resultpath.is_file() else None)
            accepted=[]
            try:
                c.require(code==0 and resultpath.is_file(),'WORKER_NATIVE_FAILURE')
                result=c.read_json(resultpath)
                c.require(len(result['rows'])==2 and result['calls']==c.call_caps(task['pair']),'PAIR_RESULT_AND_CALLS')
                for raw in result['rows']:
                    expected=task['system']+':'+task['cid']+':'+raw['axis']
                    c.require(raw['task_id']==expected,'RESULT_TASK_IDENTITY')
                    row=a.accept_owned_measurement(raw,dict(witness,task_id=expected,result_digest=c.digest(raw)),expected)
                    a.validate_measurement(row,c.COMPILER,raw['axis']);accepted.append(row)
                api.save(output/'accepted_measurements.json',dict(rows=accepted,witness=witness))
                witness['status']='PASS'
            except Exception as error:
                witness['status']='STOP_TASK';witness['reason']=type(error).__name__+':'+str(error)
                state['rejected'] += [r['task_id'] for r in task['pair']]
            api.save(output/'owned_worker_exit.json' if output.exists() else state['root']/(task['system']+'__'+task['cid']+'_exit.json'),witness)
            state['workers'].append(witness);del active[pid]
            api.append(state['root']/'ledger.jsonl',dict(event='NATIVE_REAP_AND_ACCEPTANCE',witness=witness,reservations_refunded=False))
        if now-last_observation>=10:
            api.append(resource_log,dict(fresh,owned_workers=len(active),pending_pairs=len(pending),sampled_owned_RSS_sum=current_rss))
            last_observation=now
        if now-last_report>=30:
            print(json.dumps(dict(running=len(active),pending_pairs=len(pending),completed_pairs={k:len(v['workers']) for k,v in contracts.items()},
                stopped_axes={k:len(v['rejected']) for k,v in contracts.items()},elapsed_seconds=now-started)),flush=True);last_report=now
        if pending and not active and not admit:
            # Cannot fit even one current task; preserve its non-execution reason, no fabricated cap.
            for task in pending:
                state=contracts[task['group']];state['rejected'] += [r['task_id'] for r in task['pair']]
                api.append(state['root']/'ledger.jsonl',dict(event='STOP_NOT_ADMITTED',task_ids=[r['task_id'] for r in task['pair']],
                    reason='INSUFFICIENT_EFFECTIVE_RESOURCES_FOR_ESTIMATE',memory_estimate=task['memory_estimate'],resource=fresh))
            pending=[]
        if active:time.sleep(.5)
    summaries={}
    for group,state in contracts.items():
        success=sum(w['status']=='PASS' for w in state['workers'])*2
        final=None
        try:final=api.gate(state['spec'],state['grant'])
        except Exception as error:final=dict(status='STOP',reason=str(error))
        summary=dict(contract_id=group,SOURCE=state['spec']['SOURCE'],authorized_axes=len(state['spec']['selected_task_ids']),
            accepted_axes=success,stopped_or_unperformed_axes=len(state['rejected']),worker_processes=len(state['workers']),
            all_workers_reaped=all(w['reaped'] for w in state['workers']),wall_seconds=time.monotonic()-state['start'],
            CPU_seconds=sum(w['user_CPU_seconds']+w['system_CPU_seconds'] for w in state['workers']),
            largest_single_worker_peak_RSS_bytes=max([0]+[w['peak_RSS_bytes'] for w in state['workers']]),
            calls_reserved=state['used'],final_gate=final,status='COMPLETE_MANDATORY_STOP' if success==len(state['spec']['selected_task_ids']) and final['status']=='PASS' else 'PARTIAL_OR_STOP_MANDATORY_STOP',
            next_stage_authorized=False,automatic_retry=False)
        api.save(state['root']/'summary.json',summary)
        if not (state['root']/'STOP.json').exists():api.save(state['root']/'STOP.json',dict(status=summary['status'],next_stage_authorized=False,retry=False))
        summaries[group]=summary
    api.save(api.PRIVATE/'run01/terminal.json',dict(contracts=summaries,wall_seconds=time.monotonic()-started,
        peak_sampled_sum_owned_worker_RSS_bytes=observed_peak,active_owned_workers=0,
        peak_RSS_definition='sum of RSS of owned live workers sampled; distinct from wait4 single-process maxima',mandatory_STOP=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed-directory',type=Path,required=True)
    args=p.parse_args();run(args.sealed_directory)
