"""Read-only saved metadata/byte audit. No molecular arrays, RNG or compiler."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import h46_pr_pilot_contract as c
from h46_pr_pilot_repair import canonical_tau,CUTOFF_POLICY,POLICY
from h46_m5_contract import core

def arithmetic(provider):
    space={'math':math,'require_integer_count':lambda value,**kwargs:value}
    for filename,name in [('rte.py','step_taylor_truncation_residual_bound'),
                          ('pr2_matched_accuracy_m1_execution.py','_explicit_cutoff_tolerance')]:
        path=Path(provider)/'src/trotterlib'/filename
        node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
    return space['step_taylor_truncation_residual_bound'],space['_explicit_cutoff_tolerance']

def audit(old_config):
    cfg=c.read(old_config);c.require(cfg['stage']=='H6_DESIGN_PILOT','SAVED_H6_ONLY')
    c.require(sys.executable==cfg['runtime']['python'],'FIXED_H6_ARITHMETIC_RUNTIME')
    oldroot=Path(next(x['path'] for x in cfg['sealed_files'] if x['role']=='pilot_SOURCE:scripts/h46_pr_pilot_contract.py')).parents[1]
    code='import sys;sys.path.insert(0,sys.argv[1]);import h46_pr_pilot_contract as c;v=c.read(sys.argv[2]);c.authorize(v);c.consumed(v)'
    probe=subprocess.run([sys.executable,'-B','-c',code,str(oldroot/'scripts'),str(Path(old_config).resolve())],capture_output=True,text=True)
    c.require(probe.returncode==0,'ORIGINAL_SEALS:'+probe.stderr[-1500:])
    exitrow=c.read(cfg['supervisor']['exit']);c.require(exitrow['native_exit']==1 and exitrow['reaped'],'ORIGINAL_STOP_EXIT_REAP')
    signal_cfg=c.read(cfg['origin_signal_config']);receipt=Path(signal_cfg['output'])/'H6_context_receipt.json'
    contexts=c.read(receipt);c.require(contexts['SOURCE']==signal_cfg['SOURCE'],'FIXED_SIGNAL_CONTEXT_SOURCE')
    failed=next(t for t in cfg['tasks'] if t['candidate']['formula']=='4th(m5_best)' and t['candidate']['q']==2 and t['sample']==0)
    out=Path(cfg['output']);stop_path=out/failed['id']/'STOP.json';stop=c.read(stop_path)
    c.require(stop['reason']=='finite Taylor cutoff does not meet truncation_tolerance.','EXACT_CUTOFF_STOP')
    pieces=sorted((c.read(p) for p in (out/failed['id']).glob('piece_law_*.json')),key=lambda x:x['piece_request_index'])
    c.require(len(pieces)==3,'SAVED_THREE_SUCCESSFUL_PIECES')
    lam=contexts['prefixes']['5']['lambda_R']
    candidates=set()
    for value in [lam]+[p['law']['dimensionless_step_time']/(p['signed_step_time']/p['rte_steps']) for p in pieces]:
        candidates.add(value)
        for direction in (-math.inf,math.inf):
            x=value
            for _ in range(12):x=math.nextafter(x,direction);candidates.add(x)
    matched=sorted(x for x in candidates if all(x*(p['signed_step_time']/p['rte_steps'])==p['law']['dimensionless_step_time'] for p in pieces))
    c.require(matched==[lam],'UNIQUE_LOCAL_LAMBDA_MATCHES_ALL_SAVED_PIECES')
    # The first saved positive step is monotonic in lambda; the neighboring
    # search endpoints already lie outside its rounding cell.
    first=pieces[0];u=first['signed_step_time']/first['rte_steps'];tau=first['law']['dimensionless_step_time']
    c.require(u>0 and min(candidates)*u<tau<max(candidates)*u,'LAMBDA_NEIGHBORHOOD_COVERS_FIRST_ROUNDING_CELL')
    prep=NS(rte_preparation=NS(symbolic_tail=NS(lambda_r=lam)))
    schedule=core.schedule(failed['candidate']['formula'],2,'ABS_TIME_PER_OUTER',32)
    bound,tolerance=arithmetic(cfg['provider_root']);rows=[]
    for index,step in enumerate(schedule[:4]):
        oldtau=(lam*step['signed_time'])/step['rte_steps']
        newtau=canonical_tau(prep,step['signed_time'],step['rte_steps'])
        residual=bound(newtau,4);old_tol=tolerance(oldtau,4);new_tol=tolerance(newtau,4)
        rows.append(dict(piece=index,signed_time=step['signed_time'],rte_steps=step['rte_steps'],
          old_tau=oldtau,factory_and_fixed_tau=newtau,old_tolerance=old_tol,
          factory_residual_bound=residual,fixed_tolerance=new_tol,
          old_check_PASS=residual<=old_tol,fixed_check_PASS=residual<=new_tol))
    c.require([r['old_check_PASS'] for r in rows]==[True,True,True,False]
      and all(r['fixed_check_PASS'] for r in rows),'EXACT_STOP_PATTERN_AND_FIXED_ARITHMETIC')
    registry=c.read(cfg['registry_path']);complete=[];remaining=[];law_rows=0;group_counts={};archives=0
    result_set=[]
    for task in cfg['tasks']:
        p=out/task['id']/'result.json'
        if not p.exists():remaining.append(task['id']);continue
        result=c.read(p)
        c.require(result['status']=='UNIT_COST_PASS_NOT_ACCOUNTING' and result['task']==task
          and result['SOURCE']==cfg['SOURCE'] and result['plan_sha256']==cfg['plan_sha256']
          and len(result['checks'])==8 and all(x['PASS'] for x in result['checks']),
          'COMPLETE_PAIR_NATIVE_IDENTITY')
        byte_audit=c.storage.audit_task(cfg,task)
        c.require(byte_audit==c.read(p.parent/'storage_audit.json')
          and c.sha(p.parent/'storage_audit.json')==result['storage_audit_sha256'],'EXACT_COMPLETED_PAIR_BYTES')
        archives+=byte_audit['archive_count'];candidate=task['candidate']
        steps=core.schedule(candidate['formula'],candidate['q'],candidate['r_policy'],candidate['r_value'])
        for i,step in enumerate(steps):
            piece=c.read(p.parent/f'piece_law_{i}.json')
            c.require(piece['registry_seed_PRIVATE']==registry[task['id']][f"{step['outer']}:{step['piece']}"]
              and piece['signed_step_time']==step['signed_time'] and piece['rte_steps']==step['rte_steps']
              and piece['law']['finite_taylor_order']==4 and not piece['conditioned'],'ORIGINAL_REGISTERED_PIECE_METADATA')
            c.require(piece['law']['dimensionless_step_time']==canonical_tau(prep,step['signed_time'],step['rte_steps']),
              'UNCHANGED_COMPLETED_GENERATION_TAU')
            law_rows+=1
        complete.append(task['id']);group_counts[candidate['id']]=group_counts.get(candidate['id'],0)+1
        result_set.append(dict(task=task['id'],sha256=c.sha(p)))
    c.require(len(complete)==74 and len(remaining)==322 and len(set(complete+remaining))==396,'FIXED_REGISTRATION_PARTITION')
    return dict(status='READ_ONLY_METADATA_AND_BYTE_REPAIR_AUDIT_PASS',old_SOURCE=cfg['SOURCE'],
      old_plan_sha256=cfg['plan_sha256'],old_config_sha256=c.sha(old_config),context_receipt_sha256=c.sha(receipt),
      original_STOP_sha256=c.sha(stop_path),original_exit_sha256=c.sha(cfg['supervisor']['exit']),
      original_full_stage_STOP_unchanged=True,matrix_lambda_and_saved_piece_lambda=lam,
      failure_cause='FLOAT_OPERATION_ORDER_IN_TAU_WITH_NEXTAFTER_ONLY_ACCEPTANCE_MARGIN',
      lambda_difference_required_for_this_failure=False,diagnostic_rows=rows,cutoff_policy=CUTOFF_POLICY,
      proposed_performance_policy=POLICY,complete_pairs=74,complete_axes=148,verified_completed_archives=archives,
      saved_generation_piece_rows_checked=law_rows,result_set_sha256=c.digest(sorted(result_set,key=lambda x:x['task'])),
      remaining_registered_trajectories=322,proposed_new_compile_axes_maximum=644,
      completed_by_candidate_metadata_only=group_counts,old_compile_reservations_retained=172,
      reuse_authorized=False,new_execution_authorized=False,new_valid_grants=0,
      new_signal=0,new_sampling=0,new_compile=0,new_scientific_state_actions=0,
      pilot_statistics=0,shot_accounting=0,ranking=0,AD03='DEFERRED',mandatory_STOP=True),dict(completed_ids=complete,remaining_ids=remaining)

def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--private-output',required=True)
    p.add_argument('--public-output',required=True);args=p.parse_args()
    result,partition=audit(args.old_config)
    private=Path(args.private_output);public=Path(args.public_output)
    c.require(not private.exists() and not public.exists(),'NEW_AUDIT_OUTPUTS_ONLY')
    private.parent.mkdir(parents=True,exist_ok=True);public.parent.mkdir(parents=True,exist_ok=True)
    c.save(private,partition);c.save(public,result)
    print(json.dumps(dict(status=result['status'],complete_pairs=result['complete_pairs'],remaining=result['remaining_registered_trajectories'],
      old_pattern=[r['old_check_PASS'] for r in result['diagnostic_rows']],new_pattern=[r['fixed_check_PASS'] for r in result['diagnostic_rows']],scientific_calls=0)))

if __name__=='__main__':main()
