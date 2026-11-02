"""One saved p7-versus-PR comparison; stdlib only, explicit new grant required."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
INPUT_SOURCE='a73ce48410264305f257ebe5d7c1c2c2c527fe47'
SCOPE='p7_measured_cost_single_comparison'
RESOURCES=dict(processes=1,threads=1,AS_bytes=512*2**20,RSS_bytes=256*2**20,
               CPU_seconds=30,wall_seconds=60,output_bytes=2*2**20)
CAPS=dict(cost_reaccounting=1,new_compile=0,new_signal=0,new_reference=0,new_scalar=0,
          new_sampling=0,new_shot_calculation=0,new_fit=0,candidate_reselection=0,V2=0)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def execute(grant_path,output):
    grant=read(grant_path);started=time.monotonic()
    if (grant.get('schema')!='track_a_v1_p7_measured_reaccount_grant_v1'
            or grant.get('approved_by_user') is not True or grant.get('runtime_authorization') is not True
            or grant.get('scope')!=SCOPE or grant.get('input_source_commit')!=INPUT_SOURCE
            or grant.get('one_shot') is not True or grant.get('retry') is not False
            or grant.get('next_stage_authorized') is not False
            or grant.get('call_caps')!=CAPS or grant.get('resources')!=RESOURCES
            or grant.get('execution_script_sha256')!=sha(__file__)
            or grant.get('output')!=str(output) or not 0<=started-grant['gate_monotonic_seconds']<=30):
        raise ValueError('NEW_APPROVED_FRESH_REACCOUNT_GRANT_REQUIRED')
    cpu=grant['cpu']
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):raise ValueError('CPU_BINDING')
    paths={}
    for item in grant['inputs']:
        path=Path(item['path'])
        if path.is_symlink() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('INPUT_CHANGED:'+item['name'])
        if item['name'] in paths:raise ValueError('DUPLICATE_INPUT')
        paths[item['name']]=path
    if set(paths)!={'plan','signal','comparisons','validation','terminal','cleanup','compile_audit','contract'}:
        raise ValueError('EXACT_INPUT_SET_REQUIRED')
    for path in (output,output.parent/'reaccount_ONE_SHOT'):
        if not path.is_absolute() or path.resolve()!=path or path.exists() or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_EXCLUSIVE_PATH_REQUIRED')
    os.sched_setaffinity(0,{cpu})
    resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['AS_bytes'],RESOURCES['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(RESOURCES['CPU_seconds'],RESOURCES['CPU_seconds']))
    def timeout(*args):raise ValueError('WALL_OR_STOP_SIGNAL')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    write(output.parent/'reaccount_ONE_SHOT',dict(input_source_commit=INPUT_SOURCE,grant_sha256=sha(grant_path)))
    output.mkdir(mode=0o700)
    reserved=False
    try:
        spec=importlib.util.spec_from_file_location('_p7_contract_for_accounting',paths['contract'])
        contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
        plan,signal_row,comparisons,validation,terminal,cleanup,audit=[read(paths[name]) for name in
            ('plan','signal','comparisons','validation','terminal','cleanup','compile_audit')]
        if (plan['cell_id']!=contract.CELL_ID or plan['epsilon']!=.05 or plan['factors']!=[1.,10.]
                or plan['shots']!=dict(cosine=10216,sine=14356,total=24572)
                or plan['saved_PR']['cell_id']!='B2_p5_q1_2nd_r4_K4'
                or plan['saved_PR']['G_exact']!='403423200'
                or plan['saved_PR']['scope']!='CONFIRMATION_UNION8_N32'
                or plan['saved_PR']['kind']!='SAVED_PAIRED_SAMPLE_MEAN_NOT_POPULATION_MEAN'):
            raise ValueError('REGISTERED_PLAN_CHANGED')
        if (signal_row['cell']!=contract.cell() or contract.digest(signal_row)!=audit['p7_signal_digest']
                or audit['status']!='PASS_COMPLETE_MANDATORY_STOP' or audit['input_receipts_unchanged']!=15
                or audit['frozen_model_sha256']!=contract.MODEL_SHA):raise ValueError('SAVED_SIGNAL_BINDING')
        if (terminal['status']!='COMPLETE_MANDATORY_STOP' or cleanup['status']!='COMPLETE_MANDATORY_STOP'
                or terminal['exit_code']!=0 or cleanup['native_exit_code']!=0 or not cleanup['all_reaped']
                or cleanup['first_stop'] is not None or cleanup['cleanup_errors']):raise ValueError('COMPILE_NOT_CLEAN')
        if terminal['call_counts']['native_compile_max']!=2:raise ValueError('EXACT_TWO_AXES_REQUIRED')
        if validation['validation']!=contract.validate_pair(validation['rows']):raise ValueError('AXIS_VALIDATION')
        costs={row['axis']:row['measured'] for row in validation['rows']}
        selected=[]
        for factor in plan['factors']:
            shots=signal_row['precision_rows']['0.05'][str(factor)]
            matches=[x for x in comparisons['rows'] if x['epsilon']==.05 and x['factor']==factor
                and x['bias_rule']=='signed_signal' and x['scope']=='fine_grid_S2_plus_truncated_S4']
            if len(matches)!=1:raise ValueError('UNIQUE_SAVED_COMPARISON_REQUIRED')
            old=matches[0]
            if (not shots['cost_acquisition_eligible'] or shots['boundary_guard']!='PASS'
                    or shots['N_total']!=24572 or shots['axes']['real']['shots']!=10216
                    or shots['axes']['imag']['shots']!=14356
                    or old['deterministic']['best']['cell_id']!=contract.CELL_ID
                    or old['deterministic']['best']['G_exact']!='1784443212/5'
                    or old['confirmed_PR']['best']['cell_id']!=plan['saved_PR']['cell_id']
                    or old['confirmed_PR']['best']['G_exact']!=plan['saved_PR']['G_exact']):
                raise ValueError('SAVED_SHOTS_OR_COMPARISON_CHANGED')
            selected.append((factor,shots,old))
        write(output/'reservation.json',dict(scope=SCOPE,cost_reaccounting=1,unique_comparisons=1,
            factor_rows=2,new_compile=0,new_signal=0,new_sampling=0,new_shot_calculation=0,new_fit=0))
        reserved=True
        # Only the preregistered p7 candidate and saved PR confirmation point.
        total=10216*costs['cosine']+14356*costs['sine']
        pr=Fraction(plan['saved_PR']['G_exact']);ratio=pr/total;reduction=100*(1-ratio)
        rows=[]
        for factor,shots,old in selected:
            rows.append(dict(epsilon=.05,allowance_factor=factor,bias_rule='signed_signal',
                deterministic_cell=contract.CELL_ID,saved_shots=dict(cosine=10216,sine=14356,total=24572),
                measured_RZ_per_axis=costs,deterministic_G_exact=str(total),deterministic_G=total,
                deterministic_cost_kind='MEASURED_COMPILED_AXES_TIMES_SAVED_SHOTS',
                saved_PR_cell=plan['saved_PR']['cell_id'],saved_PR_G_exact=str(pr),
                saved_PR_scope=plan['saved_PR']['scope'],saved_PR_kind=plan['saved_PR']['kind'],
                PR_over_deterministic_exact=str(ratio),PR_over_deterministic=float(ratio),
                PR_reduction_percent_exact=str(reduction),PR_reduction_percent=float(reduction),
                saved_model_deterministic_G_exact=old['deterministic']['best']['G_exact'],
                saved_model_PR_reduction_percent_exact=old['PR_reduction_percentage_points_exact'],
                observed_point_reversal_persists=bool(pr>total),candidate_reselection=False,
                population_winner_certified=False,conservative_bias_certificate=None))
        for item in grant['inputs']:
            if sha(item['path'])!=item['sha256']:raise ValueError('INPUT_CHANGED_AFTER_ACCOUNTING')
        result=dict(schema='track_a_v1_p7_measured_reaccount_result_v1',input_source_commit=INPUT_SOURCE,
            execution_script_sha256=sha(__file__),grant_sha256=sha(grant_path),scope=SCOPE,
            rows=rows,unique_comparisons=1,factor_rows=2,new_shot_calculation=False,new_signal=False,
            new_compile=False,model_refit=False,candidate_reselection=False,V2=False,
            full_grid_error_bound=False,conservative_bias_certificate=None,next_stage_authorized=False,
            mandatory_stop=True)
        write(output/'comparison.json',result)
        # VmHWM is reset by exec; ru_maxrss can retain the previous executable's peak.
        peak=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
                  if line.startswith('VmHWM:'))
        if peak>RESOURCES['RSS_bytes']:raise ValueError('RSS_CEILING')
        write(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,
            finished_UTC=datetime.now(timezone.utc).isoformat(),input_source_commit=INPUT_SOURCE,
            cost_reaccounting=1,unique_comparisons=1,factor_rows=2,wall_seconds=time.monotonic()-started,
            peak_RSS_bytes=peak,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+
                resource.getrusage(resource.RUSAGE_SELF).ru_stime,next_stage_authorized=False,automatic_retry=False))
    except BaseException as exc:
        write(output/'terminal.json',dict(status='STOP',exit_code=1,error=str(exc),reservation_charged=reserved,
            automatic_retry=False,next_stage_authorized=False))
        raise
    finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved',action='store_true')
    parser.add_argument('--grant',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute_approved:
        if args.grant or args.output:parser.error('explicit approved mode required')
        print(json.dumps(dict(scope=SCOPE,input_source_commit=INPUT_SOURCE,call_caps=CAPS,
                              resources=RESOURCES,approved=False),indent=2));return
    if not args.grant or not args.output:parser.error('grant and output required')
    execute(args.grant,args.output)


if __name__=='__main__':main()
