"""H4 fixed13 N-only batch. Default is metadata planning, never science.

--execute is for a separately permitted scientific run. No C provider, cost
aggregation, schedule optimization or automatic retry/resume is invoked.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import h4_df10_rpe_batch_provider as h4
from trottertracks import pf_pr_rpe_pipeline as pipeline

PLAN_PATH='docs/research/h4_df10_rpe_batch_preparation_20261110/PLAN.json'
SOURCE_FILES=('src/trottertracks/h4_df10_rpe_batch_provider.py','scripts/run_h4_df10_rpe_batch.py',PLAN_PATH,
 'src/trottertracks/pf_pr_rpe_pipeline.py','src/trottertracks/h4_df10_rpe_connection.py',
 'src/trottertracks/h4_df10_rpe_d1.py','src/trottertracks/h6_df19_rpe_d2.py','scripts/run_h6_df19_rpe_d2.py',
 'scripts/h46_new2_core.py','src/trotterlib/h46_m5_plan.py','src/trotterlib/rte.py',
 'src/trotterlib/df_hamiltonian.py','src/trotterlib/df_partial_s2.py','src/trotterlib/pr2_s0_s1_validation.py',
 'src/trottertracks/c1_stage3_accuracy.py','src/trotterlib/rpe_resource_accounting.py')


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def plan(packet):
    """Copy inventory and saved N only. No signal arithmetic or N evaluation."""
    task=packet['task'];cached={pipeline.key(n['identity']):n for n in packet['saved_N']};rows=[]
    for row in task['candidates']:
        n=cached.get(pipeline.key(row));cid=row['candidate']['id'];m4=cid in h4.M4_IDS
        rows.append(dict(m=row['m'],candidate_id=cid,q_RPE=row['q_RPE'],q_PF=row['q_PF'],
            N_status=n['plan']['status'] if n else 'MISSING',N_X=n['plan']['N_X'] if n else None,N_Y=n['plan']['N_Y'] if n else None,
            input_route='COPY_SAVED_N' if n else ('SAVED_SIGNAL_SCALAR' if m4 else 'NEW_REFERENCE_AND_FINITE_MEAN'),
            signal_source=packet['saved_origins'].get(cid)))
    return dict(status='TECHNICAL_PREPARATION_ONLY_NO_NEW_SCIENCE',task=deepcopy(task),rows=rows,
        static_sources=deepcopy(packet['saved_sources']),
        input_bindings={k:v['sha256'] for k,v in packet['input_specs'].items()},context_sha256=packet['context_spec']['sha256'],
        frozen_loader=dict(commit=h4.SIGNAL_CARRIER,path='scripts/h46_m5_tol_saved_input.py',
            sha256=hashlib.sha256(h4.blob(ROOT,h4.SIGNAL_CARRIER,'scripts/h46_m5_tol_saved_input.py')).hexdigest()),
        proposed_scientific_scope=dict(saved_m3_N_copied=2,saved_m3_re_evaluations=0,saved_m4_scalar_candidates=2,
            new_reference_times=[r['candidate']['T'] for r in task['candidates'] if r['m'] not in (3,4) and r['candidate']['method']=='B0'],
            reference_state_actions=22,new_finite_mean_candidates=22,PR_exact_tail_controls=11,
            new_scalar_candidates=24,XY_axis_plans=48,prices=0,G=0,eigensolves=0,sampling=0,build=0,compile=0),
        actual_errors_qualified=False,full_RPE_qualified=False,total_resources=None,automatic_retry=False,mandatory_stop=True)


def coverage(packet,records,source,*,valid=True):
    """Coverage never computes N/C/G. Missing, saved FAIL and source loss stay null."""
    bykey={pipeline.key(n['identity']):n for n in records};rows=[]
    old={pipeline.key(n['identity']) for n in packet['saved_N']}
    for row in packet['task']['candidates']:
        key=pipeline.key(row);n=bykey.get(key);lane=n['plan'] if n else {}
        status=lane.get('status','NOT_EVALUATED_AFTER_STOP')
        if not valid and key not in old:status='UNQUALIFIED_SOURCE_OR_INPUT'
        positive=all(type(lane.get('N_'+a)) is int and lane['N_'+a]>0 for a in ('X','Y'))
        if status=='CONDITIONAL_ROUND_PASS' and not positive:status='UNQUALIFIED_N_MISSING'
        passed=status=='CONDITIONAL_ROUND_PASS'
        rows.append(dict(identity=pipeline.identity(packet['task'],row),status=status,
            N_X=lane.get('N_X') if passed else None,N_Y=lane.get('N_Y') if passed else None,
            actual_N_X=None,actual_N_Y=None,C=None,G=None,
            source=deepcopy(n['source']) if n else dict(commit=source),
            origin=n.get('origin') if n else 'NOT_EVALUATED',
            reason=n.get('diagnostics',{}).get('reason') if n else 'No result before STOP',
            normalization=deepcopy(n.get('diagnostics',{}).get('normalization')) if n else None,
            normalization_is_price_multiplier=False))
    qualified={a:[r['identity']['m'] for r in rows if pipeline.arm(r['identity']['candidate'])==a and r['status']=='CONDITIONAL_ROUND_PASS'] for a in pipeline.ARMS}
    return dict(rows=rows,qualified_rounds_by_arm=qualified,
        all_13_N_rounds_conditional_pass=all(v==list(range(13)) for v in qualified.values()),
        actual_N_qualified=False,full_RPE_qualified=False,total_resources=None,mandatory_stop=True)


def execute(packet,provider,source,out):
    """Future science entry. Tests pass a mock provider and never molecular data."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(status='H4_N_BATCH_RUNNING',scientific_source=source,configuration=deepcopy(packet['task']['configuration']),
        records=deepcopy(packet['saved_N']),errors=[],scope_observed=None,
        saved_m3_re_evaluations=0,price_calls=0,G_evaluations=0,automatic_retry=False,
        actual_N_qualified=False,full_RPE_qualified=False,total_resources=None,mandatory_stop=True)
    save(out/'progress.json',result)
    try:
        batch=pipeline.run_missing(packet['task'],packet['saved_N'],[],n_provider=provider,c_provider=None)
        result['records']=batch['shots']
        result['status']='H4_N_BATCH_EVALUATED' if not batch['plan']['missing_N'] and not batch['plan']['unqualified_N'] else 'H4_STOP_UNQUALIFIED_N'
    except Exception as exc:result.update(status='H4_STOP_TECHNICAL_OR_REFERENCE',errors=[str(exc)])
    result.update(scope_observed=getattr(provider,'last_run',None),wall_seconds=time.monotonic()-started)
    for n in result['records']:save(out/(n['identity']['candidate']['id']+'.json'),n)
    save(out/'result.json',result);save(out/'COVERAGE.json',coverage(packet,result['records'],source))
    return result


def invalidate_new_records(packet,records):
    """Keep computed diagnostics but never expose source-invalid N as usable."""
    old={pipeline.key(n['identity']) for n in packet['saved_N']}
    for n in records:
        if pipeline.key(n['identity']) not in old:
            n.setdefault('diagnostics',{})['computed_plan_before_source_failure']=deepcopy(n['plan'])
            n['plan'].update(status='UNQUALIFIED_SOURCE_OR_INPUT',N_X=None,N_Y=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--execute',action='store_true',help='Separately permitted H4 N science; default copies metadata only')
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Fresh H4 output directory required; no overwrite/resume')
    packet=h4.prepare(ROOT);proposal=plan(packet)
    if not args.execute:
        out.mkdir(parents=True);save(out/'PLAN.json',proposal)
        print(json.dumps(dict(status=proposal['status'],new_science_calls=0,mandatory_stop=True)));return 0
    if json.loads((ROOT/PLAN_PATH).read_bytes())!=proposal:raise ValueError('Published H4 plan differs')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={path:h4.sha(ROOT/path) for path in SOURCE_FILES}
    for path in SOURCE_FILES:
        if (ROOT/path).read_bytes()!=h4.blob(ROOT,source,path):raise ValueError('Used scientific SOURCE differs from commit')
    for path in SOURCE_FILES[3:]:
        if (ROOT/path).read_bytes()!=h4.blob(ROOT,h4.BASELINE,path):raise ValueError('Reused baseline numerical/core SOURCE changed')
    inputs=list(packet['input_specs'].values())+[packet['context_spec']]
    provider=h4.H4NBatch(packet,dict(commit=source,path='scripts/run_h4_df10_rpe_batch.py'),lambda task:h4.load_saved_context(ROOT,packet))
    result=execute(packet,provider,source,out)
    result.update(source_files=hashes,input_bindings=proposal['input_bindings'],context_sha256=proposal['context_sha256'],
        saved_sources=proposal['static_sources'],frozen_loader=proposal['frozen_loader'],
        source_preserved=all(h4.sha(ROOT/k)==v for k,v in hashes.items()),
        input_preserved=all(h4.sha(v['path'])==v['sha256'] for v in inputs))
    valid=result['source_preserved'] and result['input_preserved']
    if not valid:
        result['status']='H4_STOP_SOURCE_OR_INPUT_CHANGED'
        invalidate_new_records(packet,result['records'])
        for n in result['records']:save(out/(n['identity']['candidate']['id']+'.json'),n)
    save(out/'result.json',result);save(out/'COVERAGE.json',coverage(packet,result['records'],source,valid=valid))
    print(json.dumps(dict(status=result['status'],records=len(result['records']),mandatory_stop=True)))
    return 0 if result['status']=='H4_N_BATCH_EVALUATED' else 1


if __name__=='__main__':sys.exit(main())
