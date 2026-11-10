"""Fixed H6 m8..12 N-only batch; real use requires explicit user permission.

Reuses the published integration provider and numerical kernels unchanged.
No prices, totals, retry/resume, or candidate/schedule optimization are run.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as pipeline

BASELINE='3b742fff26381514e12c07b51701276b7d1b26ec'
PLAN_PATH='docs/research/h6_df19_rpe_m8_12_preparation_20261110/PLAN.json'
IDS=('26608f8bf53993c7095d','53797d9e5a2eeec31a13','408142c6d1cd1b104b72','4d34a6f3f136df9ad512',
     '20a160dafc0fa7d6bce5','e25c35e31bb6c840e252','2d56c445820c4accd676','86665f416fa64ed7a1a1',
     'fb0cb7b53bbfc8080224','c9309f4a8219815577f5')
INTEGRATION_FILES=('src/trottertracks/pf_pr_rpe_pipeline.py','src/trottertracks/h6_rpe_batch_provider.py')


def validate_menu(plan, task):
    pipeline.validate_task(task)
    rows=[r for r in task['candidates'] if r['m']>=8]
    if (plan['integration_source']!=BASELINE or plan['rows']!=rows
            or tuple(r['candidate']['id'] for r in rows)!=IDS
            or plan['configuration_reuse']!=task['configuration']
            or plan['PR_allocation']!=[1,2,1,2,1,2,1,2,1,2,1]
            or plan['reference_gate']!=1e-10 or plan['norm_and_normalization_gate']!=1e-9):
        raise ValueError('Fixed published m8..12 menu/policy required')
    scope=plan['planned_scientific_scope']
    required=dict(new_reference_times=[25.6,51.2,102.4,204.8,409.6],reference_state_actions=10,finite_mean_candidates=10,PR_exact_tail_controls=5,scalar_candidates=10,XY_axis_plans=20)
    if any(scope[k]!=v for k,v in required.items()) or any(v!=0 for k,v in scope.items() if k not in required):
        raise ValueError('Fixed scientific call scope required')
    for index,row in enumerate(rows):
        m=8+index//2;c=row['candidate'];pr=index%2==1;q=1<<(m-4)
        expected=dict(system='H6',method='PR' if pr else 'FULL',T=.1*(1<<m),q=q,delta=1.6,
            formula='4th(m5_best)',prefix=7 if pr else 19,R_outer=16 if pr else None,K=2 if pr else None)
        if (row['m']!=m or row['q_RPE']!=1<<m or row['q_PF']!=q
                or any(c[k]!=v for k,v in expected.items())):
            raise ValueError('Candidate outside fixed10 scope')
    return rows


def prepare(root=ROOT, plan=None):
    """Read saved JSON only. No state arrays, N/C calculation or provider call."""
    store=pipeline.SavedArtifacts(root)
    if plan is None:plan=json.loads((Path(root)/PLAN_PATH).read_bytes())
    task=store.read(plan['task_reference'])['task'];rows=validate_menu(plan,task)
    if plan['task_reference']!=dict(commit=BASELINE,path='docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/H6_PLAN.json'):
        raise ValueError('Published task carrier required')
    reuse=store.read(plan['reuse_plan_reference']);coverage=store.read(plan['saved_coverage_reference'])
    saved_result=store.read(plan['saved_result_reference'])
    if (reuse['configuration_reuse']!=task['configuration'] or coverage['configuration']!=task['configuration']
            or reuse['scientific_binding']!=plan['scientific_binding']
            or saved_result['scientific_binding']!=plan['scientific_binding']
            or coverage['covered_rounds']!=list(range(8)) or len(coverage['rows'])!=16):
        raise ValueError('Saved H6 input/policy/coverage differs')
    ns=[]
    for old in coverage['rows']:
        ref=dict(commit=plan['saved_coverage_reference']['commit'],path=old['result_reference']['path'])
        artifact=store.read(ref)
        for field in ('base_time','energy_precision_ha','beta_total','beta_stat','alpha_total_per_method',
                      'alpha_axis','energy_prior_ha','energy_shift','decoder'):
            if not pipeline.same(artifact['configuration'][field],task['configuration'][field]):
                raise ValueError('Historical N policy differs')
        point=next(r for r in artifact['rows'] if r['candidate_id']==old['candidate_id'])
        if (artifact['scientific_source']!=old['scientific_source']
                or point['ideal_numerical_pilot']['N_X']!=old['N_X']
                or point['ideal_numerical_pilot']['N_Y']!=old['N_Y']):
            raise ValueError('Saved N provenance or integer mismatch')
        row=next(r for r in task['candidates'] if pipeline.key(r)==pipeline.key(point))
        n=pipeline.shot_record(task,row,point,dict(ref,scientific_source=artifact['scientific_source']))
        n['diagnostics']={k:v for k,v in n['diagnostics'].items() if k not in ('compiled_prices','compiled_price','G')}
        ns.append(n)
    inventory=pipeline.plan(task,ns,[])
    if tuple(inventory['missing_N'])!=IDS or inventory['unqualified_N'] or inventory['missing_candidate_slots']:
        raise ValueError('Exactly the ten new N records must be missing')
    return dict(plan=plan,task=task,rows=rows,reuse_plan=reuse,previous=coverage,
        saved_N=ns,previous_source_files=saved_result['source_files'],sources=store.sources)


def coverage(packet, new_records, source, *, valid=True):
    """Copy prior16 rows. Keep FAIL/not-evaluated distinct from qualified N."""
    task=packet['task'];byid={r['identity']['candidate']['id']:r for r in new_records}
    result=dict(configuration=deepcopy(task['configuration']),rows=deepcopy(packet['previous']['rows']),
        scheduled_rounds=list(range(13)),new_price_handoff=[],full_RPE_qualified=False,
        actual_N_qualified=False,total_resources=None,mandatory_stop=True)
    for row in packet['rows']:
        cid=row['candidate']['id'];n=byid.get(cid);lane=n['plan'] if n else {}
        positive_N=all(type(lane.get('N_'+a)) is int and lane['N_'+a]>0 for a in ('X','Y'))
        passed=valid and lane.get('status')=='CONDITIONAL_ROUND_PASS' and positive_N
        status=lane.get('status','NOT_EVALUATED_AFTER_STOP') if valid else 'UNQUALIFIED_SOURCE_OR_INPUT'
        if status=='CONDITIONAL_ROUND_PASS' and not positive_N:status='UNQUALIFIED_N_MISSING'
        nx,ny=(lane.get('N_'+a) if passed else None for a in ('X','Y'))
        result['rows'].append(dict(stage='M8_12_BATCH',candidate_id=cid,candidate=deepcopy(row['candidate']),
            m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],N_X=nx,N_Y=ny,status=status,
            actual_N_X=None,actual_N_Y=None,scientific_source=source,
            reason=n.get('diagnostics',{}).get('reason') if n else 'Batch stopped before a candidate result'))
        result['new_price_handoff'].append(dict(identity=pipeline.identity(task,row),N_X=nx,N_Y=ny,
            status=status,source=dict(commit=source,candidate_id=cid),
            normalization=deepcopy(n.get('diagnostics',{}).get('normalization')) if n else None,
            alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'],
            actual_N_X=None,actual_N_Y=None,C=None,G=None,
            normalization_is_price_multiplier=False,price_work_performed=False))
    result['qualified_rounds_by_arm']={a:[r['m'] for r in result['rows'] if pipeline.arm(r['candidate'])==a and r['status']=='CONDITIONAL_ROUND_PASS'] for a in ('PF','PR')}
    result['unqualified_rounds_by_arm']={a:[r['m'] for r in result['rows'] if pipeline.arm(r['candidate'])==a and r['status']!='CONDITIONAL_ROUND_PASS'] for a in ('PF','PR')}
    result['all_13_N_rounds_conditional_pass']=all(result['qualified_rounds_by_arm'][a]==list(range(13)) for a in ('PF','PR'))
    return result


def execute(packet, provider, source, out):
    """Explicit future scientific entry; synthetic tests inject a mock provider."""
    import run_h6_df19_rpe_d2 as io
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(status='M8_12_RUNNING',scientific_source=source,configuration=packet['task']['configuration'],
        scientific_binding=packet['plan']['scientific_binding'],previous_rounds=packet['previous']['rows'],
        new_records=[],errors=[],scope_planned=packet['plan']['planned_scientific_scope'],
        scope_observed=None,price_calls=0,G_evaluations=0,saved_round_re_evaluations=0,
        full_RPE_qualified=False,total_resources=None,automatic_retry=False,mandatory_stop=True)
    io.save(out/'progress.json',result)
    try:
        batch=pipeline.run_missing(packet['task'],packet['saved_N'],[],n_provider=provider,c_provider=None)
        result['new_records']=[n for n in batch['shots'] if n['identity']['m']>=8]
        if tuple(n['identity']['candidate']['id'] for n in result['new_records'])!=IDS:
            raise ValueError('Incomplete or changed batch menu')
        result['status']='M8_12_N_BATCH_EVALUATED' if not batch['plan']['unqualified_N'] else 'M8_12_STOP_UNQUALIFIED_N'
    except Exception as exc:
        result['status']='M8_12_STOP_TECHNICAL_OR_REFERENCE';result['errors'].append(str(exc))
    result['scope_observed']=getattr(provider,'last_run',None)
    result['wall_seconds']=time.monotonic()-started
    for n in result['new_records']:io.save(out/(n['identity']['candidate']['id']+'.json'),n)
    io.save(out/'result.json',result)
    io.save(out/'COVERAGE.json',coverage(packet,result['new_records'],source))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Fresh m8..12 batch output required; no resume or retry')
    packet=prepare();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    import run_h6_df19_rpe_d2 as io
    from trottertracks.h6_rpe_batch_provider import H6ActionBatch,load_saved_context
    files=list(packet['previous_source_files'])+list(INTEGRATION_FILES)+['scripts/run_h6_df19_rpe_m8_12.py',PLAN_PATH]
    for path in files:
        data=(ROOT/path).read_bytes()
        if data!=io.blob(source,path):raise ValueError('Used SOURCE differs from execution commit')
        if path in packet['previous_source_files'] and io.sha(ROOT/path)!=packet['previous_source_files'][path]:
            raise ValueError('Historical numerical SOURCE changed')
        if path in INTEGRATION_FILES and data!=io.blob(BASELINE,path):raise ValueError('Published integration changed')
    hashes={path:io.sha(ROOT/path) for path in files}
    tail=packet['reuse_plan']['saved_tail_context']
    index=json.loads(io.blob(tail['source_index_commit'],tail['source_index_path']))
    inputs={v['path']:v['sha256'] for v in index['saved_inputs']['H6']['inputs'].values()}
    inputs[index['T20_reference_files']['H6']['path']]=tail['sha256']
    if any(io.sha(path)!=h for path,h in inputs.items()):raise ValueError('Saved input/context differs')
    originals={}
    for ref in packet['sources']:
        commit,path=ref.split(':',1);data=io.blob(commit,path)
        if (ROOT/path).read_bytes()!=data:raise ValueError('Preserve the published saved inputs/results')
        originals[path]=data
    provider=H6ActionBatch(lambda task:load_saved_context(ROOT,packet['reuse_plan']),packet['reuse_plan'],
        dict(commit=source,path='scripts/run_h6_df19_rpe_m8_12.py'))
    result=execute(packet,provider,source,out)
    result.update(source_files=hashes,input_bindings=dict(files={k:v['sha256'] for k,v in index['saved_inputs']['H6']['inputs'].items()},tail_context_sha256=tail['sha256']),
        source_preserved=all(io.sha(ROOT/path)==h for path,h in hashes.items()),
        input_preserved=all(io.sha(path)==h for path,h in inputs.items()),
        previous_results_preserved=all((ROOT/k).read_bytes()==data for k,data in originals.items()))
    valid=all(result[k] for k in ('source_preserved','input_preserved','previous_results_preserved'))
    if not valid:result['status']='M8_12_STOP_SOURCE_OR_INPUT_CHANGED'
    io.save(out/'result.json',result);io.save(out/'COVERAGE.json',coverage(packet,result['new_records'],source,valid=valid))
    print(json.dumps(dict(status=result['status'],new_candidates=len(result['new_records']),mandatory_stop=True)))
    return 0 if result['status']=='M8_12_N_BATCH_EVALUATED' else 1


if __name__=='__main__':sys.exit(main())
