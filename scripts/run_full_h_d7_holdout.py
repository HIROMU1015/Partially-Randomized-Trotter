"""Fixed D7 independent holdouts and exact descriptive comparison; no model verdict."""
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
from decimal import Decimal, localcontext
from functools import partial
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_kappa2_p2 as P
import run_full_h_kappa2_p2_accelerated as Runner
from complete_full_h_p2_checkpoint import observe_child

CALLER='scripts/run_full_h_d7_holdout.py'
STUDY='docs/research/d_full_h_d7_holdout_20261115'
FROZEN=STUDY+'/FROZEN_JOBS.json'
MODEL=dict(commit='b5165f11bb63f9eb2a6bfe385ba75d8003b7e724',
    path='docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json')
RULE=dict(commit='f0765067a5a8b39888e36f3a8b80f1bdbda69223',
    path='docs/research/full_h_d7_holdout_rule_proposal_20261115/PROPOSAL.md')
VERDICT='NOT_ASSESSED_NO_ADOPTED_TOLERANCE'
NUMERIC_GATES={'low_control','high_control_leakage','independent_occupation_vector',
    'Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'}


def rational(value):
    if (not isinstance(value,dict) or set(value)!={'numerator','denominator'}
            or type(value['numerator']) is not int or type(value['denominator']) is not int
            or value['denominator']<=0):raise ValueError('Exact rational required')
    return Fraction(value['numerator'],value['denominator'])


def exact(value):
    value=Fraction(value)
    return dict(numerator=value.numerator,denominator=value.denominator)


def prediction_bindings(plan,reference=MODEL):
    gate=P.frozen_model_gate(plan,reference)
    frozen=json.loads(P.P1.blob(reference['commit'],reference['path']))
    if set(frozen['models'])!={'X','Y'} or len(frozen['predictions'])!=4:raise ValueError('Exactly two models/four predictions required')
    rows=[]
    for cid,q,R,T,n,pid in (('M5_ROUND',32,283,3.2,2,'FULL_H_bc8aab34256b73b68025'),
                            ('M6_ROUND',64,566,6.4,1,'FULL_H_4ec8d73571760a6830b8')):
        cell=P.selected(plan,'HOLDOUT',cid);physical=P.physical_row(cell);seeds=P.seed_table(plan,cid)
        if (cell['q'],cell['R'],cell['T'],cell['n_paths'],physical['id'])!=(q,R,T,n,pid):
            raise ValueError('Fixed D7 holdout design differs')
        for axis in ('X','Y'):
            found=[(i,r) for i,r in enumerate(frozen['predictions']) if (r['cell_id'],r['axis'])==(cid,axis)]
            if len(found)!=1:raise ValueError('Prediction key missing/duplicate')
            i,r=found[0];rational(r['predicted_RZ'])
            if (r['physical']!=physical or r['cohort']!=cell['cohort'] or r['step_seeds']!=seeds
                    or (r['q'],r['R'],r['n_paths'],r['m'])!=(q,R,n,int(cid[1]))
                    or r['features']!={k:exact(v) for k,v in P.design_api().features(cell,plan['feature_map']).items()}
                    or r.get('observed_RZ') is not None or r.get('tolerance') is not None):
                raise ValueError('Frozen prediction physical/cohort/seed/features differs')
            rows.append(dict(cell_id=cid,axis=axis,physical=physical,cohort=cell['cohort'],step_seeds=seeds,
                n_paths=n,P=deepcopy(r['predicted_RZ']),
                prediction_source=dict(reference,selector=['predictions',i]),
                model_source=dict(reference,selector=['models',axis]),rule_source=deepcopy(RULE)))
    return dict(status='BOUND_BEFORE_HOLDOUT_OBSERVATION',frozen_model_gate=gate,rows=rows,
        holdout_observation_reads=0,prediction_recomputations=0)


def axis_receipts(plan,cell,result,execution_commit):
    """Check actual path gates and original compiler records before arithmetic."""
    source=result['source'];paths=result.get('child',{}).get('paths',[])
    if (result['status']!='FINITE_CELL_COMPLETE' or result['child']['status']!='PASS'
            or source['commit']!=execution_commit or source['caller']!=CALLER or source['cell']!=cell['cell_id']
            or source['cohort']!=cell['cohort'] or source['step_seeds']!=P.seed_table(plan,cell['cell_id'])
            or len(paths)!=cell['n_paths'] or [r['index'] for r in paths]!=list(range(cell['n_paths']))):
        raise ValueError('Incomplete/foreign holdout cell receipt')
    out=[]
    for index,path in enumerate(paths):
        trial=path['trial'];checks=path['checks'];expected=NUMERIC_GATES if index==0 else set()
        if (path['row']!=P.physical_row(cell) or path['cohort']!=cell['cohort']
                or path['seeds']!=P.seed_table(plan,cell['cell_id'])[index]
                or trial['candidate']!=path['row'] or trial['cohort']!=path['cohort'] or trial['index']!=index
                or trial['status']!='PASS' or {g['check'] for g in trial['checks']}!=expected
                or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])
                or checks['builder_lowering_id']!=P.P1.LOWERING_ID or checks['parity']['status']!='PASS'
                or checks['scalar_event_phases']['status']!='PASS' or path['xy_phase_gate']['status']!='PASS'
                or checks['scalar']!=trial['scalar'] or checks['scalar_expected']!=trial['scalar']
                or checks['one_body_applications']<=0 or checks['legacy_rzz_control_instructions']!=0
                or path['sampling']!='ORIGINAL_NORMAL_SAMPLER' or path.get('parents')
                or len(path['events'])!=cell['q'] or any(len(g)!=cell['R'] for g in path['events'])):
            raise ValueError('Holdout path identity/numeric/implementation gate differs')
        for axis,a in (('X','cosine'),('Y','sine')):
            receipt=path['compile_receipts'][a];record=receipt['record'];price=record['metrics']['rz_count']
            if (record['compiler']!=cell['profile']['compiler'] or record['fingerprint']!=path['fingerprints'][a]
                    or receipt['fingerprint']!=record['fingerprint'] or record['metrics']!=trial['prices'][a]
                    or not receipt['locator']['sha256'] or type(price) is not int or price<0):
                raise ValueError('Holdout axis/compile/price differs')
            out.append(dict(cell_id=cell['cell_id'],axis=axis,index=index,cohort=cell['cohort'],
                seeds=path['seeds'],physical=path['row'],rz_count=price,source=source,
                fingerprint=record['fingerprint'],compile_receipt=receipt,
                raw_path_selector=['child','paths',index],gate_status='PASS',
                correctness_scope='NUMERIC_7_ON_INDEX0;PARITY_SCALAR_EVENT_XY_ALL_PATHS'))
    return out


def compare(bound,observations,invalid=None):
    """Four exact descriptive rows. No fit, prediction, tolerance or model test."""
    if (len(bound['rows'])!=4 or {(r['cell_id'],r['axis']) for r in bound['rows']}
            !={(c,a) for c in ('M5_ROUND','M6_ROUND') for a in ('X','Y')}):
        raise ValueError('Exactly four comparison keys required')
    invalid=invalid or {};rows=[]
    for fixed in bound['rows']:
        cid,axis=fixed['cell_id'],fixed['axis'];n=fixed['n_paths'];Pvalue=rational(fixed['P'])
        selected=[r for r in observations if r['cell_id']==cid and r['axis']==axis]
        row=dict(fixed,status='INCOMPARABLE',O=None,signed_error=None,signed_relative_error=None,
            display_signed_percent=None,relative_error_reason=None,observation_sources=[],
            tolerance=None,model_verdict=VERDICT)
        reason=invalid.get(cid)
        if not reason and (len(selected)!=n or sorted(r['index'] for r in selected)!=list(range(n))):
            reason='MISSING_OR_DUPLICATE_REQUIRED_PATH'
        if not reason:
            for obs in selected:
                if (obs['gate_status']!='PASS' or obs['cohort']!=fixed['cohort'] or obs['physical']!=fixed['physical']
                        or obs['seeds']!=fixed['step_seeds'][obs['index']]
                        or type(obs['rz_count']) is not int or obs['rz_count']<0):reason='INVALID_PATH_AXIS_BINDING';break
        row['reason']=reason
        if not reason:
            O=sum((Fraction(r['rz_count']) for r in selected),Fraction(0))/n;error=Pvalue-O
            row.update(status='VALID',O=exact(O),signed_error=exact(error),
                observation_sources=[r['locators'] for r in sorted(selected,key=lambda r:r['index'])])
            if O:
                relative=error/O;row['signed_relative_error']=exact(relative)
                with localcontext() as ctx:
                    ctx.prec=50
                    row['display_signed_percent']=str((Decimal(relative.numerator)*100/Decimal(relative.denominator)).quantize(Decimal('.000001')))
            else:row['relative_error_reason']='ZERO_OBSERVED_PRICE'
        rows.append(row)
    valid=sum(r['status']=='VALID' for r in rows)
    status=('INCOMPLETE_OR_INVALID' if valid!=4 else 'DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED'
            if any(r['relative_error_reason'] for r in rows) else 'DESCRIPTIVE_COMPLETE')
    return dict(status=status,rows=rows,valid_comparisons=valid,missing_or_invalid_comparisons=4-valid,
        model_verdict=VERDICT,tolerance=None,percent_display_decimal_places=6,
        law_expectation=False,qualified_for_candidate=False,CI=None,N=None,G=None,winner=None,mandatory_stop=True)


def child(path):
    job=json.loads(Path(path).read_text());private=Path(job['private']);start=time.monotonic()
    with (private/'PROGRESS.jsonl').open('x') as log:
        def mark(stage):
            log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,elapsed_seconds=time.monotonic()-start))+'\n');log.flush()
        mark('parallel_paths_started')
        code=Runner.child(path)
        mark('COMPLETE' if code==0 else 'STOP')
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-approved-job');parser.add_argument('--child')
    args=parser.parse_args();P.CALLER=CALLER
    if args.child:return child(args.child)
    if not args.execute_approved_job:parser.error('Explicit fixed cell job required')
    job=json.loads(Path(args.execute_approved_job).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (CALLER,'scripts/run_full_h_kappa2_p2.py',Runner.CALLER,Runner.HELPER,'scripts/complete_full_h_p2_checkpoint.py'):
        if (ROOT/path).read_bytes()!=P.P1.blob(commit,path):raise ValueError('Holdout SOURCE differs from fixed HEAD')
    frozen=json.loads(P.P1.blob(commit,FROZEN))
    if job!=frozen['jobs'][job['cell_id']]:raise ValueError('Holdout job differs from frozen inputs')
    if job['frozen_model_source']!=MODEL or job['comparison_rule']!=RULE:raise ValueError('Fixed C/C2 references differ')
    bound=prediction_bindings(job['plan'])
    if bound!=frozen['prediction_bindings']:raise ValueError('Pre-observation prediction binding differs')
    cell=P.selected(job['plan'],'HOLDOUT',job['cell_id'])
    if job['new_compile_maximum']!=2*cell['n_paths'] or not job['progress_stages']:raise ValueError('Compile scope/observability differs')
    return P.execute(job,run_child=partial(observe_child,private=job['private']))


if __name__=='__main__':sys.exit(main())
