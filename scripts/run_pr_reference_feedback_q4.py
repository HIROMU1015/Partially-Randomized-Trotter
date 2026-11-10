"""Q4-1 preparation, or a separately user-approved H/E/C numerical campaign.

Default operation is metadata-only preflight. --campaign is NOT executed by
preflight/tests and is not authorization: the operator must already have the
user's scientific execution permission. No approval-proof mechanism is used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction as F
from itertools import product
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCIENCE_OUT=ROOT/'artifacts/pr_reference_feedback_q4_1/2026-11-10/science_run01'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_reference_feedback_q4 as g
from trottertracks import pr_reference_feedback_q4_certify as c
from trottertracks import pr_reference_feedback_q4_check as checker

SOURCE_PATHS=[
    'src/trottertracks/pr_reference_feedback_q4.py',
    'src/trottertracks/pr_reference_feedback_q4_certify.py',
    'src/trottertracks/pr_reference_feedback_q4_check.py',
    'scripts/run_pr_reference_feedback_q4.py']
INPUT_PATHS=[
    'artifacts/pr_reference_q0/2026-11-09/run01/'+p for p in
    ('INPUTS.json','SOURCE.json','RESULTS.json.xz','CHECKS.json','STOP.json')]+[
    'artifacts/pr_reference_feedback_q3b/2026-11-10/run01/'+p for p in
    ('B1.json','B2.json','CHECKS.json')]+[
    'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/'+p for p in
    ('C0.json','D.json','COST.json','CHECKS.json')]


def saved(path):
    data=subprocess.check_output(['git','show',g.EVIDENCE+':'+path],cwd=ROOT)
    return json.loads(lzma.decompress(data) if path.endswith('.xz') else data)


def preflight():
    """No Arithmetic(), model construction, CI, geometry, costs or optimizer."""
    inputs=[];objects={}
    for path in INPUT_PATHS:
        data=subprocess.check_output(['git','show',g.EVIDENCE+':'+path],cwd=ROOT)
        # Schema/static metadata only. Preserve saved numeric values verbatim.
        obj=json.loads(lzma.decompress(data) if path.endswith('.xz') else data)
        objects[path]=obj
        inputs.append(dict(commit=g.EVIDENCE,path=path,bytes=len(data),
                           sha256=hashlib.sha256(data).hexdigest(),keys=sorted(obj)))
    q0=objects['artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json']
    if q0['target']!=[17,131072,32,88] or set(q0['arms'])!={'Bprime','Reference_B1'}:
        raise RuntimeError('Fixed target/arm metadata mismatch')
    statuses={
        'artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json':'Q0_POSTRUN_CHECKS_PASS',
        'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/C0.json':'ALL_CONTEXT_NATIVE_TRANSFER_CLOSED_IN_FIXED_MODEL',
        'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/D.json':'GUARANTEE_CLOSED_IN_FIXED_MODEL'}
    for path,status in statuses.items():
        if objects[path]['status']!=status:
            raise RuntimeError('Saved qualification metadata mismatch: '+path)
    sources=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in SOURCE_PATHS]
    for p in ('scripts/diagnose_pr_reference_q0.py','scripts/run_pr_reference_feedback_q3cd.py'):
        if (ROOT/p).read_text()!=g.pinned_text(p):
            raise RuntimeError('Pinned pure helper SOURCE changed: '+p)
    return dict(status='PREPARATION_ONLY_SCIENCE_NOT_EXECUTED',base=g.BASE,
                evidence=g.EVIDENCE,sources=sources,inputs=inputs,
                arithmetic_bits=g.BITS,continuous_resolution=str(g.RESOLUTION),
                H=dict(arms=['Bprime','Reference_B1'],geometries=['ideal','native'],N=[1,1252]),
                E=dict(arms=['Bprime','Reference_B1'],geometry='native',N_XY=[1,340],N_Q=[1,340]),
                scientific_campaigns_executed=0,old_evidence_changed=False,
                stop='MANDATORY_STOP_PENDING_EXPLICIT_SCIENTIFIC_EXECUTION_PERMISSION')


def model_for(arm,a):
    # SCIENTIFIC numerical binding: called ONLY by the explicit campaign path.
    inputs=saved(INPUT_PATHS[0]);row=inputs['arms'][arm]
    b2=saved('artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B2.json')
    bounds=next(r for r in b2['rows'] if r['arm']==arm)
    d=saved('artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/D.json')
    drow=next(r for r in d['rows'] if r['arm']==arm)
    time=F(bounds['time17_exact']);kappa=F(bounds['kappa_Ha']['exact'])
    # Interval normalization itself is charged by 2^-160 turns; the saved
    # center's full width bounds the midpoint-to-true-center distance safely.
    pad=a.bounds(a.r(kappa*time)/(2*a.iv.pi))[1]+g.ROOT_WIDTH
    return c.Model(F.from_float(row['raw_radius_lower']),
        F(2,5)-F.from_float(row['statistical_phase_lower']),F(4,5),F(4,5),F(1,360),
        time,kappa,F(bounds['omega_E_Ha']['exact']),F(drow['feedback_arithmetic_upper']),pad)


def check_solution(a,model,result,family,geometry,kind):
    lower=1;feasible=None
    for item in result['results']:
        n=item['n'] if kind=='Q' else item['total']
        if item['status']=='FAIL_SAVED_WITNESS':
            if family!='H' or geometry!='ideal' or n>340:
                raise ValueError('Saved ideal-H witness misapplied')
            lower=n+1
            continue
        if kind=='Q':
            if item['status']=='PASS_IMPLEMENTED':
                # Independent checking path: per-count primitive replay, never solve_q.
                if len(item['counts'])!=n+1 or [r['k'] for r in item['counts']]!=list(range(n+1)):
                    raise ValueError('Missing Q count')
                for row in item['counts']:
                    ci=g.clopper_pearson(a,n,row['k'],model.failure)
                    checker.cp_endpoints(n,row['k'],model.failure,ci)
                    regions=g.q_region(a,ci,model.rho,model.bias+model.feedback_error,model.prior)
                    width=g.width_of(regions)
                    if width is not None and width+model.restoration>model.target:
                        raise ValueError('Q energy width not sufficient')
            elif item['status']=='FAIL':
                ci=g.clopper_pearson(a,n,item['k'],model.failure)
                checker.cp_endpoints(n,item['k'],model.failure,ci)
                pts=item['pair'];bias=model.bias+model.feedback_error
                if F(pts[1]['u'])-F(pts[0]['u'])<=model.target:
                    raise ValueError('Q pair not separated')
                for p in pts:
                    u,b,rho=map(F,(p['u'],p['b'],p['rho']))
                    if not (-model.prior<=u<=model.prior and abs(b)<=bias and model.rho<=rho<=1):
                        raise ValueError('Invalid Q witness')
                    lo,hi=a.bounds(a.r(rho)*a.iv.sin(a.r(u+b)))
                    if lo<ci.inner[0] or hi>ci.inner[1]:
                        raise ValueError('Q pair outside true CI')
        else:
            checker.total_result(a,model,item,family,geometry)
        if item['status']=='FAIL' and n==lower:
            lower+=1
        if item['status']=='PASS_IMPLEMENTED':
            feasible=n
    if lower!=result['lower'] or feasible!=result['feasible']:
        raise ValueError('Gap/feasible boundary inconsistent')
    return True


def task(spec):
    stage,arm,geometry,kind=spec
    a=g.Arithmetic();model=model_for(arm,a)
    if stage=='H':
        # This shortcut is bound to the saved checker and ideal geometry only.
        old=saved('artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json')
        if old['status']!='Q0_POSTRUN_CHECKS_PASS':
            raise RuntimeError('Old Q0 witness unavailable')
        for r in old['checks']:
            if r['stronger_XY_CI_hull_lower_bound']['total_shots_through']!=340:
                raise RuntimeError('Saved lower-bound scope changed')
        result=c.solve_xy(a,model,'H',geometry,1252,341 if geometry=='ideal' else 1)
    elif kind=='Q':
        result=c.solve_q(a,model,340)
    else:
        result=c.solve_xy(a,model,'E',geometry,340)
    check_solution(a,model,result,stage,geometry,kind)
    return dict(stage=stage,arm=arm,geometry=geometry,kind=kind,
                model={k:str(v) for k,v in asdict(model).items()},result=result,
                checks='FINITE_CERTIFICATE_LEAVES_REPLAYED_BY_AUTHOR',
                not_independent_scientific_review=True)


def price_connection(rows):
    """Called only in future C, after finite mathematical/numerical checks."""
    costs=saved('artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/COST.json')
    metrics=['RZ','CX','size','depth'];accounts=c.fraction_accounting();output=[]
    for arm in ('Bprime','Reference_B1'):
        prices=next(r for r in costs['rows'] if r['arm']==arm and r['m']==17)
        prefix={metric:accounts['add']([list(map(F,r['XY_cost'][metric]['exact']))
            for r in costs['rows'] if r['arm']==arm and r['m']<17]) for metric in metrics}
        q_result=next(r['result'] for r in rows if r['arm']==arm and r['stage']=='E' and r['kind']=='Q')
        qn=q_result['feasible'];qlower=q_result['lower']
        for row in (r for r in rows if r['arm']==arm and r['kind']=='XY' and r['geometry']=='native'):
            result=row['result'];feasible=result['feasible']
            allocations=[]
            if feasible is not None:
                policy=next(r for r in result['results'] if r['status']=='PASS_IMPLEMENTED')
                allocations=[c.Allocation(p['allocation']['nx'],p['allocation']['ny'],F(p['allocation']['share']))
                             for p in policy['cells']]
            for metric in metrics:
                mu={axis:list(map(F,prices['per_shot'][axis][metric]['exact'])) for axis in ('X','Y')}
                muq=list(map(F,prices['Q_per_shot'][metric]['exact']))
                if muq!=mu['Y']:
                    raise RuntimeError('Native Q/Y price identity broken')
                xy_lower=prefix[metric][0]+result['lower']*min(mu['X'][0],mu['Y'][0])
                xy=None if not allocations else c.native_cost(prefix[metric],mu,allocations)
                # H compares saved Q-H340; E compares matched Q-E, retaining
                # old 340 only as a separately labelled fallback if Q-E unknown.
                n=340 if row['stage']=='H' else qn
                qlo_n=340 if row['stage']=='H' else qlower
                qcost=None if n is None else accounts['add']([prefix[metric],accounts['scale'](n,muq)])
                qlower_cost=prefix[metric][0]+qlo_n*muq[0]
                verdict='UNDETERMINED'
                # Cancel the shared prefix exactly. Evaluate the SAME mu_Y
                # in Q and Y; interval endpoint policies are not independent.
                saving_lower=None if n is None else min(result['lower']*min(x,y)-n*y
                    for x,y in product(mu['X'],mu['Y']))
                baseline_minus_Q_upper=None if not allocations else max(
                    c.correlated_saving(qlo_n,p,mu)[1] for p in allocations)
                if saving_lower is not None and saving_lower>0:
                    verdict='Q_BEATS_CERTIFIED_FAMILY_COST_LOWER_BOUND'
                elif baseline_minus_Q_upper is not None and baseline_minus_Q_upper<0:
                    verdict='BASELINE_FEASIBLE_COST_BEATS_Q_FAMILY_LOWER_BOUND'
                output.append(dict(arm=arm,family=row['stage'],metric=metric,status=verdict,
                    baseline_cost_lower=str(xy_lower),baseline_feasible_cost=None if xy is None else list(map(str,xy)),
                    Q_cost=None if qcost is None else list(map(str,qcost)),Q_cost_lower=str(qlower_cost),
                    baseline_family_minus_Q_lower=None if saving_lower is None else str(saving_lower),
                    baseline_policy_minus_Q_family_upper=None if baseline_minus_Q_upper is None else str(baseline_minus_Q_upper),
                    shared_prefix_cancels=True,same_mu_Y_used=True,
                    Q_H340_preserved_fallback=True,prefix_history='worst case',
                    trajectory_price='saved-law expectation',classical_cost='UNPRICED'))
    return dict(status='CONDITIONAL_SAVED_PRICE_CONNECTION_ONLY',rows=output,
                peak_qubits=costs['qubits'],new_prices_or_compiles=0,world_optimality=False)


def save(path,obj):
    data=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    if path.name.endswith('.xz'):
        data=lzma.compress(data)
    with path.open('xb') as stream:
        stream.write(data)


def campaign(out):
    # This command must be called only AFTER explicit user execution permission.
    # Fresh directory prevents overwriting an old campaign/STOP; no grant system.
    if out.resolve()!=SCIENCE_OUT.resolve():
        raise ValueError('Campaign output must be the fixed science_run01 namespace')
    out.mkdir(parents=True,exist_ok=False)
    provenance=preflight();provenance['status']='NEW_H_E_C_CAMPAIGN_SOURCE'
    provenance.pop('scientific_campaigns_executed')
    provenance.pop('stop')
    provenance['recorded_before_scientific_execution']=True
    provenance['planned_calls']=dict(H=1,E=1,C='conditional saved-price accounting')
    save(out/'SOURCE.json',provenance)
    rows=[]
    try:
        for stage,specs in [
            ('H',[('H',arm,geometry,'XY') for arm,geometry in
                  product(('Bprime','Reference_B1'),('ideal','native'))]),
            ('E',[('E',arm,'native',kind) for arm,kind in
                  product(('Bprime','Reference_B1'),('XY','Q'))])]:
            workers=min(len(specs),len(os.sched_getaffinity(0)))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(task,spec):spec for spec in specs}
                for future in as_completed(futures):
                    row=future.result();rows.append(row)
                    save(out/('_'.join(futures[future])+'.json.xz'),row)
                    print(stage,row['arm'],row['geometry'],row['kind'],row['result']['lower'],row['result']['feasible'],flush=True)
        if any(r['geometry']=='native' and r['result']['feasible'] is not None for r in rows):
            save(out/'C.json',price_connection(rows))
        else:
            save(out/'C.json',dict(status='NOT_RUN_NO_QUALIFIED_NEW_SOLUTION',new_cost_calculations=0))
        save(out/'CHECKS.json',dict(status='AUTHOR_FINITE_CERTIFICATE_CHECKS_COMPLETE',
                                  scientific_review=False,old_campaigns_rerun=False))
        save(out/'STOP.json',dict(status='MANDATORY_STOP',automatic_expansion=False))
    except BaseException as error:
        save(out/'STOP.json',dict(status='MANDATORY_STOP_ERROR',reason=repr(error),
                                 partial_results_preserved=True,automatic_retry=False))
        raise


def check_saved(folder):
    source=json.loads((folder/'SOURCE.json').read_text())
    if source['base']!=g.BASE or source['evidence']!=g.EVIDENCE:
        raise ValueError('Fixed SOURCE/input references changed')
    for record in source['sources']:
        if hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Campaign SOURCE changed: '+record['path'])
    rows=[]
    for path in sorted(folder.glob('[HE]_*.json.xz')):
        row=json.loads(lzma.decompress(path.read_bytes()))
        model=c.Model(**{k:F(v) for k,v in row['model'].items()})
        check_solution(g.Arithmetic(),model,row['result'],row['stage'],row['geometry'],row['kind'])
        rows.append(row)
    expected={(stage,arm,geometry,kind) for stage,geometries,kinds in
              [('H',['ideal','native'],['XY']),('E',['native'],['XY','Q'])]
              for arm,geometry,kind in product(('Bprime','Reference_B1'),geometries,kinds)}
    if {(r['stage'],r['arm'],r['geometry'],r['kind']) for r in rows}!=expected or len(rows)!=8:
        raise ValueError('Incomplete H/E output set')
    saved_cost=json.loads((folder/'C.json').read_text())
    if saved_cost['status']!='NOT_RUN_NO_QUALIFIED_NEW_SOLUTION' and price_connection(rows)!=saved_cost:
        raise ValueError('Saved C accounting mismatch')
    return dict(status='FINITE_CERTIFICATES_REPLAYED_NO_OPTIMIZER',rows=len(rows),
                author_check_not_independent_review=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight',type=Path,help='metadata-only preparation JSON')
    modes.add_argument('--campaign',type=Path,help='requires separate scientific execution permission')
    modes.add_argument('--check',type=Path,help='replay an approved campaign; no optimizer/rerun')
    args=parser.parse_args()
    if args.preflight:
        args.preflight.parent.mkdir(parents=True,exist_ok=True)
        save(args.preflight,preflight())
        print('PREPARATION_ONLY_SCIENCE_NOT_EXECUTED')
    elif args.check:
        print(json.dumps(check_saved(args.check),sort_keys=True))
    else:
        campaign(args.campaign)


if __name__=='__main__':
    main()
