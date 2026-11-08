"""One bounded local extension of the saved-qualified h/resolution menu.

Use the coarse study as training, including its now explicitly retired
holdout. Final validation has a DIFFERENT phase/seed namespace. A cheap
price proxy screens new conditions; reported totals use direct native MC.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
from . import track_c_closed_p3_joint_opt as j
from . import track_c_d_qualify as ql
from . import track_c_compiler_a_18round as ca
from .track_c_bottleneck_joint import new_signal
from .track_c_closed_p3_joint_opt_alpha import discrete_allocate

COARSE=j.ROOT/'artifacts/track_c_closed_p3_joint_opt/2026-11-08'
assert j.OUT!=COARSE,'The extension must have its own output namespace'
_old_load=j.cr.load_prices


def load_prices(phase):
    if phase=='training':dirs=[(COARSE/'training','coarse training'),(COARSE/'holdout','coarse holdout RECLASSIFIED AS TRAINING'),(j.OUT/'extension_training','new extension training')]
    elif phase=='holdout':dirs=[(j.OUT/'extension_holdout','fresh final holdout')]
    else:return _old_load(phase)
    result={}
    for directory,role in dirs:
        for p in sorted(directory.glob('*.json.gz')):
            with gzip.open(p,'rt') as f:r=json.load(f)
            k=j.cp.cellkey(r['t'],r['n']);rows=result.setdefault(k,{}).setdefault(str(r['rare']),[])
            r['training_or_evaluation_origin']=role;r['j']=len(rows);rows.append(r)
    return result


j.cr.load_prices=load_prices


def candidates():
    """Existing raw grid, adjacent dyadic h, neighboring/coarse mid-rates."""
    coarse=json.loads((COARSE/'MENU.json').read_text());frozen=json.loads((COARSE/'POLICY_REFINED.json').read_text());rs=load_prices('training')
    # Proxy/anchor ranking uses ONLY data explicitly designated training here.
    original_dets=j.dets
    j.dets=lambda:{(r['case']['name'],r['h']):r for r in json.loads((COARSE/'DETERMINISTIC.json').read_text())['rows']}
    anchors={};rates={};rows=[];raw=j.archived('track_c_d_internal','SIGNALS.json')['rows'];seen={tuple(s['key']) for s in coarse['rows']}
    for m in range(12,18):
        levels=set()
        for method in frozen['methods'].values():
            if method['family']=='PR':levels.add(method['selected'][m]['signal']['rate'])
        rate_pool=set()
        for rate in levels:
            for v in (rate/2,rate,2*rate,.75*rate,1.5*rate):
                if int(v)==v and 1<=v<=64:rate_pool.add(int(v))
        rates[m]=sorted(rate_pool)
        for name in ('tuned_0123','tuned_0321'):
            hh=set()
            for arm in ('canonical','closed'):
                old=[s for s in coarse['rows'] if s['family']=='PR' and s['signal']['m']==m and s['signal']['name']==name]
                pp=[j.point(s,rs,arm) for s in old];alpha=frozen['methods']['A_canonical' if arm=='canonical' else 'B_closed']['alpha'][m]
                best=min(pp,key=lambda p:j.score(p,alpha));h=best['signal']['h'];hh.update(v for v in (h/2,h,2*h) if .005<=v<=1.28)
            anchors[f'{name}:{m}']=sorted(hh)
            for h in sorted(hh):
                base=next(x for x in raw if x['name']==name and x['m']==m and x['h']==h and x['requested_h']==h and x['rate']==32)
                for rate in rates[m]:
                    key=(name,m,h,rate)
                    if key in seen:continue
                    x=new_signal(base,rate)
                    if x['PF_phase']>=.4:continue
                    rows.append(x);seen.add(key)
    j.dets=original_dets
    return coarse,rows,dict(rounds=list(range(12,18)),D_cases=['tuned_0123','tuned_0321'],h_anchors=anchors,rates=rates,
        boundary='ONE local extension only; old qualified menu retained, neighboring dyadic h and rates/mid-rates <=64. No new D coefficient/family, order, split, molecule, or sampling law.')


def primitive_bounds(signals):
    bounds=dict(ql.old()[3]);old=j.archived('track_c_d_internal','LOCAL_INTERVALS.json');bounds.update(old['primitive_bounds'])
    for study,file in [('track_c_d_internal','QUALIFICATION_EXTENSION.json'),('track_c_bottleneck','JOINT_NATIVE_SOURCE.json')]:
        for r in j.archived(study,file)['checked']:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
    missing={}
    for x in signals:
        seq=j.df.split(x['case'],x['h'])
        from .track_c_d_internal import schedule
        actual=schedule(x['case'],x['h'])
        for _,i,t in [a for a in actual if a[0]=='D']+[('D',actual[0][1],2*actual[0][2])]:
            for k,s in ql.d_counts(i,t)[1].items():
                if k not in bounds:missing[k]=s
        for recipe in x['recipe']:
            tau=j.common()['preps'][3].exact_rte_lambda_r*recipe['time']/recipe['r']
            for order in (0,2):
                for sign in (-1,1):
                    for family in ('crz','crzz'):
                        s=ca.spec(family,[sign*2*math.atan(tau/(order+1))])
                        if s['key'] not in bounds:missing[s['key']]=s
    checked=[];contexts=[]
    if missing:
        catalogue={}
        for s in ca.inventory()['catalogue'].values():
            if s['family'] not in {z['family'] for z in catalogue.values()}:catalogue[s['key']]=s
        catalogue.update(missing);unique={}
        with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            for f in as_completed(pool.submit(ca.context_compile,(mask,catalogue)) for mask in range(128)):
                r=f.result();contexts.append({k:r[k] for k in ('mask','context','post_unroll')})
                for v in r['records']:unique[v['lowering_sha256']]=v
            checked=list(pool.map(ca.interval_check,unique.values()))
        for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
    j.save_json(j.OUT/'NEW_NATIVE_INTERVALS.json',dict(**j.provenance(),contexts=contexts,checked=checked,missing_primitive_keys=list(missing),primitive_bounds=bounds))
    return bounds


def certify_cells(cells):
    old=json.loads((COARSE/'STAGE1.json').read_text());byx={r['certificate']['x_exact']:r for r in old['cells']};todo={}
    for t,n in cells:
        x=str(F.from_float(j.common()['preps'][3].exact_rte_lambda_r)*F.from_float(t)/n)
        if x not in byx:todo.setdefault(x,(t,n))
    if todo:
        with ProcessPoolExecutor(max_workers=min(len(todo),len(os.sched_getaffinity(0)))) as pool:
            for f in as_completed(pool.submit(j.cv.cell_check,v) for v in todo.values()):
                r=f.result();byx[r['certificate']['x_exact']]=r
    records=[]
    for t,n in cells:
        x=str(F.from_float(j.common()['preps'][3].exact_rte_lambda_r)*F.from_float(t)/n);a=byx[x]
        records.append({**a,'t':t,'n':n,'reused_from_cell':[a['t'],a['n']]})
    j.save_json(j.OUT/'STAGE1.json',dict(**j.provenance(),cells=records,passed=True,new_exact_microstep_certificates=len(todo),
        inherited_artificial_and_operator_binding=old['inherited_artificial_and_operator_binding']))


def proxy(s,rs,arm):
    """Screening only. Its R estimate is NEVER used in final resources."""
    x=s['signal'];q=s['closed_qualification'] if arm=='closed' else s['qualification'];d=j.dets()[(x['name'],x['h'])];g={k:np.array(v,float) for k,v in d['groups'].items()};steps=x['steps']
    D=g['first']+g['last']+(steps-1)*g['junction']+steps*g['inside'];R=np.zeros(4);refs=[]
    for recipe in x['recipe']:
        t,n=recipe['time'],recipe['r']
        k=min(rs,key=lambda k:abs(math.log((rs[k]['False'][0]['t']/rs[k]['False'][0]['n'])/(t/n)))+.25*abs(math.log(rs[k]['False'][0]['n']/n)))
        mu=sum((z[0]['stratum_probability'][arm]*np.mean([a[arm] for a in z],axis=0) for z in rs[k].values()),np.zeros(4));oldn=rs[k]['False'][0]['n']
        R+=steps*n/oldn*mu;refs.append(k)
    over=j.wrapper_price(3,x['t']);return dict(signal=x,qualification=q,mean=D+R+over),refs


def inventory():
    if (j.OUT/'MENU.json').exists():raise RuntimeError('Preserve extension menu')
    j.prepare();coarse,signals,plan=candidates();j.save_json(j.OUT/'EXTENSION_PLAN.json',dict(**j.provenance(),**plan,conditions_to_qualify=len(signals)))
    print('EXTENSION_CONDITIONS',len(signals),flush=True)
    bounds=primitive_bounds(signals);qualified=[]
    with ProcessPoolExecutor(max_workers=min(len(signals),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed(pool.submit(ql.qualify,(x,bounds)) for x in signals):qualified.append(f.result())
    j.save_json(j.OUT/'NEW_SOURCE_NATIVE_QUALIFICATION.json',dict(**j.provenance(),rows=qualified))
    lookup={j.identity(x):x for x in signals};certificate=json.loads((j.cp.INPUT/'joint_certificate.json').read_text());eligible=[];rejected=[]
    for q in qualified:
        x=lookup[j.identity(q)];p=q['dominant_overlap']
        if not q['closed'] or p<=.5 or q['dominant_accumulated_bias_abs']+math.asin(min(1.,(1-p)/p))>=.4:
            rejected.append(dict(key=list(j.identity(q)),closed=q['closed'],reason='candidate native/PF/nonresonant condition fails'));continue
        cq=j.fusion_qualify(j.supplement(q,x,certificate),x)
        if not cq['closed']:rejected.append(dict(key=list(j.identity(q)),reason='joint/diagonal supplement fails'));continue
        x['qualification']='Candidate-specific directed SOURCE/native certificate is stored alongside this new-rate signal'
        eligible.append(dict(key=list(j.identity(x)),signal=x,qualification=cq,family='PR',saved_native_qualification=q))
    cells=sorted({(r['time'],r['r']) for s in coarse['rows']+eligible for r in s['signal']['recipe']});certify_cells(cells)
    for s in eligible:s['closed_qualification']=j.cr.qualify(s)
    # These are the already measured SAME D groups, not new simulations.
    (j.OUT/'det_prices').mkdir(exist_ok=True)
    for p in (COARSE/'det_prices').glob('*.json'):shutil.copyfile(p,j.OUT/'det_prices'/p.name)
    shutil.copyfile(COARSE/'DETERMINISTIC.json',j.OUT/'DETERMINISTIC.json')
    missing={(json.dumps(s['signal']['case'],sort_keys=True),s['signal']['h']) for s in eligible if (s['signal']['name'],s['signal']['h']) not in j.dets()}
    if missing:
        with ProcessPoolExecutor(max_workers=min(len(missing),len(os.sched_getaffinity(0)))) as pool:
            for f in as_completed(pool.submit(j.det_price,(json.loads(c),h)) for c,h in missing):f.result()
        data=json.loads((j.OUT/'DETERMINISTIC.json').read_text());data['rows']=[json.loads(p.read_text()) for p in (j.OUT/'det_prices').glob('*.json')];j.save_json(j.OUT/'DETERMINISTIC.json',data);j.dets.cache_clear()
    rs=load_prices('training');frozen=json.loads((COARSE/'POLICY_REFINED.json').read_text());selected={};screen=[]
    for m in range(12,18):
        for arm in ('canonical','closed'):
            aa=frozen['methods']['A_canonical' if arm=='canonical' else 'B_closed']['alpha'][m];ranked=[]
            for s in eligible:
                if s['signal']['m']!=m:continue
                p,refs=proxy(s,rs,arm);ranked.append((j.score(p,aa),s,refs))
            ranked.sort(key=lambda r:(r[0],r[1]['key']))
            for value,s,refs in ranked[:2]:selected[tuple(s['key'])]=s
            screen.append(dict(m=m,arm=arm,candidates=[dict(key=s['key'],proxy_RZ=value,reference_cells=refs) for value,s,refs in ranked]))
    allrows=coarse['rows']+list(selected.values());actualcells=sorted({(r['time'],r['r']) for s in allrows for r in s['signal']['recipe']})
    j.save_json(j.OUT/'MENU.json',dict(**j.provenance(),rows=allrows,cells=actualcells,coarse_menu_conditions=len(coarse['rows']),
        qualified_new_conditions=len(eligible),direct_price_candidates_added=len(selected),unqualified=rejected,
        proxy_screening=screen,qualified_but_not_directly_priced=[s for s in eligible if tuple(s['key']) not in selected],
        limitation='Proxy only pre-screens new nodes; all retained coarse nodes and union of both arms top two new nodes/round get actual direct MC. Same final menu for both P3. Not a global optimum.'))
    j.save_json(j.OUT/'TRAINING_INPUTS.json',dict(**j.provenance(),sources=[str((COARSE/s).relative_to(j.ROOT)) for s in ('training','holdout')],
        role='BOTH coarse datasets are TRAINING here. They are excluded from final independent extension_holdout.',seed_namespaces=['ClosedP3:training/holdout','P3_joint_opt:training/holdout','P3_joint_opt:extension_training'],
        final_independent_namespace='P3_joint_opt:extension_holdout'))
    print('EXTENDED_MENU',len(allrows),'NEW_PRICE_NODES',len(selected),'CELLS',len(actualcells),flush=True)


def training():
    rs=load_prices('training');missing=[(t,n) for t,n in j.menu()['cells'] if j.cp.cellkey(t,n) not in rs]
    j.prices('extension_training',missing,32)
    j.freeze()


def holdout():
    f=json.loads(j.policy_path().read_text());allp=list(f['methods'].values())+list(f['ablations'].values());cells=sorted({(r['time'],r['r']) for p in allp for s in p['selected'] for r in s['signal']['recipe']})
    assert json.loads((j.OUT/'SELECTED_CORRECTNESS.json').read_text())['passed']
    # All conditions are frozen BEFORE these new seeds are generated.
    samples={j.cp.cellkey(t,n):(128 if n>=256 else 32) for t,n in cells}
    digest=hashlib.sha256(j.policy_path().read_bytes()).hexdigest()
    assert not (j.OUT/'HOLDOUT_PLAN.json').exists(),'Preserve the final independent holdout plan'
    j.save_json(j.OUT/'HOLDOUT_PLAN.json',dict(**j.provenance(),policy_sha256=digest,
        samples_per_cell=samples,independent_seed_namespace='P3_joint_opt:extension_holdout',coarse_data_role='training only',
        boundary='One local menu extension, then final independent evaluation and mandatory STOP. No further parameter family expansion.'))
    j.prices('extension_holdout',cells,samples)
    assert hashlib.sha256(j.policy_path().read_bytes()).hexdigest()==digest
    j.finish()


def run(stage):
    if stage=='inventory':inventory()
    elif stage=='training':training()
    elif stage=='correctness':
        from .track_c_closed_p3_joint_opt_checks import selected_correctness
        selected_correctness()
    elif stage=='holdout':holdout()
    elif stage=='finish':j.finish()
    else:raise ValueError(stage)
