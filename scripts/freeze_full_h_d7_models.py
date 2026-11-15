"""D7: admit ten saved calibration means, fit two exact models, freeze four forecasts."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from types import ModuleType, SimpleNamespace

from trottertracks import price_implementation_guard as G
import admit_pr_df_ld1_q3_prices as C

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/full_h_d7_model_freeze_20261115'
D1='e628c0d69c7da559369d99f323e3656a96c0728a'
HANDOFF=dict(commit=D1,path='docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.json')
A_SOURCE='7a0c9468b2499fac9fa1cd279ae2980c9dab048e'
API_PATH='src/trottertracks/finite_price_design.py'
GATE_PATH='scripts/run_full_h_kappa2_p2.py'
CELLS={'G566':(1,566,2),'G4528':(1,4528,2),'G36223':(1,36223,2),'Q8R566':(8,566,2),'P2R566':(2,566,1)}


def at(ref,*selector):return dict(ref,selector=list(selector))


def pure_functions(root,commit,path,names,scope):
    tree=ast.parse(G.C.blob(root,commit,path));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.equal({n.name for n in nodes},set(names),'Pinned metadata functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),commit+':'+path,'exec'),scope)


def metadata_api(root):
    """Original A module + original D1 gate/seed/physical functions, no science launcher."""
    D=ModuleType('full_h_D7_A_design');D.__file__=A_SOURCE+':'+API_PATH
    exec(compile(G.C.blob(root,A_SOURCE,API_PATH),D.__file__,'exec'),D.__dict__)
    blob=lambda commit,path:G.C.blob(root,commit,path)
    env=dict(ast=ast,hashlib=hashlib,json=json,blob=blob,FULL_H=dict(commit='9b83b96b83f62142c9d87858dc24a94db7ac5d61',path='scripts/h46_full_h_pure_rte.py'))
    pure_functions(root,D1,'scripts/run_full_h_kappa2_p1.py',['seed_function'],env)
    # physical_row reads only the existing cid function from this local script.
    cid_path='scripts/h46_long_time_signal.py'
    def cid_node(raw):return next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='cid')
    C.equal(ast.dump(cid_node((Path(root)/cid_path).read_bytes()),include_attributes=False),
            ast.dump(cid_node(blob(D1,cid_path)),include_attributes=False),'Original physical-id helper')
    scope=dict(ROOT=Path(root),P1=SimpleNamespace(blob=blob,seed_function=env['seed_function'],COHORT='D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1'),
        ast=ast,json=json,hashlib=hashlib,deepcopy=deepcopy,design_api=lambda:D,A_SOURCE=A_SOURCE,
        A_DESIGNS='docs/research/a_full_h_p2_pauli_b1_price_design_20261114/DESIGNS.json',
        DESIGNS=dict(D6='D6_INDEPENDENT_Q8',D7='D7_INDEPENDENT_Q8_PLUS_PAIRED_Q2',D7_H2='D7_INDEPENDENT_Q8_PLUS_PAIRED_Q2',D0='D0_EXISTING_Q1_ONLY'))
    pure_functions(root,D1,GATE_PATH,['proposal','validate','physical_row','seed_table','frozen_model_gate'],scope)
    return D,scope


def checked_evidence(root,ref,sha=None):
    if any(token in ref['path'] for token in ('M5_ROUND/','M6_ROUND/')):raise ValueError('HOLDOUT_OBSERVATION_READ_FORBIDDEN')
    if sha:C.equal(hashlib.sha256(G.C.blob(root,ref['commit'],ref['path'])).hexdigest(),sha,'Original compiler bytes/hash')
    return G.evidence(root,ref)


def check_path(binding,price,evidence,compiled,record,cell):
    C.equal(evidence['price'],price,'Price/implementation evidence linkage')
    for key in ('cell_id','index','axis','cohort','seeds','physical'):C.equal(price[key],binding[key],'Original path '+key)
    C.equal(price['source'],binding['execution_source'],'Validator SOURCE')
    C.equal(price['metrics']['rz_count'],binding['rz_count'],'Direct RZ observation')
    C.equal(compiled,price['compile_receipt']['record'],'Original embedded compiler record')
    C.equal((compiled['fingerprint'],compiled['metrics'],compiled['compiler']),
            (price['fingerprint'],price['metrics'],record['profile']['compiler']),'Compile profile/fingerprint/metrics')
    physical=price['physical'];C.equal((physical['q'],physical['R_outer'],physical['s'],physical['family'],physical['L_D']),
        (cell['q'],cell['R'],cell['s'],'FULL_H_PURE_RTE',0),'Full-H physical grouping/domain')
    C.equal((record['cell_id'],record['axis'],record['source']),(price['cell_id'],price['axis'],price['source']),'Calibration record/path axis/SOURCE')
    C.equal(price['seeds'],record['source']['step_seeds'][price['index']],'Per-path seed')
    checks=evidence['checks']
    C.equal(checks['builder_lowering_id'],record['implementation']['lowering_id'],'New ZZ SOURCE route')
    C.equal(checks['parity']['status'],'PASS','Saved instruction parity')
    C.equal(checks['scalar_event_phases']['status'],'PASS','Saved scalar/event phase gate')
    C.equal(evidence['xy_phase_gate']['status'],'PASS','Saved X/Y phase gate')
    C.equal(checks['scalar'],checks['scalar_expected'],'Saved full-H scalar')
    C.equal(checks['legacy_rzz_control_instructions'],0,'No old ZZ substitution')
    C.equal(len(evidence['trial_checks']),7 if price['index']==0 else 0,'Numerical-gate scope index0 only')
    return dict(status='ACCEPTED_SAVED_PATH_AXIS',binding=deepcopy(binding),
        numerical_gate_scope='ORIGINAL_SEVEN_ACTION_GATES_INDEX0' if price['index']==0 else 'NO_ACTION_GATE_INDEX1',
        scalar=evidence['scalar'],preparation_hash=evidence['preparation_hash'],sampling=evidence['sampling'],
        producer_SOURCE=binding['compile_producer_source'],validation_SOURCE=binding['execution_source']['commit'])


def prepare(root):
    D,api=metadata_api(root);h=G.evidence(root,HANDOFF);fixed=G.evidence(root,h['fixed_plan']);plan=fixed['plan']
    C.equal(plan,api['proposal']('D7')|{'adopted':True},'Adopted fixed D7 design')
    C.equal(h['status'],'D7_CALIBRATION_COMPLETE_MANDATORY_STOP','Complete D1 calibration')
    C.equal(h['paired']['independent_third_sample'],False,'Paired dependence preserved')
    design=D.check_plan(plan);C.equal(design['identifiability']['rank'],3,'Design rank')
    admissions=[];paths=[]
    for ref in h['available_PRICE_CALIBRATION_RECORD_V1_locators']:
        record=checked_evidence(root,ref);cid=record['cell_id'];axis=record['axis'];cell=next(c for c in plan['cells'] if c['cell_id']==cid)
        entry=dict(locator=ref,record=record,status='HELD',reason=None)
        try:
            C.equal((cell['q'],cell['R'],cell['n_paths']),CELLS[cid],'Five fixed calibration cells')
            C.equal(record['record'],'PRICE_CALIBRATION_RECORD_V1','Dedicated non-configuration record')
            C.equal((record['role'],record['route'],record['axis']),('CALIBRATION','FULL_H_RTE',axis),'Calibration domain')
            C.equal(record['features'],{k:dict(numerator=v.numerator,denominator=v.denominator) for k,v in D.features(cell,plan['feature_map']).items()},'FULL_H_GROUP_V1 features')
            for key in D.DOMAIN_KEYS:C.equal(record[key],cell[key],'Record '+key)
            C.equal(record['n_paths'],cell['n_paths'],'Saved finite cohort count')
            C.require(record['finite_sample_mean_is_law_expectation'] is False,'Finite cohort is not law expectation')
            C.equal(record['source']['cohort'],cell['cohort'],'Original cohort')
            C.equal(record['source']['step_seeds'],api['seed_table'](plan,cid),'Original seed table')
            if cid=='G36223':C.equal(ref['commit'],'f98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1','Completed G36223, not partial')
            bindings=[b for b in h['cells'][cid]['path_axis_bindings'] if b['axis']==axis]
            C.equal(sorted(b['index'] for b in bindings),list(range(cell['n_paths'])),'Unique completed path axis set')
            side=next(s for s in h['cells'][cid]['bound_sidecars'] if s['axis']==axis)
            proof=G.evidence(root,side['implementation_evidence'])
            C.equal(proof['calibration_record'],record,'Bound calibration evidence')
            C.equal((proof['implementation'],proof['profile']),(record['implementation'],record['profile']),'Implementation/profile')
            C.equal(proof['measurement'],dict(axis=axis,include_measurement=True,state_preparation_cost_included=False),'Measured wrapper/P0')
            C.equal(proof['paired_dependence'],cell.get('composed_of'),'Paired design role')
            accepted=[]
            for b in bindings:
                C.equal(b['calibration_record'],ref,'Binding record locator')
                price=checked_evidence(root,b['direct_price']);ev=checked_evidence(root,b['implementation_evidence'])
                compiled=checked_evidence(root,b['compile_record'],b['compile_record_sha256'])
                C.equal(proof['paths'][b['index']],ev,'Per-path bound proof')
                accepted.append(check_path(b,price,ev,compiled,record,cell))
            entry.update(status='ACCEPTED_CALIBRATION_RECORD',implementation_evidence=side['implementation_evidence'],
                paths_start=len(paths),path_count=len(accepted),original_correctness_scope=proof['correctness_scope'])
            paths.extend(accepted)
        except (KeyError,ValueError,OSError) as error:entry['reason']=str(error)
        admissions.append(entry)
    C.equal(len(admissions),10,'Ten calibration records')
    for axis in ('X','Y'):C.equal(sorted(r['record']['cell_id'] for r in admissions if r['record']['axis']==axis),sorted(CELLS),'Five individual records per axis')
    holdouts=[]
    for cell in plan['cells']:
        if cell['role']=='HOLDOUT':
            holdouts.append(dict(cell=cell,features={k:dict(numerator=v.numerator,denominator=v.denominator) for k,v in D.features(cell,plan['feature_map']).items()},
                step_seeds=api['seed_table'](plan,cell['cell_id']),physical=api['physical_row'](cell),observed_price=None))
    return dict(schema='FULL_H_D7_C_INPUT_ADMISSION_V1',handoff=HANDOFF,plan_source=h['fixed_plan'],plan=plan,
        identifiability=design['identifiability'],calibration=admissions,path_axes=paths,paired=h['paired'],
        historical_partial='G36223_partial and zero-compile continuation axes excluded from extra counts',
        reused_D1_publication_checks=h['D1_checked'],raw_decoder=h['raw_receipt_decoder'],holdouts=holdouts,
        admission_counts=dict(records_accepted=sum(r['status']=='ACCEPTED_CALIBRATION_RECORD' for r in admissions),records_held=sum(r['status']=='HELD' for r in admissions),path_axes=len(paths)),
        canonical_round_guard_invoked=False,new_means_computed=False,fit_executed=False)


def preopening(root,*,after_freeze=False):
    refs=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/*full*h*','refs/heads/codex/m5-m4-receipt-admission-20261114'],cwd=root,text=True).splitlines()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','docs/research'],cwd=root,text=True).splitlines()
    duplicates=[p for p in names if any(s in p.lower() for s in ('full_h_d7','full_h_p2_fit','full_h_model_freeze')) and not p.startswith(BASE+'/')]
    jobs=[]
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,comm=,args='],text=True).splitlines():
        a=line.split(None,2)
        if len(a)!=3 or a[0]==str(os.getpid()) or 'python' not in a[1]:continue
        args=shlex.split(a[2]);scripts=[Path(s).name for s in args if s.endswith('.py')]
        if any('full_h' in s and any(k in s for k in ('freeze','fit','holdout')) for s in scripts) or ('--stage' in args and 'HOLDOUT' in args and any('full_h' in s for s in scripts)):
            jobs.append(dict(pid=int(a[0]),scripts=scripts,holdout_stage='HOLDOUT' in args))
    directories=[]
    for d in (Path('/home/AbeHiromu/projects/d-full-h-p2-calibration-20261115-private'),Path('/home/AbeHiromu/projects/d-full-h-p2-calibration-20261115-private/resume01')):
        directories.append(dict(path=str(d),names=sorted(p.name for p in d.iterdir()) if d.exists() else []))
    holdout_names=[n for d in directories for n in d['names'] if any(k in n.lower() for k in ('m5_round','m6_round','holdout'))]
    holdout_refs=[r for r in refs if 'holdout' in r.lower() and 'full' in r.lower()]
    existing=(Path(root)/BASE/'run01/FROZEN_MODELS.json').exists()
    clear=not(duplicates or jobs or holdout_names or holdout_refs or (existing and not after_freeze))
    return dict(status='CLEAR_IN_CHECKED_METADATA' if clear else 'HELD_PENDING_METADATA_RESOLUTION',UTC=datetime.now(timezone.utc).isoformat(),
        public_refs=refs,duplicate_candidates=duplicates,relevant_jobs=jobs,known_directories=directories,
        holdout_name_candidates=holdout_names,holdout_ref_candidates=holdout_refs,own_output_exists=existing,holdout_price_reads=0,
        limitation='Related fixed publications, focused ref/path/job names, two known private directory names only; not an exhaustive search of unknown archives/hosts')


def calculate(root,data,source):
    D,api=metadata_api(root);plan=data['plan'];models={};forecasts=[];held=[]
    for axis in ('X','Y'):
        selected=[r for r in data['calibration'] if r['record']['axis']==axis]
        if any(r['status']!='ACCEPTED_CALIBRATION_RECORD' for r in selected):held.append(axis);continue
        records=[r['record'] for r in selected]
        model=D.fit_identified(root,records,plan=plan,axis=axis)['model']
        model['input_locators']=[r['locator'] for r in selected]
        model.update(rank=3,feature_map='FULL_H_GROUP_V1',objective='UNWEIGHTED_SUM_OF_FIVE_CELL_MEAN_SQUARED_RESIDUALS',
                     calibration_residual_definition='ABS_OBSERVED_MINUS_FITTED_CELL_MEAN',paired_dependence=data['paired'],coefficient_uncertainty=None)
        models[axis]=model
        for point in data['holdouts']:
            cell=point['cell'];value=D.c_model(root).predict(model,point['features'])
            forecasts.append(dict(cell_id=cell['cell_id'],axis=axis,m=5 if cell['cell_id']=='M5_ROUND' else 6,
                physical=point['physical'],q=cell['q'],R=cell['R'],n_paths=cell['n_paths'],cohort=cell['cohort'],
                step_seeds=point['step_seeds'],features=point['features'],predicted_RZ=dict(numerator=value.numerator,denominator=value.denominator),
                display_RZ=float(value),observed_RZ=None,model_verdict=None,tolerance=None,qualified_for_candidate=False))
    return dict(schema='FULL_H_D7_TWO_MODELS_FOUR_FORECASTS_V1',execution_source=source,input_source=dict(commit=source,path=BASE+'/INPUTS.json'),
        status='CALIBRATED_UNVALIDATED',models=models,predictions=forecasts,held_axes=held,fit_calls=len(models),prediction_calls=len(forecasts),
        cell_weights='ONE_PER_CELL_NO_PATH_COUNT_WEIGHTS_NO_XY_POOL',qualified_for_candidate=False,
        CI=None,all_law_E_C=None,sampler_ideal_distance=None,rare_contribution=None,shared_covariance=None,G=None,winner=None,mandatory_stop=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args();directory=ROOT/BASE
    if a.prepare:
        C.require(not directory.exists(),'Reuse existing prepared inputs')
        pre=preopening(ROOT);C.equal(pre['status'],'CLEAR_IN_CHECKED_METADATA','Resolve duplicate/holdout metadata')
        data=prepare(ROOT);directory.mkdir(parents=True)
        for name,value in (('INPUTS.json',data),('PREOPEN_CHECK.json',pre)):(directory/name).write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(data['admission_counts']));return
    if not a.execute:raise SystemExit('Choose --prepare or --execute')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in ('scripts/freeze_full_h_d7_models.py',BASE+'/INPUTS.json'):
        C.equal((ROOT/path).read_bytes(),G.C.blob(ROOT,source,path),'Fixed SOURCE/input')
    pre=preopening(ROOT);C.equal(pre['status'],'CLEAR_IN_CHECKED_METADATA','Resolve duplicate/holdout metadata')
    out=directory/'run01';out.mkdir(exist_ok=False);start=time.perf_counter();print('BEGIN D7 exact two-axis five-cell fit and four unopened forecasts',flush=True)
    try:result=calculate(ROOT,json.loads((directory/'INPUTS.json').read_text()),source)
    except Exception as error:
        (out/'STOP.json').write_text(json.dumps(dict(status='STOP_NO_RETRY',source=source,error=repr(error),mandatory_stop=True),indent=2)+'\n');raise
    result['preopening']=pre;result['wall_seconds']=time.perf_counter()-start
    (out/'FROZEN_MODELS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(fit_calls=result['fit_calls'],forecasts=result['prediction_calls'],held_axes=result['held_axes'],holdout_price_reads=0,wall_seconds=result['wall_seconds'],mandatory_stop=True)),flush=True)


if __name__=='__main__':main()
