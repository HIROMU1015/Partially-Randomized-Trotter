"""Fixed two-axis LD2 full-wrapper fit and candidate-owned rational allocations."""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld_split_results_20261115'
CALLER='scripts/freeze_pr_df_ld_split.py'
V0=[64,60,56,52,47,43,38,34,30,25,21,17,12]
GENERATOR=('e094b2d60c5c4468711dea244fa2f91fb587ac12','scripts/run_pauli_ld2_reallocation_q3_fixed.py')
SOLVER=('d83737b6d5241962ce39c81d39c56af3506c0cdd','src/trottertracks/price_model_validation.py')
Q3='a82b0f0ec7f0f11d9981ddb2343fcd5378a87503'
QDIR='docs/research/pr_df_fullh_q3_fixed_results_20261115/'

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def read(commit,path):return json.loads(blob(commit,path))
def enc(x):x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)
def dec(x):return Fraction(x['numerator'],x['denominator']) if isinstance(x,dict)else Fraction(x)
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def exact_functions(ref,names):
    tree=ast.parse(blob(*ref));scope=dict(Fraction=Fraction)
    code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[])
    exec(compile(code,ref[0]+':'+ref[1],'exec'),scope)
    return scope

generate=exact_functions(GENERATOR,['generate'])['generate']
solve=exact_functions(SOLVER,['_solve'])['_solve']

def fit_means(cells):
    if [r['m']for r in cells]!=[0,3,6]:raise ValueError('Complete fixed calibration required')
    if [(r['q'],r['R'])for r in cells]!=[(1,1),(8,5),(64,40)]:raise ValueError('Fixed features differ')
    x=[[Fraction(1),Fraction(r['q']),Fraction(r['q']*r['R'])]for r in cells]
    y=[dec(r['mean_C'])for r in cells]
    co=solve(x,y)
    if any(sum(c*v for c,v in zip(co,row))!=yi for row,yi in zip(x,y)):raise ValueError('Exact solve mismatch')
    return co

def freeze(source,raw):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed fit SOURCE')
    acquisition=read(raw,PUBLIC+'/acquisition/RESULT.json')
    inputs=read(raw,PUBLIC+'/ACQUISITION_INPUTS.json')
    if acquisition['status']!='ACQUIRED' or len(acquisition['rows'])!=12 or acquisition['completed_compiles']!=24:raise ValueError('Calibration incomplete')
    means=[];models={};ld2rows=[]
    config=read('a9abf8de75bf3432fcd714e868e5c14c8884b8f8','docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD2.json')
    for axis in ('X','Y'):
        axismeans=[]
        for cell in inputs['cells']:
            m=cell['m'];p=cell['request']['physical_context'];records=sorted([r for r in acquisition['rows']if r['m']==m],key=lambda r:r['index'])
            if len(records)!=4 or [r['index']for r in records]!=list(range(4)):raise ValueError('Path indices missing')
            paths=[]
            for r in records:
                v=r['record'];a=next(x for x in v['records']if x['axis']==axis)
                if r['status']!='VALID' or v['request']!=cell['request']or v['seeds']!=cell['seeds'][r['index']] or v['cohort']!=cell['cohort'] or v['index']!=r['index']or a['C']!=a['metrics']['rz_count'] or a['D']!=a['C']-a['S']:raise ValueError('Receipt identity mismatch')
                axisname={'X':'cosine','Y':'sine'}[axis]
                locator=dict(commit=raw,path=PUBLIC+f'/acquisition/m{m}/p{r["index"]}/{axisname}.json')
                receipt=read(locator['commit'],locator['path'])
                if receipt['fingerprint']!=a['fingerprint']or receipt['metrics']!=a['metrics']or receipt['compiler']!=p['price_profile']['compiler']:raise ValueError('Original receipt mismatch')
                paths.append(dict(index=r['index'],axis=axis,S=a['S'],C=a['C'],D=a['D'],fingerprint=a['fingerprint'],source=locator,
                    path_receipt=dict(commit=raw,path=PUBLIC+f'/acquisition/m{m}/p{r["index"]}/PATH_RECEIPT.json'),seeds=v['seeds'],cohort=v['cohort']))
            c=p['physical_candidate'];mean=sum(Fraction(x['C'])for x in paths)/4
            v=dict(m=m,axis=axis,physical_id=c['id'],q=c['q'],R=c['R_outer'],n_paths=4,mean_C=enc(mean),law=cell['request']['law_binding'],request=cell['request'],paths=paths)
            axismeans.append(v);means.append(v)
        co=fit_means(axismeans)
        models[axis]=dict(features=['one','q','qR'],offset=0,coefficients={n:enc(v)for n,v in zip(('one','q','qR'),co)},
            method='EXACT_SQUARE_SOLVE',calibration_rounds=[0,3,6],calibration_residuals=[enc(0)]*3,
            domain='LD2_FIXED_PREFIX_S2_DELTA0P1_K2_SCHEDULE_FAMILY_WITH_POINT_SPECIFIC_LAWS',
            calibration_sources=[dict(m=r['m'],law=r['law'],paths=r['paths'])for r in axismeans],
            validation='NOT_ASSESSED_NO_INDEPENDENT_PRICE_HOLDOUT',solver_source=dict(commit=SOLVER[0],path=SOLVER[1],selector=['_solve']))
        for row in config['rounds']:
            m=row['m'];c=row['physical_candidate'];features=(1,c['q'],c['q']*c['R_outer']);price=sum(x*y for x,y in zip(co,features))
            if price<=0:raise ValueError('Nonpositive reference; no clipping')
            ld2rows.append(dict(L_D=2,m=m,axis=axis,configuration_id=config['id'],physical_id=c['id'],q=c['q'],R_outer=c['R_outer'],C_ref=enc(price),
                application='CALIBRATION'if m in (0,3,6)else'LONG_ROUND_EXTRAPOLATION'if m>6 else'UNCHECKED_INTERPOLATION',
                law=row['law_binding'],source=dict(model=PUBLIC+'/FROZEN_LD2_MODELS.json',configuration=dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD2.json',selector=['rounds',m]))))
    ld2rows.sort(key=lambda r:(r['m'],r['axis']))
    ld1=read('85dab58d6d29d54502a7650c129bc38a9bb5ec99','docs/research/d_pr_df_ld1_reference_budget_20261115/run01/RESULT.json')
    rows=[dict(L_D=1,**r)for r in ld1['rows']]+ld2rows
    write(ROOT/PUBLIC/'FROZEN_LD2_MODELS.json',dict(source=source,raw_commit=raw,formula='C_ref=theta0+thetaq*q+thetaE*qR; full wrapper; model offset zero',models=models,means=means,
        finite_cohort_is_law_expectation=False,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',long_round_warning='m7..12 extrapolate beyond q64/R40; three-point interpolation is not a validation'))
    write(ROOT/PUBLIC/'AXIS_PRICES.json',dict(source=source,rows=rows,LD1_refit=False,LD1_prices_reused=26,new_LD2_references=26))
    return allocate(source)

def allocate(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed allocation SOURCE')
    fixed=json.loads((ROOT/PUBLIC/'FROZEN_LD2_MODELS.json').read_text())
    raw=fixed['raw_commit']
    rows=json.loads((ROOT/PUBLIC/'AXIS_PRICES.json').read_text())['rows']
    old=read(Q3,QDIR+'INPUTS.json');menu=[];cases=[]
    for L in (1,2):
        prices=[dec(r['C_ref'])for r in rows if r['L_D']==L]
        caps=[n for n in V0 for a in ('X','Y')]
        total=sum(n*c for n,c in zip(caps,prices));budget=Fraction(3,4)*total
        baseline=deepcopy(next(c for c in old['cases']if c['label']==f'PR_DF_LD{L}'))
        base_result=read(Q3,QDIR+f'run01/PR_DF_LD{L}/RESULT.json')
        menu.append(dict(L_D=L,rule='V0',shots=baseline['allocation']['shots'],cost=enc(total),budget=None,unused_budget=None,
            Q3_source=dict(commit=Q3,path=QDIR+f'run01/PR_DF_LD{L}/RESULT.json'),status=base_result['status'],disposition='REUSED_V0'))
        used={tuple(caps):menu[-1]}
        for rule in ('A_SHAPE','B_PRICE'):
            g=generate(caps,prices,budget,rule)
            if g is None:raise ValueError('Budget below one-shot floor')
            vector=g['shots'];shots=[dict(m=m,N_X=vector[2*m],N_Y=vector[2*m+1])for m in range(13)]
            cost=sum(n*c for n,c in zip(vector,prices))
            if cost!=g['conditional_cost']or cost>budget or any(not 1<=n<=cap for n,cap in zip(vector,caps)):raise ValueError('Allocation mismatch')
            label=f'LD{L}_{rule}';prior=used.get(tuple(vector))
            entry=dict(L_D=L,rule=rule,label=label,shots=shots,prices=[enc(x)for x in prices],cost=enc(cost),budget=enc(budget),unused_budget=enc(budget-cost),
                lambda_value=enc(g['lambda_value']),next_breakpoint_cost=None if g['next_breakpoint_cost']is None else enc(g['next_breakpoint_cost']),
                disposition='REUSED_DUPLICATE_VECTOR'if prior else'NEW_Q3_PENDING',reuse_label=None if prior is None else prior.get('label','V0'),
                generator=dict(commit=GENERATOR[0],path=GENERATOR[1],selector=['generate']))
            menu.append(entry);used[tuple(vector)]=entry
            if prior:continue
            case=deepcopy(baseline);case['label']=label;case['L_D']=L;case['rule']=rule
            case['allocation']=dict(allocation_id='PR_DF_SPLIT_'+label+'_3O4_01',shot_policy='USER_SUPPLIED',shots=shots)
            case['reference_cost']=enc(cost);case['budget']=enc(budget);cases.append(case)
    if len(cases)>4:raise ValueError('Q3 call cap')
    write(ROOT/PUBLIC/'ALLOCATION_INPUTS.json',dict(source=source,raw_commit=raw,menu=menu,cases=cases,maximum_new_Q3=4,
        fixed_Q3=dict(commit=Q3,path=QDIR+'INPUTS.json'),generator=dict(commit=GENERATOR[0],path=GENERATOR[1]),
        options=old['options'],statistical_contract=old['statistical_contract'],adopted_evaluation=old['adopted_evaluation'],acceleration=old['acceleration'],
        selection='Minimum actual model cost among Q3 admitted V0/A_SHAPE/B_PRICE; retain ties',price_quality='MODEL_REFERENCE_NOT_QUALIFIED_EXPECTATION',new_signal=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw');p.add_argument('--allocations-only',action='store_true');a=p.parse_args();allocate(a.source) if a.allocations_only else freeze(a.source,a.raw)
