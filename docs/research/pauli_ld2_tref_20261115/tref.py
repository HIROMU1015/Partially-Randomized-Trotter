"""PR-Pauli LD2 fixed-shot model-reference total T_ref and PF reversal boundaries.

Exact rational arithmetic only (fractions.Fraction); no rounding to integer RZ.
No Q3, path, build, compile, fit or prediction-model change. The frozen C2
coefficients are applied to the 26 axes for a diagnostic reference value only.

prepare : join fixed Git inputs (A AXIS_MAP, C2 frozen models, saved m5/m6/m9
          predictions, Q3 allocation, PF totals) and write INPUTS.json.
evaluate: requires committed script+INPUTS; evaluates the missing axes once and
          writes AXIS_TABLE.json/csv and RESULT.json (create-only).
"""
import argparse
import csv
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

sys.set_int_max_str_digits(0)
STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
AX=('df5e5c452a3814f7e04ea9cacf05c8f4bdcc047c','docs/research/a_pauli_ld2_expected_price_plan_20261115/AXIS_MAP.json')
FROZEN=('b74e4e536392c2bbe94cd371436615fc56c0564f','docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json')
M9PRED=('3f6e6604a4877dc882fd1ee7538a854464160a5b','docs/research/d_pauli_ld2_m9_longpoint_results_20261115/PREDICTIONS.json')
M9HANDOFF=('cb034e547c444ea67a4f4f90a8534c13f4db4451','docs/research/d_pauli_ld2_m9_longpoint_results_20261115/HANDOFF.md')
M9COMP=('adeae8f831b0cf7ffbf3448ad39b3c0ea259a655','docs/research/d_pauli_ld2_m9_longpoint_results_20261115/COMPARISON.json')
Q3=('50be68b803a90fd893a1499ebce74e033225144d','docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json')
PF=('1002f1dfea34de439ca1b704a0192bca473c5387','docs/research/pf6_allround_cost_20261115/RESULT.json')
SCOPE=('aae9f21799b72a03cff4d75769526c30d966aee1','docs/research/h4_fixed_shot_selection_scope_20261115/REPORT.md')
N=[48,45,42,39,36,33,29,26,23,19,16,13,9]
AXES=('X','Y')
FEATURES=('one','pf_step_count','event_count')
BASELINES={'B0I_P7_S2_16h_over_7':593555352,'FULL_PF_S2_16h_over_7':838693580}
CONFIG='d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'


def ref(pair,selector=None):
    out=dict(commit=pair[0],path=pair[1])
    if selector is not None:out['selector']=selector
    return out


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


@lru_cache(None)
def blob(commit,path):
    return git('show',commit+':'+path)


@lru_cache(None)
def doc(pair):
    return json.loads(blob(*pair))


def frac(v):
    if isinstance(v,dict)and set(v)=={'numerator','denominator'}:return Fraction(v['numerator'],v['denominator'])
    if type(v)is int:return Fraction(v)
    raise ValueError('Exact rational required: %r'%(v,))


def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)


def display(x,digits=6):
    x=Fraction(x);sign='-'if x<0 else'';x=abs(x);scale=10**digits
    q=(x*scale).numerator*2//(x*scale).denominator;q=(q+1)//2  # round half up, display only
    return '%s%d.%0*d'%(sign,q//scale,digits,q%scale)


def features(q,R):
    return {'one':Fraction(1),'pf_step_count':Fraction(q),'event_count':Fraction(q*R)}


def model_value(coefficients,feats):
    return sum(frac(coefficients[f])*feats[f]for f in FEATURES)


def saved_predictions(frozen,m9):
    out={}
    for p,source,i in [(p,FROZEN,i)for i,p in enumerate(frozen['predictions'])]+[(p,M9PRED,i)for i,p in enumerate(m9['predictions'])]:
        key=(p['m'],p['axis'])
        if key in out:raise ValueError('Duplicate saved prediction %r'%(key,))
        out[key]=dict(p=p,locator=ref(source,['predictions',i]))
    return out


def prepare():
    axis_map=doc(AX);frozen=doc(FROZEN);m9=doc(M9PRED);q3=doc(Q3);pf=doc(PF)
    issues=[]
    if axis_map['candidate']['configuration_id']!=CONFIG or axis_map['candidate']['vector_both_axes']!=N:
        raise ValueError('AXIS_MAP candidate/vector differs')
    shots=q3['record']['allocation']['shots']
    if q3['status']!='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' or [s['N_X']for s in shots]!=N or [s['N_Y']for s in shots]!=N:
        raise ValueError('Q3 LD2 allocation/status differs')
    if set(frozen['models'])!={'X','Y'} or frozen['formula']!='C_a(q,R)=beta0_a+beta1_a*q+beta2_a*q*R; a=X,Y':
        raise ValueError('Frozen model set/formula differs')
    for a in AXES:
        model=frozen['models'][a]
        if model['features']!=list(FEATURES) or model['axis']!=a:raise ValueError('Frozen model features/axis differ')
    for p in m9['predictions']:
        if p['model_source']!=ref(FROZEN,['models',p['axis']]):raise ValueError('m9 prediction model source differs')
    pf_rows={r['label']:r['total']for r in pf['rows']}
    for k,v in BASELINES.items():
        if pf_rows.get(k)!=v:raise ValueError('PF baseline differs: '+k)
    saved=saved_predictions(frozen,m9);rows=[]
    keys=[(r['m'],r['axis'])for r in axis_map['rows']]
    if len(keys)!=26 or set(keys)!={(m,a)for m in range(13)for a in AXES}:raise ValueError('AXIS_MAP must have 26 unique axes')
    for i,r in enumerate(axis_map['rows']):
        m,a=r['m'],r['axis'];base=dict(m=m,axis=a,physical_id=r['physical_id'],q=r['q'],R=r['R'],N=N[m],
            axis_map=ref(AX,['rows',i]),model=ref(FROZEN,['models',a]),evidence_class=r['evidence_class'])
        try:
            if r['configuration_id']!=CONFIG or r['N']!=N[m]:raise ValueError('AXIS_MAP row configuration/N differs')
            if type(r['q'])is not int or type(r['R'])is not int or r['q']<1 or r['R']<1:raise ValueError('q/R must be positive integers')
            f=features(r['q'],r['R'])
            s=saved.get((m,a))
            if s:
                p=s['p']
                if p['canonical_physical_id']!=r['physical_id']:raise ValueError('Saved prediction physical ID differs')
                if {k:frac(v)for k,v in p['features'].items()}!=f:raise ValueError('Saved prediction features differ from q,R')
                base.update(value_source='SAVED_FROZEN_PREDICTION',saved_prediction=s['locator'],
                            saved_predicted_RZ=p['predicted_RZ'])
            else:
                base.update(value_source='NEW_DIAGNOSTIC_APPLICATION',saved_prediction=None,saved_predicted_RZ=None)
            base['features']={k:enc(v)for k,v in f.items()}
        except Exception as error:
            issues.append(dict(m=m,axis=a,reason=str(error)));base.update(value_source='UNRESOLVED',reason=str(error))
        rows.append(base)
    comp=doc(M9COMP)
    value=dict(scope='PAULI_LD2_FIXED_SHOT_MODEL_REFERENCE_TOTAL_DIAGNOSTIC',configuration_id=CONFIG,
        allocation_id=q3['record']['allocation']['allocation_id'],N_X=N,N_Y=N,
        sources=dict(axis_map=ref(AX),frozen_models=ref(FROZEN),m9_predictions=ref(M9PRED),m9_handoff=ref(M9HANDOFF),
                     m9_comparison=ref(M9COMP),q3_allocation=ref(Q3,['record','allocation']),pf_totals=ref(PF),scope=ref(SCOPE)),
        model=dict(formula=frozen['formula'],application_scope=frozen['models']['X']['application_scope'],
                   status=frozen['models']['X']['status'],qualified_for_candidate=frozen['qualified_for_candidate'],
                   coefficients={a:frozen['models'][a]['coefficients']for a in AXES},
                   coefficients_X_equal_Y=frozen['models']['X']['coefficients']==frozen['models']['Y']['coefficients']),
        A_fixed_shot_weights_both_axes=axis_map['fixed_shot_weights']['totals_both_axes'],
        m9_independent_comparison_record=dict(rows=[{k:r[k]for k in ('m','axis','P','O','signed_relative_error')}for r in comp['rows']],
            display_signed_relative_percent='+0.274418',use='RECORD_ONLY_NOT_USED_FOR_CORRECTION_TOLERANCE_OR_CI'),
        baselines=BASELINES,axes=rows,issues=issues,
        counts=dict(saved=sum(r['value_source']=='SAVED_FROZEN_PREDICTION'for r in rows),
                    new=sum(r['value_source']=='NEW_DIAGNOSTIC_APPLICATION'for r in rows)),
        E_C=None,CI=None,G=None,winner=None,formal_qualification=None)
    write('INPUTS.json',value)
    print('prepared:',value['counts'],'issues',len(issues),flush=True)
    for i in issues:print(i,flush=True)


def compute(inputs):
    """Return per-axis rows, T_ref and boundaries. Saved values are used as-is."""
    rows=[];issues=list(inputs['issues'])
    coef=inputs['model']['coefficients']
    for r in inputs['axes']:
        if r['value_source']=='UNRESOLVED':continue
        f={k:frac(v)for k,v in r['features'].items()}
        if r['value_source']=='SAVED_FROZEN_PREDICTION':c=frac(r['saved_predicted_RZ'])
        else:c=model_value(coef[r['axis']],f)
        out=dict(r,C_ref=enc(c),C_ref_display=display(c),N_times_C_ref=enc(r['N']*c),N_times_C_ref_display=display(r['N']*c))
        if c<0:issues.append(dict(m=r['m'],axis=r['axis'],reason='NEGATIVE_MODEL_REFERENCE',C_ref=enc(c)))
        rows.append(out)
    if issues or len(rows)!=26:
        return rows,None,issues
    T=sum(frac(r['N_times_C_ref'])for r in rows)
    bounds=[]
    for label,value in inputs['baselines'].items():
        D=Fraction(value)-T
        bounds.append(dict(baseline=label,T_PF=value,D_PF=enc(D),D_PF_display=display(D),
            condition='Delta = T_LD2 - T_ref < D_PF  => LD2 lower cost; = => equal; > => LD2 higher cost'))
    return rows,dict(T_ref=enc(T),T_ref_display=display(T),boundaries=bounds),issues


def write(name,value):
    with (STUDY/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def evaluate():
    head=git('rev-parse','HEAD').decode().strip()
    for f in (Path(__file__),STUDY/'INPUTS.json'):
        if f.read_bytes()!=blob(head,str(f.relative_to(ROOT))):raise ValueError('Script/INPUTS not committed')
    inputs=json.loads((STUDY/'INPUTS.json').read_text())
    rows,result,issues=compute(inputs)
    source=ref((head,str(Path(__file__).relative_to(ROOT))))
    write('AXIS_TABLE.json',dict(source=source,units='COMPILED_RZ_MODEL_REFERENCE_PER_SHOT_AND_WEIGHTED',
        note='Model reference values (frozen C2 coefficients), not E[C]; exact rationals, not rounded',rows=rows))
    with (STUDY/'AXIS_TABLE.csv').open('x',newline='')as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['m','axis','physical_id','q','R','N','value_source','C_ref_exact','C_ref_display','N_times_C_ref_exact','N_times_C_ref_display'])
        for r in rows:
            w.writerow([r['m'],r['axis'],r['physical_id'],r['q'],r['R'],r['N'],r['value_source'],
                        '%d/%d'%(r['C_ref']['numerator'],r['C_ref']['denominator']),r['C_ref_display'],
                        '%d/%d'%(r['N_times_C_ref']['numerator'],r['N_times_C_ref']['denominator']),r['N_times_C_ref_display']])
    if issues:
        write('ISSUES.json',issues);raise ValueError('Unresolved/negative rows: total withheld; see ISSUES.json')
    T=frac(result['T_ref']);coef=inputs['model']['coefficients']['X'];w=inputs['A_fixed_shot_weights_both_axes']
    check=None
    if inputs['model']['coefficients_X_equal_Y']:
        check=w['shots']*frac(coef['one'])+w['steps']*frac(coef['pf_step_count'])+w['events']*frac(coef['event_count'])
        if check!=T:raise ValueError('A both-axes weight cross-check differs')
    write('RESULT.json',dict(source=source,inputs=ref((head,str((STUDY/'INPUTS.json').relative_to(ROOT)))),
        kind='MODEL_REFERENCE_TOTAL_NOT_E_C',units='TOTAL_COMPILED_RZ_AT_FIXED_SHOTS',**result,
        cross_check_A_both_axes_weights=dict(weights=w,equal=check==T if check is not None else None),
        counts=inputs['counts'],m9_independent_comparison_record=inputs['m9_independent_comparison_record'],
        application_scope_unchanged=inputs['model']['application_scope'],qualified_for_candidate=False,
        E_C=None,CI=None,G=None,winner=None))
    print(json.dumps(dict(T_ref=result['T_ref_display'],T_ref_exact=result['T_ref'],
        boundaries=[(b['baseline'],b['D_PF_display'])for b in result['boundaries']],cross_check=check==T),indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','evaluate'))
    (prepare if p.parse_args().mode=='prepare'else evaluate)()
