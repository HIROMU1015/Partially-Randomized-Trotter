"""PF6 all-13-round fixed-shot compiled-RZ totals: B known K + C conditional closure U.

Integer arithmetic only. No Q3, shot search, provider, fit, build or compile.
`prepare` joins fixed Git inputs and writes INPUTS.json; `analyze` requires the
committed script and INPUTS and writes the result tables (create-only).
"""
import argparse
import csv
from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
B8='540b54ab2aa01a164f913d5b18497e711ae4a29c'
B8P='docs/research/pf6_eight_round_cost_20261115/'
Q3='8dba096dc39788b8948af9eafe394f03d386c520'
Q3P='docs/research/pf_shortlist_q3_fixed_results_20261115/'
REC_FULL='c3f761f0af5b21207e64051c9c619858064450fe'
RECP='docs/research/pf_closure_price_recommendation_20261115/RECOMMENDATION.json'
N=[59,55,51,47,43,39,35,31,27,23,19,15,11]
K_ROUNDS=[0,1,2,3,4,5,6,9]; U_ROUNDS=[7,8,10,11,12]
AXES=('X','Y')
# Instruction cross-check of B's fixed known subtotals (comment(B1114), step 2).
EXPECTED_K={'FULL_PF_S2_16h_over_7':115251570,'FULL_PF_Y4_16h_over_3':153106880,'FULL_PF_M5_16h':207232138,
            'B0I_P7_S2_16h_over_7':81621086,'B0I_P7_Y4_16h_over_3':107719628,'B0I_P7_M5_16h':146850902}
LABEL_KIND={'FULL_PF':'FULL_PF','B0I_P7':'TRUNCATED_PF'}
LABEL_FORMULA={'S2':('2nd','16/7'),'Y4':('4th','16/3'),'M5':('4th(m5_best)','16')}
DIRECT='DIRECT_COMPILED_DETERMINISTIC'
CLOSURE='CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION'
PRICE_FIELDS=('price','rz_count','derived_price','derived_rz_count','C')


def ref(commit,path,selector=None):
    out=dict(commit=commit,path=path)
    if selector is not None:out['selector']=selector
    return out


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


@lru_cache(None)
def blob(commit,path):
    return git('show',commit+':'+path)


@lru_cache(None)
def document(commit,path):
    data=blob(commit,path)
    if path.endswith('.csv'):return list(csv.DictReader(io.StringIO(data.decode())))
    return json.loads(data)


def selected(source):
    value=document(source['commit'],source['path'])
    for key in source.get('selector',[]):value=value[key]
    return value


def full_sha(commit):
    return git('rev-parse',commit+'^{commit}').decode().strip()


def only(values,what):
    values=list(values)
    if len(values)!=1:raise ValueError('%s: expected one, found %d'%(what,len(values)))
    return values[0]


def integer(value):
    if isinstance(value,dict):
        if set(value)=={'numerator','denominator'}:
            if value['denominator']!=1:raise ValueError('Integer price required')
            value=value['numerator']
        elif 'rz_count'in value:return integer(value['rz_count'])
    if isinstance(value,str)and value.strip().isdigit():value=int(value)
    if type(value)is not int or value<0:raise ValueError('Nonnegative integer required: %r'%(value,))
    return value


def label_parts(label):
    route,formula=only([(k,v)for k,v in LABEL_KIND.items()if label.startswith(k+'_')],'label route '+label)
    key=label[len(route)+1:].split('_')[0]
    return formula,LABEL_FORMULA[key]


def c_rows(doc):
    """C table rows: a list, or the unique list under rows/prices/axes."""
    if isinstance(doc,list):return doc
    lists=[k for k in ('rows','prices','axes','unobserved_axes')if isinstance(doc.get(k),list)]
    return doc[only(lists,'C table row list')]


def c_price(row):
    fields=[f for f in PRICE_FIELDS if row.get(f)is not None]
    values={integer(row[f])for f in fields}
    return only(values,'C price field')


def join(b_inputs,b_sums,c_table_source,c_doc,rec_doc):
    """Join B's 156 axes (96 direct + 60 null) with C's 60 conditional prices."""
    issues=[];rows=[];candidates=[c['label']for c in b_inputs['candidates']]
    if sorted(candidates)!=sorted(EXPECTED_K):raise ValueError('Candidate labels differ')
    sums={r['label']:r for r in b_sums['rows']}
    for label in candidates:
        if sums[label]['K_i']!=EXPECTED_K[label]:raise ValueError('B K differs from instruction: '+label)
    crow=c_rows(c_doc);rec=rec_doc['rows']
    used=set()
    for a in b_inputs['axes']:
        if a['N']!=N[a['m']]:raise ValueError('Fixed allocation differs')
        base={k:a[k]for k in ('label','original_configuration_id','total_configuration_id','physical_id',
                               'original_candidate_id','total_candidate_id','m','axis','N','allocation_id')}
        if a['m']in K_ROUNDS:
            if a['status']!='ACCEPTED_DIRECT'or a['C']is None:
                issues.append(dict(base,reason='B known axis not accepted'));continue
            rows.append(dict(base,price_class=DIRECT,C=integer(a['C']),
                source=dict(B_eight_round_inputs=ref(B8,B8P+'INPUTS.json',['axes',b_inputs['axes'].index(a)]),
                            receipt=a['input_locators']['receipt'])))
            continue
        if a['status']!='UNOBSERVED_ROUND'or a['C']is not None:
            issues.append(dict(base,reason='B unknown axis is not an unobserved null row'));continue
        key=(a['original_configuration_id'],a['physical_id'],a['m'],a['axis'])
        hits=[i for i,r in enumerate(crow)if (r.get('configuration_id'),r.get('physical_id'),r.get('m'),r.get('axis'))==key]
        if len(hits)!=1:
            issues.append(dict(base,reason='C price row missing or duplicate (%d)'%len(hits)));continue
        i=hits[0];r=crow[i];used.add(i)
        try:
            (route,(formula,width))=label_parts(a['label'])
            recrow=only([x for x in rec if (x['configuration_id'],x['physical_id'],x['m'],x['axis'])==key],'recommendation row')
            for name,want in (('route',route),('formula',formula),('width_over_h',width)):
                if recrow[name]!=want:raise ValueError('Recommendation %s differs'%name)
                if r.get(name,want)!=want:raise ValueError('C table %s differs'%name)
            for name in ('q','T','model_group'):
                if name in r and r[name]!=recrow[name]:raise ValueError('C table %s differs from recommendation'%name)
            if r.get('direct_observation')is True:raise ValueError('Closure row marked as direct observation')
            if r.get('classification')==DIRECT:raise ValueError('Closure row registered as direct deterministic price')
            price=c_price(r)
        except Exception as error:
            issues.append(dict(base,reason=str(error)));continue
        rows.append(dict(base,price_class=CLOSURE,C=price,
            source=dict(C_price_table=dict(c_table_source,selector=c_table_source.get('selector',[])+[i]),
                        C_recommendation=ref(REC_FULL,RECP,['rows',rec.index(recrow)]),
                        model_group=recrow['model_group'],q=recrow['q'],
                        C_classification=r.get('classification'),C_evidence_kind=r.get('evidence_kind'),
                        C_policy=r.get('policy',r.get('adoption_policy')))))
    extra=[i for i in range(len(crow))if i not in used]
    for i in extra:issues.append(dict(reason='C price row not matched to a B unobserved axis',C_row_index=i))
    return rows,issues


def totals(rows,labels,b_sums):
    """K from B's fixed subtotal (cross-checked against the 96 direct terms), U from C."""
    out=[];sums={r['label']:r['K_i']for r in b_sums['rows']}
    for label in labels:
        mine=[r for r in rows if r['label']==label]
        keys=[(r['m'],r['axis'])for r in mine]
        if len(keys)!=26 or len(set(keys))!=26:raise ValueError('Axis count/duplicate: '+label)
        direct=[r for r in mine if r['price_class']==DIRECT];closure=[r for r in mine if r['price_class']==CLOSURE]
        if ({(r['m'],r['axis'])for r in direct}!={(m,a)for m in K_ROUNDS for a in AXES}
                or {(r['m'],r['axis'])for r in closure}!={(m,a)for m in U_ROUNDS for a in AXES}):
            raise ValueError('Direct/closure axis split differs: '+label)
        k_terms=sum(integer(r['N'])*integer(r['C'])for r in direct)
        if k_terms!=sums[label]:raise ValueError('Recomputed K differs from B subtotal: '+label)
        u_rounds=[dict(m=m,X=only([integer(r['N'])*integer(r['C'])for r in closure if r['m']==m and r['axis']=='X'],'X'),
                       Y=only([integer(r['N'])*integer(r['C'])for r in closure if r['m']==m and r['axis']=='Y'],'Y'))
                  for m in U_ROUNDS]
        for u in u_rounds:u['total']=u['X']+u['Y']
        U=sum(u['total']for u in u_rounds)
        out.append(dict(label=label,K=sums[label],U=U,total=sums[label]+U,U_rounds=u_rounds,
                        direct_axes=len(direct),closure_axes=len(closure)))
    order=sorted(out,key=lambda r:(r['total'],r['label']))
    for r in out:r['rank']=1+sum(o['total']<r['total']for o in out)
    k_order=sorted(out,key=lambda r:(r['K'],r['label']))
    for r in out:r['known_rank']=1+sum(o['K']<r['K']for o in out)
    flips=[]
    for i,a in enumerate(k_order):
        for b in k_order[i+1:]:
            dk=b['K']-a['K'];dt=b['total']-a['total']
            if (dk>0 and dt<=0)or(dk==0 and dt!=0):
                flips.append(dict(known_cheaper=a['label'],known_more_expensive=b['label'],K_gap=dk,
                    U_difference=a['U']-b['U'],total_gap=dt,result='REVERSED'if dt<0 else('TIED'if dt==0 else 'ORDER_FROM_TIE')))
    best=order[0];second=order[1]
    return out,order,flips,dict(minimum=best['label'],next=second['label'],gap=second['total']-best['total'])


def prepare(c_table):
    commit,path,*sel=c_table.split(':')
    commit=full_sha(commit);source=ref(commit,path,[json.loads(s)if s.lstrip('-').isdigit()else s for s in sel]or None)
    b_inputs=document(B8,B8P+'INPUTS.json');b_sums=document(B8,B8P+'PARTIAL_SUMS.json');q3=document(Q3,Q3P+'INPUTS.json')
    for case in q3['cases']:
        if [r['N_X']for r in case['allocation']['shots']]!=N or [r['N_Y']for r in case['allocation']['shots']]!=N:
            raise ValueError('Q3 allocation differs')
    c_doc=selected(source);rec=document(REC_FULL,RECP)
    rows,issues=join(b_inputs,b_sums,source,c_doc,rec)
    value=dict(scope='PF6_FIXED_SHOT_ALL_13_ROUND_TOTAL_COMPILED_RZ_CONDITIONAL_CLOSURE',
        B_eight_round=dict(inputs=ref(B8,B8P+'INPUTS.json'),partial_sums=ref(B8,B8P+'PARTIAL_SUMS.json'),
                           analysis_source=b_sums['source']),
        Q3_allocation=ref(Q3,Q3P+'INPUTS.json'),C_recommendation=ref(REC_FULL,RECP),C_price_table=source,
        N_X=N,N_Y=N,known_rounds=K_ROUNDS,closure_rounds=U_ROUNDS,candidates=[c['label']for c in b_inputs['candidates']],
        axes=rows,issues=issues,direct_axes=sum(r['price_class']==DIRECT for r in rows),
        closure_axes=sum(r['price_class']==CLOSURE for r in rows),
        Q3_partial_save_preserved=dict(original_API_return_available=False,CPU_peak_RAM_available=False),
        sufficient_N=None,optimal_N=None,global_RMSE=None,G=None,G_qualified=None,winner=None)
    write('INPUTS.json',value)
    print('prepared: %d direct, %d closure, %d issues'%(value['direct_axes'],value['closure_axes'],len(issues)),flush=True)
    for i in issues:print(i,flush=True)


def write(name,value):
    with (STUDY/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def analyze():
    head=git('rev-parse','HEAD').decode().strip()
    for f in (Path(__file__),STUDY/'INPUTS.json'):
        if f.read_bytes()!=blob(head,str(f.relative_to(ROOT))):raise ValueError('Script/INPUTS not committed')
    inputs=json.loads((STUDY/'INPUTS.json').read_text())
    if inputs['issues']:
        write('ISSUES.json',inputs['issues'])
        raise ValueError('Unresolved inputs: totals withheld for affected candidates; see ISSUES.json')
    if (len(inputs['axes']),inputs['direct_axes'],inputs['closure_axes'])!=(156,96,60):raise ValueError('Axis counts differ')
    b_sums=document(B8,B8P+'PARTIAL_SUMS.json')
    out,order,flips,gap=totals(inputs['axes'],inputs['candidates'],b_sums)
    source=ref(head,str(Path(__file__).relative_to(ROOT)))
    write('AXIS_TERMS.json',dict(source=source,units='COMPILED_RZ_TIMES_SHOTS',
        rows=[dict(r,N_times_C=integer(r['N'])*integer(r['C']))for r in inputs['axes']]))
    write('RESULT.json',dict(source=source,inputs=ref(head,str((STUDY/'INPUTS.json').relative_to(ROOT))),
        units='TOTAL_COMPILED_RZ_AT_FIXED_SHOTS',rows=[{k:r[k]for k in ('label','K','U','total','rank','known_rank',
        'direct_axes','closure_axes','U_rounds')}for r in order],minimum_and_next=gap,known_to_total_order_changes=flips,
        scope='CONDITIONAL_ON_ADOPTED_CLOSURE_AND_FIXED_Q3_ALLOCATION',two_percent_guarantee=None,
        sufficient_N=None,optimal_N=None,global_RMSE=None,G=None,G_qualified=None,five_method_winner=None))
    with (STUDY/'RESULT.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['label','K','U','total','rank','known_rank'],extrasaction='ignore',lineterminator='\n')
        w.writeheader();w.writerows(order)
    print(json.dumps(dict(rows=[(r['label'],r['K'],r['U'],r['total'],r['rank'])for r in order],gap=gap,flips=flips),indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','analyze'))
    p.add_argument('--c-table',help='COMMIT:PATH[:selector...] of C\'s machine-readable 60-price table')
    args=p.parse_args()
    if args.mode=='prepare':
        if not args.c_table:raise SystemExit('--c-table is required for prepare')
        prepare(args.c_table)
    else:analyze()
