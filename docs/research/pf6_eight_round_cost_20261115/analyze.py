"""Fixed saved-value PF6 partial costs; no risk, provider, fit or compile calls."""
import argparse
from copy import deepcopy
import csv
from functools import lru_cache
import gzip
import io
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'src'))
B='8dba096dc39788b8948af9eafe394f03d386c520'
BP='docs/research/pf_shortlist_q3_fixed_results_20261115/'
A='0b4b2d941cd669aaa45231e1236bc2d3158a9d12'
AP='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json'
K=[0,1,2,3,4,5,6,9]; U=[7,8,10,11,12]
N=[59,55,51,47,43,39,35,31,27,23,19,15,11]
ENTRIES={
 'm0':('c95353a44b4ce99ed86e949e408d522d6ad7e7fe','docs/research/d_pf_m0_direct_prices_20261115/run01/AXIS_PRICES.json'),
 'm12':('6e110c083ec2db2530cdda3f7b5c84d2a7d3f31c','docs/research/d_pf_m1_m2_direct_prices_20261115/run01/AXIS_PRICES.json'),
 'm3':('3b1301738507d612694bf865d904b9376e038f69','docs/research/pf_m3_saved_price_admission_20261115/ADMISSION.json'),
 'm45':('b190bd815c886446291fac8b281c3c7df7343372','docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json'),
 'm6':('a55813373b84b9128cbd10c6812c0d26143a4280','docs/research/d_pf_stage2_holdout_20261115/run01/RESULT.json'),
 'm9':('0a35101914c6833244b6e0bad7bc2767a49b5a47','docs/research/d_pf_m9_holdout_20261115/run01/RESULT.json')}


def ref(commit,path,selector=None):
    out=dict(commit=commit,path=path)
    if selector is not None:out['selector']=selector
    return out


@lru_cache(None)
def blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


@lru_cache(None)
def document(commit,path):
    data=blob(commit,path)
    if path.endswith('.gz'):data=gzip.decompress(data)
    if path.endswith('.csv'):return list(csv.DictReader(io.StringIO(data.decode())))
    return json.loads(data)


def selected(source):
    value=document(source['commit'],source['path'])
    for key in source.get('selector',[]):value=value[key]
    return value


def only(values):
    values=list(values)
    if len(values)!=1:raise ValueError('Missing or duplicate selected input: '+str(len(values)))
    return values[0]


def indexed(source,predicate):
    values=selected(source);i=only(i for i,v in enumerate(values)if predicate(v))
    return ref(source['commit'],source['path'],source.get('selector',[])+[i])


def policy_at(source,m,axis):
    value=selected(source);found=[]
    def visit(x):
        if isinstance(x,dict):
            if x.get('status')=='IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION':
                if any(r['m']==m and r['axis']==axis for r in x['checked']):found.append(x['policy'])
            else:
                for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(value)
    if not found or any(p!=found[0]for p in found):raise ValueError('Saved guard policy missing or ambiguous')
    return found[0]


def locators():
    """Follow the six fixed HANDOFF indexes; retain every original receipt."""
    rows=[]
    for name in ('m0','m12'):
        sha,path=ENTRIES[name];d=document(sha,path)
        for i,r in enumerate(d['rows']):
            rows.append(dict(configuration_id=r['configuration_id'],m=r['m'],axis=r['axis'],
                index=ref(sha,path,['rows',i]),receipt=r['receipt'],sidecar=r['sidecar'],trial=r['trial'],
                guard=ref(r['receipt']['commit'],str(Path(r['receipt']['path']).with_name('GUARD_CHECK.json'))),
                observed_rz=r['compiled_RZ']))
    sha,path=ENTRIES['m3'];d=document(sha,path)
    for i,r in enumerate(d['rows']):
        if r['status']!='ACCEPTED_SAVED_DIRECT_POINT' or r['missing']:raise ValueError('m3 admission incomplete')
        rows.append(dict(configuration_id=r['configuration_id'],m=3,axis=r['axis'],index=ref(sha,path,['rows',i]),
            receipt=ref(sha,path,['prices',i]),sidecar=ref(sha,path,['sides',i]),
            guard=ref(sha,path,['guard',i]),legacy_origin=r['origin'],observed_rz=r['saved_RZ']))
    sha,path=ENTRIES['m45'];d=document(sha,path)
    for key,g in d['groups'].items():
        for i,e in enumerate(g['entries']):
            axis=g['model_group'][-1]
            if 'price_reference'in e:
                pr=e['price_reference'];side=indexed(g['bound_sidecars'],lambda x:x['axis']==axis and x['identity']==selected(pr)['identity'])
                guard=g['guard']
            else:
                refs=e['references'];pr=indexed(refs['PRICES.json'],lambda x:x['axis']==axis)
                side=indexed(refs['SIDECARS.json'],lambda x:x['axis']==axis);guard=refs['GUARD_CHECK.json']
            price=selected(pr)
            ci=selected(g['C_inputs_source']);v=ci
            for k in e['C_inputs_selector'][1:]:v=v[k]
            if v!=e['record']:raise ValueError('Stage1 record differs from C inputs')
            rows.append(dict(configuration_id=price['configuration_id'],m=e['m'],axis=axis,
                index=ref(sha,path,['groups',key,'entries',i]),receipt=pr,sidecar=side,guard=guard,
                observed_rz=e['record']['rz_count'],calibration_input=g['C_inputs_source']))
    for name,m in [('m6',6),('m9',9)]:
        sha,path=ENTRIES[name];d=document(sha,path)
        for i,r in enumerate(d['rows']):
            pr=r['receipt'];price=selected(pr);directory=Path(pr['path']).parent
            side=indexed(ref(pr['commit'],str(directory/'SIDECARS.json')),
                         lambda x:x['axis']==r['axis'] and x['identity']==price['identity'])
            rows.append(dict(configuration_id=r['configuration_id'],m=m,axis=r['axis'],index=ref(sha,path,['rows',i]),
                receipt=pr,sidecar=side,guard=ref(pr['commit'],str(directory/'GUARD_CHECK.json')),
                observed_rz=r['observed_RZ']))
    return rows


def integer(value):
    if isinstance(value,dict)and set(value)=={'numerator','denominator'}:
        if value['denominator']!=1:raise ValueError('Integer price required')
        value=value['numerator']
    if type(value)is not int or value<0:raise ValueError('Nonnegative integer required')
    return value


def origin_price(price,axis):
    """Check public original axis evidence, without new compilation/reanalysis."""
    source=price['source'];name='cosine'if axis=='X'else'sine'
    if 'csv_selector'in source:
        row=only(v for v in document(source['commit'],source['path'])
                 if all(v[k]==w for k,w in source['csv_selector'].items()))
        return int(row['rz_count'])
    if 'key'in source:
        row=document(source['commit'],source['path'])[source['key']]
        return integer(row['unit_statistics'][name]['mean_exact'])
    if 'path'in source and 'selector'in source:
        value=selected(source);return integer(value.get('metrics',value)['rz_count'])
    return None  # New D records are checked against their explicit trial locator below.


def prepare():
    from trottertracks import rpe_price_implementation_guard as G
    b=document(B,BP+'INPUTS.json');joins=document(B,BP+'C_RESULT_JOIN.json')['candidates'];alignment=document(A,AP)
    descriptors=locators();axes=[];issues=[];candidates=[];expected=set()
    for case in b['cases']:
        join=only(j for j in joins if j['label']==case['label']);config=selected(case['configuration'])
        if case['configuration_id']!=config['id'] or join['original_configuration_id']!=config['id']:raise ValueError('B config differs')
        if (join['total_configuration_id']!=case['total_view_id'] or join['canonical_binding']!=case['binding_id']
                or join['allocation']!=case['allocation']):raise ValueError('B total binding/allocation differs')
        if [r['N_X']for r in case['allocation']['shots']]!=N or [r['N_Y']for r in case['allocation']['shots']]!=N:
            raise ValueError('Fixed allocation differs')
        candidates.append(dict(label=case['label'],configuration=case['configuration'],original_configuration_id=config['id'],
            total_configuration_id=join['total_configuration_id'],canonical_binding=join['canonical_binding'],
            allocation=case['allocation'],q3_result=ref(B,BP+join['result_path']),evidence_status=join['evidence_status'],
            original_API_return_available=False,CPU_peak_RAM_available=False,
            q3_scientific_source=document(B,BP+'run01/EXECUTION.json')['source']))
        for m in range(13):
            for axis in ('X','Y'):
                bridge=only(r for r in join['rows']if r['m']==m and r['axis']==axis)
                if bridge['N']!=N[m]:raise ValueError('B axis shot count differs')
                ai=only(i for i,r in enumerate(alignment['rows'])if r['configuration_id']==config['id']and r['m']==m and r['axis']==axis)
                ar=alignment['rows'][ai]
                if bridge['physical_id']!=ar['physical_id']or bridge['original_candidate_id']!=ar['original_canonical_id']:
                    raise ValueError('B/alignment physical or canonical ID differs')
                common=dict(label=case['label'],original_configuration_id=config['id'],total_configuration_id=join['total_configuration_id'],
                    physical_id=bridge['physical_id'],original_candidate_id=bridge['original_candidate_id'],
                    total_candidate_id=bridge['total_candidate_id'],m=m,axis=axis,N=bridge['N'],
                    allocation_id=case['allocation']['allocation_id'],B_join=ref(B,BP+'C_RESULT_JOIN.json',['candidates',joins.index(join),'rows',join['rows'].index(bridge)]),
                    alignment=ref(A,AP,['rows',ai]))
                if m in U:
                    axes.append(dict(common,status='UNOBSERVED_ROUND',C=None));continue
                key=(config['id'],m,axis);expected.add(key)
                try:
                    loc=only(r for r in descriptors if(r['configuration_id'],r['m'],r['axis'])==key)
                    price=selected(loc['receipt']);side=selected(loc['sidecar']);row=config['rounds'][m]
                    if price['identity']!=row['identity']or price['law_binding']!=row['law_binding']:raise ValueError('Price identity/law differs')
                    if price['physical_context']['physical_candidate']!=ar['physical_candidate']:raise ValueError('Physical descriptor differs')
                    if price['classification']!='DIRECT_COMPILED_DETERMINISTIC'or price['expectation']!='DETERMINISTIC':raise ValueError('Direct deterministic receipt required')
                    count=integer(price['mean']['rz_count'])
                    if count!=integer(loc['observed_rz']):raise ValueError('Receipt/index price differs')
                    original=origin_price(price,axis)
                    if original is not None and original!=count:raise ValueError('Original axis price differs')
                    if 'trial'in loc:
                        trial=selected(loc['trial']);name='cosine'if axis=='X'else'sine'
                        if trial['candidate']!=ar['physical_candidate']or trial['prices'][name]['rz_count']!=count:raise ValueError('D trial differs')
                    policy=policy_at(loc['guard'],m,axis)
                    checked=G.check(ROOT,config,[price],cohorts={(row['candidate']['id'],axis):price['cohort']},bindings=[side],policy=policy)
                    if len(checked['checked'])!=1:raise ValueError('Guard did not check the selected axis')
                    axes.append(dict(common,status='ACCEPTED_DIRECT',C=count,input_locators=loc,
                        price_generation_configuration_id=price['configuration_id'],price_generation_candidate_id=price['identity']['candidate']['id'],
                        price_generation_source=price['source'],law_binding=price['law_binding'],cohort=price['cohort'],
                        price_profile=price.get('price_profile'),physical_context=price['physical_context'],
                        implementation_evidence=side['implementation_evidence'],implementation_policy=policy,
                        guard_status=checked['status'],model_qualified=price['diagnostic']['model_qualified']))
                except Exception as error:
                    issues.append(dict(common,reason=str(error)));axes.append(dict(common,status='UNRESOLVED_INPUT',C=None,reason=str(error)))
    unexpected=[r for r in descriptors if(r['configuration_id'],r['m'],r['axis'])not in expected]
    if unexpected:raise ValueError('Unexpected source rows')
    value=dict(scope='PF6_FIXED_SHOTS_KNOWN_EIGHT_ROUND_PARTIAL_COST_ONLY',B_source=ref(B,BP+'INPUTS.json'),
        C_alignment=ref(A,AP),price_entries={k:ref(*v)for k,v in ENTRIES.items()},guard_source=G.GUARD,
        K=K,U=U,N_X=N,N_Y=N,candidates=candidates,axes=axes,issues=issues,
        original_Q3_partial_save_preserved=True,sufficient_N=None,G=None,G_qualified=None,winner=None)
    write('INPUTS.json',value)
    print('Prepared:',len([r for r in axes if r['status']=='ACCEPTED_DIRECT']),'direct axes;',len(issues),'issues',flush=True)
    for item in issues:print(item['label'],item['m'],item['axis'],item['reason'],flush=True)


def write(name,value):
    with(STUDY/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def aggregate(rows,labels):
    """Integer-only finite subset arithmetic; no price fallback or N qualification."""
    terms=[];partials=[]
    for label in labels:
        selected_rows=[r for r in rows if r['label']==label and r['m']in K]
        keys=[(r['m'],r['axis'])for r in selected_rows]
        if len(keys)!=16 or len(set(keys))!=16 or set(keys)!={(m,a)for m in K for a in ('X','Y')}:
            raise ValueError('Missing/duplicate known axes: '+label)
        if any(r['status']!='ACCEPTED_DIRECT'or r['C']is None for r in selected_rows):raise ValueError('Unresolved known axes: '+label)
        local=[]
        for r in selected_rows:
            t=deepcopy(r);t['N_times_C']=integer(r['N'])*integer(r['C']);local.append(t)
        rounds=[dict(m=m,X=only(t['N_times_C']for t in local if t['m']==m and t['axis']=='X'),
                     Y=only(t['N_times_C']for t in local if t['m']==m and t['axis']=='Y'),
                     total=sum(t['N_times_C']for t in local if t['m']==m))for m in K]
        partials.append(dict(label=label,rounds=rounds,K_i=sum(t['N_times_C']for t in local),U_i=None,
            full_13_round_cost=None,sufficient_N=None,G_qualified=None));terms.extend(local)
    pairs=[]
    for left,right in combinations(partials,2):
        # i is the smaller known partial cost; labels break a possible tie only for display.
        i,j=sorted((left,right),key=lambda r:(r['K_i'],r['label']));gap=j['K_i']-i['K_i']
        pairs.append(dict(i=i['label'],j=j['label'],K_i=i['K_i'],K_j=j['K_i'],K_j_minus_K_i=gap,
            unknown_difference='U_i - U_j',i_cheaper_condition=dict(operator='<',threshold=gap),
            equal_condition=dict(operator='=',threshold=gap),i_more_expensive_condition=dict(operator='>',threshold=gap),
            unknown_terms=[dict(m=m,axis=a,weight=N[m],term='weight*(C_i-C_j)')for m in U for a in ('X','Y')],
            full_13_round_order='UNKNOWN'))
    return terms,partials,pairs


def analyze():
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for f in (Path(__file__),STUDY/'INPUTS.json'):
        if f.read_bytes()!=blob(source,str(f.relative_to(ROOT))):raise ValueError('Analysis source/input not committed')
    inputs=json.loads((STUDY/'INPUTS.json').read_text())
    if inputs['issues']:write('ISSUES.json',inputs['issues']);raise ValueError('Unresolved inputs; no arithmetic performed')
    terms,partials,pairs=aggregate(inputs['axes'],[c['label']for c in inputs['candidates']])
    source=ref(source,str(Path(__file__).relative_to(ROOT)))
    write('AXIS_PRODUCTS.json',dict(source=source,units='TOTAL_COMPILED_RZ_AT_FIXED_SHOTS',rows=terms))
    write('PARTIAL_SUMS.json',dict(source=source,known_rounds=K,unknown_rounds=U,rows=partials,
        known_partial_order=[r['label']for r in sorted(partials,key=lambda r:(r['K_i'],r['label']))],
        formal_full_13_round_winner=None,G=None,G_qualified=None,sufficient_N=None))
    write('PAIRWISE_CONDITIONS.json',dict(source=source,rows=pairs,formal_full_13_round_winner=None))
    for name,rows,fields in [('AXIS_PRODUCTS.csv',terms,['label','physical_id','m','axis','N','C','N_times_C']),
        ('PARTIAL_SUMS.csv',partials,['label','K_i']),('PAIRWISE_CONDITIONS.csv',pairs,['i','j','K_i','K_j','K_j_minus_K_i'])]:
        with(STUDY/name).open('x',newline='')as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(partial_sums=[(r['label'],r['K_i'])for r in partials],axis_products=len(terms),pairs=len(pairs)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','analyze'))
    (prepare if p.parse_args().mode=='prepare'else analyze)()
