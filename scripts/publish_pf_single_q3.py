"""Publish one normal PF optimizer Q3 job through C's existing generic PF join.

Reads committed REQUEST/CASE_INPUTS/OUTCOME/RESULT/API_RETURN and the fixed D
registry. Never launches a worker, changes raw artifacts, chooses shots or fits
prices. First commit the price-ledger view, then its input view; the final call
emits the source-bound SUMMARY/C_RESULT_JOIN. This only avoids future/self-
referential Git locators; it is not an approval procedure.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_optimizer_connection as PF

D_SOURCE='352539e2749989b4ddc3ab74c31f486982bfe265'
REGISTRY=dict(commit=D_SOURCE,path='docs/research/d_pf_on_demand_evidence_20261116/C_RESULT_JOIN.json',selector=[])
PASS='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'


def require(ok,message):
    if not ok:raise ValueError(message)


def frac(v):
    return Fraction(int(v['numerator']),int(v['denominator'])) if isinstance(v,dict) else Fraction(v)


def enc(v):
    v=frac(v);return dict(numerator=v.numerator,denominator=v.denominator)


def ref(commit,path,*selector):return dict(commit=commit,path=str(path),selector=list(selector))


def _read(root,source):
    value=json.loads(subprocess.check_output(['git','show',source['commit']+':'+source['path']],cwd=root))
    for key in source.get('selector',[]):value=value[key]
    return value


def _files(root,commit,path):
    return subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',str(path)],cwd=root,text=True).splitlines()


def _origin(root,commit,path):
    fixed=subprocess.check_output(['git','log','-1','--format=%H',commit,'--',str(path)],cwd=root,text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}',fixed) is not None,'No fixed original file SOURCE')
    return ref(fixed,path)


def _write(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()==text:return
    require(not path.exists(),'Preserve an existing different publication: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


def _shots(allocation):
    rows=allocation['shots'] if isinstance(allocation,dict) else allocation
    require(len(rows)==13 and {r['m'] for r in rows}==set(range(13)),'13 unique actual allocation rounds required')
    rows=sorted(rows,key=lambda r:r['m'])
    require(all(type(r['N_'+a]) is int and r['N_'+a]>=0 for r in rows for a in ('X','Y')),'Actual shots must be nonnegative integers')
    return rows


def publication_views(root,study,publication_source):
    """Build metadata from exactly one committed normal Full-PF/PF job."""
    root=Path(root);study=Path(study)
    if study.is_absolute():study=study.relative_to(root)
    require('..' not in study.parts and re.fullmatch('[0-9a-f]{40}',publication_source) is not None,'Relative study and full fixed publication SHA required')
    outcomes=[p for p in _files(root,publication_source,study/'run01')
              if '/jobs/' in p and p.endswith('/OUTCOME.json')]
    require(len(outcomes)==1,'Expected exactly one completed normal job, no job selection by price/verdict')
    job=Path(outcomes[0]).parent
    origins={name:_origin(root,publication_source,job/(name+'.json')) for name in ('REQUEST','CASE_INPUTS','OUTCOME')}
    request,inputs,outcome=(_read(root,origins[k]) for k in ('REQUEST','CASE_INPUTS','OUTCOME'))
    require(request['method'] in ('FULL_PF','TRUNCATED_PF') and request['route']['kind']=='pf_b_pf6','Only the existing deterministic PF worker route is accepted')
    require(outcome['route']=='pf_b_pf6' and outcome['request_id']==request['request_id'],'Outcome is not this actual request')
    require(len(inputs['cases'])==1,'One actual worker case required')
    case=deepcopy(inputs['cases'][0]);cid=request['configuration_id'];label=case['label']
    require(label==request['request_id'],'Actual case/request label mismatch')
    require((case.get('configuration_id') or case.get('original_configuration_id'))==cid,'Actual case configuration differs')
    actual=_shots(case['allocation']);requested=[dict(m=m,N_X=x,N_Y=y)for m,(x,y)in enumerate(zip(request['N_X'],request['N_Y']))]
    require(len(request['N_X'])==len(request['N_Y'])==13 and actual==requested,'No manual or substituted allocation is accepted')
    entry=_read(root,REGISTRY)['configurations'][cid]
    require(case['configuration']==entry['configuration'] and case['signal_packet']==entry['signal_packet'],'Worker signal/configuration SOURCE differs from the fixed registry')
    conf=_read(root,entry['configuration']);price_table=_read(root,entry['prices'])
    require(conf['id']==cid,'Canonical configuration mismatch')
    identity=outcome.get('identity') or {};status=outcome['status']
    require(status in (PASS,FAIL,'INCONCLUSIVE','UNKNOWN'),'Unsupported normal worker verdict')
    lower=outcome.get('MSE_lower');upper=outcome.get('MSE_upper')
    if status!='UNKNOWN':
        require(identity.get('configuration_id')==cid and identity.get('contract_matches') is True,'Normal worker identity/contract gate failed')
        require(identity['allocation']==actual and identity['signal_packet']==entry['signal_packet'],'Normal worker allocation/signal gate failed')
        require(identity['total_configuration_id']==entry['total_configuration_id'] and identity['binding_id']==entry['binding_id'],'Normal worker canonical total binding differs')
        origins['RESULT']=_origin(root,publication_source,job/label/'RESULT.json')
        origins['API_RETURN']=_origin(root,publication_source,job/label/'API_RETURN.json')
        result=_read(root,origins['RESULT']);api=_read(root,origins['API_RETURN'])
        require(isinstance(api,list) and len(api)==1 and api[0]==result['record'],'Original API receipt/result mismatch')
        require(result['status']==status==result['adopted_q3_assessment']['status'],'Saved worker/result classification differs')
        record=api[0];risk=record['diagnostic']['deterministic_risk']
        require(frac(risk['MSE_lower'])==frac(lower) and frac(risk['MSE_upper'])==frac(upper),'Saved worker/outcome exact MSE differs')
        require(record['configuration_id']==entry['total_configuration_id'] and record['canonical_binding']==entry['binding_id'],'Original API canonical binding differs')
        require(_shots(record['allocation'])==actual and record['evaluation_target']==case['evaluation_target'],'Original API allocation/target differs')
        require(outcome['scientific_evaluation_calls']==result['scientific_evaluation_calls']==1,'Exactly one original scientific call required')
    physical=PF.physical_ids(root,conf);axes=price_table['axes']
    require(len(axes)==26 and {(r['m'],r['axis'])for r in axes}=={(m,a)for m in range(13)for a in ('X','Y')},'26 unique saved price axes required')
    costs=[]
    for i,p in enumerate(axes):
        m=p['m'];axis=p['axis'];n=actual[m]['N_'+axis]
        require(p['physical_id']==physical[m],'Price/native physical ID differs')
        require(p['kind'] in ('DIRECT_COMPILED_DETERMINISTIC','CONDITIONAL_AFFINE_PRICE_REFERENCE'),'Unsupported saved PF price kind')
        cost=frac(p['unit_price_RZ'])*n
        costs.append(dict(m=m,axis=axis,N=n,q=conf['rounds'][m]['q_PF'],physical_id=p['physical_id'],
            original_configuration_id=cid,original_candidate_id=conf['rounds'][m]['candidate']['id'],
            price_class=p['kind'],unit_price_RZ=deepcopy(p['unit_price_RZ']),C=deepcopy(p['unit_price_RZ']),
            N_times_C=enc(cost),price_source=dict(entry['prices'],selector=list(entry['prices'].get('selector',[]))+['axes',i]),
            price_evidence=deepcopy(p['price_evidence']),compiler_profile=deepcopy(conf['spec']['profile'])))
    total=sum((frac(p['N_times_C']) for p in costs),Fraction(0))
    require(total==frac(request['cost']),'Normal request quote does not match the same 26 saved prices/allocation')
    cost_view=dict(schema='D_SINGLE_PF_Q3_MODEL_QUOTE_V1',configuration_id=cid,allocation=case['allocation'],axes=costs,
        conditional_model_quote_RZ=enc(total),achieved_conditional_cost_RZ=enc(total) if status==PASS else None,
        price_scope=entry.get('price_scope'),model_error_bound_RZ=None,CI=None,
        qualification='CONDITIONAL_MODEL_QUOTE_NOT_FORMAL_G; FAILED_OR_UNKNOWN_Q3_IS_NOT_AN_ACHIEVED_COST',
        original_request_source=origins['REQUEST'],original_outcome_source=origins['OUTCOME'])
    # These are read-only publication views of the original worker inputs; the
    # worker CASE_INPUTS.json and raw API bytes remain untouched.
    priced_case=deepcopy(case);priced_case.update(base_label=cid,total_configuration_id=entry['total_configuration_id'],
        original_configuration_id=cid,conditional_cost=enc(total))
    input_view=deepcopy(inputs)
    input_view.update(cases=[priced_case],original_worker_inputs=origins['CASE_INPUTS'],original_job_sources=origins,
        publication_only=True,new_scientific_calls=0)
    return dict(study=str(study),method=request['method'],case=priced_case,inputs=input_view,cost=cost_view,outcome=outcome,origins=origins,
        rows=[dict(m=m,physical_candidate=dict(id=physical[m]))for m in range(13)])


def publish(root,study,publication_source):
    """Write views first; emit generic PF joins once their exact bytes are committed."""
    root=Path(root);view=publication_views(root,study,publication_source);directory=root/view['study']
    input_path=Path(view['study'])/'C_INPUTS.json';cost_path=Path(view['study'])/'PRICED_AXES.json'
    # Two small metadata commits keep all price and input locators fixed:
    # price ledger first, then its input view. Raw worker files never change.
    try:
        saved_cost=_read(root,ref(publication_source,cost_path))
    except (subprocess.CalledProcessError,FileNotFoundError,KeyError):
        saved_cost=None
    if saved_cost is None:
        _write(root/cost_path,view['cost'])
        return dict(status='READY_FOR_METADATA_COMMIT',phase='PRICE_LEDGER',files=[str(cost_path)],scientific_calls=0)
    require(saved_cost==view['cost'],'Committed price publication view differs from the same original job')
    cost_source=_origin(root,publication_source,cost_path)
    input_view=deepcopy(view['inputs'])
    terms=[dict(m=r['m'],axis=r['axis'],N=r['N'],price_source=dict(cost_source,selector=['axes',i]))for i,r in enumerate(saved_cost['axes'])]
    input_view['cases'][0]['price_terms']=terms
    try:saved_input=_read(root,ref(publication_source,input_path))
    except (subprocess.CalledProcessError,FileNotFoundError,KeyError):saved_input=None
    if saved_input is None:
        _write(root/input_path,input_view)
        return dict(status='READY_FOR_METADATA_COMMIT',phase='PRICED_INPUT_VIEW',files=[str(input_path)],scientific_calls=0)
    require(saved_input==input_view,'Committed input publication view differs from the same original job')
    execution=_origin(root,publication_source,input_path)
    outcome=view['outcome'];case=input_view['cases'][0];status=outcome['status'];label=case['label']
    summary_row=dict(label=label,base_label=case['base_label'],method=view['method'],
        rule='NORMAL_OPTIMIZER_SELECTED_SINGLE_Q3',configuration_id=case.get('configuration_id',case.get('original_configuration_id')),
        total_view_id=case.get('total_view_id',case['total_configuration_id']),allocation=deepcopy(case['allocation']),
        conditional_cost=deepcopy(saved_cost['conditional_model_quote_RZ']),status=status,
        MSE_lower=deepcopy(outcome.get('MSE_lower')),MSE_upper=deepcopy(outcome.get('MSE_upper')),
        result_source=view['origins']['OUTCOME'],original_API_return_saved='API_RETURN' in view['origins'],
        scientific_evaluation_calls=outcome.get('scientific_evaluation_calls',0),operational_status='UNKNOWN' if status=='UNKNOWN' else 'VALID_ORIGINAL_Q3_RECEIPT')
    summary=dict(status='NORMAL_SINGLE_PF_Q3_PUBLISHED',execution_source=execution,original_job_sources=view['origins'],
        rows=[summary_row],conditional_minima={},adopted_evaluation=deepcopy(input_view['adopted_evaluation']),
        new_scientific_calls=0,original_scientific_calls=outcome.get('scientific_evaluation_calls',0),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True,
        price_qualification=saved_cost['qualification'],achieved_conditional_cost_RZ=saved_cost['achieved_conditional_cost_RZ'])
    join=dict(kind='SINGLE_NORMAL_PF_Q3_WITH_SAVED_CONDITIONAL_PRICES',candidates=[dict(label=label,
        allocation=deepcopy(case['allocation']),model_status=status,configuration_id=summary_row['configuration_id'],
        conditional_cost=deepcopy(saved_cost['conditional_model_quote_RZ']),source=view['origins']['OUTCOME'])],
        original_job_sources=view['origins'],sufficient_N=None,G_qualified=None)
    _write(directory/'SUMMARY.json',summary);_write(directory/'C_RESULT_JOIN.json',join)
    return dict(status='PUBLISHED_GENERIC_PF_JOIN',summary=str(Path(view['study'])/'SUMMARY.json'),join=str(Path(view['study'])/'C_RESULT_JOIN.json'),scientific_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--publication-source',required=True)
    a=p.parse_args();print(json.dumps(publish(ROOT,a.study,a.publication_source),ensure_ascii=False))


if __name__=='__main__':main()
