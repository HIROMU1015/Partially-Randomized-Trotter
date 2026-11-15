"""Fixed 60-axis conditional compiler-closure prices; no fit, compile, N or G."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from trottertracks import price_implementation_guard as G
from trottertracks import price_model_validation as V
from admit_pr_df_ld1_q3_prices import at, equal, require

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/pf_closure_prices_20261115'
RECOMMENDATION=dict(commit='c3f761f0af5b21207e64051c9c619858064450fe',path='docs/research/pf_closure_price_recommendation_20261115/RECOMMENDATION.json')
FROZEN=dict(commit='12372fa6b53c878dc5dd124e036572ba0cd82061',path='docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
AUDIT=dict(commit='606881b042ed69f9a1e7f00311417f2b65855d7b',path='docs/research/d_pf_qpy_closure_audit_20261115/run01/AUDIT_RESULT.json')
D= AUDIT['commit']
M3=dict(commit='3b1301738507d612694bf865d904b9376e038f69',path='docs/research/pf_m3_saved_price_admission_20261115/ADMISSION.json')
ROUNDS=(7,8,10,11,12)
METHODS={'FULL_PF':('FULL',10),'TRUNCATED_PF':('B0I',7)}
POLICY_ID='PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1'


def key(row):
    return tuple(row[k] for k in ('configuration_id','physical_id','m','axis'))


def targets(root):
    proposal=G.evidence(root,RECOMMENDATION); frozen=G.evidence(root,FROZEN); audit=G.evidence(root,AUDIT)
    equal(len(proposal['rows']),60,'Fixed sixty rows')
    require(frozen['long_q_extrapolation_qualified'] is False,'Preserve historical global qualification')
    rows=[]
    for j,pr in enumerate(proposal['rows']):
        group=pr['model_group'];model=frozen['models'][group];r=G.evidence(root,pr['alignment'])
        p=r['physical_candidate'];first=model['input_locators'][0]
        equal((r['configuration_id'],r['physical_id'],r['m'],r['axis']),key(pr),'Original alignment identity')
        equal((p['id'],p['q'],p['T'],p['delta'],p['formula']),
              (pr['physical_id'],pr['q'],pr['T'],pr['actual_delta'],pr['formula']),'Physical schedule')
        equal((p['method'],p['prefix']),METHODS[pr['route']],'Fixed method/split')
        equal(model['model_group'],[pr['route'],pr['formula'],pr['width_over_h'],pr['axis']],'Own route/formula/axis model')
        equal(model['features'],['one','q'],'Original feature design')
        equal((model['domain']['route'],model['domain']['axis']),(pr['route'],pr['axis']),'Model domain')
        equal((first['configuration_id'],first['actual_delta']),(pr['configuration_id'],pr['actual_delta']),'Fixed configuration/width')
        require(model['qualified_for_candidate'] is False,'Historical model flag unchanged')
        a=G.evidence(root,pr['audit'])
        equal(a['group'],group,'Own audit group')
        require(all(a['A'+str(n)]=='PASS' for n in range(1,6)),'Saved audit PASS')
        equal(a['barriers'],8,'Eight recorded system-wire motifs')
        equal(pr['terminal_diagnostic_status'],'MARGIN_OK','Saved terminal diagnostic')
        require(pr['m'] in ROUNDS and pr['price'] is None,'Only adopted unobserved targets')
        rows.append(dict(configuration_id=r['configuration_id'],physical_id=p['id'],m=r['m'],axis=r['axis'],
            route=pr['route'],formula=p['formula'],q=p['q'],T=p['T'],actual_delta=p['delta'],
            canonical_id=r['original_canonical_id'],physical_candidate=deepcopy(p),model_group_id=group,
            width_over_h=pr['width_over_h'],features=dict(one=1,q=p['q']),coefficients=deepcopy(model['coefficients']),
            recommendation_row=at(RECOMMENDATION,'rows',j),alignment=deepcopy(pr['alignment']),frozen_model=deepcopy(pr['frozen_model']),
            closure_audit=deepcopy(pr['audit']),terminal_diagnostic=deepcopy(pr['saved_terminal_diagnostic']),
            price_profile=deepcopy(model['domain']['profile']),implementation=deepcopy(model['domain']['implementation']),
            scalar=deepcopy(first['scalar']),input_binding=deepcopy(first['input_binding']),split=deepcopy(first['split']),
            fragment_order=first['fragment_order'],signed_structure=deepcopy(r['structure'])))
    check_scope(rows)
    return rows,frozen['models'],deepcopy(proposal['assumptions'])


def check_scope(rows):
    require(len(rows)==60 and len({key(r) for r in rows})==60,'No missing/duplicate target axes')
    groups={r['model_group_id'] for r in rows}
    require(len(groups)==12,'Twelve model groups')
    for group in groups:
        equal(sorted(r['m'] for r in rows if r['model_group_id']==group),list(ROUNDS),'Five adopted rounds per group')


def direct_inventory(root):
    """Reuse D records and C saved admissions, retaining original price locators."""
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',D,'docs/research/d_full_pf_shortlist_prices_20261114',
        'docs/research/d_truncated_p7_t3_prices_20261114'],cwd=root,text=True).splitlines()
    price_paths=[p for p in paths if p.endswith('/PRICES.json')]
    inventory={}
    for path in price_paths:
        for i,record in enumerate(G.evidence(root,dict(commit=D,path=path))):
            if record.get('classification')!='DIRECT_COMPILED_DETERMINISTIC':continue
            p=record['physical_context']['physical_candidate']
            row=dict(configuration_id=record['configuration_id'],physical_id=p['id'],m=record['m'],axis=record['axis'])
            row['source']=dict(commit=D,path=path,selector=[i])
            inventory[key(row)]=row
    for i,row in enumerate(G.evidence(root,M3)['rows']):
        equal(row['status'],'ACCEPTED_SAVED_DIRECT_POINT','Original m3 admission')
        inventory[key(row)]={k:deepcopy(row[k]) for k in ('configuration_id','physical_id','m','axis')}
        inventory[key(row)]['source']=at(M3,'rows',i)
    # Four M5/m4 axes were C-admitted EXPLORATION receipts, not D new prices.
    for model in G.evidence(root,FROZEN)['models'].values():
        for inp in model['input_locators']:
            if inp['origin']!='C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED':continue
            row=dict(configuration_id=inp['configuration_id'],physical_id=inp['physical_candidate']['id'],m=inp['m'],axis=model['domain']['axis'])
            require(key(row) not in inventory,'Saved admission must not duplicate a new D record')
            row.update(source=deepcopy(inp['price_source']),record_source=deepcopy(inp['record_source']),
                original_price_source=deepcopy(inp['original_price_source']),original_cohort=inp['cohort'])
            inventory[key(row)]=row
    require(len(inventory)==96,'Known direct inventory should contain 96 unique axes')
    return inventory,price_paths


def preflight(root,rows):
    """Metadata inventory only: no raw/QPY, no observed-price arithmetic."""
    root=Path(root);refs=subprocess.check_output(['git','ls-remote','origin',
        'refs/heads/codex/m5-m4-receipt-admission-20261114','refs/heads/codex/d-price-records-c-bridge-20261114'],cwd=root,text=True).splitlines()
    d_head=next(x.split()[0] for x in refs if x.endswith('refs/heads/codex/d-price-records-c-bridge-20261114'))
    require(d_head==D,'New D publication needs a focused duplicate check before evaluation')
    inventory,price_paths=direct_inventory(root)
    overlap=sorted(set(inventory)&{key(r) for r in rows})
    require(not overlap,'Existing direct target found: preserve/reuse its receipt before model evaluation')
    jobs=[]
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,comm=,args='],text=True).splitlines():
        p=line.strip().split(None,2)
        if len(p)<3 or p[0]==str(os.getpid()) or 'python' not in p[1]:continue
        scripts=[Path(x).name for x in shlex.split(p[2])[1:] if x.endswith('.py')]
        if any(re.search(r'pf.*(?:closure|price|predict)|run_full_pf_shortlist|run_truncated_p7',s,re.I) for s in scripts):
            jobs.append(dict(pid=int(p[0]),scripts=scripts))
    require(not jobs,'Related ongoing job needs duplicate-scope resolution')
    old_results=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD',BASE],cwd=root,text=True).splitlines()
    require(not any(p.endswith('/PRICES.json') for p in old_results),'Reuse existing fixed prices instead of recalculating')
    return dict(status='CLEAR_IN_SCOPED_METADATA',checked_UTC=datetime.now(timezone.utc).isoformat(),public_refs=refs,
        known_direct_axes=96,direct_inventory=list(inventory.values()),price_documents_checked=price_paths,
        overlap_with_targets=overlap,related_same_user_jobs=jobs,existing_own_price_results=old_results,
        history_check='No same-scope price freeze in related C history; recommendation c3f761f0a contains no values.',
        limitation='Scoped C/D published refs, known price inventories and same-user relevant runner names; unlisted archives/other hosts are not claimed exhaustive.',
        QPY_read=False,raw_read=False,price_reaggregation=False)


def calculate(rows,models,assumptions,source):
    check_scope(rows);out=[]
    for row in rows:
        r=deepcopy(row);model=models[r['model_group_id']]
        value=V.predict(model,r['features'])
        require(value.denominator==1 and value>=0,'Exact nonnegative integer RZ required; no rounding')
        r.update(compiled_RZ=value.numerator,exact_RZ=V._enc(value),mean=dict(rz_count=value.numerator),
            classification='MODEL_PREDICTION',expectation='DETERMINISTIC_CONDITIONAL_COMPILER_MODEL',
            quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',units='RZ_COUNT_PER_MEASURED_FULL_CONTROLLED_WRAPPER',
            price_status='CONDITIONAL_CLOSURE_DERIVED_AVAILABLE',evidence_kind='FIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL',
            direct_observation=False,unobserved_circuit=True,conditional=True,conditional_use_adopted=True,
            adoption_policy=POLICY_ID,model=dict(source=r['frozen_model'],qualified_for_candidate=False,
                original_status=model['status'],validation_source=r['closure_audit'],conditional_use_adopted=True),
            source=dict(commit=source,module='scripts/freeze_pf_closure_prices.py',function='price_model_validation.predict'),
            P=0,normalization_is_price_multiplier=False,n_paths=None,cohort='DERIVED_NO_NEW_SAMPLING',
            error_bound=None,CI=None,two_percent_guarantee=False)
        out.append(r)
    return dict(schema='PF_CONDITIONAL_CLOSURE_PRICES_V1',status='COMPLETE_CONDITIONAL_PRICES',execution_source=source,
        recommendation=RECOMMENDATION,frozen_models=FROZEN,closure_audit=AUDIT,adoption_policy=POLICY_ID,
        assumptions=deepcopy(assumptions),use_scope='Same physical candidates, fixed-shot allocation, full 13-round conditional comparison only',
        rows=out,counts=dict(requested=60,calculated=len(out),reused_direct=0,held=0,groups=12,rounds_per_group=5),
        coverage=dict(existing_direct_axes=96,conditional_derived_axes=60,total_axes=156,direct_plus_derived_overlap=0),
        prediction_calls=len(out),fit_calls=0,compile_calls=0,QPY_reads=0,unconditional_error_bound=None,two_percent_guarantee=False,
        original_qualification_flags_modified=False,N=None,G=None,winner=None,mandatory_stop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true');ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    require(args.prepare != args.execute,'Choose metadata preparation or the approved 60-price execution')
    rows,models,assumptions=targets(ROOT);base=ROOT/BASE
    if args.prepare:
        require(not base.exists(),'Preserve existing preparation/results')
        check=preflight(ROOT,rows);base.mkdir()
        (base/'PREFLIGHT.json').write_text(json.dumps(check,indent=2)+'\n')
        print(json.dumps(dict(targets=len(rows),direct_inventory=check['known_direct_axes'],overlap=0,prediction_calls=0)));return
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(G.C.blob(ROOT,source,'scripts/freeze_pf_closure_prices.py')==Path(__file__).read_bytes(),'Committed runner SOURCE')
    require(G.C.blob(ROOT,FROZEN['commit'],V.MODULE)==(ROOT/V.MODULE).read_bytes(),'Unchanged original exact predictor')
    require(G.C.blob(ROOT,source,BASE+'/PREFLIGHT.json')==(base/'PREFLIGHT.json').read_bytes(),'Fixed preflight')
    out=base/'run01';require(not out.exists(),'Reuse completed output; no repeat execution')
    check=preflight(ROOT,rows);out.mkdir()
    start=time.perf_counter();print('BEGIN exact prediction of 60 fixed axes',flush=True)
    try:
        data=calculate(rows,models,assumptions,source)
        data['elapsed_seconds']=time.perf_counter()-start
        (out/'PRICES.json').write_text(json.dumps(data,indent=2)+'\n')
        with (out/'PRICES.csv').open('w',newline='') as handle:
            fields=['configuration_id','physical_id','route','formula','model_group_id','m','axis','T','q','actual_delta','compiled_RZ','classification','conditional_use_adopted','direct_observation']
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(data['rows'])
        (out/'CHECKS.json').write_text(json.dumps(dict(status='COMPLETE',source=source,preflight=check,
            missing=0,duplicates=0,existing_direct_overlap=0,prediction_calls=60,fit_calls=0,N=None,G=None,mandatory_stop=True),indent=2)+'\n')
        print(json.dumps(dict(source=source,counts=data['counts'],prediction_calls=60,seconds=data['elapsed_seconds'],mandatory_stop=True)),flush=True)
    except Exception as error:
        (out/'STOP.json').write_text(json.dumps(dict(status='FAILED_NO_AUTOMATIC_RETRY',reason=str(error),source=source,mandatory_stop=True),indent=2)+'\n')
        raise


if __name__=='__main__':main()
