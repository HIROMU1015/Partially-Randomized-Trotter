"""Exact accounting of saved F1 quotes plus admitted m0 receipts; never predict."""
import argparse
from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import time

from trottertracks import price_implementation_guard as G
from admit_pr_df_ld1_q3_prices import equal, require
from admit_full_h_d7_holdout import SHOTS, at
from run_full_h_f1_diagnostic import exact, rational
from admit_full_h_d7_holdout import ROOT

BASE='docs/research/full_h_m0_reference_total_20261115'
RULE=dict(commit='4a1cb0d7364fece475437749129fdce740fd8fc4',
          path='docs/research/full_h_m0_price_rule_20261115/PROPOSAL.json')
F1=dict(commit='a1aa38df46c92e2489095891f6082a5be1a277f5',
        path='docs/research/full_h_f1_model_diagnostic_20261115/run01/RESULT.json')
PF=dict(commit='b4296b110b129d226dbef2743c14f725a27c3bf6',
        path='docs/research/pf6_cost_budget_q3_results_20261115/RESULTS.csv')
PF_VALUES={'FULL_PF_S2_16h_over_7__B_PRICE':615304502,'B0I_P7_S2_16h_over_7__B_PRICE':435481090}
MODULE='scripts/account_full_h_reference_scenario.py'


def saved_inputs(root):
    rule=G.evidence(root,RULE);f1=G.evidence(root,F1)
    equal(rule['physical_candidate']['id'],'FULL_H_2a9a68a11d08af234a8c','Fixed m0 rule')
    require(f1['T_ref'] is None,'Original F1 null total is preserved')
    for unit in rule['units']:
        b=G.evidence(root,unit['B_join'])
        equal((b['m'],b['axis'],b['physical_id'],b['N']),
              (0,unit['axis'],unit['physical_id'],96),'B m0 physical/axis/allocation')
    rows=[]
    for i,r in enumerate(f1['rows']):
        if r['m']==0:continue
        equal(r['status'],'MODEL_SCENARIO_ONLY','Saved F1 quote status')
        require(rational(r['C_ref'])>=0,'Saved nonnegative model quote')
        equal(r['N'],SHOTS[r['m']],'Fixed scale3/2 vector')
        b=G.evidence(root,r['B_join'])
        equal((b['m'],b['axis'],b['physical_id'],b['N']),
              (r['m'],r['axis'],r['physical_id'],r['N']),'B fixed m/axis/physical/N')
        rows.append(dict(m=r['m'],axis=r['axis'],physical_id=r['physical_id'],N=r['N'],
            C_ref=deepcopy(r['C_ref']),status='ACCEPTED_SAVED_REFERENCE',
            price_origin='SAVED_F1_MODEL_REFERENCE',source=at(F1,'rows',i),
            original_configuration_id=r['original_configuration_id'],total_configuration_id=r['total_configuration_id'],
            original_candidate_id=r['original_candidate_id'],total_candidate_id=r['total_candidate_id'],
            physical_candidate=deepcopy(r['physical_candidate']),B_join=deepcopy(r['B_join']),
            model_source=deepcopy(r['model_source']),qualified_E_C=False))
    equal({(r['m'],r['axis']) for r in rows},{(m,a) for m in range(1,13) for a in ('X','Y')},'24 saved F1 axes')
    table=list(csv.DictReader(StringIO(G.C.blob(root,PF['commit'],PF['path']).decode())))
    benchmarks=[]
    for label,value in PF_VALUES.items():
        found=[(i,r) for i,r in enumerate(table) if r['label']==label]
        equal(len(found),1,'One fixed PF comparison row')
        i,r=found[0]
        equal(int(r['conditional_cost']),value,'Fixed PF achieved cost')
        equal(r['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','Saved PF Q3 model status')
        benchmarks.append(dict(label=label,T_PF=value,source=dict(PF,format='CSV',row_index=i,row_label=label),
            allocation_rule=r['rule'],N_X=r['N_X'],N_Y=r['N_Y'],qualification='CONDITIONAL_ACHIEVED_REFERENCE'))
    return rule,rows,benchmarks


def account(rows,benchmarks):
    """One scenario only; return null without summing if either receipt is held."""
    equal(len(rows),26,'26 axes')
    equal({(r['m'],r['axis']) for r in rows},{(m,a) for m in range(13) for a in ('X','Y')},'Unique 13-round schedule')
    missing=[dict(m=r['m'],axis=r['axis'],reason=r.get('reason')) for r in rows if r['status']!='ACCEPTED_SAVED_REFERENCE']
    if missing:
        return dict(status='HELD_NO_TOTAL',T=None,m0_partial=None,m0_fraction=None,m0_percent=None,
            rows=deepcopy(rows),missing=missing,comparisons=[dict(benchmark=b,T_minus_T_PF=None) for b in benchmarks])
    require(all(rational(r['C_ref'])>=0 for r in rows),'Nonnegative supplied prices')
    require(all(type(r['N']) is int and r['N']>=0 for r in rows),'Integer shot counts')
    result_rows=[]
    for r in rows:
        result_rows.append(dict(deepcopy(r),weighted_RZ=exact(r['N']*rational(r['C_ref']))))
    total=sum((rational(r['weighted_RZ']) for r in result_rows))
    m0=sum((rational(r['weighted_RZ']) for r in result_rows if r['m']==0))
    return dict(status='MODEL_SCENARIO_ONLY_WITH_EMPIRICAL_M0',T=exact(total),m0_partial=exact(m0),
        m0_fraction=exact(m0/total) if total else None,m0_percent=exact(100*m0/total) if total else None,
        rows=result_rows,missing=[],comparisons=[dict(benchmark=deepcopy(b),T_minus_T_PF=exact(total-b['T_PF']),
            relation='FULL_H_HIGHER' if total>b['T_PF'] else 'FULL_H_LOWER' if total<b['T_PF'] else 'EQUAL') for b in benchmarks])


def held_m0(rule, mapping, job, result, stop, prices, observed, compiled_records):
    """Copy a published producer STOP; observed compiles never become prices."""
    equal(mapping['rule_source'], RULE, 'Fixed m0 rule SOURCE')
    equal(job['rule_source'], RULE, 'D1 adopted price rule')
    for key, expected in (('row',rule['physical_candidate']),('profile',rule['profile']),
                          ('implementation',rule['implementation'])):
        equal(job[key], expected, 'Fixed D1 '+key)
    equal((job['cohort'],job['index'],job['seeds'],job['independent_paths']),
          (rule['proposed_acquisition']['cohort'],0,[rule['proposed_acquisition']['seed_u64']],1),'One fixed normal path')
    equal(result['source']['commit'], mapping['execution_source'], 'D1 producer SOURCE')
    equal(result['status'], 'STOP', 'Published D1 STOP, not a successful handoff')
    require(result['mandatory_stop'] is True, 'D1 mandatory STOP preserved')
    context=result['child']['context_check'];equal(context['status'],'PASS','D1 recorded context check')
    saved=context['saved_context']
    equal(saved['preparation_hash'],rule['target_preparation_hash'],'D7 preparation')
    equal(saved['scalar'],rule['scalar']['total'],'Total scalar')
    for key,value in rule['H_state_snapshot'].items():
        equal(saved['binding'][key],value,'Saved H/state/snapshot '+key)
    equal(context['compiler_profile'],rule['profile'],'Full measured wrapper/compiler/P0')
    equal(context['implementation'],rule['implementation'],'Native lowering/scalar application')
    equal(stop['accepted_axis_prices'],0,'No D1 accepted price')
    equal(stop['held_axis_prices'],2,'Two held axes')
    require(stop['C_0X'] is None and stop['C_0Y'] is None,'D1 null per-shot prices')
    equal({r['axis'] for r in prices},{'X','Y'},'Two axis-specific price statuses')
    equal({r['axis'] for r in observed},{'X','Y'},'Two original axis compile observations')
    equal((len(prices),len(observed)),(2,2),'No duplicate axis record')
    rows=[];admissions=[]
    for unit in rule['units']:
        axis=unit['axis'];i=next(i for i,r in enumerate(prices) if r['axis']==axis)
        j=next(i for i,r in enumerate(observed) if r['axis']==axis)
        original=prices[i];obs=observed[j];record=compiled_records[axis]
        equal((original['m'],original['physical_id'],original['per_shot_RZ'],original['acceptance']),
              (0,unit['physical_id'],None,'HELD_NULL_REQUIRED_EVIDENCE_INCOMPLETE'),'Original held price')
        equal(original['source'],result['source'],'Original axis producer')
        equal((obs['axis'],obs['physical_id'],obs['path_index'],obs['cohort'],obs['seed']),
              (axis,unit['physical_id'],0,job['cohort'],job['seeds'][0]),'Axis/path binding')
        equal(obs['source'],result['source'],'Observed compile producer')
        equal(obs['status'],'OBSERVED_COMPILE_ONLY_NOT_ACCEPTED_REFERENCE','No promotion of observed price')
        require(obs['per_shot_RZ'] is None,'No accepted D1 unit price')
        equal((record['compiler'],record['fingerprint'],record['metrics']['rz_count']),
              (rule['profile']['compiler'],obs['fingerprint'],obs['observed_compiled_RZ']),'Original compile record binding')
        equal((obs['measurement']['axis'],obs['measurement']['include_measurement']),(axis,True),'Original X/Y measurement')
        reason=stop['failed_requirement']
        rows.append(dict(m=0,axis=axis,physical_id=unit['physical_id'],N=96,C_ref=None,status='HELD',reason=reason,
            price_origin='D1_OBSERVED_COMPILE_ONLY_NOT_ACCEPTED_REFERENCE',qualified_E_C=False,
            **{k:deepcopy(unit[k]) for k in ('original_configuration_id','total_configuration_id',
                'original_candidate_id','total_candidate_id','B_join')},
            physical_candidate=deepcopy(rule['physical_candidate']),source=at(mapping['prices'],i)))
        admissions.append(dict(axis=axis,status='HELD_D1_REQUIRED_GATE_STOP',reason=reason,
            original_price=deepcopy(original),original_observation=deepcopy(obs),
            price_source=at(mapping['prices'],i),observation_source=at(mapping['observed'],j),
            compile_record_source=dict(commit=mapping['observed']['commit'],path=obs['compile_record']['path']),
            stop_source=mapping['stop'],accepted_per_shot_RZ=None))
    require(observed[0]['fingerprint']!=observed[1]['fingerprint'],'Distinct axis circuit fingerprints')
    return rows,admissions


def admit_m0(root,rule,mapping):
    """Current published m0 is a STOP. Do not reconstruct unpersisted gates."""
    records={}
    observed=G.evidence(root,mapping['observed'])
    for obs in observed:
        raw=G.C.blob(root,mapping['observed']['commit'],obs['compile_record']['path'])
        equal(hashlib.sha256(raw).hexdigest(),obs['compile_record']['sha256'],'Original compile bytes')
        records[obs['axis']]=json.loads(raw)
    return held_m0(rule,mapping,G.evidence(root,mapping['job']),G.evidence(root,mapping['result']),
        G.evidence(root,mapping['stop']),G.evidence(root,mapping['prices']),observed,records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--mapping',type=Path,required=True,help='Fixed C locator-only map to the published D1 STOP and receipts')
    args=p.parse_args()
    if not args.execute:p.error('Use --execute for the permitted fixed scenario')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    equal((ROOT/MODULE).read_bytes(),G.C.blob(ROOT,source,MODULE),'Fixed accounting source')
    rel=args.mapping.resolve().relative_to(ROOT)
    equal(args.mapping.read_bytes(),G.C.blob(ROOT,source,str(rel)),'Fixed D1 locator mapping')
    mapping=json.loads(args.mapping.read_text())
    out=ROOT/BASE/'run01';require(not out.exists(),'Reuse fixed result; no repeat accounting')
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    print('BEGIN fixed saved-only m0 admission and 26-axis scenario accounting',flush=True)
    try:
        rule,saved,benchmarks=saved_inputs(ROOT)
        m0,admissions=admit_m0(ROOT,rule,mapping)
        result=account(m0+saved,benchmarks)
        result.update(schema='FULL_H_M0_EMPIRICAL_FIXED_SCALE3O2_SCENARIO_V1',execution_source=source,
            sources=dict(rule=RULE,F1=F1,PF=PF,D1_mapping=dict(commit=source,path=str(rel))),
            m0_admissions=admissions,shared_m0_paths=1,
            reused_F1_model_axes=24,new_prediction_calls=0,new_paths=0,new_compiles=0,new_fit=0,new_Q3=0,
            original_F1_total_remains_null=True,model_qualification_changed=False,E_C=None,CI=None,
            law_distance=None,finite_cohort_uncertainty=None,shared_covariance=None,price_tolerance=None,
            sufficient_N=None,G_qualified=None,winner=None,wall_seconds=time.perf_counter()-start,mandatory_stop=True)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        with (out/'AXES.csv').open('w',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n')
            writer.writerow(['m','axis','physical_id','N','price_numerator','price_denominator','weighted_numerator','weighted_denominator','price_origin','status'])
            for r in result['rows']:
                price=r.get('C_ref') or {};term=r.get('weighted_RZ') or {}
                writer.writerow([r['m'],r['axis'],r['physical_id'],r['N'],price.get('numerator'),price.get('denominator'),
                    term.get('numerator'),term.get('denominator'),r['price_origin'],r['status']])
        print(json.dumps(dict(status=result['status'],m0_accepted=sum(r.get('accepted_per_shot_RZ') is not None for r in admissions),
            T=result['T'],mandatory_stop=True)),flush=True)
    except Exception as error:
        (out/'STOP.json').write_text(json.dumps(dict(source=source,status='STOP_NO_AUTOMATIC_RETRY',reason=repr(error),mandatory_stop=True),indent=2)+'\n')
        raise


if __name__=='__main__':main()
