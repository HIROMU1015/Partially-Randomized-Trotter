"""Static PF/PR integration plan. This CLI cannot run science or sum costs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as p

H6='14c4f871fb9e8d4a23a6cc3244f5435ebc65a7ce'
H4='cefc01622a3cd75dd16d30de7b9fb68274f3eddd'
H6C='a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4'


def ref(commit,path):return dict(commit=commit,path=path)


def collect(store, system):
    if system=='H6':
        coverage_ref=ref(H6,'docs/research/h6_df19_rpe_m7_results_20261110/COVERAGE.json')
        coverage=store.read(coverage_ref);saved=[]
        for row in coverage['rows']:
            origin=ref(H6,row['result_reference']['path']);artifact=store.read(origin)
            point=next(x for x in artifact['rows'] if x['candidate_id']==row['candidate_id'])
            if (point['ideal_numerical_pilot']['N_X']!=row['N_X'] or point['ideal_numerical_pilot']['N_Y']!=row['N_Y']
                    or artifact['scientific_source']!=row['scientific_source']):raise ValueError('Saved H6 coverage/result mismatch')
            saved.append((point,dict(origin,scientific_source=artifact['scientific_source'])))
        config=coverage['configuration'];source=coverage_ref
        for row in coverage['rows']:
            original=store.read(ref(H6,row['result_reference']['path']))['configuration']
            fields=['base_time','energy_precision_ha','beta_total','beta_stat','alpha_total_per_method','alpha_axis','energy_prior_ha','energy_shift','decoder']
            if any(not p.same(original[k],config[k]) for k in fields) or original['horizon']['maximum_round_index_M']!=config['horizon']['maximum_round_index_M']:
                raise ValueError('Historical N policy differs from the common task')
    elif system=='H4':
        source=ref(H4,'docs/research/h4_df10_rpe_d1_20261110/result.json');artifact=store.read(source)
        config=artifact['configuration'];saved=[(x,dict(source,scientific_source=artifact['scientific_source'])) for x in artifact['rows']]
    else:raise ValueError('H4/H6 saved carriers only')
    rows=[];existing={(r['m'],p.arm(r['candidate'])):r for r,s in saved}
    for m in range(config['horizon']['maximum_round_index_M']+1):
        for alg in p.ARMS:
            if (m,alg) in existing:
                old=existing[m,alg];rows.append({k:deepcopy(old[k]) for k in ['m','q_RPE','q_PF','candidate']});continue
            # Proposals only. No science, optimum or scientific adoption is asserted.
            base=existing[(7 if system=='H6' else 3),alg]['candidate']
            c=deepcopy(base);c['T']=math_time=config['base_time']*(1<<m)
            c['q']=(1<<(m-4)) if system=='H6' else 1<<m
            c['delta']=math_time/c['q'];c['id']=p.candidate_id(c)
            rows.append(dict(m=m,q_RPE=1<<m,q_PF=c['q'],candidate=c,adoption='PROPOSAL_NOT_SCIENTIFICALLY_ADOPTED'))
    binding=saved[0][0]['ideal_numerical_pilot']['binding']
    contract=dict(completion='saved_weighted_Hermitian_completion',fragment_order='saved_snapshot_order',
        preparation_binding=dict(snapshot=binding['snapshot'],provider_source='c31ef424175c02379eafebe8fc6c7548c1d9aabe',
            qualification='UPSTREAM_SAVED_RECEIPT_CANDIDATE_BINDING'),
        state_preparation_cost=0,input_source=source)
    task=p.task_from_saved(config,binding,rows,task_id=saved[0][0]['ideal_numerical_pilot']['target_id'],source=source,contract=contract)
    shots=[];prices=[]
    for point,origin in saved:
        row=next(r for r in task['candidates'] if p.key(r)==p.key(point))
        shots.append(p.shot_record(task,row,point,origin));prices.extend(p.saved_direct_prices(task,row,point))
    task['saved_signal_sources']={}
    if system=='H4':
        signal_catalog=store.read(ref(H4,'docs/research/h4_df10_rpe_cost_connection_20261110/COVERAGE.json'))
        task['saved_signal_sources']={r['candidate']['id']:r['signal_source'] for r in signal_catalog['candidate_coverage']
            if r.get('signal_source') and any(p.same(r['candidate'],x['candidate']) for x in task['candidates'])}
        direct=ref(H4,'docs/research/h4_df10_price_v1_20261110/HOLDOUT.json')
        existing_C={p.key(r['identity']) for r in prices}
        prices.extend(x for x in p.saved_deterministic_points(task,store.read(direct),direct) if p.key(x['identity']) not in existing_C)
    if system=='H6':
        model_sources=[ref(H6C,'docs/research/h6_df19_price_v1_20261110/'+leaf) for leaf in ['DEVELOPMENT.json','HOLDOUT.json']]
        for origin in model_sources:prices.extend(p.saved_model_prices(task,store.read(origin),origin,format='H6_V1'))
    else:
        model_sources=[ref(H4,'docs/research/h4_df10_price_v2_s1_20261110/RESULT.json')]
        for origin in model_sources:prices.extend(p.saved_model_prices(task,store.read(origin),origin,format='H4_V2_S1'))
    return task,shots,prices,model_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Fresh integration plan output required')
    store=p.SavedArtifacts(ROOT);packets={}
    for system in ['H4','H6']:
        task,shots,prices,models=collect(store,system)
        inventory=p.plan(task,shots,prices)
        packets[system]=dict(task=task,inventory=inventory,model_sources=models,
            saved_N=[dict(m=n['identity']['m'],candidate_id=n['identity']['candidate']['id'],N_X=n['plan']['N_X'],N_Y=n['plan']['N_Y'],actual=deepcopy(n['actual']),source=n['source']) for n in shots],
            saved_C_records=len(prices),numerical_prices_aggregated=False,
            batch_request=dict(N_candidates=inventory['missing_N'],C_candidates=inventory['unqualified_expected_C'],
                saved_signal_scalar_candidates=[x for x in inventory['missing_N'] if x in task['saved_signal_sources']],
                new_signal_candidates=[x for x in inventory['missing_N'] if x not in task['saved_signal_sources']],
                provider_calls='one grouped N request then one grouped price request; no retry; no totals in planning'),
            total_resources=None)
    out.mkdir(parents=True)
    for system,packet in packets.items():
        (out/(system+'_PLAN.json')).write_text(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'SOURCES.json').write_text(json.dumps(store.sources,indent=2)+'\n')
    print(json.dumps(dict(status='STATIC_INTEGRATION_PLANS_WRITTEN',systems=list(packets),new_scientific_calls=0,N_times_C_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
