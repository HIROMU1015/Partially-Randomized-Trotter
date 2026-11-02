"""Bounded stdlib-only posthoc geometry analysis; no sampling or science imports."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import time

GEOMETRIES=('0.70','0.80','0.90','1.10','1.40','1.60')
DELTAS=(0,1,10,100,200,500,1000,2000,3000,5000,10000)


def load_api(repo):
    p=repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/verify_saved_result_projection.py'
    s=importlib.util.spec_from_file_location('published_cost_arithmetic',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def config(tid):
    parts=tid.split('-rank')[1].split('-')
    return {'prefix':int(parts[0]),'q':int(parts[1][1:]),'r':int(parts[2][1:]),'K':int(parts[3][1:])}


def method(tid):
    return 'B0' if tid.startswith('PM1-B0-') else tid.split('-')[0]


def factors(row,p,e):
    return {'bias_cos':row['saved']['bias']['cosine'],'bias_sin':row['saved']['bias']['sine'],
            'B':row['saved']['normalization'],
            'headroom_cos':e/math.sqrt(2)-row['saved']['bias']['cosine'],
            'headroom_sin':e/math.sqrt(2)-row['saved']['bias']['sine'],
            'N_cos':p['shots'][0],'N_sin':p['shots'][1],'N_total':sum(p['shots']),
            'C_cos':row['_statistics'][0],'C_sin':row['_statistics'][1],
            'C_effective':p['work']/sum(p['shots']),'G':p['work'],'SE':p['SE'],
            'cov_cos_sin':row['_statistics'][3]}


def law(B,R,K):
    def ws(t):
        return [(1. if n==0 else 0.) if t==0 else math.exp(n*math.log(t)-math.lgamma(n+1))*math.hypot(1,t/(n+1)) for n in range(0,K+1,2)]
    if B==1.:
        keys=('tau_inferred_from_saved_B','p0','p2','p4','probability_nonzero_in_trajectory',
              'probability_all32_zero','expected_order2_draws_in32','expected_order4_draws_in32',
              'probability_no_order2_in32','probability_no_order4_in32')
        return {**{k:None for k in keys},'normalization_resolution':'ROUNDED_TO_ONE_UNRESOLVED'}
    if not math.isfinite(B) or B<1:raise ValueError('invalid saved normalization')
    target=math.expm1(math.log(B)/R)
    def increment(t):
        # Avoid cancellation of hypot(1,t)-1 and b(t)-1 near B=1.
        return t*t/(math.hypot(1,t)+1)+math.fsum(ws(t)[1:])
    lo,hi=0.,1.
    while increment(hi)<target:hi*=2
    for _ in range(85):
        mid=(lo+hi)/2
        if increment(mid)<target:lo=mid
        else:hi=mid
    tau=(lo+hi)/2;weights=ws(tau);total=math.fsum(weights);p=[w/total for w in weights]
    # Source p0 division may round to one for tiny masses. Sum positive masses
    # and log1p separately for this probability diagnostic.
    nonzero=math.fsum(p[1:]);logzero=math.log1p(-nonzero)
    return {'normalization_resolution':'INFERRED_FROM_ROUNDED_SAVED_B',
            'tau_inferred_from_saved_B':tau,'p0':p[0],'p2':p[1],'p4':p[2] if K==4 else None,
            'probability_nonzero_in_trajectory':-math.expm1(R*logzero),
            'probability_all32_zero':math.exp(32*R*logzero),
            'expected_order2_draws_in32':32*R*p[1],
            'expected_order4_draws_in32':32*R*p[2] if K==4 else None,
            'probability_no_order2_in32':math.exp(32*R*math.log1p(-p[1])),
            'probability_no_order4_in32':math.exp(32*R*math.log1p(-p[2])) if K==4 else None}


def regimes(rows,key):
    result=[]
    for g in GEOMETRIES:
        selected=[r for r in rows if r['geometry']==g]
        start=0
        for i in range(1,len(selected)+1):
            if i==len(selected) or selected[i][key]!=selected[start][key]:
                r=selected[start];last=selected[i-1]
                result.append({'geometry':g,'series':key,'template':r[key],
                  'epsilon_first_grid':r['epsilon'],'epsilon_last_grid':last['epsilon'],
                  'grid_points':i-start,'next_epsilon_grid':selected[i]['epsilon'] if i<len(selected) else None,
                  'boundary_kind':'bracket between executed scalar grid points, not continuous exact transition'})
                start=i
    return result


def tests(api):
    count=0
    row={'saved':{'normalization':1.,'bias':{'cosine':0.,'sine':0.},'paired_costs':[[7,8]]},'_statistics':api.statistics([[7,8]])}
    p=api.evaluate(row,.05);assert p['shots']==[math.ceil(2*math.log(80)/(.05/math.sqrt(2))**2)]*2;count+=1
    assert p['work']==sum(n*c for n,c in zip(p['shots'],(7,8)));count+=1
    pairs=[[k,k+2] for k in range(32)];row['_statistics']=api.statistics(pairs);p=api.evaluate(row,.05)
    samples=[p['shots'][0]*x+p['shots'][1]*y for x,y in pairs]
    mean=math.fsum(samples)/32;se=math.sqrt(math.fsum((x-mean)**2 for x in samples)/31/32)
    assert math.isclose(p['SE'],se,rel_tol=1e-12);count+=1
    row['saved']['bias']['cosine']=.05/math.sqrt(2);assert api.evaluate(row,.05) is None;count+=1
    for K in (2,4):
        t=.11;R=8
        w=[t**n/math.factorial(n)*math.hypot(1,t/(n+1)) for n in range(0,K+1,2)]
        d=law(math.fsum(w)**R,R,K)
        assert math.isclose(d['tau_inferred_from_saved_B'],t,rel_tol=1e-12);count+=1
        assert 0<=d['probability_all32_zero']<=1;count+=1
    assert method('PM1-B0-rank5-q1-r0-K0')=='B0';count+=1
    assert config('B2-rank3-q4-r8-K2')==dict(prefix=3,q=4,r=8,K=2);count+=1
    assert (100-80)/10<(100-85)/5;count+=1 # Max threshold need not be zero-delta winner.
    phases=[-.2,-.3,-.1];assert math.isclose(sum(phases),-.6);count+=1
    assert law(1.,8,4)['probability_all32_zero'] is None;count+=1
    tiny=law(1.+2e-15,64,2)
    assert 0<tiny['tau_inferred_from_saved_B']<1e-8 and tiny['p2']>0;count+=1
    return count


def run(repo,out):
    started=time.monotonic();out.mkdir(exist_ok=False)
    api=load_api(repo)
    base=repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13'
    scalar=repo/'artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13'
    proj=json.loads((base/'saved_candidates_v1.json').read_text());prior=json.loads((base/'summary.json').read_text())
    rows=proj['candidates'];groups=defaultdict(list)
    for r in rows:r['_statistics']=api.statistics(r['saved']['paired_costs']);groups[r['geometry']].append(r)
    assert len(rows)==1308 and all(len(groups[g])==218 for g in GEOMETRIES)
    lookup={(r['geometry'],r['template_id']):r for r in rows}
    aux=defaultdict(list)
    for v in json.loads((scalar/'saved_b0_phase_sensitivity_v1.json').read_text()):
        orig=lookup[v['geometry'],v['original_template_id']]
        aux[v['geometry']].append({'template_id':v['original_template_id']+'+I','saved':{'bias':v['counterfactual_bias'],
            'normalization':1.,'paired_costs':orig['saved']['paired_costs']},'_statistics':orig['_statistics']})
    grid=[.005*20**(i/300) for i in range(301)];grid[0],grid[-1]=.005,.1;grid=sorted(set(grid+[.05]))
    table=[];classification=[];scalar_regimes=[];nearby=[];delta_counts={delta:Counter() for delta in DELTAS}
    original_delta_counts={delta:Counter() for delta in DELTAS};b2_selection=Counter();reverse_ids=set();threshold_ids=set()
    for g in GEOMETRIES:
        for e in grid:
            pairs=[(r,api.evaluate(r,e)) for r in groups[g]]
            eligible=sorted(((r,p) for r,p in pairs if p is not None),key=lambda x:(x[1]['work'],x[0]['template_id']))
            winner,p=eligible[0];assert method(winner['template_id'])=='B2'
            minima={m:min(((r,v) for r,v in eligible if method(r['template_id'])==m),key=lambda x:x[1]['work'],default=None) for m in ('B0','B1','B2','B3')}
            det,pd=min((x for m,x in minima.items() if m in ('B0','B1') and x),key=lambda x:x[1]['work'])
            cfg=config(winner['template_id']);b2_selection[winner['template_id']]+=1
            close_candidates=[(r,v) for r,v in eligible[1:] if v['engineering_interval'][0]<=p['engineering_interval'][1]
                              and v['engineering_interval'][1]>=p['engineering_interval'][0]]
            second=eligible[1]
            for nr,nv in close_candidates:
                nearby.append({'geometry':g,'epsilon':e,'point_minimum':winner['template_id'],
                    'candidate':nr['template_id'],'G':nv['work'],'SE':nv['SE'],
                    'N_cos':nv['shots'][0],'N_sin':nv['shots'][1],
                    'gap_to_minimum_percent':100*(nv['work']/p['work']-1),
                    'status':'overlapping descriptive +-2SE; no simultaneous or postselection coverage'})
            record={'geometry':g,'epsilon':e,'eligible_total':len(eligible),'winner':winner['template_id'],**cfg,**factors(winner,p,e),
                    'best_deterministic':det['template_id'],'det_G':pd['work'],'det_N_total':sum(pd['shots']),
                    'det_C_effective':pd['work']/sum(pd['shots']),'ratio_det':p['work']/pd['work'],
                    'ratio_N_det':sum(p['shots'])/sum(pd['shots']),
                    'ratio_C_effective_det':(p['work']/sum(p['shots']))/(pd['work']/sum(pd['shots'])),
                    'reduction_percent':100*(1-p['work']/pd['work']),
                    'upper_2SE_below_det':p['work']+2*p['SE']<pd['work'],
                    'second_template':second[0]['template_id'],'second_G':second[1]['work'],'second_SE':second[1]['SE'],
                    'gap_second_percent':100*(second[1]['work']/p['work']-1),
                    'overlapping_competitors':';'.join(r['template_id'] for r,v in close_candidates),
                    'overlap_count':len(close_candidates)}
            for m,x in minima.items():
                record[m+'_eligible']=sum(method(r['template_id'])==m for r,v in eligible)
                record[m+'_template']=x[0]['template_id'] if x else None
                record[m+'_G']=x[1]['work'] if x else None
                record[m+'_ratio_B2']=x[1]['work']/p['work'] if x else None
            assert math.isclose(record['ratio_det'],record['ratio_N_det']*record['ratio_C_effective_det'],rel_tol=1e-12)
            table.append(record)
            a=sorted(((r,api.evaluate(r,e)) for r in aux[g]),key=lambda x:x[0]['template_id'])
            a=[(r,v) for r,v in a if v is not None]
            ai,pi=min(a,key=lambda x:x[1]['work'])
            max_r,max_p=max(a,key=lambda x:(p['work']-x[1]['work'])/sum(x[1]['shots']))
            threshold=(p['work']-pi['work'])/sum(pi['shots'])
            max_threshold=(p['work']-max_p['work'])/sum(max_p['shots'])
            reverse=pi['work']<p['work']
            if reverse:reverse_ids.add((g,ai['template_id']));threshold_ids.add((g,max_r['template_id']))
            cr={'geometry':g,'epsilon':e,'zero_extra_reversal':reverse,'B0_I_template':ai['template_id'],
                'B0_I_prefix':config(ai['template_id'].removesuffix('+I'))['prefix'],
                'B0_I_q':config(ai['template_id'].removesuffix('+I'))['q'],'B0_I_G0':pi['work'],
                'B2_template':winner['template_id'],'B2_G':p['work'],
                'gap_reference_G':p['work']-pi['work'],'ratio_reference_to_B2':pi['work']/p['work'],
                'threshold_for_zero_delta_winner':threshold,'max_threshold_any_B0_I':max_threshold,
                'max_threshold_template':max_r['template_id'],
                'B0_I_N_cos':pi['shots'][0],'B0_I_N_sin':pi['shots'][1],'B0_I_N_total':sum(pi['shots'])}
            classification.append(cr)
            scalar_regimes.append({'geometry':g,'epsilon':e,'B0_I_template':ai['template_id'],
                                   'zero_extra_comparison':'B0_I_below' if reverse else 'B2_below_or_equal'})
            for delta in DELTAS:
                delta_counts[delta][g]+=min(v['work']+sum(v['shots'])*delta for r,v in a)<p['work']
                original_delta_counts[delta][g]+=pi['work']+sum(pi['shots'])*delta<p['work']
    assert len(table)==len(classification)==1812
    assert sum(r['zero_extra_reversal'] for r in classification)==397
    # Verify previously fixed diagnostics; the new any-candidate envelope is
    # distinguished from holding the delta=0 candidate fixed.
    with (scalar/'scalar_reference_envelope_1812.csv').open(newline='') as f:old=list(csv.DictReader(f))
    for prev,now in zip(old,classification):
        assert prev['geometry']==now['geometry'] and float(prev['epsilon'])==now['epsilon']
        api.near(float(prev['B0_I_reference_G_zero_extra']),now['B0_I_G0'])
        api.near(float(prev['common_extra_RZ_per_shot_break_even']),now['threshold_for_zero_delta_winner'])
    switches=regimes(table,'winner')+regimes(table,'B0_template')
    scalar_switches=regimes(scalar_regimes,'B0_I_template')+regimes(scalar_regimes,'zero_extra_comparison')
    fixed=[]
    fixed_ids=['B2-rank3-q1-r4-K2','B2-rank3-q1-r4-K4','B2-rank3-q1-r8-K4',
               'B2-rank3-q2-r4-K2','B2-rank3-q4-r2-K4','PM1-B0-rank5-q1-r0-K0']
    for tid in fixed_ids:
        for e in (.005,.05,.1):
            previous=None
            for g in GEOMETRIES:
                r=lookup[g,tid];p=api.evaluate(r,e)
                v={'geometry':g,'epsilon':e,'fixed_template':tid,'eligible':p is not None}
                detail=factors(r,p,e) if p else {k:None for k in ('bias_cos','bias_sin','B','headroom_cos','headroom_sin',
                    'N_cos','N_sin','N_total','C_cos','C_sin','C_effective','G','SE','cov_cos_sin')}
                v.update(detail)
                if previous and p:
                    v.update(ratio_G_prev=p['work']/previous['G'],ratio_N_prev=sum(p['shots'])/previous['N_total'],
                             ratio_Ceff_prev=detail['C_effective']/previous['C_effective'],
                             log_G_ratio=math.log(p['work']/previous['G']),
                             log_N_ratio=math.log(sum(p['shots'])/previous['N_total']),
                             log_Ceff_ratio=math.log(detail['C_effective']/previous['C_effective']))
                    assert math.isclose(v['log_G_ratio'],v['log_N_ratio']+v['log_Ceff_ratio'],abs_tol=1e-12)
                else:
                    for k in ('ratio_G_prev','ratio_N_prev','ratio_Ceff_prev','log_G_ratio','log_N_ratio','log_Ceff_ratio'):v[k]=None
                fixed.append(v);previous=detail if p else None
    rare=[]
    for r in rows:
        if method(r['template_id']) not in ('B2','B3'):continue
        cfg=config(r['template_id']);R=cfg['q']*cfg['r'];d=law(r['saved']['normalization'],R,cfg['K'])
        rare.append({'geometry':r['geometry'],'template':r['template_id'],**cfg,'R_per_trajectory':R,
                     'B':r['saved']['normalization'],'mean_RZ_cos':r['_statistics'][0],
                     'mean_RZ_sin':r['_statistics'][1],'SE_RZ_cos':math.sqrt(r['_statistics'][2]/32),
                     'SE_RZ_sin':math.sqrt(r['_statistics'][4]/32),'paired_cost_samples':32,
                     **d,'observed_order0':'NOT_SAVED','observed_order2':'NOT_SAVED',
                     'observed_order4':'NOT_SAVED' if cfg['K']==4 else 'OUTSIDE_CUTOFF',
                     'nonzero_trajectories_observed':'NOT_SAVED','component_observations':'NOT_SAVED',
                     'probability_scope':'law inferred from rounded saved B, not observed event counts or cost bound'})
    deltarows=[{'common_delta_RZ_per_axis_shot':v,'reoptimized_auxiliary_points_below_B2':sum(delta_counts[v].values()),
                'fixed_zero_delta_candidate_points_below_B2':sum(original_delta_counts[v].values()),
                **{g:delta_counts[v][g] for g in GEOMETRIES},'scope':'unmeasured common nonnegative delta assumption'} for v in DELTAS]
    reversals=[r for r in classification if r['zero_extra_reversal']]
    reversal_groups=defaultdict(list)
    for v in reversals:
        reversal_groups[v['geometry'],v['B0_I_template'],v['B2_template']].append(v)
    grouped=[]
    for (g,a,c),values in sorted(reversal_groups.items()):
        grouped.append({'geometry':g,'B0_I_template':a,'B2_template':c,'grid_points':len(values),
            'epsilon_min':min(v['epsilon'] for v in values),'epsilon_max':max(v['epsilon'] for v in values),
            'gap_G_min':min(v['gap_reference_G'] for v in values),'gap_G_max':max(v['gap_reference_G'] for v in values),
            'threshold_min_fixed_zero_delta':min(v['threshold_for_zero_delta_winner'] for v in values),
            'threshold_max_fixed_zero_delta':max(v['threshold_for_zero_delta_winner'] for v in values),
            'scope':'grid range only; detailed membership is in scalar_classification_1812.csv'})
    h6style=next(v for v in fixed if v['geometry']=='0.90' and v['epsilon']==.05 and v['fixed_template']=='B2-rank3-q1-r4-K4')
    plans=[{'scope':'representative epsilon=.05 reversals','configurations':[[g,tid.removesuffix('+I')] for g,tid in sorted({('0.70','PM1-B0-rank5-q1-r0-K0+I'),('0.80','PM1-B0-rank5-q1-r0-K0+I'),('1.40','B0-rank3-q1-r0-K0+I')})],
            'compile_calls':6,'meaning':'exact modified costs only for these 3 fixed wrappers; not full 397-point/auxiliary-family winner coverage'},
           {'scope':'delta=0 minimizing configurations appearing at reversed grid points','configurations':[[g,tid.removesuffix('+I')] for g,tid in sorted(reverse_ids)],
            'compile_calls':2*len(reverse_ids),'meaning':'fixed reported candidates at those points; a new compiler delta can change auxiliary winner'},
           {'scope':'all registered B0 configurations with phase retained','configurations_count':120,'compile_calls':240,
            'meaning':'would measure full B0+I auxiliary class for this same q grid; requires separate science decision/source/auth/resource budget'}]
    rare_selected=[r for r in rare if r['template'] in b2_selection]
    summary={'status':'TECHNICAL_SAVED_VALUE_INTEGRATION_COMPLETE_SCIENTIFIC_DECISIONS_PENDING',
      'base_result_commit':'7c4483378047b57f514c47dad98905e3f2a9a756','base_scalar_commit':'c81f7326dde3f40686b3caefe05e2f544d8d1826',
      'science_source':'d0726e946f89f8c30a08d383fb8ff48b363629ec','candidates':1308,'grid_rows':1812,
      'candidate_precision_scalar_evaluations':395016,'all_registered_point_minima_B2':True,
      'winning_prefix_occupancy':dict(Counter(r['prefix'] for r in table)),
      'winning_q_occupancy':dict(Counter(r['q'] for r in table)),
      'winning_template_occupancy':dict(b2_selection),
      'winning_template_regimes':sum(r['series']=='winner' for r in switches),
      'all_upper_2SE_below_registered_det_points':sum(r['upper_2SE_below_det'] for r in table),
      'points_with_overlapping_competitors':sum(r['overlap_count']>0 for r in table),
      'nearby_candidate_precision_rows':len(nearby),
      'scalar_reference_reversals':397,'scalar_reversal_prefix_q_occupancy':dict(Counter(f"p{r['B0_I_prefix']}/q{r['B0_I_q']}" for r in reversals)),
      'reversal_threshold_min':min(r['threshold_for_zero_delta_winner'] for r in reversals),
      'reversal_threshold_max':max(r['threshold_for_zero_delta_winner'] for r in reversals),
      'delta_sensitivity':deltarows,'fixed_0p80_to_0p90_q1r4K4_epsilon_0p05':h6style,
      'rare_law_diagnostics':len(rare),'selected_minimum_rare_law_examples':rare_selected,
      'rare_law_normalization_unresolved_rows':sum(r['normalization_resolution']=='ROUNDED_TO_ONE_UNRESOLVED' for r in rare),
      'rare_order_counts_observed':'NOT_SAVED; reconstruction by new draws intentionally not performed',
      'minimal_compile_plans_proposals_only':plans,'B0_I_formal_baseline_adopted':False,'compiled_B0_I_cost':'UNMEASURED',
      'new_trajectory_signal_compile_eigen_Hamiltonian_GPU':0,'H8_modified':False,
      'artificial_arithmetic_checks_passed':tests(api),
      'summary_wall_seconds':time.monotonic()-started,
      'numerical_policy':'binary64 saved values, no certified population/simultaneous intervals; grid posthoc and selection dependent'}
    write_csv(out/'precision_grid_1812.csv',table);write_csv(out/'configuration_switches.csv',switches)
    write_csv(out/'scalar_classification_1812.csv',classification);write_csv(out/'scalar_comparison_switches.csv',scalar_switches)
    write_csv(out/'scalar_reversal_groups.csv',grouped);write_csv(out/'nearby_candidates.csv',nearby)
    write_csv(out/'delta_sensitivity.csv',deltarows);write_csv(out/'fixed_configuration_factors.csv',fixed)
    write_csv(out/'rare_order_law_diagnostics.csv',rare)
    write_csv(out/'scalar_threshold_extremes.csv',sorted(reversals,key=lambda r:r['threshold_for_zero_delta_winner'])[:12]+sorted(reversals,key=lambda r:-r['threshold_for_zero_delta_winner'])[:12])
    (out/'analysis_summary_v1.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    assert sum(p.stat().st_size for p in out.iterdir())<16*2**20
    print(json.dumps({k:summary[k] for k in ('status','grid_rows','winning_prefix_occupancy','winning_template_regimes',
        'all_upper_2SE_below_registered_det_points','points_with_overlapping_competitors','scalar_reversal_prefix_q_occupancy',
        'reversal_threshold_min','reversal_threshold_max','delta_sensitivity','minimal_compile_plans_proposals_only',
        'artificial_arithmetic_checks_passed','summary_wall_seconds')},ensure_ascii=False))


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.repo,a.output)


if __name__=='__main__':main()
