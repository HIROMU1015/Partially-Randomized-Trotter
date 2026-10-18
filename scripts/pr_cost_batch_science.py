"""Dispatch only after a selected contract's valid grant and byte gates.

Preparation does not call any function in this module. The A backend reuses
33c7231f's tested sampling/control/compiler functions. B/D use saved scalars.
"""
from copy import deepcopy
from fractions import Fraction as F
import importlib
import inspect
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import sys

import pr_cost_batch as b
c=b.old


def native_context(spec,data,cells,output):
    import pr_cost_native as n
    config=data['native_adapter'];system=spec['system']
    for path in reversed(config['import_paths']):sys.path.insert(0,path)
    if system!='H6':return n.load_context(system,config,cells,output)
    # p3/p4 hashes were not exported by A_SIGNAL. Bind the archived full native
    # blocks and the exact already-used prefix_view source instead of inventing
    # a historical preparation hash by restoring real inputs during preparation.
    old=c.read(config['native_config'])
    for key in ('hamiltonian','state','compiler'):
        c.require(old['binding'][key]==config['binding'][key],'ORIGIN_NATIVE_BINDING')
    module=importlib.import_module('h6_dba_native')
    m,native,ham,sector,state,qstate,preps,compiler=module.load(old)
    view=importlib.import_module('h6_dba_signal').prefix_view
    policy=data['preparation_policy']
    c.require(c.sha(inspect.getsourcefile(view))==policy['view_source_sha256'],'ORIGINAL_H6_PREFIX_VIEW_SOURCE')
    c.require(preps['faithful:p19'].preparation_hash==policy['full_p19_hash']
              and preps['faithful:p5'].preparation_hash==policy['saved_p5_hash'],'H6_ARCHIVED_PREPARATION_ROOTS')
    preparations={p:(preps['faithful:p5'] if p==5 else view(preps['faithful:p19'],ham,p,m))
                  for p in sorted({x['prefix'] for x in cells})}
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    c.require(compiler_settings_hash(compiler)==spec['binding']['compiler'],'LIVE_COMPILER')
    c.save(output/'preparation_receipt.json',dict(prefixes={str(p):x.preparation_hash for p,x in preparations.items()},
        existing_p5_unchanged=True,new_Gaussian_synthesis=0,input_generation=0))
    return SimpleNamespace(m=m,native=native,qstate=qstate,preparations=preparations,compiler=compiler,binding=spec['binding'])


def a_cost(spec,grant,data,output):
    import pr_cost_native as n
    signals=c.read(data['A_signal'])['signals'];cells=[x['cell'] for x in signals]
    index={x['cell']['id']:x for x in signals}
    c.require({k:c.digest(v) for k,v in index.items()}==data['signal_record_hashes'],'PINNED_SIXTEEN_SIGNAL_RECORDS')
    selected=c.core.acquisition_candidates(signals,cells)['selected']
    context=native_context(spec,data,selected,output)
    tasks=[]
    for task in data['new_seed_registry']['tasks']:
        record=index[task['cell']['id']]
        tasks.append(dict(task,preparation_hash=context.preparations[task['cell']['prefix']].preparation_hash,
            saved_log_B=record['log_B'],log_B_tolerance=record['allowance']['log_B_margin'],
            control_tolerance=1e-9 if spec['system']=='H4' else 1e-8))
    import h8_key_compile_parallel as pool
    from h6_dba_science import checked_results
    def callback(task,directory):return n.compile_task(context,task,directory)
    def batch(items,name):
        receipt=pool.map_owned(items,callback,cpus=grant['allowed_cpus'],
            limits=dict(max_workers=len(grant['allowed_cpus'])),directory=output/name,
            deadline=math.inf,font_seed=data['native_adapter']['font_seed'])
        c.save(output/(name+'_native_receipts.json'),receipt)
        return checked_results(receipt)
    rows=batch([t for t in tasks if t['replica']==0],'first_control')
    rows+=batch([t for t in tasks if t['replica']!=0],'remaining')
    c.require(len(rows)==spec['calls']['trajectory'],'EXACT_A_COST_COUNT')
    return dict(status='COMPLETE_MANDATORY_STOP',paired_costs=rows,scope=spec['scope'],
        original_confirmation_n32_unchanged=True,new_signal=0,new_ranking=0,
        new_quantum_shots=0,next_stage_authorized=False,automatic_retry=False)


def d_accounting(data):
    """Replace actual D prices; old PR samples/cohorts remain separate."""
    before=c.digest(data);rows=[]
    for saved in data['rows']:
        row=deepcopy(saved);cid=row['cell_id']
        c.require(row['bias_mode'] in c.core.MODES,'CANONICAL_SAVED_BIAS_MODE_REQUIRED')
        if row['eligible'] and cid in data['measured']:
            weights=row['shots'];prices=data['measured'][cid]
            c.require(set(weights)=={'cosine','sine'} and all(type(x) is int and x>0 for x in weights.values()),'SAVED_SHOTS')
            row['G_exact']=str(sum(weights[a]*prices[a]['RZ'] for a in weights))
            row['axis_costs']=deepcopy(prices);row['cost_status']='DIRECT_MEASURED_SAVED_SHOTS'
        if row['method']=='B2' and row['eligible'] and cid in data['paired_PR']:
            sample=c.core.paired_statistics(data['paired_PR'][cid],row['shots'],cohort='CONFIRMATION_N32')
            c.require(F(sample['mean_exact'])==F(saved['G_exact']),'ORIGINAL_PR_MEAN_MUST_REPRODUCE')
            row['G_SE']=sample['sample_SE'];row['population_mean_certified']=False
        rows.append(row)
    comparisons=[]
    for mode in c.core.MODES:
        for epsilon in c.core.EPS:
            for factor in (1.,10.):
                endpoint=[r for r in rows if (r['bias_mode'],r['epsilon'],r['factor'])==(mode,epsilon,factor)]
                for scope,ids in data['deterministic_sets'].items():
                    def best(domain):
                        candidates=[r for r in endpoint if r['cell_id'] in domain and r['eligible'] and r['G_exact'] is not None]
                        return min(candidates,key=lambda r:(F(r['G_exact']),r['cell_id'])) if candidates else None
                    det,pr=best(ids),best(data['PR_ids'])
                    unresolved=[r['cell_id'] for r in endpoint if r['cell_id'] in ids and r['eligible'] and r['G_exact'] is None]
                    ratio=None if det is None or pr is None or unresolved else str(100*(1-F(pr['G_exact'])/F(det['G_exact'])))
                    comparisons.append(dict(bias_mode=mode,epsilon=epsilon,factor=factor,scope=scope,
                        deterministic=det,confirmed_PR=pr,PR_reduction_percent_exact=ratio,unresolved=unresolved,
                        finite_sample_point_only=True,all_candidates_direct_measured=False,strict_bias_certificate=None))
    c.require(c.digest(data)==before,'INPUT_MUTATION')
    return dict(status='COMPLETE_MANDATORY_STOP',rows=rows,comparisons=comparisons,
        unchanged_exploration_reference=data.get('exploration_reference'),
        new_signal=0,compile=0,sampling=0,fit=0,shot_recalculation=0,next_stage_authorized=False)


def b_analysis(data):
    """Use recorded law masses; absent costs/masses produce a partial diagnostic."""
    cases=[]
    for case in data['cases']:
        conditional=[]
        for group in case['conditional_groups']:
            pairs=group['paired_costs'];stats={}
            for axis in ('cosine','sine'):
                values=[x['axes'][axis]['RZ'] for x in pairs]
                c.require(len(values)==8 and all(type(x) is int and x>0 for x in values),'CONDITIONAL_N8_INTEGER_COST')
                stats[axis]=dict(n=8,mean_exact=str(F(sum(values),8)),
                    sample_SE=statistics.stdev(values)/math.sqrt(8),population_mean_certified=False)
            conditional.append(dict(stratum=group['stratum'],recorded_log_mass=group['recorded_log_mass'],
                original_distribution=group['original_distribution'],statistics=stats))
        # Scalar analysis may classify ordinary saved trajectories after approval.
        # It does not pool conditional pilots into the ordinary n32 mean.
        ordinary=case['ordinary_samples'];strata={s:[] for s in ('zero','s2_single','s2_multi','s4')}
        for sample in ordinary:
            orders=sample['orders'];n2=orders.count(2);n4=orders.count(4)
            stratum='s4' if n4 else 's2_multi' if n2>=2 else 's2_single' if n2 else 'zero'
            strata[stratum].append(sample)
        required=['zero','s2_single','s2_multi']+(['s4'] if case['cell']['K']==4 else [])
        covered={x['stratum'] for x in conditional}|{s for s,values in strata.items() if values}
        missing=sorted(set(required)-covered)
        cases.append(dict(cell_id=case['cell']['id'],conditional=conditional,
            ordinary_counts={s:len(values) for s,values in strata.items()},ordinary_n32_mean_unchanged=True,
            missing_conditional_cost_strata=missing,unconditional_mean_RZ=None,
            unconditional_mean_status='UNRESOLVED_NO_EXACT_EXHAUSTIVE_MASS_RECEIPT',
            reason='Recorded conditional log masses do not constitute a complete exact original-law mass receipt; no renormalization',
            strict_bias_certificate=None,population_mean_certified=False))
    return dict(status='COMPLETE_MANDATORY_STOP',cases=cases,
        estimand='recorded-law/conditional-cost diagnostic; unresolved unconditional mean stays null',
        new_sampling=0,compile=0,new_signal=0,cost_ranking=0,next_stage_authorized=False)


def execute(spec,grant,output):
    data=c.read(spec['payload'])
    if spec['stage']=='A_COST':return a_cost(spec,grant,data,output)
    budget=c.Budget(output,dict(saved_scalar_pass=1));budget.take('saved_scalar_pass')
    return d_accounting(data) if spec['stage']=='D_ACCOUNTING' else b_analysis(data)
