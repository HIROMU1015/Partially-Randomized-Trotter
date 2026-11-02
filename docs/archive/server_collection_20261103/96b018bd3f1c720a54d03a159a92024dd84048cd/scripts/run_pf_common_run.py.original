"""Approved common PF stage driver; each entry owns a new grant and marker."""
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from fractions import Fraction
import math
import multiprocessing
from pathlib import Path
import pf_common_run_contract as c

def run_signal(private):
    plan=c.read(Path(private)/'plan_APPROVED.json');s=c.Stage(private,'SIGNAL',plan['signal_caps'])
    try:
        import pf_common_run_native as native
        prefixes=sorted({r['prefix'] for r in plan['signal_base'] if r['method']!='FULL'})
        ctx=native.restore(s,prefixes);matrices=native.forward_matrices(ctx)
        obs=c.resources();count=c.workers(obs,len(plan['signal_base']),512*2**20,64*2**20)
        c.require(count>0,'NO_IDLE_SIGNAL_RESOURCE')
        c.save(s.root/'resource_receipt.json',dict(observation=obs,workers=count,nested_BLAS_threads=1))
        records=c.read(Path(private)/'saved_signals.json')
        def one(row):
            value=native.new_signal(s,ctx,row,matrices);c.save(s.out/(c.cid(row)+'.json'),value);return c.cid(row),value
        with ThreadPoolExecutor(max_workers=count) as pool:
            for identity,value in pool.map(one,plan['signal_base']):records[identity]=value
        for row in plan['signal_overlays']:
            base=records[c.cid(dict(row,method='B0'))]
            value=native.overlay(s,ctx,row,base);records[c.cid(row)]=value;c.save(s.out/(c.cid(row)+'.json'),value)
        c.require(set(records)=={c.cid(r) for r in plan['rows']},'FULL_SIGNAL_REGISTRATION_COVERAGE')
        c.save(s.out/'all_signals.json',records)
        return s.finish(dict(records=len(records),new_candidate_signals=len(plan['signal_base'])+len(plan['signal_overlays'])))
    except BaseException as e:s.stop(e);raise

def qualify_record(record,row,epsilon):
    import d0_frozen_shot_rule_v1 as frozen
    from h246_saved_cost_evaluation import independent_shots
    sig=record['signal'];ev=sig['allowance_evidence'];maximum=ev['maximum_intermediate_norm'];closure=ev['raw_corrected_vector_closure']
    c.require(all(math.isfinite(x) and x>=0 for x in (maximum,closure,sig['log_B'])),'FINITE_OBSERVATIONS')
    stages=row['q']*2*(row['prefix']+1)*c.PIECES[row['formula']]
    roundoff=64.*2**-52*(1+maximum)*(1+stages)
    u=max(1e-9,10.*closure+roundoff,record['old_u'])
    margin=max(1e-12,64.*2**-52*(1+abs(sig['log_B']))*(1+stages),record['old_margin'])
    biases={a:abs(sig['total_signed_bias'][a]) for a in ('real','imag')}
    for a in biases:
        c.require(math.isfinite(biases[a]) and abs(sig['corrected'][a]-sig['reference'][a]-sig['total_signed_bias'][a])<1e-9,'BIAS_REFERENCE_CLOSURE')
    result=frozen.u_aware_shots(biases,dict(real=u,imag=u),epsilon=epsilon,log_B_upper=sig['log_B']+margin,
        alpha_axis=.025,evidence_kind='EMPIRICAL',evidence_ref='COMMON_PF_NEW_POLICY_REFERENCE_CONDITIONAL')
    for a,axis in result['axes'].items():
        h,integer=independent_shots(epsilon,biases[a],u,sig['log_B']+margin)
        c.require(h==axis['headroom'],'DECIMAL_HEADROOM_EXACT')
        if axis['shots'] is not None:c.require(integer==axis['shots'],'DECIMAL_SHOT_INTEGER_EXACT')
        if axis['status']=='ELIGIBLE' and axis['u_over_headroom']>.01:axis.update(status='UNDETERMINED',shots=None)
    statuses={x['status'] for x in result['axes'].values()}
    status='INELIGIBLE' if 'INELIGIBLE' in statuses else 'ELIGIBLE' if statuses=={'ELIGIBLE'} and all(x['shots'] is not None for x in result['axes'].values()) else 'UNDETERMINED'
    return dict(candidate=row,epsilon=epsilon,status=status,axes=result['axes'],u=u,log_B=sig['log_B'],log_B_margin=margin,
        old_u=record['old_u'],old_margin=record['old_margin'],u_never_reduced=True,empirical_not_certified=True,
        N_cos=result['axes']['real']['shots'] if status=='ELIGIBLE' else None,N_sin=result['axes']['imag']['shots'] if status=='ELIGIBLE' else None,
        decimal_crosscheck=True)

def run_qualify(private):
    p=c.read(Path(private)/'plan_APPROVED.json');s=c.Stage(private,'QUALIFY',p['qualification_caps'],['SIGNAL'])
    try:
        records=c.read(Path(private)/'SIGNAL/output/all_signals.json');rows=[]
        for row in p['rows']:
            for epsilon in c.EPS:
                s.take('endpoint',c.cid(row)+'_'+str(epsilon));rows.append(qualify_record(records[c.cid(row)],row,epsilon))
        c.save(s.out/'endpoints.json',rows)
        eligible={c.cid(x['candidate']) for x in rows if x['status']=='ELIGIBLE'}
        c.save(s.out/'eligible_ids.json',sorted(eligible))
        return s.finish(dict(endpoint_count=len(rows),eligible_candidates=len(eligible),unresolved_endpoints=sum(x['status']=='UNDETERMINED' for x in rows)))
    except BaseException as e:s.stop(e);raise

def cost_caps(rows,prefixes):
    n=len(rows)
    return dict(snapshot_restore=1,saved_reference_read=1,occupation_oracle=1,full_preparation=1,
        prefix_preparation=len(prefixes),native_evolution=n,native_piece=sum(r['q']*c.PIECES[r['formula']] for r in rows),
        independent_occupation=n,native_low=n,native_high=n,wrapper_build=2*n,uncompiled_action=2*n,
        wrapper_compile=2*n,compiled_action=2*n,sampling=0,new_signal=0,new_full_H_reference=0)

def run_cost(private):
    p=c.read(Path(private)/'plan_APPROVED.json');saved=c.read(Path(private)/'saved_prices.json')
    eligible=set(c.read(Path(private)/'QUALIFY/output/eligible_ids.json'))
    selected=[r for r in p['rows'] if c.cid(r) in eligible and c.cid(r) not in saved]
    c.require({c.cid(r) for r in selected}<=set(p['compile_pool']) and len(selected)<=p['max_compile_candidates'],'FIXED_COMPILE_POOL_ONLY')
    prefixes=sorted({r['prefix'] for r in selected if r['method']!='FULL'})
    caps=cost_caps(selected,prefixes)
    if p['system']!='H2':
        old=p['cost_cap_template'];c.require(caps['native_piece']<=old['native_signed_S2_pieces'] and caps['wrapper_compile']<=old['wrapper_compile'],'ORIGINAL_NATIVE_CAPS')
    s=c.Stage(private,'COST',caps,['SIGNAL','QUALIFY'])
    try:
        c.save(s.root/'selected_candidates.json',selected)
        if selected:
            import pf_common_run_native as native
            native.STAGE=s;native.CTX=native.restore(s,prefixes);native.SELECTED=c.read(Path(private)/'SIGNAL/output/all_signals.json')
            obs=c.resources();n=c.workers(obs,len(selected),768*2**20 if p['system']=='H4' else 512*2**20,512*2**20)
            c.require(n>0,'NO_IDLE_COST_RESOURCE')
            c.save(s.root/'resource_receipt.json',dict(observation=obs,workers=n,worker_RSS_proxy_not_cap=True,compiler_processes=1,BLAS_threads=1))
            results=[]
            # Fork after shared immutable input preparation; workers do not
            # reload or regenerate contexts. Worker entry is pickle-stable.
            with ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('fork')) as pool:
                futures={pool.submit(native.cost_one,r):r for r in sorted(selected,key=lambda x:x['q']*c.PIECES[x['formula']]*(x['prefix']+1),reverse=True)}
                processes=list(pool._processes.values())
                errors=[]
                for f in as_completed(futures):
                    try:results.append(f.result())
                    except BaseException as e:
                        errors.append(dict(candidate=futures[f],reason=str(e)))
                        for pending in futures:pending.cancel()
                c.require(not errors,'COST_WORKER_FAILURE:'+str(errors))
            # The context manager joins/reaps every owned worker before PASS.
            c.require(all(p.exitcode==0 and not p.is_alive() for p in processes),'OWN_WORKER_EXIT_AND_REAP')
            c.save(s.root/'worker_exit_receipt.json',dict(workers=[dict(pid=p.pid,exit_code=p.exitcode,reaped=True) for p in processes],
                same_parent_join=True,remaining_owned_workers=0))
            for result in results:
                identity=c.cid(result['candidate']);saved[identity]=dict(cosine=result['prices']['cosine']['metrics']['rz_count'],
                    sine=result['prices']['sine']['metrics']['rz_count'],kind='NEW_COMMON_NATIVE',result_sha256=c.sha(s.out/identity/'result.json'))
        else:results=[]
        c.save(s.out/'all_prices.json',saved)
        c.require(eligible<=set(saved),'ALL_ELIGIBLE_COSTS_ACQUIRED')
        return s.finish(dict(new_pairs=len(selected),new_axes=2*len(selected),reused_pairs=len(eligible&set(c.read(Path(private)/'saved_prices.json'))),
            maximum_native_error=max((x['error'] for r in results for x in r['checks'] if not x['check'].startswith('compiled_')),default=0.),
            maximum_compiled_error=max((x['error'] for r in results for x in r['checks'] if x['check'].startswith('compiled_')),default=0.)))
    except BaseException as e:s.stop(e);raise

def best_complete(rows,formula=None):
    selected=[r for r in rows if formula is None or r['candidate']['formula']==formula]
    if any(r['status'] not in ('ELIGIBLE','INELIGIBLE') for r in selected):return dict(status='UNRESOLVED',best=None)
    available=[r for r in selected if r['status']=='ELIGIBLE']
    if any(r.get('total_RZ_exact') is None for r in available):return dict(status='UNRESOLVED',best=None)
    if not available:return dict(status='NO_ELIGIBLE_CANDIDATE',best=None)
    winner=min(available,key=lambda r:(Fraction(r['total_RZ_exact']),c.cid(r['candidate'])))
    return dict(status='COMPLETE_REGISTERED_GRID',best=winner)

def run_account(private):
    p=c.read(Path(private)/'plan_APPROVED.json');s=c.Stage(private,'ACCOUNT',dict(weighted_PF_endpoint=5*len(p['rows']),comparison=20),['QUALIFY','COST'])
    try:
        import csv
        endpoints=c.read(Path(private)/'QUALIFY/output/endpoints.json');prices=c.read(Path(private)/'COST/output/all_prices.json')
        for ep in endpoints:
            s.take('weighted_PF_endpoint',c.cid(ep['candidate'])+'_'+str(ep['epsilon']))
            price=prices.get(c.cid(ep['candidate']));ep['price']=price;ep['total_RZ_exact']=None
            if ep['status']=='ELIGIBLE' and price:
                a,b=ep['N_cos'],ep['N_sin'];x,y=price['cosine'],price['sine']
                total=a*x+b*y
                c.require(total==int(Fraction(a)*x+Fraction(b)*y),'EXACT_PF_ACCOUNTING')
                ep['total_RZ_exact']=str(total)
        prior=list(csv.DictReader((c.ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv').open()))
        pr=[r for r in prior if r['system']==p['system'] and r['method']=='PR' and r['mode']=='USUAL_SIGNED']
        c.require(len(pr)==60,'SAVED_PR_12_GROUPS_FIVE_EPS')
        comparisons=[]
        for ep in c.EPS:
            pf_ep=[r for r in endpoints if r['epsilon']==ep]
            for formula in [*c.FORMULAS,None]:
                s.take('comparison',str(ep)+'_'+str(formula))
                pf=best_complete(pf_ep,formula)
                candidates=[r for r in pr if float(r['epsilon'])==ep and (formula is None or r['formula']==formula) and r['status']=='ELIGIBLE']
                # Preserve saved exact exploratory means, SE, n and policy.
                def pr_cost(r):return Fraction(int(r['total_RZ_numerator']),int(r['total_RZ_denominator']))
                winner=min(candidates,key=lambda r:(pr_cost(r),r['candidate_id'])) if candidates else None
                diff=ratio=None
                if pf['best'] and winner:
                    v=Fraction(pf['best']['total_RZ_exact']);u=pr_cost(winner);diff=v-u;ratio=u/v
                    c.require(math.isclose(float(ratio),float(u)/float(v),rel_tol=256*2**-52),'INDEPENDENT_RATIO_ARITHMETIC')
                comparisons.append(dict(system=p['system'],epsilon=ep,formula=formula or 'ALL_FORMULAS',PF=pf,PR_saved=winner,
                    PR_over_PF_exact=None if ratio is None else str(ratio),PF_minus_PR_exact=None if diff is None else str(diff),
                    PR_reduction_percent=None if ratio is None else float(100*(1-ratio)),
                    point_direction='UNRESOLVED' if diff is None else 'PR_LOWER' if diff>0 else 'PF_LOWER' if diff<0 else 'TIE',
                    exploratory_not_population_guarantee=True,independent_confirmation=False))
        c.save(s.out/'PF_endpoint_costs.json',endpoints);c.save(s.out/'comparisons.json',comparisons)
        c.save(s.out/'scope_limits.json',dict(PR_new_sampling=0,PR_repricing=0,PR_statistics_recomputed=0,confirmation_pooling=False,
            finite_sample_selection_effect_uncorrected=True,unobserved_rare_event_cost_bound=None,empirical_allowance_not_certificate=True,
            full_RPE_energy_cost=False,q16_included=False))
        return s.finish(dict(comparison_rows=len(comparisons),PF_endpoints=len(endpoints)))
    except BaseException as e:s.stop(e);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['signal','qualify','cost','account']);p.add_argument('--private',required=True)
    a=p.parse_args();result=globals()['run_'+a.stage](a.private);print(c.encoded(result).decode())
