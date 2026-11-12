"""Fixed proposal A/S0: two saved H4 PR means only, no new physical actions.

Real invocation is scientific reanalysis. The user's approved S0 scope covers
materialization, saved numerical gates and empirical nominal N, not sufficient
N/global RMSE. No normalizer, Hamiltonian/state loader, sampler or C/G is called.
"""
from copy import deepcopy
from fractions import Fraction
import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_rpe_rmse_pilot as P
from trottertracks.pr_r_detailed_n_p1 import checked_reference
from trottertracks.pr_r_dependent_n import saved_task
A=P.A;p=P.p
PLAN='d97a40ab719bd71fee4ad41d5acc2abe3c89904d'
BASE='ee15bbd29ec571dd2526feecdb81ed757f783557'
IDS={0:'53e00d8b07d1547d19a7',6:'2a8c6718220eec59d419'}
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
CONSUMED=('src/trottertracks/pr_rpe_rmse.py','src/trottertracks/pr_rpe_rmse_pilot.py',
 'src/trottertracks/pr_rpe_rmse_preflight.py','src/trottertracks/pf_pr_rpe_pipeline.py',
 'src/trottertracks/pr_r_dependent_n.py','src/trottertracks/pr_r_detailed_n_p1.py',
 'src/trottertracks/c1_stage3_accuracy.py','src/trotterlib/h46_m5_plan.py','scripts/h46_long_time_signal.py')


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def materialize(design,tail,source):
    """Exactly one approved pair of M/kappa; no search or alternative policy."""
    if (design['system'],design['base_time'],design['formula'],design['delta'],design['prefix'],design['K'])!=('H4',.1,'2nd',.1,1,2):
        raise ValueError('S0 outer design differs')
    if design['PF']!=dict(method='B0',prefix=7):raise ValueError('Preserve H4 B0 baseline metadata')
    lam=tail['lambda_R']
    if type(lam) not in (int,float) or not math.isfinite(lam) or lam<=0:raise ValueError('Saved positive finite lambda required')
    h=Fraction(design['base_time']);kappa=Fraction(255,128)/(Fraction(lam)**2*h*h)
    target=Fraction('0.00159360010199040');budget=target/2
    M=max(6,math.ceil(math.log2(.10*math.pi/(float(h)*float(budget)))))
    horizon=.10*math.pi/math.ldexp(float(h),M)
    if horizon>float(budget) or M>6 and .10*math.pi/math.ldexp(float(h),M-1)<=float(budget):
        raise ValueError('Unresolved empirical M ceiling; no automatic adjustment')
    policy=A.rmse_policy(target_rmse=str(target),qpe_rmse=str(budget),evolution_rmse=str(budget),
                         error_rule='minkowski',normalization_rule='finite_B_diagnostic',source=source)
    args=dict(M=M,kappa_paper=str(kappa),design=deepcopy(design),tail_norm=deepcopy(tail),policy=policy)
    return args,dict(M=M,round_count_M_plus_one=M+1,h=float(h),h_exact=str(h),lambda_R_saved=lam,
        kappa_paper_exact=str(kappa),kappa_paper_approx=float(kappa),dimensionless_x_exact='255/128',
        target_rmse_ha_exact=str(target),qpe_budget_ha_exact=str(budget),trot_budget_ha_exact=str(budget),
        target_rmse_ha=float(target),qpe_budget_ha=float(budget),trot_budget_ha=float(budget),
        empirical_qpe_horizon_ha=horizon,empirical_horizon_is_rigorous_bound=False,
        error_rule='minkowski',normalization_rule='finite_B_diagnostic')


def numeric_diagnostics(request,saved,reference):
    """Scalar-only P1 gates/disk; the old analyze/Hoeffding wrapper is not used."""
    i=request['identity'];c=i['candidate'];d=saved['diagnostics'];sig=deepcopy(d['signal'])
    if (not p.same(saved['identity']['candidate'],c) or not p.same(saved['identity']['binding'],i['binding'])
            or not p.same(saved['identity']['profile'],i['profile']) or saved['plan']['status']!='CONDITIONAL_ROUND_PASS'
            or any(d[k]!=i['structure'][k] for k in ('signed_weights','allocation','event_count'))):
        raise ValueError('Saved candidate/input/allocation/status mismatch')
    checked=checked_reference(reference,i['binding'],c['T'])['evidence']
    if not p.same(sig['reference'],checked['reference']):raise ValueError('Signal/reference SOURCE differs')
    z=lambda v:complex(v['real'],v['imag'])
    for key in ('reference','corrected','raw','exact_tail','total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias'):
        if any(type(sig[key].get(part)) not in (int,float) or not math.isfinite(sig[key][part]) for part in ('real','imag')):
            raise ValueError('Nonfinite saved signal')
    L=sig['log_B'];ev=sig['allowance_evidence'];allow=d['allowance']
    if type(L) not in (int,float) or not math.isfinite(L) or L<0:raise ValueError('Saved log B numerical range')
    if not math.isfinite(ev['maximum_intermediate_norm']) or ev['maximum_intermediate_norm']<0:raise ValueError('Saved maximum norm gate')
    if any(not math.isfinite(ev[k]) or not 0<=ev[k]<=1e-9 for k in ('raw_corrected_vector_closure','exact_tail_norm_error')):
        raise ValueError('Saved norm/independent closure gate')
    pieces=len(i['structure']['signed_weights']);events=i['structure']['event_count']
    stages=c['q']*(2*(c['prefix']+1)*pieces+pieces)+2*events
    roundoff=64*2**-52*(1+ev['maximum_intermediate_norm'])*(1+stages)
    u=allow['u'];margin=allow['log_B_margin'];pair=checked['signal_difference']
    if (allow.get('policy')!='R_AWARE_EMPIRICAL_ACTION_COUNT_V1' or allow.get('certified') is not False
            or allow['stages']!=stages or not math.isfinite(u) or not math.isfinite(margin)
            or u<max(1e-9,10*ev['raw_corrected_vector_closure']+roundoff)+pair
            or margin<max(1e-12,64*2**-52*(1+abs(L))*(1+stages))):
        raise ValueError('Preserved R-aware allowance does not cover original recipe')
    ref,corr,raw,exact=(z(sig[k]) for k in ('reference','corrected','raw','exact_tail'))
    if min(abs(ref),abs(corr),abs(raw))==0:raise ValueError('Unresolved zero signal')
    residuals=dict(total_bias=abs((corr-ref)-z(sig['total_signed_bias'])),
        outer_bias=abs((exact-ref)-z(sig['outer_PF_or_discard_bias'])),
        finite_bias=abs((corr-exact)-z(sig['finite_RTE_bias'])))
    if max(residuals.values())>math.hypot(u,u):raise ValueError('Saved bias decomposition gate')
    attenuation=math.exp(-L);closure=abs(raw-attenuation*corr)
    if closure>1e-9:raise ValueError('Scalar raw/corrected closure gate')
    energy=d['reference_solver_energy_ha'];residual=d['state_residual_ha']
    if not math.isfinite(energy) or not math.isfinite(residual) or residual<0 or d['ground_state_certified'] is not False:
        raise ValueError('Saved numerical state evidence differs')
    phase_reference=cmath.exp(-1j*energy*c['T'])  # diagnostic only, never prior/decoder branch
    reference_offset=abs(ref-phase_reference);residual_guard=c['T']*residual
    box=math.hypot(abs(z(sig['total_signed_bias']).real)+u,abs(z(sig['total_signed_bias']).imag)+u)
    disk=math.fsum((box,reference_offset,residual_guard))
    norm_error=math.fsum((closure,attenuation*(1+disk)*math.expm1(margin)))
    raw_error=math.fsum((attenuation*disk,norm_error))
    geometry=P.supplied_disk_geometry(attenuation,raw_error,error_source=dict(
        commit=P.P1,path=P.P1_DIR+'run01/result.json',candidate_id=c['id'],
        recipe='Preserved empirical conditional D1 disk; no implementation/law/ground bound'))
    if geometry['status']!='LOCAL_SUPPLIED_DISK_DIAGNOSTIC_ONLY':raise ValueError('Conditional radius numerical failure')
    if any(v is not None for v in (d['error_disk']['unconditional_ground_identification_error'],)):
        raise ValueError('Do not silently change original ground qualification')
    return dict(candidate=deepcopy(c),signal=sig,numerical_gates_passed=True,
        reference_pair=checked,bias_decomposition_residuals=residuals,
        allowance=deepcopy(allow),normalization=dict(log_B=L,B_from_saved_log=math.exp(L),
            attenuation=attenuation,scalar_raw_corrected_closure=closure,raw_margin=norm_error),
        observed=dict(corrected_radius=abs(corr),raw_radius=abs(raw),exact_tail_radius=abs(exact),
            corrected_phase_to_full_H_reference=cmath.phase(corr/ref),raw_phase_to_full_H_reference=cmath.phase(raw/ref),
            absolute_corrected_bias=abs(corr-ref),absolute_finite_RTE_bias=abs(corr-exact)),
        conditional_error_disk=dict(coordinate_box_disk=box,reference_offset=reference_offset,
            residual_guard=residual_guard,corrected_signal_disk=disk,raw_signal_error=raw_error),
        conditional_geometry=geometry,
        actual_implementation=dict(raw_error=None,law_error=None,ground_identification_error=None,
            radius_lower=None,beta_deterministic=None,RMSE_bound_ha=None,status='UNQUALIFIED_UNKNOWN_ERRORS'),
        state_energy_used_for_prior_or_branch=False,rigorous_error_bound=False)


def execute(root,out,source):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='S0_RUNNING',scientific_source=source,plan_source=PLAN,diagnostics=[],model_records=[],
        scope=dict(selection_materializations=0,saved_candidates_attempted=0,saved_candidates_reanalyzed=0,
            nominal_axis_pairs_attempted=0,nominal_axis_pairs=0,
            new_finite_means=0,new_exact_tail_controls=0,new_reference_actions=0,new_normalizer_calls=0,
            H_state_array_loads=0,sampling=0,compile=0,C=0,G=0),
        errors=[],automatic_retry=False,sufficient_N_qualified=False,full_RPE_qualified=False,mandatory_stop=True)
    save(out/'result.json',result);stage='SOURCE_AND_INPUT_BINDING'
    try:
        if blob(root,source,'scripts/run_pr_rpe_rmse_s0.py')!=(root/'scripts/run_pr_rpe_rmse_s0.py').read_bytes():raise ValueError('Runner SOURCE not fixed')
        files={}
        for path in CONSUMED:
            data=(root/path).read_bytes()
            if data!=blob(root,BASE,path):raise ValueError('Existing SOURCE differs: '+path)
            files[path]=sha(data)
        files['scripts/run_pr_rpe_rmse_s0.py']=sha((root/'scripts/run_pr_rpe_rmse_s0.py').read_bytes())
        store=p.SavedArtifacts(root);design=A.design_from_task(saved_task(root,'H4'))
        index=store.read(INDEX);spec=index['T20_reference_files']['H4'];context_data=Path(spec['path']).read_bytes()
        if sha(context_data)!=spec['sha256']:raise ValueError('H4 saved context hash differs')
        context=json.loads(context_data)
        if not p.same(context['binding'],design['binding']) or context['sector_dimension']!=36:raise ValueError('H4 context binding differs')
        prior=store.read(dict(commit=P.P1,path='docs/research/pr_r_normalization_phase0_preparation_20261111/PLAN.json'))
        tail=deepcopy(prior['saved_tails']['H4'])
        if tail['lambda_R']!=context['prefixes']['1']['lambda_R'] or tail['source']['context_sha256']!=spec['sha256']:
            raise ValueError('Saved tail value/SOURCE differs')
        old=store.read(dict(commit=P.P1,path=P.P1_DIR+'run01/result.json'))
        provenance=dict(scientific_source=source,consumed_files=files,saved_Git_sources=store.sources,
            context_SOURCE=dict(index=INDEX,field='T20_reference_files.H4',sha256=spec['sha256']),
            input_binding=design['binding'],original_sources_preserved=True,H_state_arrays_read=False,
            context_unchanged_after_execution=None)
        save(out/'SOURCE_INDEX.json',provenance)
        stage='FIXED_M_KAPPA_MATERIALIZATION'
        source_ref=dict(commit=source,path='scripts/run_pr_rpe_rmse_s0.py',plan=PLAN)
        result['scope']['selection_materializations']=1
        args,conditions=materialize(design,tail,source_ref)
        schedule=A.schedule(**args);bridge=P.b_metadata(root);config=bridge.generate(bridge.a_spec(args))
        plan=P.prepare_pilot(config,schedule,P.pilot_inventory(root),root=root)
        requests=[r for r in schedule['rows'] if r['identity']['m'] in IDS and p.arm(r['identity']['candidate'])=='PR']
        if {r['identity']['m']:r['identity']['candidate']['id'] for r in requests}!=IDS or plan['status']!='TECHNICALLY_PREPARED_NOT_SCIENTIFICALLY_QUALIFIED':
            raise ValueError('S0 candidate/preflight mismatch; no fallback')
        save(out/'CONFIGURATION.json',dict(source=source_ref,conditions=conditions,inputs=args,A_schedule=schedule,B_configuration=config))
        result['conditions']=conditions;save(out/'result.json',result)
        stage='TWO_SAVED_NUMERICAL_GATES'
        means=[]
        for request in requests:
            c=request['identity']['candidate'];saved=next(r for r in old['records'] if r['identity']['candidate']['id']==c['id'])
            ref=old['references']['H4:'+str(c['T'])]
            result['scope']['saved_candidates_attempted']+=1
            diag=numeric_diagnostics(request,saved,ref);result['diagnostics'].append(diag)
            result['scope']['saved_candidates_reanalyzed']+=1
            record=dict(candidate=deepcopy(c),signal=diag['signal'],**{k:request['identity']['structure'][k] for k in ('signed_weights','allocation','event_count')})
            means.append(A.mean_evidence(request,record,dict(commit=P.P1,path=P.P1_DIR+'run01/result.json',candidate_id=c['id'])))
            save(out/'result.json',result)
        stage='TWO_NOMINAL_N_PAIRS'
        result['scope']['nominal_axis_pairs_attempted']=2
        evaluation=A.evaluate(schedule,mean_provider=lambda all_requests:deepcopy(means))
        result['model_records']=[r for r in evaluation['records'] if r['request_id'] in {v['request_id'] for v in means}]
        if len(result['model_records'])!=2 or any(r['nominal_N_X'] is None or r['nominal_N_Y'] is None or r['status'].startswith(('INAPPLICABLE','NUMERICAL')) for r in result['model_records']):
            raise ValueError('Empirical model budget/numerical condition failed; no scientific retry')
        result['scope']['nominal_axis_pairs']=2
        if any(r['status']!='NOT_EVALUATED' for r in evaluation['records'] if r['request_id'] not in {v['request_id'] for v in means}):
            raise ValueError('Outside S0 science scope')
        handoff=P.handoff_supplied_results(plan,config,schedule,result['model_records'],source_ref,root=root)
        save(out/'N_HANDOFF.json',handoff)
        if Path(spec['path']).read_bytes()!=context_data:raise ValueError('Consumed H4 context changed during S0')
        provenance['context_unchanged_after_execution']=True
        save(out/'SOURCE_INDEX.json',provenance)
        result['status']='S0_COMPLETE_NOMINAL_UNQUALIFIED';save(out/'result.json',result)
        return result
    except Exception as exc:
        result['status']='STOPPED_S0';result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
        save(out/'result.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    result=execute(ROOT,args.out,source)
    print(json.dumps(dict(status=result['status'],scientific_source=source,conditions=result.get('conditions'),
        scope=result['scope'],rounds=[dict(m=r['identity']['m'],status=r['status'],nominal_N_X=r['nominal_N_X'],nominal_N_Y=r['nominal_N_Y']) for r in result['model_records']],errors=result['errors']),ensure_ascii=False))
    return 0 if result['status']=='S0_COMPLETE_NOMINAL_UNQUALIFIED' else 1


if __name__=='__main__':raise SystemExit(main())
