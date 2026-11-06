"""Conditional all-round accounting, retaining unknown long-q discrepancy."""
from __future__ import annotations
import json
import math
import numpy as np
from .track_c_h4 import *
from .track_c_h4_cs_price import cs_precision,rz_bound
from .track_c_h4_finite_bits import proposal_budget,integer_cdf,weight_interval
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
from .track_c_h4_price import strong_correction


def total_cost(shots,prices):
    terms=[math.log(n)+math.log(float(c)) for n,c in zip(shots,np.asarray(prices).reshape(-1)) if c>0]
    if not terms:return dict(value=0.,log10=None)
    top=max(terms);log=(top+math.log(math.fsum(math.exp(x-top) for x in terms)))/math.log(10)
    return dict(value=10**log if log<300 else None,log10=log)


def paired_price_difference(task_a,task_b,r,q,strong_b=False):
    """Paired full-minus-baseline price diagnostic, retaining X/Y covariance."""
    mean=np.zeros((2,4));cov=np.zeros((8,8))
    for rare in ((False,) if task_a['ld']==12 else (False,True)):
        def read(task):
            key=f"{task['name']}_r{r}_q{q}_rare{int(rare)}"
            return json.loads((OUT/'prices'/f'{key}.json').read_text())
        a=read(task_a);b=read(task_b)
        av=np.asarray(a['full_samples'],float)
        bv=np.asarray(b['strong_additive_samples' if strong_b and r else 'full_samples'],float)
        if strong_b and r:
            bv=bv+np.array(json.loads((OUT/'strong_D_runs'/f"{task_b['name']}_CORRECTION.json").read_text())[str(q)])[None,:,:]
        assert a['probability']==b['probability'] and av.shape==bv.shape
        delta=av-bv;mass=a['probability'];mean+=mass*delta.mean(axis=0)
        if len(delta)>1:cov+=mass**2*np.cov(delta.reshape(len(delta),8),rowvar=False,ddof=1)/len(delta)
    return mean,cov


def paired_same_shots(task_a,task_b,reference,strong_b=False):
    """Isolate prices with a common reference allocation, not a new selector."""
    shots=np.asarray(reference['selected_allocation']['shots'],float).reshape(18,2)
    total=np.zeros(4);variance=np.zeros(4);holdouts=[]
    for r in ([0] if task_a['ld']==12 else sorted(set(RVALUES))):
        one,v1=paired_price_difference(task_a,task_b,r,1,strong_b)
        two,v2=paired_price_difference(task_a,task_b,r,2,strong_b)
        eight,v8=paired_price_difference(task_a,task_b,r,8,strong_b)
        prediction=7*two-6*one
        holdouts.append(dict(r=r,actual_difference=eight.tolist(),predicted_difference=prediction.tolist(),
            MC_standard_error=np.sqrt(np.diag(v8).reshape(2,4)).tolist(),
            prediction_minus_actual_MC_standard_error=np.sqrt(np.diag(36*v1+49*v2+v8).reshape(2,4)).tolist()))
        for j in range(4):
            a=np.zeros(8);b=np.zeros(8)
            for row,ns in zip(reference['accounted_rows'],shots):
                if (0 if task_a['ld']==12 else row['r'])!=r:continue
                for axis in range(2):
                    a[axis*4+j]+=ns[axis]*(2-row['q'])
                    b[axis*4+j]+=ns[axis]*(row['q']-1)
            total[j]+=a@one.reshape(8)+b@two.reshape(8)
            variance[j]+=a@v1@a+b@v2@b
    return dict(resources=list(METRICS),full_minus_baseline_G_same_shots=total.tolist(),
        calibration_MC_standard_error=np.sqrt(np.maximum(0.,variance)).tolist(),
        reference_allocation=reference['task']['name']+':'+reference['selected_allocation']['policy'],
        holdouts=holdouts,long_q_model_error_bound=None,
        scope='paired MC-only diagnostic under the affine model; no total or simultaneous interval')


def price_moments(task,r,q,model='full',exponent=None,rres=None):
    is_t=task['pilot']=='C-S'
    metrics=['T','CX','depth','Clifford'] if is_t else list(METRICS)
    result=np.zeros((2,4));covariance=np.zeros((8,8));prob_sum=0.;paths=[]
    for rare in ((False,) if task.get('ld')==12 else (False,True)):
        if is_t:
            key=f"CS_{task['arm']}_eps{exponent}_r{r}_res{rres}_q{q}_rare{int(rare)}"
            path=OUT/'prices_T'/f'{key}.json'
        else:
            key=f"{task['name']}_r{r}_q{q}_rare{int(rare)}";path=OUT/'prices'/f'{key}.json'
        data=json.loads(path.read_text());paths.append(str(path.relative_to(OUT)))
        if is_t:
            probability=data['outer_probability']
            samples=np.array([[[sum(p['probability_given_original']*p[metric] for p in sample[axis])
                               for metric in metrics] for axis in range(2)] for sample in data['samples']])
        else:
            probability=data['probability'];name='full_samples' if model=='full' or task.get('ld')==12 else 'strong_additive_samples'
            samples=np.asarray(data[name],float)
            if model!='full' and task.get('ld')==3:
                correction_path=OUT/'strong_D_runs'/f"{task['name']}_CORRECTION.json"
                correction=np.array(json.loads(correction_path.read_text())[str(q)])
                samples=samples+correction[None,:,:]
        result+=probability*samples.mean(axis=0);prob_sum+=probability
        if len(samples)>1:
            flat=samples.reshape(len(samples),8)
            covariance+=probability**2*np.cov(flat,rowvar=False,ddof=1)/len(samples)
    assert abs(prob_sum-1)<1e-12,prob_sum
    return result,covariance,paths,metrics


def account_task(task,model='full'):
    data=json.loads((OUT/'signals'/f"{task['name']}.json").read_text())
    if not data['all_rounds_feasible']:return dict(task=task,status='INFEASIBLE_WITH_REGISTERED_POLICY')
    rows=[dict(row) for row in data['rows']];is_t=task['pilot']=='C-S';exponent=cs_precision(data) if is_t else None
    if task.get('ld')==12:
        for row in rows:
            row['registered_r_m']=row['r'];row['r']=0
            row['tail_order']=0
            row['metadata_note']='tail-free Full: raw shared-r label is ignored by its r=0 recipe'
    if is_t:
        base=json.loads((OUT/'signals/CN_S2_h0.02_ld3.json').read_text())['rows']
        for row,b in zip(rows,base):
            lam=common()['preps'][3].exact_rte_lambda_r
            original=finite_rte_distribution(lam*.02/row['r'],2)
            factors=[(original.exact_finite_distribution,row['q']*row['r'])]
            rr=row['r_res']
            if rr:
                for item in d_residual_data():
                    for theta in item['angles']:
                        delta=theta-round(theta/(math.pi/4))*math.pi/4
                        norm=finite_rte_distribution(delta/(2*rr),2).exact_finite_distribution
                        factors.append((norm,2*row['q']*rr))
                if task['arm']=='S1':
                    maximum=1.
                    for order in (0,2):
                        theta=2*math.atan(lam*.02/row['r']/(order+1))
                        delta=theta-round(theta/(math.pi/4))*math.pi/4
                        maximum=max(maximum,finite_rte_distribution(delta/(2*rr),2).exact_finite_distribution)
                    factors.append((maximum,row['q']*row['r']*rr))
            _,upper=weight_interval(factors,max(100,math.ceil(row['log_range']/math.log(10))+60))
            row['log_range_float64_diagnostic']=row['log_range']
            row['log_range']=upper;row['range']=math.exp(upper) if upper<700 else None
            row['range_bound_source']='outward interval compressed product of exact-binary normalizers'
            # Credit the unused cap using the assigned guarded epsilon and
            # full literal count/range bound, never a maximum over sampled keys.
            disk=math.exp(math.log(2*math.sqrt(2))+row['log_range']+
                          math.log(rz_bound(row['q'],row['r'],task['arm']))+exponent*math.log(10))
            center_lower=row['radius_corrected']-b['numerical_disk']-b['input_disk']
            beta=math.asin(disk/center_lower)
            assert beta<=task['cap'],(task,row['m'],beta)
            row['beta_stat']+=row['beta_synthesis']-beta
            row['beta_synthesis_registered_cap']=row['beta_synthesis']
            row['beta_synthesis']=beta;row['synthesis_disk_assigned_upper']=disk
            row['radius_lower']=center_lower-disk
    components=common()['preps'][3].rte_preparation.symbolic_tail.components
    component_count=len(components)
    for row in rows:
        if is_t:
            d_rotations=sum(len(x['angles']) for x in d_residual_data())
            draws=4*row['q']*row['r']+row['q']*(2*d_rotations+row['r'])*row['r_res']
            log_range=row['log_range']
        elif task['ld']==12:draws=0;log_range=0.
        else:draws=4*row['steps']*sum(x['r'] for x in row['recipe']);log_range=0.
        probabilities=[x.probability for x in components if x.probability>0]
        lam=common()['preps'][3].exact_rte_lambda_r
        recipes=[dict(time=.02,r=row['r'])] if is_t else row['recipe']
        for recipe in recipes:
            if recipe['r']:
                probabilities.extend(p for p in finite_rte_distribution(lam*recipe['time']/recipe['r'],2).order_probabilities if p>0)
        if is_t and row['r_res']:
            angles=[t for item in d_residual_data() for t in item['angles']]
            if task['arm']=='S1':
                angles.extend(2*math.atan(lam*.02/row['r']/(order+1)) for order in (0,2))
            for theta in angles:
                delta=theta-round(theta/(math.pi/4))*math.pi/4
                probabilities.extend(p for p in finite_rte_distribution(delta/(2*row['r_res']),2).order_probabilities if p>0)
        budget=proposal_budget(log_range,draws,component_count,min(probabilities))
        # Verify the component CDF for this precision without generating shots.
        integer_cdf([x.probability for x in components],budget['bits_per_uniform_draw'])
        if draws==0:budget['signal_error_upper']=0.
        beta=math.asin(budget['signal_error_upper']/row['radius_lower'])
        row['beta_proposal_rounding']=beta;row['finite_proposal']=budget
        row['beta_stat']-=beta;row['radius_lower']-=budget['signal_error_upper']
    fits={};prices=[];sources=[];holdouts=[]
    for row in rows:
        r=0 if task.get('ld')==12 else row['r'];rr=row.get('r_res');recipe=(r,rr)
        if recipe not in fits:
            one,v1,p1,metrics=price_moments(task,r,1,model,exponent,rr)
            two,v2,p2,_=price_moments(task,r,2,model,exponent,rr)
            eight,v8,p8,_=price_moments(task,r,8,model,exponent,rr)
            prediction=7*two-6*one
            error=abs(prediction-eight)/np.maximum(abs(eight),1.)
            holdouts.append(dict(r=r,r_res=rr,actual=eight.tolist(),prediction=prediction.tolist(),
                relative_point_error=error.tolist(),max_relative_point_error=float(error.max()),
                predicted_relative_95_halfwidth=(1.96*np.sqrt(np.diag(36*v1+49*v2).reshape(2,4))/np.maximum(abs(prediction),1.)).tolist(),
                status='LOCAL_DIAGNOSTIC_NOT_LONG_Q_CERTIFICATE'))
            fits[recipe]=(one,two,v1,v2);sources.extend(p1+p2+p8)
        one,two,_,_=fits[recipe]
        prediction=(2-row['q'])*one+(row['q']-1)*two
        if np.any(prediction<0):return dict(task=task,status='UNQUALIFIED_NEGATIVE_AFFINE_PREDICTION')
        prices.append(prediction)
    p=np.array(prices)
    candidates=shot_rows(rows,p[:,:,0])
    plan=min(candidates,key=lambda x:x['cost_log10'])
    n=plan['shots'];totals={metric:total_cost(n,p[:,:,j]) for j,metric in enumerate(metrics)}
    # Calibration is reused across rounds. Accumulate its coefficient vectors
    # before propagating covariance, instead of treating each round as fresh.
    uncertainty={}
    if max(math.log10(x) for x in n)<290:
        shotarray=np.asarray(n,float).reshape(18,2)
        for metric_idx,metric in enumerate(metrics):
            variance=0.
            for recipe,(_,_,v1,v2) in fits.items():
                a=np.zeros(8);b=np.zeros(8)
                for row,ns in zip(rows,shotarray):
                    r=0 if task.get('ld')==12 else row['r']
                    if (r,row.get('r_res'))!=recipe:continue
                    for axis in range(2):
                        a[axis*4+metric_idx]+=ns[axis]*(2-row['q'])
                        b[axis*4+metric_idx]+=ns[axis]*(row['q']-1)
                variance+=float(a@v1@a+b@v2@b)
            uncertainty[metric]=dict(calibration_MC_standard_error=math.sqrt(max(0.,variance)),
                long_q_model_error_bound=None,scope='conditional statistical diagnostic; not simultaneous or total uncertainty')
    else:
        uncertainty={metric:dict(calibration_MC_standard_error=None,long_q_model_error_bound=None,
             scope='large-range log-cost point model; no finite total uncertainty claimed') for metric in metrics}
    phases=[float(np.angle(complex(row['corrected']['real'],row['corrected']['imag']))) for row in rows]
    reconstruction=reconstruct_rpe_phase(phases,[row['q'] for row in rows])
    energy=-reconstruction['final_phase_estimate']/.02
    reconstruction.update(mean_signal_energy=energy,reference_energy=common()['energy'],
        mean_energy_error=abs(energy-common()['energy']),
        nominal_energy_width=.4/(2**17*.02),
        E0_used_by_decoder=False,scope='finite mean center; native and finite-bit errors retained as separate disks')
    return dict(task=task,model=model,status='COMPLETE_CONDITIONAL_NUMERICAL_PILOT',
        target=dict(epsilon_E=1.5936e-4,alpha_total=.05,beta_RPE=.4,delta0=.02,rounds=18),
        resource_names=metrics,prices_calibration_affine=p.tolist(),
        selected_allocation=plan,allocation_candidates=candidates,G=totals,
        uncertainty=uncertainty,holdouts=holdouts,price_sources=sorted(set(sources)),
        selection_information='exact reference signal used for numerical feasibility: oracle diagnostic, not an implementable selector',
        long_q_status='CONDITIONAL_AFFINE_MODEL_WITH_UNBOUNDED_MODEL_DISCREPANCY',
        sufficient_success='conditional on signal/error/range bounds, external alias and fresh IID shots',
        strict_synthesis_exponent=exponent,accounted_rows=rows,branch_reconstruction=reconstruction,
        strong_baseline_policy='maximal D runs between R occurrences; tail-free Full receives its actual full-wrapper price')


def aggregate():
    qualification=OUT/'NATIVE_QUALIFICATION.json'
    if qualification.exists():
        status=json.loads(qualification.read_text())
        if not status['comparison_eligible']:
            result=dict(status=status['status'],RPE_total_G=None,
                reason=status['failure'],C_N='UNDETERMINED',
                C_S='SOURCE_MEAN_RANGE_NEGATIVE_MATERIAL; NATIVE_COMPARISON_UNQUALIFIED',
                C_P='SOURCE_SIGNAL_NO_RESOLVED_PROCESSING_GAIN; NATIVE_COMPARISON_UNQUALIFIED',
                native_success_at_target='NOT_ESTABLISHED',
                completed_price_cells_Q=len(list((OUT/'prices').glob('*.json'))),
                completed_price_cells_T=len(list((OUT/'prices_T').glob('*.json'))),
                planned_price_cells_Q=306,planned_price_cells_T=108,
                scope='qualification failure; partial prices are preserved, not zero-filled or ranked',
                mandatory_STOP_after_publication=True)
            save_json(OUT/'SUMMARY.json',result)
            save_json(OUT/'ACCOUNTING_STATUS.json',dict(status=status['status'],RPE_total_G='UNQUALIFIED'))
            return result
    missing=[];answers={}
    for task in signal_tasks()+cs_tasks():
        for model in (('full','strong_additive') if task['pilot']=='C-N' else ('full',)):
            key=task['name']+':'+model
            try:
                answer=account_task(task,model);answers[key]=answer
                save_json(OUT/'accounting'/f'{key.replace(":","_")}.json',answer)
            except FileNotFoundError as e:missing.append(str(e.filename))
    save_json(OUT/'ACCOUNTING_STATUS.json',dict(completed=len(answers),missing=sorted(set(missing))))
    if missing:return dict(completed=len(answers),missing=len(set(missing)))
    cn=[t for t in signal_tasks() if t['pilot']=='C-N']
    best={model:min(cn,key=lambda t:answers[t['name']+':'+model]['G']['rz']['log10'])
          for model in ('full','strong_additive')}
    cs=[answers[t['name']+':full'] for t in cs_tasks()]
    cp=[answers[f'CP_P{i}:full'] for i in (0,1)]
    paired={t['name']:paired_same_shots(t,t,answers[t['name']+':full'],True) for t in cn}
    paired['CP_P1_minus_P0']=paired_same_shots(cp[1]['task'],cp[0]['task'],cp[0])
    save_json(OUT/'PAIRED_COMPARISONS.json',paired)
    comparison=dict(
        C_N=dict(full_selected=best['full']['name'],strong_additive_selected=best['strong_additive']['name'],
             selection_changed=best['full']['name']!=best['strong_additive']['name'],
             scope='finite-set oracle diagnostic; calibration selection, independent q8 holdout'),
        C_S=[dict(name=x['task']['name'],T_log10=x['G']['T']['log10'],
                 terminal_shots_log10=x['selected_allocation']['shots_log10'][-1]) for x in cs],
        C_P=dict(P0_RZ=cp[0]['G']['rz'],P1_RZ=cp[1]['G']['rz'],
                 conditional_ratio=10**(cp[1]['G']['rz']['log10']-cp[0]['G']['rz']['log10'])),
        unconditional_total_cost_superiority='NOT_ESTABLISHED: long-q price model discrepancy remains unbounded',
        mandatory_STOP_after_publication=True)
    save_json(OUT/'SUMMARY.json',comparison)
    return comparison
