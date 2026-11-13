"""Finite-shot inputs -> unchanged RPE decoder -> explicitly scoped loss.

No physical means, N or resource prices are computed here. Real-data calls to
simulate/exact_counts/evaluate_trials are scientific analyses requiring their
own scope. Metadata preparation and imports never execute them.
"""
from copy import deepcopy
from fractions import Fraction
import itertools
import math

from . import pr_rpe_rmse as A
from . import pr_rpe_rmse_controller as B

p = B.p
DECODER = 'consecutive_dyadic_nearest_branch_principal_energy_v1'
COUNT_RULE = 'EXACT_SIGNED_INTEGER_SUM_THEN_BINARY64_RATIO_V1'
IID = 'IID_SHOTS_INDEPENDENT_AXES_ROUNDS_AND_REPETITIONS'
DECODED = 'DECODED_SUPPLIED_COORDINATES_NOT_RMSE_QUALIFIED'
SOURCE = dict(commit='3216d27a69ac8532d2a22e78f5f94491d202d238',
    wrapper='src/trottertracks/pr_rpe_rmse.py:decode_supplied_coordinates',
    scalar_decoder='src/trotterlib/rpe_four_round_phase_validation.py:reconstruct_rpe_phase',
    statistics='src/trotterlib/rpe_hadamard_failure_validation.py')


def _finite(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(name + ': finite numeric scalar required')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ': exact integer required')
    return value


def count_mean(plus, shots):
    """Exact integer aggregation; no float cast of large N before subtraction."""
    _integer(shots,'shots',1);_integer(plus,'plus')
    if plus > shots:
        raise ValueError('plus count exceeds shots')
    exact = Fraction(2*plus-shots, shots)
    return dict(plus=plus,minus=shots-plus,shots=shots,
        exact_mean=B.a.exact_value(exact),mean=float(exact),conversion=COUNT_RULE)


def outcomes_count(outcomes):
    """Real +/-1 values or synthetic outcomes; bools/missing are not outcomes."""
    values=list(outcomes)
    if not values or any(type(v) is not int or v not in (-1,1) for v in values):
        raise ValueError('Nonempty exact +/-1 outcomes required')
    return count_mean(sum(v==1 for v in values),len(values))


def decode_counts(rounds, *, h, M, prior, decoder=DECODER, conversion=COUNT_RULE):
    """No prior clipping, energy shift, fallback or branch-rule changes.

    Prior membership is diagnostic: the existing A wrapper does not use prior
    to select aliases. Missing axes are NOT_EVALUATED, not decoder failures.
    """
    _integer(M,'M')
    if decoder != DECODER or conversion != COUNT_RULE:
        raise ValueError('Explicit supported original decoder/count conversion required')
    if isinstance(h,bool) or not math.isfinite(float(h)) or float(h)<=0:
        raise ValueError('Positive finite h required')
    if len(prior)!=2 or any(not math.isfinite(float(v)) for v in prior) or prior[0]>=prior[1]:
        raise ValueError('Ordered finite energy prior required')
    by_m={r['m']:r for r in rounds}
    if len(by_m)!=len(rounds) or any(type(m) is not int or m not in range(M+1) for m in by_m):
        raise ValueError('Duplicate or foreign measurement rounds')
    missing=[dict(m=m,axis=axis) for m in range(M+1) for axis in p.AXES
        if by_m.get(m,{}).get(axis) is None]
    if missing:
        return dict(status='NOT_EVALUATED_MISSING_COUNTS',missing=missing,energy_estimate_ha=None,
            prior=deepcopy(prior),prior_applied_to_decoder=False)
    coordinates=[];counts=[]
    for m in range(M+1):
        row={axis:count_mean(**by_m[m][axis]) for axis in p.AXES}
        coordinates.append((row['X']['mean'],row['Y']['mean']));counts.append(dict(m=m,**row))
    # Direct call to the existing A wrapper and scalar decoder. No new decoder.
    result=A.decode_supplied_coordinates(coordinates,h=float(h))
    E=result['energy_estimate_ha']
    result.update(counts=counts,count_conversion=conversion,decoder_source=deepcopy(SOURCE),
        prior=deepcopy(prior),prior_applied_to_decoder=False,
        estimate_in_supplied_prior=None if E is None else prior[0]<=E<=prior[1],
        output_range='(-pi/h,pi/h] nominal principal-phase range; binary64 not a certified real enclosure',
        alias_qualified=False,unresolved_loss=None)
    return result


def mean_from_a(config, m, physical_record, *, source_request_id, law_binding, source, target, qualification, uncertainty):
    """Copy A's supplied raw/corrected/logB, matching physical and canonical IDs.

    Does not evaluate/normalize a signal or check a numerical error gate.
    """
    B._validate(config);_integer(m,'m')
    if m>config['spec']['M']:raise ValueError('Mean round outside the configuration')
    row=config['rounds'][m];identity=physical_record['identity']
    if config['spec'].get('schedule_adapter')!='A_FIXED_GRID_510FBACD_V1':
        raise ValueError('This minimal physical adapter covers the pinned A schedule only')
    expected=dict(candidate=row['physical_candidate'],structure=row['structure'],
        binding=config['spec']['A_input']['design']['binding'],profile=config['spec']['profile'])
    if (physical_record.get('request_id')!=source_request_id or not source_request_id
            or identity.get('m')!=m or identity.get('q_RPE')!=row['q_RPE'] or identity.get('q_PF')!=row['q_PF']
            or any(not p.same(identity.get(k),v) for k,v in expected.items())
            or not p.same(law_binding,row['round_law_definition']['A_declaration']) or not source):
        raise ValueError('A physical mean identity/law/SOURCE mismatch')
    signal=physical_record.get('signal') or {}
    return dict(m=m,identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),source=deepcopy(source),
        raw=deepcopy(signal.get('raw')),corrected=deepcopy(signal.get('corrected')),
        normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_A',log_B=deepcopy(signal.get('log_B'))),
        bias={k:deepcopy(signal.get(k)) for k in ('total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')},
        target=deepcopy(target),qualification=qualification,uncertainty=deepcopy(uncertainty))


def prepare(config, means, shots, *, target, shot_policy):
    """Metadata readiness; never calculates means, shots, RMSE or G.

    Missing physical qualification does not become zero error. Simulation on
    complete unqualified means can only be a declared conditional diagnostic.
    """
    B._validate(config)
    if not shot_policy.get('source') or shot_policy.get('meaning') not in ('NOMINAL','USER_SUPPLIED','SUFFICIENT_EXTERNALLY_QUALIFIED'):
        raise ValueError('Explicit shot allocation source/meaning required')
    expected_target=dict(binding=config['spec']['binding'],energy_shift_ha=0)
    if (not p.same(target.get('binding'),expected_target['binding']) or target.get('energy_shift_ha')!=0
            or not target.get('source') or target.get('quantity')!='ORIGINAL_DF_H_TARGET_ENERGY'):
        raise ValueError('Original H/state target SOURCE without energy shift required')
    def index(values):
        result={v['m']:v for v in values}
        if len(result)!=len(values) or any(type(m) is not int or m not in range(config['spec']['M']+1) for m in result):
            raise ValueError('Duplicate or foreign mean/shot rounds')
        return result
    ms,ns=index(means),index(shots);rows=[];blockers=[]
    for request in config['rounds']:
        m=request['m'];mean=ms.get(m);n=ns.get(m);issues=[]
        if mean is not None:
            for key,value in dict(identity=request['identity'],law_binding=request['law_binding'],target=target).items():
                if not p.same(mean.get(key),value):
                    raise ValueError('Mean full V2 identity/law/target mismatch: '+key)
            if not mean.get('source'):
                raise ValueError('Physical mean numerical SOURCE required')
        if mean is None or mean.get('raw') is None or any(mean['raw'].get(k) is None for k in ('real','imag')):issues.append('MISSING_RAW_MEAN')
        if mean is None or mean.get('corrected') is None:issues.append('MISSING_CORRECTED_MEAN_METADATA')
        if mean is None or not mean.get('normalization') or mean['normalization'].get('log_B') is None:issues.append('MISSING_FINITE_NORMALIZATION_METADATA')
        if mean is None or mean.get('qualification')!='QUALIFIED':issues.append('PHYSICAL_MEAN_UNQUALIFIED')
        if mean is None or not mean.get('uncertainty'):issues.append('UNKNOWN_PHYSICAL_ERROR_CONTRACT')
        for axis in p.AXES:
            value=(n or {}).get('N_'+axis)
            if value is None:issues.append('MISSING_N_'+axis)
            else:_integer(value,'N_'+axis,1)
        rows.append(dict(m=m,identity=deepcopy(request['identity']),law_binding=deepcopy(request['law_binding']),
            physical_candidate=deepcopy(request.get('physical_candidate',request['candidate'])),
            mean=deepcopy(mean),shots=deepcopy(n),issues=issues))
        blockers.extend(issues)
    contract=config['spec']['rmse_policy']['error_contract']
    return dict(configuration_id=config['id'],M=config['spec']['M'],h=deepcopy(config['spec']['h']),
        decoder=contract.get('decoder'),prior=deepcopy(contract.get('energy_prior_ha')),
        target=deepcopy(target),shot_policy=deepcopy(shot_policy),rows=rows,
        input_complete=not any(k.startswith('MISSING_') for k in blockers),
        physical_qualification_complete=not any(k in ('PHYSICAL_MEAN_UNQUALIFIED','UNKNOWN_PHYSICAL_ERROR_CONTRACT') for k in blockers),
        blockers=sorted(set(blockers)),systematic_errors_implicitly_zero=False,
        global_RMSE_bound=None,sufficient_N=None,G=None,statistics_executed=False)


def iid_policy(policy):
    if policy.get('kind')!=IID or not policy.get('source') or not policy.get('target_law'):
        raise ValueError('Explicit IID shots/axes/rounds/repetitions and target law required')


def simulate_prepared(prepared, *, repetitions, seed, policy, allow_conditional_unqualified=False, batch_decoder=None):
    """Thin A/V2 adapter; optional batch hook preserves the existing input gates."""
    if not prepared['input_complete']:
        return dict(status='NOT_EVALUATED_MISSING_STATISTICAL_INPUT',blockers=deepcopy(prepared['blockers']),
            trials=None,RMSE=None,sufficient_N=None,G=None)
    if not prepared['physical_qualification_complete'] and not allow_conditional_unqualified:
        return dict(status='NOT_EVALUATED_UNQUALIFIED_PHYSICAL_MEANS',blockers=deepcopy(prepared['blockers']),
            trials=None,RMSE=None,sufficient_N=None,G=None)
    if prepared['decoder']!=DECODER or prepared['prior'] is None:
        raise ValueError('Fixed decoder/prior SOURCE required')
    if any(row['law_binding']['definition']['model']!=policy.get('target_law') for row in prepared['rows']):
        raise ValueError('Supplied mean law and simulation target law differ')
    rows=[dict(m=r['m'],raw=r['mean']['raw'],N_X=r['shots']['N_X'],N_Y=r['shots']['N_Y']) for r in prepared['rows']]
    result=simulate(rows,M=prepared['M'],h=float(B.a.exact_scalar(prepared['h'])),prior=prepared['prior'],
        repetitions=repetitions,seed=seed,policy=policy,batch_decoder=batch_decoder)
    result.update(configuration_id=prepared['configuration_id'],target=deepcopy(prepared['target']),
        physical_qualification_complete=prepared['physical_qualification_complete'],
        physical_error_evidence=[dict(m=r['m'],source=r['mean']['source'],bias=r['mean'].get('bias'),uncertainty=r['mean'].get('uncertainty')) for r in prepared['rows']],
        sufficient_N=None,global_RMSE_bound=None,G=None)
    return result


def simulate(rounds, *, M, h, prior, repetitions, seed, policy, batch_decoder=None):
    """Conditional binomial Monte Carlo on supplied raw coordinate means.

    No RTE trajectory sampling: each +/-1 draw is a marginal Hadamard shot.
    Independent binomials require the explicit joint-law policy. Does not infer
    actual-sampler equality, state accuracy, or sufficient quantum N.

    batch_decoder=None keeps the original scalar path. A supplied callable
    receives the same completed count arrays and exact count_mean function;
    rpe_decoder_fast.decode_sampled_counts records acceleration provenance.
    """
    import numpy as np
    iid_policy(policy);_integer(repetitions,'repetitions',1);_integer(seed,'seed')
    _integer(M,'M')
    if [r['m'] for r in rounds]!=list(range(M+1)):
        raise ValueError('All consecutive rounds required; no missing-mean simulation')
    rng=np.random.Generator(np.random.PCG64(seed));samples=[]
    for row in rounds:
        axes={}
        for axis,component in (('X','real'),('Y','imag')):
            n=_integer(row['N_'+axis],'N_'+axis,1)
            if n>int(np.iinfo(np.int64).max):
                raise ValueError('NumPy binomial int64 domain exceeded; N was not rounded')
            mean=_finite(row['raw'][component],'raw coordinate')
            if not -1<=mean<=1:raise ValueError('Raw +/-1 mean outside [-1,1]; no clipping')
            axes[axis]=rng.binomial(n,(1.+mean)/2.,size=repetitions)
        samples.append(axes)
    if batch_decoder is None:
        results=[]
        for i in range(repetitions):
            counts=[dict(m=r['m'],**{axis:dict(plus=int(samples[m][axis][i]),shots=r['N_'+axis]) for axis in p.AXES}) for m,r in enumerate(rounds)]
            result=decode_counts(counts,h=h,M=M,prior=prior)
            # Keep statuses/energies/branch diagnostics without retaining all counts.
            results.append({k:v for k,v in result.items() if k not in ('counts',)})
    else:
        results=batch_decoder(samples,rounds,M=M,h=h,prior=prior,
            count_mean=count_mean,decoder_source=SOURCE)
    return dict(trials=results,seed=seed,repetitions=repetitions,policy=deepcopy(policy),
        input_raw_means=deepcopy(rounds),numeric_distribution='NUMPY_PCG64_BINOMIAL_BINARY64_P; not exact-law arithmetic',
        simulation_is_sufficient_N_proof=False)


def exact_counts(rounds, *, h, M, prior, policy, maximum_states):
    """Exhaustive SMALL supplied count distribution, floating binomial weights.

    maximum_states is the caller's finite experiment scope, not a resource cap.
    This is not a certified numerical enclosure or a scalable 13-round solver.
    """
    from scipy.stats import binom
    iid_policy(policy);_integer(maximum_states,'maximum_states',1)
    if [r['m'] for r in rounds]!=list(range(M+1)):raise ValueError('Complete count distribution required')
    shape=[];probs=[]
    for row in rounds:
        for axis,component in (('X','real'),('Y','imag')):
            n=_integer(row['N_'+axis],'N_'+axis,1);mean=_finite(row['raw'][component],'raw coordinate')
            if not -1<=mean<=1:raise ValueError('Raw coordinate outside [-1,1]')
            shape.append(n+1);probs.append((n,(1.+mean)/2.))
    states=math.prod(shape)
    if states>maximum_states:raise ValueError('Exact enumeration exceeds supplied finite scope')
    axes=[[float(binom.pmf(k,n,pr)) for k in range(n+1)] for n,pr in probs]
    results=[]
    for ks in itertools.product(*(range(n) for n in shape)):
        weight=math.prod(axes[j][k] for j,k in enumerate(ks))
        if weight==0:continue
        rows=[dict(m=m,X=dict(plus=ks[2*m],shots=probs[2*m][0]),Y=dict(plus=ks[2*m+1],shots=probs[2*m+1][0])) for m in range(M+1)]
        results.append(dict(probability=weight,**decode_counts(rows,h=h,M=M,prior=prior)))
    return dict(trials=results,probability_mass=math.fsum(r['probability'] for r in results),count_states=states,
        arithmetic='SCIPY_BINOM_PMF_BINARY64_NOT_CERTIFIED_EXACT_PROBABILITIES')


def summarize(trials, *, target_energy_ha, unresolved_loss, iid_repetitions=False, confidence=None, loss_support=None):
    """Never drop unresolved trials or substitute an energy estimate.

    UNDEFINED -> all-trial RMSE is null when any observed trial is unresolved.
    BOUNDED_INTERVAL supplies *loss* bounds, not a decoder fallback. Only risk
    intervals are then reported; a particular failure loss is not adopted.
    """
    _finite(target_energy_ha,'target energy')
    if not trials:raise ValueError('Nonempty ensemble required')
    if any(r['status'].startswith('NOT_EVALUATED') for r in trials):
        raise ValueError('Missing measurements are not an unresolved decoder ensemble')
    kind=unresolved_loss.get('kind')
    if kind not in ('UNDEFINED','BOUNDED_INTERVAL') or not unresolved_loss.get('source'):
        raise ValueError('Explicit unresolved-trial loss policy/source required')
    fail_lo=fail_hi=None
    if kind=='BOUNDED_INTERVAL':
        fail_lo=_finite(unresolved_loss['lower'],'failure loss lower');fail_hi=_finite(unresolved_loss['upper'],'failure loss upper')
        if not 0<=fail_lo<=fail_hi:raise ValueError('Nonnegative ordered failure loss interval required')
    weighted=any('probability' in r for r in trials)
    if weighted and not all('probability' in r for r in trials):raise ValueError('Mixed frequency/probability ensemble')
    weights=[_finite(r['probability'],'probability') if weighted else 1./len(trials) for r in trials]
    if any(w<0 for w in weights) or not math.isclose(math.fsum(weights),1.,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Unit mass ensemble required; no renormalization of lost trials')
    resolved=[];failed=[];losses=[];statuses={}
    for row,w in zip(trials,weights):
        status=row['status'];statuses[status]=statuses.get(status,0)+1
        if status==DECODED:
            energy=_finite(row['energy_estimate_ha'],'decoded energy')
            loss=(energy-target_energy_ha)**2
            if not math.isfinite(loss):raise ValueError('Squared loss overflow')
            resolved.append(w);losses.append(w*loss)
        elif status.startswith('UNRESOLVED_') and row.get('energy_estimate_ha') is None:
            failed.append(w)
        else:raise ValueError('Unknown decoder output; no success coercion')
    prob_fail=math.fsum(failed);prob_ok=math.fsum(resolved);contribution=math.fsum(losses)
    endpoints=None if kind=='UNDEFINED' and prob_fail else [contribution+prob_fail*(fail_lo or 0),contribution+prob_fail*(fail_hi or 0)]
    out=dict(ensemble='ENUMERATED_BINOMIAL_WEIGHTS' if weighted else 'MONTE_CARLO_OR_SUPPLIED_TRIALS',trials=len(trials),status_counts=statuses,
        unresolved_count=sum(r['status']!=DECODED for r in trials),unresolved_probability=prob_fail,
        estimate_outside_prior_count=sum(r.get('estimate_in_supplied_prior') is False for r in trials),
        branch_error_probability=None,prior_alias_qualification=False,
        unresolved_output_energy=None,unresolved_loss_policy=deepcopy(unresolved_loss),
        resolved_MSE_contribution=contribution,conditional_on_decoded_RMSE=None if not prob_ok else math.sqrt(contribution/prob_ok),
        all_observed_trials_RMSE=math.sqrt(contribution) if prob_fail==0 else None,
        all_trial_MSE_interval=endpoints,all_trial_RMSE_interval=None if endpoints is None else [math.sqrt(v) for v in endpoints],
        population_RMSE_CI=None,unresolved_probability_CI=None,conditional_on_physical_means_only=True,
        sufficient_N=None,mathematical_global_RMSE_bound=None,physical_systematic_errors=None,G=None)
    if confidence is not None:
        if weighted or not iid_repetitions or not 0<confidence<1:
            raise ValueError('CI requires independent repeated trials and explicit confidence')
        old=p.accounting.existing._legacy('rpe_hadamard_failure_validation')
        count=out['unresolved_count'];n=len(trials)
        out['unresolved_probability_CI']=list(old._clopper_pearson_interval(count,n,confidence=confidence))
        # No population risk when failure loss is undefined, even for zero
        # observed failures: absence in this sample is not zero population mass.
        if kind=='BOUNDED_INTERVAL' and loss_support is not None:
            upper=_finite(loss_support['upper'],'squared-loss support upper')
            if not loss_support.get('source') or upper<=0 or fail_hi>upper:
                raise ValueError('Explicit uniform squared-loss bound SOURCE required')
            for row in trials:
                if row['status']==DECODED and (row['energy_estimate_ha']-target_energy_ha)**2>upper:
                    raise ValueError('Observed loss exceeds supplied support; no clipping')
            eps=upper*math.sqrt(math.log(2/(1-confidence))/(2*n))
            mse=[max(0.,endpoints[0]-eps),min(upper,endpoints[1]+eps)]
            out['population_RMSE_CI']=dict(interval=[math.sqrt(v) for v in mse],MSE_interval=mse,
                confidence=confidence,method='HOEFFDING_TWO_ONE_SIDED_BOUNDED_LOSS_ENDPOINTS',
                uniform_loss_support=deepcopy(loss_support),numeric_rigor='BINARY64_NOT_INTERVAL_ARITHMETIC',
                includes_physical_systematic_error=False,sufficient_quantum_N_qualification=False)
    return out


def trial_budget(*, loss_upper, mse_half_width, confidence):
    """Hoeffding sufficient MC repetition count, not a quantum shot allocation."""
    L=_finite(loss_upper,'loss_upper');eta=_finite(mse_half_width,'mse_half_width')
    if L<=0 or eta<=0 or not 0<confidence<1:raise ValueError('Positive loss scale/precision and proper confidence required')
    try:
        return max(1,math.ceil((L/eta)**2*math.log(2/(1-confidence))/2))
    except OverflowError as error:
        raise ValueError('MC budget numerical range unresolved; no arbitrary trial cap') from error
