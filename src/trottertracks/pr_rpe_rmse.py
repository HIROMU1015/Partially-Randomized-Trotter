"""Paper-v2 RMSE model and common paper-kappa schedule, separate from Hoeffding.

Imports/planning are passive. Numeric model evaluation on real means is a new
scientific reanalysis; only artificial inputs are exercised in this release.
"""
from copy import deepcopy
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
import math

from . import pf_pr_rpe_pipeline as common

POLICY='PR_RPE_PAPER_V2_EMPIRICAL_RMSE_V1'
KAPPA_DEFINITION='PAPER_V2_E21_KAPPA_ROTATIONS_OVER_LAMBDA_R_SQUARED_TIME_SQUARED'
MAPPING='PAPER_TOTAL_UNIFORM_OUTER_DRAFT'
B_INTERFACE=dict(commit='d562bc38d3b63df51f775aa4eb0998afbf1873e2',path='docs/research/pr_rpe_rmse_controller_20261112/INTERFACE.json')
PAPER=dict(url='https://arxiv.org/pdf/2503.05647v2',sections=['B.2','E.1','E.3'],equations=['A38','B30','E1','E2','E21','E22'])
QUALIFICATIONS=('state_and_effective_evolution_model','decoder_alias_branch',
    'involutory_tail_mapping','finite_K_RMSE_transfer','error_composition','implementation_and_sampling_law')


def positive(value,name):
    if isinstance(value,bool):raise ValueError(name+' must be positive')
    result=Fraction(value)
    if result<=0:raise ValueError(name+' must be positive')
    return result


def index(value,name):
    if type(value) is not int or value<0:raise ValueError(name+' must be a nonnegative integer')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rmse_policy(*,target_rmse,qpe_rmse,evolution_rmse,error_rule,normalization_rule,source):
    """Explicit outer error budget; no alpha, beta or horizon is inherited."""
    target,qpe=positive(target_rmse,'target_rmse'),positive(qpe_rmse,'qpe_rmse')
    evo=Fraction(evolution_rmse)
    if isinstance(evolution_rmse,bool) or evo<0:raise ValueError('Nonnegative evolution RMSE required')
    if error_rule not in ('paper_quadrature','minkowski'):raise ValueError('Explicit RMSE error rule required')
    if normalization_rule not in ('paper_kappa_envelope','finite_B_diagnostic'):raise ValueError('Explicit normalization rule required')
    if not source:raise ValueError('Model/design SOURCE required')
    return dict(policy_id=POLICY,metric='energy_RMSE',status='EMPIRICAL_RESOURCE_MODEL_NOT_SUFFICIENT_SHOTS',
        family='PR_PAPER_RMSE',
        target_rmse=str(target),qpe_rmse=str(qpe),evolution_rmse=str(evo),error_rule=error_rule,
        normalization_rule=normalization_rule,N_terminal=11,slope_D=4,
        Cmax={'PF':.08,'PR':.10},constant_qualification='PAPER_EMPIRICAL_CONSTANTS_NOT_RIGOROUS',
        source=deepcopy(source),paper=deepcopy(PAPER),high_probability_compatible=False)


def design_from_task(base):
    """Copy physical metadata only. No saved signal, normalization or N read."""
    common.validate_task(base)
    system=base['binding']['system']
    if system not in ('H4','H6'):raise ValueError('Supply an explicit design for other systems')
    pr=next(r['candidate'] for r in reversed(base['candidates']) if common.arm(r['candidate'])=='PR')
    pf=next(r['candidate'] for r in reversed(base['candidates']) if common.arm(r['candidate'])=='PF')
    return dict(system=system,base_time=base['configuration']['base_time'],delta=pr['delta'],
        formula=pr['formula'],prefix=pr['prefix'],K=pr['K'],PF=dict(method=pf['method'],prefix=pf['prefix']),
        binding=deepcopy(base['binding']),profile=deepcopy(base['profile']),contract=deepcopy(base['contract']),
        energy_prior_ha=deepcopy(base['configuration']['energy_prior_ha']),decoder=base['configuration']['decoder'],
        state_assumption=base['state_assumption'],source=deepcopy(base['source']),
        early_rounds=deepcopy([r for r in base['candidates'] if system=='H6' and r['m']<4]))


def paper_event_mapper(*,kappa_paper,lambda_R,T,q_PF,signed_weights):
    """Shared B-draft count mapper; integer candidate generation is not a bound."""
    kappa=positive(kappa_paper,'kappa_paper');lam=positive(lambda_R,'lambda_R');time=positive(T,'T')
    if type(q_PF) is not int or q_PF<1 or not signed_weights:raise ValueError('Positive qPF and signed pieces required')
    target=kappa*lam*lam*time*time;r_target=math.ceil(target)
    R=max(len(signed_weights),math.ceil(Fraction(r_target,q_PF)))
    try:counts=list(common._schedule().allocation(tuple(signed_weights),R))
    except (OverflowError,ValueError) as exc:raise ValueError('Integer allocation numerical range unresolved') from exc
    if len(counts)!=len(signed_weights) or min(counts)<1 or sum(counts)!=R:
        raise ValueError('Integer allocation does not conserve whole-outer R')
    return dict(kind=MAPPING,kappa_definition=KAPPA_DEFINITION,r_nominal=str(target),r_target=r_target,
        R_outer=R,allocation=counts,event_count=q_PF*R,minimum_events=q_PF*len(signed_weights),
        definition_source=dict(paper=PAPER,module='src/trottertracks/pr_rpe_rmse.py:paper_event_mapper'),
        finite_K_signed_piece_guarantee_inferred=False)


def schedule(*,M,kappa_paper,design,tail_norm,policy,mapping=MAPPING):
    """r*=kappa lambda_R^2 T^2; R_outer=max(J,ceil(r*/qPF)).

    r* is a continuous nominal rotation count, not a compiled RZ count. Early
    H6 rows retain the saved distinct prefix/R law and do not use kappa.
    """
    index(M,'M');kappa=positive(kappa_paper,'kappa_paper')
    if mapping!=MAPPING or policy.get('policy_id')!=POLICY:raise ValueError('Explicit paper-kappa/RMSE mapping required')
    if design['system']!=design['binding']['system']:raise ValueError('System/binding differs')
    h=positive(design['base_time'],'h');delta=positive(design['delta'],'delta')
    lam=positive(tail_norm['lambda_R'],'lambda_R')
    if (not common.same(tail_norm['binding'],design['binding']) or tail_norm['prefix']!=design['prefix']
            or not tail_norm.get('source')):raise ValueError('Tail norm SOURCE/prefix/binding mismatch')
    if type(design['K']) is not int or design['K']<0 or design['K']%2:raise ValueError('Paired finite K must be even')
    task=dict(binding=deepcopy(design['binding']),profile=deepcopy(design['profile']),contract=deepcopy(design['contract']),
        configuration=dict(policy=deepcopy(policy),base_time=float(h),M_max_index=M,round_count=M+1,
            kappa_definition=KAPPA_DEFINITION,kappa_paper=str(kappa),tail_norm=deepcopy(tail_norm),
            energy_prior_ha=deepcopy(design['energy_prior_ha']),decoder=design['decoder']))
    rows=[]
    for m in range(M+1):
        T=math.ldexp(float(h),m)
        if not math.isfinite(T):raise ValueError('Nonfinite dyadic time')
        old=[r for r in design.get('early_rounds',[]) if r['m']==m]
        if old:
            if design['system']!='H6' or m>=4 or len(old)!=2:raise ValueError('Only fixed H6 early exceptions are supported')
            for row in old:
                c=row['candidate']
                if c['T']!=T or row['q_RPE']!=1<<m or c['q']!=1 or c['delta']!=T:
                    raise ValueError('Early H6 time/outer rules differ')
                expected=('PR',5,32,2) if common.arm(c)=='PR' else ('FULL',19,None,None)
                if tuple(c[k] for k in ('method','prefix','R_outer','K'))!=expected or c['formula']!='4th(m5_best)':
                    raise ValueError('Early H6 saved law must remain unchanged')
            choices=[deepcopy(r) for r in old]
            nominal=None
        else:
            if design['system']=='H6' and m<4:raise ValueError('H6 initial rounds require explicit saved exceptions')
            q=Fraction(T)/delta
            if q.denominator!=1 or q<1:raise ValueError('Nonintegral PF grid; no implicit delta change')
            q=int(q);nominal=kappa*lam*lam*Fraction(T)**2
            prototype=dict(system=design['system'],method='PR',formula=design['formula'],prefix=design['prefix'],
                T=T,q=q,delta=float(delta),R_outer=11 if design['formula']=='4th(m5_best)' else 1,K=design['K'])
            J=len(common.structure(prototype)['signed_weights'])
            mapped=paper_event_mapper(kappa_paper=kappa,lambda_R=lam,T=T,q_PF=q,
                signed_weights=common.structure(prototype)['signed_weights'])
            prototype['R_outer']=mapped['R_outer'];prototype['id']=common.candidate_id(prototype)
            pf=dict(prototype,method=design['PF']['method'],prefix=design['PF']['prefix'],R_outer=None,K=None)
            pf['id']=common.candidate_id(pf)
            choices=[dict(m=m,q_RPE=1<<m,q_PF=q,candidate=c,structure=common.structure(c)) for c in (pf,prototype)]
        for row in choices:
            c=row['candidate'];st=common.structure(c)
            if not common.same(st,row['structure']) or c['id']!=common.candidate_id(c):raise ValueError('Canonical recipe mismatch')
            pr=common.arm(c)=='PR'
            mapping_info=dict(definition=KAPPA_DEFINITION,mapping=MAPPING,requested_kappa=str(kappa),
                kappa_applied=pr and nominal is not None,r_nominal=str(nominal) if pr and nominal is not None else None,
                rotation_event_count=st['event_count'] if pr else 0,R_outer=c['R_outer'],
                minimum_events=c['q']*len(st['allocation']) if pr else 0,
                early_rule='PRESERVED_H6_PREFIX5_R32' if old else None,
                abstract_rotation_count_is_compiled_RZ=False,
                involutory_tail_mapping_qualified=False,
                negative_signed_pieces=any(w<0 for w in st['signed_weights']),
                paper_e1_over_kappa_envelope_assumed=False)
            identity=common.identity(task,dict(row,structure=st))
            request=dict(identity=identity,policy_id=POLICY,mapping=mapping_info,
                law_binding=dict(model=st['law_model'],candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=c['q'],
                    source=dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',
                        path='src/trotterlib/rte.py' if pr else 'src/trottertracks/h6_df19_rpe_d2.py')),
                provenance=dict(design=deepcopy(design['source']),tail=deepcopy(tail_norm['source']),paper=deepcopy(PAPER)))
            request['request_id']=digest(request);rows.append(request)
    config_id=digest(dict(task=task,design=design,rows=[{k:v for k,v in r.items() if k!='request_id'} for r in rows]))
    for r in rows:
        r['identity']['configuration_id']=config_id
        r['request_id']=digest({k:v for k,v in r.items() if k!='request_id'})
    return dict(schema_version=1,policy_id=POLICY,configuration_id=config_id,task=task,rows=rows,
        B_interface_source=deepcopy(B_INTERFACE),design=deepcopy(design),
        state_assumption=design['state_assumption'],scientific_PASS_FAIL=None,
        actual_N_qualified=False,full_RPE_qualified=False)


def nominal_axis_shots(M,m,log_B):
    """Paper empirical schedule only; no confidence/sufficiency claim.

    Decimal ceilings at two precisions avoid converting enormous integer N to
    binary64. Stability is a numerical diagnostic, not an interval proof.
    """
    index(M,'M');index(m,'m')
    if m>M or type(log_B) not in (int,float) or not math.isfinite(log_B) or log_B<0:
        raise ValueError('Finite nonnegative log B and valid round required')
    if 2*log_B>math.log(float.fromhex('0x1.fffffffffffffp+1023')):
        return dict(status='NUMERICAL_RANGE_UNRESOLVED',nominal_N_X=None,nominal_N_Y=None)
    precision=max(80,math.ceil(2*log_B/math.log(10))+40);values=[]
    for prec in (precision,precision+40):
        with localcontext() as ctx:
            ctx.prec=prec
            value=(Decimal(log_B)*2).exp()*Decimal(11+4*(M-m))
            values.append(int(value.to_integral_value(rounding=ROUND_CEILING)))
    if values[0]!=values[1]:return dict(status='NUMERICAL_CEILING_UNRESOLVED',nominal_N_X=None,nominal_N_Y=None)
    return dict(status='PAPER_EMPIRICAL_NOMINAL_ONLY',nominal_N_X=values[1],nominal_N_Y=values[1],
        N_per_axis_decimal=str(values[1]),terminal_samples=11,slope=4,
        normalization_log_used=log_B,ceiling_precision_stable=True,rigorous_sufficient_N=False)


def mean_evidence(request,record,source,*,qualification=None,qualification_source=None):
    """Adapt action-only records, never old high-probability N records."""
    if any(k in record for k in ('plan','actual','N_X','N_Y')):raise ValueError('Old N is not RMSE mean evidence')
    identity=request['identity']
    if record['candidate']!=identity['candidate'] or any(record[k]!=identity['structure'][k] for k in ('signed_weights','allocation','event_count')):
        raise ValueError('Finite mean candidate/allocation mismatch')
    if not source:raise ValueError('Finite-mean SOURCE required')
    return dict(kind='FINITE_K_MEAN_ONLY',request_id=request['request_id'],identity=deepcopy(identity),
        signal=deepcopy(record['signal']),source=deepcopy(source),
        qualification=deepcopy(qualification or {}),qualification_source=deepcopy(qualification_source))


def existing_finite_mean(ctx,request,reference,source):
    """Explicit future-science leaf; caller supplies context/half-steps/reference.

    Reuses existing action and independent reference gates. No context generation,
    reference fallback, sampling, compilation, old N solver or G evaluation.
    """
    from .pr_r_detailed_n_p1 import checked_reference
    from . import h6_df19_rpe_d2 as action
    c=request['identity']['candidate'];binding=request['identity']['binding']
    if not common.same(ctx.scientific_binding,binding):raise ValueError('H/state binding mismatch')
    ref=checked_reference(reference,binding,c['T'])['evidence']['reference']
    if c['system']=='H4' and c['method']=='B0':
        from .h4_df10_rpe_batch_provider import evolve_h4
        record=evolve_h4(ctx,c,ref)
    else:record=action.evolve(ctx,c,ref)
    return mean_evidence(request,record,source)


def evaluate(schedule_record,*,mean_provider=None):
    """Model-only N estimates; real-data invocation needs scientific permission.

    provider(requests) supplies bound means and upstream applicability evidence.
    This schema deliberately lacks legacy 'plan'/'CONDITIONAL_ROUND_PASS'.
    """
    if schedule_record.get('policy_id')!=POLICY:raise ValueError('RMSE policy required')
    rows=schedule_record['rows'];cfg=schedule_record['task']['configuration'];policy=cfg['policy'];M=cfg['M_max_index']
    canonical=rmse_policy(**{k:policy[k] for k in ('target_rmse','qpe_rmse','evolution_rmse','error_rule','normalization_rule','source')})
    if not common.same(policy,canonical) or len(rows)!=2*(index(M,'M')+1):raise ValueError('RMSE policy/round count changed')
    original_rows=[]
    for r in rows:
        original=deepcopy({k:v for k,v in r.items() if k!='request_id'})
        original['identity'].pop('configuration_id',None);original_rows.append(original)
    if digest(dict(task=schedule_record['task'],design=schedule_record['design'],rows=original_rows))!=schedule_record['configuration_id']:
        raise ValueError('Whole-schedule configuration changed')
    for request in rows:
        original={k:v for k,v in request.items() if k!='request_id'};i=request['identity'];c=i['candidate']
        if (request['request_id']!=digest(original) or i['configuration']!=cfg
                or c['id']!=common.candidate_id(c) or i['structure']!=common.structure(c)
                or c['T']!=math.ldexp(cfg['base_time'],i['m'])):raise ValueError('RMSE request recipe/policy identity changed')
        if i.get('configuration_id')!=schedule_record['configuration_id']:raise ValueError('Mixed whole-schedule configuration')
    if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('Duplicate RMSE request')
    supplied=[] if mean_provider is None else mean_provider(deepcopy(rows))
    if len({v.get('request_id') for v in supplied})!=len(supplied):raise ValueError('Duplicate mean evidence')
    if any(v.get('request_id') not in {r['request_id'] for r in rows} for v in supplied):raise ValueError('Foreign evidence scope')
    byid={v['request_id']:v for v in supplied};out=[]
    for request in rows:
        result=dict(request_id=request['request_id'],identity=deepcopy(request['identity']),policy_id=POLICY,
            status='NOT_EVALUATED',nominal_N_X=None,nominal_N_Y=None,sufficient_N_X=None,sufficient_N_Y=None,
            actual_N_X=None,actual_N_Y=None,rmse_guaranteed=False,scientific_PASS_FAIL=None,C=None,G=None,
            unresolved=[],provenance=deepcopy(request['provenance']))
        ev=byid.get(request['request_id'])
        if ev is None:out.append(result);continue
        if ev.get('kind')!='FINITE_K_MEAN_ONLY' or not common.same(ev['identity'],request['identity']) or not ev.get('source'):
            raise ValueError('Mean evidence kind/identity/SOURCE mismatch')
        signal=ev['signal'];L=signal.get('log_B')
        for field in ('reference','corrected','raw','exact_tail'):
            if field not in signal or any(type(signal[field].get(k)) not in (int,float) or not math.isfinite(signal[field][k]) for k in ('real','imag')):
                raise ValueError('Finite independent signal evidence required')
        if type(L) not in (int,float) or not math.isfinite(L) or L<0:raise ValueError('Finite log B required')
        arm=common.arm(request['identity']['candidate'])
        if arm=='PF' and L!=0:raise ValueError('Deterministic PF has no RTE normalization')
        try:
            used=0. if arm=='PF' else (float(1/Fraction(cfg['kappa_paper'])) if policy['normalization_rule']=='paper_kappa_envelope' else L)
        except OverflowError:used=None
        if used is None or (arm=='PR' and policy['normalization_rule']=='paper_kappa_envelope' and used==0):
            result.update(status='NUMERICAL_RANGE_UNRESOLVED',unresolved=['Paper normalization exponent not representable']);out.append(result);continue
        result.update(provenance=dict(result['provenance'],finite_mean=deepcopy(ev['source'])),signal=deepcopy(signal),normalization_log_used=used)
        if arm=='PR' and policy['normalization_rule']=='paper_kappa_envelope' and L>used:
            result.update(status='INAPPLICABLE_NORMALIZATION_ENVELOPE',unresolved=['Measured finite normalization exceeds paper envelope']);out.append(result);continue
        result.update(nominal_axis_shots(M,request['identity']['m'],used))
        if result['nominal_N_X'] is None:out.append(result);continue
        target,qpe,evo=(Fraction(policy[k]) for k in ('target_rmse','qpe_rmse','evolution_rmse'))
        budget_ok=qpe*qpe+evo*evo<=target*target if policy['error_rule']=='paper_quadrature' else qpe+evo<=target
        horizon=policy['Cmax'][arm]*math.pi/math.ldexp(cfg['base_time'],M)
        result['empirical_qpe_rmse_ha']=horizon
        if not budget_ok or horizon>float(qpe):
            result.update(status='INAPPLICABLE_MODEL_ERROR_BUDGET',unresolved=['Outer RMSE budget/horizon fails empirical model']);out.append(result);continue
        claims=ev.get('qualification',{})
        unresolved=[k for k in QUALIFICATIONS if claims.get(k) is not True]
        if not ev.get('qualification_source'):unresolved.append('qualification_SOURCE')
        status=('INAPPLICABLE_MODEL_ASSUMPTIONS' if any(claims.get(k) is False for k in QUALIFICATIONS) else
                'UNQUALIFIED_MODEL_ASSUMPTIONS' if unresolved else 'EMPIRICAL_MODEL_INPUTS_REPORTED_COMPLETE')
        result.update(status=status,unresolved=unresolved,
            qualification=deepcopy(claims),qualification_source=deepcopy(ev.get('qualification_source')),
            normalization_interpretation='Paper envelope checked on supplied means; not inferred from signed event count' if policy['normalization_rule']=='paper_kappa_envelope' else
                'Finite-B plug-in diagnostic; paper empirical calibration not established')
        out.append(result)
    return dict(policy_id=POLICY,records=out,common_kappa_definition=KAPPA_DEFINITION,
        actual_N_qualified=False,full_RPE_qualified=False,scientific_R_adopted=False,C=None,G=None)


def b_handoff(schedule_record,evaluation,source,*,evidence_domain='UNQUALIFIED_MODEL'):
    """B published draft-shaped packets. Unknown RMSE bounds/N stay null.

    B's generate/assess implementation is not published at the pinned interface;
    exact B-generated identity equality remains an explicit integration item.
    """
    if evaluation.get('policy_id')!=POLICY or not source:raise ValueError('RMSE model/SOURCE required')
    requests={r['request_id']:r for r in schedule_record['rows']}
    if len(evaluation['records'])!=len(requests) or {v['request_id'] for v in evaluation['records']}!=set(requests):raise ValueError('Incomplete or mixed model record scope')
    design=schedule_record['design'];cfg=schedule_record['task']['configuration'];packets=[]
    for arm in ('PF','PR'):
        selected=[v for v in evaluation['records'] if common.arm(v['identity']['candidate'])==arm]
        if any(v['identity']!=requests[v['request_id']]['identity'] for v in selected):raise ValueError('Foreign RMSE identity')
        spec=dict(binding=deepcopy(design['binding']),method=arm,
            split=dict(source=deepcopy(design['source']),randomized_scope='tail_after_prefix' if arm=='PR' else 'none'),
            L_D=design['prefix'] if arm=='PR' else design['PF']['prefix'],delta_pf=design['delta'],formula=design['formula'],
            h=cfg['base_time'],M=cfg['M_max_index'],kappa=dict(kind=KAPPA_DEFINITION,value=cfg['kappa_paper'],definition_source=PAPER) if arm=='PR' else None,
            event_mapping=dict(kind='PAPER_TOTAL_UNIFORM_OUTER_SAVED_EARLY_V1' if design.get('early_rounds') else MAPPING,
                source=dict(path='src/trottertracks/pr_rpe_rmse.py',function='paper_event_mapper'),
                pf_grid='EXACT_FIXED_DELTA_WITH_EXPLICIT_SAVED_EARLY',early_overrides=deepcopy(
                    [r for r in design.get('early_rounds',[]) if common.arm(r['candidate'])==arm])),
            finite_K=design['K'] if arm=='PR' else None,law=dict(model='ideal_finite_RTE_law' if arm=='PR' else 'deterministic',source=deepcopy(source)),
            rmse_policy=deepcopy(cfg['policy']),profile=deepcopy(design['profile']))
        identities=[]
        for v in selected:
            identity=deepcopy(v['identity']);identity.pop('configuration_id')
            identity['configuration']=deepcopy(spec);identities.append(identity)
        config_id=digest(dict(spec=spec,whole_schedule=identities))
        rounds=[]
        for v,identity in zip(selected,identities):
            req=requests[v['request_id']];identity['configuration_id']=config_id
            rounds.append(dict(m=identity['m'],identity=identity,law_binding=deepcopy(req['law_binding']),
                source_request_id=v['request_id'],N_X=None,N_Y=None,
                nominal_N_X=v['nominal_N_X'],nominal_N_Y=v['nominal_N_Y'],model_status=v['status']))
        status='NOT_EVALUATED' if all(v['status']=='NOT_EVALUATED' for v in selected) else (
            'INFEASIBLE' if any(v['status'].startswith('INAPPLICABLE') for v in selected) else
            'NUMERICAL_UNSTABLE' if any(v['status'].startswith('NUMERICAL') for v in selected) else 'UNQUALIFIED')
        packets.append(dict(schema='PR_RPE_RMSE_V1',configuration_id=config_id,spec=spec,
            N_packet=dict(configuration_id=config_id,policy=deepcopy(cfg['policy']),source=deepcopy(source),
                status='SYNTHETIC_ONLY' if evidence_domain=='SYNTHETIC_ONLY' else status,evidence_domain=evidence_domain,rounds=rounds,
                rmse_evidence=dict(target_rmse_ha=cfg['policy']['target_rmse'],supplied_rmse_bound_ha=None,
                    schedule_error_contract_qualified=False,state_assumption=schedule_record['state_assumption'],
                    B_generated_identity_verified=False,finite_K_signed_mapping_qualified=False))))
    return dict(interface_source=deepcopy(B_INTERFACE),packets=packets,scientific_R_adopted=False,real_G_evaluated=False)


def decode_supplied_coordinates(coordinates,*,h):
    """Reuse existing nearest-branch decoder; no shots or success simulation."""
    from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
    positive(h,'h');phases=[]
    for x,y in coordinates:
        if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1 for v in (x,y)):
            raise ValueError('Finite binary-outcome means required')
        if x==y==0:return dict(status='UNRESOLVED_ZERO_COORDINATES',energy_estimate_ha=None)
        phases.append(math.atan2(y,x))
    decoded=reconstruct_rpe_phase(phases,[1<<m for m in range(len(phases))])
    if any(math.isclose(r['distance_from_previous'],r['next_smallest_branch_distance'],abs_tol=1e-12,rel_tol=0.) for r in decoded['rounds'][1:]):
        return dict(status='UNRESOLVED_BRANCH_TIE',energy_estimate_ha=None)
    return dict(status='DECODED_SUPPLIED_COORDINATES_NOT_RMSE_QUALIFIED',energy_estimate_ha=-decoded['final_phase_estimate']/float(h),
        decoded=decoded,prior_alias_qualified=False,rmse_guaranteed=False)
