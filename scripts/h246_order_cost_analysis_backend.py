"""Approved saved-value arithmetic; never constructs a quantum circuit or state."""
from fractions import Fraction
from collections import Counter
import importlib.util
import math
import mechanism_analysis as reused
import h246_order_cost_analysis as c

_TOKEN=object()


class Permit:
    def __init__(self,artificial,token=None):
        c.require(artificial is True or token is _TOKEN,'ANALYSIS_AUTHORIZATION_REQUIRED')
        self.artificial=artificial


def check(row,permit):
    c.require(isinstance(permit,Permit) and row['binding'].get('artificial',False) is permit.artificial,
              'REAL_DATA_REFUSED_BY_ARTIFICIAL_PERMIT')


def rational(value):
    return dict(numerator=value.numerator,denominator=value.denominator)


def exact_row(row,permit):
    check(row,permit)
    if row['status']!='ELIGIBLE' or any(v is None for v in row['N'].values()):return None
    n=len(row['pairs'])
    prices={a:Fraction(sum(p['axes'][a] for p in row['pairs']),n) for a in ('cosine','sine')}
    for a,v in prices.items():
        c.require(math.isclose(float(v),row['C'][a],rel_tol=2e-15,abs_tol=0),'SAVED_C_MEAN_CONSISTENCY')
    terms={a:row['N'][a]*prices[a] for a in prices};g=sum(terms.values(),Fraction())
    saved=float(row['G_saved'])
    c.require(abs(float(g)-saved)<=8*math.ulp(saved),'SAVED_G_ARITHMETIC_ORDER_CONSISTENCY')
    return prices,terms,g,sum(row['N'].values())


def decompose(base,other,permit):
    check(base,permit);check(other,permit)
    c.require(base['system']==other['system'] and base['epsilon']==other['epsilon'] and
              base['target']['method']==other['target']['method'] and
              base['target']['formula']=='2nd' and other['target']['formula']=='4th', 'FIXED_S2_TO_YOSHIDA4')
    a=exact_row(base,permit);b=exact_row(other,permit)
    if a is None or b is None:
        return dict(status='INELIGIBLE_OR_MISSING_NO_RATIO',system=base['system'],epsilon=base['epsilon'],
                    method=base['target']['method'],new_ratios=None)
    ca,ga,g0,n0=a;cb,gb,g1,n1=b
    cost=sum(base['N'][ax]*(cb[ax]-ca[ax]) for ax in ca)
    shots=sum(ca[ax]*(other['N'][ax]-base['N'][ax]) for ax in ca)
    cross=sum((other['N'][ax]-base['N'][ax])*(cb[ax]-ca[ax]) for ax in ca)
    c.require(cost+shots+cross==g1-g0,'ADDITIVE_C_N_CROSS_IDENTITY')
    rn=Fraction(n1,n0);rc=(g1/n1)/(g0/n0);rg=g1/g0
    weighted=sum((ga[ax]/g0)*Fraction(other['N'][ax],base['N'][ax])*(cb[ax]/ca[ax]) for ax in ca)
    c.require(rg==rn*rc==weighted,'GENERAL_TWO_AXIS_PRODUCT_IDENTITY')
    return dict(status='EXACT_ACCOUNTING_DECOMPOSITION',system=base['system'],method=base['target']['method'],
        epsilon=base['epsilon'],S2=base['candidate'],Yoshida4=other['candidate'],
        G_saved_unchanged=[base['G_saved'],other['G_saved']],G_exact_reconstructed=[rational(g0),rational(g1)],
        R_G=rational(rg),R_N_total=rational(rn),R_C_effective=rational(rc),
        R_C_axis={ax:rational(cb[ax]/ca[ax]) for ax in ca},
        additive=dict(C_at_S2_N=rational(cost),N_at_S2_C=rational(shots),cross=rational(cross),
                      total=rational(g1-g0)),
        direction='Y4_LOWER' if rg<1 else 'S2_LOWER' if rg>1 else 'TIE',
        uncertainty=[base['uncertainty'],other['uncertainty']],
        difference_SE=None,ratio_SE=None,candidate_cross_covariance='NOT_ASSUMED',
        causal_effect=None,population_guarantee=False)


def explanation(row,permit):
    check(row,permit)
    e=row['explanation']
    log_b=e['log_B_for_shots_saved']
    if log_b is None:log_b=e['log_B_base']+e['log_B_margin']
    side=dict(shots={a:row['N'][a] for a in ('cosine','sine')},explanation=dict(log_B=log_b,
        signed_bias=e['signed_bias'],allowance=e['allowance'],headroom=e['headroom'],
        alpha_axis=.025,log_shot_bound=e['log_shot_bound'],integer_status=e['integer_status']))
    # Reuse the complete verified scalar relationship function, including its rounding checks.
    try:out=reused.explanation_side(side,row['epsilon'])
    except (reused.AnalysisError,ValueError,TypeError) as exc:
        out=dict(status='UNRESOLVED',reason=str(exc))
    return dict(candidate=row['candidate'],system=row['system'],epsilon=row['epsilon'],
                saved_components=e['components'],base_log_B=e['log_B_base'],
                margin=e['log_B_margin'],relationship=out,causal_effect=None)


def structure(meta,schedule_fn,permit):
    check(meta,permit);t=meta['target']
    schedule=schedule_fn(t['formula'],t['q'],'ABS_TIME_PER_OUTER' if t['method']=='PR' else 'PER_PIECE',
                         32 if t['method']=='PR' else 1)
    if meta.get('saved_schedule') is not None:
        c.require(schedule==meta['saved_schedule'],'SAVED_H2_SCHEDULE_MISMATCH')
    result=dict(candidate=t['id'],system=t['system'],prefix=t['prefix'],q=t['q'],formula=t['formula'],
                schedule=schedule,deterministic_prefix_blocks=['one_body']+list(range(t['prefix'])),
                scalar_once=True,control='symmetric_directional',new_circuit_generation=False,
                cost_attribution='DESCRIPTIVE_STRUCTURE_NOT_A_COMPILED_COST_OR_CAUSAL_MODEL')
    if t['method']=='FULL':
        result.update(RTE_events=0,note='Full schedule rte_steps is an unused source placeholder')
        return result
    e=c.read(meta['representative_event_file'])
    c.require(e['event_digest']==meta['representative_event_digest'] and e['conditional'] is False and
              e['cohort']=='EXPLORATION','SAVED_REPRESENTATIVE_EVENT_BINDING')
    c.require(len(e['events'])==len(schedule) and all(len(events)==s['rte_steps']
                for events,s in zip(e['events'],schedule)),'SAVED_RTE_PIECE_ALLOCATION')
    orders=Counter();phases=Counter();applications=0;basis_switches=0;same_basis=0;identity=0
    for piece in e['events']:
        active=None
        for event in piece:
            orders[str(event['taylor_order'])]+=1;phases[str(event['phase']['real'])]+=1
            for app in event['application_sequence']:
                applications+=1
                if app['is_identity']:identity+=1;active=None;continue
                if active==app['basis_hash']:same_basis+=1
                else:basis_switches+=1
                active=app['basis_hash']
    result.update(representative_trajectory='Original first registered trajectory only',
                  representative_taylor_orders=dict(orders),representative_phases=dict(phases),
                  representative_applications=applications,representative_basis_switches=basis_switches,
                  representative_same_basis_adjacencies=same_basis,representative_identity_apps=identity,
                  all_trajectory_saved_strata=meta['saved_strata'],compiled_costs=meta['saved_cost_statistics'],
                  exhaustive_parameter_stream_identity=False,unobserved_event_cost=None)
    return result


def precision_summary(rows,system,method,permit):
    c.require(isinstance(permit,Permit),'ANALYSIS_AUTHORIZATION_REQUIRED')
    selected=[r for r in rows if r['system']==system and r['method']==method]
    return dict(system=system,method=method,fixed_candidates=True,reoptimized=False,
                observed=[dict(epsilon=r['epsilon'],direction=r.get('direction'),status=r['status']) for r in selected],
                missing_epsilons=[e for e in c.EPS if e not in [r['epsilon'] for r in selected]],
                direction_changes=len({r['direction'] for r in selected if 'direction' in r})>1,
                interpolation=False,cross_system_pooling=False)


def load_schedule(cfg):
    path=cfg['schedule_source']
    spec=importlib.util.spec_from_file_location('h246_sealed_original_schedule',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only schedule/weights/allocation are called. Historical SYSTEMS is not adopted.
    return module.schedule


def pipeline(session,data,cfg,*,artificial=False,schedule_fn=None):
    c.validate(data,artificial=artificial)
    if artificial:permit=Permit(True)
    else:
        c.require(type(session) is c.Session and session.used['bootstrap']==1 and
                  session.cfg is cfg,'AUTHORIZED_ANALYSIS_SESSION_REQUIRED')
        permit=Permit(False,_TOKEN)
    shots=[session.call('explanation_side',explanation,r,permit) for r in data['rows']]
    comparisons=[]
    for system in ('H2','H4','H6'):
        for method in ('FULL','PR'):
            for ep in c.EPS:
                pair=[r for r in data['rows'] if r['system']==system and r['target']['method']==method and r['epsilon']==ep]
                if len(pair)==2:comparisons.append(session.call('cost_decomposition',decompose,*pair,permit))
    schedule_fn=schedule_fn or load_schedule(cfg)
    structures=[session.call('candidate_structure',structure,m,schedule_fn,permit) for m in data['structures']]
    precision=[session.call('precision_summary',precision_summary,comparisons,s,m,permit)
               for s in ('H2','H4','H6') for m in ('FULL','PR')]
    return dict(schema=c.SERIES+'_RESULT',comparisons=comparisons,shot_relationships=shots,
                circuit_structure=structures,precision=precision,availability=data['availability'],
                limitations=c.LIMITS,science_GO=False,strict_COMPARE='BLOCKED',mandatory_STOP=True)
