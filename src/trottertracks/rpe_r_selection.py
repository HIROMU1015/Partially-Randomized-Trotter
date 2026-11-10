"""Finite R menus and supplied-record selection. No N/C providers run here.

build_menu/coverage are metadata-only. evaluate_menu is a numerical entry;
real inputs require a separate scientific execution scope. All current tests
use artificial N/C. No sampling, prediction, fit, compiler or R search grid.
"""
from copy import deepcopy

from . import pf_pr_rpe_pipeline as p

a=p.accounting
FIXED_DELTA={'H4':.1,'H6':1.6}


def build_menu(task, R_by_round, *, sampling_law):
    """Expand PR rows only; preserve every other recipe/schedule field."""
    p.validate_task(task)
    if task['binding']['system'] not in FIXED_DELTA:raise ValueError('H4/H6 menu required')
    if (not isinstance(sampling_law,dict) or not sampling_law.get('source')
            or sampling_law.get('model')!='ideal_finite_RTE_law' or not sampling_law.get('sampling')):
        raise ValueError('Explicit sourced finite RTE law/sampling definition required')
    old=task['candidates'];rows=[]
    if len({(r['m'],p.arm(r['candidate'])) for r in old})!=len(old):
        raise ValueError('Expand a fixed one-candidate-per-arm/round task')
    if any(type(m) is not int or not any(r['m']==m and p.arm(r['candidate'])=='PR' for r in old) for m in R_by_round):
        raise ValueError('R menu round outside task')
    for row in old:
        c=row['candidate']
        if p.arm(c)!='PR' or row['m'] not in R_by_round:
            rows.append(deepcopy(row));continue
        if c['delta']!=FIXED_DELTA[c['system']]:raise ValueError('Variable R must preserve H4 .1 / H6 1.6 delta')
        values=R_by_round[row['m']]
        if (not isinstance(values,(list,tuple)) or not values
                or any(type(R) is not int or R<=0 for R in values) or len(set(values))!=len(values)):
            raise ValueError('Explicit finite unique positive integer R values required')
        for R in values:
            new=deepcopy(row);new['candidate']['R_outer']=R
            new['candidate']['id']=p.candidate_id(new['candidate'])
            new['structure']=p.structure(new['candidate'])
            new['adoption']='R_MENU_PROPOSAL_NOT_SCIENTIFICALLY_ADOPTED'
            rows.append(new)
    menu=deepcopy(task);menu['candidates']=rows
    menu['sampling_law']=deepcopy(sampling_law)
    menu['variable_R_rounds']=sorted(R_by_round)
    menu['R_menu_values']={str(m):list(v) for m,v in sorted(R_by_round.items())}
    p.validate_task(menu)
    return menu


def request_context(menu, row):
    """A/C common request fields. Echo only when upstream evidence matches."""
    if not any(p.same(row,r) for r in menu['candidates']):raise ValueError('Candidate outside menu')
    c=row['candidate']
    law=dict(definition=deepcopy(menu['sampling_law']) if p.arm(c)=='PR' else {'model':'deterministic'},
             candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=row['q_PF'])
    return dict(identity=p.identity(menu,row),law_binding=law)


def _validate(menu, shots, prices, policy):
    if not menu.get('sampling_law') or 'variable_R_rounds' not in menu:raise ValueError('Use an explicit R menu')
    inventory=p.plan(menu,shots,prices,price_policy=policy)
    for row in menu['candidates']:
        c=row['candidate']
        if p.arm(c)=='PR' and row['m'] in menu['variable_R_rounds']:
            if (type(c['R_outer']) is not int or c['R_outer'] not in menu['R_menu_values'][str(row['m'])]
                    or c['delta']!=FIXED_DELTA[c['system']]):raise ValueError('Changed R menu recipe/delta')
    lookup={p.key(r):r for r in menu['candidates']}
    for record in list(shots)+list(prices):
        row=lookup[p.key(record['identity'])]
        if 'law_binding' in record and not p.same(record['law_binding'],request_context(menu,row)['law_binding']):
            raise ValueError('N/C sampling law or R binding mismatch')
    return inventory,lookup


def _candidate(menu, row, n, prices, cohort_selection, policy):
    cid=row['candidate']['id'];reasons=[];chosen=[]
    if n is None:reasons.append('MISSING_N')
    elif (n['plan']['status']!='CONDITIONAL_ROUND_PASS' or
          any(type(n['plan'].get('N_'+axis)) is not int or n['plan']['N_'+axis]<=0 for axis in p.AXES)):
        reasons.append('N_PRECISION_OR_RADIUS_GATE_FAILED')
    if n is not None and 'law_binding' not in n:reasons.append('MISSING_N_LAW_BINDING')
    for axis in p.AXES:
        cohort=cohort_selection.get((cid,axis))
        found=[v for v in prices if p.key(v['identity'])==p.key(row) and v['axis']==axis and v['cohort']==cohort]
        if len(found)!=1:reasons.append('MISSING_SELECTED_C_'+axis);continue
        value=found[0];chosen.append(value)
        if 'law_binding' not in value:reasons.append('MISSING_C_LAW_BINDING_'+axis)
        if not p._eligible_price(value,row,policy):reasons.append('UNQUALIFIED_C_'+axis)
        if value['classification']=='MODEL_PREDICTION':
            model=value.get('model') or {}
            if (not p.same(model.get('domain'),row['candidate'])
                    or not p.same(model.get('binding'),menu['binding'])
                    or model.get('compiler')!=menu['profile']['compiler']):
                reasons.append('MISSING_OR_FOREIGN_R_MODEL_DOMAIN_'+axis)
        law=value.get('law_context',{})
        if (p.arm(row['candidate'])=='PR' and value['expectation']!='FINITE_COHORT_ONLY'
                and (law.get('price_law')!=law.get('N_law') or law.get('ideal_sampler_distance_qualified') is not True)):
            reasons.append('PRICE_SAMPLING_LAW_NOT_QUALIFIED_'+axis)
    if len(chosen)==2 and chosen[0]['cohort']!=chosen[1]['cohort']:reasons.append('UNPAIRED_XY_PRICE_SCENARIO')
    return list(dict.fromkeys(reasons)),chosen


def coverage(menu, shots, prices, *, cohort_selection, price_policy='qualified_expectation'):
    """No cost arithmetic, missing N/C calculation, or provider calls."""
    if price_policy not in ('qualified_expectation','finite_cohort_scenario'):raise ValueError('Price policy required')
    inventory,_=_validate(menu,shots,prices,price_policy);nmap={p.key(n['identity']):n for n in shots}
    rows=[]
    for row in menu['candidates']:
        reasons,chosen=_candidate(menu,row,nmap.get(p.key(row)),prices,cohort_selection,price_policy)
        rows.append(dict(m=row['m'],candidate_id=row['candidate']['id'],candidate=deepcopy(row['candidate']),
            arm=p.arm(row['candidate']),eligible=not reasons,reasons=reasons,
            N_status=next(v['N_status'] for v in inventory['rows'] if v['candidate_id']==row['candidate']['id']),
            price_contexts=deepcopy(next(v['price_contexts'] for v in inventory['rows'] if v['candidate_id']==row['candidate']['id'])),
            price_uncertainty=[deepcopy(v.get('uncertainty')) for v in chosen],
            full_RPE_qualified=False))
    return dict(status='R_MENU_METADATA_ONLY',rows=rows,inventory=inventory,costs=None,
        price_policy=price_policy,external_N_C_provider_calls=0,mandatory_stop=True)


def _price_rows(chosen):
    return [dict(m=v['identity']['m'],axis=v['axis'],candidate=v['identity']['candidate'],binding=v['identity']['binding'],
        compiler=v['identity']['profile']['compiler'],scope=p.SCOPE,classification=v['classification'],
        source=v['source'],mean=v['mean'],model=v.get('model'),peak_qubits=v.get('peak_qubits')) for v in chosen]


def _aggregate(menu, shots, prices, rounds, *, partial=False):
    return a.aggregate_supplied_axes([n['plan'] for n in shots],_price_rows(prices),expected_rounds=rounds,
        alpha_total=menu['configuration']['alpha_total_per_method'],compiler=menu['profile']['compiler'],
        metrics=('rz_count',),partial_rounds=partial,exact=True)


def _label(row,axis):return f"{p.arm(row['candidate'])}:{row['m']}:{row['candidate']['id']}:{axis}"


def _bounds(menu, evidence):
    if evidence is None:return None
    expected=[_label(r,axis) for r in menu['candidates'] for axis in p.AXES]
    if (evidence.get('labels')!=expected or not evidence.get('source')
            or evidence.get('covers_stat_model_law_and_menu') is not True
            or len(evidence.get('bounds',[]))!=len(expected)):
        raise ValueError('Full menu simultaneous bounds/source/coverage required')
    result={}
    for label,bound in zip(expected,evidence['bounds']):
        if len(bound)!=2:raise ValueError('Two price endpoints required')
        lo,hi=map(a.exact_scalar,bound)
        if not 0<=lo<=hi:raise ValueError('Nonnegative ordered bounds required')
        result[label]=(lo,hi)
    return result


def evaluate_menu(menu, shots, prices, *, cohort_selection, price_policy='qualified_expectation',
                  simultaneous_bounds=None, schedule_success=None):
    """Exact supplied N*C and conditional point selection; no real-data runs by default.

    Fixed alpha/beta/h/M make the supplied point objective separable by round.
    Correlations affect uncertainty and scientific adoption; they are not set
    to zero. Missing alternatives prevent a claim over the whole finite menu.
    """
    out=coverage(menu,shots,prices,cohort_selection=cohort_selection,price_policy=price_policy)
    nmap={p.key(n['identity']):n for n in shots};table={p.key(r):r for r in menu['candidates']}
    bounds=_bounds(menu,simultaneous_bounds)
    eligible={};selected=[];selections=[];all_bounds=True
    for item in out['rows']:
        row=table[item['m'],item['candidate_id']];n=nmap.get(p.key(row))
        _,chosen=_candidate(menu,row,n,prices,cohort_selection,price_policy)
        item.update(cost=None,cost_interval=None,N_X=None if n is None else n['plan'].get('N_X'),
            N_Y=None if n is None else n['plan'].get('N_Y'),N_diagnostics=None if n is None else deepcopy(n.get('diagnostics')))
        if not item['eligible']:continue
        result=_aggregate(menu,[n],chosen,[row['m']],partial=True)
        if result['totals'] is None:
            item['eligible']=False;item['reasons'].append(result['status']);continue
        item['cost']=result['totals']['rz_count'];cost=a.exact_scalar(item['cost'])
        if bounds is not None:
            intervals=[]
            for value in chosen:
                lo,hi=bounds[_label(row,value['axis'])]
                if not lo<=a.exact_scalar(value['mean']['rz_count'])<=hi:raise ValueError('Bounds must contain supplied price')
                intervals.append((n['plan']['N_'+value['axis']]*lo,n['plan']['N_'+value['axis']]*hi))
            item['cost_interval']=[a.exact_value(sum(v[j] for v in intervals)) for j in (0,1)]
        else:all_bounds=False
        eligible.setdefault((row['m'],item['arm']),[]).append((cost,row,chosen,item))
    M=menu['configuration']['horizon']['maximum_round_index_M'];complete=True;robust=True
    for m in range(M+1):
        for arm in p.ARMS:
            options=eligible.get((m,arm),[])
            if not options:complete=False;robust=False;continue
            best=min(v[0] for v in options);ties=[v for v in options if v[0]==best]
            winner=min(ties,key=lambda v:(v[1]['candidate']['R_outer'] or 0,v[1]['candidate']['id']))
            _,row,chosen,item=winner;selected.append((row,nmap[p.key(row)],chosen))
            dominance=(bounds is not None and all(a.exact_scalar(item['cost_interval'][1])<a.exact_scalar(v[3]['cost_interval'][0])
                for v in options if v[1]['candidate']['id']!=row['candidate']['id']))
            robust &= dominance
            selections.append(dict(m=m,arm=arm,candidate_id=row['candidate']['id'],R_outer=row['candidate']['R_outer'],
                cost=item['cost'],tied_candidate_ids=[v[1]['candidate']['id'] for v in ties],
                strict_dominance_from_supplied_bounds=dominance))
    pending=[v['candidate_id'] for v in out['rows'] if any(reason.startswith(('MISSING','UNQUALIFIED_C','PRICE_SAMPLING','UNPAIRED')) for reason in v['reasons'])]
    menu_complete=complete and not pending
    totals=None
    if complete:
        totals={}
        for arm in p.ARMS:
            group=[v for v in selected if p.arm(v[0]['candidate'])==arm]
            result=_aggregate(menu,[v[1] for v in group],[price for v in group for price in v[2]],range(M+1))
            if result['totals'] is None:totals=None;complete=False;break
            totals[arm]=result['totals']
    menu_complete=menu_complete and complete
    chosen_ids=[v['candidate_id'] for v in selections]
    success_context=dict(configuration=deepcopy(menu['configuration']),binding=deepcopy(menu['binding']),
        profile=deepcopy(menu['profile']),contract=deepcopy(menu['contract']),sampling_law=deepcopy(menu['sampling_law']),
        selected_candidate_ids=chosen_ids)
    success=False
    if schedule_success is not None:
        if not p.same(schedule_success.get('context'),success_context):raise ValueError('RPE success evidence differs from selected schedule')
        success=complete and bool(schedule_success.get('source')) and all(schedule_success.get(k) is True for k in
            ('prior_alias','horizon_decoder','all_round_phase_radius','failure_budget','implementation_and_law'))
    uncertainty=p.joint_uncertainty([],[],[],None)
    uncertainty.update(price_records=[dict(candidate_id=v['identity']['candidate']['id'],axis=v['axis'],
        uncertainty=deepcopy(v.get('uncertainty'))) for _,_,values in selected for v in values],
        simultaneous_bounds_source=None if simultaneous_bounds is None else deepcopy(simultaneous_bounds['source']),
        selection_induced_uncertainty_qualified=False,cross_candidate_covariance=None,
        strict_menu_dominance=bool(robust and all_bounds and menu_complete))
    comparison=a.existing.compare_totals(
        dict(totals=None if totals is None else totals['PF'],cost_scope=p.SCOPE,intervals=None),
        dict(totals=None if totals is None else totals['PR'],cost_scope=p.SCOPE,intervals=None))
    if totals is not None:
        comparison['PR_minus_PF']=a.exact_value(a.exact_scalar(totals['PR']['rz_count'])-a.exact_scalar(totals['PF']['rz_count']))
    out.update(status='CONDITIONAL_R_MENU_POINT_SELECTION' if complete else 'INCOMPLETE_R_MENU',
        costs='EXACT_SUPPLIED_N_TIMES_C',numeric_encoding='exact_rational',selections=selections,totals=totals,
        schedule_complete=complete,menu_complete=menu_complete,pending_candidate_ids=pending,
        selection_scope='COMPLETE_SUPPLIED_MENU' if menu_complete else 'AVAILABLE_QUALIFIED_SUBSET',
        schedule_success_context=success_context,rpe_success_qualified=success,
        full_RPE_qualified=False,scientific_R_adopted=False,uncertainty=uncertainty,
        comparison=comparison,
        supplied_cost_evaluation=True,
        normalization_is_price_multiplier=False,classical_samples_are_shot_multipliers=False,
        automatic_retry=False,mandatory_stop=True)
    return out
