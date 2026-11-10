"""Static saved N/C inventory. No numerical provider, N solver or G call.

Reuse B's passive loaders and price policies. Categories are non-exclusive;
price availability, model capability and scientific adoption are separate.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

from . import pf_pr_rpe_pipeline as p

B_BASE = 'cb2b1d6b4309ca7d705f397e25e27d14e8ba125a'
H4_PREP = dict(commit='90dcc3ed20f30dfab6b22a68cc4adb9461485a98',path='docs/research/h4_df10_rpe_batch_preparation_20261110/PLAN.json')
H4_BASE_N = dict(commit='cefc01622a3cd75dd16d30de7b9fb68274f3eddd',path='docs/research/h4_df10_rpe_d1_20261110/result.json')
# Public remote observed at investigation start; never read a live B worktree.
H4_PUBLISHED_N = dict(commit='89c91c692e1d35de94d2e7afbe890aabe6d80b7e',path='docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json')
H6_RELEASE = 'f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a'
H6_PREP = dict(commit=H6_RELEASE,path='docs/research/h6_df19_rpe_m8_12_preparation_20261110/PLAN.json')
H6_RESULT = dict(commit=H6_RELEASE,path='docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json')
H6_COVERAGE = dict(commit=H6_RELEASE,path='docs/research/h6_df19_rpe_m8_12_results_20261110/run01/COVERAGE.json')


def _script(root, name):
    path=Path(root)/'scripts'/name
    spec=importlib.util.spec_from_file_location('_static_nc_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def accept_saved_N(task, previous, handoff, source):
    """Explicit published JSON overlay; copy integers/FAIL/null, never solve N.

    Equal overlaps are accepted; different evaluated records require a decision,
    not silent replacement. Missing rows remain missing. No file IO here.
    """
    if not source.get('commit') or len(source['commit'])!=40 or not source.get('path'):
        raise ValueError('PINNED_PUBLISHED_N_SOURCE_REQUIRED')
    rows={p.key(r):r for r in task['candidates']};bykey={p.key(r['identity']):deepcopy(r) for r in previous}
    incoming=handoff['shots'];seen=set()
    for record in incoming:
        key=p.key(record['identity'])
        if key not in rows or key in seen:raise ValueError('N_HANDOFF_MENU_OR_DUPLICATE')
        seen.add(key);p._match(task,rows[key],record)
        old=bykey.get(key)
        if old and (not p.same(old['plan'],record['plan']) or not p.same(old['actual'],record['actual'])):
            raise ValueError('CONFLICTING_SAVED_N_REQUIRES_EXPLICIT_SELECTION')
        bykey[key]=deepcopy(record)
    result=[bykey[p.key(row)] for row in task['candidates'] if p.key(row) in bykey]
    p.plan(task,result,[])  # Existing identity/measurement/qualification checks only.
    return result


def load_inputs(root, store, *, h4_handoff=H4_PUBLISHED_N):
    """Only immutable public Git JSON and passive saved-record functions."""
    old=_script(root,'plan_pf_pr_rpe_cost.py');systems={}
    h4_task=store.read(H4_PREP)['task'];old_h4=store.read(H4_BASE_N)
    h4_shots=[p.shot_record(h4_task,next(r for r in h4_task['candidates'] if p.key(r)==p.key(v)),v,
        dict(H4_BASE_N,scientific_source=old_h4['scientific_source'])) for v in old_h4['rows']]
    if h4_handoff is not None:
        packet=store.read(h4_handoff)
        h4_shots=accept_saved_N(h4_task,h4_shots,packet,h4_handoff)
    else:packet=None
    legacy_task,_,legacy_prices,_=old.collect(store,'H4')
    systems['H4']=dict(task=h4_task,shots=h4_shots,legacy_prices=legacy_prices,N_handoff_source=deepcopy(h4_handoff),
        N_handoff_status=packet['status'] if packet else 'BASELINE_M3_ONLY',
        N_precision_note=packet.get('precision_note') if packet else None)
    helper=_script(root,'run_h6_df19_rpe_m8_12.py')
    prepared=helper.prepare(root,plan=store.read(H6_PREP));result=store.read(H6_RESULT);coverage=store.read(H6_COVERAGE)
    if (result['status']!='M8_12_N_BATCH_EVALUATED' or not all(result[k] for k in ('source_preserved','input_preserved','previous_results_preserved'))
            or not coverage['all_13_N_rounds_conditional_pass']):raise ValueError('PUBLISHED_H6_COMPLETION_REQUIRED')
    h6_shots=prepared['saved_N']+result['new_records']
    legacy_task,_,legacy_prices,_=old.collect(store,'H6')
    systems['H6']=dict(task=prepared['task'],shots=h6_shots,legacy_prices=legacy_prices,N_handoff_source=deepcopy(H6_RESULT),
        N_handoff_status=result['status'],N_precision_note='Copy saved Python integers; no recomputation or cost arithmetic')
    for data in systems.values():
        p.validate_task(data['task']);p.plan(data['task'],data['shots'],[])
        for price in data['legacy_prices']:
            row=next(r for r in data['task']['candidates'] if p.key(r)==p.key(price['identity']))
            p._match(data['task'],row,price)
    return systems


def saved_catalog_records(provider, task):
    """Enumerate existing scalar selectors, then use the public lookup/mapper.

    _records is read only as selector metadata in these fixed provider versions.
    No path averaging, numerical entry, new descriptor or law evaluation.
    """
    prices=[];path_sources=[];seen=set()
    for row in task['candidates']:
        cid=row['candidate']['id']
        for item in provider._records:
            if item.get('candidate_id',item.get('candidate',{}).get('id'))!=cid:continue
            if item['price_unit']=='path':
                source=dict(candidate_id=cid,cohort=item['cohort'],kind=item['kind'],source=deepcopy(item['source']))
                if source not in path_sources:path_sources.append(source)
                continue
            is_pr=row['candidate']['method']=='PR'
            key=(cid,item['axis'],item['kind'],item['price_unit'],item['cohort'] if is_pr else None)
            if key in seen:continue
            seen.add(key)
            request=dict(candidate=row['candidate'],axis=item['axis'],q_RPE=row['q_RPE'],binding=task['binding'],
                profile=task['profile'],contract=task['contract'],kind=item['kind'],price_unit=item['price_unit'])
            if is_pr:request['cohort']=item['cohort']
            quote=provider.quote(request)
            if quote['value'] is None:raise ValueError('SAVED_CATALOG_SELECTOR_UNAVAILABLE')
            prices.append(provider.pipeline_record(task,row,item['axis'],quote))
    return prices,path_sources


def _context(record, origin):
    quote=record.get('provider_quote') or {};model=record.get('model')
    default_unit='cohort_mean' if p.arm(record['identity']['candidate'])=='PR' else 'deterministic'
    return dict(origin=origin,axis=record['axis'],scenario=record['cohort'],source_cohort=record.get('source_cohort',record['cohort']),
        classification=record['classification'],value=record.get('mean',{}).get('rz_count'),n_paths=record.get('n_paths'),
        price_unit=quote.get('price_unit',default_unit),
        expectation=record['expectation'],model=deepcopy(model),validation=quote.get('validation',(model or {}).get('coverage')),
        qualified_for_candidate=quote.get('qualified_for_candidate',(model or {}).get('qualified_for_candidate')),
        uncertainty=deepcopy(record.get('uncertainty')),source=deepcopy(record['source']),
        normalization_is_price_multiplier=False,aliases_are_not_independent_samples=True)


def inspect_system(data, provider):
    task=data['task'];shots=data['shots'];catalog,path_sources=saved_catalog_records(provider,task)
    # Preserve old saved direct/V1/alternative models as separate source scenarios.
    pairs=[(r,'BASELINE_SAVED') for r in data['legacy_prices']]+[(r,'PROVIDER_SAVED') for r in catalog]
    prices=[r for r,_ in pairs];inventory=p.plan(task,shots,prices,price_policy='finite_cohort_scenario')
    rows=[]
    for row in task['candidates']:
        key=p.key(row);cid=row['candidate']['id'];arm=p.arm(row['candidate'])
        n=next((v for v in shots if p.key(v['identity'])==key),None)
        plan=n['plan'] if n else {};actual=n['actual'] if n else {};contexts=[_context(v,o) for v,o in pairs if p.key(v['identity'])==key]
        available=[v for v,_ in pairs if p.key(v['identity'])==key and p._price_available(v)]
        direct=[v for v in available if v['classification'].startswith('DIRECT_COMPILED')]
        models=[v for v in available if v['classification']=='MODEL_PREDICTION']
        full_axes=lambda records:all(any(v['axis']==a for v in records) for a in p.AXES)
        passed=bool(n and plan['status']=='CONDITIONAL_ROUND_PASS' and all(type(plan.get('N_'+a)) is int and plan['N_'+a]>0 for a in p.AXES))
        selected=next(v for v in inventory['rows'] if v['candidate_id']==cid)
        request=dict(candidate=row['candidate'],axis='X',q_RPE=row['q_RPE'],binding=task['binding'],profile=task['profile'],contract=task['contract'])
        _,domain_error=provider._domain(request)  # Metadata predicate only, no numerical prediction.
        groups=[]
        if passed and selected['eligible_C_cohorts']:groups.append('SAVED_SCENARIO_INPUTS_AVAILABLE')
        if any(not (v.get('model') or {}).get('qualified_for_candidate',False) for v in models) or (domain_error is None and not full_axes(models)):
            groups.append('MODEL_REQUIRES_SCIENTIFIC_QUALIFICATION')
        if not selected['eligible_C_cohorts']:groups.append('NEW_PRICE_EVIDENCE_OR_VALIDATION_REQUIRED')
        if not passed or not (full_axes(direct) or full_axes(models)):groups.append('N_OR_SAVED_C_VALUE_MISSING')
        rows.append(dict(system=task['binding']['system'],m=row['m'],arm=arm,candidate=deepcopy(row['candidate']),q_RPE=row['q_RPE'],q_PF=row['q_PF'],
            N_status=plan.get('status','MISSING'),N_X=plan.get('N_X'),N_Y=plan.get('N_Y'),
            N_X_decimal=str(plan['N_X']) if plan.get('N_X') is not None else None,
            N_Y_decimal=str(plan['N_Y']) if plan.get('N_Y') is not None else None,
            actual_N_status=actual.get('status','UNRESOLVED'),actual_N_X=actual.get('N_X'),actual_N_Y=actual.get('N_Y'),
            N_source=deepcopy(n['source']) if n else None,N_handoff_source=deepcopy(data['N_handoff_source']),
            direct_C_available=full_axes(direct),saved_model_C_available=full_axes(models),
            eligible_finite_scenarios=selected['eligible_C_cohorts'],qualified_expectation_scenarios=selected['expected_C_cohorts'],
            price_contexts=contexts,path_price_sources=[v for v in path_sources if v['candidate_id']==cid],
            provider_domain_matches=domain_error is None,provider_domain_reason=domain_error,
            provider_numerical_model_entry_in_domain=domain_error is None,numerical_model_entry_executed=False,
            q_calibration=[1,2] if task['binding']['system']=='H6' and domain_error is None else None,
            independent_model_validation_q=[4] if task['binding']['system']=='H6' and domain_error is None else None,
            unvalidated_long_q=(row['q_PF']>4 if task['binding']['system']=='H6' and domain_error is None else
                row['q_PF']>(32 if arm=='PF' else 16) if task['binding']['system']=='H4' else False),
            unresolved=dict(actual_N=True,all_law_expected_C=arm=='PR',joint_covariance=True,long_q_systematic=True,total_CI=True),
            categories=groups,full_RPE_qualified=False,total_resources=None))
    return dict(rows=rows,task=deepcopy(task),N_handoff_status=data['N_handoff_status'],
        N_precision_note=data['N_precision_note'],static_plan=inventory,total_resources=None)


def build_coverage(root, *, h4_handoff=H4_PUBLISHED_N):
    store=p.SavedArtifacts(root);data=load_inputs(root,store,h4_handoff=h4_handoff)
    providers=dict(H4=p.h4_saved_price_provider(store.read),H6=p.h6_saved_price_provider(store.read))
    result={name:inspect_system(packet,providers[name]) for name,packet in data.items()}
    return dict(schema_version=1,baseline=B_BASE,H4_public_N_ref=deepcopy(h4_handoff),H6_public_N_ref=deepcopy(H6_RESULT),
        systems=result,sources=store.sources,categories_are_nonexclusive=True,
        scope='STATIC_SAVED_COVERAGE_AND_QUALIFICATION_ONLY',new_signal_N_price_predictions=0,new_fit_sampling_compile=0,
        N_times_C_calls=0,total_resources=None,ranking=None,joint_covariance=None,missing_covariance_is_zero=False,
        full_RPE_qualified=False,mandatory_stop=True)
