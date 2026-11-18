"""Stream finite PF/PR preparation into the ordinary four-method round.

The producer callbacks keep their existing scientific selection and contracts.
Only immutable READY records enter the controller's context while other inputs
are prepared through the same distributed executor.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import copy,deepcopy
import json
from pathlib import Path

PF_METHODS=frozenset(('FULL_PF','TRUNCATED_PF'))
PR_METHODS=frozenset(('PR_PAULI','PR_DF'))
FATAL_STOPS=frozenset(('UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD','PREPARATION_HOLD'))


def snapshot_context(context):
    """Copy writable search tables, sharing pinned readers and raw history."""
    result=dict(context)
    for field in ('candidates','expected','generation','pf_evidence_registry'):
        if field in context:result[field]=deepcopy(context[field])
    if context.get('saved')is not None:
        result['saved']=copy(context['saved']);result['saved'].table=dict(context['saved'].table)
    return result


def _completion(root,problem,spec,run,runtime):
    from . import pr_candidate_completion as PR
    settings=deepcopy(spec['candidate_completion'])
    if not(settings.get('price_registry_source')or settings.get('price_registry_file')):
        supplied=(problem.get('price_rules')or{}).get('registry_file')
        if supplied:settings['price_registry_file']=str((Path(root)/supplied).resolve())
    state=Path(run)/'completion/STATE.json'
    settings.setdefault('source',json.loads(state.read_text())['source']if state.exists()else PR.git('rev-parse','HEAD',cwd=root))
    return PR.Completion(Path(run)/'completion',settings).bind_stage_executor(runtime)


def _merge_pf(context,after,routes,after_routes,methods):
    """PF's refreshed context must not overwrite a concurrently READY PR row."""
    methods=PF_METHODS.intersection(methods)
    old_ids={c.get('configuration_id')for c in context['candidates']if c['method']in methods}
    chosen=[deepcopy(c)for c in after['candidates']if c['method']in methods]
    new_ids={c.get('configuration_id')for c in chosen}
    if 'tau'in context and 'tau'in after and context['tau']!=after['tau']:
        raise ValueError('PF preparation changed the comparison threshold')
    if 'facts'in context and 'facts'in after and context['facts']!=after['facts']:
        raise ValueError('PF preparation changed the physical comparison facts')
    context['candidates']=[c for c in context['candidates']if c['method']not in methods]+chosen
    for cid in old_ids:context['expected'].pop(cid,None)
    context['expected'].update({cid:deepcopy(value)for cid,value in after['expected'].items()if cid in new_ids})
    if context.get('saved')is not None and after.get('saved')is not None:
        for identity in list(context['saved'].table):
            if identity[0]in old_ids:context['saved'].table.pop(identity)
        context['saved'].table.update({identity:deepcopy(value)for identity,value in after['saved'].table.items()if identity[0]in new_ids})
    if 'generation'in after:
        for method in methods:
            if method in after['generation']:context.setdefault('generation',{})[method]=deepcopy(after['generation'][method])
    if 'pf_evidence_registry'in after:context['pf_evidence_registry']=deepcopy(after['pf_evidence_registry'])
    for method in methods:
        if method in after_routes:routes[method]=deepcopy(after_routes[method])
    return context,routes


def prepare_inputs(root,problem,context,spec,run,routes,executor,optimize,runtime,extra):
    """Prepare independently and search each READY snapshot with the same queue.

    Returns ``context,routes,pf_status,completion,early_result``. A fatal early
    Q3 result suppresses further Q3 calls here; already submitted preparation is
    drained and its evidence remains available. The caller owns executor.finish.
    """
    from . import pf_candidate_selection as PF,pr_candidate_completion as PR,rpe_method_optimizer as O
    if runtime is None:raise ValueError('Distributed preparation runtime required; no local heavy fallback')
    methods=list(spec['methods']);routes=deepcopy(routes)
    completion=_completion(root,problem,spec,run,runtime)if spec.get('candidate_completion')else None
    unlimited=any(value=='UNLIMITED'for value in spec['max_new_q3_per_method'].values())
    search=dict(problem['search'],max_steps_per_method=spec.get('max_steps_per_method',
        100000 if unlimited else problem['search']['max_steps_per_method']))
    search.update(spec.get('search_overrides',{}))
    pf_status=None;early_result={};fatal=False;seen=None;pf_merged=False;installed_pr_keys=None

    def install_pr():
        nonlocal routes,installed_pr_keys
        if completion is None:return
        with completion.lock:
            keys=frozenset(key for key,record in completion.state['records'].items()if record['status']=='READY_FOR_SEARCH')
        if keys==installed_pr_keys:return
        routes,records=completion.install_snapshot(context,dict(problem,execution_routes=routes),keys)
        installed_pr_keys=keys
        if completion.settings.get('evaluate_completed_only',False):
            allowed={record.get('installed_configuration_id',record.get('configuration_id'))for record in records}
            context['candidates']=[candidate for candidate in context['candidates']
                if candidate['method']not in PR_METHODS or candidate.get('configuration_id')in allowed]

    def search_ready(preparing):
        nonlocal seen,fatal
        O.apply_published(context['candidates'],routes)
        ready=frozenset((candidate['method'],candidate['configuration_id'])for candidate in context['candidates']
            if candidate.get('ready')and candidate.get('configuration_id')and candidate['method']in methods)
        if ready==seen:return
        changed={method for method,cid in ready-(seen or frozenset())}
        seen=ready
        if fatal or not changed:return
        selected=[method for method in methods if method in changed]
        executor.routes=routes;executor.expected=context['expected']
        executor.log(event='ROUND_READY_DURING_PREPARATION',ready_configuration_ids=sorted(cid for method,cid in ready),
            methods=selected,preparation_running=preparing)
        results=optimize(root,dict(problem,search=search),context=context,methods=selected,
            initial=spec.get('initial','all'),max_new=spec['max_new_q3_per_method'],executor=executor,
            extra_results=extra,science_approved=spec.get('science_approved')is True)
        early_result.update(results)
        fatal=any(result.get('stop_reason')in FATAL_STOPS for result in results.values())

    # Keep exact raw history/pinned readers shared; only writable search state
    # needs a snapshot, not another copy of the large catalog and reader cache.
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='rpe-input-preparation')as pool:
        pf_settings=spec.get('pf_completion') or problem.get('pf_completion') or {}
        prepare_pf=(PF.prepare_next if pf_settings.get('max_candidates',1)!=1 and
                    pf_settings.get('mode')=='complete' and not pf_settings.get('selected') and runtime is not None
                    else PF.prepare)
        pf_future=pool.submit(prepare_pf,root,problem,snapshot_context(context),spec,run,deepcopy(routes),stage_executor=runtime)
        pr_future=(pool.submit(PR.prepare_for_search,root,dict(problem,execution_routes=deepcopy(routes)),
            snapshot_context(context),spec,run,completion=completion,stage_executor=runtime)if completion is not None else None)
        pending={pf_future}|({pr_future}if pr_future is not None else set())
        while True:
            if pf_future.done()and not pf_merged:
                after,after_routes,pf_status=pf_future.result()
                context,routes=_merge_pf(context,after,routes,after_routes,methods);pf_merged=True
            install_pr()
            search_ready(any(not future.done()for future in pending))
            pending={future for future in pending if not future.done()}
            if not pending:break
            # Completed futures are inspected without waiting for unrelated
            # producers. The pool still drains all work on fatal/exception.
            done,_=wait(pending,timeout=.03,return_when=FIRST_COMPLETED)
            for future in done:future.result()
        if pr_future is not None:pr_future.result()
        if not pf_merged:
            after,after_routes,pf_status=pf_future.result()
            context,routes=_merge_pf(context,after,routes,after_routes,methods)
        install_pr();search_ready(False)
    executor.routes=routes;executor.expected=context['expected']
    return context,routes,pf_status,completion,early_result or None
