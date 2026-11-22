"""Stream finite PF/PR preparation into the ordinary four-method round.

The producer callbacks keep their existing scientific selection and contracts.
Only immutable READY records enter the controller's context while other inputs
are prepared through the same distributed executor.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import copy,deepcopy
import json
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event

PF_METHODS=frozenset(('FULL_PF','TRUNCATED_PF'))
PR_METHODS=frozenset(('PR_PAULI','PR_DF'))


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
    stop = Event()
    pf_arrivals = SimpleQueue()

    def stopped():
        return stop.is_set() or getattr(executor, '_user_stop_requested', lambda: False)()

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
        try:
            results=optimize(root,dict(problem,search=search),context=context,methods=selected,
                initial=spec.get('initial','all'),max_new=spec['max_new_q3_per_method'],executor=executor,
                extra_results=extra,science_approved=spec.get('science_approved')is True)
        except BaseException:
            stop.set()
            raise
        early_result.update(results)
        fatal=O.shared_stop(results)
        if fatal:
            stop.set()

    horizon_policy = ()
    if (search.get('rule') == 'RPE_SCALED_SHOTS_V1' and spec.get('science_approved') is True
            and problem.get('problem', {}).get('system') == 'H4'
            and not context.get('pf_evidence_registry', {}).get('artificial')):
        from .rpe_scaled_shots import settings
        horizon_policy = settings(search['scaled_shots'])['M']

    def with_terminal_candidates(snapshot, pf_routes, entries):
        from .pf_optimizer_connection import registry_routes
        registry = deepcopy(snapshot['pf_evidence_registry'])
        registry['configurations'].update(entries)
        return (PF._refresh(root, dict(problem, search=search), snapshot, methods, registry, False),
                registry_routes(pf_routes, registry))

    def prepare_pf_inputs(snapshot, pf_routes):
        after, pf_routes, status = prepare_pf(root,problem,snapshot,spec,run,pf_routes,stage_executor=runtime)
        if 13 in horizon_policy and PF_METHODS.intersection(methods):
            from .pf_horizon_preparation import complete_scoped
            import sys
            entries = {}
            def register(cid, entry):
                entries[cid] = entry
                pf_arrivals.put((cid, entry))
            complete_scoped(root, runtime, after['candidates'], methods, horizon_policy, pf_routes,
                            pf_settings.get('interpreter', sys.executable),
                            spec.get('executor',{}).get('prior_run_dirs', ()),
                            stop_requested=stopped, on_ready=register)
            if entries:
                after, pf_routes = with_terminal_candidates(after, pf_routes, entries)
        return after, pf_routes, status

    def prepare_pr_inputs(snapshot, pr_routes):
        after, _, _ = PR.prepare_for_search(root, dict(problem, execution_routes=pr_routes),
            snapshot, spec, run, completion=completion, stage_executor=runtime)
        if not stopped() and 13 in horizon_policy and 'PR_DF' in methods:
            from .pr_df_horizon_preparation import prepare as prepare_df_horizons
            prepare_df_horizons(root, runtime, completion, after['candidates'], horizon_policy,
                                 spec.get('executor',{}).get('prior_run_dirs', ()), stopped)

    # Keep exact raw history/pinned readers shared; only writable search state
    # needs a snapshot, not another copy of the large catalog and reader cache.
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='rpe-input-preparation')as pool:
        pf_settings=spec.get('pf_completion') or problem.get('pf_completion') or {}
        prepare_pf=(PF.prepare_next if pf_settings.get('mode')=='complete' and
                    not pf_settings.get('selected') and runtime is not None
                    else PF.prepare)
        pf_future=pool.submit(prepare_pf_inputs, snapshot_context(context), deepcopy(routes))
        pr_future=(pool.submit(prepare_pr_inputs, snapshot_context(context), deepcopy(routes))
                   if completion is not None else None)
        try:
            pending={pf_future}|({pr_future}if pr_future is not None else set())
            while True:
                if pf_future.done()and not pf_merged:
                    after,after_routes,pf_status=pf_future.result()
                    context,routes=_merge_pf(context,after,routes,after_routes,methods);pf_merged=True
                entries = {}
                while True:
                    try:
                        cid, entry = pf_arrivals.get_nowait()
                    except Empty:
                        break
                    entries[cid] = entry
                if entries and not pf_merged:
                    after, after_routes = with_terminal_candidates(snapshot_context(context), deepcopy(routes), entries)
                    context, routes = _merge_pf(context, after, routes, after_routes, methods)
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
            install_pr()
            search_ready(False)
        finally:
            stop.set()
    executor.routes=routes;executor.expected=context['expected']
    return context,routes,pf_status,completion,early_result or None
