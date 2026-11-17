"""One command: per-method RPE cost optimization, report mode or an approved search round.

Approved round (``--round ROUND.json --run-dir DIR``): methods and new-Q3 budget come from the
round; each request is executed on the method's existing Q3 route, validated, ingested, and the
next candidate is chosen, until a stop rule fires.  Re-running the same command resumes: saved
and completed results are reused, running jobs are never launched twice.

Report mode (no ``--round``; new-Q3 budget 0, nothing is launched):
  FULL     all saved Q3 known; per-method best, pending next Q3, not-ready candidates
  INGEST   one externally supplied Q3 result (artificial here) is added; the next
           request is recomputed from the updated knowledge
  REPLAY   start from listed saved seeds only; saved results answer the search
  FIXTURE  artificial Q3 backend (labelled ARTIFICIAL) to exercise several steps
  OWN_PRICE same integer vectors priced by each PR configuration's own prices
  ROUTE_DRY_CHECK (``--dry-check-routes``) builds each next request on its real route and runs
           route metadata/count/runtime checks only
and writes RESULT.json, PER_METHOD.csv, NEEDS.json, INGEST_DEMO.json,
REPLAY.json, FIXTURE.json, OWN_PRICE.json and CHECKS.json.
"""
import argparse
import csv
from contextlib import contextmanager
from fractions import Fraction
import inspect
import json
from pathlib import Path
import subprocess
import time

from trottertracks import rpe_method_optimizer as O

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(O.jsonable(value), ensure_ascii=False, indent=2) + '\n')


@contextmanager
def round_phase(run, name, on_error=None):
    """High-resolution controller timings; not a scientific progress gate."""
    started=time.perf_counter();status='ERROR';caught=None
    def record(event, **values):
        with (run / 'ROUND_STAGES.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event,phase=name,epoch=time.time(),monotonic=time.monotonic(),**values))+'\n')
    record('START')
    try:
        yield
        status='OK'
    except BaseException as error:
        caught=error
        if on_error is not None:on_error(error)
        O.record_runtime_error(run,error,name)
        try:record('ERROR',error_type=type(error).__name__,message=str(error),error_log='RUNTIME_ERRORS.jsonl')
        except OSError:pass  # original failure was also sent to stderr if saving failed
        raise
    finally:
        try:record('END',seconds=time.perf_counter()-started,status=status)
        except OSError:
            if caught is None:raise


def candidate_rows(context):
    return [dict(name=c['name'], method=c['method'], configuration=c['key'], configuration_id=c['configuration_id'],
                 ready=c['ready'], needs=c['needs'], price_status=c['price_status'], signal_source=c['signal_source'],
                 execution_needs=c.get('execution_needs'), execution_binding_hold=c.get('execution_binding_hold'),
                 labels=c['labels'], saved_points=c['saved_points'], outside_index=c.get('outside_index', False),
                 prepared_by=c.get('prepared_by'), preparation=c.get('preparation'), price_quote={k:v for k,v in(c.get('price_quote')or{}).items()if k!='model_validations'},
                 price_validation=c.get('price_validation'),price_validation_issue=c.get('price_validation_issue'),
                 kappa_value=c.get('kappa_value'), kappa_base=c.get('kappa_base'), kappa_multiplier_exact=c.get('kappa_multiplier_exact'),
                 physical_aliases=[dict(configuration_id=x['configuration_id'], kappa_value=x.get('kappa_value'))
                                   for x in c.get('physical_aliases') or []])
            for c in context['candidates']]


def table(result, path):
    with path.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['method', 'best_candidate', 'configuration', 'N_X', 'N_Y', 'status', 'MSE_lower', 'MSE_upper',
                    'cost_RZ', 'stop_reason', 'next_Q3_candidate', 'next_Q3_N_X', 'next_Q3_cost_RZ', 'ready_candidates',
                    'not_ready_candidates', 'evaluated'])
        for name, r in result.items():
            b, p = r['best'], r['pending_request']
            w.writerow([name, b and b['candidate'], b and json.dumps(b['configuration']), b and ' '.join(map(str, b['N_X'])),
                        b and ' '.join(map(str, b['N_Y'])), b and b['status'],
                        b and '%.16e' % float(b['MSE_lower']), b and '%.16e' % float(b['MSE_upper']),
                        b and b['cost_display'], r['stop_reason'], p and p['candidate'],
                        p and ' '.join(map(str, O.split(p['vector'])[0])), p and O.show(p['cost']),
                        len(r['ready_candidates']), len(r['not_ready_candidates']), r['evaluated_count']])


def fixture_function(context, method):
    """Artificial MSE (labelled): sum_m w_m (1/N_mX + 1/N_mY)/2 with late rounds weighted more."""
    t = context['tau']
    weights = [Fraction((m + 1) ** 2, 1) for m in range(13)]
    scale = t * Fraction(3, 5) / sum(w / n for w, n in zip(weights, [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]))

    def mse(config_id, vector):
        nx, ny = O.split(vector)
        value = scale * sum(w * (Fraction(1, x) + Fraction(1, y)) / 2 for w, x, y in zip(weights, nx, ny))
        return value, value
    return mse


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def resume_input(paths):
    """Resume records exactly as written by a previous run (ROUND_RESULT/RESULT methods.*.resume_records, or a list)."""
    records = []
    for path in paths or []:
        value = O.load_json(path)
        if isinstance(value, list):
            records += value
        else:
            if value.get('resume_records_exported') is False:
                raise ValueError('Compact plan output is not a resume-record export; reload original OUTCOME/catalog/resume inputs')
            for r in value.get('methods', {}).values():
                records += r.get('resume_records', [])
    return records


def scoped(problem, overrides):
    """Apply an approved search scope (formulas, width range, L_D values, configuration allow-list) to methods."""
    p = json.loads(json.dumps(problem))
    for name, change in (overrides or {}).items():
        change = dict(change)
        if 'generation' in change:  # merge: e.g. only formulas / delta_over_h_values / kappa_multipliers
            p['methods'][name]['generation'] = dict(p['methods'][name]['generation'], **change.pop('generation'))
        p['methods'][name].update(change)
    return p


def price_overrides(problem, spec):
    """Round-level price-rule settings (e.g. registry_file: a B2-extended registry with newly calibrated split entries)."""
    if spec.get('price_rules_overrides'):
        problem = json.loads(json.dumps(problem));problem['price_rules'].update(spec['price_rules_overrides'])
    if spec.get('pf_evidence_registry'):
        problem = dict(problem, pf_evidence_registry=spec['pf_evidence_registry'])
    return problem


def merged_routes(problem, spec):
    """Route rules of the round; a rule's ``configurations_file`` (a growing registry of published configurations,
    written between invocations) is merged into its ``configurations`` table and configuration_ids."""
    routes = dict(problem['execution_routes']);routes.update(json.loads(json.dumps(spec.get('routes_override', {}))))
    for rules in routes.values():
        for rule in rules:
            f = rule.get('configurations_file')
            if f:
                path = Path(f) if Path(f).is_absolute() else ROOT / f
                extra = json.loads(path.read_text()) if path.exists() else {}
                table = dict(rule.get('configurations') or {});clash = [c for c in extra if c in table and extra[c] != table[c]]
                if clash:
                    raise SystemExit('Registry changes an existing route entry: ' + ','.join(c[:8] for c in clash))
                table.update(extra);rule['configurations'] = table
                if isinstance(rule.get('configuration_ids'), list):
                    rule['configuration_ids'] = sorted(set(rule['configuration_ids']) | set(extra))
    return routes


def make_executor(root, run, routes, expected, problem, spec):
    """One normal executor/profile factory, also used by requested-width batches.

    The caller owns finish(): preparation continuations keep the same queue,
    launch budget, expected identities and prefix-cache profile alive.
    """
    run=Path(run);root=Path(root);options=spec.get('executor',{})
    scheduler=options.get('scheduler','legacy')
    if scheduler not in('legacy','concurrent'):raise ValueError('Unknown executor.scheduler')
    budgets=spec['max_new_q3_per_method']
    approved=spec.get('science_approved',any(v not in(0,None)for v in budgets.values()))
    if any(v=='UNLIMITED'for v in budgets.values())and spec.get('science_approved')is not True:
        raise ValueError('UNLIMITED requires science_approved: true in the round file')
    # New distributed rounds use worker offers rather than the legacy default
    # of one global slot. Explicit limits and persisted controller settings win.
    concurrency=options.get('max_concurrent_jobs',
        'auto' if scheduler=='concurrent' and options.get('distributed') is not None else 1)
    settings=run/'distributed'/'ALLOCATION_SETTINGS.json'
    if settings.is_file():
        concurrency=O.load_json(settings)['max_concurrent_jobs']
    if concurrency != 'auto' and (type(concurrency) is not int or concurrency < 1):
        raise ValueError('max_concurrent_jobs must be auto or a positive integer')
    if concurrency == 'auto' and options.get('distributed') is None:
        raise ValueError('Automatic global concurrency requires a distributed executor')
    common=dict(optimizer_source=head(),max_concurrent=concurrency,
        wait_for_running=options.get('wait_for_running_jobs',False),
        prior_run_dirs=[(root/d).resolve()for d in options.get('prior_run_dirs',[])])
    if scheduler=='concurrent':
        from trottertracks import rpe_optimizer_concurrent as C
        cache=options.get('prefix_cache_directory')
        if cache is not None and not Path(cache).is_absolute():cache=str(run/cache)
        executor_class=C.DistributedExecutor if options.get('distributed') is not None else C.ConcurrentExecutor
        extra={} if executor_class is C.ConcurrentExecutor else dict(distributed=options['distributed'])
        executor=executor_class(root,run,routes,expected,problem,**common,**extra,budgets=budgets,
            science_approved=approved,dp_workers=options.get('dp_workers'),prefix_cache_directory=cache,
            prefix_checkpoint_rounds=options.get('prefix_checkpoint_rounds'))
        def optimize(*args,**kwargs):return C.optimize(*args,close_executor=False,**kwargs)
    else:
        executor=O.Executor(root,run,routes,expected,problem,**common);optimize=O.optimize
    graph=spec.get('distributed_preparation')
    if graph and not getattr(executor,'launch_disabled',False) and approved:
        if not hasattr(executor,'configure_preparation_graph'):
            executor.finish();raise ValueError('Preparation tasks require the shared distributed executor')
        executor.configure_preparation_graph(graph)
    # Optional diagnostics share models across allocations/runs; never alter readiness.
    from trottertracks.rpe_price_validation import Service
    try:validation=Service(root,run,spec,executor,problem=problem)
    except Exception as error:
        try:write(run/'PRICE_VALIDATION_ISSUE.json',dict(status='INCOMPLETE',reason=type(error).__name__+':'+str(error)))
        except OSError:pass
        return executor,optimize
    executor.price_validation_service=validation
    base_optimize=optimize
    def optional_validation(context,dispatch=True):
        try:
            validation.observe([c for c in context.get('candidates',[])if c['method']in spec['methods']])
            if dispatch:validation.poll()
        except Exception as error:
            try:write(run/'PRICE_VALIDATION_ISSUE.json',dict(status='INCOMPLETE',reason=type(error).__name__+':'+str(error)))
            except OSError:pass
    def optimize_with_price_validation(*args,**kwargs):
        context=kwargs.get('context')or{}
        optional_validation(context,dispatch=False);validation.start()
        try:return base_optimize(*args,**kwargs)
        finally:optional_validation(context)
    original_finish=executor.finish
    validation_closed=False
    def finish_with_price_validation():
        nonlocal validation_closed
        if not validation_closed:
            validation_closed=True
            try:validation.close()
            except Exception:pass  # optional diagnostic persistence cannot poison Q3
        return original_finish()
    executor.finish=finish_with_price_validation
    return executor,optimize_with_price_validation


def preparation_runtime(executor,run,spec):
    """Bind heavy providers to the same queue and slots as the shot search."""
    settings=(spec.get('executor',{}).get('distributed')or{}).get('preparation')
    if not settings:return None
    if not hasattr(executor,'submit_preparation'):
        raise ValueError('Distributed preparation requires DistributedExecutor')
    from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge
    return DistributedPreparationBridge(executor,ROOT,settings,approved=spec.get('science_approved')is True)


def preparation_budgets(spec):
    settings=(spec.get('executor',{}).get('distributed')or{}).get('preparation')or{}
    return {stage:settings.get('budgets',{}).get(stage,0)for stage in ('signal','compile','audit')}


def round_mode(a, problem, extra, *, prepared_context=None, executor_setup=None, result_callback=None):
    """Approved search round: execute requests on existing routes within the round budget; resumable."""
    spec = O.load_json(a.round);run = a.run_dir.resolve();run.mkdir(parents=True, exist_ok=True);start = time.perf_counter()
    if getattr(a,'plan_only',False):
        spec=dict(spec,science_approved=False,max_new_q3_per_method={m:0 for m in spec['methods']})
        if spec.get('pf_completion'):spec['pf_completion']=dict(spec['pf_completion'],mode='plan')
    compact = spec.get('plan_output') == 'compact'
    if spec.get('plan_output') not in (None, 'full', 'compact'):
        raise ValueError('Unknown plan_output')
    if compact:
        compact_plan_scope(spec)
    problem = price_overrides(scoped(problem, spec.get('method_overrides')), spec)
    for m in spec['methods']:
        g = problem['methods'][m].get('generation') or {}
        if g.get('kappa_points_file') and not Path(g['kappa_points_file']).is_absolute():
            g['kappa_points_file'] = str(ROOT / g['kappa_points_file'])
    with round_phase(run, 'PREPARE_INPUTS_AND_PRICES'):
        context = prepared_context if prepared_context is not None else O.prepare(
            ROOT, problem, methods=spec['methods'], preparation_dir=run / 'preparation',
            generate_configurations=not (spec.get('candidate_completion')or{}).get('bootstrap_saved_only',False))
    extra = extra + resume_input([Path(x) for x in spec.get('resume_from', [])])
    stored = run / 'ROUND.json'
    previous_round=O.load_json(stored) if stored.exists() else None
    if previous_round is not None:
        old = previous_round
        if compact and old.get('plan_output') != 'compact':
            raise ValueError('Compact plan output requires a new or existing compact-plan run directory')
        for key in ('round_id', 'methods', 'initial', 'routes_override', 'method_overrides', 'search_overrides', 'pf_completion', 'pf_evidence_registry'):
            if old.get(key) != spec.get(key):
                raise SystemExit('Run directory belongs to a different approved round (' + key + ')')
        if preparation_budgets(old)!=preparation_budgets(spec):
            raise ValueError('Preparation budget changed on resume; preserve this run and use a separately adopted new round')
    else:
        write(stored, spec)
    routes = merged_routes(problem, spec)
    from trottertracks import pf_optimizer_connection as PFConnection
    if context.get('pf_evidence_registry'):
        routes = PFConnection.registry_routes(routes, context['pf_evidence_registry'])
    executor=None;runtime=None;early_result=None;streamed_preparation=None;executor_initialized=False
    executor_spec=spec
    if previous_round is not None and not (run/'distributed'/'ALLOCATION_SETTINGS.json').exists():
        executor_spec=dict(spec,executor=dict(spec.get('executor',{})))
        executor_spec['executor']['max_concurrent_jobs']=previous_round.get('executor',{}).get('max_concurrent_jobs',1)
    if (spec.get('executor',{}).get('distributed')or{}).get('preparation'):
        executor,optimize=make_executor(ROOT,run,routes,context['expected'],problem,executor_spec)
        runtime=preparation_runtime(executor,run,spec)
        if executor_setup:
            executor_setup(executor,context);executor_initialized=True
    from trottertracks import pf_candidate_selection as PFSelect
    try:
        if runtime is not None and spec.get('science_approved')is True:
            from trottertracks.rpe_preparation_round import prepare_inputs
            before={m:executor.launched(m)for m in spec['methods']}
            with round_phase(run,'STREAM_PREPARATION_AND_READY_SEARCH'):
                context,routes,pf_preparation,completion,early_result=prepare_inputs(
                    ROOT,problem,context,spec,run,routes,executor,optimize,runtime,extra)
            streamed_preparation=dict(early_methods=list(early_result or{}),launched_before=before,
                newly_launched_q3={m:executor.launched(m)-before[m]for m in spec['methods']})
            if early_result:write(run/'EARLY_SEARCH_RESULT.json',slim(early_result))
            write(run/'PREPARATION_STREAM.json',streamed_preparation)
        else:
            with round_phase(run, 'SELECT_AND_PREPARE_PF'):
                context,routes,pf_preparation = PFSelect.prepare(ROOT,problem,context,spec,run,routes)
            if spec.get('candidate_completion'):
                from trottertracks.pr_candidate_completion import prepare_for_search
                with round_phase(run, 'PREPARE_SELECTED_CANDIDATES'):
                    context,routes,completion = prepare_for_search(ROOT,dict(problem,execution_routes=routes),context,spec,run)
    except BaseException:
        if executor is not None:executor.finish()
        raise
    unlimited = any(v == 'UNLIMITED' for v in spec['max_new_q3_per_method'].values())
    steps = spec.get('max_steps_per_method', 100000 if unlimited else problem['search']['max_steps_per_method'])
    search = dict(problem['search'], max_steps_per_method=steps)  # the finite range ends an UNLIMITED round
    search.update(spec.get('search_overrides', {}))  # e.g. incumbent_scope CONFIGURATION, lookahead, seed_saved_same_L_D
    O.apply_published(context['candidates'], routes)
    budgets = spec['max_new_q3_per_method']
    approved = spec.get('science_approved', any(v not in (0, None) for v in budgets.values()))
    if executor is None:executor,optimize=make_executor(ROOT,run,routes,context['expected'],problem,executor_spec)
    else:executor.routes=routes;executor.expected=context['expected']
    try:
        if executor_setup and not executor_initialized:
            executor_setup(executor,context)
        with round_phase(run, 'SEARCH_AND_Q3'):
            fatal={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}
            if early_result and any(r.get('stop_reason')in fatal for r in early_result.values()):
                result=early_result
                missing=[m for m in spec['methods']if m not in result]
                if missing:
                    result.update(O.optimize(ROOT,dict(problem,search=search),context=context,methods=missing,
                        initial=spec.get('initial','all'),max_new={m:0 for m in missing},science_approved=False))
            else:
                result = optimize(ROOT, dict(problem, search=search), context=context, methods=spec['methods'],
                                  initial=spec.get('initial', 'all'), max_new=budgets, executor=executor,
                                  extra_results=extra, science_approved=approved)
        if spec.get('candidate_completion'):
            from trottertracks.pr_candidate_completion import continue_selected_preparation,publish_comparison_exports
            result=continue_selected_preparation(ROOT,dict(problem,search=search),context,spec,run,completion,executor,result,extra,
                                                 optimizer=optimize)
            with round_phase(run, 'EXPORT_COMPARISON'):
                publish_comparison_exports(completion,run,context)
    finally:
        executor.finish()
    if result_callback:
        result_callback(result,context,routes)
    with round_phase(run, 'SAVE_RESULTS'):
        payload = dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=spec, methods=result,
                                              optimizer_source=head(), wall_seconds=time.perf_counter() - start,
                                              pf_preparation=pf_preparation,artificial_only=bool((pf_preparation or {}).get('artificial_only')),
                                              streamed_preparation=streamed_preparation,
                                              exact_bounds='MSE bounds with more than %d digits: outward 30-digit intervals; exact in job OUTCOME.json / C' % EXACT_DIGITS)
        write(run / 'ROUND_RESULT.json', compact_plan_result(payload, problem,
            resume_inputs=[str(p) for p in (getattr(a,'resume_from',None) or [])],
            ingest_input=str(a.ingest) if getattr(a,'ingest',None) else None) if compact else slim(payload))
        table(result, run / 'PER_METHOD.csv')
        if compact:
            write(run / 'NEEDS.json', dict(schema='RPE_OPTIMIZER_COMPACT_PLAN_NEEDS_V1', round_result='ROUND_RESULT.json',
                methods={m: dict(candidates=['methods',m,'not_ready_candidates'], pending_Q3=['methods',m,'pending_request'],
                                 input_requests=['methods',m,'input_requests']) for m in result}))
        else:
            write(run / 'NEEDS.json', dict(candidates=[r for r in candidate_rows(context) if r['method'] in spec['methods']],
                                           pending_Q3={m: r['pending_request'] for m, r in result.items()}))
    if not compact and not payload.get('artificial_only') and str(run).startswith(str(ROOT/'docs/research')):
        from publish_h4_search_run import prepare as publish_search
        publish_search(ROOT,run,run.parent/'SEARCH_PUBLICATION.json',run_id=spec['round_id'])
    with (run / 'INVOCATIONS.jsonl').open('a') as f:
        f.write(json.dumps(O.jsonable(dict(time=time.time(), budget=spec['max_new_q3_per_method'],
            launched={m: executor.launched(m) for m in spec['methods']},
            stop={m: r['stop_reason'] for m, r in result.items()}, new_calls={m: r['new_calls'] for m, r in result.items()})),
            ensure_ascii=False) + '\n')
    print(json.dumps(O.jsonable({m: dict(stop=r['stop_reason'], best=r['best'] and r['best']['cost_display'],
        new_calls=r['new_calls'], spent_before=r['budget']['spent_before'],
        next=r['pending_request'] and (r['pending_request']['candidate'], O.show(r['pending_request']['cost'])))
        for m, r in result.items()}), ensure_ascii=False, indent=1))
    return result


INTERRUPTED = 'INTERRUPTED_SEARCH_POLICY_CHANGE'
EXACT_DIGITS = 60


def outward(x, digits=30, up=False):
    """Decimal with ``digits`` significant digits rounded down (or up) -- exact integer arithmetic."""
    if x == 0:
        return '0'
    sign = -1 if x < 0 else 1;x = abs(x)
    e = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** e > x:
        e -= 1
    while Fraction(10) ** (e + 1) <= x:
        e += 1
    scaled = x / Fraction(10) ** (e - digits + 1)
    n = scaled.numerator // scaled.denominator
    if (up if sign > 0 else not up) and n * scaled.denominator != scaled.numerator:
        n += 1
    return ('-' if sign < 0 else '') + '%se%d' % (n, e - digits + 1)


def slim(value):
    """Long exact rationals (MSE bounds) -> outward 30-digit interval; the exact value stays in the job's OUTCOME.json."""
    if isinstance(value, dict) and set(value) >= {'numerator', 'denominator'} and len(set(value) - {'numerator', 'denominator', 'display'}) == 0:
        value = Fraction(int(value['numerator']), int(value['denominator']))
    if isinstance(value, Fraction):
        if max(len(str(abs(value.numerator))), len(str(value.denominator))) > EXACT_DIGITS:
            return dict(lower_30=outward(value), upper_30=outward(value, up=True), exact='OMITTED_SEE_JOB_OUTCOME_JSON')
        return value
    if isinstance(value, dict):
        return {k: slim(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [slim(v) for v in value]
    return value


def compact_plan_scope(spec):
    """Compact output is a presentation-only option for explicitly unapproved zero-budget plans."""
    budgets = spec.get('max_new_q3_per_method')
    if (spec.get('science_approved') is not False or not isinstance(budgets, dict)
            or any(m not in budgets for m in spec['methods'])
            or any(type(v) is not int or v != 0 for v in budgets.values())):
        raise ValueError('plan_output: compact requires science_approved: false and explicit zero budgets')


def compact_plan_result(payload, problem, *, resume_inputs=(), ingest_input=None):
    """Project a zero-call plan before walking thousands of repeated exact MSE bounds.

    Selection is complete before this function is called. Full evaluations remain
    in the original OUTCOME/catalog or supplied resume inputs; this output is not
    an exact resume-record export. Reload those inputs for another plan.
    """
    compact_plan_scope(payload['round'])
    keep = ('method', 'best', 'stop_reason', 'pending_request', 'new_calls', 'evaluated_count',
            'ready_candidates', 'search_range', 'incumbent_scope', 'prior_rounds', 'budget',
            'input_requests', 'left', 'settings', 'prefetch_policy', 'selection_timing', 'parameter_assessment',
            'executor_metrics')
    missing_keep = ('candidate', 'configuration', 'configuration_id', 'needs', 'price_status',
                    'price_needs', 'signal_source', 'reusable', 'next')
    methods = {}
    for name, result in payload['methods'].items():
        item = {k: result[k] for k in keep if k in result}
        evaluated = result.get('evaluated') or []
        statuses, sources = {}, {}
        for row in evaluated:
            status, source = row.get('status'), row.get('source')
            statuses[status] = statuses.get(status, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        item['evaluation_summary'] = dict(count=len(evaluated), by_status=statuses, by_source=sources,
            resume_records=len(result.get('resume_records') or []), trace=len(result.get('trace') or []),
            held=len(result.get('held') or []))
        item['exact_evaluation_sources'] = dict(
            prior_run_dirs=(result.get('prior_rounds') or {}).get('run_dirs', []),
            run_jobs='jobs/<method>/<job_id>/OUTCOME.json',
            saved_catalog=problem.get('evidence', {}).get('saved_catalog_inputs'),
            resume_from=list(payload['round'].get('resume_from', [])) + list(resume_inputs), ingest=ingest_input,
            note='Reload original identity-checked OUTCOME/catalog/resume inputs; compact output omits evaluation and resume-record arrays.')
        item['not_ready_candidates'] = [{k: row[k] for k in missing_keep if k in row}
                                        for row in result.get('not_ready_candidates') or []]
        by_configuration = result.get('by_configuration')
        if by_configuration is not None:
            item['by_configuration'] = by_configuration
        generation = result.get('generation')
        if generation is not None:
            item['generation_summary'] = dict(variants=len(generation.get('variants') or []),
                                              baseline=generation.get('baseline'))
        methods[name] = item
    projected = dict(payload, schema='RPE_OPTIMIZER_COMPACT_PLAN_V1', methods=methods,
                     plan_output='compact', resume_records_exported=False)
    preparation = payload.get('pf_preparation')
    if preparation and isinstance(preparation.get('selection'), dict):
        selection = preparation['selection']
        projected['pf_preparation'] = dict(preparation, selection=dict(selection,
            candidates=(selection.get('candidates') or [])[:1],
            remaining_candidates='Omitted from compact output; first preparation candidate and original count retained.'))
    return slim(projected)


def interrupted_mode(a, problem, extra):
    """Record a round stopped early because the search policy changed (not because the rule ran out of candidates).

    The round's own rule is replayed through the round's completed jobs only (launching is disabled), so each
    configuration's trajectory, incumbent and next request are exactly where the stopped round left them.
    Completed jobs the rule had not consumed yet (lookahead) and REQUEST-only directories (never launched) are listed
    separately.  Writes ROUND_RESULT_INTERRUPTED.json, EVALUATIONS.json and EVALUATIONS.csv in the run directory.
    """
    spec = O.load_json(a.round);run = a.run_dir.resolve();start = time.perf_counter()
    old = O.load_json(run / 'ROUND.json')
    for key in ('round_id', 'methods', 'initial', 'routes_override', 'method_overrides', 'search_overrides', 'pf_completion', 'pf_evidence_registry'):
        if old.get(key) != spec.get(key):
            raise SystemExit('Run directory belongs to a different approved round (' + key + ')')
    stop = O.load_json(a.interrupted)
    problem = scoped(problem, spec.get('method_overrides'));context = O.prepare(ROOT, problem, methods=spec['methods'])
    routes = dict(problem['execution_routes']);routes.update(spec.get('routes_override', {}))
    search = dict(problem['search'], max_steps_per_method=100000);search.update(spec.get('search_overrides', {}))
    O.apply_published(context['candidates'], routes)
    executor = O.Executor(ROOT, run, routes, context['expected'], problem, optimizer_source=head(), launch_disabled=True)
    result = O.optimize(ROOT, dict(problem, search=search), context=context, methods=spec['methods'],
                        initial=spec.get('initial', 'all'), max_new={m: 0 for m in spec['methods']}, executor=executor,
                        extra_results=extra, science_approved=True)
    budget_stop = 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'
    evaluations, tau = [], context['tau']
    for name, r in result.items():
        if r['stop_reason'] == budget_stop:
            r['stop_reason'] = INTERRUPTED
        ex = executor.for_method(name);cands = {c['configuration_id']: c for c in context['candidates'] if c['method'] == name}
        adopted = {(t['configuration_id'], tuple(t['vector'])) for t in r['trace']}
        jobs = sorted((run / 'jobs' / name).glob('*/REQUEST.json')) if (run / 'jobs' / name).exists() else []
        not_adopted, unlaunched = [], []
        for req in jobs:
            job = req.parent;q = json.loads(req.read_text());c = cands.get(q['configuration_id'])
            v = O.interleave(q['N_X'], q['N_Y'])
            if not (job / 'OUTCOME.json').exists():
                unlaunched.append(dict(job=job.name, configuration_id=q['configuration_id'], N_X=q['N_X'], N_Y=q['N_Y'],
                                       launched=(job / 'LAUNCH.json').exists(), cost=O.frac(q['cost']) if isinstance(q['cost'], dict) else q['cost'],
                                       status='REQUESTED_NOT_LAUNCHED' if not (job / 'LAUNCH.json').exists() else 'LAUNCHED_NO_OUTCOME'))
                continue
            rec = ex.ingest(job, dict(configuration_id=q['configuration_id'], vector=v), calls=0)
            row = dict(configuration_id=q['configuration_id'], L_D=c and c['key'].get('L_D'),
                       kappa_multiplier=c and c['key'].get('kappa_multiplier', '1'), N_X=q['N_X'], N_Y=q['N_Y'],
                       status=rec['status'], reason=rec.get('reason'), MSE_lower=rec['MSE_lower'], MSE_upper=rec['MSE_upper'],
                       MSE_upper_over_tau=None if rec['MSE_upper'] is None else float(rec['MSE_upper'] / tau),
                       cost=O.cost(v, c['prices']), adopted_by_rule=(q['configuration_id'], v) in adopted, source='ROUND_JOB',
                       job=str(job.relative_to(ROOT)) if job.is_relative_to(ROOT) else str(job), origin=q.get('origin', {}).get('kind'),
                       outcome_sha256=O.sha256(job / 'OUTCOME.json'))
            evaluations.append(dict(row, method=name))
            if not row['adopted_by_rule']:
                not_adopted.append(row)
        for (cid, v), rec in context['saved'].table.items():
            if cid in cands and cands[cid]['ready']:
                evaluations.append(dict(method=name, configuration_id=cid, L_D=cands[cid]['key'].get('L_D'),
                                        kappa_multiplier=cands[cid]['key'].get('kappa_multiplier', '1'), N_X=O.split(v)[0], N_Y=O.split(v)[1],
                                        status=rec['status'], reason=None, MSE_lower=rec['MSE_lower'], MSE_upper=rec['MSE_upper'],
                                        MSE_upper_over_tau=None if rec['MSE_upper'] is None else float(rec['MSE_upper'] / tau),
                                        cost=rec['cost'], adopted_by_rule=True, source='SAVED', job=None, origin='C_CATALOG',
                                        label=rec.get('label')))
        for b in r.get('by_configuration') or []:
            if b['stop_reason'] == budget_stop:
                b['stop_reason'] = INTERRUPTED
            mine = [e for e in evaluations if e['method'] == name and e['configuration_id'] == b['configuration_id']]
            feas = [e for e in mine if e['status'] == O.PASS]
            cheapest = min(feas, key=lambda e: (e['cost'], e['N_X'])) if feas else None
            b['evaluated'] = dict(total=len(mine), saved=sum(e['source'] == 'SAVED' for e in mine),
                                  round_jobs_adopted=sum(e['source'] == 'ROUND_JOB' and e['adopted_by_rule'] for e in mine),
                                  round_jobs_not_adopted=sum(e['source'] == 'ROUND_JOB' and not e['adopted_by_rule'] for e in mine),
                                  feasible=len(feas), infeasible=sum(e['status'] == O.FAIL for e in mine),
                                  other=sum(e['status'] not in (O.PASS, O.FAIL) for e in mine))
            b['cheapest_feasible_evaluated'] = cheapest and dict(
                N_X=cheapest['N_X'], N_Y=cheapest['N_Y'], cost=cheapest['cost'], cost_display=O.show(cheapest['cost']),
                MSE_lower=cheapest['MSE_lower'], MSE_upper=cheapest['MSE_upper'], MSE_upper_over_tau=cheapest['MSE_upper_over_tau'],
                adopted_by_rule=cheapest['adopted_by_rule'], source=cheapest['source'], job=cheapest['job'])
        r['interrupted'] = dict(stop_reason=INTERRUPTED, computed_not_adopted=not_adopted, requested_not_completed=unlaunched,
                                note='rule trajectories replayed through completed jobs only; no Q3 launched by this record')
    write(run / 'ROUND_RESULT_INTERRUPTED.json', slim(dict(
        schema='RPE_OPTIMIZER_ROUND_INTERRUPTED_V1', status=INTERRUPTED, stop=stop, round=spec, methods=result,
        record_source=head(), wall_seconds=time.perf_counter() - start,
        distinct_from=['NO_IMPROVING_PROPOSAL', 'NO_FEASIBLE_POINT_IN_RANGE'],
        exact_bounds='MSE bounds with more than %d digits are written as outward 30-digit intervals; exact values: each job OUTCOME.json' % EXACT_DIGITS,
        next_round='load completed jobs with executor.prior_run_dirs (same job key and route) or EVALUATIONS.json resume rows')))
    write(run / 'EVALUATIONS.json', slim(dict(schema='RPE_OPTIMIZER_EVALUATIONS_V1', tau=tau, rows=evaluations,
                                         resume_records=[dict(configuration_id=e['configuration_id'], N_X=e['N_X'], N_Y=e['N_Y'],
                                                              status=e['status'], MSE_lower=O.enc(e['MSE_lower']), MSE_upper=O.enc(e['MSE_upper']),
                                                              source='ROUND_JOB:' + e['job'], job=e['job'])
                                                         for e in evaluations if e['source'] == 'ROUND_JOB' and e['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE')])))
    keys = ['method', 'L_D', 'kappa_multiplier', 'configuration_id', 'N_X', 'N_Y', 'status', 'MSE_upper_over_tau', 'cost',
            'adopted_by_rule', 'source', 'origin', 'job']
    with (run / 'EVALUATIONS.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(keys)
        for e in sorted(evaluations, key=lambda e: (e['method'], e['L_D'] or 0, Fraction(e['kappa_multiplier']), e['cost'])):
            w.writerow([' '.join(map(str, e[k])) if isinstance(e[k], list) else (O.show(e[k]) if isinstance(e[k], Fraction) else e[k])
                        for k in keys])
    with (run / 'INVOCATIONS.jsonl').open('a') as f:
        f.write(json.dumps(O.jsonable(dict(time=time.time(), mode='INTERRUPTED_RECORD', launched={m: executor.launched(m) for m in spec['methods']},
                                           stop={m: r['stop_reason'] for m, r in result.items()})), ensure_ascii=False) + '\n')
    for name, r in result.items():
        for b in r.get('by_configuration') or []:
            ce = b['cheapest_feasible_evaluated']
            print(name, b['candidate'], b['stop_reason'], 'steps', b['steps'], 'best', b['best'] and b['best']['cost_display'],
                  'cheapest_any', ce and ce['cost_display'], ce and ce['adopted_by_rule'], b['evaluated'])
        print(name, r['stop_reason'], 'not adopted', len(r['interrupted']['computed_not_adopted']),
              'requested not completed', len(r['interrupted']['requested_not_completed']))


def dry_check_routes(problem, context, full, out):
    """Build the next request(s) of each method on its real route and run metadata/count/runtime checks only."""
    rows = []
    for name, r in full.items():
        pend = r['pending_request']
        if pend is None:
            continue
        seen, picks = set(), []
        for q in [pend] + pend['queue']:
            cid = next(c['configuration_id'] for c in context['candidates'] if c['name'] == q['candidate'])
            if cid not in seen:
                seen.add(cid);picks.append(dict(q, configuration_id=cid, vector=tuple(q['vector'])))
        ex = O.Executor(ROOT, out / 'ROUTE_DRY_CHECK', problem['execution_routes'], context['expected'], problem,
                        optimizer_source=head()).for_method(name)
        for q in picks:
            rule = ex.route(q['configuration_id']);job = ex.job_dir(q)
            if rule is None:
                rows.append(dict(method=name, candidate=q['candidate'], status='NO_ROUTE_ADAPTER'));continue
            job.mkdir(parents=True, exist_ok=True)
            (job / 'REQUEST.json').write_text(json.dumps(O.jsonable(ex.request(q, job, rule)), ensure_ascii=False, indent=2) + '\n')
            for stale in ('DRY_CHECK.json', 'JOB_STAGES.jsonl'):
                (job / stale).unlink(missing_ok=True)
            code = ex.run_adapter(job, rule, 'dry-check').wait()
            check = O.load_json(job / 'DRY_CHECK.json') if (job / 'DRY_CHECK.json').exists() else dict(status='NO_OUTPUT')
            rows.append(dict(method=name, candidate=q['candidate'], route=rule['kind'], view_commit=rule['view_commit'],
                             status=check['status'], returncode=code, bounds=check.get('bounds'), template=check.get('template'),
                             science_calls=0, job=str(job.relative_to(ROOT)) if job.is_relative_to(ROOT) else str(job)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem', type=Path, required=True)
    p.add_argument('--output', type=Path, help='report-mode output directory')
    p.add_argument('--round', type=Path, help='approved search round JSON (methods, new-Q3 budget, optional routes override)')
    p.add_argument('--run-dir', type=Path, help='round directory (jobs, results); re-run the same command to resume')
    p.add_argument('--dry-check-routes', action='store_true')
    p.add_argument('--ingest', type=Path, help='JSON list of Q3 results {configuration_id,N_X,N_Y,status,MSE_lower,MSE_upper,source}')
    p.add_argument('--resume-from', type=Path, action='append', help='previous ROUND_RESULT.json/RESULT.json (resume_records) to reuse')
    p.add_argument('--interrupted', type=Path, help='with --round/--run-dir: STOP record JSON of a round stopped by a search-policy '
                   'change; replays the round through its completed jobs (no launch) and writes ROUND_RESULT_INTERRUPTED.json')
    a = p.parse_args();start = time.perf_counter()
    problem = O.load_json(a.problem)
    extra = (O.load_json(a.ingest) if a.ingest else []) + resume_input(a.resume_from)
    if a.round:
        if a.run_dir is None:
            raise SystemExit('--run-dir is required with --round')
        if a.interrupted:
            return interrupted_mode(a, problem, extra)
        return round_mode(a, problem, extra)
    out = a.output;out.mkdir(parents=True, exist_ok=True)
    context = O.prepare(ROOT, problem, preparation_dir=out / 'preparation')
    full = O.optimize(ROOT, problem, context=context, extra_results=extra)
    checks = {}
    minima = context['catalog']['method_minima']
    checks['C_catalog_minima_reproduced'] = {m: (full[m]['best'] is not None and minima.get(m) is not None
                                              and full[m]['best']['cost'] == O.frac(minima[m]['conditional_cost'])) for m in full}
    priced = [c for c in context['candidates'] if c['prices'] is not None]
    checks['own_prices_reproduce_every_saved_total'] = all(c['price_status'] == 'SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS' for c in priced)
    checks['priced_configurations'] = len(priced)
    for name, spec in problem['methods'].items():
        if spec['family'] == 'PR':
            lds = sorted({c['key']['L_D'] for c in context['candidates'] if c['method'] == name})
            checks['L_D_enumerated_' + name] = dict(values=lds, fragment_count=context['facts']['fragment_count'],
                                                     from_rule=lds == list(range(spec['L_D_range']['min'],
                                                         context['facts']['fragment_count'] + spec['L_D_range']['max_offset_from_fragment_count'] + 1)))
        else:
            checks['PF_candidates_' + name] = len([c for c in context['candidates'] if c['method'] == name])
    write(out / 'RESULT.json', dict(schema='RPE_METHOD_OPTIMIZER_RESULT_V1', rule=O.RULE, problem=str(a.problem),
                                     facts=context['facts'], tau=context['tau'], methods=full, ingested=extra,
                                     saved_catalog=problem['evidence']['saved_catalog_inputs'],
                                     scope='MINIMUM_WITHIN_EVALUATED_SET_AND_FINITE_SEARCH_RULE; no new science'))
    table(full, out / 'PER_METHOD.csv')
    write(out / 'GENERATION.json', context.get('generation', {}))
    write(out / 'PRICE_RULES.json', context.get('price_rules'))
    write(out / 'PARAMETER_ASSESSMENT.json', {m: r.get('parameter_assessment') for m, r in full.items()})
    write(out / 'NEEDS.json', dict(candidates=candidate_rows(context),
                                   pending_Q3={m: r['pending_request'] for m, r in full.items()}))

    # INGEST: an artificial result for the TRUNCATED_PF pending proposal changes the next request.
    pend = full['TRUNCATED_PF']['pending_request']
    fake = dict(configuration_id=pend['configuration_id'], N_X=O.split(pend['vector'])[0], N_Y=O.split(pend['vector'])[1],
                status=O.PASS, MSE_lower=O.enc(context['tau'] * Fraction(19, 20)), MSE_upper=O.enc(context['tau'] * Fraction(19, 20)),
                source='ARTIFICIAL_INGEST_DEMO')
    after = O.optimize(ROOT, problem, context=context, methods=['TRUNCATED_PF'], extra_results=extra + [fake])['TRUNCATED_PF']
    write(out / 'INGEST_DEMO.json', dict(ingested=fake, before_pending=pend, after_best=after['best'],
                                         after_pending=after['pending_request'], label='ARTIFICIAL_RESULT_NOT_SCIENCE'))
    checks['ingest_changes_next_request'] = (after['best']['cost'] == pend['cost'] and after['pending_request'] is not None
                                             and after['pending_request']['cost'] < pend['cost'])

    # REPLAY from listed saved seeds; saved results answer each step.
    replay = O.optimize(ROOT, problem, context=context, methods=list(problem['verification']['replay_seeds']),
                        initial=problem['verification']['replay_seeds'])
    write(out / 'REPLAY.json', replay)
    for name, r in replay.items():
        steps = r['trace']
        seed_cost = steps[0]['incumbent_before'] if steps else None
        checks['replay_' + name] = dict(seed_cost=seed_cost, best_cost=r['best'] and r['best']['cost'],
                                         reached_cheaper_feasible=bool(steps) and r['best']['cost'] < seed_cost,
                                         steps=len(steps), new_calls=r['new_calls'], stop_reason=r['stop_reason'],
                                         next_from_updated_state=all(s2['origin'].get('base_vector') == s1['vector']
                                                                     for s1, s2 in zip(steps, steps[1:])
                                                                     if s1['improved'] and s2['candidate'] == s1['candidate']
                                                                     and s2['origin']['kind'].startswith('DECREMENT')))

    # FIXTURE: artificial backend only, from no knowledge.
    fx = O.FixtureBackend(fixture_function(context, 'TRUNCATED_PF'), context['tau'])
    fixture = O.optimize(ROOT, problem, context=context, methods=['TRUNCATED_PF'], initial='none', backend=fx, max_new=12,
                          saved_is_free=False)
    write(out / 'FIXTURE.json', dict(fixture, label='ARTIFICIAL_FIXTURE_NOT_SCIENCE'))
    f = fixture['TRUNCATED_PF']
    checks['fixture'] = dict(steps=len(f['trace']), improvements=sum(t['improved'] for t in f['trace']),
                             continued_after_first_improvement=len([t for t in f['trace'] if t['improved']]) > 1
                             or (any(t['improved'] for t in f['trace']) and f['trace'][-1]['improved'] is False),
                             stop_reason=f['stop_reason'], new_calls=f['new_calls'])

    # OWN_PRICE: same vectors under each ready PR configuration's own prices.
    saved = {r['label']: (cid, v, r) for (cid, v), r in context['saved'].table.items()}
    vectors = {lab: saved[lab][1] for lab in problem['verification']['own_price_vectors_from_labels'] if lab in saved}
    pr = [c for c in context['candidates'] if c['family'] == 'PR']
    rows = O.own_price_comparison(pr, vectors)
    for row in rows:
        cid = next(c['configuration_id'] for c in pr if c['name'] == row['candidate'])
        hit = context['saved'].table.get((cid, vectors[row['vector_label']]))
        row.update(saved_status=hit and hit['status'], saved_cost_matches=hit is None or hit['cost'] == row['cost'])
    write(out / 'OWN_PRICE.json', rows)
    checks['own_price_rows'] = len(rows)
    checks['own_price_saved_costs_match'] = all(r['saved_cost_matches'] for r in rows)
    source = inspect.getsource(O)
    checks['no_known_answer_constants_in_optimizer'] = not any(s in source for s in ('395962678', '559457386', '435481090', '[59, 55'))
    if a.dry_check_routes:
        rows = dry_check_routes(problem, context, full, out)
        write(out / 'ROUTE_DRY_CHECK.json', rows)
        checks['route_dry_check'] = {r['method'] + ':' + r['candidate'].split(':')[-1]: r['status'] for r in rows}
    checks['wall_seconds'] = time.perf_counter() - start
    checks['execution_source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write(out / 'CHECKS.json', checks)
    print(json.dumps(O.jsonable({k: v for k, v in checks.items() if k != 'execution_source'}), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
