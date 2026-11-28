"""Provide scoped H-chain candidates to the ordinary distributed round."""
import base64
from collections import Counter
from copy import deepcopy
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from rpe_hchain_q3 import prepared_case
import rpe_hchain_legacy as legacy_io
from run_rpe_hchain_preparation import planned_inputs, preparation_request, publish_preparation
from trottertracks import rpe_hchain_scope as scope
from trottertracks import rpe_method_optimizer as optimizer
from trottertracks import rpe_scaled_shots as scaled
from trottertracks.rpe_local0_search import write


def selected_plan(settings, problem, methods):
    _, plan = scope.fixed_plan(settings)
    if (plan["binding"] != problem["problem"]["binding"] or plan["target"] != problem["target"]["energy_ha"]
            or plan["scope"]["M"] != problem["rpe"]["M"]):
        raise ValueError("H-chain plan differs from the applied problem")
    cases = [case for case in plan["cases"] if case["method"] in methods]
    if "case_ids" in settings:
        ids = settings["case_ids"]
        wanted = set(ids)
        cases = [case for case in cases if case["id"] in wanted]
        if len(wanted) != len(ids) or wanted != {case["id"] for case in cases}:
            raise ValueError("H-chain partition differs from the fixed candidate scope")
    return plan, cases


def candidate(plan, case):
    method, cid = case["method"], case["id"]
    system = plan["scope"]["system"]
    horizon = plan["scope"]["M"]
    pr = method == "PR_DF"
    key = dict(n_atoms=int(system[1:]), formula=case["formula"],
               width_over_h=str(optimizer.frac(case["width_over_h"])))
    key["L_D" if pr else "p"] = case["prefix"]
    row = dict(configuration_id=cid, name=f"{system}:{method}:{cid[:12]}", method=method,
               family="PR" if pr else "PF", key=key, caps=scaled.base_shots(horizon),
               labels=[], saved_points=[], M=horizon, ready=False, prices=None, signal_source=None,
               needs=["SIGNAL_OR_PRICE_EVIDENCE"], price_status="INCOMPLETE_HCHAIN_PREPARATION")
    if pr:
        key.update(kappa=str(optimizer.frac(case["kappa"])), K=case["K"])
        row["kappa_value"] = key["kappa"]
    return row


def preflight_context(problem, spec):
    scope.provider_root(spec["hchain"])
    plan, cases = selected_plan(spec["hchain"], problem, spec["methods"])
    rows = {case["id"]: candidate(plan, case) for case in cases}
    expected, routes = {}, {method: [] for method in spec["methods"]}
    distributed = spec["executor"]["distributed"]
    metadata_only = bool(spec["hchain"].get("legacy", {}).get("controller_metadata_only"))
    if metadata_only and (not distributed.get("pbs") or distributed.get("local_worker")
                         or (distributed.get("relay") or {}).get("workers")
                         or spec["hchain"].get("pbs_offline_search", {}).get("mode") != "independent"):
        raise ValueError("Metadata-only preflight requires a PBS-only independent scope")
    legacy = legacy_io.checked(spec["hchain"], plan, verify_files=True, metadata_only=metadata_only)
    if legacy is not None:
        legacy_io.install(plan, legacy, rows, routes, expected)
    return dict(candidates=list(rows.values()), expected=expected, legacy=legacy[1] if legacy else None)


def load_provider(directory,contract=None):
    original = list(sys.path)
    try:
        sys.path.insert(0, str(directory))
        provider = importlib.import_module((contract or {}).get("entry_module","run_h6_scope"))
        worker = importlib.import_module((contract or {}).get("worker_module","h6_scope_worker"))
        if any(Path(module.__file__).resolve().parent != directory for module in (provider, worker)):
            raise ValueError("Imported H-chain provider differs from its fixed checkout")
        return provider, worker
    finally:
        sys.path[:] = original


def circuit_size(cell):
    """Rank work from the fixed schedule and preliminary circuit length."""
    events = cell["q"] * (cell["R_outer"] or 0)
    return cell["q"] * (cell["prefix"] + 1) + events, events


def preparation_order(kind, cell, base_priority):
    """Among actual compiles prefer smaller estimated gate/event counts."""
    size = circuit_size(cell)
    return (size, base_priority) if kind == "price" else (base_priority, size)


def signal_batch_policy(plan, worker):
    """Use a provider-declared phase-cache policy, retaining fixed H8 defaults."""
    policy=getattr(worker,'SIGNAL_BATCH_POLICY',None)
    declared=(plan.get('provider_contract') or {}).get('signal_phase_cache_policy')
    if declared is not None:
        if policy is not None and policy!=declared:raise ValueError('Provider signal batch policy differs from fixed PLAN')
        policy=declared
    if policy is None:
        if plan['scope']['system']=='H8':
            policy={'phase_keys':{'2nd':1,'4th':2,'4th(m5_best)':6},'phase_key_limit':32}
        else:
            return {},None
    weights=policy['phase_keys'];limit=policy['phase_key_limit']
    if (not isinstance(weights,dict) or not weights or type(limit) is not int or limit<1
            or any(type(value) is not int or not 0<value<=limit for value in weights.values())):
        raise ValueError('Invalid provider signal phase-cache batch policy')
    return weights,limit


def batches(plan, run, worker, sizes, budgets):
    priority = getattr(worker,(plan.get('provider_contract') or {}).get('priority_function','h8_preparation_priority'))(plan,run,circuit_size)
    work = []
    for stage in scope.STAGES:
        if not budgets.get(stage):
            continue
        kind = "signal" if stage == "signal" else "price"
        for cell in planned_inputs(plan, stage).values():
            paths = plan["scope"]["price_paths"] if kind == "price" and cell["method"] == "PR" else 1
            for index in range(paths):
                name = f"signals/{cell['id']}.json" if kind == "signal" else f"prices/{cell['id']}-{index}.json"
                if (run / name).is_file():
                    continue
                rank = preparation_order(kind, cell, priority((kind, cell, index, name, plan["owner"])))
                work.append((*rank, stage, cell, index))
    work.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]["id"], entry[4]))
    pending = []
    phase_keys = 0
    # Each H8 signal uses the symmetric weights of its fixed PF formula.
    # Count an upper bound per target, even when times repeat across targets.
    phase_weights,phase_limit=signal_batch_policy(plan,worker)
    for _, _, stage, cell, index in work:
        keys=0
        if stage=='signal' and phase_limit is not None:
            if cell['formula'] not in phase_weights:
                raise ValueError('Provider has no signal phase-cache policy for formula: '+cell['formula'])
            keys=phase_weights[cell['formula']]
        if pending and (pending[0][0] != stage or len(pending) == sizes.get(stage, 1)
                        or (phase_limit is not None and phase_keys + keys > phase_limit)):
            yield pending
            pending = []
            phase_keys = 0
        pending.append((stage, cell, index))
        phase_keys += keys
    if pending:
        yield pending


def task_graph(executor, plan, run, worker, settings, budgets):
    definitions = []
    metadata = {k: plan[k] for k in (
        "owner", "binding", "target", "compiler", "scope", "price_cohort", "verified_saved_preparations", "provider_contract") if k in plan}
    for batch in batches(plan, run, worker, settings["batch_sizes"], budgets):
        stage = batch[0][0]
        cells = {cell["id"]: cell for _, cell, _ in batch}
        packet = dict(metadata, cases=[], designs=[])
        if stage == "signal":
            packet["cases"] = [dict(rounds=list(cells.values()))]
        elif stage == "compile":
            packet["designs"] = [dict(anchors=list(cells.values()), holdout=None)]
        else:
            packet["designs"] = [dict(anchors=[], holdout=cell) for cell in cells.values()]
        raw = json.dumps(packet, separators=(",", ":"), sort_keys=True).encode()
        checksum = hashlib.sha256(raw).hexdigest()
        executor.artifact_cache.put({checksum: base64.b64encode(raw).decode()})
        targets = [[cell["id"], index] for _, cell, index in batch]
        request = preparation_request(packet, checksum, stage, targets, source=settings["provider_source"],
                                      assets=settings["assets"], spectral_signals=settings.get("spectral_signals", False),
                                      openblas_coretype=settings.get("openblas_coretype"),fixed_plan_sha256=settings["plan"]["sha256"])
        contract = dict(request["original_contract"], fixed_plan_sha256=settings["plan"]["sha256"])
        definitions.append(dict(name=f"hchain-{stage}-{len(definitions)}", request=request, contract=contract,
                                approved=True, budget=budgets[stage]))
    return dict(tasks=definitions)


def approximate_price_priorities(plan, worker, screen):
    """Best already-available approximate RZ of each price dependency.

    No new estimate, evaluation or pruning is introduced. A missing estimate
    does not exclude the cell. The existing fixed provider defines dependency
    identity (direct anchors versus shared fitted anchors/holdout).
    """
    estimates = getattr(getattr(screen, "approximation", None), "estimates", {})
    if not estimates:
        return {}
    from fractions import Fraction
    excluded = getattr(screen, "state", {}).get("excluded", {})
    designs = {worker.design_key(d): d for d in plan["designs"]}
    result = {}
    for case in plan["cases"]:
        if case["id"] in excluded or case["id"] not in estimates:
            continue
        value = estimates[case["id"]].get("estimated_total_RZ")
        try:
            cost = Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else Fraction(value)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            continue
        if cost <= 0: continue
        for cell in case["rounds"]:
            design = designs.get(worker.group_key(cell))
            if not design: continue
            dependencies = (x for x in design["anchors"] if x["id"] == cell["id"]) if design["mode"] == "DIRECT" else (*design["anchors"], design.get("holdout"))
            for dep in dependencies:
                if dep is not None:
                    key=dep["id"]
                    result[key]=min(result.get(key,cost),cost)
    return result


def dispatch_preparation_order(kind, cell, base_priority, cost=None, *, small_price_size=64):
    """Short prices first; compare saved approximate total RZ for larger prices.

    The threshold changes ordering only, never eligibility or the fixed budget.
    Unknown costs fall back to the existing size and nearest-ready ordering.
    """
    size = circuit_size(cell)
    if kind == "signal":
        return (0, *base_priority, *size)
    if size[0] <= small_price_size:
        return (1, *size, *base_priority)
    if cost is not None:
        return (2, 0, cost, *size, *base_priority)
    return (2, 1, *size, *base_priority)


def install_preparation_priorities(executor, ranks):
    if hasattr(executor, 'set_hchain_preparation_priority'):
        executor.set_hchain_preparation_priority(ranks)
    else:
        executor.hchain_preparation_priority = ranks


def preparation_priority_descriptors(graph, executor):
    from trottertracks.rpe_preparation_queue import preparation_job_id
    descriptors = []
    for task in graph['tasks']:
        if task.get('dependencies'):
            raise ValueError('Unexpected H-chain preparation dependencies')
        request = task['request']; prep = request['preparation']
        command = prep['command']
        targets = json.loads(command[command.index('--targets')+1])
        job = preparation_job_id(request, source_manifest=executor.source_manifest, dependencies=[])
        descriptors.append((job, prep['stage'], targets))
    return descriptors


def preparation_priorities(graph, executor, plan, run, worker, screen=None, *, small_price_size=64, descriptors=None):
    """Re-rank the frozen tasks using saved completion filenames, never resize them."""
    priority = getattr(worker,(plan.get('provider_contract') or {}).get('priority_function','h8_preparation_priority'))(plan,run,circuit_size)
    inputs = {stage: planned_inputs(plan, stage) for stage in scope.STAGES}
    costs = approximate_price_priorities(plan, worker, screen)
    ranks = {}
    if descriptors is None:
        descriptors = preparation_priority_descriptors(graph, executor)
    for job, stage, targets in descriptors:
        kind = 'signal' if stage == 'signal' else 'price'
        rank = min(dispatch_preparation_order(kind, inputs[stage][cid],
                    priority((kind, inputs[stage][cid], index, None, plan['owner'])), costs.get(cid),
                    small_price_size=small_price_size) for cid, index in targets)
        ranks[job] = rank
    return ranks

def install_ready(provider, plan, run, rows, routes, expected, spec, *, final, excluded=()):
    methods = spec["methods"]
    installed = {cid for cid, row in rows.items() if row["ready"]} | set(excluded)
    caps = spec["max_new_q3_per_method"]
    finite = [method for method in methods if caps[method] != "UNLIMITED"]
    changed = set()
    remaining=[case for case in plan['cases'] if case['id'] not in installed]
    try:
        ready = (provider.select_cases(plan,remaining) if final else
                 provider.ready_batch(plan, run, methods, installed, require_all=finite))
    except (OSError,subprocess.CalledProcessError):
        # Preserve a finite method's all-ready contract. Independent unlimited
        # candidate metadata may be recovered separately after a local IO
        # error. Do not catch a whole-plan binding/identity mismatch here.
        recovered=[]
        for method in methods:
            cases=[case for case in remaining if case['method']==method]
            batches=[cases] if method in finite else [[case] for case in cases]
            for batch in batches:
                if not batch:continue
                try:
                    subset=provider.select_cases(plan,batch)
                    observed=subset if final else provider.ready_batch(subset,run,[method],installed,
                                                                       require_all=[method] if method in finite else [])
                    recovered.extend(observed['cases'])
                except (OSError,subprocess.CalledProcessError) as error:
                    for case in batch:rows[case['id']]['preparation_hold']=str(error)
        priority=plan.get('exploration_priority')
        if recovered and not final and not finite and priority:
            tier=min(priority[case['id']] for case in recovered)
            recovered=[case for case in recovered if priority[case['id']]==tier]
        ready=provider.select_cases(plan,recovered)
    if not ready["cases"]:
        return changed
    model_dir = run / "price_models"
    model_dir.mkdir(exist_ok=True)
    def models_for(subset):
        models,_=provider.price_models(subset,run,require_holdout=not final)
        for key,model in list(models.items()):
            try:
                if model['estimate'] is not None:
                    path=model_dir/(provider.digest(str(key))+'.json')
                    if path.exists():models[key]=provider.read(path)
                    else:provider.retain(path,model)
            except (OSError,ValueError,KeyError,subprocess.CalledProcessError):
                models.pop(key)  # never replace a saved model with a new one
        return models
    try:
        shared_models=models_for(ready)
    except (OSError,subprocess.CalledProcessError):
        shared_models=None
    for case in ready["cases"]:
        cid, method = case["id"], case["method"]
        try:
            models=shared_models if shared_models is not None else models_for(provider.select_cases(ready,[case]))
            axes, entry, route = prepared_case(provider, run, plan, case, models, interpreter=sys.executable)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            rows[cid]["preparation_hold"] = str(error)
            continue
        rows[cid].update(ready=True, needs=[], prices=[optimizer.frac(axis["unit_price_RZ"]) for axis in axes],
                         signal_source=entry["signal_packet"], price_axes=axes,
                         price_status="SOURCE_BOUND_HCHAIN_PRICES")
        expected[cid] = dict(M=rows[cid]["M"], signal_source=entry["signal_packet"])
        routes[method].append(route)
        changed.add(method)
    return changed


def price_screen(provider, plan, run, settings, *, full_plan=None):
    if not settings.get("saved_price_screen"):
        return None
    if plan.get("scope", {}).get("system") == "H6" and settings.get("legacy") and full_plan is not None:
        # Preserve the already accepted H6 screening, rather than fitting a new
        # model or introducing additional exclusions on a partition boundary.
        path = Path(settings["legacy"]["directory"]) / "PURE_RTE_COST_SCREEN.json"
        if path.is_file():
            from types import SimpleNamespace
            record = json.loads(path.read_bytes())
            if record["scope_owner"] != full_plan["owner"] or record["input_sha256"] != full_plan["pure_rte_screen_sha256"]:
                raise ValueError("H6 saved price screening belongs to another plan")
            ids = {case["id"] for case in plan["cases"]}
            excluded = ids.intersection(record["excluded"])
            retained = provider.select_cases(plan,[case for case in plan["cases"] if case["id"] not in excluded])
            required = {cell["id"] for stage in ("compile","audit") for cell in planned_inputs(retained,stage).values()}
            skipped = set(record["skip_price_cells"]) - required
            state = dict(skip_price_cells=sorted(skipped))
            write(run / "PURE_RTE_COST_SCREEN_INHERITED.json",dict(source=str(path),excluded=sorted(excluded),skip_price_cells=sorted(skipped)))
            return SimpleNamespace(state=state, update=lambda *_: excluded)
    path = Path(provider.__file__).with_name(provider.pure_rte.FORECAST)
    forecast = provider.read(path) if path.is_file() else None
    if forecast is not None and forecast["comparison_id"] != provider.pure_rte.comparison_id(plan):
        if full_plan is None or forecast["comparison_id"] != provider.pure_rte.comparison_id(full_plan):
            forecast = None
        else:
            # Validate the original comparison before narrowing it. The
            # estimates and scientific parameters are unchanged; no new model.
            ids = {case["id"] for case in plan["cases"]}
            reference = forecast["incumbent"]
            forecast = dict(forecast, original_comparison_id=forecast["comparison_id"],
                comparison_id=provider.pure_rte.comparison_id(plan),
                estimates=[row for row in forecast["estimates"] if row["configuration_id"] in ids],
                incumbent=reference if reference and reference["configuration_id"] in ids else None)
    return provider.build_price_pruning(plan, run, forecast=forecast)


def pruned_price_batch(request, skip_cells):
    preparation = request["preparation"]
    if preparation["stage"] not in ("compile", "audit"):
        return False
    prices = [Path(name).stem.rsplit("-", 1)[0] for name in preparation["outputs"] if name.endswith(".json")]
    return bool(prices) and all(cell in skip_cells for cell in prices)


def retire_pruned_batches(executor, pending, requests, skip_cells):
    with executor.state.cv:
        for job in list(pending):
            if not pruned_price_batch(requests[job], skip_cells):
                continue
            row = executor.store.row(job)
            if row is None:
                executor.discard_pending_preparation(job)
            elif row["state"] == "QUEUED":
                task = executor.state.tasks.get(executor.run_dir / "jobs" / row["method"] / job)
                if task is None or task.status != "QUEUED" or row["detail"].get("prelaunch_recovery"):
                    continue
                executor._cancel(task)
            elif row["state"] != "CANCELLED":
                continue
            pending.remove(job)
            executor.log(event="HCHAIN_PRICE_PRUNED", job=job)


def partition_for_run(run, spec):
    from trottertracks.rpe_pbs_offline import prepare_plan
    settings = spec["hchain"].get("pbs_offline_search")
    if "case_ids" in spec["hchain"] or not settings or settings.get("mode") != "independent":
        return None
    marker = run / "PBS_PARTITION_PLAN.json"
    if marker.is_file():
        partition = json.loads(marker.read_bytes())
        if partition["settings"]["source"] != settings["source"]:
            raise ValueError("Independent H-chain run retains its pinned optimizer")
        return partition
    if (run / "distributed" / "PREPARATION_GRAPH.json").exists():
        return None
    partition = prepare_plan(run, settings, round_spec=spec)
    if partition is not None:
        write(marker, partition)
    return partition


def wait_partition(run, executor):
    from trottertracks.rpe_pbs_offline import publish_partition_view
    manager = executor.pbs_manager
    if manager is None:
        raise ValueError("Independent H-chain partition requires a PBS manager")
    while not executor._user_stop_requested():
        publish_partition_view(run)
        if manager.independent_finished():
            return True
        if manager.detached or manager.thread is None or not manager.thread.is_alive():
            return False
        with executor.state.cv:
            if executor.state.failure:
                raise RuntimeError(executor.state.failure)
            executor.state.cv.wait(2)
    return False


HCHAIN_TERMINAL = {"SCALED_SHOT_GRID_COMPLETE", "EXPLORATORY_GRID_COMPLETE",
                   "Q3_BUDGET_EXHAUSTED", "STEP_LIMIT"}


def finish_block(run, results, methods, *, exhausted=None):
    if set(results) == set(methods) and all(r.get("stop_reason") in HCHAIN_TERMINAL for r in results.values()):
        write(run / "SEARCH_TERMINATION.json", dict(reason="HCHAIN_SCOPE_COMPLETE",
              methods={method: row["stop_reason"] for method, row in results.items()}))
    elif (exhausted and set(results)==set(methods)
          and any(r.get('stop_reason')=='WAITING_FOR_INPUTS' for r in results.values())
          and all(r.get('stop_reason') in HCHAIN_TERMINAL|{'WAITING_FOR_INPUTS'} for r in results.values())):
        # Operational terminal, never a claim that held science completed.
        write(run/'SEARCH_TERMINATION.json',dict(reason='HCHAIN_INPUTS_EXHAUSTED',
              methods={method:row['stop_reason'] for method,row in results.items()},
              withheld=exhausted,scientific_scope_complete=False))


def stream_ready_in_wave(screen, settings, spec, executor):
    """A screened wave may only stream with an explicit new ordering policy.

    Finite Q3 budgets retain the original all-ready ordering; the exclusion
    set is frozen until the active selector returns.
    """
    if not getattr(executor,'supports_nonblocking_prefetch',False):
        return False
    if screen is None:
        return True
    return (settings.get('screened_ready_streaming') is True
            and all(spec['max_new_q3_per_method'].get(method)=='UNLIMITED'
                    for method in spec['methods']))


class HChainReadySource:
    """Publish immutable late READY candidates through the shared selector API."""
    def __init__(self, executor, rows):
        self.executor = executor
        self.lock = RLock()
        self.sent = {cid for cid, row in rows.items() if row['ready']}
        self.pending = {}
        self.retry_after = {}
        self.routes = {}
        self.expected = {}

    def publish(self, rows, routes, expected, excluded):
        with self.lock:
            for cid, row in rows.items():
                if (row['ready'] and cid not in self.sent and cid not in excluded
                        and time.monotonic()>=self.retry_after.get(cid,0)):
                    self.retry_after.pop(cid,None)
                    self.pending.setdefault(row['method'], {})[cid] = deepcopy(row)
                    self.sent.add(cid)
            self.routes = {method:list(rules) for method,rules in routes.items()}
            self.expected = dict(expected)
        with self.executor.state.cv:
            self.executor.state.cv.notify_all()

    def retry_due(self):
        with self.lock:return any(time.monotonic()>=deadline for deadline in self.retry_after.values())

    def has_new(self, method):
        with self.lock:
            return bool(self.pending.get(method))

    def subscribe(self, method, context, candidates, view):
        def pending():
            return self.has_new(method)
        def take():
            with self.lock:
                rows = list(self.pending.pop(method, {}).values())
                routes = self.routes.copy()
                expected = self.expected.copy()
            trial = view.for_method(method)
            trial.routes, trial.expected = routes, expected
            new_routes, added = [], []
            bindings = dict(view.expected)
            for row in rows:
                cid = row['configuration_id']
                route = trial.route(cid)
                if route is None or trial.refresh_execution_hold(cid):
                    # Not delivered. A later publication may resolve the
                    # route or hold; do not suppress that candidate forever.
                    with self.lock:
                        self.sent.discard(cid)
                        self.retry_after[cid]=time.monotonic()+1
                    continue
                row['execution_needs'] = []
                row.pop('execution_hold', None)
                new_routes.append(dict(route, configuration_ids=[cid]))
                bindings[cid] = trial.expected_for(cid, route)
                added.append(row)
            view.routes = dict(view.routes, **{method:new_routes+view.routes.get(method,[])})
            view.expected = bindings
            return added, None
        return pending, take


def run_round(root, run, problem, spec, extra, *, make_executor, plan_only=False, partition=None):
    settings = spec["hchain"]
    plan, cases = selected_plan(settings, problem, spec["methods"])
    rows = {case["id"]: candidate(plan, case) for case in cases}
    if plan_only:
        run.mkdir(parents=True, exist_ok=True)
        write(run / "HCHAIN_PLAN.json", dict(owner=plan["owner"], scientific_calls=0,
            candidates=dict(Counter(row["method"] for row in rows.values()))))
        return
    if spec.get("science_approved") is not True:
        raise ValueError("Use the ordinary Start or Resume for an executing H-chain round")
    if partition is None:
        partition = partition_for_run(run, spec)
        if partition is not None:
            from trottertracks.rpe_pbs_offline import restrict_round, publish_partition_view
            local = [part for part in partition["parts"] if part["owner"] == "central"]
            if local:
                local_spec = restrict_round(partition, local)
                return run_round(root, run, problem, local_spec, extra,
                                 make_executor=make_executor, partition=partition)
            runtime = dict(spec, executor=dict(spec["executor"], distributed=dict(
                spec["executor"]["distributed"], pbs_offline_plan=partition)))
            executor, _ = make_executor(root, run, {m: [] for m in spec["methods"]}, {}, problem, runtime)
            try:
                write(run / "ROUND_RESULT_CENTRAL.json", dict(methods={}, scope_owner=plan["owner"]))
                complete = wait_partition(run, executor)
            finally:
                executor.finish()
            publish_partition_view(run)
            if complete and not (run / "STOP_REQUESTED.json").exists():
                write(run / "SEARCH_TERMINATION.json", dict(reason="HCHAIN_SCOPE_COMPLETE"))
            return
    provider, worker = load_provider(scope.provider_root(settings), *( [plan["provider_contract"]] if plan.get("provider_contract") else []))
    full_plan = plan
    plan = provider.select_cases(plan, cases)
    routes = {method: [] for method in spec["methods"]}
    expected, results, withheld = {}, {}, {}
    selected = set(rows)
    inherited = {}
    legacy = legacy_io.checked(settings, full_plan, verify_files=True,
                               receipt_plan=plan if "case_ids" in settings else None,
                               cache_path=run/"LEGACY_HASH_CACHE.json")
    if legacy is not None:
        if "case_ids" in settings:
            directory, history = legacy
            records = [r for r in history["records"] if r["configuration_id"] in selected]
            holds = [r for r in history["holds"] if r["configuration_id"] in selected]
            spent = dict(Counter(rows[r["configuration_id"]]["method"] for r in (*records, *holds)))
            legacy = directory, dict(history, records=records, holds=holds, spent=spent)
        legacy_io.restore_receipts(run, legacy, plan=plan if "case_ids" in settings else None)
        legacy_io.install(plan, legacy, rows, routes, expected)
        history = legacy[1]
        inherited = dict(source_job=history["source_job"], spent=history["spent"],
                         recovered=len(history["records"]), held=len(history["holds"]))
        extra = [*extra, *(record for record in history["records"] if record["configuration_id"] in selected)]
        spec = dict(spec, max_new_q3_per_method=legacy_io.remaining(spec["max_new_q3_per_method"], history))
    distributed = spec["executor"]["distributed"]
    if partition is not None:
        distributed = dict(distributed, pbs_offline_plan=partition)
        spec = dict(spec, executor=dict(spec["executor"], distributed=distributed))
    if distributed.get("live_q3_readiness") is True:
        distributed = dict(distributed)
        # Dynamically generated requests need their own real-input readiness check.
        distributed.setdefault("host_eligibility", {"schema": "RPE_Q3_HOST_ELIGIBILITY_V1", "jobs": {}})
        pbs = distributed.get("pbs")
        if pbs:
            from trottertracks.rpe_pbs import namespaced_remote_root, prefix_cache_path
            distributed.setdefault("pbs_pinned_settings", pbs)
            remote_root = namespaced_remote_root(pbs, run)
            workers = []
            for declaration in pbs["workers"]:
                profile = dict(declaration["profile"])
                profile.setdefault("prefix_cache_directory", prefix_cache_path(remote_root, declaration["host_id"]))
                workers.append(dict(declaration, profile=profile))
            distributed["pbs"] = dict(pbs, workers=workers)
        spec = dict(spec, executor=dict(spec["executor"], distributed=distributed))
    screen = (price_screen(provider, plan, run, settings, full_plan=full_plan)
              if "case_ids" in settings else price_screen(provider, plan, run, settings))
    excluded = screen.update() if screen is not None else set()
    if legacy is not None:
        legacy_io.seed_local_worker(legacy, distributed)
    # Runtime-only start fence: the durable graph and scientific scope never change.
    distributed = dict(distributed, hchain_initialize_priority=True)
    spec = dict(spec, executor=dict(spec["executor"], distributed=distributed))
    executor, optimize = make_executor(root, run, routes, expected, problem, spec)
    if legacy is not None:
        executor.prior_q3_holds.extend(row for row in legacy[1]["holds"] if row["configuration_id"] in selected)
        write(run / "HCHAIN_INHERITED.json", inherited)
    sigint = signal.signal(signal.SIGINT, lambda *_: None) if (run / "CONTROLLER_CONFIG.json").exists() else None
    result_path = run / ("ROUND_RESULT_CENTRAL.json" if partition is not None else "ROUND_RESULT.json")
    def report():
        write(result_path, optimizer.jsonable(dict(methods=results, scope_owner=plan["owner"],
              ready=[cid for cid, row in rows.items() if row["ready"]], held_tasks=withheld, inherited=inherited,
              price_pruned=sorted(excluded))), compact=True, only_changed=True)
    try:
        budgets = distributed["preparation"]["budgets"]
        graph_path = run / "distributed" / "PREPARATION_GRAPH.json"
        from trottertracks.rpe_preparation_queue import load_preparation_graph
        graph = (load_preparation_graph(graph_path) if graph_path.exists()
                 else task_graph(executor, plan, run, worker, settings, budgets))
        # Preserve the frozen graph and its budget. The old H8 worker's
        # nearest-ready-candidate priority is an execution order only.
        small_price_size = settings.get("small_price_priority_size", 64)
        if type(small_price_size) is not int or small_price_size < 1:
            raise ValueError("small_price_priority_size must be positive")
        priority_descriptors = preparation_priority_descriptors(graph, executor)
        ranks = preparation_priorities(graph, executor, plan, run, worker, screen,
            small_price_size=small_price_size, descriptors=priority_descriptors)
        last_priority_refresh = time.monotonic()
        priority_dirty = False
        if screen is not None:
            def skip(request):
                return pruned_price_batch(request, screen.state["skip_price_cells"])
            mapping = executor.configure_preparation_graph(graph, skip=skip)
            requests = {mapping[task["name"]]: task["request"] for task in graph["tasks"]}
            pending = {job for job, request in requests.items() if not skip(request)
                       or (executor.store.row(job) or {}).get("state") not in (None, "CANCELLED")}
            retire_pruned_batches(executor, pending, requests, screen.state["skip_price_cells"])
        else:
            pending = set(executor.configure_preparation_graph(graph).values())
        install_preparation_priorities(executor, ranks)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hchain-search") as pool:
            future = None
            live_source = None
            idle_marker = run/'HCHAIN_IDLE_WAIT.json'
            idle_marker.unlink(missing_ok=True)
            last_idle_check = 0.
            last_ready_install = float("-inf")
            refresh = True
            initial_ready = {row["method"] for row in rows.values() if row["ready"]
                             and spec["max_new_q3_per_method"][row["method"]] == "UNLIMITED"}
            while not executor._user_stop_requested():
                for job in executor.ready_preparations(pending, wait=False, stop_requested=executor._user_stop_requested):
                    receipt = publish_preparation(executor, job, run)
                    if receipt is None:
                        continue
                    pending.remove(job)
                    priority_dirty = True
                    refresh = True
                    if receipt["status"] != "PREPARATION_COMPLETE":
                        withheld[job] = receipt
                if priority_dirty and time.monotonic() - last_priority_refresh >= 30:
                    install_preparation_priorities(executor, preparation_priorities(graph, executor, plan, run, worker, screen,
                        small_price_size=small_price_size, descriptors=priority_descriptors))
                    last_priority_refresh = time.monotonic()
                    priority_dirty = False
                if future is not None and future.done():
                    observed = future.result()
                    results.update(observed)
                    report()
                    if optimizer.shared_stop(observed):
                        break
                    if live_source is not None:
                        if screen is not None:
                            refresh=True  # update screening only after the frozen wave ends
                        # A selector that already exited before observing a
                        # publication is restarted in the ordinary next wave.
                        initial_ready.update(method for method in spec['methods'] if live_source.has_new(method))
                    future = None
                    live_source = None
                if (future is not None and live_source is not None and refresh
                        and (not pending or time.monotonic()-last_ready_install>=1)):
                    last_ready_install=time.monotonic()
                    # The opt-in screened wave uses its original exclusion
                    # set; never re-screen while the selector is running.
                    updated_routes = {method:list(items) for method,items in routes.items()}
                    updated_expected = dict(expected)
                    changed = install_ready(provider, plan, run, rows, updated_routes, updated_expected, spec,
                                             final=not pending, excluded=excluded)
                    refresh = False
                    if changed:
                        routes, expected = updated_routes, updated_expected
                        with executor.state.cv:
                            executor.routes, executor.expected = routes, expected
                    live_source.publish(rows, routes, expected, excluded)
                elif future is not None and live_source is not None and live_source.retry_due():
                    # Recheck a held route without rereading the signal/price tree.
                    live_source.publish(rows, routes, expected, excluded)
                if future is None and (refresh or initial_ready):
                    # Explicit inherited exact replays use the same ordinary
                    # nonblocking owner reconciliation as schedule searches.
                    # Legacy holds without replay permission are untouched.
                    if any(hold.get('replay_exact') is True for hold in getattr(executor,'prior_q3_holds',())):
                        executor.replay_prior_q3(wait=False)
                    if screen is not None:
                        excluded = screen.update(results)
                        retire_pruned_batches(executor, pending, requests, screen.state["skip_price_cells"])
                        priority_dirty = True
                    refresh = False
                    last_ready_install=time.monotonic()
                    changed = initial_ready | install_ready(provider, plan, run, rows, routes, expected, spec,
                                                            final=not pending, excluded=excluded)
                    initial_ready.clear()
                    if changed or not pending:
                        streaming=stream_ready_in_wave(screen,settings,spec,executor)
                        methods = list(spec['methods']) if streaming and changed else [
                            method for method in spec['methods'] if method in changed or method not in results]
                        if methods:
                            refresh = bool(pending)  # A coarse wave may have unsubmitted ready successors.
                            prices = {cid: row["prices"] for cid, row in rows.items() if row["ready"]}
                            context = dict(candidates=deepcopy([row for cid, row in rows.items() if cid not in excluded]),
                                           tau=optimizer.tau(problem),
                                           expected=dict(expected), saved=optimizer.SavedBackend([], prices),
                                           catalog={"candidates": []}, generate=None, generation={})
                            options = {}
                            if streaming:
                                live_source = HChainReadySource(executor, rows)
                                options['ready_source'] = live_source
                            future = pool.submit(optimize, root, problem, context=context, executor=executor,
                                                 methods=methods, extra_results=extra,
                                                 max_new=spec["max_new_q3_per_method"], science_approved=True, **options)
                if future is None and not pending:
                    break
                if 'case_ids' in settings and hasattr(executor.store,'idle_revision'):
                    # Only the independent package driver knows that no
                    # selector is active. The scheduler rechecks the queue
                    # revision and owners before issuing a normal handoff.
                    if future is not None or getattr(executor,'preparation_pending',{}):
                        idle_marker.unlink(missing_ok=True)
                    elif time.monotonic()-last_idle_check >= 5:
                        last_idle_check = time.monotonic()
                        revision = executor.store.idle_revision()
                        if revision is None:
                            idle_marker.unlink(missing_ok=True)
                        elif not idle_marker.is_file():
                            write(idle_marker,dict(queue_revision=revision,reason='NO_RESERVED_WORK'))
                        elif json.loads(idle_marker.read_bytes()).get('queue_revision') != revision:
                            write(idle_marker,dict(queue_revision=revision,reason='NO_RESERVED_WORK'))
                with executor.state.cv:
                    if executor.state.failure:
                        raise RuntimeError(executor.state.failure)
                    executor.state.cv.wait(.25)
            if future is not None:
                results.update(future.result())
        for method, result in results.items():
            if result["stop_reason"] != "UNKNOWN_RESULT" and any(
                    row["method"] == method and not row["ready"] and cid not in excluded for cid, row in rows.items()):
                result.update(hchain_search_stop=result["stop_reason"], stop_reason="WAITING_FOR_INPUTS")
        report()
        if partition is not None and not optimizer.shared_stop(results):
            complete = wait_partition(run, executor)
        else:
            complete = partition is None
    finally:
        try:
            (run/'HCHAIN_IDLE_WAIT.json').unlink(missing_ok=True)
            executor.finish()
        finally:
            if sigint is not None:
                signal.signal(signal.SIGINT, sigint)
    if partition is not None:
        from trottertracks.rpe_pbs_offline import publish_partition_view
        publish_partition_view(run)
    if complete and "case_ids" in settings and not (run / "STOP_REQUESTED.json").exists():
        if partition is None:
            exhausted=None
            if not pending and withheld and hasattr(executor.store,'idle_revision') and executor.store.idle_revision() is not None:
                exhausted=dict(Counter(receipt.get('status','UNKNOWN') for receipt in withheld.values()))
            finish_block(run, results, spec["methods"],exhausted=exhausted)
        elif set(results) == set(spec["methods"]) and all(r.get("stop_reason") in HCHAIN_TERMINAL for r in results.values()):
            write(run / "SEARCH_TERMINATION.json", dict(reason="HCHAIN_SCOPE_COMPLETE"))
