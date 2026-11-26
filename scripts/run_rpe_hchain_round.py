"""Provide scoped H-chain candidates to the ordinary distributed round."""
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import signal
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
    legacy = legacy_io.checked(spec["hchain"], plan, verify_files=True)
    if legacy is not None:
        legacy_io.install(plan, legacy, rows, routes, expected)
    return dict(candidates=list(rows.values()), expected=expected, legacy=legacy[1] if legacy else None)


def load_provider(directory):
    original = list(sys.path)
    try:
        sys.path.insert(0, str(directory))
        provider = importlib.import_module("run_h6_scope")
        worker = importlib.import_module("h6_scope_worker")
        if any(Path(module.__file__).resolve().parent != directory for module in (provider, worker)):
            raise ValueError("Imported H-chain provider differs from its fixed checkout")
        return provider, worker
    finally:
        sys.path[:] = original


def circuit_size(cell):
    """Rank work from the fixed schedule and preliminary circuit length."""
    events = cell["q"] * (cell["R_outer"] or 0)
    return cell["q"] * (cell["prefix"] + 1) + events, events


def batches(plan, run, worker, sizes, budgets):
    priority = worker.h8_preparation_priority(plan, run, circuit_size)
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
                rank = priority((kind, cell, index, name, plan["owner"]))
                work.append((rank, circuit_size(cell), stage, cell, index))
    work.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]["id"], entry[4]))
    pending = []
    phase_keys = 0
    # Each H8 signal uses the symmetric weights of its fixed PF formula.
    # Count an upper bound per target, even when times repeat across targets.
    h8_phase_keys = {"2nd": 1, "4th": 2, "4th(m5_best)": 6}
    for _, _, stage, cell, index in work:
        keys = (h8_phase_keys[cell["formula"]]
                if plan["scope"]["system"] == "H8" and stage == "signal" else 0)
        if pending and (pending[0][0] != stage or len(pending) == sizes.get(stage, 1)
                        or phase_keys + keys > 32):
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
        "owner", "binding", "target", "compiler", "scope", "price_cohort", "verified_saved_preparations") if k in plan}
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
                                      assets=settings["assets"], spectral_signals=settings.get("spectral_signals", False))
        contract = dict(request["original_contract"], fixed_plan_sha256=settings["plan"]["sha256"])
        definitions.append(dict(name=f"hchain-{stage}-{len(definitions)}", request=request, contract=contract,
                                approved=True, budget=budgets[stage]))
    return dict(tasks=definitions)


def preparation_priorities(graph, executor, plan, run, worker):
    """Re-rank the frozen tasks using saved completion filenames, never resize them."""
    from trottertracks.rpe_preparation_queue import preparation_job_id

    priority = worker.h8_preparation_priority(plan, run, circuit_size)
    inputs = {stage: planned_inputs(plan, stage) for stage in scope.STAGES}
    ranks = {}
    for task in graph["tasks"]:
        if task.get("dependencies"):
            raise ValueError("Unexpected H-chain preparation dependencies")
        request = task["request"]
        prep = request["preparation"]
        stage = prep["stage"]
        command = prep["command"]
        targets = json.loads(command[command.index("--targets") + 1])
        kind = "signal" if stage == "signal" else "price"
        rank = min((priority((kind, inputs[stage][cid], index, None, plan["owner"])),
                    circuit_size(inputs[stage][cid])) for cid, index in targets)
        job = preparation_job_id(request, source_manifest=executor.source_manifest, dependencies=[])
        ranks[job] = (*rank[0], *rank[1])
    return ranks


def install_ready(provider, plan, run, rows, routes, expected, spec, *, final, excluded=()):
    methods = spec["methods"]
    installed = {cid for cid, row in rows.items() if row["ready"]} | set(excluded)
    caps = spec["max_new_q3_per_method"]
    finite = [method for method in methods if caps[method] != "UNLIMITED"]
    changed = set()
    ready = (provider.select_cases(plan, [case for case in plan["cases"] if case["id"] not in installed])
             if final else provider.ready_batch(plan, run, methods, installed, require_all=finite))
    if not ready["cases"]:
        return changed
    model_dir = run / "price_models"
    model_dir.mkdir(exist_ok=True)
    models, _ = provider.price_models(ready, run, require_holdout=not final)
    for key, model in models.items():
        if model["estimate"] is not None:
            path = model_dir / (provider.digest(str(key)) + ".json")
            if path.exists():
                models[key] = provider.read(path)
            else:
                provider.retain(path, model)
    for case in ready["cases"]:
        cid, method = case["id"], case["method"]
        try:
            axes, entry, route = prepared_case(provider, run, plan, case, models, interpreter=sys.executable)
        except (OSError, ValueError, KeyError) as error:
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


def finish_block(run, results, methods):
    if set(results) == set(methods) and all(r.get("stop_reason") in HCHAIN_TERMINAL for r in results.values()):
        write(run / "SEARCH_TERMINATION.json", dict(reason="HCHAIN_SCOPE_COMPLETE",
              methods={method: row["stop_reason"] for method, row in results.items()}))


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
    provider, worker = load_provider(scope.provider_root(settings))
    full_plan = plan
    plan = provider.select_cases(plan, cases)
    routes = {method: [] for method in spec["methods"]}
    expected, results, withheld = {}, {}, {}
    selected = set(rows)
    inherited = {}
    legacy = legacy_io.checked(settings, full_plan, verify_files=True)
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
    screen = price_screen(provider, plan, run, settings, full_plan=full_plan if "case_ids" in settings else None)
    excluded = screen.update() if screen is not None else set()
    executor, optimize = make_executor(root, run, routes, expected, problem, spec)
    if legacy is not None:
        executor.prior_q3_holds.extend(row for row in legacy[1]["holds"] if row["configuration_id"] in selected)
        write(run / "HCHAIN_INHERITED.json", inherited)
    sigint = signal.signal(signal.SIGINT, lambda *_: None) if (run / "CONTROLLER_CONFIG.json").exists() else None
    result_path = run / ("ROUND_RESULT_CENTRAL.json" if partition is not None else "ROUND_RESULT.json")
    def report():
        write(result_path, optimizer.jsonable(dict(methods=results, scope_owner=plan["owner"],
              ready=[cid for cid, row in rows.items() if row["ready"]], held_tasks=withheld, inherited=inherited,
              price_pruned=sorted(excluded))))
    try:
        budgets = distributed["preparation"]["budgets"]
        graph_path = run / "distributed" / "PREPARATION_GRAPH.json"
        graph = (json.loads(graph_path.read_bytes()) if graph_path.exists()
                 else task_graph(executor, plan, run, worker, settings, budgets))
        # Preserve the frozen graph and its budget. The old H8 worker's
        # nearest-ready-candidate priority is an execution order only.
        executor.hchain_preparation_priority = preparation_priorities(graph, executor, plan, run, worker)
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
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hchain-search") as pool:
            future = None
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
                    executor.hchain_preparation_priority = preparation_priorities(graph, executor, plan, run, worker)
                    last_priority_refresh = time.monotonic()
                    priority_dirty = False
                if future is not None and future.done():
                    observed = future.result()
                    results.update(observed)
                    report()
                    if optimizer.shared_stop(observed):
                        break
                    future = None
                if future is None and refresh:
                    if screen is not None:
                        excluded = screen.update(results)
                        retire_pruned_batches(executor, pending, requests, screen.state["skip_price_cells"])
                    refresh = False
                    changed = initial_ready | install_ready(provider, plan, run, rows, routes, expected, spec,
                                                            final=not pending, excluded=excluded)
                    initial_ready.clear()
                    if changed or not pending:
                        methods = [method for method in spec["methods"] if method in changed or method not in results]
                        if methods:
                            refresh = bool(pending)  # A coarse wave may have unsubmitted ready successors.
                            prices = {cid: row["prices"] for cid, row in rows.items() if row["ready"]}
                            context = dict(candidates=[row for cid, row in rows.items() if cid not in excluded],
                                           tau=optimizer.tau(problem),
                                           expected=expected, saved=optimizer.SavedBackend([], prices),
                                           catalog={"candidates": []}, generate=None, generation={})
                            future = pool.submit(optimize, root, problem, context=context, executor=executor,
                                                 methods=methods, extra_results=extra,
                                                 max_new=spec["max_new_q3_per_method"], science_approved=True)
                if future is None and not pending:
                    break
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
            executor.finish()
        finally:
            if sigint is not None:
                signal.signal(signal.SIGINT, sigint)
    if partition is not None:
        from trottertracks.rpe_pbs_offline import publish_partition_view
        publish_partition_view(run)
    if complete and "case_ids" in settings and not (run / "STOP_REQUESTED.json").exists():
        if partition is None:
            finish_block(run, results, spec["methods"])
        elif set(results) == set(spec["methods"]) and all(r.get("stop_reason") in HCHAIN_TERMINAL for r in results.values()):
            write(run / "SEARCH_TERMINATION.json", dict(reason="HCHAIN_SCOPE_COMPLETE"))
