"""Plan and run H6 parameter scopes on PBS using the common RPE shot optimizer."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from h6_scope import AXES, LABELS, case_rows, digest, enc, exact, group_key, read
from h6_scope_worker import read_price, write
from h6_preparation_pool import GIB


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks import price_model_validation as linear
from trottertracks import rpe_method_optimizer as optimizer
from trottertracks import rpe_optimizer_concurrent as concurrent
from trottertracks import rpe_scaled_shots as scaled


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(cell):
    q = cell["q"]
    return {"one": 1, "q": q, "qR": q * (cell["R_outer"] or 0)}


def observation(plan, cell, run, axis):
    n_paths = plan["scope"]["price_paths"] if cell["method"] == "PR" else 1
    values, sources = [], []
    for i in range(n_paths):
        path = run / "prices" / f"{cell['id']}-{i}.json"
        receipt = read_price(path)
        if (receipt["candidate"]["id"] != cell["id"] or receipt["index"] != i
                or receipt["owner"] != plan["owner"] or receipt["compiler"] != plan["compiler"]
                or receipt["candidate"] != cell
                or type(receipt["axes"][axis]["rz"]) is not int or receipt["axes"][axis]["rz"] <= 0):
            raise ValueError("H6 calibration receipt identity differs")
        values.append(receipt["axes"][axis]["rz"])
        sources.append(dict(path=str(path), sha256=sha(path)))
    return Fraction(sum(values), len(values)), sources


def design_key(design):
    first = next(iter(design["anchors"]), design.get("reference"))
    return group_key(dict(first, actual_step=design["target_step"]))


def ready_batch(plan, run, methods, completed, require_all=()):
    """Return candidate inputs whose direct prices or entire fitted group have arrived."""
    signals = {p.stem for p in (run / "signals").glob("*.json")}
    prices = {p.stem for p in (run / "prices").glob("*.json")}

    def covered(cell):
        paths = plan["scope"]["price_paths"] if cell["method"] == "PR" else 1
        return all(f"{cell['id']}-{i}" in prices for i in range(paths))

    direct, fitted = set(), set()
    for design in plan["designs"]:
        if design["mode"] == "DIRECT":
            direct.update(cell["id"] for cell in design["anchors"] if covered(cell))
        else:
            dependencies = [cell for cell in (*design["anchors"], design["holdout"]) if cell is not None]
            if all(covered(cell) for cell in dependencies):
                fitted.add(design_key(design))
    cases = [case for case in plan["cases"] if case["id"] not in completed and case["method"] in methods
             and all(cell["id"] in signals and (cell["id"] in direct or group_key(cell) in fitted)
                     for cell in case["rounds"])]
    outstanding = Counter(case["method"] for case in plan["cases"] if case["id"] not in completed)
    ready_counts = Counter(case["method"] for case in cases)
    cases = [case for case in cases if case["method"] not in require_all
             or ready_counts[case["method"]] == outstanding[case["method"]]]
    keys = {group_key(cell) for case in cases for cell in case["rounds"]}
    used = {cell["id"] for case in cases for cell in case["rounds"]}
    designs = []
    for design in plan["designs"]:
        if design_key(design) not in keys:
            continue
        if design["mode"] == "DIRECT":
            design = dict(design, anchors=[cell for cell in design["anchors"] if cell["id"] in used])
        designs.append(design)
    return dict(plan, cases=cases, designs=designs)


def price_models(plan, run, *, require_holdout=False):
    models, errors = {}, {}
    for design in plan["designs"]:
        key = design_key(design)
        calibration_step = design["calibration_step"]
        names = ("one", "q", "qR") if design["mode"] == "AFFINE_QR" else ("one", "q")
        for axis in AXES:
            try:
                if calibration_step != design["target_step"]:
                    raise ValueError("H6 price calibration must match the target physical step")
                direct = {}
                for cell in design["anchors"]:
                    if group_key(cell) != key:
                        raise ValueError("H6 calibration anchor belongs to a different price group")
                    try:
                        mean, proof = observation(plan, cell, run, axis)
                    except (OSError, ValueError, KeyError) as error:
                        if design["mode"] != "DIRECT":
                            raise
                        errors[str((key, axis, cell["id"]))] = str(error)
                        continue
                    direct[cell["id"]] = dict(price=enc(mean), source=proof)
                holdout = design["holdout"]
                if holdout is not None and group_key(holdout) != key:
                    raise ValueError("H6 holdout belongs to a different price group")
                if design["mode"] == "DIRECT":
                    models[key, axis] = dict(direct=direct, estimate=None, holdout=None)
                    if design.get("reason"):
                        errors[str((key, axis))] = design["reason"]
                    continue
                route = dict(binding=plan["binding"], mode=key[0], formula=key[1], prefix=key[2],
                             actual_step=calibration_step, K=key[4])
                pr = key[0] == "PR"
                records = [
                    linear.record(cell_id=cell["id"], role="CALIBRATION", route=route, axis=axis,
                        features={name: features(cell)[name] for name in names},
                        rz_count=direct[cell["id"]]["price"],
                        kind="DIRECT_COMPILED_FINITE_SAMPLE" if pr else "DIRECT_COMPILED_DETERMINISTIC",
                        source=direct[cell["id"]]["source"], implementation={"lowering_id": "symmetric_directional"},
                        profile={"compiler": plan["compiler"]}, n_paths=plan["scope"]["price_paths"] if pr else None,
                        law={"model": "ideal_finite_RTE_law", "K": key[4]} if pr else None)
                    for cell in design["anchors"]
                ]
                estimate = linear.fit(records, features=names,
                                      design_source={"maximum_direct_round": plan["scope"]["calibration_max_m"],
                                                     "target_step": design["target_step"]})
                estimate["extrapolation_scope"] = "CONDITIONAL_SAME_STEP"
                comparison = None
                if holdout is not None:
                    try:
                        observed, proof = observation(plan, holdout, run, axis)
                        predicted = linear.predict(estimate, features(holdout))
                        direct[holdout["id"]] = dict(price=enc(observed), source=proof)
                        comparison = dict(observed=enc(observed), predicted=enc(predicted),
                            relative_error=enc((predicted-observed)/observed), source=proof, refit=False)
                    except (OSError, ValueError) as error:
                        if require_holdout:
                            raise
                        comparison = dict(status="HOLDOUT_UNAVAILABLE", reason=str(error), refit=False)
                models[key, axis] = dict(direct=direct, estimate=estimate, holdout=comparison)
            except (OSError, ValueError, KeyError) as error:
                errors[str((key, axis))] = str(error)
    return models, errors


def retain(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f"Frozen H6 input changed: {path.name}")
    else:
        write(path, data)


def candidate_inputs(run, plan, case, models):
    cid, method = case["id"], case["method"]
    folder = run / "cases" / cid
    folder.mkdir(parents=True, exist_ok=True)
    scope = plan["scope"]
    h = float(exact(scope["h"]))
    M = scope["M"]
    axes, signals, canonical = [], [], []
    for cell in case["rounds"]:
        m = cell["m"]
        raw = read(run / "signals" / f"{cell['id']}.json")
        if raw["candidate"] != cell or raw["owner"] != plan["owner"]:
            raise ValueError("H6 signal belongs to a different physical schedule")
        signals.append(dict(m=m, candidate=raw["candidate"], signal=raw["signal"]))
        canonical.append(dict(m=m, q_PF=cell["q"], R_outer=cell["R_outer"], actual_delta_pf=cell["actual_step"]))
        for axis in AXES:
            model = models[group_key(cell), axis]
            if cell["id"] in model["direct"]:
                axis_price = model["direct"][cell["id"]]
                price = exact(axis_price["price"])
                kind, evidence = "DIRECT_COMPILED_RZ", axis_price["source"]
            else:
                estimate = model["estimate"]
                if estimate is None:
                    raise ValueError(f"H6 price missing at {m}/{axis}: insufficient same-step calibration")
                step = estimate["domain"]["route"]["actual_step"]
                if step != cell["actual_step"]:
                    raise ValueError(f"H6 price model step differs from the physical step at {m}/{axis}")
                price = linear.predict(estimate, features(cell))
                kind = "CONDITIONAL_COMPILED_PRICE_MODEL"
                proof = run / "price_models" / f"{digest(str((group_key(cell), axis)))}.json"
                evidence = dict(path=str(proof), sha256=sha(proof), direct_observation=False,
                                calibration_step=step,
                                target_step=cell["actual_step"])
            if price <= 0:
                raise ValueError(f"Nonpositive H6 price prediction at {m}/{axis}")
            axes.append(dict(m=m, axis=axis, kind=kind, status="ADMITTED", qualified_E_C=False,
                             unit_price_RZ=enc(price), price_evidence=evidence, physical_id=cell["id"]))
    evaluation = dict(binding=plan["binding"], M=M, formula=case["formula"], L_D=case["prefix"],
                      h=enc(Fraction(h)), delta_pf=enc(Fraction(h) * exact(case["width_over_h"])))
    config_path = folder / "CONFIGURATION.json"
    retain(config_path, dict(id=cid, spec=evaluation, rounds=canonical))
    signal_path = folder / "SIGNALS.json"
    retain(signal_path, dict(status="SIGNAL_PACKETS_READY", target=plan["target"], h=h, M=M,
                            compiler=plan["compiler"], run_id=plan["owner"], scope_owner=plan["owner"],
                            methods={LABELS[method]: signals}))
    price_path = folder / "REGISTERED_PRICES.json"
    entry = dict(configuration_id=cid, configuration=str(config_path), configuration_sha256=sha(config_path))
    retain(price_path, dict(scope="H6_SCOPED_CONDITIONAL_COST", compiler=plan["compiler"],
                           signal_packet_sha256=sha(signal_path), methods={LABELS[method]: dict(entry, axes=axes)}))
    return axes, dict(entry, signal_packet=str(signal_path), signal_packet_sha256=sha(signal_path)), price_path


def optimize(plan, run, risk_source, q3_slots, allocated, budgets, *, execute, artificial_backend=None,
             partial=False, route_cpus=None):
    # Uses the same optimizer and allocation rule as the normal distributed flow.
    models, errors = price_models(plan, run, require_holdout=partial)
    scope = plan["scope"]
    model_dir = run / "price_models"
    model_dir.mkdir(exist_ok=True)
    for key, model in models.items():
        if partial and model["estimate"] is None:
            continue
        path = model_dir / f"{digest(str(key))}.json"
        if path.exists():
            # Late receipts can support a new estimate in a new scope. Jobs in
            # this scope keep the original model and its cost vector.
            models[key] = read(path)
        else:
            retain(path, model)
    if not partial:
        write(run / "PRICE_MODELS.json", dict(models={str(k): v for k, v in models.items()}, errors=errors))
    routes = defaultdict(list)
    expected, candidates, prices_by_id, held = {}, [], {}, {}
    methods = {name: {"family": "PR" if name == "PR_DF" else "PF"} for name in scope["methods"]}
    horizon = scope["M"]
    search = dict(rule="RPE_SCALED_SHOTS_V1",
                  scaled_shots=scaled.settings(dict(scope["scaled_shots"], M=[horizon])),
                  work_conserving_search=True)
    if artificial_backend is not None:
        search.pop("work_conserving_search")
    problem = dict(problem={"system": "H6"}, methods=methods, rpe={"M": horizon},
                   target={"energy_ha": plan["target"]}, accuracy={"epsilon_E_decimal": "0.00159360010199040"},
                   search=search)
    for case in plan["cases"]:
        cid, method = case["id"], case["method"]
        pr = method == "PR_DF"
        key = dict(n_atoms=6, formula=case["formula"], width_over_h=str(exact(case["width_over_h"])))
        key["L_D" if pr else "p"] = case["prefix"]
        if pr:
            key.update(kappa=str(exact(case["kappa"])), K=case["K"])
        row = dict(configuration_id=cid, name=f"H6:{method}:{cid[:12]}", method=method,
                   family=methods[method]["family"], key=key, caps=scaled.base_shots(horizon),
                   labels=[], saved_points=[], M=horizon, ready=False, prices=None, signal_source=None, needs=[])
        if pr:
            row["kappa_value"] = key["kappa"]
        candidates.append(row)
        try:
            axes, entry, price_path = candidate_inputs(run, plan, case, models)
        except (OSError, KeyError, ValueError) as error:
            row.update(needs=["SIGNAL_OR_PRICE_EVIDENCE"], price_status="INCOMPLETE_H6_PREPARATION")
            held[cid] = f"{type(error).__name__}: {error}"
            continue
        prices = [exact(axis["unit_price_RZ"]) for axis in axes]
        row.update(ready=True, prices=prices, price_axes=axes, signal_source=entry["signal_packet"],
                   price_status="SOURCE_BOUND_HCHAIN_PRICES")
        prices_by_id[cid] = prices
        expected[cid] = dict(M=horizon, signal_source=entry["signal_packet"])
        routes[method].append(dict(kind="h6_df19_scoped_q3", scope_owner=plan["owner"], interpreter=sys.executable,
            view_commit="cbc471c913072de45c25a0ed22d51dff8a9b7034", risk_source=str(risk_source),
            allocated_cpus=route_cpus or max(1, allocated // q3_slots), configuration_ids=[cid], configurations={cid: entry},
            registered_prices=str(price_path), registered_prices_sha256=sha(price_path)))
    if not partial:
        write(run / "HELD_CANDIDATES.json", held)
    tau = optimizer.tau(problem)
    context = dict(candidates=candidates, saved=optimizer.SavedBackend([], prices_by_id), tau=tau,
                   catalog={"candidates": []}, generate=None, generation={})
    if artificial_backend is not None:
        results = optimizer.optimize(ROOT, problem, context=context, backend=artificial_backend,
                                     max_new=budgets, science_approved=True)
    else:
        overlap_record = run / "Q3_OVERLAP.json"
        previous = read(overlap_record) if overlap_record.exists() else {}
        if previous and previous["scope_owner"] != plan["owner"]:
            raise ValueError("Q3 overlap results belong to a different H6 scope")
        executor = concurrent.ConcurrentExecutor(ROOT, run, routes, expected, problem, budgets=budgets,
                    science_approved=execute, max_concurrent=q3_slots, wait_for_running=True,
                    optimizer_source=plan["owner"])
        for method, values in previous.get("execution_holds", {}).items():
            executor.execution_holds.setdefault(method, {}).update(values)
        results = concurrent.optimize(ROOT, problem, context=context, executor=executor,
                                      max_new=budgets, science_approved=execute,
                                      methods=list(routes) if partial else None)
    if not partial:
        write(run / "ROUND_RESULT.json", optimizer.jsonable(dict(methods=results, scope_owner=plan["owner"],
                  counts={method: sum(c["method"] == method for c in candidates) for method in methods},
                  ready=sum(c["ready"] for c in candidates), held=len(held))))
    return results, held


def prepare_with_overlap(command, run, risk_source, q3_slots, allocated, price_memory, budgets, *, execute):
    # A finite budget requires the complete priority order for that method.
    methods = {method for method, budget in budgets.items() if budget != 0}
    finite = {method for method, budget in budgets.items() if budget not in (None, "UNLIMITED")}
    if not execute or not methods:
        subprocess.run(command, check=True)
        return

    from multiprocessing import Pipe

    cpus = max(1, allocated // q3_slots)
    memory = max(1, price_memory // q3_slots)
    plan_path = run / "PLAN.json"
    record_path = run / "Q3_OVERLAP.json"
    record = (read(record_path) if record_path.exists()
              else dict(processed=[], batches=0, launched={}, execution_holds={}))
    processed = set(record["processed"])
    controller, preparation = Pipe()
    fd = preparation.fileno()
    command = [*command, "--q3-overlap-fd", str(fd), "--allocated-cpus", str(allocated)]
    with controller, preparation, subprocess.Popen(command, pass_fds=(fd,)) as process:
        preparation.close()
        plan = None
        while process.poll() is None and not record.get("failure"):
            if (run / "STOP_REQUESTED.json").exists():
                break
            if plan is None and plan_path.exists():
                plan = read(plan_path)
                if record.get("scope_owner", plan["owner"]) != plan["owner"]:
                    raise ValueError("Q3 overlap results belong to a different H6 scope")
                record["scope_owner"] = plan["owner"]
            if plan is not None:
                batch = ready_batch(plan, run, methods, processed, require_all=finite)
                if batch["cases"]:
                    try:
                        controller.send((min(q3_slots, len(batch["cases"])), cpus, memory))
                        while not controller.poll(1):
                            if process.poll() is not None:
                                break
                        slots = controller.recv() if controller.poll() else 0
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        break
                    if process.poll() is not None:
                        break
                    if slots:
                        try:
                            results, _ = optimize(batch, run, risk_source, slots, allocated, budgets, execute=True,
                                                  partial=True, route_cpus=cpus)
                            processed.update(case["id"] for case in batch["cases"])
                            record.update(processed=sorted(processed), batches=record["batches"] + 1)
                            for method, result in results.items():
                                holds = record["execution_holds"].setdefault(method, {})
                                for row in result.get("execution_holds", []):
                                    holds[row["configuration_id"]] = row
                                record["launched"][method] = record["launched"].get(method, 0) + result["new_calls"]
                                if result["stop_reason"] == "UNKNOWN_RESULT":
                                    record["failure"] = result["stop_reason"]
                            write(record_path, record)
                        finally:
                            try:
                                controller.send("release")
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                        continue
            time.sleep(5)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    if record.get("failure"):
        raise ValueError("Q3 overlap requires review: " + record["failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--saved-signals", type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--q3-slots", type=int)
    parser.add_argument("--preparation-workers", type=int)
    parser.add_argument("--signal-workers", type=int)
    parser.add_argument("--price-workers", type=int)
    parser.add_argument("--price-memory-gib", type=int, default=350)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scope = read(args.scope)
    if scope["system"] != "H6":
        parser.error("This PBS acquisition provider requires H6")
    menu = list(case_rows(scope, 19))
    scaled.settings(dict(scope["scaled_shots"], M=[scope["M"]]))
    if args.plan_only:
        if args.assets:
            assets = args.assets.resolve()
            worker = Path(__file__).resolve().with_name("h6_scope_worker.py")
            subprocess.run([sys.executable, "-B", str(worker), "--assets", str(assets), "--check"],
                           check=True, stdout=sys.stderr)
            risk = assets / "q3-pilot/risk-source/src"
            subprocess.run([sys.executable, "-B", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from trottertracks.rpe_total_dp_parallel import Accelerator; Accelerator(workers=1)", str(risk)],
                check=True)
        preview = dict(status="PLAN_ONLY", counts={m: sum(c["method"] == m for c in menu) for m in scope["methods"]},
                       candidates=menu, calibration_max_m=scope["calibration_max_m"], scientific_calls=0)
        if args.run:
            args.run.mkdir(parents=True, exist_ok=True)
            write(args.run / "SCOPE_PLAN.json", preview)
        print(json.dumps({k: v for k, v in preview.items() if k != "candidates"}))
        return 0
    if args.run is None or args.assets is None or not os.environ.get("PBS_JOBID"):
        parser.error("Preparation/execute require --run, --assets and a PBS allocation")
    import fcntl
    from trottertracks.rpe_q3_resources import resource_snapshot

    run = args.run.resolve()
    assets = args.assets.resolve()
    run.mkdir(parents=True, exist_ok=True)
    available = min(len(os.sched_getaffinity(0)), int(os.environ.get("NCPUS", len(os.sched_getaffinity(0)))))
    for count in (args.preparation_workers, args.signal_workers, args.price_workers, args.q3_slots, args.price_memory_gib):
        if count is not None and count < 1:
            parser.error("PBS resources and worker counts must be positive")
    limits = resource_snapshot(dict(slot_policy=dict(resource_limits=dict(
        cpu_count=available, memory_bytes=args.price_memory_gib * GIB))))
    available = min(available, int(limits["allowed_cpu_count"]))
    caps = [limits[k] for k in ("allowed_memory_bytes", "remaining_group_memory_bytes") if limits[k] is not None]
    # Keep space for the supervisor, shared preparation and RSS growth between samples.
    price_memory = min(caps) * 9 // 10
    default_workers = args.preparation_workers or available
    signal_workers = min(available, args.signal_workers or default_workers, max(1, price_memory // GIB))
    price_workers = min(available, args.price_workers or default_workers)
    q3_slots = min(available, args.q3_slots if args.q3_slots is not None else max(1, available // 8))
    controlled = scope.get("max_new_q3_per_method", {})
    budgets = {m: controlled.get(m, "UNLIMITED") if args.execute else 0 for m in scope["methods"]}
    if available < 1 or price_memory < 1:
        parser.error("The current PBS CPU or memory allowance is exhausted")
    with (run / ".h6_scope.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = Path(__file__).resolve().with_name("h6_scope_worker.py")
        command = [sys.executable, "-B", str(worker), "--scope", str(args.scope.resolve()),
                   "--assets", str(assets), "--run", str(run), "--signal-workers", str(signal_workers),
                   "--price-workers", str(price_workers), "--price-memory-bytes", str(price_memory)]
        if args.saved_signals:
            command.extend(["--saved-signals", str(args.saved_signals.resolve())])
        if args.continue_from:
            command.extend(["--continue-from", str(args.continue_from.resolve())])
        risk_source = assets / "q3-pilot/risk-source/src"
        prepare_with_overlap(command, run, risk_source, q3_slots, available, price_memory, budgets,
                             execute=args.execute)
        plan = read(run / "PLAN.json")
        _, held = optimize(plan, run, risk_source, q3_slots, available, budgets,
                           execute=args.execute)
    return 2 if held else 0


if __name__ == "__main__":
    raise SystemExit(main())
