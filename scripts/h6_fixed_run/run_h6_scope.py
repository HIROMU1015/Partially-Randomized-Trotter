"""Plan and run H6 parameter scopes on PBS using the common RPE shot optimizer."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from h6_scope import AXES, LABELS, case_rows, digest, enc, exact, group_key, read
from h6_scope_worker import write


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
        receipt = read(path)
        if (receipt["candidate"]["id"] != cell["id"] or receipt["index"] != i
                or receipt["owner"] != plan["owner"] or receipt["compiler"] != plan["compiler"]
                or receipt["candidate"] != cell
                or type(receipt["axes"][axis]["rz"]) is not int or receipt["axes"][axis]["rz"] <= 0):
            raise ValueError("H6 calibration receipt identity differs")
        values.append(receipt["axes"][axis]["rz"])
        sources.append(dict(path=str(path), sha256=sha(path)))
    return Fraction(sum(values), len(values)), sources


def price_models(plan, run):
    models, errors = {}, {}
    for design in plan["designs"]:
        first = design["anchors"][0]
        key = group_key(first)
        names = ("one", "q", "qR") if design["mode"] == "AFFINE_QR" else ("one", "q")
        for axis in AXES:
            try:
                direct = {}
                for cell in design["anchors"]:
                    mean, proof = observation(plan, cell, run, axis)
                    direct[cell["id"]] = dict(price=enc(mean), source=proof)
                holdout = design["holdout"]
                if design["mode"] == "DIRECT":
                    models[key, axis] = dict(direct=direct, estimate=None, holdout=None)
                    continue
                route = dict(binding=plan["binding"], mode=key[0], formula=key[1], prefix=key[2],
                             actual_step=enc(key[3]), K=key[4])
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
                                      design_source={"maximum_direct_round": plan["scope"]["calibration_max_m"]})
                comparison = None
                if holdout is not None:
                    try:
                        observed, proof = observation(plan, holdout, run, axis)
                        predicted = linear.predict(estimate, features(holdout))
                        direct[holdout["id"]] = dict(price=enc(observed), source=proof)
                        comparison = dict(observed=enc(observed), predicted=enc(predicted),
                            relative_error=enc((predicted-observed)/observed), source=proof, refit=False)
                    except (OSError, ValueError) as error:
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
                price = linear.predict(model["estimate"], features(cell))
                kind = "CONDITIONAL_COMPILED_PRICE_MODEL"
                proof = run / "price_models" / f"{digest(str((group_key(cell), axis)))}.json"
                evidence = dict(path=str(proof), sha256=sha(proof), direct_observation=False)
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


def optimize(plan, run, risk_source, q3_slots, allocated, budgets, *, execute, artificial_backend=None):
    # Uses the same optimizer and allocation rule as the normal distributed flow.
    models, errors = price_models(plan, run)
    scope = plan["scope"]
    model_dir = run / "price_models"
    model_dir.mkdir(exist_ok=True)
    for key, model in models.items():
        path = model_dir / f"{digest(str(key))}.json"
        if path.exists():
            # Late receipts can support a new estimate in a new scope. Jobs in
            # this scope keep the original model and its cost vector.
            models[key] = read(path)
        else:
            retain(path, model)
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
                   labels=[], saved_points=[], M=horizon, ready=False, prices=None, needs=[])
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
            allocated_cpus=max(1, allocated // q3_slots), configuration_ids=[cid], configurations={cid: entry},
            registered_prices=str(price_path), registered_prices_sha256=sha(price_path)))
    write(run / "HELD_CANDIDATES.json", held)
    tau = optimizer.tau(problem)
    context = dict(candidates=candidates, saved=optimizer.SavedBackend([], prices_by_id), tau=tau,
                   catalog={"candidates": []}, generate=None, generation={})
    if artificial_backend is not None:
        results = optimizer.optimize(ROOT, problem, context=context, backend=artificial_backend,
                                     max_new=budgets, science_approved=True)
    else:
        executor = concurrent.ConcurrentExecutor(ROOT, run, routes, expected, problem, budgets=budgets,
                    science_approved=execute, max_concurrent=q3_slots, wait_for_running=True,
                    optimizer_source=plan["owner"])
        results = concurrent.optimize(ROOT, problem, context=context, executor=executor,
                                      max_new=budgets, science_approved=execute)
    write(run / "ROUND_RESULT.json", optimizer.jsonable(dict(methods=results, scope_owner=plan["owner"],
              counts={method: sum(c["method"] == method for c in candidates) for method in methods},
              ready=sum(c["ready"] for c in candidates), held=len(held))))
    return results, held


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--saved-signals", type=Path)
    parser.add_argument("--q3-slots", type=int)
    parser.add_argument("--preparation-workers", type=int)
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

    run = args.run.resolve()
    assets = args.assets.resolve()
    run.mkdir(parents=True, exist_ok=True)
    available = min(len(os.sched_getaffinity(0)), int(os.environ.get("NCPUS", len(os.sched_getaffinity(0)))))
    workers = min(available, args.preparation_workers if args.preparation_workers is not None else available)
    q3_slots = min(available, args.q3_slots if args.q3_slots is not None else max(1, available // 8))
    controlled = scope.get("max_new_q3_per_method", {})
    budgets = {m: controlled.get(m, "UNLIMITED") if args.execute else 0 for m in scope["methods"]}
    if workers < 1 or q3_slots < 1:
        parser.error("PBS worker and Q3 slot counts must be positive")
    with (run / ".h6_scope.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = Path(__file__).resolve().with_name("h6_scope_worker.py")
        command = [sys.executable, "-B", str(worker), "--scope", str(args.scope.resolve()),
                   "--assets", str(assets), "--run", str(run), "--workers", str(workers)]
        if args.saved_signals:
            command.extend(["--saved-signals", str(args.saved_signals.resolve())])
        subprocess.run(command, check=True)
        plan = read(run / "PLAN.json")
        _, held = optimize(plan, run, assets / "q3-pilot/risk-source/src", q3_slots, available, budgets,
                           execute=args.execute)
    return 2 if held else 0


if __name__ == "__main__":
    raise SystemExit(main())
