"""PBS preparation worker for H6 candidates. The planned scope owns every call."""

import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import h6_preparation_pool as pool
from h6_scope import FORMULAS, calibration_design, case_rows, digest, exact, read, rounds


def write(path, data):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, allow_nan=False) + "\n")
    tmp.replace(path)


def fixed_inputs(assets):
    sys.path[:0] = [str(assets / "payload-bound"), str(assets / "representatives")]
    import run_full_m7 as fixed
    os.environ["H6_SCOPED_FINGERPRINT_SOURCE"] = str(assets / "optimizer-source/scripts/h46_b3_fast_fingerprint.py")
    return fixed, fixed.check()


def startup(assets, scope):
    fixed, (q4, helpers, _, inputs, full, _) = fixed_inputs(assets)
    from run_signals import b0i_identity
    from h6_price_acceleration import CircuitPieces
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot

    saved = assets / "payload-bound/inputs"
    ham, sector, _, state, metadata, _ = load_h6_snapshot(
        saved / "H6_snapshot.npz", read(saved / "H6_snapshot_receipt.json"), read(saved / "H6_df_receipt.json"),
    )
    cases = list(case_rows(scope, len(full.deterministic_blocks) - 1))
    prefixes = {c["prefix"] for c in cases}
    rank = len(full.deterministic_blocks) - 1
    preps = {p: helpers["prepare"](ham, p) for p in sorted(prefixes - {rank})}
    if any(p.hamiltonian_hash != inputs["scientific_binding"]["hamiltonian"] for p in preps.values()):
        raise ValueError("Prepared H6 blocks disagree with the frozen Hamiltonian")
    if rank in prefixes:
        preps[rank] = full
    lambdas = {p: preps[p].rte_preparation.symbolic_tail.lambda_r
               for p in {c["prefix"] for c in cases if c["method"] == "PR_DF"}}
    identity = {p: b0i_identity(ham, p) for p in {c["prefix"] for c in cases if c["method"] == "TRUNCATED_PF"}}
    for c in cases:
        c["rounds"] = list(rounds(scope, c, lambdas.get(c["prefix"])))
        c["id"] = digest({"binding": inputs["scientific_binding"], "case": c})  # logical scope, not the physical cache key
    # Reuse the archived occupation-basis kernel with exactly the same fragment ordering.
    scope_ = {}
    import numpy as np
    scope_["np"] = np
    q4.p.functions(q4.p.CARRIER, "scripts/h46_new2_core.py", ["occupation_one_body"], scope_)
    one_body = scope_["occupation_one_body"]
    matrices = [one_body(ham.one_body, sector)]
    for weight, fragment in zip(ham.lambdas, ham.g_matrices, strict=True):
        block = one_body(fragment, sector)
        matrices.append(weight * (block @ block))
    tail = {p: sum(matrices[p + 1:], np.zeros_like(matrices[0]))
            - preps[p].extracted_identity_coefficient * np.eye(len(state))
            for p in lambdas}
    return SimpleNamespace(cases=cases, binding=inputs["scientific_binding"], target=metadata["solver_energy"],
        compiler=fixed.COMPILER_HASH, helpers=helpers, preparations=preps,
        pieces={p: CircuitPieces(v) for p, v in preps.items()}, state=state, matrices=matrices,
        tail=tail, lambdas=lambdas, identity=identity, constant=ham.constant)


STATE = None


@lru_cache(maxsize=24)
def head_blocks(prefix, time):
    import numpy as np
    from scipy.linalg import expm

    matrices = STATE.matrices[:prefix + 1]
    forward, reverse = (np.eye(len(matrices[0]), dtype=complex) for _ in range(2))
    for block in matrices:
        step = expm(-0.5j * time * block)
        forward = step @ forward
        reverse = reverse @ step
    return forward, reverse


def scalar(cell):
    prefix = cell["prefix"]
    if cell["method"] == "PR":
        return STATE.constant + STATE.preparations[prefix].extracted_identity_coefficient
    return STATE.constant + (STATE.identity[prefix] if cell["method"] == "B0I" else 0.0)


def signal(cell):
    import numpy as np
    from trotterlib.rte import finite_rte_distribution, finite_taylor_operator

    prefix, T, q = cell["prefix"], cell["T"], cell["q"]
    state = STATE.state
    normalized = np.eye(len(state), dtype=complex)
    core = STATE.helpers["core"]
    weights = core.weights(cell["formula"])
    r = cell["R_outer"]
    counts = [None] * len(weights)
    if r is not None:
        counts = (r,) if len(weights) == 1 else core.allocation(weights, r)
    for weight, count in zip(weights, counts, strict=True):
        t = T * weight / q
        forward, reverse = head_blocks(prefix, t)
        part = forward
        if count is not None:
            lam = STATE.lambdas[prefix]
            scaled = lam * (t / count)
            norm = finite_rte_distribution(scaled, cell["K"]).exact_finite_distribution
            step = finite_taylor_operator(STATE.tail[prefix] / lam, scaled, cell["K"])
            part = np.linalg.matrix_power(step / norm, count) @ forward
        normalized = (reverse @ part) @ normalized
    z = complex(np.vdot(state, np.linalg.matrix_power(normalized, q) @ state) * np.exp(-1j * T * scalar(cell)))
    if not math.isfinite(abs(z)) or abs(z) > 1 + 1e-9:
        raise ValueError("Invalid normalized H6 raw measurement mean")
    return dict(candidate=cell, signal={"raw": {"real": z.real, "imag": z.imag}})


def price(cell, index, cohort):
    from h6_price_acceleration import PriceFingerprints, fingerprint_context, hadamard_wrapper
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost

    prep = STATE.preparations[cell["prefix"]]
    if cell["R_outer"] is None:
        requests, seeds, laws = [], [], []
    else:
        requests, seeds, laws = STATE.helpers["sample_requests"](cell, prep, cohort, index)
    compiler = CompilerSettings(("rz", "sx", "x", "cx"), None, None, 1, None, None, 17, "1.3.0")
    result = {}
    with fingerprint_context()() as session:
        fp = PriceFingerprints(session)
        evolution = STATE.pieces[cell["prefix"]].assemble(
            cell, STATE.helpers["core"], requests, scalar(cell), fp.prepare,
        )
        for axis, native in (("X", "cosine"), ("Y", "sine")):
            wrapper = hadamard_wrapper(evolution.circuit, native)
            cost = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fp.axes[native])
            if (cost.compiler_settings_hash != STATE.compiler or session.fingerprint(wrapper) != fp.axes[native]
                    or cost.rz_count <= 0):
                raise ValueError("Compiled H6 receipt fails compiler, fingerprint or price binding")
            result[axis] = dict(rz=cost.rz_count, fingerprint=fp.axes[native],
                metrics={key: getattr(cost, key) for key in ("rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")})
            del cost, wrapper
    events = [[item.to_dict() for item in request.rte_occurrence.events] for request in requests]
    return dict(candidate=cell, index=index, cohort=cohort, axes=result, compiler=STATE.compiler,
                preparation_hash=prep.preparation_hash, seeds=seeds, laws=laws, events=events)


def execute(job):
    kind, cell, index, path, owner = job
    if path.exists():
        old = read(path)
        if old["owner"] != owner or old["candidate"]["id"] != cell["id"]:
            raise ValueError("Saved H6 preparation receipt belongs to a different input")
        return
    # Ownership stays visible if PBS ends before the receipt is committed.
    with path.with_suffix(".started").open("x") as marker:
        json.dump(dict(owner=owner, job=os.environ.get("PBS_JOBID"), candidate=cell["id"]), marker)
    data = price(cell, index, "H6_SCOPE_" + owner) if kind == "price" else signal(cell)
    write(path, dict(data, owner=owner))


def collect(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path=None):
    import fcntl

    run.mkdir(parents=True, exist_ok=True)
    with (run / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return collect_owned(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path)


def collect_owned(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path):
    global STATE
    STATE = startup(assets, scope)
    cases, binding, target, compiler = STATE.cases, STATE.binding, STATE.target, STATE.compiler
    designs = calibration_design(cases, scope)
    sources = {name: digest(Path(__file__).with_name(name).read_text()) for name in (
        "h6_scope_worker.py", "h6_scope.py", "run_h6_scope.py", "h6_price_acceleration.py",
    )}
    owner = digest(dict(scope=scope, binding=binding, sources=sources,
                        preparations={str(p): pre.preparation_hash for p, pre in STATE.preparations.items()}))
    plan = dict(owner=owner, cases=cases, binding=binding, target=target, compiler=compiler, scope=scope,
                designs=list(designs.values()))
    plan_path = run / "PLAN.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("PBS work belongs to a different frozen H6 scope")
    write(plan_path, plan)
    for directory in ("signals", "prices"):
        (run / directory).mkdir(exist_ok=True)
    cells = {cell["id"]: cell for case in cases for cell in case["rounds"]}
    # A saved bundle must carry the same fixed scientific binding. PR also checks
    # its exact round candidate rather than borrowing a result from another kappa.
    if saved_signal_path:
        saved = read(saved_signal_path)
        if (saved["target"] != target or saved["compiler"] != compiler or saved["h"] != float(exact(scope["h"]))
                or saved.get("scientific_binding") != binding):
            raise ValueError("Saved H6 signals differ from the frozen physical input")
        for rows in saved["methods"].values():
            for row in rows:
                candidate = row.get("candidate")
                if not candidate or candidate["id"] not in cells:
                    continue
                cell = cells[candidate["id"]]
                if any(candidate[key] != cell[key] for key in candidate if key in cell):
                    raise ValueError("Saved H6 signal has a foreign physical schedule")
                path = run / "signals" / f"{cell['id']}.json"
                if not path.exists() and not path.with_suffix(".started").exists():
                    write(path, dict(candidate=cell, signal=row["signal"], owner=owner,
                                     reused_source=str(saved_signal_path)))
    signal_jobs = [("signal", c, 0, run / "signals" / f"{c['id']}.json", owner) for c in cells.values()]
    price_cells = {}
    for design in designs.values():
        price_cells.update((cell["id"], cell) for cell in design["anchors"])
        if design["holdout"] is not None:
            holdout = design["holdout"]
            price_cells[holdout["id"]] = holdout
    price_jobs = []
    for cell in price_cells.values():
        paths = scope["price_paths"] if cell["method"] == "PR" else 1
        price_jobs.extend(("price", cell, i, run / "prices" / f"{cell['id']}-{i}.json", owner) for i in range(paths))

    def checkpoint(metrics):
        write(run / f"{metrics['phase'].upper()}_PREPARATION_METRICS.json", metrics)

    def profile(cell):
        return pool.price_profile(cell, STATE.pieces[cell["prefix"]].bound, FORMULAS[cell["formula"]])

    errors = pool.signals(signal_jobs, execute, signal_workers, checkpoint)
    errors.update(pool.prices(price_jobs, execute, price_workers, price_memory, profile, checkpoint))
    write(run / "PREPARATION_ERRORS.json", errors)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--saved-signals", type=Path)
    parser.add_argument("--signal-workers", type=int)
    parser.add_argument("--price-workers", type=int)
    parser.add_argument("--price-memory-bytes", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _, (_, _, _, inputs, full, runtime) = fixed_inputs(args.assets.resolve())
        from h6_price_acceleration import CircuitPieces
        from run_signals import b0i_identity
        from trotterlib.rte import finite_rte_distribution, finite_taylor_operator

        if not all(map(callable, (b0i_identity, finite_rte_distribution, finite_taylor_operator))):
            raise ValueError("H6 preparation helpers are unavailable")
        CircuitPieces(full)
        print(json.dumps(dict(status="H6_INPUT_CHECK_PASS", binding=inputs["scientific_binding"], runtime=runtime)))
        raise SystemExit(0)
    resources = (args.signal_workers, args.price_workers, args.price_memory_bytes)
    if args.scope is None or args.run is None or any(value is None or value < 1 for value in resources):
        parser.error("Preparation needs a scope, output directory, positive worker counts and a price memory budget")
    raise SystemExit(collect(read(args.scope), args.assets.resolve(), args.run.resolve(), *resources, args.saved_signals))
