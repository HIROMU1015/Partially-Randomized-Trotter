"""PBS preparation worker for H6 candidates. The planned scope owns every call."""

import argparse
from functools import lru_cache
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import h6_preparation_pool as pool
from h6_scope import FORMULAS, calibration_design, case_rows, digest, exact, read, rounds


def write(path, data):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, allow_nan=False) + "\n")
    tmp.replace(path)


def write_price(path, data):
    events = data["events"]
    if events:
        archive = path.with_suffix(".events.json.xz")
        tmp = archive.with_suffix(f".{os.getpid()}.tmp")
        encoded = {}
        try:
            with tmp.open("xb") as raw, lzma.LZMAFile(raw, "wb", preset=1) as out:
                out.write(b"[")
                for batch_index, batch in enumerate(events):
                    out.write(b", [" if batch_index else b"[")
                    for event_index, event in enumerate(batch):
                        key = id(event)
                        if key not in encoded:
                            if len(encoded) == 256:
                                encoded.clear()
                            item = event if isinstance(event, dict) else event.to_dict()
                            encoded[key] = json.dumps(item, allow_nan=False).encode("ascii")
                        if event_index:
                            out.write(b", ")
                        out.write(encoded[key])
                    out.write(b"]")
                out.write(b"]\n")
            with tmp.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            info = dict(file=archive.name, size_bytes=tmp.stat().st_size, sha256=checksum)
            tmp.replace(archive)
        finally:
            tmp.unlink(missing_ok=True)
        data = dict(data, events=info)
    # Publish the small receipt last so consumers only see a completed pair.
    write(path, data)


def read_price(path):
    receipt = read(path)
    events = receipt.get("events")
    if isinstance(events, dict):
        archive = path.with_suffix(".events.json.xz")
        if events["file"] != archive.name or archive.stat().st_size != events["size_bytes"]:
            raise ValueError(f"H6 price event archive differs for {path.name}")
    return receipt


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


def measured_pair_memory(cell):
    # The separate PBS comparison measured these two PR-DF circuit shapes.
    settings = (cell["method"], cell["formula"], cell["prefix"], cell["K"], cell["q"], cell["T"])
    if settings == ("PR", "2nd", 4, 2, 7, 3.2) and cell["R_outer"] in (169, 337):
        return 8 * pool.GIB  # 4.98 GiB observed peak plus 50% headroom, rounded up.
    return 0


def price(cell, index, cohort):
    from h6_price_acceleration import PriceFingerprints, fingerprint_context, hadamard_wrapper, reuse_sampled_events
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost

    prep = STATE.preparations[cell["prefix"]]
    if cell["R_outer"] is None:
        requests, seeds, laws = [], [], []
    else:
        with pool.price_phase("sample_events"), reuse_sampled_events():
            requests, seeds, laws = STATE.helpers["sample_requests"](cell, prep, cohort, index)
    compiler = CompilerSettings(("rz", "sx", "x", "cx"), None, None, 1, None, None, 17, "1.3.0")
    result = {}
    with fingerprint_context()() as session:
        with pool.price_phase("assemble_and_fingerprint"):
            fp = PriceFingerprints(session)
            evolution = STATE.pieces[cell["prefix"]].assemble(
                cell, STATE.helpers["core"], requests, scalar(cell), fp.prepare,
            )
        def compile_axis(native, _directory, evolution, fp):
            began = time.monotonic()
            wrapper = hadamard_wrapper(evolution.circuit, native)
            cost = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fp.axes[native])
            if (cost.compiler_settings_hash != STATE.compiler or session.fingerprint(wrapper) != fp.axes[native]
                    or cost.rz_count <= 0):
                raise ValueError("Compiled H6 receipt fails compiler, fingerprint or price binding")
            receipt = dict(rz=cost.rz_count, fingerprint=fp.axes[native],
                metrics={key: getattr(cost, key) for key in ("rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")})
            return receipt, time.monotonic() - began, pool.peak_rss()

        axes = (("X", "cosine"), ("Y", "sine"))
        if pool.use_spare_axis():
            from unittest.mock import patch
            import run_h6_df19_price_v2_q4 as fork_axes

            with patch.object(fork_axes, "price_axis", compile_axis):
                prices = fork_axes.parallel_axes(None, evolution, fp)
        else:
            prices = {native: compile_axis(native, None, evolution, fp) for _, native in axes}
        for axis, native in axes:
            result[axis], seconds, rss = prices[native]
            pool.record_price_phase(f"compile_{axis}", seconds, rss)
    events = [request.rte_occurrence.events for request in requests]
    return dict(candidate=cell, index=index, cohort=cohort, axes=result, compiler=STATE.compiler,
                preparation_hash=prep.preparation_hash, seeds=seeds, laws=laws, events=events)


def completed(job):
    kind, cell, index, path, owner = job
    if not path.exists():
        return False
    old = read_price(path) if kind == "price" else read(path)
    if (old["owner"] != owner or old["candidate"] != cell
            or kind == "price" and old["index"] != index):
        raise ValueError("Saved H6 preparation receipt belongs to a different input")
    return True


def execute(job):
    kind, cell, index, path, owner = job
    if completed(job):
        return
    # Ownership stays visible if PBS ends before the receipt is committed.
    with path.with_suffix(".started").open("x") as marker:
        json.dump(dict(owner=owner, job=os.environ.get("PBS_JOBID"), candidate=cell["id"]), marker)
    if kind == "price":
        cohort = getattr(STATE, "price_cohort", "H6_SCOPE_" + owner)
        receipt = dict(price(cell, index, cohort), owner=owner)
        with pool.price_phase("write_receipt"):
            write_price(path, receipt)
    else:
        write(path, dict(signal(cell), owner=owner))


def resume_plan(plan, previous, run):
    if previous == run or any((previous / name).exists() for name in ("STOP.json", "cases", "ROUND_RESULT.json")):
        raise ValueError("H6 predecessor must be a separate preparation-stage run without a STOP")
    raw = (previous / "PLAN.json").read_bytes()
    old = json.loads(raw)
    if any(old[key] != plan[key] for key in ("scope", "binding", "target", "compiler", "cases", "designs")):
        raise ValueError("H6 predecessor plan differs from the new physical schedule")
    source = dict(run=str(previous), owner=old["owner"], plan_sha256=hashlib.sha256(raw).hexdigest())
    cohort = old.get("price_cohort", "H6_SCOPE_" + old["owner"])
    return dict(plan, owner=digest(dict(owner=plan["owner"], source=source, price_cohort=cohort)),
                continued_from=source, price_cohort=cohort)


def inherit_jobs(jobs, plan, preparations):
    source = Path(plan["continued_from"]["run"])
    old_owner = plan["continued_from"]["owner"]
    pending, inherited = [], 0
    for job in jobs:
        kind, cell, index, path, owner = job
        origin = source / path.parent.name / path.name
        if completed(job):
            continue
        if not origin.is_file():
            pending.append(job)
            continue
        raw = origin.read_bytes()
        receipt = json.loads(raw)
        if receipt["owner"] != old_owner or receipt["candidate"] != cell:
            raise ValueError(f"H6 predecessor result identity differs: {origin}")
        if kind == "price" and (
            receipt["compiler"] != plan["compiler"] or receipt["index"] != index
            or receipt["cohort"] != plan["price_cohort"]
            or receipt["preparation_hash"] != preparations[cell["prefix"]].preparation_hash
        ):
            raise ValueError(f"H6 predecessor price source differs: {origin}")
        receipt.update(owner=owner, reused_source=dict(path=str(origin), owner=old_owner,
                                                       sha256=hashlib.sha256(raw).hexdigest()))
        if kind == "price" and isinstance(receipt["events"], dict):
            read_price(origin)
            old_archive = origin.with_suffix(".events.json.xz")
            with old_archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["events"]["sha256"]:
                    raise ValueError(f"H6 predecessor event archive checksum differs: {old_archive}")
            dest = path.with_suffix(".events.json.xz")
            if not dest.exists():
                os.link(old_archive, dest)
            elif not dest.samefile(old_archive):
                raise ValueError(f"H6 continuation event archive already exists: {dest}")
            write(path, receipt)
        elif kind == "price":
            write_price(path, receipt)
        else:
            write(path, receipt)
        inherited += 1
    return pending, inherited


def collect(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path=None,
            previous=None):
    import fcntl

    run.mkdir(parents=True, exist_ok=True)
    with (run / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if previous is None:
            return collect_owned(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path)
        with (previous / ".h6_scope.lock").open("rb") as predecessor:
            fcntl.flock(predecessor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return collect_owned(scope, assets, run, signal_workers, price_workers, price_memory,
                                 saved_signal_path, previous)


def collect_owned(scope, assets, run, signal_workers, price_workers, price_memory, saved_signal_path,
                  previous=None):
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
    if previous is not None:
        plan = resume_plan(plan, previous, run)
        owner = plan["owner"]
        STATE.price_cohort = plan["price_cohort"]
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

    if previous is not None:
        signal_total, price_total = len(signal_jobs), len(price_jobs)
        signal_jobs, signals = inherit_jobs(signal_jobs, plan, STATE.preparations)
        price_jobs, prices = inherit_jobs(price_jobs, plan, STATE.preparations)
        write(run / "CONTINUED_PREPARATION.json", dict(source=plan["continued_from"],
            price_cohort=STATE.price_cohort, imported_signals=signals, imported_prices=prices,
            existing_signals=signal_total-len(signal_jobs), existing_prices=price_total-len(price_jobs),
            pending_signals=len(signal_jobs), pending_prices=len(price_jobs)))

    def checkpoint(metrics):
        write(run / f"{metrics['phase'].upper()}_PREPARATION_METRICS.json", metrics)

    def size(cell):
        return pool.price_size(cell, STATE.pieces[cell["prefix"]].bound, FORMULAS[cell["formula"]])

    errors = pool.signals(signal_jobs, execute, signal_workers, checkpoint)
    errors.update(pool.prices(price_jobs, execute, price_workers, price_memory, size, checkpoint,
                              pair_memory_for=measured_pair_memory))
    write(run / "PREPARATION_ERRORS.json", errors)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--saved-signals", type=Path)
    parser.add_argument("--continue-from", type=Path)
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
    previous = args.continue_from.resolve() if args.continue_from else None
    raise SystemExit(collect(read(args.scope), args.assets.resolve(), args.run.resolve(), *resources,
                             args.saved_signals, previous))
