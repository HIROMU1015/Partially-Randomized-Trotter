"""Run a batch of planned H-chain preparation leaves via the common worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import json
import os
from pathlib import Path
import sys


def planned_inputs(plan, stage):
    if stage == "signal":
        return {cell["id"]: cell for case in plan["cases"] for cell in case["rounds"]}
    if stage not in ("compile", "audit"):
        raise ValueError("Unknown H-chain preparation stage")
    anchors = {cell["id"]: cell for design in plan["designs"] for cell in design["anchors"]}
    if stage == "compile":
        return anchors
    return {cell["id"]: cell for design in plan["designs"] if (cell := design["holdout"]) is not None
            and cell["id"] not in anchors}


def leaves(plan, stage, targets):
    inputs = planned_inputs(plan, stage)
    if not targets:
        raise ValueError("An H-chain preparation batch must have targets")
    if len({tuple(target) for target in targets}) != len(targets):
        raise ValueError("Duplicate H-chain preparation target")
    for physical_id, index in targets:
        if physical_id not in inputs:
            raise ValueError("Preparation target is outside the specified plan and stage")
        cell = inputs[physical_id]
        paths = plan["scope"]["price_paths"] if stage != "signal" and cell["method"] == "PR" else 1
        if type(index) is not int or not 0 <= index < paths:
            raise ValueError("Preparation path index is outside the fixed plan")
        name = f"signals/{physical_id}.json" if stage == "signal" else f"prices/{physical_id}-{index}.json"
        yield cell, index, name


def preparation_request(plan, plan_sha, stage, targets, *, source, assets, spectral_signals=False):
    """Build a request for configure_preparation_graph or submit_preparation.

    The plan must already be deposited in the existing artifact cache.
    Target order and batching come from the caller's approved scope and
    preparation policy. The provider does no independent dispatch.
    """
    outputs = []
    for cell, _, name in leaves(plan, stage, targets):
        outputs.append(name)
        if stage != "signal" and cell["R_outer"] is not None and cell["R_outer"] > 0:
            outputs.append(str(Path(name).with_suffix(".events.json.xz")))
    runtime = {"packages": {"qiskit": "1.3.0", "numpy": "1.26.4"}}
    request = dict(method="HCHAIN", route=dict(kind="preparation"),
                original_contract=dict(scope_owner=plan["owner"], binding=plan["binding"], plan_sha256=plan_sha),
                preparation=dict(stage=stage, provider="HCHAIN_SCOPED", source=source,
                    inputs={"PLAN.json": plan_sha}, outputs=outputs,
                    required_paths={"hchain_assets": assets}, runtime_requirements=runtime,
                    command=["@PYTHON@", "-B", "@TRANSPORT@/scripts/run_rpe_hchain_preparation.py",
                             "--provider", "@SOURCE@/scripts/h6_fixed_run", "--assets", "@PATH:hchain_assets@",
                             "--plan", "@INPUTS@/PLAN.json", "--output", "@WORK@",
                             "--stage", stage, "--targets", json.dumps(targets, separators=(",", ":"))]))
    if stage == "signal" and spectral_signals:
        request["preparation"]["command"].append("--spectral-signals")
    return request


def preparation_graph(plan, plan_sha, ordered_batches, *, source, assets, budgets, approved):
    """Turn ordered, disjoint batches into the normal frozen round graph."""
    seen = set()
    tasks = []
    for stage, targets in ordered_batches:
        request = preparation_request(plan, plan_sha, stage, targets, source=source, assets=assets)
        outputs = set(request["preparation"]["outputs"])
        if seen & outputs:
            raise ValueError("H-chain preparation target appears in more than one batch")
        seen.update(outputs)
        tasks.append(dict(name=f"hchain-{stage}-{len(tasks)}", request=request, budget=budgets.get(stage, 0),
                          approved=approved, contract=request["original_contract"]))
    return dict(tasks=tasks)


def publish_preparation(executor, job_id, directory, *, wait=False):
    """Expose an accepted batch to the existing scoped candidate assembler."""
    result = executor.preparation_result(job_id, wait=wait, adopt=False)
    if result is None or result["status"] != "PREPARATION_COMPLETE":
        return result
    spec = executor.store.row(job_id)["request"]["preparation"]
    if spec["provider"] != "HCHAIN_SCOPED":
        raise ValueError("Expected an H-chain preparation batch")
    artifacts = result["artifacts"]
    if set(artifacts) != set(spec["outputs"]):
        raise ValueError("Accepted H-chain output names differ from the batch")
    # Consumers use the price JSON as the marker that its event file is ready.
    names = sorted(artifacts, key=lambda name: (not name.endswith(".events.json.xz"), name))
    executor.artifact_cache.materialize({name: artifacts[name] for name in names}, directory, immutable=True)
    executor.adopt_preparation(job_id)
    return result


# The provider's fixed state (including H8 shared TailSpectrum buffers) is
# initialized once before fork. Each child computes only its assigned leaf.
_SIGNAL_PROVIDER = None


def _signal_leaf(cell):
    return _SIGNAL_PROVIDER.signal(cell)


def signal_results(provider, cells, cpus):
    global _SIGNAL_PROVIDER
    workers = min(cpus, len(cells))
    if workers <= 1:
        yield from map(provider.signal, cells)
        return
    previous = _SIGNAL_PROVIDER
    _SIGNAL_PROVIDER = provider
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            # Ordered results keep the original target/output association.
            yield from pool.map(_signal_leaf, cells)
    finally:
        _SIGNAL_PROVIDER = previous


@contextmanager
def price_axis_slot(provider, cpus):
    """Use the fixed provider's existing two-axis fork path, not a new compiler."""
    if cpus <= 1:
        yield
        return
    pool = getattr(provider, "pool", None)
    if (pool is None or not hasattr(pool, "_axis_slot") or not hasattr(pool, "AXIS_RESERVED")
            or not callable(getattr(pool, "use_spare_axis", None))):
        raise ValueError("Fixed H-chain provider does not support reserved parallel price axes")
    prior = pool._axis_slot
    pool._axis_slot = mp.get_context("fork").Value("i", pool.AXIS_RESERVED)
    try:
        yield
    finally:
        pool._axis_slot = prior


def preparation_cpus():
    """Only the worker's selected per-attempt CPU reservation may enable parallelism."""
    value = int(os.environ.get("RPE_HCHAIN_PREPARATION_CPUS", "1"))
    if value < 1:
        raise ValueError("H-chain preparation CPU reservation must be positive")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return min(value, affinity)


def run(plan, stage, targets, assets, output, provider, *, prepare_spectra=None, cpus=1):
    selected = list(leaves(plan, stage, targets))
    state = provider.startup(assets, plan["scope"])
    if (state.binding, state.target, state.compiler) != (plan["binding"], plan["target"], plan["compiler"]):
        raise ValueError("H-chain preparation binding, target or compiler changed")
    if ("verified_saved_preparations" in plan
            and {str(key): value for key, value in state.preparation_sources.items()}
            != plan["verified_saved_preparations"]):
        raise ValueError("H-chain saved preparations changed")
    restored_plan = {"cases": state.cases}
    if stage != "signal":
        restored_plan["designs"] = provider.calibration_design(state.cases, plan["scope"]).values()
    restored = planned_inputs(restored_plan, stage)
    if any(restored.get(cell["id"]) != cell for cell, _, _ in selected):
        raise ValueError("H-chain preparation schedule differs from the fixed plan")
    if prepare_spectra is not None:
        prepare_spectra(state, (cell for cell, _, _ in selected))
    provider.STATE = state
    owner = plan["owner"]
    cohort = plan.get("price_cohort", "H6_SCOPE_" + owner)
    if type(cpus) is not int or cpus < 1:
        raise ValueError("H-chain preparation CPU reservation must be positive")
    if stage == "signal":
        values = signal_results(provider, [cell for cell, _, _ in selected], cpus)
        for (_, _, name), value in zip(selected, values, strict=True):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            provider.write(path, dict(value, owner=owner))
    else:
        for cell, index, name in selected:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with price_axis_slot(provider, cpus):
                value = provider.price(cell, index, cohort)
            provider.write_price(path, dict(value, owner=owner))


def main():
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      QISKIT_PARALLEL="FALSE", RAYON_NUM_THREADS="1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("signal", "compile", "audit"), required=True)
    parser.add_argument("--targets", type=json.loads, required=True)
    parser.add_argument("--spectral-signals", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.provider))
    import h6_scope_worker

    if Path(h6_scope_worker.__file__).resolve().parent != args.provider.resolve():
        raise ValueError("H-chain preparation worker is outside the fixed provider view")
    prepare_spectra = None
    if args.spectral_signals:
        from h8_scope_backend import prepare_tail_spectra
        prepare_spectra = prepare_tail_spectra
    run(json.loads(args.plan.read_bytes()), args.stage, args.targets, args.assets, args.output, h6_scope_worker,
        prepare_spectra=prepare_spectra, cpus=preparation_cpus())


if __name__ == "__main__":
    main()
