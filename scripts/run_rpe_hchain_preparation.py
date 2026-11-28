"""Run a batch of planned H-chain preparation leaves via the common worker."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import json
import hashlib
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


def preparation_request(plan, plan_sha, stage, targets, *, source, assets, spectral_signals=False, openblas_coretype=None, fixed_plan_sha256=None):
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
    if plan.get('provider_contract') is not None:
        from rpe_hchain_q3 import declared_provider_contract
        runtime['packages']=declared_provider_contract(plan['binding'],plan['provider_contract'])['packages']
    if openblas_coretype is not None:
        from trottertracks.rpe_preparation_worker import valid_openblas_core
        declared=(plan.get('provider_contract') or {}).get('openblas_coretype')
        if not valid_openblas_core(openblas_coretype,declared) or (declared is not None and declared!=openblas_coretype):
            raise ValueError("Unsupported fixed H-chain OpenBLAS core")
        runtime.update(openblas_coretype=openblas_coretype,imports=["numpy","scipy.linalg"])
        if declared is not None:runtime['declared_openblas_coretype']=declared
    request = dict(method="HCHAIN", route=dict(kind="preparation"),
                original_contract=dict(scope_owner=plan["owner"], binding=plan["binding"], plan_sha256=plan_sha),
                preparation=dict(stage=stage, provider="HCHAIN_SCOPED", source=source,
                    inputs={"PLAN.json": plan_sha}, outputs=outputs,
                    required_paths={"hchain_assets": assets}, runtime_requirements=runtime,
                    command=["@PYTHON@", "-B", "@TRANSPORT@/scripts/run_rpe_hchain_preparation.py",
                             "--provider", "@SOURCE@/"+(plan.get("provider_contract") or {}).get("provider_directory","scripts/h6_fixed_run"), "--assets", "@PATH:hchain_assets@",
                             "--plan", "@INPUTS@/PLAN.json", "--output", "@WORK@",
                             "--stage", stage, "--targets", json.dumps(targets, separators=(",", ":"))]))
    if fixed_plan_sha256 is not None:
        request['original_contract']['fixed_plan_sha256']=fixed_plan_sha256
    # H8 startup also compares a reconstructed numerical identity. Its
    # archived side can be checked in readiness without invoking startup.
    saved = plan.get('verified_saved_preparations') or {}
    export=(plan.get('provider_contract') or {}).get('saved_preparation_export')
    if export is not None:
        if not saved or not all(isinstance(v,dict) and 'saved' in v for v in saved.values()):
            raise ValueError('Declared saved preparation export requires fixed hashes')
        request['preparation']['saved_preparation_export']=dict(export,hashes={str(k):v['saved'] for k,v in saved.items()})
    elif plan['scope'].get('system') == 'H8' and saved and all(isinstance(v,dict) and 'saved' in v for v in saved.values()):
        request['preparation']['h8_saved_preparation_hashes'] = {str(k):v['saved'] for k,v in saved.items()}
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


class StartupMismatch(ValueError):
    """The numerical startup did execute; no preparation leaf may follow."""

    def __init__(self, field, expected, observed):
        self.field = field
        self.expected = expected
        self.observed = observed
        super().__init__('HCHAIN_STARTUP_MISMATCH: '+field)


def run(plan, stage, targets, assets, output, provider, *, prepare_spectra=None, cpus=1, check_startup=False, check_full_startup=False):
    if check_full_startup and (stage is not None or targets is not None or not check_startup):
        raise ValueError('Full startup check is identity-only and has no individual targets')
    selected = [] if check_full_startup else list(leaves(plan, stage, targets))
    # Only a new provider explicitly declaring this interface may choose a
    # prefix-scoped startup. Full identity checks and historical providers
    # retain the existing complete startup path and comparisons.
    if not check_full_startup and (plan.get('provider_contract') or {}).get('startup_prefixes') is True:
        prefixes=tuple(sorted({cell['prefix'] for cell,_,_ in selected}))
        state=provider.startup(assets,plan['scope'],prefixes=prefixes)
    else:
        state = provider.startup(assets, plan["scope"])
    def mismatch(field, expected, observed):
        if observed != expected:
            # Keep exact evidence in the body log; no tolerance or retry.
            print(json.dumps(dict(event='HCHAIN_STARTUP_MISMATCH', field=field,
                                  expected=expected, observed=observed), default=str), flush=True)
            raise StartupMismatch(field, expected, observed)
    for field in ('binding', 'target', 'compiler'):
        mismatch(field, plan[field], getattr(state, field))
    if 'verified_saved_preparations' in plan:
        mismatch('verified_saved_preparations', plan['verified_saved_preparations'],
                 {str(key): value for key, value in state.preparation_sources.items()})
    restored_plan = {"cases": state.cases}
    if check_full_startup:
        # The adopted plan may be a retained subset of the full scope. Check
        # every case and every complete calibration design, not just one
        # batch, against the same single numerical startup.
        restored_plan["designs"] = list(provider.calibration_design(state.cases,plan["scope"]).values())
        expected_cases=plan["cases"]
        actual_cases={case["id"]:case for case in state.cases}
        if len(actual_cases)!=len(state.cases) or len({case["id"] for case in expected_cases})!=len(expected_cases):
            raise ValueError('Duplicate case ID in full startup check')
        for case in expected_cases:
            mismatch('case:'+case['id'],case,actual_cases.get(case['id']))
        def canonical(value):
            return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        restored_designs=Counter(map(canonical,restored_plan["designs"]))
        for design in plan["designs"]:
            encoded=canonical(design)
            if restored_designs[encoded]<=0:
                mismatch('calibration_design',design,None)
            restored_designs[encoded]-=1
        counts={}
        for kind in ('signal','compile','audit'):
            expected_inputs=planned_inputs(plan,kind)
            actual_inputs=planned_inputs(restored_plan,kind)
            for cid,cell in expected_inputs.items():
                mismatch(kind+':'+cid,cell,actual_inputs.get(cid))
            counts[kind]=len(expected_inputs)
        return dict(status='HCHAIN_STARTUP_IDENTITY_PASS',leaf_calls=0,numerical_startup=True,
                    full_plan=True,cases=len(expected_cases),designs=len(plan["designs"]),
                    restored_cases=len(state.cases),restored_designs=len(restored_plan["designs"]),
                    schedules=counts)
    if stage != "signal":
        restored_plan["designs"] = provider.calibration_design(state.cases, plan["scope"]).values()
    restored = planned_inputs(restored_plan, stage)
    for cell, _, _ in selected:
        mismatch('schedule:'+cell['id'], cell, restored.get(cell['id']))
    if check_startup:
        return dict(status='HCHAIN_STARTUP_IDENTITY_PASS',leaf_calls=0, numerical_startup=True)
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
    parser.add_argument("--stage", choices=("signal", "compile", "audit"))
    parser.add_argument("--targets", type=json.loads)
    parser.add_argument("--spectral-signals", action="store_true")
    parser.add_argument("--check-startup", action="store_true",
                        help="Run numerical provider startup and compare identity only; requires scientific execution permission")
    parser.add_argument("--check-full-startup",action="store_true",
                        help="One numerical startup, check every planned case/design and no leaves")
    args = parser.parse_args()
    if args.check_full_startup:
        if not args.check_startup or args.stage is not None or args.targets is not None or args.spectral_signals:
            parser.error("Full startup requires --check-startup and no stage/targets/spectral signals")
    elif args.stage is None or args.targets is None:
        parser.error("--stage and --targets are required for a batch")
    sys.path.insert(0, str(args.provider))
    import importlib
    plan_bytes=args.plan.read_bytes()
    plan=json.loads(plan_bytes)
    h6_scope_worker=importlib.import_module((plan.get('provider_contract') or {}).get('worker_module','h6_scope_worker'))

    if Path(h6_scope_worker.__file__).resolve().parent != args.provider.resolve():
        raise ValueError("H-chain preparation worker is outside the fixed provider view")
    prepare_spectra = None
    if args.spectral_signals:
        from h8_scope_backend import prepare_tail_spectra
        prepare_spectra = prepare_tail_spectra
    try:
        result = run(plan, args.stage, args.targets, args.assets, args.output, h6_scope_worker,
            prepare_spectra=prepare_spectra, cpus=preparation_cpus(), check_startup=args.check_startup,check_full_startup=args.check_full_startup)
    except StartupMismatch as error:
        # Numerical startup has already executed; never classify this as a
        # prelaunch rejection or authorize another scientific call.
        from trottertracks.rpe_preparation_worker import put
        try:
            put(args.output/'HCHAIN_STARTUP_FAILURE.json', dict(
                schema='HCHAIN_STARTUP_FAILURE_V1', plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                stage=args.stage, field=error.field, numerical_startup=True, leaf_started=False,
                automatic_retry=False, budget_disposition='CONSUMED_STARTUP'))
        except OSError as write_error:
            print('Startup diagnostic write failed: '+str(write_error), file=sys.stderr, flush=True)
        raise
    if args.check_startup:
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
