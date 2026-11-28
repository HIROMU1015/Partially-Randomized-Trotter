"""Reuse reconciled H-chain predecessor inputs and Q3 in the common controller."""
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

from rpe_hchain_q3 import METHOD_LABELS, prepared_route, request_inputs
from trottertracks import rpe_method_optimizer as optimizer


def checked(settings, plan, *, verify_files=False, metadata_only=False, receipt_plan=None, cache_path=None):
    if "legacy" not in settings:
        return None
    fixed = settings["legacy"]
    root = Path(fixed["directory"]).resolve()
    raw = (root / "LEGACY.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != fixed["sha256"]:
        raise ValueError("Legacy H-chain handoff changed")
    history = json.loads(raw, parse_int=optimizer._outcome_int)
    if (history["schema"] != "RPE_HCHAIN_LEGACY_V1" or history["owner"] != plan["owner"]
            or history["plan_sha256"] != settings["plan"]["sha256"]):
        raise ValueError("Legacy H-chain handoff belongs to another fixed scope")
    if verify_files:
        # The PBS-only controller needs the fixed Q3 request and its bundled
        # candidate inputs. Large preparation/raw result files stay at the
        # compute site, where the normal child performs the full verification.
        needed = ({name for files in history["cases"].values() for name in files.values()} |
                  {folder + "/REQUEST.json" for folder in history["jobs"]} |
                  {"PLAN.json", "PURE_RTE_COST_SCREEN.json"}) if metadata_only else None
        allowed_receipts = receipt_names(receipt_plan) if receipt_plan is not None else None
        verified = set()
        # Disposable run-local cache, never part of the legacy evidence.
        # Bind it to the fixed manifest and root; rehash any changed identity.
        cache_path = Path(cache_path) if cache_path is not None else None
        cache_binding = dict(root=str(root), manifest_sha256=fixed["sha256"])
        cached = {}
        if cache_path is not None:
            try:
                old = json.loads(cache_path.read_bytes())
                if old.get("binding")==cache_binding and isinstance(old.get("files"),dict):
                    cached = old["files"]
            except (OSError,ValueError,AttributeError):
                pass
        updated = {}

        def verify(name):
            if name in verified:return
            expected = history["files"][name]
            relative = PurePosixPath(name)
            path = root / relative
            if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                    or not path.resolve().is_relative_to(root)):
                raise ValueError(f"Legacy H-chain evidence changed: {name}")
            identity = _identity(path)
            row = dict(identity=list(identity),sha256=expected)
            stable = time.time_ns()-max(identity[-2:])>=1_000_000_000
            if not (stable and cached.get(name)==row):
                if digest(path) != expected:
                    raise ValueError(f"Legacy H-chain evidence changed: {name}")
            if _identity(path)!=identity:
                raise ValueError(f"Legacy H-chain evidence changed: {name}")
            if stable:
                updated[name]=row
            verified.add(name)

        selected_jobs = None
        if receipt_plan is not None:
            selected = {case["id"] for case in receipt_plan["cases"]}
            all_case_files = {name for files in history["cases"].values() for name in files.values()}
            selected_case_files = {name for cid, files in history["cases"].items()
                                   if cid in selected for name in files.values()}
            # V1 legacy manifests have no folder->case index. Hash each small
            # REQUEST to bind that mapping, but avoid all unselected raw
            # results and candidate packets. The full inventory and all
            # charges are still checked below, independent of the block.
            selected_jobs = []
            all_jobs = set(history["jobs"])
            for folder in history["jobs"]:
                name = folder + "/REQUEST.json"
                verify(name)
                raw = (root/name).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=history["files"][name]:
                    raise ValueError(f"Legacy H-chain request changed: {name}")
                request = json.loads(raw)
                if request["configuration_id"] in selected:
                    selected_jobs.append(folder)
            selected_job_set = set(selected_jobs)

        def required(name):
            if needed is not None and name not in needed:return False
            if allowed_receipts is None:return True
            if name.startswith(('signals/','prices/')):
                return name in allowed_receipts
            if name in all_case_files:
                return name in selected_case_files
            folder = str(PurePosixPath(name).parent)
            if folder in all_jobs:
                return folder in selected_job_set
            return True

        for name in history["files"]:
            if required(name):verify(name)
        for name in history["files"]:
            if not metadata_only and required(name) and name.startswith("prices/") and name.endswith(".json"):
                receipt = json.loads((root / name).read_bytes())["events"]
                if isinstance(receipt, dict):
                    sidecar = str(Path(name).with_suffix(".events.json.xz"))
                    if history["files"].get(sidecar) != receipt["sha256"]:
                        raise ValueError(f"Legacy event archive differs: {name}")
        candidate_ids = {case["id"] for case in plan["cases"]}
        charges = (*history["records"], *history["holds"])
        identities = {(row["configuration_id"], tuple(row["N_X"]), tuple(row["N_Y"])) for row in charges}
        if (len(identities) != len(charges) or len(charges) != sum(history["spent"].values())
                or any(row[0] not in candidate_ids for row in identities)):
            raise ValueError("Legacy H-chain Q3 charge inventory is inconsistent")
        if cache_path is not None and updated != cached:
            # One owning package/controller writes this disposable cache.
            # Failure to cache never invalidates otherwise verified evidence.
            try:
                from trottertracks.rpe_q3_worker import atomic
                atomic(cache_path,dict(binding=cache_binding,files=updated))
            except OSError:
                pass
        if selected_jobs is not None:
            history = dict(history, jobs=selected_jobs)
    return root, history


def _identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_digest(path, identity):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    if _identity(path) != identity:
        raise ValueError('Legacy evidence changed during hash verification: '+str(path))
    return value


_verified_digest = lru_cache(maxsize=16384)(_read_digest)


def digest(path):
    path = Path(path)
    identity = _identity(path)
    # A freshly written file can have the same stat timestamp after another
    # write on a coarse-grained filesystem. Only cache stable old identities.
    if time.time_ns()-max(identity[-2:]) < 1_000_000_000:
        return _read_digest(path, identity)
    return _verified_digest(path, identity)


def receipt_names(plan):
    """Exact selected receipt closure, including the shared price design leaves."""
    from run_rpe_hchain_preparation import planned_inputs
    allowed = {f"signals/{cid}.json" for cid in planned_inputs(plan, "signal")}
    for stage in ("compile", "audit"):
        for cid, cell in planned_inputs(plan, stage).items():
            for index in range(plan["scope"]["price_paths"] if cell["method"] == "PR" else 1):
                allowed.add(f"prices/{cid}-{index}.json")
                allowed.add(f"prices/{cid}-{index}.events.json.xz")
    return allowed


def restore_receipts(run, legacy, *, plan=None):
    root, history = legacy
    allowed = receipt_names(plan) if plan is not None else None
    for name, expected in history["files"].items():
        if not name.startswith(("signals/", "prices/")) or (allowed is not None and name not in allowed):
            continue
        target = run / name
        target.parent.mkdir(exist_ok=True)
        source = root / name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        if target.is_symlink() or (not target.samefile(source) and digest(target) != expected):
            raise ValueError(f"Run already has different saved H-chain evidence: {name}")


def install(plan, legacy, rows, routes, expected):
    root, history = legacy
    # Index once: large H6 continuations contain thousands of launches.
    by_case = {}
    for folder in history["jobs"]:
        original = json.loads((root / folder / "REQUEST.json").read_bytes())
        by_case.setdefault(original["configuration_id"], []).append(original)
    for cid, files in history["cases"].items():
        if cid not in rows:
            continue
        paths = {key: root / relative for key, relative in files.items()}
        entry = dict(configuration_id=cid, configuration=str(paths["configuration"]),
                     configuration_sha256=history["files"][files["configuration"]],
                     signal_packet=str(paths["signal_packet"]),
                     signal_packet_sha256=history["files"][files["signal_packet"]])
        route = prepared_route(entry, paths["registered_prices"], owner=history.get("case_owners", {}).get(cid, plan["owner"]), target=plan["target"],
                               binding=plan["binding"], interpreter=sys.executable)
        row = rows[cid]
        method = row["method"]
        axes = json.loads(paths["registered_prices"].read_bytes())["methods"][METHOD_LABELS[method]]["axes"]
        # Verify at the current adapter boundary for every original allocation,
        # including the charged attempt with a held result.
        for original in by_case.get(cid, []):
            request_inputs(dict(original, route=route))
        row.update(ready=True, needs=[], prices=[optimizer.frac(axis["unit_price_RZ"]) for axis in axes],
                   signal_source=entry["signal_packet"], price_axes=axes, price_status="SOURCE_BOUND_HCHAIN_PRICES")
        expected[cid] = dict(M=row["M"], signal_source=entry["signal_packet"])
        routes[method].append(route)


def remaining(budgets, history):
    return {method: cap if cap == "UNLIMITED" else max(0, cap - history["spent"].get(method, 0))
            for method, cap in budgets.items()}


def resource_peaks(legacy, runtime, *, memory_root=None):
    """Translate verified old H-chain RSS into the matching current Q3 classes.

    Resource estimates are not scientific outcomes. Only the same evaluator,
    interpreter, CPU allocation and exact recorded allocation are eligible.
    """
    from trottertracks.rpe_q3_resources import measured_resource_class
    from rpe_hchain_q3 import RISK_COMMIT
    root, history = legacy
    memory_root = Path(memory_root) if memory_root is not None else root
    def key(row):
        return row["configuration_id"], tuple(row["N_X"]), tuple(row["N_Y"])
    completed = {key(row):row["status"] for row in history["records"]}
    peaks, counts = {}, {}
    rejected = 0
    for folder in history["jobs"]:
        name = folder + "/Q3_MEMORY.json"
        if name not in history["files"]:
            continue
        path = memory_root / name
        request_path = root / folder / "REQUEST.json"
        if not path.is_file():
            continue
        if digest(path) != history["files"][name] or digest(request_path) != history["files"][folder+"/REQUEST.json"]:
            raise ValueError("Legacy resource receipt changed: " + folder)
        request = json.loads(request_path.read_bytes(),parse_int=optimizer._outcome_int)
        sample = json.loads(path.read_bytes())
        route = request["route"]
        cpu = route.get("allocated_cpus")
        runtime_cpu = max(runtime.get("task_cpu_count") or 1, runtime.get("dp_workers") or 1)
        shots = optimizer.interleave(request["N_X"], request["N_Y"])
        peak = sample.get("sampled_peak_rss_bytes")
        if (route.get("kind") not in ("h6_df19_scoped_q3","h8_df28_scoped_q3")
                or route.get("view_commit") != RISK_COMMIT
                or route.get("interpreter") != runtime.get("interpreter")
                or type(cpu) is not int or cpu != runtime_cpu
                or runtime.get("dp_workers") != max(1,cpu-1)
                or sample.get("method") != request["method"] or sample.get("shots") != list(shots)
                or sample.get("status") != "FINISHED" or sample.get("usable") is not True
                or sample.get("returncode") != 0 or key(request) not in completed
                or sample.get("outcome_status") != completed[key(request)]
                or type(peak) is not int or peak <= 0):
            rejected += 1
            continue
        category = measured_resource_class(request, runtime=runtime, preparation_detail=True)
        entry = peaks.setdefault(category,dict(peak_rss_bytes=0,observed_cpu_cores=0,dp_workers=runtime_cpu))
        entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"],peak)
        counts[category] = counts.get(category,0)+1
    return dict(schema="RPE_HCHAIN_LEGACY_RESOURCE_PEAKS_V1",source_job=history["source_job"],
                task_resource_peaks=peaks,counts=counts,rejected=rejected)


def seed_resource_state(previous, seed):
    """Merge without shrinking a live class peak or altering owned jobs."""
    from copy import deepcopy
    state = deepcopy(previous)
    peaks = state.setdefault("task_resource_peaks",{})
    for category,sample in seed["task_resource_peaks"].items():
        entry = peaks.setdefault(category,{})
        for metric in ("peak_rss_bytes","observed_cpu_cores","dp_workers"):
            entry[metric] = max(entry.get(metric,0),sample.get(metric,0))
    isolated = state.get("isolated_resource_class")
    if isolated in seed["task_resource_peaks"]:
        state.pop("isolated_resource_class",None)
    state["legacy_resource_source"] = dict(source_job=seed["source_job"],counts=seed["counts"])
    return state


def seed_local_worker(legacy, distributed):
    profile = distributed.get("local_worker") or {}
    if not (profile.get("slot_policy") or {}).get("separate_task_resources"):
        return
    work = Path(profile["work_dir"])
    work.mkdir(parents=True,exist_ok=True)
    path = work / "SLOT_STATE.json"
    state = json.loads(path.read_bytes()) if path.is_file() else {}
    if state.get("legacy_resource_source",{}).get("source_job") == legacy[1]["source_job"]:
        return
    seed = resource_peaks(legacy,profile)
    if seed["task_resource_peaks"]:
        from trottertracks.rpe_q3_worker import atomic
        atomic(path,seed_resource_state(state,seed))
