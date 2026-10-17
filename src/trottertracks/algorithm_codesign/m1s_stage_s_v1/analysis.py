"""Future cap-edge evaluation. No saved input is loaded by import or toy tests."""
from multiprocessing import Pool
from fractions import Fraction as F
import os
from pathlib import Path
import resource

from .control import require_permit
from .adapter import prepare_saved
from .arithmetic import Bounds, interval_text, rational_text
from .kernels import margin_at_cap, compare_cap


def worker_count(job_count):
    """Workload and available affinity determine parallelism; no user core cap."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return min(job_count, available)


def cell_job(payload):
    """Pure scalar batch, also usable for artificial parallel/serial equivalence."""
    cell, methods, cap, K, bits, resources = payload
    B = Bounds(bits, resources)
    margins = {m: margin_at_cap(A, bias, cap, B) for m, (A, bias) in methods.items()}
    outputs = []
    a, u = methods["shared_pm"]
    for comparator in ["ordinary", "independent_pm"]:
        c, v = methods[comparator]
        data = compare_cap(a, u, c, v, margins["shared_pm"], margins[comparator], K, B)
        data["input_bounds"] = {"A_shared_lower": rational_text(a),
                                "A_comparator_lower": rational_text(c),
                                "bias_shared_upper_rad": rational_text(u),
                                "bias_comparator_upper_rad": rational_text(v)}
        data["assumptions"] = "DIRECT_TRADEOFF_POSITIVE_MARGIN_SAVED_GATE_SATISFIED"
        data["model_drift_added_again"] = False
        outputs.append((cell + (comparator,), data))
    return outputs, B.steps, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def evaluate(v1, pi3, permit, result, checkpoint):
    require_permit(permit)
    c = permit.contract
    cap, jobs, metadata, adapter_info = prepare_saved(v1, pi3, permit)
    result["comparisons"] = metadata
    for item in result["comparisons"]:
        if item["old_pi3_gate_status"] == "NOT_SATISFIED":
            item["new_stage_s_status"] = "CARRIED_PI3_NOT_SATISFIED_NOT_EVALUATED"
        elif item["split"] == "G9":
            item["new_stage_s_status"] = "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED"
    result["adapter"] = adapter_info
    result["cap"] = {"interval_rad": interval_text(cap), "endpoint_open": True,
                     "gamma_grid": None, "pi_recomputed": False}
    science = c["selected_science"]
    B = Bounds(science["arithmetic_bits"], c["resources"])
    alpha_axis = F(science["alpha_row"]) / 2
    log = B.log_above_one(2 / alpha_axis)
    K = B.pack((4 * log[0], 4 * log[1]))
    result["statistics"] = {"alpha_axis": rational_text(alpha_axis), "K_interval": interval_text(K),
                            "confidence_scope": science["confidence_scope"],
                            "fresh_IID_physical_verified": False}
    result["arithmetic"] = {"bits": science["arithmetic_bits"], "bound_steps": B.steps,
                            "refinements": 0, "scalar_margins_planned": 36, "sin_endpoint_calls_planned": 72,
                            "scalar_margins_completed_in_returned_cells": 0,
                            "sin_endpoint_calls_completed_in_returned_cells": 0,
                            "log_calls": 1, "exact_interval_strings_are_primary": True}
    workers = worker_count(len(jobs))
    result["resources"] = {"logical_cpus": os.cpu_count(), "selected_workers": workers,
                           "worker_policy": c["resources"]["worker_policy"],
                           "CPU_RAM_GPU_worker_caps": None,
                           "backend": c["resources"]["backend"],
                           "GPU_used": False, "GPU_reason": "exact rational CPU arithmetic; no GPU backend",
                           "BLAS_OpenMP_NumPy_Qiskit_imports": False,
                           "memory_available_before": [s for s in Path("/proc/meminfo").read_text().splitlines()
                                                       if s.startswith(("MemTotal:", "MemAvailable:"))],
                           "observed_worker_peak_RSS_KiB": []}
    checkpoint()
    payloads = [(cell, methods, cap, K, science["arithmetic_bits"], c["resources"])
                for cell, methods in jobs.items()]
    outputs = {tuple(item[k] for k in ["split", "R", "delta", "T", "comparator"]): item
               for item in result["comparisons"]}

    def consume(record):
        pairs, steps, rss = record
        result["arithmetic"]["bound_steps"] += steps
        if result["arithmetic"]["bound_steps"] > c["resources"]["max_bound_steps"]:
            raise RuntimeError("aggregate bound-step cap")
        result["resources"]["observed_worker_peak_RSS_KiB"].append(rss)
        result["arithmetic"]["scalar_margins_completed_in_returned_cells"] += 3
        result["arithmetic"]["sin_endpoint_calls_completed_in_returned_cells"] += 6
        for key, fields in pairs:
            outputs[key].update(fields)
        checkpoint()

    if workers > 1:
        pool = Pool(processes=workers)
        try:
            # Stable input/output ordering, independent of scheduling. No random seeds.
            for record in pool.imap(cell_job, payloads, chunksize=1):
                consume(record)
        except BaseException:
            # Only this Pool's owned workers; no signals to shared/foreign processes.
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    else:
        for payload in payloads:
            consume(cell_job(payload))
    if any(x["new_stage_s_status"] == "NOT_EVALUATED" for x in result["comparisons"]):
        raise RuntimeError("eligible outputs missing")
    result["resources"]["parent_peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["resources"]["children_peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result["resources"]["RSS_scope"] = "per-process peaks; not an aggregate concurrent peak"
    checkpoint()
