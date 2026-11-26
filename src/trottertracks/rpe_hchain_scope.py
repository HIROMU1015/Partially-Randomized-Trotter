"""Resolve a fixed H-chain plan for the ordinary RPE Start and Resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

from . import rpe_scaled_shots as scaled


STAGES = ("signal", "compile", "audit")


def fixed_plan(settings):
    record = settings["plan"]
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("H-chain fixed PLAN.json changed")
    plan = json.loads(raw)
    if not plan["cases"] or plan["scope"]["system"] != plan["binding"]["system"]:
        raise ValueError("H-chain plan lacks a consistent physical scope")
    return raw, plan


def provider_root(settings):
    path = Path(settings["provider_root"]).resolve()
    source = settings["provider_source"]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip() != source["commit"]:
        raise ValueError("H-chain provider HEAD differs")
    for name, expected in source["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"H-chain provider source differs: {name}")
    return path / "scripts/h6_fixed_run"


def pbs_profiles(distributed, node_count):
    pbs = distributed["pbs"]
    workers = pbs["workers"]
    hosts = [worker["host_id"] for worker in workers]
    if type(node_count) is not int or node_count < 1 or len(hosts) != node_count or len(set(hosts)) != len(hosts):
        raise ValueError("The fixed PBS allocation requires distinct worker profiles")
    if node_count > 1 and distributed.get("pbs_hold_node") is True:
        raise ValueError("The shared single-node pilot cannot own these allocations")
    dispatch = distributed["pbs_dispatch_gateway"]
    if (not all(dispatch.get(key) for key in ("compute_storage_sampling", "bounded_results", "defer_q3_evidence",
                                               "split_exchange", "selective_harvest"))
            or not distributed.get("pbs_gateway_storage_launcher")
            or not distributed.get("pbs_gateway_completion_workers")):
        raise ValueError("PBS compute storage and bounded completion transport must be configured together")
    sizing = pbs.setdefault("allocation_sizing", {})
    if sizing.get("per_task_resources") is not True:
        raise ValueError("The fixed PBS worker profiles require per-task CPU and memory accounting")
    sizing["fixed_node_resources"] = True
    if not all(worker["profile"].get("storage_limit") for worker in workers):
        raise ValueError("All PBS workers require the run's storage limit")
    return hosts


def resolve(config, requested):
    from .rpe_local0_search import RequestError, keys, load

    keys(requested, {"scope", "instruction", "action", "release", "run_id"}, "request")
    scope = requested["scope"]
    keys(scope, {"system", "methods", "stop", "preparation_budgets"}, "scope")
    try:
        settings = deepcopy(config["hchain"])
        _, plan = fixed_plan(settings)
        frozen = plan["scope"]
        system = frozen["system"]
        methods = scope.get("methods", list(frozen["methods"]))
        if scope.get("system", system) != system or not isinstance(methods, list) or not methods:
            raise ValueError("Choose methods and system in the fixed H-chain plan")
        if len(set(methods)) != len(methods) or any(m not in frozen["methods"] for m in methods):
            raise ValueError("Requested methods differ from the fixed H-chain plan")
        spec = load(config["round"])
        distributed = spec["executor"]["distributed"]
        if spec["executor"].get("scheduler") != "concurrent":
            raise ValueError("A shared distributed executor is required")
        host_ids = pbs_profiles(distributed, settings["pbs_nodes"])
        prep = distributed.setdefault("preparation", {}).setdefault("budgets", {})
        changes = scope.get("preparation_budgets", {})
        keys(changes, set(STAGES), "scope.preparation_budgets")
        prep.update(changes)
        stop = scope.get("stop", {})
        keys(stop, {"max_new_q3_per_method", "max_steps_per_method"}, "scope.stop")
        budgets = stop.get("max_new_q3_per_method", {m: spec["max_new_q3_per_method"][m] for m in methods})
        if not isinstance(budgets, dict):
            budgets = {m: budgets for m in methods}
        if set(budgets) != set(methods) or any(
                value != "UNLIMITED" and (type(value) is not int or value < 0)
                for value in (*budgets.values(), *prep.values())):
            raise ValueError("Use nonnegative task budgets or UNLIMITED")
        max_steps = stop.get("max_steps_per_method", spec.get("max_steps_per_method", 100000))
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("A positive optimization step limit is required")
        policy = scaled.settings(dict(frozen["scaled_shots"], M=[frozen["M"]]))
        problem = dict(problem=dict(system=system, binding=plan["binding"]), rpe=dict(M=frozen["M"]),
                       target=dict(energy_ha=plan["target"]), accuracy=dict(epsilon_E_decimal=scaled.EPSILON),
                       methods={m: dict(frozen["methods"][m], family="PR" if m == "PR_DF" else "PF") for m in methods},
                       execution_routes={}, search=dict(rule=scaled.RULE, scaled_shots=policy,
                           work_conserving_search=True, max_steps_per_method=max_steps))
        spec.update(hchain=settings, methods=methods, max_new_q3_per_method=budgets,
                    max_steps_per_method=max_steps, science_approved=requested["action"] in ("start", "prepare"))
        settings.setdefault("batch_sizes", {})
        if any(type(n) is not int or n < 1 for n in settings["batch_sizes"].values()):
            raise ValueError("Preparation batch sizes must be positive")
        canonical = dict(system=system, methods=methods, hchain_scope=frozen, execution_hosts=host_ids,
                         preparation_budgets=prep, stop=stop)
        return problem, spec, canonical
    except (OSError, KeyError, ValueError, TypeError) as error:
        if isinstance(error, RequestError):
            raise
        raise RequestError("scope.hchain", str(error)) from error
