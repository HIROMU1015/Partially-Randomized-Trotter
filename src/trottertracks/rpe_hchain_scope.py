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
    return path / settings.get("provider_directory","scripts/h6_fixed_run")


def pbs_profiles(distributed, node_count):
    # Explicit local-only deployments need no PBS gateway. Existing PBS
    # ownership is not migrated by this new-scope configuration.
    if type(node_count) is int and node_count == 0:
        if (distributed.get("pbs") or {}).get("workers"):
            raise ValueError("Zero PBS nodes conflicts with configured PBS workers")
        local = distributed.get("local_worker") or {}
        if not local.get("host_id") or local.get("slots", 0) < 1:
            raise ValueError("A local-only H-chain scope requires a local worker")
        return [local["host_id"]]
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


def verify_startup_contract(settings,plan,distributed):
    """Static declaration/placement check; no provider startup or leaf execution."""
    declaration=plan.get('provider_contract') or {}
    core=declaration.get('openblas_coretype',settings.get('openblas_coretype'))
    if plan['scope'].get('system')=='H8' and plan.get('verified_saved_preparations') and core is None:
        raise ValueError('Fixed H8 startup requires a declared OpenBLAS core')
    if core is not None:
        from .rpe_preparation_worker import valid_openblas_core
        if not valid_openblas_core(core,declaration.get('openblas_coretype')) or settings.get('openblas_coretype')!=core:
            raise ValueError('Fixed PLAN and selected OpenBLAS core differ')
        profiles=distributed.get('preparation_profiles') or {}
        if not profiles:raise ValueError('Missing H-chain preparation profiles for fixed OpenBLAS core')
        allowed=(distributed.get('preparation_provider_hosts') or {}).get('HCHAIN_SCOPED',profiles.keys())
        for host in allowed:
            profile=profiles.get(host)
            if profile is None:raise ValueError('Missing fixed H-chain preparation profile')
            rules=[rule['profile'] for rule in profile.get('preparation_profiles',())
                   if rule.get('provider')=='HCHAIN_SCOPED']
            if not rules and profile.get('preparation_provider')=='HCHAIN_SCOPED':rules=[profile]
            if not rules or any(p.get('openblas_coretype')!=core for p in rules):
                raise ValueError('H-chain preparation profile OpenBLAS differs: '+host)
    for field,declared in (('provider_source','startup_source_files'),('assets','startup_asset_files')):
        required=declaration.get(declared,())
        if not set(required)<=settings.get(field,{}).get('files',{}).keys():
            raise ValueError('H-chain startup file declaration is not fully bound: '+field)
    if declaration.get('provider_directory'):
        settings['provider_directory']=declaration['provider_directory']
    # The archived side of startup identity is fixed, so verify it once at
    # Start without invoking the numerical provider. Remote readiness still
    # checks the host's actual declared file bytes.
    saved=plan.get('verified_saved_preparations') or {}
    export=declaration.get('saved_preparation_export')
    if saved and all(isinstance(v,dict) and 'saved' in v for v in saved.values()) and (export or plan['scope'].get('system')=='H8'):
        from .rpe_preparation_worker import verify_h8_saved_preparation_metadata
        export=export or dict(path='inputs/h8_preparation_export.json',key_prefix='faithful:p',hash_field='preparation_hash')
        assets=settings['assets'];name=export['path']
        if name not in assets.get('files',{}):raise ValueError('Saved preparation export not declared in assets')
        root=Path(assets['controller_path']).resolve();relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid saved preparation export path')
        path=root/name
        if not path.resolve().is_relative_to(root):raise ValueError('Saved preparation export escapes assets')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=assets['files'][name]:
            raise ValueError('Controller saved preparation export hash differs')
        verify_h8_saved_preparation_metadata({'saved_preparation_export':dict(export,hashes={str(k):v['saved'] for k,v in saved.items()})},
            {'hchain_assets':dict(path=assets['controller_path'],files=assets['files'])})


def resolve(config, requested):
    from .rpe_local0_search import RequestError, keys, load

    keys(requested, {"scope", "instruction", "action", "release", "run_id"}, "request")
    scope = requested["scope"]
    keys(scope, {"system", "methods", "stop", "preparation_budgets"}, "scope")
    try:
        settings = deepcopy(config["hchain"])
        from .rpe_release_registry import common_worker_policy
        policy = common_worker_policy(config.get("release_worker_policy"))
        if settings.get("pbs_nodes", 0) > 0 and "pbs_offline_search" in policy:
            if "pbs_offline_search" in settings and settings["pbs_offline_search"] != policy["pbs_offline_search"]:
                raise ValueError("Conflicting independent PBS deployment settings")
            settings["pbs_offline_search"] = policy["pbs_offline_search"]
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
        host_ids = pbs_profiles(distributed, settings.get("pbs_nodes", 0))
        uses_pbs = bool((distributed.get("pbs") or {}).get("workers"))
        if not uses_pbs:
            settings.pop("pbs_offline_search", None)
            distributed["execution_hosts"] = host_ids
        if uses_pbs and policy.get("pbs_preparation_resources"):
            from .rpe_pbs import apply_preparation_resources
            apply_preparation_resources(distributed, policy["pbs_preparation_resources"])
        if uses_pbs and policy.get("pbs_q3_cpu_policies"):
            from .rpe_host_profiles import apply_pbs_q3_cpu_policies
            apply_pbs_q3_cpu_policies(distributed, policy["pbs_q3_cpu_policies"])
        if uses_pbs and policy.get("pbs_q3_measured_memory"):
            from .rpe_pbs import apply_q3_measured_memory
            apply_q3_measured_memory(distributed,policy["pbs_q3_measured_memory"])
        if uses_pbs and "pbs_q3_memory_learning" in policy:
            from .rpe_q3_memory import apply_policy
            apply_policy(distributed,policy["pbs_q3_memory_learning"])
        # Apply the same policy knobs that the H-chain shared executor consumes.
        # H4-specific provider/route choices are deliberately not substituted.
        applied = {k for k in ('pbs_offline_search', 'pbs_preparation_resources',
                               'pbs_q3_cpu_policies', 'pbs_q3_measured_memory',
                               'pbs_q3_memory_learning') if uses_pbs and k in policy}
        for field in ('ready_queue_waves', 'live_readiness_max_workers', 'live_readiness_refill_seconds',
                      'controller_queue_first', 'preparation_host_preference'):
            if field in policy:
                distributed[field] = deepcopy(policy[field]); applied.add(field)
        if policy.get('controller_parallel') is True:
            spec.setdefault('search_overrides', {})['controller_parallel'] = True
            if 'live_readiness_max_workers' not in policy:
                distributed.pop('live_readiness_max_workers', None)
            applied.add('controller_parallel')
        if uses_pbs:
            from .rpe_pbs import apply_common_worker_knobs, apply_common_artifact_staging
            applied.update(apply_common_worker_knobs(distributed,policy))
            if apply_common_artifact_staging(distributed,policy):
                applied.add('pbs_stage_artifacts_over_ssh')
        placements = policy.get('preparation_provider_hosts') or {}
        if 'HCHAIN_SCOPED' in placements:
            hosts = placements['HCHAIN_SCOPED']
            configured = distributed.get('preparation_profiles') or {}
            if not isinstance(hosts, list) or not hosts or not set(hosts) <= configured.keys():
                raise ValueError('H-chain preparation placement requires configured hosts')
            distributed.setdefault('preparation_provider_hosts', {})['HCHAIN_SCOPED'] = list(hosts)
            applied.add('preparation_provider_hosts')
        if policy:
            settings['worker_policy_disposition'] = dict(applied=sorted(applied),
                not_applied=sorted(set(policy)-applied))
        verify_startup_contract(settings,plan,distributed)
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
        required_stages={'signal'} if plan.get('cases') else set()
        if any(design.get('anchors') for design in plan.get('designs',())):required_stages.add('compile')
        if any(design.get('holdout') for design in plan.get('designs',())):required_stages.add('audit')
        if any(prep.get(stage,0) and stage not in settings['batch_sizes'] for stage in required_stages):
            raise ValueError('Declare batch_sizes for the fixed preparation stages')
        canonical = dict(system=system, methods=methods, hchain_scope=frozen, execution_hosts=host_ids,
                         preparation_budgets=prep, stop=stop)
        return problem, spec, canonical
    except (OSError, KeyError, ValueError, TypeError) as error:
        if isinstance(error, RequestError):
            raise
        raise RequestError("scope.hchain", str(error)) from error
