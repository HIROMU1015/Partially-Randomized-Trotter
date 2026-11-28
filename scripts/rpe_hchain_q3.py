"""Evaluate a prepared H-chain Q3 request with the fixed scoped evaluator."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


KIND = "hchain_scoped_q3"
RISK_COMMIT = "cbc471c913072de45c25a0ed22d51dff8a9b7034"
METHOD_LABELS = {"FULL_PF": "Full PF", "TRUNCATED_PF": "B0I", "PR_DF": "PR-DF"}
METHODS = {"Full PF": "deterministic", "B0I": "deterministic", "PR-DF": "ideal_finite_RTE_law"}
EPSILON = "0.00159360010199040"


def read(path):
    return json.loads(path.read_bytes())


def write(path, record, *, compact=False):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            formatting = dict(separators=(",", ":")) if compact else dict(indent=2)
            json.dump(record, stream, allow_nan=False, **formatting)
            stream.write("\n")
            stream.close()
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def check_registered_prices(registry, signal_sha, compiler, method, M):
    labels = set(registry["methods"])
    if (registry["signal_packet_sha256"] != signal_sha or registry["compiler"] != compiler
            or method not in labels or not labels <= set(METHODS)):
        raise ValueError("Registered prices are for a different signal packet or compiler")
    expected = [(m, axis) for m in range(M + 1) for axis in ("X", "Y")]
    entry = registry["methods"][method]
    axes = entry["axes"]
    if [(row["m"], row["axis"]) for row in axes] != expected:
        raise ValueError(f"{method} requires all registered physical-axis prices in X/Y round order")
    for row in axes:
        price = exact_price(row["unit_price_RZ"])
        if price <= 0 or not row.get("price_evidence") or row.get("status") != "ADMITTED":
            raise ValueError(f"{method} has prices without accepted evidence")
    return entry


def exact_price(value):
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


def declared_provider_contract(binding, contract):
    """Optional fixed PLAN metadata; the historical H6/H8 route stays unchanged."""
    if contract is None:return None
    required={'system','rank','packages'}
    optional={'openblas_coretype','provider_directory','entry_module','worker_module','priority_function',
              'startup_source_files','startup_asset_files','signal_phase_cache_policy','saved_preparation_export','startup_prefixes'}
    if not isinstance(contract,dict) or not required<=contract.keys() or set(contract)-required-optional:
        raise ValueError('H-chain provider contract must declare system, rank and runtime packages')
    if (not isinstance(contract['system'],str) or not contract['system']
            or type(contract['rank']) is not int or contract['rank']<1
            or (contract['system'],contract['rank'])!=(binding.get('system'),binding.get('rank'))):
        raise ValueError('H-chain provider contract differs from the fixed binding')
    packages=contract['packages']
    if (not isinstance(packages,dict) or not {'qiskit','numpy'}<=packages.keys()
            or any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in packages.items())):
        raise ValueError('H-chain provider runtime package versions must be fixed')
    from pathlib import PurePosixPath
    result=dict(system=contract['system'],rank=contract['rank'],packages=dict(packages))
    for key in optional & contract.keys():
        value=contract[key]
        if key=='openblas_coretype':
            from trottertracks.rpe_preparation_worker import valid_openblas_core
            if not valid_openblas_core(value,value):raise ValueError('Invalid fixed OpenBLAS core')
        if key in ('entry_module','worker_module','priority_function') and (not isinstance(value,str) or not value.isidentifier()):
            raise ValueError('Invalid provider entry name')
        if key in ('provider_directory','startup_source_files','startup_asset_files'):
            paths=[value] if key=='provider_directory' else value
            if not isinstance(paths,list) or any(not isinstance(path,str) or not path or PurePosixPath(path).is_absolute()
                  or '..' in PurePosixPath(path).parts for path in paths):
                raise ValueError('Invalid provider startup path')
        if key=='startup_prefixes' and type(value) is not bool:raise ValueError('Invalid prefix-scoped startup declaration')
        if key=='saved_preparation_export':
            if (not isinstance(value,dict) or set(value)!={'path','key_prefix','hash_field'}
                    or any(not isinstance(v,str) or not v for v in value.values())
                    or PurePosixPath(value['path']).is_absolute() or '..' in PurePosixPath(value['path']).parts):
                raise ValueError('Invalid saved preparation export declaration')
        result[key]=value
    return result


def supported_binding(binding,contract=None):
    if contract is not None:return declared_provider_contract(binding,contract) is not None
    return (binding.get('system'),binding.get('rank')) in (('H6',19),('H8',28))


def prepared_route(entry, price_path, *, owner, target, binding, interpreter, view=None, provider_contract=None):
    """Pack one candidate's existing, checksummed configuration and round inputs."""
    price_path = Path(price_path)
    documents = {
        "configuration": (Path(entry["configuration"]), entry["configuration_sha256"]),
        "signal_packet": (Path(entry["signal_packet"]), entry["signal_packet_sha256"]),
        "registered_prices": (price_path, None),
    }
    bundle = {}
    for name, (path, expected) in documents.items():
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if expected is not None and checksum != expected:
            raise ValueError(f"Prepared {name} bytes changed")
        bundle[name] = dict(json=raw.decode("utf-8"), sha256=checksum)
    cid = entry["configuration_id"]
    route = dict(kind=KIND, interpreter=interpreter, view_commit=RISK_COMMIT,
                 scope_owner=owner, target_energy=target, binding=binding,
                 configuration_ids=[cid], configurations={cid: bundle})
    if provider_contract is not None:
        route['provider_contract']=declared_provider_contract(binding,provider_contract)
    if view is not None:
        route["view"] = str(view)
    return route


def prepared_case(provider, directory, plan, case, models, *, interpreter, view=None):
    """Build a shared Q3 route from the scoped provider's accepted prices/signals."""
    axes, entry, price_path = provider.candidate_inputs(directory, plan, case, models)
    route = prepared_route(entry, price_path, owner=plan["owner"], target=plan["target"],
                           binding=plan["binding"], interpreter=interpreter, view=view,
                           provider_contract=plan.get("provider_contract"))
    return axes, entry, route


def request_inputs(request):
    route = request["route"]
    if route["kind"] != KIND or route["view_commit"] != RISK_COMMIT:
        raise ValueError("Wrong scoped Q3 route or risk source")
    cid = request["configuration_id"]
    bundle = route["configurations"][cid]
    for name, item in bundle.items():
        if hashlib.sha256(item["json"].encode("utf-8")).hexdigest() != item["sha256"]:
            raise ValueError(f"Prepared {name} bytes differ from the Q3 assignment")
    config = json.loads(bundle["configuration"]["json"])
    signals = json.loads(bundle["signal_packet"]["json"])
    registry = json.loads(bundle["registered_prices"]["json"])
    name = METHOD_LABELS[request["method"]]
    M, h, target = signals["M"], signals["h"], signals["target"]
    binding = config["spec"]["binding"]
    owner = route["scope_owner"]
    if (type(M) is not int or M < 0 or not 0 < h < math.inf or not math.isfinite(target)
            or target != route["target_energy"] or binding != route["binding"]
            or not supported_binding(binding,route.get("provider_contract"))
            or signals["status"] != "SIGNAL_PACKETS_READY" or signals["scope_owner"] != owner
            or signals["run_id"] != owner or config["id"] != cid or config["spec"]["M"] != M
            or request["expected"]["configuration_id"] != cid
            or request["expected"].get("M", M) != M or request.get("origin", {}).get("M", M) != M):
        raise ValueError("Scoped Q3 inputs disagree on the candidate, binding or evaluation horizon")
    method_rows = signals["methods"][name]
    if [row["m"] for row in method_rows] != list(range(M + 1)):
        raise ValueError("Scoped Q3 requires all requested rounds")
    for config_round, row in zip(config["rounds"], method_rows, strict=True):
        candidate = row.get("candidate")
        mean = row["signal"]["raw"]
        if set(mean) != {"real", "imag"} or any(not math.isfinite(v) or abs(v) > 1 for v in mean.values()):
            raise ValueError("Invalid binary64 raw measurement mean")
        if (row["m"] != config_round["m"] or candidate is not None
                and (candidate["q"] != config_round["q_PF"] or candidate["R_outer"] != config_round["R_outer"]
                     or candidate["formula"] != config["spec"]["formula"]
                     or candidate["prefix"] != config["spec"]["L_D"])):
            raise ValueError("Signal schedule differs from the canonical candidate")
    shots = request["N_X"]
    if len(shots) != M + 1 or shots != request["N_Y"] or any(type(n) is not int or n <= 0 for n in shots):
        raise ValueError("The scoped evaluator requires a positive symmetric allocation")
    signal_sha = bundle["signal_packet"]["sha256"]
    price = check_registered_prices(registry, signal_sha, signals["compiler"], name, M)
    if price["configuration_id"] != cid or price["configuration_sha256"] != bundle["configuration"]["sha256"]:
        raise ValueError("Registered prices belong to a different configuration")
    for row in price["axes"]:
        if row["kind"] == "CONDITIONAL_STEP_TRANSFER_PRICE":
            raise ValueError("Scoped Q3 requires a model calibrated at the physical step")
        if row["kind"] == "CONDITIONAL_COMPILED_PRICE_MODEL":
            step = config["rounds"][row["m"]]["actual_delta_pf"]
            evidence = row["price_evidence"]
            if evidence.get("calibration_step") != step or evidence.get("target_step") != step:
                raise ValueError("Price calibration differs from the physical step")
    cost = sum(shots[row["m"]] * exact_price(row["unit_price_RZ"]) for row in price["axes"])
    if cost != exact_price(request["cost"]):
        raise ValueError("Registered prices do not reproduce the Q3 request cost")
    source = dict(view_commit=RISK_COMMIT, source_job=owner, signal_packet_sha256=signal_sha,
                  configuration_id=cid, evaluator="rpe_total_risk", route=KIND)
    identity = dict(configuration_id=cid, M=M, contract_matches=True,
                    allocation=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(shots)])
    return name, method_rows, shots, h, target, source, identity


@contextmanager
def reuse_round_tables(risk):
    """Share exact count tables for the lifetime of one Q3 allocation."""
    build_table = risk.round_distribution
    tables = {}
    counts = {"built": 0, "reused": 0}

    def round_distribution(row, *, h, target_energy, radius, probability_rule, maximum_outcomes):
        raw = row["raw"]
        key = (row["m"], float(raw["real"]).hex(), float(raw["imag"]).hex(), row["N_X"], row["N_Y"],
               float(h).hex(), float(target_energy).hex(), float(radius).hex(), probability_rule, maximum_outcomes)
        if key not in tables:
            tables[key] = build_table(
                row, h=h, target_energy=target_energy, radius=radius, probability_rule=probability_rule,
                maximum_outcomes=maximum_outcomes,
            )
            counts["built"] += 1
        else:
            counts["reused"] += 1
        return tables[key]

    with patch.object(risk, "round_distribution", round_distribution):
        yield counts


def compute(job):
    # Scoped Q3 evaluator from 9aa6d186b452, scripts/h6_fixed_run/q3-pilot/run_q3.py.
    name, rows, shots, source, workers, h, target = job
    from trottertracks import rpe_deterministic_risk as D, rpe_total_risk as R
    from trottertracks.rpe_total_dp_parallel import installed

    start = time.monotonic()
    law = METHODS[name]
    h_exact = Fraction(h)
    threshold = Fraction(EPSILON) ** 2
    loss_cap, _ = R.loss_range(h_exact, target)
    entry = dict(
        prepared=dict(M=len(rows) - 1, h=D.a.exact_value(h_exact), decoder=D.S.DECODER,
                      input_complete=True, physical_qualification_complete=False,
                      rows=[dict(m=r["m"], mean=dict(raw=r["signal"]["raw"]),
                                 shots=dict(N_X=n, N_Y=n), law_binding=dict(definition=dict(model=law)))
                            for r, n in zip(rows, shots, strict=True)]),
        statistical_contract=dict(allow_conditional_unqualified=True,
            simulation_policy=dict(kind=D.S.IID, source=source, target_law=law),
            summary_options=dict(unresolved_loss=dict(kind="BOUNDED_INTERVAL", lower=0,
                                                       upper=D.a.exact_value(loss_cap), source=source))),
        evaluation_target=dict(energy_ha=target),
    )
    cells = [(n + 1) ** 2 for n in shots]
    frontier = 8192
    options = dict(estimator=R.T.ESTIMATOR, probability_rule=D.EXACT_AFFINE,
        phase_radii=[1.] * len(shots), maximum_round_outcomes=max(cells),
        maximum_verification_pairs=sum(a * b for a, b in zip(cells, cells[1:])),
        model_loss_contract=dict(adopted=True, source=source, scope="H6_FIXED_ALLOCATION_DIAGNOSTIC"),
        total_state_refinement=dict(source=source, maximum_frontier_states=frontier,
                                    maximum_transition_pairs=cells[0] + frontier * sum(cells[1:])),
        q3_limit=dict(MSE_upper=D.a.exact_value(threshold), scope="MODEL_MSE_FILTER_ONLY", source=source),
    )
    print(f"Q3_START {name}: {workers} children available", flush=True)
    helper = Path(__file__).resolve().parents[1] / 'src/trottertracks/rpe_q3_round_tables.py'
    spec = importlib.util.spec_from_file_location('_hchain_q3_round_tables', helper)
    acceleration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acceleration)
    with reuse_round_tables(D) as tables:
        with acceleration.parallel_good_transitions(D, workers) as enabled:
            print(f"Q3_PARTIAL_START parallel_closure={enabled}", flush=True)
            partial = D._bound(entry, options)
        print(f"Q3_PARTIAL_END elapsed={time.monotonic()-start:.1f}s", flush=True)
        with installed(workers=workers) as engine:
            total = R._total(entry, partial, options)
            timing = engine.observations
    lower = total.get("MSE_lower")
    upper = total.get("MSE_upper")
    if lower is None or upper is None:
        status = "UNKNOWN"
    elif D.a.exact_scalar(upper) <= threshold:
        status = "MODEL_BOUND_MEETS_ADOPTED_THRESHOLD"
    elif D.a.exact_scalar(lower) > threshold:
        status = "MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD"
    else:
        status = "INCONCLUSIVE"
    elapsed = time.monotonic() - start
    print(f"Q3_END {name}: {status} in {elapsed:.1f}s", flush=True)
    return dict(method=name, law=law, status=status, MSE_lower=lower, MSE_upper=upper,
                result=total, partial=partial, engine_observations=timing, round_tables=tables, wall_seconds=elapsed)


def outcome(request, proof):
    result = proof["result"]
    status, lower, upper = (result[key] for key in ("status", "MSE_lower", "MSE_upper"))
    if lower is None or upper is None:
        wanted = "UNKNOWN"
    else:
        lower_value, upper_value = exact_price(lower), exact_price(upper)
        threshold = Fraction(EPSILON) ** 2
        if not 0 <= lower_value <= upper_value:
            raise ValueError("Invalid scoped Q3 MSE bounds")
        if upper_value <= threshold:
            wanted = "MODEL_BOUND_MEETS_ADOPTED_THRESHOLD"
        elif lower_value > threshold:
            wanted = "MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD"
        else:
            wanted = "INCONCLUSIVE"
    if status != wanted:
        raise ValueError("Scoped Q3 verdict differs from the fixed MSE threshold")
    return dict(request_id=request["request_id"], route=KIND, identity=proof["identity"],
                status=status, MSE_lower=lower, MSE_upper=upper, scientific_evaluation_calls=1)


def run(request, output, mode):
    name, rows, shots, h, target, source, identity = request_inputs(request)
    proof_path = output / "HCHAIN_RESULT.json"
    outcome_path = output / "OUTCOME.json"
    if mode == "collect":
        if not proof_path.is_file():
            return 3
        proof = read(proof_path)
        if (proof["identity"] != identity or proof["source"] != source
                or proof["request_id"] != request["request_id"] or proof["result"]["method"] != name):
            raise ValueError("Collected Q3 result does not match its request")
        record = outcome(request, proof)
        if outcome_path.exists():
            if read(outcome_path) != record:
                raise ValueError("Collected Q3 outcome differs from its result")
        else:
            write(outcome_path, record)
        return 2 if record["status"] == "UNKNOWN" else 0

    view = Path(request["route"]["view"]).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=view, text=True).strip()
    if (head != RISK_COMMIT
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=view).returncode):
        raise ValueError("Scoped Q3 risk source differs from the fixed checkout")
    sys.path.insert(0, str(view / "src"))
    if mode == "dry-check":
        from trottertracks.rpe_total_dp_parallel import Accelerator

        Accelerator(workers=1)
        write(output / "DRY_CHECK.json", dict(status="DRY_CHECK_PASS", source=source, identity=identity,
                                             scientific_evaluation_calls=0))
        return 0
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    workers = request.get("execution_profile", {}).get("dp_workers")
    if workers is None:
        workers = 1
    if type(workers) is not int or workers < 1:
        raise ValueError("Scoped Q3 requires a positive DP worker count")
    receipt = dict(request_id=request["request_id"], source=source, identity=identity)
    write(output / "HCHAIN_STARTED.json", receipt)
    result = compute((name, rows, shots, source, min(workers, affinity), h, target))
    proof = dict(receipt, result=result)
    write(proof_path, proof, compact=True)
    record = outcome(request, proof)
    write(outcome_path, record)
    return 2 if record["status"] == "UNKNOWN" else 0


def main():
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("execute", "collect", "dry-check"), required=True)
    args = parser.parse_args()
    return run(read(args.request), args.output, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

