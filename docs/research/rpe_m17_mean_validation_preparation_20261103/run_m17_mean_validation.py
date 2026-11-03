"""Fixed-cell m17 preparation. Default CLI reads saved literals only.

--run selects new scalar reanalysis and H4 numerical work, and requires separate
user permission. No sampler, virtual shot, eigensolve or compiler is invoked.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V2 = "78c49d0b5f1d71e63b5959e78813237923bbc648"
PRIOR_SOURCE = "d0b1da1b04c266a1f7abf8e43966ca6e289a8695"
ROW_PATH = "docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv"
PRIOR_DIR = ROOT / "docs/research/rpe_fresh_iid_measurement_a_preparation_20261103"
Q, R, K, LD, DELTA = 131072, 32, 2, 3, .02
ATOL = 1e-10  # Numerical reproduction diagnostic, not a proved error bound.
DECIMAL_PRECISION = 80


def fixed_bytes(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"])


@lru_cache(maxsize=1)
def prior_runner():
    sys.path.insert(0, str(PRIOR_DIR))
    module = importlib.import_module("run_fresh_iid_a_v2")
    if Path(module.__file__).resolve() != PRIOR_DIR / "run_fresh_iid_a_v2.py":
        raise ValueError("Wrong prior helper module")
    for name in ("run_fresh_iid_a_v2.py", "fresh_iid_adapter.py"):
        path = (PRIOR_DIR / name).relative_to(ROOT).as_posix()
        if (ROOT / path).read_bytes() != fixed_bytes(PRIOR_SOURCE, path):
            raise ValueError("Prior helper SOURCE changed: " + path)
    return module


def fixed_literals():
    """Literal transfer only: no statistical arithmetic or H/state decode."""
    rows = csv.DictReader(io.StringIO(fixed_bytes(V2, ROW_PATH).decode()))
    selected = [r for r in rows if r["ld"] == "3" and r["round_index"] == "17"]
    if len(selected) != 1:
        raise ValueError("Selected V2 row missing/duplicated")
    row = selected[0]
    expected = (str(Q), str(R), str(K), str(DELTA), "attenuated_event_mean_signal")
    if tuple(row[k] for k in ("q_m", "r_m", "K_m", "delta_time", "saved_signal_kind")) != expected:
        raise ValueError("Fixed m17 tuple/measurement kind mismatch")
    axes = json.loads(row["axes_results_json"])
    result = {"ld": row["ld"], "m": row["round_index"], "q": row["q_m"],
              "r": row["r_m"], "K": row["K_m"], "delta": row["delta_time"],
              "saved_mean": json.loads(row["saved_complex_json"]),
              "saved_betas": [row[k] for k in ("saved_beta_PF", "saved_beta_RTE", "saved_beta_stat")],
              "mean_source": {"commit": V2, "path": ROW_PATH, "selector": "ld3/round_index17"},
              "axes": {a: {"N": axes[a]["saved_N"], "alpha": axes[a]["saved_alpha"],
                           "epsilon": axes[a]["saved_epsilon_coordinate"]} for a in ("cosine", "sine")}}
    for axis in result["axes"].values():
        if (axis["N"], axis["alpha"], axis["epsilon"]) != (
                "411", "0.016957089147912167", "0.15244325851620388"):
            raise ValueError("Fixed saved N/alpha/epsilon mismatch")
    return result


def describe():
    return {"status": "PREPARED_NOT_SCIENTIFICALLY_EXECUTED", "fixed_literals": fixed_literals(),
            "planned": {"scalar_rows": 1, "axes": 2, "DF_preparations": 1,
                        "complex_mean_paths": 2, "sampling": 0, "virtual_shots": 0,
                        "ground_state_solves": 0, "controlled_builds": 0, "compiles": 0, "costs": 0},
            "diagnostic_atol": ATOL, "u_num_bound": None,
            "decimal_scalar_precision": DECIMAL_PRECISION, "mandatory_STOP": True}


def statistical_reserve(axes):
    """New scientific scalar evaluation on real inputs; synthetic tests only now."""
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        out = {}
        for name in ("cosine", "sine"):
            raw = axes[name]
            n, alpha, epsilon = (Decimal(str(raw[k])) for k in ("N", "alpha", "epsilon"))
            if not all(x.is_finite() for x in (n, alpha, epsilon)) or n <= 0 or n != n.to_integral_value() or not 0 < alpha < 1 or epsilon <= 0:
                raise ValueError("Invalid N/alpha/epsilon")
            half_width = (Decimal(2) * (Decimal(2)/alpha).ln() / n).sqrt()
            out[name] = {"saved": dict(raw), "hoeffding_half_width": str(half_width),
                         "reserve": str(epsilon-half_width)}
        return {"axes": out, "minimum_reserve": str(min(Decimal(v["reserve"]) for v in out.values())),
                "scalar_precision": DECIMAL_PRECISION,
                "rounding_note": "High-precision scalar evaluation, not interval arithmetic"}


def grouped_averages(probability_components, coefficient_components, lambda_r, ordered, basis_lookup):
    """Two independent weight sources, one cached basis transform per group."""
    ordered = np.asarray(ordered, dtype=np.int64)
    total = math.fsum(c.probability for c in probability_components)
    if not math.isfinite(total) or total <= 0 or not math.isfinite(lambda_r) or lambda_r <= 0:
        raise ValueError("Invalid probability/coefficient normalization")
    if {c.component_id for c in probability_components} != {c.component_id for c in coefficient_components}:
        raise ValueError("Retained component identities disagree")
    groups = ({}, {})
    for components, group, probability_mode in (
            (probability_components, groups[0], True), (coefficient_components, groups[1], False)):
        for c in components:
            parity = np.zeros(len(ordered), dtype=np.int64)
            for q in c.diagonal_pauli_support:
                parity ^= (ordered >> q) & 1
            weight = c.probability/total*c.coefficient_sign if probability_mode else c.coefficient/lambda_r
            if not math.isfinite(weight):
                raise ValueError("Nonfinite component weight")
            group.setdefault(c.basis_id, np.zeros(len(ordered)))[:] += weight * (1.-2.*parity)
    matrices = [np.zeros((len(ordered), len(ordered)), dtype=complex) for _ in groups]
    for basis_id in sorted(set(groups[0]) | set(groups[1])):
        basis = basis_lookup(basis_id)
        for group, matrix in zip(groups, matrices):
            if basis_id in group:
                matrix += (basis * group[basis_id][None, :]) @ basis.conj().T
    return *matrices, total


def literal_law_mean(a_probability, distribution):
    """Full nominal categorical law, without calling the Taylor reference."""
    a_probability = np.asarray(a_probability, dtype=complex)
    identity = np.eye(a_probability.shape[0], dtype=complex)
    mean = np.zeros_like(a_probability)
    for order, probability in zip(distribution.orders, distribution.order_probabilities):
        angle = math.atan(distribution.dimensionless_step_time/(order+1))
        mean += probability * (-1)**(order//2) * (
            math.cos(angle)*identity - 1j*math.sin(angle)*a_probability
        ) @ np.linalg.matrix_power(a_probability, order)
    if not np.isfinite(mean).all():
        raise ValueError("Nonfinite literal-law mean")
    return mean


def load_numerical_api():
    """Only reached by --run or synthetic dependency injection."""
    sys.path.insert(0, str(ROOT / "src"))
    from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot
    from trotterlib.df_hamiltonian import PhysicalSector
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    from trotterlib.research_direction_full_scope import _finite_inputs
    from trotterlib.finite_rte_signal_validation import _qiskit_to_openfermion_sector_permutation, _array_fingerprint
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
    from trotterlib.rte import finite_rte_operator_moments, _validate_components
    return locals()


def prepare_context(provenance, *, api=None):
    p = prior_runner()
    provenance["fixed_inputs"] = p.source_and_input_identity()
    provenance["versions"] = {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "qiskit")}
    if provenance["versions"]["qiskit"] != "1.3.0":
        raise ValueError("Qiskit 1.3.0 required for fixed basis definitions")
    api = load_numerical_api() if api is None else api
    h = api["load_connected_cluster_hamiltonian_snapshot"](ROOT/p.H_PATH)
    if api["df_hamiltonian_hash"](h) != p.H_SHA or h.n_qubits != 8 or h.n_blocks != 12:
        raise ValueError("Fixed H identity mismatch")
    sector = api["PhysicalSector"].number_sector(n_qubits=8, n_electrons=4)
    with np.load(ROOT/p.STATE_PATH, allow_pickle=False) as saved:
        psi = np.asarray(saved["state"], dtype=complex)
        if not np.array_equal(saved["basis_indices"], sector.basis_indices) or str(saved["hamiltonian_hash"]) != p.H_SHA:
            raise ValueError("Saved state sector/H mismatch")
    fingerprint = api["_array_fingerprint"](psi)
    provenance["state_fingerprint"] = fingerprint
    if fingerprint != p.STATE_FP or abs(np.linalg.norm(psi)-1) > ATOL:
        raise ValueError("Saved physical state mismatch")
    preparation = api["prepare_df_partial_s2"](h, api["split_df_hamiltonian_by_ld"](h, LD),
                    identity_policy="extract_identity_phase", coefficient_atol=1e-12)
    components = preparation.rte_preparation.symbolic_tail.components
    provenance["preparation"] = {"hash": preparation.preparation_hash, "component_count": len(components),
                                 "deterministic_blocks": len(preparation.deterministic_blocks), "coefficient_atol": 1e-12}
    if preparation.preparation_hash != p.PREPARATION_SHA or len(components) != 218 or len(preparation.deterministic_blocks) > 4:
        raise ValueError("Preparation inventory mismatch: " + json.dumps(provenance["preparation"]))
    api["_validate_components"](components)  # Same validator used before RNG choice.
    config, distribution = api["_finite_inputs"](preparation, delta_time=DELTA, rte_steps=R, finite_taylor_order=K)
    provenance["config"] = asdict(config)
    provenance["distribution"] = distribution.to_dict()
    permutation = api["_qiskit_to_openfermion_sector_permutation"](sector)
    ordered = sector.basis_indices[permutation]
    cache, leakages = {}, []

    def basis_fn(basis_id, basis_hash, operations):
        key = basis_id, basis_hash
        if key not in cache:
            cache[key], leakage = p.restricted_basis(operations, 8, sector.basis_indices, permutation)
            leakages.append(leakage)
        return cache[key]

    def tail_basis(basis_id):
        definition = preparation.tail_extraction.basis_definition(basis_id)
        return basis_fn(basis_id, definition.metadata.basis_hash, definition.runtime_operations)

    def primitive_fn(block):
        if block.block_kind == "one_body":
            return api["one_body_diagonal_primitives"](np.asarray(block.diagonal_eigenvalues), DELTA/2)
        return api["df_squared_diagonal_primitives"](np.asarray(block.diagonal_eta), block.lam, DELTA/2)

    forward, reverse = p.numeric_half_operators(preparation, basis_fn, primitive_fn, ordered)
    a_probability, a_coefficient, total = grouped_averages(components, preparation.tail_extraction.components,
                                     preparation.exact_rte_lambda_r, ordered, tail_basis)
    reference = api["finite_rte_operator_moments"](a_coefficient, config).attenuated_event_mean_operator
    provenance.update({"component_probability_sum": total, "basis_cache_entries": len(cache),
                       "maximum_basis_sector_leakage": max(leakages, default=0.),
                       "tail_hash": preparation.tail_extraction.tail_hash})
    return {"psi": psi, "forward": forward, "reverse": reverse, "A_probability": a_probability,
            "A_coefficient": a_coefficient, "distribution": distribution, "reference_tail": reference}


def residual(left, right):
    difference = complex(left)-complex(right)
    return {"complex_abs": float(abs(difference)), "cosine_abs": float(abs(difference.real)),
            "sine_abs": float(abs(difference.imag))}


def evaluate_context(context, saved_mean, *, q=Q, r=R):
    law_event = literal_law_mean(context["A_probability"], context["distribution"])
    law_tail = np.linalg.matrix_power(law_event, r)
    means = {}
    for name, tail in (("law", law_tail), ("reference", context["reference_tail"])):
        step = context["reverse"] @ tail @ context["forward"]
        z = complex(np.vdot(context["psi"], np.linalg.matrix_power(step, q) @ context["psi"]))
        if not np.isfinite(z) or abs(z) > 1+ATOL:
            raise ValueError("Invalid final complex mean: " + repr(z))
        means[name] = {"real": z.real, "imag": z.imag}
    z_law, z_ref = (complex(means[k]["real"], means[k]["imag"]) for k in ("law", "reference"))
    comparisons = {"law_vs_reference": residual(z_law, z_ref),
                   "law_vs_saved": residual(z_law, saved_mean), "reference_vs_saved": residual(z_ref, saved_mean)}
    consistent = all(v["complex_abs"] <= ATOL for v in comparisons.values())
    result = {"means": means, "comparisons": comparisons,
              "numeric_diagnostic_status": "NUMERICALLY_CONSISTENT" if consistent else "INCONSISTENT",
              "diagnostic_atol": ATOL, "scientific_status": "UNDETERMINED",
              "operator_diagnostics_non_gating": {
                  "A_probability_vs_coefficient_frobenius": float(np.linalg.norm(context["A_probability"]-context["A_coefficient"])),
                  "tail_law_vs_reference_frobenius": float(np.linalg.norm(law_tail-context["reference_tail"]))}}
    arrays = {name: context[name] for name in ("A_probability", "A_coefficient", "forward", "reverse", "reference_tail")}
    arrays.update({"law_event_mean": law_event, "law_tail_mean": law_tail})
    return result, arrays


def reserve_interpretation(reserve, comparisons):
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        axes = {}
        for axis in ("cosine", "sine"):
            remaining = Decimal(reserve["axes"][axis]["reserve"])
            observed = comparisons["law_vs_saved"][axis+"_abs"]
            axes[axis] = {"observed_float_residual": observed,
                          "headroom_for_total_numerical_uncertainty": str(remaining-Decimal.from_float(observed))}
    positive = Decimal(reserve["minimum_reserve"]) > 0
    return {"axes": axes, "reserve_positive": positive, "u_num_bound": None, "RPE_tolerance_certified": False,
            "status": "UNDETERMINED_NUMERICAL_ERROR_BOUND_NOT_ESTABLISHED" if positive else "UNDETERMINED_NONPOSITIVE_RESERVE",
            "note": "Positive observed headroom is not a bound on true mean error. 1e-10 is diagnostic only."}


def source_metadata():
    path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    code = Path(__file__).read_bytes()
    if code != fixed_bytes("HEAD", path):
        raise ValueError("Commit this numerical SOURCE version before --run")
    return {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip(),
            "runner_sha256": hashlib.sha256(code).hexdigest()}


def write_result(output, record):
    temporary = output/".RESULT.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(output/"RESULT.json")


def execute(output):
    source = source_metadata()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"execution_status": "STARTED", "SOURCE": source, "mandatory_STOP": True,
              "stage": "saved_literals", "scientific_status": "UNDETERMINED",
              "forbidden_calls_by_construction": {"sampling": 0, "virtual_shots": 0, "ground_state_solves": 0,
                                                  "controlled_builds": 0, "compiles": 0, "cost_evaluations": 0},
              "resources": {"available_CPUs": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                            "BLAS_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}}
    try:
        write_result(output, record)
        literals = fixed_literals()
        record["fixed_literals"] = literals
        record["stage"] = "scalar_reserve"
        reserve = statistical_reserve(literals["axes"])
        record["statistical_reserve"] = reserve
        record["stage"] = "numerical_context"
        write_result(output, record)
        record["context"] = {}
        context = prepare_context(record["context"])
        record["stage"] = "mean_evaluation"
        write_result(output, record)
        target = complex(float(literals["saved_mean"]["real"]), float(literals["saved_mean"]["imag"]))
        evaluation, arrays = evaluate_context(context, target)
        record["evaluation"] = evaluation
        record["reserve_interpretation"] = reserve_interpretation(reserve, evaluation["comparisons"])
        np.savez_compressed(output/"operators.npz", **arrays)
        record["operators_npz_sha256"] = hashlib.sha256((output/"operators.npz").read_bytes()).hexdigest()
        record.update({"execution_status": "COMPLETED_MANDATORY_STOP", "stage": "completed",
                       "completed_scalar_rows": 1, "completed_DF_preparations": 1, "completed_complex_mean_paths": 2})
    except BaseException as exc:
        record["execution_status"] = "INTERRUPTED_OR_INVALID_MANDATORY_STOP"
        record["error"] = {"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), "<repository>")}
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-started
        write_result(output, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true", help="New scientific execution, only after separate user permission")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--run needs a fresh --output-dir")
    execute(args.output_dir)


if __name__ == "__main__":
    main()
