"""Static/saved-evidence audit and numeric-only artificial DF fixtures.

Never accepts a Hamiltonian, preparation archive, NPZ, or real diagonal vector.
Original function bodies are extracted from pinned Git blobs to avoid importing
the molecular/provider package initializers. Qiskit is not imported. Recording
appenders are mocks, not Qiskit circuits or new compiled measurements.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import operator
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "7e454b91185ed7b0f92a3790f800ba5d65d5f156"
SCIENCE_SOURCE = "536a3671c0ec2939ff3886e40467db26a27ec65a"
PROVIDER = "217c971dd537b1cc7fbc13742bc0f6c263ff7f45"
SAVED = "artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18"
PATHS = {
    "counter": (SCIENCE_SOURCE, "src/trotterlib/native_df_gate_model.py"),
    "ops": (PROVIDER, "src/trotterlib/df_trotter/ops.py"),
    "weights": (PROVIDER, "src/trotterlib/product_formula.py"),
    "schedule": (PROVIDER, "src/trottertracks/resource_applicability/ax2a_control_plan.py"),
    "native": (PROVIDER, "src/trottertracks/resource_applicability/ax2b_native_df_v5.py"),
    "integer": (PROVIDER, "src/trotterlib/rte.py"),
    "basis": (PROVIDER, "src/trotterlib/df_partial_s2.py"),
    "adapter": (SCIENCE_SOURCE, "scripts/h46_cost_execution.py"),
    "replay": (SCIENCE_SOURCE, "scripts/h46_same_event.py"),
}


def git_bytes(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_definitions(name, source, names=None, **dependencies):
    """Execute unchanged selected definitions, with explicit numeric/mock deps."""
    tree = ast.parse(source)
    if names is not None:
        selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in selected} != set(names):
            raise ValueError("PINNED_DEFINITION_MISSING")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    module = types.ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(dependencies)
    exec(compile(tree, name, "exec"), module.__dict__)
    return module


def functions():
    source = {key: git_bytes(*identity) for key, identity in PATHS.items()}
    m = load_definitions("_h46_s4_frozen_counter", source["counter"])
    # Numeric-only test path. This sentinel is never instantiated; symbolic
    # ParameterExpression behavior and Qiskit internals are outside this test.
    class SymbolicSentinel:
        pass
    ops = load_definitions("_h46_s4_frozen_ops", source["ops"], {
        "DiagonalEvolutionPrimitives", "_as_real", "_is_symbolic",
        "_is_effectively_zero", "_real_or_symbolic", "_append_rzz",
        "one_body_diagonal_primitives", "df_squared_diagonal_primitives",
        "append_diagonal_primitives",
    }, np=np, Any=Any, dataclass=dataclass, ParameterExpression=SymbolicSentinel)
    weights = load_definitions("_h46_s4_frozen_weights", source["weights"], {
        "yoshida_4th_list", "trotter_2nd_list", "_get_w_list"})
    integer = load_definitions("_h46_s4_integer", source["integer"], {"require_integer_count"}, np=np, operator=operator)
    schedule = load_definitions("_h46_s4_native_schedule", source["schedule"], {
        "ControlStage", "partial_s2_control_plan", "deterministic_control_plan",
    }, dataclass=dataclass, math=math, require_integer_count=integer.require_integer_count, _get_w_list=weights._get_w_list)
    native = load_definitions("_h46_s4_native_append", source["native"], {"append_directional_diagonal"})
    return types.SimpleNamespace(m=m, ops=ops, weights=weights, schedule=schedule, native=native, source=source)


class RecordingCircuit:
    """Minimal numeric uncontrolled/directional appender target; no simulation."""
    def __init__(self, m):
        self.m, self.word, self.global_phase = m, [], 0.0

    def rz(self, angle, qubit):
        self.word.append(self.m.Gate("rz", (qubit,), float(angle)))

    def rzz(self, angle, left, right):
        self.word.append(self.m.Gate("rzz", (left, right), float(angle)))

    def cx(self, left, right):
        self.word.append(self.m.Gate("cx", (left, right)))

    def circuit(self):
        return self.m.Circuit(tuple(self.word), self.global_phase)


def append_native(f, primitives, mode, ancilla):
    record = RecordingCircuit(f.m)
    if mode == "DIRECTIONAL":
        f.native.append_directional_diagonal(record, primitives, ancilla)
    elif mode == "UNCONTROLLED":
        f.ops.append_diagonal_primitives(record, primitives)
    else:
        raise ValueError("Fixture supports only the registered directional policy")
    return record.circuit()


def artificial_pair(f, eta, *, q=1, formula="4th", align_time=False):
    """One identity-basis artificial block, lam=1, T=.8. No real-system data."""
    eta = tuple(eta)
    m = f.m
    model_schedule = m.pf_schedule(1, .8, q, formula, "symmetric_directional")
    native_schedule = f.schedule.deterministic_control_plan(1, .8, q, formula)
    totals = {key: m.Cost() for key in ("model", "native")}
    phase_counts = Counter()
    stages = []
    for index, (model, native) in enumerate(zip(model_schedule, native_schedule, strict=True)):
        model_time = native.time if align_time else model.time
        pp = f.ops.df_squared_diagonal_primitives(np.asarray(eta), 1., model_time)
        pn = f.ops.df_squared_diagonal_primitives(np.asarray(eta), 1., native.time)
        mc = m.diagonal_from_primitives(pp, model.mode, len(eta))
        nc = append_native(f, pn, native.mode, len(eta))
        totals["model"] += m.count(mc)
        totals["native"] += m.count(nc)
        phase_counts["model"] += int(pp.global_phase != 0.)
        phase_counts["native"] += int(pn.global_phase != 0.)
        stages.append(dict(index=index, mode=model.mode, model_time_hex=float(model_time).hex(),
                           native_time_hex=float(native.time).hex(),
                           model_rz_indices=[v[0] for v in pp.rz], native_rz_indices=[v[0] for v in pn.rz],
                           model_rz_angles_hex=[float(v[1]).hex() for v in pp.rz],
                           native_rz_angles_hex=[float(v[1]).hex() for v in pn.rz],
                           model_phase_hex=float(pp.global_phase).hex(), native_phase_hex=float(pn.global_phase).hex()))
    delta = {k: totals["native"].as_dict()[k] - totals["model"].as_dict()[k] for k in totals["native"].as_dict()}
    return dict(eta_hex=[float(x).hex() for x in eta], q=q, formula=formula,
                counts={k:v.as_dict() for k,v in totals.items()}, delta_native_minus_model=delta,
                phase_nonzero_counts=dict(phase_counts), stages=stages)


def saved_evidence():
    read = lambda name: json.loads(git_bytes(BASE, f"{SAVED}/{name}"))
    rows = read("same_event_comparisons.json")
    provenance = read("row_provenance_index.json")
    bindings = [r for r in provenance["row_bindings"] if r["file"]=="same_event_comparisons.json"]
    if len(bindings)!=len(rows) or {b['index'] for b in bindings}!=set(range(len(rows))):
        raise ValueError("SAVED_PROVENANCE_INVENTORY")
    for binding in bindings:
        row = rows[binding['index']]
        actual = digest(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        if actual!=binding['row_digest'] or binding['arithmetic_SOURCE']!=SCIENCE_SOURCE:
            raise ValueError("SAVED_ROW_PROVENANCE_BINDING")
    mixed = [r for r in rows if r["model"] == "NATIVE_MIXED"]
    differences, matches = [], []
    for r in mixed:
        value = dict(system=r["system"], candidate=r["candidate_id"], axis=r["axis"], metric=r["metric"],
                     theory=int(r["theory_same_event"]), native=int(r["native_explicit_lowered"]),
                     compiled=r["compiled_same_event"], fingerprint=r["fingerprint"],
                     compiler_hash=r["compiler_hash"], preparation_hash=r["preparation_hash"])
        value["native_minus_theory"] = value["native"] - value["theory"]
        (differences if value["native_minus_theory"] else matches).append(value)
    if len(differences) != 8 or len(matches) != 20:
        raise ValueError("FIXED_SAVED_ROW_INVENTORY_CHANGED")
    signatures = []
    for system, cid in (("H4", "B1_p12_q1_4th"), ("H6", "B1_p19_q8_4th")):
        prefix = f"cohorts/{system}_{cid}/"
        theory = read(prefix + "paired_theory.json")
        for axis in ("cosine", "sine"):
            old = read(prefix + f"wrapper_{axis}.json")
            model = next(r["cost"] for r in theory if r["model"] == "NATIVE_MIXED" and r["axis"] == axis)
            measured = old["raw_native"]["cost"]
            delta = {k:int(measured[k])-int(model[k]) for k in model}
            inference = infer_diagonal_cardinality(delta)
            signatures.append(dict(system=system, candidate=cid, axis=axis, delta_native_minus_theory=delta,
                                   inferred_net_cardinality_change=inference,
                                   native_histogram=old["raw_native"]["histogram"],
                                   saved_preparation_hash=old["preparation_hash"],
                                   zero_unknown_primitives=not old["raw_native"]["unknown_primitives"]))
    return dict(mismatches=differences, controls=matches, saved_cost_signatures=signatures,
                verified_saved_row_bindings=len(bindings), candidate_receipts=provenance['candidate_receipts'],
                original_audit=read("model_native_difference_audit.json"))


def infer_diagonal_cardinality(delta):
    """Interpret saved differences, not a new real-system theoretical recount.

    Preconditions established by static source: identical A basis/wrapper and
    scalar emission, paired equal-time uncontrolled/directional diagonals. Let
    da/db/dg be net changes over all pairs in retained RZ, nonzero RZZ, nonzero
    phase respectively. Then dR=2da+2db+dg, dC=2da+6db,
    dRaw=4da+6db+dg. No assertion about individual real block/qubit sites.
    """
    from fractions import Fraction as F
    dr, dc, draw = (F(delta[k]) for k in ("rz", "cx", "raw"))
    db = (dc-draw+dr)/2
    dg = dr-dc+4*db
    da = (dc-6*db)/2
    if delta["sx"] != 0 or delta["x"] != 0 or any(v.denominator!=1 for v in (da,db,dg)):
        raise ValueError("PAIRED_DIAGONAL_INFERENCE_PRECONDITION")
    return dict(retained_rz=int(da), nonzero_rzz=int(db), nonzero_phase=int(dg),
                scope="NET_OVER_ALL_EQUAL_TIME_FORWARD_REVERSE_PAIRS; individual sites not observed")


def source_receipts(f):
    receipts = {}
    closure = json.loads(git_bytes(BASE, f"{SAVED}/source_closure.json"))
    scientific = next(s['source_hashes'] for s in closure['stages'] if s['stage']==4)
    for key, (commit, path) in PATHS.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{commit}:{path}"], text=True).strip()
        parsed = ast.parse(f.source[key])
        defs = {n.name: dict(start=n.lineno, end=n.end_lineno) for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        actual = digest(f.source[key])
        expected = (scientific if commit==SCIENCE_SOURCE else closure['provider_hashes'])[path]
        if actual != expected:
            raise ValueError("PINNED_SOURCE_DIFFERS_FROM_PRIOR_CLOSURE")
        receipts[key] = dict(commit=commit, path=path, blob=blob, bytes=len(f.source[key]), sha256=actual,
                             matches_prior_closure=True, definitions=defs)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path.home()) or args.output.exists():
        raise ValueError("NEW_EXCLUSIVE_HOME_OUTPUT_REQUIRED")
    f = functions()
    # Fixtures are explicit artificial zero-sum diagonals, selected without any
    # molecular/preparation data. Both directions of the discrepancy are tested.
    fixtures = [artificial_pair(f, eta, q=q) for eta in ((1., 2., -3.), (1., 5., -6.)) for q in (1,8)]
    result = dict(kind="STATIC_AND_ARTIFICIAL_ONLY", base=BASE, scientific_SOURCE=SCIENCE_SOURCE,
                  provider=PROVIDER, saved=saved_evidence(), source_receipts=source_receipts(f),
                  fixtures=fixtures, numpy_version=importlib.metadata.version("numpy"),
                  aligned_time_fixtures=[artificial_pair(f, eta, q=q, align_time=True)
                      for eta in ((1.,2.,-3.), (1.,5.,-6.)) for q in (1,8)],
                  python=sys.version, actual_scientific_calls=dict(circuit_build=0, compile=0, theoretical_recount=0,
                  numerical_reanalysis=0, sampling=0), original_scientific_implementations_modified=False,
                  qiskit_imported=any(k=="qiskit" or k.startswith("qiskit.") for k in sys.modules))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
    print(json.dumps(dict(output=str(args.output), rows=len(result["saved"]["mismatches"]), new_scientific_calls=0)))


if __name__ == "__main__":
    main()
