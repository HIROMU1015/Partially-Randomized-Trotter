#!/usr/bin/env python3
"""Audit saved exploration evidence using stdlib only; never import science code."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/representation_exploration/2026-10-10/run2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matrix(value: dict) -> list[list[complex]]:
    return [[complex(x, y) for x, y in zip(rr, ii)]
            for rr, ii in zip(value["real"], value["imag"])]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def scaled(a, c):
    return [[c * x for x in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def distance(a, b):
    return math.sqrt(sum(abs(x-y)**2 for ar, br in zip(a, b) for x, y in zip(ar, br)))


def audit(run: Path = RUN, root: Path = ROOT) -> dict:
    checks = 0

    def require(value, label):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError(label)

    def close(a, b, label, atol=2e-11):
        require(math.isclose(a, b, rel_tol=2e-11, abs_tol=atol), label)

    raw_audit = (run / "run_audit.json").read_bytes()
    receipt = json.loads(raw_audit)
    result = json.loads((run / "result.json").read_bytes())
    events = json.loads((run / "b_events.json").read_bytes())
    source = receipt["source_commit"]
    require(result["source_commit"] == source, "result/source binding")
    require(source == "25d7135f7a0285b6cf415349191b00c00acfb75f", "frozen run2 source")
    for path, expected in receipt["source_sha256"].items():
        blob = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=root)
        require(sha(blob) == expected, "source blob: " + path)
    for path, expected in receipt["outputs"].items():
        data = (run / path).read_bytes()
        require(sha(data) == expected["sha256"], "output hash: " + path)
        require(len(data) == expected["bytes"], "output size: " + path)
        require(len(data) <= receipt["caps"]["output_file_bytes"], "output cap: " + path)
    require(result["status"] == receipt["status"] ==
            "INITIAL_MECHANISMS_COMPLETE_AWAITING_GPT_REVIEW", "terminal status")
    for d in (receipt, result):
        require(d["next_stage_authorized"] is False, "next stage unauthorized")
        require(d["central_hypothesis_adopted"] is None, "no hypothesis adoption")
    require(all(x == "1" for x in receipt["threads"].values()), "single thread env")
    for field, cap in (("wall_seconds", "wall_seconds"), ("cpu_seconds", "cpu_seconds"),
                       ("peak_rss_bytes", "address_space_bytes")):
        require(0 <= receipt[field] < receipt["caps"][cap], "resource " + field)

    a, b, c = (result[k] for k in ("candidate_a", "candidate_b", "candidate_c"))
    require([len(x["rows"]) for x in (a,b,c)] == [20,9,3], "complete fixed rows")
    require(len({(x["case"], x["theta"]) for x in a["rows"]}) == 20, "unique A rows")
    costs = [x["controlled_s2_wrapper_cost"] for x in a["rows"]
             if x.get("controlled_s2_wrapper_cost")]
    costs += [x[k] for x in b["reflection_cost_rows"]
              for k in ("reflection", "controlled_base", "uncontrolled_reflection_sandwich")]
    costs += [x[k] for x in c["rows"] for k in ("thrift_wrapper_cost", "ordinary_wrapper_cost")
              if x.get(k)]
    require(len(costs) == result["resource_counts"]["compiled_circuits"] == 19,
            "compiled circuit accounting")
    for cost in costs:
        require(cost["qubits"] <= receipt["caps"]["total_circuit_qubits"], "qubit cap")
        require(cost["size"] == sum(cost[k] for k in ("rz", "cx", "sx", "x")), "native size")
        require(cost["compiled_operator_residual"] < 2e-10, "absolute compiled equality")
        require(cost["control_sensitive_repaired_residual"] < 2e-10, "additional control equality")
        if cost["global_phase_repair_radians"]:
            require(cost["certified_scalar_residual"] < 2e-10, "phase repair certificate")
    for field in ("molecular_loads", "ground_state_solves", "quantum_shots", "gpu_calls"):
        require(result["resource_counts"][field] == 0, "excluded calls " + field)

    groups = defaultdict(list)
    for event in events:
        groups[(event["t"], event["k"])].append(event)
        require(0 <= event["probability"] <= 1, "event probability")
        require(-2e-11 <= event["leakage_probability_psi0"] <= 1+2e-11, "event leakage")
        close(event["phase_real"]**2+event["phase_imag"]**2, 1, "unit event phase")
    require(len(events) == result["resource_counts"]["enumerated_b_events"] == 24678,
            "enumerated event accounting")
    require(set(groups) == {(x["t"], x["k"]) for x in b["rows"]}, "B event coverage")
    p, s, h, hb = (matrix(b["inputs"][k]) for k in ("p", "s", "h_tilde", "h_bar"))
    require(distance(p, multiply(p,p)) < 2e-11, "P projector")
    ident = [[complex(i == j) for j in range(4)] for i in range(4)]
    require(distance(s, add(scaled(p,2), scaled(ident,-1))) < 2e-11, "S definition")
    require(distance(hb, scaled(add(h, multiply(multiply(s,h),s)),.5)) < 2e-11,
            "saved symmetric generator identity")
    max_polynomial_residual = 0.0
    for row in b["rows"]:
        group = groups[(row["t"], row["k"])]
        require(len(group) == row["event_count"] <= receipt["caps"]["b_events_per_row"],
                "row event cap/count")
        require({x["index"] for x in group} == set(range(len(group))), "event indices")
        close(math.fsum(x["probability"] for x in group), row["probability_sum"], "probability sum")
        close(row["probability_sum"], 1, "normalized event dictionary")
        close(math.fsum(x["probability"]*x["leakage_probability_psi0"] for x in group),
              row["channel_leakage_probability"], "channel leakage from primary events")
        close(max(x["leakage_probability_psi0"] for x in group),
              row["individual_max_leakage_probability"], "individual leakage")
        close(math.fsum(x["probability"]*x["unfused_reflection_count"] for x in group),
              row["expected_unfused_reflections"], "reflection expectation")
        close(row["normalization"]**2, row["b_squared_variance_envelope"], "B squared")
        corr, mean = matrix(row["corrected_operator"]), matrix(row["mean_operator"])
        require(distance(corr, scaled(mean,row["normalization"])) < 2e-11, "corrected mean normalization")
        power, polynomial = ident, ident
        for degree in range(1, row["k"]+2):
            power = multiply(power, scaled(hb,-1j*row["t"]))
            polynomial = add(polynomial, scaled(power,1/math.factorial(degree)))
        residual = distance(corr, polynomial)
        max_polynomial_residual = max(max_polynomial_residual,residual)
        require(residual < 2e-11, "independent saved Taylor polynomial")
        leakage = math.sqrt(sum(abs(corr[i][j])**2 for i in (2,3) for j in (0,1)))
        require(leakage < 2e-11, "saved corrected mean block preservation")
    return {"schema_version":1, "status":"PASS", "checks":checks,
            "source_commit":source, "run_audit_sha256":sha(raw_audit),
            "verifier_sha256":sha(Path(__file__).read_bytes()),
            "primary_event_count":len(events), "compiled_circuits":len(costs),
            "independent_saved_taylor_frobenius_residual_max":max_polynomial_residual,
            "scope":"stdlib saved hashes, commit blobs, events and small saved-matrix arithmetic; no science rerun",
            "new_science_runs":0, "immutable_ci":False, "external_replication":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
