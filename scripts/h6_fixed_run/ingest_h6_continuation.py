"""Import completed fixed H6 adapter outcomes using the existing optimizer."""

import argparse
from fractions import Fraction
from pathlib import Path
import sys

from register_full_m7_prices import digest, read


METHODS = {"FULL_PF": "Full PF", "TRUNCATED_PF": "B0I", "PR_DF": "PR-DF"}


def ingest(root, problem, prices, run, method=None, *, price_path=None):
    sys.path.insert(0, str(root / "optimizer-source/src"))
    from trottertracks import rpe_method_optimizer as optimizer

    modes = read(run / "PROCESS_RETURN.json")
    threshold = Fraction(problem["accuracy"]["epsilon_E_decimal"]) ** 2
    reports = {}
    selected = METHODS if method is None else {method: METHODS[method]}
    for method, label in selected.items():
        job = run / method
        request = read(job / "REQUEST.json")
        price = prices["methods"][label]
        cid = price["configuration_id"]
        if request["method"] != method or request["configuration_id"] != cid:
            raise ValueError("Q3 request differs from the priced configuration")
        expected_sha = request["route"].get("registered_prices_sha256")
        if expected_sha is not None and price_path is not None and digest(price_path) != expected_sha:
            raise ValueError("Q3 result is bound to a different price registry")
        vector = optimizer.interleave(request["N_X"], request["N_Y"])
        units = [optimizer.frac(row["unit_price_RZ"]) for row in price["axes"]]
        cost = optimizer.cost(vector, units)
        if cost != optimizer.frac(request["cost"]):
            raise ValueError("Q3 request used a different price vector")
        if modes[method] != 0:
            reports[method] = dict(status="EXECUTION_HOLD", returncode=modes[method], job=str(job))
            continue
        outcome = read(job / "OUTCOME.json")
        if (outcome["request_id"] != request["request_id"] or outcome["route"] != request["route"]["kind"]
                or outcome["scientific_evaluation_calls"] != 1):
            raise ValueError("Q3 outcome differs from the queued request")
        executor = optimizer.Executor(
            root, run, {method: [request["route"]]}, {cid: dict(M=12)}, problem, launch_disabled=True,
        ).for_method(method)
        record = executor.ingest(job, {"configuration_id": cid, "vector": vector}, 1)
        if record["status"] not in (optimizer.PASS, optimizer.FAIL, "INCONCLUSIVE"):
            raise ValueError(f"Optimizer rejected the Q3 outcome for {method}: {record}")
        expected = (optimizer.PASS if record["MSE_upper"] <= threshold else
                    optimizer.FAIL if record["MSE_lower"] > threshold else "INCONCLUSIVE")
        if record["status"] != expected:
            raise ValueError("Q3 status differs from the fixed MSE threshold")
        candidate = dict(method=method, configuration_id=cid, ready=True, prices=units)
        _, best = optimizer.method_state([candidate], {(cid, vector): record}, threshold)
        record.update(configuration_id=cid, N_X=request["N_X"], N_Y=request["N_Y"], scenario_cost_RZ=cost,
                      price_kinds=sorted({row["kind"] for row in price["axes"]}),
                      incumbent=best is not None)
        reports[method] = optimizer.jsonable(record)
    return dict(scope=prices["scope"], run=str(run), methods=reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "problem", "prices", "run", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "q3-pilot"))
    from run_q3 import write

    report = ingest(args.root.resolve(), read(args.problem), read(args.prices), args.run.resolve(),
                    args.method, price_path=args.prices)
    if args.output.exists():
        previous = read(args.output)
        if previous == report:
            raise SystemExit(0)
        if (previous["scope"] != report["scope"] or previous["run"] != report["run"]
                or previous["methods"].keys() != report["methods"].keys()
                or any(row != report["methods"][name] and row["status"] != "EXECUTION_HOLD"
                       for name, row in previous["methods"].items())):
            raise ValueError("Q3 adoption differs from the saved report")
        write(args.output, report, replace=True)
    else:
        write(args.output, report)
