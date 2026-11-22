"""Verify completed H6 compiled receipts and preserve distinct price cohorts."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


AXES = {"cosine": "X", "sine": "Y"}
COMPILER = "1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def verify(root, signals):
    start = read(root / "INPUT.json")
    if start["compiler"] != COMPILER or start["signal_packet_sha256"] != sha(signals):
        raise ValueError("Price and supplied signal source differ")
    signal_packet = read(signals)
    verified, cohorts, fingerprints, seeds = [], defaultdict(list), set(), set()
    for result_path in sorted(root.glob("*-m*-*/RESULT.json")):
        row = read(result_path)
        leaf = result_path.parent
        method, m = row["method"], row["m"]
        label = {"B0I": "B0I", "PR": "PR-DF"}[method]
        if (row["candidate"] != signal_packet["methods"][label][m]["candidate"]
                or row["preparation_hash"] != start["preparation_hashes"][str(row["candidate"]["prefix"])]
                or (leaf / "STOP.json").exists() or set(row["axes"]) != set(AXES)):
            raise ValueError(f"Wrong source or incomplete axis receipts: {leaf.name}")
        if method == "PR":
            path = read(leaf / "PATH.json")
            path_seeds = set(path["seeds"])
            if (path["candidate"] != row["candidate"] or len(path_seeds) != len(path["seeds"])
                    or seeds.intersection(path_seeds)):
                raise ValueError("PR cohort seed collision or candidate mismatch")
            seeds.update(path_seeds)
        for native_axis, axis in AXES.items():
            receipt = row["axes"][native_axis]
            circuit = leaf / f"{native_axis}_compiled.qpy"
            price = receipt["metrics"]["rz_count"]
            if (receipt != read(leaf / f"{native_axis}_price.json") or receipt["compiler"] != COMPILER
                    or receipt["qpy"] != circuit.name or receipt["qpy_sha256"] != sha(circuit)
                    or type(price) is not int or price <= 0 or not receipt["fingerprint"]):
                raise ValueError(f"Compiled receipt or QPY differs: {leaf.name} {axis}")
            if (method, m, row["path_index"], axis) in fingerprints:
                raise ValueError("Duplicate collected method, round, path and axis")
            fingerprints.add((method, m, row["path_index"], axis))
            source = dict(path=str(result_path), sha256=sha(result_path),
                          selector=["axes", native_axis, "metrics", "rz_count"], pbs_job_id=start["run_id"])
            if method == "B0I":
                verified.append(dict(m=m, axis=axis, physical_id=row["candidate"]["id"], unit_price_RZ=price,
                    kind="DIRECT_COMPILED_DETERMINISTIC", price_evidence=source))
            else:
                cohorts[(m, axis)].append(dict(index=row["path_index"], compiled_rz=price, source=source))
    summaries = []
    for (m, axis), paths in sorted(cohorts.items()):
        mean = Fraction(sum(p["compiled_rz"] for p in paths), len(paths))
        summaries.append(dict(m=m, axis=axis, paths=paths, finite_sample_mean={
            "numerator": mean.numerator, "denominator": mean.denominator},
            planned_paths=8, population_expectation_qualified=False))
    if start.get("rounds") is not None:
        requested = {
            (method, m, index, axis)
            for method, label, paths in (("B0I", "B0I", 1), ("PR", "PR-DF", 8))
            if start.get("method") in (None, label)
            for m in start["rounds"] for index in range(paths) for axis in AXES.values()
        }
        outcome = read(root / "RESULTS.json")
        if fingerprints != requested or outcome["errors"]:
            raise ValueError("Completed receipt coverage differs from the staged job")
    return dict(status="VERIFIED_STAGED_COMPILED_RECEIPTS", compiler=COMPILER, pbs_job_id=start["run_id"],
                b0i_prices=verified, pr_finite_cohorts=summaries, total_axis_receipts=len(fingerprints),
                signal_packet_sha256=start["signal_packet_sha256"],
                previous_pr_preparation_hash=start["prior_pr_binding"],
                current_pr_preparation_hash=start["preparation_hashes"]["7"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("results", "signals"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results.resolve(), args.signals.resolve()), indent=2, allow_nan=False))
