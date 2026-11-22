"""Register the two H6 Full PF round-7 direct prices in a preview registry."""

import argparse
import hashlib
import json
from pathlib import Path


CID = "4147c55d83984e21ea4a26586c561c5a2ca56ab5e99b86a97f154747592f3b59"
PHYSICAL_ID = "5cc99f967c6a176c43fb"
AXES = {"cosine": "X", "sine": "Y"}
COMPILER = "1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha256.update(block)
    return sha256.hexdigest()


def register(preview, result_dir, output):
    result_file = result_dir / "result.json"
    result = read(result_file)
    start = read(result_dir / "STARTED.json")
    template = read(preview / "CONFIGURATION.json")
    results = result["axes"]
    candidate = {"system": "H6", "method": "FULL", "formula": "4th(m5_best)",
                 "prefix": 19, "T": 12.8, "q": 8, "delta": 1.6,
                 "R_outer": None, "K": None, "id": PHYSICAL_ID}
    if (template["id"] != CID or start["candidate"] != candidate or result["candidate"] != candidate
            or not start.get("pbs_job_id") or (result_dir / "STOP.json").exists()
            or result["status"] != "PASS_PRICE_ACQUIRED" or result["round"] != 7
            or set(results) != set(AXES) or len({r["fingerprint"] for r in results.values()}) != 2):
        raise ValueError("Not a completed pair for the fixed H6 Full PF m=7")
    table = read(preview / "PARTIAL_PRICES.json")
    if any(r["m"] == 7 for r in table["axes"]):
        raise ValueError("The base price table already has m=7 evidence")
    for name, axis in AXES.items():
        receipt = results[name]
        circuit = result_dir / f"{name}_compiled.qpy"
        price = receipt["metrics"]["rz_count"]
        if (receipt != read(result_dir / f"{name}_price.json") or receipt["compiler"] != COMPILER
                or not receipt["fingerprint"] or type(price) is not int or price <= 0
                or receipt["qpy"] != circuit.name or digest(circuit) != receipt["qpy_sha256"]):
            raise ValueError(f"Incomplete or mismatched m=7 receipt for {axis}")
        table["axes"].append(dict(m=7, axis=axis, physical_id=PHYSICAL_ID, configuration_id=CID,
                                  unit_price_RZ=price, kind="DIRECT_COMPILED_DETERMINISTIC",
                                  price_evidence={"path": str(result_file), "sha256": digest(result_file),
                                                  "selector": ["axes", name, "metrics", "rz_count"],
                                                  "pbs_job_id": start["pbs_job_id"]}))
    registry = read(preview / "REGISTRY.json")
    rules = read(preview / "ROUND.json")
    registry["configurations"][CID]["prices"] = str(output / "PARTIAL_PRICES.json")
    rules["pf_evidence_registry"] = str(output / "REGISTRY.json")
    output.mkdir(mode=0o700, exist_ok=False)
    write(output / "PARTIAL_PRICES.json", table)
    write(output / "REGISTRY.json", registry)
    write(output / "ROUND.json", rules)
    return len(table["axes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("preview", "result_dir", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(register(args.preview.resolve(), args.result_dir.resolve(), args.output.resolve()))
