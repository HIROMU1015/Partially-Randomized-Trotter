"""Profile-paired budget vectors and immutable per-stage freeze artifacts."""
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from .guard import TechnicalFailure
from .numerical import RESOURCES


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)+"\n").encode()


def identity(value):
    return sha256(canonical_bytes(value)).hexdigest()


def query_recipe(x, n, tag, vectors):
    if len(vectors) > 12:
        raise TechnicalFailure("maximum 12 budget vectors")
    dedup = {}
    for vector in vectors:
        if set(vector["resources"]) != set(RESOURCES):
            raise TechnicalFailure("budget must be a complete resource vector")
        values = tuple(str(F(vector["resources"][r])) for r in RESOURCES)
        key = identity(values)
        if key not in dedup:
            dedup[key] = {"budget_id": key, "resources": dict(zip(RESOURCES, values)), "sources": []}
        dedup[key]["sources"].append(vector["source"])
    queries = []
    for key, vector in sorted(dedup.items()):
        for resource in RESOURCES:
            caps = {r: vector["resources"][r] for r in RESOURCES if r != resource}
            queries.append({"query_id": identity((x, n, tag, key, resource)), "x": x,
                            "n": n, "tag": tag, "objective": resource,
                            "budget_id": key, "caps": caps})
    return list(dedup.values()), queries


def write_freeze(path, stage, points, input_identities, guard):
    if guard.current_phase != "BUDGET_FREEZE":
        raise TechnicalFailure("freeze written outside Phase A")
    body = {"schema": "ra_d0_budget_freeze_v2", "stage": stage,
            "input_identities": input_identities, "points": points,
            "number_of_vectors": sum(len(p["budget_vectors"]) for p in points),
            "number_of_queries": sum(len(p["queries"]) for p in points),
            "B3_solver_calls_in_this_phase": 0}
    raw = guard.write(path, body)
    digest = sha256(raw).hexdigest()
    guard.write(Path(path).with_suffix(".receipt.json"), {"SHA256": digest, "stage": stage,
                "input_identities": input_identities, "number_of_vectors": body["number_of_vectors"],
                "number_of_queries": body["number_of_queries"]})
    return digest


def load_freeze(path, expected_sha256, input_identities):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise TechnicalFailure("budget freeze SHA256 mismatch")
    body = json.loads(raw)
    if body["input_identities"] != input_identities or body["B3_solver_calls_in_this_phase"] != 0:
        raise TechnicalFailure("budget freeze identity/phase mismatch")
    for point in body["points"]:
        vectors = [{"resources": v["resources"], "source": v["sources"][0]}
                   for v in point["budget_vectors"]]
        _, expected_queries = query_recipe(point["x"], point["n"], point["tag"], vectors)
        if point["queries"] != expected_queries:
            raise TechnicalFailure("budget/query derivation mismatch")
    return body


def coverage_contexts(anchor_witnesses):
    return [x for x in ("1/8", "1/4") if not anchor_witnesses.get(x, False)]


def classification(rows, complete, technical_failure=False):
    if technical_failure or not complete:
        return "D0_TECHNICAL_INCONCLUSIVE"
    anchors = {x: any(r["x"] == x and r["tag"] == "PRIMARY_ANCHOR" and r["strict_witness"]
                      for r in rows) for x in ("1/8", "1/4")}
    if all(anchors.values()):
        return "D0_STRONG_DEGREE_LOCAL_SIGNAL"
    if any(r["strict_witness"] for r in rows):
        return "D0_LOCAL_DEGREE_LOCAL_SIGNAL"
    return "D0_NO_REGISTERED_WITNESS"
