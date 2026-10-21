"""AX-1b identities and gates. Stdlib only; importing this reads no artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

AX1A_COMMIT = "6b6959b9e76ead049a9a247dadfc1f85161916e8"
AX1A_HASHES = {
    "docs/research/track_a_ax1a_preanalysis_contract.md": "340aae4253fad7769743567176c21872adfcd0ff452839319cc1b8d8cc407059",
    "docs/research/track_a_ax1a_model_comparison_and_fit.md": "da213cccb20c9267b33089b28701375d2c932b6e6aa7a187a21d66f6101ed0a6",
    "docs/research/track_a_ax1a_evaluation_protocol.md": "441031da30be28b351f605cd16d5ca2a119e4927c918151c077ee067973f3c59",
    "artifacts/resource_applicability/track_a_ax1a/2026-10-09/input_allowlist_v1.json": "1a36c78bc70ea9d9e7b52f7a8ee04e9639ea9f488cf7ae27892710615a4bf44b",
    "artifacts/resource_applicability/track_a_ax1a/2026-10-09/execution_plan_draft_v1.json": "548c0e1b02c77596a0f802edfb47a5de28733d338b0d3fd6181395d978eaeb16",
}
AXES = ("cosine", "sine")
METRICS = ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")
FEATURES = ("intercept", "n_det", "E_rand", "n_fixed", "q")
RETAIN_PRIORITY = ("intercept", "E_rand", "n_det", "q", "n_fixed")
FLAGS = dict(ax1b_analysis_authorized=False, science_authorized=False,
             explicit_user_launch_required=True, mandatory_stop=True, next_stage_authorized=False)
CASE_CONDITIONAL = "REF_SHOT_PRED_COST_CONDITIONAL_ORACLE"
BIAS_NA = "N_A_NO_OPERATIONAL_BIAS_PREDICTOR"


class Stop(ValueError):
    def __init__(self, status: str, reason: str):
        self.status = status if status.startswith("AX1B_STOP_") else "AX1B_STOP_" + status
        self.reason = reason
        super().__init__(f"{self.status}: {reason}")


def require(ok, status, reason):
    if not ok:
        raise Stop(status, reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha256(canonical(value).encode())


def number(value, label="value", positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "SCHEMA", f"{label} is not numeric")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), "NUMERICAL_FIT", f"invalid {label}")
    return float(value)


def relative_path(path):
    require(isinstance(path, str) and path != "", "INPUT_IDENTITY", "empty/non-string path")
    p = PurePosixPath(path)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in path and str(p) == path,
            "INPUT_IDENTITY", "noncanonical relative path")
    require(p.suffix.lower() not in {".npz", ".npy", ".pkl", ".pickle"} and
            not any(x in {".runtime", "runtime", "cache", "registry"} or x.endswith("_registry") for x in p.parts),
            "INPUT_IDENTITY", "protected science storage")
    return p


def safe_path(root, relative):
    p = relative_path(relative)
    root = Path(root).resolve()
    target = root.joinpath(*p.parts)
    # Do not follow symlinks, including parent directories, outside the allowlist.
    walk = root
    for part in p.parts:
        walk = walk / part
        require(not walk.is_symlink(), "INPUT_IDENTITY", "symlink input/output forbidden")
    require(target.resolve().is_relative_to(root), "INPUT_IDENTITY", "path escapes root")
    return target


def load_contract(root):
    """Only the five pinned AX-1a contract files; never a scientific input."""
    values = {}
    for path, expected in AX1A_HASHES.items():
        data = safe_path(root, path).read_bytes()
        require(sha256(data) == expected, "CONTRACT_CONFLICT", f"AX1a bytes changed: {path}")
        if path.endswith(".json"):
            values[Path(path).name] = json.loads(data)
    allow = values["input_allowlist_v1.json"]
    plan = values["execution_plan_draft_v1.json"]
    require(len(allow["entries"]) == 45, "CONTRACT_CONFLICT", "45-input contract")
    require(digest(plan["model_configuration"]) == plan["model_configuration_sha256"], "CONTRACT_CONFLICT", "model configuration hash")
    require(all(plan[k] is v and allow[k] is v for k, v in FLAGS.items()), "CONTRACT_CONFLICT", "AX1a STOP flags")
    return allow, plan


def select_fields(obj, selector):
    """Selector syntax registered by AX1a; returns values, never follows paths."""
    parts = selector.split(".")
    def walk(value, rest):
        if not rest:
            return [value]
        key, array = (rest[0][:-2], True) if rest[0].endswith("[]") else (rest[0], False)
        if not isinstance(value, dict) or key not in value:
            return []
        node = value[key]
        if array:
            return [v for item in node for v in walk(item, rest[1:])] if isinstance(node, list) else []
        return walk(node, rest[1:])
    return walk(obj, parts)


def operational_na():
    return dict(axis_bias_pred=None, N_pred_by_axis=None, eligible_operational_pred=None,
                G_operational_pred=None, regret_operational=None, availability_status=BIAS_NA)


def check_schema(value,schema,location="$ "):
    """Validate the explicit subset used by preparation schemas; not a general JSON Schema engine."""
    predicates={"object":lambda x:isinstance(x,dict),"array":lambda x:isinstance(x,list),
                "string":lambda x:isinstance(x,str),"boolean":lambda x:type(x) is bool,
                "integer":lambda x:type(x) is int,"number":lambda x:type(x) in (int,float) and math.isfinite(x),"null":lambda x:x is None}
    kinds=schema.get("type")
    if kinds is not None:
        kinds=kinds if isinstance(kinds,list) else [kinds]
        require(any(predicates[k](value) for k in kinds),"SCHEMA",location+" type")
    if "const" in schema:
        require(type(value) is type(schema["const"]) and value==schema["const"],"SCHEMA",location+" const")
    if "enum" in schema:
        require(value in schema["enum"],"SCHEMA",location+" enum")
    if isinstance(value,dict):
        require(all(k in value for k in schema.get("required",[])),"SCHEMA",location+" required field")
        for key,child in schema.get("properties",{}).items():
            if key in value:check_schema(value[key],child,location+"."+key)
    if isinstance(value,list) and "items" in schema:
        for i,item in enumerate(value):check_schema(item,schema["items"],location+f"[{i}]")
    if "minimum" in schema and value is not None:
        require(value>=schema["minimum"],"SCHEMA",location+" minimum")
    if "pattern" in schema and value is not None:
        require(re.search(schema["pattern"],value) is not None,"SCHEMA",location+" pattern")
