"""Standard-library CLI, static configuration checks and future authority guard.

No numerical dependency or input NPZ is opened by import, status, schema or
validate-config. An authority JSON is provenance, not a substitute for actual
explicit user consent. No such authority is issued by this preparation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

STATUS = "IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED"
INPUT_PATH = "artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz"
INPUT_SHA = "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
H_SHA = "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
BASIS = "occupation_mode0_MSB_alpha_even_beta_odd"
SOURCE_PATHS = (
    "src/trottertracks/algorithm_codesign/m1s_gate0_v2/__init__.py",
    "src/trottertracks/algorithm_codesign/m1s_gate0_v2/control.py",
    "src/trottertracks/algorithm_codesign/m1s_gate0_v2/science.py",
    "scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v2.py",
    "src/trottertracks/__init__.py",
    "src/trottertracks/algorithm_codesign/__init__.py",
)
EPS_KEYS = (
    "eps_H", "eps_hermiticity_H", "eps_hermiticity_factors", "eps_phase", "eps_state_norm",
    "eps_state_residual", "eps_basis", "eps_prob", "eps_imag_coeff",
    "eps_imag_energy", "eps_imag_expectation", "eps_W",
    "eps_v", "eps_rho", "W_error_budget", "v_error_budget", "W_floor",
)
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_bytes(raw):
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("JSON byte limit exceeded")
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _sha(value, length=64):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def candidate():
    """Schema template only: nulls deliberately prevent scientific execution."""
    return {
        "schema": "m1s_gate0_execution_contract_v2",
        "status": "CANDIDATE_NOT_ADOPTED", "adopted": False,
        "input": {"path": INPUT_PATH, "sha256": INPUT_SHA,
                  "hamiltonian_hash": H_SHA, "n_qubits": 8, "df_rank": 12},
        "split_ids": ["G3", "G6", "G9"], "basis": BASIS,
        "law": {"mapping": "JW_saved_DF_v1", "identity_policy": "move_to_hd",
                "coefficient_cutoff": 0.0, "zero_policy": "exact_zero_only",
                "aggregation": "signed_before_abs_lexicographic_IXYZ"},
        "state": {"source": "state_vector", "normalize": False,
                  "label": "NUMERICAL_SECTOR_GROUND_ORACLE_NOT_THEOREM_CERTIFIED"},
        "precision": {"dtype": "complex128", **dict.fromkeys(EPS_KEYS)},
        "runtime": {"python": None, "numpy": None, "workers": 1,
                    "threads": dict.fromkeys(THREAD_KEYS, "1"), "backend": "cpu",
                    "numpy_build_files_sha256": None},
        "limits": {"wall_time_seconds": None, "memory_bytes": None,
                   "output_bytes": None, "log_bytes": None, "max_pauli_terms": 65536,
                   "max_product_contributions": None},
        "decisions": {"automatic_GO_STOP": False, "rho_threshold": None,
                      "dressing_threshold": None, "report_all_splits": True},
        "source_commit": None, "source_sha256": dict.fromkeys(SOURCE_PATHS),
        "scientific_scope_record": {"path": None, "sha256": None},
        "adoption_record": None,
    }


def validate_config(config, require_adopted=False):
    """Validate JSON structure/values only; never read the experiment input."""
    template = candidate()
    errors, pending = [], []

    def shape(value, model, prefix=""):
        if isinstance(model, dict):
            if not isinstance(value, dict) or set(value) != set(model):
                errors.append(f"{prefix}: exact object keys required")
                return
            for key in model:
                shape(value[key], model[key], f"{prefix}.{key}")
    shape(config, template)
    if errors:
        return {"valid_static_shape": False, "execution_ready": False,
                "errors": errors, "pending": pending}
    variable = {"status", "adopted", "precision", "runtime", "limits",
                "source_commit", "source_sha256", "adoption_record", "scientific_scope_record"}
    for key in set(template) - variable:
        if config[key] != template[key]:
            errors.append(f"Unapproved/unsupported condition: {key}")
    if type(config["adopted"]) is not bool or config["status"] not in (
            "CANDIDATE_NOT_ADOPTED", "ADOPTED"):
        errors.append("Invalid adoption classification")
    if (config["status"] == "ADOPTED") != config["adopted"]:
        errors.append("Inconsistent adoption classification")
    if not config["adopted"]:
        pending.append("formal_execution_contract_adoption")
    if config["precision"]["dtype"] != "complex128":
        errors.append("Only complex128 implementation prepared")
    for key in EPS_KEYS:
        value = config["precision"][key]
        if value is None:
            pending.append(key)
        elif not _number(value, positive=key not in ("eps_basis", "eps_hermiticity_factors")):
            errors.append(f"Finite nonnegative (strictly positive except exact checks) contract value required: {key}")
    for budget, target in (("W_error_budget", "eps_W"), ("v_error_budget", "eps_v")):
        a, b = config["precision"][budget], config["precision"][target]
        if _number(a) and _number(b) and a > b:
            errors.append(f"{budget} exceeds adopted accuracy target {target}")
    rt = config["runtime"]
    for key in ("python", "numpy"):
        if rt[key] is None:
            pending.append(f"runtime.{key}")
        elif not isinstance(rt[key], str) or not rt[key]:
            errors.append(f"Invalid runtime.{key}")
    if rt["backend"] != "cpu" or type(rt["workers"]) is not int or rt["workers"] != 1:
        errors.append("Prepared backend is CPU, one process")
    if rt["threads"] != template["runtime"]["threads"]:
        errors.append("Explicit single-thread runtime required")
    builds = rt["numpy_build_files_sha256"]
    if builds is None:
        pending.append("runtime.numpy_build_files_sha256")
    elif not isinstance(builds, dict) or not builds:
        errors.append("NumPy core/BLAS build identity required")
    else:
        for path, value in builds.items():
            if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or not _sha(value):
                errors.append("Invalid distribution file binding")
    lim = config["limits"]
    if lim["wall_time_seconds"] is not None:
        errors.append("No scientific wall-time cap in this candidate")
    for key in set(lim) - {"wall_time_seconds"}:
        if lim[key] is None:
            pending.append(f"limits.{key}")
        elif type(lim[key]) is not int or lim[key] <= 0:
            errors.append(f"Positive integer required: limits.{key}")
    if lim["max_pauli_terms"] != 65536:
        errors.append("Eight-qubit canonical table bound must be 65536")
    if lim["log_bytes"] is not None and lim["output_bytes"] is not None and lim["log_bytes"] != lim["output_bytes"]:
        errors.append("Native log and result share per-file RLIMIT_FSIZE; equal caps required")
    scope = config["scientific_scope_record"]
    if scope["path"] is None or scope["sha256"] is None:
        pending.append("scientific_scope_record")
    elif not isinstance(scope["path"], str) or not _sha(scope["sha256"]):
        errors.append("Scope adoption record path/hash required")
    if config["source_commit"] is None:
        pending.append("source_commit")
    elif not _sha(config["source_commit"], 40):
        errors.append("Full source commit required")
    for path, value in config["source_sha256"].items():
        if value is None:
            pending.append(f"source_sha256.{path}")
        elif not _sha(value):
            errors.append(f"Invalid source SHA256: {path}")
    if config["adoption_record"] is None:
        pending.append("adoption_record")
    elif not isinstance(config["adoption_record"], str) or not config["adoption_record"].strip():
        errors.append("Adoption record must quote actual user/GPT adoption")
    ready = not errors and not pending
    if require_adopted and not ready:
        errors.append("Fixed adopted contract with all precision/runtime limits is required")
    return {"valid_static_shape": not errors, "execution_ready": ready,
            "errors": errors, "pending": pending,
            "scientific_processing_performed": False}


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def _relative(path):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != path or not path:
        raise ValueError("Repository relative path required")
    return path


_SEAL = object()


class _Permit:
    def __init__(self, seal, config, audit):
        if seal is not _SEAL:
            raise ValueError("Authority guard required")
        self._seal = seal
        self.config = config
        self.audit = audit


def require_permit(permit):
    if not isinstance(permit, _Permit) or permit._seal is not _SEAL:
        raise ValueError("Scientific execution denied: no validated authority")


def authorize(root, config_path, auth_commit, auth_path):
    """Future committed authority + adopted contract + source + runtime binding.

    Reads bytes/JSON only. Does not create authorization, marker or output.
    Caller must have actual separate user consent; these checks bind its record.
    """
    root = Path(root).resolve()
    config_path, auth_path = _relative(config_path), _relative(auth_path)
    if not _sha(auth_commit, 40):
        raise ValueError("Full explicit authorization commit required")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != auth_commit:
        raise ValueError("Run must use the fixed authorization HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Dedicated execution checkout must be clean")
    raw = _git(root, "show", f"{head}:{auth_path}")
    if (root / auth_path).read_bytes() != raw:
        raise ValueError("Authorization working bytes differ from commit")
    auth = read_json_bytes(raw)
    keys = {"schema", "scope", "explicit_user_scientific_execution_approval",
            "user_approval_record", "contract_commit", "contract_path",
            "contract_sha256", "source_commit", "output_directory"}
    if not isinstance(auth, dict) or set(auth) != keys:
        raise ValueError("Separate authorization schema mismatch")
    if (auth["schema"] != "m1s_gate0_separate_execution_authorization_v2"
            or auth["scope"] != "Gate0_H4_DEV_G3_G6_G9_W_v_rho_once"
            or auth["explicit_user_scientific_execution_approval"] is not True
            or not isinstance(auth["user_approval_record"], str)
            or not auth["user_approval_record"].strip()):
        raise ValueError("Explicit separate Gate0 approval record required")
    cc = auth["contract_commit"]
    if not _sha(cc, 40) or cc == head or auth["contract_path"] != config_path:
        raise ValueError("Adopted contract must precede separate execution authority")
    _git(root, "merge-base", "--is-ancestor", cc, head)
    contract_raw = _git(root, "show", f"{cc}:{config_path}")
    if ((root / config_path).read_bytes() != contract_raw
            or digest(contract_raw) != auth["contract_sha256"]):
        raise ValueError("Frozen contract identity mismatch")
    config = read_json_bytes(contract_raw)
    check = validate_config(config, require_adopted=True)
    if not check["execution_ready"]:
        raise ValueError(json.dumps(check))
    sc = config["source_commit"]
    if auth["source_commit"] != sc or sc in (head, cc):
        raise ValueError("Separate frozen SOURCE commit required")
    _git(root, "merge-base", "--is-ancestor", sc, cc)
    for path in SOURCE_PATHS:
        source_raw = _git(root, "show", f"{sc}:{path}")
        if digest(source_raw) != config["source_sha256"][path] or (root / path).read_bytes() != source_raw:
            raise ValueError(f"Frozen source mismatch: {path}")
    scope_path = _relative(config["scientific_scope_record"]["path"])
    scope_raw = _git(root, "show", f"{cc}:{scope_path}")
    scope = read_json_bytes(scope_raw)
    if ((root / scope_path).read_bytes() != scope_raw
            or digest(scope_raw) != config["scientific_scope_record"]["sha256"]
            or scope.get("status") != "LIMITED_SCIENTIFIC_SCOPE_ADOPTED_BY_USER"
            or scope.get("record_kind") != "SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION"
            or scope.get("scientific_execution_authorized") is not False):
        raise ValueError("Limited scope adoption identity/classification mismatch")
    for key in ("input", "split_ids", "basis", "law", "state", "decisions"):
        if scope.get(key) != config[key]:
            raise ValueError(f"Adopted scope differs from execution contract: {key}")
    input_raw = (root / INPUT_PATH).read_bytes()
    if len(input_raw) != 8652 or digest(input_raw) != INPUT_SHA:
        raise ValueError("Frozen input byte identity mismatch before numerical import")
    if sys.version.split()[0] != config["runtime"]["python"]:
        raise ValueError("Python version mismatch")
    distribution = importlib.metadata.distribution("numpy")
    if distribution.version != config["runtime"]["numpy"]:
        raise ValueError("NumPy distribution version mismatch before import")
    for path, expected in config["runtime"]["numpy_build_files_sha256"].items():
        if digest(Path(distribution.locate_file(path)).read_bytes()) != expected:
            raise ValueError(f"NumPy build file mismatch: {path}")
    for key, value in config["runtime"]["threads"].items():
        if os.environ.get(key) != value:
            raise ValueError(f"Thread configuration mismatch: {key}")
    output = _relative(auth["output_directory"])
    if not output.startswith("artifacts/track_b_m1s_gate0_result/"):
        raise ValueError("New Gate0 result namespace required")
    if not (root / output).resolve().is_relative_to(root):
        raise ValueError("Output symlink escapes dedicated checkout")
    if (root / output).exists():
        raise ValueError("Output already exists; no implicit retry/overwrite")
    return _Permit(_SEAL, config, {"authorization_commit": head,
        "authorization_path": auth_path, "authorization_sha256": digest(raw),
        "contract_commit": cc, "contract_path": config_path,
        "contract_sha256": digest(contract_raw), "source_commit": sc,
        "source_sha256": config["source_sha256"], "output_directory": output})


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1-S Gate0 SOURCE; scientific execution disabled by default")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="No experiment input or numerical processing")
    sub.add_parser("schema", help="Print candidate JSON template; null values block execution")
    val = sub.add_parser("validate-config", help="JSON-only static checks; never reads NPZ")
    val.add_argument("config")
    exe = sub.add_parser("execute", help="Future separate committed approval AND adopted contract required")
    exe.add_argument("--contract", required=True)
    exe.add_argument("--authorization-commit", required=True)
    exe.add_argument("--authorization-path", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "status"):
            result = {"status": STATUS, "scientific_execution_authorized": False,
                      "default_action": "NO_SCIENTIFIC_PROCESSING", "mandatory_STOP": True}
        elif args.command == "schema":
            result = candidate()
        elif args.command == "validate-config":
            result = validate_config(read_json_bytes(Path(args.config).read_bytes()))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["valid_static_shape"] else 2
        else:
            root = Path(__file__).resolve().parents[4]
            permit = authorize(root, args.contract, args.authorization_commit, args.authorization_path)
            # Numerical module import is strictly after the complete byte/authority guard.
            from .science import run_authorized
            result = run_authorized(root, permit)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError,
            importlib.metadata.PackageNotFoundError) as exc:
        print(json.dumps({"status": "TECHNICAL_STOP", "reason": str(exc),
                          "automatic_scientific_GO_STOP": False}), file=sys.stderr)
        return 2
