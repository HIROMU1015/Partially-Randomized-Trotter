"""A0/B0 SOURCE/snapshot checks and existing Evidence/ScopeLedger glue.

Metadata inspection never imports scientific packages or creates output.
"""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import subprocess

from .input_package import fingerprint, read_fixture_definition
from .preflight import _check_source
from .scope import ScopeLedger, Stage1Failure, STOP
from .launcher import Evidence, rejection

SNAPSHOT_COMMIT = "33715848bfb83d49ec4e54d6eef481012a7441e4"
SNAPSHOT = "artifacts/research_b_outer_rte22/20261025/analytic-once"
SNAPSHOT_FILES = ("inputs.json", "inputs.npz", "binding.json", "graph.json", "result.json")


def snapshot_metadata(root):
    root = Path(root)
    blobs = {}
    for name in SNAPSHOT_FILES:
        path = SNAPSHOT + "/" + name
        expected = subprocess.check_output(["git", "rev-parse", SNAPSHOT_COMMIT+":"+path], cwd=root, text=True).strip()
        actual = subprocess.check_output(["git", "hash-object", "--", path], cwd=root, text=True).strip()
        if actual != expected:
            raise Stage1Failure("SNAPSHOT_BYTES_MISMATCH")
        blobs[name] = expected
    inputs = json.loads((root/SNAPSHOT/"inputs.json").read_text())
    if (inputs["recipe"]["ld"], inputs["sector_basis_indices"], inputs["occupied_modes"],
            inputs["deterministic_fragment_indices"], inputs["randomized_block_indices"]) != (2,[1,2],[0],[0,1],[2]):
        raise Stage1Failure("SNAPSHOT_CONDITIONS_MISMATCH")
    return inputs, blobs


def registered_scope(study, calls, **metadata):
    if len(set(calls)) != len(calls):
        raise ValueError("Duplicate scientific call identity")
    return dict(study=study, boundaries=dict(sorted(Counter(n for n,k in calls).items())),
        calls=[list(c) for c in calls], max_event_records=2*len(calls)+32,
        retry_allowed=False, resume_allowed=False, mandatory_stop=True, **metadata)


class FollowupLedger(ScopeLedger):
    """Use the existing scientific recorder; these studies have no old DAG rows."""
    validate_registry = staticmethod(lambda scope, graph: None)

    def __init__(self, scope):
        super().__init__(scope, None)
        self.rows = []
        self.allowed = {tuple(item) for item in scope["calls"]}

    def call(self, name, function, *, identity=None, summary=None):
        if (name, identity) not in self.allowed:
            raise Stage1Failure("UNREGISTERED_FOLLOWUP_CALL")
        return super().call(name, function, identity=identity, summary=summary)

    def partial_rows(self):
        return self.rows

    def finish(self):
        if dict(self.completed) != self.scope["boundaries"] or self.failed:
            raise Stage1Failure("INCOMPLETE_REGISTERED_SCOPE")


def preflight(root, output, source, study, scope):
    root, out = Path(root).resolve(), Path(output).resolve()
    if root != Path(__file__).resolve().parents[3]:
        raise Stage1Failure("LOADED_SOURCE_ROOT_MISMATCH")
    _check_source(root, source, output=out)
    inputs, blobs = snapshot_metadata(root)
    parent = root/"artifacts"/("research_b_outer_"+study)
    if out.exists() or out == parent or not out.is_relative_to(parent):
        raise ValueError("Fresh dedicated output required")
    dependencies = ("mpmath", "numpy") if study == "a0" else (
        "numpy", "scipy", "qiskit", "qiskit-nature", "openfermion", "openfermionpyscf", "pyscf")
    versions = {name:importlib.metadata.version(name) for name in dependencies}
    if study == "a0" and versions["mpmath"] != "1.3.0":
        raise ValueError("A0 requires verified mpmath 1.3.0 APIs")
    if study == "b0":
        if versions["qiskit"] != "1.3.0": raise ValueError("B0 compiler version mismatch")
        read_fixture_definition(root)
    return dict(root=root, output=out, source=source, scope=scope,
        inputs=inputs, snapshot_blobs=blobs, versions=versions)


def run_prepared(run, body, *, mock_only=False):
    """Single attempt: reuse exclusive output, Evidence failure/partial/STOP IO."""
    out = run["output"]
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir(mode=0o700)
    except BaseException as error:
        return dict(status="FRESH_OUTPUT_FAILED_MANDATORY_STOP",stop=STOP,error_type=type(error).__name__)
    ledger = FollowupLedger(run["scope"])
    try:
        evidence = Evidence(out, mock_only=mock_only)
    except BaseException as error:
        (out/"STOP.json").write_text(json.dumps(dict(stop=STOP,status="SETUP_FAILED_MANDATORY_STOP")))
        return dict(status="SETUP_FAILED_MANDATORY_STOP",stop=STOP,error_type=type(error).__name__)
    ledger.sink = evidence.event
    evidence.precision_status=("EMPIRICAL_90_128_U_NUM_NULL" if run["scope"]["study"]=="a0"
                               else "EMPIRICAL_COMPLEX128_NO_GAMMA_CERTIFICATE")
    try:
        evidence.write("binding.json",{k:v for k,v in run.items() if k not in ("root","output","inputs")})
        evidence.write("inputs.json",run["inputs"])
        result = body(run, ledger, evidence)
        ledger.finish()
        status=("INCONCLUSIVE_MANDATORY_STOP" if result.get("scientific_status")=="INCONCLUSIVE"
                else "COMPLETED_DESCRIPTIVE_MANDATORY_STOP")
        return evidence.finish(dict(status=status),ledger,result=result)
    except BaseException as error:
        status = "INCONCLUSIVE_MANDATORY_STOP" if getattr(error,"code","").startswith("INCONCLUSIVE") else "FAILED_MANDATORY_STOP"
        return evidence.finish(dict(status=status),ledger,error=error)


def main(study):
    import argparse
    from importlib import import_module
    from .launcher import _encode
    parser = argparse.ArgumentParser(description="Fixed "+study+"; inspect metadata or explicitly execute once")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect",action="store_true")
    mode.add_argument("--execute",action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--output")
    if study == "b0":parser.add_argument("--diagnose-first",action="store_true",help="Only original B3,+T,representative0,X; no costs")
    args = parser.parse_args()
    module = import_module("."+("a0_reference" if study == "a0" else "b0_native"), __package__)
    root = Path(__file__).resolve().parents[3]
    scope_options={"diagnostic":True} if getattr(args,"diagnose_first",False) else {}
    if args.inspect:
        print(_encode(dict(status="METADATA_ONLY_NO_SCIENCE",scope=module.scope(**scope_options))))
        return 0
    if not args.source_commit or not args.output:
        parser.error("--execute requires --source-commit and --output")
    try:
        run = preflight(root,args.output,args.source_commit,study,module.scope(**scope_options))
        status = run_prepared(run,module.execute)
    except BaseException as error:
        status = rejection(error)
    print(_encode(status))
    return 0 if status["status"] == "COMPLETED_DESCRIPTIVE_MANDATORY_STOP" else 2
