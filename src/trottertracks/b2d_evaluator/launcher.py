"""Stage-1 science launcher: fixed conditions, exclusive fresh output and STOP.

Chat permission is checked by Codex before starting this command, not proved
by the program. No approval file, packet, receipt or grant is required.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path

from .evaluation import evaluate
from .input_package import InputRecipe, load_candidate
from .preflight import verify_execution
from .scope import ScopeLedger, Stage1Failure, STOP, OUTPUTS, OUTPUT_LIMITS, candidate_graph, resolved_graph


def failure_code(error):
    if isinstance(error,Stage1Failure):
        return error.code
    message = str(error).lower()
    if type(error).__name__ in ("LinAlgError","FloatingPointError","OverflowError"):
        return "INCONCLUSIVE_NUMERICAL_PRECISION"
    if "non-empty randomized tail" in message:
        return "INCONCLUSIVE_RESIDUAL_DOMAIN"
    if isinstance(error,(PermissionError,ValueError)):
        for words,code in ((('source','fixture'),'SOURCE_MISMATCH'), (('scope','call'),'CALL_SCOPE_MISMATCH'),
            (('grid','graph','tolerance'),'CONTRACT_LAYOUT_MISMATCH'),
            (('runtime','dependency'),'ENVIRONMENT_MISMATCH'),
            (('fresh','output'),'FRESH_OUTPUT_REJECTED'),
            (('retry','resume','stop'),'RETRY_STOP_POLICY_REJECTED'),
            (('identity','input'),'INPUT_IDENTITY_MISMATCH')):
            if any(word in message for word in words):
                return code
        return "SCIENTIFIC_CONFIGURATION_REJECTED"
    if isinstance(error,FileNotFoundError):
        return "REQUIRED_SOURCE_OR_DEPENDENCY_UNAVAILABLE"
    return "EXECUTION_BOUNDARY_EXCEPTION"


def rejection(error):
    # Keep failure codes stable; do not persist arbitrary exception text.
    return dict(status="REJECTED_BEFORE_SCIENCE", reason=failure_code(error), error_type=type(error).__name__,
        science_started=False, stop=STOP)


def _json_default(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if type(value).__module__.split(".")[0] == "numpy" and getattr(value,"ndim",None) == 0:
        return value.item()
    raise TypeError("Unregistered output value type")


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      default=_json_default)


class Evidence:
    """Only this launch's newly created output, or isolated mock directory, may use this."""
    def __init__(self, output, *, mock_only=False):
        self.output, self.mock_only = Path(output), mock_only
        self.total_bytes = 0
        self.output.chmod(0o700)
        self.events = (self.output / "events.jsonl").open("x", encoding="utf-8")

    def write(self, name, value):
        if name not in OUTPUTS:
            raise Stage1Failure("OUTPUT_SCOPE_EXCEEDED")
        encoded = _encode(value)+"\n"
        if len(encoded.encode()) > OUTPUT_LIMITS["json_document_bytes"]:
            raise Stage1Failure("OUTPUT_SCOPE_EXCEEDED")
        self.audit(extra_bytes=len(encoded.encode()))
        with (self.output/name).open("x", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        self.total_bytes += len(encoded.encode())

    def event(self, value):
        encoded = _encode(value)+"\n"
        if len(encoded.encode()) > OUTPUT_LIMITS["event_record_bytes"]:
            raise Stage1Failure("LOG_SCOPE_EXCEEDED")
        if self.total_bytes+len(encoded.encode()) > OUTPUT_LIMITS["total_bytes"]-OUTPUT_LIMITS["stop_reserve_bytes"]:
            raise Stage1Failure("OUTPUT_SCOPE_EXCEEDED")
        # Actual records are bounded by ScopeLedger and each line's byte limit.
        # Audit aggregate bytes at file boundaries, avoiding a directory scan
        # for every small matrix operation's event.
        self.events.write(encoded)
        self.events.flush()
        self.total_bytes += len(encoded.encode())

    def audit(self, extra_bytes=0):
        files = list(self.output.iterdir())
        size = sum(p.stat().st_size for p in files)
        if (len(files)>OUTPUT_LIMITS["files"] or
            any(p.name not in OUTPUTS or p.is_symlink() or not p.is_file() for p in files) or
            size+extra_bytes > OUTPUT_LIMITS["total_bytes"]):
            raise Stage1Failure("OUTPUT_SCOPE_EXCEEDED")
        self.total_bytes = size

    def finish(self, status, ledger, *, result=None, error=None):
        status = dict(status, mode="MOCK_ONLY" if self.mock_only else "SCIENCE_STAGE1",
                      stop=STOP, utc=datetime.now(timezone.utc).isoformat(),
                      calls=ledger.snapshot(), precision_status="UNQUALIFIED_COMPLEX128_DESCRIPTIVE_ONLY")
        if error is not None:
            status.update(reason=failure_code(error),
                          error_type=type(error).__name__)
        failures = []
        documents = [("partial_results.json", dict(rows=ledger.partial_rows(), calls=ledger.snapshot()))]
        if result is not None:
            documents.append(("result.json", result))
        for name,value in documents:
            try:
                self.write(name,value)
            except BaseException:
                failures.append(name)
        status["evidence_write_failures"] = failures
        if failures:
            status.update(status="EVIDENCE_SAVE_FAILED_MANDATORY_STOP",reason="EVIDENCE_PERSISTENCE_FAILURE")
        for name,value in (("status.json",status), ("STOP.json",dict(
                stop=STOP, status=status["status"], reason=status.get("reason"),
                retry_allowed=False, resume_allowed=False))):
            try:
                self.write(name,value)
            except BaseException:
                failures.append(name)
        try:
            self.events.flush()
            os.fsync(self.events.fileno())
        except BaseException:
            failures.append("events.jsonl")
        finally:
            self.events.close()
        if failures:
            status.update(status="EVIDENCE_SAVE_FAILED_MANDATORY_STOP",reason="EVIDENCE_PERSISTENCE_FAILURE")
        return status


def save_inputs(inputs, output):
    # SCIENCE IO only, after fresh output creation and registered input loading.
    import numpy as np
    arrays = {f"D_{i}":value for i,value in enumerate(inputs.blocks.values())}
    arrays.update(physical=inputs.physical_hamiltonian, tail=inputs.normalized_tail, state=inputs.state)
    with (Path(output)/"inputs.npz").open("xb") as file:
        np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())


def _run(run, *, mock_only=False, input_loader=load_candidate,
                 backend_factory=None, input_saver=save_inputs, outer=False):
    """Shared continuation; artificial tests inject only mock loaders/backends."""
    make_template, ledger_type, resolve, evaluator = candidate_graph, ScopeLedger, resolved_graph, evaluate
    if outer == "rte22":
        from .outer_rte_execution import candidate_graph as make_template, OuterRTE22ScopeLedger as ledger_type, resolved_graph as resolve
        from .outer import evaluate_outer as evaluator
    elif outer:
        from .outer_execution import candidate_graph as make_template, OuterScopeLedger as ledger_type, resolved_graph as resolve
        from .outer import evaluate_outer as evaluator
    template = make_template(run.source_commit)
    ledger = ledger_type(run.call_scope, template)
    run.attach_scope(ledger)
    evidence = Evidence(run.output, mock_only=mock_only)
    ledger.sink = evidence.event
    result = None
    try:
        if mock_only and (input_loader is load_candidate or input_saver is save_inputs or backend_factory is None):
            raise Stage1Failure("MOCK_DEPENDENCIES_REQUIRED")
        evidence.write("binding.json", dict(source_commit=run.source_commit,
            call_scope=run.call_scope, environment=run.environment, output=str(run.output),
            recipe=asdict(InputRecipe()),
            mode="MOCK_ONLY" if mock_only else "SCIENCE_STAGE1"))
        context,inputs,provenance = input_loader(run.source_root, execution=run)
        evidence.write("inputs.json",provenance)
        input_saver(inputs,run.output)
        evidence.audit()
        graph = resolve(template,context,inputs,provenance)
        ledger.bind_graph(graph)
        run.check_settings(graph.settings)
        evidence.write("graph.json", dict(settings=asdict(graph.settings), graph_fingerprint=graph.fingerprint,
            nodes=[asdict(n) for n in graph.nodes], rows=graph.rows, correctness=graph.correctness))
        if backend_factory is None:
            if mock_only:
                raise Stage1Failure("MOCK_BACKEND_REQUIRED")
            if outer:
                from .outer_backend import OuterDenseBackend as Backend
            else:
                from .dense_backend import DenseBackend as Backend
            backend_factory = lambda s: Backend(s,inputs,execution=run)
        result = evaluator(graph,backend_factory,execution=run,scope=ledger,mock_only=mock_only)
        ledger.finish()
        return evidence.finish(dict(status="COMPLETED_DESCRIPTIVE_MANDATORY_STOP"),ledger,result=result)
    except BaseException as error:
        return evidence.finish(dict(status="INCONCLUSIVE_MANDATORY_STOP" if
            failure_code(error).startswith("INCONCLUSIVE") else "FAILED_MANDATORY_STOP"),
            ledger,result=result,error=error)


def launch(source_root, output, source_commit):
    """Check scientific configuration, create output exclusively, then run once."""
    return _launch(source_root, output, source_commit, outer=False)


def launch_outer(source_root, output, source_commit):
    """The separately permitted fixed 16-row outer study, using the same writer."""
    return _launch(source_root, output, source_commit, outer=True)


def launch_outer_rte22(source_root, output, source_commit):
    """The separately permitted fixed 22-row analytic study; no retry/resume."""
    return _launch(source_root, output, source_commit, outer="rte22")


def _launch(source_root, output, source_commit, *, outer):
    try:
        root = Path(source_root).resolve()
        if root != Path(__file__).resolve().parents[3]:
            raise Stage1Failure("LOADED_SOURCE_ROOT_MISMATCH")
        verifier = verify_execution
        if outer == "rte22":
            from .outer_rte_execution import verify_execution as verifier
        elif outer:
            from .outer_execution import verify_execution as verifier
        run = verifier(root,output,source_commit)
    except BaseException as error:
        return rejection(error)
    try:
        run.output.parent.mkdir(parents=True,exist_ok=True)
        run.output.mkdir(mode=0o700)
    except BaseException as error:
        # Never write into a colliding output, retry or remove an old STOP.
        return dict(status="FRESH_OUTPUT_FAILED_MANDATORY_STOP", error_type=type(error).__name__,
                    reason="EXCLUSIVE_FRESH_OUTPUT_FAILURE", science_started=False, stop=STOP)
    try:
        return _run(run, outer=outer) if outer else _run(run)
    except BaseException as error:
        # Setup failed after this launch's successful mkdir. Best-effort evidence.
        try:
            with (run.output/"STOP.json").open("x") as file:
                file.write(_encode(dict(status="SETUP_FAILED_MANDATORY_STOP",stop=STOP,
                    error_type=type(error).__name__,reason="POST_OUTPUT_SETUP_FAILURE"))+"\n")
        except BaseException:
            pass
        return dict(status="SETUP_FAILED_MANDATORY_STOP",stop=STOP,science_started=False,
                    error_type=type(error).__name__)
