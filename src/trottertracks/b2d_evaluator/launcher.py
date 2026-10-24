"""Dormant Stage-1 launcher. No production approval source is available here.

CLI execution therefore fails closed. A future operations integration must
supply the real approval verifier; a packet/string/mock cannot replace it.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path

from trottertracks.pr_dictionary_validation.authorization import claim_packet
from .evaluation import ExecutionApproval, evaluate
from .input_package import load_candidate
from .preflight import verify_execution
from .scope import ScopeLedger, Stage1Failure, STOP, OUTPUTS, OUTPUT_LIMITS, candidate_graph, resolved_graph


def production_authority():
    """No discoverable production provider; never manufacture self-approval."""
    return None


def failure_code(error):
    if isinstance(error,Stage1Failure):
        return error.code
    message = str(error).lower()
    if isinstance(error,PermissionError):
        for words,code in ((('already used','already consumed'),'GRANT_CONSUMED_NO_RESUME'),
            (('source','fixture'),'SOURCE_MISMATCH'), (('scope','call'),'CALL_SCOPE_MISMATCH'),
            (('grid','graph','tolerance'),'CONTRACT_LAYOUT_MISMATCH'),
            (('runtime','dependency'),'ENVIRONMENT_MISMATCH'),
            (('fresh','output'),'FRESH_OUTPUT_REJECTED'),
            (('retry','resume','stop'),'RETRY_STOP_POLICY_REJECTED'),
            (('grant','one-shot'),'GRANT_REJECTED'),
            (('approval','authority','record'),'APPROVAL_RECORD_REJECTED')):
            if any(word in message for word in words):
                return code
        return "AUTHORIZATION_REJECTED"
    if type(error).__name__ in ("LinAlgError","FloatingPointError","OverflowError"):
        return "INCONCLUSIVE_NUMERICAL_PRECISION"
    if isinstance(error,FileNotFoundError):
        return "REQUIRED_RECORD_UNAVAILABLE"
    if "non-empty randomized tail" in message:
        return "INCONCLUSIVE_RESIDUAL_DOMAIN"
    return "EXECUTION_BOUNDARY_EXCEPTION"


def rejection(error):
    # No untrusted exception text, paths, approval content or stack trace.
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
    """Only a newly claimed output, or isolated mock directory, may use this."""
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
    # SCIENCE IO only, after approved one-shot claim and registered input loading.
    import numpy as np
    arrays = {f"D_{i}":value for i,value in enumerate(inputs.blocks.values())}
    arrays.update(physical=inputs.physical_hamiltonian, tail=inputs.normalized_tail, state=inputs.state)
    with (Path(output)/"inputs.npz").open("xb") as file:
        np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())


def _run_claimed(verified, permit, *, mock_only=False, input_loader=load_candidate,
                 backend_factory=None, input_saver=save_inputs):
    """Testable continuation. Mocks NEVER call claim_packet or DenseBackend."""
    template = candidate_graph(verified.contract["source_commit"])
    ledger = ScopeLedger(verified.contract["call_scope"], template)
    permit.attach_scope(ledger)
    evidence = Evidence(verified.output, mock_only=mock_only)
    ledger.sink = evidence.event
    result = None
    try:
        if mock_only and (input_loader is load_candidate or input_saver is save_inputs or backend_factory is None):
            raise Stage1Failure("MOCK_DEPENDENCIES_REQUIRED")
        evidence.write("binding.json", dict(source_commit=verified.contract["source_commit"],
            contract_digest=verified.contract_digest, authorization_digest=verified.authorization_digest,
            call_scope=verified.contract["call_scope"], python_runtime=verified.contract.get("python_runtime"),
            dependency_versions=verified.contract.get("dependency_versions"),
            mode="MOCK_ONLY" if mock_only else "SCIENCE_STAGE1"))
        context,inputs,provenance = input_loader(verified.source_root, execution=permit)
        evidence.write("inputs.json",provenance)
        input_saver(inputs,verified.output)
        evidence.audit()
        graph = resolved_graph(template,context,inputs,provenance)
        ledger.bind_graph(graph)
        permit.check_settings(graph.settings)
        evidence.write("graph.json", dict(settings=asdict(graph.settings), graph_fingerprint=graph.fingerprint,
            nodes=[asdict(n) for n in graph.nodes], rows=graph.rows, correctness=graph.correctness))
        if backend_factory is None:
            if mock_only:
                raise Stage1Failure("MOCK_BACKEND_REQUIRED")
            from .dense_backend import DenseBackend
            backend_factory = lambda s: DenseBackend(s,inputs,execution=permit)
        grant = verified.authorization
        approval = ExecutionApproval(grant["approval_reference"], grant["grant_id"],
            graph.settings.source_commit,graph.fingerprint,tuple(graph.call_counts.items()),
            str(verified.output),"NO_RETRY_NO_RESUME",mock_only=mock_only)
        result = evaluate(graph,backend_factory,approval,execution=permit,scope=ledger)
        ledger.finish()
        return evidence.finish(dict(status="COMPLETED_DESCRIPTIVE_MANDATORY_STOP"),ledger,result=result)
    except BaseException as error:
        return evidence.finish(dict(status="INCONCLUSIVE_MANDATORY_STOP" if
            failure_code(error).startswith("INCONCLUSIVE") else "FAILED_MANDATORY_STOP"),
            ledger,result=result,error=error)


def launch(packet_dir, source_root, output, *, authority=None):
    """Real entry: verify before any output, claim, input or scientific import."""
    try:
        trusted_authority = production_authority()
        if trusted_authority is None:
            raise Stage1Failure("PRODUCTION_AUTHORITY_UNCONNECTED")
        if authority is not None and authority is not trusted_authority:
            raise Stage1Failure("UNREGISTERED_APPROVAL_AUTHORITY")
        authority = trusted_authority
        root = Path(source_root).resolve()
        if root != Path(__file__).resolve().parents[3]:
            raise Stage1Failure("LOADED_SOURCE_ROOT_MISMATCH")
        contract = json.loads((Path(packet_dir)/"contract.json").read_text())
        graph = candidate_graph(contract["source_commit"])
        verified = verify_execution(packet_dir,root,output,graph,authority=authority)
        # This branch also rejects every mock authority before the existing claim.
        claim = verified.for_existing_claim()
    except BaseException as error:
        return rejection(error)
    try:
        claim_packet(claim)
    except BaseException as error:
        # A racing mkdir may have consumed the marker. Never write into a
        # colliding output or remove the marker. Return a sanitized STOP record.
        return dict(status="CLAIM_FAILED_MANDATORY_STOP", error_type=type(error).__name__,
                    reason="ATOMIC_CLAIM_OR_FRESH_OUTPUT_FAILURE", science_started=False, stop=STOP)
    try:
        return _run_claimed(verified,verified.after_existing_claim())
    except BaseException as error:
        # Setup failed after this launch's successful mkdir. Best-effort evidence.
        try:
            with (verified.output/"STOP.json").open("x") as file:
                file.write(_encode(dict(status="SETUP_FAILED_MANDATORY_STOP",stop=STOP,
                    error_type=type(error).__name__,reason="POST_CLAIM_SETUP_FAILURE"))+"\n")
        except BaseException:
            pass
        return dict(status="SETUP_FAILED_MANDATORY_STOP",stop=STOP,science_started=False,
                    error_type=type(error).__name__)
