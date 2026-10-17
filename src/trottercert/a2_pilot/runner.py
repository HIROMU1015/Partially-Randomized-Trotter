"""One guarded fixed-DF pipeline, no authorization issuance or retry path."""

import contextlib
import datetime
from hashlib import sha256
from pathlib import Path
import os
import platform
import resource
import time

from ..a2_spec01_03.common import ContractError, canonical_bytes, integer, require
from .adapter import adapt_snapshot
from .contract import preflight, verify_git_sources
from .core01_generator import generate_spec01_03
from .core01_checker import check_spec01_03
from .core04_generator import generate_spec04_07
from .core04_checker import check_spec04_07
from .identity import source_identity
from .measure import BoundedLog, Timings, maximum_rat_bit_lengths
from .storage import LocalStore


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run(contract_bytes, auth_bytes, pins, *, load_input, source_check, store_factory, artificial_test=False):
    """Dependency injection is for artificial memory-store tests, never a CLI flag."""
    overall_wall, overall_cpu = time.perf_counter(), time.process_time()
    try:
        contract, auth = preflight(contract_bytes, auth_bytes, pins, artificial_test=artificial_test)
        source_check(contract)
    except Exception as exc:
        return {"schema": "A2_PILOT_RESULT_V1", "status": "PREEXECUTION_REJECTED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "input_opened": False, "marker_created": False, "mandatory_STOP": True}
    limits, task, inp = contract["limits"], contract["task"], contract["input"]
    claim = {"status": "ONE_SHOT_CONSUMED_BEFORE_INPUT_IO", "run_id": auth["run_id"],
             "contract_sha256": pins["contract_sha256"], "authorization_sha256": pins["authorization_sha256"],
             "input_manifest_id": pins["input_manifest_id"], "source": contract["source"], "claimed_at_utc": _now(),
             "artificial_memory_test_only": artificial_test, "retry": "NO_AUTOMATIC_RETRY"}
    try:
        store = store_factory(pins["contract_sha256"], limits)
        store.claim(claim)
    except Exception as exc:
        # Never overwrite an old claim/result, including interrupted claims.
        return {"schema": "A2_PILOT_RESULT_V1", "status": "CLAIM_REJECTED_OR_INCOMPLETE",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "input_opened": False, "marker_may_exist": True, "mandatory_STOP": True}
    calls, timings, saved = {}, Timings(), []
    log_limit = integer(limits["log_bytes"], natural=True)
    stdout, stderr = BoundedLog(log_limit), BoundedLog(log_limit)
    result = {"schema": "A2_PILOT_RESULT_V1", "status": "TECHNICAL_STOP", "run_id": auth["run_id"],
              "contract_sha256": pins["contract_sha256"], "authorization_sha256": pins["authorization_sha256"],
              "input_manifest_id": pins["input_manifest_id"], "source": contract["source"]["identity"],
              "state": contract["state"], "input_opened": False, "one_shot_consumed": True,
              "conditional_on_exact_ground_access": True, "ground_preparation_cost": "NOT_EVALUATED",
              "native_cost": "NOT_EVALUATED", "total_resource_improvement": "NOT_EVALUATED",
              "original_molecular_guarantee": "NOT_ESTABLISHED", "mandatory_STOP": True,
              "artificial_memory_test_only": artificial_test, "started_at_utc": _now()}
    result["runtime"] = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                         "logical_cpus": str(os.cpu_count()), "affinity_cpus": str(len(os.sched_getaffinity(0))),
                         "backend": "EXACT_FRACTION_SYMPLECTIC_COMMUTATORS_BATCHED_OCCUPATION_TRACE",
                         "resource_caps_imposed": False, "shared_environment_changed": False}

    def save(name, record):
        timings.transition("OUTPUT_SERIALIZATION_AND_WRITE")
        try:
            encoded = canonical_bytes(record)
            store.write(name, encoded)
            saved.append({"name": name, "bytes": str(len(encoded)), "sha256": sha256(encoded).hexdigest(),
                          "maximum_rat_bit_lengths": maximum_rat_bit_lengths(record)})
        finally:
            timings.transition(None)

    def invoke(name, function, *args, **kwargs):
        require(name not in calls, "CALL_SCOPE", name, "No duplicate pipeline calls")
        calls[name] = "1"
        timings.transition(name)
        try:
            return function(*args, **kwargs)
        finally:
            timings.transition(None)

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            save("contract.json", contract)
            # Preserve exact submitted bytes separately; canonical JSON is not its raw hash.
            store.write("contract.submitted.json", contract_bytes)
            store.write("authorization.submitted.json", auth_bytes)
            timings.transition("INPUT_IO_AND_PINNED_BYTES_CHECK")
            result["input_IO_attempted"] = True
            blob = load_input(contract)
            result["input_opened"] = True
            timings.transition(None)
            timings.transition("RAW_SNAPSHOT_PERSISTENCE")
            store.write("input_snapshot.npz", blob)
            saved.append({"name": "input_snapshot.npz", "bytes": str(len(blob)), "sha256": sha256(blob).hexdigest()})
            timings.transition(None)
            raw = invoke("adapter", adapt_snapshot, blob, inp["manifest"], inp["mapping"])
            save("raw_exact.json", raw)
            number, order = inp["mapping"]["sector_N"], task["fragment_order"]
            expected_model = None if inp["expected_model_id"] == "DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING" else inp["expected_model_id"]
            base = invoke("SPEC01_03_generator", generate_spec01_03, raw, particle_number=number, fragment_order=order)
            save("SPEC01_03_proposal.json", base)
            require(base["verdict"] == "EVIDENCE_GENERATED_UNVERIFIED", "GENERATION_REJECTED", "SPEC01_03", "See saved partial proposal")
            verified_base = invoke("SPEC01_03_checker", check_spec01_03, raw, base["evidence"],
                                   expected_particle_number=number, expected_fragment_order=order, expected_model_id=expected_model)
            save("SPEC01_03_checked.json", verified_base)
            require(verified_base["verdict"] == "SPEC01_03_VERIFIED_EXACT_FIXED_DF", "CHECK_REJECTED", "SPEC01_03", "Independent checker rejected")
            model_id = verified_base["context"]["model_id"]
            producer_observer = lambda label: timings.transition("SPEC04_07_generator/" + label)
            proposal = invoke("SPEC04_07_generator", generate_spec04_07, raw, base["evidence"],
                              particle_number=number, occupied_modes=task["occupied_modes"], sqrt_bits=task["sqrt_bits"],
                              comparison_policy=task["comparison_policy"], fragment_order=order,
                              expected_model_id=model_id, stage_observer=producer_observer)
            calls["internal_SPEC01_03_rechecks"] = "1"
            save("SPEC04_07_proposal.json", proposal)
            require(proposal["verdict"] == "EVIDENCE_GENERATED_UNVERIFIED", "GENERATION_REJECTED", "SPEC04_07", "See saved partial proposal")
            checker_observer = lambda label: timings.transition("SPEC04_07_checker/" + label)
            checked = invoke("SPEC04_07_checker", check_spec04_07, raw, base["evidence"], proposal["evidence"],
                             expected_particle_number=number, expected_occupied_modes=task["occupied_modes"],
                             expected_sqrt_bits=task["sqrt_bits"], expected_comparison_policy=task["comparison_policy"],
                             expected_fragment_order=order, expected_model_id=model_id, stage_observer=checker_observer)
            calls["internal_SPEC01_03_rechecks"] = "2"
            save("SPEC04_07_checked.json", checked)
            require(checked["verdict"] == "SPEC04_07_VERIFIED_EXACT_FIXED_DF", "CHECK_REJECTED", "SPEC04_07", "Independent checker rejected")
            require(calls == limits["calls"], "CALL_SCOPE", "calls", "Completed pipeline differs from fixed call inventory")
            result.update(status="COMPLETED_FIXED_DF_CERTIFICATE_COMPARISON", context=checked["context"],
                          certificate_verdict="VERIFIED_EXACT_FOR_FIXED_DF_UNDER_STATED_CHECKER_MODEL",
                          L=checked["L"], U=checked["U"], W=checked["W"], Lambda=checked["Lambda"],
                          K_SOS_upper=checked["K_SOS_upper"], Cbar2=checked["Cbar2"],
                          step_comparison=checked["step_comparison"], limitations=checked["limitations"])
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="TECHNICAL_STOP", failure={"type": type(exc).__name__, "message": str(exc)})
    finally:
        timings.transition(None)
        result.update(calls=calls, costs=timings.rows, records=saved, completed_at_utc=_now(),
                      costs_exclude_final_log_and_result_writes=True,
                      run_wall_seconds_before_finalization=format(time.perf_counter()-overall_wall, ".9f"),
                      run_CPU_seconds_before_finalization=format(time.process_time()-overall_cpu, ".9f"),
                      process_peak_RSS_KiB=str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      stdout_dropped_bytes=str(stdout.dropped_bytes), stderr_dropped_bytes=str(stderr.dropped_bytes))
        try:
            store.write("stdout.txt", bytes(stdout.data))
            store.write("stderr.txt", bytes(stderr.data))
            for name, log in (("stdout.txt", stdout), ("stderr.txt", stderr)):
                saved.append({"name": name, "bytes": str(len(log.data)), "sha256": sha256(log.data).hexdigest()})
            result["saved_bytes_before_final_record"] = str(store.written)
            store.write("result.json", canonical_bytes(result))
        except Exception as exc:
            # Do not remove/reuse the consumed claim when result persistence fails.
            return {**result, "status": "PERSISTENCE_FAILURE_STOP", "persistence_error": str(exc),
                    "record_preservation": "BEST_EFFORT_CONSUMED_CLAIM_REMAINS"}
    return result


def run_files(repository, contract_path, authorization_path, pins, *, execute=False):
    """Production entry: default closed. No artificial-test authorization accepted."""
    if execute is not True:
        return {"status": "PREEXECUTION_REJECTED", "reason": "EXPLICIT_EXECUTE_FLAG_REQUIRED", "input_opened": False, "marker_created": False, "mandatory_STOP": True}
    repository = Path(repository).resolve()
    contract_bytes, auth_bytes = Path(contract_path).read_bytes(), Path(authorization_path).read_bytes()

    def check_source(contract):
        require(str(repository) == contract["execution_store"]["canonical_repository"],
                "STORE", "canonical_repository", "Different checkout cannot bypass the fixed one-shot ledger")
        verify_git_sources(repository, contract)

    def load(contract):
        target = (repository / contract["input"]["repository_path"]).resolve()
        require(target.is_relative_to(repository), "INPUT_PATH", "snapshot", "Input must stay in intended repository")
        import subprocess
        saved = subprocess.check_output(["git", "--no-optional-locks", "show", contract["input"]["commit"] + ":" + contract["input"]["repository_path"]], cwd=repository)
        blob = target.read_bytes()
        require(blob == saved, "INPUT_COMMIT", "snapshot", "Working bytes differ from fixed input commit")
        return blob

    def store_factory(key, limits):
        directory = repository / "artifacts/qc_a2_pilot_execution"
        require(not (repository / "artifacts").is_symlink() and not directory.is_symlink()
                and directory.resolve().is_relative_to(repository),
                "STORE", "ledger", "Canonical owned ledger only")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return LocalStore(directory, key, output_bytes=integer(limits["output_bytes"], natural=True), log_bytes=integer(limits["log_bytes"], natural=True))

    return _run(contract_bytes, auth_bytes, pins, load_input=load, source_check=check_source, store_factory=store_factory)
