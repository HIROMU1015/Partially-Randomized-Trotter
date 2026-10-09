"""Fail-closed AX-1b validator/runner. Numerical imports follow authorization."""
from __future__ import annotations

import importlib.metadata
import csv
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time

from .ax1b_contract import AX1A_COMMIT, FLAGS, Stop, canonical, digest, load_contract, require, safe_path, sha256,check_schema

THREAD_VARS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")


def environment():
    try:
        return _environment_identity()
    except (importlib.metadata.PackageNotFoundError,OSError,AttributeError) as exc:
        raise Stop("ENVIRONMENT","dependency identity unavailable: "+str(exc)) from exc


def _environment_identity():
    packages={}
    for name in ("numpy","scipy","pytest"):
        dist=importlib.metadata.distribution(name)
        packages[name]=dict(version=dist.version,root=str(Path(dist.locate_file("")).resolve()),
                            metadata_sha256=sha256(dist.read_text("METADATA").encode()))
    identity=dict(python=platform.python_version(),python_releaselevel=sys.version_info.releaselevel,
                executable=sys.executable,numpy=importlib.metadata.version("numpy"),scipy=importlib.metadata.version("scipy"),
                pytest=importlib.metadata.version("pytest"),platform=platform.platform(),prefix=sys.prefix,base_prefix=sys.base_prefix,
                pythonpath=os.environ.get("PYTHONPATH"),packages=packages,
                executable_sha256=sha256(Path(sys.executable).resolve().read_bytes()),
                nnls_source_sha256=sha256(Path(importlib.metadata.distribution("scipy").locate_file("scipy/optimize/_nnls.py")).read_bytes()))
    return dict(identity,environment_fingerprint_sha256=digest(identity))


def validate_preparation(root,bundle):
    """Contract/new source bytes only. No reader of the 45 science inputs."""
    allow,plan=load_contract(root)
    require(bundle["schema_version"] in {"track_a_ax1b_preparation_manifest_v1","track_a_ax1b_prelaunch_manifest_v1"},"SCHEMA","preparation schema")
    require(bundle["ax1a_commit"]==AX1A_COMMIT,"CONTRACT_CONFLICT","wrong AX1a commit")
    require(all(bundle[k] is v for k,v in FLAGS.items()),"AUTHORIZATION","preparation must remain unauthorized")
    require(bundle["model_configuration_sha256"]==plan["model_configuration_sha256"],"CONTRACT_CONFLICT","changed model configuration")
    for f in bundle["frozen_files"]:
        require(sha256(safe_path(root,f["path"]).read_bytes())==f["sha256"],"IMPLEMENTATION","preparation source/test/schema/doc hash changed")
    schema_raw=safe_path(root,bundle["schemas"]["path"]).read_bytes()
    require(sha256(schema_raw)==bundle["schemas"]["sha256"],"SCHEMA","schemas file hash")
    schemas=json.loads(schema_raw)
    check_schema(bundle,schemas["schemas"]["preparation_manifest"])
    audit_path=bundle["synthetic_test_audit"]["path"]
    audit_raw=safe_path(root,audit_path).read_bytes()
    require(sha256(audit_raw)==bundle["synthetic_test_audit"]["sha256"],"IMPLEMENTATION","test audit hash")
    audit=json.loads(audit_raw)
    check_schema(audit,schemas["schemas"]["synthetic_test_audit"])
    require(audit["exit_code"]==0 and audit["failed"]==0 and audit["skipped"]==0 and audit["passed"]>0
            and audit["protected_access_attempts"]==0 and audit["scientific_import_attempts"]==0,"IMPLEMENTATION","synthetic suite gate")
    return allow,plan


def authorize(root,bundle,authorization,execute_saved_analysis=False,launch_authorization_sha256=None):
    """Denial is first: before contract reads, science IO, imports or fits."""
    require(execute_saved_analysis is True and authorization is not None,"AUTHORIZATION","separate authorization and explicit launch required")
    require(authorization.get("ax1b_analysis_authorized") is True and authorization.get("science_authorized") is False
            and authorization.get("explicit_user_launch_required") is True and authorization.get("mandatory_stop") is True
            and authorization.get("next_stage_authorized") is False,"AUTHORIZATION","AX1b-only execution flags")
    require(launch_authorization_sha256==digest(authorization),"AUTHORIZATION","launch must name canonical authorization hash")
    require(authorization.get("independent_review_decision")=="APPROVE_AX1B_EXECUTION" and authorization.get("user_launch_record"),
            "AUTHORIZATION","review and user launch record missing")
    require(authorization.get("preparation_manifest_sha256")==digest(bundle),"IMPLEMENTATION","preparation manifest binding")
    allow,plan=validate_preparation(root,bundle)
    commit=authorization.get("source_commit")
    require(isinstance(commit,str) and re.fullmatch(r"[0-9a-f]{40}",commit),"IMPLEMENTATION","source commit unbound")
    def git(*args):return subprocess.check_output(["git",*args],cwd=root)
    require(git("rev-parse","HEAD").decode().strip()==commit,"IMPLEMENTATION","launch HEAD differs from source commit")
    for f in bundle["frozen_files"]:
        require(sha256(git("show",commit+":"+f["path"]))==f["sha256"],"IMPLEMENTATION","source commit blob differs")
    observed=environment()
    require(sys.version_info[:2]>=(3,11) and sys.version_info.releaselevel=="final" and observed["python_releaselevel"]=="final" and observed["scipy"]=="1.14.1",
            "ENVIRONMENT","stable Python>=3.11 and exact SciPy1.14.1 required")
    require(authorization.get("environment")==observed,"ENVIRONMENT","environment not confirmed exactly")
    proof=authorization.get("analysis_environment_synthetic_audit")
    require(isinstance(proof,dict) and proof.get("exit_code")==0 and proof.get("passed",0)>0
            and all(proof.get(k)==0 for k in ["failed","skipped","protected_access_attempts","scientific_import_attempts"])
            and proof.get("real_data_fit_executed") is False,"ENVIRONMENT","same-source synthetic success in analysis environment unconfirmed")
    require(all(proof.get(k)==observed[k] for k in ["python","numpy","scipy"]),"ENVIRONMENT","synthetic proof belongs to another environment")
    if bundle.get("schema_version")=="track_a_ax1b_prelaunch_manifest_v1":
        require(proof.get("environment")==observed,"ENVIRONMENT","synthetic proof dependency origins/fingerprint differ")
    expected_source={f["path"]:f["sha256"] for f in bundle["frozen_files"] if f["path"].endswith(".py")}
    require({f["path"]:f["sha256"] for f in proof.get("source_files_after_successful_test",[])}==expected_source,
            "IMPLEMENTATION","analysis-environment test proof source hashes differ")
    limits=authorization.get("resources",{})
    for key in ("assigned_cpu_cores","ram_limit_bytes","wall_time_limit_seconds","output_disk_limit_bytes"):
        require(type(limits.get(key)) is int and limits[key]>0,"BUDGET",key+" missing/unconfirmed")
    require(limits.get("processes")==1 and limits.get("blas_threads")==1,"BUDGET","single process/BLAS1 contract")
    affinity=limits.get("cpu_affinity")
    require(isinstance(affinity,list) and all(type(core) is int for core in affinity) and len(affinity)==limits["assigned_cpu_cores"] and len(set(affinity))==len(affinity)
            and set(affinity).issubset(os.sched_getaffinity(0)),"BUDGET","assigned CPU affinity not confirmed")
    require(authorization.get("output_directory")==plan["output"]["directory"],"OUTPUT_COLLISION","output namespace differs from registered plan")
    destination=safe_path(root,authorization["output_directory"])
    require(not destination.exists(),"OUTPUT_COLLISION","output directory exists; no retry/resume")
    require(limits["output_disk_limit_bytes"]<=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,"BUDGET","insufficient available disk")
    return dict(authorized=True,authorization_sha256=digest(authorization),source_commit=commit,allowlist=allow,plan=plan,
                resources=limits,environment=observed,output_directory=str(destination))


def _limits(permit):
    limits=permit["resources"]
    for key in THREAD_VARS:
        os.environ[key]="1"
    os.sched_setaffinity(0,limits["cpu_affinity"])
    resource.setrlimit(resource.RLIMIT_AS,(limits["ram_limit_bytes"],limits["ram_limit_bytes"]))
    def timed_out(signum,frame):
        raise Stop("BUDGET","wall-time limit reached")
    signal.signal(signal.SIGALRM,timed_out)
    signal.setitimer(signal.ITIMER_REAL,limits["wall_time_limit_seconds"])


def encode_output(name,value):
    if name.endswith(".json"):
        return (canonical(value)+"\n").encode()
    if name.endswith(".jsonl"):
        return ("".join(canonical(row)+"\n" for row in value)).encode()
    if name.endswith(".md"):
        require(isinstance(value,str),"SCHEMA","Markdown output type")
        return (value.rstrip()+"\n").encode()
    require(name.endswith(".csv") and isinstance(value,list),"SCHEMA","registered output format")
    stream=io.StringIO(newline="")
    header=sorted({k for row in value for k in row})
    writer=csv.DictWriter(stream,fieldnames=header,lineterminator="\n")
    writer.writeheader()
    for row in value:
        writer.writerow({k:canonical(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    return stream.getvalue().encode()


def execute(root,bundle,authorization,execute_saved_analysis=False,launch_authorization_sha256=None):
    permit=authorize(root,bundle,authorization,execute_saved_analysis,launch_authorization_sha256)
    _limits(permit)
    started=time.monotonic()
    # These are new saved-value modules, not legacy scientific modules/runners.
    from .ax1b_data import VerifiedReader,project_saved
    from .ax1b_analysis import analyze
    reader=VerifiedReader(root,permit["allowlist"],permit)
    schemas=json.loads(safe_path(root,bundle["schemas"]["path"]).read_bytes())["schemas"]
    destination=Path(permit["output_directory"])
    destination.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        require(Path(name).name==name,"OUTPUT_COLLISION","output must be a simple filename")
        if name=="predictions.jsonl":
            for row in value:check_schema(row,schemas["prediction"])
        if name=="terminal_status.json":check_schema(value,schemas["terminal"])
        if name=="conditional_oracle_selection.csv":
            from .ax1b_evaluation import validate_selection_record
            for row in value:
                validate_selection_record(row)
                diagnostics=[row["full_set_diagnostic"],row["common_set_diagnostic"]] if row.get("row_kind")=="common_support_selection" else [row] if row.get("row_kind")=="selection_diagnostic" else []
                for diagnostic in diagnostics:check_schema(diagnostic,schemas["selection_diagnostic"])
        raw=encode_output(name,value)
        used=sum(p.stat().st_size for p in destination.iterdir() if p.is_file())
        require(used+len(raw)<=permit["resources"]["output_disk_limit_bytes"],"BUDGET","output disk cap")
        with (destination/name).open("xb") as f:
            f.write(raw)
    try:
        values={e["path"]:reader.read(e["path"]) for e in permit["allowlist"]["entries"]}
        rows=project_saved(values,permit["allowlist"])
        outputs=analyze(rows,permit["plan"],values,permit["allowlist"])
        for prediction in outputs["predictions.jsonl"]:
            prediction.update(model_source_commit=permit["source_commit"],model_configuration_sha256=permit["plan"]["model_configuration_sha256"],
                              prediction_record_time_utc=datetime.now(timezone.utc).isoformat(),prediction_timing_scope="OBSERVED_H4_DEVELOPMENT_NOT_BLIND")
        require(set(outputs)|{"input_identity_audit.json","terminal_status.json","output_manifest.json"}==set(permit["plan"]["output"]["planned_files"]),
                "CONTRACT_CONFLICT","registered output filenames differ")
        # Before outputs are accepted, recheck all source/contract/input bytes.
        validate_preparation(root,bundle)
        for e in permit["allowlist"]["entries"]:
            require(sha256(safe_path(root,e["path"]).read_bytes())==e["sha256"],"INPUT_IDENTITY","input changed during analysis")
        write("input_identity_audit.json",reader.input_audit)
        for filename,value in outputs.items():
            write(filename,value)
        write("terminal_status.json",dict(status="AX1B_COMPLETE_WITH_DECLARED_NA",mandatory_stop=True,next_stage_authorized=False,
                                          source_commit=permit["source_commit"],wall_seconds=time.monotonic()-started,
                                          authorization_sha256=permit["authorization_sha256"],environment=permit["environment"]))
        files=[dict(path=p.name,sha256=sha256(p.read_bytes())) for p in sorted(destination.iterdir())]
        write("output_manifest.json",dict(schema_version="track_a_ax1b_outputs_v1",files=files,input_allowlist_sha256=permit["plan"]["input_allowlist"]["sha256"],
             model_configuration_sha256=permit["plan"]["model_configuration_sha256"],source_commit=permit["source_commit"],mandatory_stop=True,next_stage_authorized=False))
        return dict(status="AX1B_COMPLETE_WITH_DECLARED_NA",output_manifest_sha256=sha256((destination/"output_manifest.json").read_bytes()),mandatory_stop=True,next_stage_authorized=False)
    except Exception as exc:
        status=exc.status if isinstance(exc,Stop) else "AX1B_STOP_IMPLEMENTATION"
        # Failure audit only; partial outputs never become completed evidence.
        try:
            write("failure_audit.json",dict(status=status,reason=str(exc),partial_outputs_valid=False,mandatory_stop=True,next_stage_authorized=False))
        except Exception:
            pass  # Do not bypass a full disk cap to write an audit.
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
