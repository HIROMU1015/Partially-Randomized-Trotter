"""JSON-boundary repair and saved-JSON-only aggregation of completed fits.

Old SOURCE and failed run stay byte-identical. No NPZ decode, fitting,
objective/gradient/operator evaluation, changed criteria or new candidate.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time

from . import c_common_convergence_pilot as legacy

io=legacy.io
_DECIDE=legacy.decide
_SAVE=io.save
FAMILY="artifacts/research_c_common_convergence_export_repair/2026-11-01"
SCOPE="C_COMMON_CONVERGENCE_SAVED_JSON_EXPORT_REPAIR_V1"
LIMITS=dict(saved_scalar_comparison=1)


def json_ready(value):
    """Only scalar conversion. Arrays/complex/unsupported objects fail closed."""
    if isinstance(value,dict):return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
    if type(value).__module__.split(".")[0]=="numpy":
        import numpy as np
        if isinstance(value,np.generic):return json_ready(value.item())
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError("UNSUPPORTED_JSON_VALUE: "+type(value).__name__)


def decision(*args):
    result=json_ready(_DECIDE(*args))
    # Validate inside the scientific call, before any output file is opened.
    json.dumps(result,allow_nan=False)
    return result


def save_json(path,payload):
    """Serialize before I/O, publish atomically and never overwrite a target."""
    path=Path(path)
    body=json.dumps(json_ready(payload),ensure_ascii=False,allow_nan=False,indent=2)+"\n"
    pending=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                prefix=".json-export-",delete=False) as f:
            pending=Path(f.name);f.write(body);f.flush();os.fsync(f.fileno())
        try:os.link(pending,path)
        except FileExistsError:
            if (path.name!="result.json" or not isinstance(payload,dict)
                    or payload.get("status")!="TECHNICAL_FAILURE"):raise
            # An existing partial/valid result is evidence, never replace it.
            os.link(pending,path.with_name("result_failure.json"))
    finally:
        if pending is not None:pending.unlink(missing_ok=True)


@contextmanager
def json_exports():
    """Dedicated-process compatibility shim for the unchanged paired runner."""
    if legacy.decide is not _DECIDE or io.save is not _SAVE:
        raise ValueError("DEDICATED_UNPATCHED_EXPORT_PROCESS_REQUIRED")
    try:
        legacy.decide=decision;io.save=save_json
        yield
    finally:legacy.decide=_DECIDE;io.save=_SAVE


def preflight(root,plan_path):
    """Bytes/JSON only; even saved NPZ files are hashed, never decoded."""
    root=Path(root).resolve();p=io.read(plan_path)
    if (p["scope"]!=SCOPE or p["output"]!=FAMILY+"/run01"
            or p["call_limits"]!=LIMITS or p["new_fits"]!=0):
        raise ValueError("FIXED_EXPORT_REPAIR_PLAN")
    if io.sha(root/p["source_manifest"])!=p["source_manifest_sha256"]:
        raise ValueError("REPAIR_SOURCE_MANIFEST")
    for group in (io.read(root/p["source_manifest"])["files"],p["saved_inputs"]):
        for name,digest in group.items():
            f=(root/name).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=digest:
                raise ValueError("SOURCE_OR_SAVED_INPUT: "+name)
    old=io.read(root/p["original_contract"])
    for name,digest in old["saved_inputs"].items():
        if io.sha(root/name)!=digest:raise ValueError("ORIGINAL_INPUT: "+name)
    # Preserve all 37 original files, including the invalid JSON and failed STOP.
    evidence=io.read(root/p["original_postflight"])
    for name,digest in evidence["raw_sha256"].items():
        if io.sha(root/name)!=digest:raise ValueError("ORIGINAL_RAW: "+name)
    return p


def recover(root,plan_path,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    p=preflight(root,plan_path)
    if output!=(root/p["output"]).resolve() or output.exists():
        raise ValueError("FRESH_REPAIR_OUTPUT_REQUIRED")
    old=root/p["original_output"];contract=io.read(root/p["original_contract"])
    # All science ended successfully at the call level; only export failed.
    stopped=io.read(old/"STOP.json")
    if not stopped["failed"] or stopped["scientific_units"]!=25:
        raise ValueError("EXPECTED_COMPLETED_COMPUTATION_WITH_EXPORT_STOP")
    for role in legacy.ROLES:
        if io.read(old/role/"STOP.json")["failed"]:
            raise ValueError("INCOMPLETE_FIT_EVIDENCE")
    output.mkdir(parents=True)
    calls=legacy.old.Calls(output,LIMITS,1);start=time.monotonic()
    try:
        save_json(output/"SOURCE.json",io.read(root/p["source_manifest"]))
        save_json(output/"repair_plan.json",p)
        rows={k:io.read(old/(k+"_sector.json")) for k in legacy.ROLES}
        reports={k:{name:io.read(old/k/file) for name,file in dict(
            optimizer="optimizer.json",diagnostic="final_diagnostic.json",tensor="tensor.json").items()}
            for k in legacy.ROLES}
        references={k:{name:io.read(root/file) for name,file in files.items()}
                    for k,files in contract["comparators"].items()}
        def compare():
            result=decision(rows,reports,references)
            if result["status"]!=p["expected_scientific_status"]:
                raise ValueError("PRESERVED_SCIENTIFIC_CLASSIFICATION_CHANGED")
            return result
        result=calls("saved_scalar_comparison","same decision; no fit",compare)
        save_json(output/"result.json",result)
        save_json(output/"recovery.json",dict(status="JSON_EXPORT_RECOVERED",
            original_output=p["original_output"],original_scientific_units=25,
            original_STOP_unchanged=True,original_result_is_partial=True,
            added_saved_scalar_units=1,new_fits=0,optimizer_invocations=0,
            new_objective_gradient_operator_evaluations=0,NPZ_decodes=0,
            original_fit_retry_resume=0,repair_attempt=1,scientific_status=result["status"],
            qualified_tail_comparison=result["qualified_tail_comparison"],mandatory_stop=True))
    except BaseException as error:
        calls.stop.set()
        save_json(output/"failure.json",dict(error=str(error),first_failure=calls.first_failure))
        raise
    finally:
        # Keep the old journal/STOP implementation; only this fresh repair run
        # records one saved-scalar unit and no optimizer invocation.
        legacy.existing.finish(output,calls,start)
