#!/usr/bin/env python3
"""Dedicated synthetic suite under pre-import artifact/import barriers."""
import argparse
import builtins
from datetime import datetime,timezone
import importlib.abc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
TEST="tests/tracks/resource_applicability/test_ax1b_preparation.py"


def install_boundary():
    counters=dict(protected_access_attempts=0,scientific_import_attempts=0)
    def blocked(path):
        if isinstance(path,int):return False
        try:name=os.fsdecode(os.fspath(path)).replace("\\","/")
        except TypeError:return False
        parts=name.split("/")
        if name.lower().endswith((".npz",".npy",".pkl",".pickle")) or any(x in {".runtime","runtime","cache","registry"} or x.endswith("_registry") for x in parts):return True
        if "artifacts" in parts:
            return not any(x in {"track_a_ax1a","track_a_ax1b_preparation","track_a_ax1b_prelaunch","track_a_ax1b_identity_compatibility"} for x in parts)
        return False
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if blocked(path):
                counters["protected_access_attempts"]+=1
                raise AssertionError("AX1b synthetic suite attempted scientific artifact access")
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open=wrap(builtins.open);io.open=wrap(io.open)
    os.open=wrap(os.open);os.stat=wrap(os.stat);os.lstat=wrap(os.lstat)
    class ImportBarrier(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=="trotterlib" or fullname.startswith("trotterlib."):
                counters["scientific_import_attempts"]+=1
                raise AssertionError("Legacy scientific modules forbidden in synthetic suite")
    sys.meta_path.insert(0,ImportBarrier())
    return counters


class AuditPlugin:
    def __init__(self):self.passed=self.failed=self.skipped=0
    def pytest_runtest_logreport(self,report):
        if report.failed:self.failed+=1
        elif report.skipped:self.skipped+=1
        elif report.when=="call" and report.passed:self.passed+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output",type=Path,required=True)
    args=parser.parse_args()
    for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ[var]="1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1"
    sys.path.insert(0,str(ROOT/"src"))
    os.chdir(ROOT)
    started=datetime.now(timezone.utc).isoformat();clock=time.monotonic()
    counters=install_boundary()
    import pytest
    audit=AuditPlugin()
    isolated_temp=tempfile.mkdtemp(prefix="ax1b_synthetic_")
    command=["-q","-rs","-p","no:cacheprovider","--noconftest","--collect-in-virtualenv","--ignore=artifacts","--basetemp="+isolated_temp,TEST]
    code=pytest.main(command,plugins=[audit])
    from trottertracks.resource_applicability.ax1b_execution import environment
    import hashlib
    sources=sorted([*ROOT.glob("src/trottertracks/resource_applicability/ax1b_*.py"),
                    ROOT/"scripts/resource_applicability/run_track_a_ax1b.py",
                    ROOT/"scripts/resource_applicability/run_track_a_ax1b_identity_preflight.py",Path(__file__).resolve(),ROOT/TEST])
    result=dict(schema_version="track_a_ax1b_synthetic_test_audit_v1",scope="SYNTHETIC_ONLY_NO_SAVED_SCIENCE_VALUES",
                started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-clock,
                command=[sys.executable,*sys.argv],pytest_arguments=command,python=platform.python_version(),python_full=sys.version,
                python_releaselevel=sys.version_info.releaselevel,numpy=importlib.metadata.version("numpy"),scipy=importlib.metadata.version("scipy"),pytest=pytest.__version__,
                passed=audit.passed,failed=audit.failed,skipped=audit.skipped,exit_code=int(code),processes=1,blas_threads=1,
                environment=environment(),source_files_after_successful_test=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
                legacy_science_inputs_read=0,real_data_fit_executed=False,**counters)
    args.audit_output.parent.mkdir(parents=True,exist_ok=True)
    args.audit_output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("AX1B_PREPARATION_TEST_AUDIT:",json.dumps(result,sort_keys=True))
    return int(code) if not any(counters.values()) else 1


if __name__=="__main__":
    sys.exit(main())
