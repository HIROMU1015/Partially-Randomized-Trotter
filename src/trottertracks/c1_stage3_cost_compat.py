"""Task-local serialization of Qiskit1.3 P0 compilation, old SOURCE intact.

All numerical/native/cost algorithms and logical call budgets are reused.
Only compile callbacks are mutually exclusive; independent candidate work and
Operator checks remain parallel. No shared library/venv settings are changed.
"""
import os
from pathlib import Path
import sys
import threading
import time

from . import c1_stage3_cost_only as base

VERSION = 'qiskit_p0_compile_mutex_v1'
FAMILY = 'artifacts/research_c_c1_stage3_cost_compat_science'


class CompileSerialCalls:
    def __init__(self, call):
        self.call = call
        self.lock = threading.Lock()
        self.completed = 0
        self.wait_seconds = 0.

    def __call__(self, kind, label, function):
        if kind != 'compile': return self.call(kind,label,function)
        tick = time.monotonic()
        with self.lock:
            self.wait_seconds += time.monotonic()-tick
            # Charge only after acquiring the lock. The old Calls object checks
            # its STOP before starting, so waiting jobs cannot bypass a failure.
            value = self.call(kind,label,function)
            self.completed += 1
            return value


class CompatibleBackend(base.CostBackend):
    def prepare(self, call):
        self.compile_calls = CompileSerialCalls(call)
        super().prepare(self.compile_calls)


def preflight(root, contract_path, input_path):
    v = base.preflight(root,contract_path,input_path)
    if (v['contract']['compatibility_version'] != VERSION
            or v['contract']['entry'] != 'scripts/run_research_c_c1_stage3_cost_compat.py'):
        raise ValueError('COST_COMPAT_ENTRY_VERSION')
    c=v['contract'];previous_path=v['root']/c['previous_contract']
    if base.io.sha(previous_path) != c['previous_contract_sha256']:
        raise ValueError('PREVIOUS_COST_CONTRACT')
    previous=base.io.read(previous_path)
    if any(c[k] != previous[k] for k in ('proposal','candidate_ids','call_limits','input_sha256','saved_inputs')):
        raise ValueError('CHANGED_COMPAT_SCIENTIFIC_CONDITIONS')
    return v


def execute(root, contract_path, input_path, output):
    """User-permitted repair rerun; fresh output, never old-run resume."""
    v = preflight(root,contract_path,input_path);output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/FAMILY):
        raise ValueError('FRESH_COST_COMPAT_OUTPUT_REQUIRED')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=v['root']/'.cache/stage3-cost-compat'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    sys.pycache_prefix=str(v['root']/'.cache/stage3-cost-compat/pycache')
    backend = CompatibleBackend(v,output)
    try:return base.run(backend,v['contract']['proposal'],output)
    finally:
        if output.exists():
            base.io.save(output/'SOURCE.json',v['source']);base.io.save(output/'contract.json',v['contract'])
            base.io.save(output/'runtime.json',v['runtime'])
            calls=getattr(backend,'compile_calls',None)
            base.io.save(output/'compatibility.json',dict(version=VERSION,
                P0_compile_only_serialized=True,scientific_conditions_unchanged=True,
                compile_completed=calls.completed if calls else 0,
                compile_queue_wait_seconds=calls.wait_seconds if calls else 0.,
                old_run_preserved=True))
