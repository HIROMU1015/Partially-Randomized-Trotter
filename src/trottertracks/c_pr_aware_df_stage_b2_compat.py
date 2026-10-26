"""B2 technical relaunch: serialize Qiskit's shared equivalence-library users.

Old B2 SOURCE/run01 remain unchanged. Build and P0 share one process-local
mutex; ControlledGate.inverse also calls .control(), so actual-adjoint BUILD
uses that mutex too. CAR, sampling, Operator decoding and matrix checks remain
parallel. The original SOURCE digest continues to key the same 64 samples.
"""
from contextlib import contextmanager
from pathlib import Path
import resource
import subprocess
import threading
import time

from . import c_pr_aware_df_stage_b2 as base

VERSION = 'qiskit_shared_equivalence_build_compile_adjoint_mutex_v1'
SCOPE = 'C_PR_AWARE_DF_STAGE_B2_NATIVE_COST_COMPAT_V1'
FRESH = base.FAMILY + '/2026-10-27/run02'
UNCHANGED = ('policy','allowances','compiler','gaussian','compiler_phase','sampling',
    'call_limits','maximum_logical_units','evaluation_packages')


class SharedQiskitCalls:
    def __init__(self, call):
        self.call = call; self.lock = threading.RLock()
        self.wait_seconds = 0.; self.completed = 0

    def __getattr__(self, name): return getattr(self.call,name)

    def __call__(self, kind, label, function):
        if kind not in ('kernel_build','compile'): return self.call(kind,label,function)
        tick = time.monotonic()
        with self.lock:
            self.wait_seconds += time.monotonic()-tick
            # Existing Calls checks STOP after this common lock is acquired.
            value = self.call(kind,label,function); self.completed += 1; return value

    @contextmanager
    def adjoint_builds(self):
        original = base.b.native.actual_adjoint
        def locked(circuit):
            tick = time.monotonic()
            with self.lock:
                self.wait_seconds += time.monotonic()-tick
                return original(circuit)
        base.b.native.actual_adjoint = locked
        try: yield
        finally: base.b.native.actual_adjoint = original


def preflight(root, path, input_path):
    root=Path(root).resolve();c=base.io.read(path)
    if (c['scope']!=SCOPE or c['compatibility']!=VERSION or c['fresh_output']!=FRESH
            or c['technical_relaunch']!=1 or c['retry_within_run'] or c['resume']):
        raise ValueError('B2_COMPAT_SCOPE')
    previous_path=root/c['previous_contract']
    if base.io.sha(previous_path)!=c['previous_contract_sha256']:raise ValueError('ORIGINAL_B2_CONTRACT')
    previous=base.preflight(root,previous_path,input_path)
    if (any(c[k]!=previous[k] for k in UNCHANGED)
            or c['sampling_SOURCE_sha256']!=previous['source_manifest_sha256']):
        raise ValueError('CHANGED_B2_SCIENTIFIC_OR_SAMPLING_POLICY')
    if base.io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('COMPAT_SOURCE')
    for group in (base.io.read(root/c['source_manifest'])['files'],c['previous_run_files']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or base.io.sha(target)!=h:raise ValueError('COMPAT_SOURCE_OR_PRIOR_RUN: '+p)
    failure=base.io.read(root/c['previous_failure']);stop=base.io.read(root/c['previous_STOP'])
    if (failure['error']!='Already borrowed' or failure['first_failure']['kind']!='kernel_build'
            or failure['first_failure']['error_type']!='RuntimeError' or not stop['failed']
            or stop['scientific_units']!=429 or stop['retry'] or stop['resume']):
        raise ValueError('NOT_REGISTERED_TECHNICAL_INTERRUPTION')
    return c,previous


def run(root, path, input_path, output):
    root=Path(root).resolve();c,previous=preflight(root,path,input_path);output=Path(output).resolve()
    if output!=root/FRESH or output.exists():raise ValueError('FRESH_COMPAT_NO_RESUME_OR_RETRY')
    output.mkdir(parents=True);calls=base.integrated.Calls(output,base.LIMITS,base.MAX_CALLS)
    shared=SharedQiskitCalls(calls);start=time.monotonic()
    try:
        base.io.save(output/'contract.json',c);base.io.save(output/'original_scientific_contract.json',previous)
        base.io.save(output/'SOURCE.json',dict(base.io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            sampling_SOURCE_sha256=previous['source_manifest_sha256']))
        with shared.adjoint_builds():return base.execute(previous,root,output,shared)
    except BaseException as error:
        calls.stop.set();base.io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:
        count=sum(calls.counts.values())
        base.io.save(output/'compatibility.json',dict(version=VERSION,shared_equivalence_mutex=True,
            serialized_operations=['kernel build','P0 compile','actual adjoint build only'],
            parallel_operations=['CAR reference','Operator decode','matrix checks','X/Y'],
            shared_lock_wait_seconds=shared.wait_seconds,completed_build_compile=shared.completed,
            original_sampling_SOURCE_sha256=previous['source_manifest_sha256'],scientific_conditions_unchanged=True))
        base.io.save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=count,
            call_counts=dict(calls.counts),technical_relaunch=1,retry_within_run=0,resume=0,
            previous_scientific_units=429,cumulative_scientific_units=429+count,mandatory_stop=True))
        base.io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
