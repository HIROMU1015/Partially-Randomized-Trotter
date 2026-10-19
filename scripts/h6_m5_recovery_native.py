"""Reuse frozen science; add durable, explicitly unvalidated diagnostic evidence."""
import ast
import math
import os
from pathlib import Path
import h46_m5_native as original
import h46_m5_tol_native as tol
import h6_m5_recovery_contract as c


def atomic_binary(path, writer):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # Exclusive final name; never replace old evidence.
    temporary.unlink()
    return dict(file=path.name, bytes=path.stat().st_size, sha256=c.sha(path))


def capture_wrapper(out, axis, wrapper, low, cfg, fingerprint):
    import numpy as np
    from qiskit import qpy
    folder = Path(out)/'quarantine'/axis
    folder.mkdir(parents=True, exist_ok=False)
    circuit = atomic_binary(folder/'wrapper.qpy', lambda f: qpy.dump(wrapper, f))
    state = atomic_binary(folder/'initial.npz', lambda f: np.savez(f, low=low))
    c.save(folder/'wrapper_receipt.json', dict(status='UNVALIDATED_NOT_FOR_ACCOUNTING',
        SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], axis=axis,
        wrapper_fingerprint=fingerprint, global_phase=float(wrapper.global_phase),
        circuit=circuit, state=state))


def capture_compiled(out, axis, cost):
    from qiskit import qpy
    folder = Path(out)/'quarantine'/axis
    receipt = atomic_binary(folder/'compiled.qpy', lambda f: qpy.dump(cost.transpiled_circuit, f))
    c.save(folder/'compiled_receipt.json', dict(status='UNVALIDATED_NOT_FOR_ACCOUNTING',
        circuit=receipt, global_phase=float(cost.transpiled_circuit.global_phase),
        layout=repr(cost.transpiled_circuit.layout), compiler_settings_hash=cost.compiler_settings_hash,
        pretranspile_fingerprint=cost.actual_circuit_fingerprint,
        metrics={k:getattr(cost,k) for k in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}))


def capture_vectors(out, axis, low, before, after, expected, z):
    import numpy as np
    folder = Path(out)/'quarantine'/axis
    receipt = atomic_binary(folder/'vectors.npz', lambda f: np.savez(f,
        low=low, before=before, after=after, independent_expected_high=expected))
    c.save(folder/'vectors_receipt.json', dict(status='UNVALIDATED_NOT_FOR_ACCOUNTING',
        vectors=receipt, dtype=str(after.dtype), shape=list(after.shape)))
    if not all(np.isfinite(v).all() for v in (low,before,after,expected)):
        c.save(folder/'diagnostics.json', dict(status='STOP_NONFINITE_VECTOR', acceptance_override=False))
        raise ValueError('NONFINITE_DIAGNOSTIC_VECTOR')
    half = len(low)//2
    def observable(v):
        return float(np.vdot(v[:half],v[:half]).real-np.vdot(v[half:],v[half:]).real)
    overlap = complex(np.vdot(before, after))
    phase = float(np.angle(overlap)) if overlap != 0 else None
    raw = float(np.linalg.norm(after-before))
    aligned = float(np.linalg.norm(np.exp(-1j*phase)*after-before)) if phase is not None else None
    c.save(folder/'diagnostics.json', dict(status='DIAGNOSTICS_ONLY_NO_ACCEPTANCE_OVERRIDE',
        raw_error=raw, raw_gate_PASS=math.isfinite(raw) and raw<1e-9,
        before_norm=float(np.linalg.norm(before)), after_norm=float(np.linalg.norm(after)),
        overlap_real=overlap.real, overlap_imag=overlap.imag, common_phase=phase,
        phase_aligned_error_DIAGNOSTIC_ONLY=aligned,
        low_branch_error=float(np.linalg.norm(after[:half]-before[:half])),
        high_branch_error=float(np.linalg.norm(after[half:]-before[half:])),
        hadamard_before=observable(before), hadamard_after=observable(after),
        independent_expected_hadamard=float(z.real if axis=='cosine' else z.imag),
        acceptance_override=False, full_H_reference_phase_aligned=False))


def derive_task_ast():
    """Only origin lineage and three evidence hooks differ from frozen science."""
    tree = ast.parse((c.ROOT/'scripts/h46_m5_native.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_run_task')
    source = ast.unparse(fn)
    old = "ref['SOURCE'] == config['SOURCE']"
    c.require(source.count(old)==1, 'EXACT_REFERENCE_LINEAGE_SITE')
    source = source.replace(old, "ref['SOURCE'] == config['origin_SOURCE']")
    sites = [
        ("c.reserve(config, task, axis, fingerprint)", "capture_wrapper(out, axis, wrapper, low, config, fingerprint)"),
        ("'COMPILED_IDENTITY')", 'capture_compiled(out, axis, cost)'),
        ("after = simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False), low)",
         'capture_vectors(out, axis, low, before, after, expected, z)'),
    ]
    lines = source.splitlines()
    for needle, hook in sites:
        found = [i for i,line in enumerate(lines) if needle in line]
        c.require(len(found)==1, 'EXACT_EVIDENCE_HOOK_SITE:'+needle)
        i = found[0];indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
        lines.insert(i+1, indent+hook)
    return ast.parse('\n'.join(lines))


_cache = {}
_install = c.base.rebind(original, dict(c=c))['install']
_context = c.base.rebind(tol, dict(c=c, CACHE=_cache, install=_install))['context']
_namespace = c.base.rebind(original, dict(c=c, CACHE=_cache, context=_context))
_namespace.update(capture_wrapper=capture_wrapper, capture_compiled=capture_compiled, capture_vectors=capture_vectors)
exec(compile(derive_task_ast(), str(Path(__file__).resolve())+':frozen_task_with_evidence_hooks', 'exec'), _namespace)
_run_task = _namespace['_run_task']


def run_task(cfg, stop):
    c.authorize(cfg);c.consumed(cfg)
    _install(cfg)
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    with use_fingerprint_reuse() as cache:
        try:
            return _run_task(cfg, c.TASK, stop, cache.fingerprint)
        finally:
            out = Path(cfg['output'])/c.TASK['id']
            if out.exists():c.save(out/'fingerprint_cache.json', cache.snapshot())
