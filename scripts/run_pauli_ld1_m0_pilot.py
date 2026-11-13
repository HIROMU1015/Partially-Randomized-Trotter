"""Pauli LD1/m0 pilot: native DF head + A's Pauli D2 tail, one normal path, X/Y.

Fixed target: A canonical LD1 (d905d219...), m0, T=0.1, qPF=1, R_outer=3, K2,
fresh normal path index0, at most 2 compiles, scalar c+s_R once at total T.

Reuse only: A P0 dictionary/operators (4220f300) and packet (714247619) via
`pauli_tail_price_connection`; existing sampler + long-time trial_seed family;
native DF head (`native_context`/`prepare(ham,1)`), `build_partial_native`,
measured X/Y wrapper and `compile_cached` (af284d85 / provider c31ef424); D2
(9c4006bf blobs, byte-checked) for the central-rotation control.

Two processes as for the Full-H pilot: the child calls the original install()
first; the Pauli modules absent from the provider root are registered from the
pinned 9c4006bf blobs. Index-0 gates compare the circuit with an independent
full-2^n oracle (saved P0 head blocks, A's reverse_bits, dense Pauli events).

Default: metadata preflight. `--execute-approved-pauli-b0` needs a separate
approval (sampling, native build, 2 compiles). No retry, mandatory STOP.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

CALLER = 'scripts/run_pauli_ld1_m0_pilot.py'
CONNECTION = 'src/trottertracks/pauli_tail_price_connection.py'
P0 = dict(commit='4220f3006d0d8d413e195ed948e250578513b557', directory='docs/research/df_head_pauli_tail_p0_results_20261114/run01/')
PAULI_MODULES = dict(commit='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c',
                     names=('trotterlib.pauli_signal_adapter', 'trotterlib.h4_d2_pauli_control'))
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
PROVIDER_MODULES = ('trotterlib.rte', 'trotterlib.df_trotter.circuit', 'trotterlib.rte_compiled_cost',
                    'trottertracks.resource_applicability.ax2b_native_df_v5',
                    'trottertracks.resource_applicability.h6_hash_reuse_v1')
HAMILTONIAN_CONSTANT = 2.29310124732
PREFIX, M = 1, 0
OUTPUT = 'docs/research/d_pauli_ld1_m0_pilot_results_20261114/run01'
PRIVATE_OUTPUT = Path('/home/AbeHiromu/projects/d-pauli-ld1-m0-20261114-private/run01')
SCOPE = dict(configuration='LD1', m=0, normal_paths=1, index=0, axes=['X', 'Y'], compile_maximum=2,
             new_reference=0, retry=False, mean=None, N=None, G=None)


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def connection(path=None):
    """This tree's connection module by file path (works in the provider-root child too)."""
    name = '_d_pauli_tail_price_connection'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path or (ROOT / CONNECTION))
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return sys.modules[name]


def register_pauli_modules():
    """Child: provider root lacks these two; load the pinned 9c4006bf blobs as trotterlib submodules."""
    import trotterlib
    for name in PAULI_MODULES['names']:
        if name in sys.modules:
            continue
        rel = 'src/' + name.replace('.', '/') + '.py'
        code = blob(PAULI_MODULES['commit'], rel)
        if code != (ROOT / rel).read_bytes():
            raise ValueError('Pinned Pauli module differs from this tree: ' + name)
        module = types.ModuleType(name); module.__package__ = 'trotterlib'; module.__file__ = str(ROOT / rel)
        sys.modules[name] = module; setattr(trotterlib, name.split('.')[-1], module)
        exec(compile(code, PAULI_MODULES['commit'] + ':' + rel, 'exec'), module.__dict__)
    return True


def provider_checks():
    for name in PROVIDER_MODULES:
        module = importlib.import_module(name)
        rel = 'src/' + name.replace('.', '/') + '.py'
        if Path(module.__file__).read_bytes() != blob(PROVIDER, rel):
            raise ValueError('Provider SOURCE mismatch: ' + name)
    return True


def p0_arrays():
    """A P0 saved full-2^n operators (saved mode-0-MSB basis) from the pinned commit."""
    import numpy as np
    common = np.load(io.BytesIO(blob(P0['commit'], P0['directory'] + 'COMMON_OPERATORS.npz')))
    ld1 = np.load(io.BytesIO(blob(P0['commit'], P0['directory'] + 'LD1_OPERATORS.npz')))
    return {k: ld1[k] for k in ld1.files}, common['blocks'], common['sector_basis_indices']


def pilot_core(native, ctx, dictionary, request, operators, blocks_saved, *, cohort, compile_cached):
    """Shared by the child and the artificial tests (toy ctx/dictionary/operators)."""
    import numpy as np
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib.df_trotter.circuit import simulate_statevector
    P = connection()
    prep = ctx.preparations[request['prefix']]
    if (len(prep.deterministic_blocks) != request['prefix'] + 1
            or tuple(prep.deterministic_fragment_indices) != tuple(range(request['prefix']))):
        raise ValueError('Native DF head must be one-body + fragments 0..L_D-1 in saved order')
    if prep.extracted_identity_coefficient != dictionary['saved_DF_scalar']:
        raise ValueError('Saved s_R differs from the native DF tail identity')
    gates = P.bit_order_checks(dictionary, operators, blocks_saved)
    if not gates['pass']:
        raise ValueError('P0 bit-order/label/head/tail gate failed: ' + json.dumps(gates))
    comps, labels, n = P.components(dictionary)
    if n != ctx.ham.n_qubits:
        raise ValueError('Pauli word width differs from the native system')
    scalar = P.total_scalar(dictionary, ctx.ham.constant); T = float(request['T'])
    steps, dist = P.sample_steps(comps, request, cohort, 0)
    expected = P.full_space_expected(list(blocks_saved[:request['prefix'] + 1]), steps, labels,
                                     scalar=scalar, T=T, state=ctx.qstate)
    prices = {}; compiled_checks = []
    with use_fingerprint_reuse() as fp:
        evolution = P.compose(prep.deterministic_blocks, n, steps, labels, T=T, scalar=scalar, fingerprint=fp.fingerprint)
        last = evolution.circuit.data[-1]
        once = [i for i in evolution.circuit.data if i.operation.name == 'p' and float(i.operation.params[0]) == -scalar * T]
        if (last.operation.name != 'p' or float(last.operation.params[0]) != -scalar * T or len(once) != 1):
            raise ValueError('Total scalar must be one final ancilla phase')
        checks, z, wrappers = P.index0_checks(evolution, ctx.qstate, expected)
        low = np.r_[ctx.qstate, np.zeros_like(ctx.qstate)]
        for axis, wrapper in wrappers.items():
            fingerprint = fp.fingerprint(wrapper)
            compiled, qc, reused = compile_cached(wrapper, fingerprint, True)
            prices[axis] = compiled['metrics']
            if qc is not None:
                before = simulate_statevector(wrapper.remove_final_measurements(inplace=False), low)
                after = simulate_statevector(qc.remove_final_measurements(inplace=False), low)
                error = float(np.linalg.norm(after - before))
                if error >= 1e-7:
                    raise ValueError('compiled_vector_' + axis + ':' + str(error))
                compiled_checks.append(dict(check='compiled_vector_' + axis, error=error, tolerance=1e-7))
    events = [[dict(selected=list(e.selected_component_ids), order=e.taylor_order) for e in s['events']] for s in steps]
    return dict(status='PASS', prices=prices, scalar=scalar, z=[z.real, z.imag], seeds=[s['seed'] for s in steps],
                checks=checks + compiled_checks, bit_order=gates, distribution=dict(orders=list(dist.orders),
                probabilities=list(dist.order_probabilities)), events=events, n=n, terms=len(comps))


def child(job_path):
    job = json.loads(Path(job_path).read_text())
    import h46_long_time_signal as signal
    if any(k.startswith('trottertracks.') for k in sys.modules):
        raise ValueError('Project trottertracks imported before the provider root')
    signal.install()
    provider_checks(); register_pauli_modules()
    P = connection(job['connection'])
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(job['public']), Path(job['private'])
    ctx = native.native_context('H4', (PREFIX,))
    if ctx.ham.constant != HAMILTONIAN_CONSTANT:
        raise ValueError('Hamiltonian constant differs from the bound c')
    plan = P.pilot_plan(ROOT, prefix=PREFIX, m=M, cohort=job['cohort'])
    handoff = json.loads(blob(P.PACKETS['commit'], P.PACKETS['path']))
    packet = next(p for p in handoff['packets'] if p['prefix'] == PREFIX)
    dictionary = P.load_dictionary(ROOT, packet['dictionary_identity'])
    operators, blocks, basis = p0_arrays()
    result = pilot_core(native, ctx, dictionary, plan['request'], operators, blocks,
                        cohort=job['cohort'], compile_cached=native.compile_cached)
    if result['seeds'] != plan['seeds']:
        raise ValueError('Seeds differ from the frozen plan')
    (Path(job['private']) / 'events_PRIVATE.json').write_text(json.dumps(result.pop('events')))
    Path(job['result']).write_text(json.dumps(result, indent=2, default=str))
    return 0


def _meta():
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    from trottertracks import baseline_signal_pipeline as C
    return C


def preflight(root=ROOT):
    C = _meta(); P = connection()
    for name in PAULI_MODULES['names']:
        rel = 'src/' + name.replace('.', '/') + '.py'
        if blob(PAULI_MODULES['commit'], rel) != (Path(root) / rel).read_bytes():
            raise ValueError('Pauli module differs: ' + name)
    for d in (P.NATIVE, P.WRAPPER):
        if blob(d['commit'], d['path']) != (Path(root) / d['path']).read_bytes():
            raise ValueError('Native SOURCE differs: ' + d['path'])
    plan = P.pilot_plan(root, prefix=PREFIX, m=M)
    c = P.legacy_physical_candidate(plan['request'], candidate_id=C.p.candidate_id)
    return dict(status='PREFLIGHT_METADATA_ONLY', configuration_id=plan['configuration_id'], canonical_candidate=plan['request']['candidate_id'],
                legacy_physical_candidate=c, seeds=plan['seeds'], cohort=plan['cohort'], scalar=plan['scalar'],
                dictionary_sha256=plan['dictionary_sha256'], scope=SCOPE, output=OUTPUT, private_output=str(PRIVATE_OUTPUT))


def records(root, result, *, source, evidence_commit, evidence_path):
    """Parent: price records + PR_PAULI_TAIL sidecars from a child result."""
    C = _meta(); P = connection()
    handoff = json.loads(blob(P.PACKETS['commit'], P.PACKETS['path']))
    packet = next(p for p in handoff['packets'] if p['prefix'] == PREFIX)
    plan = P.pilot_plan(root, prefix=PREFIX, m=M)
    c = P.legacy_physical_candidate(plan['request'], candidate_id=C.p.candidate_id)
    B = C.published_api(str(root)).B
    return P.price_records(packet['configuration'], M, trial_prices=result['prices'], scalar=result['scalar'],
        physical_candidate=c, native_source=dict(commit=source['commit'], path=CONNECTION), cohort=plan['cohort'],
        price_source=source, evidence_commit=evidence_commit, evidence_path=evidence_path, measurement=B.p.measurement)


def execute(root=ROOT):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (CALLER, CONNECTION):
        if blob(commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ from HEAD: ' + path)
    meta = preflight(root)
    out = Path(root) / OUTPUT; out.mkdir(parents=True, exist_ok=False); PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=False)
    source = dict(commit=commit, caller=CALLER, connection=CONNECTION, P0=P0, pauli_modules=PAULI_MODULES, provider=PROVIDER)
    (out / 'SOURCE.json').write_text(json.dumps(dict(source=source, scope=SCOPE, preflight=meta, python=sys.version,
        platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False), indent=2))
    job = dict(cohort=meta['cohort'], public=str(out / 'child_public'), private=str(PRIVATE_OUTPUT),
               result=str(PRIVATE_OUTPUT / 'child.json'), connection=str(Path(root) / CONNECTION))
    (PRIVATE_OUTPUT / 'job.json').write_text(json.dumps(job))
    start = time.perf_counter()
    code = subprocess.call([sys.executable, '-B', str(Path(root) / CALLER), '--child', str(PRIVATE_OUTPUT / 'job.json')], cwd=root)
    result = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else None
    status = 'PAULI_B0_WIRING_COMPLETE' if code == 0 and result and result['status'] == 'PASS' else 'STOP_PAULI_B0'
    if status == 'PAULI_B0_WIRING_COMPLETE':
        prices, sides, proofs = records(root, result, source=dict(source, record='RESULT.json'),
            evidence_commit='FILL_WITH_PUBLISHED_COMMIT', evidence_path=OUTPUT + '/IMPLEMENTATION_EVIDENCE.json')
        for name, value in (('PRICES.json', prices), ('SIDECARS.json', sides), ('IMPLEMENTATION_EVIDENCE.json', proofs)):
            (out / name).write_text(json.dumps(value, indent=2, default=str))
    (out / 'RESULT.json').write_text(json.dumps(dict(status=status, child_exit=code, result=result,
        elapsed_seconds=time.perf_counter() - start, mandatory_stop=True, retry_performed=False, mean=None, N=None, G=None),
        indent=2, default=str))
    print(json.dumps(dict(status=status)))
    return 0 if status == 'PAULI_B0_WIRING_COMPLETE' else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-approved-pauli-b0', action='store_true',
                        help='sample one LD1/m0 path, build and compile X/Y (separate approval)')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.execute_approved_pauli_b0:
        return execute(ROOT)
    print(json.dumps(preflight(ROOT), indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
