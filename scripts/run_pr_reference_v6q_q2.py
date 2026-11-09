"""Fixed four-fixture Q2 validation; no compile, sampling, pricing or RPE.

Scope: eaf70d88 Q2_VALIDATION_PLAN.md. Preserve candidate and original SOURCE.
Four ordered fixtures, one provider call each, old/new XY plus one Z witness.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import io
import json
import lzma
import math
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PREP = 'eaf70d886d42364a01b7e949ebc21f3b026bbe72'
BASE = 'f2638a7845685a119619382b850441f5d8d33713'
SELF = 'scripts/run_pr_reference_v6q_q2.py'
CANDIDATE = 'src/trottertracks/pr_reference_v6q.py'
MANIFEST = 'artifacts/pr_reference_v6q/2026-11-09/preparation/Q2_INPUTS.json'
TOL = 1e-9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git_bytes(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def literal(value):
    kind = type(value).__module__+'.'+type(value).__qualname__
    if isinstance(value, float):
        value_hex = float.__float__(value).hex()
    elif type(value) is int:
        value_hex = str(value)
    else:
        raise TypeError('Unsupported literal type in scientific fixture: '+kind)
    return dict(type=kind, value=value_hex)


def snapshot(qc):
    ops = []
    for inst in qc.data:
        ops.append(dict(name=inst.operation.name,
            wires=[qc.find_bit(x).index for x in inst.qubits],
            clbits=[qc.find_bit(x).index for x in inst.clbits],
            params=[literal(x) for x in inst.operation.params]))
    return dict(num_qubits=qc.num_qubits, global_phase=literal(qc.global_phase), ops=ops)


def digest(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def rational(value):
    if type(value) is int:
        return value, 1
    if isinstance(value, float):
        return float.as_integer_ratio(value)
    raise TypeError('Oracle does not coerce nonbinary literals')


def independent_sum(qc):
    # Integer common grid, independent of the provider's Fraction accumulation.
    terms = [rational(qc.global_phase)]
    for inst in qc.data:
        if inst.operation.name in ('rz', 'rzz'):
            n, d = rational(inst.operation.params[0])
            terms.append((-n, 2*d))
    denominator = max(d for n, d in terms)
    require(all(denominator % d == 0 for n, d in terms), 'Non-dyadic oracle input')
    return F(sum(n*(denominator//d) for n, d in terms), denominator)


def rounding_evidence(total, value):
    center = F.from_float(value)
    even = not (struct.unpack('>Q', struct.pack('>d', value))[0] & 1)
    left = (center + F.from_float(math.nextafter(value, -math.inf)))/2
    right = (center + F.from_float(math.nextafter(value, math.inf)))/2
    require(left < total < right or (even and total in (left, right)), 'Independent RN-even check')
    return dict(exact_sum=str(total), returned_hex=value.hex(),
        signed_rounding_residual=str(center-total), left_midpoint=str(left),
        right_midpoint=str(right), even_significand=even, passed=True)


def event_values(event):
    return dict(order=event.taylor_order, rotation=event.rotation_component_id,
        products=list(event.product_component_ids), angle=event.rotation_angle,
        phase=[event.phase.real, event.phase.imag], probability=event.event_probability,
        normalization=event.event_normalization)


def saved_event(rec):
    phase = rec['phase']
    if isinstance(phase, dict):
        phase = [phase['real'], phase['imag']]
    return dict(order=rec['order'] if 'order' in rec else rec['taylor_order'],
        rotation=rec['rotation'] if 'rotation' in rec else rec['rotation_component_id'],
        products=rec['products'] if 'products' in rec else rec['product_component_ids'],
        angle=rec['angle'] if 'angle' in rec else rec['rotation_angle'], phase=phase,
        probability=rec['probability'] if 'probability' in rec else rec['event_probability'],
        normalization=rec['normalization'] if 'normalization' in rec else rec['event_normalization'])


def core_difference(old, new, angle):
    a, b = snapshot(old), snapshot(new)
    require(a['num_qubits'] == b['num_qubits'] == 17, 'Reference register count')
    require(a['global_phase'] == b['global_phase'], 'Joint global phase changed')
    require(len(a['ops']) == len(b['ops']), 'Core operation count changed')
    hits = []
    for i, (x, y) in enumerate(zip(a['ops'], b['ops'])):
        if x['name'] == 'p' and x['wires'] == [16]:
            require({k:v for k,v in x.items() if k!='params'} ==
                    {k:v for k,v in y.items() if k!='params'}, 'Ancilla P shape changed')
            require(y['params'] == [literal(angle)], 'New P did not preserve angle bits')
            hits.append(i)
        else:
            require(x == y, 'A gate other than vacuum P changed')
    require(len(hits) == 1, 'Unique ancilla relative P required')
    require(sum(x['name']=='cswap' for x in a['ops']) == 16, 'CSWAP cascades changed')
    return dict(only_ancilla_P_may_differ=True, P_position=hits[0],
                old_P=a['ops'][hits[0]], new_P=b['ops'][hits[0]],
                source_shape_digest=digest([{k:v for k,v in x.items() if k!='params'} for x in b['ops']]))


def _xy_task(core, psi):
    sys.path.insert(0, str(ROOT/'src'))
    from trottertracks.pr_reference_interference_pilot import axis_signals
    return axis_signals(core, psi)


def complex_record(z):
    return dict(real=float(z.real), imag=float(z.imag))


def run_fixture(meta, record, psi, pool, all_literals, progress):
    import numpy as np
    import mpmath as mp
    from qiskit.quantum_info import Statevector, Operator
    from trottertracks import pr_reference_bprime as bprime
    from trottertracks import pr_reference_transfer_qualification as transfer
    from trottertracks import pr_reference_interference_pilot as old
    from trottertracks import pr_reference_v6q as candidate
    start = time.monotonic()
    progress.update(stage='reconstruct', member=meta['member'])
    cfg, dist, events = bprime.reconstructed(record)
    progress.update(reconstructed_events=[event_values(e) for e in events],
                    saved_events=[saved_event(e) for e in record[meta['event_field']]])
    require([event_values(e) for e in events] == [saved_event(e) for e in record[meta['event_field']]],
            'Reconstructed events changed saved SOURCE values')
    parts = transfer.assemble(meta['q'], 1, cfg, dist, events)
    f, g = parts['F'], parts['G']
    before = [snapshot(f), snapshot(g)]
    progress.update(stage='provider', lane_inputs=before)
    sf, sg = independent_sum(f), independent_sum(g)
    phase = candidate.exact_binary_phase(f, g)  # Exactly one provider call/fixture.
    rn = rounding_evidence(sf+sg, phase)
    progress.update(phase_old=parts['phase'], phase_new=phase, rounding=rn)
    require(before == [snapshot(f), snapshot(g)], 'Provider mutated lanes')
    baseline, bevents, replacements = bprime.build_saved(record)
    require([event_values(e) for e in bevents] == [event_values(e) for e in events], 'Bprime event mismatch')
    z = old.coherence(baseline, psi)
    old_core = old.reference_core(f, g, parts['phase'])
    new_core = old.reference_core(f, g, phase)
    shape = core_difference(old_core, new_core, phase)
    jobs = [pool.submit(_xy_task, core, psi) for core in (old_core, new_core)]
    old_z, new_z = [job.result() for job in jobs]
    xy_errors = dict(old=float(abs(old_z-z)), new=float(abs(new_z-z)))
    progress.update(stage='XY', baseline=complex_record(z), old=complex_record(old_z),
                    new=complex_record(new_z), XY_errors=xy_errors)
    require(all(x < TOL for x in xy_errors.values()), 'SOURCE XY correctness threshold failed')
    vacuum = []
    progress.update(stage='vacuum', vacuum=vacuum)
    for label, qc, exact in (('F', f, sf), ('G', g, sg)):
        require(qc.num_qubits == 8, 'Only the fixed 8-qubit lanes are qualified')
        state = Statevector.from_int(0, 256).evolve(qc).data
        with mp.workprec(192):
            target = complex(mp.exp(1j*(mp.mpf(exact.numerator)/exact.denominator)))
        goal = np.zeros(256, complex); goal[0] = target
        leak, error = float(np.linalg.norm(state[1:])), float(np.linalg.norm(state-goal))
        vacuum.append(dict(lane=label, exact_vacuum_phase=str(exact), leakage=leak, action_error=error))
        require(max(leak, error) < TOL, 'Vacuum ray/action threshold failed')
    fm = Operator(f).data
    inverse_error = float(np.linalg.norm(Operator(f.inverse()).data-fm.conj().T))
    progress.update(stage='inverse', inverse_error=inverse_error)
    require(inverse_error < TOL, 'Actual SOURCE inverse threshold failed')
    negative = None
    if meta['q'] == 1 and not meta['rare']:
        wrong = new_core.copy(); wrong.z(16)
        wrong_z = pool.submit(_xy_task, wrong, psi).result()
        flipped = float(abs(wrong_z+new_z)); mismatch = float(abs(wrong_z-z))
        progress.update(stage='negative_witness', negative_error=flipped, negative_mismatch=mismatch)
        require(flipped < TOL and mismatch > TOL, 'Fixed negative witness failed/inconclusive')
        negative = dict(ancilla_Z_appended=True, XY=complex_record(wrong_z),
                        error_to_negative_new_z=flipped, mismatch_to_correct_z=mismatch, passed=True)
    require(before == [snapshot(f), snapshot(g)], 'Validation mutated lane SOURCE')
    identity = dict(q=meta['q'], r=1, rare=meta['rare'], j=0, seed=meta['seed'])
    all_literals.append(dict(identity=identity, F=before[0], G=before[1]))
    return dict(identity=identity, member=meta['member'], member_sha256=meta['member_sha256'],
        phase_old=parts['phase'], phase_new=phase, rounding=rn,
        literal_snapshot_digest=digest(before), literal_types=sorted({p['type'] for c in before for op in c['ops'] for p in op['params']}),
        provider_input_order_and_values_preserved=True, core_shape=shape,
        baseline_XY=complex_record(z), old_reference_XY=complex_record(old_z), new_reference_XY=complex_record(new_z),
        XY_errors=xy_errors, vacuum=vacuum, inverse_error=inverse_error,
        Bprime_replacement_count=len(replacements), negative_witness=negative,
        old_tolerance=TOL, passed=True, wall_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(Path(__file__).read_bytes() == git_bytes(args.source_commit, SELF), 'Driver SOURCE mismatch')
    require((ROOT/CANDIDATE).read_bytes() == git_bytes(PREP, CANDIDATE), 'Frozen provider changed')
    manifest_raw = git_bytes(PREP, MANIFEST); manifest = json.loads(manifest_raw)
    for path, blob in manifest['source_bindings'].items():
        require(subprocess.check_output(['git','rev-parse',BASE+':'+path],cwd=ROOT,text=True).strip()==blob,
                'Original SOURCE blob mismatch')
        require((ROOT/path).read_bytes() == git_bytes(BASE,path), 'Original local SOURCE changed')
    require(not args.output.exists(), 'Fresh campaign output required')
    args.output.mkdir(parents=True)
    (args.output/'.gitignore').write_text('/common.pkl\n')
    completed = []; current = {}
    try:
        # Tiny 2x2/4x4 statevector BLAS kernels are parallelized by independent
        # old/new processes, not by a large BLAS team per tiny matrix product.
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        sys.path.insert(0, str(ROOT/'src'))
        source = dict(execution_commit=args.source_commit, provider_commit=PREP, scientific_input_commit=BASE,
            driver_sha256=sha(Path(__file__).read_bytes()), provider_sha256=sha((ROOT/CANDIDATE).read_bytes()),
            input_manifest_sha256=sha(manifest_raw), manifest=manifest, python=sys.version,
            host=os.uname().nodename, affinity_cpus=len(os.sched_getaffinity(0)),
            memory_observation=Path('/proc/meminfo').read_text(),
            execution='ordered four fixtures; two independent XY processes; process-local single-thread tiny-kernel BLAS',
            compile_sampling_price_RPE_T_calls=0)
        save(args.output/'SOURCE.json', source)
        common = manifest['common_input']; packed=git_bytes(BASE, common['path'])
        require(sha(packed)==common['archive_sha256'], 'Common archive mismatch')
        raw=lzma.decompress(packed); require(sha(raw)==common['pickle_sha256'], 'Common pickle mismatch')
        cache=args.output/'common.pkl'; cache.write_bytes(raw)
        os.environ['TRACK_C_H4_COMMON_CACHE']=str(cache.resolve())
        from trottertracks.pr_reference_interference_pilot import ground, TOL as old_tol
        from importlib.metadata import version
        require(old_tol==TOL and version('qiskit')=='1.3.0', 'Fixed runtime/threshold mismatch')
        psi=ground()  # Only loads the fixed cache/state; never solves for a state.
        archive_cache={}; all_literals=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            for meta in sorted(manifest['rows'], key=lambda x:(x['q'],x['rare'])):
                require((meta['q'],meta['r'],meta['j']) in ((1,1,0),(4,1,0)), 'Fixture expansion')
                if meta['archive'] not in archive_cache:
                    archive_cache[meta['archive']]=git_bytes(BASE,meta['archive'])
                archive=archive_cache[meta['archive']]
                require(sha(archive)==meta['archive_sha256'], 'Archive mismatch')
                with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as tar:
                    record_raw=tar.extractfile(meta['member']).read()
                require(sha(record_raw)==meta['member_sha256'], 'Member mismatch')
                record=json.loads(record_raw)
                require((record['q'],record['rare'],record['j'],record['seed'])==
                        (meta['q'],meta['rare'],meta['j'],meta['seed']), 'Saved identity mismatch')
                record['r']=1  # Fixed documented pilot r1 schema adapter, not new sampling.
                current={}
                row=run_fixture(meta,record,psi,pool,all_literals,current)
                completed.append(row)
                save(args.output/f"q{meta['q']}_rare{int(meta['rare'])}.json",row)
                print('Q2 fixture PASS',meta['q'],meta['rare'],row['XY_errors'],flush=True)
        literal_raw=json.dumps(all_literals,sort_keys=True,indent=2).encode()
        (args.output/'LITERALS.json.xz').write_bytes(lzma.compress(literal_raw))
        result=dict(status='FOUR_SAVED_FIXTURES_PASS_ALL_SUPPORT_PENDING',rows=completed,
            candidate_scientific_calls=4, reference_XY_axis_evaluations=18,
            literal_archive_sha256=sha((args.output/'LITERALS.json.xz').read_bytes()),
            literal_raw_sha256=sha(literal_raw), compile_sampling_price_RPE_T_calls=0,
            full_support_or_V6P_cost_promoted=False,next_action='MANDATORY_STOP')
        save(args.output/'RESULTS.json',result)
        save(args.output/'STOP.json',dict(status=result['status'],completed_fixtures=4,
             Q3_started=False,next_action='MANDATORY_STOP'))
    except Exception as exc:
        if current:
            save(args.output/'PARTIAL.json', current)
        save(args.output/'STOP.json',dict(status='FAIL',completed_fixtures=len(completed),
             error_type=type(exc).__name__,error=str(exc),Q3_started=False,next_action='MANDATORY_STOP'))
        raise


if __name__ == '__main__':
    main()
