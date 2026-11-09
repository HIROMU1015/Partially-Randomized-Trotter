"""Fixed Q2: two saved q1/r1 paths, two arms, four literal readout angles.

No old campaign, sampling, CT path, or RPE accounting is called. Native-only
CT1 state action is reused by AST to avoid importing its synthesis machinery.
"""
from __future__ import annotations
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
from importlib.metadata import version
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator, Statevector
from . import pr_reference_feedback_q1 as q1
from .track_c_compiler_a import BASIS, PROFILE, compile_reference

ROOT = Path(__file__).resolve().parents[2]
BASE = 'f3705321515f35c08c702e377739b2c3e6e9df13'
OUT = ROOT/'artifacts/pr_reference_feedback_q2/2026-11-10/run02'
PILOT = 'artifacts/pr_reference_interference_pilot/2026-11-06/'
CT1 = 'artifacts/pr_reference_ct1/2026-11-07/'
TOL = 1e-10
ANGLES = (('X', '0x0.0p+0'), ('Y', '0x1.921fb54442d18p+0'),
          ('positive', '0x1.b4395810624ddp-3'), ('negative', '-0x1.7020c49ba5e35p-1'))
FIXTURES = (
    dict(seed=4216572573356262833,
         event_sha256='bc3dfe95d2e57df0d6dcdc323d5876e9053b9122725f918da46b8d8a20f3758b',
         vacuum_phase_hex='0x1.8fd112eb44b80p+2'),
    dict(seed=4841711708811669773,
         event_sha256='54873b01217f8347d33f424e6aee2e65c77a432d8e499fa5c95cefc1b25b97f5',
         vacuum_phase_hex='0x1.8b84c9c9f67a8p+1'))
INPUTS = {}
COUNTS = Counter()
CORES = {}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


@lru_cache(None)
def data(path):
    raw = subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT)
    INPUTS[path] = dict(commit=BASE, bytes=len(raw), sha256=sha(raw))
    return raw


def read(path):
    return json.loads(data(path))


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)), mode='r:xz') as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}


@lru_cache(None)
def helpers():
    source = data('src/trottertracks/pr_reference_ct1.py')
    wanted = {'sample_rows', 'instructions', 'matrices_for', 'apply_batch', 'evaluate'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    require({n.name for n in nodes} == wanted, 'CT1 helpers missing')
    for node in nodes:
        node.decorator_list = []  # Own runtime-only JIT, never write old SOURCE caches.
    ns = dict(np=np, F=F, Counter=Counter, archive=archive, PILOT=PILOT, json=json)
    def forbidden():
        raise RuntimeError('CT record/word path is forbidden in Q2')
    ns.update(selected_records=forbidden, word_matrices=forbidden)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<Q2 pinned CT1 native helpers>', 'exec'), ns)
    from numba import njit
    ns['apply_batch'] = njit(cache=False)(ns['apply_batch'])
    return ns


def spec(circuit):
    return dict(n=circuit.num_qubits, lowering=helpers()['instructions'](circuit),
                global_phase=float(circuit.global_phase))


def native_action(circuit, initial):
    # ct=False: selected_records/word_matrices are unreachable. CT accounting
    # placeholders returned by the old helper are discarded, not T prices.
    output, _ = helpers()['evaluate'](spec(circuit), initial[:, None], False)
    return output[:, 0]


def layout(circuit, arm):
    n = 9 if arm == 'Bprime' else 17
    require(circuit.num_qubits == n and circuit.num_clbits == 1, 'Register layout mismatch')
    resets = [i for i in circuit.data if i.operation.name == 'reset']
    expected = [] if arm == 'Bprime' else list(range(8, 16))
    require([circuit.find_bit(i.qubits[0]).index for i in resets] == expected, 'Reset wires mismatch')
    require(all(i.operation.name == 'reset' for i in circuit.data[:len(expected)]), 'Reset is not the prefix')
    measurements = [i for i in circuit.data if i.operation.name == 'measure']
    require(len(measurements) == 1 and circuit.data[-1].operation.name == 'measure', 'Measure order mismatch')
    inst = measurements[0]
    require(circuit.find_bit(inst.qubits[0]).index == n-1 and circuit.find_bit(inst.clbits[0]).index == 0,
            'Measure mapping mismatch')
    require(not (set(circuit.count_ops())-set(BASIS)-{'measure', 'reset', 'barrier'}), 'Non-native gate')
    return dict(ancilla=n-1, target=list(range(8)), reference=expected,
                reset_prefix=expected, measure=[n-1, 0], reset_skipped_by_action_only_on_initial_vacuum=True)


def price(circuit):
    from .pr_reference_transfer_qualification import frontier_depth
    ops = dict(circuit.count_ops())
    depth, _ = frontier_depth(circuit)
    require(depth == circuit.depth(), 'Independent frontier depth mismatch')
    return dict(rz=int(ops.get('rz', 0)), cx=int(ops.get('cx', 0)),
                depth=int(depth), size=int(circuit.size()), qubits=circuit.num_qubits)


def metric_vector(p):
    return [p[k] for k in ('rz', 'cx', 'depth', 'size')]


def phase_mean(vector):
    half = len(vector)//2
    return float(np.vdot(vector[:half], vector[:half]).real-np.vdot(vector[half:], vector[half:]).real)


def raw_mean(core_output, raw):
    # Apply the ACTUAL wrapper's two final unitary instructions to the reused
    # post-core state. No second evolution of the same expensive core.
    tail = QuantumCircuit(1)
    for inst in raw.data[-3:-1]:
        require(len(inst.qubits) == 1 and raw.find_bit(inst.qubits[0]).index == raw.num_qubits-1,
                'Non-ancilla readout tail')
        tail.append(inst.operation, [0])
    result = Operator(tail).data @ core_output.reshape(2, -1)
    return phase_mean(result.reshape(-1))


def initial_state(psi, width):
    state = np.zeros(1 << width, complex)
    state[:len(psi)] = psi
    return state


def trace_pass(**kwargs):
    dag = kwargs['dag']
    ancillary = []
    for node in dag.topological_op_nodes():
        if any(dag.find_bit(q).index == dag.num_qubits()-1 for q in node.qargs):
            ancillary.append(dict(gate=node.op.name,
                                  wires=[dag.find_bit(q).index for q in node.qargs],
                                  params=[float(p) for p in node.op.params]))
    return dict(pass_name=kwargs['pass_'].name(), ops={k:int(v) for k,v in dag.count_ops().items()},
                global_phase=float(dag.global_phase), ancilla_tail=ancillary[-12:])


def prepare():
    from . import pr_reference_interference_pilot as pilot
    from . import pr_reference_bprime as bp
    from . import pr_reference_transfer_qualification as transfer
    from .track_c_h4 import common
    pre = read(CT1+'PREDECLARED.json')
    require(pre['numerical_check_tolerance'] == TOL, 'Fixed validation tolerance mismatch')
    require(data('src/trottertracks/pr_reference_feedback_q1.py') == (ROOT/'src/trottertracks/pr_reference_feedback_q1.py').read_bytes(),
            'Q1 SOURCE changed')
    env = read(CT1+'ENVIRONMENT.json')
    require(version('qiskit') == '1.3.0' and env['profile'] == PROFILE, 'Compiler context mismatch')
    for path in ('track_c_compiler_a.py', 'pr_reference_bprime.py',
                 'pr_reference_transfer_qualification.py', 'pr_reference_interference_pilot.py'):
        p = 'src/trottertracks/'+path
        old = subprocess.check_output(['git', 'show', pre['base']+':'+p], cwd=ROOT)
        require(old == data(p) == (ROOT/p).read_bytes(), 'Compiler/core SOURCE changed: '+p)
    ctinputs = read(CT1+'INPUTS.json')
    compressed = data(PILOT+'common.pkl.xz')
    require(sha(compressed) == '85d7e21a41e770596b92327d909f38f0ae091c875e2dad3651abeb660f26ae16', 'Cache archive hash mismatch')
    cache = lzma.decompress(compressed)
    require(sha(cache) == ctinputs['common_pickle_sha256'] == '68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2', 'Cache hash mismatch')
    (OUT/'common.pkl').write_bytes(cache)
    os.environ['TRACK_C_H4_COMMON_CACHE'] = str(OUT/'common.pkl')
    require(common.cache_info().currsize == 0, 'Unexpected prior scientific cache in this process')
    c = common()
    COUNTS['cache_state_load'] += 1
    psi = pilot.ground()  # Existing endian mapping; no solve or normalization.
    indices = [int(f'{int(x):08b}'[::-1], 2) for x in c['sector'].basis_indices]
    require(np.array_equal(psi[indices], c['state']), 'Ground endian mismatch')
    require(np.count_nonzero(psi) == np.count_nonzero(c['state']), 'Unexpected prepared amplitudes')
    rows = helpers()['sample_rows']()
    for p, meta in ctinputs['inputs'].items():
        if p in INPUTS:
            require(INPUTS[p]['sha256'] == meta['sha256'], 'Saved input hash mismatch: '+p)
    checks = read(CT1+'NATIVE_INPUT_CHECKS.json')
    require(checks['passed'], 'Old native validation not passed')
    old_checks = {r['name']:r for r in checks['checks']}
    native_archive = archive(CT1+'native_qpy.tar.xz')
    baselines = {}
    for name, payload in native_archive.items():
        stem = Path(name).stem
        if stem not in old_checks:
            continue
        arm = 'Bprime' if '_Bprime_' in stem else 'Reference_B1'
        circuit = qpy.load(io.BytesIO(payload))[0]
        lay = layout(circuit, arm)
        p = price(circuit)
        require(metric_vector(p) == old_checks[stem]['metric'], 'Saved QPY/price mismatch: '+stem)
        baselines[stem] = dict(member=name, qpy_sha256=sha(payload), price=p, layout=lay,
                               old_validation=old_checks[stem])
    require(len(baselines) == 8, 'Missing saved baseline QPY')
    provenance = dict(cache_pickle_sha256=sha(cache), endian_mapping='pilot.ground: reverse each 8-bit sector index',
                      same_saved_state=True, norm_diagnostic=float(np.linalg.norm(psi)),
                      compiler_profile=PROFILE, CT1_tolerance=pre['numerical_check_tolerance'])
    for rare, row in enumerate(rows):
        fixed = FIXTURES[rare]
        event_hash = sha(json.dumps(row['event_records'], sort_keys=True).encode())
        require(row['q'] == row['r'] == 1 and row['j'] == 0 and int(row['rare']) == rare,
                'Fixture coordinates mismatch')
        require(row['seed'] == fixed['seed'] and event_hash == fixed['event_sha256'], 'Ordered event identity mismatch')
        require(row['reference_phase'].hex() == fixed['vacuum_phase_hex'], 'Saved vacuum phase mismatch')
        require(pre['trajectories'][rare]['event_sha256'] == event_hash, 'PREDECLARED event mismatch')
        old = read(PILOT+f'A2_H4_rare{rare}.json')
        require(old['all_passed'] and old['seed'] == row['seed'] and old['rare'] == row['rare']
                and old['reference_phase'] == row['reference_phase'], 'Saved expected binding mismatch')
        expected = complex(old['expected']['real'], old['expected']['imag'])
        body, events, changes = bp.build_saved(row)
        require(sha(json.dumps([e.to_dict() for e in events], sort_keys=True).encode()) == event_hash,
                'Reconstructed ordered event mismatch')
        cfg, dist, _ = bp.reconstructed(row)
        assembled = transfer.assemble(1, 1, cfg, dist, events)
        require(assembled['phase'] == row['reference_phase'], 'Reconstructed vacuum phase mismatch')
        ref = pilot.reference_core(assembled['F'], assembled['G'], assembled['phase'])
        for arm, core in (('Bprime', body), ('Reference_B1', ref)):
            COUNTS['raw_cores'] += 1
            state = initial_state(psi, core.num_qubits)
            half = len(state)//2
            coherent = state.copy()/math.sqrt(2)
            coherent[half:half+len(psi)] = psi/math.sqrt(2)
            output = Statevector(coherent).evolve(core).data
            COUNTS['raw_core_state_actions'] += 1
            z = 2*np.vdot(output[:half], output[half:])
            require(abs(z-expected) < TOL, 'Raw core/saved coherence mismatch')
            baseline_arm = 'Bprime' if arm == 'Bprime' else 'reference'
            base = {axis:baselines[f'rare{rare}_{baseline_arm}_{axis}'] for axis in 'XY'}
            if arm == 'Reference_B1':
                require(all(i.operation.name in ('cswap', 'rz', 'rzz', 'xx_plus_yy', 'z', 'p')
                            for i in core.data), 'Unexpected reference structure')
            CORES[(rare, arm)] = dict(core=core, core_output=output, initial=state, expected=expected,
                identity=dict(rare=rare, q=1, r=1, j=0, seed=row['seed'], event_sha256=event_hash,
                              vacuum_phase_hex=row['reference_phase'].hex()), baseline=base,
                checks=dict(raw_z=dict(real=float(z.real), imag=float(z.imag)),
                            saved_z=old['expected'], raw_coherence_error=float(abs(z-expected)),
                            saved_expected_origin_validation=dict(core='pilot existing_direct',
                                native_absolute_error=old['errors']['existing_direct']['native']),
                            Bprime_rewrites=changes if arm=='Bprime' else None,
                            actual_inverse_SOURCE='pilot.reference_core: F.inverse()',
                            scalar_and_order='unchanged existing build_saved/assemble', phase_fit=False))
    return provenance, baselines


def one(task):
    rare, arm, angle_name, angle_hex = task
    item = CORES[(rare, arm)]
    core, expected = item['core'], item['expected']
    angle = float.fromhex(angle_hex)
    feedback = q1.FeedbackAngle(angle, None, F(0), 0,
                              definition='fixed exact binary64 Q2 parameter; no m16 feedback claim')
    raw = q1.feedback_readout(core, feedback, family=arm)
    require(raw.data[0].operation.name == 'h' and raw.data[-3].operation.name == 'p'
            and float(raw.data[-3].operation.params[0]) == -angle
            and raw.data[-2].operation.name == 'h', 'Q1 wrapper order mismatch')
    require(list(raw.data[1:1+len(core.data)]) == list(core.data), 'Q1 changed core instructions')
    trace = []
    def callback(**kwargs):
        trace.append(trace_pass(**kwargs))
    native = compile_reference(raw, callback=callback)  # Exactly one per fixed context.
    if arm == 'Reference_B1':
        full = QuantumCircuit(17, 1)
        full.reset(range(8, 16))
        full.compose(native, inplace=True)
        native = full
    lay = layout(native, arm)
    p = price(native)
    observed = phase_mean(native_action(native, item['initial']))
    source = raw_mean(item['core_output'], raw)
    target = float((np.exp(-1j*angle)*expected).real)
    error = max(abs(observed-target), abs(source-target), abs(observed-source))
    result = dict(identity={**item['identity'], 'arm':arm, 'angle_name':angle_name, 'angle_hex':angle_hex},
                  expected=target, raw=source, native=observed,
                  raw_error=abs(source-target), native_error=abs(observed-target),
                  raw_native_difference=abs(observed-source), tolerance=TOL,
                  passed=bool(error < TOL), layout=lay, price=p, ops=dict(native.count_ops()),
                  baseline_differences={axis:{k:p[k]-base['price'][k] for k in p}
                                        for axis,base in item['baseline'].items()},
                  mathematical_XY_saved_coherence_component=(expected.real if angle_name=='X' else expected.imag) if angle_name in ('X','Y') else None,
                  binary_parameter_not_mathematical_pi_over_2=(angle_name=='Y'),
                  compiler_trace=trace, common_global_phase=float(native.global_phase),
                  relative_phase_check='direct saved-coherence comparison, no fitted phase',
                  uniform_feedback_error_qualified=False, T_evaluated=False)
    name=f'rare{rare}_{arm}_{angle_name}'
    with (OUT/'native'/f'{name}.qpy').open('wb') as stream:
        qpy.dump(native, stream)
    save(OUT/f'{name}.json', result)  # Preserve each completed context before STOP.
    return result


def run():
    require(not (OUT/'RESULTS.json').exists() and not (OUT/'STOP.json').exists(), 'Q2 output already used')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'native').mkdir(exist_ok=True)
    start = time.monotonic()
    source_paths = ['src/trottertracks/'+p for p in (
        'pr_reference_feedback_q1.py', 'pr_reference_feedback_q2.py', 'pr_reference_ct1.py',
        'pr_reference_bprime.py', 'pr_reference_interference_pilot.py',
        'pr_reference_transfer_qualification.py', 'track_c_compiler_a.py', 'track_c_h4.py')]
    scientific_source = {p:sha((ROOT/p).read_bytes()) for p in source_paths}
    environment = dict(host=platform.node(), python=sys.version, executable=sys.executable,
        packages={p:version(p) for p in ('qiskit','numpy','scipy','mpmath','numba')},
        affinity=len(os.sched_getaffinity(0)), load=os.getloadavg(),
        memory=Path('/proc/meminfo').read_text(), cpu_pressure=Path('/proc/pressure/cpu').read_text(),
        memory_pressure=Path('/proc/pressure/memory').read_text(), shared_settings_changed=False)
    results = []
    try:
        provenance, baselines = prepare()
        # Compile the unchanged state-action kernel once using empty artificial
        # instructions, then share its machine code with all forked workers.
        helpers()['apply_batch'](np.zeros((2,1),complex), np.zeros(0,np.int64),
            np.zeros(0,np.int64), np.zeros(0,np.int64), np.zeros((0,2,2),complex))
        tasks = [(rare,arm,name,angle) for rare in (0,1) for arm in ('Bprime','Reference_B1') for name,angle in ANGLES]
        workers = min(len(tasks), len(os.sched_getaffinity(0)))
        # fork shares the one loaded cache and four immutable core states.
        import multiprocessing as multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(one, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                print(json.dumps(dict(context=row['identity'], passed=row['passed'], price=row['price'])), flush=True)
                require(row['passed'], 'Fixed-context correctness failure')
        COUNTS.update(wrapper_builds=16, native_compiles=16, native_state_actions=16,
                      baseline_compiles=0, old_science_runners=0, sampling=0, RPE=0, CT=0)
        require(len(results) == 16, 'Incomplete fixed contexts')
        results.sort(key=lambda r:(r['identity']['rare'],r['identity']['arm'],r['identity']['angle_name']))
        save(OUT/'RESULTS.json', dict(status='FINITE_Q2_CORRECTNESS_PASS', base_commit=BASE,
             fixed_scope='q1/r1/j0 two saved paths, both arms, four binary64 angles',
             provenance=provenance, baseline=baselines, cores=[dict(identity=v['identity'],arm=arm,**v['checks']) for (_,arm),v in CORES.items()],
             rows=results, calls=dict(COUNTS), workers=workers,
             physical_conditions='Linear H4 1.00 A STO-3G DF12 Partial S2 LD3 h=.02 K2, exact original-H ground supplied, no state-prep price',
             m17_feedback_340_or_allround_cost_qualified=False, phase_fit=False))
        with tarfile.open(OUT/'native_qpy.tar.xz', 'w:xz') as tar:
            for path in sorted((OUT/'native').glob('*.qpy')):
                tar.add(path, arcname='native/'+path.name)
        stop = dict(status='MANDATORY_STOP', reason='Fixed 16 Q2 contexts complete', next_owner='GPT/user')
    except BaseException as exc:
        stop = dict(status='MANDATORY_STOP', reason='Fixed Q2 mismatch or technical failure',
                    error=repr(exc), next_owner='GPT/user', completed_rows=len(results))
        raise
    finally:
        save(OUT/'SOURCE.json', dict(base_commit=BASE, inputs=INPUTS, source_sha256=scientific_source,
             environment=environment, elapsed_seconds=time.monotonic()-start,
             helper_reuse='CT1 pinned AST sample_rows/instructions/matrices_for(ct=False)/apply_batch/evaluate; runtime JIT cache disabled',
             old_SOURCE_raw_results_STOP_changed=False, new_approval_machinery=False,
             prohibited_calls=dict(new_SCF_DF_ground=0, sampling=0, m17_build_compile=0, Q0_rerun=0,
                 strict_guard_or_synthesis=0, RPE_or_allround_cost=0, T_prices=0)))
        save(OUT/'STOP.json', stop)


if __name__ == '__main__':
    run()
