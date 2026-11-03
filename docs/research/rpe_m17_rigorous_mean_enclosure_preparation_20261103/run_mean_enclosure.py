"""Default: fixed bytes/headers only. --run is separately permitted H4 science."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_b1_construction_precision_preparation_20261103'))
import run_b1_construction as prior

SOURCE_FILES = ['ball_adapter.py', 'mean_adapter.py', 'export_adapter.py', 'run_mean_enclosure.py', 'requirements.txt']
VERSIONS = {'python-flint': '0.9.0', 'numpy': '1.26.4', 'scipy': '1.14.1', 'qiskit': '1.3.0',
            'qiskit-nature': '0.7.2', 'openfermion': '1.6.1', 'mpmath': '1.3.0'}
PRECISIONS = (96, 160, 224)
B1_RESULT = 'artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/RESULT.json'
B1_SHA = 'd392c2430882ca239e1e4ed5f8095dc912b5ef3597313b98cf0bf51e5ea7415e'
STATUS = {'original_numeric_diagnostic_status': 'INCONSISTENT', 'old_u_num_bound': None,
          'RPE_tolerance_certified': False, 'scientific_status': 'UNDETERMINED', 'mandatory_STOP': True}


def fixed_metadata():
    meta = prior.fixed_metadata()
    inputs = meta['inputs']
    snapshot = 'artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz'
    extras = [('4591b8d75a2a6a55c74a51225f0c61648023c8ee', snapshot, '13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b'),
              ('f766a1f7fc0cd9131a21d0594da242865fe1cb0d', B1_RESULT, B1_SHA)]
    for commit, path, sha in extras:
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha or raw != prior.task_a.original_runner.fixed_bytes(commit, path):
            raise ValueError('Fixed input mismatch: '+path)
        inputs.append({'commit': commit, 'path': path, 'sha256': sha})
    # Frozen helpers: source identity for numerical reproducibility, never permission proof.
    helper_sources = [(HERE.parent/'rpe_m17_b1_construction_precision_preparation_20261103'/name,
                       '9d27f931d6a572210eb206569806f96a46c2d094') for name in ['b1_adapter.py','run_b1_construction.py']]
    for path, commit in helper_sources:
        if path.read_bytes() != prior.task_a.original_runner.fixed_bytes(commit, path.relative_to(ROOT).as_posix()):
            raise ValueError('Frozen helper changed')
    for name in ['df_rte_qiskit.py', 'rpe_hadamard_interrogation.py']:
        path = 'src/trotterlib/'+name
        if (ROOT/path).read_bytes() != prior.task_a.original_runner.fixed_bytes('4591b8d75a2a6a55c74a51225f0c61648023c8ee',path):
            raise ValueError('Fixed gate/measurement semantics changed')
    meta['b1_result'] = json.loads((ROOT/B1_RESULT).read_text())
    return meta


def describe():
    meta = fixed_metadata()
    return {'status': 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED', **STATUS, 'inputs': meta['inputs'],
            'scope': 'IID uniform-53 categorical law; exact-parameter ideal uncompiled gates; normalized saved-state ray',
            'precision_bits': PRECISIONS, 'max_complex_enclosures': 3, 'real_numeric_calls': 0,
            'backend_installed': importlib.metadata.version('python-flint') if importlib.util.find_spec('flint') else None}


def source_metadata():
    hashes = {}
    for name in SOURCE_FILES:
        path = HERE/name
        if path.read_bytes() != prior.task_a.original_runner.fixed_bytes('HEAD', path.relative_to(ROOT).as_posix()):
            raise ValueError('Freeze this numerical SOURCE before --run')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError('Pinned numerical dependency versions required')
    import flint
    if flint.__FLINT_VERSION__ != '3.6.0':
        raise ValueError('Pinned FLINT3.6.0 required')
    return {'commit': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
            'code_sha256': hashes, 'versions': versions, 'FLINT': flint.__FLINT_VERSION__}


def classify_both(value, literals, ball):
    outputs = {}
    for axis, coordinate in [('cosine','real'), ('sine','imag')]:
        saved = literals['axes'][axis]
        outputs[axis] = ball.classify_interval(getattr(value, coordinate), float(literals['saved_mean'][coordinate]),
            int(saved['N']), saved['alpha'], saved['epsilon'], bridge=None)
    statuses = [x['categorical_decision'] for x in outputs.values()]
    decision = ('CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED' if all(s == 'SUFFICIENT_CONDITION_CONFIRMED' for s in statuses)
                else 'SUFFICIENT_CONDITION_EXCLUDED' if any(s == 'SUFFICIENT_CONDITION_EXCLUDED' for s in statuses)
                else 'UNDETERMINED')
    return outputs, decision


def evaluate(export, component_law, order_law, angles, literals, output, record):
    import ball_adapter as ball
    import mean_adapter as mean
    from flint import ctx
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    ctx.threads = cpus  # One matrix trajectory; FLINT shares its own worker pool.
    record['resources'] = {'available_CPUs': cpus, 'FLINT_threads': cpus,
        'BLAS_environment': {n: os.environ.get(n) for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
    with zipfile.ZipFile(output/'enclosure_operators.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for bits in PRECISIONS:
            ctx.prec = bits
            started = time.monotonic()
            record['scientific_calls']['complex_enclosure_started'] += 1
            record['stage'] = 'enclosure_'+str(bits)
            prior.task_a.original_runner.write_result(output, record)
            value, operators, state_info = mean.enclose(export, component_law, order_law,
                q=int(literals['q']), r=int(literals['r']), orders=[0,2], angles=angles)
            phase_bound = mean.builder_phase_bound(int(literals['q']), int(literals['r']))
            # nu_cat includes the fixed builder's pi64 phase accumulation.
            # The auxiliary signed-involution mean uses exact +/-1 event phases.
            logical = {'real': ball.endpoints(value.real), 'imag': ball.endpoints(value.imag)}
            from flint import acb, arb
            value += acb(arb(0,phase_bound), arb(0,phase_bound))
            axes, decision = classify_both(value, literals, ball)
            archive.writestr(str(bits)+'.json', json.dumps({k: ball.encode_matrix_ball(v) for k,v in operators.items()}))
            item = {'bits': bits, 'axes': axes, 'categorical_decision': decision, 'state': state_info,
                    'wall_seconds': time.monotonic()-started, 'method': 'Acb exact centers + global spectral radius; binary power',
                    'logical_signed_mean_interval': logical, 'builder_phase_transfer_upper': str(ball.dyadic(phase_bound)),
                    'rigorous_for_declared_scope': True, 'precision_convergence_is_error_bound': False}
            record['enclosures'].append(item)
            prior.task_a.original_runner.write_result(output, record)
            if decision != 'UNDETERMINED':
                break
    record['categorical_decision'] = record['enclosures'][-1]['categorical_decision']


def execute(output):
    source, meta = source_metadata(), fixed_metadata()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'execution_status': 'STARTED', **STATUS, 'SOURCE': source, 'inputs': meta['inputs'],
              'stage': 'SOURCE_replay', 'enclosures': [], 'scientific_calls': {'component_basis_replay_started': 0,
              'CDF_component': 0, 'CDF_order': 0, 'complex_enclosure_started': 0},
              'bridges': {'fixed_PCG_stream_to_IID': None, 'virtual_float_clip_Bernoulli': None,
                          'compiled_circuit': None, 'noise_hardware': None, 'physical_state_preparation': None,
                          'historical_component_payload_identity': None},
              'SCF_DF_state_solve_sampling_shot_controlled_build_compile_cost': 0}
    try:
        prior.task_a.original_runner.write_result(output, record)
        import export_adapter as exporter
        import mean_adapter as mean
        record['scientific_calls']['component_basis_replay_started'] = 1
        exported = exporter.replay(ROOT, prior.task_a.original_runner)
        (output/'source_replay.json').write_text(json.dumps(exported, allow_nan=False, indent=2)+'\n')
        old_config = meta['old_result']['context']['config']
        if exported['lambda_r64'] != old_config['lambda_r']:
            raise ValueError('SOURCE replay lambda mismatch')
        record['stage'] = 'CDF'
        record['scientific_calls']['CDF_component'] = 1
        component_law = mean.categorical([c['probability64'] for c in exported['components']], component=True)
        record['scientific_calls']['CDF_order'] = 1
        order_law = mean.categorical(meta['old_result']['context']['distribution']['order_probabilities'])
        (output/'categorical_laws.json').write_text(json.dumps({'component': component_law,'order':order_law}, indent=2)+'\n')
        record['stage'] = 'enclosure'
        record['original_fixed_literals'] = meta['old_result']['fixed_literals']
        angles = meta['b1_result']['scalar_trace']['theta64']
        from fractions import Fraction
        import math
        if (len(angles) != 2 or any(not math.isfinite(2*x) or Fraction.from_float(2*x) != 2*Fraction.from_float(x) for x in angles)):
            raise ValueError('Exact finite doubling of fixed rotation angle required')
        record['fixed_theta64_hex'] = [x.hex() for x in angles]
        evaluate(exported, component_law, order_law, angles, record['original_fixed_literals'], output, record)
        record['scientific_calls']['complex_enclosure_started'] = len(record['enclosures'])
        record['output_sha256'] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                  for name in ['source_replay.json','categorical_laws.json','enclosure_operators.zip']}
        record['execution_status'], record['stage'] = 'COMPLETED_MANDATORY_STOP', 'completed'
    except BaseException as exc:
        record['execution_status'] = 'INTERRUPTED_OR_INVALID_MANDATORY_STOP'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc).replace(str(ROOT), '<repository>')}
        raise
    finally:
        record['wall_seconds'] = time.monotonic()-started
        prior.task_a.original_runner.write_result(output, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', action='store_true', help='Separately permitted scientific execution')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
    elif args.output_dir is None:
        parser.error('--run requires fresh --output-dir')
    else:
        execute(args.output_dir)


if __name__ == '__main__':
    main()
