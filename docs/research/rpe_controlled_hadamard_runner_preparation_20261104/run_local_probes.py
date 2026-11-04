"""Default: static byte/schema checks. --run needs separate user permission.

No approval-proof mechanism: --run is an explicit execution switch, not proof of
permission. This preparation task must never call it with real H4 inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT/'docs/research/rpe_controlled_hadamard_preparation_20261104'
STATUS = {'RPE_tolerance_certified': False, 'scientific_status': 'UNDETERMINED',
          'U_exec_upper': None, 'mandatory_STOP': True}
H_HASH = '56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5'
STATE_FP = 'a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90'
PREP_HASH = '89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919'
TAIL_HASH = '65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af'
PARENTS = ((0, 1), (8, 2), (10, 4), (11, 8), (13, 16), (17, 32))
CAPS = {'hamiltonian_loads': 1, 'state_loads': 1, 'preparations': 1, 'support_registrations': 1,
        'config_replays': 6, 'reference_half_pairs': 1, 'reference_components': 2,
        'reference_trajectories': 12, 'uncontrolled_builds': 12, 'uncontrolled_actions': 12,
        'controlled_raw_concatenation_builds': 12, 'controlled_boundary_optimized_builds': 12,
        'controlled_branch_actions': 48, 'wrapper_builds': 24, 'ideal_mean_actions': 24}


def load_definition(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def static_inputs():
    # Existing literal/byte validator does not deserialize numeric arrays or run science.
    plan = json.loads((HERE/'source_inputs.json').read_text())
    for item in plan['additional_fixed_sources']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('UNKNOWN/STOP: fixed source differs: '+item['path'])
    bindings = json.loads((PRIOR/'SOURCE_BINDINGS.json').read_text())
    for item in bindings['production_sources']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('UNKNOWN/STOP: production source differs: '+item['path'])
    previous = load_definition('local_probe_static_metadata', ROOT/'docs/research/rpe_18round_ideal_enclosure_preparation_20261104/run_round_enclosures.py')
    manifest, rows, raw = previous.metadata()
    selected = []
    for m, r in PARENTS:
        row = rows[m]
        if (row['m'], row['q'], row['r'], row['K'], row['delta']) != (m, 1 << m, r, 2, '0.02'):
            raise ValueError('UNKNOWN/STOP: parent tuple mismatch')
        selected.append(row)
    return manifest, selected, raw


def describe():
    _, parents, _ = static_inputs()
    return {**STATUS, 'status': 'SOURCE_PREPARED_NOT_SCIENTIFICALLY_EXECUTED',
        'parents': parents, 'probes': 12, 'q_probe_values': [1, 2], 'precision_bits': 96,
        'planned_calls': CAPS, 'new_H4_scientific_calls': 0,
        'no_sampling_compile_or_long_q_build': True}


def frozen_source():
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for name in ('run_local_probes.py', 'probe_adapter.py', 'reference_adapter.py', 'source_inputs.json', 'EXECUTION_SCOPE.md'):
        path = HERE/name
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show', head+':'+path.relative_to(ROOT).as_posix()])
        if old != path.read_bytes():
            raise ValueError('UNKNOWN/STOP: commit scientific SOURCE before execution')
    required = {'numpy': '1.26.4', 'scipy': '1.14.1', 'qiskit': '1.3.0', 'python-flint': '0.9.0'}
    actual = {p: importlib.metadata.version(p) for p in required}
    if actual != required or sys.version_info[:2] != (3, 11):
        raise ValueError('UNKNOWN/STOP: fixed numerical runtime required')
    import flint
    import ctypes
    if (flint.__FLINT_VERSION__ != '3.6.0' or sys.float_info.radix != 2 or sys.float_info.mant_dig != 53
            or ctypes.CDLL(None).fegetround() != 0):
        raise ValueError('UNKNOWN/STOP: FLINT/binary64 round-to-nearest runtime')
    actual['FLINT'] = flint.__FLINT_VERSION__
    return head, actual


def resource_context():
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    memory = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            memory = int(line.split()[1])*1024
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5)
        gpu_status = gpu.stdout.strip() if gpu.returncode == 0 else 'driver unavailable'
    except (OSError, subprocess.TimeoutExpired):
        gpu_status = 'not available'
    return {'available_cpu_affinity': available, 'available_ram_bytes': memory,
        'load_average': list(os.getloadavg()), 'gpu': gpu_status,
        'workers': min(6, available), 'worker_BLAS_threads': 1, 'reference_FLINT_threads': 1,
        'reason': 'Six independent r groups; 512-amplitude local actions dominate. Basis/reference caches are shared by payload. Small 70x70 analytic matrices do not justify FLINT thread teams.',
        'gpu_used': False, 'gpu_reason': 'Existing Qiskit Statevector CPU path for small 512-amplitude local probes; no large dense evolution.'}


def verify_preparation(prep, export):
    if (prep.preparation_hash != PREP_HASH or prep.tail_extraction.tail_hash != TAIL_HASH
            or prep.coefficient_atol != 1e-12 or prep.identity_policy != 'extract_identity_phase'
            or prep.ld != 3 or prep.num_system_qubits != 8
            or len(prep.rte_preparation.symbolic_tail.components) != 218 or len(prep.deterministic_blocks) != 4
            or list(prep.deterministic_fragment_indices) != export['deterministic_fragment_indices']
            or list(prep.randomized_block_indices) != export['randomized_fragment_indices']):
        raise ValueError('UNKNOWN/STOP: fixed preparation identity')


def construct_context(manifest, raw, record):
    """Real H4 entry, called only after explicit --run, never during preparation."""
    import numpy as np
    from trotterlib.df_hamiltonian import PhysicalSector
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    from trotterlib.finite_rte_signal_validation import _array_fingerprint, _qiskit_to_openfermion_sector_permutation
    from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot
    from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
    export = json.loads(raw['EXPORT'])
    record['counts_started']['hamiltonian_loads'] = 1
    h = load_connected_cluster_hamiltonian_snapshot(ROOT/manifest['files']['H']['path'])
    if df_hamiltonian_hash(h) != H_HASH or h.n_qubits != 8 or h.n_blocks != 12:
        raise ValueError('UNKNOWN/STOP: Hamiltonian identity')
    record['counts_started']['state_loads'] = 1
    with np.load(ROOT/manifest['files']['STATE']['path'], allow_pickle=False) as saved:
        psi = saved['state']
        sector = PhysicalSector.number_sector(n_qubits=8, n_electrons=4)
        if (not np.array_equal(saved['basis_indices'], sector.basis_indices)
                or str(saved['hamiltonian_hash']) != H_HASH or _array_fingerprint(psi) != STATE_FP):
            raise ValueError('UNKNOWN/STOP: saved state identity')
    ordered = sector.basis_indices[_qiskit_to_openfermion_sector_permutation(sector)]
    if list(ordered) != export['ordered_sector']:
        raise ValueError('UNKNOWN/STOP: endian/sector order')
    if [[float(z.real).hex(), float(z.imag).hex()] for z in psi] != export['state']:
        raise ValueError('UNKNOWN/STOP: export/raw state bytes differ')
    record['counts_started']['preparations'] = 1
    prep = prepare_df_partial_s2(h, split_df_hamiltonian_by_ld(h, 3),
                                identity_policy='extract_identity_phase', coefficient_atol=1e-12)
    verify_preparation(prep, export)
    record['counts_started']['support_registrations'] = 1
    support, proofs = register_support_restricted_bases(h, prep)
    if not all(np.isfinite(p['preserved_columns_max_abs_residual']) and p['preserved_columns_max_abs_residual'] <= 1e-10 for p in proofs):
        raise ValueError('UNKNOWN/STOP: support preserved-column residual')
    record['support_proofs'] = proofs
    return prep, support, export


def prepare_jobs(prep, support, export, parents, record, *, indices=None, parameter_path=None):
    """Generic supplied-input core; tests pass artificial 2-mode context."""
    from flint import ctx
    import probe_adapter as adapter
    import reference_adapter as ref
    ctx.prec, ctx.threads = 96, 1
    indices = adapter.select_fixed_components(prep, export) if indices is None else indices
    staged, parameters = [], []
    for row in parents:
        record['counts_started']['config_replays'] = record['counts_started'].get('config_replays', 0)+1
        config, law = adapter.signal_config(prep, row)
        profiles = adapter.profiles(prep.rte_preparation.symbolic_tail.components, law, row['r'], indices)
        parameters.append({'parent_m': row['m'], 'r': row['r'], 'signal_tau_hex': config.dimensionless_step_time.hex(),
            'model_tau_literal_not_used': row.get('model_tau64'),
            'order_angles_hex': {str(e.taylor_order): e.rotation_angle.hex() for _, _, events in profiles for e in events}})
        staged.append((row, config, law, profiles))
    record['parameter_replay'] = parameters
    # Parameters are fixed before reference matrices/actions and circuit builds.
    if parameter_path is not None:
        parameter_path.write_text(json.dumps(parameters, indent=2)+'\n')
    record['counts_started']['reference_half_pairs'] = 1
    record['counts_started']['reference_components'] = len(indices)
    reference = ref.build_reference(prep, export, indices, float(parents[0]['delta']))
    psi, psi_error = ref.to_float_vector(reference['state'], reference)
    record['reference'] = {'norm2': reference['norm2'], 'initial_conversion_error': psi_error,
        'basis_count': reference['basis_count'], 'selected_ids': [prep.rte_preparation.symbolic_tail.components[i].component_id for i in indices],
        'definition': '96-bit analytic occupation energy and exact binary64 basis/angle parameters; normalized saved ray'}
    jobs = []
    for row, config, law, profiles in staged:
        probes = []
        for q, profile, events in profiles:
            record['counts_started']['reference_trajectories'] = record['counts_started'].get('reference_trajectories', 0)+1
            values = ref.trajectory_reference(reference, events, q, row['r'])
            probes.append({'id': f"r{row['r']}_{profile}", 'q_probe': q, 'profile': profile, 'events': events, **values})
        jobs.append({'parent': row, 'config': config, 'law': law, 'psi': psi, 'probes': probes})
    return jobs


def collect_counts(record):
    counts = dict(record['counts_started'])
    for group in record.get('groups', []):
        for key, value in group['counts_started'].items():
            counts[key] = counts.get(key, 0)+value
    if any(key not in CAPS or value > CAPS[key] for key, value in counts.items()):
        raise ValueError('UNKNOWN/STOP: scientific call scope exceeded')
    record['total_counts_started'] = counts
    return counts


@contextmanager
def worker_environment():
    # Set before spawn/unpickling can import NumPy; restore this process afterward.
    keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = '1'
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(output):
    import numpy as np
    import probe_adapter as adapter
    output.mkdir(parents=True, exist_ok=False)
    record = {**STATUS, 'status': 'STARTED', 'counts_started': {}, 'groups': [], 'forbidden_calls': {
        key: 0 for key in ('new_H_or_state_solve', 'sampling', 'CDF', 'compile', 'shots', 'long_q_circuit', 'cost_or_success_reanalysis')}}
    started = time.monotonic()
    try:
        manifest, parents, raw = static_inputs()
        source, versions = frozen_source()
        record.update(source_commit=source, versions=versions, resources=resource_context())
        prep, support, export = construct_context(manifest, raw, record)
        jobs = prepare_jobs(prep, support, export, parents, record, parameter_path=output/'parameters.json')
        context = multiprocessing.get_context('spawn')
        stop = context.Event()
        with worker_environment(), ProcessPoolExecutor(max_workers=record['resources']['workers'], mp_context=context,
                initializer=adapter.initialize_worker, initargs=(prep, support, stop)) as pool:
            futures = {pool.submit(adapter.evaluate_group, job): job['parent']['r'] for job in jobs}
            for future in as_completed(futures):
                try:
                    group = future.result()
                except Exception as exc:
                    stop.set()
                    group = {'r': futures[future], 'pass': False, 'probes': [], 'counts_started': {},
                        'counts_unknown': True, 'error': type(exc).__name__+': '+str(exc), '_arrays': {}}
                arrays = group.pop('_arrays')
                path = output/f"r{group['r']}_vectors.npz"
                np.savez(path, **arrays)
                group['vectors'] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                record['groups'].append(group)
                if not group['pass']:
                    stop.set()
                collect_counts(record)
        counts = collect_counts(record)
        passed = (len(record['groups']) == 6 and all(g['pass'] for g in record['groups'])
                  and counts == CAPS and sum(len(g['probes']) for g in record['groups']) == 12)
        record['status'] = 'LOCAL_TWELVE_PROBES_CONSISTENT_MANDATORY_STOP' if passed else 'UNKNOWN_OR_INCONSISTENT_MANDATORY_STOP'
    except Exception as exc:
        record['status'] = 'STOPPED_WITH_ERROR'
        record['error'] = type(exc).__name__+': '+str(exc)
        collect_counts(record)
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        (output/'RESULT.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Real H4 execution; requires separate user permission')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(), indent=2)); return 0
    if args.output_dir is None:
        parser.error('--run requires a fresh --output-dir')
    sys.path[:0] = [str(HERE), str(ROOT/'src')]
    result = run(args.output_dir)
    print(json.dumps({k: result[k] for k in ('status', 'scientific_status', 'RPE_tolerance_certified')}))
    return 0 if result['status'] == 'LOCAL_TWELVE_PROBES_CONSISTENT_MANDATORY_STOP' else 1


if __name__ == '__main__':
    raise SystemExit(main())
