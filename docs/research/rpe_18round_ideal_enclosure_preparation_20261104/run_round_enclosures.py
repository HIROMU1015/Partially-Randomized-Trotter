"""Default: byte/schema checks only. --run is separately authorized H4 science."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATUS = {'RPE_tolerance_certified': False, 'scientific_status': 'UNDETERMINED',
          'U_virtual_IID_upper': None, 'U_exec_upper': None, 'mandatory_STOP': True}
_STOP = None


def initialize_worker(stop):
    global _STOP
    _STOP = stop


def literal_json(raw):
    return json.loads(raw, parse_float=str, parse_int=str)


def bind_rows(csv_text, grid, opt2):
    """Read saved literals, not derived scientific values. Select by explicit tuples."""
    selected = [x for x in csv.DictReader(io.StringIO(csv_text)) if x['ld'] == '3']
    models = opt2['coherent_reoptimization']['best_by_ld']['3']['best']
    if len(selected) != 18 or len(models['rounds']) != 18:
        raise ValueError('UNKNOWN/STOP: incomplete Partial schedule')
    output = []
    for m in range(18):
        matches = [x for x in selected if x['round_index'] == str(m)]
        if len(matches) != 1:
            raise ValueError('UNKNOWN/STOP: missing/duplicate round')
        row = matches[0]
        key = (str(1 << m), row['r_m'], row['K_m'])
        model = [x for x in models['rounds'] if x['round_index'] == str(m)]
        points = [(i, x) for i, x in enumerate(grid['points']) if
                  (x['q_m'], x['r_m'], x['K_m']) == key]
        if (len(model) != 1 or len(points) != 1 or row['q_m'] != key[0]
                or row['K_m'] != '2' or row['r_m'] not in ('1', '2', '4', '8', '16', '32')
                or row['delta_time'] != '0.02'
                or tuple(model[0][k] for k in ('q_m', 'r_m', 'K_m')) != key):
            raise ValueError('UNKNOWN/STOP: unsupported or inconsistent tuple')
        model = model[0]
        pi, point = points[0]
        states = [(i, s) for i, s in enumerate(point['state_results']) if s['state_label'] == 'physical_df_ground_state']
        if len(states) != 1:
            raise ValueError('UNKNOWN/STOP: physical state unavailable')
        si, state = states[0]
        mu = json.loads(row['saved_complex_json'])
        if mu != state['attenuated_event_mean_signal']:
            raise ValueError('UNKNOWN/STOP: saved mean identity mismatch')
        axes = {}
        for axis, a in json.loads(row['axes_results_json']).items():
            axes[axis] = {'N': a['saved_N'], 'alpha': a['saved_alpha'], 'epsilon': a['saved_epsilon_coordinate']}
            if (axes[axis]['N'] != model['axes'][axis]['shots']
                    or axes[axis]['alpha'] != model['axes'][axis]['alpha']
                    or axes[axis]['epsilon'] != model['epsilon_coordinate']):
                raise ValueError('UNKNOWN/STOP: fixed axis literals differ')
        if set(axes) != {'cosine', 'sine'}:
            raise ValueError('UNKNOWN/STOP: missing axis')
        output.append({'m': m, 'ld': 3, 'q': int(key[0]), 'r': int(key[1]), 'K': int(key[2]),
            'delta': row['delta_time'], 'mu0': mu, 'axes': axes,
            'beta': {kind: row['saved_beta_'+kind] for kind in ('PF', 'RTE', 'stat')},
            'signal_tau64': point['tau_m'], 'model_tau64': model['tau_m'],
            'signal_selector': f'/points/{pi}/state_results/{si}/attenuated_event_mean_signal',
            'model_selector': f'/coherent_reoptimization/best_by_ld/3/best/rounds/{m}'})
    return output


def metadata():
    manifest = json.loads((HERE/'fixed_inputs.json').read_text())
    raw = {}
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_LAZY_FETCH': '1'}
    for key, item in manifest['files'].items():
        content = (ROOT/item['path']).read_bytes()
        oid = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if oid != item['blob'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('UNKNOWN/STOP: fixed input changed: '+key)
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', item['commit']+':'+item['path']], env=env)
        if content != frozen:
            raise ValueError('UNKNOWN/STOP: fixed commit/path mismatch: '+key)
        raw[key] = content
    rows = bind_rows(raw['V2ROWS'].decode(), literal_json(raw['GRID']), literal_json(raw['OPT2']))
    if rows != manifest['rounds']:
        raise ValueError('UNKNOWN/STOP: round manifest differs from immutable saved values')
    result, export = json.loads(raw['M17RESULT']), json.loads(raw['EXPORT'])
    grid = literal_json(raw['GRID'])
    for saved in csv.DictReader(io.StringIO(raw['V2ROWS'].decode())):
        if saved['ld'] == '3' and (saved['hamiltonian_hash'] != grid['hamiltonian']['hamiltonian_hash']
                or saved['snapshot_sha256'] != manifest['files']['H']['sha256']
                or saved['saved_physical_state_fingerprint'] != grid['states']['physical_df_ground_state']['state_fingerprint']
                or saved['sector_n_electrons'] != '4'):
            raise ValueError('UNKNOWN/STOP: physical input identity differs between rounds')
    if (grid['request']['ld'] != '3' or grid['request']['delta_time'] != '0.02'
            or grid['request']['identity_policy'] != 'extract_identity_phase'
            or grid['request']['coefficient_atol'] != '1e-12'
            or grid['hamiltonian']['preparation_hash'] != export['preparation_hash']):
        raise ValueError('UNKNOWN/STOP: preparation scope differs')
    if (result['enclosures'][0]['bits'] != 96 or export['n_qubits'] != 8 or export['n_electrons'] != 4
            or len(export['ordered_sector']) != 70 or len(export['components']) != 218
            or len(export['bases']) != 11 or len(export['deterministic_blocks']) != 4
            or export['identity_policy'] != 'extract_identity_phase' or export['coefficient_atol'] != 1e-12
            or export['preparation_hash'] != manifest['preparation_hash']
            or export['tail_hash'] != manifest['tail_hash']):
        raise ValueError('UNKNOWN/STOP: shared export mismatch')
    literals = result['original_fixed_literals']
    if (literals['saved_mean'] != rows[17]['mu0'] or literals['axes'] != rows[17]['axes']
            or any(result['enclosures'][0]['axes'][a]['categorical_decision'] != 'SUFFICIENT_CONDITION_CONFIRMED'
                   for a in ('cosine', 'sine'))):
        raise ValueError('UNKNOWN/STOP: m17 result cannot be reused')
    # Only ZIP member names are inspected in metadata mode. No matrix/numeric array decode.
    with zipfile.ZipFile(io.BytesIO(raw['OPERATORS'])) as archive:
        if '96.json' not in archive.namelist():
            raise ValueError('UNKNOWN/STOP: saved 96-bit operators missing')
    return manifest, rows, raw


def describe():
    manifest, rows, _ = metadata()
    return {**STATUS, 'status': 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED', 'rounds': len(rows),
            'new_complex_enclosures': 17, 'reused_m17': manifest['files']['M17RESULT'],
            'precision_bits': 96, 'new_order_laws': 5, 'new_H4_numeric_calls': 0}


def frozen_source(manifest):
    names = ['round_adapter.py', 'run_round_enclosures.py', 'fixed_inputs.json',
             'ENCLOSURE_EXTENSION_PLAN.md', 'requirements.txt']
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_LAZY_FETCH': '1'}
    sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], env=env).decode().strip()
    for name in names:
        path = HERE/name
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', sha+':'+path.relative_to(ROOT).as_posix()], env=env)
        if path.read_bytes() != frozen:
            raise ValueError('UNKNOWN/STOP: freeze scientific SOURCE before --run')
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'python-flint')}
    if versions != {'numpy': '1.26.4', 'python-flint': '0.9.0'} or sys.version_info[:2] != (3, 11):
        raise ValueError('UNKNOWN/STOP: fixed runtime required')
    import flint
    if flint.__FLINT_VERSION__ != '3.6.0' or sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        raise ValueError('UNKNOWN/STOP: FLINT/binary64 mismatch')
    import ctypes
    if ctypes.CDLL(None).fegetround() != 0:
        raise ValueError('UNKNOWN/STOP: round-to-nearest required for scalar/CDF replay')
    return {'commit': sha, 'versions': versions, 'FLINT': flint.__FLINT_VERSION__,
            'files_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}}


def group_worker(rows, raw, threads, checkpoint=None):
    """One r per independent process: share its step, keep FLINT context process-local."""
    from flint import ctx
    import round_adapter as adapter
    if _STOP is not None and _STOP.is_set():
        return {'r': rows[0]['r'], 'rounds': [], 'stopped_before_law': True}
    progress = {'r': rows[0]['r'], 'rounds': [], 'started_m': [], 'stage': 'law_or_saved_step',
                'FLINT_threads': threads}
    def save_progress():
        if checkpoint is not None:
            temporary = checkpoint.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(progress, allow_nan=False, indent=2)+'\n')
            temporary.replace(checkpoint)
    save_progress()
    ctx.prec, ctx.threads = 96, threads
    import ctypes
    if ctypes.CDLL(None).fegetround() != 0:
        raise ValueError('UNKNOWN/STOP: worker rounding mode differs')
    result, export = json.loads(raw['M17RESULT']), json.loads(raw['EXPORT'])
    with zipfile.ZipFile(io.BytesIO(raw['OPERATORS'])) as archive:
        operators = json.loads(archive.read('96.json'))
    if rows[0]['r'] == 32:
        step = adapter.decode(operators['step'])
        trace = {'kind': 'REUSED_M17_STEP_LAW', 'new_CDF_calls': 0,
                 'theta64_hex': result['fixed_theta64_hex'], 'order_law': json.loads(raw['LAWS'])['order']}
    else:
        taus = {row['signal_tau64'] for row in rows}
        if len(taus) != 1:
            raise ValueError('UNKNOWN/STOP: inconsistent shared tau')
        orders, angles, law = adapter.replay_order_law(float(taus.pop()), 2)
        step = adapter.make_step(*(adapter.decode(operators[k]) for k in ('A', 'forward', 'reverse')),
            r=rows[0]['r'], orders=orders, angles=angles, law=law)
        trace = {'kind': 'NEW_SOURCE_REPLAY_NOT_HISTORICAL_PAYLOAD', 'new_CDF_calls': 1,
                 'theta64_hex': [x.hex() for x in angles], 'orders': orders, 'order_law': law}
    progress['law'] = trace
    save_progress()
    outputs = progress['rounds']
    for row in rows:
        if _STOP is not None and _STOP.is_set():
            break
        progress['stage'] = 'enclosure'
        progress['started_m'].append(row['m'])
        save_progress()
        item = adapter.evaluate_round(step, export['state'], result['enclosures'][0]['state']['saved_vector_norm_squared'], row)
        outputs.append(item)
        save_progress()
        if any(x['categorical_decision'] != 'SUFFICIENT_CONDITION_CONFIRMED' for x in item['axes'].values()):
            if _STOP is not None:
                _STOP.set()
            break  # no escalation/retry: return partial results, then STOP
    progress['stage'] = 'group_finished_or_stopped'
    save_progress()
    return progress


def execute(output):
    manifest, rows, raw = metadata()
    source = frozen_source(manifest)
    output.mkdir(parents=True, exist_ok=False)
    record = {**STATUS, 'SOURCE': source, 'inputs': manifest['files'], 'groups': [],
              'execution_status': 'STARTED', 'categorical_local_conditions': 'UNDETERMINED',
              'm17_recomputed': False, 'scientific_call_scope': 'five order laws; six shared steps; 17 complex enclosures at 96 bits',
              'sampling_RNG_build_compile_H_DF_state_generation': 0}
    def save():
        temporary = output/'RESULT.json.tmp'
        temporary.write_text(json.dumps(record, allow_nan=False, indent=2)+'\n')
        temporary.replace(output/'RESULT.json')
    try:
        save()
        groups = [[x for x in rows[:17] if x['r'] == r] for r in sorted({x['r'] for x in rows[:17]})]
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
        workers = min(len(groups), cpus)
        record['resources'] = {'available_CPUs': cpus, 'workers': workers, 'threads_per_worker': max(1, cpus//workers)}
        # No physical basis replay; no thread-unsafe shared FLINT precision context.
        unconfirmed = False
        process_context = multiprocessing.get_context('spawn')
        stop = process_context.Event()
        with ProcessPoolExecutor(max_workers=workers, mp_context=process_context,
                                 initializer=initialize_worker, initargs=(stop,)) as pool:
            futures = [pool.submit(group_worker, group, raw, max(1, cpus//workers),
                                   output/f"group_r{group[0]['r']}.json") for group in groups]
            try:
                for future in as_completed(futures):
                    if future.cancelled():
                        record.setdefault('cancelled_groups', 0)
                        record['cancelled_groups'] += 1
                        continue
                    try:
                        group = future.result()
                    except Exception as exc:
                        unconfirmed = True
                        stop.set()
                        record.setdefault('worker_errors', []).append({'type': type(exc).__name__, 'message': str(exc)})
                        for pending in futures:
                            pending.cancel()
                        save()
                        continue
                    record['groups'].append(group)
                    save()
                    if any(x['categorical_decision'] != 'SUFFICIENT_CONDITION_CONFIRMED'
                           for row in group['rounds'] for x in row['axes'].values()):
                        unconfirmed = True
                        stop.set()
                        for pending in futures:
                            pending.cancel()
            except BaseException:
                stop.set()
                for pending in futures:
                    pending.cancel()
                raise
        if unconfirmed:
            raise ValueError('UNKNOWN/STOP: strict local condition not confirmed or worker failed; no retry')
        import round_adapter as adapter
        old = json.loads(raw['M17RESULT'])['enclosures'][0]
        combined = [row for group in record['groups'] for row in group['rounds']]+[{'m': 17, 'axes': old['axes']}]
        record['categorical_local_conditions'] = adapter.aggregate(combined)
        record['execution_status'] = 'COMPLETED_MANDATORY_STOP'
    except BaseException as exc:
        record['execution_status'] = 'INTERRUPTED_OR_UNCONFIRMED_MANDATORY_STOP'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='separate explicit scientific execution permission required')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.run:
        if args.output_dir is None:
            parser.error('--run requires fresh --output-dir')
        execute(args.output_dir)
    else:
        print(json.dumps(describe(), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({**STATUS, 'execution_status': 'UNKNOWN_MANDATORY_STOP',
                          'error': {'type': type(exc).__name__, 'message': str(exc)}}), file=sys.stderr)
        raise SystemExit(1)
