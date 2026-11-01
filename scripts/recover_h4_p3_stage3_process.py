#!/usr/bin/env python3
"""Compile only run02's 13 saved missing targets, one fresh process per target."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from zoneinfo import ZoneInfo

import run_h4_p3_stage3 as original

ROOT = Path(__file__).resolve().parents[1]
BASE = '2da49fa4e8578227c130fc6061a0b3061276f893'
OLD = 'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run02'
OUT = ROOT / 'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03'
OWN = ('scripts/recover_h4_p3_stage3_process.py',
       'tests/test_h4_p3_stage3_process_recovery.py',
       'docs/research/h4_p3_stage3_process_recovery_20261102.md')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(out, name, value):
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_qpy_bytes(raw, row):
    if len(raw) != row['stored_bytes'] or sha(raw) != row['sha256']:
        raise ValueError('INPUT_COMPRESSED_BYTES')
    decoded = gzip.decompress(raw)
    if len(decoded) != row['decoded_bytes'] or sha(decoded) != row['decoded_QPY_sha256']:
        raise ValueError('INPUT_QPY_BYTES')
    return decoded


def compile_saved_target(key, raw, row, out, compiler, compiler_hash, qubits):
    """No builder/sampler/native preflight; input is the saved QPY byte stream."""
    start = time.monotonic()
    pid = os.getpid()
    write(out, 'workers/' + key + '.json', dict(state='STARTED', pid=pid, target=key))
    try:
        from qiskit import qpy
        kernel = original.modules()
        circuits = qpy.load(io.BytesIO(checked_qpy_bytes(raw, row)))
        if len(circuits) != 1 or circuits[0].num_qubits != qubits:
            raise ValueError('SAVED_TARGET_DIMENSION')
        circuit = circuits[0]
        if circuit.num_clbits != 1 or circuit.count_ops().get('measure') != 1:
            raise ValueError('SAVED_MEASUREMENT_BOUNDARY')
        compiled, metrics = kernel.price(circuit)
        # Saved compiler JSON uses lists; the live dataclass uses tuples.
        metrics = json.loads(json.dumps(metrics, allow_nan=False))
        if metrics['compiler'] != compiler or metrics['compiler_hash'] != compiler_hash:
            raise ValueError('COMPILER_CHANGED')
        original.OUT = Path(out)
        output = original.save_qpy('prices/' + key + '/output.qpy.gz', compiled)
        metrics.update(input_QPY=row, output_QPY=output)
        write(out, 'prices/' + key + '/metrics.json', metrics)
        worker = dict(state='COMPLETED', pid=pid, target=key,
                      compile_calls=1, elapsed_seconds=time.monotonic() - start)
        write(out, 'workers/' + key + '.json', worker)
        return key, metrics, worker
    except Exception:
        write(out, 'workers/' + key + '.json', dict(state='FAILED_NO_RETRY', pid=pid,
              target=key, error=traceback.format_exc(), elapsed_seconds=time.monotonic() - start))
        raise


def process_pool(workers):
    return ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1)


def merge_prices(saved, recovered, inputs, pending):
    if set(saved) & set(recovered) or set(recovered) != set(pending):
        raise ValueError('RECOVERY_TARGET_SET')
    if set(saved) | set(recovered) != set(inputs):
        raise ValueError('INCOMPLETE_COHORT')
    return dict(sorted({**saved, **recovered}.items()))


def interactions(prices):
    values = {}
    for arm in ('P', 'DF'):
        for index in range(4):
            for axis in ('cosine', 'sine'):
                key = f'{arm}_{index}'
                f, e, r, empty = (prices[k + '/' + axis]
                                  for k in (key + '/F', key + '/E', 'R', 'Empty'))
                values[key + '/' + axis] = {m: f[m] - r[m] - e[m] + empty[m]
                                           for m in ('RZ', 'CX', 'size')}
    return dict(formula='F - R - E + Empty', values=values,
        interpretation='fixed-program compiler nonadditivity, no unique causal allocation; depth not additive')


def saved_inputs():
    names = ('pending_targets.json', 'price_inputs.json', 'prices.json', 'preflight.json',
             'source_and_inputs.json', 'STOP.json', 'archive_members.json', 'native_evidence.zip')
    raw = {name: original.previous.gitbytes(BASE, OLD + '/' + name) for name in names}
    for name, data in raw.items():
        if (ROOT / OLD / name).read_bytes() != data:
            raise ValueError('SAVED_EVIDENCE_CHANGED:' + name)
    data = {name: json.loads(value) for name, value in raw.items() if name.endswith('.json')}
    pending = data['pending_targets.json']['targets']
    inputs, prices = data['price_inputs.json'], data['prices.json']
    if len(pending) != 13 or len(prices) != 23 or len(inputs) != 36:
        raise ValueError('FIXED_13_23_36')
    if set(pending) != set(inputs) - set(prices) or set(prices) - set(inputs):
        raise ValueError('FIXED_PENDING_SET')
    if any(row != inputs[key] for key, row in pending.items()):
        raise ValueError('PENDING_INPUT_BINDING')
    def leaves(value):
        if isinstance(value, dict):
            if 'status' in value:
                yield value['status']
            else:
                for child in value.values():
                    yield from leaves(child)
    checks = list(leaves(data['preflight.json']))
    if len(checks) != 81 or set(checks) != {'PASS'}:
        raise ValueError('SAVED_NATIVE_PREFLIGHT')
    if data['STOP.json']['status'] != 'STOP_STAGE3_PREFLIGHT_OR_EXECUTION_FAILURE_NO_RETRY':
        raise ValueError('SAVED_STOP')
    payloads = {}
    members = data['archive_members.json']['members']
    with zipfile.ZipFile(io.BytesIO(raw['native_evidence.zip'])) as archive:
        if set(archive.namelist()) != set(members):
            raise ValueError('ARCHIVE_MEMBER_SET')
        for name, meta in members.items():
            value = archive.read(name)
            if len(value) != meta['bytes'] or sha(value) != meta['sha256']:
                raise ValueError('ARCHIVE_BYTES:' + name)
        for key, row in inputs.items():
            value = archive.read(row['path'])
            checked_qpy_bytes(value, row)
            if key in pending:
                payloads[key] = value
        for key, metrics in prices.items():
            if metrics['input_QPY'] != inputs[key]:
                raise ValueError('SAVED_PRICE_INPUT:' + key)
            row = metrics['output_QPY']
            checked_qpy_bytes(archive.read(row['path']), row)
            if json.loads(archive.read('prices/' + key + '/metrics.json')) != metrics:
                raise ValueError('SAVED_PRICE_METRICS:' + key)
    return data, payloads, {OLD + '/' + name: sha(value) for name, value in raw.items()}


def execute():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    recovered, failures, workers = {}, {}, {}
    counts = dict(new_compile_targets=13, recovered_prices=0, reused_prices=23,
        sampling_calls=0, native_builds=0, native_preflight_reexecutions=0,
        saved_price_recompiles=0, new_signal_B_N_G=0, automatic_retries=0)
    source, code, evidence = {}, {}, {}
    try:
        data, payloads, evidence = saved_inputs()
        oldsource = data['source_and_inputs.json']
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        code = {**oldsource['code_sha256'], **{path: sha((ROOT / path).read_bytes()) for path in OWN}}
        for path, digest in code.items():
            if sha((ROOT / path).read_bytes()) != digest or sha(original.previous.gitbytes(commit, path)) != digest:
                raise ValueError('UNFIXED_SOURCE:' + path)
        runtime = {name: importlib.metadata.version(name) for name in oldsource['runtime']}
        if runtime != oldsource['runtime']:
            raise ValueError('RUNTIME_CHANGED')
        first = next(iter(data['prices.json'].values()))
        compiler, compiler_hash = first['compiler'], first['compiler_hash']
        if any(m['compiler'] != compiler or m['compiler_hash'] != compiler_hash
               for m in data['prices.json'].values()):
            raise ValueError('SAVED_COMPILER_DISAGREEMENT')
        available = max(1, len(os.sched_getaffinity(0)) - math.ceil(os.getloadavg()[0]))
        concurrency = min(13, available)
        source = dict(commit=commit, code_sha256=code, saved_result_commit=BASE,
            saved_scientific_SOURCE=oldsource['commit'], saved_namespace=OLD,
            input_files_sha256=evidence, conditions=oldsource['conditions'], runtime=runtime,
            compiler=compiler, compiler_hash=compiler_hash, python=sys.version,
            executable=sys.executable, execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            process_policy='spawn; max_tasks_per_child=1; no in-process parallel transpile',
            workers=concurrency, cpu_affinity_count=len(os.sched_getaffinity(0)),
            load=os.getloadavg(), saved_native_checks_reused=81,
            new_scope='exact 13 saved missing targets only, merge unchanged 23, planned interaction')
        write(OUT, 'source_and_inputs.json', source)
        pending = data['pending_targets.json']['targets']
        write(OUT, 'recovery_targets.json', pending)
        write(OUT, 'execution_status.json', dict(state='COMPILING_SAVED_13', counts=counts))
        # A fresh interpreter handles one target and exits; no shared Qiskit state.
        with process_pool(concurrency) as pool:
            futures = {pool.submit(compile_saved_target, key, payloads[key], row,
                       str(OUT), compiler, compiler_hash, 9): key for key, row in pending.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, metrics, worker = future.result()
                    recovered[key], workers[key] = metrics, worker
                except Exception:
                    failures[key] = traceback.format_exc()
                    # Let already-running independent work finish and preserve it.
                    for other in futures:
                        other.cancel()
                counts['recovered_prices'] = len(recovered)
                write(OUT, 'recovered_prices.json', recovered)
                write(OUT, 'execution_status.json', dict(state='COMPILING_SAVED_13', counts=counts,
                      failed_targets=failures, worker_processes=workers))
                print(json.dumps(dict(saved=len(recovered), failed=len(failures), target=key)), flush=True)
        if failures:
            raise RuntimeError('PROCESS_COMPILE_FAILURE:' + ','.join(sorted(failures)))
        if len({w['pid'] for w in workers.values()}) != 13:
            raise ValueError('PROCESS_REUSE')
        prices = merge_prices(data['prices.json'], recovered, data['price_inputs.json'], pending)
        write(OUT, 'prices.json', prices)
        write(OUT, 'price_provenance.json', {key: dict(input_result_commit=BASE,
              input_namespace=OLD, output_namespace=str(OUT.relative_to(ROOT)) if key in recovered else OLD,
              acquired_in='run03' if key in recovered else 'run02') for key in prices})
        write(OUT, 'interaction.json', interactions(prices))
        for path, digest in {**code, **evidence}.items():
            if sha((ROOT / path).read_bytes()) != digest:
                raise ValueError('SOURCE_OR_INPUT_CHANGED:' + path)
        finish = dict(state='COMPLETED', counts=counts, whole_price_cohort=36,
            workers_joined=True, distinct_worker_pids=len(workers),
            elapsed_seconds=time.monotonic() - start, worker_processes=workers)
        write(OUT, 'execution_status.json', finish)
        write(OUT, 'STOP.json', dict(status='STOP_STAGE3_FIXED_COHORT_COST_COMPLETE_NO_AUTOMATIC_FOLLOWUP',
              **finish, Stage4_authorized=False, automatic_followup=False))
        print(json.dumps(dict(source=commit, **finish)), flush=True)
    except Exception:
        failure = dict(state='FAILED_NO_RETRY', counts=counts, source_commit=source.get('commit'),
            error=traceback.format_exc(), failed_targets=failures, workers_joined=True,
            elapsed_seconds=time.monotonic() - start)
        write(OUT, 'recovered_prices.json', recovered)
        write(OUT, 'execution_status.json', failure)
        write(OUT, 'STOP.json', dict(status='STOP_STAGE3_PROCESS_RECOVERY_FAILURE_NO_RETRY',
              **failure, automatic_followup=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', required=True, action='store_true')
    parser.parse_args()
    execute()
