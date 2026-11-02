#!/usr/bin/env python3
"""Fixed new 32-per-arm Full price cohort, no pooling/top-up or Stage 4."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib
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
from types import SimpleNamespace as NS
import zipfile
from zoneinfo import ZoneInfo

import run_h4_p3_stage3 as original
import recover_h4_p3_stage3_process as recovery

ROOT = original.ROOT
BASE = '60515a33ce6231e9ff4ffdb40124ed845373351c'
PILOT = '90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5'
OLD = recovery.OLD
PLAN = 'docs/research/h4_p3_poststage3_cost_and_stage4_design_20261102.md'
OUT = ROOT / 'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04'
OWN = ('scripts/run_h4_p3_price_confirmation.py', 'src/trotterlib/matched_partition_price_mean.py',
       'tests/test_matched_partition_price_mean.py',
       'docs/research/h4_p3_price_confirmation_execution_20261102.md')
_CTX = None


def modules():
    kernel = original.modules()
    return kernel, importlib.import_module('stage1.matched_partition_price_mean')


def saved_native():
    names = ('source_and_inputs.json', 'law_bindings.json', 'native_bases.json',
             'preflight.json', 'STOP.json', 'archive_members.json', 'native_evidence.zip')
    raw = {name: original.previous.gitbytes(PILOT, OLD + '/' + name) for name in names}
    for name, value in raw.items():
        if (ROOT / OLD / name).read_bytes() != value:
            raise ValueError('SAVED_NATIVE_BYTES:' + name)
    saved = {name: json.loads(value) for name, value in raw.items() if name.endswith('.json')}
    def statuses(obj):
        if isinstance(obj, dict):
            if 'status' in obj:
                yield obj['status']
            else:
                for value in obj.values():
                    yield from statuses(value)
    checks = list(statuses(saved['preflight.json']))
    if len(checks) != 81 or set(checks) != {'PASS'}:
        raise ValueError('SAVED_81_NATIVE_PASS')
    payloads = {}
    members = saved['archive_members.json']['members']
    with zipfile.ZipFile(io.BytesIO(raw['native_evidence.zip'])) as archive:
        for key, row in saved['native_bases.json']['qpy'].items():
            payload = archive.read(row['path'])
            if recovery.sha(payload) != members[row['path']]['sha256']:
                raise ValueError('ARCHIVE_NATIVE_MEMBER')
            payloads[key] = recovery.checked_qpy_bytes(payload, row)
    return saved, payloads, {OLD + '/' + k: recovery.sha(v) for k, v in raw.items()}


def restore_context(context):
    kernel, adapter = modules()
    meta = context['saved']['native_bases.json']
    decoded = context['native_QPY']
    half = adapter.restored_circuit(decoded['half'])
    inverse = adapter.restored_circuit(decoded['half_actual_inverse'])
    controlled_half = adapter.restored_circuit(decoded['controlled_half'],
        fingerprint=meta['common_half_fingerprint'])
    bases = {}
    for i in range(len(meta['DF'])):
        circuit = adapter.restored_circuit(decoded[f'basis_DF_{i}'],
            fingerprint=meta['DF'][str(i)]['fingerprint'])
        bases[i] = NS(ops=adapter.basis_ops(circuit))
    n = context.get('n', 8)
    comps = adapter.bind_components(context['dictionaries'], context['normalizations'], bases,
                                    context['saved']['law_bindings.json'], n=n)
    retained = NS(n=n, q=context.get('q', 8), half=half, inverse=inverse, controlled_half=controlled_half,
        reference=context.get('reference'), reference_error=context.get('reference_error'))
    return kernel, adapter, comps, retained


def initialize_native(context):
    global _CTX
    kernel, adapter, comps, retained = restore_context(context)
    _CTX = NS(kernel=kernel, adapter=adapter, comps=comps, retained=retained,
        angle=context['angle'], out=context['out'], oracles={
            arm: kernel.DictionaryOracle(comps[arm], context['dictionaries'][arm], retained.n)
            for arm in ('P', 'DF')})


def native_worker(key, arm, events):
    c = _CTX
    start = time.monotonic()
    c.kernel.OPERATIONS.clear()
    recovery.write(c.out, 'native_workers/' + key + '.json', dict(state='STARTED', pid=os.getpid()))
    try:
        body, sidecar = c.adapter.full_only(c.comps[arm], events, c.retained, c.angle)
        original.OUT = Path(c.out)
        native_qpy = original.save_qpy('native/' + key + '.qpy.gz', body)
        expected, error = c.oracles[arm].trajectory(events, c.retained, c.angle, part='F')
        check = c.kernel.program_check(body, expected, error)
        c.kernel.require_pass({key: check})
        inputs = {key + '/F/' + axis: original.save_qpy('prices/' + key + '/F/' + axis + '/input.qpy.gz', circ)
                  for axis, circ in c.kernel.wrappers(body).items()}
        result = dict(state='COMPLETED', pid=os.getpid(), target=key, check=check,
            native_QPY=native_qpy, input_QPY=inputs, outer_residual=sidecar,
            operations=dict(c.kernel.OPERATIONS), elapsed_seconds=time.monotonic() - start)
        recovery.write(c.out, 'native_workers/' + key + '.json', result)
        return key, result
    except Exception as error:
        recovery.write(c.out, 'native_workers/' + key + '.json', dict(state='FAILED_NO_RETRY',
            pid=os.getpid(), target=key, error=traceback.format_exc(), failed_checks=getattr(error, 'checks', None),
            operations=dict(c.kernel.OPERATIONS), elapsed_seconds=time.monotonic() - start))
        raise


def execute():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    counts = dict(sampling_calls=0, trajectories=0, event_positions=0, native_Full_PASS=0,
        reference_half_exponentials=0, saved_half_matrix_evaluations=0, compile_completed=0,
        new_eigendecompositions=0, new_Givens=0, new_E_R_Empty=0,
        old_prices_recompiled=0, old_pilot_pooled=0, new_signal_B_N_G=0, automatic_retries=0)
    source, native_results, prices, compiler_workers, failures = {}, {}, {}, {}, {}
    def write(name, value): recovery.write(OUT, name, value)
    def status(state):
        write('execution_status.json', dict(state=state, counts=counts,
              elapsed_seconds=time.monotonic() - start, failed_targets=failures))
    try:
        kernel, adapter = modules()
        saved, payloads, evidence = saved_native()
        _, t, nuclear, dictionaries, normalizations, input_files = original.inputs()
        oldsource = saved['source_and_inputs.json']
        runtime = {x: importlib.metadata.version(x) for x in oldsource['runtime']}
        if runtime != oldsource['runtime']:
            raise ValueError('FIXED_RUNTIME')
        if (ROOT / PLAN).read_bytes() != original.previous.gitbytes(BASE, PLAN):
            raise ValueError('FIXED_PLAN')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        code = {**oldsource['code_sha256'], **{p: recovery.sha((ROOT / p).read_bytes()) for p in OWN},
                'scripts/recover_h4_p3_stage3_process.py': recovery.sha((ROOT / 'scripts/recover_h4_p3_stage3_process.py').read_bytes())}
        for p, digest in code.items():
            if recovery.sha((ROOT / p).read_bytes()) != digest or recovery.sha(original.previous.gitbytes(commit, p)) != digest:
                raise ValueError('UNFIXED_SOURCE:' + p)
        available = max(1, len(os.sched_getaffinity(0)) - math.ceil(os.getloadavg()[0]))
        workers = min(64, available)
        pilot_prices = json.loads(original.previous.gitbytes(PILOT, OLD.replace('run02', 'run03') + '/prices.json'))
        first = next(iter(pilot_prices.values()))
        source = dict(commit=commit, plan_commit=BASE, pilot_result_commit=PILOT,
            native_SOURCE=oldsource['commit'], code_sha256=code, input_files=input_files,
            saved_native_files_sha256=evidence, conditions=dict(oldsource['conditions'],
                seeds={a: list(v) for a,v in adapter.SEEDS.items()}, trajectories_per_arm=32),
            seeds={a: list(v) for a, v in adapter.SEEDS.items()}, new_trajectories_per_arm=32,
            saved_native_checks_reused=81, old_pilot_pooled=False, runtime=runtime,
            python=sys.version, executable=sys.executable, compiler=first['compiler'], compiler_hash=first['compiler_hash'],
            workers=workers, process_policy='native spawn; compiler spawn max_tasks_per_child=1',
            cpu_affinity=len(os.sched_getaffinity(0)), initial_load=os.getloadavg(),
            memory_observation={x.split(':')[0]: x.split(':')[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()
                                if x.startswith(('MemTotal:', 'MemAvailable:'))},
            execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat())
        write('source_and_inputs.json', source)
        context = dict(saved=saved, native_QPY=payloads, dictionaries=dictionaries,
            normalizations=normalizations, out=str(OUT), angle=kernel.scalar_angle(nuclear, 4.679666433618258))
        kernel, adapter, comps, retained = restore_context(context)
        status('SAVED_NATIVE_AND_LAW_BOUND')
        tau = float(kernel.TIME / 8 / 2)
        reference = kernel.expm(-1j * tau * t)
        counts['reference_half_exponentials'] = 1
        actual, half_error = kernel.circuit_matrix(retained.half)
        counts['saved_half_matrix_evaluations'] = 1
        half_check = kernel.compare(actual, reference, 256)
        write('preflight.json', dict(saved_system_half=half_check, saved_native_PASS_reused=81))
        kernel.require_pass({'saved_system_half': half_check})
        reference_error = kernel.norm(actual - reference) + half_error + kernel.gamma(8*256) * (1 + tau * kernel.norm(t))
        context.update(reference=reference, reference_error=reference_error)
        write('reference_half.json', dict(check=half_check, reference_error=reference_error,
              saved_half_roundoff=half_error, tau=tau, independent_exponentials=1))
        trajectories = {}
        for arm in ('P', 'DF'):
            for i, seed in enumerate(adapter.SEEDS[arm]):
                events = kernel.sample_rte_events(comps[arm].components, comps[arm].distribution, sample_count=32, seed=seed)
                key = f'{arm}_{i:02d}'
                trajectories[key] = (arm, events)
                counts['sampling_calls'] += 1; counts['trajectories'] += 1; counts['event_positions'] += len(events)
                write('trajectories/' + key + '.json', dict(arm=arm, index=i, seed=seed, events=[
                    dict(original.compact_event(e), outer_position=j//4, inner_position=j%4) for j,e in enumerate(events)]))
        status('64_TRAJECTORIES_FROZEN')
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_native, initargs=(context,)) as pool:
            futures = {pool.submit(native_worker, key, arm, events): key for key, (arm, events) in trajectories.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, result = future.result(); native_results[key] = result
                except Exception:
                    failures[key] = traceback.format_exc()
                    for other in futures: other.cancel()
                counts['native_Full_PASS'] = len(native_results)
                write('native_preflight.json', native_results); status('NATIVE_FULL_PREFLIGHT')
                print(json.dumps(dict(native_PASS=len(native_results), failures=len(failures), target=key)), flush=True)
        if failures or len(native_results) != 64:
            raise ValueError('NATIVE_COHORT_NONPASS_NO_RETRY')
        inputs = {key: row for result in native_results.values() for key, row in result['input_QPY'].items()}
        if len(inputs) != 128: raise ValueError('FIXED_128_INPUTS')
        write('price_inputs.json', inputs); status('ALL_64_FULL_PASS')
        with recovery.process_pool(workers) as pool:
            futures = {pool.submit(recovery.compile_saved_target, key, (OUT / row['path']).read_bytes(), row,
                       str(OUT), source['compiler'], source['compiler_hash'], 9): key for key,row in sorted(inputs.items())}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, metrics, worker = future.result(); prices[key] = metrics; compiler_workers[key] = worker
                except Exception:
                    failures[key] = traceback.format_exc()
                    for other in futures: other.cancel()
                counts['compile_completed'] = len(prices)
                write('prices.json', prices); status('COMPILING_FIXED_FULLS')
                print(json.dumps(dict(compiled=len(prices), failures=len(failures), target=key)), flush=True)
        if failures or len(prices) != 128: raise ValueError('INCOMPLETE_PRICE_COHORT_NO_RETRY')
        if len({w['pid'] for w in compiler_workers.values()}) != 128: raise ValueError('COMPILER_PROCESS_REUSE')
        write('price_statistics.json', adapter.price_statistics(prices))
        for p, digest in {**code, **evidence}.items():
            if recovery.sha((ROOT / p).read_bytes()) != digest: raise ValueError('SOURCE_OR_INPUT_CHANGED:' + p)
        for p, row in input_files.items():
            if recovery.sha((ROOT / p).read_bytes()) != row['sha256']: raise ValueError('FIXED_INPUT_CHANGED:' + p)
        finish = dict(state='COMPLETED', counts=counts, workers_joined=True,
            compiler_processes=128, elapsed_seconds=time.monotonic() - start)
        write('execution_status.json', finish)
        write('STOP.json', dict(status='STOP_STAGE3_PRICE_CONFIRMATION_COMPLETE_NO_AUTOMATIC_FOLLOWUP',
              **finish, Stage4_authorized=False, automatic_followup=False))
        print(json.dumps(dict(source=commit, **finish)), flush=True)
    except Exception:
        failure = dict(state='FAILED_NO_RETRY', counts=counts, source_commit=source.get('commit'),
            error=traceback.format_exc(), failed_targets=failures, workers_joined=True,
            elapsed_seconds=time.monotonic() - start)
        write('prices.json', prices); write('native_preflight.json', native_results)
        write('execution_status.json', failure)
        write('STOP.json', dict(status='STOP_STAGE3_PRICE_CONFIRMATION_FAILURE_NO_RETRY',
              **failure, automatic_followup=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--execute', required=True, action='store_true')
    parser.parse_args(); execute()
