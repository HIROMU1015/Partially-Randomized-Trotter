"""Fixed inventory, two workers, one backend call per key, no rescue."""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ..rte_development.evidence import atomic_json, digest, dumps


def validate_inventory(inventory, c):
    keys = inventory['keys']
    if len(keys) != 36 or inventory['used_unique_keys'] != 36:
        raise PermissionError('all 36 fixed keys required')
    if len({k['new_key'] for k in keys}) != 36 or len({k['ratio'] for k in keys}) != 36:
        raise PermissionError('duplicate fixed key')
    if [F(k['ratio']) for k in keys] != sorted(F(k['ratio']) for k in keys):
        raise PermissionError('fixed rational key order changed')
    for k in keys:
        spec = dict(angle_kind='atan',ratio=str(F(k['ratio'])),scale='1',epsilon='1/10000000',
                    options=c['acquisition']['options'],
                    tool_identity_sha256=c['acquisition']['tool_identity']['sha256'],
                    relative_controlled_phase='strict; signed inverse from actual adjoint')
        if (k['new_spec'] != spec or hashlib.sha256(dumps(spec).encode()).hexdigest() != k['new_key']
            or k['new_price'] is not None or k['synthesis_calls_now'] != 0):
            raise PermissionError('fixed key spec/precision/seed binding changed')
    return keys


def check_saved(saved, ratio, precision):
    from ..rte_reallocation.native import Angle
    from ..rte_reallocation.numeric import validate_saved
    return validate_saved(saved, Angle('atan',F(ratio)), F(precision))


def validate_result(result, job):
    if (result.get('status') != 'KEY_COMPLETE' or result.get('synthesis_calls') != 1
        or result.get('retries') != 0):
        raise ArithmeticError('key incomplete; no retry')
    for k,v in job.items():
        if result.get(k) != v:
            raise PermissionError('worker receipt provenance mismatch: '+k)
    return check_saved(result['saved'],job['ratio'],'1/10000000')


def acquire(root, c, identity, inventory, writer, guard):
    root = Path(root); keys = validate_inventory(inventory,c)
    jobs = writer.directory/'jobs'; jobs.mkdir()
    active, cache, receipts = {}, {}, []
    index = 0
    try:
        while index < len(keys) or active:
            guard.check()
            while index < len(keys) and len(active) < c['caps']['workers']:
                entry=keys[index]; job=dict(**identity,key=entry['new_key'],ratio=entry['ratio'],
                                            parent_pid=os.getpid(),job_index=index)
                path=jobs/f'key_{index:03d}.json'; atomic_json(path,job); index+=1
                stdout=path.with_suffix('.stdout.log').open('xb')
                stderr=path.with_suffix('.stderr.log').open('xb')
                env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',
                         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                child=subprocess.Popen([c['acquisition']['runtime_executable'],'-B',
                       'scripts/tracks/algorithm_codesign/run_d3_synthesis_key.py','--job',str(path)],
                       cwd=root,env=env,stdout=stdout,stderr=stderr)
                active[child.pid]=(child,path,job,stdout,stderr)
                guard.children.add(child.pid)
            for pid,(child,path,job,stdout,stderr) in list(active.items()):
                waited,status,usage=os.wait4(pid,os.WNOHANG)
                if not waited:continue
                child.returncode=os.waitstatus_to_exitcode(status)
                guard.reaped_cpu+=usage.ru_utime+usage.ru_stime
                guard.children.discard(pid);stdout.close();stderr.close();del active[pid]
                rp=path.with_suffix('.result.json')
                receipt=dict(key=job['key'],ratio=job['ratio'],pid=pid,exit_code=child.returncode,
                             CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_KiB=usage.ru_maxrss,
                             result_sha256=digest(rp) if rp.exists() else None)
                atomic_json(path.with_suffix('.process_receipt.json'),receipt)
                receipts.append(receipt)
                if child.returncode != 0 or not rp.exists():
                    raise RuntimeError('synthesis key process failed; no retry')
                value=json.loads(rp.read_text())
                cache[job['ratio']]=validate_result(value,job)
                if len(dumps(cache).encode()) > c['caps']['new_cache_bytes']:
                    raise RuntimeError('new cache byte cap')
            if active:time.sleep(.05)
    finally:
        for child,_,_,_,_ in active.values():
            if child.poll() is None:child.terminate()
        for pid,(child,_,_,stdout,stderr) in active.items():
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            stdout.close();stderr.close();guard.children.discard(pid)
    if len(cache) != 36 or len(receipts) != 36:
        raise ArithmeticError('incomplete new key coverage')
    return cache, receipts
