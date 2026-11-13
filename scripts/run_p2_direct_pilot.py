#!/usr/bin/env python3
"""Fixed 2-path replay gate, then 8 fresh paired paths; at most 20 compiles.

Reuses the fixed C1 context, carrier sampler/compose and accelerated price API.
No retry/resume, fit, forced-rare sampler, or subsequent science is implemented.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import p2_direct_pilot as Q
from trottertracks import p1_efficient_zz as P
from trottertracks import compiled_price_fast as F
from trottertracks import efficient_controlled_zz as E

STATE = {}
PROVIDER_FILES = ('src/trotterlib/df_rte_qiskit.py', 'src/trotterlib/rte.py',
    'src/trotterlib/rte_compiled_cost.py', 'src/trotterlib/df_partial_s2.py',
    'src/trottertracks/resource_applicability/ax2b_native_df_v5.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(Q.json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def resources():
    import os
    return dict(affinity_cpu_count=len(os.sched_getaffinity(0)), loadavg=list(os.getloadavg()),
        meminfo={k:v.strip() for line in Path('/proc/meminfo').read_text().splitlines()
                 for k,v in [line.split(':',1)] if k in ('MemTotal','MemAvailable')},
        host=platform.node(), parent_peak_rss_bytes=rss())


def worker(job):
    phase, cell_id, index = job
    plan, ctx, backend, science = (STATE[k] for k in ('plan','ctx','backend','science'))
    cell = dict(next(c for c in plan['cells'] if c['id'] == cell_id))
    c = science.m.row_for(plan, cell)['candidate']
    start = time.perf_counter()
    if phase == 'replay':
        reference = next(v for v in plan['replay'] if v['cell_id'] == cell_id)
        path = Path(reference['event_file'])
        if file_sha(path) != reference['event_sha256']:
            raise ValueError('SAVED_REPLAY_BYTES_DIFFER')
        saved = json.loads(path.read_text())
        if saved['seeds'] != reference['seeds'] or saved['conditional']:
            raise ValueError('SAVED_REPLAY_SEEDS_OR_STRATUM_DIFFER')
        requests, seeds, laws = Q.restore_requests(ctx, c, saved)
        cell['cohort'] = reference['cohort']
        del saved
    else:
        requests, seeds, laws = ctx.h['sample_requests'](c, ctx.prep, cell['cohort'], index)
        if seeds != cell['seeds'][index]:
            raise ValueError('FROZEN_FRESH_SEEDS_DIFFER')
    acquisition_seconds = time.perf_counter()-start
    batches = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    event_digest = science.m.digest(batches)
    metrics = P.event_metrics([e for b in batches for e in b])
    if phase == 'replay' and event_digest != reference['event_digest']:
        raise ValueError('REPLAY_EVENT_DIGEST_DIFFERS')
    del batches
    compile_log = []
    original = ctx.compile
    def timed_compile(*args, **kwargs):
        if len(compile_log) >= 2:
            raise ValueError('PER_PATH_COMPILE_BUDGET')
        entry = dict(started=True, status='STARTED')
        compile_log.append(entry)
        then = time.perf_counter()
        try:
            value = original(*args, **kwargs)
            entry['status'] = 'COMPLETE'
            return value
        finally:
            entry.update(seconds=time.perf_counter()-then, peak_rss_bytes=rss())
    ctx.compile = timed_compile
    directory = Path(STATE['private'])/phase/cell_id/str(index)
    try:
        then = time.perf_counter()
        with E.efficient_controlled_zz(STATE['dfq']):
            result = backend.price_requests(cell, index, directory, requests, seeds, laws, conditional=False)
        price_seconds = time.perf_counter()-then
        axes = {a: json.loads((directory/(a+'.json')).read_text()) for a in science.m.AXES}
        record = dict(result, phase=phase, m=cell['m'], seeds=seeds, laws=laws, events=metrics,
            D=[v-s for v,s in zip(result['C'],result['S'])],
            compiled={a:v['metrics'] for a,v in axes.items()},
            fingerprint_scope='FULL_WRAPPER_PRE_TRANSPILE', compile_log=compile_log,
            acquisition_seconds=acquisition_seconds, price_call_seconds=price_seconds,
            path_wall_seconds=time.perf_counter()-start, peak_rss_bytes=rss(),
            timing=json.loads((directory/'TECHNICAL_TIMING.json').read_text()), private_directory=str(directory))
        if result['new_compiles'] != len(compile_log) or result['event_digest'] != event_digest:
            raise ValueError('COMPILE_COUNT_OR_DIGEST_DIFFERS')
        if phase == 'replay':
            record['replay_checks'] = Q.compare_replay(record, reference)
            record['original_event_sha256_after'] = file_sha(reference['event_file'])
            if record['original_event_sha256_after'] != reference['event_sha256']:
                raise ValueError('ORIGINAL_REPLAY_FILE_CHANGED')
        write(directory/'PILOT_PATH.json', record)
        return record
    except BaseException:
        if directory.exists():
            write(directory/'WORKER_STOP.json', dict(job=job, compile_log=compile_log, error=traceback.format_exc()))
        raise
    finally:
        ctx.compile = original


def isolated(jobs, out):
    """All finite jobs launched once, independent per-path process peaks; no retries."""
    context = mp.get_context('fork')
    pending = []
    sys.stdout.flush()
    for job in jobs:
        recv, send = context.Pipe(duplex=False)
        def target(job=job, send=send, recv=recv):
            recv.close()
            try:
                send.send(dict(ok=True, value=worker(job)))
            except BaseException:
                send.send(dict(ok=False, error=traceback.format_exc()))
            finally:
                send.close()
        process = context.Process(target=target)
        process.start()
        send.close()
        pending.append((job,process,recv))
        print('START',job,'pid',process.pid,flush=True)
    rows = []
    for job, process, recv in pending:
        try:
            msg = recv.recv()
        except EOFError:
            msg = dict(ok=False, error='Worker exited without a result')
        process.join()
        recv.close()
        record = dict(job=job, exitcode=process.exitcode, **msg)
        rows.append(record)
        write(out/('%s_%s_%d.json'%job),record)
        print('DONE',job,'ok',msg['ok'],'exit',process.exitcode,flush=True)
    return rows


def preflight(source, path):
    if git('rev-parse','HEAD') != source or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('EXECUTION_SOURCE_MUST_BE_COMMITTED_AND_TRACKED_CLEAN')
    plan = json.loads(path.read_text())
    if Q.make_plan(ROOT) != plan:
        raise ValueError('FIXED_PLAN_DIFFERS')
    for name in ('compiled_price_fast.py','efficient_controlled_zz.py'):
        rel = 'src/trottertracks/'+name
        if (ROOT/rel).read_bytes() != P._blob(ROOT,Q.BASE,rel):
            raise ValueError('BASE_PIPELINE_CHANGED:'+rel)
    provider = Path(plan['runtime']['provider_root'])
    for rel in PROVIDER_FILES:
        if (provider/rel).read_bytes() != P._blob(ROOT,E.PROVIDER,rel):
            raise ValueError('PROVIDER_SOURCE_DIFFERS:'+rel)
    for item in plan['replay']:
        if file_sha(item['event_file']) != item['event_sha256']:
            raise ValueError('REPLAY_INPUT_SHA256_DIFFERS')
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-out',type=Path)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--source')
    ap.add_argument('--plan',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--private',type=Path)
    args = ap.parse_args()
    if args.plan_out:
        write(args.plan_out,Q.make_plan(ROOT))
        return
    if not (args.execute and args.source and args.plan and args.out and args.private):
        ap.error('--execute needs --source --plan --out --private')
    plan = preflight(args.source,args.plan)
    args.out.mkdir(parents=True,exist_ok=False)
    args.private.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    base = dict(source=args.source, base=Q.BASE, plan_sha256=file_sha(args.plan),
        compile_budget=Q.COMPILE_BUDGET, resources_before=resources(), mandatory_stop=True)
    replay, fresh = [], []
    try:
        modules = P.load_c1_backend(ROOT)
        science = modules['r_price_phase1_science']
        cls = F.make_backend_class(science.Backend,science)
        execution_plan = P.execution_plan(plan,args.source,ROOT)
        backend = cls(ROOT,execution_plan)
        ctx = backend.context('H4')
        import trotterlib.df_rte_qiskit as dfq
        import numpy, qiskit
        base['runtime'] = dict(numpy=numpy.__version__,qiskit=qiskit.__version__,python=platform.python_version())
        STATE.update(plan=plan,ctx=ctx,backend=backend,science=science,dfq=dfq,private=str(args.private))
        write(args.out/'PREFLIGHT.json',base)
        replay = isolated([('replay',c['id'],0) for c in plan['cells']],args.out)
        write(args.out/'REPLAY.json',dict(base,records=replay))
        if any(not r['ok'] or r['exitcode'] for r in replay):
            raise ValueError('REPLAY_GATE_FAILED_FRESH_NOT_STARTED')
        fresh = isolated([('fresh',c['id'],i) for c in plan['cells'] for i in range(4)],args.out)
        write(args.out/'FRESH.json',dict(base,records=fresh))
        if any(not r['ok'] or r['exitcode'] for r in fresh):
            raise ValueError('FRESH_PARTIAL_FAILURE')
        rows = [r['value'] for r in replay+fresh]
        compiles = sum(r['new_compiles'] for r in rows)
        if compiles > Q.COMPILE_BUDGET or len(rows) != 10:
            raise ValueError('FINITE_SCOPE_DIFFERED')
        summary = Q.fresh_summary(plan,[r['value'] for r in fresh],args.source)
        structural = json.loads(P._blob(ROOT,'9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f',
            'docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json'))['values']
        result = dict(base,status='P2_DIRECT_PILOT_COMPLETE',rows=rows,summary=summary,
            future_design_only=Q.planning(summary,structural),new_compiles=compiles,
            total_wall_seconds=time.perf_counter()-start,resources_after=resources(),
            all_law_E_C=None,confidence_2pct_95_achieved=False,law_distance=None,rare_contribution=None,
            sufficient_N=None,rmse_bound=None,G=None)
        write(args.out/'RESULT.json',result)
        print('P2_DIRECT_PILOT_COMPLETE','compiles',compiles,'wall',result['total_wall_seconds'],flush=True)
    except BaseException:
        write(args.out/'STOP.json',dict(base,status='STOP',error=traceback.format_exc(),replay=replay,fresh=fresh,
            total_wall_seconds=time.perf_counter()-start,resources_after=resources()))
        raise


if __name__ == '__main__':
    main()
