#!/usr/bin/env python3
"""P1 run02 continuity pre-check (sampling only, no compile).

Run with SOURCE 20fe1194 checked out (HEAD, clean tracked tree), before the run02 runner.
For each planned path (m0/m6 x path 0/1) it regenerates the trajectory exactly as the runner
does (carrier sample_requests through the pinned C1 Backend context), checks the PLAN seeds and
hashes the bytes Backend.price_requests writes as events_PRIVATE.json
(json.dump(dict(events, seeds, laws, conditional=False), indent=2, ensure_ascii=False) + newline,
UTF-8). The SHA-256 must equal the run01 file recorded in run01/STOP.json, and the preserved run01
file is re-hashed read-only. Exit 0 only if every path matches; then the runner may start.
"""
import hashlib
import json
import locale
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import p1_efficient_zz as P  # noqa: E402

SOURCE = '20fe11942369253361068dd2b00f98b26350ba8d'
DOC = ROOT/'docs/research/d_p1_efficient_czz_20261113'
PLAN_SHA = '7580ed3baf1673d0e781329c427b3d941770c6fad9beca77d004de019e1acd3c'
PRIVATE = Path('/home/AbeHiromu/projects/d-p1-efficient-czz-20261113-private')
STATE = {}


class HashWriter:
    def __init__(self):
        self.h = hashlib.sha256(); self.n = 0

    def write(self, s):
        b = s.encode('utf-8'); self.h.update(b); self.n += len(b)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def check(job):
    cell_id, index = job
    plan = STATE['plan']; ctx = STATE['ctx']; m = STATE['m']
    cell = next(c for c in plan['cells'] if c['id'] == cell_id)
    c = m.row_for(plan, cell)['candidate']; start = time.monotonic()
    requests, seeds, laws = ctx.h['sample_requests'](c, ctx.prep, cell['cohort'], index)
    events = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    w = HashWriter()
    json.dump(dict(events=events, seeds=seeds, laws=laws, conditional=False), w, ensure_ascii=False, indent=2, allow_nan=False)
    w.write('\n')
    key = 'run01/%s/%d/events_PRIVATE.json' % (cell_id, index)
    expected = STATE['run01'][key]
    run01_file = file_sha(PRIVATE/key)
    return dict(cell_id=cell_id, index=index, seeds_match=seeds == cell['seeds'][index],
                regenerated=dict(sha256=w.h.hexdigest(), bytes=w.n), run01_recorded=expected,
                run01_file_sha256_now=run01_file, event_digest=m.digest(events),
                match=(seeds == cell['seeds'][index] and w.h.hexdigest() == expected['sha256'] == run01_file and w.n == expected['bytes']),
                seconds=time.monotonic() - start)


def isolated(fn, jobs):
    ctx = mp.get_context('fork'); procs = []
    for job in jobs:
        recv, send = ctx.Pipe(duplex=False)
        def target(job=job, send=send):
            try:
                send.send(dict(ok=True, value=fn(job)))
            except BaseException:
                send.send(dict(ok=False, error=traceback.format_exc()))
        proc = ctx.Process(target=target); proc.start(); procs.append((job, proc, recv))
    out = []
    for job, proc, recv in procs:
        msg = recv.recv(); proc.join(); out.append(dict(job=list(job), exitcode=proc.exitcode, **msg))
    return out


def main():
    out = Path(sys.argv[1])
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT).decode().strip()
    if git('rev-parse', 'HEAD') != SOURCE or git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('HEAD must be the fixed SOURCE with a clean tracked tree')
    plan_path = DOC/'PLAN.json'
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != PLAN_SHA:
        raise SystemExit('PLAN differs')
    if locale.getpreferredencoding(False) != 'UTF-8':
        raise SystemExit('Backend write() encoding would not be UTF-8')
    plan = json.loads(plan_path.read_text())
    STATE['run01'] = json.loads((DOC/'run01/STOP.json').read_text())['private_partial']['files']
    exec_plan = P.execution_plan(plan, SOURCE, ROOT)
    mods = P.load_c1_backend(ROOT); backend = mods['r_price_phase1_science'].Backend(ROOT, exec_plan)
    started = time.time(); ctx = backend.context('H4')
    STATE.update(plan=plan, ctx=ctx, m=mods['r_price_phase1'])
    jobs = [(cell['id'], i) for cell in plan['cells'] for i in P.PATHS]
    rows = isolated(check, jobs)
    status = 'PASS' if all(r['ok'] and r['value']['match'] for r in rows) else 'FAIL'
    with out.open('x') as f:
        json.dump(dict(status=status, source=SOURCE, plan_sha256=PLAN_SHA, compiles=0,
                       check='regenerated events_PRIVATE.json bytes (Backend.price_requests serialization) == run01 file SHA-256',
                       rows=rows, wall_seconds=time.time() - started), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(status)
    raise SystemExit(0 if status == 'PASS' else 2)


if __name__ == '__main__':
    main()
