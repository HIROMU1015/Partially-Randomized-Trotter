#!/usr/bin/env python3
"""P1 paired development pilot with the efficient controlled ZZ (approved scope).

--plan-out writes the plan fixed before science. --execute (with the committed SOURCE and that
plan) runs: phase A, per path in parallel processes, sample the planned trajectory, check seeds,
count events and check legacy-vs-new tail parity (instruction replacement for every request,
full operator equality for m0); STOP unless all PASS. Phase B, per path, re-samples (digest must
match), and with the efficient lowering active calls the C1 Backend.price_requests (paired S and
X/Y full-wrapper compiles), recording compile wall time and peak RSS. At most 8 compiles.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import p1_efficient_zz as P  # noqa: E402
from trottertracks import efficient_controlled_zz as E  # noqa: E402

PROVIDER_FILES = ('src/trotterlib/df_rte_qiskit.py', 'src/trotterlib/rte.py', 'src/trotterlib/rte_compiled_cost.py',
                  'src/trotterlib/df_partial_s2.py', 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
STATE = {}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def isolated(fn, jobs):
    """One fork process per job (isolated peak RSS); results over pipes."""
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
        msg = recv.recv(); proc.join()
        out.append(dict(job=job, exitcode=proc.exitcode, **msg))
    return out


def _cell(job):
    plan = STATE['plan']; return next(c for c in plan['cells'] if c['id'] == job[0])


def _sample(cell, index):
    ctx = STATE['ctx']; m = STATE['m']
    c = m.row_for(STATE['plan'], cell)['candidate']
    requests, seeds, laws = ctx.h['sample_requests'](c, ctx.prep, cell['cohort'], index)
    if seeds != cell['seeds'][index]:
        raise ValueError('FROZEN_SEEDS_DIFFER')
    events = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    return requests, seeds, laws, events, m.digest(events)


def phase_a(job):
    cell = _cell(job); index = job[1]; start = time.monotonic()
    requests, seeds, laws, events, digest = _sample(cell, index)
    legacy = STATE['legacy']; new = E.make_builder_class(legacy); registry = STATE['ctx'].prep.rte_preparation.basis_registry
    parity = dict(requests=len(requests), replaced_controlled_zz=0, legacy_instructions=0, new_instructions=0, failures=[], operator=[])
    for k, req in enumerate(requests):
        old = legacy(basis_registry=registry).build_sequence(req.rte_occurrence, basis_plan=req.rte_basis_plan).circuit
        cur = new(basis_registry=registry).build_sequence(req.rte_occurrence, basis_plan=req.rte_basis_plan).circuit
        r = E.instruction_parity(old, cur)
        if r['status'] != 'PASS':
            parity['failures'].append(dict(request=k, **r)); continue
        for key in ('replaced_controlled_zz', 'legacy_instructions', 'new_instructions'):
            parity[key] += r[key]
        if cell['m'] == 0:
            import numpy as np
            from qiskit.quantum_info import Operator
            diff = float(np.abs(Operator(old).data - Operator(cur).data).max())
            parity['operator'].append(dict(request=k, max_abs_difference=diff, equal=diff < 1e-10))
    flat = [e for batch in events for e in batch]
    return dict(cell_id=cell['id'], index=index, event_digest=digest, seeds=seeds, laws=laws, events=P.event_metrics(flat),
                parity=parity, seconds=time.monotonic() - start)


def phase_b(job):
    cell = _cell(job); index = job[1]; ctx = STATE['ctx']; backend = STATE['backend']
    if getattr(STATE['dfq'].QiskitDFRTEEventCircuitBuilder, 'lowering_id', None) != E.LOWERING_ID:
        raise ValueError('Efficient lowering not active in worker')
    rss0 = rss_mb(); start = time.monotonic()
    requests, seeds, laws, events, digest = _sample(cell, index)
    if digest != STATE['phase_a'][(cell['id'], index)]:
        raise ValueError('Re-sampled trajectory differs from phase A')
    compile_log = []; original = ctx.compile
    def timed(*args, **kw):
        t = time.monotonic(); r = original(*args, **kw)
        compile_log.append(dict(seconds=time.monotonic() - t, rss_mb_after=rss_mb())); return r
    ctx.compile = timed
    directory = Path(STATE['private'])/cell['id']/str(index)
    result = backend.price_requests(cell, index, directory, requests, seeds, laws, conditional=False)
    axes = {}
    for axis in ('cosine', 'sine'):
        axes[axis] = json.loads((directory/(axis + '.json')).read_text())
    return dict(result=result, axes=axes, compile_log=compile_log, seconds=time.monotonic() - start,
                rss_mb_start=rss0, rss_mb_peak=rss_mb(), private=str(directory))


def preflight(source, plan_path):
    if git('rev-parse', 'HEAD') != source or git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('HEAD must equal --source with a clean tracked tree')
    plan = json.loads(Path(plan_path).read_text())
    fresh = P.make_plan(ROOT, plan['source_label'])
    if json.loads(json.dumps(fresh)) != plan:
        raise SystemExit('Plan differs from the regenerated fixed plan')
    root = Path(plan['runtime']['provider_root'])
    for f in PROVIDER_FILES:
        if git('rev-parse', E.PROVIDER + ':' + f) != git('hash-object', str(root/f)):
            raise SystemExit('Provider SOURCE differs: ' + f)
    import numpy, qiskit
    if (numpy.__version__, qiskit.__version__) != ('1.26.4', '1.3.0'):
        raise SystemExit('Fixed runtime numpy 1.26.4 / qiskit 1.3.0 required')
    return plan, dict(numpy=numpy.__version__, qiskit=qiskit.__version__, python=sys.version.split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-out', type=Path); ap.add_argument('--source-label', default='SET_BY_COMMIT')
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--source'); ap.add_argument('--plan', type=Path)
    ap.add_argument('--out', type=Path); ap.add_argument('--private', type=Path)
    args = ap.parse_args()
    if args.plan_out:
        write(args.plan_out, P.make_plan(ROOT, args.source_label)); return
    if not (args.execute and args.source and args.plan and args.out and args.private):
        raise SystemExit('--execute needs --source, --plan, --out and --private')
    plan, runtime = preflight(args.source, args.plan)
    args.out.mkdir(parents=True, exist_ok=False); args.private.mkdir(parents=True, exist_ok=False)
    started = time.time()
    exec_plan = P.execution_plan(plan, args.source, ROOT)     # Backend reads plan['source'] (C1 semantics)
    mods = P.load_c1_backend(ROOT); science = mods['r_price_phase1_science']
    backend = science.Backend(ROOT, exec_plan); ctx = backend.context('H4')
    import trotterlib.df_rte_qiskit as dfq
    STATE.update(plan=plan, backend=backend, ctx=ctx, m=mods['r_price_phase1'], legacy=dfq.QiskitDFRTEEventCircuitBuilder,
                 dfq=dfq, private=str(args.private))
    jobs = [(cell['id'], i) for cell in plan['cells'] for i in P.PATHS]
    base = dict(source=args.source, plan=dict(path=str(args.plan), sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest()),
                runtime=runtime, lowering=plan['lowering'], configuration_id=plan['configuration_id'], cohort=P.COHORT)
    a = isolated(phase_a, jobs)
    write(args.out/'PHASE_A.json', dict(base, phase_a=a))          # saved before any compile
    bad = [x for x in a if not x['ok'] or x['value']['parity']['failures']
           or any(not o['equal'] for o in x['value']['parity']['operator'])]
    if bad:
        write(args.out/'STOP.json', dict(base, status='STOP_TECHNICAL_PARITY_OR_SAMPLING', phase_a=a)); raise SystemExit(2)
    STATE['phase_a'] = {(x['value']['cell_id'], x['value']['index']): x['value']['event_digest'] for x in a}
    with E.efficient_controlled_zz(dfq):
        b = isolated(phase_b, jobs)
    if dfq.QiskitDFRTEEventCircuitBuilder is not STATE['legacy']:
        raise SystemExit('Legacy builder not restored')
    rows = []
    for xa, xb in zip(a, b):
        va = xa['value']
        if not xb['ok']:
            rows.append(dict(cell_id=va['cell_id'], index=va['index'], status='FAILED', error=xb['error'])); continue
        vb = xb['value']; r = vb['result']
        rows.append(dict(cell_id=va['cell_id'], index=va['index'], status='PASS', event_digest=r['event_digest'],
            seeds=va['seeds'], laws=va['laws'], events=va['events'], parity=va['parity'], C=r['C'], S=r['S'],
            D=[c - s for c, s in zip(r['C'], r['S'])], compiled={k: v['metrics'] for k, v in vb['axes'].items()},
            fingerprints=r['fingerprints'], cache_hits=r['cache_hits'], new_compiles=r['new_compiles'], compiler=r['compiler'],
            compile_log=vb['compile_log'], price_seconds=vb['seconds'], rss_mb_start=vb['rss_mb_start'], rss_mb_peak=vb['rss_mb_peak'],
            phase_a_seconds=va['seconds'], private_result=vb['private'], full_wrapper_statevector_validation=r['full_wrapper_statevector_validation']))
    compiles = sum(r.get('new_compiles', 0) for r in rows)
    status = 'P1_COMPLETE' if all(r['status'] == 'PASS' for r in rows) and compiles <= P.COMPILE_BUDGET else 'STOP_PARTIAL'
    if status != 'P1_COMPLETE':                                   # errors saved first; summary cannot hide them
        write(args.out/'STOP.json', dict(base, status=status, rows=rows, new_compiles=compiles))
    write(args.out/'RESULT.json', dict(base, status=status, rows=rows, summary=P.summarize(plan, [r for r in rows if r['status'] == 'PASS']),
        new_compiles=compiles, trajectories=len(jobs), wall_seconds=time.time() - started,
        all_law_E_C=None, sufficient_N=None, rmse_bound=None, G=None, mandatory_stop=True))
    if status != 'P1_COMPLETE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
