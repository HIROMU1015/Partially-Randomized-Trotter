"""Fixed deterministic PF m0 or m1/m2 cells, original trials/gates; no model.

Only metadata selection, process/timing observation and receipt publication are
new. The native SOURCE, scalar, wrapper, compiler and scientific gates are reused.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pf_price_stages as S
import run_full_pf_shortlist_prices as F
import run_truncated_p7_prices as T

MODULE = 'scripts/run_pf_m0_prices.py'
STUDY = 'docs/research/d_pf_m0_direct_prices_20261115'
SHORT_STUDY = 'docs/research/d_pf_m1_m2_direct_prices_20261115'
ALIGN = dict(commit='0b4b2d941cd669aaa45231e1236bc2d3158a9d12', path='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
BASELINE = '0a35101914c6833244b6e0bad7bc2767a49b5a47'
ROUTES = dict(FULL_PF=F, TRUNCATED_PF=T)
TARGETS = {'FULL_PF': {'2nd': '5669f142d693eee92583', '4th': '03df026eb51a56241eef', '4th(m5_best)': '715f0d9f996ec0e62951'},
           'TRUNCATED_PF': {'2nd': 'd2da7de820833a4c6334', '4th': '6c100b1fcb67060431a8', '4th(m5_best)': 'fc389f45fb639698ae84'}}
SOURCES = (MODULE, F.CALLER, T.CALLER, S.MODULE, 'scripts/pf_native_price_fast.py', 'scripts/h46_b3_fast_fingerprint.py')
RAW = ('RESULT.json', 'PRICES.json', 'SIDECARS_UNBOUND.json', 'IMPLEMENTATION_EVIDENCE.json')
FIELDS = ('system', 'method', 'prefix', 'formula', 'T', 'q', 'delta', 'R_outer', 'K')


def check(ok, message):
    if not ok: raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, default=str); f.write('\n')


def read(ref):
    return json.loads(S.read_pinned(ROOT, ref))


def scope(rounds):
    check(tuple(rounds) in ((0,), (1, 2)), 'Only original m0 or fixed m1/m2 scope')
    return ('m0_direct/run01', STUDY, 'M0_DIRECT_ONLY') if tuple(rounds) == (0,) else ('m1_m2_direct/run01', SHORT_STUDY, 'M1_M2_DIRECT_ONLY')


def paths(route, rounds=(0,)):
    m = ROUTES[route]
    suffix, _, _ = scope(rounds)
    return m.OUTPUT_ROOT + '/' + suffix, m.PRIVATE_ROOT / suffix


def selection(rounds=(0,)):
    _, _, stage = scope(rounds)
    alignment = read(ALIGN); chosen = {}; selected = []
    for route, module in ROUTES.items():
        chosen[route] = []
        for p in module.plans(ROOT):
            cells = []
            for m in rounds:
                req = next(r for r in p['plan']['pending']['requests'] if r['m'] == m)
                c = req['physical_candidate']
                T = (.1, .2, .4)[m]; q = 2 if m == 2 and p['formula'] == '2nd' else 1
                check(c['T'].hex() == T.hex() and c['delta'].hex() == (T/q).hex() and c['q'] == q, 'Foreign short-round T/q/actual width')
                if m == 0: check(c['id'] == TARGETS[route][p['formula']], 'Original m0 physical ID differs')
                check(c['prefix'] == (10 if route == 'FULL_PF' else 7), 'Foreign prefix')
                for axis in ('X', 'Y'):
                    matches = [(i, a) for i, a in enumerate(alignment['rows']) if (a['configuration_id'], a['m'], a['axis']) == (p['configuration_id'], m, axis)]
                    check(len(matches) == 1, 'Ambiguous alignment axis')
                    i, a = matches[0]
                    check(a['physical_candidate'] == c and a['structure']['physical'] == req['structure'] and a['structure']['canonical'] == p['plan']['configuration']['rounds'][m]['structure'], 'Physical/signed schedule differs')
                    selected.append(dict(route=route, formula=p['formula'], configuration_id=p['configuration_id'], physical_id=c['id'], physical_candidate=c,
                                         m=m, axis=axis, alignment=dict(ALIGN, selector=['rows', i])))
                cells.append(dict(m=m, formula=p['formula'], row=module.native_row(c), role=stage))
            chosen[route].append(dict(p, cells=cells))
    check(len(selected) == 12*len(rounds), 'Exact fixed axis count required')
    return chosen, selected


def duplicate_check(rounds=(0,), *, selected=None):
    """Scoped metadata delta, including aliases with different configuration names.

    A finding is held for source/axis review; it is never silently recompiled.
    No new science is called by this function.
    """
    if selected is None: _, selected = selection(rounds)
    else: selected = deepcopy(selected)  # Other fixed callers reuse this metadata audit only.
    scans = []; found = {r['physical_id']: [] for r in selected}
    def physical(c): return tuple(c.get(k) for k in FIELDS)
    targets = {physical(r['physical_candidate']): r['physical_id'] for r in selected}
    for name in ('exploration_costs', 'extension_parallel_costs', 'confirmation_costs', 'fixed_delta_exploration_costs', 'reused_compiled_prices'):
        ref = dict(commit='155529e20a8fac39e00786c40b57830b71203c4f', path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'+name+'.json.gz')
        data = json.loads(gzip.decompress(S.read_pinned(ROOT, ref)))
        hits = []
        for key, v in data.items():
            ident = targets.get(physical(v['candidate']))
            if ident:
                hit = dict(source=ref, key=key, trial_files=v.get('trial_files'), candidate=v['candidate'])
                hits.append(hit); found[ident].append(hit)
        scans.append(dict(source=ref, entries=len(data), hits=hits))
    catalog = read(S.CATALOG)['rows']; hits = []
    for v in catalog:
        ident = targets.get(physical(v['physical_candidate']))
        if ident and v.get('prices'):
            hit = dict(source=S.CATALOG, old_physical_id=v['physical_candidate_id'], receipts=v['prices'])
            hits.append(hit); found[ident].append(hit)
    scans.append(dict(source=S.CATALOG, entries=len(catalog), hits=hits))
    exact = []
    for r in selected[::2]:
        ident = r['physical_id']
        roots = [Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/cost') / c / ident / '0/result.json' for c in ('EXPLORATION', 'CONFIRMATION')]
        for route, module in ROUTES.items():
            for suffix in ('calibration/run01', 'holdout/run01', 'm9_holdout/run01', 'm0_direct/run01', 'm1_m2_direct/run01'):
                for base in (ROOT / module.OUTPUT_ROOT, module.PRIVATE_ROOT):
                    p = base / suffix / 'RESULT.json'
                    if p.exists():
                        for v in json.loads(p.read_text()).get('results', []):
                            if physical(v['cell']['row']) == physical(r['physical_candidate']):
                                found[ident].append(dict(existing_route_result=str(p)))
        for p in roots:
            exact.append(dict(path=str(p), exists=p.exists()))
            if p.exists(): found[ident].append(dict(trial=str(p)))
    jobs = S.active_price_jobs()
    # Extend the existing runner-name check to this exact m0 caller.
    for pid in subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid='], text=True).split():
        if int(pid) == os.getpid(): continue
        try: a = Path('/proc', pid, 'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, PermissionError): continue
        if a and Path(a[0]).name.startswith('python') and any(Path(x).name == Path(MODULE).name for x in a[1:]) and '--child' in a:
            try:
                job_path = Path(a[a.index('--child')+1]); job = json.loads(job_path.read_text())
                ids = [c['row']['id'] for c in job['cells']]
            except (OSError, ValueError, KeyError, IndexError): ids = None
            jobs.append(dict(pid=int(pid), physical_ids=ids, caller=MODULE))
    for ident in found:
        found[ident] += [dict(running_job=j) for j in jobs if j['physical_ids'] is None or ident in j['physical_ids']]
    for r in selected:
        r.update(action='HELD' if found[r['physical_id']] else 'NEW_COMPILE_MISSING_CONFIRMED', findings=found[r['physical_id']])
    prior = read(dict(commit=BASELINE, path='docs/research/d_pf_stage2_holdout_20261115/DUPLICATE_AND_INPUT_PREFLIGHT.json'))
    inputs = {}
    for name, v in prior['input_bytes_rechecked'].items():
        raw = Path(v['path']).read_bytes(); check(hashlib.sha256(raw).hexdigest() == v['sha256'] and len(raw) == v['bytes'], 'Saved input changed')
        inputs[name] = v
    return dict(status='METADATA_DUPLICATE_DELTA', baseline=BASELINE,
                continued_from=dict(commit='c95353a44b4ce99ed86e949e408d522d6ad7e7fe', path='docs/research/d_pf_m0_direct_prices_20261115/run01/PREFLIGHT.json') if tuple(rounds)==(1,2) else None,
                rows=selected, scans=scans, exact_private_paths=exact,
                running_jobs=jobs, inputs=inputs, coverage_limit='Known carriers/catalog, prior legacy audit, exact private paths and relevant user jobs; unlisted archives/other servers UNKNOWN',
                legacy_audit=dict(commit='bcb67b2b2bb78776db1dd8d0625a3facb37cd298', path='docs/research/d_pf_allround_price_coverage_plan_20261115/COVERAGE_DELTA.json'))


def event(path, stage, state, **extra):
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(stage=stage, state=state, monotonic=time.monotonic(), pid=os.getpid(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, **extra))+'\n'); f.flush()


@contextmanager
def span(path, stage):
    start = time.monotonic(); event(path, stage, 'START')
    try: yield
    except BaseException as e:
        event(path, stage, 'ERROR', seconds=time.monotonic()-start, error=type(e).__name__+': '+str(e)); raise
    else: event(path, stage, 'END', seconds=time.monotonic()-start)


@contextmanager
def observed_native(native, path):
    """Observe actual calls inside accelerated_native; restore on every exit."""
    import h46_b3_fast_fingerprint as fast
    import trotterlib.df_trotter.circuit as circuit
    from trottertracks.resource_applicability import ax2b_native_df_v5 as wrapper
    saved = []
    def wrap(obj, key, stage):
        original = getattr(obj, key); saved.append((obj, key, original))
        def call(*a, **kw):
            with span(path, stage): return original(*a, **kw)
        setattr(obj, key, call)
    for obj, key, stage in ((native, 'compose', 'compose'), (native, 'compile_cached', 'compile_and_archive'),
                             (fast.FingerprintSession, 'fingerprint', 'fingerprint'), (circuit, 'simulate_statevector', 'statevector_gate'),
                             (wrapper, 'build_native_hadamard_wrapper', 'measured_wrapper')):
        wrap(obj, key, stage)
    try: yield
    finally:
        for obj, key, original in reversed(saved): setattr(obj, key, original)


def worker(cell, output, progress):
    native = S._RUN['native']
    import pf_native_price_fast as fast
    with fast.accelerated_native(native), observed_native(native, progress):
        # _cell must record through the observed compiler, not its old alias.
        S._RUN['compile'] = native.compile_cached
        with span(progress, 'cell_trial'):
            result = S._cell(cell)
    result['worker_pid'] = os.getpid()
    write(output, result)


def processes(cells, directory, workers):
    """Same existing _cell worker, public multiprocessing exit/signal observation."""
    pending = list(enumerate(cells)); running = []; results = []; stopped = False
    while pending or running:
        while pending and len(running) < workers and not stopped:
            i, cell = pending.pop(0); out = directory / ('cell_%d.json' % i)
            p = mp.get_context('fork').Process(target=worker, args=(cell, out, directory / ('progress_%d.jsonl' % i)))
            p.start(); running.append((p, cell, out))
        if not running: break
        for item in list(running):
            p, cell, out = item
            if p.is_alive(): continue
            p.join(); r = json.loads(out.read_text()) if out.exists() else dict(cell=cell, ok=False, error='Worker output missing')
            r.update(worker_exit_code=p.exitcode, worker_signal=-p.exitcode if p.exitcode < 0 else None)
            r['ok'] = r['ok'] and p.exitcode == 0; results.append(r); running.remove(item)
            if not r['ok']: stopped = True
        if running: time.sleep(.1)
    return results, dict(workers=workers, cells=len(cells), started=len(results), not_started_after_failure=len(pending))


def child(job_path, *, validate_rounds=scope):
    job = json.loads(Path(job_path).read_text()); route = job['route']; module = ROUTES[route]
    validate_rounds(job.get('rounds', (0,)))
    # Install before source_gate's metadata imports can resolve the local package.
    import h46_long_time_signal as signal
    signal.install()
    # Stdlib comparison here: metadata _meta() would resolve local packages
    # before the saved native provider, defeating install() in this process.
    for path in SOURCES:
        check((ROOT/path).read_bytes() == subprocess.check_output(['git', 'show', job['execution_SOURCE']+':'+path], cwd=ROOT), 'Child SOURCE changed: '+path)
    for ref in (T.NATIVE, T.WRAPPER):
        check((ROOT/ref['path']).read_bytes() == subprocess.check_output(['git', 'show', ref['commit']+':'+ref['path']], cwd=ROOT), 'Native SOURCE changed')
    progress = Path(job['private']) / 'context_progress.jsonl'; start = time.monotonic()
    result = dict(results=[], resources=None)
    try:
        with span(progress, 'install_and_context'):
            T.provider_checks()
            import h46_long_time_native as native
            native.PRIVATE, native.PUBLIC = Path(job['private']), Path(job['public'])
            prefix = 10 if route == 'FULL_PF' else 7
            ctx = native.native_context('H4', (prefix,)); prep = ctx.preparations[prefix]
            scalar = ctx.ham.constant + (prep.extracted_identity_coefficient if route == 'TRUNCATED_PF' else 0.)
            check(scalar.hex() == job['expected_scalar'].hex(), 'Saved total scalar mismatch')
            check(prep.preparation_hash == job['preparation_hash'] and tuple(prep.deterministic_fragment_indices) == tuple(range(prefix)), 'Preparation/fragment order mismatch')
            native.CONTEXT['H4'] = ctx
        original = F.full_trial if route == 'FULL_PF' else T.checked_trial
        def checked(native, ctx, row, cohort, **kw):
            trial, facts, ok = original(native, ctx, row, cohort, **kw)
            required = {'low_control', 'high_control_leakage', 'independent_occupation_vector', 'Hadamard_cosine', 'Hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'}
            return trial, facts, ok and required <= {c['check'] for c in trial['checks']} and trial['cache_hits'] == 0
        S._RUN.update(native=native, ctx=ctx, trial=checked, cohort=job.get('cohort',module.COHORT),
                      compile=native.compile_cached, kw=dict(expected_tail_scalar=job['tail_scalar']) if route == 'TRUNCATED_PF' else {})
        result['results'], result['resources'] = processes(job['cells'], Path(job['private']), job['workers'])
        result['context_gate'] = dict(scalar=scalar, scalar_hex=scalar.hex(), preparation_hash=prep.preparation_hash, fragment_order=list(prep.deterministic_fragment_indices))
    except BaseException as e:
        result['error'] = type(e).__name__+': '+str(e)
    result.update(wall_seconds=time.monotonic()-start, context_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(Path(job['private'])/'child.json', result)
    return 0 if len(result['results']) == len(job['cells']) and all(r['ok'] for r in result['results']) else 2


def execute(rounds=(0,)):
    _, study, stage = scope(rounds)
    source = S.source_gate(ROOT, SOURCES); templates, _ = selection(rounds); preflight = duplicate_check(rounds)
    out = ROOT / study / 'run01'; out.mkdir(parents=True, exist_ok=False)
    write(out/'PREFLIGHT.json', preflight)
    from pf_common_run_contract import resources, workers
    obs = resources(); eligible = {r: [c for p in templates[r] for c in p['cells'] if all(x['action'] == 'NEW_COMPILE_MISSING_CONFIRMED' for x in preflight['rows'] if x['physical_id'] == c['row']['id'])] for r in ROUTES}
    # At present all axes are missing. Findings are never bypassed by this caller.
    # Per-axis reuse/hold requires evidence review before selecting a fresh pair.
    capacity = workers(obs, sum(map(len, eligible.values())), 2*2**30, 100*2**20)
    check(capacity >= len([v for v in eligible.values() if v]), 'Insufficient available route capacity; no science started')
    remaining = capacity; children = []; started = time.monotonic()
    for route, module in ROUTES.items():
        if not eligible[route]: continue
        rel, private = paths(route, rounds); public = ROOT / rel
        public.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
        # Already-published scalar/prep facts, no new numerical derivation.
        scalar_ref = dict(commit=BASELINE, path=module.OUTPUT_ROOT+'/holdout/run01/RESULT.json')
        check((ROOT/scalar_ref['path']).read_bytes() == S.read_pinned(ROOT, scalar_ref), 'Saved scalar/preparation evidence changed')
        old = read(scalar_ref)['results'][0]
        facts = old['facts']; n = min(len(eligible[route]), max(1, remaining//(2 if route == 'FULL_PF' and eligible['TRUNCATED_PF'] else 1))); remaining -= n
        job = dict(route=route, rounds=list(rounds), cells=eligible[route], public=str(public/'child_public'), private=str(private), workers=n,
                   expected_scalar=facts['scalar'], tail_scalar=facts['tail_identity'], preparation_hash=old['trial']['preparation_hash'], scalar_preparation_source=dict(scalar_ref, selector=['results', 0]), execution_SOURCE=source)
        write(private/'job.json', job)
        log = (private/'child.log').open('x')
        p = subprocess.Popen([sys.executable, '-B', str(ROOT/MODULE), '--child', str(private/'job.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append((route, p, log, job))
    reports = {}
    for route, p, log, job in children:
        code = p.wait(); log.close(); module = ROUTES[route]; rel, private = paths(route, rounds); public = ROOT/rel
        data = json.loads((private/'child.json').read_text()) if (private/'child.json').exists() else dict(results=[], error='Missing child output')
        prices, sides, proofs = [], [], {}
        price_source = dict(module.stage_spec()['source'], commit=source, caller=MODULE, underlying_caller=module.CALLER, stage=stage, route=route)
        for r in data['results']:
            if not r['ok']: continue
            plan = next(v for v in templates[route] if v['formula'] == r['cell']['formula'])['plan']
            a, b, c = module.price_bindings(ROOT, plan, r['cell'], r['trial'], price_source=price_source, evidence_commit=S.PLACEHOLDER, evidence_path=rel+'/IMPLEMENTATION_EVIDENCE.json')
            for price in a: price['diagnostic'].update(stage=stage, compiled_wrapper=r['wrappers']['cosine' if price['axis'] == 'X' else 'sine'])
            prices += a; sides += b; proofs.update(c)
        for name, value in zip(RAW[1:], (prices, sides, proofs)): write(public/name, value)
        data.update(source=price_source, child_exit_code=code, child_signal=-code if code < 0 else None, valid_axis_records_before_C_guard=len(prices), mandatory_stop=True)
        write(public/'RESULT.json', data); reports[route] = dict(child_exit_code=code, records=len(prices), output=rel)
    write(out/'ACQUISITION.json', dict(execution_SOURCE=source, routes=reports, resources=obs, capacity=capacity, elapsed_seconds=time.monotonic()-started,
          fit_calls=0, model_predictions=0, scientific_retries=0, mandatory_stop=True))
    print(json.dumps(reports))


def bind(route, publication, rounds=(0,)):
    _, _, stage = scope(rounds)
    rel, _ = paths(route, rounds); out = ROOT/rel; module = ROUTES[route]
    fixed = S.committed(ROOT, rel, RAW, commit=publication)
    templates, _ = selection(rounds); T._meta()
    from trottertracks import price_implementation_guard as G
    prices = json.loads((out/'PRICES.json').read_text()); sides = json.loads((out/'SIDECARS_UNBOUND.json').read_text())
    for side in sides:
        check(side['implementation_evidence']['commit'] == S.PLACEHOLDER, 'Already bound'); side['implementation_evidence']['commit'] = publication
    checks = []
    for p in templates[route]:
        cfg = p['plan']['configuration']; mine = [v for v in prices if v['configuration_id'] == cfg['id']]
        if not mine: continue
        check(all(v['m'] in rounds for v in mine), 'Foreign round')
        policy = dict(id='PF_M0_DIRECT_IMPLEMENTATION_ONLY_V1' if tuple(rounds)==(0,) else 'PF_M1_M2_DIRECT_IMPLEMENTATION_ONLY_V1', source=S.GUARD_SOURCE, profile=cfg['spec']['profile'], by_method={route:dict(native_policy=T.NATIVE_POLICY, rte_rotation_lowering=T.NO_RTE, scalar_quantity=module.SCALAR_QUANTITY)})
        cohorts = {(cfg['rounds'][v['m']]['candidate']['id'], v['axis']): module.COHORT for v in mine}
        checks.append(dict(configuration_id=cfg['id'], guard=G.check(ROOT, cfg, mine, cohorts=cohorts, bindings=sides, policy=policy)))
    check(sum(len(v['guard']['checked']) for v in checks) == len(prices), 'Unchecked price')
    write(out/'SIDECARS.json', sides); write(out/'GUARD_CHECK.json', dict(evidence=fixed, checks=checks, model_qualified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata-output', type=Path); p.add_argument('--execute-approved-m0', action='store_true')
    p.add_argument('--short-rounds', action='store_true', help='Fixed m1/m2 metadata/binding scope')
    p.add_argument('--execute-approved-m1-m2', action='store_true')
    p.add_argument('--child', type=Path); p.add_argument('--bind-route', choices=ROUTES); p.add_argument('--publication')
    a = p.parse_args()
    check(not (a.execute_approved_m0 and (a.short_rounds or a.execute_approved_m1_m2)), 'Do not mix m0 and m1/m2 scope')
    rounds = (1, 2) if a.short_rounds or a.execute_approved_m1_m2 else (0,)
    if a.metadata_output: write(a.metadata_output, duplicate_check(rounds))
    elif a.execute_approved_m1_m2: execute(rounds=(1, 2))
    elif a.execute_approved_m0: execute()
    elif a.child: return child(a.child)
    elif a.bind_route: bind(a.bind_route, a.publication, rounds)
    else: p.error('Select metadata, approved m0 execution or receipt binding')
    return 0


if __name__ == '__main__': sys.exit(main())
