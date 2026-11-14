"""Read-only revalidation of the saved P-1 run02 with the corrected scalar/event phase gate.

The original run02 (caller 72d4f4ff2, results cd56422a8) stays `STOP_WIRING_CHECK`;
this is a separate, after-the-fact acceptance under a new SOURCE. Nothing is
sampled, built, simulated or compiled: the saved X/Y wrapper/compiled QPY, events,
compile records and child result are only read, hashed and counted.

Modes (no retry; mandatory STOP after each):
  (default)                       metadata-only: originals present, public bytes = cd56422a8 blobs
  --revalidate                    + provider-root child: QPY/event/compile-record correspondence and the
                                  corrected gate; writes run02_revalidation01/REVALIDATION.json,
                                  REQUEST_PHASES.json and, only on PASS, IMPLEMENTATION_EVIDENCE.json
  --issue --evidence-commit SHA   on the PUBLISHED revalidation commit: price records, sidecars
                                  (implementation_evidence -> that commit) and C's fixed guard
                                  (c628e9e8 blob, executed unchanged); writes PRICES/SIDECARS/GUARD_CHECK

The price is a single-path direct compiled value (finite cohort, n_paths=1): never
E[C], a mean event price, a model calibration point, P-2 input or G.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import run_full_h_kappa2_p1 as P1  # noqa: E402  (imports no project package at import time)

SCRIPT = 'scripts/revalidate_full_h_kappa2_p1_run02.py'
STUDY = 'docs/research/d_full_h_kappa2_p1_results_20261114'
ORIGIN_COMMIT = 'cd56422a8cea155b0c0d88271a355f4e34f44a75'
RUN02_SOURCE_COMMIT = '72d4f4ff268d3281895328bfe2c621ec88c1f580'
ORIGINAL_CALLER = '04bead115233641233ae1fc409b2c4d89ebc2483'
RUN02 = STUDY + '/run02'
PUBLIC_ORIGINALS = (RUN02 + '/SOURCE.json', RUN02 + '/P1_RESULT.json', STUDY + '/RUN02_SCALAR_ONCE_DIAGNOSIS.json')
PRIVATE_RUN02 = Path('/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run02')
OUT = STUDY + '/run02_revalidation01'
PRIVATE_OUT = Path('/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run02_revalidation01')
CELL = 'cost/' + P1.COHORT + '/FULL_H_4de6130f0d3a66e5d695/0'
C_GUARD = dict(commit='c628e9e8447c11c984064b922cbce6c52dc72d48', path='src/trottertracks/price_implementation_guard.py')
C_GUARD_DEPENDENCY = 'src/trottertracks/baseline_signal_pipeline.py'
D_PLAN = dict(commit='5ccc1dfc341236d3fce2f745e7e9023447a027a7', path='docs/research/d_kappa_s1_price_builders_20261114/NEXT_EXECUTION.md')
C_ROUTES = dict(commit=C_GUARD['commit'], path='docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md')
POLICY_ID = 'D_FULL_H_KAPPA2_NEW_ZZ_P1_IMPLEMENTATION_ACCEPTANCE_V1'
TRIAL_GATES = ('low_control', 'high_control_leakage', 'independent_occupation_vector', 'Hadamard_cosine',
               'compiled_vector_cosine', 'Hadamard_sine', 'compiled_vector_sine')
AXIS = {'cosine': 'X', 'sine': 'Y'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def git_json(commit, path):
    return json.loads(P1.blob(commit, path))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str); f.write('\n')


def private_files():
    """Saved run02 private originals (read only). The compile cache is keyed by wrapper fingerprint."""
    cache = PRIVATE_RUN02 / 'compiled_cache'
    fps = sorted(p.name[:-len('_wrapper.qpy.gz')] for p in cache.glob('*_wrapper.qpy.gz'))
    files = {'job.json': PRIVATE_RUN02 / 'job.json', 'child.json': PRIVATE_RUN02 / 'child.json',
             'H4_context.json': PRIVATE_RUN02 / 'H4_context.json',
             'events_PRIVATE.json': PRIVATE_RUN02 / CELL / 'events_PRIVATE.json', 'result.json': PRIVATE_RUN02 / CELL / 'result.json'}
    for fp in fps:
        for suffix in ('.json', '_wrapper.qpy.gz', '_compiled.qpy.gz'):
            files['compiled_cache/' + fp + suffix] = cache / (fp + suffix)
    return files, fps


def originals(root=ROOT):
    """Presence and bytes of the originals; public bytes must equal the published cd56422a8 blobs."""
    public = {}
    for path in PUBLIC_ORIGINALS:
        data = (Path(root) / path).read_bytes()
        if data != P1.blob(ORIGIN_COMMIT, path):
            raise ValueError('Public run02 original differs from ' + ORIGIN_COMMIT + ': ' + path)
        public[path] = hashlib.sha256(data).hexdigest()
    files, fps = private_files()
    missing = [k for k, p in files.items() if not p.is_file()]
    if missing or len(fps) != 2:
        raise ValueError('Saved run02 private originals missing/incomplete: ' + json.dumps(dict(missing=missing, fingerprints=fps)))
    return dict(public_commit=ORIGIN_COMMIT, public_sha256=public,
                private_root=str(PRIVATE_RUN02), private_sha256={k: sha(p) for k, p in files.items()}, fingerprints=fps)


# ---------------------------------------------------------------- child (provider root first)

def child_gate(job_path):
    """Provider-root process: QPY/event/compile-record correspondence and the corrected gate. Read only."""
    job = read(job_path)
    import h46_long_time_signal as signal
    signal.install()
    if any(name.split('.')[0] == 'trottertracks' and name != 'trottertracks' for name in list(sys.modules)):
        raise ValueError('Project trottertracks imported before the provider root')
    provider_files = P1.provider_checks()
    from qiskit import qpy
    provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
    costs = __import__('trotterlib.rte_compiled_cost', fromlist=['x'])
    import h46_long_time_native as native
    files, fps = private_files()
    events = read(files['events_PRIVATE.json'])
    trial = read(files['result.json'])
    context = read(files['H4_context.json'])
    requests_events = events['events']
    T, n = job['row']['T'], None

    def load(path):
        return qpy.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())))[0]

    wrappers, compiled, records, axes = {}, {}, {}, {}
    for fp in fps:
        wrapper = load(files['compiled_cache/' + fp + '_wrapper.qpy.gz'])
        n = wrapper.num_qubits - 1
        anc = [i.operation.name for i in wrapper.data if [wrapper.find_bit(q).index for q in i.qubits] == [n]]
        axis = 'sine' if 'sdg' in anc else 'cosine'
        if axis in wrappers:
            raise ValueError('Two wrappers for one axis')
        wrappers[axis], axes[axis] = wrapper, fp
        compiled[axis] = load(files['compiled_cache/' + fp + '_compiled.qpy.gz'])
        records[axis] = read(files['compiled_cache/' + fp + '.json'])
    checks = {}
    for axis, fp in axes.items():
        w, c, rec = wrappers[axis], compiled[axis], records[axis]
        counts = dict(costs._gate_counts(c))
        metrics = dict(rz_count=counts.get('rz', 0), rz_depth=costs._filtered_depth(c, 'rz'), cx_count=counts.get('cx', 0),
                       cx_depth=costs._filtered_depth(c, 'cx'), total_depth=int(c.depth() or 0), circuit_size=int(c.size()))
        start, stop = P1.wrapper_evolution_bounds(w, n, axis)
        ops = P1.op_counts(w)
        checks[axis] = dict(
            fingerprint=fp, wrapper_fingerprint_recomputed=costs.canonical_qiskit_circuit_fingerprint(w),
            record_fingerprint=rec['fingerprint'], record_compiler=rec['compiler'], native_compiler_hash=native.COMPILER_HASH,
            record_metrics=rec['metrics'], compiled_metrics_recounted=metrics, trial_metrics=trial['prices'][axis],
            compiled_qubits=c.num_qubits, wrapper_qubits=w.num_qubits, compiled_basis=sorted(counts),
            wrapper_instructions=len(w.data), evolution_instructions=stop - start,
            wrapper_crzz=ops.get('crzz', 0), wrapper_crz=ops.get('crz', 0), wrapper_cz=ops.get('cz', 0))
    expected_scalar = context['constant'] + context['extracted_scalar']
    gate = P1.xy_phase_gate(wrappers, n, requests_events, trial['scalar'], expected_scalar, T, provider=provider)
    # Request table: events, sampled phases, provider-rule angle and the actual instruction on each axis.
    table = []
    for r, batch in enumerate(requests_events):
        views = [P1._event_view(e) for e in batch]
        row = dict(request=r, events=len(batch), orders={str(k): sum(v.taylor_order == k for v in views) for k in (0, 2)},
                   phase_minus_one_events=[k for k, v in enumerate(views) if v.phase == -1],
                   product_sign_minus_one=sum(a.role == 'product' and a.coefficient_sign == -1 for v in views for a in v.apps),
                   identity_applications=sum(a.is_identity for v in views for a in v.apps),
                   provider_rule_angle=P1.request_phase(batch, provider), actual={})
        for axis, g in gate['axes'].items():
            hit = [p for p in g['event_phases'] if p['request'] == r]
            row['actual'][AXIS[axis]] = None if not hit else dict(index=hit[0]['index'], angle=hit[0]['angle'])
        table.append(row)
    one_body = sum(app['component_id'].startswith('full-h-one-body:') for batch in requests_events for e in batch
                   for app in e['application_sequence'])
    return dict(provider_modules=provider_files, axes={AXIS[a]: v for a, v in checks.items()},
                phase_gate=gate, request_phases=table, one_body_applications_from_events=one_body,
                events=dict(cohort=events['cohort'], seeds=events['seeds'], conditional=events['conditional'],
                            per_request=[len(b) for b in requests_events],
                            orders={str(k): sum(e['taylor_order'] == k for b in requests_events for e in b) for k in (0, 2)}),
                scalar=dict(trial=trial['scalar'], context_total=context['scalar'], expected=expected_scalar,
                            constant=context['constant'], extracted=context['extracted_scalar'], T=T,
                            angle=-trial['scalar'] * T),
                trial_status=trial['status'], trial_index=trial['index'], preparation_hash=trial['preparation_hash'],
                context_preparation_hash=context['preparation_hash'],
                science_calls=dict(sampling=0, circuit_build=0, statevector=0, compile=0))


# ---------------------------------------------------------------- parent

def acceptance(root, orig, child, plan):
    """Every acceptance condition, evaluated on the saved originals and the child's read-only checks."""
    prepared, req, row, seeds = plan
    source = read(Path(root) / RUN02 / 'SOURCE.json')
    result = read(Path(root) / RUN02 / 'P1_RESULT.json')
    files, _ = private_files()
    job, child_saved, trial = read(files['job.json']), read(files['child.json']), read(files['result.json'])
    s = source['source']; ck = result['child']['checks']; t = result['child']['trial']
    gates = {g['check']: g for g in t['checks']}
    conditions = dict(
        run02_source=(s['commit'] == RUN02_SOURCE_COMMIT and s['caller'] == P1.CALLER and s['cohort'] == P1.COHORT
                      and all(s[k] == v for k, v in dict(full_h=P1.FULL_H, native=P1.NATIVE, wrapper=P1.WRAPPER,
                                                          provider=P1.PROVIDER, lowering=P1.NEW_ZZ).items())),
        configuration=prepared['plan']['configuration']['id'] == '54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e',
        physical_candidate=(source['row'] == row and job['row'] == row and row['id'] == 'FULL_H_4de6130f0d3a66e5d695'
                            and (row['q'], row['R_outer'], row['s'], row['K'], row['T']) == (8, 71, 568, 2, .8)),
        seeds=source['seeds'] == seeds and child['events']['seeds'] == seeds and child['events']['cohort'] == P1.COHORT,
        events=(child['events']['per_request'] == [71] * 8 and child['events']['conditional'] is False
                and child['events']['orders'] == t['orders']),
        original_status_kept=(result['status'] == 'STOP_WIRING_CHECK' and result['child']['ok'] is False
                              and ck.get('scalar_once') is False),
        child_result_matches_public=child_saved == result['child'] and trial == t,
        trial=(t['status'] == 'PASS' and t['index'] == 0 and set(t['prices']) == {'cosine', 'sine'} and t['cache_hits'] == 0),
        action_gates=(tuple(gates) == TRIAL_GATES and all(g['error'] < g['tolerance'] and not g['phase_alignment']
                                                        for g in gates.values())),
        new_zz=(ck['builder_lowering_id'] == P1.LOWERING_ID and ck['legacy_rzz_control_instructions'] == 0
                and ck['new_crz'] == 568 and all(a['wrapper_crzz'] == 0 and a['wrapper_crz'] == 568 for a in child['axes'].values())),
        parity=(ck['parity']['status'] == 'PASS' and ck['parity']['replaced_controlled_zz'] > 0
                and all(a['evolution_instructions'] == ck['parity']['new_instructions'] for a in child['axes'].values())),
        one_body=ck['one_body_applications'] > 0 and child['one_body_applications_from_events'] == ck['one_body_applications'],
        scalar_value=(child['scalar']['trial'] == child['scalar']['expected'] == ck['scalar'] == ck['scalar_expected']
                      == child['scalar']['context_total'] and t['scalar'] == ck['scalar']),
        preparation=child['preparation_hash'] == child['context_preparation_hash'] == t['preparation_hash'],
        provider_bytes=all(m['sha256'] == ck['provider_modules'][name]['sha256'] for name, m in child['provider_modules'].items())
                       and set(child['provider_modules']) == set(ck['provider_modules']),
        wrapper_fingerprints=all(a['wrapper_fingerprint_recomputed'] == a['fingerprint'] == a['record_fingerprint']
                                 for a in child['axes'].values()),
        published_fingerprints=sorted(a['fingerprint'] for a in child['axes'].values())
                               == sorted(git_json(ORIGIN_COMMIT, PUBLIC_ORIGINALS[2])['wrappers']) == orig['fingerprints'],
        compiler=all(a['record_compiler'] == a['native_compiler_hash'] for a in child['axes'].values()),
        compiled_metrics=all(a['record_metrics'] == a['compiled_metrics_recounted'] == a['trial_metrics']
                             and a['compiled_qubits'] == a['wrapper_qubits'] == 9 for a in child['axes'].values()),
        phase_gate=child['phase_gate']['status'] == 'PASS' and child['phase_gate']['same_controlled_evolution'],
        phase_gate_requests=[r['request'] for r in child['request_phases'] if r['phase_minus_one_events']] == [0, 3, 7]
                            and all(r['actual']['X'] == r['actual']['Y'] for r in child['request_phases']),
        old_gate_false_negative_reproduced=all(g['legacy_scalar_once'] is False for g in child['phase_gate']['axes'].values()),
        no_new_science=child['science_calls'] == dict(sampling=0, circuit_build=0, statevector=0, compile=0))
    return conditions


def price_source(root, orig, child, head):
    s = read(Path(root) / RUN02 / 'SOURCE.json')['source']
    return dict(s, record=RUN02 + '/P1_RESULT.json', record_commit=ORIGIN_COMMIT, origin='SAVED_RUN02_DIRECT_COMPILE',
                compile_records={ax: dict(path=str(PRIVATE_RUN02 / 'compiled_cache' / (a['fingerprint'] + '.json')),
                                          sha256=orig['private_sha256']['compiled_cache/' + a['fingerprint'] + '.json'],
                                          wrapper_fingerprint=a['fingerprint'], compiler=a['record_compiler'])
                                 for ax, a in sorted(child['axes'].items())},
                acceptance=dict(commit=head, script=SCRIPT, caller=P1.CALLER, gate=P1.PHASE_GATE_ID,
                                record=OUT + '/REVALIDATION.json', original_status='STOP_WIRING_CHECK',
                                revalidation='PASS_UNDER_CORRECTED_SCALAR_EVENT_PHASE_GATE'))


def head_commit(root=ROOT):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (SCRIPT, P1.CALLER):
        if P1.blob(commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Revalidation SOURCE bytes differ from HEAD: ' + path)
    return commit


def revalidate(root=ROOT):
    head = head_commit(root)
    started = time.perf_counter()
    orig = originals(root)
    plan = P1.plan_row(root)
    out = Path(root) / OUT; out.mkdir(parents=True, exist_ok=False)
    PRIVATE_OUT.mkdir(parents=True, exist_ok=False)
    job = dict(row=plan[2], result=str(PRIVATE_OUT / 'child_gate.json'))
    (PRIVATE_OUT / 'job.json').write_text(json.dumps(job))
    code = subprocess.call([sys.executable, '-B', str(Path(root) / SCRIPT), '--child-gate', str(PRIVATE_OUT / 'job.json')], cwd=root)
    child = read(job['result']) if code == 0 and Path(job['result']).exists() else None
    record = dict(kind='P1_RUN02_REVALIDATION', original=dict(commit=ORIGIN_COMMIT, run=RUN02, status='STOP_WIRING_CHECK',
                  run_source=RUN02_SOURCE_COMMIT, fixed_caller=ORIGINAL_CALLER, untouched=True),
                  source=dict(commit=head, script=SCRIPT, caller=P1.CALLER, gate=P1.PHASE_GATE_ID),
                  originals=orig, started_utc=datetime.now(timezone.utc).isoformat(), child_exit=code,
                  new_sampling=0, new_compile=0, new_circuit_build=0, new_statevector=0, retry=False)
    if child is None:
        record.update(status='STOP_REVALIDATION_CHILD', conditions=None)
    else:
        conditions = acceptance(root, orig, child, plan)
        failed = [k for k, v in conditions.items() if not v]
        record.update(status='REVALIDATION_PASS' if not failed else 'STOP_REVALIDATION', conditions=conditions, failed=failed,
                      child={k: v for k, v in child.items() if k != 'request_phases'})
        write(out / 'REQUEST_PHASES.json', child['request_phases'])
        if not failed:
            source = price_source(root, orig, child, head)
            record['price_source'] = source
            prices, sides, proofs = P1.price_bindings(root, plan[0], trial_for_price(child), price_source=source,
                                                      evidence_commit='0' * 40, evidence_path=OUT + '/IMPLEMENTATION_EVIDENCE.json')
            write(out / 'IMPLEMENTATION_EVIDENCE.json', proofs)
    record.update(elapsed_seconds=time.perf_counter() - started, mandatory_stop=True)
    write(out / 'REVALIDATION.json', record)
    print(json.dumps(dict(status=record['status'], failed=record.get('failed'), elapsed_seconds=record['elapsed_seconds'])))
    return 0 if record['status'] == 'REVALIDATION_PASS' else 2


def trial_for_price(child):
    """Prices from the verified compile records (= recounted compiled QPY = saved trial); scalar from the trial."""
    return dict(prices={'cosine': deepcopy(child['axes']['X']['record_metrics']),
                        'sine': deepcopy(child['axes']['Y']['record_metrics'])},
                scalar=child['scalar']['trial'])


def c_guard(root=ROOT):
    """C's fixed price_implementation_guard (c628e9e8 blob, unchanged) in this tree's package."""
    name = 'trottertracks._price_implementation_guard_c628e9e8'
    if name not in sys.modules:
        if str(Path(root) / 'src') not in sys.path:
            sys.path.insert(0, str(Path(root) / 'src'))
        import trottertracks  # noqa: F401
        if P1.blob(C_GUARD['commit'], C_GUARD_DEPENDENCY) != (Path(root) / C_GUARD_DEPENDENCY).read_bytes():
            raise ValueError("C guard dependency differs from this tree's baseline_signal_pipeline")
        module = types.ModuleType(name); module.__package__ = 'trottertracks'
        module.__file__ = C_GUARD['commit'] + ':' + C_GUARD['path']; sys.modules[name] = module
        exec(compile(P1.blob(C_GUARD['commit'], C_GUARD['path']), module.__file__, 'exec'), module.__dict__)
    return sys.modules[name]


def policy(config):
    return dict(id=POLICY_ID, source=dict(plan=D_PLAN, c_routes=C_ROUTES, scope='P-1 implementation acceptance only; '
                'not an adopted comparison scenario'), profile=deepcopy(config['spec']['profile']),
                by_method=dict(FULL_H_RTE=dict(native_policy='H46_LONG_TIME_NATIVE_COMPOSE', rte_rotation_lowering=P1.LOWERING_ID,
                                               scalar_quantity=P1.SCALAR_QUANTITY)))


def issue(root, evidence_commit):
    head_commit(root)
    record = git_json(evidence_commit, OUT + '/REVALIDATION.json')
    proofs_published = git_json(evidence_commit, OUT + '/IMPLEMENTATION_EVIDENCE.json')
    if record['status'] != 'REVALIDATION_PASS' or record.get('failed'):
        raise ValueError('Published revalidation is not PASS; no price is issued')
    acc = record['price_source']['acceptance']['commit']
    for path in (SCRIPT, P1.CALLER):
        if P1.blob(acc, path) != (Path(root) / path).read_bytes():
            raise ValueError('Acceptance SOURCE differs from the executing bytes: ' + path)
    prepared = P1.plan_row(root)[0]
    config = prepared['plan']['configuration']
    child = dict(axes=record['child']['axes'], scalar=record['child']['scalar'])
    evidence_path = OUT + '/IMPLEMENTATION_EVIDENCE.json'
    prices, sides, proofs = P1.price_bindings(root, prepared, trial_for_price(child), price_source=record['price_source'],
                                              evidence_commit=evidence_commit, evidence_path=evidence_path)
    if proofs != proofs_published:
        raise ValueError('Rebuilt implementation assertions differ from the published evidence')
    G = c_guard(root)
    cohorts = {(config['rounds'][P1.M]['candidate']['id'], a): P1.COHORT for a in ('X', 'Y')}
    checked = G.check(root, config, prices, cohorts=cohorts, bindings=sides, policy=policy(config))
    out = Path(root) / OUT
    write(out / 'PRICES.json', prices); write(out / 'SIDECARS.json', sides)
    write(out / 'GUARD_CHECK.json', dict(guard=C_GUARD, guard_dependency=dict(path=C_GUARD_DEPENDENCY, same_bytes=True),
          evidence_commit=evidence_commit, result=checked, cohorts=[[k[0], k[1], v] for k, v in cohorts.items()],
          prices_checked=len(checked['checked']), handoff='C_PRICE_ACCEPTANCE_SINGLE_PATH_FINITE_COHORT',
          not_for=['E[C]', 'mean event price', 'P-2 model calibration point', 'G', 'method comparison']))
    print(json.dumps(dict(status=checked['status'], checked=len(checked['checked']))))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--revalidate', action='store_true')
    parser.add_argument('--issue', action='store_true')
    parser.add_argument('--evidence-commit')
    parser.add_argument('--child-gate', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_gate:
        job = read(args.child_gate)
        Path(job['result']).write_text(json.dumps(child_gate(args.child_gate), indent=1, default=str))
        return 0
    if args.revalidate:
        return revalidate(ROOT)
    if args.issue:
        if not args.evidence_commit or len(args.evidence_commit) != 40:
            raise SystemExit('--issue needs the full 40-hex published revalidation commit')
        return issue(ROOT, args.evidence_commit)
    orig = originals(ROOT); plan = P1.plan_row(ROOT)
    print(json.dumps(dict(status='PREFLIGHT_METADATA_ONLY', originals=orig, row=plan[2], seeds=plan[3], output=OUT,
                          private_output=str(PRIVATE_OUT)), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
