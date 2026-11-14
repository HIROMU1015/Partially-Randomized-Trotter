"""CALIBRATION / HOLDOUT stage split for the deterministic Full PF and Truncated p7 price callers.

Stage selection, gates and packaging only. Cells, circuits, compiler, scalar,
index-0 gates and price records come unchanged from the two callers
(`plans`, `checked_trial`/`full_trial`, `price_bindings`, `fit_inputs`).

- C's CELL_PLAN (receipt-admission bundle 3368805575) is the canonical role
  table: every regenerated cell must equal its two plan rows (route, formula,
  actual width, configuration, m, axis, physical candidate, action, features,
  model group) before a stage is selected. CALIBRATION = m4/m5, HOLDOUT = m6.
- Each stage writes its own public/private namespace (fresh directories only).
- HOLDOUT requires the published, unchanged CALIBRATION outputs and C's frozen
  models (one per (route, formula, actual width, axis), fitted on exactly the
  published m4/m5 inputs); it re-checks the CALIBRATION files after the run.
- C inputs are grouped by model group; mixing groups or stages is rejected.
- The saved Full PF M5/m4 X/Y receipts admitted by C are referenced, never
  recompiled or moved to a new cohort.
- Sidecars carry an evidence placeholder until the stage outputs are published;
  `bind` then fills the published commit and runs C's guard with an explicit
  technical-compatibility probe policy (not an adopted comparison policy).

No fit, prediction, holdout judgement, N or G is computed here.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

MODULE = 'scripts/pf_price_stages.py'
ADMISSION_DIR = 'docs/research/m5_m4_receipt_admission_20261114'
BUNDLE = '3368805575d459d948e3afe7b1e54b4814f64129'      # C fixed receipt bundle (CELL_PLAN, saved inputs/prices)
ADMISSION = '2fa88065318f0f2ab13937ef610828200b35874c'   # C publication (bound sidecars, guard check)
CELL_PLAN = dict(commit=BUNDLE, path=ADMISSION_DIR + '/CELL_PLAN.json')
GUARD_SOURCE = dict(commit='c628e9e8447c11c984064b922cbce6c52dc72d48', path='src/trottertracks/price_implementation_guard.py')
STAGES = dict(CALIBRATION=(4, 5), HOLDOUT=(6,))
NEW_WRAPPERS = {('FULL_PF', 'CALIBRATION'): 10, ('FULL_PF', 'HOLDOUT'): 6,
                ('TRUNCATED_PF', 'CALIBRATION'): 12, ('TRUNCATED_PF', 'HOLDOUT'): 6}
NEW = 'NEW_COMPILE'
SAVED = 'SAVED_RECEIPT_FOR_C_ACCEPTANCE'
SAVED_ID = '0cd1527c80f7b9841a83'
PLACEHOLDER = 'FILL_WITH_PUBLISHED_COMMIT'
STAGE_FILES = ('RESULT.json', 'PRICES.json', 'SIDECARS_UNBOUND.json', 'IMPLEMENTATION_EVIDENCE.json', 'C_INPUTS.json')
BOUND_FILES = ('SIDECARS.json', 'GUARD_CHECK.json')
MODEL = 'PRICE_LINEAR_FEATURE_MODEL_V1'
PROBE = 'TECHNICAL_COMPATIBILITY_PROBE_ONLY_NOT_ADOPTED'
FEATURES = ['one', 'q']


def _T3():
    import run_truncated_p7_prices as T3
    return T3


def read_pinned(root, ref):
    C, _ = _T3()._meta()
    return C.blob(root, ref['commit'], ref['path'])


def key(group):
    return '|'.join(group)


def paths(spec, stage):
    if stage not in STAGES:
        raise ValueError('Unknown stage: %r' % (stage,))
    return '%s/%s/run01' % (spec['output'], stage.lower()), Path(spec['private']) / stage.lower() / 'run01'


def select(root, templates, route, stage):
    """Cells of one stage. The role comes from C's CELL_PLAN; any disagreement is a STOP before compile."""
    if stage not in STAGES:
        raise ValueError('Unknown stage: %r' % (stage,))
    rows = [r for r in json.loads(read_pinned(root, CELL_PLAN)) if r['route'] == route]
    used, out = 0, []
    for p in templates:
        cells = []
        for c in p['cells']:
            mine = sorted((r for r in rows if (r['formula'], r['width_over_h'], r['configuration_id'], r['m'])
                           == (p['formula'], p['width_over_h'], p['configuration_id'], c['m'])), key=lambda r: r['axis'])
            role = mine[0]['role'] if mine else None
            want = [dict(route=route, formula=p['formula'], width_over_h=p['width_over_h'],
                         configuration_id=p['configuration_id'], m=c['m'], axis=axis,
                         physical_candidate=c['row'], role=role, action=c.get('action', NEW),
                         features=dict(one=1, q=c['row']['q']), model_group=[route, p['formula'], p['width_over_h'], axis])
                    for axis in ('X', 'Y')]
            if json.loads(json.dumps(want)) != mine or c['m'] not in STAGES.get(role, ()) or c.get('role', role) != role:
                raise ValueError('C CELL_PLAN differs from the regenerated cell %s %s m%d' % (route, p['formula'], c['m']))
            used += len(mine)
            if role == stage:
                cells.append(dict(c, role=role, model_groups=[r['model_group'] for r in mine]))
        out.append(dict(p, cells=cells))
    if used != len(rows):
        raise ValueError('C CELL_PLAN rows not covered by the regenerated plans')
    return out


def new_cells(templates):
    return [c for p in templates for c in p['cells'] if c.get('action', NEW) == NEW]


def check_group(records, group, stage):
    """One C model = one (route, formula, actual width, axis) and one stage; anything else is rejected."""
    route, formula, width, axis = group
    stem = '%s:%s:%s' % (route, formula, width)
    ids = [r['cell_id'] for r in records]
    if (not records or len(set(ids)) != len(ids)
            or any(r['route'] != route or r['axis'] != axis or r['role'] != stage
                   or r['cell_id'].rsplit(':m', 1)[0] != stem for r in records)):
        raise ValueError('Records mix model groups, axes or stages: ' + key(group))
    if sorted(int(i.rsplit(':m', 1)[1]) for i in ids) != list(STAGES[stage]):
        raise ValueError('Stage rounds incomplete or foreign for ' + key(group))
    _T3().c_model()._domain(records)  # C's own one-implementation/profile/kind/law check
    return records


def saved_calibration(root, templates):
    """C-admitted saved Full PF M5/m4 X/Y: C's own input records and their SOURCE, by axis (no recompile)."""
    saved = [c for p in templates for c in p['cells'] if c.get('action') == SAVED]
    if (len(saved) != 1 or saved[0]['row']['id'] != SAVED_ID or saved[0]['m'] != 4
            or saved[0]['row']['formula'] != '4th(m5_best)' or saved[0].get('role') != 'CALIBRATION'):
        raise ValueError('Saved M5/m4 receipt cell differs from C admission')
    cell = saved[0]
    ref = {n: dict(commit=BUNDLE, path=ADMISSION_DIR + '/' + n) for n in ('SAVED_CALIBRATION_INPUTS.json', 'PRICES.json')}
    ref.update({n: dict(commit=ADMISSION, path=ADMISSION_DIR + '/' + n) for n in ('SIDECARS.json', 'GUARD_CHECK.json')})
    doc = {n: json.loads(read_pinned(root, r)) for n, r in ref.items()}
    guard = doc['GUARD_CHECK.json']['guard']
    if (guard['status'] != 'IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION'
            or sorted(x['axis'] for x in guard['checked']) != ['X', 'Y']):
        raise ValueError('C guard check of the saved receipts missing')
    out = {}
    for axis in ('X', 'Y'):
        def one(rows):
            hit = [r for r in rows if r['axis'] == axis]
            if len(hit) != 1:
                raise ValueError('Saved receipt %s axis ambiguous' % axis)
            return hit[0]
        rec, price, side = (one(doc[n]) for n in ('SAVED_CALIBRATION_INPUTS.json', 'PRICES.json', 'SIDECARS.json'))
        if (rec['cell_id'] != 'FULL_PF:4th(m5_best):16:m4' or rec['role'] != 'CALIBRATION' or rec['route'] != 'FULL_PF'
                or rec['kind'] != 'DIRECT_COMPILED_DETERMINISTIC' or rec['source'].get('key') != SAVED_ID
                or rec['features'] != {k: dict(numerator=1, denominator=1) for k in FEATURES}
                or price['physical_context']['physical_candidate'] != cell['row'] or price['source'] != rec['source']
                or rec['rz_count'] != dict(numerator=price['mean']['rz_count'], denominator=1)
                or side['implementation_evidence']['commit'] != BUNDLE):
            raise ValueError('Saved M5/m4 receipt record differs from C admission: ' + axis)
        group = next(g for g in cell['model_groups'] if g[3] == axis)
        out[key(group)] = dict(record=rec, origin='C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED',
                               references={n: dict(r, axis=axis) for n, r in ref.items()})
    return out


def source_gate(root, sources):
    """Executed callers/stage module equal HEAD; pinned native/wrapper equal their fixed blobs."""
    T3 = _T3(); C, _ = T3._meta()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in sources:
        if C.blob(root, commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ from HEAD: ' + path)
    for d in (T3.NATIVE, T3.WRAPPER):
        if C.blob(root, d['commit'], d['path']) != (Path(root) / d['path']).read_bytes():
            raise ValueError('Pinned native SOURCE differs: ' + d['path'])
    return commit


def committed(root, rel_dir, names, commit=None):
    """Stage files are tracked at `commit` (default HEAD), byte-identical, and the directory is clean."""
    git = ['git', '--no-optional-locks']
    commit = commit or subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all', '--', rel_dir], cwd=root).strip():
        raise ValueError('Stage output modified or unpublished: ' + rel_dir)
    digest = {}
    for name in names:
        if not (Path(root) / rel_dir / name).is_file():
            raise ValueError('Stage file missing: %s/%s' % (rel_dir, name))
        local = (Path(root) / rel_dir / name).read_bytes()
        try:
            published = subprocess.check_output(git + ['show', '%s:%s/%s' % (commit, rel_dir, name)], cwd=root,
                                                stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise ValueError('Stage file not published at %s: %s/%s' % (commit, rel_dir, name)) from None
        if published != local:
            raise ValueError('Stage file differs from %s: %s/%s' % (commit, rel_dir, name))
        digest[name] = hashlib.sha256(local).hexdigest()
    return dict(commit=commit, sha256=digest)


def _canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'))


def frozen_models(root, frozen, calibration_inputs):
    """C's frozen models: exactly one per model group, fitted on exactly the published m4/m5 inputs."""
    if not isinstance(frozen, dict) or len(frozen.get('commit') or '') != 40 or not frozen.get('path'):
        raise ValueError('Pinned C frozen-model SOURCE (40-hex commit and path) required')
    raw = read_pinned(root, frozen)
    models = []

    def walk(v):
        if isinstance(v, dict):
            if v.get('model') == MODEL:
                models.append(v)
            else:
                for x in v.values():
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(json.loads(raw))

    def stems(m):
        return {c['cell_id'].rsplit(':m', 1)[0] for c in m.get('calibration') or ()}
    V = _T3().c_model(); out = {}
    if not calibration_inputs or any(g['stage'] != 'CALIBRATION' for g in calibration_inputs.values()):
        raise ValueError('Published CALIBRATION inputs required')
    routes = {g['model_group'][0] for g in calibration_inputs.values()}
    if any(len(stems(m)) != 1 for m in models if (m.get('domain') or {}).get('route') in routes):
        raise ValueError('A frozen C model mixes model groups')
    for k, g in sorted(calibration_inputs.items()):
        route, formula, width, axis = g['model_group']
        mine = [m for m in models if (m.get('domain') or {}).get('route') == route and m['domain'].get('axis') == axis
                and stems(m) == {'%s:%s:%s' % (route, formula, width)}]
        if len(mine) != 1:
            raise ValueError('Exactly one frozen C model per model group required: ' + k)
        m = mine[0]
        records = [s['record'] for s in g['saved_records']] + g['new_records']
        want = sorted((r['cell_id'], _canonical(r['source'])) for r in records)
        got = sorted((c['cell_id'], _canonical(c['source'])) for c in m['calibration'])
        if (m.get('status') != 'CALIBRATED_UNVALIDATED' or m.get('features') != FEATURES or got != want
                or m.get('domain') != json.loads(json.dumps(V._domain(records))) or not m.get('design_source')
                or m.get('qualified_for_candidate') is not False or set(m.get('coefficients') or ()) != set(FEATURES)):
            raise ValueError('Frozen model is not the fixed m4/m5 calibration of ' + k)
        out[k] = dict(coefficients=deepcopy(m['coefficients']), calibration=deepcopy(m['calibration']),
                      design_source=deepcopy(m['design_source']))
    return dict(source=dict(commit=frozen['commit'], path=frozen['path']), sha256=hashlib.sha256(raw).hexdigest(),
                models=out)


def _holdout_gate(root, spec, new, frozen):
    cal_rel, _ = paths(spec, 'CALIBRATION')
    published = committed(root, cal_rel, STAGE_FILES + BOUND_FILES)
    out = Path(root) / cal_rel
    result = json.loads((out / 'RESULT.json').read_text())
    if result.get('status') != 'CALIBRATION_PRICES_COMPLETE' or result.get('stage') != 'CALIBRATION':
        raise ValueError('Complete CALIBRATION stage required before HOLDOUT')
    if PLACEHOLDER in (out / 'SIDECARS.json').read_text():
        raise ValueError('CALIBRATION sidecars still carry the evidence placeholder')
    done = {r['cell']['row']['id'] for r in result['results']}
    if done & {c['row']['id'] for c in new}:
        raise ValueError('HOLDOUT cell already acquired in CALIBRATION')
    inputs = json.loads((out / 'C_INPUTS.json').read_text())
    return dict(calibration=published, frozen_models=frozen_models(root, frozen, inputs))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def execute_stage(root, spec, stage, *, frozen=None, workers=None):
    """One stage: gates, provider-root child for the stage's NEW cells only, records in the stage namespace."""
    if stage not in STAGES or (stage == 'HOLDOUT') != (frozen is not None):
        raise ValueError('HOLDOUT needs C frozen models; CALIBRATION takes none')
    commit = source_gate(root, spec['sources'])
    templates = select(root, spec['plans'](root), spec['route'], stage)
    new = [dict(c, formula=p['formula'], row=spec['native_row'](c['row'])) for p in templates for c in p['cells']
           if c.get('action', NEW) == NEW]
    if 2 * len(new) != NEW_WRAPPERS[(spec['route'], stage)]:
        raise ValueError('Fixed %s %s scope is %d wrappers' % (spec['route'], stage, NEW_WRAPPERS[(spec['route'], stage)]))
    saved = saved_calibration(root, templates) if (spec['route'], stage) == ('FULL_PF', 'CALIBRATION') else {}
    gate = _holdout_gate(root, spec, new, frozen) if stage == 'HOLDOUT' else None
    extra = spec['job_extra'](root)  # e.g. the saved B0I tail scalar, read before any output exists
    out_rel, private = paths(spec, stage)
    out = Path(root) / out_rel
    out.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
    job = dict(stage=stage, cells=new, public=str(out / 'child_public'), private=str(private),
               result=str(private / 'child.json'), workers=workers, **extra)
    _write(private / 'job.json', job)
    start = time.perf_counter()
    code = subprocess.call([sys.executable, '-B', str(Path(root) / spec['caller']), '--child', str(private / 'job.json')],
                           cwd=root)
    child = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else dict(results=[], resources=None)
    results = child['results']
    ok = code == 0 and len(results) == len(new) and all(r['ok'] for r in results)
    source = dict(spec['source'], commit=commit, stage=stage, stage_module=MODULE, cell_plan=CELL_PLAN)
    error = None
    try:
        if ok:
            _records(root, spec, stage, templates, results, saved, source, out, out_rel)
    except Exception as exc:  # trials stay in RESULT.json; no partial price files are trusted
        ok, error = False, '%s: %s' % (type(exc).__name__, exc)
    after = None
    if gate is not None:
        try:
            after = committed(root, paths(spec, 'CALIBRATION')[0], STAGE_FILES + BOUND_FILES)
        except ValueError as exc:
            after = dict(error=str(exc))
        ok = ok and after == gate['calibration']
    status = ('%s_PRICES_COMPLETE' if ok else 'STOP_%s_PRICE_CHECK') % stage
    _write(out / 'RESULT.json', dict(status=status, stage=stage, route=spec['route'], source=source,
        cells=[dict(formula=c['formula'], m=c['m'], id=c['row']['id'], q=c['row']['q'], model_groups=c['model_groups'])
               for c in new],
        saved_calibration=sorted(saved), holdout_gate=gate, calibration_after=after, child_exit=code, error=error,
        results=results, resources=child.get('resources'), elapsed_seconds=time.perf_counter() - start,
        finished_utc=datetime.now(timezone.utc).isoformat(), mandatory_stop=True, retry_performed=False,
        fit=None, prediction=None, holdout_judgement=None, N=None, G=None))
    print(json.dumps(dict(status=status, output=out_rel)))
    return 0 if ok else 2


def _records(root, spec, stage, templates, results, saved, source, out, out_rel):
    """Price records/sidecars/evidence from the callers' unchanged price_bindings; C inputs per model group."""
    T3 = _T3()
    prices, sides, proofs, groups = [], [], {}, {}
    for p in templates:
        for c in p['cells']:
            if c.get('action', NEW) != NEW:
                continue
            r = next(x for x in results if x['cell']['row']['id'] == c['row']['id'])
            a, b, d = spec['bindings'](root, p['plan'], c, r['trial'], price_source=dict(source, record='RESULT.json'),
                                       evidence_commit=PLACEHOLDER, evidence_path=out_rel + '/IMPLEMENTATION_EVIDENCE.json')
            for x in a:
                x['diagnostic'].update(stage=stage, compiled_wrapper=deepcopy(r['wrappers']['cosine' if x['axis'] == 'X' else 'sine']))
            prices += a; sides += b; proofs.update(d)
            for rec in T3.fit_inputs(a, route=spec['route'], formula=p['formula'], width_over_h=p['width_over_h'],
                                     roles={c['m']: stage}, root=root):
                group = next(g for g in c['model_groups'] if g[3] == rec['axis'])
                groups.setdefault(key(group), dict(model_group=group, stage=stage, features=FEATURES,
                                                   new_records=[], saved_records=[]))['new_records'].append(rec)
    for k, s in saved.items():
        groups[k]['saved_records'].append(s)
    for g in groups.values():
        check_group([s['record'] for s in g['saved_records']] + g['new_records'], g['model_group'], stage)
    if len(groups) != 6:
        raise ValueError('Six model groups (3 formulas x X/Y) expected per route and stage')
    for name, value in (('PRICES.json', prices), ('SIDECARS_UNBOUND.json', sides),
                        ('IMPLEMENTATION_EVIDENCE.json', proofs), ('C_INPUTS.json', groups)):
        _write(out / name, value)


def bind(root, spec, stage, evidence_commit):
    """Fill the published evidence commit into the sidecars and run C's guard (technical probe policy)."""
    T3 = _T3(); T3._meta()  # C modules of this tree on sys.path
    from trottertracks import paper_nominal_cost as PN
    from trottertracks import price_implementation_guard as G
    rel, _ = paths(spec, stage); out = Path(root) / rel
    if len(evidence_commit or '') != 40:
        raise ValueError('Published 40-hex evidence commit required')
    if any((out / n).exists() for n in BOUND_FILES):
        raise ValueError('Stage already bound: ' + rel)
    published = committed(root, rel, STAGE_FILES, commit=evidence_commit)
    if json.loads((out / 'RESULT.json').read_text()).get('status') != '%s_PRICES_COMPLETE' % stage:
        raise ValueError('Only a complete stage is bound')
    prices = json.loads((out / 'PRICES.json').read_text())
    sides = json.loads((out / 'SIDECARS_UNBOUND.json').read_text())
    for s in sides:
        ev = s['implementation_evidence']
        if ev['commit'] != PLACEHOLDER or ev['path'] != rel + '/IMPLEMENTATION_EVIDENCE.json':
            raise ValueError('Unexpected unbound evidence locator')
        ev['commit'] = evidence_commit
    B = PN._api(root)[0].B; checks = []
    for p in select(root, spec['plans'](root), spec['route'], stage):
        config = p['plan']['configuration']
        mine = [x for x in prices if x['configuration_id'] == config['id']]
        if not mine:
            continue
        policy = dict(id=PROBE, source=dict(GUARD_SOURCE), profile=deepcopy(config['spec']['profile']),
                      by_method={spec['route']: dict(native_policy=T3.NATIVE_POLICY, rte_rotation_lowering=G.NO_RTE,
                                                     scalar_quantity=spec['scalar_quantity'])})
        cohorts = {(config['rounds'][x['m']]['candidate']['id'], x['axis']): spec['cohort'] for x in mine}
        checked = G.check(root, config, mine, cohorts=cohorts, bindings=sides, policy=policy)
        if len(checked['checked']) != len(mine):
            raise ValueError('C guard did not check every record of ' + config['id'])
        tiers = {'m%d%s' % (x['m'], x['axis']): PN._tier(B, x, spec['route']) for x in mine}
        if set(tiers.values()) != {'DETERMINISTIC_DIRECT'}:
            raise ValueError('Deterministic direct tier expected')
        checks.append(dict(configuration_id=config['id'], formula=p['formula'], guard=checked, tiers=tiers))
    if sum(len(c['guard']['checked']) for c in checks) != len(prices):
        raise ValueError('Every stage price must be guard-checked')
    _write(out / 'SIDECARS.json', sides)
    _write(out / 'GUARD_CHECK.json', dict(stage=stage, route=spec['route'], evidence=published, policy_id=PROBE,
        comparison_policy_adopted=False, checks=checks, G=None))
    return dict(status='BOUND_AND_GUARD_CHECKED', checked=len(prices), output=rel)


_RUN = {}


def _cell(cell):
    """Forked worker: one cell through the caller's unchanged trial function; records each wrapper's fingerprint."""
    native = _RUN['native']; original = _RUN['compile']; calls = []

    def recording(wrapper, fingerprint, representative):
        result = original(wrapper, fingerprint, representative)
        calls.append(dict(fingerprint=fingerprint, cache_reused=bool(result[2]), wrapper_qubits=wrapper.num_qubits,
                          compiled_qubits=None if result[1] is None else result[1].num_qubits))
        return result
    native.compile_cached = recording
    started = time.monotonic()
    try:
        trial, facts, ok = _RUN['trial'](native, _RUN['ctx'], cell['row'], _RUN['cohort'], **_RUN['kw'])
    except BaseException as exc:  # partial save; never retried
        return dict(cell=cell, ok=False, error='%s: %s' % (type(exc).__name__, exc), wall_seconds=time.monotonic() - started)
    finally:
        native.compile_cached = original
    ok = ok and len(calls) == 2
    return dict(cell=cell, trial=trial, facts=facts, ok=ok, wrappers=dict(zip(('cosine', 'sine'), calls)),
                wall_seconds=time.monotonic() - started)


def run_cells(native, ctx, trial, cohort, cells, *, kw=None, workers=None):
    """Child: independent cells in forked worker processes (the long-time runner's pattern).

    After the first failed cell no further cell is started; running cells finish
    and are saved. Workers follow the existing idle-CPU/available-RAM rule.
    """
    obs = None
    if workers is None:
        from pf_common_run_contract import resources
        obs = resources()
        workers = max(1, min(len(cells) or 1, int(obs['idle_cpu']), obs['available_ram'] // (2 * 2**30)))
    _RUN.update(native=native, ctx=ctx, trial=trial, cohort=cohort, kw=kw or {}, compile=native.compile_cached)
    results = [None] * len(cells); stopped = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        futures = {pool.submit(_cell, c): i for i, c in enumerate(cells)}
        for f in as_completed(futures):
            if f.cancelled():
                continue
            try:
                r = f.result()
            except BaseException as exc:  # e.g. a killed worker: saved as a failed cell, never retried
                r = dict(cell=cells[futures[f]], ok=False, error='%s: %s' % (type(exc).__name__, exc))
            results[futures[f]] = r
            if not r['ok'] and not stopped:
                stopped = True
                for g in futures:
                    g.cancel()
    done = [r for r in results if r is not None]
    return done, dict(workers=workers, observation=obs, cells=len(cells), started=len(done),
                      not_started_after_failure=len(cells) - len(done))
