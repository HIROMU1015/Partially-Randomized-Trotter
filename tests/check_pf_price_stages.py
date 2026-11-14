"""CALIBRATION / HOLDOUT stage split of the Full PF and Truncated p7 price callers (parent side).

Metadata, C CELL_PLAN, saved C record bindings and FICTIONAL new prices only. The provider-root child is
replaced by a fake that writes fictional trials (no build, compile or H4 input);
outputs go to temporary namespaces. C guard and C model functions are the
in-tree c628e9e8 ones; all fits replace RZ values with fictional data and only
exercise the frozen-model gate. No saved scientific values are fitted.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import pf_price_stages as S  # noqa: E402
import run_full_pf_shortlist_prices as FP  # noqa: E402
import run_truncated_p7_prices as T3  # noqa: E402
from trottertracks import price_implementation_guard as G  # noqa: E402

V = T3.c_model()
CONST = 2.29310124732
TAIL = 4.5302064712666724e-05
ROUTES = dict(FULL_PF=FP, TRUNCATED_PF=T3)
TEMPLATES = {r: m.plans() for r, m in ROUTES.items()}
REAL_READ = S.read_pinned
REAL_LEGACY = json.loads(REAL_READ(ROOT, S.CELL_PLAN))
REAL_PLAN = S.plan_rows(ROOT)
METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')


def fictional_trial(row, scalar):
    base = 1000 * row['q'] + (300 if row['method'] == 'FULL' else 0)
    prices = {a: {k: base + i + j for j, k in enumerate(METRICS)} for i, a in enumerate(('cosine', 'sine'))}
    return dict(status='PASS', candidate=row, scalar=scalar, prices=prices, checks=[], wall_seconds=0., peak_RSS_bytes=0)


def p7_fixture():
    """Fictional C acceptance interface only; no real p7 saved RZ value is used or admitted."""
    ref = dict(bundle='a' * 40, admission='a' * 40, directory='ARTIFICIAL_P7')
    p = TEMPLATES['TRUNCATED_PF'][-1]; c = p['cells'][0]
    prices, sides, proofs = T3.price_bindings(ROOT, p['plan'], c, fictional_trial(c['row'], CONST + TAIL),
        price_source=dict(fixture='ARTIFICIAL_SAVED_NOT_MEASURED', key=S.SAVED_IDS['TRUNCATED_PF']),
        evidence_commit=ref['bundle'], evidence_path='ARTIFICIAL_P7/IMPLEMENTATION_EVIDENCE.json')
    for row in prices + sides:
        row['cohort'] = 'ARTIFICIAL_ORIGINAL_COHORT'
    inputs = T3.fit_inputs(prices, route='TRUNCATED_PF', formula=p['formula'], width_over_h=p['width_over_h'],
                          roles={4: 'CALIBRATION'}, root=ROOT)
    docs = {'PRICES.json': prices, 'SIDECARS.json': sides, 'SAVED_CALIBRATION_INPUTS.json': inputs,
            'GUARD_CHECK.json': dict(fixture='ARTIFICIAL_C_INTERFACE', guard=dict(
                status='IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION',
                checked=[dict(axis=a) for a in ('X', 'Y')]))}
    return ref, {(ref['bundle'], ref['directory'] + '/' + n): json.dumps(v).encode() for n, v in docs.items()}


P7_FIXTURE, P7_DOCUMENTS = p7_fixture()
FIXTURE_PLAN = dict(commit='a' * 40, path='ARTIFICIAL_P7/CELL_PLAN.json')


class Env:
    """Temporary stage namespaces, a fake provider-root child and a fake publication record."""

    def __init__(self, route, tmp):
        self.route, self.mod = route, ROUTES[route]
        self.spec = dict(self.mod.stage_spec(), output=str(tmp / 'public'), private=tmp / 'private',
                         plans=lambda root, t=TEMPLATES[route]: deepcopy(t),
                         job_extra=(lambda root: dict(saved_tail_scalar=TAIL)) if route == 'TRUNCATED_PF' else (lambda root: {}))
        self.published, self.jobs, self.frozen = {}, [], {}
        self.on_child = None; self.drop_sine = False

    def child(self, args, cwd=None):
        job = json.loads(Path(args[-1]).read_text()); self.jobs.append(job)
        scalar = CONST + (TAIL if self.route == 'TRUNCATED_PF' else 0.)
        results = [dict(cell=c, trial=fictional_trial(c['row'], scalar), facts={}, ok=True,
                        wrappers={a: dict(fingerprint='%s_%s' % (c['row']['id'], a), cache_reused=False,
                                          wrapper_qubits=10, compiled_qubits=10) for a in ('cosine', 'sine')})
                   for c in job['cells']]
        if self.drop_sine:
            del results[-1]['trial']['prices']['sine']
        Path(job['result']).write_text(json.dumps(dict(results=results, resources=dict(workers=len(results)))))
        if self.on_child:
            self.on_child()
        return 0

    def committed(self, root, rel_dir, names, commit=None):
        digest = {}
        for n in names:
            path = Path(root) / rel_dir / n
            if (rel_dir, n) not in self.published or not path.is_file() or path.read_bytes() != self.published[(rel_dir, n)]:
                raise ValueError('not published/unchanged: ' + n)
            digest[n] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(commit=commit or 'h' * 40, sha256=digest)

    def publish(self, stage):
        rel, _ = S.paths(self.spec, stage)
        for f in Path(rel).iterdir():
            if f.is_file():
                self.published[(rel, f.name)] = f.read_bytes()

    def read(self, root, ref):
        if ref == FIXTURE_PLAN:
            return json.dumps(REAL_PLAN).encode()
        if (ref['commit'], ref['path']) in P7_DOCUMENTS:
            return P7_DOCUMENTS[(ref['commit'], ref['path'])]
        if (ref['commit'], ref['path']) in self.frozen:
            return self.frozen[(ref['commit'], ref['path'])]
        return REAL_READ(root, ref)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(S, 'source_gate', return_value='e' * 40))
        stack.enter_context(patch.object(S, 'P7_CELL_PLAN', FIXTURE_PLAN))
        stack.enter_context(patch.object(S, 'P7_ADMISSION', P7_FIXTURE))
        stack.enter_context(patch.object(S, 'duplicate_preflight', return_value=dict(status='ARTIFICIAL_NO_SCIENCE')))
        stack.enter_context(patch.object(S.subprocess, 'call', side_effect=self.child))
        stack.enter_context(patch.object(S, 'committed', side_effect=self.committed))
        stack.enter_context(patch.object(S, 'read_pinned', side_effect=self.read))
        return stack

    def run(self, stage, frozen=None):
        with self.patches():
            return S.execute_stage(ROOT, self.spec, stage, frozen=frozen, workers=1)

    def bind(self, stage):
        rel, _ = S.paths(self.spec, stage)
        def document(root, commit, path):
            return json.loads(Path(path).read_text())
        with self.patches(), patch.object(G, '_document', side_effect=document):
            return S.bind(ROOT, self.spec, stage, 'b' * 40)

    def load(self, stage, name):
        return json.loads((Path(S.paths(self.spec, stage)[0]) / name).read_text())

    def freeze(self, mutate=None):
        """Fictional C models (fictional rz values) on exactly the published calibration inputs."""
        models = []
        for g in self.load('CALIBRATION', 'C_INPUTS.json').values():
            records = [deepcopy(s['record']) for s in g['saved_records']] + deepcopy(g['new_records'])
            for i, r in enumerate(records):
                r['rz_count'] = dict(numerator=1000 + 7 * i, denominator=1)  # fictional, not a real price
            models.append(json.loads(json.dumps(V.fit(records, features=('one', 'q'),
                                                      design_source=dict(fixture='ARTIFICIAL_FROZEN_MODEL')))))
        if mutate:
            mutate(models)
        ref = dict(commit='c' * 40, path='ARTIFICIAL/FROZEN_%d.json' % len(self.frozen))
        self.frozen[(ref['commit'], ref['path'])] = json.dumps(dict(models=models)).encode()
        return ref


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='pf_stages_')); self.addCleanup(shutil.rmtree, self.tmp, True)

    def env(self, route):
        return Env(route, self.tmp / route)

    def test_revised_plan_changes_only_two_actions_and_matches_published_C_table(self):
        self.assertEqual(REAL_READ(ROOT, S.CELL_PLAN), REAL_READ(ROOT, dict(S.CELL_PLAN, commit=S.ADMISSION)))
        self.assertEqual(len(REAL_PLAN), 36)
        self.assertEqual({(r['m'], r['role']) for r in REAL_PLAN}, {(4, 'CALIBRATION'), (5, 'CALIBRATION'), (6, 'HOLDOUT')})
        delta = []
        for old, new in zip(REAL_LEGACY, REAL_PLAN, strict=True):
            if old != new:
                self.assertEqual(new, dict(old, action=S.SAVED))
                delta.append((new['route'], new['physical_candidate']['id'], new['axis']))
        self.assertEqual(delta, [('TRUNCATED_PF', S.SAVED_IDS['TRUNCATED_PF'], a) for a in ('X', 'Y')])
        if S.P7_CELL_PLAN is None:
            with self.assertRaisesRegex(ValueError, 'C p7'):
                S.admission_gate(ROOT)
        else:
            self.assertEqual(REAL_PLAN, json.loads(REAL_READ(ROOT, S.P7_CELL_PLAN)))
            self.assertEqual(S.admission_gate(ROOT), S.P7_CELL_PLAN)

    def test_real_C_admission_binds_four_original_axis_records_without_relabelling(self):
        registry = json.loads(REAL_READ(ROOT, dict(commit='e50524f4b59f268735a682bd4b2eaa8278949c89',
            path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/SAVED_RECEIPT_SOURCES.json')))
        self.assertEqual(registry['cell_plan'], S.P7_CELL_PLAN)
        seen = []
        for route, templates in TEMPLATES.items():
            chosen = S.select(ROOT, templates, route, 'CALIBRATION')
            saved = S.saved_calibration(ROOT, chosen)
            self.assertEqual(len(saved), 2)
            for value in saved.values():
                rec = value['record']; axis = rec['axis']
                ref = next(r for r in registry['receipts'] if r['route'] == route and r['axis'] == axis)
                source = ref['references']['SAVED_CALIBRATION_INPUTS.json']
                pinned = json.loads(REAL_READ(ROOT, source))
                for step in source['selector']:
                    pinned = pinned[step]
                self.assertEqual(rec, pinned)  # copy original integer/SOURCE/cohort, no new mean or fit
                self.assertEqual(rec['source']['key'], ref['physical_id'])
                self.assertEqual(rec['source']['commit'], S.SAVED_TABLE['commit'])
                self.assertEqual(ref['cohort'], 'EXPLORATION')
                seen.append((route, axis))
        self.assertEqual(set(seen), {(r, a) for r in ROUTES for a in ('X', 'Y')})

    def test_stage_split_20_calibration_12_holdout_32_new_saved_never_compiled(self):
        total = {}
        for route, templates in TEMPLATES.items():
            for stage, rounds in S.STAGES.items():
                chosen = S.select(ROOT, templates, route, stage)
                cells = [c for p in chosen for c in p['cells']]
                self.assertEqual({c['m'] for c in cells}, set(rounds))
                self.assertEqual({c['role'] for c in cells}, {stage})
                total[(route, stage)] = 2 * len(S.new_cells(chosen))
                saved = [c for c in cells if c.get('action') == S.SAVED]
                if stage == 'CALIBRATION':
                    self.assertEqual([(c['row']['id'], c['m'], c['row']['formula']) for c in saved], [(S.SAVED_IDS[route], 4, '4th(m5_best)')])
                    self.assertNotIn(S.SAVED_IDS[route], [c['row']['id'] for c in S.new_cells(chosen)])
                else:
                    self.assertEqual(saved, [])
        self.assertEqual(total, S.NEW_WRAPPERS)
        self.assertEqual(sum(v for (r, s), v in total.items() if s == 'CALIBRATION'), 20)
        self.assertEqual(sum(v for (r, s), v in total.items() if s == 'HOLDOUT'), 12)
        self.assertEqual(sum(total.values()), 32)

    def test_formula_width_q_round_and_axis_mapping(self):
        q = {'2nd': ('16/7', [7, 14, 28]), '4th': ('16/3', [3, 6, 12]), '4th(m5_best)': ('16', [1, 2, 4])}
        for route, templates in TEMPLATES.items():
            for p in templates:
                self.assertEqual((p['width_over_h'], [c['row']['q'] for c in p['cells']]), q[p['formula']])
                self.assertEqual([c['m'] for c in p['cells']], [4, 5, 6])
            for stage in S.STAGES:
                for p in S.select(ROOT, templates, route, stage):
                    for c in p['cells']:
                        self.assertEqual(c['model_groups'], [[route, p['formula'], p['width_over_h'], a] for a in ('X', 'Y')])
        full = {(p['formula'], c['m']): c['row'] for p in TEMPLATES['FULL_PF'] for c in p['cells']}
        for p in TEMPLATES['TRUNCATED_PF']:
            for c in p['cells']:
                f = full[(p['formula'], c['m'])]
                self.assertEqual((c['row']['q'], c['row']['T'], c['row']['delta']), (f['q'], f['T'], f['delta']))
                self.assertEqual(((c['row']['method'], c['row']['prefix']), (f['method'], f['prefix'])), (('B0I', 7), ('FULL', 10)))

    def test_unpublished_p7_admission_stops_both_routes_before_child_or_output(self):
        for route in ROUTES:
            e = self.env(route)
            with patch.object(S, 'source_gate', return_value='e' * 40), patch.object(S, 'P7_ADMISSION', None), \
                    patch.object(S.subprocess, 'call') as child:
                with self.assertRaisesRegex(ValueError, 'C p7'):
                    S.execute_stage(ROOT, e.spec, 'CALIBRATION')
                child.assert_not_called()
            self.assertFalse(Path(S.paths(e.spec, 'CALIBRATION')[0]).exists())
        chosen = S.select(ROOT, TEMPLATES['TRUNCATED_PF'], 'TRUNCATED_PF', 'CALIBRATION')
        with patch.object(S, 'P7_ADMISSION', None), self.assertRaisesRegex(ValueError, 'C p7'):
            S.saved_calibration(ROOT, chosen)

    def test_saved_binding_keeps_original_record_and_rejects_foreign_source_axis_physical_profile(self):
        e = self.env('TRUNCATED_PF')
        with e.patches():
            chosen = S.select(ROOT, TEMPLATES['TRUNCATED_PF'], 'TRUNCATED_PF', 'CALIBRATION')
            bound = S.saved_calibration(ROOT, chosen)
        original = json.loads(P7_DOCUMENTS[(P7_FIXTURE['bundle'], 'ARTIFICIAL_P7/SAVED_CALIBRATION_INPUTS.json')])
        self.assertEqual([v['record'] for v in bound.values()], original)
        self.assertEqual({v['record']['source']['fixture'] for v in bound.values()}, {'ARTIFICIAL_SAVED_NOT_MEASURED'})
        def changed(name, fn):
            docs = deepcopy(P7_DOCUMENTS); k = (P7_FIXTURE['bundle'], 'ARTIFICIAL_P7/' + name)
            rows = json.loads(docs[k]); fn(rows); docs[k] = json.dumps(rows).encode()
            return docs
        mutations = (
            ('PRICES.json', lambda rows: rows[0]['physical_context']['physical_candidate'].update(prefix=10)),
            ('PRICES.json', lambda rows: rows[0]['price_profile'].update(compiler='foreign')),
            ('SAVED_CALIBRATION_INPUTS.json', lambda rows: rows[0]['source'].update(key=S.SAVED_ID)),
            ('SAVED_CALIBRATION_INPUTS.json', lambda rows: rows[0].update(axis='Y')),
            ('SIDECARS.json', lambda rows: rows[0].update(cohort='NEW_COHORT')),
            ('SIDECARS.json', lambda rows: rows[0]['implementation_evidence'].update(commit='f' * 40)),
        )
        for name, fn in mutations:
            with e.patches(), patch.dict(P7_DOCUMENTS, changed(name, fn)), self.assertRaises(ValueError):
                S.saved_calibration(ROOT, chosen)

    def test_duplicate_cells_and_additional_reuse_or_running_job_are_rejected(self):
        templates = deepcopy(TEMPLATES['TRUNCATED_PF']); templates[0]['cells'].append(deepcopy(templates[0]['cells'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate cell'):
            S.select(ROOT, templates, 'TRUNCATED_PF', 'CALIBRATION')
        e = self.env('FULL_PF'); cell = deepcopy(TEMPLATES['FULL_PF'][0]['cells'][0]); ident = cell['row']['id']
        def documents(root, ref):
            import gzip
            if ref == S.SAVED_TABLE:
                return gzip.compress(json.dumps({}).encode())
            if ref == S.CATALOG:
                return json.dumps(dict(rows=[])).encode()
            return REAL_READ(root, ref)
        with patch.object(S, 'read_pinned', side_effect=documents), patch.object(S, 'active_price_jobs', return_value=[]), \
                patch.object(S, 'OLD_TRIAL_ROOT', self.tmp / 'old'):
            self.assertEqual(S.duplicate_preflight(ROOT, e.spec, [cell])['status'], 'NO_DUPLICATE_IN_CHECKED_SOURCES')
            with patch.object(S, 'active_price_jobs', return_value=[dict(pid=123, physical_ids=[ident])]):
                with self.assertRaisesRegex(ValueError, 'RUNNING_JOB'):
                    S.duplicate_preflight(ROOT, e.spec, [cell])
            trial = S.OLD_TRIAL_ROOT / ident / '0/result.json'; trial.parent.mkdir(parents=True); trial.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_TRIAL'):
                S.duplicate_preflight(ROOT, e.spec, [cell])
        def extra_saved(root, ref):
            import gzip
            if ref == S.SAVED_TABLE:
                return gzip.compress(json.dumps({ident: dict(candidate=cell['row'], trial_files=['ARTIFICIAL'])}).encode())
            return documents(root, ref)
        with patch.object(S, 'read_pinned', side_effect=extra_saved), patch.object(S, 'active_price_jobs', return_value=[]), \
                patch.object(S, 'OLD_TRIAL_ROOT', self.tmp / 'absent'):
            with self.assertRaisesRegex(ValueError, 'SAVED_TABLE_REUSE_CANDIDATE'):
                S.duplicate_preflight(ROOT, e.spec, [cell])
        def renamed(root, ref):
            import gzip
            if ref == S.SAVED_TABLE:
                return gzip.compress(json.dumps({'OLD_RENAMED_ID': dict(candidate=dict(cell['row'], id='OLD_RENAMED_ID'),
                                                                           trial_files=['ARTIFICIAL'])}).encode())
            return documents(root, ref)
        with patch.object(S, 'read_pinned', side_effect=renamed), patch.object(S, 'active_price_jobs', return_value=[]), \
                patch.object(S, 'OLD_TRIAL_ROOT', self.tmp / 'absent'):
            with self.assertRaisesRegex(ValueError, 'OLD_RENAMED_ID'):
                S.duplicate_preflight(ROOT, e.spec, [cell])

    def test_cell_plan_disagreement_stops_selection(self):
        def m6_calibration(rows):
            for r in rows:
                if r['m'] == 6:
                    r['role'] = 'CALIBRATION'
        def m5_holdout(rows):
            rows[2]['role'] = 'HOLDOUT'; rows[3]['role'] = 'HOLDOUT'
        def physical(rows):
            rows[0]['physical_candidate']['id'] = 'x' * 20
        def saved_recompiled(rows):
            for r in rows:
                if r['action'] == S.SAVED:
                    r['action'] = S.NEW
        def axis_role_split(rows):
            rows[1]['role'] = 'HOLDOUT'
        def missing(rows):
            rows.pop()
        def extra(rows):
            rows.append(deepcopy(rows[0]))
        def group(rows):
            rows[0]['model_group'][1] = '4th'
        for fn in (m6_calibration, m5_holdout, physical, saved_recompiled, axis_role_split, missing, extra, group):
            rows = deepcopy(REAL_LEGACY); fn(rows)
            with patch.object(S, 'read_pinned', return_value=json.dumps(rows).encode()):
                with self.assertRaises(ValueError, msg=fn.__name__):
                    for route, templates in TEMPLATES.items():
                        S.select(ROOT, templates, route, 'CALIBRATION')

    def test_model_group_mixing_rejected(self):
        e = self.env('TRUNCATED_PF'); e.run('CALIBRATION')
        groups = e.load('CALIBRATION', 'C_INPUTS.json')
        a, b = groups['TRUNCATED_PF|2nd|16/7|X'], groups['TRUNCATED_PF|4th|16/3|X']
        S.check_group(a['new_records'], a['model_group'], 'CALIBRATION')
        y = groups['TRUNCATED_PF|2nd|16/7|Y']
        holdout = dict(deepcopy(a['new_records'][0]), role='HOLDOUT')
        m6 = dict(deepcopy(a['new_records'][0]), cell_id='TRUNCATED_PF:2nd:16/7:m6')
        full = dict(deepcopy(a['new_records'][0]), route='FULL_PF')
        for records in ([a['new_records'][0], b['new_records'][1]], [a['new_records'][0], y['new_records'][1]],
                        [a['new_records'][0], holdout], [a['new_records'][0], m6], [a['new_records'][0], full],
                        a['new_records'][:1], [a['new_records'][0], a['new_records'][0]]):
            with self.assertRaises(ValueError):
                S.check_group(records, a['model_group'], 'CALIBRATION')
        model = V.fit(a['new_records'], features=('one', 'q'), design_source=dict(fixture='ARTIFICIAL'))
        self.assertEqual((model['method'], model['qualified_for_candidate']), ('EXACT_SQUARE_SOLVE', False))

    def test_calibration_stage_namespace_records_and_no_duplicate_run(self):
        for route in ROUTES:
            e = self.env(route)
            self.assertEqual(e.run('CALIBRATION'), 0)
            job = e.jobs[0]
            self.assertEqual(2 * len(job['cells']), S.NEW_WRAPPERS[(route, 'CALIBRATION')])
            self.assertEqual({c['m'] for c in job['cells']}, {4, 5})
            self.assertFalse(set(S.SAVED_IDS.values()) & {c['row']['id'] for c in job['cells']})
            rel, private = S.paths(e.spec, 'CALIBRATION')
            self.assertTrue(rel.endswith('/calibration/run01') and str(private).endswith('/calibration/run01'))
            self.assertEqual(e.load('CALIBRATION', 'RESULT.json')['status'], 'CALIBRATION_PRICES_COMPLETE')
            prices = e.load('CALIBRATION', 'PRICES.json')
            self.assertEqual(len(prices), S.NEW_WRAPPERS[(route, 'CALIBRATION')])
            for p in prices:
                self.assertEqual((p['classification'], p['expectation'], p['record_schema']),
                                 ('DIRECT_COMPILED_DETERMINISTIC', 'DETERMINISTIC', 'D_PRICE_RECORD_B_KINDS_V2'))
                self.assertEqual((p['source']['stage'], p['diagnostic']['stage']), ('CALIBRATION', 'CALIBRATION'))
                self.assertTrue(p['diagnostic']['compiled_wrapper']['fingerprint'].endswith('cosine' if p['axis'] == 'X' else 'sine'))
                self.assertIn(p['m'], (4, 5))
            self.assertIn(S.PLACEHOLDER, (Path(rel) / 'SIDECARS_UNBOUND.json').read_text())
            self.assertFalse((Path(rel) / 'SIDECARS.json').exists())
            groups = e.load('CALIBRATION', 'C_INPUTS.json')
            self.assertEqual(len(groups), 6)
            for k, g in groups.items():
                records = [s['record'] for s in g['saved_records']] + g['new_records']
                self.assertEqual(sorted(r['cell_id'][-2:] for r in records), ['m4', 'm5'])
                self.assertEqual({r['role'] for r in records}, {'CALIBRATION'})
                S.check_group(records, g['model_group'], 'CALIBRATION')
                saved = '4th(m5_best)' in k
                self.assertEqual(len(g['saved_records']), int(saved))
                if saved:
                    s = g['saved_records'][0]
                    self.assertEqual((s['record']['source']['key'], s['origin']), (S.SAVED_IDS[route], 'C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED'))
                    self.assertEqual(s['references']['SIDECARS.json']['commit'], S.ADMISSION if route == 'FULL_PF' else P7_FIXTURE['admission'])
                    self.assertNotEqual(s['record']['source'], g['new_records'][0]['source'])
            calls = len(e.jobs)
            with self.assertRaises(FileExistsError):
                e.run('CALIBRATION')
            self.assertEqual(len(e.jobs), calls)  # no second compile of the same cells

    def test_bind_fills_published_commit_and_passes_C_guard(self):
        for route in ROUTES:
            e = self.env(route); e.run('CALIBRATION')
            with self.assertRaises(ValueError):  # unpublished stage cannot be bound
                e.bind('CALIBRATION')
            e.publish('CALIBRATION')
            out = e.bind('CALIBRATION')
            self.assertEqual(out['checked'], S.NEW_WRAPPERS[(route, 'CALIBRATION')])
            rel = S.paths(e.spec, 'CALIBRATION')[0]
            self.assertNotIn(S.PLACEHOLDER, (Path(rel) / 'SIDECARS.json').read_text())
            check = e.load('CALIBRATION', 'GUARD_CHECK.json')
            self.assertEqual((check['policy_id'], check['comparison_policy_adopted']), (S.COMPARISON_POLICY['id'], True))
            self.assertEqual(check['comparison_policy']['holdout']['value'], dict(numerator=1, denominator=50))
            self.assertEqual(check['comparison_policy']['holdout']['scale'], 'observed')
            self.assertFalse(check['model_qualified']); self.assertIsNone(check['holdout_judgement'])
            self.assertEqual({t for c in check['checks'] for t in c['tiers'].values()}, {'DETERMINISTIC_DIRECT'})
            self.assertEqual({c['guard']['status'] for c in check['checks']}, {'IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION'})
            with self.assertRaises(ValueError):
                e.bind('CALIBRATION')

    def calibrated(self, route):
        e = self.env(route); e.run('CALIBRATION'); e.publish('CALIBRATION'); e.bind('CALIBRATION'); e.publish('CALIBRATION')
        return e

    def test_holdout_needs_published_calibration_and_frozen_models(self):
        e = self.env('FULL_PF')
        with self.assertRaises(ValueError):
            e.run('HOLDOUT')  # no frozen models
        with self.assertRaises(ValueError):
            e.run('CALIBRATION', frozen=dict(commit='c' * 40, path='x'))
        with self.assertRaises(ValueError):
            e.run('HOLDOUT', frozen=dict(commit='c' * 40, path='x'))  # no calibration yet
        e.run('CALIBRATION')
        e.publish('CALIBRATION')
        with self.assertRaises(ValueError):  # calibration published but not bound/guard-checked
            e.run('HOLDOUT', frozen=e.freeze())
        self.assertEqual(len(e.jobs), 1)
        self.assertFalse(Path(S.paths(e.spec, 'HOLDOUT')[0]).exists())

    def test_frozen_model_gate_rejects_missing_mixed_m6_and_foreign_models(self):
        e = self.calibrated('TRUNCATED_PF')
        def missing(models):
            models.pop()
        def mixed(models):
            models[0]['calibration'][1]['cell_id'] = models[2]['calibration'][1]['cell_id']  # 2nd X takes a 4th cell
        def with_m6(models):
            models[0]['calibration'][1]['cell_id'] = models[0]['calibration'][1]['cell_id'][:-1] + '6'
        def foreign_source(models):
            models[0]['calibration'][0]['source'] = dict(models[0]['calibration'][0]['source'], commit='0' * 40)
        def validated(models):
            models[0]['status'] = 'HOLDOUT_WITHIN_DECLARED_TOLERANCE'
        def extra_feature(models):
            models[0]['features'] = ['one', 'q', 'q2']
        def duplicate(models):
            models.append(deepcopy(models[0]))
        def other_axis(models):
            models[0]['domain']['axis'] = 'Y' if models[0]['domain']['axis'] == 'X' else 'X'
        for fn in (missing, mixed, with_m6, foreign_source, validated, extra_feature, duplicate, other_axis):
            with self.assertRaises(ValueError, msg=fn.__name__):
                e.run('HOLDOUT', frozen=e.freeze(fn))
        self.assertEqual(len(e.jobs), 1)
        self.assertFalse(Path(S.paths(e.spec, 'HOLDOUT')[0]).exists())

    def test_holdout_stage_acquires_m6_only_and_leaves_calibration_unchanged(self):
        for route in ROUTES:
            e = self.calibrated(route)
            rel = S.paths(e.spec, 'CALIBRATION')[0]
            before = {f.name: f.read_bytes() for f in Path(rel).iterdir()}
            frozen = e.freeze()
            self.assertEqual(e.run('HOLDOUT', frozen=frozen), 0)
            job = e.jobs[-1]
            self.assertEqual(2 * len(job['cells']), S.NEW_WRAPPERS[(route, 'HOLDOUT')])
            self.assertEqual({c['m'] for c in job['cells']}, {6})
            calibrated = {c['row']['id'] for c in e.jobs[0]['cells']}
            self.assertFalse(calibrated & {c['row']['id'] for c in job['cells']})
            self.assertEqual({f.name: f.read_bytes() for f in Path(rel).iterdir()}, before)
            result = e.load('HOLDOUT', 'RESULT.json')
            self.assertEqual(result['status'], 'HOLDOUT_PRICES_COMPLETE')
            self.assertEqual(result['holdout_gate']['calibration'], result['calibration_after'])
            self.assertEqual(sorted(result['holdout_gate']['frozen_models']['models']), sorted(e.load('CALIBRATION', 'C_INPUTS.json')))
            self.assertEqual(result['holdout_gate']['frozen_models']['source'], frozen)
            groups = e.load('HOLDOUT', 'C_INPUTS.json')
            self.assertEqual(len(groups), 6)
            for g in groups.values():
                self.assertEqual([(r['role'], r['cell_id'][-2:]) for r in g['new_records']], [('HOLDOUT', 'm6')])
                self.assertEqual(g['saved_records'], [])
            self.assertEqual({p['source']['stage'] for p in e.load('HOLDOUT', 'PRICES.json')}, {'HOLDOUT'})
            with self.assertRaises(FileExistsError):
                e.run('HOLDOUT', frozen=frozen)
            self.assertEqual(len(e.jobs), 2)

    def test_calibration_change_during_holdout_is_a_stop(self):
        e = self.calibrated('FULL_PF')
        rel = S.paths(e.spec, 'CALIBRATION')[0]
        def tamper():
            p = Path(rel) / 'PRICES.json'; p.write_text(p.read_text() + ' ')
        e.on_child = tamper
        self.assertEqual(e.run('HOLDOUT', frozen=e.freeze()), 2)
        result = e.load('HOLDOUT', 'RESULT.json')
        self.assertEqual(result['status'], 'STOP_HOLDOUT_PRICE_CHECK')
        self.assertIn('error', result['calibration_after'])

    def test_record_build_failure_saved_as_stop_without_price_files(self):
        e = self.env('TRUNCATED_PF'); e.drop_sine = True
        self.assertEqual(e.run('CALIBRATION'), 2)
        result = e.load('CALIBRATION', 'RESULT.json')
        self.assertEqual(result['status'], 'STOP_CALIBRATION_PRICE_CHECK')
        self.assertIn('KeyError', result['error'])
        self.assertEqual(len(result['results']), 5)  # trials kept as the partial result; saved cell never started
        self.assertFalse((Path(S.paths(e.spec, 'CALIBRATION')[0]) / 'PRICES.json').exists())

    def test_committed_and_source_gate_on_real_git(self):
        out = S.committed(ROOT, 'src/trottertracks', ['price_model_validation.py'])
        self.assertEqual(len(out['commit']), 40)
        with self.assertRaises(ValueError):
            S.committed(ROOT, 'src/trottertracks', ['no_such_stage_file.json'])
        self.assertEqual(S.source_gate(ROOT, ['src/trottertracks/price_model_validation.py']), out['commit'])

    def test_cli_has_only_stage_paths(self):
        for mod in ROUTES.values():
            with self.assertRaises(SystemExit):
                mod.main(['--execute-approved-full-pf' if mod is FP else '--execute-approved-t3'])
            with patch.object(mod, 'execute', return_value=0) as run:
                self.assertEqual(mod.main(['--execute-approved-stage', 'HOLDOUT', '--frozen-models', 'c' * 40 + ':F.json']), 0)
                run.assert_called_once_with(mod.ROOT, 'HOLDOUT', dict(commit='c' * 40, path='F.json'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
