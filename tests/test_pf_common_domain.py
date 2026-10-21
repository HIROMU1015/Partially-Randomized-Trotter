"""Artificial domain/authorization tests; no molecular input or circuit imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pf_common_domain', ROOT/'src/trotterlib/pf_common_domain.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def labels(real='ELIGIBLE', imag='ELIGIBLE'):
    return {e: dict(real=real, imag=imag) for e in c.EPS}


def identity_receipt():
    keys = ('hamiltonian', 'state', 'scientific_arrays', 'preparation', 'native_policy',
            'formula_coefficients', 'scalar_control_policy', 'compiler', 'runtime_closure')
    expected = {k:'toy_'+k for k in keys}
    return dict(expected=expected, observed=copy.deepcopy(expected), receipt_sha256s=['0'*64])


class DomainTests(unittest.TestCase):
    def test_full_alias_and_full_cut_rejection(self):
        row = dict(method='B1', prefix=10, q=2, formula='2nd')
        self.assertEqual(c.normalized('H4', row)['method'], 'FULL')
        for mode in ('B0', 'B0I'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'ALIAS'):
                c.normalized('H4', dict(row, method=mode))

    def test_df12_never_enters_df10_union(self):
        registrations = [dict(system=s, method='B0', prefix=p, input_comparable=True)
                         for s, ps in [('H4', range(1, 10)), ('H6', range(5, 16))] for p in ps]
        registrations.append(dict(system='H4', method='B0', prefix=11, input_comparable=False))
        rows = c.common_domain(registrations)
        self.assertEqual(len(rows), 504)
        self.assertEqual(len({c.key(r) for r in rows}), 504)
        self.assertFalse(any(r['system'] == 'H4' and r['prefix'] == 11 for r in rows))
        for s, n in [('H4', 76), ('H6', 92)]:
            for f in c.FORMULAS:
                self.assertEqual(sum(r['system'] == s and r['formula'] == f for r in rows), n)

    def test_q16_retained_outside_common_domain(self):
        r = c.normalized('H4', dict(method='B0', prefix=1, q=16, formula='4th'))
        self.assertEqual(r['q'], 16)
        self.assertNotIn(16, c.QS)

    def test_pr_and_unknown_formula_rejected(self):
        for row in [dict(method='PR', prefix=3, q=2, formula='2nd'),
                    dict(method='B0', prefix=3, q=2, formula='newformula')]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                c.normalized('H4', row)

    def test_overlay_dependency_without_cost_alias(self):
        base = dict(system='H4', method='B0', prefix=4, q=2, formula='4th(m5_best)',
                    signal_status='MISSING_SIGNAL')
        overlay = dict(base, method='B0I')
        tasks = c.signal_tasks([base, overlay])
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[1]['depends'], [tasks[0]['id']])
        self.assertNotEqual(tasks[0]['id'], tasks[1]['id'])

    def test_saved_base_removes_new_base_call(self):
        base = dict(system='H6', method='B0', prefix=6, q=4, formula='4th', signal_status='SAVED_RAW_SIGNAL')
        tasks = c.signal_tasks([base, dict(base, method='B0I', signal_status='MISSING_SIGNAL')])
        self.assertEqual([t['kind'] for t in tasks], ['B0I_ANALYTIC_OVERLAY'])
        self.assertEqual(tasks[0]['depends'], [])

    def test_cross_endpoint_axis_cannot_qualify(self):
        v = labels('INELIGIBLE', 'INELIGIBLE')
        v['0.05']['real'] = 'ELIGIBLE'; v['0.01']['imag'] = 'ELIGIBLE'
        self.assertEqual(c.paired_eligibility(v), ([], []))

    def test_unresolved_is_not_ineligible(self):
        v = labels('UNDETERMINED', 'ELIGIBLE')
        self.assertEqual(c.paired_eligibility(v), ([], list(c.EPS)))
        del v['0.05']['imag']
        with self.assertRaises(ValueError):
            c.paired_eligibility(v)

    def test_partial_or_unknown_labels_rejected(self):
        for v in [dict(labels(), extra=dict(real='ELIGIBLE', imag='ELIGIBLE')), labels('MISSING')]:
            with self.subTest(v=v), self.assertRaises(ValueError):
                c.paired_eligibility(v)

    def test_compile_only_eligible_missing_pair(self):
        tasks = [dict(id=x) for x in ('a', 'b', 'c')]
        q = dict(a=labels(), b=labels(), c=labels('INELIGIBLE', 'ELIGIBLE'))
        selected, unknown = c.select_compile_tasks(tasks, q, {'b':identity_receipt()})
        self.assertEqual([x['id'] for x in selected], ['a'])
        self.assertEqual(unknown, [])

    def test_extra_candidate_or_unregistered_reuse_rejected(self):
        for q, a in [(dict(a=labels(), b=labels()), {}), (dict(a=labels()), {'unknown':identity_receipt()})]:
            with self.subTest(q=q), self.assertRaises(ValueError):
                c.select_compile_tasks([dict(id='a')], q, a)

    def test_reuse_id_flags_or_changed_values_rejected(self):
        bad = identity_receipt(); bad['observed']['preparation'] = 'wrong'
        for reuse in [{'a'}, {'a':bad}, {'a':dict(identity_receipt(), receipt_sha256s=[])}]:
            with self.subTest(reuse=reuse), self.assertRaises(ValueError):
                c.select_compile_tasks([dict(id='a')], dict(a=labels()), reuse)

    def test_identity_requires_every_value(self):
        keys = ('hamiltonian', 'state', 'scientific_arrays', 'preparation', 'native_policy',
                'formula_coefficients', 'scalar_control_policy', 'compiler', 'runtime_closure')
        expected = {k:'saved_'+k for k in keys}
        self.assertTrue(c.identity_equal(expected, dict(expected)))
        for k in keys:
            observed = dict(expected, **{k:'DIFFERENT'})
            with self.subTest(field=k):
                self.assertFalse(c.identity_equal(expected, observed))
        self.assertFalse(c.identity_equal({}, {}))

    def test_resource_allocation_has_no_constant_worker_cap(self):
        args = dict(pending=300, available_cpu_slots=192, available_memory_bytes=1000,
                    worker_memory_bytes=1, free_disk_bytes=1000, worker_disk_bytes=1)
        self.assertEqual(c.choose_workers(**args), 192)
        self.assertEqual(c.choose_workers(**dict(args, available_memory_bytes=7)), 7)
        self.assertEqual(c.choose_workers(**dict(args, free_disk_bytes=3)), 3)
        self.assertEqual(c.choose_workers(**dict(args, available_cpu_slots=0)), 0)

    def test_invalid_resource_estimate_rejected(self):
        with self.assertRaises(ValueError):
            c.choose_workers(pending=1, available_cpu_slots=1, available_memory_bytes=1,
                             worker_memory_bytes=0, free_disk_bytes=1, worker_disk_bytes=1)

    def test_fixed_compile_science_caps(self):
        tasks = [dict(candidate=dict(q=2, formula='4th')), dict(candidate=dict(q=4, formula='4th(m5_best)'))]
        caps = c.compile_caps(tasks)
        self.assertEqual(caps['native_signed_S2_pieces'], 50)
        self.assertEqual(caps['wrapper_compile'], 4)
        self.assertEqual(caps['sampling'], 0)
        self.assertEqual(caps['new_reference'], 0)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'new'
        self.contract = dict(SOURCE='artificial_source', stage='TOY', retry=False, resume=False, caps=dict(signal=1))
        self.approval = dict(execution_authorized=True, plan_digest=c.digest(self.contract),
                             SOURCE='artificial_source', stage='TOY', grant_id='artificial_grant')
        self.fresh = dict(source_bytes=True, input_equivalence=True, preparation=True,
            compiler_runtime=True, resources=True, monotonic=time.monotonic(), plan_digest=c.digest(self.contract))

    def open(self, **kwargs):
        return c.OneShot(kwargs.get('contract', self.contract), kwargs.get('approval', self.approval),
                         kwargs.get('fresh', self.fresh), self.root)

    def test_unapproved_refusal_has_no_files_or_callback(self):
        with self.assertRaisesRegex(ValueError, 'APPROVAL'):
            self.open(approval=dict(self.approval, execution_authorized=False))
        self.assertFalse(self.root.exists())

    def test_source_stage_and_grant_changes_rejected(self):
        for k, v in [('SOURCE', 'other'), ('stage', 'other'), ('grant_id', None)]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.open(approval=dict(self.approval, **{k:v}))
        self.assertFalse(self.root.exists())

    def test_changed_plan_refused(self):
        with self.assertRaises(ValueError):
            self.open(contract=dict(self.contract, caps=dict(signal=2)))

    def test_incomplete_or_expired_gate_refused(self):
        for field in ('source_bytes', 'input_equivalence', 'preparation', 'compiler_runtime', 'resources'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.open(fresh=dict(self.fresh, **{field:False}))
        with self.assertRaisesRegex(ValueError, 'EXPIRED'):
            self.open(fresh=dict(self.fresh, monotonic=time.monotonic()-31))
        self.assertFalse(self.root.exists())

    def test_budget_and_reserved_ledger(self):
        run = self.open()
        self.assertEqual(run.call('signal', lambda:'toy_result'), 'toy_result')
        self.assertTrue((self.root/'ledger_signal_0.json').exists())
        with self.assertRaisesRegex(ValueError, 'BUDGET'):
            run.call('signal', lambda:self.fail('extra callback'))

    def test_failure_is_consumed_no_resume(self):
        run = self.open()
        def fail():
            raise RuntimeError('artificial failure')
        with self.assertRaises(RuntimeError):
            run.call('signal', fail)
        self.assertEqual(run.used, dict(signal=1))
        self.assertTrue((self.root/'CONSUMED.json').exists())
        self.assertTrue((self.root/'STOP.json').exists())
        with self.assertRaises(FileExistsError):
            self.open()
        with self.assertRaisesRegex(ValueError, 'STOPPED'):
            run.call('signal', lambda:self.fail('retry callback'))

    def test_unbudgeted_call_never_enters_callback(self):
        run = self.open()
        with self.assertRaises(ValueError):
            run.call('compile', lambda:self.fail('compile callback'))


if __name__ == '__main__':
    unittest.main()
