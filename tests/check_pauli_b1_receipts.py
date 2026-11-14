"""Artificial LD2 five-cell prices in a temporary Git repo; no H4 numerical calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]
import run_pauli_b1_prices as P


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='pauli_b1_receipt_fixture_')
        cls.root = Path(cls.temp.name); cls.E = P.receipts(); cls.plan = P.proposal(2, 'B1-CAL')
        cls.git('init', '-q')
        for name in P.REPLAY_SOURCES:
            file = cls.root / name; file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, file)
        cls.code = cls.commit('Artificial code snapshot')
        cls.cells = [c for c in cls.plan['cells'] if c['role'] == 'CALIBRATION']
        for cell in cls.cells:
            cls.save_cell(cell)
        cls.raw = cls.commit('Artificial five-cell raw results; no scientific acquisition')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def git(cls, *args):
        return subprocess.check_output(['git', *args], cwd=cls.root, stderr=subprocess.PIPE, text=True).strip()

    @classmethod
    def commit(cls, message):
        cls.git('add', '.')
        cls.git('-c', 'user.name=Artificial fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message)
        return cls.git('rev-parse', 'HEAD')

    @classmethod
    def put(cls, name, value):
        file = cls.root / name; file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value, indent=2) + '\n')

    @classmethod
    def save_cell(cls, cell):
        request, _, packet, _ = P.request_for(cls.plan, cell); dictionary = packet['dictionary_identity']
        directory = 'fixture/' + cell['cell_id']; result_path = directory + '/RESULT.json'
        seeds = P.seed_table(cls.plan, cell['cell_id'])
        source = dict(commit=cls.code, caller=P.CALLER, result_path=result_path, prefix=2, kind='B1-CAL',
                      cell=cell['cell_id'], cohort=cell['cohort'], step_seeds=seeds)
        private = cls.root / 'private' / cell['cell_id']; cache = private / 'compiled_cache'; cache.mkdir(parents=True)
        paths = []
        # A fake compiler produces only fictional counts and labelled placeholder
        # artifact bytes. These are never decoded as QPY or used for real prices.
        def fake_compile(index, axis):
            fp = hashlib.sha256(('ARTIFICIAL:' + cell['cell_id'] + str(index) + axis).encode()).hexdigest()
            metrics = dict(rz_count=100 + 10 * index + (axis == 'Y'), cx_count=7,
                           rz_depth=8, cx_depth=4, total_depth=20, circuit_size=150)
            record = dict(compiler=cell['profile']['compiler'], fingerprint=fp, metrics=metrics)
            (cache / (fp + '.json')).write_text(json.dumps(record))
            for suffix in ('_wrapper.qpy.gz', '_compiled.qpy.gz'):
                (cache / (fp + suffix)).write_bytes(('ARTIFICIAL FAKE COMPILER ' + fp + suffix).encode())
            return record
        for index in range(cell['n_paths']):
            prices, fingerprints, compiles = {}, {}, {}
            for axis, key in cls.E.AXES.items():
                record = fake_compile(index, axis); prices[key] = record['metrics']; compiles[key] = record
                fingerprints[key] = dict(fingerprint=record['fingerprint'], measurement=True, wrapper_qubits=9, cache_reused=False)
            result = dict(status='PASS', scalar=.3, n=8, prices=prices, fingerprints=fingerprints,
                compile_receipts=compiles, checks=[dict(check=k, error=0., tolerance=1e-7 if k.startswith('compiled_vector') else 1e-9)
                    for k in sorted(cls.E.CHECKS)], bit_order=dict(pass_=True), seeds=seeds[index])
            result['bit_order'] = {'pass': True}
            paths.append(dict(index=index, cohort=cell['cohort'], request=request, dictionary_identity=dictionary,
                              result=result, reused=False))
        receipts = cls.E.compile_receipts(paths, source, dictionary, private)
        output = P.records(cls.plan, cell, paths, source=source, evidence_path=directory + '/IMPLEMENTATION_EVIDENCE.json',
                           compile_receipts=receipts)
        cls.put(result_path, dict(status='FINITE_CELL_COMPLETE', child=dict(status='PASS', paths=paths),
                                  source=source, fixture='ARTIFICIAL_NOT_SCIENTIFIC'))
        for name, value in [('C_INPUTS.json', output['C_inputs']), ('PRICES.json', output['prices']),
                            ('SIDECARS_UNBOUND.json', output['sidecars']), ('IMPLEMENTATION_EVIDENCE.json', output['implementation_evidence']),
                            ('COMPILE_RECEIPTS.json', receipts)]:
            cls.put(directory + '/' + name, value)

    def bound(self, cell, commit=None):
        return P.bind_calibration_records(publication_commit=commit or self.raw,
            result_path='fixture/' + cell['cell_id'] + '/RESULT.json', root=self.root)

    def check(self, cell, bound):
        return P.read_calibration_records(self.plan, cell, bound['records'], bound['sidecars'], root=self.root)

    def mutate(self, name, change):
        # Restore this test-owned fixture only, so each rejection is isolated.
        self.git('checkout', self.raw, '--', '.')
        data = json.loads((self.root / name).read_text()); change(data); self.put(name, data)
        return self.commit('Isolated artificial mismatch')

    def test_all_five_cells_xy_trace_to_right_path_axis_and_cache(self):
        count = 0
        for cell in self.cells:
            bound = self.bound(cell); result = self.check(cell, bound)
            self.assertEqual(result['status'], 'CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION')
            for record, checked in zip(bound['records'], result['checked']):
                self.assertEqual(record['source']['commit'], self.code)
                self.assertEqual(record['source']['publication']['commit'], self.raw)
                self.assertEqual(record['source']['implementation_evidence']['commit'], self.raw)
                self.assertNotEqual(self.code, self.raw)
                for trace in checked['paths']:
                    receipt = self.E.read(self.root, trace['receipt']); path = self.E.read(self.root, trace['path'])
                    self.assertEqual(receipt['axis'], checked['axis']); self.assertEqual(receipt['index'], path['index'])
                    self.assertEqual(receipt['dictionary_identity'], path['dictionary_identity'])
                    self.assertEqual(receipt['record']['fingerprint'], trace['fingerprint'])
                    self.assertIsNone(receipt['request']['m'])
                count += 1
        self.assertEqual(count, 10)

    def test_axis_swapped_record_rejected(self):
        bound = self.bound(self.cells[0]); bound['records'].reverse()
        with self.assertRaises(ValueError): self.check(self.cells[0], bound)

    def test_missing_receipt_rejected(self):
        cell = self.cells[0]
        commit = self.mutate('fixture/S66/COMPILE_RECEIPTS.json', lambda d: d.pop('p0X'))
        with self.assertRaisesRegex(ValueError, 'receipt missing'): self.check(cell, self.bound(cell, commit))

    def test_fingerprint_mismatch_rejected(self):
        cell = self.cells[0]
        commit = self.mutate('fixture/S66/COMPILE_RECEIPTS.json', lambda d: d['p0X']['record'].update(fingerprint='f'*64))
        with self.assertRaisesRegex(ValueError, 'fingerprint'): self.check(cell, self.bound(cell, commit))

    def test_receipt_axis_path_dictionary_seed_metrics_mismatch_rejected(self):
        cell = self.cells[0]
        mutations = [lambda d: d['p0X'].update(axis='Y'), lambda d: d['p0X'].update(index=1),
            lambda d: d['p0X']['dictionary_identity'].update(dictionary_sha256='f'*64),
            lambda d: d['p0X'].update(seeds=[0]), lambda d: d['p0X']['record']['metrics'].update(rz_count=999)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                commit = self.mutate('fixture/S66/COMPILE_RECEIPTS.json', mutation)
                with self.assertRaises(ValueError): self.check(cell, self.bound(cell, commit))

    def test_raw_unbound_and_wrong_publication_rejected(self):
        cell = self.cells[0]; bound = self.bound(cell)
        bound['sidecars'][0]['implementation_evidence']['commit'] = self.code
        with self.assertRaises(ValueError): self.check(cell, bound)
        raw = self.E.read(self.root, self.E.locator('fixture/S66/SIDECARS_UNBOUND.json', commit=self.raw))
        self.assertIsNone(raw[0]['implementation_evidence']['commit'])
        with self.assertRaises(ValueError): self.check(cell, dict(records=bound['records'], sidecars=raw))

    def parent_refs(self, commit=None):
        return [dict(commit=commit or self.raw, execution_commit=self.code, path='fixture/S66/RESULT.json',
                     selector=['child', 'paths', i]) for i in (0, 1)]

    def test_parent_publication_source_request_and_order_checked_before_child(self):
        pair = next(c for c in self.cells if c['cell_id'] == 'P2R66')
        refs = self.parent_refs(); parents = P.parent_paths(self.plan, pair, refs, root=self.root)
        self.assertEqual([p['index'] for p in parents], [0, 1])
        self.assertEqual([p['receipt_source'] for p in parents], refs)
        mutations = [lambda d: d['source'].update(commit='0'*40),
            lambda d: d['child']['paths'][0]['dictionary_identity'].update(dictionary_sha256='f'*64),
            lambda d: d['child']['paths'][0]['request'].update(R=67)]
        for mutation in mutations:
            commit = self.mutate('fixture/S66/RESULT.json', mutation)
            with self.assertRaises(ValueError): P.parent_paths(self.plan, pair, self.parent_refs(commit), root=self.root)
        bad = self.parent_refs(); bad.reverse()
        plan = deepcopy(self.plan); plan['adopted'] = True
        job = dict(plan=plan, cell_id='P2R66', stage='CALIBRATION', step_seeds=P.seed_table(plan, 'P2R66'),
                   duplicate_preflight_complete=True, parents=bad, ld2_wiring_source={})
        original = P.parent_paths
        with patch.object(P, 'ld2_wiring_gate'), patch.object(P, 'parent_paths', side_effect=lambda a,b,c: original(a,b,c,root=self.root)), \
                patch.object(P.subprocess, 'call') as child:
            with self.assertRaises(ValueError): P.execute(job)
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
