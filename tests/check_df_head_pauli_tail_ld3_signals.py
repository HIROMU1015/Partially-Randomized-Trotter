"""LD3 stage-2 checks: artificial operators/state/configuration, saved JSON metadata.

Real saved arrays are never loaded: every .npz blob is replaced by an
artificial archive. No circuit, sampling, price, N or Q3.
"""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
package = ModuleType('trotterlib'); package.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules.setdefault('trotterlib', package)
from trottertracks import df_head_pauli_tail_ld3_signals as F
import run_df_head_pauli_tail_ld3_signals as R

L, p0, p1 = F.L, F.p0, F.p1


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

B_FIX = module('LD3S_B_fixture', 'tests/check_pr_rpe_signal_plan.py')
OP_FIX = module('LD3S_operator_fixture', 'tests/check_df_head_pauli_tail_ld3.py')
SOURCE = '2'*40; RAW = '3'*40; ART = '4'*40


def artificial():
    """Five-fragment operators, sector state and artificial P0 archives."""
    ham, sector, full_state, state, meta = OP_FIX.fixture()
    car = p0.car_api(ROOT)(ham, SimpleNamespace(basis_indices=np.arange(8)), full_state, meta['saved_tail_descriptors'])
    common = dict(blocks=np.asarray(car.matrices), sector_basis_indices=sector.basis_indices)
    ops = {p: dict(Pauli_nonidentity=sum(car.matrices[p+1:], np.zeros((8, 8), complex))
        - meta['saved_tail_descriptors'][str(p)]['scalar']*np.eye(8)) for p in (1, 2, 3)}
    def npz(value):
        buffer = io.BytesIO(); np.savez_compressed(buffer, **value); return buffer.getvalue()
    archives = {'COMMON_OPERATORS.npz': npz(common), **{'LD'+str(p)+'_OPERATORS.npz': npz(v) for p, v in ops.items()}}
    original = p0.blob
    def blob(root, commit, path):
        if path.endswith('.npz'):
            return archives[Path(path).name]
        return original(root, commit, path)
    loaded = (ham, sector, full_state, state, {'artificial_loader': True})
    return ham, meta, blob, loaded


def artificial256():
    """P1 load_contexts also builds LD1 full256, so this fixture is 256-dimensional."""
    rng = np.random.default_rng(1116)
    def herm():
        x = rng.normal(size=(256, 256))+1j*rng.normal(size=(256, 256)); return (x+x.conj().T)/64
    basis = np.arange(0, 256, 7)[:36]
    state = rng.normal(size=36)+1j*rng.normal(size=36); state /= np.linalg.norm(state)
    full_state = np.zeros(256, complex); full_state[basis] = state
    common = dict(blocks=np.asarray([herm() for _ in range(6)]), sector_basis_indices=basis)
    def npz(value):
        buffer = io.BytesIO(); np.savez_compressed(buffer, **value); return buffer.getvalue()
    archives = {'COMMON_OPERATORS.npz': npz(common),
        **{'LD'+str(p)+'_OPERATORS.npz': npz(dict(Pauli_nonidentity=herm())) for p in (1, 2)}}
    original = p0.blob
    def blob(root, commit, path):
        return archives[Path(path).name] if path.endswith('.npz') else original(root, commit, path)
    loaded = (SimpleNamespace(constant=.17), SimpleNamespace(basis_indices=basis), full_state, state, {'artificial': 256})
    return blob, loaded


def item(prefix, lam, scalar, *, commit=ART, directory='ARTIFICIAL/run01/'):
    return dict(payload=dict(prefix=prefix, lambda_P=lam, saved_DF_scalar=scalar, mapping='ARTIFICIAL'),
        dictionary_sha256='ARTIFICIAL_'+str(prefix), source=dict(commit=commit, path=directory+'LD'+str(prefix)+'_DICTIONARY.json'),
        P0_metrics=dict(artificial=True), P0_result_source=dict(commit=commit, path=directory+'result.json'))


def configuration(meta, lam=.7):
    b = p1.b_api(ROOT); _, original, _, _ = B_FIX.fixture(prefix=3, M=12, kappa='2')
    original = deepcopy(original); original['spec']['binding']['rank'] = 5
    it = item(3, lam, meta['saved_tail_descriptors']['3']['scalar'])
    return b, it, L.generate_configuration(ROOT, b, original, it)


def reference(ctx):
    def ref(job):
        H = sum(ctx.matrices)+ctx.constant*np.eye(len(ctx.state)); t = job['physical_context']['physical_candidate']['T']
        z = complex(np.vdot(ctx.state, expm(-1j*t*H)@ctx.state)); v = dict(real=z.real, imag=z.imag)
        return dict(binding=ctx.binding, T=t, source=dict(fixture='ARTIFICIAL_REFERENCE'),
            evidence=dict(reference=v, independent_reference=v, signal_difference=0.))
    return ref


def setup():
    ham, meta, blob, loaded = artificial()
    b, it, config = configuration(meta)
    numerical = dict(commit=SOURCE, path=p1.provider.MODULE, adapter=F.MODULE, kernel_source=F.P1_KERNEL,
        recipe=p1.provider.RECIPE)
    jobs = {m: p1.job(config, m, numerical, b) for m in F.ROUNDS}
    saved = dict(binding=config['spec']['binding'])
    with patch.object(p0, 'blob', side_effect=blob):
        ctx = F.load_context(ROOT, saved, it, {}, budget=13, loaded=loaded)
    return b, it, config, jobs, ctx, saved, blob, loaded


def evaluator(ctx):
    ref = reference(ctx)
    return lambda m, job: p1.provider.evaluate(ctx, job, ref(job), root=ROOT)


class LD3SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.item, cls.config, cls.jobs, cls.ctx, cls.saved, cls.blob, cls.loaded = setup()
        cls.blob = staticmethod(cls.blob)

    def test_b_intake_identity_reproduces_saved_LD1_LD2_LD3_descriptors(self):
        saved = L.p0_metadata(ROOT, (1, 2, 3))
        for commit, directory, prefix in ((L.SAVED_P0['commit'], L.SAVED_P0['directory'], 1),
                (L.SAVED_P0['commit'], L.SAVED_P0['directory'], 2), (F.P0_LD3['commit'], F.P0_LD3['directory'], 3)):
            full = L.dictionary_item(ROOT, commit, directory, prefix, saved)
            b_item = {k: full[k] for k in ('payload', 'dictionary_sha256', 'source')}
            self.assertEqual(F.b_intake_dictionary_identity(b_item), L.dictionary_identity(full))
        self.assertEqual(F.b_intake_dictionary_identity(
            {k: full[k] for k in ('payload', 'dictionary_sha256', 'source')}),
            L.read(ROOT, F.CONFIG['commit'], F.CONFIG['path'])['spec']['split']['tail_dictionary'])
        self.assertEqual(F.operators_locator(full), dict(commit=F.P0_LD3['commit'], path=F.P0_LD3['directory']+'LD3_OPERATORS.npz'))

    def test_context_builder_equals_P1_load_contexts_on_artificial_arrays(self):
        blob, loaded = artificial256()
        binding = dict(system='ARTIFICIAL', rank=5)
        d = {p: dict(payload=dict(prefix=p, lambda_P=.5+p, saved_DF_scalar=.01*p,
            mapping='ARTIFICIAL'), dictionary_sha256='ARTIFICIAL_'+str(p),
            source=dict(commit=p1.P0, path=p1.P0DIR+'run01/LD'+str(p)+'_DICTIONARY.json'), P0_metrics=dict(p=p))
            for p in (1, 2)}
        fixed_meta = dict(saved=dict(binding=binding), dictionaries=d)
        with patch.object(p0, 'blob', side_effect=blob), patch.object(p0, 'load_saved', return_value=loaded):
            reference = p1.load_contexts(ROOT, fixed_meta, None, {})[2]['sector36']
            mine = F.load_context(ROOT, fixed_meta['saved'], dict(d[2], P0_result_source=dict(commit=p1.P0,
                path=p1.P0DIR+'run01/result.json')), {}, budget=0)
            self.assertTrue(F.same_context(mine, reference))
            other = F.load_context(ROOT, fixed_meta['saved'], dict(d[1], P0_result_source=dict(commit=p1.P0,
                path=p1.P0DIR+'run01/result.json')), {}, budget=0)
            self.assertFalse(F.same_context(other, reference))

    def test_13_artificial_rounds_pass_and_packet_matches_B_intake_conditions(self):
        result = dict(records=[]); products = {}
        ok = F.run(self.jobs, evaluator(self.ctx), lambda n, v: products.update({n: deepcopy(v)}), result)
        self.assertTrue(ok, result.get('error')); self.assertEqual(result['status'], F.SUCCESS)
        records = {r['m']: r['record'] for r in result['records']}
        self.assertEqual(sorted(records), list(F.ROUNDS))
        self.assertEqual(len(self.ctx.normalizers), len({r['physical_context']['physical_candidate']['R_outer']
            for r in records.values()}))
        handoff = F.packet(self.config, records, self.jobs, raw_commit=RAW, science_source=SOURCE)
        self.assertEqual(handoff['status'], F.SUCCESS)
        pkt = handoff['packets'][0]
        self.assertEqual(pkt['configuration'], self.config)
        self.assertEqual(pkt['evaluated_rounds'], list(F.ROUNDS)); self.assertTrue(pkt['signal_coverage_complete'])
        self.assertEqual(pkt['dictionary_identity'], self.config['spec']['split']['tail_dictionary'])
        self.assertEqual(pkt['dictionary_identity'], F.b_intake_dictionary_identity(
            {k: self.item[k] for k in ('payload', 'dictionary_sha256', 'source')}))
        for row, canonical in zip(pkt['rounds'], self.config['rounds']):
            rec = row['mean_record']
            self.assertEqual(row['identity'], canonical['identity']); self.assertEqual(row['law_binding'], canonical['law_binding'])
            self.assertEqual(row['mean_source']['scientific_source'], rec['source']['commit'])
            self.assertTrue(rec['scalar_once']); self.assertIsNone(row['N_X']); self.assertIsNone(rec['C'])
            self.assertEqual(rec['physical_context'], self.jobs[row['m']]['physical_context'])
            self.assertEqual(rec['physical_context']['split']['tail_dictionary']['scalar_policy'], 'KEEP_SAVED_sR_AND_c_ONCE')
        row = F.summary_row(records[12], carrier=RAW, path='x', science_source=SOURCE)
        self.assertEqual(row['event_count'], row['qPF']*row['R_outer'])

    def test_first_failure_or_identity_mismatch_stops(self):
        good = evaluator(self.ctx)
        def failing(m, job):
            value = good(m, job)
            if m == 1:
                value = dict(value, status='NUMERICAL_FAILURE', numerical=dict(value['numerical'], gate_pass=False))
            return value
        result = dict(records=[])
        self.assertFalse(F.run(self.jobs, failing, lambda *a: None, result))
        self.assertEqual(result['status'], 'STOP_NUMERICAL_FAILURE'); self.assertEqual(len(result['records']), 2)
        records = {r['m']: r['record'] for r in result['records']}
        partial = F.packet(self.config, records, self.jobs, raw_commit=RAW, science_source=SOURCE)
        self.assertEqual(partial['packets'][0]['failed_rounds'], [1]); self.assertEqual(partial['packets'][0]['evaluated_rounds'], [0])
        self.assertEqual(partial['status'], 'INCOMPLETE_NUMERICAL_OR_INPUT_STOP')
        def foreign(m, job):
            return dict(good(m, job), B_candidate_id='FOREIGN')
        result = dict(records=[])
        self.assertFalse(F.run(self.jobs, foreign, lambda *a: None, result))
        self.assertEqual(result['status'], 'STOP_RETURNED_IDENTITY_SOURCE_OR_LAW_MISMATCH')
        with self.assertRaises(ValueError):
            F.run({m: j for m, j in self.jobs.items() if m}, good, lambda *a: None, dict(records=[]))
        with self.assertRaises(ValueError):
            F.packet(self.config, {0: dict(records[0], scalar_once=False)}, self.jobs, raw_commit=RAW, science_source=SOURCE)

    def test_runner_execute_and_assemble_entries_on_artificial_inputs(self):
        ref = reference(self.ctx)
        real_read, real_load = L.read, F.load_context
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'signals'; run01 = out/'run01'
            def read(root, commit, path):
                if commit == RAW:
                    return json.loads((out/Path(path).relative_to(F.DIRECTORY)).read_text())
                return real_read(root, commit, path)
            def load(root, saved, it, normalizers, *, budget, archive=None, loaded=None):
                with patch.object(p0, 'blob', side_effect=self.blob):
                    return real_load(root, saved, it, normalizers, budget=budget, loaded=self.loaded)
            fixed = (dict(self.saved, inputs={}, context_source=dict(path=__file__, sha256=p0.sha(__file__))),
                self.item, self.config, self.b)
            with patch.object(R, 'head', return_value=SOURCE), patch.object(R, 'sources', return_value=[]), \
                    patch.object(F, 'load_fixed', return_value=fixed), \
                    patch.object(R, 'ld2_context_control', return_value=dict(equal=True)), \
                    patch.object(F, 'load_context', side_effect=load), \
                    patch.object(p1.provider, 'saved_h4_reference', side_effect=lambda root, job: ref(job)):
                result = R.execute(run01, SOURCE)
                self.assertEqual(result['status'], F.SUCCESS, result.get('error'))
                self.assertEqual(result['scope']['new_mean_calls'], 13); self.assertEqual(result['not_started'], [])
                self.assertEqual(sorted(p.name for p in run01.glob('LD3_m*.json')), sorted('LD3_m%d.json' % m for m in F.ROUNDS))
                with self.assertRaises(ValueError):
                    R.execute(run01, SOURCE)
            with patch.object(R, 'head', return_value=RAW), patch.object(R, 'is_ancestor', return_value=True), \
                    patch.object(p0, 'blob', side_effect=lambda root, commit, path: (ROOT/path).read_bytes()
                        if commit == SOURCE else self.blob(root, commit, path)), \
                    patch.object(F, 'load_fixed', return_value=fixed), patch.object(L, 'read', side_effect=read):
                value = R.assemble(out, SOURCE, RAW)
            self.assertEqual(value['status'], F.SUCCESS)
            handoff = json.loads((out/'MEAN_HANDOFF.json').read_text())
            self.assertEqual(handoff['packets'][0]['rounds'][12]['mean_source'],
                dict(commit=RAW, path=F.DIRECTORY+'run01/LD3_m12.json', selector=['record'], scientific_source=SOURCE))
            self.assertEqual(len((out/'ROUND_SUMMARY.csv').read_text().splitlines()), 14)


if __name__ == '__main__':
    unittest.main()
