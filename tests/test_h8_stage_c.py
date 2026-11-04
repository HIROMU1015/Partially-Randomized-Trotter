"""Artificial metadata and three-qubit confirmation integration only."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h8_long_time_adapter as a
import h8_stage_c as c


def fixture():
    pr = a.candidate('PR', '4th(m5_best)', 8, 1, 32, 4)
    pf = a.candidate('B0I', '4th(m5_best)', 15, 1)
    endpoints = [dict(candidate=r, status='ELIGIBLE', N_cos=2, N_sin=3) for r in (pr, pf)]
    rows = [dict(candidate=pr, cohort='EXPLORATION', n=9, N_cos=2, N_sin=3,
                 mean_G_exact='90', sample_variance_G_exact='144', sample_SE_G=4.),
            dict(candidate=pf, cohort='EXPLORATION', n=1, N_cos=2, N_sin=3,
                 mean_G_exact='100', sample_variance_G_exact=None, sample_SE_G=None)]
    binding = dict(hamiltonian='artificial', compiler=a.COMPILER)
    signal = dict(status='PASS', SOURCE=c.SIGNAL_SOURCE, binding=binding,
                  results=[dict(endpoint=e) for e in endpoints])
    exploration = dict(status='COMPLETE_EXPLORATION_ONLY', native_SOURCE=c.NATIVE_SOURCE,
                       binding=binding, source_files={}, rows=rows)
    return signal, exploration


class MetadataTests(unittest.TestCase):
    def design(self):
        return c.proposal(*fixture(), evidence={'stage_B': {'sha256': 'artificial'}})

    def test_saved_ids_n_precision_and_statistics_are_copied(self):
        s, b = fixture()
        before = deepcopy(b)
        with patch.object(a, 'confirmation_n', side_effect=AssertionError('no numerical redesign')), \
             patch.object(a, 'paired_G', side_effect=AssertionError('no reaggregation')):
            design = c.proposal(s, b, {})
        self.assertEqual(design['n'], b['rows'][0]['n'])
        self.assertEqual(design['absolute_SE_G_target'], b['rows'][0]['sample_SE_G'])
        self.assertEqual(design['comparison']['PR']['saved_aggregate'], before['rows'][0])
        self.assertEqual(design['comparison_ids']['PF'], b['rows'][1]['candidate']['id'])
        self.assertEqual(b, before)
        self.assertFalse(design['numerical_n_formula_evaluated'])
        self.assertIsNone(design['confidence_level'])

    def test_changed_shots_source_and_duplicate_selection_are_rejected(self):
        for change in ('shots', 'source', 'duplicate'):
            s, b = fixture()
            if change == 'shots':
                b['rows'][0]['N_cos'] += 1
            elif change == 'source':
                b['native_SOURCE'] = 'different'
            else:
                b['rows'].append(deepcopy(b['rows'][0]))
            with self.assertRaises(ValueError):
                c.proposal(s, b, {})

    def test_fixed_design_rejects_post_hoc_n_candidate_or_direction_change(self):
        original = self.design()
        for key, value in [('n', 10), ('comparison_direction', 'PF_LOWER'),
                           ('automatic_top_up', True), ('cohort', 'EXPLORATION')]:
            with patch.object(c, 'published_proposal', return_value=original), self.assertRaises(ValueError):
                c.validate_design(dict(original, **{key: value}))

    def test_new_namespace_and_seed_collisions_with_artificial_row(self):
        row = fixture()[0]['results'][0]['endpoint']['candidate']
        first = c.fresh_seed_map(row, 2, 'a' * 40, {'input': 'artificial'}, 'fixture_run', set())
        again = c.fresh_seed_map(row, 2, 'a' * 40, {'input': 'artificial'}, 'fixture_run', set())
        self.assertEqual(first, again)
        self.assertEqual(len(first), len(set(first.values())))
        with self.assertRaisesRegex(ValueError, 'SEED_COLLISION'):
            c.fresh_seed_map(row, 2, 'a' * 40, {'input': 'artificial'}, 'fixture_run', {next(iter(first.values()))})
        original_digest = a.digest
        def collision_only_for_seeds(value):
            return '0' * 64 if isinstance(value, list) and value[0] == c.NAMESPACE else original_digest(value)
        with patch.object(a, 'digest', side_effect=collision_only_for_seeds), self.assertRaisesRegex(ValueError, 'SEED_COLLISION'):
            c.fresh_seed_map(row, 2, 'a' * 40, {}, 'fixture_run', set())
        self.assertTrue(set(first.values()).isdisjoint(c.fresh_seed_map(
            row, 2, 'a' * 40, {'input': 'artificial'}, 'another_run', set()).values()))

    def test_fixed_PF_keeps_its_origin_and_no_top_up_on_failed_precision(self):
        pr = dict(cohort='CONFIRMATION', pooled=False, mean_G_exact='80', sample_SE_G=5.)
        pf = dict(cohort='EXPLORATION', n=1, mean_G_exact='100')
        original = deepcopy(pf)
        value = c.compare_fixed_PF(pr, pf, 1.)
        self.assertTrue(value['observed_PR_lower'])
        self.assertFalse(value['descriptive_precision_met'])
        self.assertFalse(value['automatic_top_up'])
        self.assertFalse(value['population_guarantee'])
        self.assertEqual(pf, original)
        with self.assertRaisesRegex(ValueError, 'NOT_POOLED'):
            c.compare_fixed_PF(dict(pr, cohort='EXPLORATION'), pf, 1.)

    def test_preflight_never_enters_scientific_runtime_or_seed_derivation(self):
        import h8_long_time_runtime as runtime
        design = self.design()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p, body in [('scripts/h8_stage_c.py', b'new artificial code'),
                            ('scripts/h8_compiler_phase_bookkeeping.py', b'artificial phase code'),
                            ('configs/h8_long_time/stage_c_proposal_20261104.json', b'artificial proposal')]:
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                (root / p).write_bytes(body)
            seeds = []
            for i in (0, 1):
                path = root / ('seed_fixture_' + str(i))
                path.write_text('artificial metadata')
                seeds.append(dict(path=str(path), sha256=a.sha(path)))
            config = dict(SOURCE=c.NATIVE_SOURCE, source_files={}, binding=design['binding'],
                          stage_B_seed_files=seeds, confirmation_run_root=str(root / 'runs'))
            def show(argv, **kwargs):
                return (root / argv[-1].split(':', 1)[1]).read_bytes()
            with patch.object(c, 'published_proposal', return_value=design), patch.object(a, 'ROOT', root), \
                 patch.object(a, 'preflight', return_value={}), patch.object(c.subprocess, 'check_output', side_effect=show), \
                 patch.object(runtime, 'Context', side_effect=AssertionError('no physical setup')), \
                 patch.object(c, 'fresh_seed_map', side_effect=AssertionError('no real confirmation seeds')):
                result = c.preflight(design, config, 'a' * 40, 'artificial_run', 3)
            self.assertEqual(result['preflight_scientific_calls'], 0)
            self.assertEqual(result['cohort'], 'CONFIRMATION')
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['comparison_ids']['PF'], design['comparison_ids']['PF'])

    def test_import_does_not_load_scientific_dependencies(self):
        code = ('import sys; sys.path.insert(0, "scripts"); import h8_stage_c; '
                'assert "numpy" not in sys.modules; assert "qiskit" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


class ArtificialCircuitTests(unittest.TestCase):
    def test_confirmation_uses_original_seven_checks_and_phase_on_cache_fallback(self):
        import numpy as np
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import XXPlusYYGate
        from qiskit.quantum_info import Statevector
        import h46_long_time_native as native
        import h8_long_time_runtime as runtime
        sys.path.insert(0, str(ROOT / 'src'))
        import trotterlib.df_trotter.circuit as circuits
        initial = np.array([1., 0., 0., 0.], complex)
        evo = QuantumCircuit(3)
        evo.append(XXPlusYYGate(.37, .19), [0, 1])
        evo.append(XXPlusYYGate(.41, .19), [0, 1])
        def wrapper(evolution, *, axis, **kwargs):
            qc = QuantumCircuit(3, 1, name=axis)
            qc.h(2)
            qc.compose(evolution.circuit, inplace=True)
            if axis == 'sine':
                qc.sdg(2)
            qc.h(2); qc.measure(2, 0)
            return qc
        compiled = {}
        def raw_compile(qc, fp, representative):
            result = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1,
                               seed_transpiler=17, num_processes=1)
            compiled[fp] = result
            return dict(compiler=a.COMPILER, fingerprint=fp,
                        metrics=dict(rz_count=result.count_ops()['rz'])), None, True
        @contextmanager
        def reuse():
            yield NS(fingerprint=lambda qc: qc.name)
        names = {'trottertracks.resource_applicability.h6_hash_reuse_v1': NS(use_fingerprint_reuse=reuse)}
        row = fixture()[0]['results'][0]['endpoint']['candidate']
        endpoint = dict(candidate=row, status='ELIGIBLE', N_cos=2, N_sin=3, log_B=0., log_B_margin=1e-12)
        ctx = NS(np=np, qstate=initial, preps={8: NS(preparation_hash='artificial')},
                 compiler='artificial', scalar=lambda row: 0., native=NS(build_native_hadamard_wrapper=wrapper))
        fallback = lambda qc, compiler, **kw: NS(rz_count=compiled[qc.name].count_ops()['rz'],
                                                transpiled_circuit=compiled[qc.name])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, names), \
             patch.object(c, 'CONFIG', {'read_only_compiled_caches': []}), patch.object(c, 'SCOPE', {'workers': 1}), \
             patch.object(c.os, 'sched_getaffinity', return_value={0}), patch.object(runtime, 'CONTEXT', ctx), \
             patch.object(runtime, 'independent_trajectory', return_value=initial.copy()), \
             patch.object(circuits, 'simulate_statevector', side_effect=lambda qc, v: Statevector(v).evolve(qc).data), \
             patch.object(native, 'sample_requests', return_value=([], ['artificial'], [])), \
             patch.object(native, 'compose', return_value=NS(circuit=evo)), \
             patch.object(native, 'compile_cached', side_effect=raw_compile), \
             patch('trotterlib.rte_compiled_cost.transpile_and_measure_cost', side_effect=fallback):
            value = c.confirmation_price_job((row, endpoint, 'CONFIRMATION', 0, str(Path(tmp) / 'trial')))
        self.assertEqual(value['cohort'], 'CONFIRMATION')
        self.assertEqual(len(value['checks']), 7)
        self.assertEqual(len(value['compiler_phase_accounting']), 2)
        self.assertTrue(all(v['error'] < v['tolerance'] and not v['phase_alignment'] for v in value['checks']))


if __name__ == '__main__':
    unittest.main()
