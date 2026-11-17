"""Artificial acquisition leaves only; no compiler, signal, Q3 or molecule."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('price_validation_leaf', ROOT/'scripts/run_rpe_price_validation_job.py')
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def request():
    return dict(schema=W.SCHEMA, model_id='ARTIFICIAL_MODEL',
        frozen_model=dict(source=dict(commit='fixed', path='model.json'), coefficients=dict(one=1, q=2)),
        prediction_source=dict(commit='fixed', path='prediction.json'), implementation=dict(lowering_id='ARTIFICIAL'),
        selected_round=dict(m=3, q=8, R=4, physical_id='ARTIFICIAL_ROUND'),
        predicted_RZ=dict(X=W.enc(120), Y=W.enc(100)),
        acquisition=dict(kind='PR_PAULI'), independent_paths=2)


def observations():
    return [dict(path_index=p, axis=a, compiled_RZ=100 + 20*p,
        source=dict(commit='artificial', path='result.json'),
        receipt=dict(commit='artificial', path='receipt.json', selector=[str(p)+a]), valid=True)
        for p in (0, 1) for a in ('X', 'Y')]


class ValidationWorkerTests(unittest.TestCase):
    def test_real_subprocess_route_with_artificial_existing_leaf(self):
        """The subprocess and result extraction run; the leaf only writes JSON."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'ARTIFICIAL_REPO'
            root.mkdir()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            script = root/'scripts/run_pauli_split_cal.py'
            script.parent.mkdir()
            values = observations()
            script.write_text('import json\nfrom pathlib import Path\nPath("ARTIFICIAL_RESULT.json").write_text('+repr(json.dumps(dict(rows=values)))+')\n')
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Artificial', '-c', 'user.email=artificial@example.invalid',
                            'commit', '-qm', 'Artificial I/O leaf only'], cwd=root, check=True)
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            req = request()
            req['acquisition']['commands'] = [dict(cwd=str(root), argv=[sys.executable, '-B', str(script)],
                source=dict(commit=commit, files={'scripts/run_pauli_split_cal.py':hashlib.sha256(script.read_bytes()).hexdigest()}),
                observations=[dict(path_index=row['path_index'], axis=row['axis'], path=str(root/'ARTIFICIAL_RESULT.json'),
                    value_selector=['rows', i, 'compiled_RZ'], valid_selector=['rows', i, 'valid'],
                    source=row['source'], receipt=row['receipt']) for i, row in enumerate(values)])]
            result = W.run_job(req, Path(temp)/'out')
            self.assertEqual(result['status'], 'INDEPENDENT_MEASURED')
            self.assertEqual(result['rows'][0]['observed_mean_RZ'], W.enc(110))
            self.assertEqual(W.read(Path(temp)/'out/LEAF_0_RETURN.json')['returncode'], 0)

    def test_builtin_factory_is_metadata_only_and_preserves_packet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            connection = root/'src/trottertracks/pauli_tail_price_connection.py'
            connection.parent.mkdir(parents=True)
            # Deliberately artificial request helper; no physical signal/kernel.
            connection.write_text('from fractions import Fraction\nLOWERING_ID="ARTIFICIAL"\n'
                'def load_dictionary(root, identity): return {"ARTIFICIAL":True}\n'
                'def round_request(packet,m,dictionary): return {"candidate_id":"ARTIFICIAL_ROUND","q":2,"R":1,"T":Fraction(.2)}\n'
                'def seed(candidate,cohort,index,outer,piece): return 100+index*10+outer\n')
            cfg = dict(id='ARTIFICIAL_CFG', spec=dict(L_D=2, profile=dict(control='symmetric_directional', P=0,
                wrapper='full_Hadamard_with_measurement', qiskit_version='1.3.0', basis_gates=['rz','sx','x','cx'],
                optimization_level=1, seed_transpiler=17)),
                rounds=[dict(m=0, actual_delta_pf=W.enc(W.Fraction(.1)), candidate=dict(id='ARTIFICIAL_ROUND'), law_binding=dict(id='ARTIFICIAL_LAW'))])
            packet = dict(configuration=cfg, dictionary_identity=dict(source=dict(commit='p0', path='artificial/p0/DICTIONARY.json')),
                rounds=[dict(m=0, identity=dict(candidate=cfg['rounds'][0]['candidate']), law_binding=cfg['rounds'][0]['law_binding'])])
            req = request()
            req['selected_round'].update(m=0,q=2,R=1)
            source = dict(commit='signal', path='ARTIFICIAL_PACKET.json')
            packet_before = deepcopy(packet)
            def git_value(repo, locator):
                return {'packets':[packet]} if locator == source else dict(calibration=dict(step_seeds=[[1,2]]))
            def process(argv, **kwargs):
                if argv[1:3] == ['rev-parse','HEAD']:
                    return W.PAULI_SOURCE+'\n'
                if argv[1] == 'show':
                    return connection.read_bytes()
                return b''
            with mock.patch.object(W, 'git_read', side_effect=git_value), mock.patch.object(W.subprocess, 'check_output', side_effect=process):
                acquisition = W.build_pauli_acquisition(req, root=root, configuration=cfg, signal_source=source,
                    legacy_view=root, interpreter=sys.executable, execution_source=dict(commit='fixed', path=W.SCRIPT), output_dir=root/'out')
            self.assertEqual(packet, packet_before)
            self.assertEqual(acquisition['backend'], 'BUILTIN_PAULI_NORMAL_PATH')
            self.assertEqual(acquisition['seeds'], [[100,101],[110,111]])
            self.assertEqual(acquisition['seed_check']['overlap_with_calibration'], 0)
            self.assertEqual(acquisition['packet'], packet)
            self.assertFalse((root/'out').exists())

    def test_builtin_df_factory_is_one_holdout_and_no_calibration(self):
        req = request()
        req['acquisition']['kind'] = 'PR_DF'
        req['selected_round'].update(m=0, q=2, R=1)
        cfg = dict(id='ARTIFICIAL_DF', spec=dict(method='PR', L_D=2, formula='2nd', finite_K=2,
            binding=dict(hamiltonian='ARTIFICIAL_H'),
            split=dict(scalar=2.0, lambda_source=dict(commit='df', path='tail.json'))),
            rounds=[dict(m=0, candidate=dict(id='ARTIFICIAL_ROUND'), physical_candidate=dict(id='ARTIFICIAL_DF_ROUND',q=2),
                         q_PF=2, R_outer=1, structure=dict(signed_weights=[1.]))])
        from_b = mock.Mock(return_value=dict(physical_context=dict(physical_candidate=dict(id='ARTIFICIAL_DF_ROUND',q=2))))
        runtime = SimpleNamespace(D=SimpleNamespace(from_b=from_b),
            P=SimpleNamespace(trial_seed_function=lambda root: lambda c, cohort, i, outer, piece:100+i*10+outer))
        provider = SimpleNamespace(runtime=lambda: runtime, FIXED_VIEW=Path('/ARTIFICIAL'),
            SCALAR_CONSTANT=dict(commit='constant', path='constant.json'), BASE_INPUTS=dict(commit='inputs',path='inputs.json'))
        def git_value(root, loc):
            if loc == provider.SCALAR_CONSTANT:
                return dict(source=dict(hamiltonian_hash='ARTIFICIAL_H'),value=1.0)
            if loc == provider.BASE_INPUTS:
                return dict(runtime=dict(provider_root='ARTIFICIAL'), saved_input_files={},local_support={})
            return dict(calibration=dict(seeds=[[1,2]]))
        original = deepcopy(cfg)
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(W,'_df_provider',return_value=provider), mock.patch.object(W,'git_read',side_effect=git_value):
                acquisition = W.build_df_acquisition(req, root=temp, configuration=cfg,
                    configuration_source=dict(commit='config',path='config.json'), interpreter=sys.executable,
                    execution_source=dict(commit='fixed',path=W.SCRIPT), output_dir=Path(temp)/'out')
            self.assertEqual(cfg, original)
            self.assertEqual(acquisition['backend'],'BUILTIN_DF_NORMAL_PATH')
            self.assertEqual(acquisition['cell']['role'],'HOLDOUT')
            self.assertEqual(acquisition['cell']['indices'],[0,1])
            self.assertEqual(acquisition['cell']['seeds'],[[100,101],[110,111]])
            self.assertEqual(from_b.call_args.kwargs['expected_scalar'],3.0)
            self.assertFalse((Path(temp)/'out').exists())

    def test_exact_paired_mean_sign_and_source(self):
        req = request()
        original = deepcopy(req)
        result = W.summarize(req, observations())
        self.assertEqual(req, original)
        self.assertEqual(result['independent_paths'], 2)
        self.assertEqual(result['rows'][0]['observed_mean_RZ'], W.enc(110))
        self.assertEqual(result['rows'][0]['signed_relative_difference'], W.enc(W.Fraction(1, 11)))
        self.assertEqual(result['rows'][1]['signed_relative_difference'], W.enc(-W.Fraction(1, 11)))
        self.assertFalse(result['model_qualification_changed'])
        self.assertFalse(result['coefficient_refit'])
        self.assertEqual(len(result['rows'][0]['paths']), 2)

    def test_missing_axis_or_extra_independent_path_rejected(self):
        with self.assertRaises(ValueError):
            W.summarize(request(), observations()[:-1])
        with self.assertRaises(ValueError):
            W.summarize(request(), observations()+[dict(observations()[0], path_index=2)])

    def test_calibration_overlap_and_missing_prediction_rejected_before_acquisition(self):
        calls = []
        req = request()
        req['calibration_physical_ids'] = [req['selected_round']['physical_id']]
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ValueError):
                W.run_job(req, out, acquire=lambda *_: calls.append(True))
        self.assertEqual(calls, [])
        req = request()
        del req['predicted_RZ']['Y']
        with self.assertRaises(ValueError):
            W.validate_request(req)

    def test_saved_result_reuse_and_no_retry_on_failure(self):
        calls = []
        def acquire(req, out):
            calls.append(req['model_id'])
            return observations()
        with tempfile.TemporaryDirectory() as out:
            first = W.run_job(request(), out, acquire=acquire)
            second = W.run_job(request(), out, acquire=lambda *_: self.fail('retry'))
            self.assertEqual(first, second)
            self.assertTrue((Path(out)/'RESOURCE_FINAL.json').exists())
        self.assertEqual(len(calls), 1)
        with tempfile.TemporaryDirectory() as out:
            def failure(*_):
                raise ValueError('artificial unavailable leaf')
            first = W.run_job(request(), out, acquire=failure)
            self.assertEqual(first['status'], 'VALIDATION_INCOMPLETE')
            self.assertEqual(W.run_job(request(), out, acquire=lambda *_: self.fail('retry')), first)

    def test_zero_observation_relative_unknown(self):
        rows = [dict(r, compiled_RZ=0) for r in observations()]
        result = W.summarize(request(), rows)
        self.assertIsNone(result['rows'][0]['signed_relative_difference'])
        self.assertEqual(result['rows'][0]['relative_difference_reason'], 'ZERO_OBSERVED_PRICE')

    def test_deterministic_pf_allows_no_rte_events(self):
        req = request()
        req['acquisition']['kind'] = 'PF'
        req['independent_paths'] = 1
        req['selected_round']['R'] = 0
        result = W.summarize(req, observations()[:2])
        self.assertEqual(result['independent_paths'], 1)
        self.assertEqual(result['selected_round']['R'], 0)

    def test_source_gate_rejects_wrong_head_before_leaf(self):
        command = dict(cwd=str(ROOT), source=dict(commit='wrong', files={'scripts/run_pauli_split_cal.py':'bad'}),
            argv=['python', 'scripts/run_pauli_split_cal.py', '--worker', 'JOB.json'])
        with self.assertRaisesRegex(ValueError, 'HEAD differs'):
            W.check_command(command, 'PR_PAULI')


if __name__ == '__main__':
    unittest.main()
