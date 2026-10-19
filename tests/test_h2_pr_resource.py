"""Artificial arrays and disposable metadata only; never build a molecule."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import h2_pr_resource_contract as c
import h2_tol_only_input as science


class GateTests(unittest.TestCase):
    def setUp(self):
        directory = c.PRIVATE / 'artificial_tests'; directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temp.name)
        self.private = self.root / 'private'; self.private.mkdir()
        self.patch_root = patch.object(c, 'ROOT', self.root); self.patch_root.start()
        self.patch_private = patch.object(c, 'PRIVATE', self.private); self.patch_private.start()
        self.git = patch.object(c.subprocess, 'check_output', return_value=b'artificial source'); self.git.start()
        sources = {}
        for name in c.SOURCE_FILES:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'artificial source')
            sources[name] = c.sha(p)
        provider = self.private / 'provider'; provider.mkdir()
        (provider / 'toy.py').write_text('# artificial source\n')
        python = self.root / 'toy-python'; python.write_bytes(b'artificial binary')
        runtime = self.private / 'runtime.json'; c.save(runtime, {str(python): c.sha(python)})
        pending = dict(scope=c.SCOPE, stage='F', status='SEALED_PENDING_APPROVAL', SOURCE='a'*40,
            plan=c.plan(), source_hashes=sources, provider_root=str(provider), provider_SOURCE=c.PROVIDER_SOURCE,
            provider_hashes={'toy.py': c.sha(provider / 'toy.py')}, runtime_manifest=str(runtime),
            runtime_manifest_sha256=c.sha(runtime), python=str(python), python_sha256=c.sha(python),
            attempt=str(self.private / 'attempts/F_run01'), scientific_inputs=[], call_budget=c.CALLS)
        pending_path = self.private / 'pending.json'; c.save(pending_path, pending)
        self.pending = pending
        self.spec = dict(pending, status='SEALED_FOR_EXECUTION', pending_spec=str(pending_path),
                         pending_spec_sha256=c.sha(pending_path))
        receipt = self.private / 'receipt.json'
        c.save(receipt, dict(role='user', stage='F', SOURCE=self.spec['SOURCE'],
            plan_digest=c.digest(c.plan()), verbatim_user_message='ARTIFICIAL_FIXTURE_ONLY'))
        self.grant = dict(approved_by_user=True, runtime_authorization=True, stage='F',
            one_shot=True, automatic_retry=False, next_stage_authorized=False, SOURCE=self.spec['SOURCE'],
            spec_digest=c.digest(self.spec), approval_receipt=str(receipt), approval_receipt_sha256=c.sha(receipt))

    def tearDown(self):
        self.git.stop(); self.patch_private.stop(); self.patch_root.stop(); self.temp.cleanup()

    def test_disposable_valid_metadata_passes(self):
        self.assertEqual(c.verify(self.spec, self.grant)['status'], 'PASS')

    def test_pending_grant_denied_before_scientific_import(self):
        with self.assertRaisesRegex(ValueError, 'SEPARATE_EXPLICIT_F_APPROVAL_REQUIRED'):
            science.generate({}, {}, self.private / 'no-output')
        self.assertFalse((self.private / 'no-output').exists())

    def test_foreign_stage_denied(self):
        for stage in ['FP', 'D', 'A', 'S_STATS', 'P']:
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'H2_F_ONLY'):
                c.verify(self.spec, dict(self.grant, stage=stage))

    def test_retry_and_next_stage_denied(self):
        for key in ['automatic_retry', 'next_stage_authorized']:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'ONE_SHOT_STOP_REQUIRED'):
                c.verify(self.spec, dict(self.grant, **{key: True}))

    def test_changed_recipe_denied_even_with_rebound_grant_digest(self):
        spec = deepcopy(self.spec); spec['plan']['df_tol'] = 1e-6
        with self.assertRaisesRegex(ValueError, 'F_PLAN_CHANGED'):
            c.verify(spec, dict(self.grant, spec_digest=c.digest(spec)))

    def test_source_bytes_changed(self):
        (self.root / c.SOURCE_FILES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SOURCE_BYTES_CHANGED'):
            c.verify(self.spec, self.grant)

    def test_provider_bytes_changed(self):
        (Path(self.spec['provider_root']) / 'toy.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'FROZEN_PROVIDER_CHANGED'):
            c.verify(self.spec, self.grant)

    def test_runtime_bytes_changed(self):
        Path(self.spec['python']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'RUNTIME_CHANGED'):
            c.verify(self.spec, self.grant)

    def test_approval_receipt_changed(self):
        Path(self.grant['approval_receipt']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'APPROVAL_RECEIPT_BYTES'):
            c.verify(self.spec, self.grant)

    def test_foreign_and_symlink_paths_denied(self):
        with self.assertRaisesRegex(ValueError, 'H2_PRIVATE_PATH_REQUIRED'):
            c.owned(self.root / 'outside')
        link = self.private / 'link'; link.symlink_to(self.private, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'H2_PRIVATE_SYMLINK_REJECTED'):
            c.owned(link / 'file')

    def test_nonrefundable_reservation_cannot_be_repeated_or_exceeded(self):
        c.reserve(self.private, 'scf_kernel')
        with self.assertRaises(FileExistsError): c.reserve(self.private, 'scf_kernel')
        with self.assertRaisesRegex(ValueError, 'F_CALL_BUDGET'): c.reserve(self.private, 'solver_eigh', 2)
        with self.assertRaisesRegex(ValueError, 'F_CALL_BUDGET'): c.reserve(self.private, 'compile')
        self.assertEqual(c.actual_calls(self.private)['scf_kernel'], 1)

    def test_real_entrypoints_refuse_with_no_scientific_packages_available(self):
        spec_path, grant_path = self.private / 'spec.json', self.private / 'invalid.json'
        c.save(spec_path, self.pending); c.save(grant_path, dict(approved_by_user=False))
        for worker in [False, True]:
            command = ['/usr/bin/python3', '-S', '-B', str(SCRIPTS / 'run_h2_pr_f.py'),
                       '--spec', str(spec_path), '--grant', str(grant_path)]
            if worker: command.append('--worker')
            result = subprocess.run(command, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SEPARATE_EXPLICIT_F_APPROVAL_REQUIRED', result.stderr)
        self.assertFalse(Path(self.spec['attempt']).exists())


class ArtificialArrayTests(unittest.TestCase):
    def test_F_adapter_with_only_artificial_molecular_interfaces(self):
        import numpy as np
        diagonal = np.diag([0., 1., 2., 3.]).astype(complex)
        one = np.kron(np.diag([1., 2.]), np.eye(2)).astype(complex)
        mf = SimpleNamespace(converged=True, e_tot=-1., mo_coeff=np.eye(2), kernel=lambda: None)
        mol = SimpleNamespace(unit='angstrom', symmetry=False, nao_nr=lambda: 2,
                              nelectron=2, energy_nuc=lambda: 0.)
        sector = SimpleNamespace(dimension=4, basis_indices=np.array([3,6,9,12]))
        exports = {
            'scipy.linalg': dict(eigh=np.linalg.eigh),
            'openfermion': dict(MolecularData=lambda *a, **k: object(),
                low_rank_two_body_decomposition=lambda *a, **k: (np.array([.2]), np.array([one/4]), np.zeros((4,4)), 0.)),
            'openfermion.chem.molecular_data': dict(spinorb_from_spatial=lambda *a: (one, np.zeros((4,4,4,4)))),
            'openfermionpyscf._run_pyscf': dict(prepare_pyscf_molecule=lambda x: mol,
                compute_scf=lambda x: mf, compute_integrals=lambda *a: (np.eye(2), np.zeros((2,2,2,2)))),
            'trotterlib.df_hamiltonian': dict(DFHamiltonian=lambda *a: SimpleNamespace(constant=a[0],
                one_body=a[1], lambdas=a[2], g_matrices=a[3]),
                PhysicalSector=SimpleNamespace(spin_sector=lambda **k: sector),
                df_linear_operator=lambda *a, **k: (diagonal, None)),
            'trotterlib.df_partial_s2': dict(df_hamiltonian_hash=lambda x: 'toy-hamiltonian'),
            'trotterlib.pr2_s0_s1_validation': dict(_canonicalize_state_phase=lambda full, state: (full, state),
                _state_hash=lambda *a: 'toy-state'),
            'trottertracks.resource_applicability.ax2b_independent_reference': dict(occupation_df_matrix=lambda *a: diagonal),
        }
        modules = {}
        for name, attributes in exports.items():
            modules[name] = ModuleType(name); modules[name].__dict__.update(attributes)
        directory = c.PRIVATE / 'artificial_tests'; directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            output = Path(temp); (output / 'scratch').mkdir()
            with patch.dict(sys.modules, modules), patch.object(c, 'verify', return_value={'status':'ARTIFICIAL_ONLY'}):
                result = science.generate({'SOURCE':'artificial-only'}, {}, output)
            self.assertEqual(result['actual_calls'], c.CALLS)
            self.assertEqual(result['arrays']['state_vector']['shape'], [16])
            self.assertEqual(result['arrays']['sector_state_vector']['shape'], [4])
            self.assertEqual(result['rank'], 1)
            self.assertFalse(result['ground_state_certified'])

    def test_h2_projection_accepts_signed_synthetic_spin_blocks(self):
        import numpy as np
        one = np.kron(np.array([[1., .2], [.2, 2.]]), np.eye(2)).astype(complex)
        blocks = np.array([one/4, one/8]); weights = np.array([-.2, .3])
        post, projected, receipt = science.projected(one, weights, blocks, np.zeros((4,4)))
        np.testing.assert_array_equal(post, one); np.testing.assert_array_equal(projected, blocks)
        self.assertEqual(receipt['full_fock_N'], 4)
        self.assertFalse(receipt['representation_error_certified'])

    def test_h4_shape_and_excessive_projection_rejected(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError, 'DF_LAYOUT'):
            science.projected(np.eye(8), np.ones(1), np.ones((1,8,8)), np.zeros((8,8)))
        one = np.kron(np.array([[1., 1.], [0., 2.]]), np.eye(2))
        with self.assertRaisesRegex(ValueError, 'WEIGHTED_PROJECTION_BUDGET'):
            science.projected(one, np.ones(1), np.zeros((1,4,4)), np.zeros((4,4)))

    def test_snapshot_roundtrip_and_object_rejection_use_only_toy_arrays(self):
        import numpy as np
        directory = c.PRIVATE / 'artificial_tests'; directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            receipt = science.snapshot(Path(temp) / 'toy.npz', {'toy': np.eye(4)})
            self.assertEqual(receipt['arrays']['toy']['shape'], [4,4])
            with self.assertRaisesRegex(ValueError, 'NO_OBJECT_ARRAY'):
                science.snapshot(Path(temp) / 'bad.npz', {'toy': np.array([{}], dtype=object)})


if __name__ == '__main__':
    unittest.main()
