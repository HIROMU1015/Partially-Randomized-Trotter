"""Artificial saved-fit provenance/evaluation tests. No real fit or H4 decode."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_stage1 as stage
from trottertracks import c1_stage1_execution as io
from trottertracks import c1_stage1_eval_only as run


def rewrite(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@contextmanager
def artificial_packet(tmp_path):
    """Create an artificial archive and mock historical Git bytes, never a fit."""
    root = tmp_path/'root'; root.mkdir()
    (root/'src/trottertracks').mkdir(parents=True)
    library = root/'src/trottertracks/c1_stage1.py'
    library.write_text('artificial repaired source')
    archive = root/'saved'; archive.mkdir()
    h = np.diag([-.3, -.1, .2, .4]); g = np.zeros((4,)*4); c = .375
    input_path = root/'ARTIFICIAL.npz'
    np.savez(input_path, one_mo=h, eri_chemist=g, nuclear_repulsion=np.array(c),
             mo_coeff=np.eye(4), df_one_body=np.kron(h, np.eye(2)).astype(complex),
             df_lambdas=np.ones(10), df_g_matrices=np.zeros((10, 8, 8), complex),
             energy=np.array('DO_NOT_DECODE'))
    Js, Us = [], []
    for i in range(8):
        U = np.eye(4)
        for step, (p, q) in enumerate([(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]):
            angle = (i+1)*(step+2)/11; co, si = np.cos(angle), np.sin(angle)
            rotation = np.eye(4); rotation[p, p] = rotation[q, q] = co
            rotation[p, q], rotation[q, p] = si, -si; U = U@rotation
        J = np.arange(1., 17).reshape(4, 4)/31; J = (J+J.T)*(i+1)/10
        J[0, 1] = J[1, 0] = -.1*(i+1); Js.append(J); Us.append(U)
    arrays = dict(kappa=h, diag_coulomb_mats=np.array(Js), orbital_rotations=np.array(Us),
                  constant=np.array(c), optimizer_x=np.zeros(128), optimizer_jac=np.zeros(128))
    np.savez(archive/'fit_export.npz', **arrays)
    original_source = dict(implementation_commit='ARTIFICIAL_OLD_IMPLEMENTATION',
                           files={'original.py': hashlib.sha256(b'old source').hexdigest()})
    io.save(archive/'SOURCE.json', original_source)
    input_hash, export_hash = io.sha(input_path), io.sha(archive/'fit_export.npz')
    metadata = dict(schema=stage.EXPORT_SCHEMA, ffsim_commit=stage.FFSIM_COMMIT,
        compatibility_version=stage.COMPAT_VERSION, settings=stage.FIT_SETTINGS,
        z_representation=False, one_body_role='ffsim_corrected_kappa', actual_rank=8,
        optimizer=dict(success=True, status=0, fun=0.), frame_ids=[f'F{i}' for i in range(8)],
        group_ids=[f'G{i}' for i in range(8)], input_sha256=input_hash,
        source_manifest_sha256=io.sha(archive/'SOURCE.json'), export_sha256=export_hash,
        arrays={k: io.array_record(a) for k, a in arrays.items()},
        input_members={k: io.array_record(a) for k, a in dict(one_mo=h, eri_chemist=g,
                        nuclear_repulsion=np.array(c), mo_coeff=np.eye(4)).items()},
        calls=dict(input_load=1, fit=1, fit_qualification=1, tensor_reconstruct=1,
                   tensor_maxabs=1, tensor_frobenius=1))
    io.save(archive/'fit_export.json', metadata)
    old_contract = dict(input={'sha256': input_hash}, scope_and_policy={'Nalpha': 2, 'Nbeta': 2},
                        allowance={'version': stage.ALLOWANCE_VERSION})
    io.save(archive/'contract.json', old_contract)
    io.save(archive/'approval.json', dict(approved=True, source_commit=run.FIT_SOURCE_COMMIT,
        contract_sha256=io.sha(archive/'contract.json'), source_manifest_sha256=io.sha(archive/'SOURCE.json'),
        input_sha256=input_hash))
    source_path = root/'evaluation_SOURCE.json'
    io.save(source_path, dict(implementation_commit='ARTIFICIAL_EVALUATION_IMPLEMENTATION',
                             files={'src/trottertracks/c1_stage1.py': io.sha(library)}))
    fit = dict(source_commit=run.FIT_SOURCE_COMMIT, evidence_commit=run.FIT_EVIDENCE_COMMIT,
               repaired_library_commit=run.REPAIRED_COMMIT, published_source_path='published_SOURCE.json',
               files={key: dict(path='saved/'+name, sha256=io.sha(archive/name)) for key, name in
                      [('source', 'SOURCE.json'), ('metadata', 'fit_export.json'), ('export', 'fit_export.npz'),
                       ('contract', 'contract.json'), ('approval', 'approval.json')]})
    contract_path = root/'contract.json'
    contract = dict(scope=run.SCOPE, execution_authorized=False, grant_issued=False,
        new_fit_calls=0, retry=0, resume=0, raw_frame_version=run.RAW_FRAME_VERSION,
        call_limits=dict(load=dict(input_load=1, export_load=1), evaluation=stage.expected_evaluation_calls(8, 36)),
        input=old_contract['input'], allowance=old_contract['allowance'],
        scientific_conditions=old_contract['scope_and_policy'], saved_fit=fit,
        implementation_source_commit='ARTIFICIAL_EVALUATION_IMPLEMENTATION',
        source_manifest_sha256=io.sha(source_path))
    io.save(contract_path, contract)
    historical = {}
    for key, record in fit['files'].items():
        historical[(run.FIT_EVIDENCE_COMMIT, record['path'])] = (root/record['path']).read_bytes()
    historical[(run.FIT_SOURCE_COMMIT, 'published_SOURCE.json')] = (archive/'SOURCE.json').read_bytes()
    historical[('ARTIFICIAL_OLD_IMPLEMENTATION', 'original.py')] = b'old source'
    historical[(run.REPAIRED_COMMIT, 'src/trottertracks/c1_stage1.py')] = library.read_bytes()
    historical[('ARTIFICIAL_EVALUATION_IMPLEMENTATION', 'src/trottertracks/c1_stage1.py')] = library.read_bytes()

    def git_bytes(root_arg, commit, name):
        return historical[(commit, name)]

    with patch.object(run, 'P_H4_SHA256', input_hash), patch.object(run, 'SAVED_EXPORT_SHA256', export_hash), \
         patch.object(run, '_git_bytes', side_effect=git_bytes):
        v = run.preflight(root, contract_path, source_path, input_path)
        v['historical_bytes'] = historical
        yield v


def approval(v):
    output = v['root']/run.OUTPUT_ROOT/'ARTIFICIAL-run01'
    path = v['root']/'approval.json'
    io.save(path, dict(scope=run.SCOPE, authority='USER_EXPLICIT_C1_STAGE1_EVAL_ONLY_SCIENCE',
        approved=True, source_commit='ARTIFICIAL_HEAD', contract_sha256=io.sha(v['contract_path']),
        source_manifest_sha256=io.sha(v['source_path']), input_sha256=io.sha(v['input']),
        saved_export_sha256=run.SAVED_EXPORT_SHA256, fit_source_commit=run.FIT_SOURCE_COMMIT,
        fit_source_manifest_sha256=io.sha(v['saved_paths']['source']), new_fit_calls=0,
        numerical_allowance_version=stage.ALLOWANCE_VERSION, numerical_allowance_accepted=True, output=str(output)))
    return path, output


@contextmanager
def no_fit_dispatch():
    import scipy.optimize
    actual_check_output = subprocess.check_output
    def git(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'ARTIFICIAL_HEAD\n'
        if command == ['git', 'diff', '--name-only', 'HEAD']:
            return b''
        # Dependencies may read platform identity via uname; mock only Git.
        if command[0] != 'git':
            return actual_check_output(command, **kwargs)
        raise AssertionError(command)
    with patch.object(stage, 'fit_once', side_effect=AssertionError('new fit forbidden')) as fit, \
         patch.object(io, 'worker_fit', side_effect=AssertionError('worker forbidden')) as worker, \
         patch.object(io, 'execute', side_effect=AssertionError('one-fit dispatcher forbidden')) as dispatch, \
         patch.object(scipy.optimize, 'minimize', side_effect=AssertionError('optimizer forbidden')) as optimizer, \
         patch.object(run.subprocess, 'check_output', side_effect=git), \
         patch.object(io, 'runtime_identity', return_value={'role': 'artificial eval runtime'}), \
         patch.dict(run.os.environ):
        yield
    fit.assert_not_called(); worker.assert_not_called(); dispatch.assert_not_called(); optimizer.assert_not_called()


def test_preflight_and_metadata_identity_never_decode_npz(tmp_path):
    with artificial_packet(tmp_path) as v:
        with patch.object(np, 'load', side_effect=AssertionError('no numeric decode')):
            checked = run.preflight(v['root'], v['contract_path'], v['source_path'], v['input'])
        assert checked['metadata']['actual_rank'] == 8
        assert io.sha(checked['saved_paths']['source']) != io.sha(checked['source_path'])


@pytest.mark.parametrize('defect', ['input', 'export', 'evaluation_source', 'old_source', 'history'])
def test_byte_or_historical_drift_refuses_before_decode(tmp_path, defect):
    with artificial_packet(tmp_path) as v:
        if defect == 'input': v['input'].write_bytes(b'changed')
        elif defect == 'export': v['saved_paths']['export'].write_bytes(b'changed')
        elif defect == 'evaluation_source': v['source_path'].write_bytes(b'{}')
        elif defect == 'old_source': v['saved_paths']['source'].write_bytes(b'{}')
        else: v['historical_bytes'][(run.FIT_EVIDENCE_COMMIT, 'saved/fit_export.npz')] = b'wrong history'
        with patch.object(np, 'load', side_effect=AssertionError('no decode')):
            with pytest.raises(ValueError):
                run.preflight(v['root'], v['contract_path'], v['source_path'], v['input'])


@pytest.mark.parametrize('defect', ['rank', 'schema', 'source_substitution', 'optimizer', 'shape', 'frame_order'])
def test_metadata_policy_rejects_wrong_schema_or_substituted_eval_provenance(tmp_path, defect):
    with artificial_packet(tmp_path) as v:
        metadata = dict(v['metadata'])
        metadata = json.loads(json.dumps(metadata))
        if defect == 'rank': metadata['actual_rank'] = 7
        elif defect == 'schema': metadata['schema'] = 'wrong'
        elif defect == 'source_substitution': metadata['source_manifest_sha256'] = io.sha(v['source_path'])
        elif defect == 'optimizer': metadata['optimizer']['success'] = False
        elif defect == 'shape': metadata['arrays']['orbital_rotations']['shape'] = [4, 8, 4]
        else: metadata['frame_ids'].reverse()
        with pytest.raises(ValueError):
            run._metadata_policy(metadata, run.P_H4_SHA256, io.sha(v['saved_paths']['source']))


def test_old_science_approval_is_not_evaluation_approval(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        old = v['saved_paths']['approval']; output = v['root']/run.OUTPUT_ROOT/'fresh'
        with patch.object(np, 'load', side_effect=AssertionError('no decode')):
            with pytest.raises(PermissionError, match='EVALUATION_ONLY_APPROVAL'):
                run.execute(v, old, output)
        assert not output.exists()


def test_artificial_evaluation_only_full256_sector36_and_exact_calls(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        ap, output = approval(v); old_meta_bytes = v['saved_paths']['metadata'].read_bytes()
        result = run.execute(v, ap, output)
        assert result['status'] == 'STAGE1_REPRESENTATION_PASS'
        assert result['new_fit_calls'] == 0 and result['scientific_counter_units'] == 141
        assert result['calls'] == stage.expected_evaluation_calls(8, 36)
        ids = [row['id'] for row in result['records']]
        assert ids.index('full_frame_7') < ids.index('sector_adapter') < ids.index('sector_DF_CDF')
        assert ids.index('full_frame_7') < ids.index('sector_DF_XDF')
        assert len(list((output/'operators').glob('*.npz'))) == 24
        assert not (output/'fit.claim').exists()
        assert not (output/'fit_calls.jsonl').exists()
        assert io.read(output/'STOP.json')['mandatory_stop']
        assert v['saved_paths']['metadata'].read_bytes() == old_meta_bytes
        assert io.read(output/'fit_SOURCE.json') != io.read(output/'evaluation_SOURCE.json')
        assert 'ffsim' not in sys.modules
        with pytest.raises(PermissionError, match='FRESH_EVALUATION_OUTPUT'):
            run.execute(v, ap, output)


def test_full_frame_failure_stops_before_sector_equality_or_DF_errors(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        ap, output = approval(v)
        original = stage.candidate_frame
        def wrong(frame):
            H, scale = original(frame)
            return H+.01*np.eye(len(H)), scale
        with patch.object(stage, 'candidate_frame', side_effect=wrong):
            result = run.execute(v, ap, output)
        assert result['status'] == 'SCIENTIFIC_STOP_EQUALITY'
        assert result['records'][-1]['id'] == 'full_frame_0'
        assert not any(row['id'].startswith('sector_') for row in result['records'])
        assert result['calls']['spectral_norm'] == 2
        # Old attempt used107 including6 fit calls; evaluation-only excludes6.
        assert result['scientific_counter_units'] == 101
        assert not (output/'operators/CDF_sector.npz').exists()
        assert io.read(output/'STOP.json')['retry'] == 0


def test_full_base_failure_stops_before_frame_or_sector_evaluation(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        ap, output = approval(v)
        original = stage.candidate_base
        with patch.object(stage, 'candidate_base', side_effect=lambda model: original(model)+.01*np.eye(256)):
            result = run.execute(v, ap, output)
        assert result['status'] == 'SCIENTIFIC_STOP_EQUALITY'
        assert result['records'][-1]['id'] == 'full_base'
        assert result['calls'].get('frame_build', 0) == 0
        assert result['calls'].get('sector_restrict', 0) == 0
        assert not any(row['id'].startswith('sector_') for row in result['records'])


def test_interruption_preserves_partial_evaluation_STOP_no_resume(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        ap, output = approval(v)
        with patch.object(stage, 'evaluate', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt): run.execute(v, ap, output)
        assert io.read(output/'STOP.json')['status'] == 'STOP_PARTIAL_EVIDENCE'
        with pytest.raises(PermissionError): run.execute(v, ap, output)


def test_dispatch_rechecks_bytes_changed_after_preflight(tmp_path):
    with artificial_packet(tmp_path) as v, no_fit_dispatch():
        ap, output = approval(v); v['saved_paths']['export'].write_bytes(b'corrupt')
        with patch.object(np, 'load', side_effect=AssertionError('no decode')):
            with pytest.raises(ValueError): run.execute(v, ap, output)
        assert not output.exists()


def test_cli_requires_new_approval_before_reading_files_and_has_no_fit_flags(tmp_path):
    script = ROOT/'scripts/run_research_c_c1_stage1_eval_only.py'
    common = [sys.executable, str(script), '--contract', 'missing', '--source', 'missing', '--input', 'missing']
    result = subprocess.run(common+['--execute'], capture_output=True, text=True)
    assert result.returncode == 2 and 'evaluation-only approval' in result.stderr
    result = subprocess.run(common+['--worker-fit'], capture_output=True, text=True)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr
