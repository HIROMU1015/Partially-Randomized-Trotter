"""Saved-fit evaluation only; preflight is bytes/JSON and never decodes arrays.

No fit worker or one-fit dispatcher is called. Numerical loads occur only in
execute(), after a fresh evaluation-only user approval has been verified.
"""
import hashlib
import math
import os
from pathlib import Path
import subprocess
import traceback

from . import c1_stage1 as stage
from . import c1_stage1_execution as io

SCOPE = "C1_STAGE1_P_H4_SAVED_FIT_EVAL_ONLY_V1"
FIT_SOURCE_COMMIT = "da2edb465d5ae6432439fbd1df8e57512ee59c11"
FIT_EVIDENCE_COMMIT = "f65bbba4d1c54658dfffee11a960370089789030"
REPAIRED_COMMIT = "2ab9977994afa31cdfffb8629ce57c6efa83a842"
SAVED_EXPORT_SHA256 = "c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08"
P_H4_SHA256 = io.P_H4_SHA256
RAW_FRAME_VERSION = "car_density_batched_matmul_v2"
OUTPUT_ROOT = "artifacts/research_c_c1_stage1_eval_only_science"


def _git_bytes(root, commit, name):
    return subprocess.check_output(['git', 'show', commit+':'+name], cwd=root)


def _checked_file(root, record, commit=None):
    path = (root/record['path']).resolve()
    if not path.is_relative_to(root) or io.sha(path) != record['sha256']:
        raise ValueError('PROVENANCE_FILE_HASH: '+record['path'])
    if commit is not None:
        frozen = _git_bytes(root, commit, record['path'])
        if hashlib.sha256(frozen).hexdigest() != record['sha256']:
            raise ValueError('HISTORICAL_COMMIT_FILE: '+record['path'])
    return path


def _metadata_policy(metadata, input_hash, source_hash):
    """JSON declarations only; no molecular tensor/optimizer arithmetic."""
    if (metadata['schema'] != stage.EXPORT_SCHEMA
            or metadata['ffsim_commit'] != stage.FFSIM_COMMIT
            or metadata['compatibility_version'] != stage.COMPAT_VERSION
            or metadata['settings'] != stage.FIT_SETTINGS
            or metadata['z_representation'] is not False
            or metadata['one_body_role'] != 'ffsim_corrected_kappa'
            or metadata['input_sha256'] != input_hash
            or metadata['source_manifest_sha256'] != source_hash
            or metadata['export_sha256'] != SAVED_EXPORT_SHA256
            or type(metadata['actual_rank']) is not int or metadata['actual_rank'] != 8
            or metadata['optimizer']['success'] is not True
            or metadata['optimizer']['status'] != 0
            or not math.isfinite(metadata['optimizer']['fun'])
            or metadata['frame_ids'] != [f'F{i}' for i in range(8)]
            or metadata['group_ids'] != [f'G{i}' for i in range(8)]):
        raise ValueError('SAVED_FIT_METADATA_POLICY')
    shapes = dict(kappa=[4, 4], diag_coulomb_mats=[8, 4, 4],
                  orbital_rotations=[8, 4, 4], constant=[],
                  optimizer_x=[128], optimizer_jac=[128])
    if set(metadata['arrays']) != set(shapes):
        raise ValueError('SAVED_EXPORT_METADATA_SCHEMA')
    for name, shape in shapes.items():
        record = metadata['arrays'][name]
        if record['shape'] != shape or record['dtype'] != '<f8' or len(record['sha256']) != 64:
            raise ValueError('SAVED_EXPORT_METADATA_SCHEMA: '+name)
    if metadata['calls'] != dict(input_load=1, fit=1, fit_qualification=1,
                                 tensor_reconstruct=1, tensor_maxabs=1, tensor_frobenius=1):
        raise ValueError('HISTORICAL_FIT_CALLS')


def preflight(root, contract_path, source_path, input_path):
    root = Path(root).resolve()
    contract_path, source_path, input_path = map(lambda p: Path(p).resolve(),
                                                 (contract_path, source_path, input_path))
    contract, source = io.read(contract_path), io.read(source_path)
    if (contract['scope'] != SCOPE or contract['execution_authorized'] is not False
            or contract['grant_issued'] is not False or contract['new_fit_calls'] != 0
            or contract['retry'] != 0 or contract['resume'] != 0
            or contract['raw_frame_version'] != RAW_FRAME_VERSION
            or stage.RAW_FRAME_VERSION != RAW_FRAME_VERSION
            or contract['call_limits'] != dict(load=dict(input_load=1, export_load=1),
                                              evaluation=stage.expected_evaluation_calls(8, 36))):
        raise ValueError('EVAL_ONLY_CONTRACT_POLICY')
    if (io.sha(source_path) != contract['source_manifest_sha256']
            or source['implementation_commit'] != contract['implementation_source_commit']):
        raise ValueError('EVALUATION_SOURCE_IDENTITY')
    for name, want in source['files'].items():
        _checked_file(root, dict(path=name, sha256=want), source['implementation_commit'])
    # The repaired numerical library itself is reused byte-for-byte.
    repaired = _git_bytes(root, REPAIRED_COMMIT, 'src/trottertracks/c1_stage1.py')
    if hashlib.sha256(repaired).hexdigest() != io.sha(root/'src/trottertracks/c1_stage1.py'):
        raise ValueError('REPAIRED_KERNEL_SOURCE')
    if io.sha(input_path) != P_H4_SHA256 or contract['input']['sha256'] != P_H4_SHA256:
        raise ValueError('INPUT_HASH')
    fit = contract['saved_fit']
    if (fit['source_commit'] != FIT_SOURCE_COMMIT
            or fit['evidence_commit'] != FIT_EVIDENCE_COMMIT
            or fit['repaired_library_commit'] != REPAIRED_COMMIT
            or fit['files']['export']['sha256'] != SAVED_EXPORT_SHA256):
        raise ValueError('SAVED_FIT_FIXED_IDENTITIES')
    paths = {key: _checked_file(root, record, FIT_EVIDENCE_COMMIT)
             for key, record in fit['files'].items()}
    metadata, old_source = io.read(paths['metadata']), io.read(paths['source'])
    # Old SOURCE stays bound to its old published code, never to evaluation HEAD.
    published = _git_bytes(root, FIT_SOURCE_COMMIT, fit['published_source_path'])
    if hashlib.sha256(published).hexdigest() != io.sha(paths['source']):
        raise ValueError('PUBLISHED_FIT_SOURCE')
    for name, want in old_source['files'].items():
        if hashlib.sha256(_git_bytes(root, old_source['implementation_commit'], name)).hexdigest() != want:
            raise ValueError('ORIGINAL_FIT_IMPLEMENTATION: '+name)
    _metadata_policy(metadata, P_H4_SHA256, io.sha(paths['source']))
    old_contract, old_approval = io.read(paths['contract']), io.read(paths['approval'])
    if (old_approval['source_commit'] != FIT_SOURCE_COMMIT
            or old_approval['contract_sha256'] != io.sha(paths['contract'])
            or old_approval['source_manifest_sha256'] != io.sha(paths['source'])
            or old_approval['input_sha256'] != P_H4_SHA256
            or old_approval['approved'] is not True
            or contract['scientific_conditions'] != old_contract['scope_and_policy']
            or contract['allowance'] != old_contract['allowance']
            or contract['input'] != old_contract['input']):
        raise ValueError('ORIGINAL_FIT_PROVENANCE_OR_SCIENTIFIC_POLICY')
    return dict(root=root, contract_path=contract_path, source_path=source_path,
                input=input_path, contract=contract, source=source,
                saved_paths=paths, metadata=metadata, fit_source=old_source)


def verify_approval(v, approval_path, output):
    output = Path(output).resolve()
    approval = io.read(approval_path)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=v['root'], text=True).strip()
    expected = dict(scope=SCOPE, authority='USER_EXPLICIT_C1_STAGE1_EVAL_ONLY_SCIENCE',
                    approved=True, source_commit=head,
                    contract_sha256=io.sha(v['contract_path']),
                    source_manifest_sha256=io.sha(v['source_path']),
                    input_sha256=io.sha(v['input']), saved_export_sha256=SAVED_EXPORT_SHA256,
                    fit_source_commit=FIT_SOURCE_COMMIT,
                    fit_source_manifest_sha256=io.sha(v['saved_paths']['source']),
                    new_fit_calls=0, numerical_allowance_version=stage.ALLOWANCE_VERSION,
                    numerical_allowance_accepted=True, output=str(output))
    if any(approval.get(k) != value for k, value in expected.items()):
        raise PermissionError('NEW_EXPLICIT_EVALUATION_ONLY_APPROVAL_REQUIRED')
    if not output.is_relative_to(v['root']/OUTPUT_ROOT) or output.exists():
        raise PermissionError('FRESH_EVALUATION_OUTPUT_REQUIRED_NO_RESUME')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=v['root']):
        raise ValueError('TRACKED_SOURCE_DRIFT')
    return approval


def execute(verified, approval_path, output):
    # Recheck byte identities at dispatch, including any drift after preflight.
    v = preflight(verified['root'], verified['contract_path'], verified['source_path'], verified['input'])
    output = Path(output).resolve()
    approval = verify_approval(v, approval_path, output)
    runtime = io.runtime_identity(v['root'], 'eval')
    output.mkdir(parents=True)
    try:
        io.save(output/'evaluation.claim', dict(scope=SCOPE, new_fit_calls=0, resume=False))
        for name, payload in [('approval.json', approval), ('evaluation_SOURCE.json', v['source']),
                              ('fit_SOURCE.json', v['fit_source']), ('contract.json', v['contract']),
                              ('saved_fit_metadata.json', v['metadata']), ('eval_runtime.json', runtime)]:
            io.save(output/name, payload)
        provenance = dict(evaluation_source_manifest_sha256=io.sha(v['source_path']),
                          fit_source_commit=FIT_SOURCE_COMMIT,
                          fit_source_manifest_sha256=io.sha(v['saved_paths']['source']),
                          fit_export_sha256=SAVED_EXPORT_SHA256, input_sha256=io.sha(v['input']),
                          raw_frame_version=RAW_FRAME_VERSION, new_fit_calls=0,
                          historical_fit_calls=v['metadata']['calls'])
        io.save(output/'provenance.json', provenance)
        cache = output/'.cache'
        for name, leaf in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'),
                           ('PYTHONPYCACHEPREFIX', 'pycache'), ('NUMBA_CACHE_DIR', 'numba'), ('TMPDIR', 'tmp')]:
            path = cache/leaf; path.mkdir(parents=True, exist_ok=True); os.environ[name] = str(path)
        load = io.journal(output, 'load', v['contract']['call_limits']['load'])
        load('input_load'); arrays = io.load_input(v['input'], evaluation=True)
        load('export_load')
        # Reuse the original loader with ORIGINAL provenance, without editing it
        # or substituting the repaired evaluation manifest into old metadata.
        exported, metadata = io.load_export(v['saved_paths']['export'].parent,
                        dict(input=v['input'], source_path=v['saved_paths']['source']))
        if metadata != v['metadata'] or metadata['export_sha256'] != SAVED_EXPORT_SHA256:
            raise ValueError('SAVED_METADATA_CHANGED_AFTER_PREFLIGHT')
        if {k: io.array_record(arrays[k]) for k in metadata['input_members']} != metadata['input_members']:
            raise ValueError('FIT_INPUT_MEMBER_IDENTITY')
        if io.array_record(exported['constant']) != io.array_record(arrays['nuclear_repulsion']):
            raise ValueError('CONSTANT_OWNERSHIP')
        calls = stage.Calls(io.journal(output, 'evaluation', v['contract']['call_limits']['evaluation']))
        (output/'operators').mkdir()

        def keep(name, matrix):
            import numpy as np
            path = output/'operators'/f'{name}.npz'
            with path.open('xb') as stream:
                np.savez(stream, matrix=matrix)
            row = dict(name=name, file=path.name, sha256=io.sha(path), member=io.array_record(matrix))
            import json
            with (output/'operator_inventory.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')

        result = stage.evaluate(arrays['one_mo'], arrays['eri_chemist'], float(arrays['nuclear_repulsion']),
                    exported['kappa'], exported['diag_coulomb_mats'], exported['orbital_rotations'],
                    arrays['df_one_body'], arrays['df_lambdas'], arrays['df_g_matrices'],
                    nalpha=2, nbeta=2, source_identity=io.sha(v['source_path']),
                    input_identity=io.sha(v['input']), calls=calls, operator_sink=keep)
        result.update(provenance=provenance, new_fit_calls=0,
                      load_calls=dict(input_load=1, export_load=1),
                      scientific_counter_units=2+sum(calls.counts.values()))
        io.save(output/'result.json', result)
        io.save(output/'STOP.json', dict(status=result['status'], mandatory_stop=True,
                    new_fit_calls=0, retry=0, resume=0, Stage2_authorized=False, Stage3_authorized=False))
        return result
    except BaseException as error:
        io.save(output/'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if not (output/'STOP.json').exists():
            io.save(output/'STOP.json', dict(status='STOP_PARTIAL_EVIDENCE', mandatory_stop=True,
                        new_fit_calls=0, retry=0, resume=0, Stage2_authorized=False, Stage3_authorized=False))
        raise
