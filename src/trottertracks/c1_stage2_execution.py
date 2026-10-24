"""Stage2 byte preflight and separately approved one-shot dispatcher.

Proposed science is never adopted by technical preflight or artificial tests.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from . import c1_stage1_execution as io
from . import c1_stage2_native as native
from . import c1_stage2_validation as validation

SCOPE = 'C1_STAGE2_SAVED_R8_NATIVE_LIMITED_V1'
BASE = 'b5fa60f9b682a93fda417452b5d85ceb1527e550'
FAMILY = 'artifacts/research_c_c1_stage2_science'


def proposal_hash(proposal):
    return hashlib.sha256(json.dumps(proposal, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def preflight(root, contract_path, input_path):
    """Bytes/JSON only, including historical references; no numpy/Qiskit import."""
    root = Path(root).resolve()
    contract_path, input_path = Path(contract_path).resolve(), Path(input_path).resolve()
    c = io.read(contract_path)
    if (c['scope'] != SCOPE or c['execution_authorized'] is not False
            or c['proposal_adopted'] is not False or c['new_fit_calls'] != 0
            or c['retry'] != 0 or c['resume'] != 0
            or c['call_limits'] != validation.call_limits()
            or c['maximum_units'] != sum(validation.call_limits().values())
            or c['proposal_sha256'] != proposal_hash(c['recommended'])):
        raise ValueError('UNAPPROVED_STAGE2_CONTRACT_POLICY')
    source_path = root/c['source_manifest']
    if io.sha(source_path) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    source = io.read(source_path)
    for name, want in source['files'].items():
        if io.sha(root/name) != want or hashlib.sha256(subprocess.check_output(
                ['git', 'show', source['implementation_commit']+':'+name], cwd=root)).hexdigest() != want:
            raise ValueError('SOURCE_FILE: '+name)
    if io.sha(input_path) != c['input_sha256']:
        raise ValueError('ORIGINAL_INPUT')
    for name, want in c['saved_files'].items():
        if io.sha(root/name) != want or hashlib.sha256(subprocess.check_output(
                ['git', 'show', BASE+':'+name], cwd=root)).hexdigest() != want:
            raise ValueError('SAVED_STAGE1_FILE: '+name)
    old = io.read(root/c['stage1_result'])
    if old['status'] != 'STAGE1_REPRESENTATION_PASS' or not all(
            row['passed'] for row in old['records'] if row['gate']):
        raise ValueError('SAVED_STAGE1_QUALIFICATION')
    return dict(root=root, contract_path=contract_path, input=input_path,
                contract=c, source=source, source_path=source_path)


def verify_approval(v, approval_path, output):
    c, root = v['contract'], v['root']
    output = Path(output).resolve()
    approval = io.read(approval_path)
    expected = dict(scope=SCOPE, authority='USER_EXPLICIT_C1_STAGE2_SCIENCE', approved=True,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        contract_sha256=io.sha(v['contract_path']), source_manifest_sha256=io.sha(v['source_path']),
        input_sha256=io.sha(v['input']), saved_export_sha256=c['saved_export_sha256'],
        adopted_proposal_sha256=c['proposal_sha256'], scientific_conditions_adopted=True,
        output=str(output), maximum_units=c['maximum_units'], new_fit_calls=0, retry=0, resume=0)
    if any(approval.get(k) != want for k, want in expected.items()):
        raise PermissionError('SEPARATE_SCIENCE_APPROVAL_AND_PROPOSAL_ADOPTION_REQUIRED')
    if output.exists() or not output.is_relative_to(root/FAMILY):
        raise PermissionError('FRESH_STAGE2_OUTPUT_REQUIRED')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root):
        raise ValueError('TRACKED_SOURCE_DRIFT')
    return approval


def runtime(v):
    if sys.version_info[:3] != (3, 11, 1):
        raise ValueError('LOCKED_PYTHON_3_11_1_REQUIRED')
    packages = {}
    for line in (v['root']/v['contract']['environment_pins']).read_text().splitlines():
        name, want = line.split('==')
        actual = importlib.metadata.version(name)
        if actual != want:
            raise ValueError('ENVIRONMENT_PIN: '+name)
        packages[name] = actual
    return dict(python=sys.version, executable=sys.executable, packages=packages)


def execute(verified, approval_path, output):
    v = preflight(verified['root'], verified['contract_path'], verified['input'])
    approval = verify_approval(v, approval_path, output)
    environment = runtime(v)
    root, c, output = v['root'], v['contract'], Path(output).resolve()
    output.mkdir(parents=True)
    counts, events, backend = {}, [], None
    try:
        io.save(output/'evaluation.claim', dict(scope=SCOPE, new_fit=0, retry=0, resume=0))
        for name, data in [('approval.json', approval), ('contract.json', c),
                           ('SOURCE.json', v['source']), ('runtime.json', environment)]:
            io.save(output/name, data)
        # Cache location is task-owned and outside published evidence.
        cache = root/'.cache/stage2'
        for key, leaf in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'),
                          ('NUMBA_CACHE_DIR', 'numba'), ('TMPDIR', 'tmp')]:
            path = cache/leaf; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
        journal = io.journal(output, 'science', c['call_limits'])

        def mark(kind, label):
            journal(kind, label)
            counts[kind] = counts.get(kind, 0)+1

        def call(kind, label, function):
            mark(kind, label)
            value = function()
            events.append(dict(kind=kind, label=label, status='COMPLETED'))
            return value

        import numpy as np
        from . import c1_stage1 as stage1
        from . import c1_general_k_adapter as adapter

        def load_packet():
            with np.load(v['input'], allow_pickle=False) as archive:
                values = {k: archive[k] for k in ['df_one_body', 'df_lambdas', 'df_g_matrices', 'nuclear_repulsion']}
            stage1.finite_arrays(*values.values())
            if values['df_one_body'].shape != (8, 8) or values['df_lambdas'].shape != (10,) or values['df_g_matrices'].shape != (10, 8, 8):
                raise ValueError('SAVED_XDF_SCHEMA')
            return values

        packet = call('input_load', 'saved_XDF_whitelist', load_packet)
        export_parent = (root/c['saved_export']).parent
        exported, metadata = call('export_load', 'old_fit_provenance', lambda: io.load_export(
            export_parent, dict(input=v['input'], source_path=export_parent/'SOURCE.json')))
        if metadata['actual_rank'] != 8 or metadata['export_sha256'] != c['saved_export_sha256']:
            raise ValueError('SAVED_R8_REQUIRED')
        scalar = float(exported['constant'])
        if scalar != float(packet['nuclear_repulsion']):
            raise ValueError('SCALAR_OWNERSHIP')
        model = adapter.from_ffsim_number_arrays(one_body_tensor=exported['kappa'],
            diag_coulomb_mats=exported['diag_coulomb_mats'], orbital_rotations=exported['orbital_rotations'],
            constant=scalar, z_representation=False, frame_ids=metadata['frame_ids'],
            group_ids=metadata['group_ids'], source_identity=c['saved_export_sha256'],
            input_identity=c['input_sha256'], one_body_role='ffsim_corrected_kappa', enable_general_k=True)
        models = dict(CDF=call('model_prepare', 'CDF', lambda: native.prepare_cdf(model)),
                      XDF=call('model_prepare', 'XDF', lambda: native.prepare_xdf(packet['df_one_body'],
                        packet['df_lambdas'], packet['df_g_matrices'], scalar,
                        source_identity=c['input_sha256']+':saved_XDF', input_identity=c['input_sha256'])))
        if len(models['CDF'].blocks) != 9 or len(models['XDF'].blocks) != 11:
            raise ValueError('FIXED_BLOCK_COUNTS')

        def load_operator(name):
            with np.load(root/name, allow_pickle=False) as archive:
                value = archive['matrix']
            if value.shape != (256, 256) or value.dtype != np.dtype('complex128') or not np.isfinite(value).all():
                raise ValueError('SAVED_RAW_CAR_OPERATOR_SCHEMA')
            return value

        generators = dict(CDF=[call('saved_operator_load', name, lambda name=name: load_operator(name))
                               for name in c['CDF_generator_files']])
        car = call('car_context', '8 modes', lambda: stage1.car_bilinears(8))
        def xdf_base():
            return scalar*np.eye(256)+native.xdf_car_generator(packet['df_one_body'], None, car)
        generators['XDF'] = [call('xdf_generator', 'one_body+c', xdf_base)]
        for i in range(10):
            generators['XDF'].append(call('xdf_generator', f'F{i}', lambda i=i:
                native.xdf_car_generator(packet['df_g_matrices'][i], packet['df_lambdas'][i], car)))
        (output/'circuits').mkdir()
        def keep(name, qc):
            from qiskit import qpy
            with (output/'circuits'/f'{name}.qpy').open('xb') as stream:
                qpy.dump(qc, stream)
        backend = validation.QiskitBackend(models, generators, c['recommended'], mark=mark, keep=keep)
        result = validation.validate(backend, c['recommended'], call)
        result.update(scientific_units=sum(counts.values()), call_counts=counts, gates=backend.gates)
        io.save(output/'result.json', result)
        io.save(output/'STOP.json', dict(status=result['status'], mandatory_stop=True, retry=0, resume=0,
                                        new_fit=0, Stage3_authorized=False))
        return result
    except BaseException as error:
        io.save(output/'failure.json', dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), scientific_units=sum(counts.values()), call_counts=counts,
            completed=events, gates=backend.gates if backend is not None else []))
        io.save(output/'STOP.json', dict(status='STOP_PARTIAL_STAGE2', mandatory_stop=True,
                                        retry=0, resume=0, new_fit=0, Stage3_authorized=False))
        raise
