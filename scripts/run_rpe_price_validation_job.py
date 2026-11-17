"""One descriptive price holdout through existing source-bound acquisition leaves.

This is a compile-stage adapter, not a compiler or calibration runner. The
ordinary planner supplies already prepared leaf commands and immutable model,
prediction and physical-circuit metadata. Scientific permission and distributed
claim ownership remain with the existing controller/preparation queue.

No coefficient fitting, model qualification, global residual correction or Q3.
The optional ``acquire`` argument is only the artificial-test seam.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
# A DF path is already a separate worker process. Load its fixed provider
# package before the transport helper, as the existing signal worker does.
_df_view = os.environ.get('RPE_PRICE_VALIDATION_DF_VIEW')
if _df_view:
    _df_view = Path(_df_view).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=_df_view,text=True).strip() != '326feadd093920bbea29b498108aa4eae6f67a75':
        raise ValueError('Fixed DF path provider SOURCE differs')
    sys.path[:0] = [str(_df_view/'src'),str(_df_view/'scripts')]
    import trottertracks
    trottertracks.__path__.append(str(ROOT/'src/trottertracks'))
from trottertracks.rpe_preparation_paths import resolve, runtime_request


SCHEMA = 'RPE_PRICE_VALIDATION_JOB_V1'
SCRIPT = 'scripts/run_rpe_price_validation_job.py'
PAULI_SOURCE = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
DF_SOURCE = '326feadd093920bbea29b498108aa4eae6f67a75'
COMMON_OPERATORS = dict(commit='4220f3006d0d8d413e195ed948e250578513b557',
    path='docs/research/df_head_pauli_tail_p0_results_20261114/run01/COMMON_OPERATORS.npz')
KINDS = {
    'PR_PAULI': {'scripts/run_pauli_split_cal.py', 'scripts/run_pauli_ld2_m9.py',
                 'scripts/run_pauli_b1_prices.py'},
    'PR_DF': {'scripts/run_pr_df_candidate_prices.py'},
    'PF': {'scripts/run_pf_preparation_stage.py'},
}


def exact(value):
    if isinstance(value, bool):
        raise ValueError('Exact numeric value required')
    if isinstance(value, dict):
        return Fraction(int(value['numerator']), int(value['denominator']))
    return Fraction(value)


def enc(value):
    value = exact(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select(value, keys):
    for key in keys:
        value = value[key]
    return value


def git_read(root, locator):
    raw = subprocess.check_output(['git', 'show', locator['commit']+':'+locator['path']], cwd=root)
    return select(json.loads(raw), locator.get('selector', []))


def _packet(value, configuration_id):
    """Select the original saved packet; never reconstruct/relabel mean rows."""
    if isinstance(value, dict):
        if value.get('configuration', {}).get('id') == configuration_id and value.get('rounds'):
            return value
        for key in ('packets', 'packet', 'signal_packet'):
            if key in value:
                found = _packet(value[key], configuration_id)
                if found is not None:
                    return found
    if isinstance(value, list):
        matches = [p for p in (_packet(item, configuration_id) for item in value) if p is not None]
        if len(matches) > 1:
            raise ValueError('Ambiguous original signal packet')
        return matches[0] if matches else None
    return None


def _seed_values(value):
    out = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('step_seeds', 'seeds'):
                def flatten(v):
                    if type(v) is int:
                        out.add(v)
                    elif isinstance(v, list):
                        for a in v:
                            flatten(a)
                flatten(item)
            else:
                out.update(_seed_values(item))
    elif isinstance(value, list):
        for item in value:
            out.update(_seed_values(item))
    return out


def build_pauli_acquisition(request, *, root, configuration, signal_source,
                            legacy_view, interpreter, execution_source, output_dir,
                            calibration_seed_sources=()):
    """Metadata-only factory for the ordinary PR-Pauli normal-path leaf.

    Read the original packet and dictionary. Reuse its physical request and the
    existing seed recipe. No signal generation, probability evaluation or build.
    The caller freezes the returned job before queue submission. If fixed packet,
    legacy provider or calibration seed provenance is missing, keep it pending.
    """
    root, legacy_view = Path(root).resolve(), Path(legacy_view).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=legacy_view, text=True).strip() != PAULI_SOURCE:
        raise ValueError('Fixed Pauli normal-path provider view missing')
    module_path = 'src/trottertracks/pauli_tail_price_connection.py'
    expected = subprocess.check_output(['git', 'show', PAULI_SOURCE+':'+module_path], cwd=root)
    if (legacy_view/module_path).read_bytes() != expected:
        raise ValueError('Pauli request helper SOURCE differs')
    spec = importlib.util.spec_from_file_location('_validation_pauli_connection', legacy_view/module_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    if request['implementation'].get('lowering_id') != P.LOWERING_ID:
        raise ValueError('Fixed Pauli lowering contract differs')
    packet = _packet(git_read(root, signal_source), configuration['id'])
    if packet is None or packet['configuration'] != configuration:
        raise ValueError('Original canonical Pauli signal packet is unavailable or differs')
    m = request['selected_round']['m']
    packet_round = next(row for row in packet['rounds'] if row['m'] == m)
    config_round = next(row for row in configuration['rounds'] if row['m'] == m)
    if (packet_round['identity']['candidate'] != config_round['candidate'] or
            packet_round['law_binding'] != config_round['law_binding']):
        raise ValueError('Original packet candidate or law differs from configuration')
    dictionary = P.load_dictionary(root, packet['dictionary_identity'])
    physical = P.round_request(packet, m, dictionary)
    selected = request['selected_round']
    if (physical['candidate_id'], physical['q'], physical['R']) != (selected['physical_id'], selected['q'], selected['R']):
        raise ValueError('Representative circuit does not match original signal packet')
    step = exact(configuration['rounds'][m]['actual_delta_pf'])
    if Fraction(float(physical['T'])) != physical['T'] or Fraction(float(step)) != step:
        raise ValueError('Actual step or T is not represented by the fixed binary64 compiler')
    profile = configuration['spec']['profile']
    if (profile.get('control') != 'symmetric_directional' or profile.get('P') != 0 or
            profile.get('wrapper') != 'full_Hadamard_with_measurement' or
            profile.get('qiskit_version') != '1.3.0' or
            profile.get('basis_gates') != ['rz', 'sx', 'x', 'cx'] or
            profile.get('optimization_level') != 1 or profile.get('seed_transpiler') != 17):
        raise ValueError('No matching measured full-wrapper compiler profile')
    old = set()
    for source in calibration_seed_sources:
        old.update(_seed_values(git_read(root, source)))
    # Frozen model records often embed the original calibration step seeds.
    model_document = git_read(root, request['frozen_model']['source'])
    old.update(_seed_values(model_document))
    if not old:
        raise ValueError('Calibration seed provenance required for independent normal paths')
    cohort = 'RPE_SIMPLE_VALIDATION_'+request['model_id']+'_V1'
    seeds = [[P.seed(physical['candidate_id'], cohort, i, outer, 0)
              for outer in range(physical['q'])] for i in (0, 1)]
    flat = [seed for row in seeds for seed in row]
    if len(set(flat)) != len(flat) or set(flat) & old:
        raise ValueError('Independent path or calibration seed collision')
    out = Path(output_dir).resolve()
    dictionary_source = packet['dictionary_identity']['source']
    operators = dict(commit=dictionary_source['commit'],
        path=dictionary_source['path'].rsplit('/', 1)[0]+'/LD%d_OPERATORS.npz' % configuration['spec']['L_D'])
    for source in (COMMON_OPERATORS, operators):
        subprocess.check_output(['git', 'cat-file', '-e', source['commit']+':'+source['path']], cwd=root)
    return dict(kind='PR_PAULI', backend='BUILTIN_PAULI_NORMAL_PATH',
        legacy_view=str(legacy_view), interpreter=str(interpreter), execution_source=deepcopy(execution_source),
        configuration=deepcopy(configuration), signal_source=deepcopy(signal_source),
        packet=deepcopy(packet), physical_request=dict(physical, T=float(physical['T'])),
        common_operators=deepcopy(COMMON_OPERATORS), operators=operators,
        cohort=cohort, seeds=seeds, old_seed_sources=deepcopy(list(calibration_seed_sources)),
        seed_check=dict(unique_new=True, overlap_with_calibration=0, old_seed_count=len(old)),
        output_dir=str(out))


def _df_provider(root, legacy_view=None):
    root = Path(root).resolve()
    path = root/'scripts/run_pr_df_candidate_prices.py'
    spec = importlib.util.spec_from_file_location('_validation_df_candidate_prices', path)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    if legacy_view is not None:
        C.FIXED_VIEW = Path(legacy_view).resolve()
    return C


_DF_METADATA_BOOTSTRAP = r'''
import importlib.util,json,subprocess,sys
from pathlib import Path
payload=json.load(sys.stdin);root=Path(payload['root']);legacy=Path(payload['legacy_view'])
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=legacy,text=True).strip()!=payload.pop('legacy_source'):
 raise ValueError('Fixed DF metadata provider SOURCE differs')
sys.path[:0]=[str(legacy/'src'),str(legacy/'scripts')]
import trottertracks
trottertracks.__path__.append(str(root/'src/trottertracks'))
spec=importlib.util.spec_from_file_location('_isolated_df_validation',root/'scripts/run_rpe_price_validation_job.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
request=payload.pop('request')
print(json.dumps(worker._build_df_acquisition_local(request,**payload),allow_nan=False))
'''


def build_df_acquisition(request, *, root, configuration, configuration_source,
                         interpreter, execution_source, output_dir, legacy_view=None,
                         calibration_seed_sources=(), runtime=None):
    """Build original metadata in a fresh fixed-provider import context.

    The controller may already own a newer package with the same module names.
    No native preparation, sampler, compiler or scientific subprocess is run.
    """
    root=Path(root).resolve()
    payload=dict(request=request,root=str(root),configuration=configuration,
        configuration_source=configuration_source,interpreter=str(interpreter),
        execution_source=execution_source,output_dir=str(Path(output_dir).resolve()),
        legacy_view=str(Path(legacy_view or root.parent/'d2-pr-df-ld1-m3-acquisition-20261115').resolve()),
        legacy_source=DF_SOURCE,calibration_seed_sources=list(calibration_seed_sources),runtime=runtime)
    result=subprocess.run([str(interpreter),'-B','-c',_DF_METADATA_BOOTSTRAP],
        input=json.dumps(payload),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    if result.returncode:
        raise ValueError('DF metadata isolation failed: '+result.stderr[-3000:])
    return json.loads(result.stdout)


def _build_df_acquisition_local(request, *, root, configuration, configuration_source,
                         interpreter, execution_source, output_dir, legacy_view=None,
                         calibration_seed_sources=(), runtime=None):
    """Prepare exactly one DF HOLDOUT cell; never call calibration ``prepare``.

    Reuse D.from_b, the existing trial_seed function, fixed saved-input mapping
    and scalar SOURCE. Context restoration and compilation happen only in the
    eventual compile worker. Missing provider/local input data remains pending.
    """
    root = Path(root).resolve()
    s = configuration['spec']
    if s['method'] != 'PR' or s['split'].get('tail_dictionary'):
        raise ValueError('Original PR-DF configuration required')
    C = _df_provider(root, legacy_view)
    R = C.runtime()
    m = request['selected_round']['m']
    row = next(row for row in configuration['rounds'] if row['m'] == m)
    if (row['candidate']['id'], row['q_PF'], row['R_outer']) != tuple(
            request['selected_round'][key] for key in ('physical_id', 'q', 'R')):
        raise ValueError('Original DF physical round differs')
    if s['formula'] != '2nd' or s['finite_K'] != 2 or row['structure']['signed_weights'] != [1.]:
        raise ValueError('Existing DF compile worker supports S2/K2/one signed piece')
    tail = s['split']['lambda_source']
    descriptor = tail if isinstance(tail, dict) and tail.get('commit') and tail.get('path') else None
    resolved = C.scalar_metadata(configuration, descriptor)
    scalar, tail_source = resolved['scalar'], resolved['tail_source']
    physical = R.D.from_b(configuration, m, row['physical_candidate'],
        input_source=dict(configuration=configuration_source, tail=tail_source,
            scalar=dict(hamiltonian_constant=C.SCALAR_CONSTANT, tail=tail_source)),
        execution_source=deepcopy(execution_source), expected_scalar=scalar)
    candidate = physical['physical_context']['physical_candidate']
    cohort = 'RPE_SIMPLE_VALIDATION_'+request['model_id']+'_V1'
    seed = R.P.trial_seed_function(root)
    seeds = [[seed(candidate, cohort, i, outer, 0) for outer in range(candidate['q'])] for i in (0, 1)]
    old = _seed_values(git_read(root, request['frozen_model']['source']))
    for source in calibration_seed_sources:
        old.update(_seed_values(git_read(root, source)))
    if not old:
        raise ValueError('Calibration seed provenance required for independent DF paths')
    flat = [value for values in seeds for value in values]
    if len(set(flat)) != len(flat) or set(flat) & old:
        raise ValueError('DF independent path or calibration seed collision')
    base = git_read(root, C.BASE_INPUTS)
    return dict(kind='PR_DF', backend='BUILTIN_DF_NORMAL_PATH',
        configuration=deepcopy(configuration), configuration_source=deepcopy(configuration_source),
        legacy_view=str(C.FIXED_VIEW), interpreter=str(interpreter), execution_source=deepcopy(execution_source),
        cell=dict(m=m, role='HOLDOUT', request=physical, cohort=cohort, indices=[0, 1], seeds=seeds),
        runtime=deepcopy(runtime or base['runtime']), saved_input_files=deepcopy(base['saved_input_files']),
        local_support=deepcopy(base['local_support']), seed_check=dict(unique_new=True,
            overlap_with_calibration=0, old_seed_count=len(old)), output_dir=str(Path(output_dir).resolve()))


def validate_request(request):
    if request.get('schema') != SCHEMA or not request.get('model_id'):
        raise ValueError('Source-bound price-validation job required')
    model = request.get('frozen_model') or {}
    if not model.get('source') or not model.get('coefficients'):
        raise ValueError('Frozen model SOURCE and coefficients required')
    row = request.get('selected_round') or {}
    kind = (request.get('acquisition') or {}).get('kind')
    if kind not in KINDS:
        raise ValueError('Existing PF/PR acquisition route required')
    if (type(row.get('m')) is not int or row['m'] < 0 or
            type(row.get('q')) is not int or row['q'] < 1 or
            type(row.get('R')) is not int or row['R'] < (0 if kind == 'PF' else 1) or
            not row.get('physical_id')):
        raise ValueError('Actual scheduled circuit identity required')
    if row['physical_id'] in request.get('calibration_physical_ids', []):
        raise ValueError('Holdout circuit overlaps calibration')
    predictions = request.get('predicted_RZ') or {}
    if set(predictions) != {'X', 'Y'} or any(exact(v) <= 0 for v in predictions.values()):
        raise ValueError('Positive frozen X/Y predictions required')
    if not request.get('prediction_source') or not request.get('implementation'):
        raise ValueError('Prediction SOURCE and implementation contract required')
    expected = request.get('independent_paths', 2 if kind.startswith('PR_') else 1)
    if expected != (2 if kind.startswith('PR_') else 1):
        raise ValueError('Simple validation requires two PR paths or one deterministic circuit')
    return kind, expected


def check_command(command, kind):
    """Retain the existing preparation SOURCE lock before any compile starts."""
    root = Path(command['cwd']).resolve()
    source = command.get('source') or {}
    files = source.get('files') or {}
    argv = command.get('argv')
    if (not source.get('commit') or not files or not isinstance(argv, list) or
            not argv or any(not isinstance(v, str) for v in argv)):
        raise ValueError('Fixed existing acquisition command/SOURCE required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if head != source['commit']:
        raise ValueError('Acquisition SOURCE HEAD differs')
    entries = []
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Repository-relative SOURCE file required')
        file = root / relative
        if not file.is_file() or file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Acquisition SOURCE file differs: ' + name)
        if name in KINDS[kind] and any(Path(v).as_posix() in (name, file.as_posix()) for v in argv):
            entries.append(name)
    if len(entries) != 1:
        raise ValueError('Explicit existing acquisition entry required')
    if kind == 'PR_DF' and ('freeze' in argv or 'prepare' in argv):
        raise ValueError('Validation may not calibrate or refit a model')
    return root


def collect_observations(command):
    observations = []
    for binding in command.get('observations', []):
        value = read(binding['path'])
        valid = select(value, binding['valid_selector']) if binding.get('valid_selector') else True
        if valid is not True:
            raise ValueError('Original acquisition leaf did not admit the observation')
        rz = select(value, binding['value_selector'])
        if type(rz) is not int or rz < 0:
            raise ValueError('Compiled RZ observation must be an exact nonnegative integer')
        observations.append(dict(path_index=binding['path_index'], axis=binding['axis'],
            compiled_RZ=rz, source=deepcopy(binding['source']), receipt=deepcopy(binding['receipt']),
            raw_file=str(Path(binding['path']).resolve()), valid=True))
    return observations


def acquire_leaves(request, output):
    """Execute only existing prepared leaves; no implicit metadata/fit stages."""
    kind, expected_paths = validate_request(request)
    if request['acquisition'].get('backend') == 'BUILTIN_PAULI_NORMAL_PATH':
        return acquire_pauli_paths(request, output)
    if request['acquisition'].get('backend') == 'BUILTIN_DF_NORMAL_PATH':
        return acquire_pauli_paths(request, output, mode='df')
    commands = deepcopy(request['acquisition'].get('commands') or [])
    if not commands:
        raise ValueError('Prepared acquisition leaf commands are missing')
    if len(commands) > expected_paths:
        raise ValueError('One representative point only; too many acquisition leaves')
    declared = [(row.get('path_index'), row.get('axis')) for command in commands
                for row in command.get('observations', [])]
    paths = {path for path, _ in declared}
    if (len(paths) != expected_paths or len(declared) != 2*expected_paths or
            set(declared) != {(path, axis) for path in paths for axis in ('X', 'Y')}):
        raise ValueError('Freeze complete paired X/Y observation scope before acquisition')
    for command in commands:
        command['cwd'] = resolve(command['cwd'])
        command['argv'] = [resolve(argument) for argument in command['argv']]
        if os.environ.get('RPE_PREPARATION_PYTHON'):
            command['argv'][0] = os.environ['RPE_PREPARATION_PYTHON']
        for observation in command.get('observations', []):
            observation['path'] = resolve(observation['path'])
    roots = [check_command(command, kind) for command in commands]
    def run(index):
        command = commands[index]
        start = time.monotonic()
        with (output / ('LEAF_%d.log' % index)).open('x') as log:
            child = subprocess.Popen(command['argv'], cwd=roots[index], stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            put(output / ('LEAF_%d_STARTED.json' % index), dict(pid=child.pid,
                source=command['source'], argv=command['argv'], automatic_retry=False))
            code = child.wait()
        put(output / ('LEAF_%d_RETURN.json' % index), dict(returncode=code if code >= 0 else None,
            signal=-code if code < 0 else None, wall_seconds=time.monotonic()-start,
            source=command['source'], automatic_retry=False))
        if code:
            raise ValueError('Original acquisition leaf failed; no retry')
        return collect_observations(command)
    # Independent path leaves may run concurrently; a single leaf may already
    # own both paths. Scientific path and compiler caps remain its own contract.
    with ThreadPoolExecutor(len(commands)) as pool:
        result = list(pool.map(run, range(len(commands))))
    return [row for rows in result for row in rows]


def pauli_path(job_path):
    """Fixed normal-path kernel reused with this candidate's original operators."""
    job = read(job_path)
    acquisition = job['request']['acquisition']
    legacy = Path(acquisition['legacy_view']).resolve()
    folder = Path(job['folder']).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    def progress(stage, state, **detail):
        value = dict(stage=stage, state=state, monotonic_seconds=time.monotonic(), **detail)
        with (folder/'PROGRESS.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
            stream.flush()
        print('PRICE_VALIDATION_STAGE', stage, state, flush=True)
    progress('provider', 'START')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=legacy, text=True).strip() != PAULI_SOURCE:
        raise ValueError('Fixed Pauli legacy SOURCE differs')
    sys.path[:0] = [str(legacy/'scripts'), str(legacy/'src')]
    import run_pauli_b1_prices as B1
    if Path(B1.__file__).resolve() != (legacy/B1.CALLER).resolve():
        raise ValueError('Existing Pauli acquisition module path differs')
    for name in (*B1.REPLAY_SOURCES, B1.A_PATH):
        expected = subprocess.check_output(['git', 'show', PAULI_SOURCE+':'+name], cwd=legacy)
        if (legacy/name).read_bytes() != expected:
            raise ValueError('Existing Pauli acquisition SOURCE differs: '+name)
    # Same provider-first setup as the fixed split-calibration/m9 workers.
    import h46_long_time_signal as signal
    signal.install()
    B1.B0.provider_checks()
    B1.B0.register_pauli_modules()
    import qiskit
    if qiskit.__version__ != '1.3.0':
        raise ValueError('Fixed Qiskit version differs')
    import numpy as np
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = folder/'native_public', folder
    progress('provider', 'DONE')
    config = acquisition['configuration']
    prefix = config['spec']['L_D']
    progress('native_context', 'START')
    ctx = native.native_context('H4', (prefix,))
    if ctx.ham.constant != B1.B0.HAMILTONIAN_CONSTANT:
        raise ValueError('Original Hamiltonian constant differs')
    P = B1.B0.connection()
    packet = acquisition['packet']
    dictionary = P.load_dictionary(legacy, packet['dictionary_identity'])
    physical = P.round_request(packet, job['request']['selected_round']['m'], dictionary)
    physical['T'] = float(physical['T'])
    if physical != acquisition['physical_request']:
        raise ValueError('Frozen canonical round request differs')
    def array_source(locator):
        data = subprocess.check_output(['git', 'show', locator['commit']+':'+locator['path']], cwd=legacy)
        return np.load(io.BytesIO(data), allow_pickle=False)
    common = array_source(acquisition['common_operators'])
    saved = array_source(acquisition['operators'])
    operators = {name: saved[name] for name in saved.files}
    progress('native_context', 'DONE')
    compiles = []
    def compiler(wrapper, fingerprint, representative):
        if len(compiles) >= 2:
            raise ValueError('Two X/Y wrapper compiles maximum per normal path')
        compiles.append('X' if not compiles else 'Y')
        axis = compiles[-1]
        started = time.monotonic()
        progress('compile_'+axis, 'START', fingerprint=fingerprint)
        result = native.compile_cached(wrapper, fingerprint, representative)
        progress('compile_'+axis, 'DONE', wall_seconds=time.monotonic()-started)
        return result
    start = time.monotonic()
    index = job['index']
    progress('acquire_path', 'START', index=index)
    result = B1.acquire_path(native, ctx, dictionary, physical,
        cohort=acquisition['cohort'], index=index, seeds=acquisition['seeds'][index],
        operators=operators, blocks=common['blocks'], compile_cached=compiler)
    progress('acquire_path', 'DONE', index=index, wall_seconds=time.monotonic()-start)
    if result.get('status') != 'PASS' or compiles != ['X', 'Y']:
        raise ValueError('Original Pauli normal-path gate or wrapper coverage failed')
    path = dict(index=index, cohort=acquisition['cohort'], request=physical,
        dictionary_identity=packet['dictionary_identity'], result=result, reused=False)
    source = deepcopy(acquisition['execution_source'])
    source.update(kind='INDEPENDENT_SIMPLE_MODEL_VALIDATION',
        cohort=acquisition['cohort'], step_seeds=acquisition['seeds'],
        model_id=job['request']['model_id'], frozen_model=job['request']['frozen_model']['source'])
    receipts = B1.receipts().compile_receipts([path], source, packet['dictionary_identity'], folder)
    progress('save', 'START')
    put(folder/'COMPILE_RECEIPTS.json', receipts)
    with gzip.open(folder/'PATH.json.gz', 'wt', compresslevel=1) as stream:
        json.dump(path, stream, separators=(',', ':'), allow_nan=False)
    summary = {name: value for name, value in result.items() if name not in ('events', 'event_metadata')}
    put(folder/'SUMMARY.json', dict(status='PASS', index=index, source=source,
        result=summary, compile_calls=compiles, wall_seconds=time.monotonic()-start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    progress('save', 'DONE')
    return 0


def df_path(job_path):
    """Re-use the fixed selected-axis DF path worker, without calibration/fit."""
    import multiprocessing as mp
    job = read(job_path)
    acquisition = job['request']['acquisition']
    folder = Path(job['folder']).resolve()
    root = Path(__file__).resolve().parents[1]
    source = acquisition['execution_source']
    name = 'scripts/run_pr_df_candidate_prices.py'
    if (root/name).read_bytes() != subprocess.check_output(['git', 'show', source['commit']+':'+name], cwd=root):
        raise ValueError('Fixed DF candidate acquisition helper differs')
    C = _df_provider(root, acquisition['legacy_view'])
    R = C.runtime()
    for name, digest in acquisition['local_support'].items():
        if hashlib.sha256((C.FIXED_VIEW/name).read_bytes()).hexdigest() != digest:
            raise ValueError('DF local support SOURCE differs: '+name)
    cell = acquisition['cell']
    physical = cell['request']
    ham, metadata = R.restore(physical, acquisition['saved_input_files'], acquisition['runtime'])
    pc = physical['physical_context']
    runtime = R.D.NativeRuntime(root, ham, binding=pc['binding'], input_source=pc['input_source'],
        execution_source=pc['execution_source'])
    context = runtime.context(physical)
    R.PUBLIC = str(folder/'native')
    (folder/'native/acquisition').mkdir(parents=True, exist_ok=False)
    R.STATE.update(inputs=dict(acquisition, cache_root=str(folder/'isolated_cache')),
        contexts={cell['m']:context}, runtimes={cell['m']:runtime})
    R.init_price(mp.Event())
    result = C.selected_acquire_path((cell, job['index']))
    put(folder/'SUMMARY.json', dict(status='PASS' if result.get('status') == 'VALID' else 'STOP',
        result=result, source=source, metadata=metadata))
    if result.get('status') != 'VALID' or result.get('gate_status') != C.GATE:
        raise ValueError('Original DF path receipt gate failed')
    return 0


def acquire_pauli_paths(request, output, *, mode='pauli'):
    request = deepcopy(request)
    acquisition = runtime_request(request['acquisition'], ('legacy_view', 'interpreter', 'output_dir'))
    if mode == 'df':
        acquisition['runtime'] = runtime_request(acquisition['runtime'], ('provider_root',))
        acquisition['saved_input_files'] = {name:runtime_request(value, ('path',))
            for name, value in acquisition['saved_input_files'].items()}
    request['acquisition'] = acquisition
    source = acquisition['execution_source']
    script = Path(__file__).resolve()
    root = script.parents[1]
    expected = subprocess.check_output(['git', 'show', source['commit']+':'+SCRIPT], cwd=root)
    if script.read_bytes() != expected:
        raise ValueError('Validation adapter differs from frozen execution SOURCE')
    jobs = []
    for index in (0, 1):
        folder = output/('path%d' % index)
        folder.mkdir(exist_ok=False)
        job_path = folder/'JOB.json'
        put(job_path, dict(request=request, index=index, folder=str(folder)))
        jobs.append((folder, job_path))
    def run(index):
        folder, job_path = jobs[index]
        started = time.monotonic()
        with (folder/'CONSOLE.log').open('x') as log:
            child = subprocess.Popen([acquisition['interpreter'], '-B', str(script), '--'+mode+'-path', str(job_path)],
                cwd=root, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                    **({'RPE_PRICE_VALIDATION_DF_VIEW':acquisition['legacy_view']}if mode=='df'else{})))
            put(folder/'STARTED.json', dict(pid=child.pid, automatic_retry=False))
            code = child.wait()
        put(folder/'RETURN.json', dict(returncode=code if code >= 0 else None,
            signal=-code if code < 0 else None, wall_seconds=time.monotonic()-started, automatic_retry=False))
        if code:
            raise ValueError('Normal path failed; preserve evidence, no retry')
        summary = read(folder/'SUMMARY.json')
        if summary.get('status') != 'PASS':
            raise ValueError('Original normal-path gate did not pass')
        if mode == 'df':
            result = summary['result']
            observed = []
            for axis in ('X', 'Y'):
                records = [row for row in result['record']['records'] if row['axis'] == axis]
                if len(records) != 1:
                    raise ValueError('DF original axis receipt coverage differs')
                raw_name = 'cosine.json' if axis == 'X' else 'sine.json'
                files = list((folder/'native/acquisition').glob('m*/p*/axes_XY/'+raw_name))
                if len(files) != 1:
                    raise ValueError('DF original compiled-axis receipt absent')
                observed.append(dict(path_index=index, axis=axis, compiled_RZ=records[0]['C'], valid=True,
                    source=deepcopy(summary['source']), receipt=dict(artifact=str(files[0]), selector=[]),
                    raw_file=str(files[0].with_name('PATH_RECEIPT.json')),
                    gate_status=result['gate_status'],
                    numerical_action_gate=result['numerical_action_gate']))
            return observed
        return [dict(path_index=index, axis=axis,
            compiled_RZ=summary['result']['prices'][name]['rz_count'], valid=True,
            source=deepcopy(summary['source']),
            receipt=dict(artifact=str(folder/'COMPILE_RECEIPTS.json'), selector=['p%d%s' % (index, axis)]),
            raw_file=str(folder/'PATH.json.gz')) for axis, name in (('X', 'cosine'), ('Y', 'sine'))]
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(run, (0, 1)))
    return [row for path in rows for row in path]


def summarize(request, observations):
    _, expected = validate_request(request)
    by = {}
    for row in observations:
        key = row.get('path_index'), row.get('axis')
        if (type(key[0]) is not int or key[0] < 0 or key[1] not in ('X', 'Y') or key in by or
                row.get('valid') is not True or type(row.get('compiled_RZ')) is not int or
                row['compiled_RZ'] < 0 or not row.get('source') or not row.get('receipt')):
            raise ValueError('Missing, duplicate, invalid or unsourced measured axis')
        by[key] = row
    paths = sorted({p for p, _ in by})
    if len(paths) != expected or set(by) != {(p, a) for p in paths for a in ('X', 'Y')}:
        raise ValueError('Incomplete independent paths; do not replace missing observations')
    rows = []
    for axis in ('X', 'Y'):
        observed = sum((Fraction(by[p, axis]['compiled_RZ']) for p in paths), Fraction()) / expected
        prediction = exact(request['predicted_RZ'][axis])
        rows.append(dict(axis=axis, predicted_RZ=enc(prediction), observed_mean_RZ=enc(observed),
            signed_relative_difference=None if observed == 0 else enc((prediction-observed)/observed),
            relative_difference_reason='ZERO_OBSERVED_PRICE' if observed == 0 else None,
            independent_paths=expected, paths=[deepcopy(by[p, axis]) for p in paths]))
    return dict(schema='RPE_PRICE_VALIDATION_RESULT_V1', status='INDEPENDENT_MEASURED',
        model_id=request['model_id'], selected_round=deepcopy(request['selected_round']),
        frozen_model=deepcopy(request['frozen_model']), prediction_source=deepcopy(request['prediction_source']),
        predicted_RZ=deepcopy(request['predicted_RZ']),
        implementation=deepcopy(request['implementation']), rows=rows, independent_paths=expected,
        relative_difference_definition='(prediction-observed_mean)/observed_mean; positive means overprediction',
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE', model_qualification_changed=False,
        coefficient_refit=False, residual_applied_to_other_rounds=False, automatic_retry=False)


def run_job(request, output_dir, *, acquire=None):
    """One queue-owned representative point. A started job is never rerun here."""
    validate_request(request)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_file = output / 'REQUEST.json'
    if request_file.exists():
        if read(request_file) != request:
            raise ValueError('Frozen request differs')
        if (output / 'RESULT.json').exists():
            return read(output / 'RESULT.json')
        raise ValueError('Prior acquisition started or ambiguous; automatic retry forbidden')
    with request_file.open('x') as stream:
        json.dump(request, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = None
    try:
        observations = (acquire or acquire_leaves)(deepcopy(request), output)
        put(output / 'OBSERVATIONS.json', observations)
        result = summarize(request, observations)
    except BaseException:
        result = dict(schema='RPE_PRICE_VALIDATION_RESULT_V1', status='VALIDATION_INCOMPLETE',
            model_id=request['model_id'], selected_round=deepcopy(request['selected_round']),
            error=traceback.format_exc(), automatic_retry=False, model_qualification_changed=False,
            coefficient_refit=False, residual_applied_to_other_rounds=False)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
        resource_record = dict(wall_seconds=time.monotonic()-start,
            worker_user_seconds=own.ru_utime, worker_system_seconds=own.ru_stime,
            child_user_seconds=after.ru_utime-before.ru_utime,
            child_system_seconds=after.ru_stime-before.ru_stime,
            worker_peak_RSS_KiB=own.ru_maxrss, child_peak_RSS_KiB=after.ru_maxrss,
            peak_RSS_scope='worker lifetime and maximum reaped child; not concurrent sum',
            automatic_retry=False)
        put(output / 'RESOURCE_FINAL.json', resource_record)
        if result is not None:
            result['resources'] = resource_record
            put(output / 'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pauli-path', type=Path)
    parser.add_argument('--df-path', type=Path)
    args = parser.parse_args()
    job_path = args.pauli_path or args.df_path
    if job_path:
        started = time.monotonic()
        folder = Path(read(job_path)['folder'])
        try:
            return pauli_path(job_path) if args.pauli_path else df_path(job_path)
        except BaseException:
            put(folder/'FAILURE.json', dict(status='VALIDATION_INCOMPLETE',
                error=traceback.format_exc(), automatic_retry=False))
            return 2
        finally:
            own = resource.getrusage(resource.RUSAGE_SELF)
            put(folder/'RESOURCE_FINAL.json', dict(wall_seconds=time.monotonic()-started,
                worker_user_seconds=own.ru_utime, worker_system_seconds=own.ru_stime,
                worker_peak_RSS_KiB=own.ru_maxrss, peak_RSS_scope='one path worker lifetime'))
    if args.request is None or args.output is None:
        parser.error('--request and --output required')
    # Preparation bridge names its declared JSON output. The library API uses
    # the containing attempt directory so raw paths/resources travel with it.
    output = args.output.parent if args.output.suffix == '.json' else args.output
    result = run_job(read(args.request), output)
    if args.output.suffix == '.json' and args.output.name != 'RESULT.json':
        put(args.output, result)
    print(json.dumps(dict(status=result['status'], model_id=result['model_id'])))
    return 0 if result['status'] == 'INDEPENDENT_MEASURED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
