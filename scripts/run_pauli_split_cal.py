"""PR-Pauli split calibration cell for a saved split (LD5: Q8R4), 2 fresh paths x X/Y, <= 4 compiles.

The LD3 driver (``run_pauli_ld3_cal.py`` at 158bbcc7b) is reused unchanged for its generic parts:
the pinned legacy science 13ab1acfa from the dedicated checkout, the C2 m9 worker functions,
the non-configuration cell identity rule (checked against the saved LD2 Q8R66 request) and
the git helpers. Only the split inputs (prefix, dictionary/operators, configuration, cell R,
cohort and output names) become a spec. The LD3 spec reproduces the published LD3 binding.

    python -B scripts/run_pauli_split_cal.py --prefix 5 --metadata
    python -B scripts/run_pauli_split_cal.py --prefix 5 --input-preflight <scratch>
    python -B scripts/run_pauli_split_cal.py --prefix 5 --freeze
    python -B scripts/run_pauli_split_cal.py --prefix 5 --execute <binding commit>
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/run_pauli_split_cal.py'
_spec = importlib.util.spec_from_file_location('_ld3_cal_driver', ROOT / 'scripts/run_pauli_ld3_cal.py')
D3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D3)
LD3_DRIVER = dict(commit='158bbcc7b88883947b726b1f19bf037b3af3ef18', path='scripts/run_pauli_ld3_cal.py')
blob, read, write, same, head = D3.blob, D3.read, D3.write, D3.same, D3.head
AXES, COMPILE_MAXIMUM = D3.AXES, D3.COMPILE_MAXIMUM
SPECS = {
    3: dict(prefix=3, plan=D3.PLAN, config=D3.CONFIG, config_id=D3.CONFIG_ID, total_view=D3.TOTAL_VIEW,
            p0=D3.P0_LD3, cell=D3.CELL, cohort=D3.COHORT, m12=dict(id=D3.M12_ID, q=4096, R=660),
            base=D3.BASE, private=str(D3.PRIVATE), old_seed_sources=list(D3.OLD_SEEDS)),
    5: dict(prefix=5, plan=dict(commit='bf0a930fd75fb2eb693622edcf3b7204e30f5eff',
                                path='docs/research/df_head_pauli_tail_ld5_stage1_20261116/HANDOFF_B.md'),
            config=dict(commit='39a1fc7dfe9b093f0e246be595e2f7938191368b',
                        path='docs/research/df_head_pauli_tail_ld5_stage1_20261116/config_run01/CONFIGURATION_LD5.json'),
            config_id='4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb', total_view=None,
            p0=dict(commit='6255fbc422121fd777cad52763ef5a684abd46a3',
                    directory='docs/research/df_head_pauli_tail_ld5_stage1_20261116/p0_run01/'),
            cell=dict(cell_id='Q8R4', q=8, R=4, T=.1 * 8, K=2), cohort='A_PAULI_LD5_CAL_Q8R4_N2_V1',
            m12=dict(id='RMSE:795c55355aec18b321e90769af58f2b4a8d7a3dcd793b488cbc87fa3d2a1a267', q=4096, R=4),
            base='docs/research/a_pauli_ld5_cal_price_20261116',
            private='/home/AbeHiromu/projects/a-pauli-ld5-cal-20261116-private',
            old_seed_sources=list(D3.OLD_SEEDS) + [dict(commit='3074433d16dd8bbfa3fbf173e1da3606af6670d2',
                path='docs/research/a_pauli_ld3_cal_price_20261116/BINDING.json')])}
FROZEN_KEYS = ('execution_source', 'created_UTC', 'sampling', 'compiles', 'status')
REQUEST_ADAPTER = 'src/trottertracks/pauli_calibration_request.py'


def request_api():
    """Metadata-only adapter by file path; preserve provider-first worker imports."""
    name = '_a_pauli_calibration_request'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / REQUEST_ADAPTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_request_spec(path):
    """A preparation response's driver_spec or a plain request-driven spec."""
    value = json.loads(Path(path).read_text())
    spec = deepcopy(value['driver_spec'] if 'driver_spec' in value else value)
    if not isinstance(spec, dict):
        raise ValueError('Ready calibration driver_spec required')
    spec['request_driven'] = True
    request_api().validate_driver_spec(ROOT, spec)
    return spec


def select_job_spec(job):
    """New jobs carry their own validated split; old jobs retain the frozen SPECS route."""
    if 'spec' not in job:
        return SPECS[job['prefix']]
    spec = deepcopy(job['spec'])
    request_api().validate_driver_spec(ROOT, spec)
    same(spec['prefix'], job['prefix'], 'Job/spec prefix differs')
    if spec.get('reuse_only'):
        raise ValueError('Saved calibration is reuse-only; do not acquire another path')
    binding = job.get('binding')
    if not isinstance(binding, dict):
        raise ValueError('Request-driven job requires its matching frozen binding')
    same(binding['prefix'], spec['prefix'], 'Job spec/binding prefix differs')
    same(binding['configuration']['id'], spec['config_id'], 'Job spec/binding configuration differs')
    same(binding['dictionary_identity'], spec['configuration']['spec']['split']['tail_dictionary'],
         'Job spec/binding dictionary differs')
    same(binding['cohort'], spec['cohort'], 'Job spec/binding cohort differs')
    request = binding['request']; cell = spec['cell']
    same((request['prefix'], request['q'], request['R'], request['T'], request['K']),
         (spec['prefix'], cell['q'], cell['R'], cell['T'], cell['K']), 'Job spec/binding cell differs')
    if 'calibration_domain' in spec:
        same(binding.get('calibration_domain'), spec['calibration_domain'], 'Job spec/binding step domain differs')
        same(binding.get('covered_rounds'), spec['covered_rounds'], 'Job spec/binding covered rounds differ')
        step = spec['calibration_domain']['actual_step']
        same(Fraction(request['step_time']), Fraction(step['numerator'], step['denominator']),
             'Job spec/binding exact step differs')
    same(request['physical_cell']['split'], spec['configuration']['spec']['split'],
         'Job spec/binding physical split differs')
    same(request['physical_cell']['profile'], spec['configuration']['spec']['profile'],
         'Job spec/binding compiler profile differs')
    same(request['physical_cell']['binding'], spec['configuration']['spec']['binding'],
         'Job spec/binding Hamiltonian snapshot or state differs')
    same(Fraction(request['lambda_P']), Fraction(**spec['configuration']['spec']['split']['lambda_R']),
         'Job spec/binding dictionary lambda differs')
    same(binding['operators']['ld%d' % spec['prefix']],
         dict(commit=spec['p0']['commit'], path=spec['p0']['directory']+'LD%d_OPERATORS.npz' % spec['prefix']),
         'Job spec/binding operators differ')
    return spec


def old_seeds(spec):
    seeds = set(); src = spec['old_seed_sources']
    for d in read(src[0])['dictionaries'].values():
        seeds.update(s for c in d['cells'] for row in c['step_seeds'] for s in row)
    seeds.update(s for row in read(src[1])['step_seeds'] for s in row)
    for ref in src[2:]:
        data = read(ref)
        rows = data['binding']['step_seeds'] if 'binding' in data else data['step_seeds']
        seeds.update(s for row in rows for s in row)
    return seeds


def arrays(spec):
    """Saved full-2^n operators: common blocks (4220f300) + this prefix's head/tail/Pauli."""
    import numpy as np
    common = np.load(io.BytesIO(blob(D3.COMMON['commit'], D3.COMMON['path'])), allow_pickle=False)
    ops = np.load(io.BytesIO(blob(spec['p0']['commit'], spec['p0']['directory'] + 'LD%d_OPERATORS.npz' % spec['prefix'])),
                  allow_pickle=False)
    return {k: ops[k] for k in ops.files}, common['blocks']


def metadata(spec):
    same((ROOT / LD3_DRIVER['path']).read_bytes(), blob(LD3_DRIVER['commit'], LD3_DRIVER['path']), 'Reused LD3 driver differs')
    B1 = D3.legacy(); P = B1.B0.connection(); prefix = spec['prefix']; cell = spec['cell']
    config = spec.get('configuration') or read(spec['config']); s = config['spec']
    same(config['id'], spec['config_id'], 'Configuration ID differs')
    identity = s['split']['tail_dictionary']
    dictionary = P.load_dictionary(D3.PIN_ROOT, identity)
    lam = Fraction(s['split']['lambda_R']['numerator'], s['split']['lambda_R']['denominator'])
    same((s['L_D'], dictionary['prefix'], Fraction(dictionary['lambda_P'])), (prefix, prefix, lam), 'Prefix/lambda_P differs')
    same(dictionary['saved_DF_scalar'], s['split']['scalar'], 'Saved s_R differs')
    ld2_spec = read(D3.LD2_CONFIG); saved = read(D3.LD2_Q8R66)
    ld2 = D3.cell_request(P, ld2_spec, dict(q=8, R=66, T=.1 * 8, K=2), saved['lambda_P'])
    same(ld2, saved, 'Cell identity rule does not reproduce the saved LD2 Q8R66 request')
    request = D3.cell_request(P, s, cell, dictionary['lambda_P'])
    if 'calibration_domain' in spec:
        step = spec['calibration_domain']['actual_step']
        step = Fraction(step['numerator'], step['denominator'])
        same((request['q'], request['R'], request['T'], Fraction(request['step_time'])),
             (cell['q'], cell['R'], cell['T'], step), 'Cell q/R/T/exact step differs')
    else:
        same((request['q'], request['R'], request['T'], request['step_time']),
             (cell['q'], cell['R'], 0.8, 0.1), 'Cell q/R/T differs')
    m12 = config['rounds'][12]
    same((m12['candidate']['id'], m12['q_PF'], m12['R_outer']), (spec['m12']['id'], spec['m12']['q'], spec['m12']['R']),
         'm12 per-step R differs')
    seeds = [[P.seed(request['candidate_id'], spec['cohort'], i, outer, 0) for outer in range(cell['q'])] for i in (0, 1)]
    flat = [x for row in seeds for x in row]
    previous = old_seeds(spec)
    if len(set(flat)) != len(flat) or set(flat) & previous:
        raise ValueError('New or old cohort seed collision')
    value = dict(schema='A_PAULI_LD%d_CAL_BINDING_V1' % prefix, cell_id=cell['cell_id'], prefix=prefix, cohort=spec['cohort'],
        path_indices=[0, 1], axes=['X', 'Y'], n_paths=2, compile_maximum=COMPILE_MAXIMUM, request=request,
        step_seeds=seeds, dictionary_identity=identity, configuration=dict(spec.get('config') or {}, id=spec['config_id'],
            **({'embedded': True} if not spec.get('config') else {})),
        total_view=spec['total_view'], m12_canonical=dict(physical_id=spec['m12']['id'], q=spec['m12']['q'], R=spec['m12']['R'],
            relation='SAME_PER_STEP_R_AS_m12_BUT_DIFFERENT_q_T_AND_ID_NOT_A_ROUND_RECEIPT'),
        operators={'common': D3.COMMON, 'ld%d' % prefix: dict(commit=spec['p0']['commit'],
                   path=spec['p0']['directory'] + 'LD%d_OPERATORS.npz' % prefix)},
        scalar=dict(value=P.total_scalar(dictionary, B1.B0.HAMILTONIAN_CONSTANT),
                    quantity=P.SCALAR_QUANTITY, application='ONCE_AT_TOTAL_T'),
        rule_check=dict(reproduced_saved_request=dict(D3.LD2_Q8R66, candidate_id=saved['candidate_id']), equal=True),
        seed_checks=dict(new_step_seeds=len(flat), unique_new=True, old_cohort_overlap=False,
                         old_sources=spec['old_seed_sources'], old_seed_count=len(previous)),
        legacy_source=D3.PIN, legacy_checkout=str(D3.PIN_ROOT), c2_worker_pattern=D3.C2_DRIVER, driver=SCRIPT,
        plan=spec['plan'])
    if 'calibration_domain' in spec:
        value.update(calibration_domain=deepcopy(spec['calibration_domain']), covered_rounds=deepcopy(spec['covered_rounds']))
        value['m12_canonical']['relation'] = 'WHOLE_SCHEDULE_PROVENANCE_NOT_CALIBRATION_R_OR_STEP_ANCHOR'
    return value


def input_preflight(spec, scratch):
    """Provider-first child: restore H4 and prepare(ham, prefix); input/bit-order gates only."""
    B1 = D3.legacy()
    import h46_long_time_signal as signal
    signal.install(); B1.B0.provider_checks(); B1.B0.register_pauli_modules()
    import numpy, qiskit
    m = metadata(spec); P = B1.B0.connection(); prefix = spec['prefix']
    dictionary = P.load_dictionary(D3.PIN_ROOT, m['dictionary_identity'])
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(scratch) / 'public', Path(scratch) / 'private'
    ctx = native.native_context('H4', (prefix,)); prep = ctx.preparations[prefix]
    operators, blocks = arrays(spec)
    gates = P.bit_order_checks(dictionary, operators, blocks)
    comps, labels, n = P.components(dictionary)
    return dict(status='INPUT_PREFLIGHT_NO_SAMPLING_BUILD_OR_COMPILE', prefix=prefix, qiskit=qiskit.__version__,
                numpy=numpy.__version__, python=sys.version, executable=sys.executable,
                hamiltonian_constant_equal=ctx.ham.constant == B1.B0.HAMILTONIAN_CONSTANT,
                head_blocks=len(prep.deterministic_blocks), head_fragment_indices=list(prep.deterministic_fragment_indices),
                native_tail_identity=prep.extracted_identity_coefficient, saved_sR=dictionary['saved_DF_scalar'],
                native_identity_equals_saved_sR=prep.extracted_identity_coefficient == dictionary['saved_DF_scalar'],
                bit_order=gates, terms=len(comps), qubits=n, system_qubits=ctx.ham.n_qubits)


def freeze(spec):
    if spec.get('reuse_only'):
        raise ValueError('Saved calibration is reuse-only; no new binding')
    m = metadata(spec); source = head()
    same((ROOT / SCRIPT).read_bytes(), blob(source, SCRIPT), 'Commit the driver before freezing the binding')
    write(ROOT / spec['base'] / 'BINDING.json', dict(m, execution_source=dict(commit=source, path=SCRIPT, legacy_source=D3.PIN),
          created_UTC=datetime.now(timezone.utc).isoformat(), sampling=0, compiles=0, status='BINDING_FIXED_BEFORE_SAMPLING'))
    print(json.dumps(dict(candidate_id=m['request']['candidate_id'], seeds=m['step_seeds'], source=source), indent=1))


def check_result_path(path, binding, receipts=None, source=None):
    """D3.check_result_path with this binding's cohort."""
    i = path['index']; cohort = binding['cohort']
    if i not in (0, 1):
        raise ValueError('Wrong path index')
    B1 = D3.legacy()
    B1.receipts().check_path(path, request=binding['request'], dictionary=binding['dictionary_identity'],
                             cohort=cohort, index=i, seeds=binding['step_seeds'][i])
    if path['result']['sampling'] != 'ORIGINAL_NORMAL_SAMPLER':
        raise ValueError('Independent normal sampler required')
    if receipts is not None:
        for a, key in AXES.items():
            receipt = receipts['p%d%s' % (i, a)]; r = path['result']
            same((receipt['axis'], receipt['index'], receipt['cohort'], receipt['request'], receipt['seeds'],
                  receipt['dictionary_identity']),
                 (a, i, cohort, binding['request'], binding['step_seeds'][i], binding['dictionary_identity']),
                 'Individual receipt binding differs')
            same(receipt['execution_source'], source, 'Receipt execution SOURCE differs')
            same(receipt['record'], r['compile_receipts'][key], 'Original compiler record differs')
            same(receipt['record']['fingerprint'], r['fingerprints'][key]['fingerprint'], 'Fingerprint differs')
            same(receipt['record']['metrics'], r['prices'][key], 'Receipt metrics differ')
            same(receipt['record']['compiler'], binding['request']['physical_cell']['profile']['compiler'], 'Compiler differs')


def worker(job_path):
    job = json.loads(Path(job_path).read_text()); i = job['index']; folder = Path(job['private'])
    log = D3.c2().Progress(folder / 'PROGRESS.jsonl')
    try:
        spec = select_job_spec(job)
        frozen = read(job['binding_reference'])
        D3.check_binding(job['binding'], {k: v for k, v in frozen.items() if k not in FROZEN_KEYS})
        same(job['driver_commit'], frozen['execution_source']['commit'], 'Frozen driver SOURCE differs')
        same(blob(job['driver_commit'], SCRIPT), (ROOT / SCRIPT).read_bytes(), 'Running driver bytes differ')
        if 'spec' in job:
            same(blob(job['driver_commit'], REQUEST_ADAPTER), (ROOT / REQUEST_ADAPTER).read_bytes(),
                 'Request adapter SOURCE differs')
        B1 = D3.legacy()
        import h46_long_time_signal as signal
        with log.stage('install_provider'):
            signal.install(); B1.B0.provider_checks(); B1.B0.register_pauli_modules()
            import qiskit
            same(qiskit.__version__, '1.3.0', 'Qiskit differs')
            P = B1.B0.connection(); dictionary = P.load_dictionary(D3.PIN_ROOT, job['binding']['dictionary_identity'])
            import h46_long_time_native as native
            native.PUBLIC, native.PRIVATE = folder / 'native_public', folder
            ctx = native.native_context('H4', (spec['prefix'],))
            same(ctx.ham.constant, B1.B0.HAMILTONIAN_CONSTANT, 'Hamiltonian constant differs')
            operators, blocks = arrays(spec)
        compile_calls = []
        def compiler(wrapper, fp, representative):
            axis = 'X' if not compile_calls else 'Y'
            if len(compile_calls) >= 2:
                raise ValueError('Two compile maximum per path')
            compile_calls.append(axis)
            with log.stage('compile_' + axis, instructions=len(wrapper.data), fingerprint=fp):
                return native.compile_cached(wrapper, fp, representative)
        P = B1.B0.connection()
        with log.stage('acquire_path', index=i), D3.c2().accelerated_observed(P, log) as stats:
            r = B1.acquire_path(native, ctx, dictionary, job['binding']['request'], cohort=spec['cohort'], index=i,
                                seeds=job['binding']['step_seeds'][i], operators=operators, blocks=blocks,
                                compile_cached=compiler)
        path = dict(index=i, cohort=spec['cohort'], request=job['binding']['request'],
                    dictionary_identity=job['binding']['dictionary_identity'], result=r, reused=False)
        check_result_path(path, job['binding'])
        with log.stage('receipt_export'):
            receipts = B1.receipts().compile_receipts([path], job['execution_source'], job['binding']['dictionary_identity'], folder)
            check_result_path(path, job['binding'], receipts, job['execution_source'])
            write(folder / 'COMPILE_RECEIPTS.json', receipts)
        summary = {k: v for k, v in r.items() if k not in ('events', 'event_metadata')}
        summary.update(acceleration=stats, compile_calls=compile_calls)
        with log.stage('raw_save'):
            with gzip.open(folder / 'PATH.json.gz', 'wt', compresslevel=1) as f:
                json.dump(path, f, separators=(',', ':'), allow_nan=False)
            write(folder / 'SUMMARY.json', dict(status='PASS', index=i, result=summary,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        log('worker', 'DONE', index=i); return 0
    except BaseException:
        log('worker', 'STOP', error=traceback.format_exc()[-400:])
        write(folder / 'FAILURE.json', dict(status='STOP', index=i, error=traceback.format_exc(),
              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, automatic_retry=False))
        return 2


def execute(binding_commit, spec):
    if spec.get('reuse_only'):
        raise ValueError('Saved calibration is reuse-only; do not acquire another path')
    base = spec['base']; private = Path(spec['private']); prefix = spec['prefix']
    reference = dict(commit=binding_commit, path=base + '/BINDING.json')
    frozen = read(reference); m = metadata(spec)
    D3.check_binding({k: v for k, v in frozen.items() if k not in FROZEN_KEYS}, m)
    source = frozen['execution_source']
    same((ROOT / SCRIPT).read_bytes(), blob(source['commit'], SCRIPT), 'Frozen SOURCE changed')
    remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin',
                                     spec.get('execution_branch', D3.BRANCH)], cwd=ROOT, text=True).split()[0]
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', binding_commit, remote], cwd=ROOT)
    same(blob(remote, reference['path']), blob(binding_commit, reference['path']), 'Published binding differs')
    out = ROOT / base / 'run01'
    if out.exists() or private.exists():
        raise FileExistsError('Fresh scientific output required; no automatic retry')
    out.mkdir(); private.mkdir(parents=True)
    log = D3.c2().Progress(out / 'PROGRESS.jsonl')
    execution_source = dict(commit=source['commit'], caller=SCRIPT, prefix=prefix, kind='LD%d_CAL_%s' % (prefix, spec['cell']['cell_id']),
        cell=spec['cell']['cell_id'], cohort=spec['cohort'], step_seeds=m['step_seeds'], result_path=base + '/run01/RESULT.json',
        legacy_source=D3.PIN)
    write(out / 'EXECUTION.json', dict(binding=reference, execution_source=execution_source, python=sys.version,
          executable=sys.executable, started_UTC=datetime.now(timezone.utc).isoformat(),
          compile_maximum=COMPILE_MAXIMUM, automatic_retry=False))
    env = os.environ.copy()
    if not (spec.get('request_driven') or 'configuration' in spec):
        # Keep the frozen historical run's settings; new requests inherit the job environment.
        env['OPENBLAS_NUM_THREADS'] = '1'; env['OMP_NUM_THREADS'] = '1'
    processes = []
    for i in (0, 1):
        folder = private / ('path' + str(i)); folder.mkdir()
        job = dict(index=i, prefix=prefix, private=str(folder), binding=m, execution_source=execution_source,
                   driver_commit=source['commit'], binding_reference=reference)
        if spec.get('request_driven') or 'configuration' in spec:
            job['spec'] = deepcopy(spec)
        write(folder / 'JOB.json', job)
        stream = (folder / 'CONSOLE.txt').open('x')
        p = subprocess.Popen([sys.executable, '-B', str(ROOT / SCRIPT), '--worker', str(folder / 'JOB.json')],
                             cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        processes.append((i, p, stream)); log('worker', 'START', index=i, pid_child=p.pid)
    states = D3.c2().observe(processes, log)
    write(out / 'PROCESS_EXITS.json', list(states.values()))
    ok = all(s['exit_code'] == 0 for s in states.values())
    summaries, receipts, manifests = [], {}, []
    for i in (0, 1):
        folder = private / ('path' + str(i))
        for name in ('PROGRESS.jsonl', 'SUMMARY.json', 'FAILURE.json', 'CONSOLE.txt'):
            if (folder / name).exists():
                (out / ('path%d_' % i + name)).write_bytes((folder / name).read_bytes())
        if (folder / 'SUMMARY.json').exists():
            summaries.append(json.loads((folder / 'SUMMARY.json').read_text()))
        if (folder / 'COMPILE_RECEIPTS.json').exists():
            receipts.update(json.loads((folder / 'COMPILE_RECEIPTS.json').read_text()))
        if (folder / 'PATH.json.gz').exists():
            data = (folder / 'PATH.json.gz').read_bytes(); name = 'PATH%d.json.gz' % i
            (out / name).write_bytes(data)
            manifests.append(dict(index=i, path=base + '/run01/' + name, bytes=len(data),
                                  sha256=hashlib.sha256(data).hexdigest(), encoding='GZIP_JSON', selector=[]))
    ok = ok and len(summaries) == 2 and len(receipts) == 4
    write(out / 'RAW_PATHS.json', manifests); write(out / 'COMPILE_RECEIPTS.json', receipts)
    write(out / 'RESULT.json', dict(status='FINITE_LD%d_CAL_PATHS_COMPLETE' % prefix if ok else 'STOP_LD%d_CAL' % prefix,
          source=execution_source, binding=reference, summaries=summaries, raw_paths=manifests, compiles=len(receipts),
          automatic_retry=False, mean=None, fit=None, price=None, N=None, G=None, mandatory_stop=True))
    log('acquisition', 'DONE' if ok else 'STOP', valid_paths=len(summaries))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=int); ap.add_argument('--metadata', action='store_true')
    ap.add_argument('--freeze', action='store_true'); ap.add_argument('--execute'); ap.add_argument('--worker')
    ap.add_argument('--input-preflight')
    ap.add_argument('--request', help='A preparation response with driver_spec, or a plain driver spec')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    if a.request:
        spec = load_request_spec(a.request)
        if a.prefix is not None and a.prefix != spec['prefix']:
            ap.error('--prefix differs from --request')
    else:
        if a.prefix not in SPECS or a.prefix == 3 and (a.freeze or a.execute):
            ap.error('--prefix 5 (LD3 is published; its spec is a reproduction control only)')
        spec = SPECS[a.prefix]
    if (a.freeze or a.execute) and spec.get('reuse_only'):
        ap.error('Saved calibration is reuse-only; no new freeze or execution')
    if a.input_preflight:
        print(json.dumps(input_preflight(spec, a.input_preflight), indent=1, default=str)); return 0
    if a.freeze:
        freeze(spec); return 0
    if a.execute:
        return execute(a.execute, spec)
    print(json.dumps(metadata(spec), indent=2)); return 0


if __name__ == '__main__':
    sys.exit(main())
