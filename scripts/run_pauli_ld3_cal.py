"""A stage 3: PR-Pauli LD3 calibration cell Q8R660 (q=8, R=660, T=0.8), 2 fresh paths x X/Y.

Pinned legacy science 13ab1acf (the LD2 CAL and C2 m9 SOURCE) from a dedicated detached
checkout; nothing in it is modified. LD3 inputs: dictionary/operators 50640b74c, configuration
dd5b881d (421258d40). The non-configuration cell identity is built exactly as the legacy
``run_pauli_b1_prices.request_for`` builds it (binding/split/profile of the configuration
spec, route, lowering, q, R, T, K); the rule is checked by reproducing the saved LD2 Q8R66
request bit for bit. The worker is C2's m9 worker pattern (cb034e547): provider-first
child, observed stages, pure event-matrix memo and same-byte fingerprint writer, two
independent path processes, at most 2 compiles per path, receipt export, gzip raw.

No fit, price, Q3 or shot allocation here (see pauli_ld3_cal_audit.py for the audit/price).

    python -B scripts/run_pauli_ld3_cal.py --metadata
    python -B scripts/run_pauli_ld3_cal.py --freeze          # writes BINDING.json (commit/push it)
    python -B scripts/run_pauli_ld3_cal.py --execute <binding commit>
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
SCRIPT = 'scripts/run_pauli_ld3_cal.py'
BRANCH = 'codex/a-pauli-ld2-expected-price-plan-20261115'
PIN = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
PIN_ROOT = ROOT.parent / 'a-pauli-ld3-cal-source-13ab'
C2_DRIVER = dict(commit='cb034e547c444ea67a4f4f90a8534c13f4db4451', path='scripts/run_pauli_ld2_m9.py')
BASE = 'docs/research/a_pauli_ld3_cal_price_20261116'
PRIVATE = Path('/home/AbeHiromu/projects/a-pauli-ld3-cal-20261116-private')
PLAN = dict(commit='a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6', path='docs/research/a_pauli_ld_split_plan_20261116/REPORT.md')
CONFIG = dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb',
              path='docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json')
CONFIG_ID = 'dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847'
TOTAL_VIEW = dict(id='6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb',
                  commit='2de33a4f4', path='docs/research/pauli_ld3_vp_q3_results_20261116/HANDOFF.md')
P0_LD3 = dict(commit='50640b74c65c36d381d6058824745f6e38ab3c2d',
              directory='docs/research/df_head_pauli_tail_ld3_stage1_20261116/p0_run01/')
COMMON = dict(commit='4220f3006d0d8d413e195ed948e250578513b557',
              path='docs/research/df_head_pauli_tail_p0_results_20261114/run01/COMMON_OPERATORS.npz')
LD2_CONFIG = dict(commit='e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7',
                  path='docs/research/df_head_pauli_tail_p1_results_20261114/run01/PLAN.json',
                  selector=['common_kappa_configurations', '2', 'spec'])
LD2_Q8R66 = dict(commit='74955a11c', path='docs/research/d_pauli_ld2_b1_cal_results_20261115/run01/Q8R66/RESULT.json',
                 selector=['child', 'paths', 0, 'request'])
OLD_SEEDS = [dict(commit='043c7567451e8f502a1cf0f425d32efe0a50f5dc',
                  path='docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json'),
             dict(commit='6c4709f85d0f28ce7c6957e9a70c85e378d8cad0', path='docs/research/d_pauli_ld2_b1_w_results_20261115/JOB.json'),
             dict(commit='cb034e547c444ea67a4f4f90a8534c13f4db4451',
                  path='docs/research/d_pauli_ld2_m9_longpoint_results_20261115/PREDICTIONS.json')]
CELL = dict(cell_id='Q8R660', q=8, R=660, T=.1 * 8, K=2)
COHORT = 'A_PAULI_LD3_CAL_Q8R660_N2_V1'
M12_ID = 'RMSE:95014c1c1c105ae09155753b7dfa3edd40ba8900bfc2efe72152e670b1337a28'
AXES = {'X': 'cosine', 'Y': 'sine'}
COMPILE_MAXIMUM = 4


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def read(ref):
    data = json.loads(blob(ref['commit'], ref['path']))
    for key in ref.get('selector', []):
        data = data[key]
    return data


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def same(a, b, label):
    if a != b:
        raise ValueError(label)


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def c2():
    """C2's m9 driver from its pinned blob: Progress, accelerated_observed, observe (unchanged)."""
    name = '_c2_m9_driver'
    if name not in sys.modules:
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(ROOT / 'scripts' / '_c2_run_pauli_ld2_m9_cb034e5.py')
        exec(compile(blob(C2_DRIVER['commit'], C2_DRIVER['path']), C2_DRIVER['commit'] + ':' + C2_DRIVER['path'], 'exec'),
             module.__dict__)
        sys.modules[name] = module
    return sys.modules[name]


def legacy():
    same(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PIN_ROOT, text=True).strip(), PIN,
         'Legacy checkout HEAD differs')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=PIN_ROOT, text=True).strip():
        raise ValueError('Legacy checkout is not clean')
    if str(PIN_ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(PIN_ROOT / 'scripts'))
    import run_pauli_b1_prices as B1
    same(Path(B1.__file__).resolve(), (PIN_ROOT / B1.CALLER).resolve(), 'Legacy module path differs')
    for p in (*B1.REPLAY_SOURCES, B1.A_PATH):
        same((PIN_ROOT / p).read_bytes(), blob(PIN, p), 'Pinned SOURCE differs: ' + p)
    return B1


def cell_request(P, spec, cell, lambda_P):
    """``run_pauli_b1_prices.request_for`` for a non-configuration cell (same keys/order/types)."""
    identity = dict(kind='NON_CONFIGURATION_PAULI_PRICE_CELL', binding=deepcopy(spec['binding']),
                    split=deepcopy(spec['split']), profile=deepcopy(spec['profile']), route=P.ROUTE,
                    lowering=P.LOWERING_ID, q=cell['q'], R=cell['R'], T=cell['T'], K=cell['K'])
    physical = dict(m=None, prefix=spec['L_D'], q=cell['q'], R=cell['R'], K=cell['K'], T=Fraction(cell['T']),
                    step_time=cell['T'] / cell['q'], lambda_P=lambda_P,
                    candidate_id='PAULI_PRICE_CELL:' + P.canonical_sha256(identity), physical_cell=identity,
                    normalization='EXISTING_FINITE_K_RULE_NOT_EVALUATED_IN_METADATA')
    physical['T'] = float(physical['T'])
    return physical


def old_seeds():
    seeds = set()
    meta = read(OLD_SEEDS[0])
    for d in meta['dictionaries'].values():
        seeds.update(s for c in d['cells'] for row in c['step_seeds'] for s in row)
    seeds.update(s for row in read(OLD_SEEDS[1])['step_seeds'] for s in row)
    seeds.update(s for row in read(OLD_SEEDS[2])['binding']['step_seeds'] for s in row)
    return seeds


def ld3_arrays():
    """Saved full-2^n operators: common blocks (4220f300) + LD3 head/tail/Pauli (50640b74c)."""
    import numpy as np
    common = np.load(io.BytesIO(blob(COMMON['commit'], COMMON['path'])), allow_pickle=False)
    ld3 = np.load(io.BytesIO(blob(P0_LD3['commit'], P0_LD3['directory'] + 'LD3_OPERATORS.npz')), allow_pickle=False)
    return {k: ld3[k] for k in ld3.files}, common['blocks']


def metadata():
    B1 = legacy(); P = B1.B0.connection()
    config = read(CONFIG); spec = config['spec']
    same(config['id'], CONFIG_ID, 'LD3 configuration ID differs')
    identity = spec['split']['tail_dictionary']
    dictionary = P.load_dictionary(PIN_ROOT, identity)
    lam = Fraction(spec['split']['lambda_R']['numerator'], spec['split']['lambda_R']['denominator'])
    same((spec['L_D'], dictionary['prefix'], Fraction(dictionary['lambda_P'])), (3, 3, lam), 'LD3 prefix/lambda_P differs')
    same(dictionary['saved_DF_scalar'], spec['split']['scalar'], 'LD3 saved s_R differs')
    # Rule check: the same construction reproduces D2's saved LD2 Q8R66 request exactly.
    ld2_spec = read(LD2_CONFIG); saved = read(LD2_Q8R66)
    ld2 = cell_request(P, ld2_spec, dict(q=8, R=66, T=.1 * 8, K=2), saved['lambda_P'])
    same(ld2, saved, 'Cell identity rule does not reproduce the saved LD2 Q8R66 request')
    request = cell_request(P, spec, CELL, dictionary['lambda_P'])
    same((request['q'], request['R'], request['T'], request['step_time']), (8, 660, 0.8, 0.1), 'Cell q/R/T differs')
    m12 = config['rounds'][12]
    same((m12['candidate']['id'], m12['q_PF'], m12['R_outer']), (M12_ID, 4096, 660), 'm12 per-step R differs')
    seeds = [[P.seed(request['candidate_id'], COHORT, i, outer, 0) for outer in range(CELL['q'])] for i in (0, 1)]
    flat = [s for row in seeds for s in row]
    previous = old_seeds()
    if len(set(flat)) != len(flat) or set(flat) & previous:
        raise ValueError('New or old cohort seed collision')
    return dict(schema='A_PAULI_LD3_CAL_BINDING_V1', cell_id=CELL['cell_id'], prefix=3, cohort=COHORT,
        path_indices=[0, 1], axes=['X', 'Y'], n_paths=2, compile_maximum=COMPILE_MAXIMUM, request=request,
        step_seeds=seeds, dictionary_identity=identity, configuration=dict(CONFIG, id=CONFIG_ID),
        total_view=TOTAL_VIEW, m12_canonical=dict(physical_id=M12_ID, q=4096, R=660,
            relation='SAME_PER_STEP_R_AS_m12_BUT_DIFFERENT_q_T_AND_ID_NOT_A_ROUND_RECEIPT'),
        operators=dict(common=COMMON, ld3=dict(commit=P0_LD3['commit'], path=P0_LD3['directory'] + 'LD3_OPERATORS.npz')),
        scalar=dict(value=P.total_scalar(dictionary, B1.B0.HAMILTONIAN_CONSTANT),
                    quantity=P.SCALAR_QUANTITY, application='ONCE_AT_TOTAL_T'),
        rule_check=dict(reproduced_saved_request=dict(LD2_Q8R66, candidate_id=saved['candidate_id']), equal=True),
        seed_checks=dict(new_step_seeds=len(flat), unique_new=True, old_cohort_overlap=False,
                         old_sources=OLD_SEEDS, old_seed_count=len(previous)),
        legacy_source=PIN, legacy_checkout=str(PIN_ROOT), c2_worker_pattern=C2_DRIVER, driver=SCRIPT, plan=PLAN)


def check_binding(actual, expected):
    same(actual, expected, 'LD3 binding/source/cohort/seed differs')


def freeze():
    m = metadata(); source = head()
    same((ROOT / SCRIPT).read_bytes(), blob(source, SCRIPT), 'Commit the driver before freezing the binding')
    value = dict(m, execution_source=dict(commit=source, path=SCRIPT, legacy_source=PIN),
                 created_UTC=datetime.now(timezone.utc).isoformat(), sampling=0, compiles=0,
                 status='BINDING_FIXED_BEFORE_SAMPLING')
    write(ROOT / BASE / 'BINDING.json', value)
    print(json.dumps(dict(candidate_id=m['request']['candidate_id'], seeds=m['step_seeds'], source=source), indent=1))


def input_preflight(scratch):
    """Provider-first child: restore H4 and prepare(ham, 3); input/bit-order gates only.

    No sampling, circuit build, simulation or compile. Records whether the native LD3 head and
    the extracted tail identity agree with the saved dictionary before any scientific step.
    """
    B1 = legacy()
    import h46_long_time_signal as signal
    signal.install(); B1.B0.provider_checks(); B1.B0.register_pauli_modules()
    import numpy, qiskit
    m = metadata(); P = B1.B0.connection()
    dictionary = P.load_dictionary(PIN_ROOT, m['dictionary_identity'])
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(scratch) / 'public', Path(scratch) / 'private'
    ctx = native.native_context('H4', (3,)); prep = ctx.preparations[3]
    operators, blocks = ld3_arrays()
    gates = P.bit_order_checks(dictionary, operators, blocks)
    comps, labels, n = P.components(dictionary)
    return dict(status='INPUT_PREFLIGHT_NO_SAMPLING_BUILD_OR_COMPILE', qiskit=qiskit.__version__, numpy=numpy.__version__,
                python=sys.version, executable=sys.executable,
                hamiltonian_constant_equal=ctx.ham.constant == B1.B0.HAMILTONIAN_CONSTANT,
                head_blocks=len(prep.deterministic_blocks), head_fragment_indices=list(prep.deterministic_fragment_indices),
                native_tail_identity=prep.extracted_identity_coefficient, saved_sR=dictionary['saved_DF_scalar'],
                native_identity_equals_saved_sR=prep.extracted_identity_coefficient == dictionary['saved_DF_scalar'],
                bit_order=gates, terms=len(comps), qubits=n, system_qubits=ctx.ham.n_qubits)


def check_result_path(path, binding, receipts=None, source=None):
    i = path['index']
    if i not in (0, 1):
        raise ValueError('Wrong path index')
    B1 = legacy()
    B1.receipts().check_path(path, request=binding['request'], dictionary=binding['dictionary_identity'],
                             cohort=COHORT, index=i, seeds=binding['step_seeds'][i])
    if path['result']['sampling'] != 'ORIGINAL_NORMAL_SAMPLER':
        raise ValueError('Independent normal sampler required')
    if receipts is not None:
        for a, key in AXES.items():
            receipt = receipts['p%d%s' % (i, a)]; r = path['result']
            same((receipt['axis'], receipt['index'], receipt['cohort'], receipt['request'], receipt['seeds'],
                  receipt['dictionary_identity']),
                 (a, i, COHORT, binding['request'], binding['step_seeds'][i], binding['dictionary_identity']),
                 'Individual receipt binding differs')
            same(receipt['execution_source'], source, 'Receipt execution SOURCE differs')
            same(receipt['record'], r['compile_receipts'][key], 'Original compiler record differs')
            same(receipt['record']['fingerprint'], r['fingerprints'][key]['fingerprint'], 'Fingerprint differs')
            same(receipt['record']['metrics'], r['prices'][key], 'Receipt metrics differ')
            same(receipt['record']['compiler'], binding['request']['physical_cell']['profile']['compiler'], 'Compiler differs')


def acquire(B1, native, ctx, dictionary, binding, index, operators, blocks, compile_cached, log):
    """One path through the unchanged legacy acquire_path/pilot_core, observed and accelerated as C2."""
    P = B1.B0.connection()
    with log.stage('acquire_path', index=index), c2().accelerated_observed(P, log) as stats:
        r = B1.acquire_path(native, ctx, dictionary, binding['request'], cohort=COHORT, index=index,
                            seeds=binding['step_seeds'][index], operators=operators, blocks=blocks,
                            compile_cached=compile_cached)
    return r, stats


def worker(job_path):
    job = json.loads(Path(job_path).read_text()); i = job['index']; folder = Path(job['private'])
    log = c2().Progress(folder / 'PROGRESS.jsonl')
    try:
        frozen = read(job['binding_reference'])
        check_binding(job['binding'], {k: v for k, v in frozen.items()
                                       if k not in ('execution_source', 'created_UTC', 'sampling', 'compiles', 'status')})
        same(job['driver_commit'], frozen['execution_source']['commit'], 'Frozen driver SOURCE differs')
        same(blob(job['driver_commit'], SCRIPT), (ROOT / SCRIPT).read_bytes(), 'Running driver bytes differ')
        B1 = legacy()
        import h46_long_time_signal as signal
        with log.stage('install_provider'):
            signal.install(); B1.B0.provider_checks(); B1.B0.register_pauli_modules()
            import qiskit
            same(qiskit.__version__, '1.3.0', 'Qiskit differs')
            P = B1.B0.connection(); dictionary = P.load_dictionary(PIN_ROOT, job['binding']['dictionary_identity'])
            import h46_long_time_native as native
            native.PUBLIC, native.PRIVATE = folder / 'native_public', folder
            ctx = native.native_context('H4', (3,))
            same(ctx.ham.constant, B1.B0.HAMILTONIAN_CONSTANT, 'Hamiltonian constant differs')
            operators, blocks = ld3_arrays()
        compile_calls = []
        def compiler(wrapper, fp, representative):
            axis = 'X' if not compile_calls else 'Y'
            if len(compile_calls) >= 2:
                raise ValueError('Two compile maximum per path')
            compile_calls.append(axis)
            with log.stage('compile_' + axis, instructions=len(wrapper.data), fingerprint=fp):
                return native.compile_cached(wrapper, fp, representative)
        r, stats = acquire(B1, native, ctx, dictionary, job['binding'], i, operators, blocks, compiler, log)
        path = dict(index=i, cohort=COHORT, request=job['binding']['request'],
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
    except BaseException as exc:
        log('worker', 'STOP', error=repr(exc))
        write(folder / 'FAILURE.json', dict(status='STOP', index=i, error=traceback.format_exc(),
              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, automatic_retry=False))
        return 2


def execute(binding_commit):
    reference = dict(commit=binding_commit, path=BASE + '/BINDING.json')
    frozen = read(reference); m = metadata()
    check_binding({k: v for k, v in frozen.items()
                   if k not in ('execution_source', 'created_UTC', 'sampling', 'compiles', 'status')}, m)
    source = frozen['execution_source']
    same((ROOT / SCRIPT).read_bytes(), blob(source['commit'], SCRIPT), 'Frozen SOURCE changed')
    remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', BRANCH], cwd=ROOT, text=True).split()[0]
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', binding_commit, remote], cwd=ROOT)
    same(blob(remote, reference['path']), blob(binding_commit, reference['path']), 'Published binding differs')
    out = ROOT / BASE / 'run01'
    if out.exists() or PRIVATE.exists():
        raise FileExistsError('Fresh scientific output required; no automatic retry')
    out.mkdir(); PRIVATE.mkdir()
    log = c2().Progress(out / 'PROGRESS.jsonl')
    execution_source = dict(commit=source['commit'], caller=SCRIPT, prefix=3, kind='LD3_CAL_Q8R660',
        cell=CELL['cell_id'], cohort=COHORT, step_seeds=m['step_seeds'], result_path=BASE + '/run01/RESULT.json',
        legacy_source=PIN)
    write(out / 'EXECUTION.json', dict(binding=reference, execution_source=execution_source, python=sys.version,
          executable=sys.executable, started_UTC=datetime.now(timezone.utc).isoformat(),
          compile_maximum=COMPILE_MAXIMUM, automatic_retry=False))
    env = os.environ.copy(); env['OPENBLAS_NUM_THREADS'] = '1'; env['OMP_NUM_THREADS'] = '1'
    processes = []
    for i in (0, 1):
        folder = PRIVATE / ('path' + str(i)); folder.mkdir()
        job = dict(index=i, private=str(folder), binding=m, execution_source=execution_source,
                   driver_commit=source['commit'], binding_reference=reference)
        write(folder / 'JOB.json', job)
        stream = (folder / 'CONSOLE.txt').open('x')
        p = subprocess.Popen([sys.executable, '-B', str(ROOT / SCRIPT), '--worker', str(folder / 'JOB.json')],
                             cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        processes.append((i, p, stream)); log('worker', 'START', index=i, pid_child=p.pid)
    states = c2().observe(processes, log)
    write(out / 'PROCESS_EXITS.json', list(states.values()))
    ok = all(s['exit_code'] == 0 for s in states.values())
    summaries, receipts, manifests = [], {}, []
    for i in (0, 1):
        folder = PRIVATE / ('path' + str(i))
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
            manifests.append(dict(index=i, path=BASE + '/run01/' + name, bytes=len(data),
                                  sha256=hashlib.sha256(data).hexdigest(), encoding='GZIP_JSON', selector=[]))
    ok = ok and len(summaries) == 2 and len(receipts) == 4
    write(out / 'RAW_PATHS.json', manifests); write(out / 'COMPILE_RECEIPTS.json', receipts)
    write(out / 'RESULT.json', dict(status='FINITE_LD3_CAL_PATHS_COMPLETE' if ok else 'STOP_LD3_CAL', source=execution_source,
          binding=reference, summaries=summaries, raw_paths=manifests, compiles=len(receipts),
          automatic_retry=False, mean=None, fit=None, price=None, N=None, G=None, mandatory_stop=True))
    log('acquisition', 'DONE' if ok else 'STOP', valid_paths=len(summaries))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata', action='store_true'); ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--execute'); ap.add_argument('--worker'); ap.add_argument('--input-preflight')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    if a.input_preflight:
        print(json.dumps(input_preflight(a.input_preflight), indent=1, default=str)); return 0
    if a.freeze:
        freeze(); return 0
    if a.execute:
        return execute(a.execute)
    print(json.dumps(metadata(), indent=2)); return 0


if __name__ == '__main__':
    sys.exit(main())
