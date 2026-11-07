"""Dormant one-campaign C-S activation. Real execution needs separate permission.

No new law, operator engine, synthesis provider, price model or job manager.
The only matrices are the adopted two-dimensional Pauli eigenspaces, evaluated
by mpmath interval operations and the existing operator_target callback API.
"""
from dataclasses import asdict, is_dataclass
from fractions import Fraction as F
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import mpmath as mp

from . import cs_v0 as core, cs_v0_native as native
from .parent_allocation_native import operator_target
from .df_correction_comparison.intervals import record, from_record
from .rte_development.evidence import atomic_json, dumps, digest
from .same_position_residual import plan_sequence

DOC = Path('docs/research/cs_v0_repair_20261108')
OLD = Path('docs/research/cs_v0_technical_preparation_20261108')
OUTPUT = Path('artifacts/cs_v0_science/2026-11-08/campaign_run02')
BRANCH = 'codex/cs-v0-repair-20261108'
P1 = Path('artifacts/o2_a1_new_target/2026-11-06/n1')
PU = Path('artifacts/o2_a1_new_target/2026-11-06/engine_qualification_run01')
EPS = F(1, 10**6)
FINITE_CAP = F(1, 10**12)
LOCAL_CAP = F(3, 10**6)


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value  # existing dumps preserves Fraction as an exact string


def write(path, value):
    value = plain(value)
    # Size to the actual finite record; no arbitrary new storage/resource cap.
    return atomic_json(path, value, byte_cap=len((dumps(value)+'\n').encode()))


def resources():
    # Reuse the existing read-only tick parser, not its old waiting/worker policy.
    from .a1_n3_resource_admission import cpu_ticks, available_from_ticks
    cpus = set(os.sched_getaffinity(0))
    before = cpu_ticks(Path('/proc/stat').read_text())
    time.sleep(.25)  # observation window, not a CPU/runtime allocation cap
    after = cpu_ticks(Path('/proc/stat').read_text())
    _, idle = available_from_ticks(before, after, cpus)
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        if k in ('MemTotal', 'MemAvailable'):
            memory[k] = v.strip()
    return dict(cpu_count=os.cpu_count(), available_affinity=len(os.sched_getaffinity(0)),
                load_average=os.getloadavg(), memory=memory,
                cpu_pressure=Path('/proc/pressure/cpu').read_text(),
                memory_pressure=Path('/proc/pressure/memory').read_text(),
                observed_idle_core_equivalents=idle['idle_core_equivalents_exact'],
                peak_self_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def preflight(root, source):
    """Future Stage A. Reuses the old static input/runtime preflight verbatim."""
    root = Path(root)
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('full frozen SOURCE SHA required')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    if git('rev-parse', 'HEAD') != source or git('diff', '--name-only', source, '--', '.'):
        raise ValueError('SOURCE/worktree mismatch')
    if git('ls-files', '--others', '--exclude-standard', '--', 'src', 'scripts', 'tests'):
        raise ValueError('untracked execution code')
    remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/'+BRANCH).split()[0]
    if remote != source:
        raise ValueError('remote SOURCE mismatch')
    lock = json.loads((root/DOC/'SOURCE.json').read_text())
    for p, sha in lock['activation_source_sha256'].items():
        if digest(root/p) != sha:
            raise ValueError('activation SOURCE/profile mismatch: '+p)
    spec = importlib.util.spec_from_file_location('cs_v0_saved_preflight',
        root/'scripts/tracks/algorithm_codesign/cs_v0_preflight.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    checked = old.inspect(root, source_sha256=lock['effective_base_source_sha256'])
    observed = resources()
    if int(observed['memory']['MemAvailable'].split()[0]) <= 0:
        raise RuntimeError('no available memory; no resource changes permitted')
    if F(observed['observed_idle_core_equivalents']) < 1:
        raise RuntimeError('no spare core for the current sequential call; no resource changes permitted')
    return dict(old_preflight=checked, remote_source=remote,
                source_tree=git('rev-parse', source+'^{tree}'), resources=observed,
                fixed_inputs=json.loads((root/OLD/'SOURCE_AND_CHECKS.json').read_text())['fixed_inputs'],
                activation_lock_sha256=digest(root/DOC/'SOURCE.json'))


def bind_fixed(root):
    load = lambda p: json.loads((Path(root)/p).read_text())
    config = load(P1/'CONFIG.json')
    return core.bind_slots(load(P1/'PREPARATION.json'), load(P1/'LAW.json'),
        load(PU/'UNIFORM_ERROR.json'),
        lambda component: core.resolve_r_angle(component, config['distribution']),
        n=8, policy=load(P1/'POLICY.json'))


def matrix_record(matrix):
    return [[dict(real=record(z.real), imag=record(z.imag)) for z in row]
            for row in matrix.tolist()]


def local_comparison(site, coarse):
    residual = site.angle+coarse.scaled(-1)
    if coarse+residual != site.angle:
        raise ValueError('exact target/coarse/residual binding failed')
    with native.precision(100):
        matrices = {}
        for name, angle in (('target', site.angle), ('coarse', coarse), ('residual', residual)):
            theta = native.Expression.linear(angle).value(mp.iv)
            # Existing strict-RZ target diagonal, using the standard iv kernel.
            matrices[name] = mp.iv.diag([mp.iv.exp(-mp.iv.j*theta/2), mp.iv.exp(mp.iv.j*theta/2)])
        diff = matrices['residual']*matrices['coarse']-matrices['target']
        for row in diff.tolist():
            for z in row:
                for part in (z.real, z.imag):
                    lo, hi = native.enclosure(part)
                    if not lo <= 0 <= hi:
                        raise ArithmeticError('exact local composition outside interval enclosure')
        return dict(operators={k: matrix_record(v) for k, v in matrices.items()},
                    composition_difference=matrix_record(diff), exact_angle_identity=True,
                    scope='Pauli +/-1 eigenspaces only; no H4 state or full operator')


def finite_result(residual, comparison):
    table = core.residual_table(residual)
    if table['finite_error_upper'] > FINITE_CAP or len(table['events']) > 2:
        raise ValueError('finite error/event cap exceeded')
    with native.precision(100):
        identity = mp.iv.eye(2); pauli = mp.iv.diag([1, -1]); mean = mp.iv.zeros(2)
        for event in table['events']:
            if event['word']:
                raise ValueError('ordinary single-involution word did not reduce; STOP')
            def rotation(_, ratio, sign):
                theta = native.Expression('atan', ratio, 2*sign).value(mp.iv)
                return mp.iv.diag([mp.iv.exp(-mp.iv.j*theta/2), mp.iv.exp(mp.iv.j*theta/2)])
            unitary = operator_target(event, identity=lambda: identity,
                multiply=lambda a, b: a*b, involution=lambda _: pauli,
                rotation=rotation, scale=lambda a, b: a*b)
            mean += native.ivf(event['coefficient'])*unitary
        if table['exact_identity']:
            mean = identity
        exact = mp.iv.matrix([[mp.iv.mpc(from_record(z['real']), from_record(z['imag']))
            for z in row] for row in comparison['operators']['residual']])
        # All these matrices are diagonal: spectral norm is max diagonal abs.
        diagnostic = max(native.enclosure(abs(mean[i, i]-exact[i, i]))[1] for i in range(2))
        if not table['exact_identity'] and diagnostic > table['finite_error_upper']:
            raise ArithmeticError('finite mean interval exceeds adopted analytic bound')
        table.update(operator_mean=matrix_record(mean), diagnostic_operator_error_upper=diagnostic,
                     diagnostic_is_not_zero_error_claim=bool(table['exact_identity']))
    return table


def price_group(site, coarse_row, target_row, event_rows):
    """Existing literal native provider, one local price GROUP per selected spec."""
    def priced(item, phase=0):
        row, req, sign = item
        out = native.local_native(site.generator, row, req, sign, phase_i_power=phase)
        counts = dict(out['counts'])
        counts['Clifford_1Q'] = sum(g[0] in ('H', 'X', 'Y', 'Z', 'S', 's') for g in out['time_gates'])
        counts['scalar_W'] = sum(g[0] in ('W', 'w') for g in out['time_gates'])
        counts['serial_depth'] = out['serial_depth']
        return dict(native=out, counts=counts)
    target, coarse = priced(target_row), priced(coarse_row)
    branches = [dict(proposal=e['proposal'], result=priced(item, e['phase_i_power']))
                for e, item in event_rows]
    metrics = ('T', 'Tdagger_count', 'CX', 'Clifford_1Q', '1Q', 'scalar_W', 'serial_depth')
    candidate = {k: coarse['counts'][k]+sum(b['proposal']*b['result']['counts'][k] for b in branches)
                 for k in metrics}
    return dict(S0=target, coarse=coarse, branches=branches,
                S1_local_expected_counts=candidate,
                S2_local_expected_counts=candidate if site.role == 'D_DIAGONAL' else {k: target['counts'][k] for k in metrics},
                scope='selected primitive; additive literal price; full price UNKNOWN')


def run(root, source):
    """Future explicitly authorized execution, exactly one fixed output attempt.

    Never invoked by import/default CLI/tests on real input. Fresh output is a
    scientific attempt namespace, not an approval certificate or grant.
    """
    root = Path(root); out = root/OUTPUT
    out.mkdir(parents=True, exist_ok=False)  # never touch an existing attempt
    (out/'cache').mkdir(); start = time.monotonic(); cpu_start = time.process_time(); journal_index = 0
    saved = {}; cache = {}; backend = dict(started=0, returned=0)
    phase = 'A'; stop_written = False
    def journal(value):
        nonlocal journal_index
        write(out/'journal'/f'{journal_index:04d}.json', value); journal_index += 1
    calls = core.Calls(journal)
    def finish_stop(status, reason=None):
        nonlocal stop_written
        write(out/'STOP.json', dict(status=status, reason=reason, stage=phase,
            science_retry=0, additional_campaign=0, mandatory_STOP=True))
        stop_written = True
    def termination(signum, frame):
        raise InterruptedError('own campaign signal '+str(signum))
    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, termination)
        write(out/'SOURCE.json', dict(frozen_source=source, campaign_output=str(OUTPUT),
            previous_output='artifacts/cs_v0_science/2026-11-08/campaign_run01',
            technical_reexecution_after_schema_fix=True,
            technical_base='606c40d7809a5941a76c54d0bc2dd8e1a5bc1818',
            contract=str(DOC/'V0_EXECUTION_CONTRACT.md'), input_manifest=str(OLD/'SOURCE_AND_CHECKS.json'),
            native_profile=str(OLD/'NATIVE_PROFILE.json'), scope='D1/R1 joint factor0 only',
            strict_precision=EPS, finite_cap=FINITE_CAP, local_cap=LOCAL_CAP,
            call_caps=core.Calls.CAPS))
        checked = preflight(root, source); write(out/'PREFLIGHT.json', checked)
        profile = json.loads((root/OLD/'NATIVE_PROFILE.json').read_text())
        write(out/'NATIVE_PROFILE.json', profile)
        phase = 'B'
        bound = calls.invoke('binding', 'D1-R1', lambda: bind_fixed(root))
        write(out/'BINDING.json', bound)
        sites = bound['sites']
        if len(sites) != 2 or [s.role for s in sites] != ['D_DIAGONAL', 'R_DIAGONAL'] or len({s.site_id for s in sites}) != 2:
            raise ValueError('fixed D1/R1 order/scope mismatch')
        if len(bound['fixed_control_partners']) != 2 or any(p.role != 'FIXED' for p in bound['fixed_control_partners']):
            raise ValueError('FIXED partner scope mismatch')
        coarses = {s.site_id: core.coarse(s.angle) for s in sites}
        write(out/'LOCAL_PLANS.json', {a: plan_sequence(sites, coarses, arm=a) for a in ('S0', 'S1', 'S2')})
        phase = 'C'; groups = []; requests = {}
        for i, site in enumerate(sites):
            c = coarses[site.site_id]; residual = site.angle+c.scaled(-1)
            compare = calls.invoke('local_comparison', site.site_id, lambda: local_comparison(site, c))
            write(out/f'spec{i}_operators.json', compare)
            table = calls.invoke('residual_table', site.site_id, lambda: finite_result(residual, compare))
            write(out/f'spec{i}_finite.json', table)
            if sum(len(g[3]['events']) for g in groups)+len(table['events']) > 4:
                raise ValueError('total residual event cap exceeded')
            exprs = [native.Expression.linear(site.angle), native.Expression.linear(c)]
            exprs += [native.Expression('atan', e['ratio'], 2*e['rotation_sign']) for e in table['events']]
            bindings = [native.request(e, EPS, profile['tool']) for e in exprs]
            for req, sign in bindings:
                if req['key'] in requests and requests[req['key']] != req:
                    raise ValueError('duplicate key has different spec')
                requests[req['key']] = req
            groups.append((site, c, compare, table, bindings))
        if len(requests) > 8:
            raise ValueError('positive key cap exceeded before acquisition')
        write(out/'REQUESTS.json', list(requests.values()))
        for key, req in requests.items():
            def acquire_one():
                fast = native.exact_strict(native.Expression(**req['spec']['angle']))
                if fast is not None:
                    write(out/'cache'/(key+'.raw.json'), dict(sequence=fast, request=req, route='exact_fast_path'))
                def observed_synth(expr, epsilon, options):
                    backend['started'] += 1
                    journal(dict(event='BACKEND_START', key=key))
                    sequence = native.backend(expr, epsilon, options)
                    backend['returned'] += 1
                    # Preserve returned sequence BEFORE strict qualification.
                    write(out/'cache'/(key+'.raw.json'), dict(sequence=sequence, request=req, route='backend_return'))
                    journal(dict(event='BACKEND_RETURN', key=key))
                    return sequence
                row = native.acquire(req, synth=observed_synth)
                native.checked(row, req)
                write(out/'cache'/(key+'.json'), row)
                return row
            cache[key] = calls.invoke('positive_key', key, acquire_one)
        for i, (site, coarse, compare, table, bindings) in enumerate(groups):
            items = [(cache[req['key']], req, sign) for req, sign in bindings]
            errors = [F(item[0]['operator_error_upper']) for item in items]
            if any(e > EPS for e in errors):
                raise ValueError('native primitive cap exceeded')
            bounds = core.local_bounds(table, errors[1], errors[2:])
            write(out/f'spec{i}_errors.json', bounds)
            if bounds['operator_mean_error_upper'] > LOCAL_CAP:
                raise ValueError('local error cap exceeded')
            price = calls.invoke('local_price', site.site_id,
                lambda: price_group(site, items[1], items[0], list(zip(table['events'], items[2:]))))
            write(out/f'spec{i}_price.json', price)
            saved[site.site_id] = dict(role=site.role, finite_cap_pass=True, native_cap_pass=True,
                local_cap_pass=True, S0_strict_error_upper=errors[0], local_error=bounds)
        phase = 'D'
        write(out/'RESULT.json', dict(status='LOCAL_V0_COMPLETE', specs=saved,
            full_price='UNKNOWN', N_G='NOT_EVALUATED', resource_improvement='NOT_CONCLUDED'))
        finish_stop('MANDATORY_STOP_COMPLETE')
        return out
    except BaseException as exc:
        if not stop_written:
            finish_stop('MANDATORY_STOP_INCOMPLETE', type(exc).__name__+': '+str(exc))
        write(out/'RESULT.json', dict(status='LOCAL_V0_INCOMPLETE', completed_specs=saved,
            qualified_keys=list(cache), missing_local_results='UNKNOWN; see durable journal and saved spec files',
            missing_native_price='UNKNOWN', full_price='UNKNOWN', N_G='NOT_EVALUATED'))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        write(out/'CALL_COUNTS.json', dict(calls=[dict(kind=k, identity=i, state=s) for (k, i), s in calls.used.items()],
            backend=backend, qualified_keys=len(cache), completed_specs=len(saved),
            forbidden_science_calls=0, retries=0))
        write(out/'CLASSICAL_COST.json', dict(wall_seconds=time.monotonic()-start,
            process_CPU_seconds=time.process_time()-cpu_start, resources=resources(),
            cache='fresh campaign only', missing_call_returns='UNKNOWN'))
