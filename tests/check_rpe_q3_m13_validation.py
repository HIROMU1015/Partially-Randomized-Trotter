"""Isolated M=13 validation against the fixed RPE reference, artificial inputs only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXED = 'cbc471c913072de45c25a0ed22d51dff8a9b7034'
TAU = Fraction('0.00159360010199040')**2
H = 0.1
M = 13
Q = 1 << M


@contextmanager
def proposed_limit(module, old, new, *names):
    """Test a one-line capability change without editing the fixed source view."""
    path = Path(module.__file__)
    source = path.read_text()
    assert source.count(old) == 1, path
    candidate = dict(vars(module))
    exec(compile(source.replace(old, new), str(path), 'exec'), candidate)
    with patch.multiple(module, **{name: candidate[name] for name in names}):
        yield


def scalar_branches(principal, q):
    from trotterlib.rpe_four_round_phase_validation import _wrap_phase

    return np.asarray([_wrap_phase((principal + 2.0 * math.pi * branch) / q)
                       for branch in range(q)])


def compare_terminal_branch(total):
    from trotterlib.rpe_four_round_phase_validation import _phase_distance

    rows = []
    for principal in (0., -.9, 1.3, -math.pi, np.nextafter(math.pi, 0.)):
        branches = scalar_branches(principal, Q)
        order = np.argsort(branches)
        for index in (0, 1, Q//4, Q//2, 3*Q//4, Q-2, Q-1):
            left = branches[order[index]]
            right = branches[order[(index+1) % Q]]
            middle = (left + right + (2*math.pi if index == Q-1 else 0))/2
            for previous in (left, np.nextafter(middle, -np.inf), middle, np.nextafter(middle, np.inf)):
                distances = _phase_distance(branches, previous)
                branch = int(np.argmin(distances))
                ties = distances == distances[branch]
                expected = float(branches[branch])
                value, actual_branch, _, _, exact_tie = total.F._nearest(np.array(principal), np.array(previous), Q)
                result, _ = total.step(np.array(principal), np.array(previous), Q, np.array(False))
                assert int(actual_branch) == branch
                assert float(value).hex() == float(result).hex() == expected.hex()
                assert (-float(result)/H).hex() == (-expected/H).hex()
                assert bool(exact_tie) == (np.count_nonzero(ties) > 1)
                rows.append(dict(principal=float(principal).hex(), previous=float(previous).hex(),
                                 branch=branch, tie=bool(exact_tie), phase=expected.hex(), energy=(-expected/H).hex()))
    assert any(row['tie'] for row in rows)
    for previous in (-math.pi, -0.0, 0.0, np.nextafter(math.pi, 0.)):
        value, tie = total.step(np.array(0.), np.array(previous), Q, np.array(True))
        assert float(value).hex() == float(previous).hex() and not bool(tie)
    return dict(cases=len(rows), exact_ties=sum(row['tie'] for row in rows), zero_hold_cases=4,
                representative_tie=next(row for row in rows if row['tie']))


def exact_histories(rows, h):
    """Independent scalar full-branch decoding and exact binomial probabilities."""
    from trotterlib.rpe_four_round_phase_validation import _phase_distance, _wrap_phase

    axes = []
    branch_cache = {}
    for row in rows:
        support = []
        for axis, coordinate in (('X', 'real'), ('Y', 'imag')):
            n = row[f'N_{axis}']
            p = (1 + Fraction(row['raw'][coordinate])) / 2
            counts = (n,) if p == 1 else (0,) if p == 0 else range(n + 1)
            support.append([(float(Fraction(2*k-n, n)), math.comb(n, k)*p**k*(1-p)**(n-k)) for k in counts])
        axes.append(list(product(*support)))
    states = [((), Fraction(1), 0.)]
    for m, axis in enumerate(axes):
        next_states = []
        for history, weight, previous in states:
            for (x, wx), (y, wy) in axis:
                if x == y == 0:
                    phase = previous
                else:
                    principal = _wrap_phase(math.atan2(y, x))
                    if m:
                        key = (m, principal.hex())
                        if key not in branch_cache:
                            branch_cache[key] = scalar_branches(principal, 1 << m)
                        branches = branch_cache[key]
                        phase = float(branches[np.argmin(_phase_distance(branches, previous))])
                    else:
                        phase = float(principal)
                next_states.append((history + ((x, y),), weight*wx*wy, phase))
        states = next_states
    assert sum(weight for _, weight, _ in states) == 1
    return [(history, weight, -phase/h) for history, weight, phase in states]


def mse(histories, target):
    return sum(weight*(Fraction(energy)-Fraction(target))**2 for _, weight, energy in histories)


def interval(lo, hi, reference):
    assert lo <= reference <= hi, (float(lo), float(reference), float(hi))
    verdict = 'PASS' if hi <= TAU else 'FAIL' if lo > TAU else 'INCONCLUSIVE'
    return dict(reference=str(reference), lower=str(lo), upper=str(hi), verdict=verdict,
                display=[float(reference), float(lo), float(hi)])


def normal_case(toys, M, final_y, target=0.):
    raws = [(1., 1.)]*M + [(1., final_y)]
    args = toys.make(vector=(1,)*M+(2,), raws=raws, target=target)
    args['options']['q3_limit'] = dict(source=toys.MOCK, scope='MODEL_MSE_FILTER_ONLY',
                                      MSE_upper=toys.a.exact_value(TAU))
    rows = [dict(shot, raw=dict(real=x, imag=y)) for shot, (x, y) in zip(args['allocations'][0]['shots'], raws)]
    return args, rows


def check_normal(toys):
    def check(args, reference):
        result = toys.evaluate(args)['deterministic_risk']
        dp = result['bounds']['total_state_dp']
        assert [r['m'] for r in dp['rounds']] == list(range(args['config']['spec']['M'] + 1))
        assert toys.X(dp['MSE_exact']) == reference
        return interval(toys.X(result['MSE_lower']), toys.X(result['MSE_upper']), reference)

    args, rows = normal_case(toys, M, 0.)
    h = float(toys.X(args['config']['spec']['h']))
    histories = exact_histories(rows, h)
    assert len(histories) == 3 and len({energy.hex() for _, _, energy in histories}) == 3
    assert toys.evaluate(args)['partial_decoder_risk']['reason'] == 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'
    archive = {}
    for m in (11, 12):
        case, data = normal_case(toys, m, 0.)
        archive[m] = (case, toys.evaluate(case), mse(exact_histories(data, h), 0.))
    with proposed_limit(toys.D, "if prepared['M'] > 12:", "if prepared['M'] > 13:", '_bound'):
        checked = {}
        target = histories[1][2]
        for final_y, delta in ((0., 0.), (1., 0.01), (-1., 0.)):
            case, data = normal_case(toys, M, final_y, target+delta)
            reference = mse(exact_histories(data, h), target+delta)
            checked[f'final_y={final_y},target_offset={delta}'] = check(case, reference)
        assert {v['verdict'] for v in checked.values()} == {'PASS', 'FAIL'}
        for case, before, reference in archive.values():
            assert toys.evaluate(case) == before
            check(case, reference)
        beyond, _ = normal_case(toys, 14, 0.)
        assert toys.evaluate(beyond)['partial_decoder_risk']['reason'] == 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'
    return dict(M13=checked, regression_M=[11, 12], final_energies=[energy.hex() for _, _, energy in histories])


def check_large(toys, adapter):
    risk, total = toys.D, toys.R
    kernel = adapter['finite_support_kernel']()
    args = dict(h=H, probability_rule=risk.EXACT_AFFINE, threshold=TAU, risk=risk, total=total)
    rows = [dict(m=m, N_X=1, N_Y=1, raw=dict(real=1., imag=1.)) for m in range(M)]
    rows.append(dict(m=M, N_X=128, N_Y=128, raw=dict(real=1., imag=0.)))
    histories = exact_histories(rows, H)
    assert len(histories) == 129 and len({energy.hex() for _, _, energy in histories}) == 129
    anchor = histories[64][2]
    assert not kernel.supports(rows)
    archive = {m: kernel.evaluate(rows[:m]+[dict(rows[-1], m=m)], target=anchor, **args) for m in (11, 12)}
    with proposed_limit(kernel, 'len(shots) <= 13', 'len(shots) <= 14', 'supports'):
        # Evidence summarization has an independent length check.
        with proposed_limit(kernel, 'expected_rounds <= 13', 'expected_rounds <= 14', 'summarize'):
            # The original evaluator calls summarize through its module globals.
            cases = {}
            for label, target, cells in (('default', anchor, 16384), ('fail', anchor+0.01, 16384),
                                         ('forced_omission', anchor, 1)):
                kwargs = dict(args, target=target, maximum_round_cells=cells)
                first = kernel.evaluate(rows, **kwargs)
                refined = kernel.refine(rows, **kwargs)
                combined = kernel.combine([first['certificate'], refined['certificate']], TAU, expected_rounds=14)
                reference = mse(histories, target)
                cases[label] = {}
                for step, result in (('first', first), ('refined', refined), ('combined', combined)):
                    lo = kernel._decode(result['MSE_lower'])
                    hi = kernel._decode(result['MSE_upper'])
                    cases[label][step] = interval(lo, hi, reference)
                f = first['certificate']['rounds'][-1]
                assert f['m'] == M and f['represented_cells'] > 0
                omitted = 1-kernel._decode(f['window_probability'])
                retained = kernel._decode(refined['certificate']['rounds'][-1]['retained_probability'])
                assert omitted > 0 and retained < 1-omitted
                cases[label]['omitted_probability'] = str(omitted)
                cases[label]['rounding_loss'] = str(1-omitted-retained)
            for m, original in archive.items():
                assert kernel.evaluate(rows[:m]+[dict(rows[-1], m=m)], target=anchor, **args) == original
            assert not kernel.supports(rows+[dict(rows[-1], m=14)])
    assert cases['default']['combined']['verdict'] == 'PASS'
    assert cases['fail']['combined']['verdict'] == 'FAIL'
    assert cases['forced_omission']['first']['verdict'] == 'INCONCLUSIVE'
    return dict(M13=cases, regression_M=[11, 12], terminal_histories=len(histories))


def check_adapter(toys, adapter):
    # Use the ordinary CLI and real fixed intake. Replace the locator resolution
    # with artificial packets because production route locators require H4 evidence.
    kernel = adapter['finite_support_kernel']()
    path = Path(adapter['__file__'])
    source = path.read_text()
    changes = (('elif M is not None and M > 12:', 'elif M is not None and M > 13:'),
               ("request['origin'].get('M') in (11, 12)", "request['origin'].get('M') in (11, 12, 13)"),
               ("prepared['M'] >= 13", "prepared['M'] > 13"))
    for old, new in changes:
        assert source.count(old) == 1
        source = source.replace(old, new)
    candidate = dict(__name__='m13_candidate_adapter', __file__=str(path))
    exec(compile(source, str(path), 'exec'), candidate)

    with TemporaryDirectory(prefix='rpe-q3-m13-artificial-') as tmp:
        root = Path(tmp)

        def launch(main, request, out, mode):
            inputs = root/'request.json'
            inputs.write_text(json.dumps(request))
            with patch.object(sys, 'argv', [str(path), '--request', str(inputs), '--output', str(out), '--mode', mode]):
                main()
            name = {'dry-check': 'DRY_CHECK', 'execute': 'OUTCOME', 'collect': 'COLLECTED_OUTCOME'}[mode]
            return json.loads((out/f'{name}.json').read_text())

        def job(method, kind, shot_count):
            shots = [1]*M+[shot_count]
            return dict(request_id=f'artificial-{method}', configuration_id='artificial', method=method,
                        route=dict(kind=kind, view=str(root), view_commit='artificial'),
                        origin=dict(kind='RPE_SCALED_SHOTS_V1', M=M), N_X=shots, N_Y=shots,
                        execution_profile=dict(large_shot_bounds=True))

        held = launch(adapter['main'], job('FULL_PF', 'pf_b_pf6', 128), root/'original', 'dry-check')
        assert held['reason'] == 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'
        checks = {}
        with proposed_limit(kernel, 'len(shots) <= 13', 'len(shots) <= 14', 'supports'), \
             proposed_limit(kernel, 'expected_rounds <= 13', 'expected_rounds <= 14', 'summarize'), \
             patch.dict(candidate, finite_support_kernel=lambda: kernel):
            for method, kind in (('FULL_PF', 'pf_b_pf6'), ('TRUNCATED_PF', 'pf_b_pf6'),
                                 ('PR_DF', 'prdf_d2_common_entry')):
                checks[method] = {}
                for n, final_y, mode in ((2, 1., 'dry-check'), (128, 1., 'execute'), (128, -1., 'execute')):
                    target = 0.0001
                    args, rows = normal_case(toys, M, final_y, target)
                    args['allocations'][0]['shots'][-1].update(N_X=n, N_Y=n)
                    rows[-1].update(N_X=n, N_Y=n)
                    cfg, packet, allocation = (args[k] for k in ('config', 'signal_packet', 'allocations'))
                    cfg_id = cfg['id']
                    request = dict(job(method, kind, n), configuration_id=cfg_id)
                    binding = dict(id='artificial', original_configuration=cfg,
                                   total_configuration=dict(id='artificial-total'))
                    case = dict(label=request['request_id'], allocation=allocation[0],
                                evaluation_target=args['evaluation_target'], signal_packet='ARTIFICIAL')
                    inputs = {k: args[k] for k in ('statistical_contract', 'options')}
                    inputs['options']['model_loss_contract']['scope'] = 'ARTIFICIAL'
                    inputs.update(template_source='ARTIFICIAL', adopted_evaluation=dict(
                        estimator=toys.T.ESTIMATOR, target='SAVED_DF10_SOLVER_ENERGY_CONDITIONAL_PROXY',
                        loss='ALL_EVENT_SQUARED_ENERGY_ERROR', scope='MODEL_MSE_FILTER_ONLY',
                        MSE_upper=toys.a.exact_value(TAU)))
                    route = SimpleNamespace(runtime=None, bounds=candidate['count_bounds'],
                        binding=SimpleNamespace(Q=toys.R, C=SimpleNamespace(
                            interfaces=lambda _: SimpleNamespace(pipeline=args['integration']))),
                        binding_root=root, metadata=lambda _: (binding, dict(signal_packet=dict(origin_signal_packet=packet))))
                    out = root/f'{method}-{n}-{final_y}'
                    with patch.dict(candidate, build=lambda *a: (case, inputs), route=lambda *a: route):
                        result = launch(candidate['main'], request, out, mode)
                        if mode == 'dry-check':
                            assert result['status'] == 'DRY_CHECK_PASS'
                            checks[method]['ordinary_dry_check'] = result['status']
                            continue
                        recovered = launch(candidate['main'], request, out, 'collect')
                    identity = result['identity']
                    assert identity['M'] == M and identity['allocation'] == allocation[0]['shots']
                    assert result['MSE_lower'] == recovered['MSE_lower'] and result['MSE_upper'] == recovered['MSE_upper']
                    reference = mse(exact_histories(rows, float(toys.X(cfg['spec']['h']))), target)
                    checks[method][f'large_final_y={final_y}'] = interval(
                        kernel._decode(result['MSE_lower']), kernel._decode(result['MSE_upper']), reference)
                assert checks[method]['large_final_y=1.0']['reference'] != checks[method]['large_final_y=-1.0']['reference']
    return dict(original_M13=held['reason'], artificial_locators=True, methods=checks)


def check_saved(adapter):
    """The archived H4 packet extent and CLI dry-check, with zero Q3 calls."""
    requests = ROOT/'docs/research/rpe_distributed_h4_plan_20261117/requests'

    def check(method):
        path = requests/f'{method}.json'
        original = json.loads(path.read_text())
        base = original['configuration_id']
        view = Path(original['route']['view'])
        case, _ = adapter['scoped_case'](view, original)
        signal = adapter['git_json'](view, case['signal_packet'])
        actual = [row['m'] for row in signal['means']]
        assert actual == list(range(13)) and signal['configuration_id'] == base
        checks = {}
        for horizon in (12, 11):
            request = json.loads(path.read_text())
            if horizon == 11:
                projection = ['RPE_SCALED_SHOTS_V1', 'PREFIX', base, 12, 11]
                request['configuration_id'] = hashlib.sha256(json.dumps(projection, separators=(',', ':')).encode()).hexdigest()
                request['origin'] = dict(kind='RPE_SCALED_SHOTS_V1', M=11)
                request['route']['horizon_projection'] = dict(source_configuration_id=base, source_M=12, M=11)
                request['N_X'], request['N_Y'] = request['N_X'][:12], request['N_Y'][:12]
            with TemporaryDirectory(prefix='rpe-q3-m13-saved-dry-') as tmp:
                copy = Path(tmp)/'REQUEST.json'
                copy.write_text(json.dumps(request))
                subprocess.run([sys.executable, '-B', adapter['__file__'], '--request', str(copy),
                                '--output', tmp, '--mode', 'dry-check'], check=True, stdout=subprocess.DEVNULL)
                check = json.loads((Path(tmp)/'DRY_CHECK.json').read_text())
            assert check['status'] == 'DRY_CHECK_PASS' and check['metadata'] == 'PASS'
            checks[horizon] = dict(status=check['status'], route=check['route'], template=check['template'],
                                   configuration_id=request['configuration_id'])
        return method, dict(source_configuration_id=base, signal_packet=case['signal_packet'],
                            archived_rounds=actual, dry_check=checks)

    with ThreadPoolExecutor() as pool:
        return dict(pool.map(check, ('FULL_PF', 'TRUNCATED_PF', 'PR_DF')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-view', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--check-saved', action='store_true', help='Dry-check the archived server2 route locators')
    args = parser.parse_args()
    fixed = args.fixed_view.resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=fixed, text=True).strip() == FIXED
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src', 'scripts', 'tests'], cwd=fixed, check=True)
    sys.path[:0] = [str(fixed/'src'), str(fixed/'tests'), str(ROOT/'scripts')]
    import check_rpe_total_decoder_risk as toys

    adapter = runpy.run_path(str(ROOT/'scripts/run_rpe_optimizer_q3_job.py'))
    report = dict(fixed_reference=FIXED, threshold=str(TAU))
    stages = [('decoder', lambda: compare_terminal_branch(toys.T)),
              ('normal', lambda: check_normal(toys)), ('large', lambda: check_large(toys, adapter)),
              ('adapter', lambda: check_adapter(toys, adapter))]
    if args.check_saved:
        stages.append(('saved', lambda: check_saved(adapter)))
    for name, action in stages:
        print(f'{name}: start', flush=True)
        report[name] = action()
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(f'{name}: pass', flush=True)


if __name__ == '__main__':
    main()
