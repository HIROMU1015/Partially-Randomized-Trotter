"""Prepare one terminal H4 DF10 Full PF or B0I signal from the fixed snapshot."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys


PROVIDER = 'c3636265b73e9b59ce7f51dd3159c930fdd2e736'
ENTRY = 'scripts/run_rpe_pf_horizon_signal.py'
PATH_BINDER = 'scripts/run_pf_preparation_stage.py'
PATH_RESOLVER = 'src/trottertracks/rpe_preparation_paths.py'


def fixed_plan(root, request, source):
    from trottertracks import pf_evidence_preparation as evidence

    config = evidence.read(root, request['configuration'])
    prefix = evidence.read(root, request['signal_packet'])
    spec = config['spec']
    if (spec['M'] != 12 or config['id'] != prefix['configuration_id']
            or [row['m'] for row in prefix['means']] != list(range(13))
            or evidence.I.providers(str(root)).B.comparison_method(spec) not in ('FULL_PF', 'TRUNCATED_PF')):
        raise ValueError('Expected the fixed 13-round PF/B0I packet of the same candidate')

    extended = evidence.C.configure(str(root), dict(spec, M=13))
    original = evidence.I.plan(str(root), config, target=prefix['target'], signal_source=prefix['source'])
    terminal = extended['rounds'][-1]
    reference = dict(m=13, T=terminal['T'], binding=spec['binding'],
                     source=dict(source, artifact='REFERENCE.json', configuration_id=extended['id']))
    plan = evidence.I.plan(str(root), extended, target=prefix['target'], signal_source=source, references=[reference])
    for m, (before, mean) in enumerate(zip(original['pending']['requests'], prefix['means'])):
        if (any(mean[key] != before[key] for key in ('identity', 'law_binding', 'physical_candidate'))
                or not mean.get('provider_binding') or mean.get('raw') is None
                or evidence.I.price_context(original, m) != evidence.I.price_context(plan, m)):
            raise ValueError('Saved PF physical schedule or signal differs at m=' + str(m))
        evidence.I.provider_binding_check(original, mean)
    return plan, prefix


def reference_pair(root, ctx, T):
    import numpy
    from scipy.linalg import expm
    from scipy.sparse.linalg import expm_multiply
    from trottertracks import baseline_signal_numerics as numerics
    from trottertracks.r_price_phase1_science import functions

    scope = dict(np=numpy, expm=expm, expm_multiply=expm_multiply)
    functions(root, numerics.SHARED, 'src/trottertracks/h6_df19_rpe_d2.py', ('zrecord', 'reference_pair'), scope)
    return scope['reference_pair'](ctx, T)


def signal(root, plan, prefix, source):
    from trottertracks import pf_evidence_preparation as evidence

    runpy.run_path(str(Path(__file__).resolve().parents[1] / PATH_BINDER))['bind_saved_signal_paths']()
    request = plan['pending']['requests'][13]
    descriptor = request['domain']['reference']
    binding = plan['configuration']['spec']['binding']
    saved, snapshot_binding, context = evidence.N.load_shared_h4(root, binding)
    pair = reference_pair(root, saved, request['physical_candidate']['T'])
    reference = dict(pair, source=descriptor['source'], binding=binding, T=descriptor['T'])
    ctx = evidence.N.Context(root, saved, snapshot_binding, context)
    prefix_index = request['physical_candidate']['prefix']
    if request['comparison_method'] == 'TRUNCATED_PF':
        scalar_source = evidence.scalar_source(prefix_index)
        ctx.tail_scalars[prefix_index] = float(evidence.read(root, scalar_source))
        ctx.metadata['prefix_scalar_source'] = scalar_source
    raw = evidence.N.evaluate(ctx, request, reference, source=source)
    if raw['status'] != 'NUMERICAL_MEAN_AVAILABLE':
        raise ValueError('Terminal PF numerical gate: ' + str(raw.get('reason')))

    means = []
    for m, row in enumerate(prefix['means']):
        target = plan['pending']['requests'][m]
        carried = deepcopy(row)
        carried.update(identity=target['identity'], law_binding=target['law_binding'], source=source)
        carried['provider_binding'].update(numerical_source=source,
            original_numerical_source=row['source'], saved_origin=dict(m=m, source=row['source']))
        if 'implementation_source' in carried:
            carried['implementation_source'] = source
        evidence.I.provider_binding_check(plan, carried)
        means.append(carried)
    means.append(evidence.bind_mean(plan, request, raw))
    packet = dict(configuration_id=plan['configuration']['id'], source=source, target=prefix['target'], means=means)
    return reference, raw, packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider-root', required=True, type=Path)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('plan', 'execute'))
    parser.add_argument('--caller-source')
    args = parser.parse_args()
    root = args.provider_root.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != PROVIDER:
        raise ValueError('Fixed baseline signal provider differs')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src', 'scripts'], cwd=root, check=True)
    local = Path(__file__).resolve().parents[1]
    commit = args.caller_source
    if commit is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=local, text=True).strip()
    for file in (ENTRY, PATH_BINDER, PATH_RESOLVER):
        if (local / file).read_bytes() != subprocess.check_output(['git', 'show', commit + ':' + file], cwd=root):
            raise ValueError('Terminal signal runtime differs from its fixed SOURCE: ' + file)
    source = dict(commit=commit, path=ENTRY, role='FIXED_TERMINAL_SIGNAL_AND_PREFIX_ASSEMBLY', provider=PROVIDER)
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    request = json.loads(args.request.read_bytes())
    plan, prefix = fixed_plan(root, request, source)
    selected = plan['pending']['requests'][13]['physical_candidate']
    args.output.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        with (args.output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')

    write('PLAN.json', dict(parent=request, configuration=plan['configuration'],
        physical_round=selected, requested_rounds=[13], reused_rounds=list(range(13)), source=source))
    if args.mode == 'plan':
        return
    if request.get('physical_id') != selected['id']:
        raise ValueError('Execution requires the fixed terminal physical schedule')
    reference, raw, packet = signal(root, plan, prefix, source)
    write('REFERENCE.json', reference)
    write('RAW_TERMINAL.json', raw)
    write('MEAN_HANDOFF.json', dict(packet, prefix_source=request['signal_packet']))


if __name__ == '__main__':
    main()
