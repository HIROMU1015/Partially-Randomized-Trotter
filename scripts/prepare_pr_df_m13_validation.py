"""Extend the archived PR-DF LD7 packet using its fixed signal provider."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('docs/research/rpe_m13_real_validation_20261122')
PREFIX = DATA/'PR_DF_PREFIX.json'


def read(location):
    record = json.loads(subprocess.check_output(
        ['git', 'show', location['commit']+':'+location['path']], cwd=ROOT))
    for key in location.get('selector', ()):
        record = record[key]
    return record


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def plan(parent, worker, provider, metadata, mapper):
    original = read(parent['configuration'])
    packet = read(parent['signal_packet'])
    spec = deepcopy(original['spec'])
    if spec['M'] != 12 or spec['L_D'] != 7 or packet['configuration_id'] != original['id']:
        raise ValueError('Expected the archived PR-DF LD7 candidate')
    spec['M'] = spec['A_input']['M'] = 13
    config = mapper.generate(spec)
    for old, new, mean in zip(original['rounds'], config['rounds'], packet['means']):
        keys = ('physical_candidate', 'structure', 'T', 'mapping', 'q_PF', 'R_outer')
        if (any(old[key] != new[key] for key in keys)
                or mean['identity'] != old['identity'] or mean['law_binding'] != old['law_binding']):
            raise ValueError('Saved physical round differs at m=' + str(old['m']))
    pending = metadata.plan(config, {r['m']: r['physical_candidate'] for r in config['rounds']},
                            b_validate=mapper.validate, b_exact_scalar=provider._scalar,
                            numerical_recipe=provider.numerical_recipe(worker['IMPLEMENTATION']))
    return dict(configuration=config, parent=parent, physical_round=config['rounds'][-1]['physical_candidate'],
                job=pending['jobs'][-1], requested_rounds=[13], reused_rounds=list(range(13)))


def reused_records(data, packet, destination):
    rows = data['configuration']['rounds']
    for row, mean in zip(rows[:-1], packet['means']):
        carrier = read(mean['source'])
        keys = ('raw', 'corrected', 'reference', 'exact_tail')
        if (any(carrier['signal'][key] != mean[key] for key in keys)
                or carrier['B_identity'] != mean['identity'] or carrier['B_law_binding'] != mean['law_binding']):
            raise ValueError('Archived PR-DF carrier differs at m=' + str(row['m']))
        carrier.update(B_configuration_id=data['configuration']['id'], B_candidate_id=row['candidate']['id'],
                       B_identity=row['identity'], B_law_binding=row['law_binding'], horizon_origin=mean['source'])
        write(destination/f'm{row["m"]}.json', carrier)


def execute(data, provider, worker, destination, reference_location):
    config = data['configuration']
    job = data['job']
    pair = read(reference_location)
    binding = config['spec']['binding']
    keys = ('hamiltonian', 'state', 'snapshot', 'system', 'rank', 'basis', 'geometry')
    time = data['physical_round']['T']
    if provider._scalar(pair['T']) != provider._scalar(time) or any(pair['binding'][key] != binding[key] for key in keys):
        raise ValueError('Saved independent reference pair belongs to different physics')
    reference = dict(binding=binding, T=time, source=reference_location, evidence=pair)
    context = provider.load_h4_context(ROOT, config)
    print('ROUND_START m13', flush=True)
    record = provider.evaluate(context, job, reference, root=ROOT)
    write(destination/'m13.json', record)
    print('ROUND_END m13', record['status'], flush=True)
    if not worker['valid_record'](record, job):
        raise ValueError('Terminal PR-DF signal failed the unchanged numerical gate')
    reused_records(data, read(data['parent']['signal_packet']), destination)


def bundle(data, destination, carrier_commit, configuration_location):
    original = read(data['parent']['signal_packet'])
    config = data['configuration']
    if read(configuration_location) != config:
        raise ValueError('Standalone configuration differs from the fixed plan')
    result = dict(original, configuration_source=configuration_location,
                  configuration_id=config['id'], raw_result_carrier=carrier_commit, means=[])
    for row in config['rounds']:
        m = row['m']
        before = original['means'][m if m < 13 else 12]
        source = dict(commit=carrier_commit, path=str(destination.relative_to(ROOT)/f'm{m}.json'))
        record = read(source)
        signal = record['signal']
        mean = dict(before, m=m, identity=row['identity'], law_binding=row['law_binding'], source=source,
                    **{key: signal[key] for key in ('raw', 'corrected', 'reference', 'exact_tail')},
                    normalization=dict(before['normalization'], source=source,
                        **{key: signal[key] for key in ('log_B', 'B', 'attenuation')},
                        log_B_margin=record['allowance']['log_B_margin'],
                        normalizer_one_ulp_log_sensitivity=signal['allowance_evidence']['normalizer_one_ulp_log_sensitivity']),
                    bias={key: signal[key] for key in before['bias']},
                    uncertainty=dict(before['uncertainty'], empirical_allowance=record['allowance'],
                        numerical_observations=record['numerical'],
                        maximum_norm_domain=signal['allowance_evidence']['maximum_norm_domain']),
                    state_evidence=record['state_evidence'], reference_source=record['reference_source'])
        if m < 13:
            mean['horizon_origin'] = before['source']
        result['means'].append(mean)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'execute', 'bundle'))
    parser.add_argument('--view', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        worker = runpy.run_path(str(ROOT/'scripts/run_pr_df_candidate_signal_jobs.py'))
        provider, metadata, mapper = worker['fixed_runtime'](args.view)
        write(args.output, plan(json.loads((ROOT/PREFIX).read_text()), worker, provider, metadata, mapper))
        return
    fixed = json.loads(args.input.read_text())
    data = read(fixed['plan'])
    if args.mode == 'execute':
        worker = runpy.run_path(str(ROOT/'scripts/run_pr_df_candidate_signal_jobs.py'))
        provider, _, _ = worker['fixed_runtime'](args.view)
        execute(data, provider, worker, args.output.resolve(), fixed['reference'])
    else:
        write(args.output, bundle(data, (ROOT/fixed['raw_directory']).resolve(), fixed['carrier'], fixed['configuration']))


if __name__ == '__main__':
    main()
