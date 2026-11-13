"""One fixed 26-round batch: saved H4 B0I and Full-H, no reference solver.

The numerical provider is unchanged. This caller only connects the approved
scope, saved reference reader, shared context and batch-wide first-failure STOP.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import baseline_signal_pipeline as C
from trottertracks import rpe_optimization_pipeline as I

CALLER = 'scripts/run_truncated_fullh_signals.py'
SCOPE = dict(commit='6306fb3a74e7864a44a3e6cd022e9a4b062e7d44',
             path='docs/research/truncated_fullh_signal_scope_20261114/SCOPE.json')
OUTPUT = 'docs/research/truncated_fullh_signal_results_20261114/run01'
COMPLETE = 'BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED'


def read(root, descriptor):
    return json.loads(C.blob(root, descriptor['commit'], descriptor['path']))


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(root, source):
    scope = read(root, SCOPE)
    manifest = read(root, scope['source_manifest'])
    for path in (C.MODULE, C.NUMERICAL):
        if C.blob(root, scope['numerical_source']['commit'], path) != (root / path).read_bytes():
            raise ValueError('Fixed numerical SOURCE changed: ' + path)
    plans = []
    for selected in scope['selected']:
        candidate = manifest
        for key in selected['source_candidate_selector']:
            candidate = candidate[key]
        if candidate['label'] != selected['label'] or candidate['configuration']['id'] != selected['configuration_id']:
            raise ValueError('Scope/configuration differs')
        plan = I.plan(root, candidate['configuration'], target=manifest['shared']['target'],
                      signal_source=source, references=[r['reference_descriptor'] for r in candidate['rounds']])
        for req, original, row in zip(plan['pending']['requests'], candidate['rounds'], selected['rounds'], strict=True):
            if (not C.p.same(req['physical_candidate'], original['physical_candidate'])
                    or req['physical_candidate']['id'] != row['physical_candidate_id']
                    or not C.p.same(req['T'], row['T']) or req['q_RPE'] != row['q_RPE']
                    or req['q_PF'] != row['q_PF'] or req['physical_candidate']['q'] != row['q_evolution_grouping']
                    or req['physical_candidate']['R_outer'] != row['R_outer']):
                raise ValueError('Scope physical round differs')
        if len(plan['pending']['requests']) != 13 or selected['future_output'] != OUTPUT + '/' + selected['label']:
            raise ValueError('Fixed finite scope/output differs')
        plans.append((selected['label'], plan))
    return scope, manifest, plans


def reference_reader(root, shared, api):
    locator = shared['independent_reference_pairs']['source']
    saved = read(root, locator)
    if any(not C.p.same(v, shared['binding'].get(k)) for k, v in saved['binding'].items()):
        raise ValueError('Saved reference physical binding differs')

    def reader(descriptor):
        m = descriptor['m']
        if (descriptor['source'] != dict(locator, selector=['rounds', m, 'diagnostics', 'reference_pair'])
                or not C.p.same(descriptor['binding'], shared['binding'])):
            raise ValueError('Saved reference SOURCE/selector differs')
        row = saved['rounds'][m]
        if row['m'] != m or api.B.a.exact_scalar(row['physical_candidate']['T']) != api.B.a.exact_scalar(descriptor['T']):
            raise ValueError('Saved reference round/time differs')
        pair = row['diagnostics']['reference_pair']
        # The unchanged numerical provider applies the original pair gate.
        return dict(deepcopy(descriptor), **{k: deepcopy(pair[k]) for k in
                    ('reference', 'independent_reference', 'signal_difference')})
    return reader


def run_schedules(root, plans, out, context_factory, reader):
    """The provider already stops within a schedule; stop the second one too."""
    summary = dict(status='STARTED', schedules=[], errors=[], new_mean_evaluations=0,
                   saved_copies=0, N=None, C=None, G=None, mandatory_stop=True)
    stopped = False
    for label, plan in plans:
        row = dict(label=label, configuration_id=plan['configuration']['id'], status='NOT_STARTED',
                   available_means=0, coverage=[dict(m=r['m'], physical_candidate=r['physical_candidate'],
                   identity=r['identity'], law_binding=r['law_binding'], status='NOT_STARTED')
                   for r in plan['pending']['requests']])
        summary['schedules'].append(row)
        if stopped:
            continue
        print(label + ': START', flush=True)
        try:
            packet = I.produce_signals(root, plan, out / label, context_factory=context_factory,
                                       reference_provider=reader, saved=())
            result = packet['producer_result']
            write(out / label / 'B_SIGNAL_INPUT.json', I.signal_input(packet))
            row['status'] = result['status']
            summary['new_mean_evaluations'] += result['new_mean_evaluations']
            summary['saved_copies'] += result['saved_copies']
            for cell, mean in zip(row['coverage'], packet['means'], strict=True):
                cell['status'] = mean['status']
                if mean['raw'] is not None:
                    row['available_means'] += 1
            stopped = row['status'] != COMPLETE
        except Exception as error:
            row['status'] = 'STOP_CALLER_ERROR'
            summary['errors'].append(dict(type=type(error).__name__, message=str(error), label=label))
            stopped = True
        print(label + ': ' + row['status'], flush=True)
    summary['status'] = 'STOP_UNKNOWN' if stopped else COMPLETE
    return summary


def main():
    root = ROOT
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (CALLER, I.MODULE, C.MODULE, C.NUMERICAL):
        if C.blob(root, commit, path) != (root / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ: ' + path)
    source = dict(commit=commit, caller=CALLER, finite_scope=SCOPE,
                  numerical_source='5cc7c07e112ce97205948465c94441cba0042416', integration_module=I.MODULE)
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = dict(source=source, started_utc=datetime.now(timezone.utc).isoformat(),
                      python=sys.version, executable=sys.executable, platform=platform.platform(),
                      source_date='2026-11-14 Asia/Tokyo', automatic_retry=False)
    write(out / 'SOURCE.json', provenance)
    result = dict(status='STOP_PREFLIGHT_UNKNOWN', errors=[], new_mean_evaluations=0,
                  saved_copies=0, N=None, C=None, G=None, mandatory_stop=True)
    try:
        scope, manifest, plans = prepare(root, source)
        api = I.providers(str(root))
        reader = reference_reader(root, manifest['shared'], api)
        from trottertracks import baseline_signal_numerics as N
        import numpy as np
        import scipy
        shared = []

        def context_factory():
            if not shared:
                restored = N.load_shared_h4(root, manifest['shared']['binding'])
                if '7' not in restored[2]['prefixes'] or 'scalar' not in restored[2]['prefixes']['7']:
                    raise ValueError('Saved prefix7 scalar missing; no extractor fallback')
                ctx = N.Context(root, *restored)
                if 7 not in ctx.tail_scalars:
                    raise ValueError('Saved prefix7 scalar unavailable')
                write(out / 'INPUT_PROVENANCE.json', dict(binding=ctx.binding, source=ctx.metadata['source'],
                    state_dimension=len(ctx.state), matrix_count=len(ctx.matrices),
                    prefix7_scalar=ctx.tail_scalars[7], full_h_preparation=manifest['shared']['full_h_preparation'],
                    reference_source=manifest['shared']['independent_reference_pairs']['source'],
                    numpy=np.__version__, scipy=scipy.__version__, new_scalar_extraction=False))
                shared.append(ctx)
            return shared[0]

        result = run_schedules(root, plans, out, context_factory, reader)
        result['context_restorations'] = len(shared)
        result['numerical_contract'] = scope['numerical_contract']
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    result.update(elapsed_seconds=time.perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'COVERAGE.json', result)
    write(out / 'STOP.json', dict(status=result['status'], mandatory_stop=True, retry_performed=False,
          new_mean_evaluations=result['new_mean_evaluations'], N=None, C=None, G=None,
          further_science_requires_new_scope=True))
    print(json.dumps(dict(status=result['status'], evaluations=result['new_mean_evaluations'],
                          elapsed_seconds=result['elapsed_seconds']), allow_nan=False), flush=True)
    return 0 if result['status'] == COMPLETE else 2


if __name__ == '__main__':
    sys.exit(main())
