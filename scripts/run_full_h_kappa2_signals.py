"""S-1: Full-H pure RTE kappa2 (K2, q=2^m), 13 saved-input raw/corrected means.

The numerical provider is the unchanged C SOURCE (5cc7c07e bytes) used for the
R448 batch; the R448 caller's saved-reference reader is reused verbatim. The
schedule is the fixed D kappa2 candidate (26d4f23e), regenerated here through
the published B/D/C route and required to be identical. R448 signals are never
copied (saved=()), no reference/scalar is recomputed, nothing is sampled.

Default mode is a metadata-only preflight. `--execute-approved-s1` is the single
scientific entry: one batch, first failure stops, no retry, mandatory STOP.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import baseline_signal_pipeline as C  # noqa: E402
from trottertracks import rpe_optimization_pipeline as I  # noqa: E402
from trottertracks import full_h_kappa_layout as K  # noqa: E402
import run_truncated_fullh_signals as R448  # noqa: E402  (reference_reader/read/write reused)

CALLER = 'scripts/run_full_h_kappa2_signals.py'
LABEL = 'FULL_H_K2_QDYADIC_KAPPA2'
CONFIGURATION_ID = '54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e'
NUMERICAL_SOURCE = '5cc7c07e112ce97205948465c94441cba0042416'
R448_CALLER = dict(commit='5a172629db4d32a4230f8bb10a871b42807b1cd5', path='scripts/run_truncated_fullh_signals.py')
CANDIDATES = dict(commit='26d4f23e79e91385087669d4ee091632efa0c87c',
                  path='docs/research/d_full_h_kappa_schedule_20261114/KAPPA_CANDIDATES.json')
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
                path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
A_MENU = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
              path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json')
R448_PLAN = dict(commit='367dc98d832ecb4d769f2a3da3d174bc697331a5',
                 path='docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/PLAN.json')
PROPOSAL = dict(path='docs/research/d_full_h_kappa_schedule_20261114/FINITE_SCIENCE_PLAN.md',
                status='PROPOSED_NOT_ADOPTED')
OUTPUT = 'docs/research/d_full_h_kappa2_s1_results_20261114/run01'
COMPLETE = R448.COMPLETE
SCOPE = dict(rounds=13, new_mean_evaluations=13, finite_normalizers=13, new_reference_actions=0,
             scalar_extractions=0, sampling=0, circuits=0, saved_signal_copies=0, retry=False,
             N=None, C=None, G=None)


def read(d):
    return json.loads(C.blob(ROOT, d['commit'], d['path']))


def frac(d):
    return Fraction(d['numerator'], d['denominator'])


def prepare(root, source, *, api=None):
    """Regenerate the fixed kappa2 schedule and plan; metadata only (no input file)."""
    api = api or I.providers(str(root))
    pinned = read(CANDIDATES); cand = pinned['candidates'][0]
    if cand['label'] != LABEL or cand['configuration_id'] != CONFIGURATION_ID:
        raise ValueError('Fixed kappa2 candidate differs')
    manifest = read(MANIFEST); entry = manifest['candidates'][2]
    if entry['label'] != 'FULL_H_K2_QDYADIC_R448':
        raise ValueError('Manifest Full-H entry moved')
    s = read(A_MENU)['spec']
    prep = api.D.saved_full_h_preparation('H4'); prep['binding'] = deepcopy(s['binding'])
    if not C.p.same(prep, manifest['shared']['full_h_preparation']):
        raise ValueError('Full-H preparation descriptor differs from the R448 manifest')
    spec = api.D.full_h_rte_spec(s, M=12, finite_K=2, preparation=prep,
                                 kappa=K.kappa_descriptor(2, proposal_source=PROPOSAL))
    layouts, rows = K.layouts(spec, B=api.B)
    config = C.configure(root, spec, layouts=layouts, api=api)
    K.check_configuration(config, B=api.B, rows=rows)
    if config['id'] != CONFIGURATION_ID or not C.p.same(config, cand['configuration']):
        raise ValueError('Regenerated kappa2 configuration differs from the fixed candidate')
    refs = [deepcopy(r['reference_descriptor']) for r in entry['rounds']]
    plan = I.plan(root, config, target=manifest['shared']['target'], signal_source=source, references=refs)
    saved_r448 = {r['physical_candidate']['id'] for r in read(R448_PLAN)['requests']}
    requests = plan['pending']['requests']
    if len(requests) != 13 or len(cand['rounds']) != 13:
        raise ValueError('Thirteen rounds required')
    for req, row, fixed in zip(requests, rows, cand['rounds'], strict=True):
        c = req['physical_candidate']
        if (req['m'] != row['m'] or c['id'] != fixed['physical_candidate_id'] or c['q'] != row['q_grouping']
                or c['R_outer'] != row['R_per_group'] or c['s'] != row['event_count'] or c['K'] != 2
                or req['q_PF'] is not None or req['comparison_method'] != 'FULL_H_RTE'
                or req['reference_status'] != 'SUPPLIED'
                or not C.p.same(req['domain']['reference'], fixed['reference_descriptor'])):
            raise ValueError('Planned physical round differs from the fixed kappa2 row at m=' + str(row['m']))
        if c['id'] in saved_r448:
            raise ValueError('R448 physical candidate in the kappa2 plan; signals must not be reused')
        row['physical_candidate_id'] = c['id']
    return dict(plan=plan, rows=rows, manifest=manifest, candidate=cand)


def numerical_sources(root):
    for path in (C.MODULE, C.NUMERICAL):
        if C.blob(root, NUMERICAL_SOURCE, path) != (Path(root) / path).read_bytes():
            raise ValueError('Fixed numerical SOURCE changed: ' + path)
    if C.blob(root, R448_CALLER['commit'], R448_CALLER['path']) != (Path(root) / R448_CALLER['path']).read_bytes():
        raise ValueError('Reused R448 caller bytes changed')
    return True


def post_gates(records, rows):
    """Envelope gate with the provider's own one-ULP normalizer sensitivity."""
    checks = []
    for record, row in zip(records, rows, strict=True):
        c = record['physical_candidate']; sig = record['signal']; num = record['numerical']
        env = float(frac(row['paper_logB_envelope'])); sens = num.get('normalizer_one_ulp_log_sensitivity')
        logB = sig['log_B']
        ok = (record['status'] == 'NUMERICAL_MEAN_AVAILABLE' and num.get('gates_passed') is True
              and c['id'] == row['physical_candidate_id'] and c['q'] == row['q_grouping']
              and c['R_outer'] == row['R_per_group'] and c['s'] == row['event_count']
              and isinstance(sens, float) and math.isfinite(sens) and sens >= 0
              and 0 < logB <= math.nextafter(env, math.inf) + sens
              and sig['attenuation'] == math.exp(-logB) and sig['B'] == math.exp(logB))
        checks.append(dict(m=row['m'], physical_candidate_id=c['id'], log_B=logB, envelope=env,
                           one_ulp_allowance=sens, envelope_slack=None if not ok else math.nextafter(env, math.inf)+sens-logB,
                           gate='PASS' if ok else 'FAIL'))
    return checks


def summary_rows(packet):
    out = []
    for mean, rec in zip(packet['means'], packet['producer_result']['records'], strict=True):
        sig = rec['signal']; c = rec['physical_candidate']
        out.append(dict(label=LABEL, configuration_id=packet['configuration_id'], m=mean['m'],
            physical_candidate_id=c['id'], T=c['T'], q_grouping=c['q'], R_per_group=c['R_outer'], s=c['s'],
            raw_real=sig['raw']['real'], raw_imag=sig['raw']['imag'],
            corrected_real=sig['corrected']['real'], corrected_imag=sig['corrected']['imag'],
            reference_real=sig['reference']['real'], reference_imag=sig['reference']['imag'],
            B=sig['B'], log_B=sig['log_B'], attenuation=sig['attenuation'],
            gates_passed=rec['numerical']['gates_passed'], certified=rec['numerical']['certified'], status=mean['status']))
    return out


def run(root, prepared, out, *, context_factory, reader):
    """One schedule; the provider stops at the first unresolved round."""
    plan, rows = prepared['plan'], prepared['rows']
    result = dict(status='STARTED', label=LABEL, configuration_id=plan['configuration']['id'], errors=[],
                  new_mean_evaluations=0, saved_copies=0, post_gates=None, B_packet_written=False,
                  N=None, C=None, G=None, mandatory_stop=True, retry_performed=False)
    try:
        packet = I.produce_signals(root, plan, out / LABEL, context_factory=context_factory,
                                   reference_provider=reader, saved=())
        produced = packet['producer_result']
        result.update(new_mean_evaluations=produced['new_mean_evaluations'], saved_copies=produced['saved_copies'])
        if produced['saved_copies'] != 0:
            raise ValueError('Saved signal copied into kappa2')
        if produced['status'] != COMPLETE:
            result['status'] = produced['status']
            return result
        checks = post_gates(produced['records'], rows)
        result['post_gates'] = checks
        R448.write(out / LABEL / 'POST_GATES.json', checks)
        if any(c['gate'] != 'PASS' for c in checks):
            result['status'] = 'STOP_POST_GATE_UNRESOLVED'
            return result
        R448.write(out / LABEL / 'B_SIGNAL_INPUT.json', I.signal_input(packet))
        table = summary_rows(packet)
        with (out / 'SIGNALS.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
        result.update(status=COMPLETE, B_packet_written=True)
    except Exception as error:
        result['status'] = 'STOP_CALLER_ERROR'
        result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    return result


def execute(root):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (CALLER, I.MODULE, C.MODULE, C.NUMERICAL, K.MODULE, R448_CALLER['path']):
        if C.blob(root, commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ from HEAD: ' + path)
    numerical_sources(root)
    source = dict(commit=commit, caller=CALLER, numerical_source=NUMERICAL_SOURCE, candidates=CANDIDATES,
                  r448_reference_reader=R448_CALLER, integration_module=I.MODULE, layout_helper=K.MODULE)
    out = Path(root) / OUTPUT; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    R448.write(out / 'SOURCE.json', dict(source=source, scope=SCOPE, started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable, platform=platform.platform(), automatic_retry=False))
    result = dict(status='STOP_PREFLIGHT_UNKNOWN', errors=[], new_mean_evaluations=0, N=None, C=None, G=None,
                  mandatory_stop=True)
    try:
        prepared = prepare(root, source)
        api = I.providers(str(root))
        reader = R448.reference_reader(root, prepared['manifest']['shared'], api)
        from trottertracks import baseline_signal_numerics as N
        import numpy as np
        import scipy
        shared = []

        def context_factory():
            if not shared:
                ctx = N.Context(root, *N.load_shared_h4(root, prepared['manifest']['shared']['binding']))
                R448.write(out / 'INPUT_PROVENANCE.json', dict(binding=ctx.binding, source=ctx.metadata['source'],
                    state_dimension=len(ctx.state), matrix_count=len(ctx.matrices),
                    full_h_preparation=prepared['manifest']['shared']['full_h_preparation'],
                    reference_source=prepared['manifest']['shared']['independent_reference_pairs']['source'],
                    numpy=np.__version__, scipy=scipy.__version__, new_scalar_extraction=False))
                shared.append(ctx)
            return shared[0]

        result = run(root, prepared, out, context_factory=context_factory, reader=reader)
        result['context_restorations'] = len(shared)
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    result.update(elapsed_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
    R448.write(out / 'COVERAGE.json', result)
    R448.write(out / 'STOP.json', dict(status=result['status'], mandatory_stop=True, retry_performed=False,
        new_mean_evaluations=result['new_mean_evaluations'], N=None, C=None, G=None,
        further_science_requires_new_scope=True))
    print(json.dumps(dict(status=result['status'], evaluations=result['new_mean_evaluations'],
                          elapsed_seconds=result['elapsed_seconds']), allow_nan=False), flush=True)
    return 0 if result['status'] == COMPLETE else 2


def preflight(root):
    """No saved input, normalizer, signal or circuit. Prints the frozen scope."""
    numerical_sources(root)
    prepared = prepare(root, dict(commit='PREFLIGHT_ONLY', caller=CALLER, numerical_source=NUMERICAL_SOURCE))
    rows = [dict(m=r['m'], id=r['physical_candidate']['id'], q=r['physical_candidate']['q'],
                 R=r['physical_candidate']['R_outer'], s=r['physical_candidate']['s'])
            for r in prepared['plan']['pending']['requests']]
    return dict(status='PREFLIGHT_METADATA_ONLY', configuration_id=prepared['plan']['configuration']['id'],
                scope=SCOPE, rounds=rows, output=OUTPUT)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-approved-s1', action='store_true',
                        help='run the 13 saved-input means (requires the user-approved S-1 scope)')
    args = parser.parse_args(argv)
    if args.execute_approved_s1:
        return execute(ROOT)
    print(json.dumps(preflight(ROOT), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
