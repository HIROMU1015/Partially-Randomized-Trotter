"""Stage saved Full PF / B0I p7 shortlist round means as (incomplete) B packets. No science.

    python -B scripts/stage_saved_pf_means.py            # metadata preflight (configs, coverage, counts)
    python -B scripts/stage_saved_pf_means.py --write    # writes the staging directory (fresh)

The packet `source` is this caller at HEAD (its bytes must equal HEAD); each available mean
carries its saved-file origin in `provider_binding.input_source`. Incomplete packets are
written for the record only and are refused by `require_complete` as risk/N input.
"""
import argparse
import csv
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_optimization_pipeline as I  # noqa: E402
from trottertracks import saved_pf_mean_staging as S  # noqa: E402

CALLER = 'scripts/stage_saved_pf_means.py'
OUT = 'docs/research/a_saved_pf_mean_staging_20261115'
TRACK_A = dict(long_time=dict(commit='58b21d596ff8be94d436c5c08e14e2a8c6310342',
                              path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz'),
               pf_common=dict(commit='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c',
                              path='artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/signal_records.json.gz'),
               identity='artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/source_input_identity.json',
               stop='artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/MANDATORY_STOP.json')


def head_source():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for path in (S.MODULE, CALLER):
        if subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Staging SOURCE bytes differ from HEAD: ' + path)
    return dict(commit=commit, caller=CALLER, module=S.MODULE, kind=S.KIND, coverage=S.COVERAGE,
                reference_pairs=S.PHYSICS_REF, numerical_source=None, new_mean_evaluations=0)


def gz(d):
    return json.loads(gzip.decompress(subprocess.check_output(['git', 'show', d['commit'] + ':' + d['path']], cwd=ROOT)))


def s2l_audit(staged):
    """Technical record of the legacy (Track A) means; nothing is recomputed or re-gated."""
    lt, pfc = gz(TRACK_A['long_time']), gz(TRACK_A['pf_common'])
    ident = json.loads((ROOT / TRACK_A['identity']).read_text())
    stop = json.loads((ROOT / TRACK_A['stop']).read_text())
    rows = []
    for st in staged:
        for r in st['rounds']:
            if r['state'] != 'S2L_QUALIFICATION_PENDING':
                continue
            rec = lt.get(r['physical_id'])
            origin = rec.get('origin') if rec else None
            common = pfc.get(origin.split('#', 1)[1]) if origin and '#' in origin else None
            rows.append(dict(
                candidate=st['candidate'], m=r['m'], physical_id=r['physical_id'],
                legacy_records=r['legacy_sources'],
                long_time_record_reuses_pf_common=bool(common), pf_common_key=origin.split('#', 1)[1] if common else None,
                inputs=dict(hamiltonian=ident['binding']['hamiltonian'], snapshot=ident['binding']['snapshot'],
                            state=ident['binding']['state'], matches_current_binding=True),
                numerical_source=ident['SOURCE'], producer_pointer=(common or rec or {}).get('provenance'),
                saved_fields=sorted((common or rec)['signal']), saved_gate_evidence=dict(
                    allowance_evidence=(common or rec)['signal'].get('allowance_evidence'),
                    old_u=(common or rec).get('old_u'), old_margin=(common or rec).get('old_margin')),
                reference_used='Track A own saved reference (not the bba15083 pair object)',
                study_stop=stop,
                missing_for_current_acceptance=[
                    'step_unitarity_error / power_unitarity_error / norm_error with the 1e-9 gate (current policy)',
                    'backend identity with DIRECT_EXPM_SIGNED_PF_MATRIX_POWER (current direct route)',
                    'reference-pair gate evidence against bba15083 for this round object'],
                precedent='C c5466946 COVERAGE_AND_REUSE: Track A S2/.1 m3/m4 -> new direct mean (power diagnostics missing)',
                decision='PENDING_NOT_PROMOTED',
                next_science_candidate='one direct power + gates on the saved step for this round (needs approval)'))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--write', action='store_true'); args = ap.parse_args()
    source = head_source() if args.write else dict(fixture='PREFLIGHT_NOT_PUBLISHED', kind=S.KIND)
    staged = [S.stage(ROOT, c, I=I, staging_source=source) for c in S.CANDIDATES]
    for st in staged:
        try:
            S.require_complete(st['packet']); raise AssertionError('incomplete packet accepted')
        except ValueError:
            pass
    summary = [dict(candidate=st['candidate'], configuration_id=st['configuration_id'],
                    ready=[r['m'] for r in st['rounds'] if r['state'].startswith('READY')],
                    s2l=[r['m'] for r in st['rounds'] if r['state'].startswith('S2L')],
                    s3=[r['m'] for r in st['rounds'] if r['state'].startswith('S3')], complete=st['complete'])
               for st in staged]
    print(json.dumps(summary, indent=1))
    if not args.write:
        return 0
    out = ROOT / OUT; out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (out / name).open('x') as f:
            json.dump(value, f, indent=1, allow_nan=False, default=str); f.write('\n')
    for st in staged:
        write('STAGED_SIGNAL_INPUT_%s.json' % st['candidate'], I.signal_input(st['packet']))
    write('STAGING.json', dict(kind=S.KIND, source=source, status='STAGED_INCOMPLETE_NOT_RISK_INPUT', candidates=[
        {k: st[k] for k in ('candidate', 'method', 'formula', 'width_over_h', 'configuration_id', 'rounds', 'complete',
                            'risk_input_allowed')} for st in staged], summary=summary,
        new_mean_evaluations=0, new_spectra=0, numerical_reanalysis=False, sufficient_N=None, G=None, mandatory_stop=True))
    write('S2L_AUDIT.json', dict(kind='S2L_LEGACY_MEAN_TECHNICAL_AUDIT', rows=s2l_audit(staged),
                                 screen_run02_m6='outside the six shortlist candidates (Full PF S2/0.1 only)'))
    with (out / 'STAGING_STATUS.csv').open('x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate', 'configuration_id', 'm', 'axis', 'state', 'physical_id', 'canonical_id', 'q_PF',
                    'raw_bound', 'saved_mean_origin', 'coverage_status'])
        for st in staged:
            means = {m['m']: m for m in st['packet']['means']}
            for r in st['rounds']:
                for axis in ('X', 'Y'):
                    pb = means[r['m']].get('provider_binding') or {}
                    origin = (pb.get('input_source') or {}).get('saved_mean')
                    w.writerow([st['candidate'], st['configuration_id'], r['m'], axis, r['state'], r['physical_id'],
                                r['canonical_id'], r['q_PF'], means[r['m']]['raw'] is not None,
                                '' if not origin else origin['commit'] + ':' + origin['path'], r['coverage_status']])
    return 0


if __name__ == '__main__':
    sys.exit(main())
