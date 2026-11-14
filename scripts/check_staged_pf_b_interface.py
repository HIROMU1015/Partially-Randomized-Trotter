"""Compatibility of the staged Full PF / B0I shortlist packets with B's existing intake (9c0e12d8).

Run inside an unmodified checkout of B's commit (B code is imported from there, never edited):

    cd <B checkout @ 9c0e12d87> && python -B <A tree>/scripts/check_staged_pf_b_interface.py \
        --b-root . --packets-commit <published staging commit>

Checks (metadata/artificial only; B's kernel/risk/N are never called):
  1. The configuration regenerated with B's own providers has the staged configuration id.
  2. The real staged packet (incomplete) is REFUSED by T.bind_signal (pinned SOURCE path).
  3. A variant whose missing rounds carry the packet SOURCE is still REFUSED (whole-schedule /
     signal preparation) -- B's completeness requirement is not relaxed.
  4. An ARTIFICIAL complete fixture (staged available rounds kept, missing rounds filled with
     artificial values and B's own physical context) is ACCEPTED, i.e. the staged rounds'
     identity / law / physical context / provider binding pass B's existing checks.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
from unittest.mock import patch

OUT = 'docs/research/a_saved_pf_mean_staging_20261115'
LABELS = (('FULL_PF_S2_16h_over_7', 'FULL', '2nd', Fraction(16, 7)), ('FULL_PF_Y4_16h_over_3', 'FULL', '4th', Fraction(16, 3)),
          ('FULL_PF_M5_16h', 'FULL', '4th(m5_best)', Fraction(16)), ('B0I_P7_S2_16h_over_7', 'B0I', '2nd', Fraction(16, 7)),
          ('B0I_P7_Y4_16h_over_3', 'B0I', '4th', Fraction(16, 3)), ('B0I_P7_M5_16h', 'B0I', '4th(m5_best)', Fraction(16)))
BASE_SPEC = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
                 path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json', selector=['spec'])
FIXTURE = dict(fixture='ARTIFICIAL_COMPLETE_FIXTURE_NOT_H4_DATA')


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--b-root', required=True); ap.add_argument('--packets-commit'); ap.add_argument('--dry-run-packets-dir')
    args = ap.parse_args(); root = Path(args.b_root).resolve()
    sys.path[:0] = [str(root / 'src'), str(root / 'tests')]
    from trottertracks import rpe_total_binding as T
    api = T.C.interfaces(str(root)).pipeline; deps = api.providers(str(root))
    H = Fraction(.1); out = []
    for label, method, formula, width in LABELS:
        if args.dry_run_packets_dir:   # unpublished local files: the pinned-SOURCE comparison is mocked
            packet = json.loads((Path(args.dry_run_packets_dir) / ('STAGED_SIGNAL_INPUT_%s.json' % label)).read_text())
            source = dict(fixture='DRY_RUN_LOCAL_STAGED_FILE')
        else:
            source = dict(commit=args.packets_commit, path='%s/STAGED_SIGNAL_INPUT_%s.json' % (OUT, label))
            packet = T._selected(root, source)
        spec = deps.D.full_pf_spec(T._selected(root, BASE_SPEC), formula=formula,
                                   delta_pf=dict(numerator=(width * H).numerator, denominator=(width * H).denominator), M=12, rank=10)
        if method == 'B0I':
            spec['L_D'] = 7
            spec['split'].update(L_D=7, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
            spec['split']['retained_hamiltonian'].update(construction='B0I', prefix=7)
        config = T.B.generate(spec)
        row = dict(candidate=label, config_id_equal=config['id'] == packet['configuration_id'])
        bridge = T.create(root, config, original_source=dict(fixture='A_REGENERATED_SHORTLIST_CONFIGURATION'))
        try:
            with patch.object(T, '_selected', return_value=packet) if 'fixture' in source else _null():
                T.bind_signal(root, bridge, packet, original_packet_source=source)
            row['real_staged_packet'] = 'ACCEPTED (unexpected)'
        except ValueError as error:
            row['real_staged_packet'] = 'REFUSED: ' + str(error)[:160]
        variant = deepcopy(packet)
        for mean in variant['means']:
            if mean['raw'] is None:
                mean['source'] = deepcopy(packet['source'])
        try:
            with patch.object(T, '_selected', return_value=variant):
                T.bind_signal(root, bridge, variant, original_packet_source=FIXTURE)
            row['missing_rounds_with_packet_source'] = 'ACCEPTED (unexpected)'
        except Exception as error:
            row['missing_rounds_with_packet_source'] = 'REFUSED: %s: %s' % (type(error).__name__, str(error)[:160])
        request = api.plan(str(root), config, target=packet['target'], signal_source=packet['source'])
        fixture = deepcopy(packet); filled = []
        for mean in fixture['means']:
            if mean['raw'] is None:
                ctx = api.price_context(request, mean['m']); filled.append(mean['m'])
                mean.update(physical_candidate=deepcopy(ctx['physical_candidate']), source=deepcopy(packet['source']),
                            raw=dict(real=.25, imag=-.5), corrected=dict(real=.25, imag=-.5),
                            normalization=dict(log_B=0., B=1., attenuation=1.), status='ARTIFICIAL_FIXTURE',
                            provider_binding=dict(physical_context=ctx, numerical_source=deepcopy(packet['source'])))
        try:
            with patch.object(T, '_selected', return_value=fixture):
                view = T.bind_signal(root, bridge, fixture, original_packet_source=FIXTURE)
            kept = [m for m in view['signal_packet']['means'] if m['m'] not in filled]
            row['artificial_complete_fixture'] = 'ACCEPTED'
            row['staged_rounds_kept_unchanged'] = all(
                a['raw'] == b['raw'] and a['reference'] == b['reference'] and a['normalization'] == b['normalization']
                for a, b in zip(kept, [m for m in packet['means'] if m['m'] not in filled]))
            row['new_canonical_configuration'] = view['configuration']['id']
            row['science_calls'] = view['science_calls']
        except Exception as error:
            row['artificial_complete_fixture'] = 'REFUSED: %s: %s' % (type(error).__name__, str(error)[:200])
        row['artificial_rounds'] = filled
        out.append(row)
    print(json.dumps(dict(B_commit='9c0e12d87bf83429e3411d295b16c47d913283d2', packets_commit=args.packets_commit, rows=out), indent=1))
    ok = all(r['config_id_equal'] and r['real_staged_packet'].startswith('REFUSED')
             and r['missing_rounds_with_packet_source'].startswith('REFUSED') and r['artificial_complete_fixture'] == 'ACCEPTED'
             and r.get('staged_rounds_kept_unchanged') for r in out)
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
