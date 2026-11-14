"""Signal coverage per physical candidate x round x axis for the five H4 methods (metadata only).

Reads pinned Git blobs (existing coverage/readiness files and saved round records). It does
NOT compute any new mean, spectrum, normalizer, MSE, N, price or G. Saved values are only
compared for identity (same physical ID, same inputs, agreement of already-saved numbers).

    python -B docs/research/a_signal_candidate_identity_coverage_20261114/BUILD_SIGNAL_COVERAGE.py --check
writes SIGNAL_COVERAGE.csv and SAVED_ROUND_AUDIT.json next to this file (fails on a broken invariant).
"""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SRC = dict(
    A_SIGNAL='714247619afb4714fd663eb15c0470ef32bdc7af',        # PR-DF/Pauli packets, Full PF H_eff scan + follow-up
    A_S2_01='69ee0ff8898960d5f9959af8d34d89f223caec87',         # Full PF S2/.1 H_eff pilot (m0/m12 direct means)
    TRUNC='5fb2c32e39e63267ea1748861269c362a93a1b86',           # B0I p7 S2/.1 + Full-H R448 packets, T2 cells
    S1='ce6aacfaca8249c510baa2d958849cffc7649cdc',              # Full-H kappa2 S-1 packet
    P1='a04470c0ad489eea1798856e648f863f9092cfb1',              # Full-H kappa2 P-1 price (revalidated)
    B0='8502dd33ad16dc012fb3558668ef3e4cd8a4405c',              # Pauli LD1 m0 B0 price (raw commit)
    B_READY='99930491819cabd94609c42005a8637ac06a787d',         # B READINESS / SIGNAL_APPLICABILITY
    C_ADMIT='855573064b2c17cfe20eb94a01ece1a554141b0b',         # C CELL_PLAN_V2, PR_DF_COVERAGE
    C_BASE='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',          # C 39-round baseline manifest (Full PF S2/.1 plan)
    SCREEN='699178305cec241b83220f6c307c65e0539a7856',          # four-method PF screen run02 (m6 saved, m12 gate STOP)
    TA_LONG='58b21d596ff8be94d436c5c08e14e2a8c6310342',                                        # Track A long-time signal_records.json.gz
    TA_COMMON='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c',                                      # Track A pf_common H4 signal_records.json.gz
)
FORM = {'S2': '2nd', 'Y4': '4th', 'M5': '4th(m5_best)'}
SHORTLIST = ('S2_16h_over_7', 'Y4_16h_over_3', 'M5_16h')


def blob(c, p):
    return subprocess.check_output(['git', 'show', c + ':' + p], cwd=ROOT)


def js(c, p):
    return json.loads(blob(c, p))


def ls(c, p):
    return subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', c, p], cwd=ROOT, text=True).split()


def cid(system, method, formula, prefix, T, q, R=None, K=None):
    """Track A / D physical candidate id (h46_long_time_signal.cid)."""
    return hashlib.sha256(json.dumps((system, method, formula, prefix, T, q, R, K)).encode()).hexdigest()[:20]


# ---------------------------------------------------------------- saved deterministic PF round records

def saved_pf_records():
    recs = []

    def add(src, c, raw, note, gate=None):
        calc = cid(c['system'], c['method'], c['formula'], c['prefix'], c['T'], c['q'], c.get('R_outer'), c.get('K'))
        if c.get('id') and c['id'] != calc:
            raise ValueError('Saved physical id differs from its own fields: ' + src)
        recs.append(dict(source=src, physical_id=calc, method=c['method'], prefix=c['prefix'], formula=c['formula'],
                         T=c['T'], q=c['q'], raw=raw, note=note, gate=gate))
    for c, base, note in ((SRC['A_S2_01'], 'docs/research/h4_full_pf_heff_results_20261113/run01', 'A_HEFF_PILOT_DIRECT_MEAN'),
                          (SRC['A_SIGNAL'], 'docs/research/h4_full_pf_heff_scan_results_20261114/run01', 'A_HEFF_SCAN_DIRECT_MEAN'),
                          (SRC['A_SIGNAL'], 'docs/research/h4_full_pf_followup_results_20261114/run01', 'A_FOLLOWUP_DIRECT_MEAN')):
        for p in ls(c, base):
            if '/round_' in p:
                d = js(c, p)
                add('%s:%s' % (c, p), d['physical_candidate'], d.get('direct_PF_mean'), note, d.get('gate_pass'))
    for p in ls(SRC['TRUNC'], 'docs/research/truncated_pf_t2_results_20261114/run01'):
        if '/round_' in p:
            d = js(SRC['TRUNC'], p)
            add('%s:%s' % (SRC['TRUNC'], p), d['physical_candidate'], d.get('direct_PF_mean'), 'T2_TWO_TIME_DIRECT_MEAN', d.get('gate_pass'))
    for p in ls(SRC['SCREEN'], 'docs/research/h4_four_method_signal_screen_results_20261113/run02'):
        if p.endswith('_m6.json'):
            d = js(SRC['SCREEN'], p)
            add('%s:%s' % (SRC['SCREEN'], p), d['candidate'], d['signal'].get('raw'), 'SCREEN_RUN02_M6_RUN_LEVEL_STOP',
                d.get('numerical', {}).get('gate_pass') if isinstance(d.get('numerical'), dict) else None)
    lt = json.loads(gzip.decompress(blob(SRC['TA_LONG'], 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz')))
    for k, v in lt.items():
        c = v['candidate']
        if c['system'] == 'H4' and c['method'] in ('FULL', 'B0I'):
            add('%s:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz#%s' % (SRC['TA_LONG'], k),
                c, v['signal'].get('raw'), 'TRACK_A_LONG_TIME_MEAN')
    pfc = json.loads(gzip.decompress(blob(SRC['TA_COMMON'], 'artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/signal_records.json.gz')))
    for k, v in pfc.items():
        _, method, prefix, q, formula = k.split('_', 4)
        if method in ('FULL', 'B0I'):   # T=0.8 (Track A long-time cites these records as its T=0.8 assets)
            add('%s:artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/signal_records.json.gz#%s' % (SRC['TA_COMMON'], k),
                dict(system='H4', method=method, prefix=int(prefix), q=int(q), formula=formula, T=0.8), v['signal'].get('raw'),
                'TRACK_A_PF_COMMON_T08_MEAN')
    return recs


def agreement(hits):
    vals = [complex(h['raw']['real'], h['raw']['imag']) for h in hits if h.get('raw')]
    return None if len(vals) < 2 else max(abs(a - b) for a in vals for b in vals)


CURRENT = {'A_HEFF_PILOT_DIRECT_MEAN', 'A_HEFF_SCAN_DIRECT_MEAN', 'A_FOLLOWUP_DIRECT_MEAN', 'T2_TWO_TIME_DIRECT_MEAN'}
LEGACY = {'TRACK_A_LONG_TIME_MEAN', 'TRACK_A_PF_COMMON_T08_MEAN', 'SCREEN_RUN02_M6_RUN_LEVEL_STOP'}


def pf_status(hits, packet=False):
    notes = {h['note'] for h in hits}
    if packet:
        return 'S1_SAVED_13ROUND_PACKET_SOURCE_BOUND'
    if notes & CURRENT and all(h['gate'] in (True, None) for h in hits if h['note'] in CURRENT):
        return 'S2_SAVED_ROUND_MEAN_PACKET_BRIDGE_AND_INTAKE_CHECK_NEEDED'
    if notes & LEGACY:
        return 'S2L_SAVED_LEGACY_MEAN_QUALIFICATION_NEEDED'
    return None


# ---------------------------------------------------------------- price-side projections

def cell_plan():
    out = {}
    for x in js(SRC['C_ADMIT'], 'docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json'):
        out[(x['physical_candidate']['id'], x['axis'])] = '%s %s (%s)' % (x['role'], x['action'], x['route'])
    return out


def build(check=False):
    recs = saved_pf_records()
    by = collections.defaultdict(list)
    for r in recs:
        by[r['physical_id']].append(r)
    plan = cell_plan()
    rows, audit = [], []

    def emit(method, cand, conf, m, T, q, width, R, phys, canon, status, sources, agree, spectrum, price, identity, note):
        for axis in ('X', 'Y'):
            rows.append(dict(method=method, candidate=cand, configuration_id=conf, m=m, axis=axis, T=T, q=q,
                             width=width, R_or_events=R, physical_id=phys, canonical_id=canon, signal_status=status,
                             signal_sources=';'.join(sources), saved_value_max_abs_disagreement=agree, spectrum_source=spectrum,
                             price_side=price.get(axis, '') if isinstance(price, dict) else price, signal_price_identity=identity, note=note))

    # --- Full PF shortlist and B0I p7 shortlist (follow-up C_HANDOFF rounds + T2 configurations)
    fu = js(SRC['A_SIGNAL'], 'docs/research/h4_full_pf_followup_results_20261114/C_HANDOFF.json')
    t2 = {(r['formula'], r['requested_delta_over_h']): r['configuration_id'] for r in csv.DictReader(io.StringIO(
        blob(SRC['TRUNC'], 'docs/research/truncated_pf_t2_results_20261114/CONFIGURATIONS.csv').decode())) if r['prefix'] == '7'}
    widths = {'S2_16h_over_7': '16/7', 'Y4_16h_over_3': '16/3', 'M5_16h': '16'}
    for c in fu['configurations']:
        if c['label'] not in SHORTLIST:
            continue
        f = FORM[c['label'][:2]]
        for r in c['rounds']:
            m = r['m']; T = 0.1 * 2 ** m; q = r['q_PF']; pc = r['physical_candidate']
            assert pc['id'] == cid('H4', 'FULL', f, 10, T, q), (c['label'], m)
            for method, prefix, conf in (('FULL', 10, c['configuration_id']), ('B0I', 7, t2[(f, widths[c['label']])])):
                phys = cid('H4', method, f, prefix, T, q)
                hits = by.get(phys, [])
                status = pf_status(hits) or 'S3_SAVED_STEP_SPECTRUM_NEW_MEAN_NOT_EXECUTED'
                spec = (json.dumps(r['effective_spectrum_source'], sort_keys=True) if method == 'FULL'
                        else 'T2 %s cell_p7_%s_* SPECTRUM.npz (step %s)' % (SRC['TRUNC'][:9], f.replace('(', '_').replace(')', ''), r['step_cell']))
                price = {a: plan.get((phys, a), 'NO_PLANNED_PRICE_CELL (model/extrapolation)') for a in ('X', 'Y')}
                label = ('FULL_PF ' if method == 'FULL' else 'TRUNCATED_B0I_P7 ') + c['label']
                emit(method, label, conf, m, T, q, r['step_cell'], None, phys, r['canonical_candidate_id'] if method == 'FULL' else '',
                     status, ['%s [%s]' % (h['source'], h['note']) for h in hits], agreement(hits),
                     spec if status.startswith('S3') else '', price, 'PHYSICAL_ID_EQUAL' if any(v != 'NO_PLANNED_PRICE_CELL (model/extrapolation)' for v in price.values()) else 'NO_PRICE_CELL', '')
                audit.append(dict(candidate=label, m=m, physical_id=phys, status=status, hits=hits, agreement=agreement(hits)))

    # --- Full PF S2/.1 (C 39-round manifest, never executed) and B0I p7 S2/.1 (saved packet)
    man = js(SRC['C_BASE'], 'docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
    fpf = next(c for c in man['candidates'] if c['label'] == 'FULL_PF_S2_D01')
    ready = js(SRC['B_READY'], 'docs/research/rpe_required_n_cross_method_preparation_20261114/READINESS.json')
    series = {s['series']: s for s in ready['series']}
    for r in fpf['rounds']:
        m = r['m']; pc = r['physical_candidate']; phys = pc['id']; hits = by.get(phys, [])
        status = pf_status(hits) or 'S3_SAVED_STEP_SPECTRUM_NEW_MEAN_NOT_EXECUTED'
        emit('FULL', 'FULL_PF S2_0.1 (C manifest FULL_PF_S2_D01)', fpf['configuration'].get('id', ''), m, pc['T'], pc['q'], '2nd @ 1h', None,
             phys, '', status, ['%s [%s]' % (h['source'], h['note']) for h in hits], agreement(hits),
             'A 69ee0ff8 h4_full_pf_heff_results_20261113/run01 (2nd @ 1h)' if status.startswith('S3') else '',
             'm3/m4 D e7dca5f7 receipt metadata EXACT (C c5466946 coverage); others none' if m in (3, 4) else 'NO_PRICE_CELL',
             'NO_PRICE_CELL', '39-round plan c5466946 item; never executed (scope 6306fb3a ran B0I/R448 only)')
        audit.append(dict(candidate='FULL_PF S2_0.1', m=m, physical_id=phys, status=status, hits=hits, agreement=agreement(hits)))
    tr = series['TRUNCATED_PF_P7_B0I']
    for r in tr['rows']:
        phys = r['physical_candidate_id']; hits = by.get(phys, [])
        emit('B0I', 'TRUNCATED_B0I_P7 S2_0.1', tr['original_configuration_id'], r['m'], float(r['T']['numerator']) / r['T']['denominator'],
             r['q_PF'], '2nd @ 1h', None, phys, r['original_canonical_candidate_id'], 'S1_SAVED_13ROUND_PACKET_SOURCE_BOUND',
             ['%s:%s' % (tr['packet_source']['commit'], tr['packet_source']['path'])] + ['%s [%s]' % (h['source'], h['note']) for h in hits],
             agreement(hits), '', 'NO_PRICE_FOR_THIS_CANDIDATE', 'NO_PRICE_CELL', 'B metadata intake PASS (B 99930491)')

    # --- Full-H kappa2 and R448
    p1 = js(SRC['P1'], 'docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/PRICES.json')
    s1 = js(SRC['S1'], 'docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json')
    s1m = {m['m']: m for m in s1['means']}
    k2 = series['FULL_H_KAPPA2']
    for r in k2['rows']:
        m = r['m']
        ident = ''
        price = 'NO_PRICE (P-2 proposal only)'
        if m == 3:
            eq = s1m[3]['identity'] == p1[0]['identity'] and s1m[3]['law_binding'] == p1[0]['law_binding']
            ident = 'IDENTITY_AND_LAW_EQUAL' if eq else 'MISMATCH'
            price = 'P-1 single path direct (a04470c0a), not E[C]'
        emit('FULL_H_RTE', 'FULL_H kappa2 K2 q=2^m', k2['original_configuration_id'], m, float(r['T']['numerator']) / r['T']['denominator'], None,
             None, r['event_count'], r['physical_candidate_id'], r['original_canonical_candidate_id'], 'S1_SAVED_13ROUND_PACKET_SOURCE_BOUND',
             ['%s:%s' % (k2['packet_source']['commit'], k2['packet_source']['path'])], None, '', price, ident or 'NO_PRICE_CELL',
             'B metadata intake PASS (B 99930491)')
    r448 = js(SRC['TRUNC'], 'docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/B_SIGNAL_INPUT.json')
    for mean in r448['means']:
        pc = mean.get('provider_binding', {}).get('physical_context', {}).get('physical_candidate', {})
        emit('FULL_H_RTE', 'FULL_H R448 K2 q=2^m (separate candidate)', r448['configuration_id'], mean['m'], None, pc.get('q'), None,
             pc.get('s') or pc.get('event_count'), pc.get('id', ''), mean['identity']['candidate']['id'],
             'S1_SAVED_13ROUND_PACKET_NO_INTAKE_RECORD',
             ['%s:docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/B_SIGNAL_INPUT.json' % SRC['TRUNC']],
             None, '', 'NO_PRICE', 'NO_PRICE_CELL', 'not a current comparison candidate; never combine with kappa2')

    # --- PR-Pauli
    mh = js(SRC['A_SIGNAL'], 'docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')
    b0 = js(SRC['B0'], 'docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02/PRICES.json')
    for p in mh['packets']:
        for r in p['rounds']:
            ident, price = 'NO_PRICE_CELL', 'MISSING'
            if p['prefix'] == 1 and r['m'] == 0:
                eq = r['identity'] == b0[0]['identity'] and r['law_binding'] == b0[0]['law_binding']
                ident = 'IDENTITY_AND_LAW_EQUAL' if eq else 'MISMATCH'
                price = 'B0 single path direct (00f74f86/8502dd33), not E[C]'
            emit('PR_PAULI', 'PR-Pauli LD%d' % p['prefix'], p['configuration']['id'], r['m'], None, None, None, None,
                 r['candidate_id'], r['candidate_id'], 'S1_SAVED_13ROUND_PACKET_SOURCE_BOUND',
                 ['%s:docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json#prefix%d/m%d' % (SRC['A_SIGNAL'], p['prefix'], r['m'])],
                 None, '', price, ident, 'canonical id = physical id (C Pauli bridge); dictionary-bound')

    # --- PR-DF
    cov = {(r['label'], int(r['m']), r['axis']): r for r in csv.DictReader(io.StringIO(
        blob(SRC['C_ADMIT'], 'docs/research/truncated_p7_admission_pr_df_evidence_20261114/PR_DF_COVERAGE.csv').decode()))}
    for ld in ('LD1', 'LD2'):
        s = js(SRC['A_SIGNAL'], 'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_%s.json' % ld)
        for mean in s['means']:
            c = mean['identity']['candidate']; m = mean['m']
            price = {}
            ident = 'NO_PRICE_CELL'
            for a in ('X', 'Y'):
                row = cov.get(('PR-' + ld, m, a))
                if row:
                    if row['physical_id'] != c['source_candidate_id']:
                        ident = 'PHYSICAL_MISMATCH'
                    elif ident != 'PHYSICAL_MISMATCH':
                        ident = 'PHYSICAL_ID_EQUAL'
                    price[a] = 'direct_path=%s cohort=%s E[S]=%s tier=%s' % (row['saved_direct_path_price'], row['finite_cohort_available'],
                                                                       row['structural_expectation_available'], row['G_paper_tier'])
            emit('PR_DF', 'PR-DF ' + ld, s['configuration_id'], m, None, c['q'], None, c['R_outer'], c['source_candidate_id'], c['id'],
                 'S1B_SAVED_13ROUND_PACKET_TOTAL_VIEW_BRIDGE_PENDING',
                 ['%s:docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_%s.json#m%d' % (SRC['A_SIGNAL'], ld, m)],
                 None, '', price, ident, 'B: versioned decoder-only bridge needed (fixed grid A_FIXED_GRID_510FBACD_V1)')
    if check:
        assert len(rows) == 2 * (13 * 3 * 2 + 13 + 13 + 13 + 13 + 13 * 2 + 13 * 2), len(rows)
        assert all(r['signal_price_identity'] != 'MISMATCH' and r['signal_price_identity'] != 'PHYSICAL_MISMATCH' for r in rows)
        assert not [r for r in rows if r['signal_status'] is None]
    return rows, audit, recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); args = ap.parse_args()
    rows, audit, recs = build(args.check)
    with open(HERE / 'SIGNAL_COVERAGE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = collections.Counter((r['candidate'], r['signal_status']) for r in rows if r['axis'] == 'X')
    (HERE / 'SAVED_ROUND_AUDIT.json').write_text(json.dumps(dict(
        kind='SAVED_DETERMINISTIC_PF_ROUND_AUDIT_METADATA_ONLY', sources=SRC, indexed_saved_records=len(recs),
        new_means=0, new_spectra=0, numerical_reanalysis=False,
        status_counts_per_candidate=[dict(candidate=k[0], status=k[1], rounds=v) for k, v in sorted(summary.items())],
        rounds=[dict(a, hits=[dict(source=h['source'], note=h['note'], gate=h['gate']) for h in a['hits']]) for a in audit]),
        indent=1, default=str) + '\n')
    for (cand, status), n in sorted(summary.items()):
        print('%-48s %-58s %2d' % (cand, status, n))


if __name__ == '__main__':
    main()
