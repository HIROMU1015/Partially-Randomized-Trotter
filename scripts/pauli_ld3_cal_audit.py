#!/usr/bin/env python3
"""A stage 3: S0-r1 step audit of the LD3 Q8R660 cell and the LD3 26-axis reference price.

Rules: docs/research/a_pauli_ld3_cal_price_20261116/AUDIT_PRICE_RULES.md (fixed before any
LD3 QPY exists). The per-path audit is the unchanged S0-r1 function
``pauli_ld2_step_audit.audit_path`` (5a18b792a: P1 binding, ordered ancilla-CX matching with
identity pairs excluded, barrier cuts, X/Y token equality, final run). This script builds the
inputs from the committed raw receipts, derives the per-wire deterministic parts (entry F/P,
suffix S, end E) from the two LD3 paths, and computes

    C_ref(m, a) = A3 + H3 q_m + E3 q_m R_m,
    A3 = F + S + E + final_run - (P + S),  H3 = P + S  (all RZ, summed over wires),
    E3 = (sum of the 14 later-step body RZ: steps 2..8 of each path, X axis) / (14 * 660).

At most 8 QPY (2 paths x X/Y x wrapper/compiled) are read; no build, sampling, transpile,
compile, fit, E[C], CI or G.

    python -B scripts/pauli_ld3_cal_audit.py --build <raw commit>
    python -B scripts/pauli_ld3_cal_audit.py --run
    python -B scripts/pauli_ld3_cal_audit.py --analyze
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A  # noqa: E402  (S0-r1 rule functions, unchanged)

OUT = ROOT / 'docs/research/a_pauli_ld3_cal_price_20261116'
RUN = OUT / 'audit_run01'
WORK = RUN / 'work'
R1_SOURCE = '5a18b792a11e78931ddf7d458eeff1aa8edaac5c'
CONFIG = dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb',
              path='docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json')
CONFIG_ID = 'dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847'
TOTAL_VIEW_ID = '6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb'
TOTAL_VIEW_SOURCE = dict(commit='2de33a4f4', path='docs/research/pauli_ld3_vp_q3_results_20261116/HANDOFF.md')
LD2_TREF = dict(commit='a7eceecba', path='docs/research/pauli_ld2_tref_20261115/RESULT.json', selector=['T_ref'])
VP = [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]
VP_SOURCE = dict(commit='2de33a4f4', path='docs/research/pauli_ld3_vp_q3_results_20261116/REPORT.md',
                 allocation='USER_SUPPLIED_VP_X_AND_Y_EACH')
Q, R, PATHS = 8, 660, 2
LATER_INTERVALS = PATHS * (Q - 1)          # 14
MAX_QPY_READS = 8


def exact_json(x):
    x = Fraction(x)
    return dict(numerator=x.numerator, denominator=x.denominator)


# ---------------------------------------------------------------- inputs (metadata + stat only)
def build(raw_commit):
    base = 'docs/research/a_pauli_ld3_cal_price_20261116/run01'
    receipts = A.git_json(raw_commit, base + '/COMPILE_RECEIPTS.json')
    result = A.git_json(raw_commit, base + '/RESULT.json')
    binding = A.git_json(raw_commit, 'docs/research/a_pauli_ld3_cal_price_20261116/BINDING.json')
    if result['status'] != 'FINITE_LD3_CAL_PATHS_COMPLETE' or len(receipts) != 4:
        raise A.AuditStop('RAW_NOT_COMPLETE')
    fingerprints = {s['index']: {('X' if k == 'cosine' else 'Y'): v for k, v in s['result']['fingerprints'].items()}
                    for s in result['summaries']}
    paths = {}
    for sel, rec in sorted(receipts.items()):
        axis, index, req = rec['axis'], rec['index'], rec['request']
        if (req != binding['request'] or (req['q'], req['R'], req['prefix']) != (Q, R, 3)
                or rec['cohort'] != binding['cohort'] or rec['seeds'] != binding['step_seeds'][index]):
            raise A.AuditStop('LD3_RECEIPT_BINDING')
        fp = rec['record']['fingerprint']
        if fingerprints[index][axis]['fingerprint'] != fp:
            raise A.AuditStop('LD3_FINGERPRINT_BINDING')
        files = {}
        for f in rec['artifacts_private'] + [dict(path=rec['native_cache']['path'], sha256=rec['native_cache']['sha256'])]:
            p = Path(f['path'])
            kind = ('wrapper' if f['path'].endswith('_wrapper.qpy.gz') else 'compiled' if f['path'].endswith('_compiled.qpy.gz')
                    else 'cache_json' if f['path'].endswith(fp + '.json') else None)
            if kind:
                files[kind] = dict(path=f['path'], sha256=f['sha256'], bytes=f.get('bytes'), exists=p.exists(),
                                   size_on_disk=p.stat().st_size if p.exists() else None)
        entry = dict(fingerprint=fp, cache_reused=fingerprints[index][axis]['cache_reused'],
                     metrics={k: int(rec['record']['metrics'][k]) for k in A.METRICS},
                     receipt=dict(commit=raw_commit, path=base + '/COMPILE_RECEIPTS.json', selector=[sel]), files=files)
        pid = f'Q8R660/p{index}'
        common = dict(path_id=pid, kind='LD3_CALIBRATION', cell='Q8R660', index=index, m=None, q=req['q'], R=req['R'],
                      T_hex=float(req['T']).hex(), physical_id=req['candidate_id'], cohort=rec['cohort'],
                      composed_of=None, acceptance=raw_commit)
        rec_p = paths.setdefault(pid, dict(axes={}, **common))
        rec_p['axes'][axis] = entry
    reads = sum(1 for p in paths.values() for a in p['axes'].values() for k in ('wrapper', 'compiled') if k in a['files'])
    if len(paths) != PATHS or reads > MAX_QPY_READS or any(sorted(p['axes']) != ['X', 'Y'] for p in paths.values()):
        raise A.AuditStop('LD3_INPUT_COUNT')
    config = A.git_json(CONFIG['commit'], CONFIG['path'])
    if config['id'] != CONFIG_ID:
        raise A.AuditStop('LD3_CONFIGURATION')
    scalar = binding['scalar']['value']
    data = dict(schema='A_PAULI_LD3_CAL_AUDIT_INPUTS_V1',
                rules='docs/research/a_pauli_ld3_cal_price_20261116/AUDIT_PRICE_RULES.md',
                rule_functions=dict(commit=R1_SOURCE, path='scripts/pauli_ld2_step_audit.py'),
                raw=raw_commit, configuration=dict(CONFIG, id=CONFIG_ID),
                scalar=dict(value_hex=float(scalar).hex(), value=scalar, source=dict(commit=raw_commit,
                            path='docs/research/a_pauli_ld3_cal_price_20261116/BINDING.json', selector=['scalar', 'value'])),
                round_T={str(r['m']): r['T'] for r in config['rounds']},
                qpy_reads=dict(total=reads, limit=MAX_QPY_READS),
                paths=[paths[k] for k in sorted(paths)], private_contents_read=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'INPUTS_AUDIT.json').write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps(dict(paths=len(paths), reads=reads, scalar=scalar,
                          missing=[f['path'] for p in paths.values() for a in p['axes'].values()
                                   for f in a['files'].values() if not f['exists']])))


# ---------------------------------------------------------------- run (unchanged S0-r1 audit_path)
def run(workers=2):
    A.check_runtime()
    A.RUN, A.WORK = RUN, WORK          # redirect outputs only; rule functions are unchanged
    RUN.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'INPUTS_AUDIT.json').read_text())
    scalar = float.fromhex(data['scalar']['value_hex'])
    started = time.monotonic(); outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(A.run_path, (p, scalar)): p['path_id'] for p in data['paths']}
        for future in as_completed(futures):
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                outcomes.append(dict(path_id=futures[future], status='WORKER_FAILED', reason=repr(exc)))
            print(json.dumps(outcomes[-1])[:300], flush=True)
    (RUN / 'RUN_ALL.json').write_text(json.dumps(dict(workers=workers, elapsed_seconds=time.monotonic() - started,
        outcomes=sorted(outcomes, key=lambda x: x['path_id'])), indent=1) + '\n')
    return 0 if all(o['status'] == 'DONE' for o in outcomes) else 2


# ---------------------------------------------------------------- deterministic parts and coefficients
def edges(arrays, q):
    """Per-wire S (common suffix of all segments), F/P (common entry of first/later segments), E."""
    import numpy as np
    pids = sorted(arrays); n = len(arrays[pids[0]])
    S = {w: A.lcs_len([arrays[p][w]['segs'][s][0] for p in pids for s in range(q)]) for w in range(n)}
    first = {w: [arrays[p][w]['segs'][0][0][:len(arrays[p][w]['segs'][0][0]) - S[w]] for p in pids] for w in range(n)}
    later = {w: [arrays[p][w]['segs'][s][0][:len(arrays[p][w]['segs'][s][0]) - S[w]] for p in pids for s in range(1, q)]
             for w in range(n)}
    F = {w: A.lcp_len(first[w]) for w in range(n)}
    P = {w: A.lcp_len(later[w]) for w in range(n)}
    ref = arrays[pids[0]]
    rz = lambda arr, a, b: int(arr[1][a:b].sum())
    per_wire = []
    for w in range(n):
        s1, s2 = ref[w]['segs'][0], ref[w]['segs'][1]
        ends = [arrays[p][w]['end'][0] for p in pids]
        per_wire.append(dict(wire=w, ancilla=w == n - 1, suffix_len=S[w], suffix_rz=rz(s1, len(s1[0]) - S[w], len(s1[0])),
                             first_prefix_len=F[w], first_prefix_rz=rz(s1, 0, F[w]),
                             later_prefix_len=P[w], later_prefix_rz=rz(s2, 0, P[w]),
                             end_len=len(ref[w]['end'][0]), end_rz=int(ref[w]['end'][1].sum()),
                             end_identical_all_paths=all(len(e) == len(ends[0]) and np.array_equal(e, ends[0]) for e in ends)))
    const = dict(first_prefix_rz=sum(e['first_prefix_rz'] for e in per_wire),
                 later_prefix_rz=sum(e['later_prefix_rz'] for e in per_wire),
                 suffix_rz=sum(e['suffix_rz'] for e in per_wire), end_rz=sum(e['end_rz'] for e in per_wire))
    return dict(per_wire=per_wire, constants=const), S, F, P


def bodies(arrays, q, S, F, P):
    """Body RZ of every segment: segment minus its entry (F or P) and suffix S, summed over wires."""
    out = {}
    for pid, wires in arrays.items():
        rows = []
        for s in range(q):
            pre = F if s == 0 else P
            seg = body = 0
            for w, wa in enumerate(wires):
                h, rz, _ = wa['segs'][s]
                cut = len(h) - S[w]
                seg += int(rz.sum()); body += int(rz[pre[w]:cut].sum())
            rows.append(dict(step=s + 1, context='FIRST' if s == 0 else 'LATER', segment_rz=seg, body_rz=body))
        out[pid] = rows
    return out


def coefficients(const, final_run_rz, segment_bodies):
    """A3/H3 from directly counted deterministic parts; E3 = later bodies / (14 * 660)."""
    later = [r['body_rz'] for rows in segment_bodies.values() for r in rows if r['context'] == 'LATER']
    if len(later) != LATER_INTERVALS:
        raise ValueError('Exactly 14 later-step intervals required')
    A3 = const['first_prefix_rz'] + const['suffix_rz'] + const['end_rz'] + final_run_rz - (const['later_prefix_rz'] + const['suffix_rz'])
    H3 = const['later_prefix_rz'] + const['suffix_rz']
    E3 = Fraction(sum(later), LATER_INTERVALS * R)
    return dict(A3=A3, H3=H3, E3=E3, later_body_sum=sum(later), intervals=len(later), R=R,
                A3_parts=dict(first_entry_F=const['first_prefix_rz'], suffix_S=const['suffix_rz'], end_E=const['end_rz'],
                              final_run=final_run_rz, minus_per_step_H3=H3))


def price_table(config, coef, vp=VP):
    """26 axes C_ref = A3 + H3 q + E3 q R (exact) from the fixed schedule; VP-weighted total."""
    rows = []; total = Fraction(0)
    for r in config['rounds']:
        m, q, Rm = r['m'], r['q_PF'], r['R_outer']
        if r['candidate']['q'] != q or r['candidate']['R_outer'] != Rm:
            raise ValueError('Schedule fields disagree')
        c = coef['A3'] + coef['H3'] * q + coef['E3'] * q * Rm
        for axis in ('X', 'Y'):
            n = vp[m]; total += n * c
            rows.append(dict(m=m, axis=axis, physical_id=r['candidate']['id'], q=q, R=Rm, events=q * Rm, N=n,
                             value_source='MODEL_APPLICATION_FROM_Q8R660_DECOMPOSITION',
                             C_ref_exact=str(c), C_ref_display=float(c), N_times_C_ref_exact=str(n * c),
                             N_times_C_ref_display=float(n * c)))
    return rows, total


# ---------------------------------------------------------------- analysis, rule support, price
def analyze():
    import numpy as np
    data = json.loads((OUT / 'INPUTS_AUDIT.json').read_text())
    A.RUN, A.WORK = RUN, WORK
    scalar = float.fromhex(data['scalar']['value_hex'])
    config = A.git_json(CONFIG['commit'], CONFIG['path'])
    results, stops = {}, []
    for p in data['paths']:
        f = RUN / ('PATH_' + p['path_id'].replace('/', '_') + '.json')
        if f.exists():
            results[p['path_id']] = json.loads(f.read_text())
        else:
            stops.append(p['path_id'])
    ok = sorted(pid for pid, r in results.items() if all(r['axes'][a].get('P1') == 'PASS'
                and str(r['axes'][a].get('P2', '')).startswith('PASS') for a in ('X', 'Y')))
    checks = dict(P1_all=len(results) == PATHS and all(r['axes'][a].get('P1') == 'PASS' for r in results.values() for a in ('X', 'Y')),
                  P2_all=len(ok) == PATHS,
                  events_equal_R=all(results[p]['axes'][a].get('events_equal_R') for p in ok for a in ('X', 'Y')),
                  completeness=all(results[p]['axes'][a].get('completeness') for p in ok for a in ('X', 'Y')))
    out = dict(schema='A_PAULI_LD3_CAL_AUDIT_RESULT_V1', stops=stops, paths_ok=ok, checks=checks, rule_supported=False)
    price = dict(schema='A_PAULI_LD3_REFERENCE_PRICE_V1', status='STOP_RULE_NOT_SUPPORTED', configuration_id=CONFIG_ID,
                 configuration_source=CONFIG, total_view=dict(id=TOTAL_VIEW_ID, source=TOTAL_VIEW_SOURCE),
                 rule='C_ref(m,a) = A3 + H3*q_m + E3*q_m*R_m', coefficients=None, rows=None, calibration_cell=None,
                 VP=dict(N=VP, source=VP_SOURCE), T_LD3=None, LD2=None, kind='MODEL_REFERENCE_PRICE_NOT_E_C',
                 E_C=None, CI=None, G=None, winner=None)
    if len(ok) == PATHS:
        arrays = {pid: A.load_x_arrays(pid, Q, results[pid]['n_wires']) for pid in ok}
        edge, S, F, P = edges(arrays, Q)
        segs = bodies(arrays, Q, S, F, P)
        # P3 (a)-(e) as S0-r1, on the two LD3 paths
        sigs = {st['signature'] for pid in ok for a in ('X', 'Y') for st in results[pid]['axes'][a]['steps']}
        barriers = {json.dumps(results[pid]['axes'][a]['barrier']) for pid in ok for a in ('X', 'Y')}
        digests = {m['ancilla_D_region_digest'] for pid in ok for m in results[pid]['axes']['X']['markers']}
        n = len(edge['per_wire'])
        cover = []
        for w in range(n):
            need = max(results[pid]['axes']['X']['markers'][s]['from_second_last_block'][w] for pid in ok for s in range(Q))
            need_all = max(results[pid]['axes']['X']['markers'][s]['from_first_block'][w] for pid in ok for s in range(Q))
            cover.append(dict(wire=w, suffix_len=S[w], need_from_second_last_block=need, need_from_first_block=need_all,
                              covers_second_last_block=S[w] >= need, covers_first_block=S[w] >= need_all))
        blocks = {st['d_blocks'] for pid in ok for a in ('X', 'Y') for st in results[pid]['axes'][a]['steps']}
        P3 = dict(uncompiled_D_region_identical=len(sigs) == 1, barrier_identical=len(barriers) == 1,
                  compiled_ancilla_D_region_identical=len(digests) == 1,
                  suffix_covers_last_two_D_blocks_all_wires=all(c['covers_second_last_block'] for c in cover),
                  system_end_edge_identical=all(e['end_identical_all_paths'] for e in edge['per_wire'][:-1]),
                  suffix_covers_all_D_blocks_all_wires_informational=all(c['covers_first_block'] for c in cover),
                  D_blocks_per_step=sorted(blocks),
                  identity_pairs_per_step=sorted({st['identity_pairs'] for pid in ok for a in ('X', 'Y')
                                                  for st in results[pid]['axes'][a]['steps']}))
        P3['status'] = 'PASS' if all(v for k, v in P3.items() if isinstance(v, bool) and not k.endswith('informational')) else 'FAIL'
        # P5 X/Y, wrapper input, final run and the 13 LD3 round margins (relation from S0-r1)
        fr = {pid: {a: results[pid]['axes'][a]['final_run'] for a in ('X', 'Y')} for pid in ok}
        form_ok = all(fr[p][a]['names'] == ['rz', 'sx', 'rz'] for p in ok for a in ('X', 'Y'))
        final_rz = {fr[p][a]['rz'] for p in ok for a in ('X', 'Y')}
        p5_rows, theta = [], {}
        for a in ('X', 'Y'):
            t1 = {fr[p][a]['angles_hex'][0] for p in ok} if form_ok else set()
            t2 = {fr[p][a]['angles_hex'][1] for p in ok} if form_ok else set()
            theta[a] = dict(theta1_hex=sorted(t1), theta2_hex=sorted(t2), unique=len(t1) == 1 and len(t2) == 1)
            T_cal = Fraction(float.fromhex(data['paths'][0]['T_hex']))
            for m, Tm in sorted(data['round_T'].items(), key=lambda x: int(x[0])):
                Tm = Fraction(Tm['numerator'], Tm['denominator'])
                if theta[a]['unique']:
                    t = A.wrap(float.fromhex(next(iter(t1))) - scalar * float(Tm - T_cal))
                    cls = 'MARGIN_OK' if abs(t) >= A.MARGIN_THRESHOLD else 'UNKNOWN_SMALL_MARGIN'
                else:
                    t, cls = None, 'UNKNOWN_FORM_OR_NONUNIQUE'
                p5_rows.append(dict(m=int(m), axis=a, T=str(Tm), theta1_predicted=t, margin=abs(t) if t is not None else None,
                                    final_run=cls, final_run_rz=2 if cls == 'MARGIN_OK' else None))
        P5 = dict(compiled_X_equals_Y_except_final_run=all(results[p]['xy']['compiled_equal_except_final_run'] for p in ok),
                  uncompiled_X_Y_differ_only_by_sdg=all(results[p]['xy']['uncompiled_only_sdg_differs'] for p in ok),
                  wrapper_input_checks=all(results[p]['axes'][a]['wrapper_input']['p_equals_minus_scalar_T']
                                           and results[p]['axes'][a]['wrapper_input']['synthesized_matches_compiled']
                                           for p in ok for a in ('X', 'Y')),
                  final_run_form_rz_sx_rz=form_ok, final_run_rz=sorted(final_rz), theta=theta,
                  relation='FROM_S0_R1_LD2_FIVE_T_VALUES_NOT_RETESTED_AT_ONE_T',
                  margins_all_ok=all(r['final_run'] == 'MARGIN_OK' for r in p5_rows))
        P5['status'] = 'PASS' if all(P5[k] is True for k in ('compiled_X_equals_Y_except_final_run',
                                     'uncompiled_X_Y_differ_only_by_sdg', 'wrapper_input_checks', 'final_run_form_rz_sx_rz',
                                     'margins_all_ok')) and len(final_rz) == 1 and all(t['unique'] for t in theta.values()) else 'FAIL'
        fr_rz = next(iter(final_rz)) if len(final_rz) == 1 else None
        # P6 completeness and the exact decomposition for every path and axis
        decomposition = {}
        coef = coefficients(edge['constants'], fr_rz, segs) if fr_rz is not None else None
        for pid in ok:
            body_sum = sum(r['body_rz'] for r in segs[pid])
            for a in ('X', 'Y'):
                saved = next(p for p in data['paths'] if p['path_id'] == pid)['axes'][a]['metrics']['rz_count']
                restored = coef['A3'] + coef['H3'] * Q + body_sum if coef else None
                decomposition[f'{pid}:{a}'] = dict(saved_rz=saved, restored=restored, exact=restored == saved,
                                                   segments_rz=sum(results[pid]['axes'][a]['segment_rz']),
                                                   end_rz=results[pid]['axes'][a]['end_rz'],
                                                   final_run_rz=fr[pid][a]['rz'], sum_body_rz=body_sum)
        P6 = dict(status='PASS' if decomposition and all(d['exact'] for d in decomposition.values()) else 'FAIL',
                  decomposition=decomposition)
        support = all(checks.values()) and P3['status'] == 'PASS' and P5['status'] == 'PASS' and P6['status'] == 'PASS'
        later = [r['body_rz'] for rows in segs.values() for r in rows if r['context'] == 'LATER']
        mean = Fraction(sum(later), len(later)); var = sum((Fraction(v) - mean) ** 2 for v in later) / (len(later) - 1)
        out.update(P3=P3, P5=P5, P6=P6, edges=edge, coverage=cover, segments=segs,
                   later_body_statistics=dict(n=len(later), mean=float(mean), mean_exact=str(mean), mean_over_R=float(mean / R),
                                              sample_variance=float(var), variance_over_R=float(var / R),
                                              min=min(later), max=max(later), use='DESCRIPTIVE_ONLY_NOT_CI'),
                   rule_supported=support)
        with open(RUN / 'FINAL_RUN.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(p5_rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(p5_rows)
        with open(RUN / 'SEGMENTS.csv', 'w', newline='') as fh:
            rows = [dict(path_id=pid, **r) for pid, rs in segs.items() for r in rs]
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
        (RUN / 'EDGES.json').write_text(json.dumps(dict(edge, coverage=cover), indent=1) + '\n')
        if support:
            rows, total = price_table(config, coef)
            ld2 = A.git_json(LD2_TREF['commit'], LD2_TREF['path'])['T_ref']
            T2 = Fraction(ld2['numerator'], ld2['denominator'])
            cell = []
            for pid in ok:
                body_sum = sum(r['body_rz'] for r in segs[pid])
                for a in ('X', 'Y'):
                    d = decomposition[f'{pid}:{a}']
                    cell.append(dict(path_id=pid, axis=a, physical_id=data['paths'][0]['physical_id'], q=Q, R=R,
                                     value_source='DIRECT_COUNT_CALIBRATION_CELL', compiled_RZ=d['saved_rz'],
                                     first_body_rz=segs[pid][0]['body_rz'], later_body_rz_sum=body_sum - segs[pid][0]['body_rz'],
                                     receipt=next(p for p in data['paths'] if p['path_id'] == pid)['axes'][a]['receipt']))
            model_cell = coef['A3'] + coef['H3'] * Q + coef['E3'] * Q * R
            price.update(status='PRICE_RULE_SUPPORTED_MODEL_REFERENCE',
                         coefficients=dict(A3=coef['A3'], H3=coef['H3'], E3=exact_json(coef['E3']), E3_display=float(coef['E3']),
                                           E3_unit='RZ_PER_EVENT', later_body_sum=coef['later_body_sum'],
                                           intervals=coef['intervals'], A3_parts=coef['A3_parts'],
                                           evidence=dict(edges='audit_run01/EDGES.json', segments='audit_run01/SEGMENTS.csv',
                                                         audit='audit_run01/AUDIT_RESULT.json')),
                         rows=rows, calibration_cell=dict(rows=cell, model_value_at_cell=exact_json(model_cell),
                                                          model_value_at_cell_display=float(model_cell),
                                                          note='Cell ID is not the m12 round ID; its receipts are calibration only'),
                         T_LD3=exact_json(total), T_LD3_display=float(total),
                         LD2=dict(T_ref=exact_json(T2), T_ref_display=float(T2), source=LD2_TREF,
                                  method='C2_FROZEN_LINEAR_MODEL_b74e_NOT_THE_S0_R1_DECOMPOSITION'),
                         difference_LD3_minus_LD2=exact_json(total - T2), difference_display=float(total - T2),
                         ratio_LD3_over_LD2=exact_json(total / T2), ratio_display=float(total / T2))
            with open(OUT / 'PRICE_26AXIS.csv', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (OUT / 'PRICE_26AXIS.json').write_text(json.dumps(price, indent=1, default=str) + '\n')
    out.update(compiles=0, sampling=0, builds=0, fits=0, private_writes=0, qpy_reads=data['qpy_reads'])
    (RUN / 'AUDIT_RESULT.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps(dict(checks=checks, P3=(out.get('P3') or {}).get('status'), P5=(out.get('P5') or {}).get('status'),
                          P6=(out.get('P6') or {}).get('status'), supported=out.get('rule_supported'),
                          coefficients=price['coefficients'], T_LD3=price.get('T_LD3_display'),
                          ratio=price.get('ratio_display')), default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.build:
        build(args.build)
    if args.run:
        return run(args.workers)
    if args.analyze:
        analyze()
    return 0


if __name__ == '__main__':
    sys.exit(main())
