#!/usr/bin/env python3
"""S0-r1 step audit of a request-driven split calibration cell and its 26-axis reference price (any prefix).

B2 generalisation of A's ``scripts/pauli_split_cal_audit.py`` at 1097c37f9 (LD5-only SPECS): the spec now comes from the
calibration request that A's ``prepare_pauli_calibration.py`` (7d6a8c385) produced and the driver
``run_pauli_split_cal.py --request`` froze and executed.  The rule functions (S0-r1, ``pauli_ld2_step_audit`` at
5a18b792a) and the LD3 counting functions (``pauli_ld3_cal_audit`` at 158bbcc7b) are reused unchanged (byte-equal copies);
every check (P1, P2, P3, P5 incl. the wrapper margin at all 13 round T for X and Y, P6) and the counting rule are the
same:

    A = F + S + E + final_run - (P + S),   H = P + S,
    E = (sum of the 14 later-step body RZ: steps 2..8 of each path, X axis) / (14 * R).

Changes: spec from --request/--config; coefficient keys A<p>/H<p>/E<p> (what D2's PAULI_AUDIT mode reads); the LD5-vs-LD3
allocation comparison is reported only for prefix 5.  At most 8 QPY read; no build, sampling, transpile, compile, fit.

    python -B scripts/pauli_split_cal_audit_request.py --request CAL.CALIBRATION_REQUEST.json --config COMMIT:PATH --build <raw commit>
    python -B scripts/pauli_split_cal_audit_request.py --request ... --config ... --run
    python -B scripts/pauli_split_cal_audit_request.py --request ... --config ... --analyze
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
sys.set_int_max_str_digits(0)  # B summaries carry exact rationals with long integers
import pauli_ld3_cal_audit as AU  # noqa: E402  (LD3 stage-3 counting functions, unchanged)

LD3_AUDIT = dict(commit='158bbcc7b88883947b726b1f19bf037b3af3ef18', path='scripts/pauli_ld3_cal_audit.py')
ALLOCATIONS = dict(
    VP=dict(N=AU.VP, source=AU.VP_SOURCE,
            LD3_total=dict(commit='f25a05f8a92aab43cfbe30fc80b974054156aac1',
                           path='docs/research/a_pauli_ld3_cal_price_20261116/PRICE_26AXIS.json', selector=['T_LD3'])),
    FULL_PF__B_PRICE=dict(N=[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2],
            source=dict(commit='fb4d78781', path='docs/research/pauli_ld3_pf_budget_q3_results_20261116/INPUTS.json',
                        selector=['cases', 3, 'allocation'], allocation_id='PAULI_LD3_FULL_PF__B_PRICE_VP_CAP_01'),
            LD3_total=dict(commit='fb4d78781', path='docs/research/pauli_ld3_pf_budget_q3_results_20261116/SUMMARY.json',
                           selector=['achieved_conditional_cost'])))
LD3_PRICE = dict(commit='f25a05f8a92aab43cfbe30fc80b974054156aac1', path='docs/research/a_pauli_ld3_cal_price_20261116/PRICE_26AXIS.json')
LD3_CONFIG = AU.CONFIG
SPECS = {}  # filled from the calibration request (see spec_from_request)


def spec_from_request(request_path, config_ref):
    """The audit spec of A's LD5 SPECS entry, built from a frozen calibration request instead of a constant."""
    value = json.loads(Path(request_path).read_text())
    s = value.get('driver_spec', value)
    commit, path = config_ref.split(':', 1)
    cell = s['cell']
    return dict(prefix=int(s['prefix']), cell=cell['cell_id'], q=int(cell['q']), R=int(cell['R']), paths=2, out=s['base'],
                config=dict(commit=commit, path=path), config_id=s['config_id'], signals=None)


def selected(ref):
    data = A.git_json(ref['commit'], ref['path'])
    for key in ref.get('selector', []):
        data = data[key]
    return data


def paths_of(spec):
    out = ROOT / spec['out']
    return out, out / 'audit_run01', out / 'audit_run01' / 'work'


def build(spec, raw_commit):
    base = spec['out'] + '/run01'
    receipts = A.git_json(raw_commit, base + '/COMPILE_RECEIPTS.json')
    result = A.git_json(raw_commit, base + '/RESULT.json')
    binding = A.git_json(raw_commit, spec['out'] + '/BINDING.json')
    if result['status'] != 'FINITE_LD%d_CAL_PATHS_COMPLETE' % spec['prefix'] or len(receipts) != 4:
        raise A.AuditStop('RAW_NOT_COMPLETE')
    fingerprints = {s['index']: {('X' if k == 'cosine' else 'Y'): v for k, v in s['result']['fingerprints'].items()}
                    for s in result['summaries']}
    paths = {}
    for sel, rec in sorted(receipts.items()):
        axis, index, req = rec['axis'], rec['index'], rec['request']
        if (req != binding['request'] or (req['q'], req['R'], req['prefix']) != (spec['q'], spec['R'], spec['prefix'])
                or rec['cohort'] != binding['cohort'] or rec['seeds'] != binding['step_seeds'][index]):
            raise A.AuditStop('RECEIPT_BINDING')
        fp = rec['record']['fingerprint']
        if fingerprints[index][axis]['fingerprint'] != fp:
            raise A.AuditStop('FINGERPRINT_BINDING')
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
        pid = '%s/p%d' % (spec['cell'], index)
        common = dict(path_id=pid, kind='LD%d_CALIBRATION' % spec['prefix'], cell=spec['cell'], index=index, m=None,
                      q=req['q'], R=req['R'], T_hex=float(req['T']).hex(), physical_id=req['candidate_id'], cohort=rec['cohort'],
                      composed_of=None, acceptance=raw_commit)
        paths.setdefault(pid, dict(axes={}, **common))['axes'][axis] = entry
    reads = sum(1 for p in paths.values() for a in p['axes'].values() for k in ('wrapper', 'compiled') if k in a['files'])
    if len(paths) != spec['paths'] or reads > AU.MAX_QPY_READS or any(sorted(p['axes']) != ['X', 'Y'] for p in paths.values()):
        raise A.AuditStop('INPUT_COUNT')
    config = A.git_json(spec['config']['commit'], spec['config']['path'])
    if config['id'] != spec['config_id']:
        raise A.AuditStop('CONFIGURATION')
    scalar = binding['scalar']['value']
    data = dict(schema='A_PAULI_LD%d_CAL_AUDIT_INPUTS_V1' % spec['prefix'],
                rules=spec['out'] + '/AUDIT_PRICE_RULES.md',
                rule_functions=dict(commit=AU.R1_SOURCE, path='scripts/pauli_ld2_step_audit.py'), counting_functions=LD3_AUDIT,
                raw=raw_commit, configuration=dict(spec['config'], id=spec['config_id']),
                scalar=dict(value_hex=float(scalar).hex(), value=scalar,
                            source=dict(commit=raw_commit, path=spec['out'] + '/BINDING.json', selector=['scalar', 'value'])),
                round_T={str(r['m']): r['T'] for r in config['rounds']}, qpy_reads=dict(total=reads, limit=AU.MAX_QPY_READS),
                paths=[paths[k] for k in sorted(paths)], private_contents_read=False)
    out, _, _ = paths_of(spec); out.mkdir(parents=True, exist_ok=True)
    (out / 'INPUTS_AUDIT.json').write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps(dict(paths=len(paths), reads=reads, scalar=scalar)))


def run(spec, workers=2):
    A.check_runtime()
    out, A.RUN, A.WORK = paths_of(spec)
    A.RUN.mkdir(parents=True, exist_ok=True)
    data = json.loads((out / 'INPUTS_AUDIT.json').read_text())
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
    (A.RUN / 'RUN_ALL.json').write_text(json.dumps(dict(workers=workers, elapsed_seconds=time.monotonic() - started,
        outcomes=sorted(outcomes, key=lambda x: x['path_id'])), indent=1) + '\n')
    return 0 if all(o['status'] == 'DONE' for o in outcomes) else 2


def coefficients(const, final_run_rz, segment_bodies, *, paths, q, R):
    """AU.coefficients with this cell's R: A/H from counted parts; E = later bodies / (intervals * R)."""
    intervals = paths * (q - 1)
    later = [r['body_rz'] for rows in segment_bodies.values() for r in rows if r['context'] == 'LATER']
    if len(later) != intervals:
        raise ValueError('Exactly %d later-step intervals required' % intervals)
    H = const['later_prefix_rz'] + const['suffix_rz']
    A_ = const['first_prefix_rz'] + const['suffix_rz'] + const['end_rz'] + final_run_rz - H
    return dict(A=A_, H=H, E=Fraction(sum(later), intervals * R), later_body_sum=sum(later), intervals=intervals, R=R,
                events_in_denominator=intervals * R,
                A_parts=dict(first_entry_F=const['first_prefix_rz'], suffix_S=const['suffix_rz'], end_E=const['end_rz'],
                             final_run=final_run_rz, minus_per_step_H=H))


def weights(config, n):
    """VP weights: shots, steps and events summed over X and Y."""
    w = dict(shots=0, steps=0, events=0)
    for r in config['rounds']:
        k = 2 * n[r['m']]
        w['shots'] += k; w['steps'] += k * r['q_PF']; w['events'] += k * r['q_PF'] * r['R_outer']
    return w


def compare(coef, ld3_coef, config, ld3_config):
    """Per allocation: LD5 and LD3 totals, decomposed into intercept/head/event parts, and differences."""
    out = {}
    for name, alloc in ALLOCATIONS.items():
        w5, w3 = weights(config, alloc['N']), weights(ld3_config, alloc['N'])
        part5 = dict(intercept=coef['A'] * w5['shots'], head=coef['H'] * w5['steps'], event=coef['E'] * w5['events'])
        part3 = dict(intercept=ld3_coef['A'] * w3['shots'], head=ld3_coef['H'] * w3['steps'], event=ld3_coef['E'] * w3['events'])
        T5, T3 = sum(part5.values()), sum(part3.values())
        saved = selected(alloc['LD3_total']); saved = Fraction(saved['numerator'], saved['denominator'])
        if T3 != saved:
            raise ValueError('LD3 coefficients do not reproduce the saved LD3 total: ' + name)
        out[name] = dict(N_per_axis=alloc['N'], allocation_source=alloc['source'], weights_LD5=w5, weights_LD3=w3,
            T_LD5=AU.exact_json(T5), T_LD5_display=float(T5), T_LD3=AU.exact_json(T3), T_LD3_display=float(T3),
            T_LD3_source=alloc['LD3_total'], difference_LD5_minus_LD3=AU.exact_json(T5 - T3), difference_display=float(T5 - T3),
            ratio_LD5_over_LD3=AU.exact_json(T5 / T3), ratio_display=float(T5 / T3),
            parts_LD5={k: float(v) for k, v in part5.items()}, parts_LD3={k: float(v) for k, v in part3.items()},
            part_differences={k: float(part5[k] - part3[k]) for k in part5},
            part_differences_exact={k: str(part5[k] - part3[k]) for k in part5},
            Q3_for_LD5='NOT_EVALUATED')
    return out


def analyze(spec):
    data = json.loads((ROOT / spec['out'] / 'INPUTS_AUDIT.json').read_text())
    out, A.RUN, A.WORK = paths_of(spec)
    scalar = float.fromhex(data['scalar']['value_hex'])
    config = A.git_json(spec['config']['commit'], spec['config']['path'])
    ld3_config = A.git_json(LD3_CONFIG['commit'], LD3_CONFIG['path'])
    q, R, npaths = spec['q'], spec['R'], spec['paths']
    results, stops = {}, []
    for p in data['paths']:
        f = A.RUN / ('PATH_' + p['path_id'].replace('/', '_') + '.json')
        if f.exists():
            results[p['path_id']] = json.loads(f.read_text())
        else:
            stops.append(p['path_id'])
    ok = sorted(pid for pid, r in results.items() if all(r['axes'][a].get('P1') == 'PASS'
                and str(r['axes'][a].get('P2', '')).startswith('PASS') for a in ('X', 'Y')))
    checks = dict(P1_all=len(results) == npaths and all(r['axes'][a].get('P1') == 'PASS' for r in results.values() for a in ('X', 'Y')),
                  P2_all=len(ok) == npaths,
                  events_equal_R=all(results[p]['axes'][a].get('events_equal_R') for p in ok for a in ('X', 'Y')),
                  completeness=all(results[p]['axes'][a].get('completeness') for p in ok for a in ('X', 'Y')))
    res = dict(schema='A_PAULI_LD%d_CAL_AUDIT_RESULT_V1' % spec['prefix'], stops=stops, paths_ok=ok, checks=checks,
               rule_supported=False)
    ld3_price = selected(LD3_PRICE)
    ld3_coef = dict(A=ld3_price['coefficients']['A3'], H=ld3_price['coefficients']['H3'],
                    E=Fraction(ld3_price['coefficients']['E3']['numerator'], ld3_price['coefficients']['E3']['denominator']))
    price = dict(schema='A_PAULI_LD%d_REFERENCE_PRICE_V1' % spec['prefix'], status='STOP_RULE_NOT_SUPPORTED',
                 configuration_id=spec['config_id'], configuration_source=spec['config'], signal_packet=spec['signals'],
                 total_view=None, rule='C_ref(m,a) = A + H*q_m + E*q_m*R_m', coefficients=None, rows=None,
                 calibration_cell=None, comparison=None, LD3_coefficients=dict(A3=ld3_coef['A'], H3=ld3_coef['H'],
                 E3=AU.exact_json(ld3_coef['E']), source=dict(LD3_PRICE, selector=['coefficients'])),
                 kind='MODEL_REFERENCE_PRICE_NOT_E_C', Q3_for_LD5='NOT_EVALUATED', E_C=None, CI=None, G=None, winner=None)
    if len(ok) == npaths:
        arrays = {pid: A.load_x_arrays(pid, q, results[pid]['n_wires']) for pid in ok}
        edge, S, F, P = AU.edges(arrays, q)
        segs = AU.bodies(arrays, q, S, F, P)
        sigs = {st['signature'] for pid in ok for a in ('X', 'Y') for st in results[pid]['axes'][a]['steps']}
        barriers = {json.dumps(results[pid]['axes'][a]['barrier']) for pid in ok for a in ('X', 'Y')}
        digests = {m['ancilla_D_region_digest'] for pid in ok for m in results[pid]['axes']['X']['markers']}
        n = len(edge['per_wire']); cover = []
        for w in range(n):
            need = max(results[pid]['axes']['X']['markers'][s]['from_second_last_block'][w] for pid in ok for s in range(q))
            need_all = max(results[pid]['axes']['X']['markers'][s]['from_first_block'][w] for pid in ok for s in range(q))
            cover.append(dict(wire=w, suffix_len=S[w], need_from_second_last_block=need, need_from_first_block=need_all,
                              covers_second_last_block=S[w] >= need, covers_first_block=S[w] >= need_all))
        P3 = dict(uncompiled_D_region_identical=len(sigs) == 1, barrier_identical=len(barriers) == 1,
                  compiled_ancilla_D_region_identical=len(digests) == 1,
                  suffix_covers_last_two_D_blocks_all_wires=all(c['covers_second_last_block'] for c in cover),
                  system_end_edge_identical=all(e['end_identical_all_paths'] for e in edge['per_wire'][:-1]),
                  suffix_covers_all_D_blocks_all_wires_informational=all(c['covers_first_block'] for c in cover),
                  D_blocks_per_step=sorted({st['d_blocks'] for pid in ok for a in ('X', 'Y') for st in results[pid]['axes'][a]['steps']}),
                  identity_pairs_per_step=sorted({st['identity_pairs'] for pid in ok for a in ('X', 'Y')
                                                  for st in results[pid]['axes'][a]['steps']}))
        P3['status'] = 'PASS' if all(v for k, v in P3.items() if isinstance(v, bool) and not k.endswith('informational')) else 'FAIL'
        fr = {pid: {a: results[pid]['axes'][a]['final_run'] for a in ('X', 'Y')} for pid in ok}
        form_ok = all(fr[p][a]['names'] == ['rz', 'sx', 'rz'] for p in ok for a in ('X', 'Y'))
        final_rz = {fr[p][a]['rz'] for p in ok for a in ('X', 'Y')}
        p5_rows, theta = [], {}
        T_cal = Fraction(float.fromhex(data['paths'][0]['T_hex']))
        for a in ('X', 'Y'):
            t1 = {fr[p][a]['angles_hex'][0] for p in ok} if form_ok else set()
            t2 = {fr[p][a]['angles_hex'][1] for p in ok} if form_ok else set()
            theta[a] = dict(theta1_hex=sorted(t1), theta2_hex=sorted(t2), unique=len(t1) == 1 and len(t2) == 1)
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
        coef = coefficients(edge['constants'], fr_rz, segs, paths=npaths, q=q, R=R) if fr_rz is not None else None
        decomposition = {}
        for pid in ok:
            body_sum = sum(r['body_rz'] for r in segs[pid])
            for a in ('X', 'Y'):
                saved = next(p for p in data['paths'] if p['path_id'] == pid)['axes'][a]['metrics']['rz_count']
                restored = coef['A'] + coef['H'] * q + body_sum if coef else None
                decomposition['%s:%s' % (pid, a)] = dict(saved_rz=saved, restored=restored, exact=restored == saved,
                    segments_rz=sum(results[pid]['axes'][a]['segment_rz']), end_rz=results[pid]['axes'][a]['end_rz'],
                    final_run_rz=fr[pid][a]['rz'], sum_body_rz=body_sum)
        P6 = dict(status='PASS' if decomposition and all(d['exact'] for d in decomposition.values()) else 'FAIL',
                  decomposition=decomposition)
        support = all(checks.values()) and P3['status'] == 'PASS' and P5['status'] == 'PASS' and P6['status'] == 'PASS'
        later = [r['body_rz'] for rows in segs.values() for r in rows if r['context'] == 'LATER']
        mean = Fraction(sum(later), len(later)); var = sum((Fraction(v) - mean) ** 2 for v in later) / (len(later) - 1)
        res.update(P3=P3, P5=P5, P6=P6, edges=edge, coverage=cover, segments=segs, rule_supported=support,
                   later_body_statistics=dict(n=len(later), mean=float(mean), mean_exact=str(mean), mean_over_R=float(mean / R),
                                              sample_variance=float(var), variance_over_R=float(var / R),
                                              min=min(later), max=max(later), use='DESCRIPTIVE_ONLY_NOT_CI'))
        with open(A.RUN / 'FINAL_RUN.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(p5_rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(p5_rows)
        with open(A.RUN / 'SEGMENTS.csv', 'w', newline='') as fh:
            rows = [dict(path_id=pid, **r) for pid, rs in segs.items() for r in rs]
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
        (A.RUN / 'EDGES.json').write_text(json.dumps(dict(edge, coverage=cover), indent=1) + '\n')
        direct = []
        for pid in ok:
            for a in ('X', 'Y'):
                d = decomposition['%s:%s' % (pid, a)]
                direct.append(dict(path_id=pid, axis=a, physical_id=data['paths'][0]['physical_id'], q=q, R=R,
                                   value_source='DIRECT_COUNT_CALIBRATION_CELL', compiled_RZ=d['saved_rz'],
                                   first_body_rz=segs[pid][0]['body_rz'], later_body_rz_sum=d['sum_body_rz'] - segs[pid][0]['body_rz'],
                                   receipt=next(p for p in data['paths'] if p['path_id'] == pid)['axes'][a]['receipt']))
        price['calibration_cell'] = dict(rows=direct, decomposition_supported=support,
                                         note='Calibration cell %s (q=%d, R=%d) is not a 13-round receipt (m=null)' % (spec['cell'], q, R))
        price['partial_counts'] = dict(constants=edge['constants'], final_run_rz=fr_rz,
                                       later_body_sum=coef and coef['later_body_sum'])
        if support:
            rows, _ = AU.price_table(config, dict(A3=coef['A'], H3=coef['H'], E3=coef['E']))
            for r in rows:
                r['value_source'] = 'MODEL_APPLICATION_FROM_%s_DECOMPOSITION' % spec['cell'].upper()
                r['direct_observation_at_this_q_R'] = False
            comparison = compare(coef, ld3_coef, config, ld3_config) if spec['prefix'] == 5 else None
            model_cell = coef['A'] + coef['H'] * q + coef['E'] * q * R
            p_ = spec['prefix']
            price.update(status='PRICE_RULE_SUPPORTED_MODEL_REFERENCE',
                         coefficients={'A%d' % p_: coef['A'], 'H%d' % p_: coef['H'], 'E%d' % p_: AU.exact_json(coef['E']),
                                       'E%d_display' % p_: float(coef['E']), 'E%d_unit' % p_: 'RZ_PER_EVENT',
                                       'later_body_sum': coef['later_body_sum'], 'intervals': coef['intervals'],
                                       'events_in_denominator': coef['events_in_denominator'], 'A_parts': coef['A_parts'],
                                       'H%d_minus_H3' % p_: coef['H'] - ld3_coef['H'], 'A%d_minus_A3' % p_: coef['A'] - ld3_coef['A'],
                                       'evidence': dict(edges='audit_run01/EDGES.json', segments='audit_run01/SEGMENTS.csv',
                                                        audit='audit_run01/AUDIT_RESULT.json')},
                         rows=rows, comparison=comparison,
                         model_value_at_cell=AU.exact_json(model_cell), model_value_at_cell_display=float(model_cell),
                         stage1_break_even_guides=None)
            with open(out / 'PRICE_26AXIS.csv', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (out / 'PRICE_26AXIS.json').write_text(json.dumps(price, indent=1, default=str) + '\n')
    res.update(compiles=0, sampling=0, builds=0, fits=0, private_writes=0, qpy_reads=data['qpy_reads'])
    (A.RUN / 'AUDIT_RESULT.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(dict(checks=checks, P3=(res.get('P3') or {}).get('status'), P5=(res.get('P5') or {}).get('status'),
                          P6=(res.get('P6') or {}).get('status'), supported=res['rule_supported'],
                          coefficients=price['coefficients'],
                          comparison={k: (v['T_LD5_display'], v['T_LD3_display'], v['ratio_display'])
                                      for k, v in (price['comparison'] or {}).items()}), default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, help='frozen calibration request (driver_spec)')
    parser.add_argument('--config', required=True, help='COMMIT:PATH of the committed configuration JSON of the request')
    parser.add_argument('--build'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    spec = spec_from_request(args.request, args.config)
    if args.build:
        build(spec, args.build)
    if args.run:
        return run(spec, args.workers)
    if args.analyze:
        analyze(spec)
    return 0


if __name__ == '__main__':
    sys.exit(main())
