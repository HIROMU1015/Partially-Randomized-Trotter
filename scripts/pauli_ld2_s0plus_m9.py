#!/usr/bin/env python3
"""S0+: read-only step-decomposition audit of C2's saved LD2 m9 circuits (q=512, R=526).

Rules: docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9/AUDIT_RULES_S0PLUS.md (fixed before
the m9 QPY were opened). The per-path audit is the unchanged S0-r1 function
`pauli_ld2_step_audit.audit_path` (SOURCE 5a18b792a: P1 binding, ordered ancilla-CX matching with
identity pairs excluded, segmentation, X/Y body equality, final run). This script only builds the
m9 inputs, freezes the run02 deterministic parts (entry F/P, suffix S, end E per wire) as a token
reference, and applies them to the m9 segments. At most 8 QPY (2 paths x X/Y x wrapper/compiled)
are read; no build, sampling, transpile, compile, fit, price prediction, E[C], CI or G.

    python -B scripts/pauli_ld2_s0plus_m9.py --build
    python -B scripts/pauli_ld2_s0plus_m9.py --run --workers 2
    python -B scripts/pauli_ld2_s0plus_m9.py --summarize
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A  # noqa: E402  (S0-r1 rule functions, unchanged)

S0P = ROOT / 'docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9'
RUN = S0P / 'run03'
WORK = RUN / 'work'
RUN02 = ROOT / 'docs/research/a_pauli_ld2_step_audit_20261115/run02'
M9_RAW = ('a7fdd200da8d499e93965d63b5b2e5b01e0b8940', 'docs/research/d_pauli_ld2_m9_longpoint_results_20261115/run01')
M9_PRICES = ('c9986c82f866bfa57a7142124746f89fc42f9838', 'docs/research/d_pauli_ld2_m9_longpoint_results_20261115/run01/PRICES.json')
M9_FINAL = ('cb034e547c444ea67a4f4f90a8534c13f4db4451', 'docs/research/d_pauli_ld2_m9_longpoint_results_20261115')
R1_SOURCE = '5a18b792a11e78931ddf7d458eeff1aa8edaac5c'
AXIS_MAP = ('df5e5c452a3814f7e04ea9cacf05c8f4bdcc047c',
            'docs/research/a_pauli_ld2_expected_price_plan_20261115/AXIS_MAP.json')
REFERENCE_PATHS = dict(first='S66/p0', later='Q8R66/p0')   # any segment carries the common parts
MAX_QPY_READS = 8
INTERCEPT, PER_STEP = 308, 1900


# ---------------------------------------------------------------- inputs and frozen reference
def run02_arrays(path_id, q, n):
    import numpy as np
    z = np.load(RUN02 / 'work' / (path_id.replace('/', '_') + '_X.npz'))
    return [dict(segs=[(z[f'w{w}_s{s}_h'], z[f'w{w}_s{s}_rz'], z[f'w{w}_s{s}_mq']) for s in range(1, q + 1)],
                 end=(z[f'w{w}_end_h'], z[f'w{w}_end_rz'], z[f'w{w}_end_mq'])) for w in range(n)]


def make_reference(per_wire, first, later):
    """Per-wire entry (F: first step, P: later step), suffix S and end E token hashes with RZ counts."""
    wires = []
    for w, e in enumerate(per_wire):
        h1, rz1, _ = first[w]['segs'][0]
        h2, rz2, _ = later[w]['segs'][1]
        F, P, S = e['first_prefix_len'], e['later_prefix_len'], e['suffix_len']
        he, rze, _ = first[w]['end']
        wires.append(dict(wire=w,
                          F=[format(int(x), '016x') for x in h1[:F]], F_rz=int(rz1[:F].sum()),
                          P=[format(int(x), '016x') for x in h2[:P]], P_rz=int(rz2[:P].sum()),
                          S=[format(int(x), '016x') for x in h1[len(h1) - S:]], S_rz=int(rz1[len(h1) - S:].sum()),
                          E=[format(int(x), '016x') for x in he], E_rz=int(rze.sum())))
    return wires


def build():
    """m9 inputs (metadata + stat only) and the frozen run02 token reference (no QPY read)."""
    receipts = A.git_json(M9_RAW[0], M9_RAW[1] + '/COMPILE_RECEIPTS.json')
    result = A.git_json(M9_RAW[0], M9_RAW[1] + '/RESULT.json')
    prices = A.git_json(*M9_PRICES)
    axis_map = A.git_json(*AXIS_MAP)
    m9_id = next(r['physical_id'] for r in axis_map['rows'] if r['m'] == 9 and r['axis'] == 'X')
    reused = {s['index']: {('X' if k == 'cosine' else 'Y'): v for k, v in s['result']['fingerprints'].items()}
              for s in result['summaries']}
    paths = {}
    for sel, rec in sorted(receipts.items()):
        axis, index = rec['axis'], rec['index']
        req = rec['request']
        if req['candidate_id'] != m9_id or (req['m'], req['q'], req['R']) != (9, 512, 526):
            raise A.AuditStop('M9_RECEIPT_PHYSICAL')
        fp = rec['record']['fingerprint']
        if reused[index][axis]['fingerprint'] != fp:
            raise A.AuditStop('M9_FINGERPRINT_BINDING')
        files = {}
        for f in rec['artifacts_private'] + [dict(path=rec['native_cache']['path'], sha256=rec['native_cache']['sha256'])]:
            p = Path(f['path'])
            kind = ('wrapper' if f['path'].endswith('_wrapper.qpy.gz') else 'compiled' if f['path'].endswith('_compiled.qpy.gz')
                    else 'cache_json' if f['path'].endswith(fp + '.json') else None)
            if kind:
                files[kind] = dict(path=f['path'], sha256=f['sha256'], bytes=f.get('bytes'), exists=p.exists(),
                                   size_on_disk=p.stat().st_size if p.exists() else None)
        entry = dict(fingerprint=fp, cache_reused=reused[index][axis]['cache_reused'],
                     metrics={k: int(rec['record']['metrics'][k]) for k in A.METRICS},
                     receipt=dict(commit=M9_RAW[0], path=M9_RAW[1] + '/COMPILE_RECEIPTS.json', selector=[sel]),
                     files=files)
        pid = f'M9_ROUND/p{index}'
        common = dict(path_id=pid, kind='C2_M9', cell='M9_ROUND', index=index, m=9, q=req['q'], R=req['R'],
                      T_hex=float(req['T']).hex(), physical_id=req['candidate_id'], cohort=rec['cohort'],
                      composed_of=None, acceptance=M9_FINAL[0])
        rec_p = paths.setdefault(pid, dict(axes={}, **common))
        rec_p['axes'][axis] = entry
    avg = {p['axis']: p['metrics']['rz_count'] for p in prices}
    for axis in ('X', 'Y'):
        vals = [paths[f'M9_ROUND/p{i}']['axes'][axis]['metrics']['rz_count'] for i in (0, 1)]
        if Fraction(sum(vals), 2) != Fraction(avg[axis]).limit_denominator(4):
            raise A.AuditStop('M9_PRICE_TRANSCRIPTION')
    reads = sum(1 for p in paths.values() for a in p['axes'].values() for k in ('wrapper', 'compiled') if k in a['files'])
    if len(paths) != 2 or reads > MAX_QPY_READS:
        raise A.AuditStop('M9_INPUT_COUNT')
    run02_inputs = json.loads((A.OUT / 'INPUTS.json').read_text())
    data = dict(schema='A_PAULI_LD2_S0PLUS_M9_INPUTS_V1',
                rules='docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9/AUDIT_RULES_S0PLUS.md',
                rule_functions=dict(commit=R1_SOURCE, path='scripts/pauli_ld2_step_audit.py'),
                sources=dict(raw=M9_RAW[0], prices=M9_PRICES[0], final=M9_FINAL[0], axis_map=AXIS_MAP[0]),
                scalar=run02_inputs['scalar'], round_T=run02_inputs['round_T'],
                qpy_reads=dict(total=reads, limit=MAX_QPY_READS),
                paths=[paths[k] for k in sorted(paths)], private_contents_read=False)
    S0P.mkdir(parents=True, exist_ok=True)
    (S0P / 'INPUTS_M9.json').write_text(json.dumps(data, indent=1) + '\n')
    # frozen run02 reference (token hashes as hex) -------------------------------------------
    edges = json.loads((RUN02 / 'EDGES.json').read_text())
    final_run = list(csv.DictReader(open(RUN02 / 'FINAL_RUN.csv')))
    n = len(edges['per_wire'])
    first = run02_arrays(REFERENCE_PATHS['first'], 1, n)
    later = run02_arrays(REFERENCE_PATHS['later'], 8, n)
    wires = make_reference(edges['per_wire'], first, later)
    const = dict(F=sum(x['F_rz'] for x in wires), P=sum(x['P_rz'] for x in wires),
                 S=sum(x['S_rz'] for x in wires), E=sum(x['E_rz'] for x in wires))
    if const != dict(F=edges['constants']['first_prefix_rz'], P=edges['constants']['later_prefix_rz'],
                     S=edges['constants']['suffix_rz'], E=edges['constants']['end_rz']):
        raise A.AuditStop('REFERENCE_CONSTANTS')
    if const['F'] + const['S'] + const['E'] + 2 - (const['P'] + const['S']) != INTERCEPT or const['P'] + const['S'] != PER_STEP:
        raise A.AuditStop('REFERENCE_INTERCEPT_SLOPE')
    pred = {r['axis']: float(r['theta1']) for r in final_run if r['m'] == '9'}
    theta2 = {}
    for axis in ('X', 'Y'):
        ref = json.loads((RUN02 / 'PATH_M6_ROUND_p0.json').read_text())['axes'][axis]['final_run']
        theta2[axis] = ref['angles_hex'][1]
    ref_doc = dict(schema='A_PAULI_LD2_RUN02_DETERMINISTIC_REFERENCE_V1',
                   source=dict(run02_result='43755e0caa831697c286e56d20eb911e935ce596', rules=R1_SOURCE,
                               first_segment_of=REFERENCE_PATHS['first'], later_segment_of=REFERENCE_PATHS['later']),
                   token_hash='blake2b-8 of repr((name, qubits, float.hex params)), as in pauli_ld2_step_audit.token_hash',
                   constants=dict(**const, final_run=2, intercept=INTERCEPT, per_step=PER_STEP),
                   final_run_prediction_m9=dict(theta1=pred, theta2_hex=theta2), wires=wires)
    (S0P / 'REFERENCE_RUN02.json').write_text(json.dumps(ref_doc, indent=1) + '\n')
    print(json.dumps(dict(paths=len(paths), reads=reads, constants=const, prediction=pred,
                          missing=[f['path'] for p in paths.values() for a in p['axes'].values()
                                   for f in a['files'].values() if not f['exists']])))


# ---------------------------------------------------------------- run (unchanged S0-r1 audit_path)
def run(workers):
    A.check_runtime()
    A.RUN, A.WORK = RUN, WORK          # redirect outputs only; rule functions are unchanged
    RUN.mkdir(parents=True, exist_ok=True)
    data = json.loads((S0P / 'INPUTS_M9.json').read_text())
    scalar = float.fromhex(data['scalar']['value_hex'])
    started = time.monotonic()
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(A.run_path, (p, scalar)): p['path_id'] for p in data['paths']}
        for future in as_completed(futures):
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                outcomes.append(dict(path_id=futures[future], status='WORKER_FAILED', reason=repr(exc)))
            print(json.dumps(outcomes[-1])[:300], flush=True)
    summary = dict(workers=workers, elapsed_seconds=time.monotonic() - started,
                   outcomes=sorted(outcomes, key=lambda x: x['path_id']))
    (RUN / 'RUN_ALL.json').write_text(json.dumps(summary, indent=1) + '\n')
    return 0 if all(o['status'] == 'DONE' for o in outcomes) else 2


# ---------------------------------------------------------------- reference check and statistics
def check_against_reference(arrays, q, ref_wires):
    """Token-level match of every segment with the frozen run02 entry/suffix and of the end edge.

    Returns dict(status, mismatch, later_body_rz, first_body_rz). The body RZ used for statistics is
    segment RZ minus the run02 constants (entry + suffix), defined whenever segments exist.
    """
    import numpy as np
    to_arr = lambda xs: np.array([int(x, 16) for x in xs], dtype=np.uint64)
    refs = [dict(F=to_arr(r['F']), P=to_arr(r['P']), S=to_arr(r['S']), E=to_arr(r['E']), F_rz=r['F_rz'],
                 P_rz=r['P_rz'], S_rz=r['S_rz']) for r in ref_wires]
    mismatch = None
    bodies = []
    for s in range(q):
        seg_rz = 0
        for w, ref in enumerate(refs):
            h, rz, _ = arrays[w]['segs'][s]
            seg_rz += int(rz.sum())
            pre = ref['F'] if s == 0 else ref['P']
            if mismatch is None:
                if len(h) < len(pre) + len(ref['S']):
                    mismatch = dict(step=s + 1, wire=w, part='segment_shorter_than_reference')
                elif not np.array_equal(h[:len(pre)], pre):
                    k = int(np.flatnonzero(h[:len(pre)] != pre)[0])
                    mismatch = dict(step=s + 1, wire=w, part='entry', position=k)
                elif not np.array_equal(h[len(h) - len(ref['S']):], ref['S']):
                    tail = h[len(h) - len(ref['S']):]
                    k = int(np.flatnonzero(tail != ref['S'])[-1])
                    mismatch = dict(step=s + 1, wire=w, part='suffix', position_from_end=len(ref['S']) - k)
        const = sum((r['F_rz'] if s == 0 else r['P_rz']) + r['S_rz'] for r in refs)
        bodies.append(seg_rz - const)
    if mismatch is None:
        for w, ref in enumerate(refs[:-1]):
            if not np.array_equal(arrays[w]['end'][0], ref['E']):
                mismatch = dict(step=None, wire=w, part='end_edge')
                break
    return dict(status='PASS' if mismatch is None else 'FAIL', mismatch=mismatch,
                first_body_rz=bodies[0], later_body_rz=bodies[1:])


def describe(values, R):
    n = len(values)
    mean = Fraction(sum(values), n)
    var = sum((Fraction(v) - mean) ** 2 for v in values) / (n - 1) if n > 1 else None
    return dict(n=n, mean=float(mean), mean_exact=str(mean), mean_over_R=float(mean / R),
                sample_variance=float(var) if var is not None else None,
                sd=math.sqrt(var) if var is not None else None,
                variance_over_R=float(var / R) if var is not None else None, min=min(values), max=max(values))


def summarize():
    data = json.loads((S0P / 'INPUTS_M9.json').read_text())
    ref = json.loads((S0P / 'REFERENCE_RUN02.json').read_text())
    A.RUN, A.WORK = RUN, WORK
    rows, stats, out_paths = [], [], {}
    pooled = []
    for p in data['paths']:
        pid = p['path_id']
        f = RUN / ('PATH_' + pid.replace('/', '_') + '.json')
        if not f.exists():
            out_paths[pid] = dict(status='NO_RESULT')
            continue
        r = json.loads(f.read_text())
        res = dict(P1={a: r['axes'][a].get('P1') for a in ('X', 'Y')},
                   P2={a: r['axes'][a].get('P2') for a in ('X', 'Y')},
                   P2_detail={a: r['axes'][a].get('P2_detail') for a in ('X', 'Y')},
                   identity_pairs={a: sorted({s['identity_pairs'] for s in r['axes'][a].get('steps', [])}) for a in ('X', 'Y')},
                   events_equal_R={a: r['axes'][a].get('events_equal_R') for a in ('X', 'Y')},
                   products={a: r['axes'][a].get('products_total') for a in ('X', 'Y')},
                   pairs_checked=r['axes']['X'].get('pair_checks'),
                   XY_body_equal=r['xy']['compiled_equal_except_final_run'],
                   uncompiled_XY_only_sdg=r['xy']['uncompiled_only_sdg_differs'],
                   saved_rz={a: p['axes'][a]['metrics']['rz_count'] for a in ('X', 'Y')})
        fr = {}
        for a in ('X', 'Y'):
            run_ = r['axes'][a]['final_run']
            ok_form = run_['names'] == ['rz', 'sx', 'rz']
            t1 = float.fromhex(run_['angles_hex'][0]) if ok_form else None
            pred = ref['final_run_prediction_m9']['theta1'][a]
            fr[a] = dict(names=run_['names'], rz=run_['rz'], theta1=t1, theta1_predicted_run02=pred,
                         theta1_matches=ok_form and abs(A.wrap(t1 - pred)) <= A.RELATION_TOLERANCE,
                         theta2_matches=ok_form and run_['angles_hex'][1] == ref['final_run_prediction_m9']['theta2_hex'][a],
                         wrapper_input=r['axes'][a]['wrapper_input'])
        res['final_run'] = fr
        segmented = all(str(r['axes'][a].get('P2', '')).startswith('PASS') for a in ('X', 'Y'))
        if segmented:
            arrays = A.load_x_arrays(pid, p['q'], r['n_wires'])
            chk = check_against_reference(arrays, p['q'], ref['wires'])
            body_sum = chk['first_body_rz'] + sum(chk['later_body_rz'])
            total = INTERCEPT + PER_STEP * p['q'] + body_sum
            res['reference_match'] = chk['status']
            res['reference_mismatch'] = chk['mismatch']
            res['decomposition'] = dict(intercept=INTERCEPT, per_step=PER_STEP, q=p['q'], sum_body_rz=body_sum,
                                        restored=total, saved=p['axes']['X']['metrics']['rz_count'],
                                        exact=total == p['axes']['X']['metrics']['rz_count'],
                                        segments_rz=r['axes']['X']['segment_rz'] and sum(r['axes']['X']['segment_rz']),
                                        end_rz=r['axes']['X']['end_rz'], final_run_rz=r['axes']['X']['final_run']['rz'])
            res['first_step_body_rz'] = chk['first_body_rz']
            st = describe(chk['later_body_rz'], p['R'])
            stats.append(dict(path_id=pid, context='LATER', R=p['R'], independent_paths=1, **st))
            pooled.extend(chk['later_body_rz'])
            for i, b in enumerate([chk['first_body_rz']] + chk['later_body_rz'], 1):
                rows.append(dict(path_id=pid, step=i, context='FIRST' if i == 1 else 'LATER', R=p['R'], body_rz=b,
                                 segment_rz=r['axes']['X']['segment_rz'][i - 1]))
        out_paths[pid] = res
    if pooled:
        stats.append(dict(path_id='POOLED_2_PATHS', context='LATER', R=data['paths'][0]['R'],
                          independent_paths=len({r['path_id'] for r in rows}), **describe(pooled, data['paths'][0]['R'])))
    for name, table in (('SEGMENTS_M9.csv', rows), ('STEP_STATS_M9.csv', stats)):
        keys = list(dict.fromkeys(k for row in table for k in row))
        with open(RUN / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator='\n')
            w.writeheader()
            w.writerows(table)
    result = dict(schema='A_PAULI_LD2_S0PLUS_M9_RESULT_V1', paths=out_paths, statistics=stats,
                  P4='NOT_APPLICABLE_NO_PAIRED_COMPOSITION_IN_M9',
                  compiles=0, sampling=0, builds=0, fits=0, price_predictions=0, private_writes=0)
    (RUN / 'AUDIT_RESULT_M9.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(json.dumps({pid: dict(P2=v.get('P2'), ref=v.get('reference_match'), dec=(v.get('decomposition') or {}).get('exact'),
                                xy=v.get('XY_body_equal')) for pid, v in out_paths.items()}, default=str))
    print(json.dumps(stats, default=str)[:1200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
    if args.run:
        return run(args.workers)
    if args.summarize:
        summarize()
    return 0


if __name__ == '__main__':
    sys.exit(main())
