"""Zero-compile closure audit of saved deterministic-PF compiled QPY.

Plan and rules: docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md §6 (dd435d07e)
and docs/research/d_pf_qpy_closure_audit_20261115/AUDIT_RULES.md.

Reads saved private QPY and cache JSON read-only (at most 56 QPY files). Never builds,
transpiles or compiles circuits, never writes private files, never computes prices of
unobserved rounds and never re-estimates coefficients. Wire tokens reuse the existing
wire comparison of 51d324ea check_xy_wrapper_locality_artificial.py (parameter encoding
switched to float.hex as fixed by the rules).

    python -B scripts/pf_qpy_closure_audit.py --build-inputs
    python -B scripts/pf_qpy_closure_audit.py --measure
    python -B scripts/pf_qpy_closure_audit.py --run --workers N
    python -B scripts/pf_qpy_closure_audit.py --summarize
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gc
import gzip
import io
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/d_pf_qpy_closure_audit_20261115'
RUN = OUT / 'run01'
ROUTES = {
    'FULL_PF': dict(output='docs/research/d_full_pf_shortlist_prices_20261114',
                    private='/home/AbeHiromu/projects/d-full-pf-shortlist-20261114-private'),
    'TRUNCATED_PF': dict(output='docs/research/d_truncated_p7_t3_prices_20261114',
                         private='/home/AbeHiromu/projects/d-truncated-p7-t3-20261114-private'),
}
STAGES = (('calibration', 'b190bd815c886446291fac8b281c3c7df7343372', (4, 5)),
          ('holdout', 'a55813373b84b9128cbd10c6812c0d26143a4280', (6,)),
          ('m9_holdout', '0a35101914c6833244b6e0bad7bc2767a49b5a47', (9,)))
FORMULAS = {'2nd': '16/7', '4th': '16/3', '4th(m5_best)': '16'}
SCHEDULE_LABEL = {'2nd': 'S2', '4th': 'Y4', '4th(m5_best)': 'M5'}
AXES = {'X': 'cosine', 'Y': 'sine'}
FROZEN = ('12372fa6b53c878dc5dd124e036572ba0cd82061',
          'docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
SCHEDULE = ('dd435d07e3f0490165ff49889be9c71f573bd5db',
            'docs/research/d_pf_unobserved_round_closure_20261115/SCHEDULE_IDENTITY.json')
WIRE_COMPARISON = ('51d324ea9c83bc12f1e866da0fa409cef926f74a',
                   'docs/research/pr_resource_repo_wide_audit_20261114/checks/check_xy_wrapper_locality_artificial.py')
UNOBSERVED = (7, 8, 10, 11, 12)
MARGIN_THRESHOLD = 1e-6          # final-run RZ omission margin (REPORT §6.2)
RELATION_TOLERANCE = 1e-9        # T-dependent angle vs slope -scalar (instruction A4)
OMISSION_ATOL = 1e-12            # Qiskit 1.3.0 ANGLE_ZERO_EPSILON (documentation only)
METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')
QISKIT = '1.3.0'
MAX_QPY_READS = 56
TWO_PI = 2 * math.pi


class AuditStop(RuntimeError):
    """Mandatory stop conditions of REPORT §6.5 (partial results are kept)."""


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def git_json(commit, path):
    return json.loads(git('show', f'{commit}:{path}'))


def wrap(angle):
    """Map to (-pi, pi]."""
    value = math.fmod(angle + math.pi, TWO_PI)
    if value <= 0:
        value += TWO_PI
    return value - math.pi


def group_id(route, formula, axis):
    return f'{route}|{formula}|{FORMULAS[formula]}|{axis}'


# ---------------------------------------------------------------- wire tokens (reused)
def hex_params(params):
    return tuple(float(x).hex() for x in params)


def wires_without_final_ancilla_run(t, encode=hex_params, intern=None):
    """51d324ea `wires_without_final_ancilla_run`, unchanged control flow.

    Per-wire ordered operation records; the ancilla list is cut after its last two-qubit
    gate. Differences: parameter encoding is a callable (audit: float.hex), records may be
    interned to integers, a qubit->index map replaces repeated find_bit, and the cut tail
    records are returned together with simple counts.
    """
    a = t.num_qubits - 1
    wires = [[] for _ in range(t.num_qubits)]
    tail = []
    index = {q: t.find_bit(q).index for q in t.qubits}
    counts = dict(rz=0, cx=0, size=0, measure=0, other=0)
    for inst in t.data:
        op = inst.operation
        if getattr(op, '_directive', False):
            continue
        counts['size'] += 1
        name = op.name
        if name == 'measure':
            counts['measure'] += 1
            continue
        if name in counts:
            counts[name] += 1
        idx = tuple(index[q] for q in inst.qubits)
        rec = (name, idx, encode(op.params))
        key = rec if intern is None else intern(rec)
        for q in idx:
            wires[q].append(key)
        if a in idx:
            tail = [] if len(idx) > 1 else tail + [rec]
    if tail:
        wires[a] = wires[a][:-len(tail)]
    return wires, tail, counts


class Interner:
    def __init__(self):
        self.ids, self.tokens = {}, []

    def __call__(self, token):
        value = self.ids.get(token)
        if value is None:
            value = self.ids[token] = len(self.tokens)
            self.tokens.append(token)
        return value


# ---------------------------------------------------------------- periodic decomposition
def longest_true_run(mask):
    import numpy as np
    if mask.size == 0:
        return 0, 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]
    if starts.size == 0:
        return 0, 0
    lengths = ends - starts
    best = int(np.argmax(lengths))           # first maximal run on ties
    return int(starts[best]), int(lengths[best])


def extract_wire(seqs):
    """Canonical (H, B, n, E, c_w) from the largest q; linear length checked on all q."""
    import numpy as np
    qs = sorted(seqs)
    qmin, qmax = qs[0], qs[-1]
    lengths = {q: int(len(seqs[q])) for q in qs}
    num, den = lengths[qmax] - lengths[qmin], qmax - qmin
    if den <= 0 or num % den:
        return dict(ok=False, reason='NON_INTEGER_PERIOD', lengths=lengths)
    p = num // den
    if p <= 0:
        return dict(ok=False, reason='NONPOSITIVE_PERIOD', lengths=lengths, p=p)
    if any(lengths[q] != lengths[qmin] + p * (q - qmin) for q in qs):
        return dict(ok=False, reason='NONLINEAR_LENGTH', lengths=lengths, p=p)
    S = seqs[qmax]
    if len(S) <= p:
        return dict(ok=False, reason='SEQUENCE_SHORTER_THAN_PERIOD', lengths=lengths, p=p)
    s, run = longest_true_run(S[:-p] == S[p:])
    if run == 0:
        return dict(ok=False, reason='NO_PERIODIC_RUN', lengths=lengths, p=p)
    n = (run + p) // p
    return dict(ok=True, p=p, s=s, n=n, c_w=qmax - n, H=S[:s], B=S[s:s + p], E=S[s + n * p:],
                lengths=lengths)


def verify_group(wire_seqs):
    """A2: common c; E_w absorbs surplus copies; exact equality on every audited q."""
    import numpy as np
    parts = [extract_wire(seqs) for seqs in wire_seqs]
    qs = sorted(wire_seqs[0])
    if not all(part['ok'] for part in parts):
        return dict(pass_=False, c=None, parts=parts, per_q={q: False for q in qs})
    c = max(part['c_w'] for part in parts)
    per_q = {}
    for part, seqs in zip(parts, wire_seqs):
        extra = c - part['c_w']
        part['E'] = np.concatenate((np.tile(part['B'], extra), part['E']))
        part['per_q'] = {}
        for q in qs:
            reps = q - c
            ok = reps >= 0 and np.array_equal(seqs[q], np.concatenate((part['H'], np.tile(part['B'], max(reps, 0)), part['E'])))
            part['per_q'][q] = bool(ok)
            per_q[q] = per_q.get(q, True) and bool(ok)
    return dict(pass_=all(per_q.values()), c=c, parts=parts, per_q=per_q)


# ---------------------------------------------------------------- inputs
def frozen_coefficients():
    data = git_json(*FROZEN)
    out = {}
    for key, model in data['models'].items():
        coeff = {name: Fraction(v['numerator'], v['denominator']) for name, v in model['coefficients'].items()}
        if coeff['one'].denominator != 1 or coeff['q'].denominator != 1:
            raise AuditStop('NON_INTEGER_FROZEN_COEFFICIENT:' + key)
        out[key] = dict(a=int(coeff['one']), b=int(coeff['q']))
    return out


def schedule_rounds():
    data = git_json(*SCHEDULE)
    if not data['all_pass'] or data['checks'] != 540:
        raise AuditStop('SCHEDULE_IDENTITY_NOT_FIXED_PASS')
    route_of = {'FULL_PF': 'FULL_PF', 'TRUNCATED_PF': 'TRUNCATED_PF'}
    out = {}
    for group in data['groups']:
        for row in group['rounds']:
            out[(route_of[group['route']], group['formula'], row['m'])] = dict(
                T_hex=row['T_hex'], q=row['q'], physical_id=row['physical_id'], status=row['status'])
    return out


def build_inputs():
    schedule = schedule_rounds()
    frozen = frozen_coefficients()
    targets, wrappers, receipts = [], [], []
    for route, spec in ROUTES.items():
        for stage, commit, rounds in STAGES:
            base = f"{spec['output']}/{stage}/run01"
            result, prices = git_json(commit, base + '/RESULT.json'), git_json(commit, base + '/PRICES.json')
            for name in ('RESULT.json', 'PRICES.json'):
                fixed = git('rev-parse', f'{commit}:{base}/{name}').strip()
                head = git('rev-parse', f'HEAD:{base}/{name}').strip()
                receipts.append(dict(commit=commit, path=f'{base}/{name}', blob=fixed, head_blob=head,
                                     unchanged_at_HEAD=fixed == head))
            for i, res in enumerate(result['results']):
                cell, trial = res['cell'], res['trial']
                row = cell.get('row') or trial['candidate']
                m, formula = cell['m'], row['formula']
                if m not in rounds or (m == 4 and formula == '4th(m5_best)'):
                    continue
                sched = schedule[(route, formula, m)]
                for axis, wire in AXES.items():
                    sel = [j for j, p in enumerate(prices) if p['m'] == m and p['axis'] == axis
                           and p['identity']['candidate']['formula'] == formula]
                    if len(sel) != 1:
                        raise AuditStop(f'PRICE_RECEIPT_NOT_UNIQUE:{route}:{formula}:{m}:{axis}')
                    fp = res['wrappers'][wire]['fingerprint']
                    cache = Path(spec['private']) / stage / 'run01' / 'compiled_cache'
                    compiled = cache / f'{fp}_compiled.qpy.gz'
                    target = dict(group=group_id(route, formula, axis), route=route, formula=formula, axis=axis,
                                  m=m, q=row['q'], T=float(row['T']).hex(), T_schedule=sched['T_hex'],
                                  physical_id=row['id'], scalar=float(trial['scalar']).hex(), fingerprint=fp,
                                  cache_reused=res['wrappers'][wire].get('cache_reused'),
                                  result=dict(commit=commit, path=base + '/RESULT.json', selector=['results', i]),
                                  price=dict(commit=commit, path=base + '/PRICES.json', selector=[sel[0]]),
                                  receipt_metrics=prices[sel[0]]['metrics'], trial_metrics=trial['prices'][wire],
                                  compiled=str(compiled), cache_json=str(cache / f'{fp}.json'),
                                  compiled_exists=compiled.exists(),
                                  compiled_bytes=compiled.stat().st_size if compiled.exists() else None)
                    targets.append(target)
                    if m == 9:
                        wrapper = cache / f'{fp}_wrapper.qpy.gz'
                        wrappers.append(dict(group=target['group'], m=m, fingerprint=fp, path=str(wrapper),
                                             exists=wrapper.exists(),
                                             bytes=wrapper.stat().st_size if wrapper.exists() else None))
    unobserved = {}
    for route in ROUTES:
        for formula in FORMULAS:
            for m in UNOBSERVED:
                sched = schedule[(route, formula, m)]
                unobserved[f'{route}|{formula}|{m}'] = dict(T_hex=sched['T_hex'], q=sched['q'],
                                                           physical_id=sched['physical_id'])
    reads = len(targets) + len(wrappers)
    if reads > MAX_QPY_READS:
        raise AuditStop('READ_BUDGET')
    data = dict(schema='D_PF_QPY_CLOSURE_AUDIT_INPUTS_V1', rules='docs/research/d_pf_qpy_closure_audit_20261115/AUDIT_RULES.md',
                plan=dict(commit=SCHEDULE[0], path='docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md#6'),
                frozen=dict(commit=FROZEN[0], path=FROZEN[1], coefficients=frozen),
                schedule=dict(commit=SCHEDULE[0], path=SCHEDULE[1]), wire_comparison=dict(commit=WIRE_COMPARISON[0], path=WIRE_COMPARISON[1]),
                thresholds=dict(margin=MARGIN_THRESHOLD, relation=RELATION_TOLERANCE, omission_atol_documented=OMISSION_ATOL),
                qpy_reads=dict(compiled=len(targets), wrapper=len(wrappers), total=reads, limit=MAX_QPY_READS),
                receipts=receipts, targets=targets, wrappers=wrappers, unobserved=unobserved,
                private_contents_read=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'INPUTS.json').write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps(dict(targets=len(targets), wrappers=len(wrappers), reads=reads,
                          missing=[t['compiled'] for t in targets if not t['compiled_exists']] +
                                  [w['path'] for w in wrappers if not w['exists']],
                          receipts_unchanged=all(r['unchanged_at_HEAD'] for r in receipts))))


# ---------------------------------------------------------------- reading
def load_qpy(path):
    from qiskit import qpy
    with open(path, 'rb') as handle:          # read-only
        raw = gzip.decompress(handle.read())
    circuits = qpy.load(io.BytesIO(raw))
    if len(circuits) != 1:
        raise AuditStop('QPY_CIRCUIT_COUNT:' + str(path))
    return circuits[0]


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Progress:
    def __init__(self, path, label):
        self.path, self.label = path, label

    def __call__(self, event, **data):
        record = dict(time=time.time(), label=self.label, event=event, max_rss_mib=round(rss_mib(), 1), **data)
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(record) + '\n')


def check_runtime():
    import qiskit
    if qiskit.__version__ != QISKIT:
        raise AuditStop('QISKIT_VERSION:' + qiskit.__version__)


def a1_bind(target):
    cache = json.loads(Path(target['cache_json']).read_text())
    checks = dict(compiled_exists=Path(target['compiled']).exists(),
                  cache_fingerprint=cache.get('fingerprint') == target['fingerprint'],
                  cache_metrics_equal_receipt=all(cache['metrics'].get(k) == target['receipt_metrics'].get(k) for k in METRICS),
                  trial_metrics_equal_receipt=all(target['trial_metrics'].get(k) == target['receipt_metrics'].get(k) for k in METRICS),
                  cache_compiler=cache.get('compiler') == '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b',
                  T_matches_schedule=target['T'] == target['T_schedule'],
                  not_cache_reused=target['cache_reused'] is False)
    return checks


def final_run_angles(tail):
    names = [rec[0] for rec in tail]
    angles = [float.fromhex(rec[2][0]) for rec in tail if rec[0] == 'rz']
    return names, angles


def synthesize_input_run(tail_records):
    """Expected compiled final run from the uncompiled ancilla input run (no transpile)."""
    import numpy as np
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from qiskit.synthesis import OneQubitEulerDecomposer
    mapping = get_standard_gate_name_mapping()
    matrix = np.eye(2, dtype=complex)
    for name, params in tail_records:
        gate = mapping[name]
        if params:
            gate = type(gate)(*params)
        matrix = np.asarray(gate.to_matrix()) @ matrix
    circuit = OneQubitEulerDecomposer('ZSX')(matrix, simplify=True)
    return [inst.operation.name for inst in circuit.data], [float(inst.operation.params[0]) for inst in circuit.data if inst.operation.name == 'rz']


def wrapper_input_tail(circuit):
    a = circuit.num_qubits - 1
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    tail = []
    for inst in reversed(circuit.data):
        idx = tuple(index[q] for q in inst.qubits)
        if a not in idx or inst.operation.name == 'measure':
            continue
        if len(idx) > 1:
            break
        tail.append((inst.operation.name, [float(p) for p in inst.operation.params]))
    return list(reversed(tail))


def audit_group(group, targets, wrapper, frozen, unobserved, progress_path):
    import numpy as np
    progress = Progress(progress_path, group)
    progress('group_start', targets=len(targets))
    check_runtime()
    intern = Interner()
    by_q, observed = {}, []
    for target in sorted(targets, key=lambda t: t['q']):
        a1 = a1_bind(target)
        record = dict(m=target['m'], q=target['q'], fingerprint=target['fingerprint'], A1=a1)
        if not a1['compiled_exists']:
            record.update(A1_status='UNKNOWN_MISSING')
            observed.append(record)
            continue
        started = time.monotonic()
        progress('load_start', m=target['m'], q=target['q'], bytes=target['compiled_bytes'])
        circuit = load_qpy(target['compiled'])
        wires, tail, counts = wires_without_final_ancilla_run(circuit, intern=intern)
        nq = circuit.num_qubits
        del circuit
        gc.collect()
        saved = target['receipt_metrics']
        if counts['rz'] != saved['rz_count']:
            progress('stop', reason='RZ_TOKEN_TOTAL_MISMATCH', tokens=counts['rz'], saved=saved['rz_count'])
            raise AuditStop(f"RZ_TOKEN_TOTAL_MISMATCH:{group}:m{target['m']}:{counts['rz']}!={saved['rz_count']}")
        a1['token_rz_equal_saved'] = True
        a1['token_cx_equal_saved'] = counts['cx'] == saved['cx_count']
        a1['token_size_equal_saved'] = counts['size'] == saved['circuit_size']
        a1['qubits'] = nq
        record.update(A1_status='PASS' if all(v is True for k, v in a1.items() if k != 'qubits') else 'UNKNOWN_BINDING',
                      counts=counts, final_run=[[r[0], list(r[2])] for r in tail],
                      load_seconds=round(time.monotonic() - started, 3))
        by_q[target['q']] = dict(target=target, wires=[np.asarray(w, dtype=np.int64) for w in wires], tail=tail, counts=counts)
        observed.append(record)
        progress('load_end', m=target['m'], q=target['q'], seconds=record['load_seconds'], instructions=counts['size'])
    result = dict(group=group, observed=observed)
    bound = all(r.get('A1_status') == 'PASS' for r in observed) and len(by_q) == len(targets)
    result['A1'] = 'PASS' if bound else 'UNKNOWN'
    if not by_q:
        result.update(A2='UNKNOWN', A3='UNKNOWN', A4='UNKNOWN', A5='UNKNOWN')
        return result
    qs = sorted(by_q)
    nwires = len(by_q[qs[0]]['wires'])
    wire_seqs = [{q: by_q[q]['wires'][w] for q in qs} for w in range(nwires)]
    a2 = verify_group(wire_seqs)
    result['A2'] = 'PASS' if a2['pass_'] and bound else ('FAIL' if bound else 'UNKNOWN')
    result['c'] = a2['c']
    result['audited_q'] = qs
    result['A2_per_q'] = {str(q): v for q, v in a2['per_q'].items()}
    result['wires'] = []
    frozen_ab = frozen[group]
    rz_body = rz_edges = 0
    ancilla = nwires - 1
    for w, part in enumerate(a2['parts']):
        info = dict(wire=w, ok=part['ok'], reason=part.get('reason'), lengths={str(k): v for k, v in part['lengths'].items()})
        if part['ok']:
            body = [intern.tokens[i] for i in part['B']]
            rz_b = sum(t[0] == 'rz' for t in body)
            rz_e = sum(intern.tokens[i][0] == 'rz' for i in part['H']) + sum(intern.tokens[i][0] == 'rz' for i in part['E'])
            rz_body += rz_b
            rz_edges += rz_e
            cx = sum(t[0] == 'cx' for t in body)
            barrier = False
            if w != ancilla:
                cyc = body + body[:2]
                barrier = any(cyc[i][0] == 'cx' and cyc[i][1] == (ancilla, w) and cyc[i + 1][0] == 'rz'
                              and cyc[i + 1][1] == (w,) and cyc[i + 2] == cyc[i] for i in range(len(body)))
            info.update(p=part['p'], c_w=part['c_w'], head=len(part['H']), tail=len(part['E']), body_rz=rz_b,
                        edge_rz=rz_e, body_cx=cx, body_has_cx=cx > 0,
                        directional_barrier=None if w == ancilla else barrier,
                        per_q={str(k): v for k, v in part.get('per_q', {}).items()})
        result['wires'].append(info)
    a3 = dict(body_rz=rz_body, frozen_b=frozen_ab['b'], frozen_a=frozen_ab['a'])
    if a2['pass_']:
        a3['body_equals_b'] = rz_body == frozen_ab['b']
        a3['per_q'] = {}
        for q in qs:
            final_rz = sum(r[0] == 'rz' for r in by_q[q]['tail'])
            outside = by_q[q]['counts']['rz'] - (q - a2['c']) * rz_body
            a3['per_q'][str(q)] = dict(edges_rz=rz_edges, final_run_rz=final_rz, outside=outside,
                                       expected=frozen_ab['a'] + a2['c'] * frozen_ab['b'],
                                       ok=outside == frozen_ab['a'] + a2['c'] * frozen_ab['b'] and outside == rz_edges + final_rz)
        result['A3'] = 'PASS' if a3['body_equals_b'] and all(v['ok'] for v in a3['per_q'].values()) else 'FAIL'
    else:
        result['A3'] = 'UNKNOWN'
    result['A3_detail'] = a3
    # A4 final run
    a4 = dict(forms={}, sx={}, theta1={}, theta2={})
    for q in qs:
        names, angles = final_run_angles(by_q[q]['tail'])
        a4['forms'][str(q)] = names
        a4['sx'][str(q)] = names.count('sx')
        if names == ['rz', 'sx', 'rz']:
            a4['theta1'][str(q)] = angles[0].hex()
            a4['theta2'][str(q)] = angles[1].hex()
    standard = all(a4['forms'][str(q)] == ['rz', 'sx', 'rz'] for q in qs)
    a4['standard_form'] = standard
    q9 = qs[-1]
    t9 = by_q[q9]['target']
    scalar = float.fromhex(t9['scalar'])
    a4['scalar'] = t9['scalar']
    a4['scalar_common'] = all(by_q[q]['target']['scalar'] == t9['scalar'] for q in qs)
    relation_ok = False
    if standard:
        a4['theta2_constant'] = len(set(a4['theta2'].values())) == 1
        th9 = float.fromhex(a4['theta1'][str(q9)])
        a4['relation_residual'] = {}
        for q in qs:
            dT = float(Fraction(float.fromhex(by_q[q]['target']['T'])) - Fraction(float.fromhex(t9['T'])))
            res = wrap(float.fromhex(a4['theta1'][str(q)]) - th9 + scalar * dT)
            a4['relation_residual'][str(q)] = res
        relation_ok = a4['scalar_common'] and a4['theta2_constant'] and all(abs(v) <= RELATION_TOLERANCE for v in a4['relation_residual'].values())
    a4['relation_ok'] = relation_ok
    if wrapper is not None and wrapper['exists']:
        progress('wrapper_load_start', bytes=wrapper['bytes'])
        circuit = load_qpy(wrapper['path'])
        tail_in = wrapper_input_tail(circuit)
        del circuit
        gc.collect()
        expected_phase = (-scalar * float.fromhex(t9['T']))
        p_params = [params[0] for name, params in tail_in if name == 'p']
        names, angles = synthesize_input_run(tail_in)
        compiled_names, compiled_angles = final_run_angles(by_q[q9]['tail'])
        a4['wrapper_input'] = dict(names=[n for n, _ in tail_in], params=[[float(x).hex() for x in p] for _, p in tail_in],
                                   p_equals_minus_scalar_T=len(p_params) == 1 and p_params[0] == expected_phase,
                                   synthesized_form=names,
                                   synthesized_matches_compiled=names == compiled_names and len(angles) == len(compiled_angles)
                                   and all(abs(wrap(x - y)) <= RELATION_TOLERANCE for x, y in zip(angles, compiled_angles)))
        progress('wrapper_load_end')
    result['A4_detail'] = a4
    result['A4'] = 'PASS' if relation_ok else 'UNKNOWN'
    result['A5'] = 'PASS' if a2['pass_'] and all(w.get('body_has_cx') for w in result['wires']) else ('FAIL' if a2['pass_'] else 'UNKNOWN')
    # unobserved final-run margins (no price values)
    margins = []
    route, formula, _, axis = group.split('|')
    for m in UNOBSERVED:
        info = unobserved[f'{route}|{formula}|{m}']
        row = dict(group=group, axis=axis, m=m, q=info['q'], physical_id=info['physical_id'], T=info['T_hex'])
        if relation_ok:
            dT = float(Fraction(float.fromhex(info['T_hex'])) - Fraction(float.fromhex(t9['T'])))
            angle = wrap(float.fromhex(a4['theta1'][str(q9)]) - scalar * dT)
            row.update(theta1_extrapolated=angle, margin=abs(angle),
                       final_run=('MARGIN_OK' if abs(angle) >= MARGIN_THRESHOLD else 'UNKNOWN_SMALL_MARGIN'))
        else:
            row.update(theta1_extrapolated=None, margin=None, final_run='UNKNOWN_A4')
        margins.append(row)
    result['unobserved'] = margins
    progress('group_end', A1=result['A1'], A2=result['A2'], A3=result['A3'], A4=result['A4'], A5=result['A5'])
    return result


def run_group(job):
    group, targets, wrapper, frozen, unobserved = job
    path = RUN / ('GROUP_' + group.replace('|', '_').replace('/', 'o').replace('(', '').replace(')', '') + '.json')
    try:
        result = audit_group(group, targets, wrapper, frozen, unobserved, RUN / 'PROGRESS.jsonl')
        path.write_text(json.dumps(result, indent=1, default=str) + '\n')
        return dict(group=group, status='DONE', path=str(path.relative_to(ROOT)), max_rss_mib=round(rss_mib(), 1))
    except BaseException as exc:
        stop = dict(group=group, status='STOP', reason=str(exc), traceback=traceback.format_exc(), max_rss_mib=round(rss_mib(), 1))
        path.with_suffix('.STOP.json').write_text(json.dumps(stop, indent=1) + '\n')
        return stop


def jobs_from_inputs(selected=None):
    data = json.loads((OUT / 'INPUTS.json').read_text())
    frozen = data['frozen']['coefficients']
    groups = {}
    for target in data['targets']:
        groups.setdefault(target['group'], []).append(target)
    wrappers = {w['group']: w for w in data['wrappers']}
    return [(g, groups[g], wrappers.get(g), frozen, data['unobserved'])
            for g in sorted(groups) if selected is None or g in selected], data


def measure():
    """Load and tokenize the smallest compiled target once to size the run (§6.5)."""
    check_runtime()
    data = json.loads((OUT / 'INPUTS.json').read_text())
    target = min(data['targets'], key=lambda t: t['compiled_bytes'])
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    circuit = load_qpy(target['compiled'])
    loaded = time.monotonic() - started
    wires, tail, counts = wires_without_final_ancilla_run(circuit, intern=Interner())
    out = dict(group=target['group'], m=target['m'], q=target['q'], bytes=target['compiled_bytes'],
               instructions=counts['size'], load_seconds=loaded, total_seconds=time.monotonic() - started,
               max_rss_mib=rss_mib(), rz_tokens=counts['rz'], saved_rz=target['receipt_metrics']['rz_count'])
    (RUN / 'MEASURE.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


def run(workers, selected=None):
    check_runtime()
    RUN.mkdir(parents=True, exist_ok=True)
    jobs, _ = jobs_from_inputs(selected)
    started = time.monotonic()
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_group, job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                outcomes.append(future.result())
            except BaseException as exc:   # worker died (OOM/signal)
                outcomes.append(dict(group=futures[future], status='WORKER_FAILED', reason=repr(exc)))
            print(json.dumps(outcomes[-1])[:300], flush=True)
    summary = dict(workers=workers, elapsed_seconds=time.monotonic() - started, outcomes=sorted(outcomes, key=lambda x: x['group']))
    (RUN / ('RUN_' + ('ALL' if selected is None else 'PART') + '.json')).write_text(json.dumps(summary, indent=1) + '\n')
    return 0 if all(o['status'] == 'DONE' for o in outcomes) else 2


def summarize():
    data = json.loads((OUT / 'INPUTS.json').read_text())
    groups = sorted({t['group'] for t in data['targets']})
    rows, observed, unobserved = [], [], []
    for group in groups:
        path = RUN / ('GROUP_' + group.replace('|', '_').replace('/', 'o').replace('(', '').replace(')', '') + '.json')
        if not path.exists():
            rows.append(dict(group=group, A1='UNKNOWN', A2='UNKNOWN', A3='UNKNOWN', A4='UNKNOWN', A5='UNKNOWN', note='NO_RESULT'))
            continue
        res = json.loads(path.read_text())
        detail = res.get('A3_detail', {})
        a4 = res.get('A4_detail', {})
        rows.append(dict(group=group, audited_q=' '.join(map(str, res.get('audited_q', []))), c=res.get('c'),
                         A1=res['A1'], A2=res['A2'], A3=res['A3'], A4=res['A4'], A5=res['A5'],
                         body_rz=detail.get('body_rz'), frozen_b=detail.get('frozen_b'), frozen_a=detail.get('frozen_a'),
                         final_form='/'.join(sorted({'-'.join(v) for v in a4.get('forms', {}).values()})),
                         wrapper_p_ok=(a4.get('wrapper_input') or {}).get('p_equals_minus_scalar_T'),
                         wrapper_synth_ok=(a4.get('wrapper_input') or {}).get('synthesized_matches_compiled'),
                         barriers=sum(bool(w.get('directional_barrier')) for w in res.get('wires', [])),
                         system_wires=len(res.get('wires', [])) - 1))
        for rec in res['observed']:
            observed.append(dict(group=group, m=rec['m'], q=rec['q'], fingerprint=rec['fingerprint'], A1=rec['A1_status'],
                                 rz_tokens=(rec.get('counts') or {}).get('rz'),
                                 final_run='-'.join(r[0] for r in rec.get('final_run', [])),
                                 A2_q=res.get('A2_per_q', {}).get(str(rec['q']))))
        unobserved.extend(res['unobserved'])
    for name, table in (('AUDIT_TABLE.csv', rows), ('OBSERVED_POINTS.csv', observed), ('UNOBSERVED_AXES.csv', unobserved)):
        keys = list(dict.fromkeys(k for row in table for k in row))
        with open(RUN / name, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
    status = dict(groups=len(rows), all_pass=all(all(r[k] == 'PASS' for k in ('A1', 'A2', 'A3', 'A4', 'A5')) for r in rows),
                  unobserved_axes=len(unobserved),
                  final_run_margin_ok=sum(r['final_run'] == 'MARGIN_OK' for r in unobserved),
                  min_margin=min((r['margin'] for r in unobserved if r['margin'] is not None), default=None))
    (RUN / 'AUDIT_RESULT.json').write_text(json.dumps(dict(schema='D_PF_QPY_CLOSURE_AUDIT_RESULT_V1', status=status,
                                                            table=rows), indent=1) + '\n')
    print(json.dumps(status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-inputs', action='store_true')
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--groups', nargs='*')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.build_inputs:
        build_inputs()
    if args.measure:
        measure()
    if args.run:
        return run(args.workers, set(args.groups) if args.groups else None)
    if args.summarize:
        summarize()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
