#!/usr/bin/env python3
"""S0: zero-compile step-decomposition audit of saved PR-Pauli LD2 compiled QPY.

Plan: docs/research/a_pauli_ld2_expected_price_plan_20261115/REPORT.md §5 (df5e5c452).
Rules: docs/research/a_pauli_ld2_step_audit_20261115/AUDIT_RULES.md (fixed before any real QPY
was opened; run01 executed SOURCE 2f3bf20ce). This revision implements S0-r1
(AUDIT_RULES_R1.md, approved re-analysis of the same 52 QPY) and writes to run02/. The saved private QPY and cache JSON are read read-only (at most 52 QPY: 13 paths
x X/Y x {wrapper, compiled}). The audit never builds, samples, transpiles or compiles a
circuit, never writes private files, never fits or predicts prices and never computes
E[C], CI or G. Wire tokens follow the PF closure audit (d3338c7ef `pf_qpy_closure_audit.py`:
name, qubit tuple, float.hex params; final ancilla 1q run cut after its last multi-qubit gate).

    python -B scripts/pauli_ld2_step_audit.py --build-inputs
    python -B scripts/pauli_ld2_step_audit.py --measure
    python -B scripts/pauli_ld2_step_audit.py --run --workers N
    python -B scripts/pauli_ld2_step_audit.py --summarize
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/a_pauli_ld2_step_audit_20261115'
RUN = OUT / 'run02'           # S0-r1 (approved re-analysis of the same 52 QPY); run01 = SOURCE 2f3bf20ce
WORK = RUN / 'work'           # regenerable token arrays, git-ignored

PLAN = ('df5e5c452a3814f7e04ea9cacf05c8f4bdcc047c', 'docs/research/a_pauli_ld2_expected_price_plan_20261115')
W_ACCEPT = '1e2639a1be46ebea618b15591fb1e886bc499d33'
W_RAW = ('6c4709f85d0f28ce7c6957e9a70c85e378d8cad0', 'docs/research/d_pauli_ld2_b1_w_results_20261115/run01')
CAL = ('5dac74959ed9ea49465d393fc792ef5b4727edba', 'docs/research/d_pauli_ld2_b1_cal_results_20261115')
HOLDOUT = ('705f59ea863e1b9fe90e023f1cb1057d00366b16', 'docs/research/d_pauli_ld2_b1_holdout_results_20261115')
M9_FINAL = ('cb034e547c444ea67a4f4f90a8534c13f4db4451', 'docs/research/d_pauli_ld2_m9_longpoint_results_20261115')
PF_TOKENS = ('d3338c7ef', 'scripts/pf_qpy_closure_audit.py')
SCALAR_SOURCE = ('ff536ed2a6effc6c01899a7dcde76e5361448b85',
                 'docs/research/d_pauli_ld2_b1_cal_results_20261115/run01/S66/RESULT.json')

COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
QISKIT = '1.3.0'
METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')
MAX_QPY_READS = 52
ANGLE_MATCH = 1e-12          # compiled barrier rz vs uncompiled angle (wrapped)
# S0-r1 (proposed after run01, NOT the rule run01 used): a D-region pair whose uncompiled rz angle
# is below Qiskit 1.3.0's identity threshold is removed by the compiler together with its CX
# pair, deterministically in every step. run01 (SOURCE 2f3bf20ce) had no such exclusion.
IDENTITY_PAIR_EPS = 1e-12
S0R1_IDENTITY_PAIRS = True
RELATION_TOLERANCE = 1e-9    # final-run theta1 vs slope -scalar (as PF A4)
MARGIN_THRESHOLD = 1e-6      # final-run RZ omission margin (as PF A4)
SYNTH_TOLERANCE = 1e-9       # ZSX synthesis of the wrapper input run vs compiled run
TWO_PI = 2 * math.pi
TAIL_EVENT, TAIL_PRODUCT, D_OP = 'TAIL_EVENT', 'TAIL_PRODUCT', 'D'


class AuditStop(RuntimeError):
    """Mandatory STOP conditions (AUDIT_RULES §6); partial results are kept."""


class StructureFail(RuntimeError):
    """A rule-level FAIL (no relaxation, no retry); `detail` records where it happened."""

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail


# ---------------------------------------------------------------- utilities
def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True).stdout


def git_json(commit, path):
    return json.loads(git('show', f'{commit}:{path}'))


def wrap(angle):
    value = math.fmod(angle + math.pi, TWO_PI)
    if value <= 0:
        value += TWO_PI
    return value - math.pi


def safe_hex(p):
    try:
        return float(p).hex()
    except (TypeError, ValueError):
        return 'NONFLOAT:' + type(p).__name__


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def token_hash(rec):
    return int.from_bytes(hashlib.blake2b(repr(rec).encode(), digest_size=8).digest(), 'little')


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Progress:
    def __init__(self, path, label):
        self.path, self.label = path, label

    def __call__(self, event, **data):
        rec = dict(time=time.time(), label=self.label, event=event, max_rss_mib=round(rss_mib(), 1), **data)
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(rec) + '\n')


def check_runtime():
    import qiskit
    if qiskit.__version__ != QISKIT:
        raise AuditStop('QISKIT_VERSION:' + qiskit.__version__)


def load_qpy(path):
    from qiskit import qpy
    with open(path, 'rb') as handle:          # read-only
        raw = gzip.decompress(handle.read())
    circuits = qpy.load(io.BytesIO(raw))
    if len(circuits) != 1:
        raise AuditStop('QPY_CIRCUIT_COUNT:' + str(path))
    return circuits[0]


# ---------------------------------------------------------------- tokens
def tokenize(circuit, *, hash_tokens=True):
    """Compiled-circuit tokens: (name, qubits, float.hex params) per instruction, per-wire positions.

    Directives are skipped; `measure` is counted but not tokenized (as the PF audit). Counts
    follow the PF audit: size counts every non-directive instruction including measure.
    """
    n = circuit.num_qubits
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    instrs, hashes, wires = [], [], [[] for _ in range(n)]
    counts = dict(rz=0, cx=0, size=0, measure=0, other=0)
    cache = {}
    for inst in circuit.data:
        op = inst.operation
        if getattr(op, '_directive', False):
            continue
        counts['size'] += 1
        name = op.name
        if name == 'measure':
            counts['measure'] += 1
            continue
        if name in ('rz', 'cx'):
            counts[name] += 1
        else:
            counts['other'] += 1
        qb = tuple(index[q] for q in inst.qubits)
        rec = (name, qb, tuple(safe_hex(p) for p in op.params))
        j = len(instrs)
        instrs.append(rec)
        if hash_tokens:
            h = cache.get(rec)
            if h is None:
                h = cache[rec] = token_hash(rec)
            hashes.append(h)
        for x in qb:
            wires[x].append(j)
    return dict(n=n, instrs=instrs, hashes=hashes, wires=wires, counts=counts)


def final_ancilla_run(tok):
    """Ancilla positions after its last multi-qubit instruction (PF definition)."""
    anc = tok['n'] - 1
    pos = tok['wires'][anc]
    last = -1
    for k, j in enumerate(pos):
        if len(tok['instrs'][j][1]) > 1:
            last = k
    return pos[last + 1:], last


def final_run_angles(tok, run_positions):
    names = [tok['instrs'][j][0] for j in run_positions]
    angles = [float.fromhex(tok['instrs'][j][2][0]) for j in run_positions if tok['instrs'][j][0] == 'rz']
    return names, angles


def synthesize_input_run(tail_records):
    """Expected compiled final run from the uncompiled ancilla input run (no transpile; PF A4)."""
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


# ---------------------------------------------------------------- uncompiled structure (AUDIT_RULES §2)
def uncompiled_structure(circuit, q_req):
    """Steps of the saved uncompiled wrapper from its ancilla operations (construction order).

    Per step: tail = ancilla 2q ops up to the step's ancilla `p` (crz = order-0 event, cx/cy/cz =
    product token); D-region = ancilla ops after that `p` up to the next tail `crz` (or the final
    `p`). D-region ops must be `cx(anc, o)` in triples cx(anc,o), rz(o), cx(anc,o) (consecutive in
    construction order) separated by ancilla `rz` (block markers). Returns the expected compiled
    ancilla-CX target list with labels.
    """
    n = circuit.num_qubits
    anc = n - 1
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    ops = []
    for inst in circuit.data:
        op = inst.operation
        if getattr(op, '_directive', False):
            continue
        ops.append((op.name, tuple(index[q] for q in inst.qubits), list(op.params)))
    anc_k = [k for k, (name, qb, _) in enumerate(ops) if anc in qb and name != 'measure']
    p_marks = [k for k in anc_k if ops[k][0] == 'p' and len(ops[k][1]) == 1]
    if len(p_marks) != q_req + 1:
        raise StructureFail(f'ANCILLA_P_COUNT:{len(p_marks)}!=q+1={q_req + 1}')
    final_p = p_marks[-1]
    tail_ps = set(p_marks[:-1])
    steps, pre, post = [], [], []
    state, cur = 'TAIL', dict(tail=[], d=[], tail_1q=[])
    for k in anc_k:
        name, qb, params = ops[k]
        if k == final_p:
            if state != 'D':
                raise StructureFail('FINAL_P_OUTSIDE_D_REGION')
            steps.append(cur)
            state = 'POST'
            continue
        if state == 'POST':
            post.append(name)
            continue
        if state == 'TAIL':
            if k in tail_ps:
                state = 'D'
                cur['p'] = safe_hex(params[0])
                continue
            if len(qb) == 2:
                if qb[0] != anc:
                    raise StructureFail('ANCILLA_NOT_CONTROL_IN_TAIL')
                cur['tail'].append((k, name, qb[1]))
            elif not steps and not cur['tail']:
                pre.append(name)
            else:
                cur['tail_1q'].append(name)
            continue
        # state == 'D'
        if name == 'crz':
            steps.append(cur)
            cur = dict(tail=[(k, name, qb[1])], d=[], tail_1q=[])
            state = 'TAIL'
            continue
        if len(qb) == 2:
            if name != 'cx' or qb[0] != anc:
                raise StructureFail('UNEXPECTED_D_REGION_2Q:' + name)
            cur['d'].append(('cx', k, qb[1]))
        elif name == 'rz':
            cur['d'].append(('rz', k, safe_hex(params[0])))
        else:
            raise StructureFail('UNEXPECTED_D_REGION_1Q:' + name)
    if state != 'POST' or len(steps) != q_req:
        raise StructureFail(f'STEP_COUNT:{len(steps)}!=q={q_req}')
    expected = []          # (target, label, step, extra)
    for s, st in enumerate(steps, 1):
        st['n_event'] = sum(1 for _, nm, _ in st['tail'] if nm == 'crz')
        st['n_product'] = sum(1 for _, nm, _ in st['tail'] if nm != 'crz')
        for k, nm, t in st['tail']:
            for _ in range(2 if nm == 'crz' else 1):
                expected.append((t, TAIL_EVENT if nm == 'crz' else TAIL_PRODUCT, s, None))
        # D-region: blocks split by ancilla rz markers; every cx belongs to a triple
        blocks, block, gps, pairs = [], [], [], []
        d_list = st['d']
        i = 0
        while i < len(d_list):
            kind = d_list[i][0]
            if kind == 'rz':
                gps.append(d_list[i][2])
                blocks.append(block)
                block = []
                i += 1
                continue
            _, k1, o = d_list[i]
            if i + 1 >= len(d_list) or d_list[i + 1][0] != 'cx' or d_list[i + 1][2] != o:
                raise StructureFail('D_CX_NOT_PAIRED')
            k2 = d_list[i + 1][1]
            mid = ops[k1 + 1]
            if not (k2 == k1 + 2 and mid[0] == 'rz' and mid[1] == (o,)):
                raise StructureFail('D_PAIR_NOT_CX_RZ_CX')
            identity = S0R1_IDENTITY_PAIRS and abs(float(mid[2][0])) < IDENTITY_PAIR_EPS
            pair = dict(target=o, angle=safe_hex(mid[2][0]), block=len(blocks), identity=identity)
            block.append(pair)
            pairs.append(pair)
            if not identity:
                for half in (0, 1):
                    expected.append((o, D_OP, s, dict(pair=len(pairs) - 1, half=half)))
            i += 2
        if block:
            blocks.append(block)
        st['blocks'] = blocks
        st['gps'] = gps
        st['pairs'] = pairs
        st['identity_pairs'] = sum(1 for pr in pairs if pr['identity'])
        st['signature'] = hashlib.sha256(json.dumps(
            [[(p['target'], p['angle']) for p in b] for b in blocks] + [gps]).encode()).hexdigest()
    return dict(n=n, anc=anc, steps=steps, expected=expected, pre=pre, post=post,
                final_p=safe_hex(ops[final_p][2][0]), ops_count=len(ops))


def uncompiled_signature(circuit):
    """X/Y comparison of uncompiled wrappers: (name, qubits, params) list."""
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    return [(i.operation.name, tuple(index[q] for q in i.qubits), tuple(safe_hex(p) for p in i.operation.params))
            for i in circuit.data if not getattr(i.operation, '_directive', False)]


# ---------------------------------------------------------------- compiled segmentation (AUDIT_RULES §3)
def segment_compiled(tok, struct):
    """Align compiled ancilla CX tokens with the uncompiled expectation and cut every wire.

    Cut of step s on wire w = compiled position of the last D-region ancilla CX of step s that
    targets w (ancilla: last D-region CX of step s). Raises StructureFail on any rule failure.
    """
    n, anc = tok['n'], tok['n'] - 1
    instrs, wires = tok['instrs'], tok['wires']
    run_pos, _ = final_ancilla_run(tok)
    run_set = set(run_pos)
    anc_2q = [j for j in wires[anc] if len(instrs[j][1]) > 1]
    for j in anc_2q:
        name, qb, _ = instrs[j]
        if name != 'cx' or qb[0] != anc:
            raise StructureFail('COMPILED_ANCILLA_2Q_NOT_CX_FROM_ANCILLA:' + name)
    exp = struct['expected']
    got = [instrs[j][1][1] for j in anc_2q]
    want = [e[0] for e in exp]
    if got != want:
        m = min(len(got), len(want))
        i = next((k for k in range(m) if got[k] != want[k]), m)
        detail = dict(first_mismatch_index=i, compiled_count=len(got), expected_count=len(want),
                      expected_entry=[exp[i][0], exp[i][1], exp[i][2], exp[i][3]] if i < len(want) else None,
                      compiled_target=got[i] if i < len(got) else None,
                      compiled_instruction_index=anc_2q[i] if i < len(anc_2q) else None,
                      expected_window=[[e[0], e[1], e[2], e[3]] for e in exp[max(0, i - 3):i + 4]],
                      compiled_window=got[max(0, i - 3):i + 4])
        import difflib
        ops = [op for op in difflib.SequenceMatcher(a=want, b=got, autojunk=False).get_opcodes() if op[0] != 'equal']
        detail['difference_blocks'] = len(ops)
        detail['differences'] = [dict(tag=tag, expected_range=[i1, i2], compiled_range=[j1, j2],
                                      expected_entries=[[e[0], e[1], e[2], e[3]] for e in exp[i1:i2][:12]],
                                      compiled_targets=got[j1:j2][:12])
                                 for tag, i1, i2, j1, j2 in ops[:10]]
        raise StructureFail(f'ANCILLA_CX_ALIGNMENT:{len(got)}vs{len(want)}:first_mismatch={i}', detail)
    q = len(struct['steps'])
    # pair verification (barrier structure of every D-region pair)
    pos_on = [{j: k for k, j in enumerate(w)} for w in wires]
    anc_pos = pos_on[anc]
    pair_cx = {}
    for j, e in zip(anc_2q, exp):
        if e[1] == D_OP:
            pair_cx.setdefault((e[2], e[3]['pair']), []).append(j)
    pair_checks = dict(pairs=0, exact_hex=0)
    for (s, pid), (j1, j2) in pair_cx.items():
        pair = struct['steps'][s - 1]['pairs'][pid]
        o = pair['target']
        k1, k2 = pos_on[o][j1], pos_on[o][j2]
        if k2 - k1 != 2 or anc_pos[j2] - anc_pos[j1] != 1:
            raise StructureFail(f'BARRIER_PAIR_NOT_ADJACENT:step{s}:pair{pid}')
        mid = instrs[wires[o][k1 + 1]]
        if mid[0] != 'rz' or mid[1] != (o,):
            raise StructureFail(f'BARRIER_PAIR_MIDDLE_NOT_RZ:step{s}:pair{pid}')
        got, want = float.fromhex(mid[2][0]), float.fromhex(pair['angle'])
        if abs(wrap(got - want)) > ANGLE_MATCH:
            raise StructureFail(f'BARRIER_RZ_ANGLE:step{s}:pair{pid}')
        pair_checks['pairs'] += 1
        pair_checks['exact_hex'] += int(mid[2][0] == pair['angle'])
    # cuts
    cuts = [[None] * q for _ in range(n)]
    for j, e in zip(anc_2q, exp):
        if e[1] == D_OP:
            s = e[2]
            cuts[e[0]][s - 1] = j       # last D CX on target wire (overwritten in order)
            cuts[anc][s - 1] = j
    for w in range(n):
        if any(c is None for c in cuts[w]):
            raise StructureFail(f'WIRE_WITHOUT_CUT:{w}')
        if cuts[w] != sorted(cuts[w]):
            raise StructureFail(f'CUTS_NOT_MONOTONE:{w}')
    seg_of = []
    for w in range(n):
        seg_of.append({j: bisect_left(cuts[w], j) + 1 for j in wires[w]})
    for j, (name, qb, _) in enumerate(instrs):
        if len(qb) > 1 and len({seg_of[w][j] for w in qb}) != 1:
            raise StructureFail(f'MULTI_QUBIT_TOKEN_ACROSS_CUT:{j}')
    # per-wire segments (positions), end edge, ancilla final run excluded
    segments = [[[] for _ in range(q)] for _ in range(n)]
    ends = [[] for _ in range(n)]
    for w in range(n):
        for j in wires[w]:
            if w == anc and j in run_set:
                continue
            s = seg_of[w][j]
            (ends[w] if s > q else segments[w][s - 1]).append(j)
    if ends[anc]:
        raise StructureFail('ANCILLA_END_EDGE_NOT_EMPTY')
    # D-region markers per segment (for P3b/P3c): first D CX of the last two blocks / first block
    markers = []
    d_by_step = {}
    for j, e in zip(anc_2q, exp):
        if e[1] == D_OP:
            d_by_step.setdefault(e[2], []).append((j, e))
    for s, st in enumerate(struct['steps'], 1):
        nb = len(st['blocks'])
        need = {}
        for j, e in d_by_step.get(s, []):
            blk = st['pairs'][e[3]['pair']]['block']
            for w in (e[0], anc):
                for label, first_block in (('from_second_last_block', max(nb - 2, 0)), ('from_first_block', 0)):
                    if blk >= first_block:
                        key = (label, w)
                        if key not in need:
                            need[key] = j
        # wires without a pair in the second-to-last/last blocks fall back to their cut
        m = {}
        for label in ('from_second_last_block', 'from_first_block'):
            m[label] = []
            for w in range(n):
                seg = segments[w][s - 1]
                j0 = need.get((label, w), cuts[w][s - 1])
                k0 = seg.index(j0)
                m[label].append(len(seg) - k0)
        d_first = need.get(('from_first_block', anc))
        seg_a = segments[anc][s - 1]
        anc_d = [tok['hashes'][j] for j in seg_a[seg_a.index(d_first):]]
        m['ancilla_D_region_digest'] = hashlib.sha256(json.dumps(anc_d).encode()).hexdigest()
        markers.append(m)
    return dict(cuts=cuts, segments=segments, ends=ends, final_run=run_pos, markers=markers,
                pair_checks=pair_checks, anc_cx=len(anc_2q))


def wire_arrays(tok, positions):
    import numpy as np
    h = np.array([tok['hashes'][j] for j in positions], dtype=np.uint64)
    rz = np.array([tok['instrs'][j][0] == 'rz' for j in positions], dtype=bool)
    mq = np.array([len(tok['instrs'][j][1]) > 1 for j in positions], dtype=bool)
    return h, rz, mq


# ---------------------------------------------------------------- inputs (AUDIT_RULES §1)
def build_inputs():
    plan_axis = git_json(PLAN[0], PLAN[1] + '/AXIS_MAP.json')
    rows = {(r['m'], r['axis']): r for r in plan_axis['rows']}
    round_T = {r['m']: r['T'] for r in plan_axis['rows'] if r['axis'] == 'X'}
    paths = {}

    def add(path_id, axis, entry, common):
        rec = paths.setdefault(path_id, dict(path_id=path_id, axes={}, **common))
        for key, value in common.items():
            if rec.get(key) != value:
                raise AuditStop(f'INPUT_PATH_INCONSISTENT:{path_id}:{key}')
        rec['axes'][axis] = entry

    def files_entry(files, fingerprint):
        out = {}
        for f in files:
            p = f['path']
            kind = ('wrapper' if p.endswith('_wrapper.qpy.gz') else 'compiled' if p.endswith('_compiled.qpy.gz')
                    else 'cache_json' if p.endswith(fingerprint + '.json') else None)
            if kind is None:
                continue
            path = Path(p)
            out[kind] = dict(path=p, sha256=f['sha256'], bytes=f.get('bytes'), exists=path.exists(),
                             size_on_disk=path.stat().st_size if path.exists() else None)
        return out

    # W m0 (wiring)
    prices = git_json(W_RAW[0], W_RAW[1] + '/PRICES.json')
    trace = git_json(W_RAW[0], W_RAW[1] + '/RECEIPT_TRACE.json')
    for sel, pr in enumerate(prices):
        axis = pr['axis']
        cand = pr['identity']['candidate']
        if cand['id'] != rows[(0, axis)]['physical_id']:
            raise AuditStop('W_PHYSICAL_ID')
        tr = trace[axis]
        entry = dict(fingerprint=tr['fingerprint'], cache_reused=tr['cache_reused'],
                     metrics={k: int(pr['metrics'][k]) for k in METRICS},
                     receipt=dict(commit=W_RAW[0], path=W_RAW[1] + '/PRICES.json', selector=[sel],
                                  trace=dict(commit=W_RAW[0], path=W_RAW[1] + '/RECEIPT_TRACE.json', selector=[axis])),
                     files=files_entry(tr['files'], tr['fingerprint']))
        T = Fraction(cand['T']['numerator'], cand['T']['denominator'])
        add('W_M0/p0', axis, entry, dict(kind='WIRING', cell='W_M0', index=0, m=0, q=cand['q'], R=cand['R_outer'],
                                          T_hex=float(T).hex(), physical_id=cand['id'], cohort=pr['cohort'],
                                          composed_of=None, acceptance=W_ACCEPT))
    # CAL 5 cells (9 paths, 18 receipts)
    handoff = git_json(CAL[0], CAL[1] + '/C_HANDOFF.json')
    cells = {c['cell_id']: c for c in handoff['cells']}
    obs = list(csv.DictReader(io.StringIO(git('show', f'{CAL[0]}:{CAL[1]}/OBSERVATIONS.csv').decode())))
    obs_by_fp = {o['fingerprint']: o for o in obs}
    receipt_cache = {}
    for item in handoff['individual_compile_receipts']:
        loc = item['receipt']
        key = (loc['commit'], loc['path'])
        if key not in receipt_cache:
            receipt_cache[key] = git_json(*key)
        rec = receipt_cache[key][loc['selector'][0]]
        cell = cells[item['cell_id']]
        fp = rec['record']['fingerprint']
        if fp != item['fingerprint'] or rec['axis'] != item['axis'] or rec['index'] != item['index']:
            raise AuditStop('CAL_RECEIPT_BINDING:' + item['cell_id'])
        req = rec['request']
        if req['candidate_id'] != cell['physical_id'] or (req['q'], req['R']) != (cell['q'], cell['R']):
            raise AuditStop('CAL_PHYSICAL:' + item['cell_id'])
        files = rec['artifacts_private'] + [dict(path=rec['native_cache']['path'], sha256=rec['native_cache']['sha256'])]
        entry = dict(fingerprint=fp, cache_reused=obs_by_fp[fp]['cache_reused'] == 'True',
                     metrics={k: int(rec['record']['metrics'][k]) for k in METRICS},
                     receipt=loc, files=files_entry(files, fp))
        composed = None
        if cell.get('paired_dependence'):
            composed = ['S66/p0', 'S66/p1'] if cell['paired_dependence'] == ['S66/path0', 'S66/path1'] else None
            if composed is None:
                raise AuditStop('PAIRED_DEPENDENCE_UNEXPECTED')
        add(f"{item['cell_id']}/p{item['index']}", item['axis'], entry,
            dict(kind='CALIBRATION', cell=item['cell_id'], index=item['index'], m=None, q=req['q'], R=req['R'],
                 T_hex=float(req['T']).hex(), physical_id=req['candidate_id'], cohort=rec['cohort'],
                 composed_of=composed, acceptance=CAL[0]))
    # holdout m5/m6 (3 paths, 6 receipts)
    adm = git_json(HOLDOUT[0], HOLDOUT[1] + '/RECEIPT_ADMISSION.json')
    for unit in adm['units']:
        for p in unit['paths']:
            loc = p['receipt']
            key = (loc['commit'], loc['path'])
            if key not in receipt_cache:
                receipt_cache[key] = git_json(*key)
            rec = receipt_cache[key][loc['selector'][0]]
            fp = rec['record']['fingerprint']
            if fp != p['fingerprint'] or rec['axis'] != unit['axis']:
                raise AuditStop('HOLDOUT_RECEIPT_BINDING')
            if rec['record']['metrics']['rz_count'] != p['compiled_RZ']:
                raise AuditStop('HOLDOUT_RZ_TRANSCRIPTION')
            req = rec['request']
            if req['candidate_id'] != rows[(unit['m'], unit['axis'])]['physical_id']:
                raise AuditStop('HOLDOUT_PHYSICAL_ID')
            files = rec['artifacts_private'] + [dict(path=rec['native_cache']['path'], sha256=rec['native_cache']['sha256'])]
            entry = dict(fingerprint=fp, cache_reused=p['cache_reused'],
                         metrics={k: int(rec['record']['metrics'][k]) for k in METRICS},
                         receipt=loc, files=files_entry(files, fp))
            add(f"{unit['cell_id']}/p{p['index']}", unit['axis'], entry,
                dict(kind='HOLDOUT', cell=unit['cell_id'], index=p['index'], m=unit['m'], q=req['q'], R=req['R'],
                     T_hex=float(req['T']).hex(), physical_id=req['candidate_id'], cohort=rec['cohort'],
                     composed_of=None, acceptance=HOLDOUT[0]))
    scalar_doc = git_json(*SCALAR_SOURCE)
    scalar = scalar_doc['child']['paths'][0]['result']['scalar']
    reads = sum(1 for p in paths.values() for a in p['axes'].values() for k in ('wrapper', 'compiled') if k in a['files'])
    if len(paths) != 13 or any(sorted(p['axes']) != ['X', 'Y'] for p in paths.values()):
        raise AuditStop('INPUT_PATH_COUNT')
    if reads > MAX_QPY_READS:
        raise AuditStop('READ_BUDGET')
    data = dict(schema='A_PAULI_LD2_STEP_AUDIT_INPUTS_V1',
                rules='docs/research/a_pauli_ld2_step_audit_20261115/AUDIT_RULES.md',
                plan=dict(commit=PLAN[0], path=PLAN[1] + '/REPORT.md#5'),
                sources=dict(wiring=dict(acceptance=W_ACCEPT, raw=W_RAW[0]), calibration=CAL[0],
                             holdout=HOLDOUT[0], m9_final_reference_only=M9_FINAL[0], pf_tokens=PF_TOKENS[0]),
                scalar=dict(value_hex=float(scalar).hex(), value=scalar,
                            source=dict(commit=SCALAR_SOURCE[0], path=SCALAR_SOURCE[1],
                                        selector=['child', 'paths', 0, 'result', 'scalar'])),
                round_T=round_T,
                thresholds=dict(angle_match=ANGLE_MATCH, relation=RELATION_TOLERANCE, margin=MARGIN_THRESHOLD,
                                synthesis=SYNTH_TOLERANCE),
                qpy_reads=dict(total=reads, limit=MAX_QPY_READS),
                paths=[paths[k] for k in sorted(paths)], private_contents_read=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'INPUTS.json').write_text(json.dumps(data, indent=1) + '\n')
    missing = [f['path'] for p in paths.values() for a in p['axes'].values() for f in a['files'].values() if not f['exists']]
    print(json.dumps(dict(paths=len(paths), reads=reads, missing=missing, scalar=scalar)))


# ---------------------------------------------------------------- per-path worker (P1, P2, P5 parts)
def p1_bind(path, axis, entry, tok):
    files = entry['files']
    cache = json.loads(Path(files['cache_json']['path']).read_text())
    checks = dict(
        wrapper_sha256=sha256_file(files['wrapper']['path']) == files['wrapper']['sha256'],
        compiled_sha256=sha256_file(files['compiled']['path']) == files['compiled']['sha256'],
        cache_json_sha256=sha256_file(files['cache_json']['path']) == files['cache_json']['sha256'],
        cache_fingerprint=cache.get('fingerprint') == entry['fingerprint'],
        cache_compiler=cache.get('compiler') == COMPILER,
        cache_metrics_equal_receipt=all(cache.get('metrics', {}).get(k) == entry['metrics'][k] for k in METRICS),
        not_cache_reused=entry['cache_reused'] is False,
        token_rz_equal_receipt=tok['counts']['rz'] == entry['metrics']['rz_count'],
        token_cx_equal_receipt=tok['counts']['cx'] == entry['metrics']['cx_count'],
        token_size_equal_receipt=tok['counts']['size'] == entry['metrics']['circuit_size'],
        compiled_only_basis_gates=tok['counts']['other'] == sum(1 for r in tok['instrs'] if r[0] in ('sx', 'x')),
    )
    return checks


def audit_path(path, scalar, progress_path):
    import numpy as np
    prog = Progress(progress_path, path['path_id'])
    out = dict(path_id=path['path_id'], kind=path['kind'], cell=path['cell'], index=path['index'], m=path['m'],
               q=path['q'], R=path['R'], T_hex=path['T_hex'], composed_of=path['composed_of'], axes={})
    arrays = {}
    unc_sig = {}
    streams = {}
    for axis in ('X', 'Y'):
        entry = path['axes'][axis]
        prog('load_start', axis=axis, file='wrapper')
        t0 = time.monotonic()
        wrapper = load_qpy(entry['files']['wrapper']['path'])
        prog('load_done', axis=axis, file='wrapper', seconds=time.monotonic() - t0, instructions=len(wrapper.data))
        unc_sig[axis] = (wrapper.num_qubits - 1, uncompiled_signature(wrapper))
        res = dict()
        try:
            struct = uncompiled_structure(wrapper, path['q'])
            res['structure'] = 'PASS'
        except StructureFail as exc:
            struct = None
            res['structure'] = 'FAIL:' + str(exc)
            res['structure_detail'] = exc.detail
        input_tail = wrapper_input_tail(wrapper)
        del wrapper
        prog('load_start', axis=axis, file='compiled')
        t0 = time.monotonic()
        compiled = load_qpy(entry['files']['compiled']['path'])
        tok = tokenize(compiled)
        del compiled
        prog('load_done', axis=axis, file='compiled', seconds=time.monotonic() - t0, instructions=tok['counts']['size'])
        p1 = p1_bind(path, axis, entry, tok)
        res['P1_checks'] = p1
        run_set_xy = set(final_ancilla_run(tok)[0])
        streams[axis] = [np.array([tok['hashes'][j] for j in tok['wires'][w]
                                   if not (w == tok['n'] - 1 and j in run_set_xy)], dtype=np.uint64)
                         for w in range(tok['n'])]
        if not (p1['wrapper_sha256'] and p1['compiled_sha256']):
            raise AuditStop(f"QPY_SHA256_MISMATCH:{path['path_id']}:{axis}")
        if not p1['token_rz_equal_receipt']:
            raise AuditStop(f"RZ_TOKEN_MISMATCH:{path['path_id']}:{axis}:{tok['counts']['rz']}!={entry['metrics']['rz_count']}")
        res['P1'] = 'PASS' if all(p1.values()) else 'FAIL'
        res['counts'] = tok['counts']
        # P5 final run and wrapper input
        run_pos, _ = final_ancilla_run(tok)
        names, angles = final_run_angles(tok, run_pos)
        res['final_run'] = dict(names=names, angles_hex=[a.hex() for a in angles],
                                rz=sum(1 for nm in names if nm == 'rz'))
        exp_names, exp_angles = synthesize_input_run([(nm, pa) for nm, pa in input_tail])
        p_angles = [pa[0] for nm, pa in input_tail if nm == 'p']
        T = float.fromhex(path['T_hex'])
        res['wrapper_input'] = dict(names=[nm for nm, _ in input_tail],
                                    p_equals_minus_scalar_T=len(p_angles) == 1 and p_angles[0] == -scalar * T,
                                    synthesized_matches_compiled=(exp_names == names and len(exp_angles) == len(angles)
                                                                  and all(abs(wrap(a - b)) <= SYNTH_TOLERANCE
                                                                          for a, b in zip(exp_angles, angles))))
        if struct is None:
            res['P2'] = 'FAIL'
            out['axes'][axis] = res
            continue
        res['steps'] = [dict(step=s, events=st['n_event'], products=st['n_product'], tail_1q=st['tail_1q'],
                             d_blocks=len(st['blocks']), d_pairs=len(st['pairs']), signature=st['signature'],
                             identity_pairs=st['identity_pairs'], tail_p_hex=st.get('p'))
                        for s, st in enumerate(struct['steps'], 1)]
        res['events_equal_R'] = all(st['n_event'] == path['R'] for st in struct['steps'])
        res['products_total'] = sum(st['n_product'] for st in struct['steps'])
        res['barrier'] = [(p['target'], p['angle']) for p in struct['steps'][-1]['blocks'][-1]] if struct['steps'][-1]['blocks'] else []
        try:
            seg = segment_compiled(tok, struct)
            res['P2'] = 'PASS'
        except StructureFail as exc:
            res['P2'] = 'FAIL:' + str(exc)
            res['P2_detail'] = exc.detail
            out['axes'][axis] = res
            continue
        res['pair_checks'] = seg['pair_checks']
        res['markers'] = seg['markers']
        n = tok['n']
        per_wire = []
        for w in range(n):
            segs = [wire_arrays(tok, positions) for positions in seg['segments'][w]]
            end = wire_arrays(tok, seg['ends'][w])
            per_wire.append(dict(segs=segs, end=end))
        seg_rz = [int(sum(per_wire[w]['segs'][s][1].sum() for w in range(n))) for s in range(path['q'])]
        end_rz = int(sum(per_wire[w]['end'][1].sum() for w in range(n)))
        res['segment_rz'] = seg_rz
        res['end_rz'] = end_rz
        res['completeness'] = sum(seg_rz) + end_rz + res['final_run']['rz'] == tok['counts']['rz']
        arrays[axis] = per_wire
        out['axes'][axis] = res
        prog('segmented', axis=axis, steps=path['q'])
    # X/Y comparison (P5)
    xy = dict(uncompiled_only_sdg_differs=None, compiled_equal_except_final_run=None)
    (anc_x, ux), (anc_y, uy) = unc_sig['X'], unc_sig['Y']
    removed = [r for r in uy if r[0] == 'sdg' and r[1] == (anc_y,)]
    xy['uncompiled_only_sdg_differs'] = (anc_x == anc_y and len(removed) == 1
                                         and [r for r in uy if not (r[0] == 'sdg' and r[1] == (anc_y,))] == ux)
    if 'X' in streams and 'Y' in streams:
        xy['compiled_equal_except_final_run'] = bool(
            len(streams['X']) == len(streams['Y'])
            and all(np.array_equal(a, b) for a, b in zip(streams['X'], streams['Y'])))
    out['xy'] = xy
    # persist X arrays (regenerable work file) for cross-path analysis
    if 'X' in arrays:
        WORK.mkdir(parents=True, exist_ok=True)
        blobs = {}
        for w, wa in enumerate(arrays['X']):
            for s, (h, rz, mq) in enumerate(wa['segs'], 1):
                blobs[f'w{w}_s{s}_h'], blobs[f'w{w}_s{s}_rz'], blobs[f'w{w}_s{s}_mq'] = h, rz, mq
            h, rz, mq = wa['end']
            blobs[f'w{w}_end_h'], blobs[f'w{w}_end_rz'], blobs[f'w{w}_end_mq'] = h, rz, mq
        np.savez_compressed(WORK / (path['path_id'].replace('/', '_') + '_X.npz'), **blobs)
        out['n_wires'] = len(arrays['X'])
    prog('done')
    return out


def run_path(job):
    path, scalar = job
    target = RUN / ('PATH_' + path['path_id'].replace('/', '_') + '.json')
    try:
        check_runtime()
        result = audit_path(path, scalar, RUN / 'PROGRESS.jsonl')
        target.write_text(json.dumps(result, indent=1, default=str) + '\n')
        return dict(path_id=path['path_id'], status='DONE', pid=os.getpid(), max_rss_mib=round(rss_mib(), 1))
    except BaseException as exc:
        stop = dict(path_id=path['path_id'], status='STOP', reason=str(exc), traceback=traceback.format_exc(),
                    pid=os.getpid(), max_rss_mib=round(rss_mib(), 1))
        target.with_suffix('.STOP.json').write_text(json.dumps(stop, indent=1) + '\n')
        return stop


def measure():
    check_runtime()
    data = json.loads((OUT / 'INPUTS.json').read_text())
    best = min(((p, a) for p in data['paths'] for a in ('X',)), key=lambda pa: pa[0]['axes'][pa[1]]['files']['compiled']['bytes'])
    path, axis = best
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    circuit = load_qpy(path['axes'][axis]['files']['compiled']['path'])
    loaded = time.monotonic() - t0
    tok = tokenize(circuit)
    out = dict(path_id=path['path_id'], axis=axis, bytes=path['axes'][axis]['files']['compiled']['bytes'],
               instructions=tok['counts']['size'], load_seconds=loaded, total_seconds=time.monotonic() - t0,
               max_rss_mib=rss_mib(), rz_tokens=tok['counts']['rz'], saved_rz=path['axes'][axis]['metrics']['rz_count'])
    (RUN / 'MEASURE.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


def run(workers):
    check_runtime()
    RUN.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'INPUTS.json').read_text())
    scalar = float.fromhex(data['scalar']['value_hex'])
    jobs = [(p, scalar) for p in data['paths']]
    started = time.monotonic()
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_path, job): job[0]['path_id'] for job in jobs}
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


# ---------------------------------------------------------------- cross-path analysis (P3, P4, P5, P6)
def lcp_len(arrays):
    import numpy as np
    if not arrays:
        return 0
    ref = arrays[0]
    L = len(ref)
    for a in arrays[1:]:
        m = min(L, len(a))
        if m == 0:
            return 0
        neq = np.flatnonzero(ref[:m] != a[:m])
        L = int(neq[0]) if neq.size else m
    return L


def lcs_len(arrays):
    return lcp_len([a[::-1] for a in arrays])


def leveled_compare(x, y, px, py):
    """P4 level of two segments x, y (h, rz, mq arrays) whose deterministic entry prefixes are px, py.

    c = longest common suffix. EXACT: everything outside the entry prefixes is identical
    (len(x) - c <= px and len(y) - c <= py). LEADING_1Q_RUN_ONLY: the non-common parts beyond the
    prefixes contain no multi-qubit token (a 1q junction run only). DIFFER otherwise.
    """
    import numpy as np
    c = lcs_len([x[0], y[0]])
    dx, dy = len(x[0]) - c, len(y[0]) - c
    if dx <= px and dy <= py:
        return 'EXACT'
    if not x[2][px:dx].any() and not y[2][py:dy].any():
        return 'LEADING_1Q_RUN_ONLY'
    return 'DIFFER'


def load_x_arrays(path_id, q, n):
    import numpy as np
    z = np.load(WORK / (path_id.replace('/', '_') + '_X.npz'))
    wires = []
    for w in range(n):
        segs = [(z[f'w{w}_s{s}_h'], z[f'w{w}_s{s}_rz'], z[f'w{w}_s{s}_mq']) for s in range(1, q + 1)]
        end = (z[f'w{w}_end_h'], z[f'w{w}_end_rz'], z[f'w{w}_end_mq'])
        wires.append(dict(segs=segs, end=end))
    return wires


def summarize():
    import numpy as np
    data = json.loads((OUT / 'INPUTS.json').read_text())
    scalar = float.fromhex(data['scalar']['value_hex'])
    results = {}
    stops = []
    for p in data['paths']:
        f = RUN / ('PATH_' + p['path_id'].replace('/', '_') + '.json')
        if f.exists():
            results[p['path_id']] = json.loads(f.read_text())
        else:
            stops.append(p['path_id'])
    table = []
    ok_paths = [pid for pid, r in results.items()
                if all(str(r['axes'][a].get('P2', '')).startswith('PASS') for a in ('X', 'Y'))]
    n = max((results[pid].get('n_wires', 0) for pid in ok_paths), default=0)
    arrays = {pid: load_x_arrays(pid, results[pid]['q'], n) for pid in ok_paths}
    # P3: deterministic boundary form ---------------------------------------------------------
    struct_paths = [pid for pid, r in results.items() if all(r['axes'][a].get('structure') == 'PASS' for a in ('X', 'Y'))]
    sigs = {st['signature'] for pid in struct_paths for a in ('X', 'Y') for st in results[pid]['axes'][a]['steps']}
    barriers = {json.dumps(results[pid]['axes'][a]['barrier']) for pid in struct_paths for a in ('X', 'Y')}
    anc_digests = {m['ancilla_D_region_digest'] for pid in ok_paths for m in results[pid]['axes']['X']['markers']}
    all_segs = {w: [arrays[pid][w]['segs'][s] for pid in ok_paths for s in range(results[pid]['q'])] for w in range(n)}
    S = {w: lcs_len([h for h, _, _ in all_segs[w]]) for w in range(n)}
    cover = {}
    for w in range(n):
        need_f0 = max(results[pid]['axes']['X']['markers'][s]['from_second_last_block'][w]
                      for pid in ok_paths for s in range(results[pid]['q']))
        need_f1 = max(results[pid]['axes']['X']['markers'][s]['from_first_block'][w]
                      for pid in ok_paths for s in range(results[pid]['q']))
        cover[w] = dict(suffix_len=S[w], need_from_second_last_block=need_f0, need_from_first_block=need_f1,
                        covers_second_last_block=S[w] >= need_f0, covers_first_block=S[w] >= need_f1)
    first = {w: [] for w in range(n)}
    later = {w: [] for w in range(n)}
    for pid in ok_paths:
        for w in range(n):
            for s, (h, rz, mq) in enumerate(arrays[pid][w]['segs']):
                cut = len(h) - S[w]
                (first if s == 0 else later)[w].append(h[:cut])
    F = {w: lcp_len(first[w]) for w in range(n)}
    P = {w: lcp_len(later[w]) if later[w] else 0 for w in range(n)}
    ends_equal = {}
    for w in range(n - 1):
        ends = [arrays[pid][w]['end'][0] for pid in ok_paths]
        ends_equal[w] = all(len(e) == len(ends[0]) and np.array_equal(e, ends[0]) for e in ends)
    ref_pid = ok_paths[0] if ok_paths else None

    def rz_of(arr, start, stop):
        return int(arr[1][start:stop].sum())
    edges = dict(per_wire=[])
    for w in range(n):
        ex = arrays[ref_pid][w]['segs'][0]
        later_ex = next((arrays[pid][w]['segs'][1] for pid in ok_paths if results[pid]['q'] > 1), None)
        edges['per_wire'].append(dict(
            wire=w, ancilla=w == n - 1, suffix_len=S[w], suffix_rz=rz_of(ex, len(ex[0]) - S[w], len(ex[0])),
            first_prefix_len=F[w], first_prefix_rz=rz_of(ex, 0, F[w]),
            later_prefix_len=P[w], later_prefix_rz=rz_of(later_ex, 0, P[w]) if later_ex is not None else None,
            end_len=len(arrays[ref_pid][w]['end'][0]), end_rz=int(arrays[ref_pid][w]['end'][1].sum()),
            end_identical_all_paths=ends_equal.get(w), coverage=cover[w]))
    const = dict(suffix_rz=sum(e['suffix_rz'] for e in edges['per_wire']),
                 first_prefix_rz=sum(e['first_prefix_rz'] for e in edges['per_wire']),
                 later_prefix_rz=sum(e['later_prefix_rz'] or 0 for e in edges['per_wire']),
                 end_rz=sum(e['end_rz'] for e in edges['per_wire']))
    edges['constants'] = const
    P3 = dict(uncompiled_D_region_identical=len(sigs) == 1, barrier_identical=len(barriers) == 1,
              compiled_ancilla_D_region_identical=len(anc_digests) == 1,
              suffix_covers_last_two_D_blocks_all_wires=all(c['covers_second_last_block'] for c in cover.values()),
              system_end_edge_identical=all(ends_equal.values()),
              suffix_covers_all_D_blocks_all_wires_informational=all(c['covers_first_block'] for c in cover.values()))
    P3['status'] = ('UNKNOWN_NO_SEGMENTED_PATH' if not ok_paths else
                    'PASS' if all(v for k, v in P3.items() if not k.endswith('informational')) else 'FAIL')
    P3['uncompiled_part_paths'] = len(struct_paths)
    # P6 tables: segments and bodies ----------------------------------------------------------
    seg_rows = []
    decomposition = {}
    for pid in ok_paths:
        r = results[pid]
        total = 0
        for s in range(r['q']):
            pre = F if s == 0 else P
            seg_rz = body_rz = pre_rz = suf_rz = 0
            body_len = 0
            for w in range(n):
                h, rz, mq = arrays[pid][w]['segs'][s]
                L = len(h)
                cut = L - S[w]
                k = pre[w]
                seg_rz += int(rz.sum())
                suf_rz += int(rz[cut:].sum())
                pre_rz += int(rz[:k].sum())
                body_rz += int(rz[k:cut].sum())
                body_len += cut - k
            total += seg_rz
            independent = r['composed_of'] is None
            seg_rows.append(dict(path_id=pid, kind=r['kind'], cell=r['cell'], index=r['index'], m=r['m'], q=r['q'],
                                 R=r['R'], step=s + 1, context='FIRST' if s == 0 else 'LATER',
                                 events=r['axes']['X']['steps'][s]['events'],
                                 products=r['axes']['X']['steps'][s]['products'], segment_rz=seg_rz,
                                 prefix_rz=pre_rz, body_rz=body_rz, suffix_rz=suf_rz, body_tokens=body_len,
                                 independent_sample=independent))
        ends = sum(int(arrays[pid][w]['end'][1].sum()) for w in range(n))
        fr = r['axes']['X']['final_run']['rz']
        saved = data_metric(data, pid, 'X', 'rz_count')
        decomposition[pid] = dict(segments_rz=total, end_rz=ends, final_run_rz=fr, sum=total + ends + fr, saved_rz=saved,
                                  exact=total + ends + fr == saved,
                                  form=dict(first_prefix=const['first_prefix_rz'], later_prefix=const['later_prefix_rz'],
                                            suffix=const['suffix_rz'], end=const['end_rz']))
    # P4: paired composition ------------------------------------------------------------------
    P4 = dict(status='UNKNOWN', reason='INPUT_MISSING')
    if all(x in ok_paths for x in ('P2R66/p0', 'S66/p0', 'S66/p1')):
        child, a, b = arrays['P2R66/p0'], arrays['S66/p0'], arrays['S66/p1']
        seg1 = all(len(child[w]['segs'][0][0]) == len(a[w]['segs'][0][0]) and np.array_equal(child[w]['segs'][0][0], a[w]['segs'][0][0])
                   for w in range(n))
        body2 = [leveled_compare(child[w]['segs'][1], b[w]['segs'][0], P[w], F[w]) for w in range(n)]
        end_equal = all(np.array_equal(child[w]['end'][0], b[w]['end'][0]) for w in range(n))
        rz_child = data_metric(data, 'P2R66/p0', 'X', 'rz_count')
        rz_a, rz_b = data_metric(data, 'S66/p0', 'X', 'rz_count'), data_metric(data, 'S66/p1', 'X', 'rz_count')
        fr = {pid: results[pid]['axes']['X']['final_run']['rz'] for pid in ('P2R66/p0', 'S66/p0', 'S66/p1')}
        delta = rz_child - rz_a - rz_b
        explained = (const['later_prefix_rz'] - const['first_prefix_rz'] - const['end_rz']
                     + fr['P2R66/p0'] - fr['S66/p0'] - fr['S66/p1'])
        if not seg1 or any(x == 'DIFFER' for x in body2):
            status = 'FAIL'
        elif all(x == 'EXACT' for x in body2):
            status = 'PASS_EXACT'
        else:
            status = 'PASS_LEADING_1Q_RUN_ONLY'
        P4 = dict(status=status, first_segment_identical=seg1, second_body_vs_S66p1_per_wire=body2,
                  end_edge_identical=end_equal, rz_delta_child_minus_parents=delta,
                  rz_delta_explained_by_constants=explained,
                  rz_identity_exact=delta == explained if status == 'PASS_EXACT' else None)
    # P5: X/Y and final-run relation; margins for the 13 rounds -------------------------------
    p5_rows = []
    relation = {}
    for axis in ('X', 'Y'):
        pts = []
        for pid in ok_paths + [x for x in results if x not in ok_paths]:
            fr = results[pid]['axes'][axis].get('final_run')
            if not fr:
                continue
            pts.append((Fraction(float.fromhex(results[pid]['T_hex'])), fr, pid))
        forms = {tuple(fr['names']) for _, fr, _ in pts}
        ref = max(pts, key=lambda x: x[0])
        ok_form = forms == {('rz', 'sx', 'rz')}
        theta2 = {fr['angles_hex'][1] for _, fr, _ in pts} if ok_form else set()
        resid = []
        if ok_form:
            t1_ref = float.fromhex(ref[1]['angles_hex'][0])
            for T, fr, pid in pts:
                t1 = float.fromhex(fr['angles_hex'][0])
                resid.append(abs(wrap(t1 - t1_ref + scalar * float(T - ref[0]))))
        status = ('PASS' if ok_form and len(theta2) == 1 and max(resid) <= RELATION_TOLERANCE else 'UNKNOWN')
        relation[axis] = dict(status=status, forms=sorted('-'.join(f) for f in forms), theta2_unique=len(theta2) == 1,
                              max_residual=max(resid) if resid else None, reference=ref[2], reference_T_hex=float(ref[0]).hex(),
                              observed_T=sorted({float(T) for T, _, _ in pts}))
        for m, Tm in sorted(data['round_T'].items(), key=lambda x: int(x[0])):
            Tm = Fraction(Tm['numerator'], Tm['denominator'])
            if status == 'PASS':
                t1 = wrap(float.fromhex(ref[1]['angles_hex'][0]) - scalar * float(Tm - ref[0]))
                margin = abs(t1)
                cls = 'MARGIN_OK' if margin >= MARGIN_THRESHOLD else 'UNKNOWN_SMALL_MARGIN'
            else:
                t1, margin, cls = None, None, 'UNKNOWN_RELATION'
            p5_rows.append(dict(m=int(m), axis=axis, T=str(Tm), theta1=t1, margin=margin, final_run=cls))
    xy_flags = [results[pid]['xy']['compiled_equal_except_final_run'] for pid in results]
    xy_ok = (None if any(f is None for f in xy_flags) or not xy_flags else all(xy_flags))
    wrapper_ok = all(results[pid]['axes'][a]['wrapper_input']['p_equals_minus_scalar_T']
                     and results[pid]['axes'][a]['wrapper_input']['synthesized_matches_compiled']
                     for pid in results for a in ('X', 'Y'))
    sdg_ok = all(results[pid]['xy']['uncompiled_only_sdg_differs'] for pid in results)
    P5 = dict(status='PASS' if xy_ok is True and sdg_ok and wrapper_ok and all(v['status'] == 'PASS' for v in relation.values())
              and all(r['final_run'] == 'MARGIN_OK' for r in p5_rows) else 'FAIL_OR_UNKNOWN',
              compiled_X_equals_Y_except_final_run=xy_ok,
              uncompiled_X_Y_differ_only_by_sdg=sdg_ok,
              wrapper_input_checks=wrapper_ok, relation=relation,
              final_run_rz={pid: {a: results[pid]['axes'][a]['final_run']['rz'] for a in ('X', 'Y')} for pid in results})
    # step statistics (independent X-axis bodies; P2R66 excluded) -----------------------------
    stats = []
    groups = {}
    for row in seg_rows:
        if not row['independent_sample']:
            continue
        groups.setdefault((row['R'], row['context']), []).append(row['body_rz'])
    for (R, ctx), vals in sorted(groups.items()):
        nval = len(vals)
        mean = Fraction(sum(vals), nval)
        if nval >= 2:
            var = Fraction(sum((Fraction(v) - mean) ** 2 for v in vals), nval - 1)
            note = 'ONE_DEGREE_OF_FREEDOM' if nval == 2 else ''
        else:
            var, note = None, 'VARIANCE_NOT_IDENTIFIABLE_N_LT_2'
        stats.append(dict(R=R, context=ctx, n=nval, mean_body_rz=float(mean), mean_exact=str(mean),
                          sample_variance=float(var) if var is not None else None,
                          sample_variance_exact=str(var) if var is not None else None,
                          sd=math.sqrt(var) if var is not None else None,
                          variance_over_R=float(var / R) if var is not None else None,
                          min=min(vals), max=max(vals), note=note))
    # per-path table ------------------------------------------------------------------------
    for pid in sorted(results):
        r = results[pid]
        table.append(dict(path_id=pid, kind=r['kind'], m=r['m'], q=r['q'], R=r['R'],
                          P1='/'.join(r['axes'][a].get('P1', 'NA') for a in ('X', 'Y')),
                          P2='/'.join(str(r['axes'][a].get('P2', 'NA'))[:40] for a in ('X', 'Y')),
                          events_equal_R=all(r['axes'][a].get('events_equal_R') for a in ('X', 'Y')),
                          products=sum(r['axes'][a].get('products_total') or 0 for a in ('X', 'Y')),
                          pairs_checked=r['axes']['X'].get('pair_checks', {}).get('pairs'),
                          pairs_exact_hex=r['axes']['X'].get('pair_checks', {}).get('exact_hex'),
                          XY_equal=r['xy']['compiled_equal_except_final_run'],
                          final_run_X='-'.join(r['axes']['X']['final_run']['names']),
                          final_run_Y='-'.join(r['axes']['Y']['final_run']['names']),
                          decomposition_exact=decomposition.get(pid, {}).get('exact'),
                          completeness=all(r['axes'][a].get('completeness') for a in ('X', 'Y'))))
    for name, rows in (('AUDIT_TABLE.csv', table), ('SEGMENTS.csv', seg_rows), ('STEP_STATS.csv', stats),
                       ('FINAL_RUN.csv', p5_rows)):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(RUN / name, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (RUN / 'EDGES.json').write_text(json.dumps(edges, indent=1) + '\n')
    P1 = 'PASS' if all(r['axes'][a].get('P1') == 'PASS' for r in results.values() for a in ('X', 'Y')) and not stops else 'FAIL_OR_STOP'
    P2 = 'PASS' if len(ok_paths) == len(data['paths']) and all(t['events_equal_R'] for t in table) else 'FAIL'
    P6 = ('UNKNOWN_NO_SEGMENTED_PATH' if not decomposition else
          'PASS' if all(d['exact'] for d in decomposition.values()) and len(decomposition) == len(data['paths']) else 'FAIL')
    result = dict(schema='A_PAULI_LD2_STEP_AUDIT_RESULT_V1', P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
                  P6=dict(status=P6, decomposition=decomposition, segments=len(seg_rows)),
                  stops=stops, paths_passing_P2=len(ok_paths), paths=len(data['paths']),
                  products_total=sum(t['products'] for t in table),
                  identity_pair_rule=S0R1_IDENTITY_PAIRS,
                  compiles=0, sampling=0, builds=0, fits=0, price_predictions=0, private_writes=0)
    (RUN / 'AUDIT_RESULT.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(json.dumps(dict(P1=P1, P2=P2, P3=P3['status'], P4=P4['status'], P5=P5['status'], P6=P6,
                          paths_ok=len(ok_paths), stops=stops), default=str))


def data_metric(data, pid, axis, key):
    path = next(p for p in data['paths'] if p['path_id'] == pid)
    return path['axes'][axis]['metrics'][key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-inputs', action='store_true')
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.build_inputs:
        build_inputs()
    if args.measure:
        measure()
    if args.run:
        return run(args.workers)
    if args.summarize:
        summarize()
    return 0


if __name__ == '__main__':
    sys.exit(main())
