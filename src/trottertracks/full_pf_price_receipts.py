"""Receipt trace for saved Full PF T=0.8 compiled prices reused from pf_common. Metadata only.

Every long-time reused price is followed back to its origin without recomputing:
long-time carrier -> pf_common PF_endpoint_costs price of the same candidate ->
(a) the pf_common native_results record whose saved result.json SHA-256 equals
the recorded result_sha256, (b) the M5 pilot result.json named by the proof
(pilot group ID = digest of the candidate fields, recomputed with the original
pilot_group code), or (c) a candidate receipt. Candidate, H/state/snapshot
binding, compiler, wrapper with measurement, control, native policy, saved gate
checks and the saved RZ count are compared per axis (X=cosine, Y=sine).

No Hamiltonian/state load, circuit build/compile/transpile, RZ recount, fit or
G. Saved values are compared for identity only.
"""
import ast
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import types

from . import full_pf_saved_inventory as I

D = I.D
p = D.p
EXACT, CONDITIONAL, MISMATCH, UNAVAILABLE = I.EXACT, I.CONDITIONAL, I.MISMATCH, 'SOURCE_UNAVAILABLE'
PILOT = ('3eca46148070ba4d4b0f5815ef45943407cc959e', 'artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/')
PILOT_SOURCE = 'c7d53f04c3be131078ee91e8bef3c78ca434ed3e'
DOMAIN = 'src/trotterlib/pf_common_domain.py'
PREPARE_DOMAIN = 'scripts/prepare_pf_common_domain.py'
M5_NATIVE = 'scripts/h46_m5_native.py'
COST = 'scripts/pf_common_run_native.py'
PLAN = 'src/trotterlib/h46_m5_plan.py'
BINDING_FIELDS = ('T', 'basis', 'compiler', 'control', 'geometry', 'hamiltonian', 'profile', 'rank', 'rank_policy',
                  'series', 'snapshot', 'state', 'system')
CANDIDATE_KEYS = ('formula', 'method', 'prefix', 'q', 'system')
# Whitespace-free SOURCE fragments; the long-time compose shares the deterministic lowering.
DET_LOWERING = ("build_deterministic_native(blocks,num_system_qubits=n,T=s['signed_time'],q=1,formula='2nd',scalar=0.,"
                "control_policy='symmetric_directional',max_instructions=base+1)")
PF_COMMON_TEXT = ("build_native_hadamard_wrapper(evolution,axis=ax,include_measurement=True",
                  "CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')",
                  "old.compose(dict(row,policy=None,r_value=None),prep,(),scalar,native,QuantumCircuit,fp.fingerprint)",
                  "old.c.core.schedule(row['formula'],row['q'],'PER_PIECE',1)")
PILOT_TEXT = ("build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True",
              "CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)")


def _squash(text):
    return ''.join(text.split())


def canonical_sha(record):
    """SHA-256 of a saved result.json (sorted keys, indent 2, trailing newline)."""
    return hashlib.sha256((json.dumps(record, sort_keys=True, indent=2) + '\n').encode()).hexdigest()


@lru_cache(maxsize=None)
def _text(commit, path):
    return D._blob(commit, path).decode()


def _sha_blob(commit, path):
    return hashlib.sha256(D._blob(commit, path)).hexdigest()


def _function(commit, path, name):
    return next(n for n in ast.parse(_text(commit, path)).body if isinstance(n, ast.FunctionDef) and n.name == name)


@lru_cache(maxsize=1)
def pilot_group_function():
    """The original pilot_group (prepare_pf_common_domain) with the pinned digest/RANKS."""
    scope = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=[_function(D.BASE, DOMAIN, 'digest')], type_ignores=[]), 'pinned_domain', 'exec'), scope)
    domain = types.SimpleNamespace(RANKS=D._constant(D.BASE, DOMAIN, 'RANKS'), digest=scope['digest'])
    group = {'c': domain}
    exec(compile(ast.Module(body=[_function(D.BASE, PREPARE_DOMAIN, 'pilot_group')], type_ignores=[]), 'pinned_prepare', 'exec'), group)
    return group['pilot_group']


def load(k):
    """pf_common native results, endpoint prices, identities; pilot inventory. Pinned Git only."""
    out = dict(systems={}, native_policy=D._constant(D.BASE, DOMAIN, 'NATIVE'),
               pilot_inventory={v['path']: v['sha256'] for v in json.loads(D._blob(PILOT[0], PILOT[1] + 'inventory.json'))})
    for system, (commit, path) in I.PF_COMMON.items():
        native = json.loads(gzip.decompress(D._blob(commit, path + 'native_results.json.gz')))
        prices, conflicts = {}, []
        for ep in json.loads(gzip.decompress(D._blob(commit, path + 'PF_endpoint_costs.json.gz'))):
            if ep.get('price') is None:
                continue
            cid = p.candidate_id(dict(ep['candidate'], T=.8, R_outer=None, K=None))
            if cid in prices and not p.same(prices[cid], ep['price']):
                conflicts.append(cid)
            prices[cid] = ep['price']
        out['systems'][system] = dict(commit=commit, path=path, by_sha={canonical_sha(r): r for r in native},
            inventory=json.loads(D._blob(commit, path + 'native_result_inventory.json')), prices=prices, price_conflicts=conflicts,
            identity=json.loads(D._blob(commit, path + 'source_input_identity.json')),
            end_audit=json.loads(D._blob(commit, path + 'end_audit.json')))
    return out


def _binding_diff(binding, expected):
    return [f for f in BINDING_FIELDS if not p.same(binding.get(f), expected.get(f))]


def source_checks(ctx, k):
    """Native SOURCE text facts: wrapper+measurement, compiler, compose identity, formula SOURCE."""
    commits = {PILOT_SOURCE}
    for s in ctx['systems'].values():
        commits |= {s['identity']['SOURCE'], s['end_audit'].get('inherited_SOURCE')} - {None}
    compose = {c: ast.dump(_function(c, M5_NATIVE, 'compose')) for c in sorted(commits)}
    long_path = D.SOURCES['pf_native']['path']
    long_time = _squash(ast.get_source_segment(_text(D.BASE, long_path), _function(D.BASE, long_path, 'compose')))
    m5 = _squash(ast.get_source_segment(_text(PILOT_SOURCE, M5_NATIVE), _function(PILOT_SOURCE, M5_NATIVE, 'compose')))
    plan_sha = {c: _sha_blob(c, PLAN) for c in sorted(commits | {D.BASE})}
    items = dict(
        compose_identical_across_pilot_and_pf_common=len(set(compose.values())) == 1,
        deterministic_lowering_shared_with_long_time_compose=DET_LOWERING in long_time and DET_LOWERING in m5,
        long_time_schedule_and_scalar=("'PER_PIECE',row['R_outer']ifrequestselse1,T=row['T'])" in long_time
                                       and "qc.p(-scalar*row['T'],n)" in long_time),
        pf_common_schedule_and_scalar=("policy=group['policy']or'PER_PIECE';density=group['r_value']or1" in m5
                                       and 'qc.p(-scalar*.8,n)' in m5),
        pilot_cost_text=all(_squash(t) in _squash(_text(PILOT_SOURCE, M5_NATIVE)) for t in PILOT_TEXT),
        pf_common_cost_text={c: all(_squash(t) in _squash(_text(c, COST)) for t in PF_COMMON_TEXT) for c in sorted(commits - {PILOT_SOURCE})},
        formula_source_sha=plan_sha, formula_source_identical=len(set(plan_sha.values())) == 1)
    recorded = {}
    for system, s in ctx['systems'].items():
        files = {f['path']: f['sha256'] for f in s['identity']['source_files']}
        recorded[system] = dict(cost=files.get(COST) == _sha_blob(s['identity']['SOURCE'], COST),
                                plan=files.get(PLAN) == plan_sha[D.BASE], m5_native=files.get(M5_NATIVE) == _sha_blob(s['identity']['SOURCE'], M5_NATIVE),
                                binding=_binding_diff(s['identity']['binding'], I.expected_binding(k, system)))
    items['pf_common_recorded_SOURCE_sha'] = recorded
    ok = (items['compose_identical_across_pilot_and_pf_common'] and items['deterministic_lowering_shared_with_long_time_compose']
          and items['long_time_schedule_and_scalar'] and items['pf_common_schedule_and_scalar'] and items['pilot_cost_text']
          and all(items['pf_common_cost_text'].values()) and items['formula_source_identical']
          and all(v['cost'] and v['plan'] and v['m5_native'] and not v['binding'] for v in recorded.values()))
    return dict(status=EXACT if ok else MISMATCH, items=items, commits=sorted(commits),
                note='SOURCE text comparison only; no module is imported and no circuit is built')


def _axis(status, reasons, **kw):
    return dict(status=status, reasons=reasons, **kw)


def _native(c, reused, ctx, k):
    """(a) result_sha256 -> pf_common native_results record."""
    s = ctx['systems'][c['system']]; sha = reused['result_sha256']; record = s['by_sha'].get(sha)
    source = dict(commit=s['commit'], path=s['path'] + 'native_results.json.gz', result_sha256=sha)
    if record is None:
        return dict(kind='PF_COMMON_NATIVE_RESULT', source=source, axes={a: _axis(UNAVAILABLE, ['NATIVE_RESULT_SHA_NOT_FOUND']) for a in I.AXES})
    common = [] if canonical_sha(record) == sha else ['RESULT_SHA_DIFFERS']
    if record['candidate'] != {f: c[f] for f in CANDIDATE_KEYS} or record['candidate']['method'] != 'FULL':
        common.append('CANDIDATE_DIFFERS')
    if not any(v['result_sha256'] == sha and v['candidate'] == record['candidate'] for v in s['inventory']):
        common.append('NOT_IN_NATIVE_RESULT_INVENTORY')
    expected = I.expected_binding(k, c['system'])
    if record['hamiltonian'] != expected['hamiltonian'] or record['state'] != expected['state']:
        common.append('H_OR_STATE_DIFFERS')
    if record['status'] != 'PASS' or record['native_policy'] != ctx['native_policy']:
        common.append('NOT_ACCEPTED_NATIVE_POLICY')
    if record['SOURCE'] not in (s['identity']['SOURCE'], s['end_audit'].get('inherited_SOURCE')):
        common.append('UNRECORDED_NATIVE_SOURCE')
    if not all(v['PASS'] for v in record['checks']):
        common.append('SAVED_GATE_CHECK_FAILED')
    axes = {}
    for axis, label in I.AXES.items():
        price = record['prices'].get(label) or {}; reasons = list(common)
        names = {v['check'] for v in record['checks']}
        archives = {v['path']: v['restoration_PASS'] for v in record['archives']}
        if price.get('compiler') != p.PROFILE['compiler']:
            reasons.append('COMPILER_DIFFERS')
        if (price.get('metrics') or {}).get('rz_count') != reused[label]:
            reasons.append('RZ_COUNT_DIFFERS')
        if price.get('preparation_hash') != record['preparation_hash']:
            reasons.append('PREPARATION_HASH_DIFFERS')
        if not {'hadamard_' + label, 'compiled_vector_' + label} <= names:
            reasons.append('AXIS_GATE_CHECK_MISSING')
        if not (archives.get(label + '_wrapper.qpy.gz') and archives.get(label + '_compiled.qpy.gz')):
            reasons.append('AXIS_ARCHIVE_NOT_RECORDED')
        axes[axis] = _axis(MISMATCH if reasons else EXACT, reasons, rz_count=(price.get('metrics') or {}).get('rz_count'),
                           fingerprint=price.get('fingerprint'), compiler=price.get('compiler'))
    return dict(kind='PF_COMMON_NATIVE_RESULT', source=source, native_SOURCE=record['SOURCE'], preparation_hash=record['preparation_hash'],
                candidate=record['candidate'], checks={v['check']: v['error'] for v in record['checks']}, axes=axes)


def _pilot(c, reused, proof, ctx, k):
    """(b) pilot proof -> M5 pilot result.json named by the recomputed pilot group ID."""
    group = pilot_group_function()(c['system'], 'FULL', c['formula'])
    rel = c['system'] + '_TOL_PILOT_DET/' + group['id'] + '__0/result.json'
    source = dict(commit=PILOT[0], path=PILOT[1] + rel, pilot_group=group)
    common = []
    if proof.get('source') != 'pilot_' + group['id'] + '_result':
        common.append('PILOT_GROUP_ID_DIFFERS_FROM_CANDIDATE')
    if (group['q'], group['prefix'], group['method']) != (c['q'], c['prefix'], 'FULL'):
        common.append('PILOT_GROUP_CANDIDATE_DIFFERS')
    try:
        raw = D._blob(PILOT[0], PILOT[1] + rel)
    except Exception:
        return dict(kind='M5_PILOT_RESULT', source=source, axes={a: _axis(UNAVAILABLE, common + ['PILOT_RESULT_NOT_IN_GIT']) for a in I.AXES})
    result = json.loads(raw)
    source['sha256'] = hashlib.sha256(raw).hexdigest()
    if ctx['pilot_inventory'].get(rel) != source['sha256']:
        common.append('PILOT_INVENTORY_SHA_DIFFERS')
    if (result['task'] != group['id'] + '__0' or result['status'] != 'PASS' or result['native_policy'] != ctx['native_policy']
            or result['SOURCE'] != PILOT_SOURCE):
        common.append('PILOT_NOT_ACCEPTED')
    diff = _binding_diff(result['binding'], I.expected_binding(k, c['system']))
    if diff:
        common.append('BINDING_DIFFERS:' + ','.join(diff))
    records = {v['axis']: v for v in result['records']}
    proofs = {v['axis']: v for v in proof.get('axes') or []}
    checks = {v['check']: v['error'] for v in result['checks']}
    axes = {}
    for axis, label in I.AXES.items():
        r, q = records.get(label) or {}, proofs.get(label) or {}; reasons = list(common)
        if r.get('compiler_settings_hash') != p.PROFILE['compiler']:
            reasons.append('COMPILER_DIFFERS')
        if not r or r.get('fingerprint') != q.get('fingerprint') or r.get('metrics') != q.get('metrics'):
            reasons.append('PROOF_DIFFERS_FROM_PILOT_RECORD')
        if (r.get('metrics') or {}).get('rz_count') != reused[label]:
            reasons.append('RZ_COUNT_DIFFERS')
        if not {'hadamard_' + label, 'compiled_vector_' + label} <= set(checks):
            reasons.append('AXIS_GATE_CHECK_MISSING')
        axes[axis] = _axis(MISMATCH if reasons else EXACT, reasons, rz_count=(r.get('metrics') or {}).get('rz_count'),
                           fingerprint=r.get('fingerprint'), compiler=r.get('compiler_settings_hash'))
    return dict(kind='M5_PILOT_RESULT', source=source, native_SOURCE=result['SOURCE'], preparation_hash=result['preparation_hash'],
                checks=checks, axes=axes)


def _receipt(c, reused, proof):
    """(c) a candidate receipt carried in the proof (one axis)."""
    r = proof['receipt']; label = proof['axis']; reasons = []
    if (r.get('method') != 'FULL' or r.get('prefix') != c['prefix'] or r.get('formula') != c['formula']
            or r.get('q') != c['q'] or r.get('system') != c['system']):
        reasons.append('RECEIPT_CANDIDATE_DIFFERS')
    if r.get('compiler_settings_hash') != p.PROFILE['compiler']:
        reasons.append('COMPILER_DIFFERS')
    if r.get('rz_count') != reused[label]:
        reasons.append('RZ_COUNT_DIFFERS')
    axis = next(a for a, v in I.AXES.items() if v == label)
    return dict(kind='CANDIDATE_RECEIPT', source=dict(proof_source=proof.get('source'), acceptance_SOURCE=r.get('acceptance_SOURCE'),
                receipt_sha256=r.get('receipt_sha256'), result_sha256=r.get('result_sha256')),
                preparation_hash=r.get('preparation_hash'),
                axes={axis: _axis(MISMATCH if reasons else EXACT, reasons, rz_count=r.get('rz_count'), fingerprint=r.get('fingerprint'),
                                  compiler=r.get('compiler_settings_hash'))})


def trace(c, k, ctx, sources):
    """All receipts of one reused price; X/Y judged separately. None if no reused price."""
    reused = k['files']['reused_compiled_prices.json.gz'].get(c['id'])
    if reused is None:
        return None
    s = ctx['systems'][c['system']]
    origin = s['prices'].get(c['id'])
    chain = []
    if reused.get('reuse_provenance') != s['path'] + 'PF_endpoint_costs.json.gz':
        chain.append('REUSE_PROVENANCE_NOT_PF_COMMON_ENDPOINTS')
    if origin is None:
        chain.append('NO_PF_COMMON_ENDPOINT_PRICE')
    elif not p.same(origin, {f: v for f, v in reused.items() if f not in ('candidate', 'origin', 'reuse_provenance')}):
        chain.append('CARRIER_PRICE_DIFFERS_FROM_PF_COMMON')
    if c['id'] in s['price_conflicts']:
        chain.append('PF_COMMON_ENDPOINT_PRICES_DISAGREE')
    if not p.same(reused['candidate'], c):
        chain.append('CARRIER_CANDIDATE_DIFFERS')
    evidence = []
    if reused.get('result_sha256'):
        evidence.append(_native(c, reused, ctx, k))
    for proof in reused.get('proof') or []:
        if proof.get('kind') == 'DIRECT_PAIRED_NATIVE_PILOT':
            evidence.append(_pilot(c, reused, proof, ctx, k))
        elif 'receipt' in proof:
            evidence.append(_receipt(c, reused, proof))
    preps = {e.get('preparation_hash') for e in evidence if e.get('preparation_hash')}
    if len(preps) > 1:
        chain.append('PREPARATION_HASH_DIFFERS_BETWEEN_RECEIPTS')
    axes = {}
    for axis in I.AXES:
        items = [e['axes'][axis] for e in evidence if axis in e['axes']]
        fps = {v.get('fingerprint') for v in items if v.get('fingerprint')}
        reasons = list(chain) + (['FINGERPRINTS_DISAGREE'] if len(fps) > 1 else [])
        if sources['status'] != EXACT:
            reasons.append('NATIVE_SOURCE_TEXT_CHECK_FAILED')
        statuses = [v['status'] for v in items]
        if reasons or MISMATCH in statuses:
            status = MISMATCH
        elif EXACT in statuses:
            status = EXACT          # saved native/pilot result or candidate receipt matches every field
        elif UNAVAILABLE in statuses:
            status = UNAVAILABLE
        else:
            status = CONDITIONAL; reasons.append('NO_TRACEABLE_RECEIPT')
        axes[axis] = dict(status=status, reasons=reasons + [r for v in items for r in v['reasons']],
                          rz_count=reused[I.AXES[axis]], evidence_kinds=[e['kind'] for e in evidence if axis in e['axes']])
    return dict(candidate_id=c['id'], kind=reused['kind'], chain=dict(reasons=chain, pf_common=dict(commit=s['commit'],
                path=s['path'] + 'PF_endpoint_costs.json.gz')), evidence=evidence, axes=axes)


def private_preparation_check(k, ctx):
    """Optional read-only check: long-time FULL trial preparation hash equals pf_common's."""
    out = {}
    for system, s in ctx['systems'].items():
        hashes = {r['preparation_hash'] for r in s['by_sha'].values() if r['candidate']['method'] == 'FULL'}
        entry = next((v for v in k['files']['exploration_costs.json.gz'].values()
                      if v['candidate']['system'] == system and v['candidate']['method'] == 'FULL' and v.get('trial_files')), None)
        path = Path(entry['trial_files'][0]) if entry else None
        if path is None or not path.exists():
            out[system] = dict(status='PRIVATE_NOT_READ', pf_common_full_preparation_hashes=sorted(hashes)); continue
        trial = json.loads(path.read_bytes())
        out[system] = dict(status=EXACT if hashes == {trial['preparation_hash']} else MISMATCH, trial=str(path),
                           long_time_trial_preparation_hash=trial['preparation_hash'], pf_common_full_preparation_hashes=sorted(hashes))
    return out


def update(cov, k, ctx=None):
    """New coverage: reused-price receipts traced; only statuses those receipts change are updated."""
    ctx = ctx or load(k); sources = source_checks(ctx, k)
    new = deepcopy(cov); changes = []; traces = []
    for row in new['rows']:
        if row['group'] == 'PHYSICAL_MISMATCH':
            continue
        result = trace(row['physical_candidate'], k, ctx, sources)
        if result is None:
            continue
        traces.append(result)
        source = next(s for s in row['prices'] if s['kind'].startswith('DIRECT_COMPILED_DETERMINISTIC_REUSED_'))
        before = dict(X=row['X'], Y=row['Y'], group=row['group'])
        for axis in I.AXES:
            old = source['axes'][axis]; status = result['axes'][axis]['status']
            # CONDITIONAL entries take the traced result; EXACT ones change only on a contradiction.
            if old['status'] == CONDITIONAL or status in (MISMATCH, UNAVAILABLE):
                if status != old['status']:
                    old.update(previous_status=old['status'], previous_reasons=old['reasons'])
                old.update(status=status, reasons=result['axes'][axis]['reasons'])
        compilers = {v['compiler'] for e in result['evidence'] for v in e['axes'].values() if v.get('compiler')}
        source['compiler'] = next(iter(compilers)) if len(compilers) == 1 else None
        source['receipt_trace'] = dict(evidence=[dict(kind=e['kind'], source=e['source'], native_SOURCE=e.get('native_SOURCE'),
                                       preparation_hash=e.get('preparation_hash'), axes=e['axes']) for e in result['evidence']],
                                       chain=result['chain'])
        row['X'], row['Y'] = I._best(row['prices'], 'X'), I._best(row['prices'], 'Y')
        priced = row['X'] in (EXACT, CONDITIONAL) and row['Y'] in (EXACT, CONDITIONAL)
        has_signal = row['signal']['status'] in (EXACT, CONDITIONAL)
        row['group'] = ('SIGNAL_AND_XY_PRICE' if has_signal and priced else 'SIGNAL_ONLY_PRICE_MISSING' if has_signal
                        else 'PRICE_ONLY_SIGNAL_MISSING' if priced else 'ALL_MISSING')
        after = dict(X=row['X'], Y=row['Y'], group=row['group'])
        if after != before:
            changes.append(dict(physical_candidate_id=row['physical_candidate_id'], system=row['system'], formula=row['formula'],
                                T=row['T'], q_PF=row['q_PF'], before=before, after=after, reused_kind=result['kind'],
                                evidence_kinds=sorted({e['kind'] for e in result['evidence']})))
    def count(field):
        return dict(Counter(r[field] for r in new['rows']))
    new.update(status='FULL_PF_SAVED_EVIDENCE_COVERAGE_RECEIPTS_TRACED_METADATA_ONLY', previous_coverage_commit='92ff17dededea6e1ee273b2eee6a08e292ee9e5b',
               receipt_source_checks=sources, receipt_traces=traces, changes=changes,
               totals=dict(new['totals'], X=count('X'), Y=count('Y'), groups=count('group')),
               private_preparation_check=private_preparation_check(k, ctx), values_recomputed=False, new_compile=0, rz_recount=0)
    return new
