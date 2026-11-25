"""Per-method RPE cost optimizer, first version: saved evidence + a finite search rule.

Pipeline (one call of ``optimize``):

1. Problem input (JSON): Hamiltonian data locators, target, accuracy, methods with
   their candidate ranges, search budget, evidence locators.  All H4-specific
   paths, dimensions and labels live in that file, not here.
2. Saved evidence: C's catalog is rebuilt live from C's fixed INPUTS
   (``saved_rpe_cost_catalog.Catalog``), so every Q3 verdict, allocation and
   26-axis price locator is the one C already accepted.
3. Candidate space: PF formula x width from the PF candidate index within the
   requested range; PR ``L_D`` from the DF fragment count (read from the H data
   receipt) and the method definition.  Each candidate is attached to its own
   configuration, signal and 26-axis unit prices; missing items are reported.
4. Search (``FINITE_LOCAL_DESCENT_V1``): round-wise X/Y shots are the variables.
   From each configuration's cheapest feasible allocation, propose single-round
   decrements (step size gated by the incumbent's Q3 headroom); configurations
   without a feasible point get generator seeds at the method's best cost.
   Proposals strictly cheaper than the method incumbent are ordered by cost.
   No proposal is excluded because of another allocation's failure (no
   monotonicity assumption); only the identical configuration+allocation is
   reused.  Each proposal is answered by a Q3 backend: saved results, completed
   jobs of the approved round, or the executor that launches the method's
   existing Q3 route (``scripts/run_rpe_optimizer_q3_job.py``) within the
   approved budget.  Otherwise the proposal is emitted as a request and the
   method stops.
   ``search.incumbent_scope = "CONFIGURATION"`` applies the same rule to every
   configuration separately (its own incumbent; nothing is dropped because another
   configuration is cheaper), e.g. to compare kappa or L_D on equal terms.
   ``search.rule = "METHOD_MINIMUM_PRIORITY_V2"`` (``rpe_priority_search``) instead keeps one
   queue per method that lowers the method minimum first, gives every few picks to other / new
   configurations, defers allocations dominated by a clear failure, and plans the next kappa.
   ``executor.prior_run_dirs`` (round file) loads completed jobs of earlier rounds (same job key
   and route, identity-checked) so a stopped round's evaluations are reused without a new Q3.

No signal, price, compile or Q3 is computed here.  Costs are each candidate's own
saved unit prices times the integer shots, validated against C's saved totals.
"""
import ast
from collections import deque
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from .baseline_signal_pipeline import blob
from .saved_rpe_cost_catalog import Catalog, Reader, display, frac

PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
AXES = ('X', 'Y')
RULE = 'FINITE_LOCAL_DESCENT_V1'
UNIT_KEYS = ('C_ref_exact', 'C_ref', 'C', 'unit_price_RZ')


_ERROR_LOCK = threading.Lock()


def record_runtime_error(run, error, phase, **context):
    """Save the active exception before any thread/child drain can delay it."""
    row=dict(event='RUNTIME_EXCEPTION',controller_state='DRAINING_AFTER_EXCEPTION',phase=phase,error_type=type(error).__name__,
        message=str(error),traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)),
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),epoch=time.time(),monotonic=time.monotonic(),
        pid=os.getpid(),thread=threading.current_thread().name,**context)
    try:
        Path(run).mkdir(parents=True,exist_ok=True)
        with _ERROR_LOCK, (Path(run)/'RUNTIME_ERRORS.jsonl').open('a')as stream:
            stream.write(json.dumps(row,default=str)+'\n');stream.flush();os.fsync(stream.fileno())
    except OSError as logging_error:
        # Keep the original exception. A broken result disk is still fatal.
        print(row['traceback'],file=sys.stderr,flush=True)
        print('Runtime error record could not save: '+repr(logging_error),file=sys.stderr,flush=True)
    return row


def is_local_unknown(answer):
    return answer is not None and answer.get('status')=='UNKNOWN' and answer.get('failure_scope')=='CONFIGURATION'


def enc(value):
    value = Fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def show(value):
    return None if value is None else display(enc(value))


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interleave(nx, ny):
    return tuple(v for pair in zip(nx, ny) for v in pair)


def split(vector):
    return list(vector[0::2]), list(vector[1::2])


def cost(vector, prices):
    return sum(Fraction(n) * c for n, c in zip(vector, prices))


# ---------------------------------------------------------------- problem input
def hamiltonian_facts(problem):
    """Read the H data receipt; derive fragment count and dimensions from its array records."""
    data = problem['problem']['hamiltonian_data']
    receipt_path, snapshot_path = Path(data['receipt']['path']), Path(data['snapshot']['path'])
    if sha256(receipt_path) != data['receipt']['sha256']:
        raise ValueError('H data receipt differs from the problem input')
    receipt = load_json(receipt_path)
    if sha256(snapshot_path) != data['snapshot']['sha256'] or receipt['sha256'] != data['snapshot']['sha256']:
        raise ValueError('H data snapshot differs from the problem input/receipt')
    arrays = receipt['arrays']
    fragments, orbitals, _ = arrays['g_matrices']['shape']
    states = arrays['state_vector']['shape'][0]
    if problem['problem']['system'] == 'H6' and 'rank' not in receipt:
        metadata = receipt['metadata']
        identity = dict(rank=metadata['hamiltonian_metadata']['df_rank_actual'],
                        hamiltonian_hash=metadata['hamiltonian_hash'], state_hash=metadata['state_hash'])
    else:
        identity = {k: receipt[k] for k in ('rank', 'hamiltonian_hash', 'state_hash')}
    facts = dict(fragment_count=fragments, n_qubits=int(math.log2(states)),
                 orbital_dimension=orbitals, sector_dimension=arrays['sector_state_vector']['shape'][0],
                 snapshot_sha256=receipt['sha256'], **identity)
    expected = data.get('expected', {})
    for key, value in expected.items():
        if key in facts and facts[key] != value:
            raise ValueError('H data fact differs from the problem input: ' + key)
    if facts['fragment_count'] != facts['rank'] or 2 ** facts['n_qubits'] != states:
        raise ValueError('Inconsistent H data receipt')
    return facts


def tau(problem):
    return Fraction(problem['accuracy']['epsilon_E_decimal']) ** 2


def check_comparison(problem, facts, comparison):
    """Saved Q3 are reusable only under the same physical and evaluation conditions."""
    binding, accuracy = comparison['binding'], comparison['accuracy']
    expected = dict(system=problem['problem']['system'], snapshot=facts['snapshot_sha256'],
                    hamiltonian=facts['hamiltonian_hash'], state=facts['state_hash'])
    for key, value in expected.items():
        if binding[key] != value:
            raise ValueError('Saved catalog binding differs: ' + key)
    for key in ('estimator', 'target', 'loss', 'epsilon_E_decimal', 'scope'):
        if accuracy[key] != problem['accuracy'][key]:
            raise ValueError('Saved catalog accuracy differs: ' + key)
    if frac(accuracy['MSE_upper']) != tau(problem):
        raise ValueError('Saved catalog threshold differs from epsilon_E squared')
    if comparison['target']['energy_ha'] != problem['target']['energy_ha'] or comparison['M'] != problem['rpe']['M']:
        raise ValueError('Saved catalog target/RPE differs')


def load_catalog(root, problem):
    """Rebuild C's catalog from C's fixed INPUTS (reuse of C's reader, normalizer and checks)."""
    reader = Reader(Path(root))
    source = problem['evidence']['saved_catalog_inputs']
    from .pf_optimizer_connection import read_source
    inputs = read_source(root, reader, source)
    catalog = Catalog(reader, inputs['comparison'])
    for item in inputs['sources']:
        catalog.add(dict(item, commit=reader.commit(item['commit'])))
    return reader, inputs, catalog.result()


def pinned_function(reader, source):
    tree = ast.parse(blob(reader.root, reader.commit(source['commit']), source['path']))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == source['function']]
    if len(nodes) != 1:
        raise ValueError('Pinned helper unavailable')
    scope = dict(Fraction=Fraction)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source['commit'] + ':' + source['path'], 'exec'), scope)
    return scope[source['function']]


# ---------------------------------------------------------------- saved evidence
def point_vector(point):
    return interleave(point['N_X'], point['N_Y'])


def resolve_unit(reader, axis):
    for key in UNIT_KEYS:
        if axis.get(key) is not None:
            return frac(axis[key])
    loc = axis['price_source'];row = reader.read(loc)
    for key in UNIT_KEYS:
        if isinstance(row, dict) and row.get(key) is not None:
            return frac(row[key])
    raise ValueError('No unit price in source row')


def configuration_prices(reader, points, *, rounds=None):
    """Own axis prices of one configuration, checked against every saved total."""
    if rounds is None:
        lengths={len(point_vector(p))//len(AXES) for p in points if p.get('conditional_cost') is not None}
        if len(lengths)>1:
            return None,'INCONSISTENT_SAVED_HORIZON'
        rounds=next(iter(lengths),13)
    axis_count=rounds*len(AXES)
    priced = [p for p in points if len(p['price_axes']) == axis_count and p['conditional_cost'] is not None]
    if not priced:
        return None, 'NO_SAVED_%d_AXIS_PRICE'%axis_count
    units = {}
    for p in priced:
        for axis in p['price_axes']:
            unit = resolve_unit(reader, axis);key = (axis['m'], axis['axis'])
            if units.setdefault(key, unit) != unit:
                return None, 'CONFLICTING_SAVED_UNIT_PRICES'
    if sorted(units) != [(m, a) for m in range(rounds) for a in AXES]:
        return None, 'INCOMPLETE_%d_AXIS_PRICE'%axis_count
    prices = [units[(m, a)] for m in range(rounds) for a in AXES]
    for p in priced:
        if cost(point_vector(p), prices) != frac(p['conditional_cost']):
            return None, 'UNIT_PRICES_DO_NOT_REPRODUCE_SAVED_TOTAL'
    if any(c <= 0 for c in prices):
        return None, 'NONPOSITIVE_PRICE'
    return prices, 'SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS'


def record_from_point(point, prices):
    status = point['q3_status']
    lower = None if point['MSE_lower'] is None else frac(point['MSE_lower'])
    upper = None if point['MSE_upper'] is None else frac(point['MSE_upper'])
    return dict(status=status, MSE_lower=lower, MSE_upper=upper, source='SAVED', point_id=point['point_id'],
                label=point['label'], result_sources=point['result_sources'],
                cost=None if prices is None else cost(point_vector(point), prices))


class SavedBackend:
    """Answers a proposal only when C's catalog holds the same configuration, allocation and contract."""

    def __init__(self, points, prices_by_config):
        self.table = {}
        for p in points:
            if p['q3_status'] is None:
                continue
            self.table[(p['configuration_id'], point_vector(p))] = record_from_point(p, prices_by_config.get(p['configuration_id']))

    def answer(self, proposal, allow_new=False):
        hit = self.table.get((proposal['configuration_id'], proposal['vector']))
        return None if hit is None else dict(hit, calls=0)


class FixtureBackend:
    """Artificial Q3 for tests/demos only: callable(config_id, vector) -> (lower, upper)."""

    def __init__(self, function, threshold):
        self.function, self.threshold = function, threshold

    def answer(self, proposal, allow_new=True):
        if not allow_new:
            return None
        lower, upper = self.function(proposal['configuration_id'], proposal['vector'])
        status = PASS if upper <= self.threshold else FAIL if lower > self.threshold else 'INCONCLUSIVE'
        return dict(status=status, MSE_lower=Fraction(lower), MSE_upper=Fraction(upper), source='ARTIFICIAL_FIXTURE', calls=1)


class ChainBackend:
    def __init__(self, *backends):
        self.backends = backends

    def answer(self, proposal):
        for backend in self.backends:
            hit = backend.answer(proposal)
            if hit is not None:
                return hit
        return None

    def hold_configuration(self, proposal, answer):
        for backend in self.backends:
            if hasattr(backend, 'hold_configuration'):
                backend.hold_configuration(proposal, answer)


# ---------------------------------------------------------------- candidate space
def delta_ok(value, bounds):
    lo, hi = (Fraction(b) for b in bounds)
    return lo <= Fraction(value) <= hi


def enumerate_candidates(reader, problem, facts, catalog):
    points = [p for p in catalog['candidates']]
    by_config = {}
    for p in points:
        by_config.setdefault(p['configuration_id'], []).append(p)
    from .pf_optimizer_connection import indexed_prefix, read_source, scoped_index_row, width_inside
    index = read_source(Path(reader.root), reader, problem['evidence']['pf_candidate_index'])
    out = []
    for name, spec in problem['methods'].items():
        caps = spec['caps']
        allowed = spec.get('configuration_ids')  # optional allow-list, applied to every candidate source
        if spec['family'] == 'PF':
            rank = facts['fragment_count']
            def inside(formula, delta, p):
                return (formula in spec['formulas'] and delta is not None and width_inside(delta, spec)
                        and (spec.get('p_values') is None or p in spec['p_values']))
            indexed = [r for r in index['candidates'] if scoped_index_row(name, spec, r)]
            seen = {r['configuration_id'] for r in indexed}
            for r in indexed:
                prefix = rank if name == 'FULL_PF' else indexed_prefix(reader, r, rank)
                if inside(r['formula'], r['requested_delta_over_h'], prefix):
                    out.append(dict(method=name, family='PF', key=dict(p=prefix, formula=r['formula'], delta_over_h=r['requested_delta_over_h']),
                                    configuration_id=r['configuration_id'], index_signal=r['source_bound_13round_packet'], caps=caps))
            for cid, ps in by_config.items():
                if ps[0]['method'] == name and cid not in seen:
                    # Catalog-only configurations get the same formula/width range test (width = delta_pf / h).
                    delta = None if ps[0].get('delta_pf') is None else str(frac(ps[0]['delta_pf']) / frac(ps[0]['h']))
                    catalog_prefix = rank if name == 'FULL_PF' else ps[0].get('L_D')
                    if catalog_prefix is None:
                        continue  # Cannot place this saved row in a physical prefix scope
                    if inside(ps[0]['formula'], delta, catalog_prefix):
                        out.append(dict(method=name, family='PF', key=dict(p=catalog_prefix, formula=ps[0]['formula'], delta_over_h=delta),
                                        configuration_id=cid, index_signal=None, caps=caps, outside_index=True))
        else:
            lo = spec['L_D_range']['min'];hi = facts['fragment_count'] + spec['L_D_range']['max_offset_from_fragment_count']
            values = spec.get('L_D_values')
            for ld in range(lo, hi + 1):
                if values is not None and ld not in values:
                    continue
                configs = sorted({cid for cid, ps in by_config.items() if ps[0]['method'] == name and ps[0]['L_D'] == ld})
                widths = spec.get('requested_delta_pf_values')
                if widths is not None:
                    # Multi-width rounds scope saved candidates by the exact
                    # requested width, before generation/preparation.  Do not
                    # reinterpret a binary64 declaration as a decimal rational.
                    from .rpe_kappa_schedule import _frac as exact_width
                    allowed_widths = {exact_width(v) for v in widths}
                    configs = [cid for cid in configs if by_config[cid][0].get('delta_pf') is not None
                               and exact_width(by_config[cid][0]['delta_pf']) in allowed_widths]
                formulas = (spec.get('generation') or {}).get('formulas')
                if formulas is not None:
                    configs = [cid for cid in configs if by_config[cid][0].get('formula') in formulas]
                if not configs and not spec.get('saved_configurations_only'):
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=None, index_signal=None, caps=caps))
                for cid in configs:
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=cid, index_signal=None, caps=caps))
    out = [c for c in out if problem['methods'][c['method']].get('configuration_ids') is None
           or c['configuration_id'] in problem['methods'][c['method']]['configuration_ids']]
    for c in out:
        ps = by_config.get(c['configuration_id'], [])
        signals = [p['signal_source'] for p in ps if p.get('signal_source')] or ([c['index_signal']] if c['index_signal'] else [])
        rounds=len(c['caps']);axis_count=len(AXES)*rounds
        prices, price_status = configuration_prices(reader, ps, rounds=rounds) if ps else (None, 'NO_SAVED_%d_AXIS_PRICE'%axis_count)
        labels = sorted({p['label'] for p in ps})
        metadata=next((p for p in ps if len(p.get('price_axes',[]))==axis_count and p.get('conditional_cost')is not None),ps[0]if ps else {})
        c.update(labels=labels, signal_source=signals[0] if signals else None, prices=prices, price_status=price_status,
                 saved_points=[p['point_id'] for p in ps],
                 price_axes=deepcopy(metadata.get('price_axes',[])),
                 price_evidence=deepcopy(metadata.get('price_evidence')),
                 configuration_source=deepcopy(metadata.get('configuration_source')))
        if c['family'] == 'PF' and c.get('configuration_source') and c.get('signal_source'):
            # A previous search may have published a complete input binding in
            # the saved catalog without installing it in this run's registry.
            matches = [p for p in ps if p.get('configuration_source') == c['configuration_source']
                       and p.get('signal_source') == c['signal_source']]
            for field in ('price_registry_source', 'q3_template_source'):
                locators = [p[field] for p in matches if p.get(field)]
                if locators and all(loc == locators[0] for loc in locators):
                    c[field] = deepcopy(locators[0])
        needs = []
        if c['configuration_id'] is None:
            needs.append('CONFIGURATION_AND_SCHEDULE')
        if c['signal_source'] is None:
            needs.append(f'SIGNAL_{rounds}ROUND_PACKET')
        if prices is None:
            needs.append(f'PRICE_{axis_count}AXIS')
        c['needs'] = needs
        c['ready'] = not needs
        c['name'] = c['method'] + ':' + ','.join('%s=%s' % kv for kv in c['key'].items()) + ':' + (c['configuration_id'] or 'NONE')[:8]
    return out


# ---------------------------------------------------------------- search
def headroom(record, threshold):
    return None if record['MSE_upper'] is None else record['MSE_upper'] / threshold


def method_state(candidates, knowledge, threshold):
    """Cheapest feasible record per configuration and for the method (own-price cost)."""
    best = {};incumbent = None
    for c in candidates:
        if not c['ready']:
            continue
        rows = [(cost(v, c['prices']), v, r) for (cid, v), r in knowledge.items() if cid == c['configuration_id'] and r['status'] == PASS]
        if rows:
            t, v, r = min(rows, key=lambda x: (x[0], x[1]))
            best[c['configuration_id']] = dict(cost=t, vector=v, record=r, candidate=c)
            if incumbent is None or t < incumbent['cost']:
                incumbent = best[c['configuration_id']]
    return best, incumbent


def axis_groups(prices, m):
    if prices[2 * m] == prices[2 * m + 1]:
        return [('X', 'Y')]
    return [('X', 'Y'), ('X',), ('Y',)]


def proposals(candidates, knowledge, threshold, search, generate):
    best, incumbent = method_state(candidates, knowledge, threshold)
    limit = None if incumbent is None else incumbent['cost']
    out = []
    large = Fraction(search['large_step_headroom'])
    for c in candidates:
        if not c['ready'] or c.get('execution_needs'):
            continue
        cid, prices = c['configuration_id'], c['prices']
        caps = tuple(n for n in c['caps'] for _ in AXES)
        raw = []
        if cid in best:
            base = best[cid];h = headroom(base['record'], threshold)
            steps = search['decrement_steps'] if h is not None and h < large else [min(search['decrement_steps'])]
            for m in range(len(c['caps'])):
                for group in axis_groups(prices, m):
                    for d in steps:
                        v = list(base['vector'])
                        for a in group:
                            v[2 * m + AXES.index(a)] -= d
                        if min(v) >= 1:
                            raw.append((tuple(v), dict(kind='DECREMENT_FROM_CONFIGURATION_INCUMBENT', m=m, axes=list(group), d=d,
                                                       base_vector=list(base['vector']), base_headroom=None if h is None else '%.6f' % float(h))))
        else:
            budget = limit if limit is not None else cost(caps, prices)
            for rule in search['seed_rules']:
                g = generate(list(caps), list(prices), budget, rule)
                if g is not None:
                    raw.append((tuple(g['shots']), dict(kind='GENERATOR_SEED', rule=rule, budget=str(budget))))
            if limit is None:
                raw.append((caps, dict(kind='CAP_VECTOR_SEED')))
            for seed in c.get('sibling_seeds', []):  # saved feasible incumbents of same-method, same-L_D configurations
                raw.append((tuple(seed['vector']), dict(kind='SAVED_SAME_L_D_INCUMBENT_SEED', from_configuration=seed['configuration_id'],
                                                        from_label=seed.get('label'))))
        for v, origin in raw:
            if (cid, v) in knowledge:
                continue
            t = cost(v, prices)
            if limit is not None and t >= limit:
                continue
            out.append(dict(configuration_id=cid, candidate=c['name'], vector=v, cost=t, origin=origin, method=c['method'],
                            shot_change=sum(abs(a - b) for a, b in zip(v, origin.get('base_vector', v)))))
    unique = {}
    for item in out:
        unique.setdefault((item['configuration_id'], item['vector']), item)
    ordered = sorted(unique.values(), key=lambda x: (x['cost'], x['shot_change'], x['candidate'], x['vector']))
    return ordered, incumbent


HOLD = {'IN_PROGRESS': 'JOB_IN_PROGRESS', 'NO_ROUTE': 'NO_ROUTE_ADAPTER', 'STALE_JOB': 'STALE_JOB_NEEDS_REVIEW',
        'HOST_ELIGIBILITY_HOLD': 'HOST_ELIGIBILITY_HOLD'}
TASK_HOLD_STOPS = frozenset(HOLD.values())


def shared_stop(result):
    """The optimizer reports UNKNOWN_RESULT only if it cannot isolate the fault."""
    return any(row.get('stop_reason') == 'UNKNOWN_RESULT' for row in result.values())


def hold_candidate(candidate, proposal, answer, backend):
    """No reuse/retry of an unresolved job; isolate its canonical candidate.

    Saved feasible evidence remains usable. Numerical UNKNOWN is deliberately
    outside this path and keeps the existing fatal handling.
    """
    row=dict(scope='CONFIGURATION',candidate=candidate['name'],configuration_id=candidate['configuration_id'],
             method=candidate['method'],status=answer['status'],reason=answer.get('reason')or HOLD.get(answer['status'],'LOCAL_UNKNOWN_REQUIRES_REVIEW'),
             job=answer.get('job'),vector=list(proposal['vector']),
             new_scientific_calls=answer.get('scientific_evaluation_calls')if is_local_unknown(answer)else 0,
             launched_attempts=answer.get('calls',0))
    candidate['execution_hold']=row
    candidate['execution_needs']=list(dict.fromkeys(candidate.get('execution_needs',[])+['SAVED_JOB_REVIEW_REQUIRED']))
    if hasattr(backend,'hold_configuration'):backend.hold_configuration(proposal,answer)
    return row


def route_supports_configuration(rule, cid, reader):
    """A wildcard declaration cannot make an absent saved case executable.

    Only inspect published metadata; the caller's cached Reader avoids reading
    a template once per allocation. Numerical/intake guards still run in the worker.
    """
    allowed = rule.get('configuration_ids')
    if allowed not in (None, '*') and cid not in allowed:
        return False
    cid = (rule.get('horizon_projection') or {}).get('source_configuration_id', cid)
    if rule.get('kind') in ('pauli_b_common_entry', 'prdf_d2_common_entry'):
        entry = (rule.get('configurations') or {}).get(cid) or {}
        return all(entry.get(k) is not None for k in
                   ('configuration', 'signal_packet', 'binding_id', 'total_configuration_id')) and bool(rule.get('contract_template'))
    if rule.get('kind') in ('pf_b_pf6', 'pauli_b_ld3', 'pauli_b_ld5', 'prdf_d2_split'):
        for loc in rule.get('templates', []):
            source = dict(loc, selector=loc.get('inputs_selector', []))
            inputs = reader.read(source)
            if any((c.get('configuration_id') or c.get('original_configuration_id')) == cid for c in inputs['cases']):
                return True
        return False
    return True  # artificial/custom adapters retain their own contract


def run_method(name, candidates, knowledge, backend, search, threshold, generate, max_new, routes=None, spent=0, free=None,
               science_approved=True, prefetch=None):
    """Finite loop: select, ask backend, update knowledge, repeat until a stop rule fires.

    The budget is checked BEFORE the backend is called: a proposal is sent to the backend only if it can be
    answered without a new Q3 (``free``: same configuration+allocation already saved or already a job of this
    round) or if new-call budget remains.  Otherwise it is returned as the unexecuted next request.
    ``spent`` counts new calls already launched for this method in the same approved round (resume).
    UNKNOWN (gate/persistence) stops the method; a model failure (lower bound > tau) does not.
    ``prefetch`` (unlimited approved rounds only): called with the next proposals in the rule's own order so their
    jobs run concurrently; the rule still consumes answers one at a time in that order (same trajectory).
    """
    trace, held, new_calls, pending, stop = [], [], spent, None, None
    ready = [c for c in candidates if c['ready']];missing = [c['name'] for c in candidates if not c['ready'] or c.get('execution_needs')]
    limit = 0 if not science_approved else (float('inf') if max_new is None else max_new)
    large_headroom = Fraction(search['large_step_headroom'])
    from itertools import count
    max_steps=search['max_steps_per_method']
    work_slice=search.get('work_slice_steps')
    for step in count() if max_steps is None else range(max_steps):
        if work_slice is not None and step>=work_slice:
            stop='WORK_SLICE_YIELDED';break
        props, incumbent = proposals(ready, knowledge, threshold, search, generate)
        if not props:
            # Exhausted within the approved finite range; distinguish candidates that still lack inputs.
            stop = 'NO_IMPROVING_PROPOSAL_INPUTS_MISSING' if missing else 'NO_IMPROVING_PROPOSAL'
            if ready and incumbent is None:
                stop = 'NO_FEASIBLE_POINT_IN_RANGE'
            if pending and pending.get('hold_reason'):
                stop = 'HOST_ELIGIBILITY_HOLD'
            break
        choice = props[0];allow = new_calls < limit
        if prefetch is not None and limit == float('inf') and science_approved:
            # Look ahead only where failures are likely (seeds, or an incumbent within the large-step headroom); far from
            # the threshold the first proposal usually succeeds and lookahead jobs would be discarded.
            h = choice['origin'].get('base_headroom')
            near = choice['origin'].get('kind') != 'DECREMENT_FROM_CONFIGURATION_INCUMBENT' or h is None or Fraction(h) >= large_headroom
            prefetch(props[:search.get('lookahead', 1) if near else 1])
        is_free = free is not None and free(choice)
        answer = backend.answer(choice) if backend is not None and (is_free or allow) else None
        if is_local_unknown(answer):
            candidate=next(c for c in ready if c['configuration_id']==choice['configuration_id'])
            new_calls+=answer.get('calls',0)
            held.append(hold_candidate(candidate,choice,answer,backend))
            continue
        hold = None if answer is None else HOLD.get(answer['status'])
        if hold:
            candidate=next(c for c in ready if c['configuration_id']==choice['configuration_id'])
            held.append(hold_candidate(candidate,choice,answer,backend))
            if pending is None and answer['status'] == 'HOST_ELIGIBILITY_HOLD':
                pending = dict(choice, route=(routes or {}).get(name), job=answer.get('job'),
                               host_eligibility=answer.get('host_eligibility'), hold_reason=answer.get('reason'))
            continue
        if answer is None or hold:
            pending = dict(choice, route=(routes or {}).get(name), queue=[dict(candidate=p['candidate'], cost=p['cost'],
                vector=list(p['vector']), origin=p['origin']) for p in props[:5]], job=None if answer is None else answer.get('job'))
            if hold:
                stop = hold
                if answer['status'] == 'HOST_ELIGIBILITY_HOLD':
                    pending['host_eligibility'] = answer.get('host_eligibility')
                    pending['hold_reason'] = answer.get('reason')
            elif not science_approved and not is_free:
                stop = 'NEW_SCIENCE_NOT_APPROVED'
            elif limit == 0 and not is_free:
                stop = 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'
            elif not allow and not is_free:
                stop = 'Q3_BUDGET_EXHAUSTED'
            else:
                stop = 'NEW_Q3_REQUIRED_NO_EXECUTOR'
            break
        new_calls += answer['calls']
        key = (choice['configuration_id'], choice['vector'])
        knowledge[key] = dict(answer, cost=choice['cost'])
        _, after = method_state(ready, knowledge, threshold)
        nx, ny = split(choice['vector'])
        trace.append(dict(step=step + 1, candidate=choice['candidate'], configuration_id=choice['configuration_id'],
                          vector=list(choice['vector']), N_X=nx, N_Y=ny, cost=choice['cost'],
                          origin=choice['origin'], answer_source=answer['source'], status=answer['status'],
                          MSE_lower=answer.get('MSE_lower'), MSE_upper=answer['MSE_upper'], calls=answer['calls'], job=answer.get('job'),
                          result_sources=answer.get('result_sources'), point_id=answer.get('point_id'),
                          reason=answer.get('reason'),
                          incumbent_before=None if incumbent is None else incumbent['cost'],
                          incumbent_after=None if after is None else after['cost'],
                          improved=after is not None and (incumbent is None or after['cost'] < incumbent['cost']),
                          proposals_considered=len(props)))
        if answer['status'] not in (PASS, FAIL):
            held.append(dict(candidate=choice['candidate'], vector=list(choice['vector']), status=answer['status'],
                             reason=answer.get('reason')))
        if answer['status'] == 'UNKNOWN':
            stop = 'UNKNOWN_RESULT';break
    else:
        stop = 'STEP_LIMIT'
    _, incumbent = method_state(ready, knowledge, threshold)
    return dict(method=name, stop_reason=stop, trace=trace, held=held, new_calls=new_calls - spent, spent_before=spent,
                pending_request=pending, incumbent=incumbent, not_ready_in_range=missing)


# ---------------------------------------------------------------- executor (approved rounds only)
def contract_digest(problem):
    return hashlib.sha256(json.dumps([problem['accuracy'], problem['target']['energy_ha'], problem['rpe']], sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=32)
def _ten_to(digits):
    return 10 ** digits


def _outcome_int(text):
    """Read a saved decimal integer exactly, with smaller conversions."""
    if len(text) <= 3072:
        return int(text)
    if text.startswith('-'):
        return -_outcome_int(text[1:])
    middle = len(text) // 2
    return _outcome_int(text[:middle]) * _ten_to(len(text) - middle) + _outcome_int(text[middle:])


_PREDECODED_OUTCOME = threading.local()


def _outcome_metadata_int(text):
    return int(text) if len(text) <= 3072 else None


def job_outcome(job, *, metadata_only=False):
    """Accept separately collected evidence without overwriting the old STOP."""
    prepared = getattr(_PREDECODED_OUTCOME, 'value', None)
    if prepared is not None and prepared[0] == job:
        return prepared[1]
    original=Path(job)/'OUTCOME.json';path=Path(job)/'COLLECTED_OUTCOME.json'
    parse_int = _outcome_metadata_int if metadata_only else _outcome_int
    old=json.loads(original.read_text(), parse_int=parse_int)
    if path.exists() and old.get('status')=='UNKNOWN':
        new=json.loads(path.read_text(), parse_int=parse_int)
        gatefile=Path(job)/'COLLECT_SUPERVISION.json'
        gate=json.loads(gatefile.read_text()).get('process_exit_gate',{}) if gatefile.exists()else{}
        if (new.get('mode')=='collect' and new.get('collection_source') and
            new.get('original_outcome_sha256')==hashlib.sha256(original.read_bytes()).hexdigest() and
            gate.get('all_exit_codes_zero') is True):return new,path
    return old,original


class Executor:
    """Launch the method's existing Q3 route for one request at a time; resume from the round directory.

    Job directory jobs/<method>/<job_id>/ holds REQUEST.json, LAUNCH.json (pid), PROCESS_RETURN.json and the
    adapter's OUTCOME.json plus all worker artifacts.  A job is launched only when it has no OUTCOME, no live
    process and no previous launch; a dead launch without OUTCOME is first re-validated in collect mode (no
    science) and otherwise reported for review (never retried automatically).

    ``max_concurrent`` > 1 lets several jobs run at once (lookahead / per-configuration searches); each launch
    takes a slot until its process exits.  ``wait_for_running``: a live job of an earlier invocation is waited
    for (polled) instead of being reported as IN_PROGRESS.
    """

    def __init__(self, root, run_dir, routes, expected, problem, adapter='scripts/run_rpe_optimizer_q3_job.py', optimizer_source=None,
                 max_concurrent=1, wait_for_running=False, poll_seconds=10, prior_run_dirs=(), launch_disabled=False):
        self.root, self.run_dir, self.routes, self.expected = Path(root), Path(run_dir), routes, expected
        self.adapter, self.optimizer_source, self.contract = Path(root) / adapter, optimizer_source, contract_digest(problem)
        self.method = None
        self.exploratory_cost = problem.get('search', {}).get('exploratory_cost') is True
        self.lock, self.slots = threading.Lock(), threading.BoundedSemaphore(max(1, max_concurrent))
        self.events, self.waiters, self.uncounted = {}, [], set()
        self.wait_for_running, self.poll_seconds = wait_for_running, poll_seconds
        # Completed jobs of earlier rounds (same job key, same route for the configuration) answer without a new Q3.
        self.prior_run_dirs, self.launch_disabled = [Path(d) for d in prior_run_dirs], launch_disabled
        self.pf_template_reader, self.pf_template_cache = Reader(self.root), {}
        # READY refreshes consult the same prior jobs with a different candidate subset.
        # Only cache the identity used to skip unrelated requests; validate matching jobs afresh.
        self.prior_request_configurations = {}
        previous=self.run_dir/'ROUND_RESULT.json'
        saved=load_json(previous).get('methods',{})if previous.exists()else {}
        self.execution_holds={m:{r['configuration_id']:r for r in v.get('execution_holds',[])}for m,v in saved.items()}

    def log(self, **event):
        with self.lock:
            with (self.run_dir / 'PROGRESS.jsonl').open('a') as f:
                f.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), method=self.method, **event),
                                   default=str) + '\n')

    def for_method(self, method):
        view = Executor.__new__(Executor)
        view.__dict__.update(self.__dict__)  # shares lock, slots, events and counters; only the method differs
        view.method = method
        return view

    def hold_configuration(self,proposal,answer):
        row=dict(scope='CONFIGURATION',configuration_id=proposal['configuration_id'],method=self.method,
                 status=answer['status'],reason=answer.get('reason')or HOLD.get(answer['status'],'LOCAL_UNKNOWN_REQUIRES_REVIEW'),job=answer.get('job'))
        with self.lock:self.execution_holds.setdefault(self.method,{})[proposal['configuration_id']]=row
        self.log(event='CANDIDATE_HELD',**{k:v for k,v in row.items()if k!='method'})

    def refresh_execution_hold(self,cid):
        """Release only evidence the existing reuse/identity guards now accept."""
        row=self.execution_holds.get(self.method,{}).get(cid)
        if row is None:return None
        release=(row['status'] in ('NO_ROUTE', 'HOST_ELIGIBILITY_HOLD') and not row.get('job')
                 and self.route(cid) is not None)
        if not release and row.get('job'):
            job=Path(row['job']);job=job if job.is_absolute()else self.root/job
            try:
                request=load_json(job/'REQUEST.json');done=load_json(job/'PROCESS_RETURN.json')
                proposal=dict(configuration_id=cid,vector=interleave(request['N_X'],request['N_Y']))
                if(done['returncode']==0 and not done.get('signal')and self.job_dir(proposal).name==job.name
                        and self.same_route(job,cid)and(job/'OUTCOME.json').exists()):
                    release=self.ingest(job,proposal,calls=0)['status']in(PASS,FAIL,'INCONCLUSIVE')
            except(OSError,ValueError,KeyError,TypeError):pass
        if release:
            with self.lock:self.execution_holds[self.method].pop(cid,None)
            self.log(event='CANDIDATE_HOLD_RESOLVED',configuration=cid,source='EXISTING_REUSE_GUARDS',new_calls=0)
            return None
        return row

    def route(self, cid):
        for rule in self.routes.get(self.method, []):
            try:
                if route_supports_configuration(rule, cid, self.pf_template_reader):
                    return rule
            except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError):
                continue  # unresolved source is an input need, never a wildcard fallback
        return None

    def job_dir(self, proposal):
        rule = self.route(proposal['configuration_id']) or {}
        key = json.dumps([self.method, proposal['configuration_id'], list(proposal['vector']), rule.get('kind'), self.contract])
        return self.run_dir / 'jobs' / self.method / hashlib.sha256(key.encode()).hexdigest()[:20]

    def launched(self, method):
        base = self.run_dir / 'jobs' / method
        return sum(1 for d in base.iterdir() if (d / 'LAUNCH.json').exists()) if base.exists() else 0

    def alive(self, launch, job):
        pid = launch['pid']
        try:
            os.kill(pid, 0)
            cmd = Path('/proc/%d/cmdline' % pid).read_bytes().decode(errors='replace')
        except (OSError, ProcessLookupError):
            return False
        return str(job) in cmd

    def expected_for(self, cid, rule):
        entry = ((rule or {}).get('configurations') or {}).get(cid) or {}
        base = dict(self.expected.get(cid) or {k: entry.get(k) for k in ('total_configuration_id', 'binding_id')})
        return dict(base, configuration_id=cid)

    def request(self, proposal, job, rule):
        nx, ny = split(proposal['vector'])
        return dict(request_id=job.name, method=self.method, configuration_id=proposal['configuration_id'], N_X=nx, N_Y=ny,
                    cost=None if proposal['cost'] is None else enc(proposal['cost']), origin=proposal['origin'], route=rule,
                    expected=self.expected_for(proposal['configuration_id'], rule), optimizer_source=self.optimizer_source)

    def run_adapter(self, job, rule, mode):
        cmd = [rule['interpreter'], '-B', str(self.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (job / (mode.upper() + '.log')).open('a') as log:  # the child keeps its own descriptor
            proc = subprocess.Popen(cmd, cwd=str(self.root), env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc

    def pf_template_source(self, rule, cid):
        """Effective first PF template SOURCE; unrelated additions are harmless."""
        cid = (rule.get('horizon_projection') or {}).get('source_configuration_id', cid)
        templates=rule['templates'];key=(cid,json.dumps(templates,sort_keys=True))
        if key in self.pf_template_cache:
            return self.pf_template_cache[key]
        for loc in templates:
            source=dict(commit=self.pf_template_reader.commit(loc['commit']),path=loc['path'],
                        selector=list(loc.get('inputs_selector',[])))
            inputs=self.pf_template_reader.read(source)
            if any((case.get('configuration_id')or case.get('original_configuration_id'))==cid for case in inputs['cases']):
                self.pf_template_cache[key]=source
                return source
        raise ValueError('No effective PF template for prior candidate')

    def prior_route_fields(self, rule, cid):
        """Compare only the route fields and immutable inputs used by this job."""
        omitted = {'configurations', 'configuration_ids', 'existing_caller', 'configurations_file'}
        if rule.get('kind') == 'pf_b_pf6':
            omitted.add('templates')
        source_cid = (rule.get('horizon_projection') or {}).get('source_configuration_id', cid)
        return dict(route={k: v for k, v in rule.items() if k not in omitted},
                    entry=(rule.get('configurations') or {}).get(source_cid))

    def same_route(self, job, cid, *, request=None):
        """A prior job is reusable only if it ran the same route (kind, worker view, contract, signal packet, binding)."""
        try:
            old = json.loads((job / 'REQUEST.json').read_text()) if request is None else request
            now = self.route(cid) or {}
            before = old.get('route') or {}
            return (old.get('configuration_id') == cid and before.get('kind') == now.get('kind')
                    and old.get('expected') == json.loads(json.dumps(self.expected_for(cid, now), default=str))
                    and self.prior_route_fields(before, cid) == self.prior_route_fields(now, cid)
                    and (now.get('kind') != 'pf_b_pf6'
                         or self.pf_template_source(before, cid) == self.pf_template_source(now, cid)))
        except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError):
            return False

    def prior_job(self, proposal, *, exclude=()):
        name = self.job_dir(proposal).name
        # Prefer an exact saved outcome across all prior directories. Retain
        # unmatched and unfinished ownership when no reusable result exists.
        fallback = None
        for marker in ('OUTCOME.json', 'LAUNCH.json'):
            for base in self.prior_run_dirs:
                job = base / 'jobs' / self.method / name
                if not (job / marker).exists() or (exclude and job.resolve() in exclude):
                    continue
                if marker == 'OUTCOME.json' and self.same_route(job, proposal['configuration_id']):
                    return job
                if fallback is None:
                    fallback = job
        return fallback

    def prior_same_route(self, job, cid, *, request=None):
        return self.same_route(job, cid, request=request)

    def ingest_prior(self, job, proposal):
        return self.ingest(job, proposal, calls=0)

    def has_job(self, proposal):
        job = self.job_dir(proposal)
        return (job / 'OUTCOME.json').exists() or (job / 'LAUNCH.json').exists() or self.prior_job(proposal) is not None

    def prior_records(self, candidates):
        """Every completed prior-round job of this method for a ready candidate, validated like a new job (same key,
        same route, identity checks).  Returns ({(cid, vector): record}, refused rows)."""
        cache = getattr(self, 'preparation_history_cache', None)
        if cache is not None:
            return cache.prior_records(self, candidates)[:2]
        cache_dir = getattr(self, 'optimizer_history_directory', None)
        if cache_dir is not None:
            from .rpe_plan_history_cache import HistoryCache
            import sqlite3
            try:
                with HistoryCache(cache_dir / (self.method + '.sqlite3')) as history:
                    known, refused, stats = history.prior_records(self, candidates, details=False)
                self.log(event='PRIOR_HISTORY_CACHE', hits=stats['cache_hits'],
                         scanned=stats['scanned_jobs'], validated=stats['validated_jobs'],
                         process_decodes=stats['process_decodes'],
                         unrelated_request_hits=stats['unrelated_request_hits'],
                         context_changes=stats['context_changes'],
                         seconds=stats['wall_seconds'])
                return known, refused
            except (OSError, sqlite3.Error) as exc:
                self.log(event='PRIOR_HISTORY_CACHE_UNAVAILABLE', reason=str(exc))
        ready = {c['configuration_id']: c for c in candidates if c['ready'] and c['configuration_id']}
        declared = {c['configuration_id'] for c in candidates if c.get('configuration_id')}
        known, refused = {}, []
        held_jobs = {(Path(item['parent_directory'])/'jobs'/item['method']/item['job_id']).resolve()
                     for item in getattr(self,'prior_q3_holds', ())}
        for base in self.prior_run_dirs:
            for job in sorted((base / 'jobs' / self.method).glob('*/OUTCOME.json')):
                job = job.parent
                if held_jobs and job.resolve() in held_jobs:
                    refused.append(dict(job=str(job), reason='PRIOR_Q3_PERSISTENCE_HOLD'))
                    continue
                request = job / 'REQUEST.json'
                try:
                    stat = request.stat()
                    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    cached = self.prior_request_configurations.get(request)
                    if cached is not None and cached[0] == signature and cached[1] not in ready:
                        refused.append(dict(job=str(job), reason='CONFIGURATION_PREPARATION_PENDING'
                                            if cached[1] in declared else 'CONFIGURATION_NOT_IN_THIS_ROUND'))
                        continue
                    q = json.loads(request.read_text())
                except (OSError, ValueError):
                    refused.append(dict(job=str(job), reason='NO_REQUEST'));continue
                cid = q.get('configuration_id')
                self.prior_request_configurations[request] = (signature, cid)
                c = ready.get(cid)
                if c is None:
                    refused.append(dict(job=str(job), reason='CONFIGURATION_PREPARATION_PENDING'
                                        if cid in declared else 'CONFIGURATION_NOT_IN_THIS_ROUND'));continue
                proposal = dict(configuration_id=c['configuration_id'], vector=interleave(q['N_X'], q['N_Y']))
                if self.job_dir(proposal).name != job.name or not self.prior_same_route(job, c['configuration_id'], request=q):
                    refused.append(dict(job=str(job), reason='DIFFERENT_JOB_KEY_OR_ROUTE'));continue
                r = self.ingest_prior(job, proposal)
                if r['status'] not in (PASS, FAIL, 'INCONCLUSIVE'):
                    refused.append(dict(job=str(job), reason=r['status'] + ':' + str(r.get('reason'))));continue
                known[(c['configuration_id'], proposal['vector'])] = dict(r, source='PRIOR_ROUND_JOB',
                    cost=None if c['prices'] is None else cost(proposal['vector'], c['prices']))
        return known, refused

    def launch(self, proposal, rule):
        """Start one job without waiting; a waiter thread records the return and frees the slot."""
        prior = self.prior_job(proposal)
        if prior is not None:
            return prior
        job = self.job_dir(proposal);done = threading.Event()
        if self.launch_disabled:
            raise RuntimeError('launch disabled (saved-job replay): ' + job.name)
        with self.lock:
            if (job / 'LAUNCH.json').exists() or (job / 'OUTCOME.json').exists() or job in self.events:
                return job
            job.mkdir(parents=True, exist_ok=True)  # a REQUEST-only directory (stopped while waiting for a slot) is reused
            self.events[job] = done
        (job / 'REQUEST.json').write_text(json.dumps(self.request(proposal, job, rule), ensure_ascii=False, indent=2, default=str) + '\n')
        self.slots.acquire()
        start = time.time();proc = self.run_adapter(job, rule, 'execute')
        (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=proc.pid, started=start, kind=rule['kind'])) + '\n')
        with self.lock:
            self.uncounted.add(job)
        self.log(event='LAUNCH', job=job.name, configuration=proposal['configuration_id'][:8], N=split(proposal['vector'])[0],
                 cost=show(proposal['cost']), origin=proposal['origin'].get('kind'))

        def wait():
            try:
                code = proc.wait()
                (job / 'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=code if code >= 0 else None,
                    signal=-code if code < 0 else None, wall_seconds=time.time() - start, automatic_retry=False)) + '\n')
                status = json.loads((job / 'OUTCOME.json').read_text())['status'] if (job / 'OUTCOME.json').exists() else None
                self.log(event='EXIT', job=job.name, returncode=code, wall_seconds=round(time.time() - start, 1), status=status)
            finally:
                self.slots.release();done.set()
        t = threading.Thread(target=wait, daemon=False);t.start()
        with self.lock:
            self.waiters.append(t)
        return job

    def prefetch(self, proposals):
        """Launch jobs for proposals that have no saved answer and no job yet (unlimited approved rounds only)."""
        for proposal in proposals:
            rule = self.route(proposal['configuration_id'])
            if rule is not None and not self.has_job(proposal):
                self.launch(proposal, rule)

    def finish(self):
        """Wait for every job launched by this process (lookahead jobs the rule did not need included)."""
        for t in list(self.waiters):
            t.join()

    def take_count(self, job):
        with self.lock:
            if job in self.uncounted:
                self.uncounted.discard(job)
                return 1
        return 0

    def answer(self, proposal, allow_new=True):
        rule = self.route(proposal['configuration_id']);job = self.job_dir(proposal)
        if job in self.events:  # launched by this process (directly or as lookahead)
            self.events[job].wait()
        if (job / 'OUTCOME.json').exists():
            return self.ingest(job, proposal, calls=self.take_count(job))
        if not (job / 'LAUNCH.json').exists():
            prior = self.prior_job(proposal)
            if prior is not None:
                if not self.prior_same_route(prior,proposal['configuration_id']):
                    return dict(status='STALE_JOB',job=str(prior),calls=0,source='PRIOR_ROUND_JOB',owner_review_required=True,
                                reason='PRIOR_OWNER_ROUTE_OR_SOURCE_UNRESOLVED_REVIEW_REQUIRED')
                if (prior / 'OUTCOME.json').exists():
                    return dict(self.ingest_prior(prior, proposal), source='PRIOR_ROUND_JOB')
                try:
                    launch = json.loads((prior / 'LAUNCH.json').read_text())
                    running = self.alive(launch, prior)
                except (OSError, ValueError, KeyError, TypeError):
                    running = False
                return dict(status='IN_PROGRESS' if running else 'STALE_JOB', job=str(prior), calls=0,
                            source='PRIOR_ROUND_JOB', owner_review_required=True,
                            reason='PRIOR_OWNER_JOB_RUNNING' if running else 'PRIOR_OWNER_JOB_INCOMPLETE_REVIEW_REQUIRED')
        if (job / 'LAUNCH.json').exists():
            launch = json.loads((job / 'LAUNCH.json').read_text())
            if job in self.events:
                return self.scope_unknown(job,proposal,dict(status='UNKNOWN', reason='ADAPTER_EXIT_WITHOUT_OUTCOME', MSE_lower=None, MSE_upper=None,
                            source='NEW_Q3_JOB', calls=self.take_count(job), job=str(job)))
            while not (job / 'PROCESS_RETURN.json').exists() and self.alive(launch, job):
                if not self.wait_for_running:
                    return dict(status='IN_PROGRESS', job=str(job))
                time.sleep(self.poll_seconds)
            if (job / 'OUTCOME.json').exists():
                return self.ingest(job, proposal, calls=0)
            if rule is not None:
                self.run_adapter(job, rule, 'collect').wait()
            if (job / 'OUTCOME.json').exists():
                return self.ingest(job, proposal, calls=0)
            return dict(status='STALE_JOB', job=str(job))
        if not allow_new:
            return None
        if rule is None:
            return dict(status='NO_ROUTE', job=None)
        job = self.launch(proposal, rule)
        return self.answer(proposal, allow_new=False)

    def scope_unknown(self,job,proposal,answer):
        """Only evidenced local failures are isolated; no UNKNOWN is accepted.

        Scope preflight (zero numerical calls) and explicit SIGINT/SIGTERM exit
        are local. SIGKILL/OOM, numerical/identity/SOURCE failures, persistence
        faults and unclassified errors keep global protection. No retry occurs.
        """
        if answer is None or answer.get('status')!='UNKNOWN':return answer
        answer=dict(answer,failure_scope='GLOBAL')
        if getattr(getattr(self,'state',None),'failure',None):return answer
        reason=answer.get('reason');job=Path(job)
        pre_api = ('COUNT_ONLY_SCOPE_EXCEEDED', 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN')
        if reason not in (*pre_api, 'ADAPTER_EXIT_WITHOUT_OUTCOME'):return answer
        try:
            request=load_json(job/'REQUEST.json');nx,ny=split(proposal['vector'])
            if(request['configuration_id']!=proposal['configuration_id']or request['N_X']!=nx or request['N_Y']!=ny
                or request['request_id']!=self.job_dir(proposal).name or not self.same_route(job,proposal['configuration_id'])):
                return answer
            if reason in pre_api and type(answer.get('scientific_evaluation_calls'))is int and answer['scientific_evaluation_calls']==0:
                answer.update(failure_scope='CONFIGURATION',failure_kind='PRE_API_SCOPE_REJECTION')
            elif reason=='ADAPTER_EXIT_WITHOUT_OUTCOME':
                receipt=load_json(job/'PROCESS_RETURN.json')
                if receipt.get('returncode')is None and receipt.get('signal')in(2,15):
                    answer.update(failure_scope='CONFIGURATION',failure_kind='WORKER_INTERRUPTED')
        except(OSError,ValueError,KeyError,TypeError):pass
        return answer

    def ingest(self, job, proposal, calls):
        """Accept a job outcome only for the same configuration, allocation, total view/binding/signal and contract."""
        out, outcome_path = job_outcome(job);ident = out.get('identity') or {}
        nx, ny = split(proposal['vector']);rule = self.route(proposal['configuration_id'])
        want = self.expected_for(proposal['configuration_id'], rule)
        status, reason = out['status'], out.get('reason')
        if status in (PASS, FAIL, 'INCONCLUSIVE'):
            checks = dict(configuration=ident.get('configuration_id') == proposal['configuration_id'],
                          allocation=ident.get('allocation') == [dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(nx, ny))],
                          contract=ident.get('contract_matches') is True)
            for key in ('total_configuration_id', 'binding_id', 'M'):
                if want.get(key) is not None:
                    checks[key] = ident.get(key) == want[key]
            if (risk := (rule or {}).get('horizon_risk_source')) is not None:
                checks['horizon_risk_source'] = ident.get('M') == 13 and out.get('horizon_risk_source') == risk
            if not all(checks.values()):
                status, reason = 'UNKNOWN', 'IDENTITY_MISMATCH:' + ','.join(k for k, v in checks.items() if not v)
        source = 'ARTIFICIAL_JOB' if out.get('artificial') else 'NEW_Q3_JOB'
        lower = out.get('MSE_lower')
        upper = out.get('MSE_upper')
        lower = lower if lower is None or isinstance(lower, Fraction) else frac(lower)
        upper = upper if upper is None or isinstance(upper, Fraction) else frac(upper)
        result = dict(status=status, reason=reason, MSE_lower=lower, MSE_upper=upper, source=source, calls=calls,
                      scientific_evaluation_calls=out.get('scientific_evaluation_calls'),
                      job=str(job.relative_to(self.root)) if job.is_relative_to(self.root) else str(job), label=None)
        origin = proposal.get('origin')
        if origin is None and (job / 'REQUEST.json').is_file():
            origin = load_json(job / 'REQUEST.json').get('origin')
        if origin and origin.get('kind') == 'RPE_SCALED_SHOTS_V1':
            result['allocation_rule'] = origin['kind']
        return self.scope_unknown(job, proposal, result)


def science_needs(needs):
    return [need for need in needs if not need.startswith('PRICE_')]


def apply_published(candidates, routes, *, exploratory=False):
    """A generated configuration whose configuration, 13-round packet and Q3 route are published becomes searchable.

    The route rule's ``configurations`` table (configuration + MEAN_HANDOFF locators, total binding ids) is the
    single record of what was published; prices stay those of the price step (PriceRules), never filled here.
    """
    published = {}
    for name, rules in routes.items():
        for rule in rules:
            for cid, entry in (rule.get('configurations') or {}).items():
                published[(name, cid)] = entry
    cleared = ('CONFIGURATION_PUBLICATION', 'SIGNAL_PACKET_ASSEMBLY_FROM_REUSED_ROUNDS', 'Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION')
    for c in candidates:
        entry = published.get((c['method'], c['configuration_id']))
        if entry is None or not c.get('generated'):
            continue
        c['needs'] = [n for n in c['needs'] if n not in cleared and not n.startswith('SIGNAL_ROUNDS:')]
        c['signal_source'] = entry['signal_packet'];c['published_inputs'] = entry
        c['ready'] = (not science_needs(c['needs']) if exploratory else not c['needs'] and c['prices'] is not None)
    return candidates


def sibling_seeds(candidates, knowledge):
    """For configurations without a feasible point: the saved feasible incumbent of each other configuration of the
    same method and L_D (cheapest by that configuration's own prices), as an extra seed of the same rule."""
    best = {}
    for c in candidates:
        rows = [(cost(v, c['prices']), v, r) for (cid, v), r in knowledge.items()
                if c['ready'] and cid == c['configuration_id'] and r['status'] == PASS and r.get('source') == 'SAVED']
        if rows:
            t, v, r = min(rows, key=lambda x: (x[0], x[1]))
            best[c['configuration_id']] = dict(vector=list(v), configuration_id=c['configuration_id'], label=r.get('label'))
    for c in candidates:
        if c['ready'] and c['configuration_id'] not in best and c['family'] == 'PR':
            c['sibling_seeds'] = [b for cid, b in best.items() for o in candidates
                                  if o['configuration_id'] == cid and o['key'].get('L_D') == c['key'].get('L_D')]
    return candidates


# ---------------------------------------------------------------- PR configuration generation (metadata only)
def round_key(config, row):
    """Physical identity of one round: what the signal and the compiled circuit depend on (kappa only via R)."""
    s = config['spec'];st = row['structure'];split = s.get('split') or {}
    return json.dumps(dict(method=s.get('method'), L_D=s.get('L_D'), formula=s.get('formula'), K=s.get('finite_K'),
                           law=(s.get('law') or {}).get('model'), lambda_R=split.get('lambda_R'),
                           tail=split.get('lambda_source') or split.get('dictionary_identity'), T=row.get('T'), q=row.get('q_PF'),
                           R=row.get('R_outer'), formula_id=st.get('formula_id'), weights=st.get('signed_weights'),
                           allocation=st.get('allocation'), events=st.get('event_count')), sort_keys=True)


def schedule_keys(config):
    return [round_key(config, r) for r in sorted(config['rounds'], key=lambda r: r['m'])]


def dec_kappa(config):
    """spec.kappa.value as an exact rational (kappa itself, not a multiplier)."""
    v = config['spec']['kappa']['value']
    return Fraction(int(v['numerator']), int(v['denominator'])) if isinstance(v, dict) else Fraction(str(v))


def variant_grid(spec):
    g = spec['generation']
    for formula in g['formulas']:
        for delta in g['delta_over_h_values']:
            for kappa in g['kappa_multipliers']:
                yield dict(formula=formula, delta_over_h=delta, kappa_multiplier=kappa)


def generate_variants(root, problem, reader, catalog, candidates, methods=None, preparation_dir=None):
    """For each PR method with a ``generation`` range: regenerate every saved configuration over formula x
    step width x kappa with the method's fixed mapper (subprocess, route view, metadata only), then
    deduplicate by physical schedule.  Identical schedules alias the saved configuration (its evaluations are
    reused, never re-evaluated); new schedules become candidates listing exactly which inputs are missing."""
    out = {}
    for name, spec in problem['methods'].items():
        if spec['family'] != 'PR' or not spec.get('generation') or (methods is not None and name not in methods):
            continue
        g = spec['generation'];prepared = None
        if g.get('preparation') and any(c['method'] == name and c['configuration_id'] is None for c in candidates):
            # A's PR-Pauli preparation entry gets the search range's candidates without a configuration BEFORE the id filter
            # below: complete configurations are registered (even without price), the rest stay preparation requests.
            from . import rpe_candidate_preparation as R
            import tempfile
            if preparation_dir:
                prepared = R.prepare_unregistered(root, name, g['preparation'], candidates, g['baseline'], Path(preparation_dir) / name)
            else:  # library call without a run directory: responses are kept in the candidates, files are discarded
                with tempfile.TemporaryDirectory(prefix='rpe_prep_') as d:
                    prepared = R.prepare_unregistered(root, name, g['preparation'], candidates, g['baseline'], Path(d))
        saved = [c for c in candidates if c['method'] == name and c['configuration_id'] and not c.get('generated')]
        points = {}
        for point in catalog['candidates']:
            points.setdefault(point['configuration_id'], point)
        configs = {c['configuration_id']: reader.read(points[c['configuration_id']]['configuration_source']) for c in saved}
        base_keys = {cid: schedule_keys(cfg) for cid, cfg in configs.items()}
        for c in saved:
            c['schedule_keys'] = base_keys[c['configuration_id']]
            if c['family'] == 'PR' and (configs[c['configuration_id']].get('spec') or {}).get('kappa'):
                c['kappa_value'] = str(dec_kappa(configs[c['configuration_id']]))
        signal_keys = {k for c in saved if c['signal_source'] for k in base_keys[c['configuration_id']]}
        price_keys = {}
        for c in saved:
            if c['prices'] is not None:
                for m, k in enumerate(base_keys[c['configuration_id']]):
                    for a, axis in enumerate(AXES):
                        price_keys[(k, axis)] = c['prices'][2 * m + a]
        items = []
        for c in saved:
            loc = points[c['configuration_id']]['configuration_source']
            for v in variant_grid(spec):
                items.append(dict(item_id='%s|%s|%s|%s' % (c['configuration_id'][:12], v['formula'], v['delta_over_h'], v['kappa_multiplier']),
                                  base=loc, changes={k: val for k, val in v.items()
                                                     if str(val) != str(g['baseline'][k])}, variant=v, base_configuration_id=c['configuration_id']))
        result = run_generator(root, g, items)
        rows = []
        for item, gen in zip(items, result['items']):
            base = next(c for c in saved if c['configuration_id'] == item['base_configuration_id'])
            row = dict(variant=item['variant'], L_D=base['key']['L_D'], base_configuration_id=base['configuration_id'],
                       status=gen['status'], reason=gen.get('reason'), base_reproduced=gen.get('base_reproduced'),
                       supported_formulas=gen.get('supported_formulas'), generator=dict(kind=g['kind'], view_commit=g['view_commit']))
            if gen['status'] == 'GENERATED':
                keys = schedule_keys(gen['configuration'])
                same = next((cid for cid, bk in base_keys.items() if bk == keys), None)
                differ = [m for m, (a, b) in enumerate(zip(keys, base_keys[base['configuration_id']])) if a != b]
                row.update(configuration_id=gen['configuration_id'], rounds=gen['structure'], rounds_differing_from_base=differ,
                           physically_identical_to=same)
                if same is not None:
                    alias = next(c for c in candidates if c['configuration_id'] == same)
                    alias.setdefault('physical_aliases', []).append(dict(variant=item['variant'], configuration_id=gen['configuration_id']))
                    row['disposition'] = 'IDENTICAL_SCHEDULE_REUSES_SAVED_EVALUATIONS'
                else:
                    missing_signal = [m for m, k in enumerate(keys) if k not in signal_keys]
                    missing_price = [dict(m=m, axis=a) for m, k in enumerate(keys) for a in AXES if (k, a) not in price_keys]
                    needs = ['CONFIGURATION_PUBLICATION']
                    if missing_signal:
                        needs.append('SIGNAL_ROUNDS:' + ','.join(map(str, missing_signal)))
                    if len(missing_signal) < len(keys):
                        needs.append('SIGNAL_PACKET_ASSEMBLY_FROM_REUSED_ROUNDS')
                    if missing_price:
                        needs.append('PRICE_AXES:%d_of_%d' % (len(missing_price), len(keys)*len(AXES)))
                    needs.append('Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION')
                    prices = None if missing_price else [price_keys[(k, a)] for k in keys for a in AXES]
                    cand = dict(method=name, family='PR', key=dict(L_D=base['key']['L_D'], formula=item['variant']['formula'],
                                delta_over_h=item['variant']['delta_over_h'], kappa_multiplier=item['variant']['kappa_multiplier']),
                                configuration_id=gen['configuration_id'], index_signal=None, caps=base['caps'], generated=True,
                                generated_from=base['configuration_id'], labels=[], signal_source=None, prices=prices,
                                price_status='REUSED_FOR_ALL_AXES' if prices else 'MISSING_FOR_%d_AXES' % len(missing_price),
                                saved_points=[], needs=needs, ready=False, generated_configuration=gen['configuration'], schedule_keys=keys,
                                kappa_value=str(dec_kappa(gen['configuration'])) if (gen['configuration'].get('spec') or {}).get('kappa') else None,
                                reusable=dict(signal_rounds=[m for m in range(len(keys)) if m not in missing_signal],
                                              price_axes=len(keys)*len(AXES) - len(missing_price)))
                    cand['name'] = name + ':' + ','.join('%s=%s' % kv for kv in cand['key'].items()) + ':' + gen['configuration_id'][:8]
                    if not any(c['configuration_id'] == cand['configuration_id'] for c in candidates):
                        candidates.append(cand)
                    row.update(disposition='NEW_SCHEDULE_CANDIDATE', needs=needs, reusable=cand['reusable'], candidate=cand['name'])
            else:
                row['disposition'] = 'NOT_GENERATED'
                row['needs'] = ['GENERATOR_SUPPORT:' + (gen.get('reason') or gen['status'])]
            rows.append(row)
        kappa_rows = prepare_kappa_points(root, name, g, candidates, points, configs, preparation_dir)
        out[name] = dict(kind=g['kind'], view_commit=g['view_commit'], preparation=prepared, kappa_points=kappa_rows,
                         preparation_source=(g.get('preparation') or {}).get('view_commit'),
                         supported_formulas=result['items'][0].get('supported_formulas') if result['items'] else None,
                         baseline=g['baseline'], range=dict(formulas=g['formulas'], delta_over_h=g['delta_over_h_values'],
                         kappa_multipliers=g['kappa_multipliers']), h='fixed RPE base time (not a search variable)', variants=rows)
    return out


def prepare_kappa_points(root, name, g, candidates, points, configs, preparation_dir):
    """Absolute-kappa points of the round (``generation.kappa_points`` or ``kappa_points_file``: [{L_D, kappa}]) through A's
    preparation entry; base = the saved configuration of that L_D (or one A registered), multiplier = kappa / base kappa
    (exact), and the generated spec.kappa.value must equal the target exactly before the point is registered."""
    kp = list(g.get('kappa_points') or [])
    f = g.get('kappa_points_file')
    if f and Path(f).exists():
        kp += json.loads(Path(f).read_text())
    if not kp or not g.get('preparation'):
        return None
    from . import rpe_candidate_preparation as R
    import tempfile
    bases = {}
    for c in candidates:
        if c['method'] != name or not c['configuration_id'] or c.get('kappa_value') is None:
            continue
        ld = c['key'].get('L_D');saved_base = not c.get('generated') and c['configuration_id'] in points
        a_registered = c.get('prepared_by') and c.get('configuration_source') and c['key'].get('kappa_multiplier', '1') == '1' and 'kappa' not in c['key']
        if (saved_base or a_registered) and ld not in bases:
            src = points[c['configuration_id']]['configuration_source'] if saved_base else c['configuration_source']
            bases[ld] = dict(source=src, configuration_id=c['configuration_id'], kappa_value=c['kappa_value'])
    base = g['baseline']
    # a point may name its own width delta_pf (JSON value kept: '1/5' exact, 0.2 binary64); otherwise the baseline delta/h
    proposals = [dict(kind='KAPPA_POINT', family=dict(L_D=int(p['L_D']), formula=base['formula'], delta_over_h=base['delta_over_h'])
                      if p.get('delta_pf') is None else dict(L_D=int(p['L_D']), formula=base['formula'], delta_over_h=None,
                                                             delta_pf=p['delta_pf']),
                      kappa=str(Fraction(str(p['kappa'])))) for p in kp]
    if preparation_dir:
        return R.prepare_kappa(root, name, g['preparation'], candidates, proposals, bases, Path(preparation_dir) / name, label='kappa_points')
    with tempfile.TemporaryDirectory(prefix='rpe_prep_') as d:
        return R.prepare_kappa(root, name, g['preparation'], candidates, proposals, bases, Path(d), label='kappa_points')


def run_generator(root, g, items):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        req, out = Path(d) / 'REQUEST.json', Path(d) / 'OUT.json'
        req.write_text(json.dumps(dict(kind=g['kind'], view=g['view'], view_commit=g['view_commit'],
                                       items=[{k: v for k, v in i.items() if k in ('item_id', 'base', 'changes')} for i in items])))
        subprocess.run([g['interpreter'], '-B', str(Path(root) / 'scripts/run_rpe_optimizer_config_generator.py'),
                        '--request', str(req), '--output', str(out)], cwd=str(root), check=True, stdout=subprocess.DEVNULL,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        return json.loads(out.read_text())


def quote_prices(root, problem, candidates, catalog, methods=None):
    """D2's PriceRules (explicit policy, MODEL for the saved comparison) for every PR candidate.

    Saved configurations: the quoted unit prices must reproduce the catalog's own unit prices (hence every saved
    total).  Generated configurations: a complete 26-axis quote becomes the candidate's prices; otherwise the
    missing axes, reasons and required calibrations are carried into NEEDS.  Price status is kept separate from
    signal/Q3 readiness.
    """
    pr = problem['price_rules'];policy = pr['policy']
    if policy != 'MODEL':
        raise ValueError('The saved comparison is reproduced with policy MODEL; change it only by an explicit new round')
    targets = [c for c in candidates if c['family'] == 'PR' and c['configuration_id'] and (methods is None or c['method'] in methods)]
    if not targets:
        return dict(policy=policy, quotes=[])
    import tempfile
    reader=Reader(Path(root))
    saved_points={}
    for point in catalog['candidates']:saved_points.setdefault(point['configuration_id'],[]).append(point)
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / 'PROBE.json';(Path(d) / 'P.json').write_text(json.dumps(dict(policy=policy, items=[])))
        run_quotes(root, pr, Path(d) / 'P.json', probe)
        registry = {v['configuration_id']: k for k, v in json.loads(probe.read_text())['registry'].items()}
        items = []
        names = set(registry.values())
        for c in targets:
            base = c.get('generated_from') or c['configuration_id']
            # A split registered after the saved catalog (e.g. a newly calibrated PAULI_LD<p>) is found by its entry name.
            entry_name = ('PAULI_LD%s' if c['method'] == 'PR_PAULI' else 'DF_LD%s') % c['key'].get('L_D')
            if base not in registry and entry_name not in names:
                c['price_quote'] = dict(policy=policy, status='NO_REGISTERED_PRICE_RULE_FOR_BASE');continue
            config=c.get('generated_configuration')
            if config is None:
                source=c.get('configuration_source')or(saved_points.get(c['configuration_id'])or[{}])[0].get('configuration_source')
                if source is None:
                    c['price_quote']=dict(policy=policy,status='CANDIDATE_CONFIGURATION_SOURCE_MISSING');continue
                config=reader.read(source)
            if config['id']!=c['configuration_id']:raise ValueError('PRICE_QUOTE_CONFIGURATION_ID_DIFFERS')
            items.append(dict(item_id=c['name'], registry=registry[base] if base in registry else entry_name,configuration=config))
        req, out = Path(d) / 'REQ.json', Path(d) / 'OUT.json'
        req.write_text(json.dumps(dict(policy=policy, items=items)))
        run_quotes(root, pr, req, out)
        result = json.loads(out.read_text())
    by_name = {c['name']: c for c in targets};report = []
    for q in result['items']:
        c = by_name[q['item_id']];rows = q.get('rows') or []
        if q['complete']and q.get('configuration_id')!=c['configuration_id']:
            raise ValueError('PRICE_QUOTE_RETURNED_ANOTHER_CONFIGURATION')
        quoted = [frac(r['unit_price_RZ']) for r in rows] if q['complete'] else None
        counts = {}
        for r in rows:
            counts[r['classification']] = counts.get(r['classification'], 0) + 1
        missing = [dict(m=r['m'], axis=r['axis'], reasons=(r.get('needs') or {}).get('reasons'),
                        calibration=(r.get('needs') or {}).get('calibration') or (r.get('needs') or {}).get('missing_quantity'))
                   for r in rows if r['unit_price_RZ'] is None]
        c['price_model_validations']=q.get('model_validations',[])
        c['price_quote'] = dict(policy=q.get('policy', policy), status=q['status'], complete=q['complete'], registry=q['registry'],
                                classifications=counts, extrapolated_axes=sum(1 for r in rows if r.get('extrapolation')),
                                reason=q.get('reason'), implementation_source=pr['implementation_source'])
        entry = dict(candidate=c['name'], configuration_id=c['configuration_id'], generated=bool(c.get('generated')),
                     registry=q['registry'], status=q['status'], complete=q['complete'], classifications=counts)
        if not c.get('generated'):
            entry['reproduces_saved_unit_prices'] = None if c['prices'] is None else (quoted == list(c['prices']))
            if c['prices'] is None and q['complete']:
                points=[p for p in saved_points.get(c['configuration_id'],[])if p.get('conditional_cost')is not None]
                axes=[(r['m'],r['axis'])for r in rows]
                rounds=len(c['caps']);axis_count=rounds*len(AXES)
                valid=(c['price_status']==f'NO_SAVED_{axis_count}_AXIS_PRICE'and all(not p.get('price_axes')for p in points)
                       and axes==[(m,a)for m in range(rounds)for a in AXES]and all(v>0 for v in quoted)and points
                       and all(cost(point_vector(p),quoted)==frac(p['conditional_cost'])for p in points))
                entry['reproduces_all_saved_totals']=bool(valid)
                if valid:
                    c.update(prices=quoted,price_status='SOURCE_BOUND_MODEL_QUOTE_REPRODUCES_ALL_SAVED_TOTALS')
                    c['needs']=[n for n in c['needs']if n!=f'PRICE_{axis_count}AXIS'];c['ready']=not c['needs']
                    entry['note']='Own MODEL quote, same canonical configuration, exact match to every saved total'
                else:entry['note']='Own quote does not reproduce saved totals; retain price need'
        else:
            c['needs'] = [n for n in c['needs'] if not n.startswith('PRICE_AXES')]
            if q['complete']:
                c['prices'] = quoted;c['price_status'] = 'PRICE_RULES_%s_COMPLETE' % policy
            else:
                c['prices'] = None;c['price_status'] = 'PRICE_RULES_%s_INCOMPLETE' % policy
                c['needs'].append('PRICE_AXES:%d_of_%d' % (len(missing), len(rows)))
                c['price_needs'] = missing
            c['ready'] = not c['needs']
            entry['missing_axes'] = len(missing)
            entry['missing_reasons'] = sorted({r for m in missing for r in (m['reasons'] or [])})
        report.append(entry)
    return dict(policy=policy, implementation_source=pr['implementation_source'], handoff=pr.get('handoff'), quotes=report)


def run_quotes(root, pr, request, output):
    cmd = [pr['interpreter'], '-B', str(Path(root) / 'scripts/run_rpe_optimizer_price_quotes.py'), '--view', pr['view'],
           '--request', str(request), '--output', str(output)]
    if pr.get('registry_file'):  # a registry extended with newly calibrated split entries (sources are committed locators)
        f = Path(pr['registry_file']);cmd += ['--registry', str(f if f.is_absolute() else Path(root) / f)]
    subprocess.run(cmd, cwd=str(root), check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))


def parameter_assessment(name, candidates, generation, summary):
    """Per (L_D, formula, delta): the kappa rows with their schedule identity, readiness and best feasible cost."""
    if not generation:
        return None
    best = {}
    for e in summary['evaluated']:
        if e['status'] == PASS and e['cost'] is not None:
            best[e['configuration_id']] = min(best.get(e['configuration_id'], e['cost']), e['cost'])
    groups = {}
    for row in generation['variants']:
        v = row['variant'];cid = row.get('physically_identical_to') or row.get('configuration_id')
        groups.setdefault((row['L_D'], row['base_configuration_id'], v['formula'], v['delta_over_h']), []).append(dict(
            kappa_multiplier=v['kappa_multiplier'], disposition=row['disposition'], configuration_id=cid,
            rounds_differing_from_base=row.get('rounds_differing_from_base'), needs=row.get('needs'),
            best_feasible_cost=best.get(cid)))
    out = []
    for (ld, base, formula, delta), rows in groups.items():
        if all(r['disposition'] == 'IDENTICAL_SCHEDULE_REUSES_SAVED_EVALUATIONS' for r in rows):
            decision = 'SAME_PHYSICAL_SCHEDULE_FOR_ALL_KAPPA_IN_RANGE'
        elif all(r['best_feasible_cost'] is not None for r in rows):
            decision = 'DECIDABLE_FROM_MINIMUM_FEASIBLE_COST'
        else:
            decision = 'UNDECIDED_INPUTS_OR_EVALUATIONS_MISSING'
        out.append(dict(L_D=ld, base_configuration_id=base, formula=formula, delta_over_h=delta, kappa_rows=rows, kappa_decision=decision))
    return out


# ---------------------------------------------------------------- orchestration
def initial_knowledge(saved, candidates, initial='all'):
    """Known results at start: all saved, only the listed saved labels (replay), or none."""
    known = {}
    if initial == 'none':
        return known
    ids = {c['configuration_id'] for c in candidates if c['configuration_id']}
    for (cid, v), r in saved.table.items():
        if cid in ids and (initial == 'all' or r['label'] in initial):
            known[(cid, v)] = dict(r, calls=0)
    return known


def ingest(knowledge, candidates, results):
    """Add externally obtained Q3 results (e.g. a newly approved call) to the knowledge."""
    ids = {c['configuration_id']: c for c in candidates if c['configuration_id']}
    for r in results:
        if r['configuration_id'] not in ids or r.get('status') not in (PASS, FAIL, 'INCONCLUSIVE'):
            continue
        v = interleave(r['N_X'], r['N_Y']);c = ids[r['configuration_id']]
        knowledge[(r['configuration_id'], v)] = dict(status=r['status'], MSE_lower=frac(r['MSE_lower']), MSE_upper=frac(r['MSE_upper']),
                                                     source=r.get('source', 'INGESTED'), calls=0, label=r.get('label'),
                                                     cost=None if c['prices'] is None else cost(v, c['prices']))
    return knowledge


def summarize_method(name, candidates, result, knowledge, threshold):
    inc = result['incumbent']
    best = None
    if inc is not None:
        c = inc['candidate'];nx, ny = split(inc['vector']);r = inc['record']
        best = dict(candidate=c['name'], configuration_id=c['configuration_id'], configuration=c['key'], labels=c['labels'],
                    N_X=nx, N_Y=ny, MSE_lower=r['MSE_lower'], MSE_upper=r['MSE_upper'], status=r['status'],
                    cost=inc['cost'], cost_display=show(inc['cost']), evidence=r['source'], point_id=r.get('point_id'),
                    result_sources=r.get('result_sources'))
    evaluated = [dict(candidate=next(c['name'] for c in candidates if c['configuration_id'] == cid), configuration_id=cid,
                      vector=list(v), N_X=split(v)[0], N_Y=split(v)[1], status=r['status'], MSE_lower=r.get('MSE_lower'),
                      MSE_upper=r.get('MSE_upper'), cost=None if r.get('cost') is None else r['cost'], source=r['source'],
                      result_sources=r.get('result_sources'), point_id=r.get('point_id'), job=r.get('job'), label=r.get('label'))
                 for (cid, v), r in knowledge.items() if any(c['configuration_id'] == cid for c in candidates)]
    unevaluated = [dict(candidate=c['name'], configuration=c['key'], configuration_id=c['configuration_id'], needs=c['needs'] + c.get('execution_needs', []),
                        execution_binding_hold=c.get('execution_binding_hold'),
                        price_status=c['price_status'], price_quote=c.get('price_quote'), price_needs=c.get('price_needs'),
                        signal_source=c['signal_source'], reusable=c.get('reusable'),
                        next='Q3 of a generated/decremented allocation after the listed items exist')
                   for c in candidates if not c['ready'] or c.get('execution_needs')]
    return dict(method=name, best=best, stop_reason=result['stop_reason'], pending_request=result['pending_request'],
                trace=result['trace'], held=result['held'], new_calls=result['new_calls'],
                evaluated_count=len(evaluated), evaluated=evaluated, not_ready_candidates=unevaluated,
                ready_candidates=[c['name'] for c in candidates if c['ready']],
                search_range=dict(candidates=[c['name'] for c in candidates], rule=result.get('rule', RULE)),
                resume_records=resume_records(evaluated), by_configuration=result.get('by_configuration'),
                incumbent_scope=result.get('incumbent_scope', 'METHOD'))


def resume_records(evaluated):
    """Evaluations obtained in this run (not C's saved points), in the ``ingest`` format: resume input as written."""
    return [dict(configuration_id=e['configuration_id'], N_X=e['N_X'], N_Y=e['N_Y'], status=e['status'],
                 MSE_lower=None if e['MSE_lower'] is None else enc(e['MSE_lower']),
                 MSE_upper=None if e['MSE_upper'] is None else enc(e['MSE_upper']), source=e['source'], job=e.get('job'))
            for e in evaluated if e['source'] != 'SAVED' and e['status'] in (PASS, FAIL, 'INCONCLUSIVE')]


def own_price_comparison(candidates, vectors):
    """Same integer vector priced by each ready configuration's own 26-axis prices (arithmetic only)."""
    rows = []
    for label, v in vectors.items():
        for c in candidates:
            if c['ready'] and c['prices'] is not None:
                rows.append(dict(vector_label=label, candidate=c['name'], cost=cost(v, c['prices'])))
    return rows


def jsonable(value):
    if isinstance(value, Fraction):
        portable = max(len(str(abs(value.numerator))), len(str(value.denominator))) > 1000
        to = str if portable else int  # very long exact bounds stay readable by default JSON parsers (as in C)
        return dict(numerator=to(value.numerator), denominator=to(value.denominator), display=display(enc(value)))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != 'generated_configuration' and (k not in ('prices', 'candidate') or not isinstance(v, (list, dict)))}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scope_absolute_kappa(context, problem):
    """Apply the same exact PR design domain to every candidate source."""
    if not any(v.get('absolute_kappa_bounds') is not None or v.get('absolute_kappa_values') is not None
               for v in problem['methods'].values()):
        return context
    from .rpe_kappa_schedule import _frac
    saved = {p['configuration_id']: p for p in context['catalog']['candidates']}
    kept, excluded = [], []
    for candidate in context['candidates']:
        setting = problem['methods'][candidate['method']]
        bounds = setting.get('absolute_kappa_bounds')
        grid = setting.get('absolute_kappa_values')
        if bounds is None and grid is None:
            kept.append(candidate)
            continue
        meta = saved.get(candidate.get('configuration_id'), {})
        generated = (candidate.get('generated_configuration') or {}).get('spec', {})
        allowed = setting.get('L_D_values')
        ld = candidate.get('L_D', candidate.get('key', {}).get('L_D', generated.get('L_D', meta.get('L_D'))))
        if allowed is not None and ld not in allowed:
            excluded.append(dict(configuration_id=candidate['configuration_id'], L_D=ld,
                                 reason='OUTSIDE_REQUESTED_L_D_VALUES'))
            continue
        formulas = (setting.get('generation') or {}).get('formulas')
        formula = generated.get('formula', candidate.get('key', {}).get('formula', meta.get('formula')))
        if formulas is not None and candidate.get('configuration_id') is not None and formula not in formulas:
            excluded.append(dict(configuration_id=candidate['configuration_id'], formula=formula,
                                 reason='OUTSIDE_REQUESTED_PR_FORMULAS'))
            continue
        widths = setting.get('requested_delta_pf_values')
        width = generated.get('delta_pf', meta.get('delta_pf'))
        if widths is not None and candidate.get('configuration_id') is not None and (
                width is None or _frac(width) not in {_frac(v) for v in widths}):
            excluded.append(dict(configuration_id=candidate['configuration_id'],
                                 reason='OUTSIDE_REQUESTED_PR_WIDTHS'))
            continue
        if candidate.get('configuration_id') is None:
            kept.append(candidate)
            continue
        value = candidate.get('kappa_value')
        if value is None and candidate.get('generated_configuration'):
            value = candidate['generated_configuration']['spec']['kappa']['value']
        if value is None:
            value = saved.get(candidate['configuration_id'], {}).get('kappa')
        if isinstance(value, dict) and 'value' in value:
            value = value['value']
        if value is None:
            excluded.append(dict(configuration_id=candidate['configuration_id'],
                                 reason='MISSING_ABSOLUTE_KAPPA_FOR_SCOPE_CHECK'))
            continue
        kappa = _frac(value)
        inside_bounds = bounds is None or _frac(bounds[0]) <= kappa <= _frac(bounds[1])
        inside_grid = grid is None or kappa in {_frac(v) for v in grid}
        if inside_bounds and inside_grid:
            candidate['kappa_value'] = str(kappa)
            kept.append(candidate)
        else:
            excluded.append(dict(configuration_id=candidate['configuration_id'], kappa=str(kappa),
                                 reason='OUTSIDE_REQUESTED_ABSOLUTE_KAPPA_VALUES' if not inside_grid
                                 else 'OUTSIDE_INCLUSIVE_ABSOLUTE_KAPPA_BOUNDS'))
    context['candidates'] = kept
    context.setdefault('scope_excluded', []).extend(excluded)
    return context


def optimize(root, problem, *, context=None, methods=None, initial='all', extra_results=(), backend=None, max_new=None,
             executor=None, saved_is_free=True, science_approved=True, ready_source=None, scaled_finalizers=None,
             close_executor=True, on_feasible=None, price_work_pending=None):
    """Run the per-method search.

    initial: 'all' saved results, 'none', or {method: [saved labels]} (replay from seeds).
    backend: Q3 answer source; default = saved results (C's catalog).  With ``executor`` (an approved round),
    saved results are tried first, then completed/new jobs of that round.  ``max_new`` is an int or a
    {method: int} budget of new Q3 calls; calls already launched in the round count against it.
    saved_is_free: the backend answers C's saved configuration+allocation without a new call (set False for a
    backend that does not consult the saved results, e.g. a stand-alone artificial fixture).
    """
    context = scope_absolute_kappa(context or prepare(root, problem), problem)
    search = problem['search'];threshold = context['tau']
    max_new = search['max_new_q3_per_method'] if max_new is None else max_new
    out = {}
    for name in methods or problem['methods']:
        cands = [c for c in context['candidates'] if c['method'] == name]
        if ready_source is not None:
            # Each live selector owns its candidate dictionaries and route view.
            cands = deepcopy(cands)
        if executor is not None:
            ex = executor.for_method(name)
            for c in cands:
                if c['ready']:
                    c['execution_needs'] = ([] if ex.route(c['configuration_id']) is not None
                                            else ['Q3_ROUTE_CONFIGURATION_OR_TEMPLATE'])
                    hold=ex.refresh_execution_hold(c['configuration_id'])
                    if hold:
                        c['execution_hold']=hold;c['execution_needs'].append('SAVED_JOB_REVIEW_REQUIRED')
                    else:c.pop('execution_hold',None)
        base_candidates = cands
        if search.get('rule') == 'RPE_SCALED_SHOTS_V1':
            from . import rpe_scaled_shots as scaled
            policy = scaled.settings(search['scaled_shots'])
            cands = scaled.horizon_candidates(base_candidates, problem, policy,
                                              ex if executor is not None else None, context.get('catalog'))
        parallel_history = (ready_source is not None and getattr(ex, 'optimizer_history_directory', None) is not None
                            and getattr(ex, 'preparation_history_cache', None) is None
                            and getattr(ex, 'supports_nonblocking_prefetch', False))
        budget = max_new.get(name, 0) if isinstance(max_new, dict) else max_new
        budget = None if budget in (None, 'UNLIMITED') else budget
        history_lanes = 1
        if parallel_history and search.get('controller_parallel') is True:
            from .rpe_q3_resources import controller_cpu_count
            history_lanes = max(1, controller_cpu_count() // max(1, len(problem['methods'])))
        stream_initial = (parallel_history and search.get('rule') == 'RPE_SCALED_SHOTS_V1'
                          and search.get('work_conserving_search') is True
                          and (budget is None or search.get('controller_parallel') is True)
                          and bool(ex.prior_run_dirs))
        initial_ready = []
        if stream_initial:
            # Each scaled allocation series has its own minimum. Its history
            # must be complete before it can propose work.
            initial_ready = [c for c in cands if c['ready'] and c.get('configuration_id')]
            cands = [c for c in cands if not (c['ready'] and c.get('configuration_id'))]
        start = initial if isinstance(initial, str) else initial.get(name, [])
        knowledge = ingest(initial_knowledge(context['saved'], cands, start), cands, extra_results)
        table = context['saved'].table if saved_is_free else {}
        prefetch = None
        if executor is not None:
            answer, spent = ChainBackend(context['saved'], ex), ex.launched(name)
            free = lambda p, ex=ex: (p['configuration_id'], p['vector']) in table or ex.has_job(p)
            if getattr(ex, 'supports_nonblocking_prefetch', False) and search.get('rule') == 'METHOD_MINIMUM_PRIORITY_V2':
                prefetch = lambda ps, ex=ex: ex.prefetch([p for p in ps if (p['configuration_id'], p['vector']) not in table], scope='METHOD')
            else:
                prefetch = lambda ps, ex=ex: ex.prefetch([p for p in ps if (p['configuration_id'], p['vector']) not in table])
        else:
            answer, spent = (context['saved'] if backend is None else backend), 0
            free = lambda p: (p['configuration_id'], p['vector']) in table
        if stream_initial:
            prior = dict(run_dirs=[str(d) for d in ex.prior_run_dirs], loaded=0, already_known=0,
                         refused=0, refused_reasons=[], streamed=True)
        elif executor is not None and executor.prior_run_dirs:
            ingest_start = time.perf_counter()
            executor.for_method(name).log(event='PRIOR_INGEST_START')
            known, refused = ex.prior_records(cands)
            executor.for_method(name).log(event='PRIOR_INGEST_END', seconds=time.perf_counter()-ingest_start,
                                          reused=len(known), refused=len(refused))
            fresh = {k: v for k, v in known.items() if k not in knowledge}
            knowledge.update(fresh)
            prior = dict(run_dirs=[str(d) for d in executor.prior_run_dirs], loaded=len(fresh), already_known=len(known) - len(fresh),
                         refused=len(refused), refused_reasons=sorted({r['reason'] for r in refused}))
        else:
            prior = None
        prior_q3_holds = [item for item in getattr(executor, 'prior_q3_holds', ()) if item['method'] == name]
        if prior_q3_holds and search.get('rule') not in ('METHOD_MINIMUM_PRIORITY_V2', 'RPE_SCALED_SHOTS_V1'):
            raise ValueError('Prior Q3 holds require an isolated-candidate search')
        if search.get('rule') in ('METHOD_MINIMUM_PRIORITY_V2', 'RPE_SCALED_SHOTS_V1'):
            from trottertracks import rpe_priority_search as P
            refresh_ready = None
            history_pool = None
            ready_history = {}
            initial_batches = deque()
            late_batches = deque()
            remaining = iter(initial_ready)
            size = 1
            while batch := list(islice(remaining, size)):
                initial_batches.append(batch)
                # Fit the first placements quickly and bound each later scan.
                size = min(size * 2, 8)
            if ready_source is not None:
                pending_ready, take_ready = ready_source.subscribe(name, context, base_candidates, ex)

                def validate_ready(view, added, saved_snapshot, initial_batch=False):
                    saved = ingest(initial_knowledge(saved_snapshot or context['saved'], added, start), added, extra_results)
                    began = time.perf_counter()
                    prior_saved, refused = view.prior_records(added)
                    view.log(event='PRIOR_READY_INGEST_END', seconds=time.perf_counter()-began,
                             configurations=len(added), reused=len(prior_saved),
                             lane='initial' if initial_batch else 'late')
                    fresh = {key: value for key, value in prior_saved.items() if key not in saved}
                    if initial_batch:
                        scoped = [row for row in refused if row['reason'] != 'CONFIGURATION_NOT_IN_THIS_ROUND']
                        with ex.state.cv:
                            prior['loaded'] += len(fresh)
                            prior['already_known'] += len(prior_saved) - len(fresh)
                            prior['refused'] += len(scoped)
                            prior['refused_reasons'] = sorted(set(prior['refused_reasons']) | {row['reason'] for row in scoped})
                    saved.update(fresh)
                    return added, saved

                initial_lanes = [('initial', i) for i in range(history_lanes)]
                late_lane = ('late', 0)

                def ready_pending():
                    return (any(future.done() for future in ready_history.values())
                            or (late_lane not in ready_history and (bool(late_batches) or pending_ready()))
                            or (bool(initial_batches) and any(lane not in ready_history for lane in initial_lanes)))

                def wake_selector(_):
                    with ex.state.cv:
                        ex.state.cv.notify_all()

                def stage_history(lane, added, saved_snapshot=None):
                    view = ex.for_method(name)
                    view.routes, view.expected = dict(ex.routes), dict(ex.expected)
                    view.log(event='PRIOR_READY_INGEST_START', configurations=len(added), lane=lane[0])
                    future = history_pool.submit(validate_ready, view, added, saved_snapshot, lane[0] == 'initial')
                    ready_history[lane] = future
                    future.add_done_callback(wake_selector)

                def await_ready():
                    with ex.state.cv:
                        while not ready_pending():
                            if ((price_work_pending is not None and price_work_pending())
                                    or not ready_history or ex.state.closed or ex.state.failure is not None
                                    or (ex.run_dir/'STOP_REQUESTED.json').exists()):
                                return False
                            ex.state.cv.wait(.5)
                        return True

                ex.ready_pending = ready_pending

                def refresh_ready():
                    added, saved = [], {}
                    for lane in (late_lane, *initial_lanes):
                        future = ready_history.get(lane)
                        if future is not None and future.done():
                            del ready_history[lane]
                            checked, known = future.result()
                            added.extend(checked)
                            saved.update(known)
                    if late_lane not in ready_history and not late_batches and pending_ready():
                        arrived, saved_snapshot = take_ready()
                        if arrived and search.get('rule') == 'RPE_SCALED_SHOTS_V1':
                            arrived = scaled.horizon_candidates(arrived, problem, policy, ex, context.get('catalog'))
                        if arrived and history_pool is None:
                            checked, known = validate_ready(ex, arrived, saved_snapshot)
                            added.extend(checked)
                            saved.update(known)
                        else:
                            # Keep subsequent cohorts bounded, with one validation in
                            # flight per lane. Older cohorts cannot occupy the late lane.
                            remaining = iter(arrived)
                            size = 1
                            while batch := list(islice(remaining, size)):
                                late_batches.append((batch, saved_snapshot))
                                size = min(size * 2, 8)
                    if history_pool is not None:
                        if late_lane not in ready_history and late_batches:
                            stage_history(late_lane, *late_batches.popleft())
                        for lane in initial_lanes:
                            if lane not in ready_history and initial_batches:
                                stage_history(lane, initial_batches.popleft())
                    ids = {c['configuration_id'] for c in added}
                    cands[:] = [c for c in cands if c['configuration_id'] not in ids] + added
                    return added, saved
            log = None
            if executor is not None:
                ex_log = executor.for_method(name)
                log = lambda row, ex_log=ex_log: ex_log.log(event='SEARCH_ADOPT' if getattr(ex_log, 'supports_nonblocking_prefetch', False) else 'ADOPT', configuration=row['configuration_id'][:8], job=row.get('job'), cls=row['cls'],
                                                           status=row['status'], cost=show(row['cost']),
                                                           method_min=row['method_min_after'] and show(row['method_min_after']),
                                                           **({k: row[k] for k in ('order_reason', 'order_baseline') if k in row}))
            standby = None
            if search.get('work_conserving_search') is True:
                if executor is None or not getattr(ex, 'prefetch_standby', None):
                    raise ValueError('work_conserving_search requires the concurrent Q3 executor')
                def standby(choice, proposals):
                    def saved(p):
                        return (p['configuration_id'], p['vector']) in table
                    ex.prefetch_standby(None if choice is None or saved(choice) else choice, [p for p in proposals if not saved(p)],
                        **({'retain_independent': True} if search.get('rule') == 'RPE_SCALED_SHOTS_V1'
                           and hasattr(ex, 'retire_queued_allocations') else {}))
            if search.get('rule') == 'RPE_SCALED_SHOTS_V1':
                read_completed = None
                refresh_completions = None
                answer_now = None
                await_dispatch = None
                if executor is not None and price_work_pending is not None:
                    candidate_ready = getattr(ex, 'ready_pending', lambda: False)
                    ex.ready_pending = lambda: price_work_pending() or candidate_ready()
                if search.get('controller_parallel') is True and getattr(executor, 'answer_now', None):
                    def answer_now(proposal):
                        cached = context['saved'].answer(proposal)
                        return cached if cached is not None else ex.answer_now(proposal)
                    await_dispatch = ex.await_dispatch
                if executor is not None and hasattr(ex, 'read_scaled_completions'):
                    if hasattr(ex, 'recover_scaled_completions'):
                        ex.recover_scaled_completions()
                    ex.completion_pending = ex.scaled_completion_pending
                    read_completed = ex.read_scaled_completions
                    refresh_completions = getattr(ex, 'refresh_scaled_completions', None)

                def after_drain(finalize):
                    def update(summary):
                        resolved = finalize()
                        if resolved is not None:
                            summary['rule_adopted_best'] = summary['best']
                            summary.update(summarize_method(name, cands, resolved, knowledge, threshold))
                            summary['input_requests'] = resolved['input_requests']
                            summary['scaled_shots'] = resolved['scaled_shots']
                    scaled_finalizers[name] = update

                def submit_standby(choice, proposals):
                    if standby is not None:
                        standby(choice, proposals)
                    elif prefetch is not None:
                        prefetch(proposals[:max(0, search.get('lookahead', 1)-1)])
                from contextlib import nullcontext
                from concurrent.futures import ThreadPoolExecutor
                method_scope = problem['methods'][name]
                if (method_scope.get('family') == 'PR' and method_scope.get('L_D_values') is None
                        and method_scope.get('configuration_ids') is None and 'L_D_range' in method_scope):
                    domain = method_scope['L_D_range']
                    upper = context['facts']['fragment_count'] + domain['max_offset_from_fragment_count']
                    method_scope = dict(method_scope, L_D_values=range(domain['min'], upper + 1))
                with (ThreadPoolExecutor(max_workers=1 + history_lanes if stream_initial else 1, thread_name_prefix='rpe-ready-history')
                      if parallel_history else nullcontext()) as history_pool:
                    result = scaled.run(name, cands, knowledge, answer, search, threshold, budget,
                                        spent=spent, free=free, science_approved=science_approved,
                                        method_scope=method_scope, refresh_ready=refresh_ready, submit_standby=submit_standby,
                                        await_ready=await_ready if parallel_history else None,
                                        answer_now=answer_now, await_dispatch=await_dispatch,
                                        on_feasible=on_feasible, price_work_pending=price_work_pending,
                                        retire_allocations=getattr(ex, 'retire_queued_allocations', None) if executor is not None else None,
                                        read_completed=read_completed, refresh_completions=refresh_completions,
                                        after_drain=after_drain if scaled_finalizers is not None else None,
                                        held=prior_q3_holds,
                                        flagged_stop=(lambda: (ex.run_dir/'STOP_REQUESTED.json').exists()) if executor is not None else None)
                    if executor is not None and (ex.run_dir/'STOP_REQUESTED.json').exists():
                        for future in ready_history.values():
                            future.cancel()
                    for future in ready_history.values():
                        if not future.cancelled():
                            future.result()
            else:
                result = P.run_priority(name, cands, knowledge, answer, search, threshold, context['generate'], budget,
                                        baseline=(problem['methods'][name].get('generation') or {}).get('baseline'),
                                        routes=problem.get('q3_routes'), spent=spent, free=free, science_approved=science_approved,
                                        prefetch=prefetch, on_missing=search.get('on_missing', 'STOP'), facts=context.get('facts'), log=log,
                                        prefetch_state=getattr(ex, 'prefetch_state', None) if executor is not None else None,
                                        refresh_ready=refresh_ready, standby=standby,
                                        search_observations=ex.search_observations if standby is not None else None,
                                        search_profile=ex.search_profile_for_proposal if standby is not None else None,
                                        prior_q3_holds=prior_q3_holds,
                                        retire_priced_out=getattr(ex,'retire_priced_out',None) if executor is not None else None)
                prep = (problem['methods'][name].get('generation') or {}).get('preparation')
                if prep and search.get('prepare_kappa_proposals'):
                    result['input_requests'] = prepare_kappa_requests(root, problem, context, name, prep, result['input_requests'])
        elif search.get('incumbent_scope') == 'CONFIGURATION':
            result = run_by_configuration(name, cands, knowledge, answer, search, threshold, context['generate'], budget,
                                          problem.get('q3_routes'), spent, free, science_approved, prefetch)
        else:
            result = run_method(name, cands, knowledge, answer, search, threshold, context['generate'], budget,
                                problem.get('q3_routes'), spent, free, science_approved)
        out[name] = summarize_method(name, cands, result, knowledge, threshold)
        if executor is not None:
            out[name]['execution_holds']=list(executor.execution_holds.get(name,{}).values())
        out[name]['prior_rounds'] = prior
        for key in ('unevaluated', 'input_requests', 'left', 'settings', 'prefetch_policy', 'selection_timing', 'prior_q3_holds', 'self_price_floor_deferred', 'scaled_shots', 'dispatch_wait'):
            if key in result:
                out[name][key] = result[key]
        out[name]['budget'] = dict(max_new_q3='UNLIMITED_WITHIN_RANGE' if budget is None else budget, science_approved=science_approved,
                                   spent_before=spent, new_calls=result['new_calls'])
        out[name]['generation'] = context.get('generation', {}).get(name)
        out[name]['parameter_assessment'] = parameter_assessment(name, cands, context.get('generation', {}).get(name), out[name])
    if executor is not None and close_executor:
        executor.finish()
    return out


def prepare_kappa_requests(root, problem, context, name, prep, requests):
    """Kappa proposals of the V2 rule -> A's preparation entry (base = the saved configuration of that L_D); complete
    configurations are registered (not ready: signals / route still missing) and quoted with D2 PriceRules MODEL."""
    from . import rpe_candidate_preparation as R
    import tempfile
    points = {}
    for pt in context['catalog']['candidates']:
        points.setdefault(pt['configuration_id'], pt)
    bases = {c['key'].get('L_D'): dict(source=points[c['configuration_id']]['configuration_source'], configuration_id=c['configuration_id'])
             for c in context['candidates'] if c['method'] == name and c['configuration_id'] in points and not c.get('generated')}
    proposals = [r for r in requests if r['kind'] == 'KAPPA_PROPOSAL']
    before = {id(c) for c in context['candidates']}
    if context.get('preparation_dir'):
        rows = R.prepare_kappa(root, name, prep, context['candidates'], proposals, bases, Path(context['preparation_dir']) / name)
    else:
        with tempfile.TemporaryDirectory(prefix='rpe_prep_') as d:
            rows = R.prepare_kappa(root, name, prep, context['candidates'], proposals, bases, Path(d))
    new = [c for c in context['candidates'] if id(c) not in before]
    if new and problem.get('price_rules'):
        quote_prices(root, problem, new, context['catalog'], [name])
    by_item = {(r['proposal']['family']['L_D'], r['proposal']['kappa_multiplier']): r for r in rows}
    for r in requests:
        if r['kind'] == 'KAPPA_PROPOSAL':
            row = by_item.get((r['family']['L_D'], r['kappa_multiplier'])) or {}
            c = next((x for x in context['candidates'] if row.get('configuration_id') and x['configuration_id'] == row['configuration_id']), None)
            r['preparation'] = dict(disposition=row.get('disposition'), status=row.get('status'), configuration_id=row.get('configuration_id'),
                                    signal_counts=row.get('signal_counts'), next_actions=row.get('next_actions'),
                                    needs=c and c['needs'], price_status=c and c.get('price_status'))
    return requests


EXHAUSTED = ('NO_IMPROVING_PROPOSAL', 'NO_FEASIBLE_POINT_IN_RANGE')
STOP_PRIORITY = ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD', 'JOB_IN_PROGRESS', 'Q3_BUDGET_EXHAUSTED',
                 'NEW_SCIENCE_NOT_APPROVED', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', 'NEW_Q3_REQUIRED_NO_EXECUTOR', 'STEP_LIMIT')


def run_by_configuration(name, candidates, knowledge, backend, search, threshold, generate, max_new, routes=None, spent=0,
                         free=None, science_approved=True, prefetch=None):
    """The same rule on every ready configuration separately (incumbent scope CONFIGURATION).

    Each configuration descends from its own feasible point; a configuration is never cut because another one is
    cheaper.  With an unlimited approved budget the configurations run concurrently (independent searches, each with
    the rule's own lookahead); with a finite budget they run one after another sharing that budget.
    """
    if search.get('seed_saved_same_L_D'):
        sibling_seeds(candidates, knowledge)
    ready = [c for c in candidates if c['ready']];missing = [c['name'] for c in candidates if not c['ready']]
    unlimited = science_approved and max_new is None

    def one(c, used):
        kn = {k: v for k, v in knowledge.items() if k[0] == c['configuration_id']}
        return c, kn, run_method(name, [c], kn, backend, search, threshold, generate, max_new, routes, used, free, science_approved,
                                 prefetch if unlimited else None)
    if unlimited and len(ready) > 1:
        with ThreadPoolExecutor(len(ready)) as pool:
            parts = list(pool.map(lambda c: one(c, spent), ready))
    else:
        parts, used = [], spent
        for c in ready:
            parts.append(one(c, used));used += parts[-1][2]['new_calls']
    trace, held, by, new_calls, pending = [], [], [], 0, None
    for c, kn, r in parts:
        knowledge.update(kn)
        trace += [dict(t, configuration_step=t['step']) for t in r['trace']];held += r['held'];new_calls += r['new_calls']
        pending = pending or r['pending_request']
        inc = r['incumbent']
        by.append(dict(candidate=c['name'], configuration_id=c['configuration_id'], key=c['key'], stop_reason=r['stop_reason'],
                       steps=len(r['trace']), new_calls=r['new_calls'], pending_request=r['pending_request'],
                       best=None if inc is None else dict(vector=list(inc['vector']), cost=inc['cost'], cost_display=show(inc['cost']),
                                                          status=inc['record']['status'], MSE_lower=inc['record'].get('MSE_lower'),
                                                          MSE_upper=inc['record'].get('MSE_upper'), source=inc['record'].get('source'),
                                                          job=inc['record'].get('job'), label=inc['record'].get('label'))))
    stops = [b['stop_reason'] for b in by]
    if all(x in EXHAUSTED for x in stops):
        stop = 'NO_IMPROVING_PROPOSAL_INPUTS_MISSING' if missing else 'NO_IMPROVING_PROPOSAL'
    else:
        stop = next(x for x in STOP_PRIORITY + tuple(stops) if x in stops)
    _, incumbent = method_state(ready, knowledge, threshold)
    return dict(method=name, stop_reason=stop, trace=trace, held=held, new_calls=new_calls, spent_before=spent, pending_request=pending,
                incumbent=incumbent, not_ready_in_range=missing, by_configuration=by, incumbent_scope='CONFIGURATION')


def prepare(root, problem, generate_configurations=True, methods=None, preparation_dir=None, *, saved_catalog=None):
    facts = hamiltonian_facts(problem)
    if saved_catalog is None:
        reader, inputs, catalog = load_catalog(root, problem)
    else:
        # Same invocation, same committed catalog input. Consumers may
        # annotate their own candidate rows without touching the snapshot.
        _, inputs, raw = saved_catalog
        reader, catalog = Reader(Path(root)), deepcopy(raw)
    check_comparison(problem, facts, inputs['comparison'])
    candidates = enumerate_candidates(reader, problem, facts, catalog)
    from . import pf_optimizer_connection as PF
    PF.enumerate_prefixes(root, reader, problem, candidates, methods)
    generation = generate_variants(root, problem, reader, catalog, candidates, methods, preparation_dir) if generate_configurations else {}
    scope = scope_absolute_kappa(dict(candidates=candidates, catalog=catalog), problem)
    candidates = scope['candidates']
    price_rules = quote_prices(root, problem, candidates, catalog, methods) if problem.get('price_rules') else None
    artificial_pf = problem.get('artificial_pf_evidence') is True
    pf_registry = PF.load_registry(root, reader, problem.get('pf_evidence_registry'),allow_artificial=artificial_pf)
    pf_expected = PF.hydrate(root, reader, problem, candidates, pf_registry, facts,allow_artificial=artificial_pf)
    if problem.get('search', {}).get('exploratory_cost') is True:
        for c in candidates:
            if c['configuration_id'] and c['signal_source'] and not science_needs(c['needs']):
                c['ready'] = True
    prices = {c['configuration_id']: c['prices'] for c in candidates if c['prices'] is not None}
    saved = SavedBackend(catalog['candidates'], prices)
    expected = {}
    for point in catalog['candidates']:
        expected.setdefault(point['configuration_id'], dict(total_configuration_id=point.get('total_configuration_id'),
                                                            binding_id=point.get('binding_id'), signal_source=point.get('signal_source')))
    for cid, entry in pf_expected.items():
        if cid in expected:
            for key in ('total_configuration_id', 'binding_id'):
                if expected[cid].get(key) is not None and entry.get(key) is not None and expected[cid][key] != entry[key]:
                    raise ValueError('PF registry conflicts with saved Q3 binding: ' + key)
        expected[cid] = dict(expected.get(cid) or {}, **{k: v for k, v in entry.items() if v is not None})
    return dict(facts=facts, reader=reader, catalog=catalog, candidates=candidates, saved=saved, tau=tau(problem),
                expected=expected, generate=pinned_function(reader, problem['generator']), generation=generation,
                price_rules=price_rules, preparation_dir=preparation_dir, pf_evidence_registry=pf_registry,
                scope_excluded=scope.get('scope_excluded', []))
