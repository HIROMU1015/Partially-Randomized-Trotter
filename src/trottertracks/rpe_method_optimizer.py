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
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .baseline_signal_pipeline import blob
from .saved_rpe_cost_catalog import Catalog, Reader, display, frac

PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
AXES = ('X', 'Y')
RULE = 'FINITE_LOCAL_DESCENT_V1'
UNIT_KEYS = ('C_ref_exact', 'C_ref', 'C', 'unit_price_RZ')


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
    facts = dict(fragment_count=fragments, rank=receipt['rank'], n_qubits=int(math.log2(states)),
                 orbital_dimension=orbitals, sector_dimension=arrays['sector_state_vector']['shape'][0],
                 hamiltonian_hash=receipt['hamiltonian_hash'], state_hash=receipt['state_hash'],
                 snapshot_sha256=receipt['sha256'])
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
    inputs = reader.document(source['commit'], source['path'])
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


def configuration_prices(reader, points):
    """Own 26-axis unit prices of one configuration, checked against every saved total."""
    priced = [p for p in points if len(p['price_axes']) == 26 and p['conditional_cost'] is not None]
    if not priced:
        return None, 'NO_SAVED_26_AXIS_PRICE'
    units = {}
    for p in priced:
        for axis in p['price_axes']:
            unit = resolve_unit(reader, axis);key = (axis['m'], axis['axis'])
            if units.setdefault(key, unit) != unit:
                return None, 'CONFLICTING_SAVED_UNIT_PRICES'
    if sorted(units) != [(m, a) for m in range(13) for a in AXES]:
        return None, 'INCOMPLETE_26_AXIS_PRICE'
    prices = [units[(m, a)] for m in range(13) for a in AXES]
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


# ---------------------------------------------------------------- candidate space
def delta_ok(value, bounds):
    lo, hi = (Fraction(b) for b in bounds)
    return lo <= Fraction(value) <= hi


def enumerate_candidates(reader, problem, facts, catalog):
    points = [p for p in catalog['candidates']]
    by_config = {}
    for p in points:
        by_config.setdefault(p['configuration_id'], []).append(p)
    index = reader.document(problem['evidence']['pf_candidate_index']['commit'], problem['evidence']['pf_candidate_index']['path'])
    out = []
    for name, spec in problem['methods'].items():
        caps = spec['caps']
        allowed = spec.get('configuration_ids')  # optional allow-list, applied to every candidate source
        if spec['family'] == 'PF':
            prefix = facts['fragment_count'] if name == 'FULL_PF' else 7
            def inside(formula, delta, p=prefix):
                return (formula in spec['formulas'] and delta is not None and delta_ok(delta, spec['delta_over_h_range'])
                        and (spec.get('p_values') is None or p in spec['p_values']))
            indexed = [r for r in index['candidates'] if r['method'] == spec['index_method']]
            seen = {r['configuration_id'] for r in indexed}
            for r in indexed:
                if inside(r['formula'], r['requested_delta_over_h']):
                    out.append(dict(method=name, family='PF', key=dict(p=prefix, formula=r['formula'], delta_over_h=r['requested_delta_over_h']),
                                    configuration_id=r['configuration_id'], index_signal=r['source_bound_13round_packet'], caps=caps))
            for cid, ps in by_config.items():
                if ps[0]['method'] == name and cid not in seen:
                    # Catalog-only configurations get the same formula/width range test (width = delta_pf / h).
                    delta = None if ps[0].get('delta_pf') is None else str(frac(ps[0]['delta_pf']) / frac(ps[0]['h']))
                    catalog_prefix = (ps[0].get('L_D') if name != 'FULL_PF' and ps[0].get('L_D') is not None else prefix)
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
                if not configs and not spec.get('saved_configurations_only'):
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=None, index_signal=None, caps=caps))
                for cid in configs:
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=cid, index_signal=None, caps=caps))
    out = [c for c in out if problem['methods'][c['method']].get('configuration_ids') is None
           or c['configuration_id'] in problem['methods'][c['method']]['configuration_ids']]
    for c in out:
        ps = by_config.get(c['configuration_id'], [])
        signals = [p['signal_source'] for p in ps if p.get('signal_source')] or ([c['index_signal']] if c['index_signal'] else [])
        prices, price_status = configuration_prices(reader, ps) if ps else (None, 'NO_SAVED_26_AXIS_PRICE')
        labels = sorted({p['label'] for p in ps})
        c.update(labels=labels, signal_source=signals[0] if signals else None, prices=prices, price_status=price_status,
                 saved_points=[p['point_id'] for p in ps])
        needs = []
        if c['configuration_id'] is None:
            needs.append('CONFIGURATION_AND_SCHEDULE')
        if c['signal_source'] is None:
            needs.append('SIGNAL_13ROUND_PACKET')
        if prices is None:
            needs.append('PRICE_26AXIS')
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
        if not c['ready']:
            continue
        cid, prices = c['configuration_id'], c['prices']
        caps = tuple(n for n in c['caps'] for _ in AXES)
        raw = []
        if cid in best:
            base = best[cid];h = headroom(base['record'], threshold)
            steps = search['decrement_steps'] if h is not None and h < large else [min(search['decrement_steps'])]
            for m in range(13):
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


def route_supports_configuration(rule, cid, reader):
    """A wildcard declaration cannot make an absent saved case executable.

    Only inspect published metadata; the caller's cached Reader avoids reading
    a template once per allocation. Numerical/intake guards still run in the worker.
    """
    allowed = rule.get('configuration_ids')
    if allowed not in (None, '*') and cid not in allowed:
        return False
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
    ready = [c for c in candidates if c['ready']];missing = [c['name'] for c in candidates if not c['ready']]
    limit = 0 if not science_approved else (float('inf') if max_new is None else max_new)
    large_headroom = Fraction(search['large_step_headroom'])
    for step in range(search['max_steps_per_method']):
        props, incumbent = proposals(ready, knowledge, threshold, search, generate)
        if not props:
            # Exhausted within the approved finite range; distinguish candidates that still lack inputs.
            stop = 'NO_IMPROVING_PROPOSAL_INPUTS_MISSING' if missing else 'NO_IMPROVING_PROPOSAL'
            if ready and incumbent is None:
                stop = 'NO_FEASIBLE_POINT_IN_RANGE'
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
        hold = None if answer is None else HOLD.get(answer['status'])
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


def job_outcome(job):
    """Accept separately collected evidence without overwriting the old STOP."""
    original=Path(job)/'OUTCOME.json';path=Path(job)/'COLLECTED_OUTCOME.json'
    old=json.loads(original.read_text())
    if path.exists() and old.get('status')=='UNKNOWN':
        new=json.loads(path.read_text())
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
        self.lock, self.slots = threading.Lock(), threading.BoundedSemaphore(max(1, max_concurrent))
        self.events, self.waiters, self.uncounted = {}, [], set()
        self.wait_for_running, self.poll_seconds = wait_for_running, poll_seconds
        # Completed jobs of earlier rounds (same job key, same route for the configuration) answer without a new Q3.
        self.prior_run_dirs, self.launch_disabled = [Path(d) for d in prior_run_dirs], launch_disabled
        self.pf_template_reader, self.pf_template_cache = Reader(self.root), {}

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

    def route(self, cid):
        for rule in self.routes.get(self.method, []):
            if rule.get('configuration_ids') in (None, '*') or cid in rule['configuration_ids']:
                return rule
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
                    cost=enc(proposal['cost']), origin=proposal['origin'], route=rule,
                    expected=self.expected_for(proposal['configuration_id'], rule), optimizer_source=self.optimizer_source)

    def run_adapter(self, job, rule, mode):
        cmd = [rule['interpreter'], '-B', str(self.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (job / (mode.upper() + '.log')).open('a') as log:  # the child keeps its own descriptor
            proc = subprocess.Popen(cmd, cwd=str(self.root), env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc

    def pf_template_source(self, rule, cid):
        """Effective first PF template SOURCE; unrelated additions are harmless."""
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

    def same_route(self, job, cid):
        """A prior job is reusable only if it ran the same route (kind, worker view, contract, signal packet, binding)."""
        try:
            old = json.loads((job / 'REQUEST.json').read_text())
            now = self.route(cid) or {};before=old.get('route')or{}
            pf=before.get('kind')==now.get('kind')=='pf_b_pf6'
            omitted=('configurations','configuration_ids','existing_caller','configurations_file')+(('templates',)if pf else())
            fixed=lambda r:{k:v for k,v in r.items()if k not in omitted}
            entry=lambda r:(r.get('configurations')or{}).get(cid)
            if not (old.get('configuration_id')==cid and fixed(before)==fixed(now) and entry(before)==entry(now)
                    and old.get('expected')==json.loads(json.dumps(self.expected_for(cid,now),default=str))):
                return False
            return not pf or self.pf_template_source(before,cid)==self.pf_template_source(now,cid)
        except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError):
            return False

    def prior_job(self, proposal):
        name = self.job_dir(proposal).name
        # A same-key prior LAUNCH remains owned there, even without an outcome.
        for marker in ('OUTCOME.json', 'LAUNCH.json'):
            for base in self.prior_run_dirs:
                job = base / 'jobs' / self.method / name
                if (job / marker).exists():
                    return job
        return None

    def has_job(self, proposal):
        job = self.job_dir(proposal)
        return (job / 'OUTCOME.json').exists() or (job / 'LAUNCH.json').exists() or self.prior_job(proposal) is not None

    def prior_records(self, candidates):
        """Every completed prior-round job of this method for a ready candidate, validated like a new job (same key,
        same route, identity checks).  Returns ({(cid, vector): record}, refused rows)."""
        cache = getattr(self, 'preparation_history_cache', None)
        if cache is not None:
            return cache.prior_records(self, candidates)[:2]
        ready = {c['configuration_id']: c for c in candidates if c['ready'] and c['configuration_id']}
        known, refused = {}, []
        for base in self.prior_run_dirs:
            for job in sorted((base / 'jobs' / self.method).glob('*/OUTCOME.json')):
                job = job.parent
                try:
                    q = json.loads((job / 'REQUEST.json').read_text())
                except (OSError, ValueError):
                    refused.append(dict(job=str(job), reason='NO_REQUEST'));continue
                c = ready.get(q.get('configuration_id'))
                if c is None:
                    refused.append(dict(job=str(job), reason='CONFIGURATION_NOT_IN_THIS_ROUND'));continue
                proposal = dict(configuration_id=c['configuration_id'], vector=interleave(q['N_X'], q['N_Y']))
                if self.job_dir(proposal).name != job.name or not self.same_route(job, c['configuration_id']):
                    refused.append(dict(job=str(job), reason='DIFFERENT_JOB_KEY_OR_ROUTE'));continue
                r = self.ingest(job, proposal, calls=0)
                if r['status'] not in (PASS, FAIL, 'INCONCLUSIVE'):
                    refused.append(dict(job=str(job), reason=r['status'] + ':' + str(r.get('reason'))));continue
                known[(c['configuration_id'], proposal['vector'])] = dict(r, source='PRIOR_ROUND_JOB', cost=cost(proposal['vector'], c['prices']))
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
                if not self.same_route(prior,proposal['configuration_id']):
                    return dict(status='STALE_JOB',job=str(prior),calls=0,source='PRIOR_ROUND_JOB',owner_review_required=True,
                                reason='PRIOR_OWNER_ROUTE_OR_SOURCE_UNRESOLVED_REVIEW_REQUIRED')
                if (prior / 'OUTCOME.json').exists():
                    return dict(self.ingest(prior, proposal, calls=0), source='PRIOR_ROUND_JOB')
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
                return dict(status='UNKNOWN', reason='ADAPTER_EXIT_WITHOUT_OUTCOME', MSE_lower=None, MSE_upper=None,
                            source='NEW_Q3_JOB', calls=self.take_count(job), job=str(job))
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

    def ingest(self, job, proposal, calls):
        """Accept a job outcome only for the same configuration, allocation, total view/binding/signal and contract."""
        out, outcome_path = job_outcome(job);ident = out.get('identity') or {}
        nx, ny = split(proposal['vector']);want = self.expected_for(proposal['configuration_id'], self.route(proposal['configuration_id']))
        status, reason = out['status'], out.get('reason')
        if status in (PASS, FAIL, 'INCONCLUSIVE'):
            checks = dict(configuration=ident.get('configuration_id') == proposal['configuration_id'],
                          allocation=ident.get('allocation') == [dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(nx, ny))],
                          contract=ident.get('contract_matches') is True)
            for key in ('total_configuration_id', 'binding_id'):
                if want.get(key) is not None:
                    checks[key] = ident.get(key) == want[key]
            if not all(checks.values()):
                status, reason = 'UNKNOWN', 'IDENTITY_MISMATCH:' + ','.join(k for k, v in checks.items() if not v)
        source = 'ARTIFICIAL_JOB' if out.get('artificial') else 'NEW_Q3_JOB'
        lower = None if out.get('MSE_lower') is None else frac(out['MSE_lower'])
        upper = None if out.get('MSE_upper') is None else frac(out['MSE_upper'])
        return dict(status=status, reason=reason, MSE_lower=lower, MSE_upper=upper, source=source, calls=calls,
                    job=str(job.relative_to(self.root)) if job.is_relative_to(self.root) else str(job), label=None)


def apply_published(candidates, routes):
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
        c['ready'] = not c['needs'] and c['prices'] is not None
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
                    if len(missing_signal) < 13:
                        needs.append('SIGNAL_PACKET_ASSEMBLY_FROM_REUSED_ROUNDS')
                    if missing_price:
                        needs.append('PRICE_AXES:%d_of_26' % len(missing_price))
                    needs.append('Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION')
                    prices = None if missing_price else [price_keys[(k, a)] for k in keys for a in AXES]
                    cand = dict(method=name, family='PR', key=dict(L_D=base['key']['L_D'], formula=item['variant']['formula'],
                                delta_over_h=item['variant']['delta_over_h'], kappa_multiplier=item['variant']['kappa_multiplier']),
                                configuration_id=gen['configuration_id'], index_signal=None, caps=base['caps'], generated=True,
                                generated_from=base['configuration_id'], labels=[], signal_source=None, prices=prices,
                                price_status='REUSED_FOR_ALL_AXES' if prices else 'MISSING_FOR_%d_AXES' % len(missing_price),
                                saved_points=[], needs=needs, ready=False, generated_configuration=gen['configuration'], schedule_keys=keys,
                                kappa_value=str(dec_kappa(gen['configuration'])) if (gen['configuration'].get('spec') or {}).get('kappa') else None,
                                reusable=dict(signal_rounds=[m for m in range(13) if m not in missing_signal],
                                              price_axes=26 - len(missing_price)))
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
            items.append(dict(item_id=c['name'], registry=registry[base] if base in registry else entry_name,
                              configuration=c.get('generated_configuration') if c.get('generated') else None))
        req, out = Path(d) / 'REQ.json', Path(d) / 'OUT.json'
        req.write_text(json.dumps(dict(policy=policy, items=items)))
        run_quotes(root, pr, req, out)
        result = json.loads(out.read_text())
    by_name = {c['name']: c for c in targets};report = []
    for q in result['items']:
        c = by_name[q['item_id']];rows = q.get('rows') or []
        quoted = [frac(r['unit_price_RZ']) for r in rows] if q['complete'] else None
        counts = {}
        for r in rows:
            counts[r['classification']] = counts.get(r['classification'], 0) + 1
        missing = [dict(m=r['m'], axis=r['axis'], reasons=(r.get('needs') or {}).get('reasons'),
                        calibration=(r.get('needs') or {}).get('calibration') or (r.get('needs') or {}).get('missing_quantity'))
                   for r in rows if r['unit_price_RZ'] is None]
        c['price_quote'] = dict(policy=q.get('policy', policy), status=q['status'], complete=q['complete'], registry=q['registry'],
                                classifications=counts, extrapolated_axes=sum(1 for r in rows if r.get('extrapolation')),
                                reason=q.get('reason'), implementation_source=pr['implementation_source'])
        entry = dict(candidate=c['name'], configuration_id=c['configuration_id'], generated=bool(c.get('generated')),
                     registry=q['registry'], status=q['status'], complete=q['complete'], classifications=counts)
        if not c.get('generated'):
            entry['reproduces_saved_unit_prices'] = None if c['prices'] is None else (quoted == list(c['prices']))
            if c['prices'] is None and q['complete']:
                entry['note'] = 'saved catalog has no price for this configuration; quote not adopted for saved points'
        else:
            c['needs'] = [n for n in c['needs'] if not n.startswith('PRICE_AXES')]
            if q['complete']:
                c['prices'] = quoted;c['price_status'] = 'PRICE_RULES_%s_COMPLETE' % policy
            else:
                c['prices'] = None;c['price_status'] = 'PRICE_RULES_%s_INCOMPLETE' % policy
                c['needs'].append('PRICE_AXES:%d_of_26' % len(missing))
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
    unevaluated = [dict(candidate=c['name'], configuration=c['key'], configuration_id=c['configuration_id'], needs=c['needs'],
                        price_status=c['price_status'], price_quote=c.get('price_quote'), price_needs=c.get('price_needs'),
                        signal_source=c['signal_source'], reusable=c.get('reusable'),
                        next='Q3 of a generated/decremented allocation after the listed items exist')
                   for c in candidates if not c['ready']]
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
            if c['ready']:
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


def optimize(root, problem, *, context=None, methods=None, initial='all', extra_results=(), backend=None, max_new=None,
             executor=None, saved_is_free=True, science_approved=True):
    """Run the per-method search.

    initial: 'all' saved results, 'none', or {method: [saved labels]} (replay from seeds).
    backend: Q3 answer source; default = saved results (C's catalog).  With ``executor`` (an approved round),
    saved results are tried first, then completed/new jobs of that round.  ``max_new`` is an int or a
    {method: int} budget of new Q3 calls; calls already launched in the round count against it.
    saved_is_free: the backend answers C's saved configuration+allocation without a new call (set False for a
    backend that does not consult the saved results, e.g. a stand-alone artificial fixture).
    """
    context = context or prepare(root, problem)
    search = problem['search'];threshold = context['tau']
    max_new = search['max_new_q3_per_method'] if max_new is None else max_new
    out = {}
    for name in methods or problem['methods']:
        cands = [c for c in context['candidates'] if c['method'] == name]
        start = initial if isinstance(initial, str) else initial.get(name, [])
        knowledge = ingest(initial_knowledge(context['saved'], cands, start), cands, extra_results)
        budget = max_new.get(name, 0) if isinstance(max_new, dict) else max_new
        budget = None if budget in (None, 'UNLIMITED') else budget
        table = context['saved'].table if saved_is_free else {}
        prefetch = None
        if executor is not None:
            ex = executor.for_method(name)
            answer, spent = ChainBackend(context['saved'], ex), ex.launched(name)
            free = lambda p, ex=ex: (p['configuration_id'], p['vector']) in table or ex.has_job(p)
            if getattr(ex, 'supports_nonblocking_prefetch', False) and search.get('rule') == 'METHOD_MINIMUM_PRIORITY_V2':
                prefetch = lambda ps, ex=ex: ex.prefetch([p for p in ps if (p['configuration_id'], p['vector']) not in table], scope='METHOD')
            else:
                prefetch = lambda ps, ex=ex: ex.prefetch([p for p in ps if (p['configuration_id'], p['vector']) not in table])
        else:
            answer, spent = (context['saved'] if backend is None else backend), 0
            free = lambda p: (p['configuration_id'], p['vector']) in table
        if executor is not None and executor.prior_run_dirs:
            ingest_start = time.perf_counter()
            executor.for_method(name).log(event='PRIOR_INGEST_START')
            known, refused = executor.for_method(name).prior_records(cands)
            executor.for_method(name).log(event='PRIOR_INGEST_END', seconds=time.perf_counter()-ingest_start,
                                          reused=len(known), refused=len(refused))
            fresh = {k: v for k, v in known.items() if k not in knowledge}
            knowledge.update(fresh)
            prior = dict(run_dirs=[str(d) for d in executor.prior_run_dirs], loaded=len(fresh), already_known=len(known) - len(fresh),
                         refused=len(refused), refused_reasons=sorted({r['reason'] for r in refused}))
        else:
            prior = None
        if search.get('rule') == 'METHOD_MINIMUM_PRIORITY_V2':
            from trottertracks import rpe_priority_search as P
            log = None
            if executor is not None:
                ex_log = executor.for_method(name)
                log = lambda row, ex_log=ex_log: ex_log.log(event='SEARCH_ADOPT' if getattr(ex_log, 'supports_nonblocking_prefetch', False) else 'ADOPT', configuration=row['configuration_id'][:8], job=row.get('job'), cls=row['cls'],
                                                           status=row['status'], cost=show(row['cost']),
                                                           method_min=row['method_min_after'] and show(row['method_min_after']))
            result = P.run_priority(name, cands, knowledge, answer, search, threshold, context['generate'], budget,
                                    baseline=(problem['methods'][name].get('generation') or {}).get('baseline'),
                                    routes=problem.get('q3_routes'), spent=spent, free=free, science_approved=science_approved,
                                    prefetch=prefetch, on_missing=search.get('on_missing', 'STOP'), facts=context.get('facts'), log=log,
                                    prefetch_state=getattr(ex, 'prefetch_state', None) if executor is not None else None)
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
        out[name]['prior_rounds'] = prior
        for key in ('unevaluated', 'input_requests', 'left', 'settings', 'prefetch_policy', 'selection_timing'):
            if key in result:
                out[name][key] = result[key]
        out[name]['budget'] = dict(max_new_q3='UNLIMITED_WITHIN_RANGE' if budget is None else budget, science_approved=science_approved,
                                   spent_before=spent, new_calls=result['new_calls'])
        out[name]['generation'] = context.get('generation', {}).get(name)
        out[name]['parameter_assessment'] = parameter_assessment(name, cands, context.get('generation', {}).get(name), out[name])
    if executor is not None:
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


def prepare(root, problem, generate_configurations=True, methods=None, preparation_dir=None):
    facts = hamiltonian_facts(problem)
    reader, inputs, catalog = load_catalog(root, problem)
    check_comparison(problem, facts, inputs['comparison'])
    candidates = enumerate_candidates(reader, problem, facts, catalog)
    from . import pf_optimizer_connection as PF
    PF.enumerate_prefixes(root, reader, problem, candidates, methods)
    generation = generate_variants(root, problem, reader, catalog, candidates, methods, preparation_dir) if generate_configurations else {}
    price_rules = quote_prices(root, problem, candidates, catalog, methods) if problem.get('price_rules') else None
    artificial_pf = problem.get('artificial_pf_evidence') is True
    pf_registry = PF.load_registry(root, reader, problem.get('pf_evidence_registry'),allow_artificial=artificial_pf)
    pf_expected = PF.hydrate(root, reader, problem, candidates, pf_registry, facts,allow_artificial=artificial_pf)
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
                price_rules=price_rules, preparation_dir=preparation_dir, pf_evidence_registry=pf_registry)
