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
   Proposals strictly cheaper than the method incumbent are ordered by cost;
   those dominated by a known failure are skipped (heuristic, recorded).  Each
   proposal is answered by a Q3 backend (saved results, or an artificial
   fixture in tests); without an answer and without remaining budget, the
   proposal is emitted as a machine-readable request and the method stops.

No signal, price, compile or Q3 is computed here.  Costs are each candidate's own
saved unit prices times the integer shots, validated against C's saved totals.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from .baseline_signal_pipeline import blob
from .saved_rpe_cost_catalog import Catalog, Reader, display, frac

PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
AXES = ('X', 'Y')
RULE = 'FINITE_LOCAL_DESCENT_V1'
UNIT_KEYS = ('C_ref_exact', 'C_ref', 'C')


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

    def answer(self, proposal):
        hit = self.table.get((proposal['configuration_id'], proposal['vector']))
        return None if hit is None else dict(hit, calls=0)


class FixtureBackend:
    """Artificial Q3 for tests/demos only: callable(config_id, vector) -> (lower, upper)."""

    def __init__(self, function, threshold):
        self.function, self.threshold = function, threshold

    def answer(self, proposal):
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
        if spec['family'] == 'PF':
            rows = [r for r in index['candidates'] if r['method'] == spec['index_method'] and r['formula'] in spec['formulas']
                    and delta_ok(r['requested_delta_over_h'], spec['delta_over_h_range'])]
            seen = set()
            for r in rows:
                seen.add(r['configuration_id'])
                out.append(dict(method=name, family='PF', key=dict(formula=r['formula'], delta_over_h=r['requested_delta_over_h']),
                                configuration_id=r['configuration_id'], index_signal=r['source_bound_13round_packet'], caps=caps))
            for cid, ps in by_config.items():
                if ps[0]['method'] == name and cid not in seen:
                    out.append(dict(method=name, family='PF', key=dict(formula=ps[0]['formula'], delta_over_h=None),
                                    configuration_id=cid, index_signal=None, caps=caps, outside_index=True))
        else:
            lo = spec['L_D_range']['min'];hi = facts['fragment_count'] + spec['L_D_range']['max_offset_from_fragment_count']
            for ld in range(lo, hi + 1):
                configs = sorted({cid for cid, ps in by_config.items() if ps[0]['method'] == name and ps[0]['L_D'] == ld})
                if not configs:
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=None, index_signal=None, caps=caps))
                for cid in configs:
                    out.append(dict(method=name, family='PR', key=dict(L_D=ld), configuration_id=cid, index_signal=None, caps=caps))
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
    out, skipped = [], []
    large = Fraction(search['large_step_headroom'])
    for c in candidates:
        if not c['ready']:
            continue
        cid, prices = c['configuration_id'], c['prices']
        caps = tuple(n for n in c['caps'] for _ in AXES)
        fails = [v for (k, v), r in knowledge.items() if k == cid and r['status'] == FAIL]
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
        for v, origin in raw:
            if (cid, v) in knowledge:
                continue
            t = cost(v, prices)
            if limit is not None and t >= limit:
                continue
            item = dict(configuration_id=cid, candidate=c['name'], vector=v, cost=t, origin=origin,
                        shot_change=sum(abs(a - b) for a, b in zip(v, origin.get('base_vector', v))))
            dominated = next((f for f in fails if all(a >= b for a, b in zip(f, v))), None)
            if dominated is not None:
                skipped.append(dict(item, skip='DOMINATED_BY_KNOWN_FAILURE', failure_vector=list(dominated)))
                continue
            out.append(item)
    unique = {}
    for item in out:
        unique.setdefault((item['configuration_id'], item['vector']), item)
    ordered = sorted(unique.values(), key=lambda x: (x['cost'], x['shot_change'], x['candidate'], x['vector']))
    return ordered, skipped, incumbent


def run_method(name, candidates, knowledge, backend, search, threshold, generate, max_new, routes=None):
    """Finite loop: select, ask backend, update knowledge, repeat until a stop rule fires."""
    trace, held, new_calls, pending, stop = [], [], 0, None, None
    ready = [c for c in candidates if c['ready']]
    for step in range(search['max_steps_per_method']):
        props, skipped, incumbent = proposals(ready, knowledge, threshold, search, generate)
        if not props:
            stop = 'NO_IMPROVING_PROPOSAL';break
        choice = props[0];answer = backend.answer(choice) if backend is not None else None
        if answer is None or (answer['calls'] and new_calls >= max_new):
            pending = dict(choice, route=(routes or {}).get(name), queue=[dict(candidate=p['candidate'], cost=p['cost'],
                vector=list(p['vector']), origin=p['origin']) for p in props[:5]])
            if answer is not None:
                stop = 'Q3_BUDGET_EXHAUSTED'
            elif new_calls >= max_new:
                stop = 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'
            else:
                stop = 'NEW_Q3_REQUIRED_NO_EXECUTOR_IN_THIS_VERSION'
            break
        new_calls += answer['calls']
        key = (choice['configuration_id'], choice['vector'])
        knowledge[key] = dict(answer, cost=choice['cost'])
        _, after = method_state(ready, knowledge, threshold)
        trace.append(dict(step=step + 1, candidate=choice['candidate'], vector=list(choice['vector']), cost=choice['cost'],
                          origin=choice['origin'], answer_source=answer['source'], status=answer['status'],
                          MSE_upper=answer['MSE_upper'], calls=answer['calls'],
                          incumbent_before=None if incumbent is None else incumbent['cost'],
                          incumbent_after=None if after is None else after['cost'],
                          improved=after is not None and (incumbent is None or after['cost'] < incumbent['cost']),
                          proposals_considered=len(props), proposals_skipped=len(skipped)))
        if answer['status'] not in (PASS, FAIL):
            held.append(dict(candidate=choice['candidate'], vector=list(choice['vector']), status=answer['status']))
        if answer['status'] == 'UNKNOWN':
            stop = 'UNKNOWN_RESULT';break
    else:
        stop = 'STEP_LIMIT'
    _, incumbent = method_state(ready, knowledge, threshold)
    return dict(method=name, stop_reason=stop, trace=trace, held=held, new_calls=new_calls, pending_request=pending,
                incumbent=incumbent)


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
        if r['configuration_id'] not in ids:
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
    evaluated = [dict(candidate=next(c['name'] for c in candidates if c['configuration_id'] == cid), vector=list(v),
                      status=r['status'], cost=None if r.get('cost') is None else r['cost'], source=r['source'])
                 for (cid, v), r in knowledge.items() if any(c['configuration_id'] == cid for c in candidates)]
    unevaluated = [dict(candidate=c['name'], configuration=c['key'], configuration_id=c['configuration_id'], needs=c['needs'],
                        price_status=c['price_status'], signal_source=c['signal_source'],
                        next='Q3 of a generated/decremented allocation after the listed items exist')
                   for c in candidates if not c['ready']]
    return dict(method=name, best=best, stop_reason=result['stop_reason'], pending_request=result['pending_request'],
                trace=result['trace'], held=result['held'], new_calls=result['new_calls'],
                evaluated_count=len(evaluated), evaluated=evaluated, not_ready_candidates=unevaluated,
                ready_candidates=[c['name'] for c in candidates if c['ready']],
                search_range=dict(candidates=[c['name'] for c in candidates], rule=RULE))


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
        return {str(k): jsonable(v) for k, v in value.items() if k not in ('prices', 'candidate') or not isinstance(v, (list, dict))}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def optimize(root, problem, *, context=None, methods=None, initial='all', extra_results=(), backend=None, max_new=None):
    """Run the per-method search.

    initial: 'all' saved results, 'none', or {method: [saved labels]} (replay from seeds).
    backend: Q3 answer source; default = saved results (C's catalog).  Any answer the
    backend cannot give, beyond ``max_new`` new calls, becomes the pending request.
    """
    context = context or prepare(root, problem)
    search = problem['search'];threshold = context['tau']
    max_new = search['max_new_q3_per_method'] if max_new is None else max_new
    backend = context['saved'] if backend is None else backend
    out = {}
    for name in methods or problem['methods']:
        cands = [c for c in context['candidates'] if c['method'] == name]
        start = initial if isinstance(initial, str) else initial.get(name, [])
        knowledge = ingest(initial_knowledge(context['saved'], cands, start), cands, extra_results)
        result = run_method(name, cands, knowledge, backend, search, threshold, context['generate'], max_new,
                            problem.get('q3_routes'))
        out[name] = summarize_method(name, cands, result, knowledge, threshold)
    return out


def prepare(root, problem):
    facts = hamiltonian_facts(problem)
    reader, inputs, catalog = load_catalog(root, problem)
    check_comparison(problem, facts, inputs['comparison'])
    candidates = enumerate_candidates(reader, problem, facts, catalog)
    prices = {c['configuration_id']: c['prices'] for c in candidates if c['prices'] is not None}
    saved = SavedBackend(catalog['candidates'], prices)
    return dict(facts=facts, reader=reader, catalog=catalog, candidates=candidates, saved=saved, tau=tau(problem),
                generate=pinned_function(reader, problem['generator']))
