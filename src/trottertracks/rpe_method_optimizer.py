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
import time

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
            def inside(formula, delta):
                return (formula in spec['formulas'] and delta is not None and delta_ok(delta, spec['delta_over_h_range']))
            indexed = [r for r in index['candidates'] if r['method'] == spec['index_method']]
            seen = {r['configuration_id'] for r in indexed}
            for r in indexed:
                if inside(r['formula'], r['requested_delta_over_h']):
                    out.append(dict(method=name, family='PF', key=dict(formula=r['formula'], delta_over_h=r['requested_delta_over_h']),
                                    configuration_id=r['configuration_id'], index_signal=r['source_bound_13round_packet'], caps=caps))
            for cid, ps in by_config.items():
                if ps[0]['method'] == name and cid not in seen:
                    # Catalog-only configurations get the same formula/width range test (width = delta_pf / h).
                    delta = None if ps[0].get('delta_pf') is None else str(frac(ps[0]['delta_pf']) / frac(ps[0]['h']))
                    if inside(ps[0]['formula'], delta):
                        out.append(dict(method=name, family='PF', key=dict(formula=ps[0]['formula'], delta_over_h=delta),
                                        configuration_id=cid, index_signal=None, caps=caps, outside_index=True))
        else:
            lo = spec['L_D_range']['min'];hi = facts['fragment_count'] + spec['L_D_range']['max_offset_from_fragment_count']
            values = spec.get('L_D_values')
            for ld in range(lo, hi + 1):
                if values is not None and ld not in values:
                    continue
                configs = sorted({cid for cid, ps in by_config.items() if ps[0]['method'] == name and ps[0]['L_D'] == ld})
                if not configs:
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


HOLD = {'IN_PROGRESS': 'JOB_IN_PROGRESS', 'NO_ROUTE': 'NO_ROUTE_ADAPTER', 'STALE_JOB': 'STALE_JOB_NEEDS_REVIEW'}


def run_method(name, candidates, knowledge, backend, search, threshold, generate, max_new, routes=None, spent=0, free=None,
               science_approved=True):
    """Finite loop: select, ask backend, update knowledge, repeat until a stop rule fires.

    The budget is checked BEFORE the backend is called: a proposal is sent to the backend only if it can be
    answered without a new Q3 (``free``: same configuration+allocation already saved or already a job of this
    round) or if new-call budget remains.  Otherwise it is returned as the unexecuted next request.
    ``spent`` counts new calls already launched for this method in the same approved round (resume).
    UNKNOWN (gate/persistence) stops the method; a model failure (lower bound > tau) does not.
    """
    trace, held, new_calls, pending, stop = [], [], spent, None, None
    ready = [c for c in candidates if c['ready']];missing = [c['name'] for c in candidates if not c['ready']]
    limit = 0 if not science_approved else (float('inf') if max_new is None else max_new)
    for step in range(search['max_steps_per_method']):
        props, incumbent = proposals(ready, knowledge, threshold, search, generate)
        if not props:
            # Exhausted within the approved finite range; distinguish candidates that still lack inputs.
            stop = 'NO_IMPROVING_PROPOSAL_INPUTS_MISSING' if missing else 'NO_IMPROVING_PROPOSAL';break
        choice = props[0];allow = new_calls < limit
        is_free = free is not None and free(choice)
        answer = backend.answer(choice) if backend is not None and (is_free or allow) else None
        hold = None if answer is None else HOLD.get(answer['status'])
        if answer is None or hold:
            pending = dict(choice, route=(routes or {}).get(name), queue=[dict(candidate=p['candidate'], cost=p['cost'],
                vector=list(p['vector']), origin=p['origin']) for p in props[:5]], job=None if answer is None else answer.get('job'))
            if hold:
                stop = hold
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


class Executor:
    """Launch the method's existing Q3 route for one request at a time; resume from the round directory.

    Job directory jobs/<method>/<job_id>/ holds REQUEST.json, LAUNCH.json (pid), PROCESS_RETURN.json and the
    adapter's OUTCOME.json plus all worker artifacts.  A job is launched only when it has no OUTCOME, no live
    process and no previous launch; a dead launch without OUTCOME is first re-validated in collect mode (no
    science) and otherwise reported for review (never retried automatically).
    """

    def __init__(self, root, run_dir, routes, expected, problem, adapter='scripts/run_rpe_optimizer_q3_job.py', optimizer_source=None):
        self.root, self.run_dir, self.routes, self.expected = Path(root), Path(run_dir), routes, expected
        self.adapter, self.optimizer_source, self.contract = Path(root) / adapter, optimizer_source, contract_digest(problem)
        self.method = None

    def for_method(self, method):
        self.method = method
        return self

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

    def request(self, proposal, job, rule):
        nx, ny = split(proposal['vector'])
        expected = dict(self.expected.get(proposal['configuration_id'], {}), configuration_id=proposal['configuration_id'])
        return dict(request_id=job.name, method=self.method, configuration_id=proposal['configuration_id'], N_X=nx, N_Y=ny,
                    cost=enc(proposal['cost']), origin=proposal['origin'], route=rule, expected=expected,
                    optimizer_source=self.optimizer_source)

    def run_adapter(self, job, rule, mode):
        cmd = [rule['interpreter'], '-B', str(self.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (job / (mode.upper() + '.log')).open('a') as log:  # the child keeps its own descriptor
            proc = subprocess.Popen(cmd, cwd=str(self.root), env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc

    def has_job(self, proposal):
        job = self.job_dir(proposal)
        return (job / 'OUTCOME.json').exists() or (job / 'LAUNCH.json').exists()

    def answer(self, proposal, allow_new=True):
        rule = self.route(proposal['configuration_id']);job = self.job_dir(proposal)
        if (job / 'OUTCOME.json').exists():
            return self.ingest(job, proposal, calls=0)
        if (job / 'LAUNCH.json').exists():
            launch = json.loads((job / 'LAUNCH.json').read_text())
            if not (job / 'PROCESS_RETURN.json').exists() and self.alive(launch, job):
                return dict(status='IN_PROGRESS', job=str(job))
            if rule is not None:
                self.run_adapter(job, rule, 'collect').wait()
            if (job / 'OUTCOME.json').exists():
                return self.ingest(job, proposal, calls=0)
            return dict(status='STALE_JOB', job=str(job))
        if not allow_new:
            return None
        if rule is None:
            return dict(status='NO_ROUTE', job=None)
        job.mkdir(parents=True)
        (job / 'REQUEST.json').write_text(json.dumps(self.request(proposal, job, rule), ensure_ascii=False, indent=2, default=str) + '\n')
        start = time.time();proc = self.run_adapter(job, rule, 'execute')
        (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=proc.pid, started=start, kind=rule['kind'])) + '\n')
        code = proc.wait()
        (job / 'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=code if code >= 0 else None, signal=-code if code < 0 else None,
                                                                 wall_seconds=time.time() - start, automatic_retry=False)) + '\n')
        if not (job / 'OUTCOME.json').exists():
            return dict(status='UNKNOWN', reason='ADAPTER_EXIT_WITHOUT_OUTCOME', MSE_lower=None, MSE_upper=None,
                        source='NEW_Q3_JOB', calls=1, job=str(job))
        return self.ingest(job, proposal, calls=1)

    def ingest(self, job, proposal, calls):
        """Accept a job outcome only for the same configuration, allocation, total view/binding/signal and contract."""
        out = json.loads((job / 'OUTCOME.json').read_text());ident = out.get('identity') or {}
        nx, ny = split(proposal['vector']);want = self.expected.get(proposal['configuration_id'], {})
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


def variant_grid(spec):
    g = spec['generation']
    for formula in g['formulas']:
        for delta in g['delta_over_h_values']:
            for kappa in g['kappa_multipliers']:
                yield dict(formula=formula, delta_over_h=delta, kappa_multiplier=kappa)


def generate_variants(root, problem, reader, catalog, candidates, methods=None):
    """For each PR method with a ``generation`` range: regenerate every saved configuration over formula x
    step width x kappa with the method's fixed mapper (subprocess, route view, metadata only), then
    deduplicate by physical schedule.  Identical schedules alias the saved configuration (its evaluations are
    reused, never re-evaluated); new schedules become candidates listing exactly which inputs are missing."""
    out = {}
    for name, spec in problem['methods'].items():
        if spec['family'] != 'PR' or not spec.get('generation') or (methods is not None and name not in methods):
            continue
        g = spec['generation'];saved = [c for c in candidates if c['method'] == name and c['configuration_id']]
        points = {}
        for point in catalog['candidates']:
            points.setdefault(point['configuration_id'], point)
        configs = {c['configuration_id']: reader.read(points[c['configuration_id']]['configuration_source']) for c in saved}
        base_keys = {cid: schedule_keys(cfg) for cid, cfg in configs.items()}
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
                                saved_points=[], needs=needs, ready=False,
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
        out[name] = dict(kind=g['kind'], view_commit=g['view_commit'], supported_formulas=result['items'][0].get('supported_formulas') if result['items'] else None,
                         baseline=g['baseline'], range=dict(formulas=g['formulas'], delta_over_h=g['delta_over_h_values'],
                         kappa_multipliers=g['kappa_multipliers']), h='fixed RPE base time (not a search variable)', variants=rows)
    return out


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
                        price_status=c['price_status'], signal_source=c['signal_source'],
                        next='Q3 of a generated/decremented allocation after the listed items exist')
                   for c in candidates if not c['ready']]
    return dict(method=name, best=best, stop_reason=result['stop_reason'], pending_request=result['pending_request'],
                trace=result['trace'], held=result['held'], new_calls=result['new_calls'],
                evaluated_count=len(evaluated), evaluated=evaluated, not_ready_candidates=unevaluated,
                ready_candidates=[c['name'] for c in candidates if c['ready']],
                search_range=dict(candidates=[c['name'] for c in candidates], rule=RULE),
                resume_records=resume_records(evaluated))


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
        return {str(k): jsonable(v) for k, v in value.items() if k not in ('prices', 'candidate') or not isinstance(v, (list, dict))}
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
        if executor is not None:
            ex = executor.for_method(name)
            answer, spent = ChainBackend(context['saved'], ex), ex.launched(name)
            free = lambda p, ex=ex: (p['configuration_id'], p['vector']) in table or ex.has_job(p)
        else:
            answer, spent = (context['saved'] if backend is None else backend), 0
            free = lambda p: (p['configuration_id'], p['vector']) in table
        result = run_method(name, cands, knowledge, answer, search, threshold, context['generate'], budget,
                            problem.get('q3_routes'), spent, free, science_approved)
        out[name] = summarize_method(name, cands, result, knowledge, threshold)
        out[name]['budget'] = dict(max_new_q3='UNLIMITED_WITHIN_RANGE' if budget is None else budget, science_approved=science_approved,
                                   spent_before=spent, new_calls=result['new_calls'])
        out[name]['generation'] = context.get('generation', {}).get(name)
        out[name]['parameter_assessment'] = parameter_assessment(name, cands, context.get('generation', {}).get(name), out[name])
    return out


def prepare(root, problem, generate_configurations=True, methods=None):
    facts = hamiltonian_facts(problem)
    reader, inputs, catalog = load_catalog(root, problem)
    check_comparison(problem, facts, inputs['comparison'])
    candidates = enumerate_candidates(reader, problem, facts, catalog)
    generation = generate_variants(root, problem, reader, catalog, candidates, methods) if generate_configurations else {}
    prices = {c['configuration_id']: c['prices'] for c in candidates if c['prices'] is not None}
    saved = SavedBackend(catalog['candidates'], prices)
    expected = {}
    for point in catalog['candidates']:
        expected.setdefault(point['configuration_id'], dict(total_configuration_id=point.get('total_configuration_id'),
                                                            binding_id=point.get('binding_id'), signal_source=point.get('signal_source')))
    return dict(facts=facts, reader=reader, catalog=catalog, candidates=candidates, saved=saved, tau=tau(problem),
                expected=expected, generate=pinned_function(reader, problem['generator']), generation=generation)
