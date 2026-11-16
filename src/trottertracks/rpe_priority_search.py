"""METHOD_MINIMUM_PRIORITY_V2: lower each method's cheapest feasible cost early, and keep exploring new L_D / kappa.

One ordered queue per method instead of one exhaustive descent per configuration.  Every pick is classified:

  UPDATE    cost < current method minimum M, not predicted infeasible; ordered by cost (largest possible drop of M first)
  EXPLORE   taken every ``explore_every``-th pick (or when UPDATE is empty): first configurations without a feasible
            point (new kappa / new L_D seeds, incl. allocations transferred from the nearest kappa), then configurations
            inside ``explore_window`` x M (least explored first), then the rest by incumbent cost
  SMALL     an UPDATE whose saving is below ``small_gain_fraction`` of its configuration incumbent (tail improvements)
  DEFERRED  componentwise <= a failed allocation of the same configuration whose MSE upper / tau >= 1 + margin;
            evaluated only when nothing else is left -- never dropped (the bound is not exactly monotone near tau)

Moves are those of FINITE_LOCAL_DESCENT_V1 (decrement d in {1,2} per round/axis group from the configuration
incumbent, d = 1 only when headroom >= large_step_headroom; generator, cap and saved-sibling seeds without a feasible
point) plus TRANSFER seeds: the cheapest feasible allocations of the nearest kappa of the same (L_D, formula, width).
Nothing is ever skipped because another configuration is cheaper; finishing a configuration's small improvements is
not a condition for moving on.

Kappa plan per (L_D, formula, width) from current incumbents: minimum at an edge -> extend geometrically
(kappa*^2 / neighbour); interior minimum -> arithmetic midpoints towards both neighbours (cheaper side first) while the
ratio exceeds ``kappa_min_ratio``.  Proposals without configuration / price / signal / route become input requests
(no Q3 slot); so do L_D values without a configuration.

Answer sources: ``backend.answer(proposal)``.  ``on_missing='SKIP'`` (saved-log replay) records a proposal without
an answer as UNEVALUATED and continues; ``'STOP'`` (rounds) stops with the next request, as V1 does.
"""
from collections import defaultdict
from fractions import Fraction
import math
import time

from trottertracks import rpe_method_optimizer as O

RULE_V2 = 'METHOD_MINIMUM_PRIORITY_V2'
DEFAULTS = dict(explore_every=4, explore_window='3/2', small_gain_fraction='1/10000', deferral_fail_margin='1/10000',
                transfer_count=2, kappa_bounds=['1/4', '8'], kappa_min_ratio='9/8', lookahead=1, adaptive_lookahead=False,
                coarse_gain_fraction=None, lookahead_mode='CHAIN', adaptive_hybrid=False,
                allocation_exchange=None)
CLASS_ORDER = ('UPDATE', 'EXPLORE', 'FINE', 'SMALL', 'DEFERRED')
# FINE (only with coarse_gain_fraction): a non-small saving below coarse_gain_fraction of the configuration's own best.  It is
# taken only when no UPDATE / EXPLORE pick is left in any configuration, so every schedule gets its coarse descent (and new
# schedules their seeds) before one schedule's small shot reductions are explored; FINE picks are never dropped.
# lookahead_mode INDEPENDENT: free slots take the next picks of OTHER configurations (one per configuration, in V2's order)
# instead of chaining speculative picks along one configuration (CHAIN, adaptive_lookahead); HYBRID: the independent heads
# first, then chain speculation for the slots still free.


def settings(search):
    s = dict(DEFAULTS, **{k: v for k, v in search.items() if k in DEFAULTS})
    s.update(explore_window=Fraction(s['explore_window']), small_gain_fraction=Fraction(s['small_gain_fraction']),
             deferral_fail_margin=Fraction(s['deferral_fail_margin']), kappa_min_ratio=Fraction(s['kappa_min_ratio']),
             kappa_bounds=[Fraction(x) for x in s['kappa_bounds']])
    s['coarse_gain_fraction'] = None if s.get('coarse_gain_fraction') is None else Fraction(str(s['coarse_gain_fraction']))
    return s


def family(c, baseline):
    k = c['key']
    return (k.get('L_D'), str(k.get('formula', baseline.get('formula'))), str(k.get('delta_over_h', baseline.get('delta_over_h'))))


def kappa_of(c, baseline):
    return Fraction(str(c['key'].get('kappa_multiplier', baseline.get('kappa_multiplier', '1'))))


def kappa_value(c):
    """kappa itself (spec.kappa.value, exact) when the candidate carries it."""
    v = c.get('kappa_value')
    return None if v is None else Fraction(str(v))


def leq(a, b):
    return all(x <= y for x, y in zip(a, b))


def hybrid_limits(heads, requested, observation=None):
    """Bound independent-first previews using measured completed speculation only.

    A launch is not a failure to adopt while its worker is still running.  With
    missing completion counters the initial depth is retained, and the report
    marks utilization as unmeasured.  Capacity changes never stop live jobs.
    """
    observation = observation or {}
    capacity = max(1, int(observation.get('capacity') or requested))
    total = min(max(0, requested), capacity)
    independent = min(heads, total)
    spare = total - independent
    completed = observation.get('speculative_completed')
    adopted = observation.get('speculative_adopted')
    ratio = None
    if completed is not None and adopted is not None and completed >= 8:
        # One pseudo-success and one pseudo-failure avoid an abrupt permanent
        # zero-depth state after a small unlucky sample.  This is scheduling,
        # not a model of MSE as a function of shots.
        ratio = Fraction(min(int(adopted), int(completed)) + 1, int(completed) + 2)
        spare = min(spare, max(1, math.ceil(spare * ratio))) if spare else 0
    return dict(capacity=capacity, preview_limit=total, independent_heads=independent,
                chain_slots=spare, completed_speculation=completed,
                adopted_speculation=adopted, adoption_fraction=None if ratio is None else str(ratio))


def exchange_moves(base, prices, caps, steps, settings, excluded=()):
    """Cost-decreasing expensive-round -> cheap-round moves, with no MSE prediction.

    Existing nearest-kappa TRANSFER seeds copy whole allocations between
    configurations.  These moves instead change two rounds of one incumbent.
    They remain opt-in, inside the existing shot caps, and need their own Q3.
    """
    if not settings or not settings.get('enabled', False):
        return []
    count = int(settings.get('max_moves', 8))
    adds = [int(x) for x in settings.get('add_steps', [1, 2])]
    if count < 0 or any(x < 1 for x in adds):
        raise ValueError('allocation_exchange needs nonnegative max_moves and positive add_steps')
    original = O.cost(base, prices);excluded = set(excluded)
    out = []
    for remove_m in range(13):
        for give in O.axis_groups(prices, remove_m):
            for d in steps:
                reduced = list(base)
                for axis in give:
                    reduced[2 * remove_m + O.AXES.index(axis)] -= d
                if min(reduced) < 1:
                    continue
                for add_m in range(13):
                    if add_m == remove_m:
                        continue
                    for receive in O.axis_groups(prices, add_m):
                        for a in adds:
                            vector = list(reduced)
                            for axis in receive:
                                vector[2 * add_m + O.AXES.index(axis)] += a
                            vector = tuple(vector)
                            if vector in excluded or any(n > cap for n, cap in zip(vector, caps)) or O.cost(vector, prices) >= original:
                                continue
                            out.append((vector, dict(kind='COST_DECREASING_ROUND_EXCHANGE',
                                remove_m=remove_m, remove_axes=list(give), remove_shots=d,
                                add_m=add_m, add_axes=list(receive), add_shots=a, base_vector=list(base))))
    unique = {}
    for vector, origin in out:
        unique.setdefault(vector, origin)
    ordered = sorted(unique.items(), key=lambda x: (O.cost(x[0], prices),
                     sum(abs(a-b) for a, b in zip(x[0], base)), x[0]))
    return ordered[:count]


class ConfigState:
    """Per-configuration incumbent, maximal margin failures and cached local candidates."""

    def __init__(self, c):
        self.c, self.cid, self.dirty, self.cache = c, c['configuration_id'], True, []
        self.fails, self.best, self.picks, self.explore_picks, self.evaluated, self.promising = [], None, 0, 0, 0, None
        self.fail_streak = 0  # consecutive failed picks: lookahead depth for this configuration (1 + streak)

    def add_failure(self, v):
        if any(leq(v, f) for f in self.fails):
            return
        self.fails = [f for f in self.fails if not leq(f, v)] + [v]

    def deferred(self, v):
        return any(leq(v, f) for f in self.fails)


def run_priority(name, candidates, knowledge, backend, search, threshold, generate, max_new, *, baseline=None, routes=None,
                 spent=0, free=None, science_approved=True, prefetch=None, on_missing='STOP', facts=None, log=None,
                 prefetch_state=None):
    s = settings(search);baseline = baseline or {}
    ready = [c for c in candidates if c['ready'] and c['configuration_id']]
    states = {c['configuration_id']: ConfigState(c) for c in ready}
    limit = 0 if not science_approved else (float('inf') if max_new is None else max_new)
    large = Fraction(search['large_step_headroom']);margin = 1 + s['deferral_fail_margin']
    trace, held, unevaluated, skipped = [], [], [], set()
    new_calls, pending, stop, since_explore, step = spent, None, None, 0, 0
    prefetch_metrics = dict(decisions=0, maximum_preview=0, chain_proposals=0, latest=None)
    selection_timing = dict(candidate_classification_seconds=0.0, candidate_classification_calls=0,
                            preview_order_seconds=0.0, preview_order_calls=0,
                            backend_answer_seconds=0.0, backend_answer_calls=0)
    if search.get('seed_saved_same_L_D'):
        O.sibling_seeds(candidates, knowledge)

    def absorb(cid, v, r):
        st = states[cid];st.dirty = True
        if r['status'] == O.PASS:
            t = O.cost(v, st.c['prices'])
            if st.best is None or (t, v) < (st.best[0], st.best[1]):
                st.best = (t, v, r)
                for o in states.values():  # transfer seeds of configurations without a feasible point follow their neighbours
                    if o.best is None:
                        o.dirty = True
        elif r['status'] == O.FAIL and r.get('MSE_upper') is not None and r['MSE_upper'] / threshold >= margin:
            st.add_failure(v)
    for (cid, v), r in list(knowledge.items()):
        if cid in states:
            absorb(cid, v, r)

    def method_min():
        rows = [(st.best[0], cid) for cid, st in states.items() if st.best is not None]
        return min(rows) if rows else None

    def transfers(st):
        fam = family(st.c, baseline);near = []
        absolute = kappa_value(st.c) is not None
        kv = (lambda c: kappa_value(c)) if absolute else (lambda c: kappa_of(c, baseline))
        k0 = kv(st.c)
        for o in states.values():
            if o is st or family(o.c, baseline) != fam or kv(o.c) is None:
                continue
            feas = sorted((O.cost(v, st.c['prices']), v) for (cid, v), r in knowledge.items() if cid == o.cid and r['status'] == O.PASS)
            if feas:
                near.append((abs(math.log(kv(o.c) / k0)), o, feas))
        out = []
        for dist, o, feas in sorted(near, key=lambda x: (x[0], kv(x[1].c)))[:2]:
            for t, v in feas[:s['transfer_count']]:  # re-priced with this configuration's own prices (O.cost above)
                out.append((v, dict(kind='TRANSFER_FROM_NEAREST_KAPPA', from_configuration=o.cid,
                                    from_kappa=str(kv(o.c)), to_kappa=str(k0), kappa_is_absolute=absolute)))
        return out

    def local(st):
        if not st.dirty:
            return st.cache
        c, prices = st.c, st.c['prices'];caps = tuple(n for n in c['caps'] for _ in O.AXES);raw = []
        if st.best is not None:
            t0, base, rec = st.best;h = O.headroom(rec, threshold)
            steps = search['decrement_steps'] if h is not None and h < large else [min(search['decrement_steps'])]
            for m in range(13):
                for group in O.axis_groups(prices, m):
                    for d in steps:
                        v = list(base)
                        for a in group:
                            v[2 * m + O.AXES.index(a)] -= d
                        if min(v) >= 1:
                            raw.append((tuple(v), dict(kind='DECREMENT_FROM_CONFIGURATION_INCUMBENT', m=m, axes=list(group), d=d,
                                                       base_vector=list(base), base_headroom=None if h is None else '%.6f' % float(h))))
            if s['allocation_exchange']:
                excluded = {v for cid, v in knowledge if cid == st.cid} | {v for cid, v in skipped if cid == st.cid}
                raw += exchange_moves(base, prices, caps, steps, s['allocation_exchange'], excluded)
        else:
            budget = O.cost(caps, prices)
            for rule in search['seed_rules']:
                g = generate(list(caps), list(prices), budget, rule)
                if g is not None:
                    raw.append((tuple(g['shots']), dict(kind='GENERATOR_SEED', rule=rule, budget=str(budget))))
            raw.append((caps, dict(kind='CAP_VECTOR_SEED')))
            for seed in c.get('sibling_seeds', []):
                raw.append((tuple(seed['vector']), dict(kind='SAVED_SAME_L_D_INCUMBENT_SEED', from_configuration=seed['configuration_id'],
                                                        from_label=seed.get('label'))))
            raw += transfers(st)
        seen, out = set(), []
        for v, origin in raw:
            key = (st.cid, v)
            if key in knowledge or key in skipped or v in seen:
                continue
            t = O.cost(v, prices)
            if st.best is not None and t >= st.best[0]:
                continue
            seen.add(v)
            out.append(dict(configuration_id=st.cid, candidate=c['name'], vector=v, cost=t, origin=origin, method=c['method'],
                            deferred=st.deferred(v), shot_change=sum(abs(a - b) for a, b in zip(v, origin.get('base_vector', v)))))
        out.sort(key=lambda x: (x['cost'], x['shot_change'], x['vector']))
        st.cache, st.dirty = out, False
        return out

    def classify():
        """SMALL = saving below small_gain_fraction of the configuration's own best (any cost); UPDATE = not small and
        cheaper than M; EXPLORE = per non-leading configuration its non-small, non-deferred candidates (head first)."""
        selection_started = time.perf_counter()
        mm = method_min();M = None if mm is None else mm[0];lead = None if mm is None else mm[1]
        groups = {k: [] for k in CLASS_ORDER};explore, queues = [], {}
        for cid, st in states.items():
            items = local(st);open_ = []
            for p in items:
                if p['deferred']:
                    groups['DEFERRED'].append(dict(p, cls='DEFERRED'));continue
                small = st.best is not None and (st.best[0] - p['cost']) < s['small_gain_fraction'] * st.best[0]
                fine = (s['coarse_gain_fraction'] is not None and st.best is not None
                        and (st.best[0] - p['cost']) < s['coarse_gain_fraction'] * st.best[0])
                if small:
                    groups['SMALL'].append(dict(p, cls='SMALL'))
                elif fine:
                    groups['FINE'].append(dict(p, cls='FINE'))
                elif M is None or p['cost'] < M:
                    groups['UPDATE'].append(dict(p, cls='UPDATE'))
                else:
                    open_.append(p)
            st.promising = sum(1 for p in items if not p['deferred'] and not (st.best is not None and (st.best[0] - p['cost'])
                                                                                < s['small_gain_fraction'] * st.best[0]))
            if cid != lead and open_:
                if st.best is None:
                    rank = (0, open_[0]['cost'], 0)
                elif M is not None and st.best[0] <= s['explore_window'] * M:
                    rank = (1, st.explore_picks, st.best[0])
                else:
                    rank = (2, st.best[0], 0)
                explore.append((rank, cid, dict(open_[0], cls='EXPLORE', explore_rank=rank[0])))
                queues[cid] = [dict(p, cls='EXPLORE', explore_rank=rank[0]) for p in open_]
        for k in groups:
            groups[k].sort(key=lambda x: (x['cost'], x['shot_change'], x['candidate'], x['vector']))
        explore.sort(key=lambda x: (x[0], x[1]))
        groups['EXPLORE'] = [p for _, _, p in explore]
        groups['_queues'] = [queues[cid] for _, cid, _ in explore]
        selection_timing['candidate_classification_seconds'] += time.perf_counter() - selection_started
        selection_timing['candidate_classification_calls'] += 1
        return groups, M

    def depth(cid):
        if s.get('lookahead_mode') in ('INDEPENDENT', 'HYBRID'):
            return 1
        return max(1, min(s['lookahead'], 1 + states[cid].fail_streak)) if s.get('adaptive_lookahead') else s['lookahead']

    def order(groups, k, observe=False):
        """The next k picks if no state changed (lookahead only): UPDATE in cost order with every explore_every-th pick
        going round-robin over the EXPLORE configurations, each contributing its next candidate.  With adaptive_lookahead a
        configuration contributes at most 1 + its current run of failed picks (a passing pick moves its incumbent and makes
        deeper speculation stale; a failing streak keeps the incumbent, so deeper candidates stay valid)."""
        selection_started = time.perf_counter()
        used = defaultdict(int)

        def take(p):
            if used[p['configuration_id']] >= depth(p['configuration_id']):
                return False
            used[p['configuration_id']] += 1;return True
        out, u, since, qs, i = [], [p for p in groups['UPDATE']], since_explore, [list(q) for q in groups['_queues']], 0
        guard = 0
        while len(out) < k and guard < 10000:
            guard += 1
            live = [q for q in qs if q and used[q[0]['configuration_id']] < depth(q[0]['configuration_id'])]
            u = [p for p in u if used[p['configuration_id']] < depth(p['configuration_id'])]
            if u and not (live and since >= s['explore_every'] - 1):
                p = u.pop(0)
                if take(p):
                    out.append(p);since += 1
            elif live:
                q = live[i % len(live)];p = q.pop(0);i += 1
                if take(p):
                    out.append(p);since = 0
            else:
                rest = [p for p in groups['FINE'] + groups['SMALL'] + groups['DEFERRED'] if take(p)];out += rest[:k - len(out)];break
        adaptive = s.get('adaptive_hybrid') and s.get('lookahead_mode') == 'HYBRID'
        limits = None
        if adaptive:
            limits = hybrid_limits(len(out), k, prefetch_state() if prefetch_state is not None else None)
            out = out[:limits['preview_limit']]
            extra = chains(out, limits['chain_slots'])
            out += extra
            if observe:
                prefetch_metrics['decisions'] += 1
                prefetch_metrics['maximum_preview'] = max(prefetch_metrics['maximum_preview'], len(out))
                prefetch_metrics['chain_proposals'] += len(extra)
                prefetch_metrics['latest'] = limits
        elif (s.get('adaptive_lookahead') and s.get('lookahead_mode') != 'INDEPENDENT') or s.get('lookahead_mode') == 'HYBRID':
            out += chains(out)  # legacy: retained byte-for-byte ordering when the opt-in is off
        seen, uniq = set(), []
        for p in out:
            if (p['configuration_id'], p['vector']) not in seen:
                seen.add((p['configuration_id'], p['vector']));uniq.append(p)
        selection_timing['preview_order_seconds'] += time.perf_counter() - selection_started
        selection_timing['preview_order_calls'] += 1
        return uniq

    def chain_next(st, v, exclude):
        """The pick V2 makes next for this configuration if allocation v passes: the cheapest non-small, non-deferred
        decrement of v (same steps rule with the incumbent's headroom, same order: cost, shot change, vector)."""
        prices = st.c['prices'];h = O.headroom(st.best[2], threshold)
        steps = search['decrement_steps'] if h is not None and h < large else [min(search['decrement_steps'])]
        tv = O.cost(v, prices);best = None
        for m in range(13):
            for group in O.axis_groups(prices, m):
                for d in steps:
                    w = list(v)
                    for a in group:
                        w[2 * m + O.AXES.index(a)] -= d
                    w = tuple(w)
                    if min(w) < 1 or (st.cid, w) in knowledge or (st.cid, w) in skipped or w in exclude or st.deferred(w):
                        continue
                    t = O.cost(w, prices)
                    if t >= tv or (tv - t) < s['small_gain_fraction'] * tv:
                        continue
                    key = (t, d * len(group), w)
                    if best is None or key < best[0]:
                        best = (key, dict(configuration_id=st.cid, candidate=st.c['name'], vector=w, cost=t, method=st.c['method'],
                                          origin=dict(kind='CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES', base_vector=list(v)), deferred=False,
                                          shot_change=d * len(group), cls='CHAIN'))
        return None if best is None else best[1]

    def chains(preview, maximum=None):
        """Speculation along the likely path for configurations whose last pick passed: assume their next pick passes too
        and prefetch the pick that would follow from it, up to ``lookahead`` deep.  Consumption stays one answer at a time in
        V2's order, so the adopted trajectory is unchanged; a wrong guess is only an unused (reusable) job."""
        extra = []
        if maximum == 0:
            return extra
        frontiers = []
        for cid in dict.fromkeys(p['configuration_id'] for p in preview):
            st = states[cid]
            if st.best is None or st.fail_streak > 0:
                continue
            head = next(p for p in preview if p['configuration_id'] == cid);v, seen = head['vector'], {head['vector']}
            if maximum is not None:
                frontiers.append([st, v, seen, 0])
                continue
            for _ in range(s['lookahead'] - 1):
                nxt = chain_next(st, v, seen)
                if nxt is None:
                    break
                extra.append(nxt);seen.add(nxt['vector']);v = nxt['vector']
        # Capacity-limited chains grow one round at a time across independent
        # configurations, so the first configuration cannot consume all spares.
        while maximum is not None and len(extra) < maximum and frontiers:
            pending_frontiers = []
            for st, v, seen, depth_ in frontiers:
                if len(extra) >= maximum:
                    break
                nxt = chain_next(st, v, seen)
                if nxt is None or depth_ >= s['lookahead'] - 1:
                    continue
                extra.append(nxt);seen.add(nxt['vector'])
                pending_frontiers.append([st, nxt['vector'], seen, depth_ + 1])
            frontiers = pending_frontiers
        return extra

    while step < search['max_steps_per_method']:
        groups, M = classify()
        if groups['UPDATE'] and not (groups['EXPLORE'] and since_explore >= s['explore_every'] - 1):
            choice = groups['UPDATE'][0]
        elif groups['EXPLORE']:
            choice = groups['EXPLORE'][0]
        elif groups['FINE']:
            choice = groups['FINE'][0]
        elif (groups['SMALL'] or groups['DEFERRED']) and search.get('small_and_deferred') == 'STOP':
            stop = 'ONLY_SMALL_OR_DEFERRED_LEFT';break
        elif groups['SMALL']:
            choice = groups['SMALL'][0]
        elif groups['DEFERRED']:
            choice = groups['DEFERRED'][0]
        else:
            stop = 'NO_CANDIDATE_LEFT_IN_RANGE' if M is not None else 'NO_FEASIBLE_POINT_IN_RANGE'
            break
        allow = new_calls < limit;is_free = free is not None and free(choice)
        if prefetch is not None and science_approved and s['lookahead'] > 1 and new_calls < limit:
            # finite budgets prefetch too (rounds driven in batches); unconsumed jobs stay reusable knowledge
            width = s['lookahead']
            if s.get('adaptive_hybrid') and s.get('lookahead_mode') == 'HYBRID' and prefetch_state is not None:
                # lookahead remains the maximum depth of each speculative
                # chain; many independent heads may fill a larger global pool.
                width = max(width, int(prefetch_state().get('capacity') or width))
            if limit != float('inf'):
                width = int(min(width, limit - new_calls))
            prefetch(order(groups, width, observe=True))
        answer_started = time.perf_counter()
        answer = backend.answer(choice) if backend is not None and (is_free or allow) else None
        selection_timing['backend_answer_seconds'] += time.perf_counter() - answer_started
        selection_timing['backend_answer_calls'] += int(backend is not None and (is_free or allow))
        hold = None if answer is None else O.HOLD.get(answer['status'])
        if answer is None and on_missing == 'SKIP':
            skipped.add((choice['configuration_id'], choice['vector']));states[choice['configuration_id']].dirty = True
            unevaluated.append(dict(step_index=len(trace), cls=choice['cls'], candidate=choice['candidate'],
                                    configuration_id=choice['configuration_id'], N_X=O.split(choice['vector'])[0],
                                    N_Y=O.split(choice['vector'])[1], cost=choice['cost'], origin=choice['origin'],
                                    status='UNEVALUATED_NO_ANSWER_IN_LOG'))
            continue
        if answer is None or hold:
            pending = dict(choice, route=(routes or {}).get(name), queue=[dict(candidate=p['candidate'], cost=p['cost'], cls=p['cls'],
                           vector=list(p['vector']), origin=p['origin']) for p in order(groups, 5)],
                           job=None if answer is None else answer.get('job'))
            if answer is not None and answer['status'] == 'HOST_ELIGIBILITY_HOLD':
                pending['host_eligibility'] = answer.get('host_eligibility')
                pending['hold_reason'] = answer.get('reason')
            stop = hold or ('NEW_SCIENCE_NOT_APPROVED' if not science_approved and not is_free else
                            'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET' if limit == 0 and not is_free else
                            'Q3_BUDGET_EXHAUSTED' if not allow and not is_free else 'NEW_Q3_REQUIRED_NO_EXECUTOR')
            break
        step += 1;new_calls += answer['calls']
        if choice['cls'] == 'EXPLORE':
            since_explore = 0;states[choice['configuration_id']].explore_picks += 1
        elif choice['cls'] == 'UPDATE':
            since_explore += 1
        key = (choice['configuration_id'], choice['vector']);knowledge[key] = dict(answer, cost=choice['cost'])
        st = states[choice['configuration_id']];before_cfg = None if st.best is None else st.best[0]
        st.picks += 1;absorb(choice['configuration_id'], choice['vector'], knowledge[key])
        st.fail_streak = 0 if answer['status'] == O.PASS else st.fail_streak + 1
        after = method_min();nx, ny = O.split(choice['vector'])
        row = dict(step=step, cls=choice['cls'], candidate=choice['candidate'], configuration_id=choice['configuration_id'],
                   vector=list(choice['vector']), N_X=nx, N_Y=ny, cost=choice['cost'], origin=choice['origin'],
                   answer_source=answer['source'], status=answer['status'], MSE_lower=answer.get('MSE_lower'),
                   MSE_upper=answer['MSE_upper'], calls=answer['calls'], job=answer.get('job'), reason=answer.get('reason'),
                   method_min_before=M, method_min_after=None if after is None else after[0],
                   configuration_best_before=before_cfg, configuration_best_after=None if st.best is None else st.best[0],
                   improved=after is not None and (M is None or after[0] < M),
                   improved_configuration=st.best is not None and (before_cfg is None or st.best[0] < before_cfg))
        trace.append(row)
        if log is not None:
            log(row)
        if answer['status'] not in (O.PASS, O.FAIL):
            held.append(dict(candidate=choice['candidate'], vector=list(choice['vector']), status=answer['status'], reason=answer.get('reason')))
        if answer['status'] == 'UNKNOWN':
            stop = 'UNKNOWN_RESULT';break
    else:
        stop = 'STEP_LIMIT'
    groups, M = classify()
    _, incumbent = O.method_state(ready, knowledge, threshold)
    by = []
    for cid, st in states.items():
        b = st.best
        by.append(dict(candidate=st.c['name'], configuration_id=cid, key=st.c['key'], picks=st.picks, explore_picks=st.explore_picks,
                       kappa_value=None if kappa_value(st.c) is None else str(kappa_value(st.c)), promising=st.promising,
                       left=dict((k, sum(1 for p in groups[k] if p['configuration_id'] == cid)) for k in CLASS_ORDER),
                       best=None if b is None else dict(vector=list(b[1]), cost=b[0], cost_display=O.show(b[0]), status=b[2]['status'],
                                                        MSE_lower=b[2].get('MSE_lower'), MSE_upper=b[2].get('MSE_upper'),
                                                        source=b[2].get('source'), job=b[2].get('job'), label=b[2].get('label'))))
    plan = [] if search.get('kappa_neighbourhood') else kappa_plan(candidates, states, s, baseline, knowledge)
    requests = plan + new_ld_requests(candidates, states, M, facts, baseline)
    return dict(method=name, rule=RULE_V2, stop_reason=stop, trace=trace, held=held, new_calls=new_calls - spent, spent_before=spent,
                pending_request=pending, incumbent=incumbent, not_ready_in_range=[c['name'] for c in candidates if not c['ready']],
                by_configuration=by, incumbent_scope='METHOD_PRIORITY', unevaluated=unevaluated,
                prefetch_policy=prefetch_metrics,
                selection_timing=selection_timing,
                left={k: len(v) for k, v in groups.items() if not k.startswith('_')}, input_requests=requests, settings={k: str(v) if isinstance(v, Fraction) else v
                                                                                                for k, v in s.items()})


def kappa_plan(candidates, states, s, baseline, knowledge):
    """Next kappa values per (L_D, formula, width) from current incumbents (input requests; nothing is generated here)."""
    fams = defaultdict(list)
    for c in candidates:
        if c['family'] == 'PR' and c['configuration_id']:
            st = states.get(c['configuration_id'])
            fams[family(c, baseline)].append((kappa_of(c, baseline), None if st is None or st.best is None else st.best[0], c))
    out = []
    lo, hi = s['kappa_bounds']
    for fam, rows in sorted(fams.items(), key=lambda x: str(x[0])):
        known = {k for k, _, _ in rows}
        feas = sorted((k, t, c) for k, t, c in rows if t is not None)
        if len(feas) < 2:
            continue
        ks = [k for k, _, _ in feas];i = min(range(len(feas)), key=lambda j: (feas[j][1], feas[j][0]))
        kstar, tstar, cstar = feas[i];props = []
        if i == len(feas) - 1:
            props.append((kstar * kstar / ks[i - 1], 'EDGE_EXTEND_UP', [cstar, feas[i - 1][2]]))
        elif i == 0:
            props.append((kstar * kstar / ks[1], 'EDGE_EXTEND_DOWN', [cstar, feas[1][2]]))
        else:
            sides = sorted([(feas[i - 1][1], i - 1), (feas[i + 1][1], i + 1)])
            for _, j in sides:
                if max(kstar, ks[j]) / min(kstar, ks[j]) > s['kappa_min_ratio']:
                    props.append(((kstar + ks[j]) / 2, 'INTERIOR_REFINE', [cstar, feas[j][2]]))
        for k, why, near in props:
            if k in known or not (lo <= k <= hi):
                out.append(dict(kind='KAPPA_PROPOSAL_NOT_REQUESTED', family=dict(L_D=fam[0], formula=fam[1], delta_over_h=fam[2]),
                                kappa_multiplier=str(k), reason=why, why_not='ALREADY_PRESENT' if k in known else 'OUTSIDE_KAPPA_BOUNDS'))
                continue
            seeds = []
            for c in near:
                feasible = sorted((O.cost(v, c['prices']), v) for (cid, v), r in knowledge.items() if cid == c['configuration_id'] and r['status'] == O.PASS)
                seeds += [dict(from_configuration=c['configuration_id'], from_kappa=str(kappa_of(c, baseline)), N_X=O.split(v)[0],
                               cost_in_source_configuration=t) for t, v in feasible[:s['transfer_count']]]
            out.append(dict(kind='KAPPA_PROPOSAL', family=dict(L_D=fam[0], formula=fam[1], delta_over_h=fam[2]), kappa_multiplier=str(k),
                            reason=why, kappa_minimum_so_far=str(kstar), family_best_cost=tstar, from_kappas=[str(kappa_of(c, baseline)) for c in near],
                            needs=['CONFIGURATION_GENERATION', 'PRICE_26AXIS_MODEL', 'SIGNAL_13ROUND_PACKET', 'Q3_ROUTE_ENTRY'],
                            initial_candidates=seeds, priority=tstar,
                            note='transfer seeds are re-priced with the new configuration and evaluated with its own signals'))
    return sorted(out, key=lambda r: (r['kind'] != 'KAPPA_PROPOSAL', r.get('priority') or 0))


def new_ld_requests(candidates, states, M, facts, baseline):
    """Configurations that cannot be evaluated yet (no configuration, signal, price or route), nearest L_D to the leader first."""
    lead = None
    if M is not None:
        lead = next((st.c['key'].get('L_D') for st in states.values() if st.best is not None and st.best[0] == M), None)
    out = []
    for c in candidates:
        if c['ready']:
            continue
        ld = c['key'].get('L_D')
        prep = c.get('preparation') or {}
        out.append(dict(kind='NEW_L_D' if c['configuration_id'] is None else 'CONFIGURATION_INPUTS_MISSING', candidate=c['name'],
                        configuration=c['key'], configuration_id=c['configuration_id'], needs=c['needs'],
                        price_needs=c.get('price_needs'), distance_to_leading_L_D=None if lead is None or ld is None else abs(ld - lead),
                        preparation=prep and dict(status=prep.get('status'), preparation_status=prep.get('preparation_status'),
                                                  next_actions=prep.get('next_actions'), readiness=prep.get('readiness'),
                                                  descriptor_status=prep.get('descriptor_status'))))
    return sorted(out, key=lambda r: (r['distance_to_leading_L_D'] is None, r['distance_to_leading_L_D'] or 0, str(r['configuration'])))


# ---------------------------------------------------------------- kappa neighbourhood (absolute kappa)
NEIGHBOURHOOD_DEFAULTS = dict(center='2', step_ratio='3/2', bounds=['1/2', '8'], min_ratio='9/8', extend_margin='1/100',
                              settle_after_picks=80, family_window='3/2')


def neighbourhood_settings(nb):
    s = dict(NEIGHBOURHOOD_DEFAULTS, **(nb or {}))
    return dict(center=Fraction(str(s['center'])), step_ratio=Fraction(str(s['step_ratio'])),
                bounds=[Fraction(str(x)) for x in s['bounds']], min_ratio=Fraction(str(s['min_ratio'])),
                extend_margin=Fraction(str(s['extend_margin'])), settle_after_picks=int(s['settle_after_picks']),
                family_window=Fraction(str(s['family_window'])))


def kappa_neighbourhood_plan(points, nb, sched=None):
    """Next absolute-kappa points per L_D family around the centre (no automatic doubling / halving).

    points: dicts with L_D, kappa (Fraction), status (EVALUATED / PENDING / HELD / ALIAS), best (Fraction or None),
    promising (non-small, non-deferred candidates left), picks.  A point is SETTLED when it has no promising candidate
    or has had settle_after_picks picks; a settled point without a feasible allocation counts as infinitely expensive.
    Rules, one new point per family per decision, all recorded with the costs used:
      CENTER                 the centre is evaluated first in every family that can be prepared
      FIRST_NEIGHBOURS       centre / r and centre * r once the centre has a feasible point and the family is active
      INTERIOR_REFINE        minimum inside the sampled set: midpoint towards the cheaper neighbour first, while the
                             kappa ratio exceeds min_ratio
      EDGE_EXTEND_ONE_STEP   minimum at an edge AND cheaper than the inner neighbour by >= extend_margin: one step r
      EDGE_REFINE            minimum at an edge but the gain is below the margin (or the bound is reached): midpoint
                             towards the inner neighbour while the ratio exceeds min_ratio
      STOP / WAIT            nothing left under these rules / some point not settled yet
    A family is active while its best <= family_window x the neighbourhood minimum over all families.
    With ``sched`` (exact kappa -> 13-round R vector of the family, see rpe_kappa_schedule) the rule works on physical
    schedules: points with the same schedule are one point; the first neighbours are the first DIFFERENT schedules below
    and above the centre (FIRST_DISTINCT_SCHEDULE_DOWN/UP), and a proposed kappa whose schedule is already sampled is
    moved to the next different schedule in its direction (never dropped because of the kappa step size).
    """
    s = neighbourhood_settings(nb);lo, hi = s['bounds'];r = s['step_ratio'];INF = None
    fams = defaultdict(list)
    for p in points:
        fams[p['L_D']].append(p)

    def settled(p):  # an ALIAS follows the configuration it is physically identical to
        return p['status'] in ('EVALUATED', 'ALIAS') and (p.get('promising') == 0 or (p.get('picks') or 0) >= s['settle_after_picks'])
    inside = lambda k: lo <= k <= hi
    bests = [p['best'] for p in points if p['status'] in ('EVALUATED', 'ALIAS') and p.get('best') is not None and inside(p['kappa'])]
    m_nb = min(bests) if bests else None
    proposals, decisions = [], []
    def R(ld, k):
        return None if sched is None else sched.R(ld, k)

    def snap(ld, k, toward, have_R, stop_at=None):
        """k, or the next schedule not yet sampled in direction ``toward`` ('up'/'down'); None if none before stop_at."""
        if sched is None:
            return k
        guard = 0
        while k is not None and R(ld, k) in have_R and guard < 1000:
            k = sched.next(ld, k, toward);guard += 1
        if k is None or not inside(k) or (stop_at is not None and (k >= stop_at if toward == 'up' else k <= stop_at)):
            return None
        if stop_at is not None and R(ld, k) == R(ld, stop_at):
            return None
        return k

    for ld in sorted(fams):
        pts = [p for p in fams[ld] if inside(p['kappa'])];have = {p['kappa'] for p in fams[ld]}
        have_R = {R(ld, p['kappa']) for p in fams[ld]} if sched is not None and sched.known(ld) else set()
        if sched is not None and sched.known(ld):  # one point per physical schedule (the one nearest the centre)
            by_R = {}
            for p in sorted(pts, key=lambda p: (abs(p['kappa'] - s['center']), p['kappa'])):
                by_R.setdefault(R(ld, p['kappa']), p)
            pts = list(by_R.values())
        usable = [p for p in pts if p['status'] != 'HELD']
        if not usable:
            held = [p for p in pts if p['status'] == 'HELD']
            decisions.append(dict(L_D=ld, decision='STOP_FAMILY_INPUTS_HELD', reason='; '.join(str(p.get('reason')) for p in held)))
            continue
        if s['center'] not in have:
            proposals.append(dict(L_D=ld, kappa=s['center'], reason='CENTER'));decisions.append(dict(L_D=ld, decision='CENTER'))
            continue
        fam_best = min((p['best'] for p in usable if p.get('best') is not None), default=None)
        waiting = [str(p['kappa']) for p in usable if not settled(p)]
        centre = next(p for p in fams[ld] if p['kappa'] == s['center'])
        if centre['status'] == 'HELD':
            decisions.append(dict(L_D=ld, decision='STOP_FAMILY_CENTER_HELD', reason=centre.get('reason')));continue
        if centre.get('best') is None and not settled(centre):  # neighbours once the centre has a feasible point
            decisions.append(dict(L_D=ld, decision='WAIT', waiting=waiting));continue
        if fam_best is not None and m_nb is not None and fam_best > s['family_window'] * m_nb:
            decisions.append(dict(L_D=ld, decision='STOP_FAMILY_OUTSIDE_WINDOW', family_best=fam_best, neighbourhood_min=m_nb,
                                  window=str(s['family_window'])))
            continue
        if sched is not None and sched.known(ld):
            first = []
            for toward in ('down', 'up'):
                k = sched.next(ld, s['center'], toward)
                if k is not None and inside(k) and R(ld, k) not in have_R:
                    first.append((k, 'FIRST_DISTINCT_SCHEDULE_' + toward.upper(), sched.describe(ld, s['center'], toward)))
            if first:
                for k, why, info in first:
                    proposals.append(dict(L_D=ld, kappa=k, reason=why, centre_best=centre.get('best'), schedule_change=info))
                decisions.append(dict(L_D=ld, decision='FIRST_DISTINCT_SCHEDULES', kappas=[str(k) for k, _, _ in first]));continue
        else:
            first = [k for k in (s['center'] / r, s['center'] * r) if inside(k) and k not in have]
            if first:
                for k in first:
                    proposals.append(dict(L_D=ld, kappa=k, reason='FIRST_NEIGHBOURS', centre_best=centre.get('best')))
                decisions.append(dict(L_D=ld, decision='FIRST_NEIGHBOURS', kappas=[str(k) for k in first]));continue
        if waiting:
            decisions.append(dict(L_D=ld, decision='WAIT', waiting=waiting));continue
        rows = sorted((p['kappa'], p['best'] if p.get('best') is not None else INF) for p in usable)
        cost = lambda c: (1, 0) if c is None else (0, c)
        i = min(range(len(rows)), key=lambda j: (cost(rows[j][1]), abs(rows[j][0] - s['center'])))
        kstar, cstar = rows[i]
        if cstar is None:
            decisions.append(dict(L_D=ld, decision='STOP_FAMILY_NO_FEASIBLE_POINT'));continue
        new, why, evidence = None, None, dict(kappa_min=str(kstar), cost_min=cstar)
        if 0 < i < len(rows) - 1:
            for j in sorted((i - 1, i + 1), key=lambda j: cost(rows[j][1])):
                mid = (kstar + rows[j][0]) / 2
                if rows[j][1] == cstar and sched is None:  # equal cost on both ends (plateau): nothing to refine there
                    continue
                toward = 'up' if rows[j][0] > kstar else 'down'
                mid = snap(ld, mid, toward, have_R, stop_at=rows[j][0]) if sched is not None and sched.known(ld) else mid
                if mid is not None and max(kstar, rows[j][0]) / min(kstar, rows[j][0]) > s['min_ratio'] and mid not in have:
                    new, why = mid, 'INTERIOR_REFINE';evidence.update(neighbour=str(rows[j][0]), neighbour_cost=rows[j][1]);break
        else:
            j = 1 if i == 0 else len(rows) - 2
            if j < 0 or j >= len(rows) or j == i:  # a single sampled point: step outward on both sides
                for k in (kstar / r, kstar * r):
                    if inside(k) and k not in have:
                        new, why = k, 'EDGE_EXTEND_ONE_STEP';break
            else:
                kin, cin = rows[j];step = kstar / r if i == 0 else kstar * r
                outward = 'down' if i == 0 else 'up'
                if sched is not None and sched.known(ld) and inside(step):
                    step = snap(ld, step, outward, have_R) or step  # the step kappa's schedule may already be sampled
                gain_ok = cin is None or cstar <= cin * (1 - s['extend_margin'])
                evidence.update(inner=str(kin), inner_cost=cin, margin=str(s['extend_margin']))
                if gain_ok and inside(step) and step not in have and R(ld, step) not in have_R:
                    new, why = step, 'EDGE_EXTEND_ONE_STEP'
                else:
                    mid = (kstar + kin) / 2
                    if sched is not None and sched.known(ld):
                        mid = snap(ld, mid, 'up' if kin > kstar else 'down', have_R, stop_at=kin)
                    if mid is not None and cin != cstar and max(kstar, kin) / min(kstar, kin) > s['min_ratio'] and mid not in have:
                        new, why = mid, 'EDGE_REFINE'
                        evidence['edge_extension_skipped'] = ('gain below margin' if not gain_ok else
                                                              'outside bounds' if not inside(step) else 'already sampled')
        if new is None:
            decisions.append(dict(L_D=ld, decision='STOP_FAMILY_RULE_EXHAUSTED', **evidence))
        else:
            proposals.append(dict(L_D=ld, kappa=new, reason=why, **evidence))
            decisions.append(dict(L_D=ld, decision=why, kappa=str(new), **evidence))
    return proposals, decisions, m_nb
