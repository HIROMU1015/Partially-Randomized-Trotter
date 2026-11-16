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

from trottertracks import rpe_method_optimizer as O

RULE_V2 = 'METHOD_MINIMUM_PRIORITY_V2'
DEFAULTS = dict(explore_every=4, explore_window='3/2', small_gain_fraction='1/10000', deferral_fail_margin='1/10000',
                transfer_count=2, kappa_bounds=['1/4', '8'], kappa_min_ratio='9/8', lookahead=1)
CLASS_ORDER = ('UPDATE', 'EXPLORE', 'SMALL', 'DEFERRED')


def settings(search):
    s = dict(DEFAULTS, **{k: v for k, v in search.items() if k in DEFAULTS})
    s.update(explore_window=Fraction(s['explore_window']), small_gain_fraction=Fraction(s['small_gain_fraction']),
             deferral_fail_margin=Fraction(s['deferral_fail_margin']), kappa_min_ratio=Fraction(s['kappa_min_ratio']),
             kappa_bounds=[Fraction(x) for x in s['kappa_bounds']])
    return s


def family(c, baseline):
    k = c['key']
    return (k.get('L_D'), str(k.get('formula', baseline.get('formula'))), str(k.get('delta_over_h', baseline.get('delta_over_h'))))


def kappa_of(c, baseline):
    return Fraction(str(c['key'].get('kappa_multiplier', baseline.get('kappa_multiplier', '1'))))


def leq(a, b):
    return all(x <= y for x, y in zip(a, b))


class ConfigState:
    """Per-configuration incumbent, maximal margin failures and cached local candidates."""

    def __init__(self, c):
        self.c, self.cid, self.dirty, self.cache = c, c['configuration_id'], True, []
        self.fails, self.best, self.picks, self.explore_picks, self.evaluated = [], None, 0, 0, 0

    def add_failure(self, v):
        if any(leq(v, f) for f in self.fails):
            return
        self.fails = [f for f in self.fails if not leq(f, v)] + [v]

    def deferred(self, v):
        return any(leq(v, f) for f in self.fails)


def run_priority(name, candidates, knowledge, backend, search, threshold, generate, max_new, *, baseline=None, routes=None,
                 spent=0, free=None, science_approved=True, prefetch=None, on_missing='STOP', facts=None, log=None):
    s = settings(search);baseline = baseline or {}
    ready = [c for c in candidates if c['ready'] and c['configuration_id']]
    states = {c['configuration_id']: ConfigState(c) for c in ready}
    limit = 0 if not science_approved else (float('inf') if max_new is None else max_new)
    large = Fraction(search['large_step_headroom']);margin = 1 + s['deferral_fail_margin']
    trace, held, unevaluated, skipped = [], [], [], set()
    new_calls, pending, stop, since_explore, step = spent, None, None, 0, 0
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
        fam = family(st.c, baseline);k0 = kappa_of(st.c, baseline);near = []
        for o in states.values():
            if o is st or family(o.c, baseline) != fam:
                continue
            feas = sorted((O.cost(v, st.c['prices']), v) for (cid, v), r in knowledge.items() if cid == o.cid and r['status'] == O.PASS)
            if feas:
                near.append((abs(math.log(kappa_of(o.c, baseline) / k0)), o, feas))
        out = []
        for dist, o, feas in sorted(near, key=lambda x: (x[0], kappa_of(x[1].c, baseline)))[:2]:
            for t, v in feas[:s['transfer_count']]:
                out.append((v, dict(kind='TRANSFER_FROM_NEAREST_KAPPA', from_configuration=o.cid,
                                    from_kappa=str(kappa_of(o.c, baseline)), to_kappa=str(kappa_of(st.c, baseline)))))
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
        mm = method_min();M = None if mm is None else mm[0];lead = None if mm is None else mm[1]
        groups = {k: [] for k in CLASS_ORDER};explore = []
        for cid, st in states.items():
            items = local(st)
            for p in items:
                if p['deferred']:
                    groups['DEFERRED'].append(dict(p, cls='DEFERRED'))
                elif M is None or p['cost'] < M:
                    small = st.best is not None and (st.best[0] - p['cost']) < s['small_gain_fraction'] * st.best[0]
                    groups['SMALL' if small else 'UPDATE'].append(dict(p, cls='SMALL' if small else 'UPDATE'))
            if cid != lead:
                head = next((p for p in items if not p['deferred']), None)
                if head is not None:
                    if st.best is None:
                        rank = (0, head['cost'], 0)
                    elif M is not None and st.best[0] <= s['explore_window'] * M:
                        rank = (1, st.explore_picks, st.best[0])
                    else:
                        rank = (2, st.best[0], 0)
                    explore.append((rank, cid, dict(head, cls='EXPLORE', explore_rank=rank[0])))
        for k in groups:
            groups[k].sort(key=lambda x: (x['cost'], x['shot_change'], x['candidate'], x['vector']))
        groups['EXPLORE'] = [p for _, _, p in sorted(explore, key=lambda x: (x[0], x[1]))]
        return groups, M

    def order(groups, k):
        """The next k picks if no state changed (used for lookahead only)."""
        out, u, e, since = [], list(groups['UPDATE']), list(groups['EXPLORE']), since_explore
        while len(out) < k and (u or e or groups['SMALL'] or groups['DEFERRED']):
            if u and not (e and since >= s['explore_every'] - 1):
                out.append(u.pop(0));since += 1
            elif e:
                out.append(e.pop(0));since = 0
            else:
                rest = groups['SMALL'] + groups['DEFERRED'];out += rest[:k - len(out)];break
        seen, uniq = set(), []
        for p in out:
            if (p['configuration_id'], p['vector']) not in seen:
                seen.add((p['configuration_id'], p['vector']));uniq.append(p)
        return uniq

    while step < search['max_steps_per_method']:
        groups, M = classify()
        if groups['UPDATE'] and not (groups['EXPLORE'] and since_explore >= s['explore_every'] - 1):
            choice = groups['UPDATE'][0]
        elif groups['EXPLORE']:
            choice = groups['EXPLORE'][0]
        elif groups['SMALL']:
            choice = groups['SMALL'][0]
        elif groups['DEFERRED']:
            choice = groups['DEFERRED'][0]
        else:
            stop = 'NO_CANDIDATE_LEFT_IN_RANGE' if M is not None else 'NO_FEASIBLE_POINT_IN_RANGE'
            break
        allow = new_calls < limit;is_free = free is not None and free(choice)
        if prefetch is not None and limit == float('inf') and science_approved and s['lookahead'] > 1:
            prefetch(order(groups, s['lookahead']))
        answer = backend.answer(choice) if backend is not None and (is_free or allow) else None
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
                       left=dict((k, sum(1 for p in groups[k] if p['configuration_id'] == cid)) for k in CLASS_ORDER),
                       best=None if b is None else dict(vector=list(b[1]), cost=b[0], cost_display=O.show(b[0]), status=b[2]['status'],
                                                        MSE_lower=b[2].get('MSE_lower'), MSE_upper=b[2].get('MSE_upper'),
                                                        source=b[2].get('source'), job=b[2].get('job'), label=b[2].get('label'))))
    plan = kappa_plan(candidates, states, s, baseline, knowledge)
    requests = plan + new_ld_requests(candidates, states, M, facts, baseline)
    return dict(method=name, rule=RULE_V2, stop_reason=stop, trace=trace, held=held, new_calls=new_calls - spent, spent_before=spent,
                pending_request=pending, incumbent=incumbent, not_ready_in_range=[c['name'] for c in candidates if not c['ready']],
                by_configuration=by, incumbent_scope='METHOD_PRIORITY', unevaluated=unevaluated,
                left={k: len(v) for k, v in groups.items()}, input_requests=requests, settings={k: str(v) if isinstance(v, Fraction) else v
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
        out.append(dict(kind='NEW_L_D' if c['configuration_id'] is None else 'CONFIGURATION_INPUTS_MISSING', candidate=c['name'],
                        configuration=c['key'], configuration_id=c['configuration_id'], needs=c['needs'],
                        price_needs=c.get('price_needs'), distance_to_leading_L_D=None if lead is None or ld is None else abs(ld - lead)))
    return sorted(out, key=lambda r: (r['distance_to_leading_L_D'] is None, r['distance_to_leading_L_D'] or 0, str(r['configuration'])))
