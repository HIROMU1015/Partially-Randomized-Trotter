"""Finite whole-allocation search on a run-defined RPE multiplier grid."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from hashlib import sha256
import json

RULE = 'RPE_SCALED_SHOTS_V1'
EPSILON = '0.00159360010199040'
DEFAULT = dict(M=[11, 12, 13], grid=dict(min_k=-12, max_k=16, divisor=4), compare_paper=True)


def settings(value=None):
    supplied = {} if value is None else value
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT):
        raise ValueError('Expected M, grid and/or compare_paper')
    result = dict(DEFAULT, **supplied)
    horizon, grid = result['M'], result['grid']
    if (not isinstance(horizon, list) or not horizon
            or any(type(m) is not int or m < 0 for m in horizon)):
        raise ValueError('M must list nonnegative maximum round indices')
    if (not isinstance(grid, dict) or set(grid) != {'min_k', 'max_k', 'divisor'}
            or any(type(v) is not int for v in grid.values())
            or grid['min_k'] > grid['max_k'] or grid['divisor'] < 1):
        raise ValueError('grid requires integer min_k <= max_k and positive divisor')
    if type(result['compare_paper']) is not bool:
        raise ValueError('compare_paper must be a boolean')
    return dict(M=sorted(set(horizon)), grid=dict(grid), compare_paper=result['compare_paper'])


def base_shots(M):
    return [11 + 4 * (M - m) for m in range(M + 1)]


def dyadic_count(b, k, divisor):
    """Return ceil(b * 2**(k/divisor)) using only integer comparisons."""
    power = b ** divisor
    if k < 0:
        denominator = 1 << -k
        target = (power + denominator - 1) // denominator
    else:
        target = power << k
    lo, hi = 1, 1 << ((target.bit_length() + divisor - 1) // divisor)
    while lo < hi:
        n = (lo + hi) // 2
        if n ** divisor >= target:
            hi = n
        else:
            lo = n + 1
    return lo


def paper_counts(bases, kappa):
    """Certify each ceiling using enclosing Decimal bounds for exp(2/kappa)."""
    if kappa <= 0:
        raise ValueError('Paper comparison requires positive absolute kappa')
    argument = Fraction(2) / kappa
    precision = 32
    while True:
        with localcontext() as ctx:
            ctx.prec = precision
            ctx.rounding = ROUND_FLOOR
            lo = (Decimal(argument.numerator) / Decimal(argument.denominator)).exp().next_minus()
            low = [int((lo * b).to_integral_value(rounding=ROUND_CEILING)) for b in bases]
            ctx.rounding = ROUND_CEILING
            hi = (Decimal(argument.numerator) / Decimal(argument.denominator)).exp().next_plus()
            high = [int((hi * b).to_integral_value(rounding=ROUND_CEILING)) for b in bases]
        if low == high:
            return low
        precision *= 2


def allocation_menu(M, policy, *, kappa=None):
    """A unique interleaved vector, with every generating multiplier retained."""
    menu = {}
    base = base_shots(M)

    def add(shots, alias):
        vector = tuple(n for n in shots for _ in ('X', 'Y'))
        menu.setdefault(vector, []).append(alias)

    grid = policy['grid']
    for k in range(grid['min_k'], grid['max_k'] + 1):
        divisor = grid['divisor']
        exponent = Fraction(k, divisor)
        add([dyadic_count(b, k, divisor) for b in base],
            dict(kind='GRID', k=k, divisor=divisor, value=f'2^({exponent})'))
    add(base, dict(kind='UNIT_REFERENCE', value='1'))
    if policy['compare_paper'] and kappa is not None:
        kappa = Fraction(kappa)
        add(paper_counts(base, kappa),
            dict(kind='PAPER_REFERENCE', kappa=str(kappa), value=f'exp(2/({kappa}))'))
    return menu


def projection_id(configuration_id, source_M, M):
    identity = json.dumps([RULE, 'PREFIX', configuration_id, source_M, M], separators=(',', ':'))
    return sha256(identity.encode()).hexdigest()


def _classification(answer, threshold):
    from . import rpe_method_optimizer as O
    status = answer.get('status')
    lo, up = answer.get('MSE_lower'), answer.get('MSE_upper')
    if status in (O.PASS, O.FAIL, 'INCONCLUSIVE') and lo is not None and up is not None and 0 <= lo <= up:
        return O.PASS if up <= threshold else O.FAIL if lo > threshold else 'INCONCLUSIVE'
    if (status in ('NO_ROUTE', 'HOST_ELIGIBILITY_HOLD', 'STALE_JOB', 'TECHNICAL_HOLD') or
            answer.get('failure_kind') in ('PRE_API_SCOPE_REJECTION', 'WORKER_INTERRUPTED')):
        return 'TECHNICAL_HOLD'
    return 'IN_PROGRESS' if status == 'IN_PROGRESS' else 'UNKNOWN'


def scope_coverage(method_scope, horizons, rows):
    """Match the declared finite design axes before claiming scope exhaustion."""
    from itertools import islice, product
    from math import prod
    from .rpe_scope_map import scope_axes, coordinate, row_coordinates

    scope = method_scope or {}
    axes = scope_axes(scope, horizons)

    seen = set()
    for row in rows:
        if not row['configuration_id']:
            continue
        seen.add(coordinate(axes, row_coordinates(row)))
    values = list(axes.values())
    requested = prod(map(len, values))
    represented = sum(all(v in grid for v, grid in zip(key, values)) for key in seen)
    missing = requested - represented
    sample = (islice((row for row in product(*values) if row not in seen), 32) if missing else ())
    return dict(axes=list(axes), absolute_kappa_bounds=scope.get('absolute_kappa_bounds'),
                requested=requested, represented=represented, missing_count=missing,
                missing_sample=[dict(zip(axes, row)) for row in sample])


def run(name, candidates, knowledge, backend, search, threshold, max_new, *, spent=0, free=None,
        science_approved=True, refresh_ready=None, submit_standby=None, held=(), flagged_stop=None,
        read_completed=None, after_drain=None, await_ready=None, method_scope=None,
        refresh_completions=None, answer_now=None, await_dispatch=None,
        on_feasible=None, price_work_pending=None, retire_allocations=None):
    """Search the fixed allocation grid, with no assumed MSE monotonicity."""
    from decimal import DecimalException
    from . import rpe_method_optimizer as O
    from .rpe_schedule_search import coarse_priority

    if threshold != Fraction(EPSILON)**2:
        raise ValueError('Scaled search must use the exact saved R08 MSE threshold')
    policy = settings(search['scaled_shots'])
    exploratory = search.get('exploratory_cost') is True
    feasibility_first = search.get('feasibility_first', False) is True and exploratory
    groups = {}
    registered = {}
    yielded = set()
    deferred = {}
    retired = set()
    legacy = O.method_state([c for c in candidates if c.get('prices') is not None],
                            {key: row for key, row in knowledge.items() if row.get('allocation_rule') != RULE},
                            threshold)[1]
    known = dict(knowledge)
    knowledge.clear()
    held_keys = {(r['configuration_id'], O.interleave(r['N_X'], r['N_Y'])): r for r in held}
    trace, holds = [], []
    used, steps, completed_steps = 0, 0, 0
    pending, fatal, requested_stop = None, False, None
    step_limit = search.get('max_steps_per_method')
    work_slice = search.get('work_slice_steps')
    evaluated = (O.PASS, O.FAIL, 'INCONCLUSIVE')
    budget = 0 if not science_approved else max_new
    # The queue spends no finite budget on optional points within a series.
    # For an unlimited run, preflight/dispatch can fill spare capacity ahead
    # of the next decision. Selection still takes one head per series.
    configured_depth = search.get('lookahead', 1)
    depth = (configured_depth if search.get('work_conserving_search') is True and budget is None
             and type(configured_depth) is int and configured_depth > 1 and step_limit is None
             and submit_standby is not None else 1)

    def absorb(c, row, answer):
        status = _classification(answer, threshold)
        row.update(status=status, MSE_lower=answer.get('MSE_lower'), MSE_upper=answer.get('MSE_upper'),
                   evidence=answer.get('source'), job=answer.get('job'), reason=answer.get('reason'))
        if status in evaluated:
            key = (c['configuration_id'], row['vector'])
            verified = dict(answer, status=status, cost=row['cost'])
            knowledge[key] = known[key] = verified
            if (exploratory and on_feasible is not None and status == O.PASS
                    and row['smallest_grid_k'] is not None):
                on_feasible(name, c, row)

    native_parents = set()

    def register(c):
        M, cid = c['M'], c['configuration_id']
        parent = c.get('source_configuration_id')
        if parent and c.get('horizon_preparation'):
            if (c['method'], parent, M) in native_parents:
                return
        elif parent and c.get('ready') and not c.get('execution_needs'):
            native_parents.add((c['method'], parent, M))
            # A projection awaiting inputs is replaced by its collected native
            # configuration. Other parents and selected shot attempts are independent.
            for key, previous in list(groups.items()):
                pending_candidate = previous['candidate']
                if (key[0] == c['method'] and key[2] == M and pending_candidate.get('horizon_preparation')
                        and pending_candidate.get('source_configuration_id') == parent and not pending_candidate['ready']):
                    del groups[key]
        coordinates = dict(c.get('key') or {})
        if c.get('kappa_value') is not None:
            coordinates['kappa'] = c['kappa_value']
        priority = coarse_priority(search.get('coarse_axes', {}), name, coordinates)
        group = dict(candidate=c, rows=[], scope_priority=priority)
        groups[c['method'], cid or c['name'], M] = group
        if c.get('unsupported_horizon'):
            return
        prices = c.get('prices')
        known_prices = prices is not None and len(prices) == 2 * (M + 1) and all(p >= 0 for p in prices)
        ready = bool(cid) and c['ready'] and not c.get('execution_needs') and (known_prices or exploratory)
        kappa = c.get('kappa_value')
        if policy['compare_paper'] and name.startswith('PR_') and kappa is None:
            group['paper_comparison_pending'] = 'ABSOLUTE_KAPPA_MISSING'
        try:
            menu = allocation_menu(M, policy, kappa=kappa)
        except (DecimalException, OverflowError, ValueError) as error:
            group['paper_comparison_pending'] = type(error).__name__
            menu = allocation_menu(M, policy)
        for vector, aliases in menu.items():
            grid = [a['k'] for a in aliases if a['kind'] == 'GRID']
            nx, ny = O.split(vector)
            row = dict(configuration_id=cid, candidate=c['name'], method=name, M=M, vector=vector, N_X=nx, N_Y=ny,
                       origin=dict(kind=RULE, M=M, aliases=aliases),
                       aliases=aliases, smallest_grid_k=min(grid) if grid else None,
                       cost=O.cost(vector, prices) if known_prices else None,
                       status='PENDING' if ready else 'INPUT_WAITING')
            saved = known.get((cid, vector))
            if saved is not None and (known_prices or exploratory):
                absorb(c, row, saved)
            elif (cid, vector) in held_keys:
                row.update(status='TECHNICAL_HOLD', reason='INHERITED_Q3_HOLD')
            group['rows'].append(row)
            if ready and (cid, vector) not in held_keys:
                registered[cid, vector] = (c, row)

    def best_for(group, grid):
        return min((row for row in group['rows'] if row['status'] == O.PASS and row['cost'] is not None and
                    (row['smallest_grid_k'] is not None) == grid),
                   key=lambda r: (r['cost'], r['smallest_grid_k'] or 0, r['vector']), default=None)

    priorities = {}

    def allocation_order(row):
        return (priorities.get(id(row), 1), groups[name, row['configuration_id'], row['M']]['scope_priority'],
                row['cost'] is None, row['cost'] if row['cost'] is not None else sum(row['vector']),
                row['M'], row['configuration_id'])

    def pending_allocations(group, limit=1):
        best = best_for(group, True)
        feasible_k = min((r['smallest_grid_k'] for r in group['rows']
                          if r['status'] == O.PASS and r['smallest_grid_k'] is not None), default=None) if exploratory else None
        grid = []
        reference = []
        for row in group['rows']:
            if row['status'] != 'PENDING':
                continue
            k = row['smallest_grid_k']
            if k is None:
                reference.append(row)
            elif exploratory and feasible_k is not None and k > feasible_k:
                # A higher common multiplier cannot lower any integer shot count.
                row.update(status='COST_PRUNED', reason='OWN_GRID_PASS_DOMINATES_INTEGER_COUNTS')
            elif best is not None and (row['cost'], k) >= (best['cost'], best['smallest_grid_k']):
                row.update(status='COST_PRUNED', reason='OWN_GRID_PASS_COST_OR_TIE')
            else:
                grid.append(row)
        if grid:
            if feasibility_first:
                all_grid = sorted((r for r in group['rows'] if r['smallest_grid_k'] is not None),
                                  key=lambda r: r['smallest_grid_k'])
                if feasible_k is None:
                    # Probe the largest and then a middle distinct vector.
                    # A FAIL does not imply anything about other shot counts.
                    probes = []
                    for probe in (all_grid[-1], all_grid[len(all_grid)//2]):
                        if probe['status'] == 'PENDING' and probe in grid and probe not in probes:
                            priorities[id(probe)] = 0
                            probes.append(probe)
                    if probes:
                        # The default fixed scope still runs one probe at a
                        # time. Only a new explicitly configured unlimited
                        # scope can reserve an independent second probe.
                        prefetch = search.get('feasibility_probe_prefetch') is True and budget is None
                        return probes[:limit] if prefetch else probes[:1]
                    for r in grid:
                        priorities[id(r)] = 2
                else:
                    for r in grid:
                        priorities[id(r)] = 1
            order = lambda r: (*allocation_order(r), r['smallest_grid_k'])
            return [min(grid, key=order)] if limit == 1 else sorted(grid, key=order)[:limit]
        return [min(reference, key=lambda r: (*allocation_order(r), r['vector']))] if reference else []

    def record(c, row, answer, *, selected):
        nonlocal used, fatal
        used += answer.get('calls', 0)
        absorb(c, row, answer)
        group = groups.get((c['method'], c['configuration_id'] or c['name'], c['M']))
        if group is not None:
            group.pop('pending_cache', None)
        trace.append(dict(row, calls=answer.get('calls', 0), answer_source=answer.get('source'),
                          cls='SCALED_SHOTS', selected_by_rule=selected))
        if row['status'] in ('TECHNICAL_HOLD', 'IN_PROGRESS', 'UNKNOWN'):
            holds.append(dict(row))
        if row['status'] == 'UNKNOWN' and answer.get('failure_scope') == 'GLOBAL':
            fatal = True

    def collect_completed():
        changed = False
        if read_completed is not None:
            for key, answer in read_completed(lambda key: registered[key][1] if key in registered else None):
                c, row = registered[key]
                if row['status'] in evaluated:
                    continue
                record(c, row, answer, selected=False)
                changed = True
        return changed

    for c in candidates:
        register(c)
    if refresh_completions is not None:
        refresh_completions({cid for cid, _ in registered})
    while True:
        if refresh_ready is not None:
            added, values = refresh_ready()
            for key, value in values.items():
                known.setdefault(key, value)
            for c in added:
                register(c)
            if added and refresh_completions is not None:
                refresh_completions({c['configuration_id'] for c in added})
        if collect_completed():
            deferred.clear()
        if fatal:
            break
        if price_work_pending is not None and price_work_pending():
            requested_stop = 'WORK_SLICE_YIELDED'
            break
        # A pending dispatch does not change any series. Recompute a group's
        # head only after its own evidence changes or it is registered anew.
        series = []
        for group in groups.values():
            if 'pending_cache' not in group:
                group['pending_cache'] = pending_allocations(group, depth)
            if group['pending_cache']:
                series.append(group['pending_cache'])
        if retire_allocations is not None:
            terminal = {key for key, (_, row) in registered.items()
                        if row['status'] in evaluated or row['status'] in ('COST_PRUNED', 'TECHNICAL_HOLD')}
            if terminal - retired:
                retire_allocations(terminal - retired)
                retired.update(terminal)
        heads = [rows[0] for rows in series]
        proposals = [row for row in heads if (row['configuration_id'], row['vector']) not in deferred]
        if not proposals and deferred:
            if await_dispatch is not None and await_dispatch(tuple(deferred.values())):
                deferred.clear()
                continue
            break
        if not proposals:
            if depth > 1:
                submit_standby(None, [])
            if await_ready is not None and not (flagged_stop is not None and flagged_stop()) and await_ready():
                continue
            break
        references = []
        if submit_standby is not None and search.get('prefetch_independent_references') and budget is None:
            head_ids = {id(row) for row in heads}
            references = [row for group in groups.values() for row in group['rows']
                          if row['status'] == 'PENDING' and row['smallest_grid_k'] is None and id(row) not in head_ids]
            references.sort(key=allocation_order)
        if step_limit is not None and steps >= step_limit:
            requested_stop = 'STEP_LIMIT'
            break
        draining = False
        if work_slice is not None and completed_steps >= work_slice:
            mode = search.get('ready_work_at_slice')
            can_continue = mode == 'refill' and budget is None and submit_standby is not None
            if mode == 'drain' and budget is None and free is not None:
                from itertools import chain
                # Yield to the adaptive outer planner after the reservations.
                proposals = [row for row in chain(proposals, references) if free(row)]
                references = []
                can_continue = bool(proposals)
                draining = True
            if not can_continue:
                requested_stop = 'WORK_SLICE_YIELDED'
                break
        proposals.sort(key=lambda r: ((r['configuration_id'], r['vector']) in yielded,
                                      free is None or not free(r), r['smallest_grid_k'] is None,
                                      *allocation_order(r)))
        # A yielded job still owns its allocation and counts against the queue
        # budget. At the cap, keep collecting existing jobs ahead of new work.
        capped = budget is not None and spent + used >= budget
        choice = next((r for r in proposals if not capped or (free is not None and free(r))), proposals[0])
        is_free = free is not None and free(choice)
        if flagged_stop is not None and flagged_stop():
            pending = choice
            requested_stop = 'USER_REQUESTED_STOP'
            break
        if not is_free and budget is not None and spent + used >= budget:
            pending = choice
            break
        if submit_standby is not None and science_approved and not capped:
            standby = [r for r in proposals if r is not choice]
            if deferred and not draining:
                standby.extend(r for r in heads if (r['configuration_id'], r['vector']) in deferred)
            standby += references
            if depth > 1 and not draining:
                # Give every series its next point before opening a deeper
                # layer. Within a series the cheapest unresolved points lead.
                from itertools import zip_longest
                for layer in zip_longest(*(rows[1:] for rows in series)):
                    standby.extend(sorted((r for r in layer if r is not None),
                                          key=allocation_order))
            submit_standby(choice, standby)
        answer = answer_now(choice) if answer_now is not None else (
            backend.answer(choice) if backend is not None else None)
        if answer is not None and answer.get('status') == 'DISPATCH_PENDING':
            deferred[choice['configuration_id'], choice['vector']] = dict(answer, configuration_id=choice['configuration_id'])
            continue
        if answer is not None and answer.get('status') == 'RESELECT_READY':
            yielded.add((choice['configuration_id'], choice['vector']))
            continue
        if answer is None or answer.get('status') in ('BUDGET_EXHAUSTED', 'STOP_REQUESTED'):
            if (answer is not None and answer['status'] == 'BUDGET_EXHAUSTED' and deferred
                    and await_dispatch is not None and await_dispatch(tuple(deferred.values()))):
                deferred.clear()
                continue
            pending = choice
            if answer is not None:
                requested_stop = ('USER_REQUESTED_STOP' if answer['status'] == 'STOP_REQUESTED'
                                  else 'Q3_BUDGET_EXHAUSTED')
            break
        c = groups[name, choice['configuration_id'], choice['M']]['candidate']
        steps += 1
        record(c, choice, answer, selected=True)
        if choice['status'] in evaluated:
            completed_steps += 1
        if fatal:
            break

    def summarize():
        rows = []
        winners = []
        unresolved = {'PENDING', 'INPUT_WAITING', 'TECHNICAL_HOLD', 'IN_PROGRESS', 'UNKNOWN', 'INCONCLUSIVE'}
        for group in groups.values():
            c = group['candidate']
            pending_allocations(group)
            best = best_for(group, True)
            compare = best_for(group, False)
            blockers = [r for r in group['rows'] if r['smallest_grid_k'] is not None and r['status'] in unresolved
                        and (best is None or r['cost'] is None
                             or (r['cost'], r['smallest_grid_k']) < (best['cost'], best['smallest_grid_k']))]
            verdict = ('EVALUATED_MINIMUM' if blockers else 'GRID_MINIMUM_CONFIRMED') if best is not None else (
                'NO_CONFIRMED_GRID_POINT' if blockers else 'NO_GRID_POINT_MEETS_LIMIT')
            feasible = None
            if exploratory:
                feasible = min((r for r in group['rows'] if r['status'] == O.PASS
                                and r['smallest_grid_k'] is not None),
                               key=lambda r: r['smallest_grid_k'], default=None)
                if best is not None:
                    verdict = 'PROVISIONAL_GRID_MINIMUM' if not blockers else 'PROVISIONAL_EVALUATED_MINIMUM'
                elif feasible is not None:
                    verdict = 'MSE_PASS_COST_UNAVAILABLE'
            if c.get('unsupported_horizon'):
                verdict = 'OUTSIDE_QUALIFIED_EVALUATOR_DOMAIN'
            key = dict(c.get('key', {}), M=c['M'])
            spec = (c.get('generated_configuration') or {}).get('spec', {})
            for field in ('formula', 'delta_pf', 'h', 'L_D'):
                if spec.get(field) is not None:
                    key[field] = spec[field]
            if c.get('kappa_value') is not None:
                key['absolute_kappa'] = str(c['kappa_value'])
            row = dict(configuration_id=c['configuration_id'], source_configuration_id=c.get('source_configuration_id'),
                       method=name, M=c['M'], configuration=key, configuration_source=c.get('configuration_source'),
                       needs=c['needs'] + c.get('execution_needs', []),
                       price_status=c.get('price_status'), prices=c.get('prices'),
                       price_evidence=c.get('price_evidence'), price_quote=c.get('price_quote'),
                       signal_source=c.get('signal_source'),
                       best=best, minimum_status=verdict, unresolved_competitors=len(blockers), comparison_best=compare,
                       comparison_delta_RZ=None if best is None or compare is None else compare['cost']-best['cost'],
                       paper_comparison_pending=group.get('paper_comparison_pending'),
                       allocations=group['rows'], promising=len(blockers), required_rounds=c.get('required_rounds'))
            if exploratory:
                row.update(mse_feasible=feasible, cost_status='PROVISIONAL' if c.get('prices') else 'UNAVAILABLE',
                           price_source=c.get('price_source'))
            if 'horizon_preparation' in c:
                row['horizon_preparation'] = c['horizon_preparation']
            rows.append(row)
            if best is not None:
                winners.append((best['cost'], c['M'], c['configuration_id'], best, c))

        def overall(winners):
            if winners:
                cost, _, _, row, c = min(winners, key=lambda w: w[:3])
                return dict(cost=cost, vector=row['vector'], candidate=c, record=knowledge[c['configuration_id'], row['vector']])

        winning = overall(winners)
        baseline = overall([w for w in winners if w[1] == 12])
        def external(value):
            if value is None:
                return None
            c, record = value['candidate'], value['record']
            vector = value['vector']
            row = next((r for r in rows if r['configuration_id'] == c['configuration_id'] and r['M'] == c['M']), None)
            return dict(configuration_id=c['configuration_id'], M=c['M'], configuration=c['key'],
                N_X=O.split(vector)[0], N_Y=O.split(vector)[1], cost=value['cost'],
                MSE_lower=record['MSE_lower'], MSE_upper=record['MSE_upper'], evidence=record['source'],
                aliases=None if row is None or row['best'] is None or row['best']['vector'] != vector else row['best']['aliases'])
        def scope_status(best, groups):
            if best is None:
                return 'NO_CONFIRMED_GRID_POINT'
            limit = best['cost']
            can_improve = any(r['smallest_grid_k'] is not None and
                              (r['status'] in unresolved or exploratory and r['status'] == O.PASS and r['cost'] is None)
                              and (r['cost'] is None or r['cost'] < limit)
                              for g in groups for r in g['allocations'])
            if exploratory:
                return 'PROVISIONAL_EVALUATED_MINIMUM' if can_improve else 'PROVISIONAL_GRID_MINIMUM'
            return 'EVALUATED_MINIMUM' if can_improve else 'MINIMUM_WITHIN_REGISTERED_CANDIDATES'
        coverage = scope_coverage(method_scope, policy['M'], rows)
        has_pending = any(r['status'] == 'PENDING' for g in rows for r in g['allocations'])
        budget_exhausted = (pending is not None and has_pending and budget is not None
                            and spent + used >= budget)
        if fatal:
            stop = 'UNKNOWN_RESULT'
        elif requested_stop and (requested_stop not in ('WORK_SLICE_YIELDED', 'Q3_BUDGET_EXHAUSTED') or has_pending):
            stop = requested_stop
        elif budget_exhausted:
            stop = 'Q3_BUDGET_EXHAUSTED'
        elif (not rows or coverage['missing_count']) or any(r['status'] in unresolved for g in rows for r in g['allocations']) or (
                exploratory and any(row.get('mse_feasible') is not None and row.get('cost_status') == 'UNAVAILABLE' for row in rows)):
            stop = 'UNRESOLVED_SCALED_SCOPE'
        elif any(row['minimum_status'] == 'OUTSIDE_QUALIFIED_EVALUATOR_DOMAIN' for row in rows):
            stop = 'SCALED_SCOPE_INCLUDES_UNSUPPORTED_HORIZON'
        else:
            stop = 'EXPLORATORY_GRID_COMPLETE' if exploratory else 'SCALED_SHOT_GRID_COMPLETE'
        horizon_requests = {}
        for row in rows:
            missing = row.get('required_rounds')
            if not missing:
                continue
            # Base inputs belong to the existing candidate preparation path.
            # All shot vectors for this source share one horizon request.
            preparation = row.get('horizon_preparation')
            rounds = preparation['extension_rounds'] if preparation is not None else missing
            key = (row['source_configuration_id'], row['configuration_id'], row['M'])
            horizon_requests[key] = dict(kind='HORIZON_INPUTS_MISSING', method=name, M=row['M'],
                configuration_id=row['configuration_id'], source_configuration_id=row['source_configuration_id'],
                configuration=row['configuration'], signal_rounds=rounds,
                price_round_axes=[dict(m=m, axis=axis) for m in rounds for axis in ('X','Y')])
            if preparation is not None:
                horizon_requests[key].update(preparation)
        input_requests = list(horizon_requests.values())
        compared = [(r['comparison_best']['cost'], r['M'], r['configuration_id'], r['comparison_best'])
                    for r in rows if r['comparison_best'] is not None]
        reference = min(compared, key=lambda r:r[:3])[3] if compared else None
        shortlist = {}
        if exploratory:
            by_horizon = {}
            estimated_order = []
            for cost, M, cid, _, _ in sorted(winners, key=lambda w: w[:3]):
                lead = by_horizon.setdefault(M, cost)
                estimated_order.append(dict(configuration_id=cid, M=M, cost=cost, leads_horizon=cost == lead))
            shortlist = dict(cost_stage='EXPLORATORY_SHORTLIST', estimated_order=estimated_order,
                unpriced_feasible=[dict(configuration_id=r['configuration_id'], M=r['M'])
                                   for r in rows if r.get('mse_feasible') and r['best'] is None])
        return dict(rule=RULE, incumbent_scope='CONFIGURATION_AND_HORIZON', incumbent=winning,
                    input_requests=input_requests,
                    trace=trace, held=holds, dispatch_wait=list(deferred.values()), pending_request=pending if pending is None or pending['status'] == 'PENDING' else None, new_calls=used, stop_reason=stop,
                    by_configuration=rows, scaled_shots=dict(policy=policy, scope_coverage=coverage, M12_best=external(baseline),
                        M12_minimum_status=scope_status(baseline, [r for r in rows if r['M'] == 12]),
                        all_M_best=external(winning), all_M_minimum_status=scope_status(winning, rows),
                        comparison_only_best=reference,
                        comparison_only_delta_RZ=None if reference is None or winning is None else reference['cost']-winning['cost'],
                        legacy_free_allocation_reference=external(legacy), **shortlist))

    if after_drain is not None:
        def finalize():
            if collect_completed():
                return summarize()

        after_drain(finalize)
    return summarize()


def horizon_candidates(candidates, problem, policy, executor=None, catalog=None):
    """Keep base identities for archival reuse and bind shorter views to their owners."""
    by_id = {r['configuration_id']: r for r in (catalog or {}).get('candidates', [])}
    exploratory = problem.get('search', {}).get('exploratory_cost') is True
    sources = {(c['method'], c.get('source_configuration_id') or c.get('configuration_id'),
                c.get('M', problem['rpe']['M'])) for c in candidates if c.get('configuration_id')}
    result, added_routes = [], []
    def unsupported(method, M, route):
        if problem.get('problem', {}).get('system') != 'H4' or (route or {}).get('kind') == 'artificial':
            return False
        from run_rpe_optimizer_q3_job import M13_ROUTES
        return M > 13 or (M == 13 and not any(name == method for name, _ in M13_ROUTES))
    for original in candidates:
        original_id = original['configuration_id']
        original_M = original.get('M', problem['rpe']['M'])
        c = dict(original, M=original_M)
        known = by_id.get(original_id) or {}
        c['key'] = dict(c['key'])
        for field in ('formula','delta_pf','h'):
            if known.get(field) is not None:
                c['key'].setdefault(field,known[field])
        kappa = known.get('kappa') or c.get('absolute_kappa')
        if c.get('kappa_value') is None and isinstance(kappa, dict):
            v = kappa.get('value', kappa)
            if isinstance(v, dict) and 'numerator' in v:
                c['kappa_value'] = str(Fraction(int(v['numerator']), int(v['denominator'])))
        if c.get('kappa_value') is None and isinstance(kappa, str):
            c['kappa_value'] = kappa
        base_route = executor.route(original_id) if executor is not None and original_id and c['ready'] else None
        if problem.get('problem', {}).get('system') == 'H4' and original_M > 12:
            fixed = (base_route or {}).get('horizon_risk_source')
            parent = (base_route or {}).get('horizon_parent')
            if not (original_M == 13 and fixed and parent and c.get('source_configuration_id')
                    and (exploratory or len(c.get('prices') or ()) == 28)):
                c['execution_needs'] = list(dict.fromkeys([*c.get('execution_needs', []),
                    'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN']))
        if original_M in policy['M']:
            if unsupported(c['method'], original_M, base_route):
                c.update(unsupported_horizon=True, ready=False)
            result.append(c)
        for M in policy['M']:
            if M == original_M or (original_id and (c['method'], original_id, M) in sources) or original.get('source_configuration_id'):
                continue
            item = dict(c, M=M, name=c['name'] + ':M=' + str(M), source_configuration_id=original_id,
                        key=dict(c['key'], M=M), needs=list(c['needs']), execution_needs=list(c.get('execution_needs', [])))
            if original_id:
                item['configuration_id'] = projection_id(original_id, original_M, M)
            if unsupported(c['method'], M, base_route):
                item.update(ready=False, prices=None, unsupported_horizon=True, required_rounds=[],
                            price_status='OUTSIDE_QUALIFIED_EVALUATOR_DOMAIN')
                item['execution_needs'].append('NO_QUALIFIED_Q3_ROUTE_FOR_METHOD_AND_HORIZON')
                result.append(item)
                continue
            price_count = 2 * (original_M + 1)
            if (M < original_M and c['ready'] and original_id
                    and (exploratory or c.get('prices') is not None and len(c['prices']) == price_count)):
                if c.get('prices') is not None and len(c['prices']) == price_count:
                    item['prices'] = c['prices'][:2 * (M + 1)]
                    item['price_status'] = 'REUSED_SOURCE_HORIZON_PREFIX'
                item['price_axes'] = c.get('price_axes', [])[:2 * (M + 1)]
                native = executor.route(item['configuration_id']) if executor is not None else None
                native_ids = (native or {}).get('configuration_ids')
                if isinstance(native_ids, (list, tuple)) and item['configuration_id'] in native_ids:
                    item['execution_needs'] = [need for need in item['execution_needs']
                                               if need != 'Q3_ROUTE_CONFIGURATION_OR_TEMPLATE']
                    hold = executor.refresh_execution_hold(item['configuration_id'])
                    if hold:
                        item['execution_hold'] = hold
                        item['execution_needs'].append('SAVED_JOB_REVIEW_REQUIRED')
                elif base_route is not None:
                    base_rule = executor.expected_for(original_id, base_route)
                    projection = dict(source_configuration_id=original_id, source_M=original_M, M=M)
                    added_routes.append(dict(base_route, configuration_ids=[item['configuration_id']],
                                             horizon_projection=projection))
                    executor.expected[item['configuration_id']] = dict(base_rule, M=M)
                elif executor is not None:
                    item['execution_needs'].append('SOURCE_HORIZON_ROUTE_UNAVAILABLE')
            else:
                extension = list(range(original_M + 1, M + 1))
                item.update(ready=False, prices=None, price_status='HORIZON_INPUTS_MISSING',
                            required_rounds=extension if c['ready'] else list(range(M + 1)),
                            needs=list(dict.fromkeys(item['needs'] + ['SIGNAL_AND_PRICE_FOR_HORIZON'])))
                if problem.get('problem', {}).get('system') == 'H4' and M > 12:
                    item['execution_needs'].append('OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN')
                item['horizon_preparation'] = dict(source_M=original_M, base_inputs_ready=c['ready'],
                    base_preparation_needs=list(c['needs']), extension_rounds=extension,
                    evaluator_preflight_needs=list(dict.fromkeys(item['execution_needs'])))
            result.append(item)
    if added_routes:
        ids = {v['configuration_ids'][0] for v in added_routes}
        remaining = [v for v in executor.routes.get(executor.method, [])
                     if not (v.get('horizon_projection') and v['configuration_ids'][0] in ids)]
        executor.routes[executor.method] = added_routes + remaining
    return result
