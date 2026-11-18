"""Pure candidate ordering from public metadata and completed observations only.

The caller owns candidate generation, execution, budgets and scientific gates.
No archive, mapper, outcome file or calibration registry is accessed here.
"""
from fractions import Fraction
import math
import statistics

PASS = {'PASS', 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'}
FAIL = {'FAIL', 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'}


def _seconds(value):
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def completed_observations(observations, *, now, exclude_id=None):
    """Exclude unfinished/future records and the candidate's own held-out record."""
    return [o for o in observations if o.get('finished_at') is not None
            and o['finished_at'] <= now and o.get('id') != exclude_id]


def estimate_seconds(action, *, now, observations=()):
    """Estimate preparation plus Q3; absence of evidence remains None.

    SHOT_Q3.seconds is Q3 service time. A NEW_SCHEDULE observation may instead
    include preparation, so its Q3 leg requires an explicit q3_seconds field.
    Profile equality is deliberate: worker/concurrency changes are not measured
    by old timings. Explicit action priors are assumptions, never measurements.
    """
    prep = action.get('preparation') or {}
    reused = (prep.get('signal_reuse') is True and prep.get('price_reuse') is True
              and prep.get('requires_calibration') is False)
    preparation = 0.0 if reused else _seconds(prep.get('seconds'))
    prep_source = 'REUSED_INPUTS_NO_PREPARATION' if reused else prep.get('source', 'PREPARATION_TIME_UNKNOWN')
    q3 = _seconds(action.get('q3_seconds'))
    source, timing_ids = 'EXPLICIT_ASSUMED_PRIOR' if q3 is not None else 'Q3_TIME_UNKNOWN', []
    if q3 is None:
        records = []
        for o in completed_observations(observations, now=now, exclude_id=action['id']):
            if o.get('method') != action.get('method') or o.get('profile') != action.get('profile'):
                continue
            seconds = _seconds(o.get('q3_seconds') if 'q3_seconds' in o else
                               o.get('seconds') if o.get('kind') == 'SHOT_Q3' else None)
            if seconds is not None:
                records.append((o, seconds))
        same = [(o, t) for o, t in records if action.get('configuration_id') is not None
                and o.get('configuration_id') == action['configuration_id']]
        if same:
            records = same
        if records:
            q3 = statistics.median(t for _, t in records)
            source = 'COMPLETED_CONFIGURATION_MEDIAN' if same else 'COMPLETED_METHOD_PROFILE_MEDIAN'
            timing_ids = [o['id'] for o, _ in records]
    total = None if preparation is None or q3 is None else preparation + q3
    reasons = []
    if preparation is None:
        reasons.append('PREPARATION_TIME_UNKNOWN')
        if prep.get('requires_calibration') is True:
            reasons.append('CALIBRATION_REQUIRED_TIME_UNKNOWN')
    if q3 is None:
        reasons.append('Q3_TIME_UNKNOWN_NO_FIXED_OR_COMPLETED_PRIOR')
    return dict(preparation_seconds=preparation, preparation_source=prep_source, q3_seconds=q3,
                q3_source=source, total_seconds=total, timing_observation_ids=timing_ids, unknown_reasons=reasons)


def _pass_rate(action, observations, now):
    records = [o for o in completed_observations(observations, now=now, exclude_id=action['id'])
               if o.get('method') == action.get('method') and o.get('status') in PASS | FAIL]
    same = [o for o in records if action.get('configuration_id') is not None
            and o.get('configuration_id') == action['configuration_id']]
    if same:
        records = same
    passed = sum(o['status'] in PASS for o in records)
    return dict(value=Fraction(passed+1, len(records)+2), passed=passed, samples=len(records),
                source='COMPLETED_CONFIGURATION_BETA_1_1' if same else
                       'COMPLETED_METHOD_BETA_1_1' if records else 'NEUTRAL_BETA_1_1',
                observation_ids=[o['id'] for o in records],
                limitation='Empirical ordering feature with shrinkage; no MSE monotonicity or pass guarantee.')


class Selector:
    """ROUND05 preserves a_rank; the opt-in alternative explores periodically."""
    def __init__(self, policy='ROUND05', explore_every=4):
        if policy not in ('ROUND05', 'IMPROVEMENT_PER_TIME_V1'):
            raise ValueError('Unknown candidate-order policy')
        if type(explore_every) is not int or explore_every < 1:
            raise ValueError('Positive integer explore_every required')
        self.policy, self.explore_every = policy, explore_every
        self.choices, self.selected = 0, {}

    def snapshot(self):
        return dict(policy=self.policy, explore_every=self.explore_every, choices=self.choices, selected=dict(self.selected))

    @classmethod
    def restore(cls, state):
        selector = cls(state['policy'], state['explore_every'])
        selector.choices, selector.selected = state['choices'], dict(state['selected'])
        return selector

    def choose(self, actions, *, method_best, tau, now, observations=()):
        actions = list(actions)
        if len({a['id'] for a in actions}) != len(actions):
            raise ValueError('Action IDs must be unique; keep exact width identities distinct')
        order = sorted(range(len(actions)), key=lambda i: (tuple(actions[i]['a_rank']), i))
        rows = []
        for i, action in enumerate(actions):
            timing = estimate_seconds(action, now=now, observations=observations)
            probability = _pass_rate(action, observations, now)
            point = action.get('cost')
            saving = None if point is None or method_best is None else max(Fraction(method_best)-Fraction(point), Fraction(0))
            upper = action.get('parent_MSE_upper')
            headroom = None if upper is None or tau is None else (Fraction(tau)-Fraction(upper))/Fraction(tau)
            total = timing['total_seconds']
            score = None if saving is None or total is None or total == 0 else saving*probability['value']/Fraction(str(total))
            rows.append(dict(id=action['id'], kind=action['kind'], method=action.get('method'), L_D=action.get('L_D'),
                width=action.get('width'), a_rank=list(action['a_rank']), point_cost=point,
                profile=action.get('profile'), preparation=dict(action.get('preparation') or {}),
                explicit_q3_prior=action.get('q3_seconds'),
                price_floor=action.get('price_floor'), price_floor_role='BOUND_ONLY_NEVER_PREDICTED_POINT_COST',
                parent_cost=action.get('parent_cost'), parent_MSE_upper=upper, parent_headroom=headroom,
                immediate_saving=saving, pass_rate=probability, timing=timing, score=score,
                zero_time_positive_saving=total == 0 and saving is not None and saving > 0,
                previous_selections=self.selected.get(action['id'], 0), untried=action.get('untried', False)))
        baseline, chosen, mode = (order[0] if order else None), None, 'EMPTY'
        if order:
            chosen, mode = baseline, 'ROUND05_A_RANK'
            if self.policy == 'IMPROVEMENT_PER_TIME_V1':
                unknown = [i for i in order if rows[i]['immediate_saving'] is None or rows[i]['timing']['total_seconds'] is None]
                untried = [i for i in order if actions[i].get('untried')]
                # Fresh shot IDs must not crowd unknown schedules out forever.
                explore = ([i for i in unknown if actions[i]['kind'] == 'NEW_SCHEDULE'] or unknown or
                           [i for i in untried if actions[i]['kind'] == 'NEW_SCHEDULE'] or untried)
                if (self.choices+1) % self.explore_every == 0 and explore:
                    chosen = min(explore, key=lambda i: (rows[i]['previous_selections'], order.index(i)))
                    mode = 'PERIODIC_UNTRIED_OR_UNKNOWN_EXPLORATION'
                else:
                    timed = [i for i in order if rows[i]['zero_time_positive_saving'] or
                             rows[i]['score'] is not None and rows[i]['score'] > 0]
                    def head(i):
                        return rows[i]['parent_headroom'] if rows[i]['parent_headroom'] is not None else Fraction(0)
                    if timed:
                        chosen = min(timed, key=lambda i: (-rows[i]['zero_time_positive_saving'],
                            -(rows[i]['score'] or 0), -head(i), order.index(i)))
                        mode = 'IMMEDIATE_SAVING_X_EMPIRICAL_PASS_PER_ESTIMATED_TIME'
                    else:
                        chosen = min(order, key=lambda i: (-(rows[i]['immediate_saving'] or 0), -head(i), order.index(i)))
                        mode = 'POINT_SAVING_A_RANK_FALLBACK_TIMING_OR_POSITIVE_SCORE_UNAVAILABLE'
            self.choices += 1
            identity = actions[chosen]['id']
            self.selected[identity] = self.selected.get(identity, 0)+1
        return dict(chosen=None if chosen is None else actions[chosen], trace=dict(policy=self.policy, selection=self.choices,
            now=now, baseline_id=None if baseline is None else actions[baseline]['id'],
            chosen_id=None if chosen is None else actions[chosen]['id'], changed_from_a_rank=chosen != baseline,
            reason=mode, method_best=method_best, tau=tau, ranks=rows,
            invariant='Score never rejects an allocation, proves infeasibility or assumes MSE monotonicity.'))
