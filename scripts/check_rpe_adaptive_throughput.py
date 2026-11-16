"""Artificial regressions and clock models for opt-in HYBRID / allocation exchanges.

No real signal, calibration, compilation, Q3 or scientific MSE is evaluated.
The clock models reuse the actual V2 candidate/adoption code.  They report
modeled elapsed and job service time separately, never label the latter CPU.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
import check_rpe_priority_search as CP


class Clock:
    """Causal fake executor. Only answer() reveals artificial result values."""

    def __init__(self, function, slots, duration=10, completed=None, max_launches=None):
        self.function, self.slots, self.duration = function, slots, duration
        self.t, self.finish_times, self.speculative, self.adopted = 0.0, {}, set(), set()
        self.previews = [];self.completed = dict(completed or {});self.max_launches = max_launches

    def key(self, p):
        return p['configuration_id'], tuple(p['vector'])

    def active(self):
        return sum(f > self.t for f in self.finish_times.values())

    def launch(self, p):
        key = self.key(p)
        if key in self.completed or key in self.finish_times:
            return True
        if self.max_launches is not None and len(self.finish_times) >= self.max_launches:
            return False
        if self.active() >= self.slots:
            return False
        self.finish_times[key] = self.t + self.duration
        if p.get('cls') == 'CHAIN':
            self.speculative.add(key)
        return True

    def prefetch(self, proposals):
        self.previews.append([(p['configuration_id'], tuple(p['vector']), p.get('cls')) for p in proposals])
        for p in proposals:
            if not self.launch(p):
                break

    def state(self):
        done = {k for k in self.speculative if self.finish_times[k] <= self.t}
        return dict(capacity=self.slots, active_jobs=self.active(), speculative_launched=len(self.speculative),
                    speculative_running=len(self.speculative - done), speculative_completed=len(done),
                    speculative_adopted=len(self.adopted & self.speculative))

    def answer(self, proposal):
        key = self.key(proposal)
        if key in self.completed:
            hit = dict(self.completed[key], calls=0);self.adopted.add(key);return hit
        if key not in self.finish_times:
            if self.active() >= self.slots:
                self.t = min(f for f in self.finish_times.values() if f > self.t)
            if not self.launch(proposal):
                return None
        self.t = max(self.t, self.finish_times[key]);self.adopted.add(key)
        lo, up = self.function(key[0], key[1])
        return dict(status=O.PASS if up <= CP.TAU else O.FAIL, MSE_lower=lo, MSE_upper=up,
                    source='ARTIFICIAL_CLOCK', calls=1)

    def measurement(self):
        return dict(kind='ARTIFICIAL_CONSTANT_DURATION_CLOCK',
                    elapsed_to_last_adoption=self.t,
                    elapsed_including_outstanding=max([self.t] + list(self.finish_times.values())),
                    summed_job_service_seconds=len(self.finish_times) * self.duration,
                    total_CPU_seconds=None, CPU_reason='No real CPU worker in a clock model',
                    jobs_launched=len(self.finish_times), jobs_adopted=len(self.adopted),
                    complete_outcome_reuses=len(self.adopted & self.completed.keys()),
                    speculative_launched=len(self.speculative), speculative_adopted=len(self.adopted & self.speculative),
                    unused_launched=len(self.finish_times.keys() - self.adopted))


def fixtures(count):
    base = sum(w / n for w, n in zip(CP.W, CP.CAPS))
    candidates = [CP.config('C%d' % i, i + 1, '1', Fraction(100 + i, 100)) for i in range(count)]
    function = CP.mse_fn({c['configuration_id']: Fraction(4 + (i % 3), 10) / base for i, c in enumerate(candidates)})
    return candidates, function, CP.seeded(candidates, function)


def run(candidates, function, knowledge, *, adaptive, slots=8, steps=120, budget=None, completed=None, exchange=None):
    model = Clock(function, slots, completed=completed)
    search = dict(CP.SEARCH, max_steps_per_method=steps, lookahead=16, lookahead_mode='HYBRID',
                  adaptive_hybrid=adaptive, allocation_exchange=exchange)
    result = P.run_priority('PR_PAULI', deepcopy(candidates), dict(knowledge), model, search, CP.TAU,
                            CP.none_generator, budget, prefetch=model.prefetch, prefetch_state=model.state,
                            free=lambda p: model.key(p) in model.completed)
    return result, model


def signature(result):
    return [(t['configuration_id'], t['vector'], t['status']) for t in result['trace']]


def checks():
    rows = []
    def check(name, ok, **details):
        rows.append(dict(check=name, status='PASS' if ok else 'FAIL', **details))

    # Running jobs must not be counted as failed speculation, and incomplete
    # counters do not invent a measured adoption ratio.
    unknown = P.hybrid_limits(2, 16, dict(capacity=8, speculative_launched=100, speculative_adopted=0))
    low = P.hybrid_limits(2, 16, dict(capacity=8, speculative_completed=20, speculative_adopted=1))
    high = P.hybrid_limits(2, 16, dict(capacity=8, speculative_completed=20, speculative_adopted=20))
    dense = P.hybrid_limits(12, 16, dict(capacity=8, speculative_completed=20, speculative_adopted=1))
    check('completed_only_adoption_and_capacity', unknown['chain_slots'] == 6 and unknown['adoption_fraction'] is None
          and low['chain_slots'] < high['chain_slots'] and dense['chain_slots'] == 0,
          missing_counters=unknown, low_adoption=low, high_adoption=high, independent_queue=dense)

    results = {}
    for count in (1, 12):
        cands, function, knowledge = fixtures(count)
        old, old_clock = run(cands, function, knowledge, adaptive=False)
        new, new_clock = run(cands, function, knowledge, adaptive=True)
        check('same_adoptions_%d_configurations' % count,
              signature(old) == signature(new) and old['incumbent']['cost'] == new['incumbent']['cost']
              and all(len(p) <= 8 for p in new_clock.previews), adopted=len(new['trace']),
              baseline=old_clock.measurement(), adaptive=new_clock.measurement())
        # Reopen after every fake job has finished, preserving complete outcomes
        # as the real executor does.  This verifies reuse and trajectory rather
        # than asserting that a callback itself persists scheduler counters.
        cached = {}
        for cid, vector in new_clock.finish_times:
            lo, up = function(cid, vector)
            cached[(cid, vector)] = dict(status=O.PASS if up <= CP.TAU else O.FAIL, MSE_lower=lo, MSE_upper=up,
                                         source='ARTIFICIAL_COMPLETE_OUTCOME')
        again, again_clock = run(cands, function, knowledge, adaptive=True, completed=cached)
        check('resume_completed_outcomes_%d_configurations' % count,
              signature(new) == signature(again) and not cached.keys() & again_clock.finish_times.keys(),
              reused=len(again_clock.adopted & cached.keys()), fresh_jobs=len(again_clock.finish_times),
              duplicate_launches=len(cached.keys() & again_clock.finish_times.keys()),
              limit='This fake restart resets utilization counters and can issue different unused speculation; real restart counters are tested at the executor entry')
        results[str(count)] = dict(legacy=old_clock.measurement(), adaptive=new_clock.measurement(),
                                   same_adoption_order=signature(old) == signature(new))

    # Finite preview is clipped before it reaches the executor.  Legacy chains
    # appended beyond k; executor reservations still enforce the actual budget.
    cands, function, knowledge = fixtures(3)
    finite, clock = run(cands, function, knowledge, adaptive=True, budget=3, steps=3)
    check('finite_prefetch_preview_never_exceeds_remaining_budget',
          all(len(p) <= 3-i for i, p in enumerate(clock.previews)), previews=[len(p) for p in clock.previews])
    cands, function, knowledge = fixtures(40)
    large, clock = run(cands, function, knowledge, adaptive=True, slots=32, steps=1)
    check('independent_queue_can_fill_pool_larger_than_chain_depth',
          len(clock.previews[0]) == 32 and len({p[0] for p in clock.previews[0]}) == 32
          and not any(p[2] == 'CHAIN' for p in clock.previews[0]),
          configured_chain_depth=16, pool_slots=32, initial_independent_heads=len(clock.previews[0]))

    # Independent fixture with one defined feasible swap.  Every pure decrement
    # fails, while moving one shot from expensive m12 to cheap m0 passes.
    c = CP.config('E', 5, '1', 1);c['caps'] = [8] * 13
    prices = [Fraction(1)] * 26;prices[24] = prices[25] = 100;c['prices'] = prices
    base = tuple([4] * 26);target = list(base);target[24] -= 1;target[25] -= 1;target[0] += 1;target[1] += 1;target = tuple(target)
    def exchange_truth(cid, vector):
        value = Fraction(1, 2) if tuple(vector) in (base, target) else Fraction(2)
        return value, value
    kn = {('E', base): dict(status=O.PASS, MSE_lower=Fraction(1, 2), MSE_upper=Fraction(1, 2), source='SAVED', calls=0)}
    no, _ = run([c], exchange_truth, kn, adaptive=True, steps=1000)
    yes, _ = run([c], exchange_truth, kn, adaptive=True, steps=1000,
                 exchange=dict(enabled=True, max_moves=32, add_steps=[1]))
    hit = next((t for t in yes['trace'] if tuple(t['vector']) == target), None)
    check('same_configuration_exchange_escapes_decrement_local_minimum',
          no['incumbent']['vector'] == base and yes['incumbent']['vector'] == target and hit is not None
          and hit['origin']['kind'] == 'COST_DECREASING_ROUND_EXCHANGE',
          decrement_only_cost=str(no['incumbent']['cost']), exchange_cost=str(yes['incumbent']['cost']),
          Q3_to_exchange=None if hit is None else hit['step'],
          exchange_candidates_evaluated=sum(t['origin']['kind'] == 'COST_DECREASING_ROUND_EXCHANGE' for t in yes['trace']))
    generated = P.exchange_moves(base, prices, tuple([8] * 26), [1, 2], dict(enabled=True, max_moves=32))
    check('exchange_preserves_caps_and_requires_lower_cost', generated and all(min(v) >= 1 and max(v) <= 8
          and O.cost(v, prices) < O.cost(base, prices) for v, origin in generated), candidates=len(generated))
    # After a candidate is answered or absent from a saved log the next bounded
    # window must advance instead of repeatedly proposing the same top moves.
    earlier = {v for v, origin in generated}
    later = P.exchange_moves(base, prices, tuple([8] * 26), [1, 2], dict(enabled=True, max_moves=32), excluded=earlier)
    check('exchange_window_advances_past_evaluated_and_unknown', later and not earlier.intersection(v for v, origin in later),
          first_window=len(earlier), next_window=len(later))
    return dict(schema='RPE_ADAPTIVE_THROUGHPUT_ARTIFICIAL_V1', checks=rows, clock_models=results,
                passed=sum(r['status'] == 'PASS' for r in rows), failed=sum(r['status'] == 'FAIL' for r in rows),
                source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                science='Artificial truth tables and fake worker clock; real scientific calls = 0')


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    result = checks();(args.output / 'CHECKS.json').write_text(json.dumps(O.jsonable(result), indent=2) + '\n')
    for row in result['checks']:
        print(row['check'], row['status'], flush=True)
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
