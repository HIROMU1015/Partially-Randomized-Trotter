"""Reveal-after-selection replay on a fixed ready set from round04.

The verified ledger indexes REQUEST identities without reading numerical
outcomes. An OUTCOME is opened only after V2 selects that allocation. Unknown
allocations stop the replay and are reported as unevaluated, not infeasible.
This isolates scheduling/allocation order from the outer schedule-wide rule.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
import rpe_round_tools as T


class SavedClock:
    """Nonblocking fake pool with saved wall durations; no access to future MSE."""

    def __init__(self, index, seeds, slots, maximum):
        self.index, self.seeds, self.slots, self.maximum = index, seeds, slots, maximum
        self.now, self.finishes, self.speculative, self.adopted = 0.0, {}, set(), set()
        self.opened, self.missing, self.curve = {}, [], []
        durations = [r['seconds'] for r in index.values() if r['seconds'] is not None]
        self.mean_seconds = sum(durations) / len(durations) if durations else 1.0

    @staticmethod
    def key(proposal):
        return proposal['configuration_id'], tuple(proposal['vector'])

    def active(self):
        return sum(v > self.now for v in self.finishes.values())

    def launch(self, proposal):
        key = self.key(proposal)
        if key in self.seeds or key in self.finishes:
            return True
        if len(self.finishes) >= self.maximum or self.active() >= self.slots:
            return False
        seconds = self.index.get(key, {}).get('seconds')
        self.finishes[key] = self.now + (self.mean_seconds if seconds is None else seconds)
        if proposal.get('cls') == 'CHAIN':
            self.speculative.add(key)
        return True

    def prefetch(self, proposals):
        for proposal in proposals:
            if not self.launch(proposal):
                break

    def state(self):
        complete = {k for k in self.speculative if self.finishes[k] <= self.now}
        return dict(capacity=self.slots, active_jobs=self.active(), speculative_launched=len(self.speculative),
                    speculative_completed=len(complete), speculative_running=len(self.speculative - complete),
                    speculative_adopted=len(self.adopted & self.speculative))

    def answer(self, proposal):
        key = self.key(proposal)
        if key in self.seeds:
            return dict(self.seeds[key], calls=0)
        # Crucial: choice is fixed before its numerical OUTCOME is opened.
        row = self.index.get(key)
        if row is None:
            self.missing.append(dict(configuration_id=key[0], vector=list(key[1]), cls=proposal.get('cls'),
                origin=proposal.get('origin'), status='UNEVALUATED_NO_ANSWER_IN_LOG'))
            return None
        if key not in self.finishes:
            if self.active() >= self.slots:
                self.now = min(f for f in self.finishes.values() if f > self.now)
            if not self.launch(proposal):
                return None
        self.now = max(self.now, self.finishes[key])
        path = row['outcome'];data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Saved outcome changed since fixed ledger: ' + str(path))
        value = json.loads(data);self.opened[key] = str(path);self.adopted.add(key)
        self.curve.append(dict(key=key, seconds=self.now, launched=len(self.finishes)))
        return dict(status=value['status'], MSE_lower=None if value.get('MSE_lower') is None else O.frac(value['MSE_lower']),
                    MSE_upper=None if value.get('MSE_upper') is None else O.frac(value['MSE_upper']),
                    source='SAVED_OUTCOME_REVEALED_AFTER_SELECTION', calls=1)

    def projection(self):
        service = sum((self.index.get(k, {}).get('seconds') or self.mean_seconds) for k in self.finishes)
        selected_CPU = [self.index[k]['CPU_seconds'] for k in self.adopted if self.index[k]['CPU_seconds'] is not None]
        return dict(clock='RECORDED_JOB_WALL_WITH_CONSTANT_MEAN_FOR_UNKNOWN_SPECULATION',
                    modeled_elapsed_to_last_adoption=self.now,
                    modeled_elapsed_including_outstanding=max([self.now] + list(self.finishes.values())),
                    summed_modeled_job_service_seconds=service,
                    observed_CPU_seconds_of_selected_jobs=sum(selected_CPU),
                    selected_jobs_with_CPU_measurement=len(selected_CPU),
                    modeled_total_CPU_seconds=None,
                    CPU_limit='A changed concurrency/worker profile cannot inherit old CPU or wall timings as a real measurement',
                    jobs_launched=len(self.finishes), recorded_results_revealed=len(self.opened),
                    selected_unknown_allocations=self.missing,
                    launched_without_saved_answer=sum(k not in self.index for k in self.finishes),
                    speculative_launched=len(self.speculative), speculative_adopted=len(self.speculative & self.adopted))


def make_index(ledger, roots):
    index = {};missing = []
    with ledger.open() as stream:
        for row in csv.DictReader(stream):
            if row['verified'] != 'True':
                continue
            job = next((r / 'jobs' / 'PR_PAULI' / row['job'] for r in roots
                        if (r / 'jobs' / 'PR_PAULI' / row['job'] / 'REQUEST.json').exists()), None)
            if job is None:
                missing.append(dict(run=row['run'], job=row['job']));continue
            request = json.loads((job / 'REQUEST.json').read_text())
            key = request['configuration_id'], O.interleave(request['N_X'], request['N_Y'])
            ret = json.loads((job / 'PROCESS_RETURN.json').read_text()) if (job / 'PROCESS_RETURN.json').exists() else {}
            resource_path = job / job.name / 'RESOURCE_FINAL.json'
            resource = json.loads(resource_path.read_text()) if resource_path.exists() else {}
            index[key] = dict(outcome=job / row['outcome_file'], sha256=row['outcome_sha256'],
                              seconds=ret.get('wall_seconds'), CPU_seconds=resource.get('process_tree_CPU_seconds'))
    return index, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--slots', type=int, default=32)
    args = parser.parse_args();start = time.perf_counter();args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_worktree.resolve()
    round_path = baseline / 'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json'
    spec = O.load_json(round_path)
    # Existing metadata-only context construction, with all locators resolved
    # against the fixed B2 worktree so no result or signal workspace is changed.
    old_root_T, old_root_CLI = T.ROOT, T.CLI.ROOT
    print('CONTEXT_START metadata and saved sources only', flush=True)
    try:
        T.ROOT = baseline;T.CLI.ROOT = baseline
        spec, problem, context, routes = T.round_context(baseline / 'configs/rpe_method_optimizer/h4_df10_problem.json',
                    round_path, args.output / 'preparation')
    finally:
        T.ROOT, T.CLI.ROOT = old_root_T, old_root_CLI
    context_seconds = time.perf_counter()-start
    print('CONTEXT_END seconds=%.3f' % context_seconds, flush=True)
    roots = [(baseline / p).resolve() for p in spec['executor']['prior_run_dirs']]
    ledger = baseline / 'docs/research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/ledger/LEDGER.csv'
    index, absent = make_index(ledger, roots)
    print('INDEX_END verified_jobs=%d missing=%d' % (len(index), len(absent)), flush=True)
    candidates = [c for c in context['candidates'] if c['method'] == 'PR_PAULI' and c['ready']]
    seed = O.initial_knowledge(context['saved'], candidates, 'all')
    search = dict(problem['search'], **spec['search_overrides'], max_steps_per_method=args.budget)
    results = {}
    variants = dict(BASELINE_HYBRID={}, ADAPTIVE_HYBRID=dict(adaptive_hybrid=True),
                    EXCHANGE=dict(adaptive_hybrid=True, allocation_exchange=dict(enabled=True, max_moves=8, add_steps=[1, 2])))
    for name, change in variants.items():
        clock = SavedClock(index, seed, args.slots, args.budget)
        result = P.run_priority('PR_PAULI', candidates, dict(seed), clock, dict(search, **change), context['tau'],
                    context['generate'], args.budget, prefetch=clock.prefetch, prefetch_state=clock.state,
                    free=lambda p: SavedClock.key(p) in seed,
                    baseline=(problem['methods']['PR_PAULI'].get('generation') or {}).get('baseline'))
        projection = clock.projection()
        snapshots = {x['key']: x for x in clock.curve}
        curve = [];running_best = O.method_state(candidates, seed, context['tau'])[1]
        running_best = None if running_best is None else running_best['cost']
        initial_best = running_best
        for trace in result['trace']:
            if trace['status'] == O.PASS:
                running_best = trace['cost'] if running_best is None else min(running_best, trace['cost'])
            key = trace['configuration_id'], tuple(trace['vector'])
            snap = snapshots.get(key, dict(seconds=0, launched=0))
            curve.append(dict(seconds=snap['seconds'], launched=snap['launched'], best_cost=running_best))
        levels = [Fraction(3792207339, 7), Fraction(3620059111, 7), Fraction(3065260145, 7),
                  Fraction(410199237)*Fraction(101,100), Fraction(410199237)]
        reach = []
        for target in levels:
            first = dict(seconds=0, launched=0) if initial_best is not None and initial_best <= target else next(
                (dict(seconds=x['seconds'], launched=x['launched']) for x in curve if x['best_cost'] is not None and x['best_cost'] <= target), None)
            reach.append(dict(target_cost=target, reached=first))
        projection.update(stop_reason=result['stop_reason'], best_adopted_cost=result['incumbent'] and result['incumbent']['cost'],
                          calls_adopted=result['new_calls'], prefetch_policy=result['prefetch_policy'],
                          selection_timing=result['selection_timing'], target_reach=reach,
                          adopted_allocations=[dict(configuration_id=t['configuration_id'], N_X=t['N_X'], N_Y=t['N_Y'],
                                                   status=t['status'], cost=t['cost'], origin=t['origin']) for t in result['trace']])
        results[name] = projection
        print(name, 'adopted', result['new_calls'], 'launched', len(clock.finishes), 'unknown', len(clock.missing),
              'best', result['incumbent'] and O.show(result['incumbent']['cost']), flush=True)
    summary = dict(schema='RPE_FIXED_READY_SET_REPLAY_V1', index_jobs=len(index), missing_raw_jobs=absent,
                   compared_budget=args.budget, slots=args.slots, ready_configurations=len(candidates),
                   context_seconds=context_seconds, tool_wall_seconds=time.perf_counter() - start, variants=results,
                   limits=['Fixed ready configurations isolate allocation/scheduling; outer schedule-wide choices and preparation latency are excluded.',
                           'The first selected unknown allocation stops each trajectory; it remains unevaluated and is not treated as a failure.',
                           'Saved results were generated by earlier policies and censor this comparison.',
                           'Wall is a timing model using old job durations, not a new scientific timing measurement.',
                           'All outcome records are opened only after selection; original outcomes and STOPs remain unchanged.'],
                   real_scientific_calls=0)
    (args.output / 'SAVED_REPLAY.json').write_text(json.dumps(O.jsonable(summary), indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
