#!/usr/bin/env python3
"""One-shot future BF-1. Draft authorization always rejects before input I/O."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT/'src'))

from trottertracks.algorithm_codesign.domain import Domain, Point
from trottertracks.algorithm_codesign.freeze import (EXECUTION_ID, PREPARATION_REL, canonical, git,
    sha, verify_launch, write_new)
from trottertracks.algorithm_codesign.pilot import Evaluator, Limits, classify, search, secondary_frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', type=Path, required=True)
    arguments = parser.parse_args()
    prep = ROOT/PREPARATION_REL
    plan = json.loads((prep/'source_plan.json').read_text())
    authorization = json.loads(arguments.authorization.read_text())
    domain_bytes = (prep/'domain_manifest.json').read_bytes()
    tests_bytes = (prep/'synthetic_semantic_report.json').read_bytes()
    source_commit = verify_launch(ROOT, plan, authorization, domain_bytes, tests_bytes)
    if plan['domain_fingerprint'] != sha(canonical(json.loads(domain_bytes))):
        raise ValueError('Domain manifest modified')
    common = Path(git(ROOT, 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = ROOT/common
    registry = common/'track-b-bf1-one-shot'
    registry.mkdir(exist_ok=True)
    # A global repository marker prevents repeating this execution from a
    # second worktree. Its namespace is separate from A's runtime registry.
    write_new(registry/(EXECUTION_ID+'.json'), dict(execution_id=EXECUTION_ID,
              source_commit=source_commit, consumed=True, retry_authorized=False))
    output = ROOT/plan['output_relative']
    limits = Limits()
    output.mkdir(parents=True, exist_ok=False)
    caps = plan['resource_caps']
    resource.setrlimit(resource.RLIMIT_CPU, (caps['cpu_seconds'], caps['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_AS, (caps['rss_bytes'], caps['rss_bytes']))
    def expired(*args):
        raise RuntimeError('STOP_WALL_TIME_CAP')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(caps['wall_seconds'])
    rows = output/'cells.jsonl'
    written = 0
    def on_cell(row):
        nonlocal written
        data = canonical(row)+b'\n'
        written += len(data)
        if written > caps['output_bytes']-16*1024**2:
            raise RuntimeError('STOP_OUTPUT_CAP')
        with rows.open('ab') as stream:
            stream.write(data)
        limits.check()
    result = dict(schema='bf1_one_shot_result_v1', execution_id=EXECUTION_ID,
                  source_commit=source_commit, plan_fingerprint=sha(canonical(plan)),
                  mandatory_stop=True, automatic_next_stage=None, BF2_authorized=False,
                  retry_authorized=False, counters=dict(molecular_generation=0, trajectories=0,
                                                        circuits=0, compilations=0, gpu_queries=0))
    try:
        # This import/call is deliberately AFTER every source/auth/one-shot gate.
        from trottertracks.algorithm_codesign.science_input import load_authorized_task
        task, input_audit = load_authorized_task()
        result['input_audit'] = input_audit
        evaluator = Evaluator(task, limits=limits, on_cell=on_cell)
        frozen = json.loads(domain_bytes)
        initial = [Point.from_record(p) for p in frozen['initial_points']]
        fixed = [Point.from_record(p) for p in frozen['fixed_references']]
        domain = Domain()
        points, searches = {}, {}
        for arm in ('O', 'L', 'F'):
            points[arm], searches[arm] = search(domain, initial, evaluator, arm)
        reference_points = {p.identity: p for p in points['O']+points['L']+fixed}
        F_points = {p.identity: p for p in points['F']}
        reference = evaluator.rescore(list(reference_points.values()), .01)
        finite = evaluator.rescore(list(F_points.values()), .01)
        L_ids = {p.identity for p in points['L']}
        leading = [r for r in reference if r['coefficient'] in L_ids]
        decision = classify(finite, reference, leading)
        frontier = secondary_frontier(finite+reference)
        all_points = {p.identity: p for p in list(reference_points.values())+list(F_points.values())}
        bridge = evaluator.rescore(list(all_points.values()), .05)
        result.update(status='BF1_COMPLETE_MANDATORY_STOP', decision=decision, searches=searches,
                      primary_F=finite, primary_reference=reference, bridge=bridge,
                      secondary_frontier=frontier,
                      ideal_cells=len(evaluator.ideal), finite_cells=len(evaluator.finite),
                      exact_target=[evaluator.target.real, evaluator.target.imag],
                      coefficients=[p.record() for p in all_points.values()])
    except Exception as exc:
        result.update(status='BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY',
                      outcome='INCONCLUSIVE', exception_type=type(exc).__name__, reason=str(exc))
    finally:
        signal.alarm(0)
        result['wall_seconds'] = time.monotonic()-limits.wall
        result['cpu_seconds'] = time.process_time()-limits.cpu
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        data = canonical(result)
        if len(data)+written > caps['output_bytes']:
            result = {k: v for k, v in result.items() if k not in ('primary_F', 'primary_reference', 'bridge', 'searches', 'coefficients')}
            result.update(status='BF1_INCOMPLETE_OUTPUT_CAP_MANDATORY_STOP', outcome='INCONCLUSIVE')
        write_new(output/'result.json', result)
        print(result['status'])
        print('MANDATORY STOP: review Track B RQ/novelty/endpoint before any next stage.')


if __name__ == '__main__':
    main()
