"""Read-only metadata plan using the fixed implementation's existing W helper."""
import collections
import datetime
from fractions import Fraction
import hashlib
import gzip
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
SOURCE = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99')
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))
from trottertracks import rpe_schedule_search as W
from trottertracks import rpe_method_optimizer as O


def encoded(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {k: encoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    return value


def main():
    snapshot = BASE / 'inputs/server2_snapshot/run01/KAPPA_STATE.json.gz'
    snapshot_bytes = gzip.decompress(snapshot.read_bytes())
    state = json.loads(snapshot_bytes)
    spec = json.loads((BASE / 'inputs/SHOT_PLAN_ROUND.json').read_text())
    families = {k: W.Family(k, v['rows']) for k, v in state['families'].items()
                if v['status'] == 'READY_METADATA'}
    points = []
    for p in state['points']:
        costs = [O.frac(v) for v in (p.get('best'), p.get('confirmed_all_best')) if v is not None]
        best = min(costs) if costs else None
        status = ('FEASIBLE' if best is not None else 'NO_FEASIBLE_FOUND' if p['status'] == 'EVALUATED'
                  else 'UNAVAILABLE' if p['status'] in ('HELD', 'PRUNED') else 'PENDING')
        points.append(dict(family=p['family'], kappa=Fraction(p['kappa']), status=status,
                           best=best, min_lower_over_tau=p.get('min_lower_over_tau')))
    settings = spec['search_overrides']['schedule_wide']
    by_method = {}
    for method in ('PR_PAULI', 'PR_DF'):
        scoped = {k: v for k, v in families.items() if state['families'][k]['method'] == method}
        proposals, decisions, report = W.plan_wide(scoped, [p for p in points if p['family'] in scoped], settings)
        rows = []
        for rank, proposal in enumerate(proposals, 1):
            key = proposal['family']
            family = families[key]
            R = family.R(proposal['kappa'])
            duplicates = [p for p in state['points'] if p['family'] == key and family.R(p['kappa']) == R]
            rows.append(dict(priority_rank=rank, **proposal,
                method=method, L_D=state['families'][key]['L_D'], formula=state['families'][key]['formula'],
                requested_delta_pf=state['families'][key]['requested_delta_pf'],
                q=[r['q'] for r in family.rows], R=R,
                actual_delta_pf=[O.frac(state['families'][key]['configuration']['spec']['h']) * 2**r['m'] / r['q'] for r in family.rows],
                existing_physical_schedule_matches=[{k:p.get(k) for k in ('family','kappa','status','configuration_id','reason')} for p in duplicates],
                dispatchability='PLAN_ONLY_NOT_DISPATCHED_HISTORICAL_R06_STOP_PRESERVED',
                required_before_Q3=['candidate-specific configuration and signals','candidate-specific 26-axis price and route','separate scientific execution scope']))
        by_method[method] = dict(proposals=rows, decisions=decisions, report=report,
            existing_points_by_status=dict(collections.Counter(p['status'] for p in state['points'] if p.get('method') == method)))
    mixed_proposals, mixed_decisions, mixed_report = W.plan_wide(families, points, settings)
    original_points = []
    for p in state['points']:
        status = ('FEASIBLE' if p.get('best') is not None else 'NO_FEASIBLE_FOUND' if p['status'] == 'EVALUATED'
                  else 'UNAVAILABLE' if p['status'] in ('HELD', 'PRUNED') else 'PENDING')
        original_points.append(dict(family=p['family'], kappa=Fraction(p['kappa']), status=status,
            best=None if p.get('best') is None else O.frac(p['best']), min_lower_over_tau=p.get('min_lower_over_tau')))
    original_proposals, original_decisions, original_report = W.plan_wide(families, original_points, settings)
    result = dict(schema='DISTRIBUTED_H4_READONLY_WIDTH_FRONTIER_V1',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source='28d976b99cdf8329adcc1b79327e329e947ef774',
        helper='src/trottertracks/rpe_schedule_search.py:plan_wide',
        snapshot_path=str(snapshot), snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(), snapshot_encoding='gzip',
        existing_scientific_stop=state.get('stop'), existing_points=len(state['points']),
        rules=settings, by_method=by_method,
        fixed_28d_WidthDriver_plan_mixed=dict(proposals=original_proposals, decisions=original_decisions, report=original_report),
        mixed_with_confirmed_best=dict(proposals=mixed_proposals, decisions=mixed_decisions, report=mixed_report),
        qualification='Supplementary pure metadata W helper calls, independently per method as R06 adopted. No WidthDriver mutation, no Q3/collect/signal/price/compile/queue calls. Historical R06 stop remains active; these proposals are not execution authorization. Family bests are limited to the snapshot known context; current C catalog minima are reported separately. No price_floor callback is used, matching WidthDriver.plan.',
        fixed_28d_difference='28d WidthDriver.plan calls W once across methods and uses only point.best. R06 independently scopes methods and considers confirmed_all_best; no implementation changes are applied to fixed SOURCE here.')
    destination = BASE / 'checks/WIDTH_FRONTIER.json'
    destination.write_text(json.dumps(encoded(result), ensure_ascii=False, indent=2) + '\n')
    print(destination)
    for method, data in by_method.items():
        print(method, 'proposals', len(data['proposals']), 'stop', data['report']['stop'])
        for p in data['proposals']:
            print(p['priority_rank'], p['family'], p['kappa'], p['reason'], 'duplicates',len(p['existing_physical_schedule_matches']))


if __name__ == '__main__':
    main()
