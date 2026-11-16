"""Create an unapproved next-round example; never start scientific work."""
import argparse
from copy import deepcopy
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(base, output, run_dir, saved_root, *, slots=32, workers=8, adaptive=False, overlap=False):
    base, output, run_dir, saved_root = map(lambda p: Path(p).resolve(), (base, output, run_dir, saved_root))
    if output.exists() or run_dir.exists():
        raise ValueError('Use a new output file and run directory; existing runs are preserved')
    spec = deepcopy(json.loads(base.read_text()))
    spec['round_id'] += '_THROUGHPUT_EXAMPLE'
    spec['science_approved'] = False
    spec['executor'].update(scheduler='concurrent', max_concurrent_jobs=slots, dp_workers=workers,
        wait_for_running_jobs=True, prefix_cache_directory=str(run_dir / 'prefix_cache'),
        prefix_checkpoint_rounds=[5, 11])
    spec['executor']['prior_run_dirs'] = [str((saved_root / p).resolve()) for p in spec['executor'].get('prior_run_dirs', [])]
    spec['continues']['run_dir'] = str((saved_root / spec['continues']['run_dir']).resolve())
    spec['search_overrides']['adaptive_hybrid'] = adaptive
    spec['overlap_ready_batch'] = overlap
    spec['method_overrides']['PR_PAULI']['generation']['kappa_points_file'] = str(run_dir / 'KAPPA_POINTS.json')
    registry = b'{}\n'
    for route in spec['routes_override']['PR_PAULI']:
        if route.get('configurations_file'):
            old = saved_root / route['configurations_file']
            route['configurations_file'] = str(run_dir / 'ROUTE_REGISTRY.json')
            registry = old.read_bytes() if old.exists() else b'{}\n'
    price_settings = spec['price_rules_overrides']
    saved_prices = saved_root / price_settings['registry_file']
    if saved_prices.exists():
        prices = saved_prices.read_bytes()
    else:
        source = price_settings['source']
        prices = subprocess.check_output(['git', 'show', source['commit'] + ':' + source['path']], cwd=saved_root)
    json.loads(prices)  # validate the copied registry before creating the new run
    price_settings['registry_file'] = str(run_dir / 'PRICE_RULES_REGISTRY.json')
    output.parent.mkdir(parents=True, exist_ok=True);run_dir.mkdir(parents=True)
    (run_dir / 'ROUTE_REGISTRY.json').write_bytes(registry)
    (run_dir / 'PRICE_RULES_REGISTRY.json').write_bytes(prices)
    output.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT / 'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json')
    p.add_argument('--output', type=Path, required=True);p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--saved-root', type=Path, required=True)
    p.add_argument('--slots', type=int, default=32);p.add_argument('--workers', type=int, default=8)
    p.add_argument('--adaptive', action='store_true');p.add_argument('--overlap', action='store_true')
    a = p.parse_args()
    if a.slots < 1 or a.workers < 1:raise ValueError('slots/workers must be positive')
    spec = prepare(a.base, a.output, a.run_dir, a.saved_root, slots=a.slots, workers=a.workers,
                   adaptive=a.adaptive, overlap=a.overlap)
    print(json.dumps(dict(round=str(a.output.resolve()), run=str(a.run_dir.resolve()), science_approved=spec['science_approved'])))


if __name__ == '__main__':main()
