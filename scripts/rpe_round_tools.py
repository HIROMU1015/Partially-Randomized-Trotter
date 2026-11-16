"""Shared helpers for saved-result tools: the context of a round exactly as ``run_rpe_method_optimizer.py --round`` builds it
(scoped problem, price overrides, kappa points, merged route registry, published routes), without launching anything."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def round_context(problem_path, round_path, preparation_dir, points_file=None):
    spec = O.load_json(round_path)
    problem = CLI.price_overrides(CLI.scoped(O.load_json(problem_path), spec.get('method_overrides')), spec)
    for m in spec['methods']:
        g = problem['methods'][m].get('generation') or {}
        if points_file is not None:
            g['kappa_points_file'] = str(Path(points_file).resolve())
        elif g.get('kappa_points_file') and not Path(g['kappa_points_file']).is_absolute():
            g['kappa_points_file'] = str(ROOT / g['kappa_points_file'])
    context = O.prepare(ROOT, problem, methods=spec['methods'], preparation_dir=Path(preparation_dir))
    routes = CLI.merged_routes(problem, spec)
    O.apply_published(context['candidates'], routes)
    return spec, problem, context, routes


def executor(problem, context, routes, run_dir, prior_run_dirs):
    return O.Executor(ROOT, Path(run_dir), routes, context['expected'], problem, launch_disabled=True,
                      prior_run_dirs=[(ROOT / d).resolve() for d in prior_run_dirs])
