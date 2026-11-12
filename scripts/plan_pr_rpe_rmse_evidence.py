"""Read an explicit finite V2 menu and report evidence gaps; no science or G."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.pr_rpe_rmse_evidence import plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--menu', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    menu = json.loads(args.menu.read_text())
    # No candidate/policy defaults, loaders, network lookups, providers or runners.
    result = plan(menu['entries'], price_policy=menu['price_policy'],
        required_methods=menu['required_methods'], formula_menu=menu.get('formula_menu'),
        unexplored_scope=menu['unexplored_scope'])
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
