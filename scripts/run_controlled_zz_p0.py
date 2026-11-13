#!/usr/bin/env python3
"""P0 controlled-ZZ lowering check (approved scope: at most 8 small-circuit compiles, 0 sampling).

--plan-out writes the fixed design (no compile). --execute requires the committed
SOURCE, the plan written before execution, the fixed compiler stack and provider
bytes; it compiles exactly the eight planned circuits with the existing
rte_compiled_cost.transpile_and_measure_cost and writes RESULT.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import controlled_zz_p0 as Z  # noqa: E402

PROVIDER_FILES = ('src/trotterlib/rte_compiled_cost.py', 'src/trotterlib/rte.py', 'src/trotterlib/df_rte_qiskit.py')
COUNTER_FILES = ('src/trottertracks/h4_df10_price_v1.py', 'src/trottertracks/track_d_native_resources.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def plan(store):
    data = Z.inputs(store)
    return dict(status='P0_PLAN_FIXED_BEFORE_COMPILE', design=Z.design(data), read_sources=dict(store.sources),
                approved_scope='max 8 small-circuit compiles, 0 sampling; no P1/P2/P3, no S-prime implementation, no fit')


def preflight(source):
    if git('rev-parse', 'HEAD') != source or git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('HEAD must equal --source with a clean tracked tree')
    for f in PROVIDER_FILES:
        if git('rev-parse', Z.PROVIDER + ':' + f) != git('hash-object', f):
            raise SystemExit('Provider SOURCE differs: ' + f)
    for f in COUNTER_FILES:
        if git('rev-parse', Z.COUNTER + ':' + f) != git('hash-object', f):
            raise SystemExit('Structural counter differs: ' + f)
    import numpy, qiskit
    if (numpy.__version__, qiskit.__version__) != ('1.26.4', '1.3.0'):
        raise SystemExit('Fixed runtime numpy 1.26.4 / qiskit 1.3.0 required')
    return dict(numpy=numpy.__version__, qiskit=qiskit.__version__, python=sys.version.split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-out', type=Path)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--source'); ap.add_argument('--plan', type=Path); ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    store = Z.p.SavedArtifacts(ROOT)
    if args.plan_out:
        write(args.plan_out, plan(store)); return
    if not (args.execute and args.source and args.plan and args.out):
        raise SystemExit('--execute needs --source, --plan and --out')
    runtime = preflight(args.source)
    fixed = json.loads(args.plan.read_text())
    if fixed['design'] != json.loads(json.dumps(Z.design(Z.inputs(store)))):
        raise SystemExit('Design differs from the plan fixed before execution')
    args.out.mkdir(parents=True, exist_ok=False)
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    settings = CompilerSettings(*Z.COMPILER_ARGS)
    result = Z.execute(store, lambda qc, s: transpile_and_measure_cost(qc, s), settings)
    if len(result['compile_calls']) != Z.COMPILE_BUDGET:
        raise SystemExit('Exactly eight planned compiles required')
    result.update(source=args.source, runtime=runtime, plan=dict(path=str(args.plan), sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest()),
                  read_sources=dict(store.sources), mandatory_stop=True)
    write(args.out/'RESULT.json', result)


if __name__ == '__main__':
    main()
