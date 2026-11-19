"""Acquire only the previously missing Full DF19/M5 short-round prices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = '73dcbe23678bfc50d5654f09e600d4cfbe75e697'
D2 = '95d01b6ab1d820921fcd2521324e41e82389713a'
PLAN = 'docs/research/h6_df19_rpe_d2_preparation_20261110/PLAN.json'
PILOT = 'docs/research/h6_df19_price_v2_20261110/PILOT_INPUTS.json'
AXES = ('cosine', 'sine')
PRICE = None


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def saved(ref, path):
    return json.loads(git(ROOT, 'show', ref + ':' + path))


def cases():
    rows = [(r['m'], r['candidate']) for r in saved(D2, PLAN)['rows']
            if r['m'] in (0, 1, 2) and r['candidate']['method'] == 'FULL']
    if [(m, c['T'], c['q'], c['delta']) for m, c in rows] != [
            (0, .1, 1, .1), (1, .2, 1, .2), (2, .4, 1, .4)]:
        raise ValueError('Unexpected Full PF short-round schedule')
    if any((c['system'], c['prefix'], c['formula'], c['R_outer'], c['K']) !=
           ('H6', 19, '4th(m5_best)', None, None) for _, c in rows):
        raise ValueError('Unexpected Full PF physical candidate')
    return rows


def runtime():
    import run_h6_df19_price_v2_q4 as q4
    p = q4.p
    if git(p.ROOT, 'rev-parse', 'HEAD').decode().strip() != DEPENDENCY or git(
            p.ROOT, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('H6 price helper view differs from the fixed dependency')
    if (p.np.__version__, p.qiskit.__version__) != ('1.26.4', '1.3.0'):
        raise ValueError('H6 acquisition runtime differs')
    spec = saved(DEPENDENCY, PILOT)
    cfg = p.read(p.CONFIG)['systems']['H6']
    if cfg['binding'] != spec['scientific_binding'] or p.COMPILER != spec['compiler_hash']:
        raise ValueError('Saved binding or compiler differs')
    for name, meta in cfg['inputs'].items():
        raw = Path(meta['path']).read_bytes()
        if p.sha(raw) != meta['sha256'] or len(raw) != meta['bytes'] or meta['sha256'] != spec['input_sha256'][name]:
            raise ValueError('Stored input differs from H6 price pilot')
    return q4, p, spec, cfg


def prepare(q4, p, spec, cfg):
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib import rte_compiled_cost
    from trotterlib.rte import CompilerSettings
    from trottertracks.resource_applicability import ax2b_native_df_v5

    p.verify_provider()
    inputs = cfg['inputs']
    ham, *_ = load_h6_snapshot(Path(inputs['H6_snapshot.npz']['path']),
                               p.read(inputs['H6_snapshot_receipt.json']['path']),
                               p.read(inputs['H6_df_receipt.json']['path']))
    helpers = p.helpers()
    prep = helpers['prepare'](ham, 19)
    if (prep.preparation_hash != spec['preparation_hash']['19']
            or prep.hamiltonian_hash != cfg['binding']['hamiltonian']
            or prep.deterministic_fragment_indices != tuple(range(19)) or
            p.AXES != AXES):
        raise ValueError('Full19 preparation differs')
    rte_compiled_cost.transpile = partial(p.qiskit.transpile, num_processes=1)
    return q4, helpers, prep, ham.constant, use_fingerprint_reuse


def acquire(output, case):
    m, candidate = case
    q4, helpers, prep, scalar, fingerprint_scope = PRICE
    leaf = output / f'm{m}'
    leaf.mkdir()
    print(f'm{m} BUILD_START', flush=True)
    start = time.monotonic()
    with fingerprint_scope() as fp:
        evolution = helpers['compose'](candidate, prep, [], scalar, fp.fingerprint)
        q4.p.write(leaf/'NATIVE.json', dict(candidate=candidate, scalar=scalar,
                   fingerprint=evolution.circuit_fingerprint, instructions=len(evolution.circuit.data)))
        print(f'm{m} BUILD_DONE; START_X_Y', flush=True)
        axes = q4.parallel_axes(leaf, evolution, fp)
    result = dict(m=m, candidate=candidate, scalar=scalar, preparation_hash=prep.preparation_hash,
                  axes={'X': axes['cosine'], 'Y': axes['sine']}, wall_seconds=time.monotonic()-start)
    q4.p.write(leaf/'RESULT.json', result)
    print(f'm{m} COMPLETE', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true', help='compile both axes of the three fixed cases')
    args = ap.parse_args()
    source = git(ROOT, 'rev-parse', 'HEAD').decode().strip()
    script = Path(__file__)
    if script.read_bytes() != git(ROOT, 'show', source+':scripts/'+script.name):
        raise ValueError('Commit this caller before preflight')
    rows = cases()
    q4, p, spec, cfg = runtime()
    output = args.output.resolve()
    output.mkdir(parents=True)
    p.write(output/'INPUTS.json', dict(source=source, dependency=DEPENDENCY, scientific_helper_source=p.CARRIER,
        provider=p.PROVIDER, compiler=p.COMPILER, input_sha256=spec['input_sha256'],
        configuration='Full19/M5, h=0.1, requested width 1.6', cases=rows,
        maximum_compile=2*len(rows), interpreter=os.path.realpath(os.sys.executable)))
    if not args.execute:
        return
    global PRICE
    PRICE = prepare(q4, p, spec, cfg)
    print('Full19 preparation passed', flush=True)
    with ProcessPoolExecutor(max_workers=len(rows), mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(partial(acquire, output), rows))
    p.write(output/'AXIS_PRICES.json', dict(source=source, rows=results,
        compile_count=sum(not a['cache_hit'] for r in results for a in r['axes'].values())))


if __name__ == '__main__':
    main()
