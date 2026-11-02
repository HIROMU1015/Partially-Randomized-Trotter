"""Parent witness for the one-shot S_STATS child; no scientific provider imports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import h4_dp_contract as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--grant', type=Path, required=True)
    args = p.parse_args()
    spec, grant = c.read(args.spec), c.read(args.grant)
    c.require(spec.get('stage') == grant.get('stage') == 'S_STATS', 'S_STATS_ONLY')
    c.require(spec.get('status') == 'SEALED_FOR_EXECUTION' and grant.get('approved_by_user') is True
              and grant.get('runtime_authorization') is True, 'DRAFT_UNAUTHORIZED')
    run = Path(spec['output']).parent
    c.require(run.exists() and not (run/'native_exit.json').exists(), 'UNUSED_S_STATS_WITNESS')
    start = time.monotonic()
    with (run/'stdout.log').open('x') as out, (run/'stderr.log').open('x') as err:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('h4_s_stats.py')),
            '--spec', str(args.spec), '--grant', str(args.grant)], stdout=out, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1'))
        code = proc.wait()
    result = Path(spec['output'])/'result.json'
    c.save(run/'native_exit.json', dict(SOURCE=spec['SOURCE'], exit_code=code, same_parent_reaped=True,
        wall_seconds=time.monotonic()-start, result_sha256=c.sha(result) if code == 0 and result.exists() else None,
        next_stage_authorized=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
