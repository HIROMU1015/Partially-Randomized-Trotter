#!/usr/bin/env python3
"""Run the ordinary native worker with a small explicitly selected transport overlay.

Deploy as scripts/run_rpe_q3_worker.py beside the overlay's src/trottertracks
modules and NATIVE_WORKER_OVERLAY.json. The fixed base runner/provider remains
unchanged. Detached supervisors re-enter this same selected worker script.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def load_overlay():
    root = Path(__file__).resolve().parents[1]
    selection = json.loads((root/'NATIVE_WORKER_OVERLAY.json').read_bytes())
    base = Path(selection['base_root']).resolve()
    names = ('rpe_q3_timing', 'rpe_host_profiles', 'rpe_q3_resources', 'rpe_worker_storage', 'rpe_q3_worker')
    if set(selection['modules']) != set(names):
        raise ValueError('Incomplete native worker operational overlay')
    sys.path.insert(0, str(base/'src'))
    import trottertracks
    for name in names:
        path = root/'src/trottertracks'/(name+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != selection['modules'][name]:
            raise ValueError('Native worker operational module differs: '+name)
        qualified = 'trottertracks.'+name
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        setattr(trottertracks, name, module)
    return base


def main():
    base = load_overlay()
    if sys.argv[1:] == ['--check-imports']:
        print('NATIVE_WORKER_IMPORTS_PASS')
        return
    runpy.run_path(str(base/'scripts/run_rpe_q3_worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
