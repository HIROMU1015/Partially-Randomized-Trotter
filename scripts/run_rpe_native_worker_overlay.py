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


def check_imports(worker, resources, storage, timing, profiles):
    # Check the actual bound attributes, not only sys.modules paths.
    references = ((worker,'Resources',resources),(worker,'Storage',storage),
                  (worker,'stamp',timing.stamp),(worker,'timing_append',timing.append),
                  (worker,'_select_q3_profile',profiles.select_host_profile),
                  (worker,'validate_adapter_profile',profiles.validate_adapter_profile),
                  (worker,'check_runtime_profile',profiles.check_runtime_profile),
                  (resources,'stamp',timing.stamp))
    for module,name,expected in references:
        if getattr(module,name,None) is not expected:
            raise ValueError('Native worker overlay reference differs: '+name)


def main():
    base = load_overlay()
    if sys.argv[1:] == ['--check-imports']:
        from trottertracks import (rpe_q3_worker as worker, rpe_q3_resources as resources,
            rpe_worker_storage as storage, rpe_q3_timing as timing, rpe_host_profiles as profiles)
        check_imports(worker,resources,storage,timing,profiles)
        print('NATIVE_WORKER_IMPORTS_PASS')
        print(json.dumps({name:getattr(sys.modules.get('trottertracks.'+name),'__file__',None)
                          for name in ('rpe_q3_timing','rpe_host_profiles','rpe_q3_resources',
                                       'rpe_worker_storage','rpe_q3_worker')}))
        return
    if '--mailbox' in sys.argv and sys.argv[sys.argv.index('--mailbox')+1:sys.argv.index('--mailbox')+2] == ['storage-sample']:
        # The selected operational worker can expose a newer mailbox action
        # while its frozen scientific checkout retains the older CLI choices.
        import argparse
        from trottertracks.rpe_q3_worker import mailbox, read
        parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True)
        parser.add_argument('--mailbox',choices=['storage-sample'],required=True)
        args=parser.parse_args()
        try:
            print(json.dumps(mailbox(read(args.profile),args.mailbox,json.load(sys.stdin)),allow_nan=False),flush=True)
        except Exception as error:
            print(json.dumps(dict(ok=False,error=type(error).__name__+': '+str(error))),flush=True)
            raise SystemExit(2)
        return
    runpy.run_path(str(base/'scripts/run_rpe_q3_worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
