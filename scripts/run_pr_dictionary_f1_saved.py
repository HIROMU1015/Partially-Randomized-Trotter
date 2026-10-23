"""Future F1 entry; no execution grant is supplied by F1-PRE.

Only stdlib and the unchanged authorization gate load before consumption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args(); root=Path(args.source_root).resolve()
    sys.path.insert(0,str(root/'src'))
    from trottertracks.pr_dictionary_validation.authorization import verify_packet, claim_packet
    from trottertracks.pr_dictionary_validation.f1_saved_plan import iter_cases
    verified=verify_packet(args.packet,root,args.output)
    contract=verified['contract']
    if contract.get('adapter')!='F1_SAVED_B_V1':
        raise PermissionError('Saved-B adapter contract required')
    if not sys.flags.isolated or sys.version_info[:3]!=(3,11,9) or sys.prefix==sys.base_prefix:
        raise PermissionError('Isolated dedicated CPython 3.11.9 venv required')
    if contract.get('interpreter')!=str(Path(sys.executable).absolute()):
        raise PermissionError('Wrong dedicated interpreter')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head!=contract.get('source_commit'):
        raise PermissionError('Wrong frozen SOURCE commit')
    # All imported local modules, including eager trotterlib imports, are bound.
    paths={str(p.relative_to(root)) for base in ('src/trotterlib','src/trottertracks')
           for p in (root/base).rglob('*.py')}
    paths.update(('scripts/run_pr_dictionary_f1.py','scripts/run_pr_dictionary_f1_saved.py'))
    if not paths.issubset(contract['source_files']):
        raise PermissionError('Incomplete local SOURCE closure')
    if verified['inventory']['cases']!=list(iter_cases()):
        raise PermissionError('Inventory payload/order differs from fixed ordinal plan')
    if contract.get('price_interval_policy')!='UNQUALIFIED_EXPLICIT_FALLBACK_NO_CERTIFIED_SELECTION':
        raise PermissionError('Separate approval required to change interval policy')
    if contract.get('interval_provider_qualification'):
        raise PermissionError('This preparation does not qualify the interval provider')
    required={'handoff/f1_bundle.json','inputs/physical.json','inputs/catalog.json',
              *(f'arms/{a}.json' for a in ('SS','SP','PS','PP')),
              *(f'laws/{a}.json' for a in ('SS','SP','PS','PP','catalog'))}
    if not required.issubset(verified['inputs']):
        raise PermissionError('Missing original B inputs; regeneration prohibited')
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in verified['inputs']):
        raise PermissionError('Noncanonical input snapshot path')
    claim_packet(verified)
    try:
        with (verified['output']/'binding_snapshot.json').open('x') as f:
            json.dump(dict(contract=contract,authorization_sha256=verified['authorization_digest']),f)
        for name,data in verified['inputs'].items():
            target=verified['output']/'input_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(data)
        # A single fresh grant covers every registered attempt, with no retry.
        from trottertracks.pr_dictionary_validation.f1_saved_execution import execute_saved_f1
        result=execute_saved_f1(verified)
    except BaseException as error:
        result=dict(status='FAILED_MANDATORY_STOP_NO_RETRY',
                    error_type=type(error).__name__,error=str(error))
        raise
    finally:
        stop=dict(result,authorization=verified['authorization_digest'],F2_authorized=False)
        with (verified['output']/'STOP.json').open('x') as f:
            json.dump(stop,f,ensure_ascii=False,allow_nan=False); f.write('\n')
        records=[]
        for path in sorted(verified['output'].rglob('*')):
            if path.is_file():
                hasher=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1<<20),b''): hasher.update(block)
                records.append(dict(path=str(path.relative_to(verified['output'])),
                                    bytes=path.stat().st_size,sha256=hasher.hexdigest()))
        with (verified['output']/'output_hashes.json').open('x') as f:
            json.dump(dict(records=records,self_excluded=True),f)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
