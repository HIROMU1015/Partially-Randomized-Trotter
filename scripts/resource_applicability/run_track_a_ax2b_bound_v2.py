#!/usr/bin/env python3
"""Default metadata only. Execution requires a separately pinned user grant."""
import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_bound_launch_v2 import (
        preparation, validate_launch, environment, verify_input, verify_sources)
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_limits import exclusive_json, install_worker_limits
from trottertracks.resource_applicability.ax2b_launch_watchdog_v2 import supervise


def bounded_json(path):
    if path.stat().st_size > 4*2**20:
        raise ValueError('JSON_INPUT_SIZE')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('H4_LIMITED','H6_TECHNICAL'),default='H6_TECHNICAL')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        if args.worker or args.manifest or args.authorization or args.authorization_sha256:
            parser.error('Launch arguments need explicit --execute and separate grant.')
        exclusive_json(args.output,preparation(args.kind))
        print('H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')
        return 0
    if not args.manifest or not args.authorization or not args.authorization_sha256:
        parser.error('Separate pinned authorization and sealed manifest required.')
    if file_hash(args.authorization) != args.authorization_sha256:
        raise ValueError('AUTHORIZATION_FILE_HASH')
    manifest,auth = bounded_json(args.manifest),bounded_json(args.authorization)
    if manifest.get('kind') != args.kind:
        raise ValueError('CLI_SCOPE_BINDING')
    cpu = validate_launch(ROOT,manifest,auth,requested=True,output=args.output,worker=args.worker)
    caps = manifest['plan']['caps_proposed']
    if not args.worker:
        args.output.mkdir(parents=False,exist_ok=False)
        exclusive_json(args.output/'launch_binding.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(auth)})
        exclusive_json(args.output/'frozen_preparation.json',manifest)
        exclusive_json(args.output/'authorization.json',auth)
        command = [sys.executable,str(Path(__file__).resolve()),'--kind',args.kind,'--execute','--worker',
                   '--manifest',str(args.manifest.resolve()),'--authorization',str(args.authorization.resolve()),
                   '--authorization-sha256',args.authorization_sha256,'--output',str(args.output.resolve())]
        result = supervise(command,args.output,kind=args.kind,caps=caps)
        print(result['status'])
        return 0 if result['status'].endswith('_COMPLETE') else 1
    install_worker_limits(cpu,caps['address_space_bytes'],caps['output_bytes']-65536)
    exclusive_json(args.output/'worker_claim.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(auth),
                                                  'assigned_cpu':cpu,'resume':False})
    # First numerical imports, after all gates/one-shot claim/limits.
    from trottertracks.resource_applicability.ax2b_h6_controller import BoundedWriter
    from trottertracks.resource_applicability.ax2b_molecular_ports_v2 import MolecularPort
    writer = BoundedWriter(args.output,byte_cap=caps['output_bytes'],diagnostics_cap=caps['diagnostics'])
    port = MolecularPort(ROOT,manifest,writer,started=STARTED)
    try:
        port.setup(); port.correctness(); port.costs()
        verify_sources(ROOT,manifest['source_commit'],manifest['source_hashes'])
        verify_input(ROOT,manifest['input_binding'],manifest['kind'])
        if environment() != manifest['environment']:
            raise ValueError('ENVIRONMENT_CHANGED_DURING_RUN')
        status,reason = args.kind+'_COMPLETE',None
    except Exception as error:
        status,reason = args.kind+'_STOP',type(error).__name__+':'+str(error)[:512]
    writer.write('worker_terminal.json',{'status':status,'reason':reason,'completed_correctness_cells':port.completed,
                'compiled_wrappers':port.compiled,'calls':port.calls.used,'N':None,'G':None,
                'mandatory_stop':True,'next_stage_authorized':False,'numerical_allowance_certified':False,
                'accuracy_eligibility':'UNDETERMINED'},terminal=True)
    return 0 if status.endswith('_COMPLETE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
