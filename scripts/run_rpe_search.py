"""Common operation entry: all hosts send requests to the selected controller."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_operation_client import ACTIONS,OperationGateway,gateway_status,operate,read,start_gateway


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config','-Config',type=Path)
    parser.add_argument('--action','-Action',type=str.lower,choices=ACTIONS,default='preview')
    parser.add_argument('--request-file','-RequestFile',type=Path)
    parser.add_argument('--run-id','-RunId');parser.add_argument('--release','-Release')
    parser.add_argument('--instruction','-Instruction');parser.add_argument('--no-install',action='store_true')
    parser.add_argument('--gateway-config',type=Path)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--serve',action='store_true')
    modes.add_argument('--start-gateway',action='store_true');modes.add_argument('--status-gateway',action='store_true')
    modes.add_argument('--stop-gateway',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.serve or args.start_gateway or args.status_gateway or args.stop_gateway:
            if not args.gateway_config:parser.error('--gateway-config is required')
            if args.serve:return OperationGateway(args.gateway_config).serve()
            if args.start_gateway:result=start_gateway(args.gateway_config,Path(__file__).resolve())
            elif args.stop_gateway:
                config=read(args.gateway_config);(Path(config['state_directory'])/'STOP').touch();result=dict(status='STOP_REQUESTED')
            else:result=gateway_status(args.gateway_config)
        else:
            if not args.config:parser.error('--config is required')
            result=operate(read(args.config),args.action.lower(),
                args.request_file.read_bytes()if args.request_file else b'{}',run_id=args.run_id,
                release=args.release,instruction=args.instruction,install=not args.no_install)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='CLIENT_HOLD',error=type(error).__name__+': '+str(error),scientific_calls=0),ensure_ascii=False))
        return 2


if __name__=='__main__':raise SystemExit(main())
