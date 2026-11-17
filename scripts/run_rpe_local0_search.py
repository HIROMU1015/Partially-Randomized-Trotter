"""Server2 ordinary entry gateway. Requests are JSON, never shell fragments."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_local0_search import Gateway,RequestError,execute_run
from trottertracks.rpe_release_registry import load_service

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=pathlib.Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request',type=pathlib.Path);group.add_argument('--execute-run',type=pathlib.Path)
    parser.add_argument('--invocation')
    args=parser.parse_args();config=load_service(args.config)
    if args.execute_run:
        if not args.invocation:parser.error('--execute-run requires --invocation')
        execute_run(config,args.execute_run,args.invocation);return
    try:value=Gateway(config).handle(json.loads(args.request.read_text()))
    except RequestError as error:value=dict(status='NOT_STARTED',started=False,errors=[dict(field=error.field,reason=error.reason)])
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
