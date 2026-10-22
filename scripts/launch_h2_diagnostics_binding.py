"""Isolated stdlib launcher; no site/.pth, no scientific import before execute."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approval');parser.add_argument('--grant')
    args=parser.parse_args()
    if args.execute and not (args.approval and args.grant):
        parser.error('EXPLICIT_SCIENCE_APPROVAL_AND_NEW_GRANT_REQUIRED')
    cfg=json.loads(Path(args.contract).read_text())
    runtime=cfg['runtime']
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        parser.error('LAUNCH_REQUIRES_PYTHON_I_S_B')
    if str(Path(sys.executable).resolve())!=runtime['python_target']:
        parser.error('SEALED_INTERPRETER_REQUIRED')
    sys.path[:]=[str(Path(__file__).resolve().parent)]+runtime['sys_path']
    import h2_common_production_contract as c
    if args.execute:result=c.execute(cfg,c.read(args.approval),c.read(args.grant))
    else:result=c.preflight(cfg)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
