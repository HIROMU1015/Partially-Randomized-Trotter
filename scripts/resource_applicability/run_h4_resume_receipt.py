#!/usr/bin/env python3
"""Prepare clock/native-stop/byte receipts only; never stop or launch a run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import continuation as c
from trottertracks.resource_applicability.h4_geometry.identity import require
from trottertracks.resource_applicability.h4_geometry.gates import THREAD_ENV


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('capture-clock','capture-stop','receipt'))
    parser.add_argument('--parent',choices=('run10','resume01'),default='run10')
    parser.add_argument('--started',required=True)
    parser.add_argument('--clock')
    parser.add_argument('--proof')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    protocol=c
    if args.parent=='resume01':
        from trottertracks.resource_applicability.h4_geometry import continuation_resume01 as protocol
    require(sys.flags.safe_path and sys.dont_write_bytecode and
            all(os.environ.get(k)==v for k,v in THREAD_ENV.items()),'receipt Python -P -B / thread1')
    started=protocol.read_json(args.started,expected=protocol.STARTED_EXPECTED)
    if args.mode=='capture-clock':value=protocol.capture_clock(started)
    else:
        require(args.clock is not None,'fixed live clock context needed')
        clock=protocol.read_json(args.clock,expected=protocol.CLOCK_EXPECTED)
        if args.mode=='capture-stop':value=protocol.capture_stop_proof(started,clock)
        else:
            require(args.proof is not None,'native stop proof needed before snapshot')
            value=protocol.make_receipt(ROOT,args.started,args.clock,args.proof)
    path=c.private_path(args.output)
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    require(len(raw)<=c.MAX_RECEIPT and not path.exists(),'exclusive bounded receipt output')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    from trottertracks.resource_applicability.h4_geometry.resources import fsync_directory
    fsync_directory(path.parent)
    print(json.dumps(c.path_ref(path)))


if __name__=='__main__':main()
