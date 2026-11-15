#!/usr/bin/env python3
"""Passive DF signal plan/stage/bundle entry. No scientific execute mode."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_df_signal_entry import DFSignalEntry,FixedRuntime,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('plan','stage','bundle'))
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--signal-view',type=Path)
    args=ap.parse_args()
    try:
        entry=DFSignalEntry(args.root,runtime=FixedRuntime(args.root,args.signal_view))
        req=json.loads(args.input.read_text())
        if args.mode=='bundle':
            if 'carrier' in req:
                c=req['carrier'];base=c['directory'].rstrip('/')
                cfgsrc=dict(commit=c['commit'],path=base+'/CONFIGURATION.json')
                config=entry.reader.read(cfgsrc)
                sources={r['m']:dict(commit=c['commit'],path=base+'/raw/m'+str(r['m'])+'.json')for r in config['rounds']}
            else:
                cfgsrc=req['configuration_source'];sources=req['raw_sources']
            result=entry.assemble(cfgsrc,sources)
            write(args.output,result)
        else:
            value=req.get('configuration_source',req)
            cfg=entry.configuration(value)
            if args.mode=='plan':write(args.output,entry.plan(cfg))
            else:entry.stage(cfg,args.output,obtained=req.get('obtained'))
    except (ValueError,KeyError,OSError) as exc:
        print(json.dumps(dict(status='SIGNAL_INPUT_OR_EVIDENCE_ERROR',reason=str(exc))),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
