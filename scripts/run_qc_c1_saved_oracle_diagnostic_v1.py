#!/usr/bin/env python3
"""Separate authorization required before registered saved-data metrics."""
import argparse,json,sys,resource,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.representation_exploration.c1_saved_oracle_execution_v1 import authorize,preflight,read,safe_write
from trottertracks.representation_exploration.c1_saved_oracle_spec_v1 import COMPARISONS,SEAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorization');p.add_argument('--preflight',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if args.preflight:
        print(json.dumps(preflight(root),indent=2));return
    out,receipt=authorize(root,args.authorization)
    tic=time.monotonic()
    try:
        from trottertracks.representation_exploration.c1_saved_oracle_diagnostic_v1 import diagnose
        result=diagnose(read(root/COMPARISONS),read(root/SEAL),receipt=receipt)
        safe_write(out,'diagnostic.json',result)
        safe_write(out,'resource_use.json',{'wall_seconds':time.monotonic()-tic,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                          'workers':1,'threads':1,'new_eigensolver_PF_compile_calls':0})
        safe_write(out,'stop.json',{'status':'POSTHOC_COMPLETE_INDEPENDENT_AUDIT_REQUIRED','retry':0,'mandatory_stop':True})
        print(json.dumps({'status':'POSTHOC_COMPLETE_INDEPENDENT_AUDIT_REQUIRED','mandatory_stop':True}))
    except Exception as e:
        safe_write(out,'failure.json',{'error':type(e).__name__,'message':str(e)[:1000],'retry':0,'mandatory_stop':True})
        raise
if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'error':type(error).__name__,'message':str(error)[:1000],'mandatory_stop':True}),file=sys.stderr)
        sys.exit(1)
