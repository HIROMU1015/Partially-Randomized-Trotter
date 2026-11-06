#!/usr/bin/env python3
"""Artificial/mock-only test entry. No science flags or campaign execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    results=[]
    for name in ('test_a1_connections.py','test_a1_control_variates.py','test_a1_n3_connection.py'):
        command=[sys.executable,'-B','-m','unittest','discover','-s','tests/tracks/algorithm_codesign','-p',name,'-v']
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/(name+'.log')).write_text(run.stdout)
        result=dict(test=name,command=command,returncode=run.returncode)
        results.append(result)
        if run.returncode:break
    passed=all(r['returncode']==0 for r in results) and len(results)==3
    (out/'TESTS.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',results=results,
        scope='invented laws/states/gates/word tapes and mock compile prices only; no real RNG',
        scientific_calls=0,N3_executed=False),indent=2)+'\n')
    (out/'STOP.json').write_text(json.dumps(dict(status='MANDATORY_STOP',N3_authorized=False,scientific_calls=0),indent=2)+'\n')
    print(json.dumps(dict(status='PASS' if passed else 'FAIL',output=str(out))))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
