"""Labelled I/O and sleep fixture; no molecular signal, compiler or scientific DP."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['signal','compile','audit'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path)
    p.add_argument('--sleep',type=float,default=.15);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    time.sleep(a.sleep)
    result={'ARTIFICIAL':True,'value':'EXACT_SYNTHETIC_SHARED_CALIBRATION','scientific_calls':0}
    if a.stage=='compile':
        circuit=b'// ARTIFICIAL ONLY; NO QUANTUM CIRCUIT COMPILE\n'+b'x'*131072
        (a.output/'circuit.txt').write_bytes(circuit)
        result['circuit_sha256']=hashlib.sha256(circuit).hexdigest()
    if a.stage=='audit':
        receipt=json.loads((a.inputs/'compiled/receipt.json').read_text())
        assert receipt['ARTIFICIAL'] and receipt['circuit_sha256']==hashlib.sha256((a.inputs/'compiled/circuit.txt').read_bytes()).hexdigest()
        result.update(verified_compile_receipt=True,circuit_sha256=receipt['circuit_sha256'])
    (a.output/'receipt.json').write_text(json.dumps(result,sort_keys=True)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
