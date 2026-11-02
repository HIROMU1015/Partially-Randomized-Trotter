#!/usr/bin/env python3
"""Independent audit only after separately authorized consumed attempt."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.representation_exploration.c1_saved_oracle_execution_v1 import checker_receipt,read,safe_write,claim_audit
from trottertracks.representation_exploration.c1_saved_oracle_spec_v1 import COMPARISONS,SEAL
from trottertracks.representation_exploration.c1_saved_oracle_checker_v1 import verify_result

def main():
    root=Path(__file__).resolve().parents[1];out,receipt=checker_receipt(root)
    if (out/'failure.json').exists():raise ValueError('FAILED_DIAGNOSTIC_NO_AUDIT_RETRY')
    claim_audit(out)
    try:
        report=verify_result(read(root/COMPARISONS),read(root/SEAL),read(out/'diagnostic.json'),receipt=receipt)
        safe_write(out,'independent_audit.json',report)
        print(json.dumps(report,indent=2))
    except Exception as error:
        safe_write(out,'independent_audit_failure.json',{'error':type(error).__name__,'message':str(error)[:1000],'retry':0,'mandatory_stop':True})
        raise

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'error':type(error).__name__,'message':str(error)[:1000],'mandatory_stop':True}),file=sys.stderr)
        sys.exit(1)
