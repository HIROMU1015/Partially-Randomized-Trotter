#!/usr/bin/env python3
"""Separate approval required: one read-only numerical re-audit of saved cells."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_reaudit_v2 import authorize,preflight
from trottertracks.ground_pf_repartition.qc_onebody_repartition_execution_v1 import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--authorization',type=Path);a=p.parse_args()
    context=None;trace=None
    try:
        if a.preflight:
            print(json.dumps(preflight(ROOT),indent=2));return 0
        context=authorize(ROOT,a.authorization)  # All gating before numerical import/load.
        import numpy as np
        from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_checker_v2 import audit,AuditTrace,require
        trace=AuditTrace();trace.stage='saved_input_decode'
        c=context.numerical_contract
        with np.load(ROOT/c['saved_arrays'],allow_pickle=False) as archive:data={k:np.array(archive[k]) for k in archive.files}
        seal=json.loads((context.saved_attempt/'inherited_selector_seal.json').read_text())
        raw=json.loads((context.saved_attempt/'raw.json').read_text())
        if trace.snapshot()['peak_RSS_bytes']>=context.contract['limits']['aggregate_RSS_bytes']:raise ValueError('AUDIT_DECODE_RSS_CAP')
        result=audit(data,seal,raw,c,context.saved_attempt,trace=trace)
        result.update(scope_id=context.contract['scope_id'],original_results_modified=False,old_authorization_reused=False)
        payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if len(payload.encode())>context.contract['limits']['output_bytes']:raise ValueError('OUTPUT_CAP')
        (context.output/'independent_audit.json').write_text(payload);print(payload);return 0
    except Exception as e:
        result={'status':'REAUDIT_V2_STOP_NO_RETRY','reason':str(e)[:4096],'mandatory_stop':True}
        if context is not None:
            result['diagnostic']=getattr(e,'diagnostic',trace.snapshot() if trace is not None else {'stage':'numerical_import'})
            write_json(context.output/'independent_audit.json',result)
        print(json.dumps(result,allow_nan=False),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
