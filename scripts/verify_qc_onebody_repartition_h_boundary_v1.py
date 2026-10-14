#!/usr/bin/env python3
"""One independent audit after issued attempt; never launches science."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_execution_v1 import preflight,ARTIFACT,OUTPUT,SCOPE
from trottertracks.ground_pf_repartition.qc_onebody_repartition_execution_v1 import digest,write_json
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=a.directory.resolve()
    try:
        if d!=(ROOT/OUTPUT).resolve():raise ValueError('REGISTERED_DIRECTORY_REQUIRED')
        preflight(ROOT);marker=json.loads((d/'attempt_consumed.json').read_text());identity=json.loads((d/'result_identity.json').read_text())
        if marker['scope_id']!=SCOPE or marker['attempt']!=1 or marker['retry']!=0:raise ValueError('CONSUMED_ATTEMPT_REQUIRED')
        c=json.loads((ROOT/ARTIFACT/'contract.json').read_text())
        if identity['contract_SHA256']!=digest(ROOT/ARTIFACT/'contract.json') or identity['source_manifest_SHA256']!=digest(ROOT/ARTIFACT/'source_manifest.json') or identity['fixed_files']!=c['fixed_files'] or identity['authorization_SHA256']!=marker['authorization_SHA256']:raise ValueError('RESULT_IDENTITY')
        seal=json.loads((d/'inherited_selector_seal.json').read_text())
        if digest(d/'inherited_selector_seal.json')!=digest(ROOT/c['saved_selector_seal']):raise ValueError('SELECTOR_CHANGED')
        if (d/'audit_consumed.json').exists():raise ValueError('AUDIT_NO_RETRY')
        write_json(d/'audit_consumed.json',{'scope_id':SCOPE,'attempt':1,'retry':0})
        import numpy as np
        from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_checker_v1 import audit
        with np.load(ROOT/c['saved_arrays'],allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        raw=json.loads((d/'raw.json').read_text());result=audit(data,seal,raw,c,d)
        write_json(d/'independent_audit.json',result);print(json.dumps(result,indent=2));return 0
    except Exception as e:
        result={'status':'INDEPENDENT_AUDIT_STOP_NO_RETRY','reason':str(e)[:4096],'mandatory_stop':True}
        if (d/'audit_consumed.json').exists():write_json(d/'independent_audit.json',result)
        print(json.dumps(result),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
