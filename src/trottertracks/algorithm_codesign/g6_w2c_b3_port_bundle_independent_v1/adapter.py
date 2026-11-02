"""Independent Bundle acceptance plus new Port proof; no producer imports."""
from pathlib import Path
import json
from .. import g6_w2c_b3_bundle_independent_adapter_v1 as BASE
from .. import g6_w2c_b3_offline_result as V
from ..g6_w2c_b3_parallel_offline_adapter import safe_values
from ..g6_w2c_b3_bundle_independent_identity_v1 import read,equal
from . import dispatch

def verify(root,reference):
    result=BASE.verify(root,reference);result['schema']='b3_port_bundle_independent_local_v1';result['port_dispatch']=dict(complete=False,verified_trials=0)
    if result['status'] in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'):return result
    root=Path(root);out=root/'output'
    if not(out/'final.json').exists() or not(out/'operations.jsonl').exists():return result
    try:
        c=read(out/'candidate.json');m=read(out/'manifest.json');r=safe_values(read(out/'final.json'))
        equal(r['port_binding'],m['endpoint_binding'],'RESULT_PORT_BINDING')
        for name in ('summary.json','receipt.json'):
            p=out/name
            if p.exists() and p.read_bytes().strip():equal(read(p)['port_binding'],m['endpoint_binding'],'DOCUMENT_PORT_BINDING:'+name)
            elif result['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE':raise FileNotFoundError(str(p))
        proof=dispatch.verify(out,c,m,r,m['tasks']);complete=result['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'
        if complete:
            V.require(proof['scheduler_complete_joined']==proof['verified_trials']==m['planned']==r['states']['COMPLETED'] and not proof['unjoined_or_partial'] and proof['domains']==8,'FULL_GLOBAL_PORT_COVERAGE')
        proof['complete']=complete;result['port_dispatch']=proof
    except (V.Invalid,KeyError,ValueError,TypeError,IndexError,OSError,OverflowError) as e:
        # An unreadable/empty final document already bound as failed save stays
        # incomplete; no Port proof may upgrade missing evidence to success.
        if result['status']=='INCOMPLETE_EVIDENCE' and isinstance(e,(FileNotFoundError,json.JSONDecodeError)):
            result['evidence_gaps'].append('Port proof unavailable after failed persistence:'+str(e))
        else:result.update(status='INTEGRITY_MISMATCH',issues=result['issues']+[str(e)],comparison_eligible=False)
    return result
