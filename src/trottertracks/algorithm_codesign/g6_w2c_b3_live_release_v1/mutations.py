"""Semantic changes to saved artificial payloads, including resealed originals."""
from copy import deepcopy
from pathlib import Path
import json
from . import offline as O
from ..g6_w2c_b3_real_adapter_v1 import codec as C
from .. import g6_w2c_b3_offline_result as V


def reseal(p,field,value):
    a=p['evidence']['adapted'];doc=C.restore(a['codec'],source_identity=a['source_identity']);data=doc.read()
    data[field]=value;a['codec']=C.convert(C.Document.create(data),source_identity=a['source_identity'])


CHANGES={
 'global_trial':lambda p:p['evidence']['connection'].__setitem__('global_trial_id','wrong'),
 'source':lambda p:p['evidence']['connection'].__setitem__('source_binding','0'*64),
 'runtime':lambda p:p['evidence']['connection'].__setitem__('runtime_binding','0'*64),
 'session':lambda p:p['evidence']['connection'].__setitem__('session','0'*64),
 'worker':lambda p:p['evidence']['connection'].__setitem__('worker_pid',0),
 'trial':lambda p:p.__setitem__('trial_id',p['trial_id'][:-1]+'1'),
 'event':lambda p:p['evidence']['adapted'].__setitem__('event',[]),
 'event_fingerprint':lambda p:p['evidence']['adapted'].__setitem__('event_fingerprint','0'*64),
 'digest':lambda p:p['evidence']['adapted']['codec']['f3'].__setitem__('digest','0'*64),
 'sidecar':lambda p:p['evidence']['adapted']['codec'].__setitem__('driver_digest','0'*64),
 'seed':lambda p:p['evidence']['consumption'].__setitem__('domain','SCIENCE_UNAUTHORIZED'),
 'width':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('width',159),
 'counter':lambda p:p['evidence']['consumption'].__setitem__('counter_after',999),
 'missing_angle':lambda p:p.__setitem__('angle',None),
 'requery':lambda p:p['angle']['requery_input'].__setitem__('child',99),
 'angle_digest':lambda p:p['angle']['angle']['f3'].__setitem__('digest','0'*64),
 'audit_digest':lambda p:p['evidence']['audit']['original_driver_seal'].__setitem__('digest','0'*64),
 'counter_limit':lambda p:p['evidence']['boundary_counts'].__setitem__('scheduled_sampling',999),
 'resealed_original_trial':lambda p:reseal(p,'trial_id','wrong'),
 'resealed_original_source':lambda p:reseal(p,'source_id','wrong'),
}


def run(root,c,m):
    task=next(t for t in m['tasks'] if t['condition']=='S' and t['arm']=='ordinary' and t['layer']=='B')
    path=Path(root)/'output'/(task['key']+'.jsonl')
    original=V.decode(json.loads(path.read_bytes().splitlines()[0])['payload'])
    O.payload(original,c,m,task)
    rows=[]
    for name,change in CHANGES.items():
        p=deepcopy(original);change(p)
        try:O.payload(p,c,m,task)
        except Exception as exc:rows.append(dict(name=name,status='REJECTED',reason=str(exc)))
        else:rows.append(dict(name=name,status='ACCEPTED_UNEXPECTEDLY'))
    return dict(total=len(rows),PASS=sum(r['status']=='REJECTED' for r in rows),
        FAIL=sum(r['status']!='REJECTED' for r in rows),mutations=rows,
        same_author=True,scope='payload semantic mutations; not a rebuild of historical independent Storage reseals')
