"""Isolated subject process. Candidate functions are never the acceptance oracle."""
import importlib.abc
import json
from pathlib import Path
import sys

FORBIDDEN={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision',
           'g6_f3_generator','g6_w2c_b3_driver','g6_w2c_cost_preparation'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.',1)[-1] in FORBIDDEN:
            raise RuntimeError('SUBJECT_PRELOAD_DENIAL:'+fullname)
sys.meta_path.insert(0,Guard())
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/b3-independent-server-r2-20261020/runtime/src')
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C,result as R

def one(q):
    op=q['op']
    if op=='encode_examples':
        from fractions import Fraction as F
        values=[None,True,False,0,-1,2**4096,F(-11,17),(),[],(1,[F(2,3),False]),
                {3:'three',(1,2):{'unicode':'日本語'}},{'int':'0x1'}, {'rat':['0x1','0x2']}]
        return [C.Document.create(v).transport() for v in values]
    if op=='read':
        record=C.Document(**q['record']);value=record.read()
        return dict(record=C.Document.create(value).transport(),sidecar=C.convert(record,source_identity=q.get('source','TEST_SOURCE')))
    if op=='restore':
        return C.restore(q['sidecar'],source_identity=q['source']).transport()
    if op=='audit':
        return R.canonical_audit(C.DriverDocument(**q['seal']),**q['expected'])
    if op=='plain_unsupported':
        return C.Document.create(q['input_value']).transport()
    if op in ('result','failure','g7','angle'):
        record=C.Document(**q['record'])
        if op=='result':return R.adapt_f3(record,expected=q['expected'])
        if op=='failure':return R.failure(record).to_dict()
        if op=='g7':return R.adapt_g7(C.Document(**q['event']).read(),observation=record,expected=q['expected'],arm=q['arm'])
        adapted=C.Document(**q['adapted']).read()
        kwargs=dict(requery_input=C.Document(**q['query']).read(),requery_event=C.Document(**q['event']).read(),angle_record=record,synthetic=True)
        if q['arm']=='full_return':return R.angle_binding(adapted,**kwargs)
        return R.canonical_angle_binding(adapted,**kwargs,input_payload={'id':'ARTIFICIAL_INPUT_V1'},policy_payload={'id':'ARTIFICIAL_PRECISION_V1'})
    if op=='bits':
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import bits as B
        b=B.ArtificialBits(q['arm'],q['axis'],namespace=q.get('namespace',B.SEED))
        if 'counter' in q:b.counter=q['counter']
        if 'domain' in q:b.domain=q['domain']
        for width in q['widths']:
            try:b.bits(width)
            except BaseException as exc:
                return dict(rejected=True,reason=str(exc),draws=b.trace,counter=b.counter,
                            consumed_bits=b.consumed_bits,failure_position=b.failure_position,stopped=b.stopped)
        return dict(draws=b.trace,counter=b.counter,consumed_bits=b.consumed_bits)
    if op=='closed':
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import bits as B,contract as K
        from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import ports as P
        observations=[]
        for label,call in [('production_entry',K.production_entry),('real_bits',B.real_bits)]:
            try:call();observations.append(dict(name=label,rejected=False))
            except BaseException as exc:observations.append(dict(name=label,rejected=True,reason=str(exc)))
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.bridge import ProductionFactory,ProductionBits
        for factory in (ProductionFactory,ProductionBits):
            try:factory()();observations.append(dict(name=factory.__name__,rejected=False))
            except BaseException as exc:observations.append(dict(name=factory.__name__,rejected=True,reason=str(exc)))
        # Call the finders directly with names as data. Never attempt science import.
        P.install_guard()
        for finder in sys.meta_path:
            if type(finder) is P.BlockScience:
                for name in FORBIDDEN:
                    try:finder.find_spec('trottertracks.algorithm_codesign.'+name);observations.append(dict(name=name,rejected=False))
                    except BaseException as exc:observations.append(dict(name=name,rejected=True,reason=str(exc)))
        return observations
    raise ValueError('UNKNOWN_PROBE')

def main():
    rows=[]
    for q in json.load(sys.stdin):
        try:rows.append(dict(id=q['id'],ok=True,result=C.canonical(one(q))))
        except BaseException as exc:rows.append(dict(id=q['id'],ok=False,type=type(exc).__name__,reason=str(exc)))
    print(json.dumps(dict(rows=rows,science_modules_loaded=[x for x in sys.modules if x.rsplit('.',1)[-1] in FORBIDDEN])))

if __name__=='__main__':main()
