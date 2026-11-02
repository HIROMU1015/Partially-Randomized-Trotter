"""Owned V2 fake inputs and source-bound original v3 worker factories."""
import b3_parallel_integrated_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_storage_v2 as R


class Fault:
    def __init__(self,point='',match=None,repeat=False):
        self.point,self.match,self.repeat=point,match or{},repeat
        self.events=[]

    def __call__(self,point,**context):
        if point==self.point and all(context.get(k)==v for k,v in self.match.items())and(not self.events or self.repeat):
            self.events.append(dict(point=point,context=context))
            raise OSError('ARTIFICIAL_STORAGE_FAILURE:'+point)


def fixture(root,settings):
    c=R.candidate(F.ROOT);inner,ledger,_=F.fixture(root,settings)
    inner['scope']=inner['r1_reference']['scope']='ARTIFICIAL_STORAGE_REPAIR_V2'
    inner['claim_id']=inner['r1_reference']['claim_id']='ARTIFICIAL_STORAGE_CLAIM_V2'
    inner['pilot_candidate_binding']=R.validate(c)
    inner['storage_protocol']=c['storage_protocol']
    return c,inner,ledger,F.authorize(inner,root,ledger)


def invoke(root,c,inner,ledger,auth,fault=None,admission_fault=None):
    return R.execute_artificial(c,inner,root=root,ledger=ledger,authority=auth,
        endpoint_factory=F.ArtificialFactory(inner['runtime_manifest']),bits_factory=F.bits_factory,
        storage_fault=fault or Fault(),admission_fault=admission_fault or Fault())
