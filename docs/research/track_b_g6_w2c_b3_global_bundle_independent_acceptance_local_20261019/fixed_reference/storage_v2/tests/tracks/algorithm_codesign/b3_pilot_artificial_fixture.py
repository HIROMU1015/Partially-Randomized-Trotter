"""Fresh private fake scope; unchanged v3 worker factories contain no authority."""
from pathlib import Path
import b3_parallel_integrated_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_candidate as C


def fixture(root,settings=None):
    c=C.candidate(F.ROOT)
    inner,ledger,_=F.fixture(root,settings or F.P.Settings(profile_n=16,codec='fused',workload='light'))
    inner['scope']=inner['r1_reference']['scope']='ARTIFICIAL_PILOT_FORWARD_V1'
    inner['claim_id']=inner['r1_reference']['claim_id']='ARTIFICIAL_PILOT_CLAIM_FORWARD_V1'
    inner['pilot_candidate_binding']=C.validate(c)
    auth=F.authorize(inner,root,ledger)
    return c,inner,ledger,auth


def invoke(root,c,inner,ledger,auth,**kwargs):
    return C.execute_artificial(c,inner,root=root,ledger=ledger,authority=auth,
        endpoint_factory=F.ArtificialFactory(inner['runtime_manifest']),bits_factory=F.bits_factory,**kwargs)
