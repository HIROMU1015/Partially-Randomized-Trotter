"""Unadopted adapter candidate; fixed historical SOURCE bytes remain untouched."""
from copy import deepcopy
from pathlib import Path
import sys
from . import codec as C, bits as Bits, binding as Binding
from ..g6_w2c_b3_port_bundle_v1 import contract as OLD
from ..g6_w2c_b3_api_binding_v1 import static as S
from .. import g6_w2c_b3_parallel_admission as A

SOURCE,PORT=OLD.SOURCE,OLD.PORT
PKG=SOURCE+'g6_w2c_b3_real_adapter_v1/'
DOC='docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/'
CLOSED=OLD.CLOSED
FINAL_SLOT,DELIVERY_SLOT,OBSERVATION_SLOT=OLD.FINAL_SLOT,OLD.DELIVERY_SLOT,OLD.OBSERVATION_SLOT
encoded,digest,plan=OLD.encoded,OLD.digest,OLD.plan
SCHEMA='b3_closed_real_adapter_candidate_v1'
MANIFEST='b3_closed_real_adapter_manifest_v1'
BASE='62967269e611b61e73b70abe20e998d9c7eaaf50'


def endpoint_binding(pins):
    return dict(schema='b3_closed_real_adapter_binding_v1',artificial_only=True,
        endpoint='trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.adapter.Endpoint',
        adapter='trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.adapter.ArtificialFactory',
        bits='trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.bits.ArtificialBits',
        adapter_source={p:h for p,h in pins.items() if p.startswith(PKG)},
        seed_namespace=Bits.SEED,real_dispatch=False,backend='SYNTHETIC_DOCUMENTS_ONLY',
        required_real_release='SEPARATE_VERSIONED_SOURCE_FORMAL_CONTRACT_AND_APPROVAL')


def candidate(root):
    root=Path(root).resolve();c=deepcopy(OLD.candidate(root))
    pins=dict(c['source_pins'])
    paths=[str(p.relative_to(root)) for p in (root/PKG).glob('*.py')]
    paths += [str(p.relative_to(root)) for p in (root/DOC).glob('*.py')]
    paths += ['tests/tracks/algorithm_codesign/test_g6_w2c_b3_real_adapter_v1.py',
              'tests/tracks/algorithm_codesign/test_g6_w2c_b3_real_adapter_integration_v1.py']
    pins.update({p:C.digest_bytes((root/p).read_bytes()) for p in paths})
    c.update(schema=SCHEMA,**CLOSED,source_pins=dict(sorted(pins.items())),source_release_candidate=digest(dict(sorted(pins.items()))),
        python=S.runtime(),fixed_integration_commit=BASE,endpoint_binding=endpoint_binding(pins),
        artificial_seed_namespace=Bits.SEED,integration='ONE_GLOBAL_COORDINATOR_CLOSED_DOCUMENT_ADAPTER_V1',
        port_name='CLOSED_ADAPTER_V1',scientific_API_dispatch='NOT_IMPLEMENTED',
        codec_policy='ORIGINAL_F3_DOCUMENT_DIGEST_PLUS_DRIVER_PROJECTION_SIDECAR',
        monitoring=S.monitoring(),api_binding_candidate=Binding.specification())
    runtime=A.manifest(str(root),c['source_pins'])
    runtime['artificial_fixture']=dict(path=PKG+'adapter.py',sha256=pins[PKG+'adapter.py'])
    c['worker_runtime']=runtime
    return c


def manifest(c,scope,*,port_variant='CLOSED_ADAPTER_V1',port_faults=None,**kw):
    C.need(port_variant=='CLOSED_ADAPTER_V1','NO_BACKEND_SELECTOR')
    m=OLD.OLD.manifest(c,scope,**kw)
    m.update(schema=MANIFEST,seed_namespace=Bits.SEED,port_variant=port_variant,
             port_faults=port_faults or [],endpoint_binding=digest(c['endpoint_binding']))
    return m


def guard(c,m):
    C.need(encoded(c)==encoded(candidate(c['runtime_root'])),'ADAPTER_SOURCE_INPUT_RUNTIME_BINDING')
    C.need(not any(n.rsplit('.',1)[-1] in S.SCIENCE for n in sys.modules),'SCIENCE_ALREADY_LOADED')
    expected=manifest(c,m['global_scope'],profile_n=m['profile_n'],modes=tuple(m['modes']),
        result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'],call_limit=m['call_limit'],
        condition_faults=m['condition_faults'],sync_failure=m['sync_failure'],parent_fault=m['parent_fault'],
        port_variant=m['port_variant'],port_faults=m['port_faults'])
    C.need(encoded(m)==encoded(expected),'EXACT_GLOBAL_MANIFEST')
    C.need(m['global_scope'].startswith('ARTIFICIAL_ONLY/'),'ARTIFICIAL_SCOPE')
    C.need(type(m['call_limit']) is int and 0<=m['call_limit']<=m['planned']<=16896,'GLOBAL_CALL_CAP')
    for channel in ('result','log'):
        C.need(type(m['byte_limits'][channel]) is int and 0<m['byte_limits'][channel]<=c['proposal'][channel+'_bytes'],
               'GLOBAL_BYTE_CAP')
    for path,h in c['source_pins'].items():
        C.need(C.digest_bytes((Path(c['runtime_root'])/path).read_bytes())==h,'SOURCE_BYTES:'+path)
    for path,h in c['scientific_reference']['input_pins'].items():
        C.need(C.digest_bytes((Path(c['runtime_root'])/path).read_bytes())==h,'INPUT_BYTES:'+path)
    for actual,expected in OLD.pins_loaded(c):C.need(actual==expected,'LOADED_MODULE_LOCATION')
    return digest(m)


def production_entry(*args,**kwargs):
    raise RuntimeError('NO_REAL_DISPATCH_IMPLEMENTATION_IN_THIS_RELEASE')
