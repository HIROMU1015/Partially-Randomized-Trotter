"""Versioned integration contract. Original bundle and port bytes are pinned."""
from pathlib import Path
from copy import deepcopy
import sys
from .. import g6_w2c_b3_bundle_contract_v1 as OLD
from ..g6_w2c_b3_api_binding_v1 import static as S
from .. import g6_w2c_b3_parallel_admission as A

SOURCE=OLD.SOURCE
DOC='docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019/'
PKG=SOURCE+'g6_w2c_b3_port_bundle_v1/'
PORT=SOURCE+'g6_w2c_b3_api_binding_v1/'
SCHEMA='b3_closed_port_global_bundle_candidate_v1'
MANIFEST='b3_closed_port_global_bundle_manifest_v1'
CLOSED=OLD.CLOSED
FINAL_SLOT,DELIVERY_SLOT,OBSERVATION_SLOT=OLD.FINAL_SLOT,OLD.DELIVERY_SLOT,OLD.OBSERVATION_SLOT
encoded,digest,plan=OLD.encoded,OLD.digest,OLD.plan
BASE='d24cccbecdf0928fe977080dd7b28c591df0089e'
PORT_COMMIT='82896d53fa97235fa848d6c0833eb193af8b99d6'
NEW_FILES=[PKG+'bridge.py']
FIXED_PINS={'docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/pilot.schema.json': '773d61abdd2e87d9a740e7ea9af671aeee66cadee42ce3c13a3cfcda6833cfd3', 'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/bundle.schema.json': 'a83bdd9a965264de4a8c4e54bc14b0203b3baf42afca89f274bc428c7b5d6d93', 'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/fixed_scientific_reference.json': '54b97c5d9de78532eaa225d89fa540586ec3ecefe1e456d8c7fdc3ea10f82d03', 'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/run_case.py': 'b7c3183ffb6a6480f6981934cc2283eb92975944b2da72e1e6cfeacb0df6c1d2', 'src/trottertracks/__init__.py': '2d51bfa08b2b1e000b86ec53bd463e786529307a7270078600e3b03995e90285', 'src/trottertracks/algorithm_codesign/__init__.py': 'b6503716cfd51d52ca9038e6de5a411eb78380d30451f81df9b81c31b1be090f', 'src/trottertracks/algorithm_codesign/g6_f3_contract.py': '754c27f72f3356b3072ced4a00bef174b8922d2efedd0eff18d86a887fa598e1', 'src/trottertracks/algorithm_codesign/g6_f3_generator.py': '2ccce19571cdf33527abc92f049f35ef4e0d33d8a876132b2d1d70f6e63ad751', 'src/trottertracks/algorithm_codesign/g6_f3_precision.py': 'c2136326042fcf95240ecae41436cd953a1f857543ec9797e57f8fcbbe158706', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_admission.py': 'b718047a96353f60bfda4ee08694e610be7e53c701185d1ceda539146f1ae750', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/__init__.py': '794bb6d12f78f64073f932cbf47ed18f30145a315bac89bddb89b9bf7eb7c781', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/authority.py': '5cf47cc928ac757bec21b0fd39e1afaacc0b326864fae247ef296f8fd924d786', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/exercise.py': '442264facd91d55cd7eb288651a5a83cfaa42c22942674bdffab219db9f2db07', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/offline.py': 'bb8a2c97967c3372e8246a002ce58398c0daeba2a382444cde1fa0b2a5770c9e', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/ports.py': 'b3269593d207779b90433bd9e7487e289b8fe67833efa26d3cf3d62216f434cb', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/static.py': '7d16a25d3ca3c1cc5648cc4ad1c96c579fc6c4f75957ff996d22792a4784c4e7', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_bridge_v1.py': 'cf35e8a2e6af0da6e9a680943300efddab9157e1b00a7325c6340f877be3b485', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_contract_v1.py': 'bb292de38691d39b2687759da7ccefdbb1cc046544f560905638a6564a85cc12', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_fixture_v1.py': '0b44a56d029e88355dc0909e4670cc08225676bb82aca80ad189a90030537e05', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_offline_v1.py': '2e0a94f200745380c583f2cd69356628e507bd4a3e6f73077c8c1278adc1857b', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_store_v1.py': 'c680640ccf7c00520c67fd9dcd0b3769ca6ac3980f35d5875f996d689b555fd4', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_trace_v1.py': '964e9605d59ddce2de6fe373bc451ba83c19d0a1448c44c791c33af87d46a067', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py': '5e6904e782526e96cc6a2dc9e94642a8237b4aa69ce8743aec24ccde5f212694', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_exact_serialization.py': 'e05326d95d6c8e91d779e23db3283ca2cd41b6eab2fb1024707c112c1c93087e', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_host.py': '49e753184b66b22b41287acade1c9ac6a43249321481c6fe53149b43c2bd03b9', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_integration_acceptance.py': 'd67b1b63047e91d80fd37c63a4fee9474187d37e85c0123e8ec441bd96f2b4dd', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_offline_cli.py': 'c8d8187d477999d67baff92088812892f5f36d5d5c25605210e12379870be883', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_offline_cost.py': 'cec5584ca988a3865cd5ff8d2c5f387a44ac8c17da0ddd5c5196dcc8cfb5c578', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_offline_result.py': '88d052c90dd140756d4153e2ad8d21a22d26677f656f76c2afe3bc2635c246cd', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_admission.py': '08fcac3171106ed3a8e1551ed12cbdfb5dfdf7e41a26b10c4242b41fc1d63c59', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_candidate.py': '3324b93087a104511c8c18f7bd12e38d643ff6614b53ccd29883b7cd112ff76d', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_driver.py': '4cf515a439334fcf893e1219f64c769a3bd9ba8ea6bf8eb9cedcc8b12ed7813f', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_host.py': 'c3c7269bdd2d0f459248de22f2d64952d02a4170b606ad5a466b0caeca04f9b6', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_observer.py': '50533aec6e6b007df23372d1c4c0b01bc6dc6348a83fede4aef5e80e4abd66ef', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_offline_adapter.py': '74371b9ea9fccd59a501ec99d44578b1747dbcc373d30831bff68ad085261c4d', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_offline_cli.py': 'f241c1be5e9acb0f252c537f854eff70c2d981565a30e6c1aa028d04fe38309e', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_offline_cost.py': '73051803a8a39491effbe04d26e1053652c183f7eba66d90fa9f4dae83f84051', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_overhead.py': 'ff9213fa6ccbd65b8c20e9eec1576d2c43d5dc3b29a035f50271582804ba392e', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_scheduler.py': '48cbcbec290404b71392b1fe24742e0d07236c0bebbc6f2e5bfe641109e614ed', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_verifier_core.py': 'e915594d780d2ef43a3dfd80017220389917267235fbb1d81c25e7c4b99ae4a1', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_candidate.py': 'c3bc48ee7dd85b77645c763b5d883710c5f20e68116498f81b4723a474c489fa', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_adapter.py': 'd8015cbc07b970addcc741c804cb826e243018f2706f6954a0f4ae30cd9473d0', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_cli.py': '8bc85e4ff6309501d8f2dc1852cfe510586d3ce91702749ad72f251f389acc6d', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_identity.py': '90cfab6d2b544773391b41558be1f84c1e7957697d4c07b44cabdc280397e867', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_trace.py': '8380044d544c8374331b36372074cb2227c83797be132e17611120e7666a62cb', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_offline.py': '14f99fc953001907f160bab3f614c4ca552ee0845bfdd1612abde8a4df15e14c', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_storage_v2.py': 'f97b917e49f694ec4a34546b27dde74c1db4096cfe05179fb61682f604c2ddab', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_readiness.py': 'f6cbb170b99bd67926c66eeb982df67a1dd4506dcc8fc40f5016cf482341b00a', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_storage_verify_v2.py': '605e0fdc1d829a77c93104aa236777461dd7f9dfbdc3978b09713f0eaf290fae', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_acceptance.py': '8cd352e84059e38c8961be4d53b8c39a4c0ceb845d0e83bdee1d5805db9f7d18', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_adapter.py': 'd5563b33ed26f32e535a1e22d0fb639f384fae3cdd5b901b9050a53f313110bb', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_cli.py': '487d9be2cde7873c297a2ca21b403958c755b6bb3047b57a77c61adc588d3c41', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_cost.py': '5d8eb20c9f9020a7445a047d2329426c82b52745e0fc61744e5666b05df4495e', 'src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_identity.py': '0430b1ced11cd5b8d876bf3c674ee33d62e7e8f8c0bf466f944b497f7e069b2c', 'src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py': '35a4a0d8b28fd1a50791cc2afd09ac86800a1f37b488dc4cee5906864ea7f0e1', 'src/trottertracks/algorithm_codesign/g7_generator.py': 'bb4177d26f1cee9721c2b083de01f4aa309f3ad39f22597cfae3b662881269fb', 'src/trottertracks/algorithm_codesign/g7_provider.py': '17fff6d4d9fa9c9728091f6f01a4e6af1a389e8949c6e4a2ec6e3112f2fe3c62', 'src/trottertracks/algorithm_codesign/return_aggregation.py': 'f716d3c69fc1063e47b7618022662715f6b34a9c1f35cc3e7ed26813f5ac69f0'}

def endpoint_binding(pins):
    return dict(schema='b3_ARTIFICIAL_port_injection_binding_v1',artificial_only=True,
        endpoint='trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1.ports.ArtificialEndpoint',
        bits='trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1.ports.ArtificialBits',
        adapter='trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1.adapter.ArtificialFactory',
        bits_factory='trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1.adapter.bits_factory',
        original_port_commit=PORT_COMMIT,port_source={p:h for p,h in pins.items()if p.startswith(PORT)},
        adapter_source={p:h for p,h in pins.items()if p.startswith(PKG)},
        seed_namespace=S.SEED,variants=['PORT_V1','PORT_V2'],
        outcome='fixed Outcome plus dispatch evidence; versioned artificial event payload tag',
        real_dispatch=False,source_mutation_allowed=False)

def candidate(root):
    root=Path(root).resolve();c=deepcopy(OLD.candidate(root))
    for path,h in FIXED_PINS.items():S.need(S.sha((root/path).read_bytes())==h,'IMMUTABLE_BUNDLE_PORT_SOURCE:'+path)
    paths=set(c['source_pins'])
    paths.update(str(p.relative_to(root))for directory in (PORT,PKG)for p in (root/directory).glob('*.py'))
    paths.update([DOC+'run_case.py',DOC+'run_matrix.py',DOC+'integration.schema.json',
                  'tests/tracks/algorithm_codesign/test_g6_w2c_b3_port_bundle_v1.py'])
    pins={p:S.sha((root/p).read_bytes())for p in sorted(paths)}
    c.update(schema=SCHEMA,status='ARTIFICIAL_ONLY_UNADOPTED_EXECUTION_CLOSED',source_pins=pins,
        source_release_candidate=digest(pins),python=S.runtime(),
        fixed_bundle_commit=BASE,fixed_port_commit=PORT_COMMIT,
        endpoint_binding=endpoint_binding(pins),production_gate='NO_IMPLEMENTED_SCIENCE_DISPATCH',
        artificial_seed_namespace=S.SEED,integration='ONE_GLOBAL_COORDINATOR_REAL_PORT_DISPATCH_V1')
    runtime=A.manifest(str(root),pins)
    runtime['artificial_fixture']=dict(path=PKG+'adapter.py',sha256=pins[PKG+'adapter.py'])
    c['worker_runtime']=runtime
    return c

def manifest(c,scope,*,port_variant='PORT_V1',port_faults=None,**kw):
    if port_variant not in ('PORT_V1','PORT_V2'):raise ValueError('ARTIFICIAL_PORT_VARIANT')
    m=OLD.manifest(c,scope,**kw)
    m.update(schema=MANIFEST,seed_namespace=S.SEED,port_variant=port_variant,port_faults=port_faults or [],
             endpoint_binding=digest(c['endpoint_binding']))
    return m

def guard(c,m):
    S.need(encoded(c)==encoded(candidate(c['runtime_root'])),'INTEGRATION_SOURCE_RUNTIME_CONTRACT')
    S.need(not any(n.rsplit('.',1)[-1]in S.SCIENCE for n in sys.modules),'SCIENCE_ALREADY_LOADED')
    for name,value in CLOSED.items():
        S.need(type(m.get(name))is type(value)and m[name]==value,'MANIFEST_CLOSED:'+name)
    expected=manifest(c,m['global_scope'],profile_n=m['profile_n'],modes=tuple(m['modes']),
        result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'],call_limit=m['call_limit'],
        condition_faults=m['condition_faults'],sync_failure=m['sync_failure'],parent_fault=m['parent_fault'],
        port_variant=m['port_variant'],port_faults=m['port_faults'])
    S.need(encoded(expected)==encoded(m),'EXACT_GLOBAL_MANIFEST')
    S.need(type(m['call_limit'])is int and 0<=m['call_limit']<=m['planned']<=16896,'GLOBAL_CALL_CAP')
    S.need(m['global_scope'].startswith('ARTIFICIAL_ONLY/'),'ARTIFICIAL_SCOPE')
    for ch in ('result','log'):
        S.need(type(m['byte_limits'][ch])is int and 0<m['byte_limits'][ch]<=c['proposal'][ch+'_bytes'],'GLOBAL_BYTE_CAP')
    for path,h in c['scientific_reference']['input_pins'].items():
        S.need(S.sha((Path(c['runtime_root'])/path).read_bytes())==h,'FIXED_INPUT:'+path)
    for path in pins_loaded(c):
        S.need(path[0]==path[1],'LOADED_MODULE_LOCATION')
    return digest(m)

def pins_loaded(c):
    root=Path(c['runtime_root'])
    for name,module in list(sys.modules.items()):
        if not name.startswith('trottertracks.')or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        relative='src/'+name.replace('.','/')+('.py'if not path.name=='__init__.py'else'/__init__.py')
        if relative in c['source_pins']:yield path,root/relative
