"""Adopted fixed pilot schedule; fake manifests cannot authorize science."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import sys
from .guard import install,require
from . import spec as S
from ..g6_w2c_b3_final_live_v1 import contract as OLD
PKG=S.PKG
DOC='docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/'
ARTIFICIAL_SEED='ARTIFICIAL_LIMITED_PILOT_LIVE_V1'
BASE='f6088e2cc453816b30e985e52b10696830844fa9'
SCENARIOS=('normal','coverage','P1_angle','S_angle','receipt_close','unknown_ack','scientific_mismatch','worker_exit')

def candidate(root):
    install();root=Path(root).resolve();c=deepcopy(OLD.candidate(root));pins=dict(c['source_pins'])
    for path in list((root/PKG).glob('*.py'))+[root/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_limited_pilot_live_v1.py']:
        if path.is_file():pins[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
    c.update(schema='b3_limited_pilot_live_source_v1',fixed_base=BASE,source_pins=dict(sorted(pins.items())),
        source_release_candidate=S.digest(pins),call_budgets=S.budgets(),science_authority='REQUIRES_BOUND_USER_APPROVAL',
        endpoint_binding=dict(wrapper=PKG+'wrapper.py',resolver=PKG+'resolver.py',gate=PKG+'gate.py',activation=PKG+'activation.py'),
        tasks=S.tasks(),pilot_input=S.INPUT,pilot_precision=S.PRECISION,pilot_A6=S.ANGLE)
    from .. import g6_w2c_b3_parallel_admission as A
    c['worker_runtime']=A.manifest(str(root),c['source_pins'])
    c['worker_runtime']['artificial_fixture']=dict(path=PKG+'wrapper.py',sha256=pins[PKG+'wrapper.py'])
    from .resolver import static_binding
    c['final_runtime_binding']=static_binding(root)
    return c


def base_manifest(c):
    return dict(schema='b3_limited_pilot_execution_manifest_v1',**S.CLOSED,SCIENCE_RUN_AUTHORIZED=False,
        approval_status='PENDING',execution_state='NOT_STARTED',candidate_binding=S.digest(c),
        source_release_candidate=c['source_release_candidate'],science_SOURCE=S.SCIENCE,
        input=S.INPUT,precision=S.PRECISION,A6=S.ANGLE,seed_namespace=S.SEED,seed_domain='namespace/arm/axis',mode_in_seed=False,
        tasks=S.tasks(),planned=16896,modes=['S','P1'],profile_n=256,workers=8,
        phase_order=['S_correctness','S_baseline','S_instrumented','P1_correctness','P1_baseline','P1_instrumented'],
        runtime=c['python'],worker_runtime=c['worker_runtime'],runtime_binding=S.digest(c['worker_runtime']),
        call_budgets=S.budgets(),byte_limits=dict(result=1<<36,log=1<<28),one_shot=True,real_grant=None,
        coverage='CORRECTNESS_FIXED16_ACCEPTED_GE1_ELSE_INCONCLUSIVE_AND_STOP_NO_NEXT_PHASE',
        retention='CORRECTNESS_FULL_PROFILE_FIRST16_FULL_REST_EXACT_COMPACT_DIGEST_RESULT_BITS',
        observer='BASELINE_OFF_INSTRUMENTED_CACHED_ALL_CALL_RETURN',codec='fused',
        global_trial_identity='run_id/mode/phase/pass/arm/layer/axis/index',
        monitoring=dict(outer='pre-admission through all joins/finalization/STOP; export excluded',coarse_fine_added=False,
            estimand='DESCRIPTIVE_FIXED_PREFIX_NO_IID_CI',CPU='parent own + reaped child hierarchy; worker own END separately',
            worker_trial='generator through retention before emit; storage/IPC outside this trial window',blocks=[16,64,256]),
        scenario='normal',approval_record=None,activation_record=None)


def manifest(c,scope,destination,*,scenario='normal'):
    require(scope.startswith('ARTIFICIAL_ONLY/RELEASE/') and scenario in SCENARIOS,'ARTIFICIAL_MANIFEST')
    m=base_manifest(c);m.update(seed_namespace=ARTIFICIAL_SEED,scenario=scenario,global_scope=scope,
        global_claim=S.digest(['ARTIFICIAL_PILOT_CLAIM',scope,S.digest(c)]),run_id=S.digest(['ARTIFICIAL_PILOT_RUN',scope,S.digest(c)]),
        storage_destination=str(Path(destination).resolve()),state_directory=str(Path(destination).resolve().parent))
    return m


def run_manifest_candidate(c,*,source_commit=None):
    m=base_manifest(c);run='B3_LIMITED_PILOT_16896/20261020_V1_'+c['source_release_candidate'][:16]
    root=Path(c['runtime_root'])/'artifacts/track_b_g6_w2c_b3_limited_pilot_16896/2026-10-20'/c['source_release_candidate'][:16]
    m.update(run_id=run,global_scope=run,global_claim=S.digest([run,S.digest(c)]),measured_SOURCE=source_commit,
        state_directory=str(root),storage_destination=str(root/'output'))
    return m

verify_source_commit=OLD.verify_source_commit

def loaded(c):
    root=Path(c['runtime_root']);prefix='trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1'
    for name,module in list(sys.modules.items()):
        if name==prefix or name.startswith(prefix+'.'):
            path=PKG+('__init__.py' if name==prefix else name[len(prefix)+1:].replace('.','/')+'.py')
            require(path in c['source_pins'] and Path(module.__file__).resolve()==root/path and
                sha256((root/path).read_bytes()).hexdigest()==c['source_pins'][path],'PILOT_LOADED_SOURCE:'+name)


def preflight(c,m):
    install();require(S.encoded(c)==S.encoded(candidate(c['runtime_root'])),'PILOT_SOURCE_RUNTIME')
    require(S.encoded(m)==S.encoded(manifest(c,m['global_scope'],m['storage_destination'],scenario=m['scenario'])),'PILOT_ARTIFICIAL_EXACT_MANIFEST')
    loaded(c)
    for p,h in c['input_pins'].items():require(sha256((Path(c['runtime_root'])/p).read_bytes()).hexdigest()==h,'PILOT_INPUT:'+p)
    return S.digest(m)


def preflight_runtime(c,m,permit):
    from .gate import require_live
    require_live(permit,c,m)
    require(S.encoded(c)==S.encoded(candidate(c['runtime_root'])),'PILOT_SOURCE_RUNTIME')
    expected=run_manifest_candidate(c,source_commit=m['measured_SOURCE'])
    mutable={'approval_record','activation_record','approval_status','SCIENCE_RUN_AUTHORIZED','artificial_only','run_ready',
        'content_adopted','contract_grants_execution','science_release','scientific_adoption'}
    require({k:v for k,v in m.items() if k not in mutable}=={k:v for k,v in expected.items() if k not in mutable},'PILOT_FIXED_ADOPTED_CONTRACT')
    require(S.encoded(m)==S.encoded(permit.manifest),'PILOT_PERMIT_MANIFEST');loaded(c);verify_source_commit(c,m['measured_SOURCE'])
    for p,h in c['input_pins'].items():require(sha256((Path(c['runtime_root'])/p).read_bytes()).hexdigest()==h,'PILOT_INPUT:'+p)
    return S.digest(m)
