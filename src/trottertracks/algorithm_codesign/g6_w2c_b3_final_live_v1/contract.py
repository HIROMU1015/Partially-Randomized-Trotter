"""Preparation approval is distinct from formal adoption and execution approval."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import ast
import sys
from .guard import install, require
from ..g6_w2c_b3_live_preflight_v1 import spec as S
from ..g6_w2c_b3_live_release_v1 import contract as OLD

PKG=S.SOURCE+'g6_w2c_b3_final_live_v1/'
DOC='docs/research/track_b_g6_w2c_b3_final_live_connection_simple_run_control_20261020/'
BASE='841ac8920e41e93e2e79574f6ecdc6213a570f88'
ARTIFICIAL_SEED='ARTIFICIAL_FINAL_LIVE_V1'


def call_budgets():
    b=deepcopy(S.budgets())
    b.update(event_witness=512,additional_sampling=0)
    return b


def counter_semantics():
    return dict(event_witness='512 is the driver correctness boundary. Nested B event_witness (<=256) is observed separately inside B sampling audit, never silently called a total function bound.',
        F3_prepare='2 includes the two nested Full constructors within primary_constructions16',
        B_event_requery='one per B accepted event; Full packet requery is a nested operation',
        bit_draws='384 canonical samples * at most6 + 128 Full samples * at most7 =3200; H160; fixed m5; no audit/requery draw; no retry',
        internal_total='NOT_ESTABLISHED: root/law/predicate/packet internals are not an aggregate call authorization')


def authority_requirements():
    return dict(schema='b3_simple_run_control_contract_v1',approver='USER',
        current_approval='PREPARATION_ONLY',run_authorized=False,
        required_bindings=['SOURCE','input_precision','API_contract','runtime_libraries','worker_lineage',
            'seed','global_scope_claim','call_budgets','global_bytes','storage_destination','monitoring',
            'retry_resume','STOP','receipt_caller'],
        one_shot=True,retry=0,resume=False,mandatory_STOP=True,
        states=['NOT_STARTED','START_RESERVED/SPENT','COMPLETED','STOPPED','UNKNOWN'],
        mechanism='owned exclusive intent files + SQLite UNIQUE transaction; no refund',
        activation='separate explicit USER approval + adopted exact SOURCE/manifest; current activation closed',
        removed_external_prerequisites=['external_keys','credential_issuance_revocation','authorization_server',
            'multiple_approval_authorities','physical_power_loss_certification'],
        contract_delta='user instruction 2026-10-20 section5; proposed for NEW scope only; old claims/STOP unchanged',
        physical_power_loss_durability='NOT_ESTABLISHED',malicious_tamper_resistance='NOT_CLAIMED')


def candidate(root):
    install();root=Path(root).resolve();c=deepcopy(OLD.candidate(root))
    pins=dict(c['source_pins'])
    paths=list((root/PKG).glob('*'))+list((root/DOC).glob('*.py'))
    paths += [root/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_final_live_v1.py']
    pins.update({str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    c.update(schema='b3_final_live_candidate_v1',status='FINAL_LIVE_PREPARATION_UNAUTHORIZED',
        preparation_approval='USER_APPROVED_PREPARATION_SCOPE',SCIENCE_RUN_AUTHORIZED=False,
        fixed_base=BASE,source_pins=dict(sorted(pins.items())),source_release_candidate=S.digest(pins),
        call_budgets=call_budgets(),counter_semantics=counter_semantics(),
        real_authority_requirements=authority_requirements(),science_loader='IMPLEMENTED_GATED_NOT_CALLED',
        artificial_seed_namespace=ARTIFICIAL_SEED)
    from .. import g6_w2c_b3_parallel_admission as A
    c['worker_runtime']=A.manifest(str(root),c['source_pins'])
    c['worker_runtime']['artificial_fixture']=dict(path=PKG+'wrapper.py',sha256=pins[PKG+'wrapper.py'])
    from .resolver import static_binding
    c['final_runtime_binding']=static_binding(root)
    c['authority_requirements']=authority_requirements()
    c['scientific_API_dispatch']='IMPLEMENTED_GATED_ACTIVATION_CLOSED'
    c['endpoint_binding']=dict(schema='b3_final_runtime_factory_v1',
        wrapper=PKG+'wrapper.py',resolver=PKG+'resolver.py',bits=PKG+'bits.py',
        gate=PKG+'gate.py',activation=PKG+'activation.py',
        built_in_artificial_only_currently=True,live_compatibility='NOT_ESTABLISHED')
    return c


def manifest(c,scope,destination,*,scenario='normal',result_bytes=1<<30,log_bytes=64<<20):
    require(type(scope) is str and scope.startswith('ARTIFICIAL_ONLY/RELEASE/'),'TEST_SCOPE_ONLY')
    require(scenario in ('normal','coverage','S_angle','P1_angle','scientific_mismatch','receipt_close',
                         'byte_cap','call_cap','worker_exit','unknown_ack'),'BUILTIN_SCENARIO')
    require(type(result_bytes) is int and 0<result_bytes<=1<<30 and type(log_bytes) is int
            and 0<log_bytes<=64<<20,'BYTE_CAP')
    return dict(schema='b3_release_test_manifest_v1',**S.CLOSED,
        preparation_approval='USER_APPROVED_PREPARATION_SCOPE',SCIENCE_RUN_AUTHORIZED=False,
        candidate_binding=S.digest(c),source_release_candidate=c['source_release_candidate'],
        global_scope=scope,global_claim=S.digest(['TEST_ONLY_CLAIM',scope,S.digest(c)]),
        run_id=S.digest(['TEST_ONLY_RUN',scope,S.digest(c)]),tasks=S.tasks(),planned=512,modes=['S','P1'],
        seed_namespace=ARTIFICIAL_SEED,proposed_scientific_namespace=S.SEED,scenario=scenario,
        call_budgets=call_budgets(),byte_limits=dict(result=result_bytes,log=log_bytes),
        storage_destination=str(Path(destination).resolve()),runtime_binding=S.digest(c['worker_runtime']),
        authority_binding=S.digest(dict(requirements=authority_requirements(),candidate=S.digest(c),
            destination=str(Path(destination).resolve()),scope=scope)),
        monitor_binding=S.digest(c['monitoring']),real_grant=None,real_authority=False,
        retention='ALL_512_FULL_DOCUMENTS_NO_ADAPTIVE_REDUCTION')


def preflight(c,m):
    install();require(S.encoded(c)==S.encoded(candidate(c['runtime_root'])),'CANDIDATE_SOURCE_API_RUNTIME')
    expected=manifest(c,m['global_scope'],m['storage_destination'],scenario=m['scenario'],
        result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'])
    require(S.encoded(m)==S.encoded(expected),'MANIFEST_AUTHORITY_SEED_BUDGET_STOP_BINDING')
    for p,h in c['input_pins'].items():
        require(sha256((Path(c['runtime_root'])/p).read_bytes()).hexdigest()==h,'INPUT_PIN:'+p)
    for actual,expected_path in OLD.OLD.OLD.OLD.pins_loaded(c):
        require(actual==expected_path,'LOADED_LOCATION')
    check_loaded_management(c)
    return S.digest(m)


verify_source_commit=OLD.verify_source_commit


def run_manifest_candidate(c, *, source_commit=None):
    """Approval target template. Never grants or consumes a real permission."""
    run_id='B3_CORRECTNESS_512/20261020_FINAL_V1_'+c['source_release_candidate'][:16]
    state_root=Path(c['runtime_root'])/'artifacts/track_b_g6_w2c_b3_correctness_512/2026-10-20'/c['source_release_candidate'][:16]
    return dict(schema='b3_correctness_run_manifest_candidate_v1',**S.CLOSED,
        SCIENCE_RUN_AUTHORIZED=False,approval_status='PENDING_EXPLICIT_USER_EXECUTION_APPROVAL',
        execution_state='NOT_STARTED',run_id=run_id,global_scope=run_id,global_claim=S.digest([run_id,S.digest(c)]),
        measured_SOURCE=source_commit,science_SOURCE=S.SCIENCE,
        source_release_candidate=c['source_release_candidate'],candidate_binding=S.digest(c),
        input=S.INPUT,precision=S.PRECISION,A6=S.ANGLE,
        runtime=c['python'],worker_runtime=c['worker_runtime'],runtime_binding=S.digest(c['worker_runtime']),
        seed_namespace=S.SEED,seed_domain='namespace/arm/axis',mode_in_seed=False,
        global_trial_identity='run_id/mode/correctness/baseline/arm/layer/axis/index',
        tasks=S.tasks(),planned=512,modes=['S','P1'],call_budgets=call_budgets(),
        byte_limits=dict(result=1<<30,log=64<<20),storage_destination=str(state_root/'output'),state_directory=str(state_root),
        one_shot=True,real_grant=None,approval_record=None,activation_record=None,
        retention='ALL_512_FULL_DOCUMENTS_NO_ADAPTIVE_REDUCTION',
        coverage='INCONCLUSIVE_COVERAGE_IF_ANY_CELL_ACCEPTED_ZERO_NO_EXTENSION',
        stop_conditions=['SOURCE_INPUT_RUNTIME_MISMATCH','UNAUTHORIZED','BITS_CONSUMPTION',
            'RESULT_EVIDENCE_CODEC','ANGLE_REQUERY','SCIENTIFIC_INVARIANT','WORKER','STORAGE',
            'CALL_BYTE_BUDGET','UNKNOWN','ENTER_AFTER_STOP'],
        monitoring=deepcopy(c['monitoring']),simple_control_contract=authority_requirements())


def preflight_runtime(c,m,permit):
    from .gate import require_live
    require_live(permit,c,m)
    require(S.encoded(c)==S.encoded(candidate(c['runtime_root'])),'RUNTIME_SOURCE_API_BINDING')
    require(m==permit.manifest,'RUNTIME_APPROVAL_MANIFEST')
    expected=run_manifest_candidate(c,source_commit=m['measured_SOURCE'])
    mutable={'run_id','global_scope','global_claim','storage_destination','state_directory',
        'approval_record','activation_record','approval_status','SCIENCE_RUN_AUTHORIZED',
        'run_ready','content_adopted','contract_grants_execution','science_release','scientific_adoption',
        'artificial_only'}
    require({k:v for k,v in m.items() if k not in mutable}==
            {k:v for k,v in expected.items() if k not in mutable},'RUNTIME_FIXED_CONTRACT')
    require(m['SCIENCE_RUN_AUTHORIZED'] is True and m['retry']==0 and m['resume'] is False
            and m['mandatory_STOP'] is True,'RUNTIME_EXECUTION_APPROVAL')
    check_loaded_management(c)
    verify_source_commit(c,m['measured_SOURCE'])
    for p,h in c['input_pins'].items():
        require(sha256((Path(c['runtime_root'])/p).read_bytes()).hexdigest()==h,'RUNTIME_INPUT:'+p)
    return S.digest(m)


def check_loaded_management(c):
    prefix='trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1'
    root=Path(c['runtime_root'])
    for name,module in list(sys.modules.items()):
        if name==prefix or name.startswith(prefix+'.'):
            path=PKG+('__init__.py' if name==prefix else name[len(prefix)+1:].replace('.','/')+'.py')
            require(path in c['source_pins'] and Path(module.__file__).resolve()==root/path,
                'FINAL_LOADED_LOCATION:'+name)
            require(sha256((root/path).read_bytes()).hexdigest()==c['source_pins'][path],
                'FINAL_LOADED_SOURCE:'+name)
