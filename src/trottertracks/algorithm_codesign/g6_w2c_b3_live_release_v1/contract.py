"""Preparation approval is distinct from formal adoption and execution approval."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import ast
from .guard import install, require
from ..g6_w2c_b3_live_preflight_v1 import contract as OLD, spec as S

PKG=S.SOURCE+'g6_w2c_b3_live_release_v1/'
DOC='docs/research/track_b_g6_w2c_b3_live_correctness_release_preparation_20261020/'
BASE='6c6d3376e7ff206219f5e58355f1e1080fafba04'
ARTIFICIAL_SEED='ARTIFICIAL_LIVE_RELEASE_V1'


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
    return dict(schema='b3_real_authority_requirements_v1',real_authority_implemented=False,
        required_bindings=['SOURCE','input_precision','API_contract','runtime_libraries','worker_lineage',
            'seed','global_scope_claim','call_budgets','global_bytes','storage_destination','monitoring',
            'retry_resume','STOP','receipt_caller'],
        external_decisions=['store_owner','grant_issuer','credential_key_custody','revocation_rotation',
            'atomicity_ACK_unknown_contract','power_loss_durability','worker_containment',
            'receipt_caller_durability','execution_environment_responsibility'],
        test_prototype='Owned SQLite FULL synchronous + durable attempt intent; TEST_ONLY no credentials',
        real_store_access=False)


def candidate(root):
    install();root=Path(root).resolve();c=deepcopy(OLD.candidate(root))
    pins=dict(c['source_pins'])
    paths=list((root/PKG).glob('*'))+list((root/DOC).glob('*.py'))
    paths += [root/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_live_release_v1.py']
    pins.update({str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    c.update(schema='b3_live_release_candidate_v1',status='LIVE_RELEASE_CANDIDATE_UNAUTHORIZED',
        preparation_approval='USER_APPROVED_PREPARATION_SCOPE',SCIENCE_RUN_AUTHORIZED=False,
        fixed_base=BASE,source_pins=dict(sorted(pins.items())),source_release_candidate=S.digest(pins),
        call_budgets=call_budgets(),counter_semantics=counter_semantics(),
        real_authority_requirements=authority_requirements(),science_loader='STATIC_INACTIVE_SOURCE_ONLY',
        artificial_seed_namespace=ARTIFICIAL_SEED)
    from .. import g6_w2c_b3_parallel_admission as A
    c['worker_runtime']=A.manifest(str(root),c['source_pins'])
    c['worker_runtime']['artificial_fixture']=dict(path=PKG+'wrapper.py',sha256=pins[PKG+'wrapper.py'])
    # Verify the dormant resolver as syntax; do not import/compile/execute it.
    ast.parse((root/PKG/'scientific_resolver.py.inactive').read_bytes())
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
    for actual,expected_path in OLD.OLD.OLD.pins_loaded(c):
        require(actual==expected_path,'LOADED_LOCATION')
    return S.digest(m)


verify_source_commit=OLD.verify_source_commit
