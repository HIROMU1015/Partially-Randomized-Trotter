"""Closed 16,896 proposal and static binding checks; no pilot execution release."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
from .guard import install,require,Closed
from . import contract as K
from ..g6_w2c_b3_live_preflight_v1 import spec as S

OLD='docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/'
PRIOR='docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/'
BASE='22ef92cef31f98e3e31f27a209405f81f5c4a0e4'


def tasks():
    rows=[];offset=0
    for condition in ('S','P1'):
        for phase,pass_name,n in [('correctness','baseline',16),('profile','baseline',256),('profile','instrumented',256)]:
            for arm in S.ARMS:
                for layer in ('A','B'):
                    rows.append(dict(condition=condition,phase=phase,pass_name=pass_name,arm=arm,layer=layer,
                        trials_per_axis=n,cells=2,offset=offset,scheduled_sampling=2*n))
                    offset+=2*n
    require(offset==16896,'PILOT_PROPOSAL_ARITHMETIC')
    return rows


def candidate(root,*,metadata_source=None):
    install();root=Path(root).resolve();c=K.candidate(root)
    old_path=root/OLD/'pilot_contract_candidate.json';old=json.loads(old_path.read_bytes())
    if metadata_source is not None:require(type(metadata_source) is str and len(metadata_source)==40
        and all(x in '0123456789abcdef' for x in metadata_source),'METADATA_SOURCE_FULL_SHA')
    return dict(schema='b3_post512_pilot_preflight_candidate_v1',status='UNADOPTED_EXECUTION_CLOSED',
        preparation_only=True,SCIENCE_RUN_AUTHORIZED=False,artificial_only=True,content_adopted=False,
        run_ready=False,contract_grants_execution=False,science_release=None,scientific_adoption=None,
        approval_record=None,activation_record=None,science_authorization=None,global_scope=None,
        global_claim=None,run_id=None,state_directory=None,storage_destination=None,retry=0,resume=False,mandatory_STOP=True,
        fixed_input=deepcopy(S.INPUT),precision=deepcopy(S.PRECISION),A6=deepcopy(S.ANGLE),
        science_reference=S.SCIENCE,metadata_preparation_SOURCE=metadata_source,prior_512_public_HEAD=BASE,
        prior_512_role='REFERENCE_ONLY_NOT_REUSED_OR_DEDUCTED',future_pilot_release_SOURCE=None,
        source_release_candidate=c['source_release_candidate'],source_pins=c['source_pins'],input_pins=c['input_pins'],
        worker_runtime=c['worker_runtime'],runtime_binding=S.digest(c['worker_runtime']),
        old_proposal_identity=dict(path=OLD+'pilot_contract_candidate.json',sha256=sha256(old_path.read_bytes()).hexdigest(),
            measured_SOURCE='b9cabcfc6cf304ab65fc8a540c33ce2173f6cbaa'),
        proposal=dict(adopted=False,profile_n=256,conditions=['S','P1'],order='S_THEN_P1',repetitions=1,
            scheduled_sampling=16896,breakdown=dict(correctness=512,baseline=8192,instrumented=8192),
            tasks=tasks(),seed_namespace=old['proposal']['seed_namespace'],seed_adopted=False,
            domain='namespace/arm/axis; mode/layer/pass excluded; new streams restart corresponding prefixes',
            IID_claim=False,codec='fused',observer='cached_all_call_return',
            result_bytes=old['proposal']['result_bytes'],log_bytes=old['proposal']['log_bytes'],
            byte_budgets_adopted=False,retention_adopted=False,
            retention_candidate='correctness full; profile full first16/axis then full constructed+hashed only, as old source; requires adoption and checker extension',
            CPU_estimand='DESCRIPTIVE_FIXED_PREFIX; parent + reaped child hierarchy; worker interpreter END observations separately; no coarse/fine addition',
            coverage_adopted=False,coverage_conflict='old pilot says STOP; adopted 512 says INCONCLUSIVE_COVERAGE with S/P1 continuation if technically complete',
            new_calls_not_included=['additional sampling','adaptive extension','native compile','Full-N']),
        execution_blocks=['UNADOPTED_SCIENTIFIC_DESIGN','PILOT_RELEASE_NOT_IMPLEMENTED',
            'FUTURE_SOURCE_AND_MANIFEST_NOT_FIXED','FRESH_SCOPE_NOT_ASSIGNED','NO_EXECUTION_APPROVAL'])


def validate(value,root):
    require(type(value) is dict and value==candidate(root,metadata_source=value.get('metadata_preparation_SOURCE')),
        'PILOT_CLOSED_SOURCE_CONTRACT_BINDING')
    for p,h in {**value['source_pins'],**value['input_pins']}.items():
        require(sha256((Path(root)/p).read_bytes()).hexdigest()==h,'PILOT_STATIC_PIN:'+p)
    return S.digest(value)


def production_entry(*args,**kwargs):
    raise Closed('PILOT_UNADOPTED_NO_EXECUTABLE_RELEASE_NO_AUTHORIZATION')
