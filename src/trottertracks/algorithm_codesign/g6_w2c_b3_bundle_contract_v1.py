"""Unadopted bundle contract data. No producer or scientific imports."""
from hashlib import sha256
from pathlib import Path
import json,sys

DOC='docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/'
SOURCE='src/trottertracks/algorithm_codesign/'
SCHEMA='b3_storage_v2_global_bundle_candidate_v1'
FINAL_SLOT=4*1024*1024
DELIVERY_SLOT=4096
OBSERVATION_SLOT=4096
CLOSED=dict(content_adopted=False,run_ready=False,science_release=None,scientific_adoption=None,
            contract_grants_execution=False,mandatory_STOP=True,artificial_only=True,retry=0,resume=False)
NEW_FILES=[SOURCE+'g6_w2c_b3_bundle_'+n+'_v1.py' for n in ('contract','bridge','fixture','store','offline','trace')]+[DOC+'bundle.schema.json',DOC+'fixed_scientific_reference.json',DOC+'run_case.py']

def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()

def digest(value):return sha256(encoded(value)).hexdigest()

def runtime():
    import json as j,fractions,multiprocessing,multiprocessing.connection,sqlite3,_sqlite3,_json,_hashlib
    libs={}
    for m in (j,fractions,multiprocessing,multiprocessing.connection,sqlite3,_sqlite3,_json,_hashlib):
        p=Path(m.__file__).resolve();libs[m.__name__]=dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest())
    p=Path(sys.executable).resolve()
    return dict(executable=str(p),executable_sha256=sha256(p.read_bytes()).hexdigest(),version=sys.version,cache_tag=sys.implementation.cache_tag,libraries=libs)

def plan(n=256):
    tasks=[];offset=0
    arms=('ordinary','partial_return_tail','closed_P3_tail','full_return')
    for mode_index,mode in enumerate(('S','P1')):
        for phase_index,(phase,pass_name) in enumerate((('correctness','baseline'),('profile','baseline'),('profile','instrumented'))):
            for i in range(8):
                arm,layer=arms[i//2],('A','B')[i%2];count=16 if phase_index==0 else n
                tasks.append(dict(key=f'{mode}__{phase}_{pass_name}_{arm}_{layer}',condition=mode,phase_index=3*mode_index+phase_index,phase=phase,pass_name=pass_name,arm=arm,layer=layer,pair_index=i,n=count,offset=offset));offset+=2*count
    return tasks,offset

def candidate(root):
    root=Path(root).resolve();ref=json.loads((root/DOC/'fixed_scientific_reference.json').read_bytes())
    fixed=list(ref['source_pins'])
    # Pin the actual technical import closure as well as the scientific references.
    fixed+=['src/trottertracks/__init__.py',SOURCE+'__init__.py']
    fixed += [str(p.relative_to(root)) for p in sorted((root/SOURCE).glob('g6_w2c_b3_*.py')) if 'bundle_' not in p.name]
    pins={p:sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(fixed+NEW_FILES))}
    for p,h in ref['source_pins'].items():
        if pins[p]!=h:raise ValueError('FIXED_SCIENTIFIC_SOURCE:'+p)
    return dict(schema=SCHEMA,status='UNADOPTED_EXECUTION_CLOSED',**CLOSED,runtime_root=str(root),python=runtime(),source_pins=pins,
        scientific_reference_commit='76ca3c4089b201d55da0fede08e1f77733ac38a5',scientific_reference=ref,
        source_release_candidate=digest(pins),fixed_order=['S','P1'],
        proposal=dict(profile_n=256,cells=16,correctness_per_cell=16,baseline_per_cell=256,instrumented_per_cell=256,
            per_condition=dict(correctness=256,baseline=4096,instrumented=4096,total=8448),total_calls=16896,
            result_bytes=1<<36,log_bytes=1<<28,repetition=1,codec='fused',observer='cached_all_call_return',workers=8,start_method='spawn',
            science_seed_namespace='PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1',seed_domain='namespace/arm/axis',mode_is_seed_domain=False),
        storage_protocol='GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1',production_gate='NO_IMPLEMENTED_SCIENCE_DISPATCH',
        finalization=dict(final_slot=FINAL_SLOT,delivery_slot=DELIVERY_SLOT,observation_slot=OBSERVATION_SLOT,receipt_after_storage_close=True,marker_after_receipt_close=True,
            final_self_close_requires_external_delivery=True,delivery_close_requires_caller_observation=True,fsync_durability_claim=False))

def manifest(c,scope,*,profile_n=16,modes=('S','P1'),result_bytes=None,log_bytes=None,call_limit=None,condition_faults=None,sync_failure='',parent_fault=''):
    if profile_n<16 or profile_n>256 or tuple(modes) not in (('S',),('P1',),('S','P1')):raise ValueError('ARTIFICIAL_PLAN')
    tasks,total=plan(profile_n)
    # Projection tests retain global offset/identity; they can never be bundle success.
    return dict(schema='b3_bundle_artificial_manifest_v1',**CLOSED,candidate_binding=digest(c),source_release_candidate=c['source_release_candidate'],
        global_scope=scope,global_claim=digest(['ARTIFICIAL_ONLY',scope,digest(c)]),run_id=digest(['RUN',scope,digest(c)]),
        scientific_input=digest(c['scientific_reference']['scientific_reference']['input']),seed_namespace='ARTIFICIAL_R1_PARALLEL',
        profile_n=profile_n,modes=list(modes),tasks=tasks,planned=total,static_scientific_candidate_calls=16896,
        call_limit=total if call_limit is None else call_limit,
        byte_limits=dict(result=c['proposal']['result_bytes'] if result_bytes is None else result_bytes,log=c['proposal']['log_bytes'] if log_bytes is None else log_bytes),
        condition_faults=condition_faults or {},sync_failure=sync_failure,parent_fault=parent_fault,
        codec='fused',observer='cached_all_call_return',workers=8,start_method='spawn')
