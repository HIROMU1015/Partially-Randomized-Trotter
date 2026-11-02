"""Pinned bundle identities, independently derived from the published data contract.

Reads bytes only. No producer, scheduler, Admission or scientific imports.
Historical runtime hashes are declarations, not proof that this host has those bytes.
"""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
import sys
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_v3_offline_identity as I

PUBLIC='d24cccbecdf0928fe977080dd7b28c591df0089e'
MEASURED='e099dcc461247e81eff981867dc28c3f3b16df5e'
DOC='docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/'
CLOSED=dict(content_adopted=False,run_ready=False,science_release=None,scientific_adoption=None,
            contract_grants_execution=False,mandatory_STOP=True,artificial_only=True,retry=0,resume=False)
SLOTS=4*1024*1024+8192

def read(path):return V.plain(Path(path).read_bytes())
def same(a,b):return I.strict_equal(a,b)
def equal(a,b,code):V.require(same(a,b),code)
def hashed(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def current_python():
    import fractions,multiprocessing,multiprocessing.connection,sqlite3,_sqlite3,_json,_hashlib
    libs={}
    for mod in (json,fractions,multiprocessing,multiprocessing.connection,sqlite3,_sqlite3,_json,_hashlib):
        p=Path(mod.__file__).resolve();libs[mod.__name__]=dict(path=str(p),sha256=V.file_digest(p))
    exe=Path(sys.executable).resolve()
    return dict(executable=str(exe),executable_sha256=V.file_digest(exe),version=sys.version,cache_tag=sys.implementation.cache_tag,libraries=libs)

def plan(n):
    V.require(type(n)is int and 16<=n<=256,'PROFILE_N')
    tasks=[];offset=0
    for j,mode in enumerate(('S','P1')):
        local,total=G.plan(n)
        tasks.extend(dict(t,key=mode+'__'+t['key'],condition=mode,phase_index=t['phase_index']+3*j,offset=t['offset']+offset)for t in local)
        offset+=total
    return tasks,offset

class Reference:
    def __init__(self,tree,trusted_candidate=None,*,allow_runtime_projection=False,verify_local_runtime=True):
        self.tree=Path(tree).resolve();inv=read(self.tree/DOC/'source_inventory.json')
        equal(inv['measured_runtime_SOURCE_commit'],MEASURED,'FIXED_MEASURED_SOURCE')
        self.template=read(self.tree/DOC/'technical_contract_candidate.json')
        self.pins=inv['measured_runtime_pins']
        equal(self.template['source_pins'],self.pins,'FIXED_SOURCE_PIN_SET')
        equal(self.template['scientific_reference'],read(self.tree/DOC/'fixed_scientific_reference.json'),'FIXED_SCIENTIFIC_REFERENCE')
        self.schema=read(self.tree/DOC/'bundle.schema.json')
        self.trusted=deepcopy(self.template if trusted_candidate is None else trusted_candidate)
        projected=deepcopy(self.trusted)
        if allow_runtime_projection:
            for k in ('runtime_root','python'):projected[k]=self.template[k]
        equal(projected,self.template,'UNDECLARED_CANDIDATE_DELTA')
        self.projected=allow_runtime_projection
        self.local_bytes=self.projected and verify_local_runtime
        if self.local_bytes:equal(self.trusted['python'],current_python(),'LOCAL_RUNTIME_ACTUAL_BYTES')
        self.verify_bytes()

    def verify_bytes(self):
        for p,h in self.pins.items():equal(V.file_digest(V.safe_path(self.tree,p)),h,'SOURCE_BYTES:'+p)
        for p,h in self.template['scientific_reference']['input_pins'].items():
            equal(V.file_digest(V.safe_path(self.tree,p)),h,'INPUT_BYTES:'+p)
        if self.local_bytes:
            runtime=Path(self.trusted['runtime_root']).resolve()
            for p,h in self.pins.items():equal(V.file_digest(V.safe_path(runtime,p)),h,'LOCAL_RUNTIME_SOURCE_BYTES:'+p)
            for p,h in self.template['scientific_reference']['input_pins'].items():equal(V.file_digest(V.safe_path(runtime,p)),h,'LOCAL_RUNTIME_INPUT_BYTES:'+p)

    def identity(self,c,m):
        self.verify_bytes();equal(c,self.trusted,'TRUSTED_CANDIDATE')
        equal(c['schema'],'b3_storage_v2_global_bundle_candidate_v1','CANDIDATE_SCHEMA')
        equal(m['schema'],'b3_bundle_artificial_manifest_v1','MANIFEST_SCHEMA')
        for obj in (c,m):
            for k,v in CLOSED.items():equal(obj[k],v,'CLOSED_FLAG:'+k)
        equal(c['status'],'UNADOPTED_EXECUTION_CLOSED','CLOSED_STATUS')
        equal(c['fixed_order'],['S','P1'],'FIXED_ORDER')
        equal(c['storage_protocol'],'GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1','STORAGE_PROTOCOL')
        equal(c['production_gate'],'NO_IMPLEMENTED_SCIENCE_DISPATCH','PRODUCTION_GATE')
        equal(c['source_release_candidate'],hashed(c['source_pins']),'SOURCE_RELEASE_DIGEST')
        equal(m['source_release_candidate'],c['source_release_candidate'],'MANIFEST_SOURCE_DIGEST')
        equal(m['candidate_binding'],hashed(c),'CANDIDATE_DIGEST')
        py=c['python'];V.require(Path(c['runtime_root']).is_absolute(),'RUNTIME_ROOT')
        V.require(V.is_hash(py['executable_sha256'])and py['version']and py['cache_tag'],'RUNTIME_IDENTITY')
        equal(set(py['libraries']),{'json','fractions','multiprocessing','multiprocessing.connection','sqlite3','_sqlite3','_json','_hashlib'},'RUNTIME_LIBRARY_SET')
        for row in py['libraries'].values():V.require(V.is_hash(row['sha256'])and Path(row['path']).is_absolute(),'RUNTIME_LIBRARY_IDENTITY')
        proposal=c['proposal']
        equal(proposal,self.template['proposal'],'FIXED_PROPOSAL')
        V.require(16*16+2*16*256==8448 and 2*8448==proposal['total_calls']==16896,'STATIC_CALL_ARITHMETIC')
        tasks,total=plan(m['profile_n'])
        equal(m['tasks'],tasks,'GLOBAL_TASKS_OFFSETS');equal(m['planned'],total,'GLOBAL_PLANNED')
        equal(m['static_scientific_candidate_calls'],16896,'STATIC_16896')
        V.require(m['modes']in (['S'],['P1'],['S','P1']),'MODE_PROJECTION')
        V.require(type(m['call_limit'])is int and 0<=m['call_limit']<=total,'GLOBAL_CALL_CAP')
        for ch in ('result','log'):
            n=m['byte_limits'][ch];V.require(type(n)is int and 0<n<=proposal[ch+'_bytes'],'GLOBAL_BYTE_LIMIT:'+ch)
        for k,v in dict(seed_namespace='ARTIFICIAL_R1_PARALLEL',codec='fused',observer='cached_all_call_return',workers=8,start_method='spawn').items():equal(m[k],v,'COMPARISON_BINDING:'+k)
        V.require(type(m['global_scope'])is str and m['global_scope'].startswith('ARTIFICIAL_ONLY/'),'ARTIFICIAL_SCOPE')
        equal(m['global_claim'],hashed(['ARTIFICIAL_ONLY',m['global_scope'],hashed(c)]),'GLOBAL_CLAIM')
        equal(m['run_id'],hashed(['RUN',m['global_scope'],hashed(c)]),'GLOBAL_RUN_ID')
        equal(m['scientific_input'],hashed(c['scientific_reference']['scientific_reference']['input']),'SCIENTIFIC_INPUT')
        required=set(CLOSED)|{'schema','candidate_binding','source_release_candidate','global_scope','global_claim','run_id','scientific_input','seed_namespace','profile_n','modes','tasks','planned','static_scientific_candidate_calls','call_limit','byte_limits','condition_faults','sync_failure','parent_fault','codec','observer','workers','start_method'}
        equal(set(m),required,'MANIFEST_FIELDS')
        return dict(tasks=tasks,planned=total,candidate_digest=hashed(c),manifest_digest=hashed(m),run_id=m['run_id'],
                    historical_runtime_bytes_proven=False,local_runtime_bytes_proven=self.local_bytes,runtime_projection=self.projected,real_authority_verified=False)
