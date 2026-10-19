"""Pinned data bindings; historical runtime hashes remain declarations."""
import hashlib,json,sqlite3
from pathlib import Path
from .. import g6_w2c_b3_offline_result as V
from ..g6_w2c_b3_v3_offline_identity import strict_equal

SEED='ARTIFICIAL_API_BINDING_V1'
MEASURED='78c571005fe766c61b32dc867e006531099ff608'
CLOSED=dict(run_ready=False,content_adopted=False,contract_grants_execution=False,science_release=None,scientific_adoption=None,mandatory_STOP=True,artificial_only=True,retry=0,resume=False)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def equal(a,b,why):V.require(strict_equal(a,b),why)
def read(p):return V.plain(Path(p).read_bytes())
def files(c,root):
    for p,h in c['source_pins'].items():equal(V.file_digest(V.safe_path(root,p)),h,'SOURCE_BYTES:'+p)
    for p,h in c['scientific_reference']['input_pins'].items():equal(V.file_digest(V.safe_path(root,p)),h,'INPUT_BYTES:'+p)
def candidate(c,trusted,root):
    equal(c,trusted,'TRUSTED_CANDIDATE');equal(c['schema'],'b3_closed_api_binding_candidate_v1','CANDIDATE_SCHEMA')
    for k,v in CLOSED.items():equal(c[k],v,'CLOSED:'+k)
    equal(c['production_dispatch'],False,'NO_PRODUCTION_DISPATCH');equal(c['artificial_seed_namespace'],SEED,'ARTIFICIAL_SEED')
    equal(c['source_release_candidate'],digest(c['source_pins']),'SOURCE_RELEASE_HASH')
    equal(c['scientific_reference_sha256'],digest(c['scientific_reference']),'INPUT_REF_HASH')
    files(c,root)
    return dict(source_pins=len(c['source_pins']),input_pins=len(c['scientific_reference']['input_pins']),historical_runtime_bytes_proven=False,actual_runtime='SEPARATE_LOCAL_ENVIRONMENT')
def expected_binding(c,logical):
    logical=Path(logical);scope='ARTIFICIAL_ONLY/API_PORT/'+logical.name
    return dict(schema='b3_FAKE_API_authority_binding_v1',formal_source_release=None,
        source_release_candidate=c['source_release_candidate'],scientific_input_hash=c['scientific_reference_sha256'],
        runtime_hash=digest(c['python']),contract_hash=digest(c),global_scope=scope,
        global_claim=digest(['FAKE_API_CLAIM',scope,digest(c)]),call_budget=16896,result_budget=1<<36,log_budget=1<<28,
        seed_namespace=SEED,retry=0,resume=False,worker_policy='fresh spawn, owned workers; S sequential / P1 fixed candidate eight',
        storage_destination=str(logical),monitoring_STOP_hash=digest(c['monitoring']),artificial_only=True,real_grant=None,real_authority=False)
def authority(root,c,logical):
    b=read(root/'authority_binding.json');equal(b,expected_binding(c,logical),'AUTHORITY_BINDING')
    with sqlite3.connect((root/'FAKE_ONLY.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        rows=db.execute('SELECT scope,claim,binding,status,spent FROM claims').fetchall();history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
    equal(rows,[(b['global_scope'],b['global_claim'],digest(b),'STOPPED',1)],'CLAIM_SPENT_STOP_SNAPSHOT')
    equal(history,[(1,b['global_scope'],'CLAIM_COMMITTED',digest(b)),(2,b['global_scope'],'STOP_COMMITTED',digest(b))],'CLAIM_SAVED_HISTORY')
    equal(read(root/'fake_snapshot.json'),dict(schema='b3_FAKE_ONLY_snapshot_v1',claim=list(rows[0]),history=[list(x) for x in history],real_authority=False),'SNAPSHOT_BINDING')
    return dict(snapshot_checked=True,complete_atomic_history_proven=False,real_authority=False)
