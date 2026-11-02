"""Fixed pilot preparation identities. No producer/science/authority imports."""
from pathlib import Path
from copy import deepcopy
import sqlite3
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_v3_offline_identity as I
from . import g6_w2c_b3_parallel_verifier_core as G
PUBLIC='76ca3c4089b201d55da0fede08e1f77733ac38a5'
CONTENT='f3ba0cd09d3c09c5d04a6b48c3d6c9da63fff867'
MEASURED='b9cabcfc6cf304ab65fc8a540c33ce2173f6cbaa'
PREFIX='docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/'
CANDIDATE_SOURCE='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_candidate.py'
CHECKER_SOURCE='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_offline.py'


class Reference:
    def __init__(self,tree,legacy):
        self.tree=Path(tree).resolve();self.legacy=I.Reference(legacy)
        inv=V.plain((self.tree/PREFIX/'source_inventory.json').read_text())
        self.pins={r['path']:r['sha256'] for r in inv['files']+inv['protected']}
        self.template=V.plain((self.tree/PREFIX/'pilot_contract_candidate.json').read_text())
        self.schema=V.plain((self.tree/PREFIX/'pilot.schema.json').read_text())
        V.require(inv['measured_source_commit']==MEASURED,'PILOT_MEASURED_SOURCE')
        for r in inv['files']+inv['protected']:
            V.require(V.file_digest(V.safe_path(self.tree,r['path']))==r['sha256'],'REFERENCE_SOURCE:'+r['path'])

    def candidate(self,c):
        I.validate(c,self.schema,{})
        # Published inputs are fixed, not an authority to change N/seed/modes.
        V.require(I.strict_equal(c,self.template),'FIXED_PILOT_CANDIDATE_IDENTITY')
        for k,v in dict(artificial_only=True,contract_grants_execution=False,content_adopted=False,
                        run_ready=False,science_release=None,scientific_adoption=None,retry=0,resume=False,
                        mandatory_STOP=True,claim_id=None).items():
            V.require(I.strict_equal(c.get(k),v),'PILOT_CLOSED_FLAG:'+k)
        p=c['proposal'];n=p['profile_n']
        V.require(type(n) is int and n in (64,256,1024) and p['modes']==['S','P1'] and
                  p['correctness_calls']==512 and p['online_calls']==64*n and p['total_calls']==512+64*n,
                  'PILOT_SCOPE_COUNTS')
        V.require(p['codec']=='fused' and p['observer']=='cached_all_call_return' and p['retry']==0 and
                  p['resume'] is False and p['science_authorization'] is None and
                  p['seed_namespace']=='PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1','PILOT_COMPARISON_SEED')
        for path,h in {**c['source_pins'],**c['input_pins']}.items():
            V.require(V.file_digest(V.safe_path(self.tree,path))==h,'PILOT_SOURCE_INPUT_BYTES:'+path)
        py=c['python'];V.require(V.is_hash(py['executable_sha256']) and py['version'] and py['cache_tag'],'PILOT_RUNTIME_SHAPE')
        V.require(all(V.is_hash(x['sha256']) for x in py['libraries'].values()),'PILOT_LIBRARY_PINS')
        return V.identity(c)

    def inner(self,c,inner):
        digest=self.candidate(c)
        V.require(inner['pilot_candidate_binding']==digest,'PILOT_INNER_BINDING')
        binding=self.legacy.contract(inner)
        old=V.plain((self.tree/c['input_pins_reference']).read_text()) if 'input_pins_reference' in c else V.plain((self.tree/I.PREFIX/'measurements/benchmark/light-PF-0-contract.json').read_text())
        keys=('input','policy','original_source_commits','saved_evidence','saved_rows','cells','decisions','acceptance','call_budget','one_shot')
        V.require(I.strict_equal(c['scientific_reference'],{k:old['candidate'][k] for k in keys}),'FROZEN_SCIENTIFIC_REFERENCE')
        return dict(pilot_digest=digest,inner_digest=V.identity(inner),inner_binding=binding,
                    historical_python_bytes_verified=False,real_authorization_verified=False)


def admission_snapshot(path,inner):
    """Only explicitly supplied retained FAKE store, immutable read-only URI."""
    path=Path(path).resolve();V.require(path.name=='FAKE_ONLY_claims.sqlite3','FAKE_STORE_ONLY')
    before=V.file_digest(path)
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        V.require(db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'FAKE_STORE_INTEGRITY')
        meta=db.execute('SELECT store_id,version FROM metadata').fetchall()
        rows=db.execute('SELECT scope,claim_id,contract_digest,binding,nonce,status,owner_pid,stop_record FROM claims').fetchall()
        V.require(len(meta)==len(rows)==1,'ONE_FAKE_STORE_CLAIM')
        V.require(type(meta[0][1]) is int and meta[0][1]==1,'FAKE_BINDING:ledger_store_version')
        scope,claim,digest,b,nonce,status,pid,stop=rows[0];b=V.plain(b)
        V.require(scope==inner['scope'] and scope.startswith('ARTIFICIAL_') and claim==inner['claim_id'] and
                  digest==V.identity(inner),'FAKE_SCOPE_CLAIM_CONTRACT')
        expected=dict(formal_digest=digest,driver_contract_digest=V.identity(inner['candidate']),
                      input_pins_digest=V.identity(inner['input_pins']),source_pins_digest=V.identity(inner['source_pins']),
                      scope=scope,claim_id=claim,ledger_store_id=meta[0][0])
        for k,v in expected.items():V.require(b[k]==v,'FAKE_BINDING:'+k)
        V.require(status in ('RESERVED','ADMITTED','STOPPED') and type(pid) is int and nonce,'FAKE_SPENT_STATUS')
        if status=='STOPPED':
            s=V.plain(stop);V.require(s['schema']=='b3_admission_stop_v1' and s['mandatory_STOP'] is True and
                                    type(s['retry']) is int and s['retry']==0 and s['resume'] is False,'FAKE_STOP_FLAGS')
        V.require(V.file_digest(path)==before,'FAKE_STORE_MUTATED')
        return dict(status='VERIFIED_FAKE_SPENT_SNAPSHOT',store_id=meta[0][0],scope=scope,claim_id=claim,
                    claim_status=status,sha256=before,snapshot_proves_atomic_history=False,
                    original_private_approval_bytes_available=False,real_authorization_verified=False)
    finally:db.close()


def scope_plans():
    """Static arithmetic/identity contract, no bitstream or sampling."""
    rows=[]
    for n in (64,256,1024):
        tasks,total=G.plan(n)
        rows.append(dict(profile_n=n,cells=16,conditions=['S','P1'],correctness_per_condition=256,
                         baseline_per_condition=16*n,instrumented_per_condition=16*n,
                         per_condition=total,total=2*total,global_identity='condition / bundle digest / local trial id',
                         production_two_condition_bridge='NOT_IMPLEMENTED',scientifically_adopted=False))
    return rows
