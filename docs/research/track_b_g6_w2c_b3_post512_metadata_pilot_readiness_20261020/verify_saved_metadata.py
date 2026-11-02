"""Limited old-512 metadata/identity read-only verification; reuses old science checks."""
from pathlib import Path
from hashlib import sha256
import json,sys,tarfile,sqlite3,subprocess,ast,os,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import guard as G
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S
BASE='22ef92cef31f98e3e31f27a209405f81f5c4a0e4';OLD=ROOT/'docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020'

def read(path):return json.loads(Path(path).read_bytes())
def h(path):return sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

if __name__=='__main__':
    inventory=read(OLD/'raw_inventory.json');source=read(OLD/'source_inventory.json');m=read(OLD/'approved_manifest.json');summary=read(OLD/'result_summary.json');saved=read(OLD/'saved_evidence_check.json')
    oldroot=Path(inventory['original_state_root']);oldwork=Path(m['worker_runtime']['runtime_root'])
    assert h(OLD/'raw_scientific.tar.gz')==inventory['archive']['sha256']
    rawchecks=[]
    for row in inventory['files']:
        assert h(oldroot/row['path'])==row['sha256'];rawchecks.append(dict(path=row['path'],sha256=row['sha256'],unchanged=True))
    # Fixed science checks are reused, not rerun/reaggregated. Inspect only mode
    # flags, source/manifest/pid joins and the already saved document counts.
    observations=0;source_bindings=set();live_workers=set()
    with tarfile.open(OLD/'raw_scientific.tar.gz','r:gz') as archive:
        report=json.load(archive.extractfile('run/output/final.json'));receipt=json.load(archive.extractfile('run/output/receipt.json'))
        delivery=json.load(archive.extractfile('run/output/delivery.json'));caller=json.load(archive.extractfile('run/output/caller.json'))
        assert report['artificial_only'] is True and delivery['artificial_only'] is False and caller['artificial_only'] is False
        for task in m['tasks']:
            for line in archive.extractfile('run/output/'+task['key']+'.jsonl').read().splitlines():
                payload=V.decode(json.loads(line)['payload']);e=payload['evidence']
                assert e['artificial_only'] is False and e['consumption']['artificial_only'] is False
                assert e['runtime_origin']=='PINNED_G7_F3'
                assert {n.rsplit('.',1)[-1] for n in e['guard']['core_loaded']}==set(G.CORE)
                assert e['connection']['manifest_binding']==S.digest(m)
                source_bindings.add(e['connection']['source_binding']);live_workers.add(e['guard']['pid']);observations+=1
        assert observations==512 and source_bindings=={source['source_release_candidate']}
    assert sha256((oldroot/'output/final.json').read_bytes()).hexdigest()==delivery['final_sha256']
    assert sha256((oldroot/'output/delivery.json').read_bytes()).hexdigest()==caller['delivery_sha256']
    assert sha256((oldroot/'output/receipt.json').read_bytes()).hexdigest()==report['receipt_sha256']
    dbpath=oldroot/'SIMPLE_RUN.sqlite3';before=h(dbpath)
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro&immutable=1',uri=True)
    claims=db.execute('select scope,claim,binding,status,spent,receipt from claims').fetchall();runs=db.execute('select run_id,scope,execution_state from runs').fetchall();db.close()
    assert h(dbpath)==before and len(claims)==1 and claims[0][3:5]==('STOPPED',1) and runs[0][2]=='COMPLETED'
    assert claims[0][:3]==(m['global_scope'],m['global_claim'],S.digest(m))
    assert summary['technical_completed']==saved['trials']==saved['full_documents']==report['states']['COMPLETED']==512
    assert summary['by_mode']==dict(S=256,P1=256) and saved['receipt_caller_join'] is True
    pins=[]
    for p,expected in {**source['source_pins'],**source['input_pins']}.items():
        assert h(oldwork/p)==expected and sha256(git('show',BASE+':'+p)).hexdigest()==expected
        pins.append(dict(path=p,sha256=expected))
    runtime=source['runtime'];assert h(runtime['executable'])==runtime['executable_sha256']
    for row in runtime['libraries'].values():assert h(row['path'])==row['sha256']
    core=[]
    for name in sorted(G.CORE):
        path='src/trottertracks/algorithm_codesign/'+name+'.py';expected=sha256(git('show',source['resolver_binding']['science_reference']+':'+path)).hexdigest()
        assert h(ROOT/path)==h(oldwork/path)==expected;core.append(dict(path=path,sha256=expected))
    roots=[oldwork,ROOT.parent/'g6-w2c-b3-final-live-connection-simple-run-control-20261020',ROOT.parent/'g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020',ROOT.parent/'g6-w2c-b3-real-adapter-independent-server-20261020-r2']
    states=[]
    for root in roots:states.append(dict(path=str(root),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)))
    finalrow=next(x for x in rawchecks if x['path']=='output/final.json')
    correction=dict(schema='b3_post512_metadata_correction_sidecar_v1',target_public_commit=BASE,
        target_measured_SOURCE=source['measured_SOURCE'],target_raw_archive=inventory['archive'],
        target_final=dict(path='output/final.json',sha256=finalrow['sha256']),
        corrections=[dict(field='output/final.json:artificial_only',original=True,correct_display=False,
            meaning='USER_APPROVED_LIVE',evidence='512 per-trial/bit flags False, Core6 loaded in16 workers, exact approval/source binding, SPENT intents and persistent COMPLETED claim'),
            dict(field='SimpleStore snapshot:real_authority',original=False,correct_display=False,
                meaning='external_authority_used=False; user_approved_live_execution=True; approval_mode=USER_SIMPLE_ONE_SHOT. This legacy field never measured USER authorization.'),
            dict(field='live_API_compatibility',original='NOT_ESTABLISHED',correct_display='NOT_ESTABLISHED',
                additional_observation='FIXED_MANIFEST_512_TRIALS_COMPLETED_AND_SAVED_CHECKED',general_guarantee='NOT_ESTABLISHED',
                unverified='outside adopted input/precision/paths; real F01 failure did not occur')],
        applies_only_to=dict(run_id=m['run_id'],manifest_sha256=h(OLD/'approved_manifest.json'),global_claim=m['global_claim']),
        raw_modified=False,science_result_modified=False,new_scientific_result=False,new_scientific_operations=0)
    save('metadata_correction_sidecar.json',correction)
    save('saved_512_evidence_verification.json',dict(status='PASS_LIMITED_METADATA_IDENTITY_CHECK',fixed_public_HEAD=BASE,
        scope='metadata, bytes, source/input/runtime, joins and persistent state only; reuse previous full numerical/document/Codec/prefix checks',
        raw_files=rawchecks,original_archive_sha256=inventory['archive']['sha256'],source_input_pins=pins,
        runtime_identity_unchanged=True,science_core_pins=core,metadata_trials_checked=observations,worker_pids=sorted(live_workers),
        actual_claims=claims,actual_runs=runs,DB_sha256_unchanged=before,receipt_delivery_caller_join=True,
        reused_checks=[dict(path=str((OLD/n).relative_to(ROOT)),sha256=h(OLD/n)) for n in ['saved_evidence_check.json','result_summary.json','source_inventory.json','approved_manifest.json','raw_inventory.json']],
        reused_results=dict(completed=512,by_mode=summary['by_mode'],coverage=summary['coverage_all_cells'],bit_draws=summary['bit_draw_requests'],
            event_invariants='previous fixed check PASS; not recomputed',Exact_Codec='previous fixed check PASS; not rerun',AB_SP1_prefix='previous fixed check PASS; not rerun'),
        metadata_only_issues=True,new_nonmetadata_inconsistency_found=False,new_scientific_analysis=False,scientific_operations=0,
        protected_worktrees=states,guard=G.observation(),separate_reviewer=False))
    instruction=Path('/home/AbeHiromu/.codex/attachments/18a14bda-1481-4e21-a218-27df9d71632c/貼り付けたテキスト.txt').read_bytes()
    save('user_instruction.json',dict(purpose='TECHNICAL_METADATA_AND_CLOSED_PILOT_PREPARATION_ONLY',instruction_text=instruction.decode(),sha256=sha256(instruction).hexdigest(),science_execution_approval=False))
    resource=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg(),meminfo=Path('/proc/meminfo').read_text(),pressure={name:(Path('/proc/pressure')/name).read_text() for name in ('cpu','memory','io')},observed_ns=time.time_ns())
    save('resource_observation.json',resource)
    assert not G.observation()['core_loaded'] and not G.ATTEMPTS
    print(json.dumps(dict(raw_files=len(rawchecks),metadata_trials=observations,source_input_pins=len(pins),runtime_libraries=len(runtime['libraries']),science_imports=0,sampling=0),indent=2))
