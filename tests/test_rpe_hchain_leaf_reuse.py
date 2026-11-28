"""Artificial accepted preparation receipts; never call a scientific provider."""
from contextlib import closing
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from rpe_hchain_preparation_reuse import reuse_prior_leaves
from rpe_hchain_reporting import intermediate_report
from run_rpe_hchain_preparation import preparation_request
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_q3_coordinator import digest


def fixture(tmp_path, *, stage='signal', changed_candidate=False):
    prior=tmp_path/'prior';run=tmp_path/'new';prior.mkdir();run.mkdir()
    for name in ('LAUNCH.json','EXIT.json'):
        (prior/name).write_text(json.dumps({'invocation_id':'ended'}))
    cell=dict(id='leaf',method='PR',R_outer=2)
    another=dict(id='other',method='PR',R_outer=2)
    plan=dict(owner='fixed-owner',binding={'system':'H8'},target=-1,compiler='fixed',
              scope={'system':'H8','price_paths':1},price_cohort='fixed-cohort',
              cases=[dict(rounds=[cell,another])],
              designs=[dict(anchors=[cell,another],holdout=None)])
    packet=dict(plan,cases=[dict(rounds=[cell])] if stage=='signal' else [],
                designs=[] if stage=='signal' else [dict(anchors=[cell],holdout=None)])
    settings=dict(plan={'sha256':'a'*64},provider_source={'commit':'b'*40,'files':{'source.py':'c'*64}},
                  assets={'kind':'directory','files':{'asset':'d'*64},'controller_path':'fixed'},
                  openblas_coretype='HASWELL')
    old=ArtifactCache(prior/'distributed/artifacts')
    cache=ArtifactCache(run/'distributed/artifacts',prior_roots=[old.root])
    def put(raw):
        checksum=hashlib.sha256(raw).hexdigest()
        old.put({checksum:base64.b64encode(raw).decode()})
        return checksum
    packet_sha=put(json.dumps(packet).encode())
    request=preparation_request(packet,packet_sha,stage,[['leaf',0]],
                source=settings['provider_source'],assets=settings['assets'],openblas_coretype='HASWELL')
    request['request_id']='job'
    req_sha=digest(request)
    prep=request['preparation']
    name=prep['outputs'][0];manifest={}
    value={'owner':plan['owner'],'candidate':dict(cell,id='wrong') if changed_candidate else cell,'data':'saved exact bytes'}
    if stage!='signal':
        side=prep['outputs'][1];manifest[side]=put(b'artificial events')
        value.update(index=0,cohort=plan['price_cohort'],compiler=plan['compiler'],
                     events={'sha256':manifest[side],'size_bytes':len(b'artificial events'),'file':Path(side).name})
    manifest[name]=put(json.dumps(value).encode())
    process={'returncode':0,'signal':None}
    receipt=dict(source=prep['source'],returncode=0,signal=None,
        runtime={'openblas_coretype':'HASWELL','packages':prep['runtime_requirements']['packages']},
        numerical_environment={'OPENBLAS_CORETYPE':'HASWELL','OPENBLAS_NUM_THREADS':'1',
                               'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    outcome=dict(status='PREPARATION_COMPLETE',request_id='job',request_sha256=req_sha,
                 stage=stage,source_manifest=prep['source'],artifacts=manifest)
    stage_dir=prior/'distributed/transfers/job/attempt';stage_dir.mkdir(parents=True)
    files={}
    for filename,obj in (('PROCESS_RETURN.json',process),('OUTCOME.json',outcome),
                         ('PREPARATION_RECEIPT.json',receipt)):
        raw=json.dumps(obj).encode();(stage_dir/filename).write_bytes(raw)
        files[filename]=base64.b64encode(raw).decode()
    contract={'evaluation':{'fixed_plan_sha256':settings['plan']['sha256']}}
    db=prior/'distributed/QUEUE.sqlite3'
    with closing(sqlite3.connect(db)) as con, con:
        con.execute('CREATE TABLE jobs(job_id,attempt_id,host_id,request,contract,request_sha256,completion_sha256,eligibility,state,adopted_epoch,method)')
        con.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)',('job','attempt','synthetic',
                    json.dumps(request),json.dumps(contract),req_sha,digest(files),'null','ACCEPTED',1,'PREPARATION:'+stage))
    executor=SimpleNamespace(prior_run_dirs=[prior],artifact_cache=cache)
    return executor,plan,settings,run,prior,name,manifest,old,db


@pytest.mark.parametrize('stage',['signal','compile'])
def test_exact_leaf_reuse_without_new_owner(tmp_path,stage):
    executor,plan,settings,run,prior,name,manifest,old,_=fixture(tmp_path,stage=stage)
    assert reuse_prior_leaves(executor,plan,settings,{stage:1},run)==1
    assert (run/name).samefile(old.path(manifest[name]))
    assert not (run/'distributed/QUEUE.sqlite3').exists()
    rows=[json.loads(line) for line in (run/'distributed/HCHAIN_PREPARATION_REUSE.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['job_id']=='job' and rows[0]['artifacts']==manifest
    assert reuse_prior_leaves(executor,plan,settings,{stage:1},run)==0
    for path,sha in manifest.items():
        assert (run/path).samefile(old.path(sha))


@pytest.mark.parametrize('change',['owner','plan','core','unpinned','unaccepted','unadopted','digest','active','missing','candidate'])
def test_changed_or_unresolved_predecessor_is_not_reused(tmp_path,change):
    executor,plan,settings,run,prior,name,manifest,old,db=fixture(tmp_path,stage='compile',changed_candidate=change=='candidate')
    if change=='owner':plan['owner']='different'
    elif change=='plan':settings['plan']['sha256']='e'*64
    elif change=='core':settings['openblas_coretype']='NEHALEM'
    elif change=='unpinned':settings.pop('openblas_coretype')
    elif change in ('unaccepted','unadopted'):
        with closing(sqlite3.connect(db)) as con,con:
            con.execute("UPDATE jobs SET state='UNKNOWN'" if change=='unaccepted' else 'UPDATE jobs SET adopted_epoch=NULL')
    elif change=='digest':(prior/'distributed/transfers/job/attempt/PROCESS_RETURN.json').write_text('{}')
    elif change=='active':(prior/'EXIT.json').unlink()
    elif change=='missing':old.path(manifest[next(n for n in manifest if n.endswith('.xz'))]).unlink()
    assert reuse_prior_leaves(executor,plan,settings,{'compile':1},run)==0
    assert not (run/name).exists()


def test_intermediate_report_omits_duplicate_exact_arrays_only():
    bound={'numerator':10**1000,'denominator':10**1100+7}
    full={'scope_owner':'fixed','held_tasks':{'x':'hold'},'methods':{'FULL_PF':{
        'best':{'MSE_upper':bound},'stop_reason':'WAITING_FOR_INPUTS',
        'by_configuration':[{'allocations':[{'MSE_upper':bound}]}],
        'evaluated':[{'MSE_upper':bound}]*30,'resume_records':[{'MSE_upper':bound}]*30}}}
    compact=intermediate_report(full)
    result=compact['methods']['FULL_PF']
    assert result['intermediate_evaluation_counts']=={'evaluated':30,'resume_records':30}
    assert result['best']==full['methods']['FULL_PF']['best']
    assert result['by_configuration']==full['methods']['FULL_PF']['by_configuration']
    assert compact['resume_records_exported'] is False and compact['held_tasks']==full['held_tasks']
    assert len(json.dumps(compact))<len(json.dumps(full))/10
    assert len(full['methods']['FULL_PF']['resume_records'])==30

def test_lossless_compressed_proof_and_collect_match_legacy(tmp_path,monkeypatch):
    import gzip
    import rpe_hchain_q3 as q
    proof=dict(request_id='fixture',source={'fixed':'source'},identity={'id':'fixed'},
               result=dict(method='Full PF',status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',
                           MSE_lower=0,MSE_upper=0,intermediate=[10**1000]*30))
    old=tmp_path/'plain.json';compressed=tmp_path/'HCHAIN_RESULT.json.gz'
    q.write(old,proof,compact=True)
    q.write_compressed_proof(compressed,proof)
    assert gzip.decompress(compressed.read_bytes())==old.read_bytes()
    assert compressed.stat().st_size<old.stat().st_size/10
    monkeypatch.setattr(q,'request_inputs',lambda request:('Full PF',[],[],1,0,proof['source'],proof['identity']))
    request={'request_id':'fixture','hchain_proof_storage':'GZIP_JSON_V1'}
    assert q.run(request,tmp_path,'collect')==0
    assert json.loads((tmp_path/'OUTCOME.json').read_text())==q.outcome(request,proof)
    assert not (tmp_path/'HCHAIN_RESULT.json').exists()
    assert q.run(request,tmp_path,'collect')==0

def test_existing_queue_is_not_treated_as_new_graph(tmp_path):
    executor,plan,settings,run,prior,name,manifest,old,db=fixture(tmp_path)
    from trottertracks.rpe_q3_store import JobStore
    executor.store=JobStore(run/'distributed/QUEUE.sqlite3')
    executor.store.reserve('FULL_PF','existing',{'route':{'kind':'artificial'}},1,True)
    assert reuse_prior_leaves(executor,plan,settings,{'signal':1},run)==0
    assert not (run/name).exists()


def test_partition_progress_cannot_masquerade_as_exact_resume_export():
    from trottertracks.rpe_pbs_offline import compact_block_report
    report=intermediate_report({'methods':{'FULL_PF':{'evaluated':[{'x':1}],'resume_records':[{'x':1}]}}})
    compact=compact_block_report(report)
    assert compact['resume_records_exported'] is False
    assert compact['report_phase']=='INTERMEDIATE'
