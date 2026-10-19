"""One isolated artificial case; raw and fake ledger retained, never science."""
from pathlib import Path
import json,multiprocessing as mp,os,shutil,sys,tarfile,tempfile,time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_storage_artificial_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_storage_verify_v2 as V


def owned_parent(root,c,inner,store,anchor):
    l=F.R.A.A.ClaimLedger(Path(root)/'ledger',store)
    F.invoke(Path(root),c,inner,l,F.R.A.A.PinnedApproval(anchor))


class Writer:
    def __init__(self,wrapped,partial):self.wrapped,self.partial,self.calls=wrapped,partial,0
    def write(self,data):
        self.calls+=1
        if self.partial and self.calls==1:return self.wrapped.write(data[:13])
        raise OSError('INDEPENDENT_ARTIFICIAL_TRACE_RESERVE_WRITE_FAILURE')
    def __getattr__(self,key):return getattr(self.wrapped,key)


def main():
    spec=json.loads(sys.argv[1]);dest=Path(sys.argv[2]).resolve();dest.mkdir(parents=True,exist_ok=False)
    settings=F.F.P.Settings(profile_n=16,codec='fused',workload='light',mode=spec['mode'],**spec.get('settings',{}))
    fault=F.Fault(spec.get('point',''),spec.get('match',{}));writer_events=[]
    admission_fault=F.Fault(spec.get('admission_point',''))
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as t:
        root=Path(t);c,i,l,a=F.fixture(root,settings);r=None;error=None
        original=F.R.Trace.emit
        def emit(self,op,**fields):
            if spec.get('writer')and not writer_events and op==spec.get('writer_op','RESERVE'):
                writer_events.append(dict(op=op,seq=self.seq,trial_id=fields.get('trial_id')))
                self.file=Writer(self.file,spec['writer']=='partial')
            return original(self,op,**fields)
        if spec.get('parent_crash'):
            process=mp.get_context('spawn').Process(target=owned_parent,args=(str(root),c,i,l.store_id,a.anchor))
            process.start();process.join(30);assert not process.is_alive()and process.exitcode==86
            # Owned workers observe loss of their exact parent and exit naturally.
            pids=[];journal=root/'artificial-output/control.jsonl'
            if journal.exists():
                for line in journal.read_text().splitlines():
                    row=json.loads(line)
                    if row.get('event')=='WORKER_START':pids.append(row['pid'])
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                alive=[]
                for pid in pids:
                    try:
                        state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                        if state!='Z':alive.append(pid)
                    except FileNotFoundError:pass
                if not alive:break
                time.sleep(.05)
            assert not alive if pids else True
            error='OWNED_PARENT_EXIT_86';verification=dict(status='INCOMPLETE_EVIDENCE',host_cpu_ns=None,worker_cpu_ns=None)
        else:
            try:
                with patch.object(F.R.Trace,'emit',emit):r=F.invoke(root,c,i,l,a,fault,admission_fault)
            except BaseException as exc:error=repr(exc)
            # Preserve source-produced bytes before a checker assertion can
            # fail. A development rejection must not erase the triggering raw.
            for name,obj in(('pilot_contract.json',c),('inner_contract.json',i),
                ('observations.json',dict(spec=spec,report=r,error=error,verification_pending=True))):
                (dest/name).write_bytes(F.F.P.safe_json(obj)+b'\n')
            shutil.copyfile(l.path,dest/'FAKE_ONLY_claims.sqlite3')
            if(root/'artificial-output').exists():
                with tarfile.open(dest/'artificial_raw.tar.gz','w:gz')as tf:tf.add(root/'artificial-output',arcname='artificial-output')
            if r is not None:
                verification=V.verify(root/'artificial-output',c,i,r)
                assert r['complete']==(spec['name']=='normal'),(spec,r.get('primary_failure'),r['evidence_v2'])
                assert r['reserved']==768-r['states']['NOT_STARTED']==sum(r['evidence_v2']['next_indices'].values())
                assert not r['evidence_v2']['current_tickets']
                if spec.get('writer')=='fail':assert r['reserved']>=1 and r['started_observed']==0
                if spec.get('point')=='trace.write'and spec.get('match',{}).get('op')=='ENTER':
                    assert r['started_observed']==0
                if spec.get('point')=='trace.write'and spec.get('match',{}).get('op')=='COMPLETE':
                    assert r['states']['COMPLETED']==0 and r['states']['TERMINATED_WITH_UNKNOWN_RESULT']>=1
            else:
                assert error and spec.get('early'),(spec,error)
                verification=dict(status='INCOMPLETE_EVIDENCE',report_available=False,error=error)
        status=l.row(i['scope'])[4]
        assert status in(('ADMITTED','RESERVED')if spec.get('parent_crash')else(('ADMITTED','STOPPED')if spec.get('admission_point')else('STOPPED',)))
        # Same-store reuse is rejected even after an IO failure; never refund.
        try:F.R.ArtificialAdmission(c,i,root,l,a).admit_artificial(F.R.D.identity(i['candidate']))
        except Exception as exc:reuse_error=type(exc).__name__
        else:raise AssertionError('SPENT_SCOPE_REUSED')
        assert spec['name']=='normal'or fault.events or writer_events or admission_fault.events or spec.get('parent_crash')or spec.get('settings'),spec
        observations=dict(spec=spec,report=r,error=error,verification=verification,injected=fault.events+writer_events+admission_fault.events,
            fake_claim_status=status,reuse_denied=reuse_error,scientific_calls=0,real_authorizations_grants_claims=0)
        for name,obj in(('pilot_contract.json',c),('inner_contract.json',i),('observations.json',observations)):
            (dest/name).write_bytes(F.F.P.safe_json(obj)+b'\n')
        shutil.copyfile(l.path,dest/'FAKE_ONLY_claims.sqlite3')
        if (root/'artificial-output').exists():
            with tarfile.open(dest/'artificial_raw.tar.gz','w:gz')as tf:tf.add(root/'artificial-output',arcname='artificial-output')
        forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1]in F.F.P.FORBIDDEN];assert not forbidden
        result=dict(case=dest.name,status=verification['status'],report_available=r is not None,
            reserved=r['reserved']if r else None,started=r['started_observed']if r else None,
            completed=r['states']['COMPLETED']if r else None,unknown=r['states']['TERMINATED_WITH_UNKNOWN_RESULT']if r else None,
            trace_complete=r['evidence_v2']['trace_complete']if r else False,
            statefile_complete=r['evidence_v2']['statefile_complete']if r else False,
            claim=status,reuse_denied=True,forbidden_science_imports=forbidden,scientific_calls=0,successful=True)
        (dest/'case_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
