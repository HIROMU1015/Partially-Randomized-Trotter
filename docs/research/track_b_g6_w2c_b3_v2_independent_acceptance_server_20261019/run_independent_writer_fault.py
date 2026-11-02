"""External writer fault: snapshot RAM exactly when actual RESERVE write raises.

Separate producer subprocess only; the independent consumer is never imported.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch
import json,os,shutil,sys,tarfile,tempfile
HERE=Path(__file__).resolve().parent;RUNTIME=HERE/'injection_runtime'
sys.dont_write_bytecode=True;sys.path[:0]=[str(RUNTIME/'src'),str(RUNTIME/'tests/tracks/algorithm_codesign')]
import b3_storage_artificial_fixture as F

def main():
    mode=sys.argv[1];dest=HERE/'independent_writer_faults_final'/mode;dest.mkdir(parents=True,exist_ok=False);events=[]
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-independent-writer-') as tmp:
        root=Path(tmp);c,inner,ledger,auth=F.fixture(root,F.R.P.Settings(profile_n=16,mode=mode,codec='fused',workload='light'))
        original=F.R.Coordinator._record
        class FailWriter:
            def __init__(self,wrapped,coordinator):self.wrapped=wrapped;self.owner=coordinator
            def write(self,data):
                owner=self.owner;ticket=next(iter(owner.current.values()));task=ticket['worker']
                observation=dict(at='actual_file_write_before_OSError',reserved=owner.reservations,started=owner.started,state=dict(owner.states.counts),ticket=deepcopy(ticket),cursor=owner.next_indices[task],active_cursor=next(row['next'] for row in owner.active.values() if row['task']['key']==task),physical_state_sha256=sha256((owner.target/'trial_states.bin').read_bytes()).hexdigest())
                events.append(observation)
                assert observation['reserved']==observation['cursor']==observation['active_cursor']==1
                assert observation['started']==0 and observation['ticket']['index']==0 and observation['ticket']['state']=='START_RESERVED' and observation['state']['START_RESERVED']==1
                raise OSError('INDEPENDENT_ARTIFICIAL_TRACE_RESERVE_WRITE_FAILURE')
            def __getattr__(self,key):return getattr(self.wrapped,key)
        def record(owner,op,**fields):
            if op=='RESERVE' and not events:owner.trace.file=FailWriter(owner.trace.file,owner)
            return original(owner,op,**fields)
        with patch.object(F.R.Coordinator,'_record',record):report=F.invoke(root,c,inner,ledger,auth)
        assert events and report['reserved']==1 and report['started_observed']==0 and report['states']['CANCELLED_BEFORE_START']==1 and report['complete'] is False
        for name,value in [('pilot_contract.json',c),('inner_contract.json',inner),('observations.json',dict(report=report,writer_observations=events,producer_self_checker_called=False,scientific_calls=0))]:
            (dest/name).write_bytes(F.R.P.safe_json(value)+b'\n')
        shutil.copyfile(ledger.path,dest/'FAKE_ONLY_claims.sqlite3')
        with tarfile.open(dest/'artificial_raw.tar.gz','w:gz') as tf:tf.add(root/'artificial-output',arcname='artificial-output')
        forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in F.R.P.FORBIDDEN];assert not forbidden
        result=dict(mode=mode,actual_fault_count=len(events),at_fault=events[0],final=dict(reserved=report['reserved'],started=report['started_observed'],cancelled=report['states']['CANCELLED_BEFORE_START'],complete=report['complete']),source_sha256=sha256((RUNTIME/F.R.SOURCE).read_bytes()).hexdigest(),forbidden_scientific_imports=forbidden,scientific_calls=0)
        (dest/'fault_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
