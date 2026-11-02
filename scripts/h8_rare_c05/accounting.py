"""Private transactional receipts, conditional execution gate and identity cache."""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
from .sampling import digest,require


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)


@contextmanager
def lock(path):
    with Path(path).open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


class Budget:
    def __init__(self,directory,limits):
        self.root=Path(directory);self.limits=dict(limits);self.root.mkdir(exist_ok=True,parents=True)
    def take(self,kind,n,key):
        require(type(n) is int and n>0 and kind in self.limits,'CALL_KIND_COUNT')
        require(Path(key).name==key and key not in ('.','..'),'SAFE_CALL_KEY')
        with lock(self.root/'ledger.lock'):
            p=self.root/'ledger.json';d=read(p) if p.exists() else {'used':{},'limits':self.limits,'nonrefundable':True}
            require(d['limits']==self.limits,'BUDGET_CHANGED')
            require(d['used'].get(kind,0)+n<=self.limits[kind],'BUDGET_EXCEEDED')
            reservation=self.root/(key+'.reserved.json')
            fd=os.open(reservation,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:json.dump(dict(kind=kind,count=n,nonrefundable=True),f);f.flush();os.fsync(f.fileno())
            d['used'][kind]=d['used'].get(kind,0)+n;atomic(p,d)
    def used(self):return read(self.root/'ledger.json')['used'] if (self.root/'ledger.json').exists() else {}


_FACTORY_PROOF=object()

@dataclass(frozen=True)
class Permit:
    mode:str
    contract_sha256:str
    condition:str
    manifest_digest:str=""
    bridge_digest:str=""
    proof:object=None


def check_permit(permit,mock):
    require(isinstance(permit,Permit) and permit.mode==('MOCK' if mock else 'SCIENCE')
            and (mock or permit.proof is _FACTORY_PROOF),'EXECUTION_AUTHORIZATION_REQUIRED')


def begin_approved(contract_path,grant_path,marker,approval_path,byte_rows):
    """Implemented for a future explicitly approved C1; NEVER called on real files in C0.5."""
    c=read(contract_path)
    require(c.get('approved') is True and c.get('runtime_authorization') is True,'CONTRACT_NOT_APPROVED')
    require(c.get('numeric_model_approved') is True,'NUMERICAL_CONVENTION_NOT_APPROVED')
    require(c.get('new_execution_SOURCE') and c.get('frozen_scientific_budget'),'UNSEALED_SOURCE_OR_BUDGET')
    require(c['condition'] in ('H8_0.05','H8_0.0005'),'FIXED_H8_CONDITION')
    grant=read(grant_path);approval=read(approval_path)
    require(approval.get('explicit_C1_scientific_execution_authorized') is True,'NO_USER_C1_RECEIPT')
    require(approval['condition']==c['condition'] and approval['contract_sha256']==sha(contract_path),'APPROVAL_SCOPE')
    require(grant['budget']==c['frozen_scientific_budget']==approval['budget'],'APPROVED_SCIENCE_BUDGET')
    require(grant['contract_sha256']==sha(contract_path) and grant['approval_receipt_sha256']==sha(approval_path),'GRANT_BINDING')
    require(grant['SOURCE']==c['new_execution_SOURCE'] and grant['condition']==c['condition'],'GRANT_SCOPE')
    require(c.get('old_cohort_seed_exclusion_complete') is True,'OLD_SEED_EXCLUSION_INCOMPLETE')
    require(c.get('fork_safety_verified') is True and c.get('live_session_library_verified') is True,'LIVE_FORK_OR_LIBRARY_NOT_VERIFIED')
    require(grant.get('one_shot') is True and not grant.get('retry') and not grant.get('resume'),'ONE_SHOT_ONLY')
    require(byte_rows and {r['role'] for r in byte_rows}>={'SOURCE','input','runtime','provider','session_library'},'INCOMPLETE_BYTE_CLOSURE')
    require(digest(byte_rows)==c['private_byte_manifest_digest'],'MANIFEST_NOT_BOUND')
    for row in byte_rows:require(sha(row['path'])==row['sha256'],'SOURCE_INPUT_RUNTIME_CHANGED')
    fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(dict(consumed=True,nonrefundable=True,grant_sha256=sha(grant_path)),f)
    return Permit('SCIENCE',sha(contract_path),c['condition'],digest(byte_rows),c['runtime_bridge_config_digest'],_FACTORY_PROOF)


IDENTITY_FIELDS={'SOURCE','provider','runtime','session_library','hamiltonian','state','snapshot','preparation',
                 'candidate','q','r','K','delta','event_digest','control','scalar','axis','measurement','compiler','fingerprint','numeric_policy'}
class IdentityCache:
    def __init__(self,path): self.root=Path(path);self.root.mkdir(parents=True,exist_ok=True)
    def get_or_compute(self,identity,compute):
        require(IDENTITY_FIELDS<=set(identity) and all(identity[k] is not None for k in IDENTITY_FIELDS),'INCOMPLETE_CACHE_IDENTITY')
        require(identity['axis'] in ('cosine','sine') and identity['measurement'] is True,'FULL_WRAPPER_IDENTITY')
        key=digest(identity);directory=self.root/key;directory.mkdir(exist_ok=True)
        with lock(directory/'lock'):
            p=directory/'complete.json'
            if p.exists():
                record=read(p)
                require(record['identity']==identity and record['value_sha256']==digest(record['value']),'CACHE_CORRUPTION_OR_IDENTITY_MISMATCH')
                return record['value'],True
            require(not (directory/'started.json').exists(),'INCOMPLETE_CACHE_ATTEMPT_NO_RETRY')
            atomic(directory/'started.json',{'nonrefundable_attempt':True})
            value=compute();atomic(p,dict(identity=identity,value=value,value_sha256=digest(value)))
            return value,False


def sample_statistics(values):
    """Pure saved-value arithmetic; used on toy arrays ONLY in C0.5."""
    from fractions import Fraction
    from decimal import Decimal,localcontext
    n=len(values)
    if not n:return dict(n=0,mean=None,variance=None,SE=None)
    mean=Fraction(sum(values),n)
    var=sum((Fraction(v)-mean)**2 for v in values)/(n-1) if n>1 else None
    with localcontext() as ctx:
        ctx.prec=100
        se=str((Decimal(var.numerator)/Decimal(var.denominator)/n).sqrt()) if var is not None else None
    return dict(n=n,mean=str(mean),variance=str(var) if var is not None else None,SE=se)


def paired_pilot_summary(results,shots,candidate,phase):
    from .sampling import STRATA
    groups={s:[] for s in STRATA};seen=set()
    for row in results:
        require(row['candidate']==candidate and row['phase']==phase,'NO_COHORT_POOLING')
        require(row['task_id'] not in seen,'DUPLICATE_ACCOUNTING_ROW');seen.add(row['task_id'])
        axes=row['axes'];require(set(axes)=={'cosine','sine'},'PAIRED_METRICS')
        require(axes['cosine']['event_digest']==axes['sine']['event_digest']==row['event_digest'],'PAIRED_ACCOUNTING_EVENT')
        groups[row['stratum']].append(sum(shots[a]*axes[a]['metrics']['RZ'] for a in ('cosine','sine')))
    return dict(groups={s:sample_statistics(v) for s,v in groups.items()},unknown_population_means={s:None for s in STRATA},
                population_ranking_guarantee=False,all_support_cost_upper_bound=None,confidence_interval=None)


def independent_pilot_moments(results,shots,summary):
    """Independent integer raw-moment route; no primary centered-stat function call."""
    from fractions import Fraction
    for stratum,stats in summary['groups'].items():
        values=[sum(shots[a]*r['axes'][a]['metrics']['RZ'] for a in ('cosine','sine')) for r in results if r['stratum']==stratum]
        n=len(values);require(stats['n']==n,'INDEPENDENT_N')
        mean=Fraction(sum(values),n) if n else None
        variance=(Fraction(sum(v*v for v in values))-Fraction(sum(values)**2,n))/(n-1) if n>1 else None
        require(stats['mean']==(str(mean) if mean is not None else None),'INDEPENDENT_MEAN')
        require(stats['variance']==(str(variance) if variance is not None else None),'INDEPENDENT_VARIANCE')
    return dict(status='PASS_INDEPENDENT_INTEGER_MOMENTS_NOT_POPULATION_GUARANTEE',certificate=False)
