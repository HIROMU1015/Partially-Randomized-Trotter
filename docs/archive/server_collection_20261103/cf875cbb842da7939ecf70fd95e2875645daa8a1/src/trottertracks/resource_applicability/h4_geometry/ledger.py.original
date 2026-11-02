"""Immutable completion records, external expected identity and durable reservations."""
from contextlib import contextmanager
import fcntl
import json
import os
import copy
from .identity import Stop, require, fingerprint, wrapper_key,hash_id

METRICS=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')


def wire(identity,axis,seed,index,numerical):
    hash_id(numerical)
    return dict(schema_version='h4-full-wrapper-checkpoint-v1', artifact_scope='SCIENCE_EXECUTION',
        geometry=identity['geometry'],candidate_template=identity['template'],axis=axis,
        hamiltonian_sha256=identity['H'],df_sha256=identity['DF'],state_sha256=identity['state'],
        input_fingerprint=identity['input'],candidate_fingerprint=fingerprint('h4-candidate-v1',identity),
        source_commit=identity['source'],compiler_fingerprint=identity['compiler'],environment_fingerprint=identity['environment'],
        wrapper_semantics=identity['wrapper_semantics'],trajectory_seed=seed,trajectory_index=index,
        wrapper_key=wrapper_key(identity,axis,seed,index),numerical_circuit_fingerprint=numerical,
        sample_weight=1. if index is None else 1/32,mandatory_stop=True)


def completion_digest(record):
    return fingerprint('h4-completion-record-v1',record)


def validate_complete(record,expected,registry,external_digest):
    require(completion_digest(record)==external_digest,'external completion digest')
    require(record.get('status')=='COMPLETE','non-COMPLETE owner')
    require(all(record.get(k)==v for k,v in expected.items()),'independent expected identity')
    require(registry.get(record['wrapper_key'])==record['numerical_circuit_fingerprint'],'independent numerical registry')
    require(set(record.get('metrics',{}))==set(METRICS) and
            all(type(v) is int and v>=0 for v in record['metrics'].values()),'complete metrics')


def validate_reuse_identity(owner,request):
    """Identity comparison only; this never permits reuse of RESERVED results."""
    require(owner['wrapper_key']!=request['wrapper_key'],'self cache link')
    fields=('geometry','candidate_template','hamiltonian_sha256','df_sha256','state_sha256','input_fingerprint',
            'candidate_fingerprint','source_commit','compiler_fingerprint','environment_fingerprint','wrapper_semantics',
            'axis','numerical_circuit_fingerprint')
    require(all(owner[k]==request[k] for k in fields),'cross geometry/cell/axis cache')


def reuse_owner(owner,owner_expected,request,registry,external_digest):
    validate_complete(owner,owner_expected,registry,external_digest)
    require(owner.get('cache_reuse') is False and owner.get('cache_owner_wrapper_key') is None,'cache owner chain')
    validate_reuse_identity(owner,request)
    require(owner.get('actual_transpile_invocation_id') is not None,'owner invocation missing')
    require(registry.get(request['wrapper_key'])==request['numerical_circuit_fingerprint'],'request numerical registry')
    return owner['metrics']


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


class Ledger:
    """One fresh map session. RESERVED/orphan states are fatal, not resumable.

    Append-only numbered snapshots are published exclusively under a single
    writer lock. A record is published before its ledger completion; a crash in
    either window leaves the durable invocation charged and requires review.
    """
    def __init__(self,budget,*,cap=74784):
        self.budget,self.cap,self.version=budget,cap,0
        self.entries,self.reservations,self.registry,self.expected={},{},{},{}
        self.saved_entries,self.saved_reservations,self.chain={},{},None
        budget.write('ledger.lock',b'')
        self.fd=os.open(budget.root/'ledger.lock',os.O_RDWR|os.O_NOFOLLOW)
        self._save()

    def close(self):
        os.close(self.fd)

    @contextmanager
    def locked(self):
        fcntl.flock(self.fd,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self.fd,fcntl.LOCK_UN)

    def _save(self):
        payload={'schema_version':'h4-completion-ledger-delta-v1','version':self.version,
                 'previous_digest':self.chain,
                 'entries':{k:v for k,v in self.entries.items() if self.saved_entries.get(k)!=v},
                 'reservations':{k:v for k,v in self.reservations.items() if self.saved_reservations.get(k)!=v},
                 'mandatory_stop':True}
        self.budget.write('ledger-%06d.json'%self.version,json_bytes(payload))
        self.chain=fingerprint('h4-ledger-delta-v1',payload)
        self.saved_entries.update(copy.deepcopy(payload['entries']))
        self.saved_reservations.update(copy.deepcopy(payload['reservations']))
        self.version+=1

    def register(self,expected):
        key=expected['wrapper_key']
        require(key not in self.expected,'duplicate logical wrapper')
        self.registry[key]=expected['numerical_circuit_fingerprint'];self.expected[key]=dict(expected)

    def reserve(self,key):
        with self.locked():
            require(key in self.expected and key not in self.entries and key not in self.reservations,'duplicate reservation')
            require(len(self.reservations)<self.cap,'actual invocation cap before compile')
            invocation='science-%06d'%(len(self.reservations)+1)
            self.reservations[key]={'status':'RESERVED','invocation':invocation}
            self._save()
        return invocation

    def complete(self,key,metrics,*,owner_key=None,interrupt_after_record=False):
        with self.locked():
            expected=self.expected[key];require(key not in self.entries,'duplicate completion')
            if owner_key is None:
                require(key in self.reservations and self.reservations[key]['status']=='RESERVED','compile not reserved')
                invocation=self.reservations[key]['invocation']
            else:
                require(owner_key in self.entries,'missing owner')
                owner=self.read(owner_key)
                metrics=reuse_owner(owner,self.expected[owner_key],expected,self.registry,self.entries[owner_key]['digest'])
                invocation=None
            record={**expected,'status':'COMPLETE','metrics':dict(metrics),'cache_reuse':owner_key is not None,
                    'cache_owner_wrapper_key':owner_key,'actual_transpile_invocation_id':invocation}
            digest=completion_digest(record)
            validate_complete(record,expected,self.registry,digest)
            self.budget.write('record-'+key+'.json',json_bytes(record))
            if interrupt_after_record:
                raise Stop('synthetic interruption after record; charged RESERVED remains')
            self.entries[key]={'status':'COMPLETE','digest':digest,'cache_owner_wrapper_key':owner_key}
            if owner_key is None:
                self.reservations[key]['status']='COMPLETE'
            self._save()
            return record

    def read(self,key):
        require(key in self.entries,'missing completion owner')
        fd=os.open(self.budget.root/('record-'+key+'.json'),os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            record=json.loads(stream.read())
        validate_complete(record,self.expected[key],self.registry,self.entries[key]['digest'])
        return record

    def audit(self):
        entries,reservations,chain={},{},None
        for version in range(self.version):
            payload=json.loads((self.budget.root/('ledger-%06d.json'%version)).read_bytes())
            require(payload['version']==version and payload['previous_digest']==chain,'ledger chain/digest')
            chain=fingerprint('h4-ledger-delta-v1',payload)
            entries.update(payload['entries']);reservations.update(payload['reservations'])
        require(chain==self.chain and entries==self.entries and reservations==self.reservations,'external ledger head/state')
        require(set(self.entries)==set(self.expected)==set(self.registry),'unresolved logical wrapper STOP')
        require(all(r['status']=='COMPLETE' for r in self.reservations.values()),'unresolved consumed reservation STOP')
        actual={p.name[7:-5] for p in self.budget.root.glob('record-*.json')}
        require(actual==set(self.entries),'orphan record STOP')
        require(not list(self.budget.root.glob('*.pending')),'orphan temporary STOP')
        for key in self.entries:
            record=self.read(key)
            owner=record['cache_owner_wrapper_key']
            if owner is not None:
                require(owner in self.entries,'missing owner')
                reuse_owner(self.read(owner),self.expected[owner],self.expected[key],self.registry,self.entries[owner]['digest'])
        return {'logical_wrappers':len(self.entries),'actual_invocations':len(self.reservations)}
