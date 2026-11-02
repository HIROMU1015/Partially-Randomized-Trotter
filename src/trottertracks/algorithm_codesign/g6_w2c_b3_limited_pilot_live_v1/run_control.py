"""Simple owned research run state. Exclusive intents spend run AND scope.

Derived from live_release_v1.authority.TestStore (841ac892). No keys, issuer,
revocation or external authorization service. Physical power-loss guarantees
and protection against malicious owner modifications are not claimed.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import os,sqlite3,json
from .guard import require,Closed
from . import spec as S


def private(root):
    p=Path(root).absolute()
    require(p==p.resolve() and p.is_dir() and p.stat().st_uid==os.getuid(),'OWNED_DEDICATED_ROOT')
    require(not p.is_symlink(),'STATE_SYMLINK')
    return p


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def owner_start(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


@dataclass(frozen=True)
class MockApproval:
    binding: str
    scope: str
    claim: str
    artificial_only: bool=True


def mock_approval(m):
    require(m['global_scope'].startswith('ARTIFICIAL_ONLY/RELEASE/') and
        m['SCIENCE_RUN_AUTHORIZED'] is False and m['artificial_only'] is True
        and m['real_grant'] is None,'MOCK_APPROVAL_ONLY')
    return MockApproval(S.digest(m),m['global_scope'],m['global_claim'])


class SimpleStore:
    def __init__(self,root,*,create=True,permit=None,candidate=None,manifest=None):
        self.root=private(root)
        # Artificial fixtures use dedicated test roots; other roots require
        # the exact source-bound live permit.
        if not self.root.name.startswith('b3-release-test-'):
            from .gate import require_live
            require_live(permit,candidate,manifest)
        self.path=self.root/'SIMPLE_RUN.sqlite3';self.intents=self.root/'intents'
        if create:
            require(not self.path.exists() and not self.intents.exists(),'NEW_DEDICATED_STORE_ONLY')
            self.intents.mkdir(mode=0o700)
            with sqlite3.connect(self.path) as db:
                db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE claims(scope TEXT PRIMARY KEY, claim TEXT UNIQUE, binding TEXT, status TEXT, spent INTEGER CHECK(spent=1), receipt TEXT)')
                db.execute('CREATE TABLE history(seq INTEGER PRIMARY KEY, scope TEXT, event TEXT, binding TEXT)')
                db.execute('CREATE TABLE runs(run_id TEXT PRIMARY KEY, scope TEXT UNIQUE, execution_state TEXT, owner_pid INTEGER, owner_start TEXT)')
            db.close()
            os.chmod(self.path,0o600);fsync_dir(self.root)
        self._locations()

    def _locations(self):
        private(self.root)
        for p in (self.path,self.intents):
            require(p.exists() and p.resolve()==p and not p.is_symlink()
                and p.stat().st_uid==os.getuid(),'STATE_STORE_LOCATION')
    @contextmanager
    def connect(self):
        self._locations();db=sqlite3.connect(self.path,timeout=60,isolation_level=None)
        try:
            db.execute('PRAGMA synchronous=FULL')
            yield db
        finally:db.close()
    def _intent(self,m):return self.intents/('scope-'+S.digest(m['global_scope'])+'.json')
    def _run_intent(self,m):return self.intents/('run-'+S.digest(m['run_id'])+'.json')
    def _intent_binding(self,m):
        expected=dict(scope=m['global_scope'],run_id=m['run_id'],claim=m['global_claim'],binding=S.digest(m),spent=True)
        for p in (self._intent(m),self._run_intent(m)):
            require(p.is_file() and p.resolve()==p and not p.is_symlink(),'SPENT_INTENT_REQUIRED')
            require(json.loads(p.read_text())==expected,'SPENT_INTENT_BINDING')

    def claim(self,m,approval,*,crash='none'):
        require(type(approval) is MockApproval and approval==mock_approval(m),'MOCK_APPROVAL_BINDING')
        return self._claim(m,crash=crash)
    def claim_live(self,c,m,permit):
        from .gate import require_live
        require_live(permit,c,m)
        require(str(self.root)==m['state_directory'],'LIVE_STATE_DESTINATION')
        return self._claim(m)
    def _claim(self,m,*,crash='none'):
        require(m['retry']==0 and type(m['retry']) is int and m['resume'] is False,'NO_RETRY_RESUME')
        require(crash in ('none','after_intent','before_commit','after_commit'),'CRASH_POINT')
        raw=S.encoded(dict(scope=m['global_scope'],run_id=m['run_id'],claim=m['global_claim'],binding=S.digest(m),spent=True))
        for p in (self._run_intent(m),self._intent(m)):
            try:fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            except FileExistsError as exc:raise Closed('RUN_OR_SCOPE_SPENT_OR_UNKNOWN') from exc
            # Any partial/empty intent remains spent. Never unlink/refund.
            with os.fdopen(fd,'wb',buffering=0) as stream:
                view=memoryview(raw)
                while view:
                    n=stream.write(view);require(type(n) is int and n>0,'INTENT_SHORT_WRITE');view=view[n:]
                os.fsync(stream.fileno())
            fsync_dir(self.intents)
        if crash=='after_intent':os._exit(61)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO claims VALUES(?,?,?,?,1,NULL)',(m['global_scope'],m['global_claim'],S.digest(m),'SPENT'))
            db.execute('INSERT INTO runs VALUES(?,?,?,?,?)',(m['run_id'],m['global_scope'],'START_RESERVED/SPENT',os.getpid(),owner_start(os.getpid())))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],'START_RESERVED/SPENT',S.digest(m)))
            if crash=='before_commit':os._exit(62)
            db.execute('COMMIT')
        if crash=='after_commit':os._exit(63)
        return dict(status='START_RESERVED/SPENT',binding=S.digest(m))

    def execution_state(self,m):
        with self.connect() as db:row=db.execute('SELECT execution_state,owner_pid,owner_start FROM runs WHERE run_id=?',(m['run_id'],)).fetchone()
        if row is None:
            return 'UNKNOWN' if self._run_intent(m).exists() or self._intent(m).exists() else 'NOT_STARTED'
        if row[0]=='START_RESERVED/SPENT' and owner_start(row[1])!=row[2]:return 'UNKNOWN'
        return row[0]

    def enter(self,m):
        self._intent_binding(m)
        with self.connect() as db:
            row=db.execute('SELECT claim,binding,status,spent FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            owner=db.execute('SELECT owner_pid,owner_start FROM runs WHERE run_id=?',(m['run_id'],)).fetchone()
        require(row==(m['global_claim'],S.digest(m),'SPENT',1) and
            self.execution_state(m)=='START_RESERVED/SPENT' and owner==(os.getpid(),owner_start(os.getpid())),'STOP_UNKNOWN_OR_OWNER_NO_ENTER')

    def stop(self,m,*,unknown=False):
        self._intent_binding(m);state=self.execution_state(m)
        terminal='UNKNOWN' if unknown or state=='UNKNOWN' else 'STOPPED'
        if state in ('COMPLETED','UNKNOWN') and not unknown:terminal=state
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT claim,binding,status FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            if row is None:
                db.execute('INSERT INTO claims VALUES(?,?,?,?,1,NULL)',(m['global_scope'],m['global_claim'],S.digest(m),'UNKNOWN_STOPPED'))
                db.execute('INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?)',(m['run_id'],m['global_scope'],'UNKNOWN',os.getpid(),owner_start(os.getpid())))
            else:
                require(row[:2]==(m['global_claim'],S.digest(m)),'STOP_BINDING')
                db.execute('UPDATE claims SET status=? WHERE scope=?',('UNKNOWN_STOPPED' if terminal=='UNKNOWN' else 'STOPPED',m['global_scope']))
                db.execute('UPDATE runs SET execution_state=? WHERE run_id=?',(terminal,m['run_id']))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],terminal,S.digest(m)));db.execute('COMMIT')

    def link_receipt(self,m,digest):
        self._intent_binding(m)
        require(type(digest) is str and len(digest)==64,'RECEIPT_HASH')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status,receipt FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            require(row==('STOPPED',None) and self.execution_state(m)=='STOPPED','RECEIPT_TERMINAL_ONCE')
            db.execute('UPDATE claims SET receipt=? WHERE scope=?',(digest,m['global_scope']))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],'RECEIPT_LINK',digest));db.execute('COMMIT')

    def mark_completed(self,m):
        """Only after durable caller/output evidence has closed; no new ENTER."""
        self._intent_binding(m)
        out=Path(m['storage_destination'])
        from hashlib import sha256
        caller=json.loads((out/'caller.json').read_text());delivery=json.loads((out/'delivery.json').read_text())
        final=json.loads((out/'final.json').read_text())
        require(caller['observed'] is True and caller['delivery_sha256']==sha256((out/'delivery.json').read_bytes()).hexdigest()
            and delivery['final_sha256']==sha256((out/'final.json').read_bytes()).hexdigest()
            and final['complete'] is True and final['states']['COMPLETED']==m['planned']==16896
            and final['manifest_binding']==S.digest(m),'COMPLETION_CALLER_EVIDENCE')
        digest=sha256((out/'receipt.json').read_bytes()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            require(db.execute('SELECT receipt,status FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()==(digest,'STOPPED')
                and self.execution_state(m)=='STOPPED','COMPLETION_SPENT_RECEIPT')
            db.execute('UPDATE runs SET execution_state=? WHERE run_id=?',('COMPLETED',m['run_id']))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],'COMPLETED',digest));db.execute('COMMIT')

    def snapshot(self,m):
        with self.connect() as db:
            row=db.execute('SELECT scope,claim,binding,status,spent,receipt FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
        from .metadata import context
        mode=context(m)
        binding_matches=row is None or row[:3]==(m['global_scope'],m['global_claim'],S.digest(m))
        state=self.execution_state(m)
        return dict(schema='b3_simple_run_state_v1',claim=list(row) if row else None,
            history=[list(x) for x in history],execution_state=state,run_status=state,execution_metadata=mode,execution_metadata_verified=binding_matches,
            claim_spent=bool(row and row[4]==1) or self._intent(m).exists() or self._run_intent(m).exists(),
            intent_exists=self._intent(m).exists(),run_intent_exists=self._run_intent(m).exists(),
            real_authority=False,external_authority_used=False,real_authority_semantics='LEGACY_ALIAS_FOR_EXTERNAL_AUTHORITY_USED',physical_power_loss_durability='NOT_ESTABLISHED',retry=0,resume=False)
