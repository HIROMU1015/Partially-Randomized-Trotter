"""Owned TEST_ONLY persistence prototype. Never opens a production store.

An exclusive durable attempt intent spends the test scope even if the process
dies before SQLite COMMIT. Missing ACK is never a permission to try again.
SQLite FULL/fsync requests are tested; physical power-loss durability is not.
"""
from dataclasses import dataclass
from pathlib import Path
import sqlite3,os,json
from .guard import require,Closed
from ..g6_w2c_b3_live_preflight_v1 import spec as S


def private(root):
    p=Path(root).absolute()
    require(p.resolve()==p and p.parent==Path('/tmp') and p.name.startswith('b3-release-test-')
        and p.is_dir() and p.stat().st_uid==os.getuid(),'OWNED_TEST_ROOT_ONLY')
    return p


@dataclass(frozen=True)
class TestGrant:
    binding: str
    scope: str
    claim: str
    issuer: str='BUILTIN_TEST_ONLY_NO_CREDENTIAL'
    real_authority: bool=False


def test_grant(m):
    require(m['global_scope'].startswith('ARTIFICIAL_ONLY/RELEASE/') and m['real_authority'] is False
            and m['real_grant'] is None and m['SCIENCE_RUN_AUTHORIZED'] is False,'TEST_GRANT_ONLY')
    return TestGrant(S.digest(m),m['global_scope'],m['global_claim'])


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class TestStore:
    def __init__(self,root,*,create=True):
        self.root=private(root);self.path=self.root/'TEST_ONLY.sqlite3';self.intents=self.root/'intents'
        if create:
            require(not self.path.exists() and not self.intents.exists(),'NEW_TEST_STORE_ONLY')
            self.intents.mkdir(mode=0o700)
            with sqlite3.connect(self.path) as db:
                db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE claims(scope TEXT PRIMARY KEY, claim TEXT UNIQUE, binding TEXT, status TEXT, spent INTEGER CHECK(spent=1), receipt TEXT)')
                db.execute('CREATE TABLE history(seq INTEGER PRIMARY KEY, scope TEXT, event TEXT, binding TEXT)')
                db.execute('CREATE TABLE revoked(binding TEXT PRIMARY KEY)')
            os.chmod(self.path,0o600);fsync_dir(self.root)
        require(self.path.is_file() and not self.path.is_symlink() and self.intents.is_dir()
                and not self.intents.is_symlink(),'TEST_STORE_FILES')

    def connect(self):
        require(private(self.root)==self.root and self.path.resolve()==self.path
            and self.path.stat().st_uid==os.getuid(),'TEST_STORE_LOCATION')
        db=sqlite3.connect(self.path,timeout=60,isolation_level=None)
        db.execute('PRAGMA synchronous=FULL');return db

    def _intent(self,m):return self.intents/(S.digest(m['global_scope'])+'.json')

    def _validate(self,m,grant):
        expected=test_grant(m)
        require(type(grant) is TestGrant and grant==expected,'TEST_GRANT_BINDING')
        require(m['retry']==0 and type(m['retry']) is int and m['resume'] is False,'RETRY_RESUME')
        with self.connect() as db:
            require(db.execute('SELECT 1 FROM revoked WHERE binding=?',(grant.binding,)).fetchone() is None,'TEST_REVOKED')

    def claim(self,m,grant,*,crash='none'):
        self._validate(m,grant)
        require(crash in ('none','after_intent','before_commit','after_commit'),'TEST_CRASH_POINT')
        path=self._intent(m)
        try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError as exc:raise Closed('TEST_SCOPE_SPENT_OR_ACK_UNKNOWN') from exc
        try:
            raw=S.encoded(dict(scope=m['global_scope'],claim=m['global_claim'],binding=S.digest(m),
                               spent=True,artificial_only=True))
            with os.fdopen(fd,'wb',buffering=0) as stream:
                view=memoryview(raw)
                while view:
                    n=stream.write(view);require(type(n) is int and n>0,'INTENT_SHORT_WRITE');view=view[n:]
                os.fsync(stream.fileno())
            fsync_dir(self.intents)
            if crash=='after_intent':os._exit(61)
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO claims VALUES(?,?,?,?,1,NULL)',
                    (m['global_scope'],m['global_claim'],S.digest(m),'SPENT'))
                db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',
                    (m['global_scope'],'CLAIM_COMMIT',S.digest(m)))
                if crash=='before_commit':os._exit(62)
                db.execute('COMMIT')
            if crash=='after_commit':os._exit(63)
            return dict(status='ADMITTED_TEST_ONLY',binding=grant.binding)
        except BaseException:
            # Never unlink the attempt intent or refund the scope.
            raise

    def _intent_binding(self,m):
        p=self._intent(m);require(p.is_file() and not p.is_symlink(),'SPENT_INTENT_REQUIRED')
        value=json.loads(p.read_text())
        require(value==dict(scope=m['global_scope'],claim=m['global_claim'],binding=S.digest(m),
                             spent=True,artificial_only=True),'SPENT_INTENT_BINDING')

    def stop(self,m):
        self._intent_binding(m)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT claim,binding,status FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            if row is None:
                db.execute('INSERT INTO claims VALUES(?,?,?,?,1,NULL)',
                    (m['global_scope'],m['global_claim'],S.digest(m),'UNKNOWN_STOPPED'))
            else:
                require(row[:2]==(m['global_claim'],S.digest(m)),'STOP_BINDING')
                if row[2] not in ('STOPPED','UNKNOWN_STOPPED'):
                    db.execute("UPDATE claims SET status='STOPPED' WHERE scope=?",(m['global_scope'],))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',
                       (m['global_scope'],'STOP',S.digest(m)));db.execute('COMMIT')

    def enter(self,m):
        """Parent-side test admission assertion, never a scientific dispatch API."""
        self._intent_binding(m)
        with self.connect() as db:
            row=db.execute('SELECT claim,binding,status,spent FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
        require(row==(m['global_claim'],S.digest(m),'SPENT',1),'STOP_OR_UNKNOWN_NO_ENTER')

    def link_receipt(self,m,digest):
        self._intent_binding(m);require(type(digest) is str and len(digest)==64,'RECEIPT_HASH')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status,receipt FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            require(row is not None and row[0]=='STOPPED' and row[1] is None,'RECEIPT_TERMINAL_ONCE')
            db.execute('UPDATE claims SET receipt=? WHERE scope=?',(digest,m['global_scope']))
            db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',
                       (m['global_scope'],'RECEIPT_LINK',digest));db.execute('COMMIT')

    def revoke_test(self,grant):
        require(type(grant) is TestGrant and grant.real_authority is False,'TEST_ONLY_REVOKE')
        with self.connect() as db:db.execute('INSERT OR IGNORE INTO revoked VALUES(?)',(grant.binding,))

    def snapshot(self,m):
        with self.connect() as db:
            row=db.execute('SELECT scope,claim,binding,status,spent,receipt FROM claims WHERE scope=?',
                           (m['global_scope'],)).fetchone()
            history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
        return dict(schema='b3_TEST_ONLY_persistent_store_snapshot_v1',claim=list(row) if row else None,
            history=[list(x) for x in history],intent_exists=self._intent(m).exists(),real_authority=False,
            requested_durability='SQLITE_FULL_PLUS_INTENT_FILE_AND_DIRECTORY_FSYNC',
            physical_power_loss_durability='NOT_ESTABLISHED')


class RealAuthority:
    def admit(self,*args,**kwargs):
        raise Closed('REAL_OWNER_ISSUER_CREDENTIAL_DURABILITY_UNCONFIGURED')
