"""Private SQLite fake one-shot harness. Deliberately has no authority API."""
from pathlib import Path
from hashlib import sha256
import sqlite3,os
from .g6_w2c_b3_bundle_contract_v1 import digest

class Denied(RuntimeError):pass

def private(root):
    p=Path(root).absolute()
    if p.resolve()!=p or p.parent!=Path('/tmp') or not p.name.startswith('b3-bundle-artificial-') or not p.is_dir() or p.stat().st_uid!=os.getuid():raise Denied('PRIVATE_FAKE_TEMP_ONLY')
    return p

class FakeStore:
    """Commit reservation before ACK; scope can never be deleted/released."""
    def __init__(self,root):
        self.root=private(root);self.path=self.root/'FAKE_ONLY.sqlite3'
        if self.path.exists():raise Denied('NEW_FAKE_STORE_ONLY')
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE claims(scope TEXT PRIMARY KEY, claim TEXT UNIQUE NOT NULL, binding TEXT NOT NULL, status TEXT NOT NULL, spent INTEGER NOT NULL CHECK(spent=1))')
            db.execute('CREATE TABLE history(seq INTEGER PRIMARY KEY, scope TEXT, event TEXT, binding TEXT)')
    def _connect(self):
        root=private(self.root)
        if self.path!=root/'FAKE_ONLY.sqlite3' or self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_uid!=os.getuid():raise Denied('FAKE_STORE_PATH_BINDING')
        return sqlite3.connect(self.path,timeout=30,isolation_level=None)
    def claim(self,m,fault=lambda p:None):
        if m['retry']!=0 or m['resume'] is not False:raise Denied('RETRY_RESUME_FORBIDDEN')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                db.execute('INSERT INTO claims VALUES(?,?,?,?,1)',(m['global_scope'],m['global_claim'],digest(m),'SPENT'))
                db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],'CLAIM_COMMITTED',digest(m)))
                db.execute('COMMIT')
            except sqlite3.IntegrityError as exc:
                db.execute('ROLLBACK');raise Denied('SCOPE_OR_CLAIM_SPENT') from exc
        fault('admission.commit_ack')
    def stop(self,m,fault=lambda p:None):
        fault('admission.stop')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT claim,binding,status,spent FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            if row is None or row[:2]!=(m['global_claim'],digest(m)) or row[3]!=1:
                db.execute('ROLLBACK');raise Denied('FAKE_STOP_BINDING')
            if row[2]!='STOPPED':
                db.execute("UPDATE claims SET status='STOPPED' WHERE scope=?",(m['global_scope'],))
                db.execute('INSERT INTO history(scope,event,binding) VALUES(?,?,?)',(m['global_scope'],'STOP_COMMITTED',digest(m)))
            db.execute('COMMIT')
    def snapshot(self,m):
        with self._connect() as db:
            row=db.execute('SELECT scope,claim,binding,status,spent FROM claims WHERE scope=?',(m['global_scope'],)).fetchone()
            history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
        return dict(schema='b3_FAKE_ONLY_snapshot_v1',claim=list(row) if row else None,history=[list(x)for x in history],real_authority=False)
