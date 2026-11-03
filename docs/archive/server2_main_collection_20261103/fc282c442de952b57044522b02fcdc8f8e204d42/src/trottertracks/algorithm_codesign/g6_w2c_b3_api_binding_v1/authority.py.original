"""Binding candidate using only the fixed private FAKE_ONLY SQLite store."""
from pathlib import Path
from ..g6_w2c_b3_bundle_store_v1 import FakeStore
from .static import SEED, digest, need

def binding(c, root, *, scope=None):
    root=Path(root).resolve()
    scope=scope or 'ARTIFICIAL_ONLY/API_PORT/'+root.name
    need(scope.startswith('ARTIFICIAL_ONLY/API_PORT/'), 'FRESH_ARTIFICIAL_SCOPE_ONLY')
    return dict(schema='b3_FAKE_API_authority_binding_v1',
        formal_source_release=None, source_release_candidate=c['source_release_candidate'],
        scientific_input_hash=c['scientific_reference_sha256'], runtime_hash=digest(c['python']),
        contract_hash=digest(c), global_scope=scope, global_claim=digest(['FAKE_API_CLAIM',scope,digest(c)]),
        call_budget=16896, result_budget=1<<36, log_budget=1<<28,
        seed_namespace=SEED, retry=0, resume=False,
        worker_policy='fresh spawn, owned workers; S sequential / P1 fixed candidate eight',
        storage_destination=str(root), monitoring_STOP_hash=digest(c['monitoring']),
        artificial_only=True, real_grant=None, real_authority=False)

def approval(b):
    return dict(schema='b3_FAKE_approval_v1', binding_hash=digest(b), artificial_only=True,
                execution_permission='ARTIFICIAL_INTERFACE_ONLY', real_grant=None)

class ArtificialAuthority:
    def __init__(self, c, root):
        self.c,self.root=c,Path(root).resolve()
        self.expected=binding(c,root)
        self.store=FakeStore(root)

    def claim(self,b,a,*,fault=lambda p:None):
        need(b==self.expected and a==approval(self.expected), 'FAKE_APPROVAL_BINDING_MISMATCH')
        self.store.claim(b,fault)

    def stop(self,b):
        need(b==self.expected, 'FAKE_STOP_BINDING_MISMATCH')
        self.store.stop(b)

    def snapshot(self):
        return self.store.snapshot(self.expected)
