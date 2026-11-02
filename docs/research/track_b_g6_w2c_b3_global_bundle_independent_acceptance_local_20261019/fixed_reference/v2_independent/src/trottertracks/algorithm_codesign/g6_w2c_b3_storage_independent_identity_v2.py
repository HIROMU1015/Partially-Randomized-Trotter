"""V2 consumer identity: pinned public bytes, no Producer/Admission imports."""
from copy import deepcopy
from pathlib import Path
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_v3_offline_identity as I
from . import g6_w2c_b3_pilot_independent_identity as P
PUBLIC='596781daeba253418ba35f4b7964312a54caf233'
MEASURED='45bf46b4a833e71f138903fb511f2741e54043b7'
PREFIX='docs/research/track_b_g6_w2c_b3_trace_reservation_accounting_repair_20261019/'
SOURCE='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_storage_v2.py'
CHECKER='src/trottertracks/algorithm_codesign/g6_w2c_b3_storage_verify_v2.py'
class Reference:
    def __init__(self,public_tree,old_tree,legacy,trusted_candidate):
        self.tree=Path(public_tree);self.old=P.Reference(old_tree,legacy);self.legacy=self.old.legacy
        inv=V.plain((self.tree/PREFIX/'source_inventory.json').read_text())
        V.require(inv['measured_SOURCE_commit']==MEASURED and inv['fixed_checker_commit']=='49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866','V2_SOURCE_COMMITS')
        self.pins={r['path']:r['sha256'] for r in inv['files']+inv['protected']}
        for p,h in self.pins.items():V.require(V.file_digest(V.safe_path(self.tree,p))==h,'V2_REFERENCE_BYTES:'+p)
        self.schema=V.plain((self.tree/PREFIX/'storage_v2.schema.json').read_text())
        self.template=deepcopy(trusted_candidate)
        # Derive all non-environment changes from the frozen V1 contract.
        expected=deepcopy(self.old.template)
        expected.update(schema='b3_pilot_storage_repair_candidate_v2',storage_protocol='MONOTONE_ACCOUNTING_EXPLICIT_INCOMPLETE_EVIDENCE_V2')
        for p in (SOURCE,CHECKER,PREFIX+'storage_v2.schema.json'):expected['source_pins'][p]=self.pins[p]
        for k in ('runtime_root','python'):expected[k]=self.template[k]
        V.require(I.strict_equal(self.template,expected),'V2_UNDECLARED_CONTRACT_DELTA')
        self.python_claim=self.template['python']
    def inner(self,c,inner):
        I.validate(c,self.schema,{})
        V.require(I.strict_equal(c,self.template),'V2_FIXED_CANDIDATE')
        for path,h in c['source_pins'].items():
            V.require(self.pins.get(path)==h and V.file_digest(V.safe_path(self.tree,path))==h,'V2_SOURCE_PIN:'+path)
        for path,h in c['input_pins'].items():
            V.require(V.file_digest(V.safe_path(self.old.tree,path))==h,'V2_INPUT_PIN:'+path)
        V.require(inner['pilot_candidate_binding']==V.identity(c),'V2_INNER_PILOT_BINDING')
        binding=self.legacy.contract(inner)
        V.require(inner['runtime_manifest']['runtime_root']==c['runtime_root'] and I.strict_equal(inner['runtime_manifest']['python'],c['python']),'V2_RUNTIME_CANDIDATE_BINDING')
        V.require(inner['settings']['mode'] in c['proposal']['modes'] and inner['settings']['codec']==c['proposal']['codec'],'V2_INNER_VARIANT')
        return dict(pilot_digest=V.identity(c),inner_digest=V.identity(inner),inner_binding=binding,historical_python_bytes_independently_proven=False,real_authorization_verified=False)
