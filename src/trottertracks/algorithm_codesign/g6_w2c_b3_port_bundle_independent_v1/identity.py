"""Strict pinned integration identities, independent of the Producer contract."""
from copy import deepcopy
from pathlib import Path
import sys
from .. import g6_w2c_b3_offline_result as V
from ..g6_w2c_b3_bundle_independent_identity_v1 import equal,read,hashed,plan,CLOSED

class Reference:
    def __init__(self,tree,template,*,trusted=None,allow_runtime_projection=False,verify_local_runtime=False):
        self.tree=Path(tree);self.template=deepcopy(template);self.trusted=deepcopy(template if trusted is None else trusted)
        projected=deepcopy(self.trusted)
        if allow_runtime_projection:
            for key in ('runtime_root','python','worker_runtime'):projected[key]=self.template[key]
        equal(projected,self.template,'UNDECLARED_CANDIDATE_DELTA')
        self.projected=allow_runtime_projection;self.local_runtime=verify_local_runtime;self.verify_bytes()
    def verify_bytes(self):
        c=self.template
        if self.local_runtime:
            local=self.trusted;equal(Path(local['python']['executable']).resolve(),Path(sys.executable).resolve(),'LOCAL_EXECUTABLE');equal(local['python']['version'],sys.version,'LOCAL_PYTHON_VERSION')
            for py in (local['python'],local['worker_runtime']['python']):
                equal(V.file_digest(py['executable']),py['executable_sha256'],'LOCAL_EXECUTABLE_BYTES')
                for row in py['libraries'].values():equal(V.file_digest(row['path']),row['sha256'],'LOCAL_LIBRARY_BYTES')
            for p,h in {**local['source_pins'],**local['scientific_reference']['input_pins']}.items():equal(V.file_digest(V.safe_path(Path(local['runtime_root']),p)),h,'LOCAL_RUNTIME_SOURCE:'+p)
        for p,h in c['source_pins'].items():equal(V.file_digest(V.safe_path(self.tree,p)),h,'FIXED_SOURCE:'+p)
        for p,h in c['scientific_reference']['input_pins'].items():equal(V.file_digest(V.safe_path(self.tree,p)),h,'FIXED_INPUT:'+p)
    def identity(self,c,m):
        equal(c,self.trusted,'TRUSTED_INTEGRATION_CANDIDATE');self.verify_bytes()
        equal(c['schema'],'b3_closed_port_global_bundle_candidate_v1','CANDIDATE_SCHEMA');equal(m['schema'],'b3_closed_port_global_bundle_manifest_v1','MANIFEST_SCHEMA')
        for obj in (c,m):
            for k,v in CLOSED.items():equal(obj[k],v,'CLOSED:'+k)
        equal(c['status'],'ARTIFICIAL_ONLY_UNADOPTED_EXECUTION_CLOSED','STATUS')
        equal(c['source_release_candidate'],hashed(c['source_pins']),'SOURCE_RELEASE_DIGEST')
        equal(m['candidate_binding'],hashed(c),'MANIFEST_CANDIDATE');equal(m['source_release_candidate'],c['source_release_candidate'],'MANIFEST_SOURCE')
        equal(c['fixed_bundle_commit'],'d24cccbecdf0928fe977080dd7b28c591df0089e','ORIGINAL_BUNDLE')
        equal(c['fixed_port_commit'],'82896d53fa97235fa848d6c0833eb193af8b99d6','ORIGINAL_PORT')
        equal(c['fixed_order'],['S','P1'],'S_THEN_P1');equal(c['production_gate'],'NO_IMPLEMENTED_SCIENCE_DISPATCH','CLOSED_GATE')
        equal(c['integration'],'ONE_GLOBAL_COORDINATOR_REAL_PORT_DISPATCH_V1','INTEGRATION_KIND')
        equal(c['storage_protocol'],'GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1','STORAGE_PROTOCOL')
        binding=c['endpoint_binding'];equal(m['endpoint_binding'],hashed(binding),'ENDPOINT_BINDING')
        equal(binding['real_dispatch'],False,'NO_REAL_DISPATCH');equal(binding['artificial_only'],True,'ARTIFICIAL_ENDPOINT')
        equal(binding['port_source'],{p:h for p,h in c['source_pins'].items() if '/g6_w2c_b3_api_binding_v1/' in p},'PORT_PIN_SET')
        equal(binding['adapter_source'],{p:h for p,h in c['source_pins'].items() if '/g6_w2c_b3_port_bundle_v1/' in p},'ADAPTER_PIN_SET')
        equal(len(binding['port_source']),6,'SIX_PORT_FILES')
        for key in ('source_pins','runtime_root'):equal(c['worker_runtime'][key],c[key],'WORKER_RUNTIME:'+key)
        fixture=c['worker_runtime']['artificial_fixture'];equal(fixture,dict(path='src/trottertracks/algorithm_codesign/g6_w2c_b3_port_bundle_v1/adapter.py',sha256=c['source_pins']['src/trottertracks/algorithm_codesign/g6_w2c_b3_port_bundle_v1/adapter.py']),'WORKER_FACTORY_PIN')
        p=c['proposal'];equal(p,self.template['proposal'],'STATIC_PROPOSAL');equal(p['total_calls'],16896,'STATIC_16896')
        equal(p['result_bytes'],1<<36,'RESULT_64GiB');equal(p['log_bytes'],1<<28,'LOG_256MiB')
        tasks,total=plan(m['profile_n']);equal(m['tasks'],tasks,'GLOBAL_TASKS_OFFSETS');equal(m['planned'],total,'GLOBAL_PLANNED')
        equal(m['static_scientific_candidate_calls'],16896,'STATIC_CALLS')
        V.require(m['modes'] in (['S'],['P1'],['S','P1']),'MODES');V.require(type(m['call_limit'])is int and 0<=m['call_limit']<=total,'GLOBAL_CALL_LIMIT')
        for ch in ('result','log'):V.require(type(m['byte_limits'][ch])is int and 0<m['byte_limits'][ch]<=p[ch+'_bytes'],'GLOBAL_BYTE_LIMIT:'+ch)
        for k,v in dict(seed_namespace='ARTIFICIAL_API_BINDING_V1',codec='fused',observer='cached_all_call_return',workers=8,start_method='spawn').items():equal(m[k],v,'EXECUTION_BINDING:'+k)
        V.require(m['port_variant'] in ('PORT_V1','PORT_V2') and type(m['port_faults'])is list,'PORT_FIXTURE_VARIANT')
        V.require(type(m['global_scope'])is str and m['global_scope'].startswith('ARTIFICIAL_ONLY/'),'FAKE_GLOBAL_SCOPE')
        equal(m['global_claim'],hashed(['ARTIFICIAL_ONLY',m['global_scope'],hashed(c)]),'GLOBAL_CLAIM')
        equal(m['run_id'],hashed(['RUN',m['global_scope'],hashed(c)]),'GLOBAL_RUN_ID')
        equal(m['scientific_input'],hashed(c['scientific_reference']['scientific_reference']['input']),'SCIENTIFIC_INPUT')
        required=set(CLOSED)|{'schema','candidate_binding','source_release_candidate','global_scope','global_claim','run_id','scientific_input','seed_namespace','profile_n','modes','tasks','planned','static_scientific_candidate_calls','call_limit','byte_limits','condition_faults','sync_failure','parent_fault','codec','observer','workers','start_method','endpoint_binding','port_variant','port_faults'}
        equal(set(m),required,'MANIFEST_FIELDS')
        return dict(tasks=tasks,planned=total,candidate_digest=hashed(c),manifest_digest=hashed(m),run_id=m['run_id'],historical_runtime_bytes_proven=False,local_runtime_bytes_proven=self.local_runtime,runtime_projection=self.projected,real_authority_verified=False)
