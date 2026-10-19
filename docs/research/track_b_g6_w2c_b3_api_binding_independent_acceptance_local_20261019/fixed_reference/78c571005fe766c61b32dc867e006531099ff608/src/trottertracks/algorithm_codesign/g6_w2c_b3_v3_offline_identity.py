"""Pinned v3 schemas and offline contract/runtime identity, standard library only.

This reads bytes and records; it never imports producer or science SOURCE, checks
an old server filesystem, opens an authority store, or grants execution.
"""
from copy import deepcopy
from pathlib import Path
import re
from . import g6_w2c_b3_offline_result as V

PUBLIC = '7ae4bd0b951dae05380538cd5c22a3ead5328ab2'
MEASURED = '860fef4a3a8ebb5a46628f4e0364e5934b5af416'
PREFIX = 'docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019/'
SCHEMA_DIR = PREFIX + 'output_schema/'
NAMES = ('contract','envelope','f01','pair','receipt','result','runtime')
ROLES = {n:'src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_'+n+'.py'
         for n in ('driver','observer','scheduler','admission','host')}
ROLES['serialization'] = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_exact_serialization.py'


def strict_equal(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(strict_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(strict_equal(x,y) for x,y in zip(a,b))
    return a==b


def validate(value,schema,registry,path='$'):
    """Evaluate every assertion keyword used by the seven pinned JSON schemas.

    Local urn refs only; no URL resolution, code execution or third-party runtime.
    This is a validator for this pinned vocabulary, not general JSON Schema.
    """
    if '$ref' in schema:
        V.require(schema['$ref'] in registry,'UNKNOWN_SCHEMA_REF')
        validate(value,registry[schema['$ref']],registry,path)
    if 'anyOf' in schema:
        matched=False
        for choice in schema['anyOf']:
            try:
                validate(value,choice,registry,path);matched=True;break
            except V.Invalid:
                pass
        V.require(matched,'SCHEMA_ANYOF:'+path)
    if 'type' in schema:
        names=schema['type'] if isinstance(schema['type'],list) else [schema['type']]
        kinds={'object':dict,'array':list,'integer':int,'boolean':bool,'string':str,'null':type(None)}
        V.require(any(type(value) is kinds[n] for n in names),'SCHEMA_TYPE:'+path)
    if 'const' in schema:
        V.require(strict_equal(value,schema['const']),'SCHEMA_CONST:'+path)
    if 'enum' in schema:
        V.require(any(strict_equal(value,v) for v in schema['enum']),'SCHEMA_ENUM:'+path)
    if 'minimum' in schema and type(value) in (int,float):
        V.require(value>=schema['minimum'],'SCHEMA_MINIMUM:'+path)
    if 'pattern' in schema and isinstance(value,str):
        V.require(re.search(schema['pattern'],value) is not None,'SCHEMA_PATTERN:'+path)
    if isinstance(value,dict):
        V.require(all(k in value for k in schema.get('required',[])),'SCHEMA_REQUIRED:'+path)
        properties=schema.get('properties',{})
        for k,v in value.items():
            if k in properties:
                validate(v,properties[k],registry,path+'.'+k)
            elif schema.get('additionalProperties') is False:
                raise V.Invalid('SCHEMA_ADDITIONAL_PROPERTY:'+path+'.'+k)
            elif isinstance(schema.get('additionalProperties'),dict):
                validate(v,schema['additionalProperties'],registry,path+'.'+k)
    if isinstance(value,list) and 'items' in schema:
        for i,v in enumerate(value):
            validate(v,schema['items'],registry,path+'['+str(i)+']')


class Reference:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.inventory=V.plain((self.root/'source_inventory.json').read_text())
        V.require(self.inventory['source_commit']==MEASURED,'MEASURED_SOURCE_COMMIT')
        expected={r['path']:r['sha256'] for r in self.inventory['measured_files']+
                  self.inventory['protected_runtime_and_artificial_dependencies']}
        # The public R1 source pins also include the two package init files.
        expected.update({'src/trottertracks/__init__.py':'2d51bfa08b2b1e000b86ec53bd463e786529307a7270078600e3b03995e90285',
                         'src/trottertracks/algorithm_codesign/__init__.py':'b6503716cfd51d52ca9038e6de5a411eb78380d30451f81df9b81c31b1be090f'})
        self.expected=expected
        self.schemas={};self.registry={}
        self.schema_pins={}
        for name in (*NAMES,'protocol'):
            rel=SCHEMA_DIR+('protocol.json' if name=='protocol' else name+'.schema.json')
            p=V.safe_path(self.root/'tree',rel)
            V.require(V.file_digest(p)==expected[rel],'PINNED_SCHEMA_BYTES:'+name)
            self.schema_pins[rel]=expected[rel]
            doc=V.plain(p.read_text());self.schemas[name]=doc
            if '$id' in doc:self.registry[doc['$id']]=doc
        protocol=self.schemas['protocol']
        V.require(protocol['schema']=='b3_parallel_wire_and_storage_v3','PROTOCOL_SCHEMA')
        from . import g6_w2c_b3_parallel_verifier_core as G
        V.require(protocol['state_codes']=={s:i for i,s in enumerate(G.STATES)} and
                  protocol['budgets']['result']==G.BUDGET['result_bytes'] and
                  protocol['budgets']['log']==G.BUDGET['log_bytes'],'PROTOCOL_CORE_MAPPING')

    def validate(self,name,value):
        validate(value,self.schemas[name],self.registry)

    def source_pins(self,pins):
        V.require(type(pins) is dict and pins,'SOURCE_PINS_REQUIRED')
        for rel,pin in pins.items():
            V.require(self.expected.get(rel)==pin,'SOURCE_PIN_REFERENCE:'+rel)
            V.require(V.file_digest(V.safe_path(self.root/'tree',rel))==pin,'SOURCE_BYTES:'+rel)

    def runtime(self,runtime,pins):
        self.validate('runtime',runtime)
        V.require(runtime['source_pins']==pins and runtime['source_roles']==ROLES and
                  runtime['schema_pins']==self.schema_pins,'RUNTIME_SOURCE_SCHEMA_ROLES')
        V.require(runtime['entrypoint']==dict(path=ROLES['host'],api='execute_approved_v3',sha256=pins[ROLES['host']]),'RUNTIME_ENTRYPOINT')
        V.require(set(runtime['import_locations'])==set(ROLES),'IMPORT_ROLE_COVERAGE')
        for n,rel in ROLES.items():
            V.require(runtime['import_locations'][n]==dict(module='trottertracks.algorithm_codesign.'+Path(rel).stem,path=rel,sha256=pins[rel]),'IMPORT_LOCATION_RECORD:'+n)
        V.require(Path(runtime['runtime_root']).is_absolute(),'RECORDED_RUNTIME_ROOT')
        py=runtime['python']
        V.require(V.is_hash(py['executable_sha256']) and py['version'] and py['cache_tag'],'PYTHON_IDENTITY_SHAPE')
        V.require(set(py['libraries'])=={'json','fractions','multiprocessing','multiprocessing.connection','sqlite3','_sqlite3','_json','_hashlib'},'PYTHON_LIBRARY_COVERAGE')
        V.require(all(V.is_hash(v['sha256']) and Path(v['path']).is_absolute() for v in py['libraries'].values()),'PYTHON_LIBRARY_IDENTITY')
        self.source_pins(pins)

    def contract(self,c):
        self.validate('contract',c)
        V.require(c['schema_pins']==self.schema_pins,'CONTRACT_SCHEMA_PINS')
        r1=c['r1_reference'];candidate=deepcopy(r1['candidate']);candidate['source_pins']=c['source_pins']
        V.require(r1['schema']=='b3_formal_admission_v2' and r1['content_adopted'] is True and
                  r1['contract_grants_execution'] is False and type(r1['retry']) is int and r1['retry']==0 and
                  r1['resume'] is False and r1['mandatory_STOP'] is True,'ARTIFICIAL_R1_REFERENCE_FLAGS')
        V.require(c['candidate']==candidate,'R1_SCIENTIFIC_POLICY_CHANGED')
        self.static_policy(candidate)
        V.require(c['scope']==r1['scope'] and c['claim_id']==r1['claim_id'],'R1_SCOPE_CLAIM_BINDING')
        V.require(all(c['source_pins'].get(p)==h for p,h in r1['source_pins'].items()) and
                  set(c['source_pins'])==set(r1['source_pins'])|set(ROLES.values()),'R1_SOURCE_PIN_COVERAGE')
        V.require(all(c['input_pins'].get(p)==h for p,h in r1['input_pins'].items()),'R1_INPUT_PIN_BINDING')
        V.require(all(c['input_pins'].get(p)==h for p,h in self.schema_pins.items()),'INPUT_SCHEMA_PINS')
        self.runtime(c['runtime_manifest'],c['source_pins'])
        fixture=c['runtime_manifest']['artificial_fixture']
        V.require(fixture==dict(path='tests/tracks/algorithm_codesign/b3_parallel_integrated_fixture.py',sha256=c['input_pins'].get('artificial-fixture.py')),'ARTIFICIAL_FIXTURE_PIN')
        V.require(V.file_digest(self.root/'tree'/fixture['path'])==fixture['sha256'],'ARTIFICIAL_FIXTURE_BYTES')
        s=c['settings'];V.require(s['codec'] in ('reference','fused') and s['mode'] in ('S','P1','P2'),'VARIANT')
        V.require(c['measurement_condition']==('sequential_pairs' if s['mode']=='S' else 'concurrent_pairs') and
                  c['output_schema']=='b3_parallel_result_v3' and c['observer']=='cached_all_call_return' and
                  c['transport']=='coalesced_reserve_enter','MEASUREMENT_SEMANTICS')
        # Local offline identity checks do not execute validate_r1 or read saved scientific values.
        return dict(formal_digest=V.identity(c),runtime_manifest_sha256=V.identity(c['runtime_manifest']),
                    source_pins_sha256=V.identity(c['source_pins']),driver_digest=V.identity(c['candidate']))

    @staticmethod
    def static_policy(c):
        """Compare frozen contract literals only; never load/evaluate saved science."""
        fixed={'id':'P5_general_order','p':['1/5','3/10','1/2'],'x':'5/7','m':5}
        policy={'root_bits':256,'probability_bits':160,'eta':'1/1000000000000',
                'rho':'1/1000000000000','epsilon_axis':'1/200','primitive_error':'1/1000000',
                'alpha_axis':'1/320','alpha_familywise_16_axes':'1/20','sigma':1}
        V.require(strict_equal(c['input'],fixed) and strict_equal(c['policy'],policy),'STATIC_INPUT_POLICY')
        ns=(1174526,894664,880312,1014322)
        cells=[dict(arm=a,layer=l,axis=x,N=n) for a,n in zip(V.ARMS,ns)
               for l in ('A','B') for x in ('X','Y')]
        V.require(strict_equal(c['cells'],cells) and c['passes']==['baseline','instrumented'] and
                  c['correctness_trials_per_cell']==16,'STATIC_CELLS')
        b=c['call_budget']
        for k,n in dict(correctness_sampling_calls=256,baseline_full_N_sampling_calls=15855296,
                        instrumented_full_N_sampling_calls=15855296,total_sampling_calls=31710848).items():
            V.require(type(b[k]) is int and b[k]==n,'STATIC_BUDGET:'+k)
        V.require(strict_equal(c['decisions']['angle'],dict(tau='1/1000000',K_phi=32,J=32,
                  mode='MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE')),'STATIC_A6')
        one=c['one_shot']
        V.require(one['bundles']==1 and type(one['automatic_retry']) is int and one['automatic_retry']==0 and
                  one['resume'] is False and one['old_claim_reuse'] is False and one['mandatory_STOP'] is True,
                  'STATIC_ONE_SHOT')
        V.require(c['angle_proposal']['physical_RZ_changed'] is False and
                  c['angle_proposal']['angle_refinement_or_search'] is False,'STATIC_NO_PHYSICAL_CHANGE')
