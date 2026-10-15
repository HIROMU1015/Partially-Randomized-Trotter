"""H6 V2 static inspection / future separately authorized theory conversion.

Default --plan is stdlib only. --inspect-archive authenticates saved descriptions
and reports hashes, without producing primitives or evaluating a gate count.
The numerical mode requires a new source/input/profile/law/receipt-bound grant.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


a=load('_v2_cost_adapter',ROOT/'src/trotterlib/native_df_h6_cost_adapter.py')
archive_api=load('_v2_description_graph',ROOT/'src/trotterlib/native_df_archive_metadata.py')
schema=load('_v2_connection_schema',ROOT/'src/trotterlib/native_df_metadata_binding.py')
model=load('_v2_gate_model',ROOT/'src/trotterlib/native_df_gate_model.py')
inspection=load('_v2_saved_inspection',ROOT/'scripts/inspect_native_df_preparation_archive.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(),object_pairs_hook=archive_api._pairs,
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def static_archive(path):
    checked=inspection.inspect_archive(path,a.ARCHIVE_SHA256)
    graph=archive_api.FrozenMetadataGraph.from_file(path,expected_sha256=a.ARCHIVE_SHA256)
    descriptions={key:a.preparation_description(graph,key) for key in graph.archive['codec']['roots']}
    return dict(schema='track_a_v2_static_adapter_receipt_v1',archive_sha256=a.ARCHIVE_SHA256,
        roots={k:dict(description_sha256=a.digest(v),blocks=len(v['blocks']),
                      complete_components=len(v['components']),
                      preparation_hash=v['preparation_hash'],primitive_lists_evaluated=False,
                      probabilities_evaluated=False,costs_evaluated=False) for k,v in descriptions.items()},
        existing_inspection_status=checked['status'],native_build=0,real_theory_calls=0,
        scientific_imports=0,runtime_authorization=False)


def verify_grant(config,grant,*,config_sha256,source,fresh):
    if (grant.get('schema')!='track_a_v2_h6_theory_grant_v1' or
        grant.get('approved_by_user') is not True or grant.get('runtime_authorization') is not True or
        grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False or
        grant.get('SOURCE')!=source or grant.get('config_sha256')!=config_sha256 or
        grant.get('plan_sha256')!=a.digest(a.plan()) or grant.get('resources')!=a.plan()['resources'] or
        grant.get('law_policy')!='EXACT_RENORMALIZATION_OF_SAVED_BINARY64_WEIGHTS' or
        grant.get('scope') not in ('metadata_primitives','theory') or
        (grant.get('scope')=='theory' and config.get('binding_gate_complete') is not True)):
        raise ValueError('NEW_H6_V2_SEALED_GRANT_AND_COMPLETE_BINDING_REQUIRED')
    cpus=grant.get('allowed_cpus')
    if (not isinstance(cpus,list) or len(cpus)!=1 or type(cpus[0]) is not int or cpus[0] not in os.sched_getaffinity(0)):
        raise ValueError('EXPLICIT_ONE_CPU_ASSIGNMENT_REQUIRED')
    if (fresh.get('source')!=source or fresh.get('config_sha256')!=config_sha256 or fresh.get('allowed_cpus')!=cpus or
        fresh.get('resources_idle_verified') is not True or fresh.get('profile_verified') is not True or
        fresh.get('protected_jobs_unchanged') is not True or fresh.get('external_RSS_monitor_ready') is not True or
        type(fresh.get('monotonic_seconds')) not in (int,float) or
        not 0<=time.monotonic()-fresh['monotonic_seconds']<=30):
        raise ValueError('FRESH_RESOURCE_PROFILE_GATE_REQUIRED')
    if config.get('plan_sha256')!=a.digest(a.plan()):raise ValueError('CONFIG_PLAN_CHANGED')


class Permit:
    law_policy='EXACT_RENORMALIZATION_OF_SAVED_BINARY64_WEIGHTS'
    def __init__(self,config,grant,output):
        self.config,self.grant,self.output=config,grant,output
        self.started=time.monotonic();self.used={}
        self.primitive_cache={};self.occurrence_cache={}

    def require(self,scope):
        if scope!='h6_v2_numeric':raise ValueError('SCOPE')
        self.guard()

    def guard(self):
        ceiling=self.grant['resources']
        if time.monotonic()-self.started>ceiling['wall_seconds']:raise ValueError('WALL_CEILING')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>ceiling['RSS_bytes']:raise ValueError('RSS_CEILING')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())>ceiling['output_bytes']:
            raise ValueError('OUTPUT_CEILING')

    def take(self,name):
        self.guard()
        caps=dict(timed_primitive_records=2048,model_candidate_axis_rows=416,
                  complete_component_law_conversion=416,expected_occurrence_evaluations=144)
        if name not in caps or self.used.get(name,0)>=caps[name]:raise ValueError('CALL_CAP:'+name)
        self.used[name]=self.used.get(name,0)+1
        with (self.output/'reservations.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=name,reserved=self.used[name]))+'\n');f.flush();os.fsync(f.fileno())

    def verify_producer(self,producer):
        expected=Path(self.config['provider_root'])/'src/trotterlib/df_trotter/ops.py'
        if Path(producer.__file__).resolve()!=expected.resolve() or sha(expected)!=self.config['provider_hashes']['src/trotterlib/df_trotter/ops.py']:
            raise ValueError('PINNED_ORIGINAL_PRIMITIVE_PRODUCER_REQUIRED')

    def order_law(self,candidate,description):
        # A separate source-bound receipt supplies tau/B/component semantics.
        receipt=self.config['order_law_receipts'][candidate['id']]
        lam=float.fromhex(description['lambda_r']['float_hex'])
        tau=lam*.8/candidate['q']/candidate['r']
        if (tau.hex()!=receipt['tau_hex'] or candidate['R']!=candidate['q']*candidate['r'] or
            receipt['K']!=candidate['K'] or receipt['component_law_source']!='complete_saved_catalog' or
            receipt['probability_policy']!=self.law_policy or receipt['native_law_binding_verified'] is not True):
            raise ValueError('TAU_NORMALIZATION_COMPONENT_LAW_BINDING')
        orders={int(k):Fraction(v[0],v[1]) for k,v in receipt['orders'].items()}
        if sum(orders.values())!=1:raise ValueError('ORDER_PROBABILITY_MASS')
        # No canonical-law recomputation or inferred missing orders here.
        return tau,orders

    def write(self,name,value,*,terminal=False):
        if not terminal:self.guard()
        def encode(v):
            if isinstance(v,Fraction):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
            if hasattr(v,'as_dict'):return v.as_dict()
            if hasattr(v,'__dataclass_fields__'):return dict(vars(v))
            raise TypeError(type(v).__name__)
        if Path(name).name!=name:raise ValueError('OUTPUT_PATH')
        with (self.output/name).open('x') as f:json.dump(value,f,indent=2,default=encode,allow_nan=False);f.write('\n')


def execute(config_path,grant_path,fresh_path):
    config,grant,fresh=map(read,(config_path,grant_path,fresh_path))
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    verify_grant(config,grant,config_sha256=sha(config_path),source=source,fresh=fresh)
    for root,hashes,fixed in ((ROOT,config['source_hashes'],source),
        (Path(config['provider_root']),config['provider_hashes'],a.PROVIDER)):
        for name,expected in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('SOURCE_PATH')
            fixed_bytes=subprocess.check_output(['git','show',fixed+':'+name],cwd=root)
            if sha(root/name)!=expected or hashlib.sha256(fixed_bytes).hexdigest()!=expected:
                raise ValueError('SOURCE_BYTES_CHANGED:'+name)
    for r in config['files']:
        p=Path(r['path'])
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('INPUT_BYTES')
    for name,expected in config['runtime_file_hashes'].items():
        if sha(name)!=expected:raise ValueError('RUNTIME_PROFILE')
    # Independently fixed expected envelopes, referenced receipts, paired-row
    # provenance and full variant binding must precede every numerical call.
    for item in config['connection_envelopes'] if grant['scope']=='theory' else []:
        expected=read(item['expected_path']);record=read(item['record_path'])
        receipts=item['receipt_paths']
        a.validate_full_binding(record,expected,schema=schema,receipt_loader=lambda f:Path(receipts[f]).read_bytes())
    for item in config['compiled_pairs'] if grant['scope']=='theory' else []:
        a.paired_compiled_rows(item['rows'],expected=item['expected'])
    if grant['scope']=='theory' and (not config['connection_envelopes'] or not config['compiled_pairs']):
        raise ValueError('EMPTY_BINDING_SET')
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('THREAD_ONE_ENV_REQUIRED')
    os.sched_setaffinity(0,set(grant['allowed_cpus']))
    output=Path(config['output']);lock=Path(config['one_shot_lock'])
    for p in (output,lock):
        if not p.is_absolute() or p.resolve()!=p or not p.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('EXCLUSIVE_PRIVATE_PATH')
    if output.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    with lock.open('x') as f:f.write(source+'\n')
    output.mkdir();permit=Permit(config,grant,output)
    scratch=output/'scratch';scratch.mkdir();mpl=scratch/'matplotlib';mpl.mkdir()
    font=config['font_seed']
    if sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
    (mpl/font['name']).write_bytes(Path(font['path']).read_bytes())
    os.environ['MPLCONFIGDIR']=str(mpl);os.environ['NUMBA_CACHE_DIR']=str(scratch/'numba')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    ceilings=grant['resources']
    resource.setrlimit(resource.RLIMIT_AS,(ceilings['AS_bytes'],ceilings['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(ceilings['CPU_seconds'],ceilings['CPU_seconds']))
    def timeout(*args):raise ValueError('WALL_CEILING')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(ceilings['wall_seconds'])
    try:
        graph=archive_api.FrozenMetadataGraph.from_file(config['archive_path'],expected_sha256=a.ARCHIVE_SHA256)
        descriptions={k:a.preparation_description(graph,k) for k in graph.archive['codec']['roots']}
        sys.path.insert(0,str(Path(config['provider_root'])/'src'))
        producer=importlib.import_module('trotterlib.df_trotter.ops')
        permit.verify_producer(producer)
        if grant['scope']=='metadata_primitives':
            if config['expected_preparation_hashes']!={key:value['preparation_hash'] for key,value in descriptions.items()}:
                raise ValueError('STATIC_PREPARATION_IDENTITIES_CHANGED')
            requests=set()
            for item in config['metadata_tasks']:
                c=item['cell'];description=descriptions[item['archive_root']]
                for stage in model.pf_schedule(len(description['blocks']),.8,c['q'],c['formula'],'symmetric_directional'):
                    requests.add((item['archive_root'],stage.block,float(stage.time).hex()))
            if len(requests)>2048:raise ValueError('TIMED_PRIMITIVE_PLAN_EXCEEDS_CAP')
            records=[]
            for key,index,time_hex in sorted(requests):
                permit.take('timed_primitive_records');b=descriptions[key]['blocks'][index]
                coefficients=[float.fromhex(v['float_hex']) for v in b['diagonal_coefficients']]
                timed=(producer.one_body_diagonal_primitives(coefficients,float.fromhex(time_hex))
                    if b['kind']=='DFDeterministicOneBodySpec' else producer.df_squared_diagonal_primitives(
                        coefficients,float.fromhex(b['lam']['float_hex']),float.fromhex(time_hex)))
                records.append(dict(archive_root=key,block=index,time_hex=time_hex,
                    basis_hash=b['basis']['basis_hash'],coefficient_node_sha256=b['original_node_sha256'],
                    rz=list(timed.rz),rzz=list(timed.rzz),global_phase=float(timed.global_phase)))
            permit.write('timed_primitive_receipts.json',dict(archive_sha256=a.ARCHIVE_SHA256,
                original_producer_SOURCE=a.PROVIDER,records=records,theoretical_cost_evaluated=False))
            permit.write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,
                SOURCE=source,scope='metadata_primitives',calls=permit.used,new_native_build=0,
                new_compile=0,real_theory_calls=0,next_stage_authorized=False),terminal=True)
            return
        for item in config['evaluation_tasks']:
            permit.guard();c=item['cell'];description=descriptions[item['archive_root']]
            if item['candidate_binding_verified'] is not True:raise ValueError('UNRESOLVED_CANDIDATE_VARIANT')
            for variant in ('A','B'):
                for axis in ('cosine','sine'):
                    cost=a.evaluate_candidate(description,c,model=model,schema=schema,producer=producer,
                        permit=permit,variant=variant,axis=axis,scalar_phase=item['scalar_phase'])
                    permit.write(c['id']+'_'+variant+'_'+axis+'.json',dict(cell=c,model=variant,axis=axis,
                        cost=cost,phase_receipt_sha256=item['phase_receipt_sha256'],
                        literal_native_policy='A_prefix_B_tail',FT_T_count=None,Toffoli_count=None))
        permit.write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,
            SOURCE=source,scope='theory',calls=permit.used,new_compile=0,new_native_build=0,next_stage_authorized=False),terminal=True)
    except BaseException as exc:
        # Failure receipts are retained; no lock refund or automatic retry.
        permit.write('terminal_STOP.json',dict(status='STOP',reason=str(exc),calls=permit.used,retry=False),terminal=True)
        raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group();group.add_argument('--plan',action='store_true')
    group.add_argument('--inspect-archive');group.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name)
    args=p.parse_args()
    if args.execute_approved:
        if not all((args.config,args.grant,args.fresh_gate)):p.error('all sealed inputs required')
        execute(args.config,args.grant,args.fresh_gate)
    elif args.inspect_archive:
        resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))
        result=static_archive(args.inspect_archive)
        if {'numpy','scipy','qiskit','pyscf','cupy','trottertracks'} & set(sys.modules):raise RuntimeError('SCIENTIFIC_IMPORT')
        print(json.dumps(result,indent=2))
    else:
        if any((args.config,args.grant,args.fresh_gate)):p.error('real input arguments require --execute-approved')
        print(json.dumps(a.plan(),indent=2))


if __name__=='__main__':main()
