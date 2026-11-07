"""CT-1 finite saved q1/r1 pilot; prior SOURCE/results/STOP remain read-only."""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
from importlib.metadata import version
import hashlib,io,json,lzma,math,os,platform,subprocess,sys,tarfile,time
import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Operator,Statevector
from .track_c_h4 import common,save_json,ROOT,diagonal_primitives
from .track_c_compiler_a import compile_reference,PROFILE
from .track_c_compiler_a_18round import I,upper,lower
from . import pr_reference_bprime as bp
from . import pr_reference_transfer_qualification as transfer
from . import pr_reference_interference_pilot as pilot
from . import track_c_h4_synthesis as synth
from .pr_reference_ct0 import validate_record

BASE='f5f16fad64748236b0160284dc6c3017974b0132'
OUT=ROOT/'artifacts/pr_reference_ct1/2026-11-07'
PILOT='artifacts/pr_reference_interference_pilot/2026-11-06/'
BP='artifacts/pr_reference_bprime/2026-11-07/'
TR='artifacts/pr_reference_transfer_qualification/2026-11-06/'
CT='artifacts/track_c2_b_pilot/2026-11-06/'
INPUTS={}
def data(path):
    raw=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    INPUTS[path]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return raw

def read(path):
    raw=data(path);return json.loads(lzma.decompress(raw) if path.endswith('.xz') else raw)

def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        return {m.name:t.extractfile(m).read() for m in t if m.isfile()}

def instructions(q):
    return [[i.operation.name,[q.find_bit(z).index for z in i.qubits],list(map(float,i.operation.params))] for i in q.data]

def native_circuit(spec):
    q=QuantumCircuit(spec['n']);q.global_phase=spec.get('global_phase',0.)
    for name,wires,params in spec['lowering']:
        getattr(q,name)(*params,*wires)
    return q

def attach():
    if (OUT/'STOP.json').exists():raise RuntimeError('CT-1 mandatory STOP')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')


def sample_rows():
    rows=[json.loads(v) for v in archive(PILOT+'raw_prices.tar.xz').values()]
    return [dict(next(x for x in rows if x['q']==1 and x['j']==0 and x['rare']==rare),r=1) for rare in (False,True)]

def prepare_native(task):
    rare,arm,axis=task;attach();row=sample_rows()[int(rare)]
    if arm=='Bprime':
        body,events,changes=bp.build_saved(row);native=compile_reference(pilot.wrapper(body,axis))
    else:
        cfg,dist,events=bp.reconstructed(row);a=transfer.assemble(1,1,cfg,dist,events)
        assert a['phase']==row['reference_phase']
        native=compile_reference(pilot.wrapper(pilot.reference_core(a['F'],a['G'],a['phase']),axis))
        full=QuantumCircuit(17,1);full.reset(range(8,16));full.compose(native,inplace=True);native=full
    name=f'rare{int(rare)}_{arm}_{axis}'
    with (OUT/'native'/f'{name}.qpy').open('wb') as f:qpy.dump(native,f)
    if arm=='reference':expected=row['values']['reference'][axis=='Y']
    else:
        saved=[json.loads(v) for v in archive(BP+'paired_Bprime_prices.tar.xz').values()]
        v=next(x for x in saved if x['q']==1 and x['j']==0 and x['rare']==rare)
        expected=v['Bprime_values'][axis=='Y']
    actual=pilot.metric(native);assert actual==expected,(name,actual,expected)
    return dict(name=name,new_compile=1,ops=dict(native.count_ops()),metric=actual,saved_metric=expected,passed=True)

def records_at(exponent):
    records={};excluded=[]
    raw=data(CT+'raw_synthesis.tar.xz')
    bundles=read(CT+'BUNDLES.json')
    assert hashlib.sha256(raw).hexdigest()==next(x['sha256'] for x in bundles if x['path']=='raw_synthesis.tar.xz')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz') as t:
        for m in t:
            v=json.load(t.extractfile(m))
            if v['precision_exponent']!=exponent:continue
            validate_record(v,m.name)
            if not v.get('guard_method'):excluded.append(v['key']);continue
            records[v['theta_hex']]=v
    return records,excluded


def local_specs():
    old=[json.loads(v) for v in archive(TR+'local_catalogue_boundary.tar.xz').values()]
    new=[json.loads(v) for v in archive(BP+'Bprime_local_catalogue.tar.xz').values()]
    c={a:{r['name']:r for r in batch['records']} for a,batch in [
        ('reference',next(x for x in old if x['request']==['reference',True,True,True])),
        ('Bprime',next(x for x in new if x['request']==[True,True]))]}
    result=[]
    def add(name,role,rec,arm='both'):
        result.append(dict(name=name,role=role,arm=arm,n=rec['n'],lowering=rec['lowering'],global_phase=0.,origin='saved local native catalogue'))
    ref=c['reference'];b=c['Bprime']
    gaussian=next(r for r in ref.values() if r['meta'].get('family')=='xx_plus_yy' and r['meta']['params'][0]!=0.)
    add('Gaussian_primitive','D_Gaussian_primitive',gaussian)
    for name in ('basis:D0','basis_inverse:D0'):
        add(name,'D_frame_actual_source_inverse' if 'inverse' in name else 'D_frame',ref[name])
    # A physical diagonal RZZ, plus controlled D half-angle/scalar, without
    # duplicating entire deterministic sweeps already in the selected wholes.
    for family in ('rzz','crz'):
        rec=next(r for r in (ref if family=='rzz' else b).values() if r['meta'].get('family')==family and r['meta']['params'][0]!=0.)
        add('D_'+family,'D_'+family,rec,'reference' if family=='rzz' else 'Bprime')
    constants=read(BP+'V2_INPUTS_CONSTANTS.json')
    for order in (0,2):
        angle=float.fromhex(next(x['actual_hex'] for x in constants['angles'] if x['r']==1 and x['order']==order))
        for zz in (False,True):
            name=('bprime:' if zz else 'crz:')+angle.hex()
            # Both signs are already in the saved catalogue. Positive one is
            # the minimal controlled case; its actual inverse is tested later.
            add(f'RTE_control_{order}_{int(zz)}',f'RTE_order{order}_'+('ZZ' if zz else 'Z'),b[name],'Bprime')
            ops=[['cx',[0,1],[]],['rz',[1],[angle]],['cx',[0,1],[]]] if zz else [['rz',[0],[angle]]]
            add(f'RTE_reference_{order}_{int(zz)}',f'RTE_order{order}_'+('ZZ' if zz else 'Z'),dict(n=2 if zz else 1,lowering=ops),'reference')
    for name in ('cswap','h','sdg'):
        add(name,'CSWAP' if name=='cswap' else 'XY_wrapper',ref[name],'reference' if name=='cswap' else 'both')
    # Numeric scalar/event/vacuum P are read from the actual saved source
    # identities and native passthrough rule; fixed in stage_a below.
    return result

def stage_a():
    if (OUT/'PREDECLARED.json').exists():raise RuntimeError('Already predeclared; no resampling or target expansion')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'native').mkdir(exist_ok=True)
    raw=lzma.decompress(data(PILOT+'common.pkl.xz'));(OUT/'common.pkl').write_bytes(raw);attach()
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        executable=sys.executable,packages={p:version(p) for p in ('qiskit','pygridsynth','numpy','scipy','mpmath','numba')},
        affinity=len(os.sched_getaffinity(0)),meminfo=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        io_pressure=Path('/proc/pressure/io').read_text(),profile=PROFILE,shared_settings_changed=False))
    samples=sample_rows();saved=archive(PILOT+'native_circuits.tar.xz');checks=[]
    for row in samples:
        name=f'rare{int(row["rare"])}_reference_X';payload=saved[f'circuits/q1_rare{int(row["rare"])}_reference.qpy']
        assert hashlib.sha256(payload).hexdigest()==row['representative_native_qpy_sha256']['reference']
        (OUT/'native'/f'{name}.qpy').write_bytes(payload)
        native=qpy.load(io.BytesIO(payload))[0];assert pilot.metric(native)==row['values']['reference'][0]
        checks.append(dict(name=name,new_compile=0,metric=pilot.metric(native),saved_QPY_reused=True,passed=True))
    tasks=[(rare,arm,axis) for rare in (False,True) for arm in ('Bprime','reference') for axis in 'XY' if arm=='Bprime' or axis=='Y']
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        checks.extend(pool.map(prepare_native,tasks))
    save_json(OUT/'NATIVE_INPUT_CHECKS.json',dict(checks=checks,new_native_compiles=6,old_reference_QPY_reused=2,passed=True))
    specs=local_specs()
    scalar=-.02*(common()['preps'][3].constant_coefficient+common()['preps'][3].extracted_identity_coefficient)
    dscalar=2*float(diagonal_primitives(common()['preps'][3].deterministic_blocks[0],.01).global_phase)
    for name,angle,role in [('physical_scalar',scalar,'scalar_P'),('D_scalar',dscalar,'D_scalar_P')]:
        if angle is not None:specs.append(dict(name=name,role=role,arm='Bprime',n=1,global_phase=0.,lowering=[['rz',[0],[angle]]],origin='same scalar P -> native RZ passthrough'))
    for row in samples:
        phase=0.
        for ev in row['event_records']:
            if ev['taylor_order']==2:phase+=math.pi
        for ev in row['event_records']:
            for app in ev['application_sequence']:
                if app['role']=='product' and app['coefficient_sign']<0:phase+=math.pi
        for role,angle,arm in [('event_P',phase,'Bprime'),('vacuum_P',row['reference_phase'],'reference')]:
            if angle!=0:specs.append(dict(name=role+'_rare'+str(int(row['rare'])),role=role,arm=arm,n=1,global_phase=0.,lowering=[['rz',[0],[angle]]],origin='saved trajectory actual binary phase'))
    save_json(OUT/'LOCAL_TARGETS.json',dict(records=specs))
    roles=defaultdict(set);used=defaultdict(Counter)
    for path in sorted((OUT/'native').glob('*.qpy')):
        q=qpy.load(io.BytesIO(path.read_bytes()))[0]
        for i in q.data:
            if i.operation.name=='rz':
                key=abs(float(i.operation.params[0])).hex();roles[key].add('whole:'+path.stem);used[path.stem][float(i.operation.params[0]).hex()]+=1
    for spec in specs:
        for name,wires,params in spec['lowering']:
            if name=='rz':roles[abs(float(params[0])).hex()].add('local:'+spec['name'])
    prior=read('artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json')['rows'][0]
    endpoints={arm:{axis:prior['arms'][arm]['axis'][axis] for axis in 'XY'} for arm in ('Bprime','reference')}
    sufficient=min(F(x['uniform_gate_error_strict_upper']) for a in endpoints.values() for x in a.values())
    # Registered engine/catalogue specifications only; no precision grid.
    offered=tuple(range(-13,-5));eligible=[e for e in offered if F(10)**e<=sufficient]
    if not eligible:raise RuntimeError('STOP: no registered precision specification meets all four q1 conditions')
    exponent=max(eligible);records,excluded=records_at(exponent)
    reused={k:records[k] for k in roles if k in records};new=sorted(set(roles)-set(reused))
    (OUT/'reused').mkdir(exist_ok=True)
    for k,v in reused.items():save_json(OUT/'reused'/f'{v["key"]}.json',v)
    pre=dict(base=BASE,q=1,r=1,axes=['X','Y'],trajectory_selection='saved j0 in each stratum; fixed before any CT price',
        trajectories=[dict(q=1,r=1,rare=x['rare'],j=0,seed=x['seed'],event_sha256=hashlib.sha256(json.dumps(x['event_records'],sort_keys=True).encode()).hexdigest(),
            events=[dict(order=e['taylor_order'],rotation=e['rotation_component_id'],products=e['product_component_ids']) for e in x['event_records']],
            vacuum_phase_hex=x['reference_phase'].hex()) for x in samples],
        exponent=exponent,epsilon=str(F(10)**exponent),precision_choice='coarsest registered specification satisfying all four CT0 q1 worst-case / saved-N conditions',
        CT0_sufficient_gate_error_upper=str(sufficient),CT0_q1_endpoints=prior,
        all_positive_keys=sorted(roles),roles={k:sorted(v) for k,v in roles.items()},
        existing_qualified_keys=sorted(reused),new_synthesis_keys=new,excluded_legacy_record_keys=excluded,
        actual_signed_occurrences={k:dict(v) for k,v in used.items()},
        whole_dependency_closure='every key required by fixed eight whole circuits; no optional duplicate key additions',
        local_targets=[x['name'] for x in specs],planned_tests=['strict positive/negative word and exact adjoint','written product order and W','full-space local native vs CT',
        'finite whole ground and deterministic full-space state, X/Y','telescoping/raw disk/old integer N'],
        phase_fit=False,all_support_price=False,all18_T=False,precision_optimization=False,
        numerical_check_tolerance=1e-10,
        simulation='same word as an exact-ring 2x2 matrix per native RZ; not new physical gate synthesis or fusion')
    save_json(OUT/'PREDECLARED.json',pre)
    save_json(OUT/'INPUTS.json',dict(base=BASE,inputs=INPUTS,common_pickle_sha256=hashlib.sha256(raw).hexdigest(),new_SCF_DF_ground_sampling=0))
    (OUT/'STAGE_A_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(stage='A',whole_circuits=8,new_compiles=6,exponent=exponent,keys=len(roles),reused=len(reused),new=len(new)),flush=True))
    return pre
