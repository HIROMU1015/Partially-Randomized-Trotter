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
    print(json.dumps(dict(stage='A',whole_circuits=8,new_compiles=6,exponent=exponent,keys=len(roles),reused=len(reused),new=len(new))),flush=True)
    return pre


def predeclared():
    attach();return json.loads((OUT/'PREDECLARED.json').read_text())

def acquire_one(angle):
    pre=predeclared();assert angle in pre['new_synthesis_keys']
    synth.OUT=OUT
    v=synth.acquire(float.fromhex(angle),pre['exponent'])
    validate_record(v);assert v.get('guard_method') and v['theta_hex']==angle
    return dict(angle=angle,key=v['key'],T=v['T'],guard=v['strict_error_upper'],status='PASS')

def stage_b():
    pre=predeclared()
    if (OUT/'KEY_REUSE_AND_SYNTHESIS.json').exists():raise RuntimeError('Stage B already complete')
    tasks=pre['new_synthesis_keys'];workers=min(len(tasks),len(os.sched_getaffinity(0)));result=[]
    # Bound only the number of outstanding jobs by actual CPU affinity.
    # A scientific failure stops new submissions; preserve all in-flight output.
    todo=iter(tasks);pool=ProcessPoolExecutor(max_workers=workers);running={}
    try:
        for _ in range(workers):
            angle=next(todo,None)
            if angle is not None:running[pool.submit(acquire_one,angle)]=angle
        while running:
            done=next(as_completed(running));angle=running.pop(done);result.append(done.result())
            if len(result)%20==0:print('strict synthesis completed',len(result),'/',len(tasks),flush=True)
            angle=next(todo,None)
            if angle is not None:running[pool.submit(acquire_one,angle)]=angle
    except BaseException as exc:
        for f in running:f.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='strict acquisition failure',error=repr(exc),completed=len(result),automatic_precision_change=False))
        raise
    else:pool.shutdown(wait=True)
    records=selected_records();assert set(records)==set(pre['all_positive_keys'])
    inventory=[]
    for angle,v in sorted(records.items()):
        inventory.append(dict(angle=angle,key=v['key'],exponent=v['precision_exponent'],
            source='reused' if angle in pre['existing_qualified_keys'] else 'new',
            spec=v['spec'],sequence_sha256=v['sequence_sha256'],strict_error_upper=v['strict_error_upper'],
            T=v['T'],Tdagger=v['Tdagger'],global_W=v['global_W'],guard_method=v['guard_method'],route=v['route'],
            roles=pre['roles'][angle],positive_word_negative_actual_adjoint=True))
    save_json(OUT/'KEY_REUSE_AND_SYNTHESIS.json',dict(status='PASS',exponent=pre['exponent'],epsilon=pre['epsilon'],
        reused=len(pre['existing_qualified_keys']),new=len(tasks),workers=workers,records=inventory,
        unregistered_synthesis_keys=0,old_cache_modified=False,global_phase_fit=False))
    print('Stage B PASS',len(records),'words',flush=True)


def selected_records():
    pre=predeclared();records={}
    for folder in ('reused','synthesis'):
        for path in (OUT/folder).glob('*.json'):
            assert not path.name.endswith('.failed.json'),path
            v=json.loads(path.read_text());validate_record(v)
            assert v.get('guard_method') and v['precision_exponent']==pre['exponent']
            angle=v['theta_hex'];assert angle in pre['all_positive_keys'] and angle not in records
            records[angle]=v
    return records


@lru_cache(None)
def word_matrices():
    pre=predeclared();records=selected_records()
    from .track_c2_b_verify import lower_native
    synth.OUT=OUT
    from pygridsynth.domega_unitary import DOmegaUnitary as Exact
    inverse={'H':'H','T':'t','t':'T','S':'SSS','X':'X','W':'WWWWWWW'}
    mp.iv.dps=100;mp.mp.dps=100
    matrices={};checks=[]
    for angle,v in sorted(records.items()):
        theta=float.fromhex(angle);word=v['sequence']
        guard=synth.strict_guard(word,theta);assert 0<=guard<=F(pre['epsilon'])
        adj=''.join(inverse[g] for g in reversed(word))
        assert Exact.from_gates((word+adj).replace('t','TTTTTTT'))==Exact.identity()
        im=synth.exact_word_matrix(word)
        u=np.array([[complex(float(z.real.mid),float(z.imag.mid)) for z in row] for row in im])
        native=np.diag(np.exp(np.array([-.5j,.5j])*theta))
        observed=float(np.linalg.norm(u-native,2));allocation=[dict(role='one',angle=angle,exponent=pre['exponent'])]
        emitter=[]
        for sign in (1,-1):
            q=QuantumCircuit(1);q.rz(sign*theta,0)
            emitted=lower_native(q,allocation,{(angle,pre['exponent']):v},'one')
            actual=Operator(emitted).data;expected=u if sign==1 else u.conj().T
            mismatch=float(np.linalg.norm(actual-expected,2))
            assert mismatch<=pre['numerical_check_tolerance']
            emitter.append(dict(sign=sign,exact_word_vs_emitter_operator_error=mismatch,
                emitted_T=int(emitted.count_ops().get('t',0)+emitted.count_ops().get('tdg',0)),
                global_phase_float=float(emitted.global_phase)))
            assert emitter[-1]['emitted_T']==v['T']
        assert observed<=float(guard)+pre['numerical_check_tolerance']
        matrices[angle]=u
        checks.append(dict(angle=angle,key=v['key'],strict_recomputed_Frobenius_upper=str(guard),
            observed_spectral_error=observed,exact_adjoint_identity=True,emitter=emitter,passed=True))
    save_json(OUT/'WORD_CORRECTNESS.json',dict(status='PASS',records=checks,guard_dps=100,
        exact_algebra='cyclotomic ring; actual adjoint including W; written product reversed by existing emitter',
        mathematical_guard='strict exact logical word including W; no phase fit',
        observed_operator_numbers='binary64 diagnostics; tolerance is not an operator certificate',
        emitted_global_phase='Qiskit float representation is compared separately; exact W scalar retained in word semantics',passed=True))
    return matrices


def matrices_for(spec,ct):
    records=selected_records() if ct else None
    wm=word_matrices() if ct else None
    codes=[];q0=[];q1=[];mats=[];bound=F();T=Td=0;counts=Counter();W=0
    for name,qs,params in spec['lowering']:
        if name in ('measure','reset','barrier'):continue
        if name=='cx':
            codes.append(1);q0.append(qs[0]);q1.append(qs[1]);mats.append(np.eye(2));continue
        codes.append(0);q0.append(qs[0]);q1.append(0)
        if name=='rz':
            theta=float(params[0]);key=abs(theta).hex();counts[theta.hex()]+=1
            if ct:
                v=records[key];u=wm[key]
                if theta<0:u=u.conj().T
                bound+=F(v['strict_error_upper']);T+=v['T'];Td+=v['Tdagger'] if theta>=0 else v['T']-v['Tdagger'];W+=(1 if theta>=0 else -1)*v['global_W']
            else:u=np.diag(np.exp(np.array([-.5j,.5j])*theta))
        elif name=='sx':u=np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
        elif name=='x':u=np.array([[0,1],[1,0]],complex)
        else:raise ValueError('unqualified native op '+name)
        mats.append(u)
    return (np.asarray(codes,np.int64),np.asarray(q0,np.int64),np.asarray(q1,np.int64),np.asarray(mats,np.complex128)),dict(
        strict_joint_operator_upper=str(bound),T=T,Tdagger=Td,global_W_signed=W,
        exact_native_global_phase=spec.get('global_phase',0.),signed_RZ_occurrences=dict(counts))


# No suitable existing matrix-action kernel was found in the focused SOURCE
# search. This small gate-by-gate evaluator avoids millions of repeated H/T
# evaluations on 17 qubits; it changes no physical word or circuit schedule.
from numba import njit
@njit(cache=True)
def apply_batch(states,codes,q0,q1,mats):
    v=states.copy();dim=v.shape[0];cols=v.shape[1]
    for k in range(len(codes)):
        a=1<<q0[k]
        if codes[k]==1:
            b=1<<q1[k]
            for i in range(dim):
                if (i&a)!=0 and (i&b)==0:
                    j=i|b
                    for c in range(cols):v[i,c],v[j,c]=v[j,c],v[i,c]
        else:
            for base in range(0,dim,2*a):
                for s in range(a):
                    i=base+s;j=i+a
                    for c in range(cols):
                        x=v[i,c];y=v[j,c]
                        v[i,c]=mats[k,0,0]*x+mats[k,0,1]*y
                        v[j,c]=mats[k,1,0]*x+mats[k,1,1]*y
    return v


def evaluate(spec,states,ct):
    args,account=matrices_for(spec,ct)
    return np.exp(1j*spec.get('global_phase',0.))*apply_batch(states,*args),account


def simulator_check():
    # Every supported gate and both CX directions, both phase and wire order.
    q=QuantumCircuit(3);q.rz(.37,0);q.sx(2);q.cx(2,0);q.x(1);q.cx(0,2);q.global_phase=.23
    spec=dict(n=3,lowering=instructions(q),global_phase=float(q.global_phase))
    observed,_=evaluate(spec,np.eye(8,dtype=complex),False)
    error=float(np.linalg.norm(observed-Operator(q).data,2));assert error<1e-12
    save_json(OUT/'TECHNICAL_TESTS.json',dict(passed=True,full_space_three_qubit_simulator_vs_Qiskit=error,
        supported_gates=['RZ','SX','X','CX'],both_CX_directions=True,nonzero_global_phase=True,
        whole_dense_unitary=False,new_physical_conditions=False))


def local_one(spec):
    pre=predeclared();n=spec['n'];dim=1<<n
    args,acc=matrices_for(spec,True)
    a,_=evaluate(spec,np.eye(dim,dtype=complex),False)
    c=apply_batch(np.eye(dim,dtype=complex),*args)*np.exp(1j*spec.get('global_phase',0.))
    # One SVD per <=256-dimensional local full space. No phase alignment.
    with __import__('threadpoolctl').threadpool_limits(limits=max(1,len(os.sched_getaffinity(0))//len(pre['local_targets']))):
        error=float(np.linalg.norm(c-a,2))
    bound=F(acc['strict_joint_operator_upper']);assert error<=float(bound)+pre['numerical_check_tolerance'],spec['name']
    # Native return circuits and synthesized inverse are different notions.
    # Inverting the synthesized local operator preserves its exact error bound.
    inverse_error=float(np.linalg.norm(c.conj().T-a.conj().T,'fro'))
    return dict(name=spec['name'],role=spec['role'],arm=spec['arm'],qubits=n,
        full_space=True,phase_fit=False,observed_spectral_error=error,
        actual_adjoint_observed_Frobenius_error=inverse_error,account=acc,passed=True)


def whole_spec(path):
    q=qpy.load(io.BytesIO(path.read_bytes()))[0]
    # Reset is only the identical initial reference register preparation.
    reset_positions=[j for j,i in enumerate(q.data) if i.operation.name=='reset']
    assert not reset_positions or reset_positions==list(range(8))
    assert [i.operation.name for i in q.data].count('measure')==1
    return dict(name=path.stem,n=q.num_qubits,global_phase=float(q.global_phase),lowering=instructions(q),
        initial_reference_resets=len(reset_positions),native_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def whole_one(spec):
    attach();n=spec['n'];dim=1<<n;psi=pilot.ground()
    states=np.empty((dim,2),complex);states[:,0]=0;states[:256,0]=psi
    # Full joint-space deterministic probe; no PR/event sampling or new H.
    states[:,1]=np.array([1 if j%3 else 1j for j in range(dim)],complex)/math.sqrt(dim)
    native,_=evaluate(spec,states,False);ct,acc=evaluate(spec,states,True)
    norms=np.linalg.norm(ct-native,axis=0);bound=F(acc['strict_joint_operator_upper'])
    assert max(norms)<=float(bound)+1e-10
    outcomes=[]
    signs=np.where((np.arange(dim)&(1<<(n-1)))==0,1.,-1.)
    for j,label in enumerate(('exact_ground_reference_vacuum','deterministic_full_joint_probe')):
        sn=float(np.sum(signs*abs(native[:,j])**2));sc=float(np.sum(signs*abs(ct[:,j])**2))
        assert abs(sn-sc)<=2*float(bound)+1e-10
        outcomes.append(dict(state=label,state_l2_error=float(norms[j]),native_measurement=sn,CT_measurement=sc,
            measurement_difference=abs(sn-sc),native_norm=float(np.linalg.norm(native[:,j])),CT_norm=float(np.linalg.norm(ct[:,j]))))
    return dict(name=spec['name'],qubits=n,initial_reference_resets=spec['initial_reference_resets'],
        native_file_sha256=spec['native_file_sha256'],outcomes=outcomes,account=acc,
        full_joint_operator_certificate='unitary telescoping of strict per-RZ guards; finite-state diagnostics are not the norm proof',
        whole_state_simulation='identical exact-word 2x2 actions at native positions, checked against existing word emitter; global W retained',
        phase_fit=False,passed=True)


def error_and_shots(whole):
    pre=predeclared();mp.iv.prec=192
    rho0=pre['CT0_q1_endpoints']['arms']['Bprime']['rho0'];rows=[]
    for arm in ('Bprime','reference'):
        old=pre['CT0_q1_endpoints']['arms'][arm];N=old['saved_shots_per_axis']
        peraxis={axis:max(F(x['account']['strict_joint_operator_upper']) for x in whole if '_'+arm+'_'+axis==x['name'][x['name'].find('_'):]) for axis in 'XY'}
        actual=max(peraxis.values())
        # This all-support count is a sufficient precision CONDITION only.
        # Finite CT-1 words do not implement the as-yet-unqualified all-key policy.
        uniform=max(x['worst_case_RZ_upper'] for x in old['axis'].values())*F(pre['epsilon'])
        for kind,error in [('finite_pilot_maximum',actual),('all_support_uniform_precision_condition',uniform)]:
            extra=2*mp.iv.sqrt(I(2))*(I(error.numerator)/error.denominator)
            d=I(old['d_old'])+extra;ratio=d/I(rho0)
            rho=I(rho0)-d;beta=I(old['B_lower'])-mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))
            needed=4*mp.iv.log(I(1440))/(rho*rho*mp.iv.sin(beta)**2)
            passed=lower(rho)>0 and lower(beta)>0 and upper(needed)<=N
            assert passed,(arm,kind)
            rows.append(dict(arm=arm,kind=kind,operator_error_upper=str(error),synthesis_raw_disk_upper=upper(extra),
                d_old=old['d_old'],d_total_upper=upper(d),rho_lower=lower(rho),beta_stat_lower=lower(beta),
                saved_N=N,required_continuous_N_upper=upper(needed),saved_N_sufficient=True,
                conditional_all_keys_policy=(kind=='all_support_uniform_precision_condition')))
    result=dict(status='SAVED_Q1_N_BUDGET_CONDITIONS_PASS',rows=rows,alpha='1/720',phase_cap='.4',
        failure_total='36/720=.05, other rounds unchanged native model',energy_accuracy_Ha=1.5936e-4,
        compiler_and_phase_disks_added_once=True,new_N_schedule=False,
        full_law_CT_RPE_success='NOT_QUALIFIED: only fixed pilot words acquired; uniform all-support word policy remains unimplemented',
        finite_pilot_does_not_estimate_PR_mean=True,all_support_vacuum_price='UNQUALIFIED',all18_T='NOT_ACQUIRED')
    save_json(OUT/'ERROR_AND_SHOTS.json',result);return result


def stage_c():
    pre=predeclared();word_matrices();simulator_check()
    specs=json.loads((OUT/'LOCAL_TARGETS.json').read_text())['records']
    local=[]
    with ProcessPoolExecutor(max_workers=min(len(specs),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(local_one,s) for s in specs]):
            local.append(f.result())
    save_json(OUT/'LOCAL_CORRECTNESS.json',dict(status='PASS_LOCAL_CORRECTNESS',records=local,passed=True,
        observed_numbers='binary64 full-space local matrices; no phase fit; numerical tolerance separated from interval certificates'))
    whole=[];specs=[whole_spec(p) for p in sorted((OUT/'native').glob('*.qpy'))]
    with ProcessPoolExecutor(max_workers=min(len(specs),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(whole_one,s) for s in specs]):
            row=f.result();whole.append(row);print('whole checked',row['name'],flush=True)
    save_json(OUT/'JOINT_CORRECTNESS.json',dict(status='PASS_PILOT_JOINT_CORRECTNESS',records=whole,passed=True,
        scope='two saved q1/r1 trajectories, Bprime/reference, X/Y; not the all-support CT mean',
        full_joint_norm_proof='strict local guards + unitary telescoping, not finite states',whole_dense_unitary_built=False))
    e=error_and_shots(whole)
    save_json(OUT/'PILOT_T_COUNTS.json',dict(scope='finite saved pilot circuits only',rows=[dict(name=r['name'],
        T=r['account']['T'],Tdagger=r['account']['Tdagger'],signed_RZ_occurrences=r['account']['signed_RZ_occurrences']) for r in whole],
        performance_winner=None,all18_T=None,all_support_expected_T=None,T_depth=None,minimum_T=False))
    save_json(OUT/'SOURCE.json',dict(base=BASE,source='src/trottertracks/pr_reference_ct1.py',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage_A_snapshot='STAGE_A_SOURCE.py',stage_A_sha256=hashlib.sha256((OUT/'STAGE_A_SOURCE.py').read_bytes()).hexdigest(),
        reused_sources=['track_c_h4_synthesis.acquire/strict_guard/exact_word_matrix','track_c2_b_verify.lower_native',
            'pr_reference_ct0.validate_record and saved q1 headroom','pr_reference_bprime.build_saved/reconstructed',
            'pr_reference_transfer_qualification.assemble','pr_reference_interference_pilot.reference_core/wrapper/ground',
            'track_c_compiler_a.compile_reference','compiler_a_18round.I/upper/lower'],
        source_growth='Stage A source snapshot retained; later acquisition/correctness functions added before their execution. No running source or old SOURCE changed.',
        independent_review=False))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='Authorized CT-1 finite scope complete',
        local='PASS_LOCAL_CORRECTNESS',joint='PASS_PILOT_JOINT_CORRECTNESS',shots=e['status'],
        full_support_word_price='UNQUALIFIED',all18_T='NOT_ACQUIRED',old_SOURCE_results_STOP_preserved=True,
        next_owner='GPT/user: scientific interpretation and choice of any independent next scope'))
    print('CT-1 mandatory STOP',flush=True)
