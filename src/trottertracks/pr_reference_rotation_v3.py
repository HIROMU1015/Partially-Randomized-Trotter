"""Saved-data-only V3: provenance-aware RZ resources; no compile or synthesis."""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import hashlib,io,json,lzma,math,os,platform,subprocess,sys,tarfile
from importlib.metadata import version
import mpmath as mp
from .track_c_h4 import ROOT,common,save_json,diagonal_primitives
from . import pr_reference_bprime as bp
from .pr_reference_bprime_accounting import distributions
from .track_c_compiler_a_18round import I,upper,lower

BASE='43bfc5918703e8dcd020be4fcf96bb8dfd99f909'
OUT=ROOT/'artifacts/pr_reference_rotation_v3/2026-11-07'
PILOT='artifacts/pr_reference_interference_pilot/2026-11-06/'
TRANSFER='artifacts/pr_reference_transfer_qualification/2026-11-06/'
BPRIME='artifacts/pr_reference_bprime/2026-11-07/'
CATEGORIES=('Clifford','T_type','arbitrary','unresolved')

def data(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        return {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}

def initialize():
    if OUT.exists():raise RuntimeError('Fresh V3 output required')
    OUT.mkdir(parents=True)
    raw=lzma.decompress(data(PILOT+'common.pkl.xz'));(OUT/'common.pkl').write_bytes(raw)
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')
    save_json(OUT/'INPUTS.json',dict(base=BASE,common_sha256=hashlib.sha256(raw).hexdigest(),
        new_SCF_DF_ground_solve_sampling_compile_synthesis=0))
    save_json(OUT/'ENVIRONMENT.json',dict(platform=platform.platform(),host=platform.node(),python=sys.version,
        executable=sys.executable,packages={p:version(p) for p in ('qiskit','numpy','scipy','mpmath')},
        affinity=len(os.sched_getaffinity(0)),memory=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        shared_settings_changed=False))

def attach():
    if (OUT/'STOP.json').exists():raise RuntimeError('Mandatory STOP')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')

def canonical(k):return float(F(k,4))*math.pi

@lru_cache(None)
def numeric_class(theta_hex):
    """No tolerance snapping: canonical float alone supplies no exact origin."""
    theta=float.fromhex(theta_hex)
    if not math.isfinite(theta):return 'unresolved'
    if theta==0:return 'Clifford'
    # Canonical binary encodings without source proof remain unresolved.
    k=round(theta/(math.pi/4))
    if theta==canonical(k):return 'unresolved'
    mp.iv.prec=192
    angle=I(theta);grid=I(k)*mp.iv.pi/4
    if upper(abs(angle))>1e12:return 'unresolved'
    # An outward enclosure excluding the nearest pi/4 grid point proves
    # non-membership; adjacent grid points are also checked.
    for j in (k-1,k,k+1):
        delta=angle-I(j)*mp.iv.pi/4
        if lower(delta)<=0<=upper(delta):return 'unresolved'
    return 'arbitrary'

def classify(theta,pi_quarters=None,origin=None):
    """Resource type from exact analytic provenance, plus literal-angle caveat."""
    if isinstance(theta,str):return dict(category='unresolved',literal_category='unresolved',angle_key='symbolic:'+theta,origin=origin)
    theta=float(theta);key=theta.hex()
    if pi_quarters is None:
        cat=numeric_class(key)
        return dict(category=cat,literal_category=cat,angle_key=key,origin=origin)
    k=int(pi_quarters)
    if theta!=canonical(k):
        return dict(category='unresolved',literal_category='unresolved',angle_key=key,origin=origin,provenance_encoding_mismatch=True)
    cat='Clifford' if k%2==0 else 'T_type'
    mp.iv.prec=192;res=upper(abs(I(theta)-I(k)*mp.iv.pi/4))
    return dict(category=cat,literal_category='Clifford' if theta==0 else 'unresolved',angle_key=key,
        exact_provenance_pi_quarters=k,angle_rounding_residual_upper=res,origin=origin,
        exact_native_Clifford_T_claim=theta==0)

def vec(entries,field='category'):
    c=Counter(x[field] for x in entries);return {k:c[k] for k in CATEGORIES}
def add(*counters):
    out=Counter()
    for c in counters:out.update(c)
    return out
def scaled(c,n):return Counter({k:v*n for k,v in c.items()})

@lru_cache(None)
def catalogues():
    old=[json.loads(x) for x in archive(TRANSFER+'local_catalogue_boundary.tar.xz').values()]
    new=[json.loads(x) for x in archive(BPRIME+'Bprime_local_catalogue.tar.xz').values()]
    return old,new

@lru_cache(None)
def records():
    old,new=catalogues()
    return {a:{r['name']:r for r in row['records']} for a,row in [
        ('registered',next(x for x in old if x['request']==['Simon_Love',True,True,True])),
        ('reference',next(x for x in old if x['request']==['reference',True,True,True])),
        ('Bprime',next(x for x in new if x['request']==[True,True]))]}

def rzvalues(record):return [(i,p[0]) for i,(op,_,p) in enumerate(record['lowering']) if op=='rz']

def local_entries(record):
    meta=record['meta'];fam=meta.get('family');result=[]
    for index,value in rzvalues(record):
        k=None;origin=f'{fam}:{index}'
        if fam=='xx_plus_yy':
            theta,beta=meta['params']
            fixed={1:2,2:-2,4:2,6:0,10:12,11:0,15:12,17:-2,19:-2,20:2,21:2,23:2,24:2,25:2}
            if index in fixed:k=fixed[index]
            elif index in (0,18):
                # Cached numeric angle(np...) has no saved symbolic branch.
                # Even a canonical encoding is not enough to promote beta.
                origin='cached numeric Gaussian beta; symbolic branch not retained'
            elif index in (8,13):origin='pi - source_theta/2 (physical Gaussian angle)'
            else:raise AssertionError(('unrecognized XX RZ position',index))
        elif fam=='crzz':
            if index not in (19,21):
                # Fixed CCX/H slots, verified against Qiskit1.3 definitions.
                k=round(float(value)/(math.pi/4));assert abs(k) in (1,2)
                origin='fixed CCX/H analytic pi constant, slot '+str(index)
            else:origin='controlled RTE actual theta/2'
        elif fam in ('cz','cswap','h','sdg','z'):
            k=round(float(value)/(math.pi/4));origin='fixed '+fam+' analytic pi constant, slot '+str(index)
        result.append(dict(native_index=index,**classify(value,k,origin)))
    return result

def real_input_witness():
    import numpy as np
    p=common()['preps'][3];bs=list(p.deterministic_blocks)
    definitions=p.rte_preparation.basis_registry._definitions
    result=[]
    # Inspect cached matrices and SOURCE angles from ops.py / Qiskit Nature.
    # These observations do not recover symbolic branches or promote types.
    ham=common()['ham']
    for name,a in [('one_body',ham.one_body),('g_matrices',ham.g_matrices)]:
        a=np.asarray(a);real=not np.iscomplexobj(a) or not np.any(a.imag!=0)
        assert real
        result.append(dict(name=name,dtype=str(a.dtype),shape=list(a.shape),all_imaginary_parts_zero=real))
    ops=[*[(f'D{i}',b.runtime_basis_operations) for i,b in enumerate(bs)],*[(k,d.runtime_operations) for k,d in definitions.items()]]
    phases=[]
    for name,seq in ops:
        for g,_ in seq:
            if g.name=='rz':assert float(g.params[0]) in (0.,math.pi,-math.pi)
            elif g.name=='xx_plus_yy':assert float(g.params[1]) in (-math.pi/2,math.pi/2,-3*math.pi/2)
            else:raise AssertionError(g.name)
        phases.append(dict(basis=name,RZ_phase_values=sorted(set(float(g.params[0]) for g,_ in seq if g.name=='rz')),
            XX_beta_values=sorted(set(float(g.params[1]) for g,_ in seq if g.name=='xx_plus_yy'))))
    save_json(OUT/'REAL_BASIS_WITNESS.json',dict(matrices=result,phases=phases,
        source='ops.py / Qiskit Nature BogoliubovTransform; cached numeric phase observations only',
        promotes_cached_phases_to_Clifford=False,
        literal_float_vs_symbolic='analytic provenance only; binary pi encoding residual retained separately'))
    return result

TOKENS={}
def token(entry):
    key='|'.join((entry['category'],entry['literal_category'],entry['angle_key']))
    TOKENS.setdefault(key,entry)
    return key
def profile(entries):return Counter(token(x) for x in entries)
def physical(theta,origin):return profile([classify(theta,origin=origin)])
def unresolved(name):return profile([classify(name,origin='trajectory-dependent angle; not enumerated')])
def summary(p,field='category'):
    c=defaultdict(F)
    for k,n in p.items():c[TOKENS[k][field]]+=n
    return {k:c[k] for k in CATEGORIES}
def histogram(p):
    c=Counter()
    for k,n in p.items():
        key=TOKENS[k]['angle_key']
        if key in ('-0x0.0p+0','0x0.0p+0'):key='0x0.0p+0'
        c[key]+=n
    return +c
def actual_hist(record):
    return Counter(('0x0.0p+0' if float(v)==0 else float(v).hex()) for _,v in rzvalues(record))

@lru_cache(None)
def local_map():
    result={}
    for arm,recs in records().items():
        for rec in recs.values():
            fam=rec['meta'].get('family')
            if fam=='xx_plus_yy':
                key=tuple(float(x).hex() for x in rec['meta']['params'])
                result[key]=profile(local_entries(rec))
    return result

def basis(seq):
    out=Counter()
    for g,_ in seq:
        if g.name=='rz':out.update(physical(float(g.params[0]),'cached numeric Gaussian phase; no symbolic phase branch'))
        elif g.name=='xx_plus_yy':out.update(local_map()[tuple(float(x).hex() for x in g.params)])
        else:raise AssertionError(g.name)
    return out

@lru_cache(None)
def basis_profiles():
    p=common()['preps'][3];defs={f'D{i}':x.runtime_basis_operations for i,x in enumerate(p.deterministic_blocks)}
    defs.update({k:x.runtime_operations for k,x in p.rte_preparation.basis_registry._definitions.items()})
    result={}
    for name,seq in defs.items():
        result[name]=basis(seq)
        inverse=[(g.inverse(),qs) for g,qs in reversed(seq)]
        result['inverse:'+name]=basis(inverse)
    return result

def pair(name):return add(basis_profiles()[name],basis_profiles()['inverse:'+name])

def fixed(family):
    for arm in ('registered','reference'):
        rec=next((r for r in records()[arm].values() if r['meta'].get('family')==family),None)
        if rec:return profile(local_entries(rec))
    raise KeyError(family)

def diagonal(index,t,kind='plain'):
    p=common()['preps'][3];z=diagonal_primitives(p.deterministic_blocks[index],t);out=pair('D'+str(index))
    for _,_,angle in z.rzz:
        if float(angle)!=0:
            assert kind!='control'
            out.update(physical(-float(angle) if kind=='directional' else float(angle),'D ZZ diagonal'))
    for _,angle in z.rz:
        theta=-float(angle) if kind=='directional' else float(angle)
        if kind=='control':out.update(physical(theta/2,'controlled D RZ +half'));out.update(physical(-theta/2,'controlled D RZ -half'))
        else:out.update(physical(theta,'D single RZ'))
    phi=float(z.global_phase)
    if phi and kind!='plain':out.update(physical(2*phi if kind=='directional' else phi,'D ancilla-relative scalar P'))
    return out

@lru_cache(None)
def deterministic_profiles():
    out={}
    for k in range(4):out[f'block:{k}:half']=diagonal(k,.01)
    out['block:0:full']=diagonal(0,.02)
    out['left_all']=add(*(diagonal(k,.01) for k in range(4)))
    out['left_no0']=add(*(diagonal(k,.01) for k in range(1,4)))
    out['right_all']=add(*(diagonal(k,.01,'directional') for k in range(4)))
    out['right_no0']=add(*(diagonal(k,.01,'directional') for k in range(1,4)))
    out['boundary_control']=diagonal(0,.02,'control')
    return out

def inventory():
    local_map();basis_profiles();deterministic_profiles();rows=[]
    for arm,recs in records().items():
        for name,r in recs.items():
            m=r['meta'];p=None
            if m['kind']=='basis':p=basis_profiles()[m['id']]
            elif m['kind']=='basis_inverse':p=basis_profiles()['inverse:'+m['id']]
            elif name in deterministic_profiles():p=deterministic_profiles()[name]
            elif m.get('family'):p=profile(local_entries(r))
            elif m['kind']=='bprime':
                theta=m['theta'];p=add(physical(theta/2,'Bprime actual +half'),physical(-theta/2,'Bprime actual -half'))
            if p is None:continue
            assert histogram(p)==actual_hist(r),(arm,name,'RZ provenance projection mismatch')
            assert sum(p.values())==r['ops'].get('rz',0)
            rows.append(dict(arm=arm,name=name,categories={k:int(v) for k,v in summary(p).items()},
                literal_categories={k:int(v) for k,v in summary(p,'literal_category').items()},
                RZ=sum(p.values()),CX=r['ops'].get('cx',0),histogram=histogram(p),tokens=dict(p)))
    # Context identity was established in Bprime V1; inspect saved variants,
    # not a new compiler run. Their RZ histograms must still be identical.
    old,new=catalogues();checks=[]
    for c in old+new:
        for r in c['records']:
            if r['name'] in deterministic_profiles():
                assert actual_hist(r)==histogram(deterministic_profiles()[r['name']])
                checks.append(c.get('key',str(c['request']))+':'+r['name'])
    save_json(OUT/'LOCAL_INVENTORY.json',dict(rows=rows,context_projection_checks=checks,
        categories='exact analytic source types with binary encoding residual; unresolved numeric phases remain unresolved',
        literal_ledger='nonzero binary pi encodings are unresolved for exact native Clifford+T purposes'))
    return rows

@lru_cache(None)
def crzz_constants():
    rec=next(x for x in records()['registered'].values() if x['meta'].get('family')=='crzz')
    entries=[e for e in local_entries(rec) if e['native_index'] not in (19,21)]
    return profile(entries)

def rotation(arm,theta,zz):
    if arm=='reference':return physical(theta,'RTE native physical theta')
    p=add(physical(theta/2,'controlled RTE +half'),physical(-theta/2,'controlled RTE -half'))
    if arm=='registered' and zz:p.update(crzz_constants())
    return p

def integer_pi_phase(theta=None,n=None):
    if theta is None:
        return profile([dict(category='Clifford',literal_category='unresolved',angle_key='symbolic:integer_pi_event_phase',
            origin='exact event/product sign integer*pi before floating evaluation',exact_native_Clifford_T_claim=False)])
    mp.iv.prec=192
    return profile([dict(category='Clifford',literal_category='Clifford' if theta==0 else 'unresolved',angle_key=float(theta).hex(),
        origin='integer*pi event/product phase',exact_provenance_pi_quarters=4*n,
        angle_rounding_residual_upper=upper(abs(I(theta)-n*mp.iv.pi)),exact_native_Clifford_T_claim=theta==0)])

def dparts(arm,q):
    d=deterministic_profiles()
    if arm!='reference':
        raw=scaled(add(d['left_all'],d['right_all']),q)
        corr=scaled(add(d['left_no0'],d['right_no0'],d['boundary_control'],scaled(add(d['left_all'],d['right_all']),-1)),q-1)
    else:
        raw=scaled(add(*(diagonal(k,t) for k in range(4) for t in (.01,-.01))),q)
        n=q//2
        corr=Counter() if q==1 else scaled(add(*(add(diagonal(0,2*t),scaled(diagonal(0,t),-2)) for t in (.01,-.01))),n-1)
    return {'D_unfused':raw,'D_boundary_fusion_difference':corr}

def expected_parts(arm,q,r,bits,axis):
    p=common()['preps'][3];components,probs,p2,pzz,es=distributions(bits,r)
    parts=dparts(arm,q);pb=defaultdict(F)
    for c,pi in zip(components,probs):pb[c.basis_id]+=pi
    parts['RTE_basis_frames']=add(*(scaled(pair(name),q*(pi+(r*(1+2*p2)-1)*pi*(1-pi))) for name,pi in pb.items()))
    rot=Counter();tau=p.rte_preparation.symbolic_tail.lambda_r*(.02/r)
    for c,pi in zip(components,probs):
        for order,po in ((0,1-p2),(2,p2)):
            theta=2*c.coefficient_sign*math.atan(tau/(order+1))
            rot.update(scaled(rotation(arm,theta,len(c.diagonal_pauli_support)==2),q*r*pi*po))
    parts['RTE_central_rotation']=rot
    parts['RTE_products']=scaled(fixed('z' if arm=='reference' else 'cz'),2*q*r*p2*es)
    parts['RTE_event_sign_phase']=Counter() if arm=='reference' else scaled(integer_pi_phase(),q*(1-(1-p2)**r))
    if arm=='reference':parts['scalar_or_vacuum_correction']=unresolved('vacuum_phase[m='+str(int(math.log2(q)))+']')
    else:
        scalar=-.02*q*(p.constant_coefficient+p.extracted_identity_coefficient)
        parts['scalar_or_vacuum_correction']=physical(scalar,'controlled physical scalar P')
    parts['CSWAP']=scaled(fixed('cswap'),16) if arm=='reference' else Counter()
    parts['XY_wrapper']=add(scaled(fixed('h'),2),fixed('sdg') if axis=='Y' else Counter())
    return parts

@lru_cache(None)
def samples():return bp.saved_samples()

def saved_path_parts(row,arm,axis):
    p=common()['preps'][3];_,_,events=bp.reconstructed(row);comp={c.component_id:c for c in p.rte_preparation.symbolic_tail.components}
    q=row['q'];r=row['r'];parts=dparts(arm,q);frames=Counter();rot=Counter();products=Counter();phases=Counter()
    for j in range(q):
        last=None;phase=0.;npi=0
        occurrence=events[j*r:(j+1)*r]
        # Fixed SOURCE accumulates all event sign phases before products.
        for e in occurrence:
            if e.taylor_order==2:phase+=math.pi;npi+=1
        for e in occurrence:
            for cid in [*e.product_component_ids,e.rotation_component_id]:
                c=comp[cid]
                if c.basis_id!=last:frames.update(pair(c.basis_id));last=c.basis_id
            c=comp[e.rotation_component_id];theta=2*c.coefficient_sign*e.rotation_angle
            if arm=='reference' and q>1 and j>=q//2:theta=-theta
            if arm=='reference' and q==1:pass # q1 rotation is in G, positive.
            rot.update(rotation(arm,theta,len(c.diagonal_pauli_support)==2))
            for cid in e.product_component_ids:
                c=comp[cid];products.update(scaled(fixed('z' if arm=='reference' else 'cz'),len(c.diagonal_pauli_support)))
                if c.coefficient_sign<0:phase+=math.pi;npi+=1
        if arm!='reference' and phase:phases.update(integer_pi_phase(phase,npi))
    parts.update(RTE_basis_frames=frames,RTE_central_rotation=rot,RTE_products=products,RTE_event_sign_phase=phases)
    if arm=='reference':parts['scalar_or_vacuum_correction']=physical(row.get('phase',row.get('reference_phase')),'saved trajectory vacuum correction')
    else:parts['scalar_or_vacuum_correction']=physical(-.02*q*(p.constant_coefficient+p.extracted_identity_coefficient),'controlled physical scalar P')
    parts['CSWAP']=scaled(fixed('cswap'),16) if arm=='reference' else Counter()
    parts['XY_wrapper']=add(scaled(fixed('h'),2),fixed('sdg') if axis=='Y' else Counter())
    return parts

def aggregate(parts):return add(*parts.values())

def validate_cohort():
    braw=[json.loads(x) for x in archive(BPRIME+'paired_Bprime_prices.tar.xz').values()]
    checks=[];projections={}
    for row in samples():
        nr=next(x for x in braw if (x['q'],x['r'],x['rare'],x['j'])==(row['q'],row['r'],row['rare'],row['j']))
        for arm in ('registered','Bprime','reference'):
            for i,axis in enumerate('XY'):
                parts=saved_path_parts(row,arm,axis);p=aggregate(parts)
                actual=nr['Bprime_values'][i][0] if arm=='Bprime' else row['values']['Simon_Love' if arm=='registered' else 'reference'][i][0]
                dropped=sum(p.values())-actual
                if arm=='reference':
                    assert 0<=dropped<=sum(parts['RTE_products'].values())
                    if dropped:
                        parts['source_product_Z_cancellation']=scaled(fixed('z'),-dropped);p=aggregate(parts)
                else:assert dropped==0,(row['q'],arm,'RZ total mismatch')
                assert sum(p.values())==actual
                key=f'q{row["q"]}_r{row["r"]}_rare{int(row["rare"])}_j{row["j"]}_{arm}_{axis}'
                projections[key]=histogram(p)
                checks.append(dict(key=key,RZ=actual,source_product_Z_RZ_removed=dropped,
                    categories={k:int(v) for k,v in summary(p).items()},literal_categories={k:int(v) for k,v in summary(p,'literal_category').items()},passed=True))
    save_json(OUT/'COHORT_CHECKS.json',dict(checks=checks,native_recompiles=0,passed=True))
    return projections

def qpy_one(task):
    path,name,payload=task
    from qiskit import qpy
    q=qpy.load(io.BytesIO(payload))[0];base=Path(name).name
    qn=int(base.split('_')[0][1:]);rare=int(base.split('_')[1][4:]);arm=base.split('_',2)[2][:-4]
    arm={'Simon_Love':'registered','Bprime':'Bprime','reference':'reference'}[arm]
    h=Counter()
    for inst in q.data:
        if inst.operation.name=='rz':
            a=float(inst.operation.params[0]);h['0x0.0p+0' if a==0 else a.hex()]+=1
    return dict(archive=path,file=name,key=f'q{qn}_r1_rare{rare}_j0_{arm}_X',histogram=h,RZ=sum(h.values()))

def validate_qpy(projections=None):
    tasks=[]
    for path in (PILOT+'native_circuits.tar.xz',TRANSFER+'native_circuits.tar.xz',BPRIME+'Bprime_native_qpy.tar.xz'):
        tasks.extend((path,name,payload) for name,payload in archive(path).items()
            if Path(name).name.split('_',2)[2][:-4] in ('Simon_Love','Bprime','reference'))
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        rows=list(pool.map(qpy_one,tasks))
    if projections is None:
        projections={};checks={x['key']:x for x in json.loads((OUT/'COHORT_CHECKS.json').read_text())['checks']}
        for row in rows:
            key=row['key'];match=checks[key]
            qn=int(key.split('_')[0][1:]);rare=bool(int(key.split('_')[2][4:]))
            arm=key.rsplit('_',1)[0].split('_',4)[4]
            saved=next(s for s in samples() if s['q']==qn and s['rare']==rare and s['j']==0)
            parts=saved_path_parts(saved,arm,'X')
            if match['source_product_Z_RZ_removed']:
                parts['source_product_Z_cancellation']=scaled(fixed('z'),-match['source_product_Z_RZ_removed'])
            projections[key]=histogram(aggregate(parts))
    for row in rows:assert row['histogram']==projections[row['key']],(row['file'],'native histogram mismatch')
    save_json(OUT/'QPY_CHECKS.json',dict(rows=rows,passed=True,new_compile=False,new_simulation=False))
    return rows

def synthetic_tests():
    tests=[]
    for k in (-12,-6,-2,-1,0,1,2,4,12):
        x=classify(canonical(k),k,'verified analytic pi slot')
        assert x['category']==('Clifford' if k%2==0 else 'T_type')
        if k:assert x['literal_category']=='unresolved' and x['angle_rounding_residual_upper']>0
        tests.append('provenance pi quarters '+str(k))
    # Nearness and even identical numeric encodings without provenance cannot
    # acquire an exact Clifford/T type.
    assert classify(math.pi/4)['category']=='unresolved'
    assert classify(math.nextafter(math.pi/4,math.inf),1,'wrong encoding')['category']=='unresolved'
    assert classify(math.nextafter(math.pi/4,math.inf))['category']!='T_type'
    assert classify(.123)['category']=='arbitrary'
    assert classify(-.123)['category']=='arbitrary'
    assert classify('unbound trajectory')['category']=='unresolved'
    assert classify(0.)['category']=='Clifford'
    assert vec(local_entries(next(r for r in records()['registered'].values() if r['meta'].get('family')=='crzz')))['T_type']==14
    assert vec(local_entries(next(r for r in records()['reference'].values() if r['meta'].get('family')=='cswap')))['T_type']==7
    tests.extend(['no tolerance snapping','missing provenance','symbolic unresolved','sign and zero','CRZZ14T origin','CSWAP7T origin'])
    save_json(OUT/'SYNTHETIC_TESTS.json',dict(passed=True,tests=tests,scientific_performance_claim=False))
    return tests

def audit_synthesis_records():
    path='artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz';count=0;angles=set();routes=Counter();bad=[]
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        for m in t.getmembers():
            if not m.isfile() or not m.name.endswith('.json'):continue
            row=json.load(t.extractfile(m));count+=1
            if row['status']!='PASS':bad.append(m.name)
            angles.add(row['theta_hex']);routes[row['route']]+=1
    assert not bad
    result=dict(source_path=path,records_read=count,distinct_positive_binary_angle_keys=len(angles),routes=routes,
        prior_model='Full Y4 / LD12, not this Partial S2/LD3 schedule',
        target='strict RZ exact binary angle; up_to_phase=False, W retained, precision/seed in key',
        reused_T_prices=0,new_synthesis_calls=0,new_strict_guard_evaluations=0,
        reason='V3 selects no synthesis precision or error allocation, so cached T prices cannot supply total T',
        existing_tolerance_classifier='native_df_h6_cost_adapter.rotation_classes is diagnostic and not used',
        key_reuse='signed float.hex for actual angles; abs cache key only a lookup convention, not gate sharing')
    save_json(OUT/'SYNTHESIS_REUSE_AUDIT.json',result)
    return result

def account_rounds():
    assert json.loads((OUT/'SYNTHETIC_TESTS.json').read_text())['passed']
    assert json.loads((OUT/'QPY_CHECKS.json').read_text())['passed']
    saved=json.loads(data(BPRIME+'TOTALS_18ROUND.json'))
    prior=json.loads(data(TRANSFER+'STAGE4_RPE.json'))
    totals={arm:{ledger:{bound:Counter() for bound in ('lower','upper')} for ledger in ('analytic_provenance','literal_native')} for arm in ('registered','Bprime','reference')}
    rows=[];detail=[]
    for row,prev in zip(saved['rows'],prior['rows']):
        q=row['q'];r=row['r'];bits=row['bits'];rounds={}
        for arm in totals:
            n=row['shots_per_axis'] if arm=='Bprime' else prev['arms']['Simon_Love' if arm=='registered' else 'reference']['shots_per_axis']
            axes={}
            for axis in 'XY':
                parts=expected_parts(arm,q,r,bits,axis);pu=aggregate(parts);pl=pu.copy()
                if arm=='reference':pl.update(scaled(parts['RTE_products'],-1))
                prices=row['price_per_axis'][axis]
                price_key={'registered':'registered','Bprime':'Bprime','reference':'reference'}[arm]
                expected_upper=F(prices['registered_RZ_upper' if arm=='registered' else price_key+'_RZ_upper'])
                expected_lower=F(prices['registered_RZ_lower' if arm=='registered' else price_key+'_RZ_lower'])
                assert sum(pu.values())==expected_upper and sum(pl.values())==expected_lower,(row['m'],arm,axis,'old RZ price mismatch')
                ledgers={}
                for ledger,field in [('analytic_provenance','category'),('literal_native','literal_category')]:
                    lo=summary(pl,field);up=summary(pu,field)
                    assert all(lo[k]>=0 and up[k]>=lo[k] for k in CATEGORIES)
                    for k in CATEGORIES:
                        totals[arm][ledger]['lower'][k]+=n*lo[k];totals[arm][ledger]['upper'][k]+=n*up[k]
                    ledgers[ledger]={'lower':{k:str(v) for k,v in lo.items()},'upper':{k:str(v) for k,v in up.items()}}
                axes[axis]=dict(shots=n,RZ_lower=str(expected_lower),RZ_upper=str(expected_upper),ledgers=ledgers)
                detail.append(dict(m=row['m'],q=q,r=r,bits=bits,arm=arm,axis=axis,shots=n,
                    parts={k:{token:str(v) for token,v in p.items()} for k,p in parts.items()},
                    reference_lower_removes='all product-Z RZs' if arm=='reference' else None))
            rounds[arm]=axes
        rows.append(dict(m=row['m'],q=q,r=r,bits=bits,arms=rounds))
    result={}
    def fi(x):return I(x.numerator)/x.denominator
    mp.iv.prec=192
    for arm,ledgers in totals.items():
        result[arm]={}
        for ledger,bounds in ledgers.items():
            lo=bounds['lower'];up=bounds['upper']
            result[arm][ledger]=dict(categories={k:dict(lower=0 if lo[k]==0 else lower(fi(lo[k])),upper=0 if up[k]==0 else upper(fi(up[k])),
                lower_fraction=str(lo[k]),upper_fraction=str(up[k])) for k in CATEGORIES},
                complete_arbitrary_count_interval=dict(lower=lower(fi(lo['arbitrary'])),
                    upper=upper(fi(up['arbitrary']+up['unresolved'])),
                    meaning='unresolved may include any category; not assigned arbitrarily'))
    comparisons={}
    for ledger in ('analytic_provenance','literal_native'):
        b=totals['Bprime'][ledger];rr=totals['reference'][ledger]
        resolved_diff=b['lower']['arbitrary']-rr['upper']['arbitrary']
        worst_diff=b['lower']['arbitrary']-(rr['upper']['arbitrary']+rr['upper']['unresolved'])
        best_diff=(b['upper']['arbitrary']+b['upper']['unresolved'])-rr['lower']['arbitrary']
        comparisons[ledger]=dict(resolved_arbitrary_saving_lower=lower(fi(resolved_diff)),
            resolved_arbitrary_saving_percent_lower=lower(100*fi(resolved_diff)/fi(b['lower']['arbitrary'])),
            complete_arbitrary_saving_interval=[lower(fi(worst_diff)),upper(fi(best_diff))],
            complete_arbitrary_reduction_qualified=worst_diff>0,
            note='resolved subset comparison does not settle unresolved rotations')
    main=dict(status='PARTIAL_CLASSIFICATION_COMPLETE_ARBITRARY_ADVANTAGE_UNQUALIFIED',
        primary_ledger='literal_native',
        totals=result,comparisons=comparisons,rows=rows,
        inherited_CX_RZ_shots_qubits=saved['totals'],
        classification='analytic provenance types and literal native exactness reported separately',
        actual_native_angles_or_circuits_changed=False,RPE_shots_or_errors_changed=False,
        total_T='NOT_ACQUIRED; T-type RZ count is not minimum T or strict synthesis cost',
        V5_started=False,new_compile_synthesis_sampling=0)
    # Full signed angle keys, roles, finite-bit multiplicities and per-round
    # contributions; compressed saved result, not a synthesis request.
    payload=dict(tokens=TOKENS,rows=detail)
    (OUT/'ANGLE_KEY_MULTIPLICITIES.json.xz').write_bytes(lzma.compress(json.dumps(payload,sort_keys=True).encode()))
    save_json(OUT/'CLASSIFIED_18ROUND.json',main)
    return main

def finish_stop():
    assert (OUT/'CLASSIFIED_18ROUND.json').exists()
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='V3 finite scope complete; unresolved symbolic/numeric classifications preserved',
        old_SOURCE_raw_results_STOPs_preserved=True,new_compile_synthesis_sampling=0,V5_started=False,
        next_owner='GPT/user: interpret unresolved rotation classes and decide whether to separately authorize V5'))
