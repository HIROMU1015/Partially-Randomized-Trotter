"""All-support local interval and floating phase bounds, fixed SOURCE only."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy,hashlib,json,math,os
import mpmath as mp
import numpy as np
from .pr_reference_transfer_qualification import OUT,common,save_json,data
from . import track_c_compiler_a_18round as iv
from .pr_reference_transfer_cost_bounds import catalog_bounds

BITS=192
ROUND_PATH='artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json'


def source_matrix(f,params):
    if f in ('xx_plus_yy','crz','crzz','cz','h','sdg'):return iv.source_matrix(iv.spec(f,params))
    if f=='z':
        a=iv.eye(2);a[1][1]=iv.C(-1);return a
    if f=='cswap':
        a=[[iv.C() for _ in range(8)] for _ in range(8)]
        for j in range(8):
            k=j
            if j&1 and ((j>>1)&1)!=((j>>2)&1):k=j^6
            a[k][j]=iv.C(1)
        return a
    raise ValueError(f)


def interval_one(record):
    mp.iv.prec=BITS
    f=record['meta']['family'];params=record['meta']['params']
    a=source_matrix(f,params);b=iv.native_matrix(record['n'],record['lowering']);d=len(a)
    approx=lambda z:complex(float(z.real.mid),float(z.imag.mid))
    i,j=max(((i,j) for i in range(d) for j in range(d)),key=lambda ij:abs(approx(a[ij[0]][ij[1]])))
    phase=float(np.angle(approx(b[i][j])/approx(a[i][j])))
    w=iv.expi(iv.I(-phase));ss=iv.I(0)
    for i in range(d):
        for j in range(d):
            z=a[i][j]-b[i][j]*w;ss+=z.real*z.real+z.imag*z.imag
    return dict(key=record['key'],family=f,params=params,upper=iv.upper(mp.iv.sqrt(ss)),
        common_joint_gauge=phase,method='192-bit outward interval Frobenius >= operator norm; one joint gauge')


def actual_records():
    catalog,_=catalog_bounds();records={};analytic=[]
    for ctx in catalog:
        local=[r for r in ctx['records'] if r['meta']['kind']=='local']
        for r in local:
            f=r['meta']['family']
            if f in ('rzz','crz'):
                a=r['meta']['params'][0]
                expected=([['cx',[0,1],[]],['rz',[1],[a]],['cx',[0,1],[]]] if f=='rzz' else
                    [['rz',[1],[a/2]],['cx',[0,1],[]],['rz',[1],[-a/2]],['cx',[0,1],[]]])
                assert r['lowering']==expected,(ctx['key'],f)
                analytic.append(dict(context=ctx['key'],family=f,param=a,rule='exact native RZZ / CRZ, half-angle is binary scaling'))
                continue
            if f=='crzz':continue
            record={k:r[k] for k in ('meta','n','lowering')}
            key=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
            records[key]=dict(record,key=key)
        # Substitute actual math.atan angles into the compiler's affine CRZZ template.
        cr=[r for r in local if r['meta'].get('family')=='crzz']
        if cr:
            template=cr[0];theta=template['meta']['params'][0]
            def symbolic(r):
                alpha=r['meta']['params'][0];out=copy.deepcopy(r['lowering']);hits=0
                for inst in out:
                    if inst[0]=='rz' and abs(inst[2][0])==abs(alpha/2):
                        inst[2]=['+half' if inst[2][0]==alpha/2 else '-half'];hits+=1
                assert hits==2
                return out
            shape=symbolic(template)
            assert all(symbolic(r)==shape for r in cr)
            for r in (1,2,4,8,16,32):
                tau=common()['preps'][3].exact_rte_lambda_r*.02/r
                for order in (0,2):
                    for sign in (-1,1):
                        a=sign*2*math.atan(tau/(order+1));out=copy.deepcopy(shape)
                        for inst in out:
                            if inst[0]=='rz' and inst[2][0] in ('+half','-half'):
                                inst[2]=[a/2 if inst[2][0]=='+half' else -a/2]
                        record=dict(meta=dict(family='crzz',params=[a]),n=3,lowering=out)
                        key=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
                        records[key]=dict(record,key=key)
        p=next(r for r in ctx['records'] if r['name']=='p_symbolic')
        assert p['lowering']==[['rz',[0],['symbolic_correction']]]
    return list(records.values()),analytic


def I(x=0):return iv.I(x)
def up(x):return iv.upper(x)
def low(x):return iv.lower(x)
def gamma(n):
    nu=I(int(n))*I(2)**-53
    assert up(nu)<1
    return nu/(1-nu)


def phase_bound(q,r):
    """Bound the fixed naive vacuum_phase algorithm; no precision change."""
    p=common()['preps'][3];n=max(1,q//2);tau=I(math.tau)
    half=[common_diagonal(b,.01) for b in p.deterministic_blocks]
    full=common_diagonal(p.deterministic_blocks[0],.02)
    # Boundary one-body fusion is exact binary doubling, including scalar.
    assert float(full.global_phase)==2*float(half[0].global_phase)
    assert list(full.rz)==[(j,2*a) for j,a in half[0].rz]
    assert len(full.rzz)==0
    allbases=[b.runtime_basis_operations for b in p.deterministic_blocks]
    allbases += [d.runtime_operations for d in p.rte_preparation.basis_registry._definitions.values()]
    assert all(sum(g.name=='rz' for g,_ in ops)<=8 for ops in allbases)
    basis_abs=max(up(sum((abs(I(g.params[0])/2) for g,_ in ops if g.name=='rz'),I(0))) for ops in allbases)
    angles=lambda prim:[a for _,a in prim.rz]+[a for _,_,a in prim.rzz]
    drift=2*sum((abs(sum((I(a)/2 for a in angles(z)),I(0))) for z in half),I(0))
    local_diag=max(up(sum((abs(I(a)/2) for a in angles(z)),I(0))) for z in [*half,full])
    max_phi=max(abs(float(z.global_phase)) for z in [*half,full])
    assert max_phi<float(math.tau)
    tau_r=p.exact_rte_lambda_r*.02/r
    max_rot=abs(math.atan(tau_r))
    # Per half: <=3r basis runs/step, 8 RZ per frame, one rotation/event.
    counts=(336+49*r)*n+24
    max_prefix=tau+n*(drift+r*I(max_rot))+2*I(basis_abs)+I(local_diag)
    naive=gamma(counts)*max_prefix
    naive_total=2*naive+gamma(1)*2*(max_prefix+naive)
    period_error=abs(tau-2*mp.iv.pi)
    def rem_bound(x):
        # Standard correctly-rounded IEEE add/sub/remainder model; normalize period is float(2pi).
        return gamma(4)*(abs(x)+2*tau)+(abs(x)/tau+2)*period_error
    scalar=float(-.02*(p.constant_coefficient+p.extracted_identity_coefficient))
    # q/n are powers of two, so scaling the same binary scalar introduces no reassociation rounding.
    assert float(-.02*n*(p.constant_coefficient+p.extracted_identity_coefficient))==n*scalar
    normalization=2*(20*n*rem_bound(2*tau)+n*rem_bound(3*r*I(math.pi))+rem_bound(n*I(abs(scalar))))
    return dict(naive_accumulation_upper=up(naive_total),normalization_vs_controlled_source_upper=up(normalization),
        phase_error_upper=up(naive_total+normalization),RZ_subtractions_per_half_upper=counts,
        exact_prefix_absolute_upper=up(max_prefix),half_steps_upper=n)


def common_diagonal(b,t):
    from .track_c_h4 import diagonal_primitives
    return diagonal_primitives(b,t)


def qualify():
    certificate=json.loads((OUT/'STAGE2_CX_CERTIFICATE.json').read_text())
    if not certificate['pass_to_stage3']:raise RuntimeError('Stage2 not passed')
    records,analytic=actual_records()
    with ProcessPoolExecutor(max_workers=min(len(records),len(os.sched_getaffinity(0)))) as pool:
        results=list(pool.map(interval_one,records))
    save_json(OUT/'STAGE3_LOCAL_INTERVALS.json',dict(bits=BITS,records=results,analytic_RZZ_CRZ_checks=analytic,
        P_rule='P(theta) -> RZ(theta), exact up to one common JOINT phase; actual theta preserved',
        actual_CRZZ_angles='math.atan from fixed SOURCE, affine half-angle template verified in every reachable context'))
    mp.iv.prec=BITS
    errors={}
    for x in results:
        k=(x['family'],tuple(float(v).hex() for v in x['params']))
        errors[k]=max(errors.get(k,0),x['upper'])
    def err(g):
        if g.name in ('rz','cx'):return 0.
        return errors[(g.name,tuple(float(x).hex() for x in g.params))]
    def basis_error(ops):
        return up(sum((I(err(g))+I(err(g.inverse())) for g,_ in ops),I(0)))
    p=common()['preps'][3]
    derror=[basis_error(b.runtime_basis_operations) for b in p.deterministic_blocks]
    tail=max(basis_error(d.runtime_operations) for d in p.rte_preparation.basis_registry._definitions.values())
    family=lambda f:max(x['upper'] for x in results if x['family']==f)
    rots=family('crzz');cz=family('cz');z=family('z');cswap=family('cswap')
    boundary=2*I(family('h'))+I(family('sdg'))
    oldrows=next(c for c in json.loads(data(ROUND_PATH)) if c['name']=='CN_S2_h0.02_ld3')['rows']
    rows=[]
    for oldrow in oldrows:
        q=oldrow['q'];r=oldrow['r'];ph=phase_bound(q,r)
        bD=(q+1)*I(derror[0])+2*q*sum((I(x) for x in derror[1:]),I(0))
        cD=(2 if q==1 else q+2)*I(derror[0])+2*q*sum((I(x) for x in derror[1:]),I(0))
        B=bD+3*q*r*I(tail)+q*r*I(rots)+4*q*r*I(cz)+boundary
        C=cD+3*q*r*I(tail)+4*q*r*I(z)+16*I(cswap)+boundary
        shared=2*mp.iv.sqrt(I(2))*I(oldrow['source_build_and_finite_RTE_operator_upper'])
        shared+=I(oldrow['event_phase_disk_inherited'])+I(oldrow['finite_bit_proposal']['directed_signal_error_upper'])
        nominal_rho=I(oldrow['raw_radius_lower'])+I(oldrow['raw_extra_disk_upper'])
        arms={}
        for name,error,phase in [('Simon_Love',B,I(0)),('reference',C,I(ph['phase_error_upper']))]:
            disk=shared+2*mp.iv.sqrt(I(2))*error+phase
            radius=nominal_rho-disk
            beta=I('.4')-I(oldrow['PF_phase_upper'])-mp.iv.asin(disk/nominal_rho) if up(disk)<low(nominal_rho) else I(-1)
            arms[name]=dict(native_joint_operator_error_upper=up(error),native_raw_disk_upper=up(2*mp.iv.sqrt(I(2))*error),
                phase_generation_extra_raw_disk_upper=up(phase),total_raw_disk_upper=up(disk),
                raw_radius_lower=low(radius),statistical_phase_lower=low(beta),
                all_support_sufficient=low(radius)>0 and low(beta)>0)
        rows.append(dict(m=oldrow['m'],q=q,r=r,phase_arithmetic=ph,old_source_shared_raw_disk_upper=up(shared),
            old_compiler_disk_replaced_not_added=oldrow['compiler_disk_inherited'],
            nominal_raw_radius_lower=low(nominal_rho),arms=arms))
    passed=all(a['all_support_sufficient'] for row in rows for a in row['arms'].values())
    out=dict(status='ALL18_ERROR_QUALIFIED_IN_FIXED_ARITHMETIC_MODEL' if passed else 'UNQUALIFIED_INSUFFICIENT_BOUND',
        pass_to_stage4=bool(passed),rows=rows,
        scope='saved Hermitian H4 matrix model, exact ground supply, ideal native primitive RZ/SX/X/CX, fixed finite-bit proposals',
        arithmetic_assumptions='binary64 round-to-nearest, no overflow/underflow; remainder error bounded by gamma4*(|x|+2tau)+(|x|/tau+2)*|tau-2pi|; 192-bit directed interval gate bounds',
        phase_normalization_source='Qiskit 1.3.0 CircuitData::set_global_phase uses rem_euclid(2*f64::PI)',
        full_space=True,particle_sector_projection=False,
        state_or_E0_used_for_phase=False,old_correctness_suite_rerun=False,
        source_error_reused='old SOURCE/finite-RTE/phase/proposal once; old compiler replaced; new phase generation separately bounded',
        T_synthesis_error='none: arbitrary-angle native gate model, no Clifford+T synthesis')
    save_json(OUT/'STAGE3_ALL_SUPPORT.json',out)
    return out
