"""Opt-in relative Pauli-rotation enclosure with an exact Clifford scalar lemma.

No molecular loader/compiler. Plan formation uses exact dyad/pi algebra and
signed binary Clifford tableaux. Peeling is a charged gate approximation, never
an exact-zero assertion. Numerical evaluation needs separate science permission.
"""
from collections import Counter
from fractions import Fraction as F
import math
from . import c_cx_structural_blocks as blocks
from . import c_compiled_raw_local as local
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z
from .c_per_word_population_range import _digest

VERSION='relative_Pauli_charged_peeling_Clifford_scalar_Arb192_v1'
THRESHOLD=F(1,2**40)
PI_SELECTOR=float.fromhex('0x1.921fb54442d18p+1')


def _angle(pi=F(),rad=F()):return dict(pi=str(F(pi)),rad=str(F(rad)))
def _add(a,b):return _angle(F(a['pi'])+F(b['pi']),F(a['rad'])+F(b['rad']))
def _neg(a):return _angle(-F(a['pi']),-F(a['rad']))
def _zero(a):return F(a['pi'])==0 and F(a['rad'])==0


def _approx(a):
    # Selector only;any integer choice is algebraically valid. Exact angles
    # remain in the ledger and are enclosed with Arb,including pi roundoff.
    try:return float(F(a['pi']))*PI_SELECTOR+float(F(a['rad']))
    except OverflowError:return math.inf


def _cliffords():
    from qiskit.quantum_info import Clifford
    H=Clifford.from_label('H');S=Clifford.from_label('S')
    sx=H.compose(S).compose(H)
    return dict(id=Clifford.from_label('I'),x=Clifford.from_label('X'),z=Clifford.from_label('Z'),
        h=H,s=S,sdg=S.adjoint(),sx=sx,sxdg=sx.adjoint(),
        cx=Clifford.from_dict(dict(destabilizer=['+XX','+XI'],stabilizer=['+IZ','+ZZ'])))


def normal_form(expansion,*,collect_rotations=True):
    """U=e^(i phi) F R_m ... R_1, with F the actual declared Clifford word."""
    import numpy as np
    from qiskit.quantum_info import Clifford,Pauli
    record=expansion['record'];local.validate(record);n=record['qubits'];cache=_cliffords()
    inverse=Clifford(np.eye(2*n,dtype=bool));word=[];rotations=[];pi=F();rad=F(expansion['phase_exact_rad'])
    def cliff(name,qs):
        nonlocal inverse
        if name=='id':return
        inverse=inverse.compose(cache[name].adjoint(),qargs=qs,front=True)
        word.append(dict(name=name,qubits=list(qs),params_hex=[]))
    def rz(theta,q):
        nonlocal pi
        k=round(theta/F(PI_SELECTOR/2));delta=_angle(-F(k,2),theta);pi-=F(k,4)
        if collect_rotations and not _zero(delta):
            zz=np.zeros(n,dtype=bool);zz[q]=True
            P=Pauli((zz,np.zeros(n,dtype=bool))).evolve(inverse,frame='s')
            if P.phase not in (0,2):raise ValueError('HERMITIAN_PAULI_CONJUGATION')
            if P.phase==2:delta=_neg(delta)
            x=sum(int(v)<<i for i,v in enumerate(P.x));zz=sum(int(v)<<i for i,v in enumerate(P.z))
            if not (x or zz):raise ValueError('NONTRIVIAL_PAULI_REQUIRED')
            rotations.append(dict(x=x,z=zz,angle=delta))
        name={0:'id',1:'s',2:'z',3:'sdg'}[k%4];cliff(name,[q])
    for g in record['gates']:
        name=g['name'];qs=g['qubits']
        if name in cache:cliff(name,qs)
        elif name in ('rz','ry','p') and len(qs)==1:
            theta=F(float.fromhex(g['params_hex'][0]));q=qs[0]
            if name=='p':rad+=theta/2
            if name=='ry':cliff('sx',qs)
            rz(theta,q)
            if name=='ry':cliff('sxdg',qs)
        else:raise ValueError('EXPANDED_ONEQ_CX_REQUIRED')
    return dict(rotations=rotations,rotations_collected=collect_rotations,clifford_record=dict(fixed.identity(n),gates=word),
        inverse_tableau=inverse.tableau.astype(int).tolist(),phase=_angle(pi,rad),
        theorem='exact RY=SXdg RZ SX;P=e^(i theta/2) RZ;RZ=e^(-ik pi/4) S^k RZ(theta-k pi/2);Heisenberg transport')


def _key(r):return r['x'],r['z']
def _commute(a,b):return ((a['x']&b['z']).bit_count()+(a['z']&b['x']).bit_count())%2==0


def reduce_relative(rotations,threshold=THRESHOLD):
    """Exact commuting/fusion plus explicitly charged removal of small rotations."""
    if not isinstance(threshold,F) or not 0<=threshold<=1:raise ValueError('CHARGED_PEEL_POLICY')
    seq=list(rotations);peeled=[];centers=[];center_pi=0;passes=0;fusions=0
    def center(r):
        nonlocal center_pi
        approx=_approx(r['angle']);k=round(approx/(2*PI_SELECTOR)) if math.isfinite(approx) else 0
        if not k:return r
        angle=_add(r['angle'],_angle(-2*k));center_pi+=k
        centers.append(dict(original=r['angle'],centered=angle,compensating_pi=k))
        return dict(r,angle=angle)
    def discard(r):
        if _zero(r['angle']):return True
        approx=_approx(r['angle'])
        if math.isfinite(approx) and abs(approx)<=float(threshold):
            peeled.append(r);return True
        return False
    while True:
        passes+=1;out=[];initial=len(seq)
        for r in seq:
            r=center(r)
            if discard(r):continue
            j=len(out)-1
            while j>=0 and _commute(r,out[j]):
                if _key(r)==_key(out[j]):break
                j-=1
            if j>=0 and _key(r)==_key(out[j]):
                previous=out.pop(j);r=center(dict(r,angle=_add(previous['angle'],r['angle'])));fusions+=1
                if discard(r):continue
                out.insert(j,r)
            else:out.append(r)
        if len(out)==initial:return dict(remaining=out,peeled=peeled,center_pi=center_pi,centers=centers,passes=passes,fusions=fusions)
        seq=out


def make_plan(left,right,*,saved=None,threshold=THRESHOLD):
    saved=local.plan(left,right) if saved is None else saved
    blocks.verify_saved_plan(saved,left,right)
    forms=[normal_form(e) for e in saved['expansions']]
    C,R=forms;equal=C['inverse_tableau']==R['inverse_tableau']
    relative=C['rotations']+[dict(r,angle=_neg(r['angle'])) for r in reversed(R['rotations'])]
    reduced=reduce_relative(relative,threshold)
    cliff=dict(fixed.identity(left['qubits']),gates=C['clifford_record']['gates']+fixed.adjoint(R['clifford_record'])['gates'])
    cliff,proof=blocks.exact_rewrite(cliff)
    # Reuse only registered exact Clifford commutations;no numerical synthesis.
    sk=local.skeleton(dict(record=cliff),commute_CX=True)
    gates=[]
    for event in sk['events']:
        for q,body in zip(event['gate']['qubits'],event['blocks']):gates.extend(dict(g,qubits=[q]) for g in body)
        gates.append(event['gate'])
    for q,body in enumerate(sk['terminal']):gates.extend(dict(g,qubits=[q]) for g in body)
    cliff=dict(cliff,gates=gates)
    phase=_add(C['phase'],_neg(R['phase']));phase=_add(phase,_angle(reduced['center_pi']))
    value=dict(version=VERSION,qubits=left['qubits'],left_id=_digest(left),right_id=_digest(right),
        primitive_plan_id=saved['plan_id'],threshold_exact=str(threshold),normal_forms=forms,
        relative_clifford=cliff,relative_reduction=reduced,phase=phase,projective_Clifford_equal=equal,
        applicable=equal,clifford_rewrite=proof,
        theorem='signed generator equality implies FRdag FC=gI;one-column Arb encloses g;charged relative Pauli telescope',
        no_Hamiltonian_target_substitution=True)
    return saved,dict(value,plan_id=_digest(value))


def _rotation_job(angle):
    from flint import ctx,arb
    with z._BALL_LOCK,ctx.workprec(192):
        theta=z._real(F(angle['pi']))*arb.pi()+z._real(F(angle['rad']))
        upper=z._fraction((2*(theta/4).sin()).abs_upper())
        return dict(upper_exact=str(min(F(2),upper)),precision_bits=192,rounding_enclosed=True,
            theorem='nonidentity Hermitian Pauli:||exp(-i theta P/2)-I||2=2|sin(theta/4)|')


def _scalar_job(plan):
    from flint import ctx
    if not plan['projective_Clifford_equal']:raise ValueError('EXACT_SCALAR_LEMMA_REQUIRED')
    with z._BALL_LOCK,ctx.workprec(192):
        v,error=fixed._column(plan['relative_clifford'],0,{})
        phase=local._phase(F(plan['phase']['pi']),F(plan['phase']['rad']))
        upper=z._fraction((phase*v[0,0]-1).abs_upper())+error
        return dict(upper_exact=str(min(F(2),upper)),column=0,columns_evaluated=1,
            propagation_radius_exact=str(error),precision_bits=192,rounding_enclosed=True,
            exact_scalar_lemma=True,finite_probe_operator_promotion=False,phase_fit=False)


def evaluate(left,right,*,saved=None,plan=None,parallel=True,threshold=THRESHOLD):
    z.runtime_model()
    actual_saved,actual=make_plan(left,right,saved=saved,threshold=threshold)
    if plan is not None and actual!=plan:raise ValueError('COMPLETE_RELATIVE_PLAN_IDENTITY')
    saved,plan=actual_saved,actual
    if not plan['applicable']:return dict(status='NOT_APPLICABLE_PROJECTIVE_CLIFFORD',upper_exact=None,
        full_operator_coverage=False,dense_fallback_used=False,dimension=1<<left['qubits'])
    definitions={};dc=Counter()
    for e in saved['expansions']:
        for key,d in e['definitions'].items():
            if key in definitions and {k:v for k,v in d.items() if k!='multiplicity'}!={k:v for k,v in definitions[key].items() if k!='multiplicity'}:
                raise ValueError('PRIMITIVE_DEFINITION_CONFLICT')
            definitions[key]=d;dc[key]+=d['multiplicity']
    dw,dr=local._map(local._definition_job,definitions.values(),parallel);dw=dict(zip(definitions,dw))
    angles={};counts=Counter()
    for r in plan['relative_reduction']['peeled']+plan['relative_reduction']['remaining']:
        key=_digest(r['angle']);angles[key]=r['angle'];counts[key]+=1
    aw,ar=local._map(_rotation_job,angles.values(),parallel);aw=dict(zip(angles,aw))
    scalar=_scalar_job(plan)
    translation=sum((F(dw[k]['upper_exact'])*dc[k] for k in dw),F())
    rotation=sum((F(aw[k]['upper_exact'])*counts[k] for k in aw),F())
    upper=min(F(2),translation+rotation+F(scalar['upper_exact']))
    return dict(status='VALIDATED_RELATIVE_PAULI_NATIVE_PAIR',version=VERSION,kind='rigorous_enclosure',
        norm='full_operator_2',dimension=1<<left['qubits'],full_operator_coverage=True,
        precision_bits=192,rounding_enclosed=True,phase_fit=False,sector_projection=False,
        left_id=plan['left_id'],right_id=plan['right_id'],plan_id=plan['plan_id'],threshold_exact=str(threshold),
        upper_exact=str(upper),lower_exact='0',translation_upper_exact=str(translation),
        rotation_upper_exact=str(rotation),scalar_upper_exact=scalar['upper_exact'],
        scalar_witness=scalar,rotation_witnesses=aw,rotation_multiplicities=dict(counts),
        definition_witnesses=dw,definition_multiplicities=dict(dc),
        definition_resources=dr,rotation_resources=ar,
        scalar_columns_evaluated=1,complete_basis_columns_evaluated=0,
        exact_projective_scalar_lemma=True,finite_probe_operator_promotion=False,
        charged_rotation_occurrences=sum(counts.values()),remaining_rotations=len(plan['relative_reduction']['remaining']),
        peeled_rotations=len(plan['relative_reduction']['peeled']),dense_fallback_used=False,
        nonunitary_literal_target_supported=False,all_word_uniform=False,calibration_ready=False,Stage_C_ready=False,actual_CA_achieved=None)


def compare(left,right,**kwargs):return evaluate(left,right,**kwargs)


def check_binding(cert,left,right):
    local.validate(left);local.validate(right)
    if (cert.get('status')!='VALIDATED_RELATIVE_PAULI_NATIVE_PAIR' or cert.get('version')!=VERSION
            or cert.get('left_id')!=_digest(left) or cert.get('right_id')!=_digest(right)
            or left['qubits']!=right['qubits'] or cert.get('dimension')!=1<<left['qubits']
            or cert.get('precision_bits')!=192 or cert.get('rounding_enclosed') is not True
            or any(cert.get(k) is not False for k in ('phase_fit','sector_projection','finite_probe_operator_promotion',
                'dense_fallback_used','nonunitary_literal_target_supported','all_word_uniform','calibration_ready','Stage_C_ready'))
            or cert.get('full_operator_coverage') is not True or cert.get('exact_projective_scalar_lemma') is not True
            or cert.get('scalar_columns_evaluated')!=1 or cert.get('complete_basis_columns_evaluated')!=0
            or cert.get('actual_CA_achieved') is not None):raise ValueError('RELATIVE_CERTIFICATE_SCOPE')
    terms=[F(cert[k]) for k in ('translation_upper_exact','rotation_upper_exact','scalar_upper_exact')]
    if min(terms)<0 or F(cert['upper_exact'])!=min(F(2),sum(terms,F())) or F(cert['lower_exact'])!=0:
        raise ValueError('RELATIVE_TELESCOPE_COMPOSITION')
    return True
