"""Coefficient-only C1 rules. Exact arithmetic, no spectral oracle or optimizer."""
from fractions import Fraction as F
from .c1_macrocycle_algebra_v1 import (REPRESENTATIVES, kernel, kernel_error,
    macrocycle, macro_error, catalog, canonical_stages)

PAULI = {
    ('I','I'): ('I',0), ('I','X'): ('X',0), ('I','Y'): ('Y',0), ('I','Z'): ('Z',0),
    ('X','I'): ('X',0), ('Y','I'): ('Y',0), ('Z','I'): ('Z',0),
    ('X','X'): ('I',0), ('Y','Y'): ('I',0), ('Z','Z'): ('I',0),
    ('X','Y'): ('Z',1), ('Y','Z'): ('X',1), ('Z','X'): ('Y',1),
    ('Y','X'): ('Z',3), ('Z','Y'): ('X',3), ('X','Z'): ('Y',3)}


def pauli_product(a, b):
    if len(a) != len(b): raise ValueError('PAULI_LENGTH')
    chars, phase = [], 0
    for x,y in zip(a,b):
        z,p = PAULI[x,y]; chars.append(z); phase += p
    return ''.join(chars), phase % 4


def cmul(a,b):
    return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])


def sparse_product(a,b):
    out = {}
    for x,u in a.items():
        for y,v in b.items():
            z,p = pauli_product(x,y)
            w = cmul(cmul(u,v), ((F(1),F(0)),(F(0),F(1)),
                                 (F(-1),F(0)),(F(0),F(-1)))[p])
            old = out.get(z,(F(0),F(0)))
            out[z] = (old[0]+w[0],old[1]+w[1])
    return {z:w for z,w in out.items() if w != (0,0)}


def substitute(poly, fragments):
    n = len(next(iter(fragments['A'])))
    out = {}
    for word,c in poly.items():
        prod = {'I'*n:(F(c),F(0))}
        for name in word:
            prod = sparse_product(prod,{p:(F(v),F(0)) for p,v in fragments[name].items()})
        for p,(r,i) in prod.items():
            a,b = out.get(p,(F(0),F(0))); out[p]=(a+r,b+i)
    if any(i for r,i in out.values()): raise ValueError('NONHERMITIAN_ERROR_POLYNOMIAL')
    return {p:r for p,(r,i) in out.items() if r}


def expectation(poly, bitstring):
    total = F(0)
    for p,c in poly.items():
        if not any(x in 'XY' for x in p):
            total += c * (-1)**sum(b=='1' and x=='Z' for b,x in zip(bitstring,p))
    return total


def l1(poly): return sum(map(abs,poly.values()),F(0))


def analytic_bound(Lam,gamma,dnorm,width,index):
    """Exact rational coefficient/gap bound, not a floating point certificate."""
    h=F(1)/(8*Lam*2**index)
    if gamma is None or h*h*dnorm>gamma/4: return None
    K=F(2,3)*(4*Lam)**5
    return h*h*width+h**4*(K+2*dnorm*dnorm/gamma)


def validate_input(row):
    if set(row) != {'id','qubits','fragments','information_states','diagonal_gap'}:
        raise ValueError('INPUT_FIELDS_OR_ORACLE')
    if row['qubits'] != 2 or set(row['fragments']) != {'A','B','C'}:
        raise ValueError('FIXED_THREE_FRAGMENT_TWO_QUBIT_SCOPE')
    fragments = {}
    for name,terms in row['fragments'].items():
        if not terms or len(terms)>4: raise ValueError('FRAGMENT_SIZE')
        parsed = {}
        for p,c in terms.items():
            if len(p)!=2 or any(x not in 'IXYZ' for x in p): raise ValueError('PAULI_SCHEMA')
            if not isinstance(c,str) or len(c)>32: raise ValueError('RATIONAL_SCHEMA')
            value=F(c)
            if not value: raise ValueError('ZERO_TERM')
            parsed[p]=value
        for p in parsed:
            for q in parsed:
                if pauli_product(p,q)[1] != pauli_product(q,p)[1]:
                    raise ValueError('WITHIN_FRAGMENT_NONCOMMUTING')
        fragments[name]=parsed
    for j,k in (('A','B'),('A','C'),('B','C')):
        comm = sparse_product({p:(v,F(0)) for p,v in fragments[j].items()},
                              {p:(v,F(0)) for p,v in fragments[k].items()})
        rev = sparse_product({p:(v,F(0)) for p,v in fragments[k].items()},
                             {p:(v,F(0)) for p,v in fragments[j].items()})
        if comm==rev: raise ValueError('COMMUTING_FRAGMENTS')
    if row['information_states'] != ['00','11']: raise ValueError('INFORMATION_STATE_SCOPE')
    # This input class makes the A gap obtainable from literal diagonal coefficients.
    A=fragments['A']
    if set(A) != {'II','ZI','IZ','ZZ'} or any(A[p]>=0 for p in ('ZI','IZ','ZZ')):
        raise ValueError('DIAGONAL_REFERENCE_CLASS')
    a,b,j=(-A[p] for p in ('ZI','IZ','ZZ'))
    if F(row['diagonal_gap']) != 2*min(a+j,b+j,a+b): raise ValueError('DIAGONAL_GAP')
    return fragments


def information(row,fragments,state):
    gammaA=F(row['diagonal_gap']); v=l1(fragments['B'])+l1(fragments['C'])
    certified=state=='00' and 2*v<gammaA
    return {'state':state,'d_rho':min(F(2),2*v/(gammaA-v)) if certified else F(2),
            'gamma_lower':gammaA-2*v if certified else None,
            'source':'COEFFICIENT_PERTURBATION' if certified else 'TRIVIAL_TRACE_DISTANCE',
            'state_measurement_shots':0,'state_preparation_X':state.count('1'),
            'classical_information_is_toy_specific':True}


def native_counts(stages,fragments,periodic=True):
    """Exact adjacent sharing; controlled Pauli rotations, no compile or T estimate."""
    stages=list(canonical_stages(stages))
    if periodic and len(stages)>1 and stages[0][0]==stages[-1][0]: stages.pop()
    rz=cx=basis=0
    for name,_ in stages:
        for p in fragments[name]:
            k=sum(x!='I' for x in p)
            rz+=2 if k else 1
            cx+=4*k-2 if k else 0
            basis+=2*p.count('X')+4*p.count('Y')
    return {'RZ':rz,'CX':cx,'basis_Clifford':basis,'depth':None,'T':None,'ancilla':0,
            'controlled_global_scalar_phase_included':True}


def methods():
    out=[]
    for j,order in enumerate(REPRESENTATIVES):
        for m in (1,2,3,4):
            stages=tuple((p,t/F(m)) for _ in range(m) for p,t in kernel(order))
            out.append({'id':f'S{j}_m{m}','kind':'SINGLE' if m==1 else 'SUBDIVISION',
                        'stages':stages,'D':{w:v/F(m*m) for w,v in kernel_error(order).items()},
                        'j':j,'m':m})
    for row in catalog():
        j,k,a=row['j'],row['k'],F(row['a'])
        out.append({'id':row['id'],'kind':'MACRO','stages':macrocycle(j,k,a),
                    'D':macro_error(j,k,a),'j':j,'k':k,'a':a,
                    'known_uniform_mixture':a==F(1,3)})
    return out


def select(row,price):
    """Seal coefficient-only choices before any reference eigensolver is imported."""
    fragments=validate_input(row)
    if price not in (3,6): raise ValueError('FIXED_PRICE_HYPOTHESES')
    result={'input_id':row['id'],'price_CX_per_RZ':price,'states':{}}
    for state in row['information_states']:
        info=information(row,fragments,state); rows=[]
        for method in methods():
            D=substitute(method['D'],fragments); d=expectation(D,state)
            n=native_counts(method['stages'],fragments)
            g=n['RZ']+n['basis_Clifford']+price*n['CX']
            joint=abs(d)+info['d_rho']*l1(D)
            box=joint
            if method['kind']=='MACRO':
                a,j,k=method['a'],method['j'],method['k']
                dj=substitute(kernel_error(REPRESENTATIVES[j]),fragments)
                dk=substitute(kernel_error(REPRESENTATIVES[k]),fragments)
                box=abs(d)+info['d_rho']*(2*a**3*l1(dj)+(1-2*a)**3*l1(dk))
            rows.append({'id':method['id'],'kind':method['kind'],'d_approx':str(d),
                         'D_l1':str(l1(D)),'joint_width':str(joint),'box_width':str(box),
                         'J2_empirical':str(g*g*abs(d)),'J2_joint':str(g*g*joint),
                         'J2_box':str(g*g*box),'counts':n,'g_hypothesis':g,
                         'D_Pauli_terms':len(D),
                         'max_coefficient_bit_length':max((max(v.numerator.bit_length(),v.denominator.bit_length()) for v in D.values()),default=0),
                         'known_uniform_mixture':method.get('known_uniform_mixture',False)})
        # Same information, dictionary, precision and sharing for each comparator.
        pools={'best_single':lambda r:r['kind']=='SINGLE',
               'uniform_order_mixture':lambda r:r['known_uniform_mixture'],
               'C1':lambda r:True,'strong_known_composition':lambda r:True}
        choices={}
        for score in ('empirical','joint','box'):
            choices[score]={name:min((r for r in rows if pred(r)),
                                    key=lambda r:(F(r['J2_'+score]),r['id']))['id']
                            for name,pred in pools.items()}
        Lam=sum(l1(terms) for terms in fragments.values())
        precision_choices={}
        for epsilon in (F(1,100000),F(1,1000000)):
            precision_choices[str(epsilon)]={}
            for r in rows:
                eligible=[index for index in range(6) if
                    (value:=analytic_bound(Lam,info['gamma_lower'],F(r['D_l1']),F(r['joint_width']),index)) is not None
                    and value<=epsilon/2]
                precision_choices[str(epsilon)][r['id']]=min(eligible) if eligible else None
        result['states'][state]={'information':{k:str(v) if isinstance(v,F) else v
                                               for k,v in info.items()},
                                 'candidates':rows,'choices':choices,
                                 'coefficient_only_precision_choices':precision_choices,
                                 'strong_known_composition_equals_C1_by_construction':True}
    return result
