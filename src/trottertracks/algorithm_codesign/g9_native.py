"""Specified three-qubit provider, exact Q(sqrt(2)) collection and literal gates."""
from fractions import Fraction as F
from math import factorial
from .return_aggregation import Interval, root_interval, dyadic_distribution


class A:
    def __init__(self,a=0,b=0):self.a,self.b=F(a),F(b)
    def __add__(self,z):
        z=alg(z);return A(self.a+z.a,self.b+z.b)
    __radd__=__add__
    def __neg__(self):return A(-self.a,-self.b)
    def __sub__(self,z):return self+-alg(z)
    def __mul__(self,z):
        z=alg(z);return A(self.a*z.a+2*self.b*z.b,self.a*z.b+self.b*z.a)
    __rmul__=__mul__
    def __truediv__(self,z):return self*F(1,F(z))
    def __bool__(self):return bool(self.a or self.b)
    def __eq__(self,z):z=alg(z);return (self.a,self.b)==(z.a,z.b)
    def interval(self):
        rt=root_interval(F(2),256)
        return Interval(self.a+self.b*(rt.lo if self.b>=0 else rt.hi),self.a+self.b*(rt.hi if self.b>=0 else rt.lo))
    def json(self):return [str(self.a),str(self.b)]


def alg(z):return z if isinstance(z,A) else A(z)
TIMES={('X','Y'):('Z',1),('Y','X'):('Z',3),('Y','Z'):('X',1),('Z','Y'):('X',3),('Z','X'):('Y',1),('X','Z'):('Y',3)}


def pauli_product(a,b):
    result='';phase=0
    for x,y in zip(a,b):
        if x=='I':z,q=y,0
        elif y=='I':z,q=x,0
        elif x==y:z,q='I',0
        else:z,q=TIMES[x,y]
        result+=z;phase+=q
    return result,phase%4


def poly_product(a,b):
    out={}
    for (x,p),c in a.items():
        for (y,q),d in b.items():
            axis,phase=pauli_product(x,y);phase=(phase+p+q)%4
            key=axis,phase%2;out[key]=out.get(key,A())+(-1 if phase>=2 else 1)*c*d
    return {k:v for k,v in out.items() if v}


def provider_polynomials():
    c=A(0,F(1,2))
    return [{('ZII',0):A(1)},{('IZI',0):c,('XYI',0):c},
            {('IIZ',0):c,('IXY',0):A(F(1,2)),('ZYY',0):A(F(-1,2))}]


def exact_target(p,x):
    Q=provider_polynomials();R={}
    for pi,q in zip(p,Q):
        for k,v in q.items():R[k]=R.get(k,A())+F(pi)*v
    power={('III',0):A(1)};target={('III',0):A(1)}
    for n in range(1,6):
        power=poly_product(R,power)
        phase=(-n)%4;factor=F(x)**n/factorial(n)
        for (axis,j),v in power.items():
            ph=(phase+j)%4;key=axis,ph%2
            target[key]=target.get(key,A())+(-1 if ph>=2 else 1)*factor*v
    return {k:v for k,v in target.items() if v},Q


def cts_events(p,x,H=160,K=256,eta=F(1,10**12),rho=F(1,10**12)):
    target,_=exact_target(p,x);real=[];odd=[];mean_enclosure_error=F(0)
    for (axis,phase),v in sorted(target.items()):
        if phase==0:
            v=v-A(axis=='III')
            if not v:continue
        z=v.interval()
        if z.lo<=0<=z.hi:raise ArithmeticError('CTS sign unresolved')
        sign=1 if z.lo>0 else -1;absz=Interval(sign*(z.lo if sign>0 else z.hi),sign*(z.hi if sign>0 else z.lo))
        midpoint=absz.midpoint;mean_enclosure_error+=(absz.hi-absz.lo)/2
        (real if phase==0 else odd).append((axis,sign,midpoint,absz))
    if any(axis=='III' for axis,_,_,_ in odd):raise ArithmeticError('scalar odd phase not in registered native template')
    L=sum(v for _,_,v,_ in odd);norm=root_interval(1+L*L,K)
    if norm.hi-norm.lo>2*rho*norm.lo:raise ArithmeticError('CTS norm precision')
    # norm-midpoint / sqrt(1+L²) differs by at most rho, affecting I+odd part.
    mean_enclosure_error+=rho*(1+L)
    if mean_enclosure_error>8*rho:raise ArithmeticError('common coefficient bias exceeded')
    events=[]
    for axis,sign,coefficient,_ in real:
        events.append({'pauli':axis,'coefficient':coefficient,'phase_i_power':0 if sign>0 else 2,'ratio':F(0),'rotation_sign':0})
    for axis,sign,coefficient,_ in odd:
        events.append({'pauli':axis,'coefficient':norm.midpoint*coefficient/L,'phase_i_power':0,'ratio':L,'rotation_sign':-sign})
    B=sum(e['coefficient'] for e in events)
    law=dyadic_distribution(tuple(Interval(e['coefficient']/B,e['coefficient']/B) for e in events),H,eta)
    for e,q in zip(events,law):e.update(proposal=q,weight=e['coefficient']/q)
    return events,{'literal_finite_Theorem1_specialization':True,'first_operator_moment_not_channel':True,
        'identity_even_correction_kept_separate':True,'target_exact_Qsqrt2':{k[0]+':'+str(k[1]):v.json() for k,v in target.items()},
        'rational_rotation_tangent':str(L),'coefficient_mean_error_upper':str(mean_enclosure_error),
        'normalizer_rational':str(B),'Pauli_access':'explicit cheap I1 collection in this synthetic provider context'}


def inverse(g):
    name,*w=g
    if name=='R':return ('R',w[0],w[1],-w[2])
    return ({'T':'t','t':'T','S':'s','s':'S','W':'w','w':'W'}.get(name,name),*w)


def adjoint(gates):return [inverse(g) for g in reversed(gates)]


def simplify(gates):
    out=[]
    for g in gates:
        if out and inverse(out[-1])==g:out.pop()
        else:out.append(g)
    return out


def basis(axis):
    out=[]
    for i,P in enumerate(axis):
        if P=='X':out.append(('H',i))
        elif P=='Y':out.extend([('s',i),('H',i)])
    return out


def parity(axis):
    support=[i for i,p in enumerate(axis) if p!='I']
    return [('CX',i,support[-1]) for i in support[:-1]]


def exact_rotation(axis):
    support=[i for i,p in enumerate(axis) if p!='I'];B=basis(axis);ladder=parity(axis)
    # Global e^(i*pi/8) vs R_P(pi/4) cancels only within V ... V†.
    return B+ladder+[('T',support[-1])]+adjoint(ladder)+adjoint(B)


def V(label):
    return [] if label==0 else exact_rotation('XXI') if label==1 else exact_rotation('ZZI')+exact_rotation('IXX')


def CQ(label,control=3):
    v=V(label);return v+[('CZ',control,label)]+adjoint(v)


def cpauli(axis,control=3):
    out=[]
    for i,p in enumerate(axis):
        if p=='X':out.append(('CX',control,i))
        elif p=='Z':out.append(('CZ',control,i))
        elif p=='Y':out.extend([('s',i),('CX',control,i),('S',i)])
    return out


def crot(axis,ratio,sign=1,control=3):
    support=[i for i,p in enumerate(axis) if p!='I'];t=support[-1];B=basis(axis);ladder=parity(axis)
    return B+ladder+[('R',t,str(ratio),sign),('CX',control,t),('R',t,str(ratio),-sign),('CX',control,t)]+adjoint(ladder)+adjoint(B)


def native_ir(event,helper=False):
    gates=[('Z',3)] if event['phase_i_power']==2 else []
    if 'pauli' in event:
        gates+=cpauli(event['pauli']) if not event['rotation_sign'] else crot(event['pauli'],event['ratio'],event['rotation_sign'])
    else:
        for i in reversed(event['word']):gates+=CQ(i)
        child=event['child']
        if helper:
            gates += [('H',4)]+CQ(child,4)+[('H',4),('R',4,str(event['ratio']),1),('CX',3,4),('R',4,str(event['ratio']),-1),('CX',3,4),('H',4)]+CQ(child,4)+[('H',4)]
        else:
            v=V(child);axis=''.join('Z' if i==child else 'I' for i in range(3))
            gates+=v+crot(axis,event['ratio'])+adjoint(v)
    lowered=[]
    for gate in gates:
        if gate[0]=='CZ':lowered.extend([('H',gate[2]),('CX',gate[1],gate[2]),('H',gate[2])])
        else:lowered.append(gate)
    return simplify(lowered)


def cost(gates,cache):
    counts={'T':0,'CX':0,'CZ':0,'1Q':0}
    for g in gates:
        if g[0]=='R':
            row=cache[g[2]];counts['T']+=row['T_count'];counts['1Q']+=row['one_qubit_count']
        elif g[0] in ('CX','CZ'):counts[g[0]]+=1
        else:counts['1Q']+=g[0] not in ('W','w');counts['T']+=g[0] in ('T','t')
    # CZ = H CX H, same exact common gate basis for every arm.
    counts['1Q']+=2*counts['CZ'];counts['CX']+=counts.pop('CZ')
    counts['strict_error_upper']=2*F(1,10**6) if any(g[0]=='R' for g in gates) else F(0)
    return counts
