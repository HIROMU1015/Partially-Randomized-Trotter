"""Phase-preserving two-system-qubit native IR. No synthesis or cost scoring."""
from dataclasses import dataclass
from fractions import Fraction as F
from .model import AXES, pauli_mul, reduce_indices


@dataclass(frozen=True)
class Angle:
    kind: str
    value: F
    scale: F=F(1)
    def __post_init__(self):
        if self.kind not in ('atan','pi'):raise ValueError('unsupported angle')
        object.__setattr__(self,'value',F(self.value));object.__setattr__(self,'scale',F(self.scale))
    @property
    def key(self):return f'{self.kind}:{self.value}:scale:{self.scale}'
    def neg(self):return Angle(self.kind,self.value,-self.scale)


@dataclass(frozen=True)
class Gate:
    name: str
    wires: tuple=()
    angle: object=None
    phase: int=0


def invert(g):
    if g.name=='RZ':return Gate('RZ',g.wires,g.angle.neg())
    if g.name=='S':return Gate('Sdg',g.wires)
    if g.name=='Sdg':return Gate('S',g.wires)
    if g.name=='GLOBAL':return Gate('GLOBAL',phase=(-g.phase)%4)
    if g.name not in ('H','X','Y','Z','CX'):raise ValueError('unknown exact inverse')
    return g


def adjoint(circuit):return [invert(g) for g in reversed(circuit)]


def basis_xx():
    return [Gate('H',(0,)),Gate('H',(1,)),Gate('CX',(0,1)),
            Gate('RZ',(1,),Angle('pi',F(1,8))),Gate('CX',(0,1)),Gate('H',(0,)),Gate('H',(1,))]


def pauli_gates(axis,controlled):
    out=[]
    for wire,p in enumerate(axis):
        if p=='I':continue
        if not controlled:out.append(Gate(p,(wire,)))
        elif p=='X':out.append(Gate('CX',(2,wire)))
        elif p=='Z':out += [Gate('H',(wire,)),Gate('CX',(2,wire)),Gate('H',(wire,))]
        elif p=='Y':out += [Gate('Sdg',(wire,)),Gate('CX',(2,wire)),Gate('S',(wire,))]
        else:raise ValueError('unknown Pauli axis')
    return out


def pauli_rotation(axis,angle,controlled):
    active=[j for j,p in enumerate(axis) if p!='I']
    if not active:raise ValueError('identity rotation needs explicit scalar handling')
    basis=[]
    for wire in active:
        if axis[wire]=='X':basis.append(Gate('H',(wire,)))
        if axis[wire]=='Y':basis += [Gate('Sdg',(wire,)),Gate('H',(wire,))]
    target=active[-1]; parity=[Gate('CX',(wire,target)) for wire in active[:-1]]
    if controlled:
        half=Angle(angle.kind,angle.value,angle.scale/2)
        middle=[Gate('RZ',(target,),half),Gate('CX',(2,target)),
                Gate('RZ',(target,),half.neg()),Gate('CX',(2,target))]
    else:middle=[Gate('RZ',(target,),angle)]
    return basis+parity+middle+adjoint(parity)+adjoint(basis)


def involution(context,index,controlled,angle=None):
    if isinstance(index,str):axis=index;basis=[]
    elif context in AXES:axis=AXES[context][index];basis=[]
    elif context=='distinct_basis' and index in (0,1):
        axis=('ZI','IZ')[index];basis=basis_xx() if index==1 else []
    else:raise ValueError('no registered involution implementation')
    middle=pauli_gates(axis,controlled) if angle is None else pauli_rotation(axis,angle,controlled)
    # V first in circuit time, V dagger last gives V dagger P V.
    return basis+middle+adjoint(basis)


def simplify(circuit):
    """Common exact adjacent inverse cancellation. No angle arithmetic fusion.

    Equality is exact symbolic equality, including signed basis RZ(pi/8).
    No cancellation across sampled events or distinct stochastic occurrences.
    """
    out=[]
    for g in circuit:
        if out and invert(g)==out[-1]:out.pop()
        else:out.append(g)
    return out


def lower_event(context,event,controlled):
    word=event.word;phase=event.phase;sign=event.rotation_sign
    q=event.b/event.a if event.b else F(0)
    if event.complement:
        if not event.a or not event.b:raise ValueError('degenerate complement')
        q=event.a/event.b;sign=-sign;phase=(phase-event.rotation_sign)%4
        word=word+(event.rotation,)
    if context in AXES:
        axis='II'
        for ref in word:
            axis,extra=pauli_mul(ref if isinstance(ref,str) else AXES[context][ref],axis)
            phase=(phase+extra)%4
        word=() if axis=='II' else (axis,)
    else:word=reduce_indices(word)
    circuit=[]
    if phase:
        if controlled:
            circuit += {1:[Gate('S',(2,))],2:[Gate('Z',(2,))],3:[Gate('Sdg',(2,))]}[phase]
        else:circuit.append(Gate('GLOBAL',phase=phase))
    for ref in word:circuit += involution(context,ref,controlled)
    if q:
        angle=Angle('atan',q,F(2*sign))
        circuit += involution(context,event.rotation,controlled,angle)
    return simplify(circuit)


def planned_angles(xs):
    from .model import cts_ls
    angles={}
    for raw in xs:
        x=F(raw);rho=(x+x**3/6)/(1+x*x/2)
        for q in (x,x/3,rho,cts_ls('pauli_commuting',x),cts_ls('pauli_noncommuting',x)):
            for scale in (-2,-1,1,2):
                a=Angle('atan',q,F(scale));angles[a.key]=a
    for sign in (-1,1):
        a=Angle('pi',F(1,8),F(sign));angles[a.key]=a
    return dict(sorted(angles.items()))
