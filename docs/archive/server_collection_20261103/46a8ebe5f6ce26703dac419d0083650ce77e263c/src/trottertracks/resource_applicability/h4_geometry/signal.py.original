"""Canonical finite-RTE, paired sampling, saved-value precision/resource maps."""
import math
from .identity import require, Stop, step_seed, trajectory_seed, fingerprint


def distribution(tau,K):
    require(math.isfinite(tau) and type(K) is int and K in (2,4), 'canonical cutoff')
    orders=list(range(0,K+1,2))
    weights=[(1. if n==0 else 0.) if tau==0 else
             math.exp(n*math.log(abs(tau))-math.lgamma(n+1))*math.hypot(1,tau/(n+1)) for n in orders]
    B=math.fsum(weights)
    require(math.isfinite(B) and B>0, 'finite normalization')
    return orders,[w/B for w in weights],B


def sample_events(components,template,parent):
    import numpy as np
    require(components and template['r']>0, 'nonempty random tail')
    lam=math.fsum(c['abs_coefficient'] for c in components)
    p=[c['abs_coefficient']/lam for c in components]
    tau=lam*template['delta']/template['r']
    orders,probs,B=distribution(tau,template['K'])
    used=set();all_events=[]
    def rng(outer,short,occurrence,kind):
        seed=step_seed(parent,outer,short,occurrence,kind)
        require(seed not in used,'duplicate step seed STOP')
        used.add(seed)
        return np.random.Generator(np.random.PCG64(seed))
    for outer in range(template['q']):
        events=[]
        for short in range(template['r']):
            order=int(rng(outer,short,0,'order').choice(orders,p=probs))
            # First draw is rotation; remaining draws are circuit-order products.
            selected=[components[int(rng(outer,short,j,'component').choice(len(components),p=p))]
                      for j in range(order+1)]
            events.append({'order':order,'rotation':selected[0], 'products':selected[1:],
                           'angle':math.atan(tau/(order+1)), 'outer_step':outer,'short_step':short})
        all_events.append(events)
    return all_events,B**(template['q']*template['r'])


def trajectory_seeds(identity):
    seeds=[trajectory_seed(identity,i,actual_inputs_frozen=True,signal_launch=True) for i in range(32)]
    require(len(set(seeds))==32,'duplicate trajectory seeds STOP')
    return seeds


def finite_polynomial(H,tau,K):
    import numpy as np
    total=np.eye(len(H),dtype=complex);term=total.copy()
    for degree in range(1,K+2):
        term=term@H*(-1j*tau/degree);total+=term
    return total


def corrected_signal(deterministic,tail,constant,state,template):
    import numpy as np
    from scipy.linalg import expm
    q,delta=template['q'],template['T']/template['q']
    require(delta==template['delta'], 'delta=T/q')
    half=[expm(-1j*delta/2*H) for H in deterministic]
    current=np.asarray(state,dtype=complex).copy()
    B=1.
    if template['method'] in ('B2','B3'):
        require(tail is not None and template['r']>0,'random tail')
        lam,H=tail;tau=lam*delta/template['r']
        _o,_p,b=distribution(tau,template['K'])
        central=np.linalg.matrix_power(finite_polynomial(H,tau,template['K']),template['r'])
        B=math.exp(q*template['r']*math.log(b))
        require(abs(B-b**(q*template['r']))<=1e-12*max(1,B),'normalization consistency')
    else:
        require(template['r']==0 and template['K']==0, 'baseline shape')
        central=np.eye(len(state),dtype=complex)
    for _ in range(q):
        current*=np.exp(-1j*delta*constant)
        for U in half:
            current=U@current
        current=central@current
        for U in reversed(half):
            current=U@current
    corrected=complex(np.vdot(state,current))
    return {'corrected':corrected,'raw':corrected/B,'normalization':B}


class GeometryPreparation:
    """Private, one-geometry driver cache; never loaded from another run.

    Only method and split rank affect prepare(). The frozen input arrays stay
    read-only while this object is alive. Cached basis circuits are read by the
    builder, which composes copies rather than modifying the basis itself.
    """
    def __init__(self,arrays):
        from types import MappingProxyType
        from .circuits import diagonalize_block
        self._arrays=arrays
        self._inputs={k:arrays[k] for k in ('one','G','lambdas','nuclear','H')}
        for value in self._inputs.values():
            value.setflags(write=False)
        def block(matrix,lam=None):
            value=diagonalize_block(matrix,lam)
            return MappingProxyType({**value,'coefficients':MappingProxyType(dict(value['coefficients']))})
        self.one=block(arrays['one'])
        self.df=[block(g,float(lam)) for lam,g in zip(arrays['lambdas'],arrays['G'],strict=True)]
        self._operators={};self._dense={};self._prepared={}

    def check(self,arrays):
        require(arrays is self._arrays and all(arrays.get(k) is value and not value.flags.writeable
                for k,value in self._inputs.items()),'geometry preparation input changed')

    def operator(self,block):
        from qiskit.quantum_info import Operator
        key=id(block)
        if key not in self._operators:
            value=Operator(block['basis']).data.copy();value.setflags(write=False)
            self._operators[key]=value
        return self._operators[key]

    @staticmethod
    def result(value):
        preparation,det,tail=value
        return ({**preparation,'deterministic':list(preparation['deterministic']),
                 'components':[dict(c) for c in preparation['components']]},list(det),tail)


def prepare(arrays,template,*,common=None):
    if common is None:
        return _prepare(arrays,template)
    common.check(arrays)
    key=(template['method'],template['L_D'])
    require(key[0] in ('B0','B1','B2','B3') and type(key[1]) is int and 0<=key[1]<=len(common.df),
            'geometry preparation template')
    if key not in common._prepared:
        result=_prepare(arrays,template,common=common)
        if result[2] is not None:result[2][1].setflags(write=False)
        common._prepared[key]=result
    return common.result(common._prepared[key])


def _prepare(arrays,template,*,common=None):
    import numpy as np
    from qiskit.quantum_info import Operator
    from .circuits import diagonalize_block
    from .inputs import reverse_bits
    perm=[reverse_bits(i) for i in range(256)]
    one=diagonalize_block(arrays['one']) if common is None else common.one
    df=([diagonalize_block(g,float(lam)) for lam,g in zip(arrays['lambdas'],arrays['G'],strict=True)]
        if common is None else common.df)
    rank=template['L_D'];det=[one,*df[:rank]]
    constant=float(arrays['nuclear']);components=[]
    if template['method'] in ('B2','B3'):
        for block in df[rank:]:
            constant+=block['coefficients'][()]
            for support,c in block['coefficients'].items():
                if support and c!=0:
                    components.append({'block':block,'support':support,'abs_coefficient':abs(c),'sign':1 if c>0 else -1})
        require(bool(components), 'random candidate has empty tail; review required')
    # Deterministic dense action from exactly the same Gaussian diagonal coefficients.
    def dense(block):
        if common is not None and id(block) in common._dense:return common._dense[id(block)]
        U=Operator(block['basis']).data if common is None else common.operator(block)
        diagonal=np.zeros(256)
        for support,c in block['coefficients'].items():
            diagonal+=c*np.asarray([(-1)**sum((i>>p)&1 for p in support) for i in range(256)])
        value=(U*diagonal)@U.conj().T
        if common is not None:
            value.setflags(write=False);common._dense[id(block)]=value
        return value
    det_dense=[dense(b) for b in det]
    lam=math.fsum(c['abs_coefficient'] for c in components)
    tail=np.zeros((256,256),dtype=complex)
    for c in components:
        U=Operator(c['block']['basis']).data if common is None else common.operator(c['block'])
        diagonal=np.asarray([(-1)**sum((i>>p)&1 for p in c['support']) for i in range(256)])
        tail+=c['sign']*c['abs_coefficient']*(U*diagonal)@U.conj().T
    modeled=constant*np.eye(256)+sum(det_dense,np.zeros_like(tail))+tail
    expected=arrays['H'][np.ix_(perm,perm)]
    if template['method']=='B0':
        expected=float(arrays['nuclear'])*np.eye(256)+dense(one)+sum([dense(b) for b in df[:rank]],np.zeros_like(tail))
    require(np.linalg.norm(modeled-expected,2)<=1e-9,'circuit/signal input reconstruction')
    return dict(n=8,deterministic=det,components=components,constant=constant),det_dense,(lam,tail/lam) if components else None


def candidate_identity(input_identity,template,source_hash,compiler,environment):
    from .gaussian_structure import SEMANTICS
    require(input_identity['geometry'] in ('0.70','0.80','0.90','1.10','1.40','1.60'), 'candidate geometry')
    return {'geometry':input_identity['geometry'],**{k:input_identity[k] for k in ('H','DF','state','input')},
            'template':fingerprint('h4-template-v1',template),'source':source_hash,
            'compiler':compiler,'environment':environment,'wrapper_semantics':SEMANTICS}


def epsilon_grid():
    values=[0.005*(0.1/0.005)**(i/300) for i in range(301)]
    values[0],values[-1]=0.005,0.1
    return sorted(set(values+[0.05]))


def shots(B,bias,epsilon):
    require(math.isfinite(B) and B>=1 and math.isfinite(bias) and bias>=0 and math.isfinite(epsilon) and epsilon>0,'shot inputs')
    allowance=epsilon/math.sqrt(2)-bias
    return None if allowance<=0 else math.ceil(2*B*B/(allowance*allowance)*math.log(2/0.025))


def resource_point(saved,epsilon,P=0):
    import numpy as np
    require(math.isfinite(P) and P>=0,'common preparation P')
    counts=[shots(saved['normalization'],saved['bias'][axis],epsilon) for axis in ('cosine','sine')]
    if any(c is None for c in counts):
        return {'epsilon':epsilon,'eligible':False,'shots':None,'work':None,'SE':None,'engineering_interval':None}
    samples=np.asarray(saved['paired_costs'],dtype=float)
    require(samples.shape in ((32,2),(1,2)) and np.all(np.isfinite(samples)) and np.all(samples>=0),'paired logical costs')
    means=np.mean(samples,axis=0);point=float(np.dot(counts,means)+sum(counts)*P)
    se=0.
    if len(samples)==32:
        covariance=np.cov(samples,rowvar=False,ddof=1)
        variance=float(np.asarray(counts)@covariance@np.asarray(counts)/32)
        require(variance>=-1e-9*max(1,point*point),'paired covariance variance')
        se=math.sqrt(max(0.,variance))
    return {'epsilon':epsilon,'eligible':True,'shots':counts,'work':point,'SE':se,
            'engineering_interval':[point-2*se,point+2*se]}


def display_map(saved,P=0):
    # Pure saved-value operation: never samples, builds or compiles.
    return [resource_point(saved,e,P) for e in epsilon_grid()]


def exact_ties(points):
    eligible=[p for p in points if p['work'] is not None]
    if not eligible:
        return []
    minimum=min(p['work'] for p in eligible)
    return [p for p in eligible if p['work']==minimum]
