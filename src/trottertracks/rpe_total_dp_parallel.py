"""Exact outcome-wise aggregation and process parallelism for total-state DP.

State propagation and weighted exact sums are accelerated. The original
decoder, residual geometry, frontier selection, moment definitions and gates
are reused from rpe_total_risk.total_state_dp. No running job is modified.
"""
import ast
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import inspect
import hashlib
import math
import multiprocessing
import os
import textwrap
from unittest.mock import patch

from . import rpe_total_risk as R

CONTEXT = None
Mass = namedtuple('Mass', 'numerator denominator')
REFERENCE_SOURCE='710aa69b663ef3313439762d25c80a652a03948f'
REFERENCE_DP_AST_SHA256='3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8'


def _accumulate(res,key,mass,coefficient):
    """LCM denominator accumulation; normalize only when a value is read.

    Avoid repeatedly gcd-reducing huge probability numerators. This supports
    arbitrary exact rational coefficients, including the original loss caps.
    """
    if not coefficient or not mass.numerator:return
    coefficient=Fraction(coefficient)
    n=mass.numerator*coefficient.numerator;d=mass.denominator*coefficient.denominator
    current=res[key]
    if isinstance(current,Fraction):current=Mass(current.numerator,current.denominator)
    if current.numerator:
        if d==current.denominator:
            n+=current.numerator
        else:
            g=math.gcd(d,current.denominator)
            n=current.numerator*(d//g)+n*(current.denominator//g)
            d=d*(current.denominator//g)
    res[key]=Mass(n,d)
    res.setdefault('_cached',{}).pop(key,None)


def _res_value(res,key):
    current=res[key]
    if isinstance(current,Fraction):return current
    cache=res.setdefault('_cached',{})
    if key not in cache:cache[key]=Fraction(current.numerator,current.denominator)
    return cache[key]


def _terminal_moments(states,denominator,h,target):
    """Same binary64 energies and exact squared loss; one final reduction/sum."""
    sums={key:Fraction(0) for key in ('mse','m1','m2','eta','flagged')}
    for (flag,state),mass in states.items():
        w=Mass(mass,denominator);energy=R.a.exact_scalar(-float.fromhex(state)/h)
        loss=(energy-target)**2
        _accumulate(sums,'mse',w,loss);_accumulate(sums,'m1',w,energy)
        _accumulate(sums,'m2',w,energy*energy)
        _accumulate(sums,'flagged' if flag else 'eta',w,1 if flag else loss)
    return tuple(_res_value(sums,key) for key in ('mse','m1','m2','eta','flagged'))


def _initialize(context):
    global CONTEXT
    CONTEXT = context


def _outcomes(bounds, context=None):
    """Sum previous integer masses before multiplying by one outcome weight.

    For an outcome j, all prior states reaching the SAME (flag,binary64 phase)
    have the SAME integer factor w_j. Thus sum_i mass_i*w_j = w_j*sum_i mass_i.
    All additions/multiplications stay arbitrary-precision integers.
    """
    import numpy as np
    flags, estimates, masses, groups, q, m = CONTEXT if context is None else context
    new={}; zero_mass=tie_mass=0; worst=None
    for zero,phase,weight in groups[bounds[0]:bounds[1]]:
        principal=np.full(len(estimates),phase); zeros=np.full(len(estimates),zero,dtype=bool)
        if m:
            values,ties=R.T.step(principal,estimates,q,zeros)
            if not np.all(np.isfinite(values)):
                return None,0,0,None,'NONFINITE_TOTAL_DECODER_TRANSITION'
            if np.any(values < -math.pi) or np.any(values > math.pi):
                return None,0,0,None,'TOTAL_DECODER_OUTSIDE_PRINCIPAL_RANGE'
            move=np.abs(values-estimates);move=np.minimum(move,2*math.pi-move)
            excess=float(np.max(move-math.pi/q));worst=excess if worst is None else max(worst,excess)
        else:
            values=R.T.initial(principal,zeros);ties=np.zeros(len(estimates),dtype=bool)
        if zero:
            zero_mass+=int(np.sum(masses))*weight
        elif np.any(ties):
            tie_mass+=int(np.sum(masses[ties]))*weight
        next_flags=flags | zero | ties
        bits=values.view(np.uint64)
        for flag in (False,True):
            indices=np.flatnonzero(next_flags==flag)
            if not len(indices):continue
            indices=indices[np.argsort(bits[indices],kind='stable')]
            ordered=bits[indices]
            starts=np.r_[0,1+np.flatnonzero(ordered[1:]!=ordered[:-1])]
            sums=np.add.reduceat(masses[indices],starts)
            for bit,mass in zip(ordered[starts],sums):
                key=(flag,int(bit));new[key]=new.get(key,0)+int(mass)*weight
    return new,zero_mass,tie_mass,worst,None


def _phase_hex(bits):
    import struct
    return struct.unpack('=d',struct.pack('=Q',bits))[0].hex()


class Accelerator:
    """One callable with configurable computational workers, not scientific scope.

    workers=None uses available CPU affinity, bounded by the number of tasks.
    Small automatically selected tasks stay serial to avoid process overhead.
    outcome_batch controls scheduling granularity only, not retained mass.
    """
    def __init__(self, *, workers=None, outcome_batch=8):
        if workers is not None:R.S._integer(workers,'workers',1)
        R.S._integer(outcome_batch,'outcome_batch',1)
        self.workers=workers;self.outcome_batch=outcome_batch;self.observations=[]
        self.kernel=self._kernel()

    def _kernel(self):
        # Reuse the original DP body, delegating transitions and exact weighted
        # sums. Geometric bounds, pruning and moment equations stay in one place.
        if not inspect.isfunction(R.total_state_dp):
            raise ValueError('Nested/concurrent acceleration contexts are not supported')
        tree=ast.parse(textwrap.dedent(inspect.getsource(R.total_state_dp)))
        if hashlib.sha256(ast.dump(tree.body[0],include_attributes=False).encode()).hexdigest()!=REFERENCE_DP_AST_SHA256:
            raise ValueError('Reference total DP SOURCE changed; do not silently adapt')
        replacement=ast.parse("""
new,pairs,zero_mass,tie_mass,worst,error = _propagate(states,groups,table['q'],m,pairs,maximum_pairs,previous,drop,worst)
if error is not None:
    out['reason']=error
    return out
""").body
        class Replace(ast.NodeTransformer):
            count=0;terminal_count=0;drop_count=0
            def visit_FunctionDef(self,node):
                if node.name=='drop':
                    # Keep the original geometry expressions; only replace its
                    # weighted rational sums by deferred exact accumulation.
                    for v in ast.walk(node):
                        if (isinstance(v,ast.AugAssign) and isinstance(v.target,ast.Subscript)
                                and getattr(v.target.value,'id',None)=='res'
                                and v.target.slice.value!='events'):
                            self.drop_count+=1
                    class Sums(ast.NodeTransformer):
                        def visit_AugAssign(self,v):
                            if (not isinstance(v.target,ast.Subscript)
                                    or getattr(v.target.value,'id',None)!='res' or v.target.slice.value=='events'):
                                return v
                            expression=v.value
                            if isinstance(expression,ast.Name) and expression.id=='mass':coefficient=ast.Constant(1)
                            elif isinstance(expression,ast.BinOp) and getattr(expression.left,'id',None)=='mass':
                                coefficient=expression.right
                            elif isinstance(expression,ast.IfExp) and getattr(expression.orelse,'id',None)=='mass':
                                coefficient=ast.IfExp(test=expression.test,body=expression.body,orelse=ast.Constant(1))
                            elif isinstance(expression,ast.IfExp) and getattr(expression.body,'id',None)=='mass':
                                coefficient=ast.IfExp(test=expression.test,body=ast.Constant(1),orelse=expression.orelse)
                            else:raise ValueError('Original residual expression changed')
                            return ast.Expr(ast.Call(func=ast.Name('_accumulate',ast.Load()),
                                args=[ast.Name('res',ast.Load()),v.target.slice,ast.Name('mass',ast.Load()),coefficient],keywords=[]))
                    node=Sums().visit(node)
                return self.generic_visit(node)
            def visit_Subscript(self,node):
                if (getattr(node.value,'id',None)=='res' and isinstance(node.ctx,ast.Load)
                        and isinstance(node.slice,ast.Constant) and node.slice.value!='events'):
                    return ast.Call(func=ast.Name('_res_value',ast.Load()),args=[node.value,node.slice],keywords=[])
                return self.generic_visit(node)
            def visit_Call(self,node):
                if (getattr(node.func,'id',None)=='drop' and len(node.args)==4
                        and isinstance(node.args[2],ast.Call) and getattr(node.args[2].func,'id',None)=='Fraction'):
                    node.args[2].func.id='Mass'
                return self.generic_visit(node)
            def visit_For(self,node):
                target=node.target
                if (isinstance(target,ast.Tuple) and len(target.elts)==2
                        and isinstance(node.iter,ast.Call) and getattr(node.iter.func,'id',None)=='sorted'
                        and isinstance(target.elts[0],ast.Tuple)
                        and [getattr(n,'id',None) for n in target.elts[0].elts]==['flag','state']
                        and getattr(target.elts[1],'id',None)=='mass'):
                    self.count+=1;return replacement
                if (isinstance(target,ast.Tuple) and len(target.elts)==2
                        and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Attribute)
                        and node.iter.func.attr=='items' and getattr(node.iter.func.value,'id',None)=='states'
                        and isinstance(target.elts[0],ast.Tuple)
                        and [getattr(n,'id',None) for n in target.elts[0].elts]==['flag','state']
                        and getattr(target.elts[1],'id',None)=='mass'):
                    self.terminal_count+=1
                    return ast.parse('mse,m1,m2,eta,flagged = _terminal_moments(states,denominator,h,E)').body
                return self.generic_visit(node)
        transform=Replace();tree=transform.visit(tree)
        if (transform.count,transform.terminal_count,transform.drop_count)!=(1,1,8):
            raise ValueError('DP transition/residual/moment body changed; acceleration needs review')
        scope=dict(R.__dict__);scope['_propagate']=self.propagate
        scope.update(_accumulate=_accumulate,_res_value=_res_value,_terminal_moments=_terminal_moments,Mass=Mass)
        exec(compile(ast.fix_missing_locations(tree),'rpe_total_dp_parallel:original_DP_body','exec'),scope)
        return scope['total_state_dp']

    def propagate(self,states,groups,q,m,pairs,maximum_pairs,previous,drop,worst):
        import numpy as np
        ordered=sorted(states.items(),key=lambda kv:(-kv[1],str(kv[0])))
        # The original loop consumes whole state rows in this exact order.
        # Preselect the SAME prefix; all omitted states use the original drop.
        count=min(len(ordered),max(0,(maximum_pairs-pairs)//len(groups)))
        active=ordered[:count]
        for (flag,state),mass in ordered[count:]:
            drop(flag,state,Mass(mass,previous),m-1)
        if not active:return {},pairs,0,0,worst,None
        flags=np.array([key[0] for key,mass in active],dtype=bool)
        estimates=np.array([0. if key[1] is None else float.fromhex(key[1]) for key,mass in active])
        masses=np.array([mass for key,mass in active],dtype=object)
        context=flags,estimates,masses,groups,q,m
        batches=[(i,min(i+self.outcome_batch,len(groups))) for i in range(0,len(groups),self.outcome_batch)]
        available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
        workers=min(self.workers or available,len(batches))
        if self.workers is None and len(active)*len(groups)<100000:workers=1
        self.observations.append(dict(m=m,active_states=count,omitted_states=len(ordered)-count,
            outcome_groups=len(groups),transition_pairs=count*len(groups),workers=workers,batches=len(batches)))
        results=None;pool=None
        if workers>1:
            method='fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context(method),
                initializer=_initialize,initargs=(context,))
            results=pool.map(_outcomes,batches)
        else:
            results=(_outcomes(batch,context) for batch in batches)
        new={};zero_mass=tie_mass=0
        try:
            for values,z,t,excess,error in results:
                if error is not None:return {},pairs,0,0,worst,error
                for (flag,bits),mass in values.items():
                    key=flag,_phase_hex(bits);new[key]=new.get(key,0)+mass
                zero_mass+=z;tie_mass+=t
                if excess is not None:worst=excess if worst is None else max(worst,excess)
        finally:
            if pool is not None:pool.shutdown(wait=True)
        return new,pairs+count*len(groups),zero_mass,tie_mass,worst,None

    def __call__(self,tables,*,h,target,scope):
        self.observations=[]
        return self.kernel(tables,h=h,target=target,scope=scope)


@contextmanager
def installed(*,workers=None,outcome_batch=8):
    """Explicit opt-in in THIS process only; never patches an external worker."""
    engine=Accelerator(workers=workers,outcome_batch=outcome_batch)
    with patch.object(R,'total_state_dp',engine):yield engine
