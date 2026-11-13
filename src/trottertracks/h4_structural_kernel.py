"""Fixed H4 K0/K1: source parity first, then four ideal-law mean S values.

No trajectories, full-path S counter, circuit construction, compiler, fit or G.
Native Z arithmetic is sequential binary64; only probability algebra is dense.
"""
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import h4_pr_physical_closure as c
from .mean_compiled_price import MeanKernel, MASS_TOLERANCE

BASE='a637ed1b813dc3952339cb05f36beb79168ace23'
B_LATEST='f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8'
COUNTER='8457c443554af5fadaa3ba36f97f6316ca3c5050'
COUNTER_PATH='src/trottertracks/h4_df10_price_v1.py'
OLD='docs/research/h4_pr_physical_closure_20261113/'
SOURCE_FILES=('src/trottertracks/h4_structural_kernel.py','scripts/run_h4_structural_kernel.py',
              'src/trottertracks/h4_pr_physical_closure.py','src/trottertracks/mean_compiled_price.py',
              'src/trottertracks/s0a_structure_context.py',COUNTER_PATH)
U=np.finfo(float).eps/2
TINY=float(np.nextafter(0.,1.))
ACTIVE=None


def zkey(z):return None if z is None else float(z).hex()
def zvalue(z):return None if z is None else float.fromhex(z)


def integer_weights(values):
    ratios=[float(v).as_integer_ratio() for v in values]
    denominator=max(b for a,b in ratios)
    weights=[a*(denominator//b) for a,b in ratios]
    if any(w<0 for w in weights) or sum(weights)<=0:raise ValueError('POSITIVE_PROBABILITY_WEIGHTS')
    return weights,sum(weights)


def fraction_bounds(f):
    x=float(f);v=Fraction.from_float(x)
    return (float(np.nextafter(x,-np.inf)) if v>f else x,
            float(np.nextafter(x,np.inf)) if v<f else x)


def rounded(x,inner):
    """Outward nonnegative IEEE dot bound, gamma_(2*inner), incl. underflow."""
    gamma=fraction_bounds(Fraction(2*inner,2**53-2*inner))[1]
    eta=2*inner*TINY
    denominator_hi=np.nextafter(1+gamma,np.inf);denominator_lo=np.nextafter(1-gamma,-np.inf)
    return (np.maximum(0.,np.nextafter(np.nextafter(x/denominator_hi,-np.inf)-eta,-np.inf)),
            np.nextafter(np.nextafter(x/denominator_lo,np.inf)+eta,np.inf))


def dot_bounds(al,ah,bl,bh):
    return rounded(al@bl,al.shape[-1])[0],rounded(ah@bh,ah.shape[-1])[1]


def add_bounds(al,ah,bl,bh):
    return rounded(al+bl,1)[0],rounded(ah+bh,1)[1]


class Band:
    """Existing MeanKernel plus outward probability/reward error enclosure."""
    def __init__(self,central,pl,ph,gl,gh):
        self.central=central
        self.pl=np.minimum(pl,central.transition);self.ph=np.maximum(ph,central.transition)
        self.gl=np.minimum(gl,central.reward);self.gh=np.maximum(gh,central.reward)
        if any(not np.isfinite(a).all() for a in (self.pl,self.ph,self.gl,self.gh)):
            raise ValueError('NONFINITE_NUMERICAL_ENCLOSURE')

    @classmethod
    def identity(cls,states):
        m=MeanKernel.identity(states);return cls(m,m.transition,m.transition,m.reward,m.reward)

    def then(self,b):
        pl,ph=dot_bounds(self.pl,self.ph,b.pl,b.ph)
        gl,gh=dot_bounds(self.pl,self.ph,b.gl,b.gh)
        gl,gh=add_bounds(self.gl,self.gh,gl,gh)
        return Band(self.central.then(b.central),pl,ph,gl,gh)

    def repeat(self,count):
        if type(count) is not int or count<0:raise ValueError('NONNEGATIVE_REPETITIONS')
        result=Band.identity(self.central.states);base=self
        while count:
            if count&1:result=result.then(base)
            count>>=1
            if count:base=base.then(base)
        return result

    def expected(self,index,terminal):
        p=np.zeros(len(self.central.states));p[index]=1.
        value=self.central.expected(p,terminal)[0]
        l,h=dot_bounds(self.pl,self.ph,terminal,terminal)
        l,h=add_bounds(self.gl,self.gh,l,h)
        lo=float(l[index]);hi=float(h[index])
        if not lo<=value<=hi:raise ValueError('MEAN_OUTSIDE_ENCLOSURE')
        return value,lo,hi


def discrete(states,nominal,raw_weights,transition):
    """Do not repair nominal source probabilities; enclose ideal weight ratios."""
    weights,total=integer_weights(raw_weights);lookup={s:i for i,s in enumerate(states)}
    table=[[transition(s,j) for j in range(len(weights))] for s in states]
    m=MeanKernel.from_events(states,range(len(weights)),nominal,lambda s,j:table[lookup[s]][j])
    pl=np.zeros_like(m.transition);ph=pl.copy();gl=np.zeros_like(m.reward);gh=gl.copy()
    for i,row in enumerate(table):
        nums=[0]*len(states);reward=0
        for j,(end,cost) in enumerate(row):
            nums[lookup[end]]+=weights[j];reward+=weights[j]*cost
        for j,num in enumerate(nums):pl[i,j],ph[i,j]=fraction_bounds(Fraction(num,total))
        gl[i],gh[i]=fraction_bounds(Fraction(reward,total))
    return Band(m,pl,ph,gl,gh)


def mixture(a,b,nominal,raw_weights):
    weights,total=integer_weights(raw_weights)
    m=MeanKernel(a.central.states,nominal[0]*a.central.transition+nominal[1]*b.central.transition,
                 nominal[0]*a.central.reward+nominal[1]*b.central.reward)
    lows=[];highs=[];rl=[];rh=[]
    for k,v in enumerate((a,b)):
        lo,hi=fraction_bounds(Fraction(weights[k],total))
        lo=min(lo,nominal[k]);hi=max(hi,nominal[k])
        lows.append(rounded(lo*v.pl,1)[0]);highs.append(rounded(hi*v.ph,1)[1])
        rl.append(rounded(lo*v.gl,1)[0]);rh.append(rounded(hi*v.gh,1)[1])
    pl,ph=add_bounds(lows[0],highs[0],lows[1],highs[1]);gl,gh=add_bounds(rl[0],rh[0],rl[1],rh[1])
    return Band(m,pl,ph,gl,gh)


def source_tail_functions(root):
    tree=ast.parse(c.blob(root,COUNTER,COUNTER_PATH))
    A=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Accounting')
    tail=next(n for n in A.body if isinstance(n,ast.FunctionDef) and n.name=='tail')
    event_loop=next(n for n in tail.body if isinstance(n,ast.For))
    app_loop=next(n for n in event_loop.body if isinstance(n,ast.For))
    # Preserve the whole original application body, including its validation.
    loop=ast.For(target=app_loop.target,iter=ast.parse('[(i, app)]',mode='eval').body,body=app_loop.body,orelse=[])
    def function(name,args,body):
        f=ast.parse('def '+name+'('+args+'):\n pass').body[0];f.body=body+ast.parse('return active, phase').body
        return f
    f=function('one_app','self, active, phase, app, i, unsigned',[loop])
    g=function('close','self, active, phase',tail.body[tail.body.index(event_loop)+1:])
    scope={'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=[f,g],type_ignores=[])),COUNTER+':tail_parts','exec'),scope)
    return scope['one_app'],scope['close']


def signature(word):
    return tuple((o['name'],tuple(o['wires']),zkey(o['angle'])) for o in word)


class Model:
    def __init__(self,root,context,saved):
        self.A=context.Accounting;self.Trace=c.tracer(self.A);self.blocks=context.blocks[0];self.scalar=context.scalar
        prep=context.ctx.prep;self.components=prep.rte_preparation.symbolic_tail.components
        if prep.preparation_hash!=saved['preparation']['hash'] or len(self.components)!=324:raise ValueError('SAVED_PREPARATION_HASH')
        self.registry=[];self.keys=[];self.components_meta=[]
        for comp,old in zip(self.components,saved['preparation']['components'],strict=True):
            key=(comp.basis_id,comp.basis_hash)
            native_ops=[dict(name=o.name,qubits=list(o.qubits),parameters=list(o.parameters)) for o in comp.basis_change_operations]
            # Original C1 calls e.to_dict() before JSON encoding. asdict retains
            # tuples and matrix_sha256=None; do not replace these with head lists.
            ops=tuple(asdict(o) for o in comp.basis_change_operations)
            if key not in self.keys:self.keys.append(key);self.registry.append(ops)
            b=self.keys.index(key)
            old_basis=next(v for v in saved['preparation']['basis_registry'] if (v['id'],v['hash'])==key)
            if (native_ops!=old_basis['operations'] or comp.component_id!=old['id'] or comp.coefficient_abs!=old['coefficient_abs']
                or comp.probability.hex()!=old['probability_hex'] or list(comp.diagonal_pauli_support)!=old['support'] or comp.coefficient_sign!=old['sign']):
                raise ValueError('SAVED_COMPONENT_ORDER_OR_COEFFICIENT')
            self.components_meta.append(dict(b=b,support=tuple(comp.diagonal_pauli_support),sign=comp.coefficient_sign))
        if len(self.registry)!=9:raise ValueError('NINE_BASES')
        self.one_app,self.close=source_tail_functions(root)
        self.laws=saved['laws'];self.raw=[v.probability for v in self.components]
        self.nominal=[p/math.fsum(self.raw) for p in self.raw]
        self.switch={};self.app={};self.closing={};self.source_parity=0
        self.prepare_words()

    def active(self,b):return None if b<0 else (self.keys[b],self.registry[b])

    def initial(self,b):
        t=self.Trace()
        if b<0:t.pending_basis=(self.blocks[-1]['basis'],False,'head_basis')
        return t

    def prepare_words(self):
        for old in range(-1,9):
            for new in range(9):
                t=self.initial(old)
                if old!=new:
                    if old>=0:t.basis(self.registry[old],False,'tail_basis')
                    t.basis(self.registry[new],True,'tail_basis')
                t.flush_basis();self.switch[old,new]=t.word
        for law in self.laws:
            R=law['R_outer']
            for j,meta in enumerate(self.components_meta):
                for role,order in (('product',2),('rotation',0),('rotation',2)):
                    angle=meta['sign']*law['rotation_angles'][str(order)]
                    key=(R,j,role,order)
                    word=c.application_word(self.Trace,meta['support'],role,angle)
                    self.app[key]=word
                    app=self.app_dict(j,role,0 if role=='product' or order==0 else 2)
                    for flag in (False,True):
                        t=self.initial(meta['b']);active,phase=self.one_app(t,self.active(meta['b']),math.pi if flag else 0.,app,app['application_index'],angle)
                        if signature(t.word)!=signature(word) or active[0]!=self.keys[meta['b']]:raise ValueError('SOURCE_APPLICATION_WORD_PARITY')
                        if bool(phase)!=bool(flag or (role=='product' and meta['sign']<0)):raise ValueError('SOURCE_PHASE_STEP_PARITY')
                        self.source_parity+=1
        # Basis branch is independent of role/support; check every old/new pair.
        for old in range(-1,9):
            for new in range(9):
                j=next(i for i,v in enumerate(self.components_meta) if v['b']==new)
                t=self.initial(old);app=self.app_dict(j,'product',0)
                self.one_app(t,self.active(old),0.,app,0,0.)
                expected=self.switch[old,new]+self.app[self.laws[0]['R_outer'],j,'product',2]
                if signature(t.word)!=signature(expected):raise ValueError('SOURCE_BASIS_FLUSH_PARITY')
                self.source_parity+=1
        t=self.Trace();t.diagonal(self.blocks[0]['diagonal'],True,'head_directional')
        eff=c.effects(t.word)
        if not all(e['reset'] for e in eff):raise ValueError('OUTER_EXIT_NOT_FIXED')
        self.exit=[e['exit_z'] for e in eff]
        self.prefix={}
        for first in (True,False):
            t=self.Trace()
            if first:t.gate('h',(8,),role='wrapper')
            else:t.pending_basis=(self.blocks[0]['basis'],False,'head_basis')
            for block in self.blocks:t.block(block,False)
            if t.pending_basis[:2]!=(self.blocks[-1]['basis'],False):raise ValueError('HEAD_ENTRY_QUEUE')
            self.prefix[first]=t.word
        for b in range(-1,9):
            for flag in (False,True):
                t=self.initial(b);self.close(t,self.active(b),math.pi if flag else 0.)
                for block in reversed(self.blocks):t.block(block,True)
                if t.pending_basis[:2]!=(self.blocks[0]['basis'],False):raise ValueError('REVERSE_HEAD_QUEUE')
                control=[o for o in t.word if 8 in o['wires']]
                leading=[]
                for op in control:
                    if op['name']!='rz':break
                    leading.append(op['angle'])
                if leading!=([math.pi] if flag else []):raise ValueError('PHASE_CONTROL_PREFIX_PARITY')
                self.closing[b,flag]=t.word
        self.suffix={}
        for law in self.laws:
            for axis in ('cosine','sine'):
                t=self.Trace();t.pending_basis=(self.blocks[0]['basis'],False,'head_basis')
                T=.1 if law['R_outer']==2 else 6.4
                if self.scalar*T!=0:t.gate('p',(8,),(-self.scalar*T,),'scalar_once')
                if axis=='sine':t.gate('sdg',(8,),role='wrapper')
                t.gate('h',(8,),role='wrapper');t.gate('measure',(8,),role='measurement');t.flush_basis()
                self.suffix[law['R_outer'],axis]=t.word

    def app_dict(self,j,role,index):
        v=self.components_meta[j]
        return dict(application_index=index,is_identity=False,coefficient_sign=v['sign'],role=role,
            basis_id=self.keys[v['b']][0],basis_hash=self.keys[v['b']][1],basis_change_operations=self.registry[v['b']],
            diagonal_pauli_support=list(v['support']))


class Wire:
    def __init__(self,model,q):self.m=model;self.q=q;self.cache={};self.parity=0

    def step_word(self,word,z):
        local=[o for o in word if self.q in o['wires']];key=(signature(local),zkey(z))
        if key not in self.cache:
            predicted=c.projected_step(z,local,self.q)
            a=self.m.A();a.pending_z[self.q]=z
            for op in local:a._native(op['name'],op['wires'],op['angle'])
            actual=(a.pending_z[self.q],a.folded._counts.get('rz',0))
            if (zkey(predicted[0]),predicted[1])!=(zkey(actual[0]),actual[1]):raise ValueError('EXACT_LOCAL_REWARD_OR_Z_PARITY')
            self.cache[key]=predicted;self.parity+=1
        return self.cache[key]

    def application(self,state,j,role,order,R):
        b,z,flag=state;meta=self.m.components_meta[j]
        z,a=self.step_word(self.m.switch[b,meta['b']],zvalue(z))
        z,d=self.step_word(self.m.app[R,j,role,order],z)
        return (meta['b'],zkey(z),bool(flag or (role=='product' and meta['sign']<0))),a+d

    def build(self,law):
        R=law['R_outer'];entry=[];prefix=[]
        for first in (True,False):
            z,cost=self.step_word(self.m.prefix[first],None if first else self.m.exit[self.q])
            entry.append((-1,zkey(z),False));prefix.append(cost)
        states=list(dict.fromkeys(entry));cursor=0
        # Finite enclosure from reset proof, not a CPU/worker or ad hoc search cap.
        values={s[1] for s in states}|{None}
        for b in range(9):values.add(zkey(c.wire_effect(c.basis_word(self.m.Trace,self.m.registry[b],True),self.q)['exit_z']))
        for j in range(324):
            for role,order in (('product',2),('rotation',0),('rotation',2)):
                e=c.wire_effect(self.m.app[R,j,role,order],self.q)
                if e['reset']:values.add(zkey(e['exit_z']))
        bound=2*10*len(values);seen=set(states)
        while cursor<len(states):
            s=states[cursor];cursor+=1;ends=[(s[0],s[1],True)]
            for j in range(324):
                for role,order in (('product',2),('rotation',0),('rotation',2)):
                    ends.append(self.application(s,j,role,order,R)[0])
            for end in ends:
                if end[1] not in values:raise ValueError('PENDING_Z_ENCLOSURE_FAILED')
                if end not in seen:seen.add(end);states.append(end)
            if len(states)>bound:raise ValueError('FINITE_STATE_BOUND_FAILED')
        states=tuple(states)
        role_kernels=[]
        for role,order in (('product',2),('rotation',0),('rotation',2)):
            role_kernels.append(discrete(states,self.m.nominal,self.m.raw,lambda s,j:self.application(s,j,role,order,R)))
        product,r0,r2=role_kernels
        phase=discrete(states,[1.],[1.],lambda s,j:((s[0],s[1],True),0))
        event=mixture(r0,phase.then(product).then(product).then(r2),law['order_probabilities'],law['order_weights'])
        terminal=[]
        for s in states:
            z,cost=self.step_word(self.m.closing[s[0],s[2]],zvalue(s[1]))
            if zkey(z)!=zkey(self.m.exit[self.q]):raise ValueError('OUTER_EXIT_DEPENDS_ON_HISTORY')
            terminal.append(cost)
        suffix={}
        for axis in ('cosine','sine'):
            z,cost=self.step_word(self.m.suffix[R,axis],self.m.exit[self.q])
            suffix[axis]=cost+int(z is not None and z!=0.)
            # Isolated source terminal flush, never an actual full trajectory.
            a=self.m.A();a.pending_z[self.q]=z;a._flush_z(self.q)
            if a.folded._counts.get('rz',0)!=int(z is not None and z!=0.):raise ValueError('TERMINAL_FLUSH_PARITY')
        # K0: propagate probability only; no physical mean reward evaluation yet.
        zero=Band(MeanKernel(states,event.central.transition,np.zeros(len(states))),event.pl,event.ph,np.zeros(len(states)),np.zeros(len(states)))
        propagated=zero.repeat(R)
        entry_columns=[i for i,s in enumerate(states) if s[0]<0]
        if np.any(propagated.central.transition[:,entry_columns]!=0.):raise ValueError('UNREACHED_ENTRY_TERMINAL_MASS')
        deviation=float(np.max(np.abs(propagated.central.transition.sum(axis=1)-1)))
        if deviation>MASS_TOLERANCE:raise ValueError('PROBABILITY_MASS_TOLERANCE')
        ml,mh=dot_bounds(propagated.pl,propagated.ph,np.ones(len(states)),np.ones(len(states)))
        if np.any(ml>1) or np.any(mh<1):raise ValueError('IDEAL_MASS_OUTSIDE_ENCLOSURE')
        high=np.linalg.matrix_power(event.central.transition.astype(np.longdouble),R)
        if np.any(high<propagated.pl) or np.any(high>propagated.ph):raise ValueError('LONGDOUBLE_PROBABILITY_CHECK')
        return dict(kernel=event,entry_indices=[states.index(v) for v in entry],prefix=prefix,
            terminal=np.array(terminal,dtype=float),suffix=suffix,
            metadata=dict(wire=self.q,R=R,state_count=len(states),proved_enclosure=bound,
                exact_local_table_parities=self.parity,max_probability_mass_defect=deviation,
                longdouble_probability_check=True,all_terminal_exits_fixed=True,
                prefix_reward=prefix,suffix_terminal_reward=suffix,full_path_S_evaluations=0))


def worker(q):
    w=Wire(ACTIVE,q);return q,[w.build(law) for law in ACTIVE.laws]


def law_metadata_parity(context,model):
    """Finite sign templates, not all component tuples or any trajectory."""
    positive=next(i for i,c in enumerate(model.components) if c.coefficient_sign>0)
    negative=next(i for i,c in enumerate(model.components) if c.coefficient_sign<0)
    maximum=0.;checks=0;gamma=16*U/(1-16*U)
    for law in model.laws:
        dist=SimpleNamespace(orders=(0,2),unnormalized_order_weights=tuple(law['order_weights']),
            dimensionless_step_time=law['tau'],exact_finite_distribution=law['event_normalization'])
        for rotation in range(324):
            for products in ((),(positive,positive),(positive,negative),(negative,positive),(negative,negative)):
                order=len(products);indices=(rotation,*products)
                event=context.rte._make_event(indices,model.components,dist,order//2)
                payload=event.to_dict()
                for app,i in zip(payload['application_sequence'],(*products,rotation),strict=True):
                    expected_ops=model.registry[model.components_meta[i]['b']]
                    if type(app['basis_change_operations']) is not type(expected_ops) or app['basis_change_operations']!=expected_ops:
                        raise ValueError('ORIGINAL_TO_DICT_BASIS_REPRESENTATION_PARITY')
                expected=law['order_probabilities'][order//2]*math.prod(model.nominal[i] for i in indices)
                error=abs(event.event_probability-expected)/event.event_probability
                maximum=max(maximum,error);checks+=1
                if (event.rotation_angle.hex()!=law['rotation_angles'][str(order)].hex()
                    or event.phase!=complex((-1)**(order//2))
                    or event.selected_component_ids!=tuple(model.components[i].component_id for i in (*products,rotation))):
                    raise ValueError('SOURCE_EVENT_ORDER_PHASE_ANGLE_PARITY')
                ratio=abs(Fraction.from_float(event.event_probability)-Fraction.from_float(expected))/Fraction.from_float(event.event_probability)
                if ratio>Fraction(16,2**53-16):raise ValueError('PROBABILITY_FACTORIZATION_ROUNDOFF_BOUND')
    return dict(metadata_records_checked=checks,max_relative_probability_factorization_difference=maximum,
        derived_relative_bound=gamma,full_tuple_enumeration=False,trajectory_sampling=False)


def check_inputs(root):
    plan,old=c.check_publication(root)
    official=c.read(root,B_LATEST,c.B_PATH)
    if official['id']!=plan['B_V2_configuration_id']:raise ValueError('LATEST_B_CONFIGURATION')
    rows={r['m']:r for r in official['rounds']}
    for request in plan['focus_requests']:
        for field in ('identity','law_binding','physical_candidate'):
            if rows[request['m']][field]!=request[field]:raise ValueError('LATEST_B_REQUEST_BINDING')
    saved=c.read(root,BASE,OLD+'run02/RESULT.json');proof=c.read(root,BASE,OLD+'REDUCTION.json')
    if not all(proof['premises'].values()):raise ValueError('CLOSURE_PREMISES_NOT_CONFIRMED')
    return plan,saved,dict(commit=B_LATEST,path=c.B_PATH,configuration_id=official['id'])


def run_K0(root,source,progress):
    global ACTIVE
    if float(np.nextafter(0.,1.)*2)==0.:raise ValueError('GRADUAL_UNDERFLOW_REQUIRED')
    for path in SOURCE_FILES:
        if (Path(root)/path).read_bytes()!=c.blob(root,source,path):raise ValueError('FROZEN_SOURCE_CHANGED:'+path)
    plan,saved,B=check_inputs(root)
    from .s0a_saved_structure import saved as c1_saved
    from .s0a_structure_context import Context,no_circuits
    with no_circuits():context=Context(root,c1_saved(root,'PLAN.json'),'H4')
    if float(context.scalar).hex()!=saved['scalar']['hex']:raise ValueError('SAVED_SCALAR_BINDING')
    ACTIVE=Model(root,context,saved)
    law_checks=law_metadata_parity(context,ACTIVE)
    workers=min(9,len(os.sched_getaffinity(0)));results={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(worker,q) for q in range(9)]
        for f in as_completed(futures):
            q,data=f.result();results[q]=data;progress('K0 wire '+str(q)+' complete')
    # Control phase value is eliminated only at the proved source reset barrier.
    wire=Wire(ACTIVE,8);theta=0.;phase_checks=0
    for k in range(3*128+1):
        for initial in (None,math.pi/2,ACTIVE.exit[8]):
            word=[dict(name='rz',wires=[8],angle=theta),dict(name='cx',wires=[8,0],angle=None)]
            end,cost=wire.step_word(word,initial)
            expected=int((theta if initial is None else initial+theta)!=0.)
            if end is not None or cost!=expected:raise ValueError('PHASE_CONTROL_BARRIER_PARITY')
            phase_checks+=1
        theta+=math.pi
    summary=dict(status='PASS',execution_source=source,B_binding=B,preparation_hash=saved['preparation']['hash'],
        state_representation='active_basis/full_operations, exact pending_Z_hex, phase_order2_seen; exact fixed outer boundary',
        source_native_branch_parities=ACTIVE.source_parity,phase_barrier_checks=phase_checks,
        law_metadata_parity=law_checks,
        wires=[v['metadata'] for q in range(9) for v in results[q]],
        workers=workers,probability_mass_tolerance=MASS_TOLERANCE,
        exact_local_and_terminal_parity=True,all_legal_continuations='Finite source alphabet parity + closed state graph + source boundary maps; induction',
        probability_contract='NUMERICAL_CONTRACT.md',K1_started=False,
        full_path_S=0,sampling=0,build_compile=0,fit=0,N_G=0)
    ACTIVE=None
    return plan,saved,results,summary


def run_K1(results,laws):
    output=[]
    for index,law in enumerate(laws):
        q=law['q_PF'];R=law['R_outer']
        for axis in ('cosine','sine'):
            values=[];lows=[];highs=[]
            for wire in range(9):
                v=results[wire][index];tail=v['kernel'].repeat(R)
                first=tail.expected(v['entry_indices'][0],v['terminal'])
                steady=tail.expected(v['entry_indices'][1],v['terminal'])
                # The single state is the proved exact outer exit, not a reset.
                g=steady[0]+v['prefix'][1]
                one=Band(MeanKernel(('FIXED_OUTER_EXIT',),np.ones((1,1)),np.array([g])),
                    np.ones((1,1)),np.ones((1,1)),np.array([steady[1]+v['prefix'][1]]),np.array([steady[2]+v['prefix'][1]]))
                rest=one.repeat(q-1).expected(0,np.array([0.]))
                fixed=v['prefix'][0]+v['suffix'][axis]
                value=first[0]+rest[0]+fixed
                lo,hi=add_bounds(np.array(first[1]),np.array(first[2]),np.array(rest[1]),np.array(rest[2]))
                lo,hi=add_bounds(lo,hi,np.array(float(fixed)),np.array(float(fixed)))
                if not float(lo)<=value<=float(hi):raise ValueError('K1_WIRE_ENCLOSURE')
                values.append(value);lows.append(float(lo));highs.append(float(hi))
            value=math.fsum(values)
            lo=fraction_bounds(sum((Fraction.from_float(v) for v in lows),Fraction()))[0]
            hi=fraction_bounds(sum((Fraction.from_float(v) for v in highs),Fraction()))[1]
            # Also cover literal source event-probability factorization rounding.
            gamma=Fraction(16,2**53-16);events=q*R
            lo=fraction_bounds(Fraction.from_float(lo)*(1-gamma)**events)[0]
            hi=fraction_bounds(Fraction.from_float(hi)*(1+gamma)**events)[1]
            if not lo<=value<=hi:raise ValueError('K1_TOTAL_ENCLOSURE')
            output.append(dict(m=law['m'],candidate_id=law['candidate_id'],axis=axis,q_PF=q,R_outer=R,
                quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',expectation='IDEAL_FINITE_K_SOURCE_LAW',
                mean_S=value,numerical_enclosure=[lo,hi],status='K0_PARITY_PASS_K1_NUMERIC_ENCLOSURE',
                source_probability_factorization_relative_bound=fraction_bounds((1+gamma)**events-1)[1],
                mean_compiled_C=None,compiled_price_qualified=False,sampler_ideal_distance=None,
                compiler_residual=None,joint_covariance=None))
    return output


def save_kernels(out,results):
    arrays={};states={}
    for q in range(9):
        for v in results[q]:
            name='wire'+str(q)+'_R'+str(v['metadata']['R']);k=v['kernel']
            states[name]=dict(states=k.central.states,entry_indices=v['entry_indices'],prefix=v['prefix'],suffix=v['suffix'])
            for field,value in (('P',k.central.transition),('g',k.central.reward),('Plo',k.pl),('Phi',k.ph),('glo',k.gl),('ghi',k.gh),('terminal',v['terminal'])):
                arrays[name+'_'+field]=value
    np.savez_compressed(out/'KERNELS.npz',**arrays)
    return states,hashlib.sha256((out/'KERNELS.npz').read_bytes()).hexdigest()


def execute_batch(root,source,out,*,k0=run_K0,k1=run_K1):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    try:
        plan,saved,results,summary=k0(root,source,lambda msg:print(msg,flush=True))
        if summary['status']!='PASS':
            write('K0.json',summary);write('STOP.json',dict(status='K0_'+summary['status']+'_MANDATORY_STOP',K1_executed=False));return
        # Durable PASS and arrays precede the only real K1 call.
        states,sha=save_kernels(out,results);summary['kernel_npz_sha256']=sha
        write('KERNEL_STATES.json',states);write('K0.json',summary)
        values=k1(results,saved['laws'])
        for r in values:
            request=next(v for v in plan['focus_requests'] if v['m']==r['m'])
            r['identity']=request['identity'];r['law_binding']=request['law_binding'];r['kernel_source']=source
        write('K1.json',dict(status='FOUR_IDEAL_FINITE_LAW_MEAN_STRUCTURAL_PRICES_COMPLETE',source=source,values=values,
            quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',compiled_price_records_produced=0,
            mean_compiled_C=None,full_rpe_resources=None,qualified_for_compiled_price=False))
        write('STOP.json',dict(status='K0_PASS_K1_FOUR_VALUES_COMPLETE_MANDATORY_STOP',source=source,automatic_follow_on=False))
    except Exception as error:
        label=str(error)
        fail=any(s in label for s in ('PARITY','ENCLOSURE_FAILED','STATE_BOUND_FAILED','EXIT_DEPENDS_ON_HISTORY','ENTRY_QUEUE','REVERSE_HEAD_QUEUE'))
        status='FAIL' if fail else 'UNKNOWN'
        if not (out/'K0.json').exists():write('K0.json',dict(status=status,source=source,error=type(error).__name__+': '+label,K1_started=False))
        write('STOP.json',dict(status='KERNEL_'+status+'_MANDATORY_STOP',source=source,error=type(error).__name__+': '+label,
            K1_complete=False,automatic_follow_on=False))
        raise
