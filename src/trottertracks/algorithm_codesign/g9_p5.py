"""Independent degree-five grouping; no Green kernel or event table input."""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import factorial
from .return_aggregation import Interval, root_interval, dyadic_distribution, dyadic_index
from .g7_generator import _event


class P5Closed:
    def __init__(self, p, x, root_bits=256, probability_bits=160,
                 eta=F(1, 10**12), rho=F(1, 10**12)):
        self.p, self.x = tuple(map(F, p)), F(x)
        if min(self.p) <= 0 or sum(self.p) != 1 or not 0 < self.x <= 1:
            raise ValueError('positive normalized p and 0<x<=1')
        self.m, self.arm = 5, 'closed_P5_full'
        self.K, self.H, self.eta, self.rho = root_bits, probability_bits, F(eta), F(rho)
        self.t = tuple(self.x**n / factorial(n) for n in range(6))
        self.chi, self.mu3, self.mu4, self.mu5 = [sum(v**k for v in self.p) for k in (2, 3, 4, 5)]
        self.a0 = 1-self.chi*self.t[2]+(2*self.chi**2-self.mu4)*self.t[4]
        self.child0 = tuple(v*(self.t[1]-(2*self.chi-v*v)*self.t[3]
                         +(5*self.chi**2-4*self.chi*v*v+2*v**4-2*self.mu4)*self.t[5]) for v in self.p)
        self.s0 = sum(self.child0)
        # DP states are (previous label, remaining length), O(4L) states.
        @lru_cache(None)
        def completion(previous, remaining):
            return F(1) if remaining == 0 else sum(v*completion(i, remaining-1)
                for i,v in enumerate(self.p) if i != previous)
        self.completion = completion
        self.r4 = tuple(v*completion(i,3) for i,v in enumerate(self.p))
        self.groups = [('root', -1, -1, self.a0*self.a0+self.s0*self.s0)]
        for j,pj in enumerate(self.p):
            for k,pk in enumerate(self.p):
                if j != k:
                    a,s = self.pair_scalars(j,k)
                    self.groups.append(('two', j,k, (pj*pk)**2*(a*a+s*s)))
        for j,r in enumerate(self.r4):
            if r:
                self.groups.append(('four',j,-1,r*r*(self.t[4]**2+self.t[5]**2*(1-self.p[j])**2)))
        self.roots = tuple(root_interval(g[3],self.K) for g in self.groups)
        if any(z.lo<=0 or z.hi-z.lo>2*self.rho*z.lo for z in self.roots):
            raise ArithmeticError('group coefficient resolution')
        self.B = Interval(sum(z.lo for z in self.roots),sum(z.hi for z in self.roots))
        self.group_law = dyadic_distribution(tuple(Interval(z.lo/self.B.hi,z.hi/self.B.lo)
                                                    for z in self.roots),self.H,self.eta)

    def law(self, values):
        return dyadic_distribution(tuple(Interval(v,v) for v in values),self.H,self.eta)

    def pair_scalars(self,j,k):
        pj,pk=self.p[j],self.p[k]
        a=self.t[2]-self.t[4]*(3*self.chi-pj*pj-pk*pk)
        s=(1-pj)*self.t[3]-self.t[5]*((1-pj)*(4*self.chi-pj*pj-pk*pk)-(self.mu3-pj**3))
        return a,s

    def level2(self,j,k):
        pj,pk=self.p[j],self.p[k];a,s=self.pair_scalars(j,k)
        b=tuple(F(0) if i==j else v*(self.t[3]-self.t[5]*(4*self.chi-pj*pj-pk*pk-v*v))
                for i,v in enumerate(self.p))
        if sum(b)!=s:raise ArithmeticError('closed conditional mass')
        return a,s,b

    def parent_coefficients(self,word):
        word=tuple(word)
        if not word:return self.a0,self.s0,self.child0
        if len(word)==2:
            a,s,b=self.level2(*word);w=self.p[word[0]]*self.p[word[1]]
            return w*a,w*s,tuple(w*v for v in b)
        if len(word)==4:
            w=F(1)
            for i in word:w*=self.p[i]
            b=tuple(F(0) if i==word[0] else self.t[5]*w*v for i,v in enumerate(self.p))
            return self.t[4]*w,sum(b),b
        raise ValueError('even P5 parent')

    def word_law(self, first, word):
        ideal,proposal=F(1),F(1)
        for pos,i in enumerate(word[1:]):
            prev=word[pos];remain=2-pos
            values=tuple(F(0) if k==prev else v*self.completion(k,remain)/self.completion(prev,remain+1)
                         for k,v in enumerate(self.p))
            # Zero alternatives are removed before dyadic rounding.
            ids=tuple(k for k,v in enumerate(values) if v)
            law=self.law(tuple(values[k] for k in ids))
            ideal*=values[i];proposal*=law[ids.index(i)]
        return ideal,proposal

    def event(self,index,word,child):
        mode,j,k,_=self.groups[index];word=tuple(word)
        q=self.group_law[index];w=F(1)
        if mode=='root':
            if word:raise ValueError('root word')
            ids=tuple(range(len(self.p)));child_mass=tuple(v/self.s0 for v in self.child0)
            ratio=self.s0/self.a0
        elif mode=='two':
            if word!=(j,k):raise ValueError('two group')
            a,s,b=self.level2(j,k);ids=tuple(i for i in range(len(self.p)) if i!=j)
            child_mass=tuple(b[i]/s for i in ids);ratio=s/a
        else:
            if len(word)!=4 or word[0]!=j or any(a==b for a,b in zip(word,word[1:])):
                raise ValueError('raw-reduced four group')
            wi,wq=self.word_law(j,word);w*=wi;q*=wq
            ids=tuple(i for i in range(len(self.p)) if i!=j)
            child_mass=tuple(self.p[i]/(1-self.p[j]) for i in ids)
            ratio=self.x*(1-self.p[j])/5
        child_law=self.law(child_mass);ci=ids.index(child)
        w*=child_mass[ci];q*=child_law[ci]
        return _event(word,child,ratio,len(word),self.roots[index].midpoint*w,q)

    def sample(self,bits):
        draw=lambda law:dyadic_index(law,self.H,bits.bits(self.H))
        index=draw(self.group_law);mode,j,k,_=self.groups[index]
        if mode=='root':word=()
        elif mode=='two':word=(j,k)
        else:
            word=(j,)
            for remaining in (2,1,0):
                prev=word[-1];ids=tuple(i for i in range(len(self.p)) if i!=prev)
                values=tuple(self.p[i]*self.completion(i,remaining)/self.completion(prev,remaining+1) for i in ids)
                word+= (ids[draw(self.law(values))],)
        a,s,b=self.parent_coefficients(word);ids=tuple(i for i,v in enumerate(b) if v)
        child=ids[draw(self.law(tuple(b[i]/s for i in ids)))]
        return self.event(index,word,child)

    def reference_events(self):
        # Explicitly reference-only, never called by sample/production.
        L=len(self.p)
        for index,(mode,j,k,_) in enumerate(self.groups):
            words=[()] if mode=='root' else [(j,k)] if mode=='two' else (
                (j,)+tail for tail in product(range(L),repeat=3)
                if all(a!=b for a,b in zip((j,)+tail,tail)))
            for word in words:
                _,_,b=self.parent_coefficients(word)
                for child,v in enumerate(b):
                    if v:yield self.event(index,word,child)


def independent_raw_coefficients(p,x):
    """Small formal word reference: first-adjacent deletion, no Green import."""
    out={};p=tuple(map(F,p));x=F(x)
    for n in range(6):
        for word in product(range(len(p)),repeat=n):
            q=x**n/factorial(n)
            for i in word:q*=p[i]
            reduced=list(word)
            while any(a==b for a,b in zip(reduced,reduced[1:])):
                i=next(i for i in range(len(reduced)-1) if reduced[i]==reduced[i+1])
                del reduced[i:i+2]
            axis=n%2;sign=(1,-1,-1,1)[n%4]
            key=(tuple(reduced),axis);out[key]=out.get(key,F(0))+sign*q
    return out


def audit_p5(p,x):
    g=P5Closed(p,x);ref=independent_raw_coefficients(p,x);checks=0
    L=len(p)
    for l in (0,2,4):
        for word in product(range(L),repeat=l):
            if any(a==b for a,b in zip(word,word[1:])):continue
            a,s,b=g.parent_coefficients(word);sign=(-1)**(l//2)
            assert a>0 and s>0 and sign*a==ref[(word,0)]
            for child,coef in enumerate(b):
                if coef:assert -sign*coef==ref[((child,)+word,1)]
            checks+=1
    assert len(g.groups)<=L*L+1
    events=list(g.reference_events());assert sum(e['proposal'] for e in events)==1
    return {'parents_checked':checks,'events':len(events),'groups':len(g.groups),
            'formal_coefficients_exact':True,'digital_proposal_sum':'1','B_interval':{'lo':str(g.B.lo),'hi':str(g.B.hi)},
            'angles':sorted({str(e['ratio']) for e in events})}
