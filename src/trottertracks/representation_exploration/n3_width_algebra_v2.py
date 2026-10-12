"""Memoized CAR coefficient assembly for fixed weighted SOS proposals.

Normal ordered words contain positive creation and negative annihilation labels.
The independent checker uses explicit ladder actions and does not import this.
"""
from functools import lru_cache
from fractions import Fraction as F
from itertools import combinations


@lru_cache(maxsize=200000)
def normal(word):
    for i in range(len(word)-1):
        x,y=word[i:i+2]
        if (x>0)==(y>0) and x==y: return ()
        swap=(x<0<y) or ((x>0)==(y>0) and abs(x)>abs(y))
        if swap:
            result={}
            for w,c in normal(word[:i]+(y,x)+word[i+2:]):result[w]=result.get(w,0)-c
            if x<0<y and -x==y:
                for w,c in normal(word[:i]+word[i+2:]):result[w]=result.get(w,0)+c
            return tuple((w,c) for w,c in sorted(result.items()) if c)
    return ((word,1),)


def add(*polys):
    ans={}
    for p in polys:
        for w,c in p.items():ans[w]=ans.get(w,F(0))+c
    return {w:c for w,c in ans.items() if c}


def scale(p,c): return {w:x*c for w,x in p.items() if x*c}
def mono(word): return dict(normal(tuple(word)))
def dagger(p):
    ans={}
    for w,c in p.items():ans=add(ans,scale(mono(tuple(-x for x in reversed(w))),c))
    return ans
def multiply(p,q):
    ans={}
    for w,c in p.items():
        for z,b in q.items():ans=add(ans,scale(mono(w+z),c*b))
    return ans


def number(m): return add(*(mono((i+1,-i-1)) for i in range(m)))
def one_body(A): return add(*(scale(mono((i+1,-j-1)),A[i][j]) for i in range(len(A)) for j in range(len(A))))
def density(J):return add(*(scale(mono((i+1,-i-1,j+1,-j-1)),x) for (i,j),x in J.items()))


def generators(m):
    pairs=list(combinations(range(m),2))
    return {'particle':[mono((-i-1,)) for i in range(m)],
            'hole':[mono((i+1,)) for i in range(m)],
            'D':[mono((-j-1,-i-1)) for i,j in pairs],
            'Q':[mono((i+1,j+1)) for i,j in pairs],
            'G':[mono((i+1,-j-1)) for i in range(m) for j in range(m)]}


@lru_cache(maxsize=8)
def structure(m,nu):
    blocks={}; keys={()}
    for name,ops in generators(m).items():
        terms=[]
        for i in range(len(ops)):
            for j in range(i,len(ops)):
                p=multiply(dagger(ops[i]),ops[j])
                if i!=j:p=add(p,multiply(dagger(ops[j]),ops[i]))
                terms.append((i,j,p));keys.update(p)
        blocks[name]={'size':len(ops),'terms':terms}
    N=add(number(m),{():-F(nu)})
    multipliers=[N]
    for i in range(m):
        for j in range(i,m):
            p=mono((i+1,-j-1))
            if i!=j:p=add(p,mono((j+1,-i-1)))
            multipliers.append(multiply(p,N))
    for p in multipliers:keys.update(p)
    return blocks,tuple(multipliers),tuple(sorted(keys))


def assemble_gram(block,raw):
    return add(*(scale(p,raw[i][j]) for i,j,p in block['terms']))
