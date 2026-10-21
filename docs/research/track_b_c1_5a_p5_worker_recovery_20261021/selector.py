"""An exact integer cumulative selector over an already sealed dyadic group law.

P5 coefficient/normalizer/event methods are inherited unchanged. Only group index
lookup changes; other dyadic selections retain the frozen implementation.
"""
from bisect import bisect_right
from common import require
from bridge import connect
connect()
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.return_aggregation import dyadic_index

class Prefix:
    def __init__(self,law,width):
        scale=1<<width;total=0;ends=[]
        for q in law:
            v=q*scale
            require(v.denominator==1 and v>=0,'EXACT_NONNEGATIVE_DYADIC')
            total+=v.numerator;ends.append(total)
        require(total==scale,'NORMALIZED_DYADIC_LAW')
        self.width,self.ends,self.scale=width,tuple(ends),scale
    def index(self,value):
        require(type(value) is int and 0<=value<self.scale,'INVALID_BIT_INTEGER')
        return bisect_right(self.ends,value)

class IndexedP5(P5Closed):
    @classmethod
    def attach(cls,generator):
        require(type(generator) is P5Closed,'FROZEN_P5_REQUIRED')
        obj=object.__new__(cls);obj.__dict__.update(generator.__dict__)
        obj.selector=Prefix(obj.group_law,obj.H)
        return obj
    def sample(self,bits):
        draw=lambda law:(self.selector.index(bits.bits(self.H)) if law is self.group_law
                         else dyadic_index(law,self.H,bits.bits(self.H)))
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
