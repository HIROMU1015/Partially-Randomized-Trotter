"""Opt-in Arb192 full-basis native-pair certificates; no molecular loader.

Exact common unitary factors may be cancelled for operator/Frobenius norms.
All residual basis columns and outputs are evaluated; phase is never fitted.
This reduction is NOT applicable to literal nonunitary CAR targets.
"""
from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z

VERSION='Arb192_full_basis_batch_exact_unitary_factors_v1'


def reduce_pair(left,right):
    fixed._validate(left);fixed._validate(right)
    if left['qubits']!=right['qubits']:raise ValueError('SAME_REGISTER_REQUIRED')
    a,b=deepcopy(left),deepcopy(right);prefix=0
    while prefix<min(len(a['gates']),len(b['gates'])) and a['gates'][prefix]==b['gates'][prefix]:prefix+=1
    a['gates']=a['gates'][prefix:];b['gates']=b['gates'][prefix:];suffix=0
    while suffix<min(len(a['gates']),len(b['gates'])) and a['gates'][-1-suffix]==b['gates'][-1-suffix]:suffix+=1
    if suffix:a['gates']=a['gates'][:-suffix];b['gates']=b['gates'][:-suffix]
    return a,b,dict(prefix_gates=prefix,suffix_gates=suffix,global_phases_unchanged=True,
        proof='U_A=exp(i phi_A) L A R,U_B=exp(i phi_B) L B R;L,R exact unitary;both norms invariant',
        literal_target_reduction_supported=False)


@lru_cache(None)
def layout(n,qs):
    d=1<<n;mask=sum(1<<q for q in qs)
    bases=tuple(i for i in range(d) if i&mask==0)
    offsets=tuple(sum(((k>>j)&1)<<q for j,q in enumerate(qs)) for k in range(1<<len(qs)))
    return tuple(tuple(base|offset for base in bases) for offset in offsets)


def midpoint_batch(ball):
    from flint import acb,acb_mat
    d,k=ball.nrows(),ball.ncols()
    middle=acb_mat([[acb(ball[i,j].real.mid(),ball[i,j].imag.mid()) for j in range(k)] for i in range(d)])
    delta=[F(z._norm(acb_mat([[ball[i,j]-middle[i,j]] for i in range(d)]))['upper_exact']) for j in range(k)]
    return middle,delta


def action(record,columns,cache):
    from flint import acb,acb_mat
    n=record['qubits'];d=1<<n;k=len(columns)
    v=acb_mat([[int(i==column) for column in columns] for i in range(d)]);error=[F()]*k
    for gate in record['gates']:
        key=(z.PRECISION,gate['name'],tuple(gate['params_hex']))
        if key not in cache:cache[key]=fixed._local(key[1],[float.fromhex(x) for x in key[2]])
        indices=layout(n,tuple(gate['qubits']));size=len(indices);blocks=len(indices[0])
        packed=acb_mat([[v[indices[i][j],c] for c in range(k) for j in range(blocks)] for i in range(size)])
        product=cache[key]*packed;out=acb_mat(d,k)
        for i in range(size):
            for c in range(k):
                for j in range(blocks):out[indices[i][j],c]=product[i,c*blocks+j]
        v,delta=midpoint_batch(out);error=[a+b for a,b in zip(error,delta)]
    phase=(acb(0,1)*z._real(float.fromhex(record['phase_hex']))).exp()
    v,delta=midpoint_batch(z._scale(v,phase));return v,[a+b for a,b in zip(error,delta)]


_PAIR=None
_CACHE=None
def init_pair(a,b):
    global _PAIR,_CACHE
    _PAIR=(a,b);_CACHE={}


def pair_batch(columns):
    from flint import ctx,acb_mat
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        a,ea=action(_PAIR[0],columns,_CACHE);b,eb=action(_PAIR[1],columns,_CACHE)
        rows=[]
        for j,column in enumerate(columns):
            norm=z._norm(acb_mat([[a[i,j]-b[i,j]] for i in range(a.nrows())]));radius=ea[j]+eb[j]
            rows.append(dict(column=column,upper_exact=str(F(norm['upper_exact'])+radius),
                lower_exact=str(max(F(),F(norm['lower_exact'])-radius)),
                evaluation_radius_left_exact=str(ea[j]),evaluation_radius_right_exact=str(eb[j])))
        return rows


def compare(left,right,*,batch_size=8,parallel=True,cancel_common=True):
    from flint import ctx,arb
    from .c_per_word_population_range import _digest
    z.runtime_model();fixed._validate(left);fixed._validate(right)
    if left['qubits']!=right['qubits'] or type(batch_size) is not int or batch_size<1:
        raise ValueError('REGISTER_OR_BATCH_SCHEMA')
    d=1<<left['qubits'];a,b,proof=reduce_pair(left,right) if cancel_common else (left,right,None)
    batches=[list(range(i,min(i+batch_size,d))) for i in range(0,d,batch_size)]
    plan=fixed.resource_plan(d);plan['workers']=min(plan['workers'],len(batches))
    work=d*d*(len(a['gates'])+len(b['gates']))
    if parallel and work>100000 and plan['workers']>1:
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init_pair,initargs=(a,b)) as pool:
            rows=[row for batch in pool.map(pair_batch,batches) for row in batch]
    else:
        init_pair(a,b);rows=[row for batch in batches for row in pair_batch(batch)]
        plan.update(workers_used=1,reason='small residual or explicit artificial serial comparison')
    rows.sort(key=lambda x:x['column'])
    if [r['column'] for r in rows]!=list(range(d)):raise ValueError('COMPLETE_ORTHONORMAL_BASIS_REQUIRED')
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        u=z._fraction(sum((z._real(F(r['upper_exact']))**2 for r in rows),arb(0)).sqrt().upper())
    return dict(status='VALIDATED_BATCHED_NATIVE_PAIR',kind='rigorous_enclosure',norm='full_operator_2',
        dimension=d,columns_evaluated=d,precision_bits=z.PRECISION,rounding_enclosed=True,phase_fit=False,
        upper_exact=str(min(F(2),u)),lower_exact=str(max(F(r['lower_exact']) for r in rows)),
        columns=rows,left_id=_digest(left),right_id=_digest(right),factor_reduction=proof,
        column_basis='full computational basis of norm-equivalent residual;not original column identities when factors removed',
        original_full_operator_coverage=True,nonunitary_literal_target_supported=False,
        batch_size=batch_size,resource_plan=plan,whole_law_bound=None,actual_CA_achieved=None)
