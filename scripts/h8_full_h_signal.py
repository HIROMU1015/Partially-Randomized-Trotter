"""Few H8 pure-H means; saved eigenstate shortcut with sparse vector check."""
import cmath
import math
import time
import numpy as np
from scipy.sparse import eye
import h8_full_h_pure_rte as s


def polynomial(value,dt,K,normalization=1.,*,raw=False,horner=False):
    # K0 is the same degree-one paired-Taylor numerator; original H8 API
    # accepted only K2/4. No change to its definition for those orders.
    if K==0:
        out=1.+(-1j*dt)*value
        return out/normalization if raw else out
    return s.old.finite_piece(1.+0j,lambda x:value*x,dt,K,normalization,raw=raw,horner=horner)


def record(ctx,row):
    from trotterlib.rte import finite_rte_distribution
    prep=ctx.preps[0];count=row['s'];dt=1.6/count
    scalar=ctx.scalar(row);energy=ctx.energy-scalar
    law=finite_rte_distribution(prep.exact_rte_lambda_r*dt,row['K'])
    b=law.exact_finite_distribution;log_B=count*math.log(b)
    if row['K']==0:
        p=1.-1j*dt*energy;ph=p
    else:
        p=polynomial(energy,dt,row['K']);ph=polynomial(energy,dt,row['K'],horner=True)
    phase=cmath.exp(-1j*scalar*1.6)
    corrected=phase*cmath.exp(count*cmath.log(p))
    independent=phase*cmath.exp(count*cmath.log(ph))
    raw=phase*cmath.exp(count*cmath.log(p/b))
    exact=cmath.exp(-1j*1.6*ctx.energy)
    # For F=P/b and ||F(H)||<=1, telescope F(H)^s-F(e)^s.
    # The reported bound uses measured residual, not an interval certificate.
    eta=dt*max(prep.exact_rte_lambda_r,abs(energy))
    derivative=sum(eta**k/math.factorial(k) for k in range(row['K']+1))
    residual_bound=math.exp(log_B)*1.6*derivative*ctx.residual/b*max(1.,abs(p/b))**(count-1)
    discrepancy=abs(corrected-independent);closure=abs(raw-math.exp(-log_B)*corrected)
    s.require(max(discrepancy,closure,residual_bound)<1e-9,'SCALAR_FINITE_MEAN_CLOSURE')
    cr=lambda z:dict(real=float(z.real),imag=float(z.imag))
    rec=dict(candidate=row,old_u=max(ctx.reference_error,residual_bound,discrepancy,closure),old_margin=0.,
        origin='PURE_H_SAVED_EIGENSTATE_FINITE_POLYNOMIAL',phase_alignment=False,
        signal=dict(reference=cr(ctx.reference),corrected=cr(corrected),raw=cr(raw),exact_tail=cr(exact),
            total_signed_bias=cr(corrected-ctx.reference),outer_PF_or_discard_bias=cr(exact-ctx.reference),
            finite_RTE_bias=cr(corrected-exact),log_B=log_B,
            allowance_evidence=dict(maximum_intermediate_norm=max(1.,abs(corrected)),raw_corrected_vector_closure=closure)),
        recurrence_difference=discrepancy,eigenstate_shortcut=dict(energy=ctx.energy,
            tail_energy=energy,residual=ctx.residual,measured_residual_propagation_bound=residual_bound,
            bound_is_interval_certificate=False,formula='exp(-i*T*scalar)*P_(K+1)(-i*T*(E-scalar)/s)^s'))
    from run_pf_common_run import qualify_record
    endpoint=qualify_record(rec,row,.0005)
    products=math.fsum(k*p for k,p in zip(law.orders,law.order_probabilities))
    endpoint['proxy_C_work']=count*(1+products)
    endpoint['proxy_G_work']=None if endpoint['status']!='ELIGIBLE' else (endpoint['N_cos']+endpoint['N_sin'])*endpoint['proxy_C_work']
    endpoint['proxy_is_compiled_price_or_lower_bound']=False
    return dict(record=rec,endpoint=endpoint)


def sparse_vector_check(ctx,item):
    row=item['endpoint']['candidate'];prep=ctx.preps[0]
    started=time.monotonic()
    # Exact sparse algebra from the saved 4900-dimensional sector, no dense H.
    matrix=ctx.one.copy()
    for w,g in zip(ctx.ham.lambdas,ctx.blocks):matrix=matrix+w*(g@g)
    matrix=matrix-prep.extracted_identity_coefficient*eye(len(ctx.state),format='csr')
    errors=[]
    for v in (ctx.state,np.eye(1,len(ctx.state),0,dtype=complex)[0],np.eye(1,len(ctx.state),len(ctx.state)-1,dtype=complex)[0]):
        errors.append(float(np.linalg.norm(matrix@v-ctx.tails[0][0]@v)))
    s.require(max(errors)<1e-9,'CSR_VERSUS_SAVED_DF_ACTION')
    from trotterlib.rte import finite_rte_distribution
    dt=1.6/row['s'];b=finite_rte_distribution(prep.exact_rte_lambda_r*dt,row['K']).exact_finite_distribution
    v=ctx.state.copy();maximum=1.
    for i in range(row['s']):
        if row['K']==0:v=v-1j*dt*(matrix@v)
        else:v=s.old.finite_piece(v,lambda x:matrix@x,dt,row['K'],b,raw=False)
        maximum=max(maximum,float(np.linalg.norm(v)))
    v*=np.exp(-1j*1.6*ctx.scalar(row))
    z=item['record']['signal']['corrected'];expected=complex(z['real'],z['imag'])*ctx.state
    error=float(np.linalg.norm(v-expected))
    s.require(error<1e-9,'FULL_STEP_SPARSE_VECTOR_POLYNOMIAL_CHECK')
    result=dict(status='PASS',candidate=row,sector_dimension=len(ctx.state),CSR_nonzeros=matrix.nnz,
        matrixfree_action_errors=errors,full_step_vector_error=error,maximum_intermediate_norm=maximum,
        wall_seconds=time.monotonic()-started,dense_full_matrix_allocated=False,new_target_reference_calls=0)
    s.save(s.PUBLIC/'signal_vector_validation.json',result)
    return result


def main():
    s.require(s.read(s.PUBLIC/'validation.json')['status']=='PASS','PHASE_A_PASS_REQUIRED')
    ctx=s.context('signal');lam=ctx.preps[0].exact_rte_lambda_r
    # Pure K2 low-bias cost proxy s*B^2 suggests s~2(lambda*T)^2.
    # Three nearby budgets and one K0 check only; no boundary iteration.
    center=max(192,int(math.ceil((2*(lam*1.6)**2/32)/64))*64)
    rows=[s.candidate(r,2) for r in (center-128,center,center+128)]+[s.candidate(center,0)]
    items=[record(ctx,r) for r in rows]
    if not any(x['endpoint']['status']=='ELIGIBLE' for x in items if x['endpoint']['candidate']['K']==2):
        items.append(record(ctx,s.candidate(center,4)))
    s.save(s.PUBLIC/'signals.json',dict(items=items,scope='SMALL_FIXED_NEIGHBORHOOD_NOT_GRID_SEARCH',
        center_R=center,proxy='H46 pure-H event-count*(1+expected product order)*fixed shots; uncalibrated H8 work units',
        reference=ctx.reference_record,new_reference_calls=0,new_SCF_DF_state=0))
    eligible=[x for x in items if x['endpoint']['status']=='ELIGIBLE']
    if not eligible:
        s.save(s.PUBLIC/'selection.json',dict(status='NO_ELIGIBLE_SMALL_CANDIDATE',candidates=[],prices_started=False))
        return
    choice=min(eligible,key=lambda x:x['endpoint']['proxy_G_work'])
    sparse_vector_check(ctx,choice)
    s.save(s.PUBLIC/'selection.json',dict(status='FIXED_BEFORE_PRICES',candidates=[choice['endpoint']['candidate']],
        endpoints=[choice['endpoint']],cohort=s.COHORT,n=9,independent_confirmation=False,
        rule='one eligible minimum of uncalibrated event-work proxy among the four/five recorded signal candidates',
        scope_expansion=False,conditional_price_sampling=False))
    print(s.read(s.PUBLIC/'selection.json'))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        s.save(s.PUBLIC/'signal_STOP.json',dict(reason=str(exc),type=type(exc).__name__))
        raise
