"""Instrumented spectral / ascending matrix paths. Imported without science IO.

Both paths retain the pinned signed schedule and numerator degrees 0..5.
Only an authorized production Backend calls these functions on real inputs.
"""
import math
import h2_common_production_contract as c


def path(backend, candidate, independent):
    s=backend.s; s.verify()
    np=backend.np;ctx=backend.ctx;pr=candidate.method=='PR'
    from scipy.linalg import expm
    steps=backend.steps(candidate)
    columns=2 if pr else 1
    v=np.repeat(ctx.oracle.initial[:,None],columns,axis=1)
    trace=[];work=dict(deterministic=0,tail=0);logs=[];raw_log=0.

    def observe(label, values, scale_log):
        c.require(np.isfinite(values).all(), 'NONFINITE_DIAGNOSTIC_STAGE')
        norms=np.linalg.norm(values,axis=0)
        for j,norm in enumerate(norms):
            # The raw column is attenuated; report its norm in corrected units.
            value=float(norm)*(math.exp(scale_log) if pr and j==1 else 1.)
            c.policy.number(value,'stage_norm')
            trace.append(dict(stage=len(trace),label=label,column=j,
                              corrected_unit_norm=value,raw_log_scale=scale_log if j==1 else 0.))

    observe('initial',v,raw_log)
    specs=ctx.oracle.specs[:candidate.prefix+1]
    for i,step in enumerate(steps):
        t=step['signed_time'];r=step['rte_steps']
        for direction in ('forward','reverse'):
            for index in (range(len(specs)) if direction=='forward' else reversed(range(len(specs)))):
                spec=specs[index]
                if independent:
                    v=expm(-.5j*t*ctx.matrices[index])@v
                else:
                    # Same spectral_apply algebra, with every executed stage observed.
                    v=spec[2]@v;observe('det_basis_in',v,raw_log)
                    v=np.exp(-.5j*t*spec[0])[:,None]*v;observe('det_diagonal',v,raw_log)
                    v=spec[1]@v
                work['deterministic']+=columns
                observe('det_'+direction,v,raw_log)
            if direction!='forward' or not pr:continue
            tau=ctx.lambda_R*t/r
            if independent:
                b=math.fsum(abs(tau)**k/math.factorial(k)*math.hypot(1.,tau/(k+1)) for k in (0,2,4))
                for short in range(r):
                    term=v.copy();total=term.copy();observe('tail_term_0',term,raw_log)
                    for degree in range(1,6):
                        term=(-1j*t/(r*degree))*(ctx.tail@term)
                        work['tail']+=columns
                        observe('tail_term_'+str(degree),term,raw_log)
                        total+=term;observe('tail_sum_'+str(degree),total,raw_log)
                    v=total;observe('tail_corrected_step',v,raw_log)
                    v[:,1]/=b;raw_log+=math.log(b)
                    observe('tail_raw_divide',v,raw_log)
            else:
                b=backend.distribution(tau,4).exact_finite_distribution
                f=backend.core.polynomial(ctx.tail_spec[0],t/r,4)**r
                factors=np.column_stack((f,f/(b**r)))
                v=ctx.tail_spec[2]@v;work['tail']+=columns;observe('tail_basis_in',v,raw_log)
                v=factors*v;raw_log+=r*math.log(b);observe('tail_diagonal_power',v,raw_log)
                v=ctx.tail_spec[1]@v;work['tail']+=columns;observe('tail_basis_out',v,raw_log)
            logs.append(r*math.log(b))
        observe('piece_'+str(i),v,raw_log)
    scalar=ctx.ham.constant+(ctx.tail_identity if pr else 0.)
    v*=np.exp(-.8j*scalar);observe('scalar_once',v,raw_log)
    logB=math.fsum(logs) if independent else sum(logs)
    c.policy.number(logB,'log_B')
    signals=[complex(np.vdot(ctx.oracle.initial,v[:,j])) for j in range(columns)]
    return dict(vectors=v,signals=signals,log_B=logB,work=work,trace=trace,
        work_definition='per-vector deterministic propagator; actual tail matrix-vector application',
        trace_definition='all executed state stages including spectral bases and Taylor terms/sums; raw column in corrected units')


def join(backend, candidate, primary, independent):
    np=backend.np;pr=candidate.method=='PR'
    def scaled(row,index):return math.exp(row['log_B']) if pr and index==1 else 1.
    se=[abs(primary['signals'][i]*scaled(primary,i)-independent['signals'][i]*scaled(independent,i))
        for i in range(len(primary['signals']))]
    ve=[float(np.linalg.norm(primary['vectors'][:,i]*scaled(primary,i)-
                            independent['vectors'][:,i]*scaled(independent,i))) for i in range(len(se))]
    log_error=abs(primary['log_B']-independent['log_B'])
    closure=abs(math.exp(primary['log_B'])*primary['signals'][1]-primary['signals'][0]) if pr else 0.
    for value in se+ve+[log_error,closure]:
        c.policy.number(value,'diagnostic_error');c.require(value<1e-9,'DIAGNOSTIC_GATE')
    ref=backend.reference(candidate,primary['signals'][0])
    trace=primary['trace']+independent['trace']
    ev='results/'+candidate.id+'_diagnostics.json'
    obs=lambda value,unit,scope:c.policy.Observation(value,unit,scope,ev)
    inputs=c.policy.AdaptiveInputs(candidate.id,'SAVED_SCIENCE',backend.reference_id,
        obs(backend.reference_error,'CORRECTED_SIGNAL','H2_REFERENCE_COMBINED_DIAGNOSTIC'),
        tuple(obs(x,'CORRECTED_SIGNAL','PRIMARY_VS_INDEPENDENT_SIGNAL') for x in se),
        tuple(obs(x,'CORRECTED_STATE_VECTOR','PRIMARY_VS_INDEPENDENT_VECTOR') for x in ve),
        obs(closure,'CORRECTED_SIGNAL','B_RAW_MINUS_CORRECTED_SIGNAL'),
        obs(log_error,'LOG_NORMALIZATION','TWO_PATH_LOG_B_DIFFERENCE'),
        primary['work'],independent['work'],
        obs(max(t['corrected_unit_norm'] for t in trace),'CORRECTED_STATE_VECTOR','BOTH_PATHS_ALL_REQUIRED_STAGES'),
        'BOTH_PATHS_ALL_REQUIRED_STAGES',ev,c.policy.NATIVE_POLICY,backend.input_digest)
    inputs.validate(backend.reference_id)
    c.save(backend.s.root/ev,dict(candidate=candidate.id,series=c.SERIES,
        inputs=c.asdict(inputs),primary_trace=primary['trace'],independent_trace=independent['trace'],
        log_B=primary['log_B'],reference_binding=ref,certified=False,
        policy_semantics='new instrumented common diagnostic series; old u is not reused'))
    return dict(inputs=inputs,primary=primary,independent=independent,bias=ref['bias'])
