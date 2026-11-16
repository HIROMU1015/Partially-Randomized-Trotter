"""Numerical leaf for the fixed 20-cell screen; artificial tests only so far.

Uses existing signed schedules/spectra/Taylor polynomials and NumPy powers.
No circuit builder, sampler, shot solver, decoder or price provider is imported.
"""
from copy import deepcopy
import ast
from functools import lru_cache
import math
from types import SimpleNamespace


@lru_cache(maxsize=1)
def allowance_formula(root):
    """Original arithmetic only; caller supplies its own applicable norm gates.

    Pure Full-H uses saved full-H references, not a newly propagated exact-tail
    column. Its missing norm observation must not be fabricated to call the
    old PR action gate. The empirical allowance itself grants no qualification.
    """
    from . import h4_pr_m1_rmse as M
    tree=ast.parse(M.blob(root,M.V.S0_SCIENCE,'src/trottertracks/pr_r_dependent_n.py'))
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze').body
    def assigns(n,name):return isinstance(n,ast.Assign) and any(getattr(t,'id',None)==name for t in n.targets)
    start=next(i for i,n in enumerate(body) if assigns(n,'pieces'))
    stop=next(i for i,n in enumerate(body) if assigns(n,'point'))
    fn=ast.parse('def allowance(row,record,reference):\n c=row["candidate"];sig=record["signal"];ev=sig["allowance_evidence"];pair=reference["signal_difference"]\n').body[0]
    fn.body+=body[start:stop]+[ast.parse("return dict(u=u,log_B_margin=margin,stages=stages,roundoff=roundoff,certified=False,policy='R_AWARE_EMPIRICAL_ACTION_COUNT_V1')").body[0]]
    scope=dict(math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen_empirical_allowance_formula','exec'),scope)
    return scope['allowance']


def context(saved, *, root, preparation, tail7):
    import numpy as np
    from .pr_rpe_power_mean import kernels
    k=kernels(root)
    specs=tuple(k.spectrum(m) for m in saved.matrices)
    non_scalar=sum(saved.matrices,np.zeros_like(saved.matrices[0]))
    full_spec=k.spectrum(non_scalar-preparation['scalar']['extracted_identity']*np.eye(len(saved.state)))
    residuals=[float(np.linalg.norm((s[1]*s[0])@s[2]-m,'fro')) for s,m in zip(specs,saved.matrices)]
    residuals.append(float(np.linalg.norm((full_spec[1]*full_spec[0])@full_spec[2]-
        (non_scalar-preparation['scalar']['extracted_identity']*np.eye(len(saved.state))),'fro')))
    return SimpleNamespace(state=saved.state,constant=saved.ham.constant,specs=specs,full_spec=full_spec,
        tail7=tail7,cache={},spectral_residuals=residuals)


def deterministic_step(ctx,c,weights):
    import numpy as np
    # Cached local step matrices, not circuits or full event sequences.
    key=(c['formula'],c['prefix'],c['T']/c['q'])
    if key not in ctx.cache:
        U=np.eye(len(ctx.state),dtype=np.complex128)
        for w in weights:
            t=c['T']*w/c['q']
            for spec in (*ctx.specs[:c['prefix']+1],*reversed(ctx.specs[:c['prefix']+1])):
                U=spec[1]@(np.exp(-.5j*t*spec[0])[:,None]*(spec[2]@U))
        ctx.cache[key]=U
    return ctx.cache[key]


def evaluate(ctx,row,reference,*,root,preparation):
    import numpy as np
    from . import h4_pr_m1_rmse as M
    from .pr_rpe_power_mean import kernels
    from trotterlib.rte import finite_rte_distribution
    c=row['candidate']; st=row['structure']; T=c['T']; z=lambda v:complex(v['real'],v['imag'])
    ref=z(reference['reference']); pair=reference['signal_difference']
    if not all(math.isfinite(v) for v in (ref.real,ref.imag,z(reference['independent_reference']).real,z(reference['independent_reference']).imag)):
        raise ValueError('Nonfinite saved reference')
    if not math.isfinite(pair) or not 0<=pair<=1e-10:raise ValueError('Saved reference pair gate')
    if abs(z(reference['independent_reference'])-ref)>1e-10:raise ValueError('Saved reference consistency')
    sensitivity=0.; logB=0.; b=1.; normalizers=0
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        if row['comparison_method']=='FULL_H_RTE':
            if st['deterministic_blocks']!=0 or preparation['one_body_policy']!='RANDOMIZED':
                raise ValueError('No DF B3 substitution')
            q=c['q']; R=c['R_outer']; s=q*R
            # Preserve the original grouped event-time operation order. T/s is
            # not substituted when binary64 rounding differs for non-dyadic q.
            dt=(T/q)/R; tau=preparation['lambda_R']*dt
            b=finite_rte_distribution(tau,c['K']).exact_finite_distribution;normalizers=1
            if not math.isfinite(b) or b<=1:raise ValueError('Normalizer precision unresolved')
            P=kernels(root).evolve.__globals__['polynomial'](ctx.full_spec[0],dt,c['K'])
            one=P**R
            # Divide at group-operator level, before independently propagating
            # raw and corrected powers. Never set raw_signal=corrected/B.
            factors=np.stack([one**q,(one/(b**R))**q],axis=1)
            vectors=ctx.full_spec[1]@(factors*(ctx.full_spec[2]@ctx.state)[:,None])
            vectors*=np.exp(-1j*T*(ctx.constant+preparation['scalar']['extracted_identity']))
            corrected,raw=(complex(np.vdot(ctx.state,vectors[:,j])) for j in (0,1))
            logB=s*math.log(b);closure=float(np.linalg.norm(vectors[:,1]-math.exp(-logB)*vectors[:,0]))
            maximum=float(max(1.,np.linalg.norm(vectors,axis=0).max()))
            norm_error=None; unitary_error=None
            sensitivity=s*(math.log(math.nextafter(b,math.inf))-math.log(math.nextafter(b,0.)))
            backend='PURE_FULL_H_GROUPED_SPECTRAL_POWER'
        else:
            U=deterministic_step(ctx,c,st['signed_weights'])
            powered=np.linalg.matrix_power(U,c['q'])
            vector=powered@ctx.state; vector*=np.exp(-1j*T*ctx.constant)
            corrected=complex(np.vdot(ctx.state,vector))
            if c['method']=='B0I':corrected*=np.exp(-1j*T*ctx.tail7['scalar'])
            raw=corrected;closure=0.;maximum=float(max(1.,np.linalg.norm(vector)))
            norm_error=abs(float(np.linalg.norm(vector))-1.)
            unitary_error=float(np.linalg.norm(powered.conj().T@powered-np.eye(len(ctx.state)),'fro'))
            if norm_error>1e-9 or unitary_error>1e-9:raise ValueError('Deterministic power norm/unitarity gate')
            backend='SIGNED_S2_MACRO_STEP_MATRIX_POWER'
    values=(corrected.real,corrected.imag,raw.real,raw.imag,logB,closure,maximum,sensitivity)
    if not all(math.isfinite(v) for v in values) or closure>1e-9:raise ValueError('Finite mean normalization closure gate')
    attenuation=math.exp(-logB)
    if attenuation==0:raise ValueError('Attenuation underflow')
    ev=dict(maximum_intermediate_norm=maximum,raw_corrected_vector_closure=closure,
        maximum_norm_domain='POWER_OUTPUT_BOUNDARIES_NOT_EVERY_EVENT',exact_tail_norm_error=norm_error)
    record=dict(candidate=deepcopy(c),signed_weights=deepcopy(st['signed_weights']),
                allocation=deepcopy(st['allocation']),event_count=st['event_count'],signal=dict(allowance_evidence=ev,log_B=logB))
    # Existing empirical R-aware formula only; analyze()/Hoeffding never called.
    allow=allowance_formula(str(root))(
        dict(candidate=c,structure=st),record,reference)
    allow.update(log_B_margin=max(allow['log_B_margin'],sensitivity),normalizer_sensitivity=sensitivity,
        rigorous_bound=False,source=dict(commit=M.V.S0_SCIENCE,path='src/trottertracks/pr_r_dependent_n.py',function='analyze:allowance_prefix'),
        logical_stages_conservative_for_full_h_zero_head=True,
        full_h_exact_tail_norm_observation=None if row['comparison_method']=='FULL_H_RTE' else norm_error)
    zz=lambda v:dict(real=float(v.real),imag=float(v.imag))
    if max(abs(raw.real),abs(raw.imag))>1+allow['u']:raise ValueError('Raw outcome range unresolved')
    pure=row['comparison_method']=='FULL_H_RTE'
    phase=lambda value:None if abs(value)==0 or abs(ref)==0 else math.atan2((value*ref.conjugate()).imag,(value*ref.conjugate()).real)
    return dict(candidate=deepcopy(c),structure=deepcopy(st),backend=backend,
        signal=dict(reference=deepcopy(reference['reference']),corrected=zz(corrected),raw=zz(raw),
            total_signed_bias=zz(corrected-ref),log_B=logB,attenuation=attenuation,
            B=math.exp(logB),allowance_evidence=ev,exact_tail=zz(ref if pure else corrected),
            outer_PF_or_discard_bias=zz(0j if pure else corrected-ref),finite_RTE_bias=zz(corrected-ref if pure else 0j),
            exact_tail_is_saved_reference=pure),
        local=dict(absolute_corrected_bias=abs(corrected-ref),corrected_phase_to_reference=phase(corrected),raw_phase_to_reference=phase(raw),
            corrected_radius=abs(corrected),raw_radius=abs(raw),reference_radius=abs(ref)),allowance=allow,
        numerical=dict(norm_error=norm_error,power_unitarity_error=unitary_error,closure=closure,
            spectral_residuals=deepcopy(ctx.spectral_residuals),gates_passed=True,certified=False),
        scalar_once=True,raw_corrected_independent_propagation=pure,raw_equals_corrected_at_B1=not pure,finite_normalizer_calls=normalizers,
        reference_recomputed=False,N_X=None,N_Y=None,global_RMSE_bound=None,scientific_PASS_FAIL=None,
        qualification='LOCAL_SCREEN_NOT_GLOBAL_RMSE')
