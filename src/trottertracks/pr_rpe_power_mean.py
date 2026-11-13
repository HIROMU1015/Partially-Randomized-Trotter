"""Explicit finite-mean leaf reusing archived spectral polynomial powers.

No inputs or numerical libraries are loaded on import. Real molecular calls
require their own scientific permission. No sampling, decoder, prices or N.
"""
import ast
from copy import deepcopy
import math
from types import SimpleNamespace
from . import h4_pr_m1_rmse as previous
p=previous.p
SOURCE='e417ffc9f52b13c18f5493193ce510d858664259'
CORE='scripts/h46_new2_core.py'
SIGNAL='scripts/h46_long_time_signal.py'


def kernels(root):
    import numpy as np
    from scipy.linalg import eigh
    from trotterlib.rte import finite_rte_distribution
    scope=dict(np=np,eigh=eigh,math=math)
    for name in ('allocation','polynomial','spectral_apply','spectrum'):
        previous.function(root,SOURCE,CORE,name,scope)
    previous.function(root,SOURCE,SIGNAL,'complex_record',scope)
    sensitivity=[]
    def distribution(tau,K):
        result=finite_rte_distribution(tau,K);b=result.exact_finite_distribution
        if tau==0 or not math.isfinite(b) or b<=1:raise ValueError('Unresolved finite-normalizer precision')
        sensitivity.append(math.log(math.nextafter(b,math.inf))-math.log(math.nextafter(b,0.)))
        return result
    scope.update(core=SimpleNamespace(weights=p._schedule().weights),finite_rte_distribution=distribution)
    node=next(n for n in ast.parse(previous.blob(root,SOURCE,SIGNAL)).body if isinstance(n,ast.FunctionDef) and n.name=='evolve_batch')
    # Strip local helper imports only, and expose existing final vectors. All
    # propagation, polynomial powers, normalization and SOURCE gates are intact.
    node.body=[n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    assert isinstance(node.body[-1],ast.Return)
    node.body[-1]=ast.parse('return records, v, maximum').body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),SOURCE+':'+SIGNAL,'exec'),scope)
    return SimpleNamespace(spectrum=scope['spectrum'],evolve=scope['evolve_batch'],sensitivity=sensitivity)


def spectral_context(ctx,root):
    """Spectra of existing head/tail matrices, never a new ground state."""
    import numpy as np
    k=kernels(root)
    matrices=list(ctx.matrices[:2])+[ctx.tails[1]['matrix']];specs=[k.spectrum(m) for m in matrices]
    residuals=[dict(reconstruction_frobenius=float(np.linalg.norm((s[1]*s[0])@s[2]-m,'fro')),
        isometry_frobenius=float(np.linalg.norm(s[2]@s[1]-np.eye(len(s[0])),'fro')),
        source_gate_tolerance=1e-9,rigorous_bound=False) for m,s in zip(matrices,specs)]
    return SimpleNamespace(ham=ctx.ham,state=ctx.state,specs=tuple(specs[:2]),
        tails={1:dict(spec=specs[2],scalar=ctx.tails[1]['scalar'],lam=ctx.tails[1]['lambda_R'])},
        refs={},scientific_binding=deepcopy(ctx.scientific_binding),spectral_residuals=residuals)


def evolve(ctx,row,reference,*,root):
    import numpy as np
    c=row['candidate'];st=row['structure']
    if c['method']!='PR' or c['prefix']!=1 or c['K']!=2 or c['formula']!='2nd':
        raise ValueError('Fixed H4 S2/prefix1/K2 only')
    if c['id']!=p.candidate_id(c) or not p.same(st,p.structure(c)):raise ValueError('Candidate/structure differs')
    k=kernels(root);view=SimpleNamespace(**vars(ctx));view.refs={c['T']:complex(reference['real'],reference['imag'])}
    records,v,maximum=k.evolve(view,[c]);out=records[0]
    ev=out['signal']['allowance_evidence']
    ev['exact_tail_norm_error']=abs(float(np.linalg.norm(v[:,0]))-1.)
    ev['maximum_norm_domain']='ACTUAL_SPECTRAL_BLOCK_BOUNDARIES'
    ev['norm_envelope_rationale']='Hermitian polynomial is normal; exact-arithmetic power norm squared is convex in exponent. Floating-point equivalence remains empirical.'
    ev['normalizer_one_ulp_log_sensitivity']=c['q']*c['R_outer']*k.sensitivity[0]
    if not math.isfinite(ev['normalizer_one_ulp_log_sensitivity']) or math.exp(-out['signal']['log_B'])==0:
        raise ValueError('Unresolved normalization range')
    out.update(signed_weights=deepcopy(st['signed_weights']),allocation=deepcopy(st['allocation']),event_count=st['event_count'],
        backend=dict(kind='ARCHIVED_SPECTRAL_POLYNOMIAL_POWER',source=dict(commit=SOURCE,path=SIGNAL),
            logical_event_count=st['event_count'],tail_block_applications=c['q'],
            spectral_residuals=deepcopy(ctx.spectral_residuals),
            raw_corrected_independent_propagation=True,scalar_once=True,approximate_algorithm_substitution=False))
    return out
