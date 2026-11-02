"""Keep only discarded-tail identity, using the immutable B0 circuit builder."""
import cmath
import math
from ..h4_geometry.identity import require, fingerprint, hash_id

METHOD = 'B0_PLUS_I_VALIDATION'
SEMANTICS = 'h4-b0i-ordinary-structured-jw-validation-v1'
NAMESPACE = 'h4-b0i-pilot6-validation-20261013'
ORIGINAL_SOURCE = 'd0726e946f89f8c30a08d383fb8ff48b363629ec'


def tail_scalar_lower(matrices, lambdas, prefix):
    """Trace identity on the lower-Hermitian convention; no eigensolver."""
    require(len(matrices)==len(lambdas) and type(prefix) is int and 0<=prefix<=len(matrices), 'DF tail shape/prefix')
    values=[]
    for matrix, lam in zip(matrices[prefix:],lambdas[prefix:],strict=True):
        n=len(matrix);require(n>0 and all(len(row)==n for row in matrix),'square DF fragment')
        require(math.isfinite(float(lam)), 'finite lambda')
        trace=[];squares=[]
        for i in range(n):
            diagonal=complex(matrix[i][i])
            require(diagonal.imag==0 and math.isfinite(diagonal.real),'real finite diagonal')
            trace.append(diagonal.real);squares.append(diagonal.real**2)
            for j in range(i):
                lower=complex(matrix[i][j])
                require(math.isfinite(lower.real) and math.isfinite(lower.imag),'finite lower entry')
                squares.append(2*abs(lower)**2)
        value=float(lam)/4*(math.fsum(trace)**2+math.fsum(squares))
        require(math.isfinite(value),'finite identity coefficient');values.append(value)
    return math.fsum(values)


def template(prefix, q, T=.8):
    require(type(prefix) is int and prefix>=0 and type(q) is int and q in (1,2,4,8),'B0+I domain')
    require(type(T) is float and math.isfinite(T) and T>0,'finite positive time')
    return dict(method=METHOD,template_id=f'B0I-validation-p{prefix}-q{q}',L_D=prefix,q=q,
                r=0,K=0,T=T,delta=T/q,PF_order=2,control='ordinary',wrapper_semantics=SEMANTICS)


def add_tail_identity(b0, spec, scalar):
    require(spec['method']==METHOD and spec['r']==spec['K']==0,'separate deterministic method')
    require(set(b0)=={'n','deterministic','components','constant'} and not b0['components'], 'original B0 preparation only')
    require(len(b0['deterministic'])==spec['L_D']+1, 'kept one-body plus prefix; no tail blocks')
    require(type(scalar) is float and math.isfinite(scalar) and math.isfinite(b0['constant']), 'finite tail/constant')
    # No kept-block coefficients are touched, no extra scalar gates are appended.
    return {**b0,'deterministic':list(b0['deterministic']),
            'components':[], 'constant':b0['constant']+scalar,
            'b0i_tail_identity_applied':scalar,'b0_original_constant':b0['constant']}


def evolution(b0, spec, scalar):
    from ..h4_geometry.circuits import build_evolution
    return build_evolution(add_tail_identity(b0,spec,scalar),spec,[[] for _ in range(spec['q'])])


def rotate_saved_signal(z, T, scalar):
    require(all(math.isfinite(float(v)) for v in (z.real,z.imag,T,scalar)),'finite saved signal')
    return cmath.exp(-1j*T*scalar)*z


def bound_fingerprint(spec, scalar, axis, input_identity, source_commit, numeric):
    require(axis in ('cosine','sine') and spec['method']==METHOD,'validation identity axis/method')
    require(len(source_commit)==40 and all(c in '0123456789abcdef' for c in source_commit),'actual source commit')
    for k in ('H','DF','state','input','compiler','environment'):hash_id(input_identity[k])
    hash_id(numeric)
    return fingerprint('h4-b0i-validation-wrapper-v1',dict(namespace=NAMESPACE,method=METHOD,
        template=spec,input=dict(input_identity),source=source_commit,original_source=ORIGINAL_SOURCE,
        tail_scalar=scalar,phase_angle=-spec['T']*scalar,axis=axis,control='ordinary',
        semantics=SEMANTICS,numeric_circuit_fingerprint=numeric,
        trajectory_seed=None,trajectory_index=None,old_B0_cache_reuse=False))
