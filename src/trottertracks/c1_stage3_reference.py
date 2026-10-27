"""Small-matrix Stage3 reference; dormant until a separately permitted call.

Accepts original CAR generators, not Qiskit operators or candidate K expansion.
No saved molecular values are loaded here. Do not restrict generators to a
sector unless all relevant intermediate evolutions preserve that sector.
"""
from fractions import Fraction
from .c1_stage3_accuracy import _legacy
from . import c1_stage2_native as native


def _ordered_exponentials(generators, schedule, cache=None,*,arithmetic=None):
    import numpy as np
    from scipy.linalg import expm
    result = (np.eye(len(generators[0]), dtype=complex) if arithmetic is None
              else arithmetic.eye(len(generators[0])))
    def evaluate(index,time):
        return (expm(-1j*float(time)*generators[index]) if arithmetic is None
                else arithmetic.exponential(time,generators[index]))
    for index, time in schedule:
        key = (index,Fraction(time))
        if cache is None:
            factor=evaluate(index,time)
        else:
            if key not in cache:cache[key]=evaluate(index,time)
            factor=cache[key]
        result = factor @ result
    return result


def reference_kernels(generators, step_time, *, ld, identity_tail=0., rte_config=None,
                      exponential_cache=None):
    """Return ideal PF and normalization-corrected finite RTE kernels.

    generators[0] includes c exactly once through its two half evolutions.
    Remaining generators are original frames. Tail identity is extracted and
    applied exactly; it is absent from the sampling norm lambda_R.
    """
    import numpy as np
    from scipy.linalg import expm
    generators = tuple(np.asarray(h, dtype=complex) for h in generators)
    if not generators or type(ld) is not int or not 0 <= ld < len(generators):
        raise ValueError('Valid original-frame prefix required')
    d = len(generators[0]); eye = np.eye(d,dtype=complex)
    if any(h.shape != (d,d) or not np.isfinite(h).all() for h in generators):
        raise ValueError('Compatible finite generator matrices required')
    h = Fraction(step_time)
    if ld == len(generators)-1:
        pf = _ordered_exponentials(generators,native.ordered_schedule(len(generators),h),exponential_cache)
        return dict(pf=pf,corrected=pf,config=None,step_time=float(h))
    if rte_config is None or not np.isclose(float(h),rte_config.evolution_time,rtol=1e-14,atol=0):
        raise ValueError('Finite-RTE configuration must match the PF kernel time')
    forward = tuple((i,h/2) for i in range(ld+1))
    # Cache must be scoped to one immutable arm/model, never shared between
    # CDF and XDF. It reuses only exact block/time exponential requests.
    left = _ordered_exponentials(generators,forward,exponential_cache)
    right = _ordered_exponentials(generators,tuple(reversed(forward)),exponential_cache)
    tail = sum(generators[ld+1:],np.zeros((d,d),complex))-identity_tail*eye
    phase = np.exp(-1j*float(h)*identity_tail)
    pf = phase*right @ expm(-1j*float(h)*tail) @ left
    corrected_tail = _legacy('rte').finite_rte_corrected_operator(tail/rte_config.lambda_r,rte_config)
    return dict(pf=pf,corrected=phase*right @ corrected_tail @ left,config=rte_config,step_time=float(h))


def reference_round(kernels, full_model, *, repetitions, state=None, exact_operator=None):
    """Matrix powers avoid trajectory enumeration and full long-q circuits.

    PF and corrected finite-RTE discrepancies are in the same operator norm.
    Scaling by A to observed signal units occurs only in assess_round().
    """
    import numpy as np
    from scipy.linalg import expm
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('Positive coherent repetition count required')
    # Caller supplies elapsed physical time rather than inferring it from a
    # matrix phase; no exact-time/approximate-time comparison ambiguity.
    full, time = full_model
    if not np.isclose(float(time),kernels['step_time']*repetitions,rtol=1e-14,atol=0):
        raise ValueError('Reference elapsed time differs from repeated kernel time')
    config = kernels['config']
    attenuation = 1. if config is None else _legacy('rte').finite_rte_attenuation(
        config,tail_evolutions=repetitions)
    if attenuation == 0:
        # Do not first form a huge normalization-corrected matrix power when
        # the observed signal has already underflowed at this fixed precision.
        return dict(status='INFEASIBLE_OBSERVED_RADIUS',attenuation=0.,
                    pf_operator_error=None,rte_corrected_operator_error=None)
    exact = expm(-1j*float(time)*np.asarray(full)) if exact_operator is None else np.asarray(exact_operator)
    pf = np.linalg.matrix_power(kernels['pf'],repetitions)
    corrected = np.linalg.matrix_power(kernels['corrected'],repetitions)
    result=dict(pf_operator_error=float(np.linalg.norm(pf-exact,2)),
                rte_corrected_operator_error=float(np.linalg.norm(corrected-pf,2)),
                attenuation=attenuation)
    if state is not None:
        psi=np.asarray(state,dtype=complex)
        if not np.isclose(np.vdot(psi,psi),1.,rtol=0,atol=1e-12):
            raise ValueError('Normalized target state required')
        result['raw_signal']=complex(np.vdot(psi,attenuation*corrected@psi))
    return result


def sample_rpe_phases(signals, shots_per_axis, *, trials, seed):
    """Reuse binomial marginals and existing phase reconstruction.

    This simulates classical marginal outcomes only. It neither executes
    quantum shots nor enumerates exponentially many per-shot PR trajectories.
    Signals must already describe independent trajectories per actual shot.
    """
    import numpy as np
    if type(trials) is not int or trials<1 or len(signals)!=len(shots_per_axis) or not len(signals):
        raise ValueError('Matching round data and positive trial count required')
    if any(type(n) is int and n>np.iinfo(np.int64).max for n in shots_per_axis):
        return dict(status='VIRTUAL_BINOMIAL_INTEGER_RANGE_EXCEEDED',
                    virtual_trials=0,actual_quantum_shots_executed=0)
    rng=np.random.default_rng(seed);phases=[]
    for z,n in zip(signals,shots_per_axis):
        z=complex(z)
        if type(n) is not int or n<1 or not np.isfinite([z.real,z.imag]).all() or abs(z)>1+1e-12:
            raise ValueError('Valid normalized observed signal and positive shots required')
        x=2*rng.binomial(n,(1+np.clip(z.real,-1,1))/2,size=trials)/n-1
        y=2*rng.binomial(n,(1+np.clip(z.imag,-1,1))/2,size=trials)/n-1
        phases.append(np.arctan2(y,x))
    estimates,_,ambiguous=_legacy('rpe_four_round_phase_validation')._vectorized_reconstruction(
        phases,[1<<m for m in range(len(phases))])
    return dict(phase_estimates=estimates.tolist(),ambiguous=ambiguous.tolist(),
                virtual_trials=trials,actual_quantum_shots_executed=0)
