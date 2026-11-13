"""Artificial-only PF stability experiment. No molecular input options."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def experiment():
    import numpy as np
    import scipy
    from trotterlib.h46_m5_plan import weights
    from trottertracks import pf_power_stability as D
    from trottertracks.h4_four_method_screen_mean import deterministic_step
    from trottertracks.pr_rpe_power_mean import kernels
    tick=time.monotonic();records=[];quality=[]
    for kind,seed in (('regular',20261113),('clustered_zero',20261114)):
        matrices,state=D.artificial_blocks(kind,seed)
        archived=kernels(ROOT)
        systems={'archived_evr':tuple(archived.spectrum(a) for a in matrices)}
        for driver in ('evd','ev'):
            systems[driver]=tuple(D.explicit_spectrum(a,driver) for a in matrices)
        for backend,specs in systems.items():
            quality.append(dict(fixture=kind,seed=seed,backend=backend,
                blocks=[D.spectral_metrics(a,s) for a,s in zip(matrices,specs)]))
        exp_cache={};contexts={k:SimpleNamespace(state=state,specs=s,cache={}) for k,s in systems.items()}
        for formula in ('2nd','4th','4th(m5_best)'):
            for delta in (.1,.4):
                w=weights(formula);independent=D.expm_step(matrices,w,delta,exp_cache)
                steps={'expm':independent}
                candidate=dict(formula=formula,prefix=10,T=delta,q=1)
                for backend,ctx in contexts.items():
                    steps[backend]=deterministic_step(ctx,candidate,w)
                for q in (1,64,1024,4096):
                    ref,_=D.power_diagnostics(independent,q,state)
                    for backend,step in steps.items():
                        powered,metrics=D.power_diagnostics(step,q,state)
                        records.append(dict(fixture=kind,seed=seed,formula=formula,delta=delta,
                            backend=backend,pieces=len(w),block_exponentials_per_step=22*len(w),
                            **metrics,power_difference_to_expm_fro=float(np.linalg.norm(powered-ref,'fro'))))
    # Injected control, NOT an observed LAPACK output: reconstruction of A=0
    # cannot see a small Gram defect in its zero-eigenvalue subspace.
    n=36;vectors=np.eye(n,dtype=complex);vectors[0,1]=3e-12
    spec=(np.zeros(n),vectors,vectors.conj().T);zero=np.zeros((n,n),complex)
    state=np.zeros(n,complex);state[:2]=1/np.sqrt(2)
    ctx=SimpleNamespace(state=state,specs=(spec,),cache={})
    injected=[]
    for formula in ('2nd','4th(m5_best)'):
        step=deterministic_step(ctx,dict(formula=formula,prefix=0,T=0.,q=1),weights(formula))
        for q in (1,4096):
            _,row=D.power_diagnostics(step,q,state)
            injected.append(dict(formula=formula,**row))
    return dict(schema='ARTIFICIAL_PF_POWER_STABILITY_V1',scope='ARTIFICIAL_ONLY',
        numpy=np.__version__,scipy=scipy.__version__,dimension=36,blocks=11,
        archived_source='e417ffc9f52b13c18f5493193ce510d858664259',
        production_screen_source='256caad3414a80de02fb68eca89c13c957ed6354',
        spectral_quality=quality,records=records,
        injected_zero_control=dict(description='Manually injected nonorthogonal zero eigensystem; not an observed solver output',
            quality=D.spectral_metrics(zero,spec),records=injected),
        elapsed_seconds=time.monotonic()-tick,real_H4_calls=0,saved_numerical_reanalysis=0,
        production_changed=False,scientific_qualification=None,mandatory_stop=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Fresh artificial output required')
    result=experiment();a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(records=len(result['records']),elapsed_seconds=result['elapsed_seconds'],real_H4_calls=0)))


if __name__=='__main__':main()
