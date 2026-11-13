"""Approved one-cell H4 diagnostic: two predeclared paths, no fallback/retry."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import h4_four_method_screen as screen

HELPER_SOURCE='dd24dd329949c255d5e37b90ff8993fa8df7159b'
RECIPE_SOURCE='256caad3414a80de02fb68eca89c13c957ed6354'
MODULE='scripts/run_pf_power_stability_h4.py'


def selected_cell(planned):
    row=next(r for r in planned['fresh'] if r['label']=='F2-COARSE' and r['m']==12)
    c=row['candidate']
    if (c['id'],c['method'],c['formula'],c['prefix'],c['T'],c['q'],c['delta'])!=(
            '4f0aca817edccc7fbe17','FULL','2nd',10,409.6,1024,.4):
        raise ValueError('Fixed one-cell scope differs')
    if row['structure']['signed_weights']!=[1.]:raise ValueError('S2 signed recipe differs')
    return deepcopy(row)


def observe_route(step,c,state,constant):
    import numpy as np
    from trottertracks.pf_power_stability import power_diagnostics
    powered,metrics=power_diagnostics(step,c['q'],state)
    propagated=powered@state
    propagated*=np.exp(-1j*c['T']*constant)  # Original total-T scalar once.
    mean=complex(np.vdot(state,propagated))
    if not math.isfinite(mean.real) or not math.isfinite(mean.imag):raise ValueError('Nonfinite mean')
    metrics.update(step_state_norm=float(np.linalg.norm(step@state)),power_state_norm=float(np.linalg.norm(propagated)))
    return dict(numerical=metrics,mean=dict(real=mean.real,imag=mean.imag),
        raw_equals_corrected_at_B1=True,log_B=0.,B=1.,attenuation=1.,scalar_once=True,
        status='LOCAL_NUMERICAL_GATE_PASS' if metrics['gate_pass'] else 'LOCAL_NUMERICAL_GATE_FAIL',
        N=None,RMSE=None,scientific_PASS_FAIL=None),powered,propagated


def execute(out,source):
    import numpy as np
    import scipy
    from trottertracks import h4_pr_rmse_completion as old
    from trottertracks import h4_four_method_screen_mean as leaf
    from trottertracks import pf_power_stability as diagnostic
    from trottertracks.pr_rpe_power_mean import kernels
    out=Path(out)
    if out.exists():raise ValueError('Fresh diagnostic output required')
    bound={MODULE:source,'src/trottertracks/pf_power_stability.py':HELPER_SOURCE,
        screen.MODULE:RECIPE_SOURCE,screen.MEAN:RECIPE_SOURCE,
        'src/trottertracks/pr_rpe_power_mean.py':RECIPE_SOURCE,
        'src/trottertracks/h4_pr_rmse_completion.py':RECIPE_SOURCE,
        'src/trottertracks/h4_df10_rpe_batch_provider.py':RECIPE_SOURCE,
        'src/trottertracks/h6_df19_rpe_d2.py':RECIPE_SOURCE,
        'scripts/h46_new2_core.py':RECIPE_SOURCE}
    for path,commit in bound.items():
        if (ROOT/path).read_bytes()!=screen.M.blob(ROOT,commit,path):raise ValueError('Scientific SOURCE differs: '+path)
    planned=screen.plan(ROOT)
    fixed=json.loads(screen.M.blob(ROOT,RECIPE_SOURCE,'docs/research/h4_four_method_signal_screen_preparation_20261113/PLAN.json'))
    if planned!=fixed:raise ValueError('Fixed recipe PLAN differs')
    row=selected_cell(planned);c=row['candidate'];out.mkdir(parents=True)
    def save(name,value):
        with (out/name).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save('SCOPE.json',dict(candidate=row,binding=planned['binding'],profile=planned['profile'],
        sources=bound,recipe_source=RECIPE_SOURCE,helper_source=HELPER_SOURCE,
        maximum_complex_means=2,new_reference_actions=0,normalizer_calls=0,
        paths=['ARCHIVED_SPECTRAL','DIRECT_EXPM'],gate_failure_is_observed_not_fallback=True))
    started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    result=dict(source=source,recipe_source=RECIPE_SOURCE,helper_source=HELPER_SOURCE,
        status='STARTED',routes=[],errors=[],new_reference_actions=0,normalizer_calls=0,
        automatic_retry=False,global_RMSE_bound=None,sufficient_N=None,winner=None,mandatory_stop=True)
    try:
        if (np.__version__,scipy.__version__)!=('1.26.4','1.14.1'):raise ValueError('Fixed numerical runtime differs')
        physics=screen.p.SavedArtifacts(ROOT).read(screen.PHYS)
        pair=deepcopy(physics['rounds'][12]['diagnostics']['reference_pair'])
        z=lambda v:complex(v['real'],v['imag'])
        ref=z(pair['reference']);other=z(pair['independent_reference'])
        if (not all(math.isfinite(v) for v in (ref.real,ref.imag,other.real,other.imag,pair['signal_difference']))
                or not 0<=pair['signal_difference']<=1e-10 or abs(ref-other)>1e-10):raise ValueError('Saved reference gate')
        save('REFERENCE.json',dict(source=row['reference'],evidence=pair,recomputed=False))
        saved,state_metadata=old.load_context(ROOT,planned['binding'])
        if saved.ham.constant!=planned['full_h_preparation']['scalar']['constant']:raise ValueError('Saved scalar differs')
        archived=kernels(ROOT)
        specs=tuple(archived.spectrum(a) for a in saved.matrices)
        metrics=[dict(block=i,**diagnostic.spectral_metrics(a,s)) for i,(a,s) in enumerate(zip(saved.matrices,specs))]
        save('SPECTRAL.json',dict(source=kernels.__module__,blocks=metrics,
            shared_input_diagnostics=True,direct_expm_eigenvectors=None,direct_expm_spectral_reconstruction=None,
            direct_expm_reason='No eigendecomposition in direct expm path'))
        ctx=SimpleNamespace(state=saved.state,specs=specs,cache={})
        archive_step=leaf.deterministic_step(ctx,c,row['structure']['signed_weights'])
        exp_cache={};direct_step=diagnostic.expm_step(saved.matrices,row['structure']['signed_weights'],c['delta'],exp_cache)
        block_rows=[]
        for i,s in enumerate(specs):
            spectral=(s[1]*np.exp(-.5j*c['delta']*s[0]))@s[2]
            direct=exp_cache[i,.5*c['delta']]
            block_rows.append(dict(block=i,
                spectral_half_unitarity=float(np.linalg.norm(spectral.conj().T@spectral-np.eye(len(saved.state)),'fro')),
                direct_half_unitarity=float(np.linalg.norm(direct.conj().T@direct-np.eye(len(saved.state)),'fro')),
                half_difference_fro=float(np.linalg.norm(spectral-direct,'fro'))))
        save('BLOCK_FACTORS.json',dict(half_time=.5*c['delta'],blocks=block_rows))
        arrays={};powers={};vectors={}
        # Both paths are predeclared, irrespective of either finite gate result.
        # Nonfinite/input errors still stop this finite experiment immediately.
        for name,step in (('ARCHIVED_SPECTRAL',archive_step),('DIRECT_EXPM',direct_step)):
            record,powered,vector=observe_route(step,c,saved.state,saved.ham.constant)
            record.update(path=name,candidate=deepcopy(c),binding=deepcopy(planned['binding']))
            mean=z(record['mean']);record['saved_reference']=pair['reference']
            record['local_signal_bias']=dict(real=(mean-ref).real,imag=(mean-ref).imag)
            save(name+'.json',record);result['routes'].append(record)
            arrays[name+'_step']=step;arrays[name+'_power']=powered
            powers[name]=powered;vectors[name]=vector
        np.savez(out/'OPERATORS.npz',**arrays)
        result['comparison']=dict(
            step_difference_fro=float(np.linalg.norm(archive_step-direct_step,'fro')),
            power_difference_fro=float(np.linalg.norm(powers['ARCHIVED_SPECTRAL']-powers['DIRECT_EXPM'],'fro')),
            propagated_vector_difference=float(np.linalg.norm(vectors['ARCHIVED_SPECTRAL']-vectors['DIRECT_EXPM'])),
            complex_mean_difference_abs=abs(z(result['routes'][0]['mean'])-z(result['routes'][1]['mean'])),
            direct_expm_spectral_orthogonality=None,direct_expm_has_no_eigenvectors=True,
            new_acceptance_threshold_adopted=False)
        result['state_metadata']=state_metadata
        result['status']='TWO_PATH_DIAGNOSTIC_COMPLETE_NOT_GLOBAL_QUALIFICATION'
    except Exception as error:
        result['status']='STOP_UNRESOLVED_NO_RETRY';result['errors'].append(dict(type=type(error).__name__,message=str(error)))
    save('result.json',result)
    save('EXECUTION.json',dict(source=source,started_UTC=started,finished_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-tick,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        import_paths=[str(ROOT/'src'),str(ROOT/'scripts')],status=result['status'],complex_means_saved=len(result['routes']),
        new_reference_actions=0,normalizer_calls=0,automatic_retry=False,mandatory_stop=True))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True)
    a=p.parse_args();r=execute(a.out,a.source)
    print(json.dumps(dict(status=r['status'],complex_means_saved=len(r['routes']),errors=r['errors'])))


if __name__=='__main__':main()
