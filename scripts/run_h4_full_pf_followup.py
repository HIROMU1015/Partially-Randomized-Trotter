"""Approved five-cell Full PF follow-up; one saved M5 window check, no retry."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import prepare_h4_full_pf_followup as planner
import run_h4_full_pf_heff_scan as existing

previous=existing.previous
E=existing.E
PREPARATION='d48d8732ee7ef521aab5d67a0945439364f5a2f8'
PLAN_PATH='docs/research/h4_full_pf_followup_plan_20261114/PLAN.json'
MODULE='scripts/run_h4_full_pf_followup.py'
COMPLETE='FIVE_CELL_HEFF_FOLLOWUP_COMPLETE_NOT_RPE_QUALIFIED'


def load_plan():
    plan=json.loads(previous.blob(PREPARATION,PLAN_PATH))
    if not previous.P.same(planner.build(),plan):raise ValueError('Fixed follow-up PLAN differs')
    cells=[(i,c) for i,c in enumerate(plan['step_cells']) if c['new_science_required'] and c['branch']['mode'] is not None]
    if len(cells)!=5 or plan['metadata_counts']['held_new_actual_steps']!=1:
        raise ValueError('Approved five-cell scope differs')
    _,reference,operator,tolerances,policy=existing.load_plan()
    operator['identity']=dict(binding=plan['shared']['binding'])
    return plan,cells,reference,operator,tolerances,policy


def check_saved_window(saved,branch,delta,tolerances):
    """One saved-spectrum check, no new operator/Schur/H/reference action."""
    import numpy as np
    result=dict(status='UNKNOWN',branch=deepcopy(branch),old_delta=delta,metrics={},errors=[],
        new_operator_or_eigensolve=0,old_signal_reanalysis=0,physical_alias_qualified=False,
        ground_identification_qualified=False,global_RMSE_bound=None)
    def require(test,reason):
        if not test:raise ValueError(reason)
    try:
        require(saved['status']=='EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE','Saved spectrum unresolved')
        require(not saved['renormalized'] and not saved['unitary_repaired'],'Saved numerical repair differs')
        require(branch['mode']=='ENERGY_WINDOW' and branch['purpose']=='EFFECTIVE_SPECTRUM_WINDOW','Explicit spectral window required')
        lo,hi=branch['window'];center=lo+(hi-lo)/2
        require(lo<hi and (hi-lo)*delta<2*np.pi,'Old-step branch window width')
        sp=saved['spectrum'];energies=np.asarray(sp['energies_ha'],float);weights=np.asarray(sp['state_weights'],float)
        values=np.asarray(sp['unitary_eigenvalues']['real'])+1j*np.asarray(sp['unitary_eigenvalues']['imag'])
        require(energies.ndim==1 and len(energies)>0 and weights.shape==values.shape==energies.shape,'Saved spectral shape')
        require(all(np.isfinite(a).all() for a in (energies,weights,values)),'Nonfinite saved spectrum')
        require(np.all(np.diff(energies)>=0) and np.all(weights>=0),'Saved spectral order/weights')
        rotated=np.angle(np.exp(1j*center*delta)*values);lifted=center-rotated/delta
        metrics=result['metrics']
        metrics.update(energy_lift_difference_max_ha=float(np.max(abs(lifted-energies))),
            weight_sum_error=abs(float(np.sum(weights))-1.),
            selected_cut_clearance_rad=float(np.min(np.pi-abs(rotated))),
            window_edge_clearance_rad=float(np.min(np.minimum(lifted-lo,hi-lifted))*delta))
        require(metrics['energy_lift_difference_max_ha']<=tolerances['generator_hermiticity_ha'],'Window changes saved energy branch')
        require(metrics['weight_sum_error']<=tolerances['reconstruction'],'Saved spectral weight sum')
        require(min(metrics['selected_cut_clearance_rad'],metrics['window_edge_clearance_rad'])>tolerances['phase_guard_rad'],'Saved spectrum/window cut or edge')
        groups=E._groups(lifted,weights,tolerances['energy_cluster_ha']);original=sp['energy_groups']
        require([g['indices'] for g in groups]==[g['indices'] for g in original],'Window changes saved energy groups')
        metrics['group_weight_difference_max']=max(abs(g['weight']-o['weight']) for g,o in zip(groups,original))
        require(metrics['group_weight_difference_max']<=tolerances['reconstruction'],'Window changes saved group weights')
        d=saved['diagnostics'];limits=dict(input_state_norm_error=1e-12,unitarity_defect_fro=tolerances['unitarity'],
            H_eff_hermiticity_defect_fro=tolerances['generator_hermiticity_ha'],reference_eigen_residual_fro=tolerances['generator_hermiticity_ha'])
        for k in ('schur_offdiagonal_fro','eigenvalue_modulus_error','eigenvector_gram_defect_fro','unitary_eigen_residual_fro',
                  'exponential_reconstruction_fro','weight_sum_error','reference_eigenvector_gram_defect_fro','reference_weight_sum_error'):
            limits[k]=tolerances['reconstruction']
        result['saved_numerical_gates']={k:dict(value=d[k],limit=v,pass_gate=bool(np.isfinite(d[k]) and d[k]<=v)) for k,v in limits.items()}
        require(all(g['pass_gate'] for g in result['saved_numerical_gates'].values()),'Saved numerical gate failed')
        overlaps=saved['reference_comparison']['reference_lowest_group_to_effective_groups_trace_overlap']
        require(len(overlaps)==len(groups) and np.isfinite(overlaps).all(),'Saved reference/group correspondence')
        result.update(status='SAVED_WINDOW_COMPATIBILITY_DIAGNOSTIC_PASS',dimension=len(energies),
            energy_groups_unchanged=True,state_weights_unchanged=True,
            original_lowest_group_max_trace_group_indices=[i for i,v in enumerate(overlaps) if v==max(overlaps)],
            original_lowest_group_max_trace_overlap=max(overlaps),reference_energy_not_used_to_choose_branch=True)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:
        result['errors']=[dict(type=type(error).__name__,message=str(error))]
    return E._finite_payload(result)


def execute(out,source):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required; no retry/resume')
    out.mkdir(parents=True)
    def write(directory,name,value):
        directory.mkdir(parents=True,exist_ok=True)
        if name.endswith('.npz'):
            import numpy as np
            with (directory/name).open('xb') as f:np.savez(f,**value)
        else:
            with (directory/name).open('x') as f:f.write(json.dumps(E._finite_payload(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();current=None
    result=dict(status='STARTED',source=source,preparation=PREPARATION,cell_results=[],errors=[],
        counts=dict(context_load_attempts=0,new_PF_step_attempts=0,saved_M5_08_window_checks=0),
        original_H_eigh=0,new_reference_actions=0,old_saved_signal_reanalysis=0,normalizer_N_MC_C_G=0,
        M5_32h_executed=False,automatic_retry=False,automatic_extension=False,
        sufficient_N=None,rpe_bias_ha=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True)
    try:
        fixed={MODULE:source,'scripts/run_h4_full_pf_heff_scan.py':source,
            'scripts/prepare_h4_full_pf_followup.py':PREPARATION,'scripts/prepare_h4_full_pf_heff_scan.py':existing.PREPARATION,
            'src/trottertracks/pf_effective_hamiltonian.py':existing.PREPARATION,
            'scripts/run_h4_full_pf_heff.py':'3a26b9ba70f7c5022a7e07356fe1df8b620545e8',
            'src/trottertracks/pf_power_stability.py':'dd24dd329949c255d5e37b90ff8993fa8df7159b'}
        for path,commit in fixed.items():
            if (ROOT/path).read_bytes()!=previous.blob(commit,path):raise ValueError('Scientific SOURCE differs: '+path)
        plan,cells,reference,operator,tolerances,policy=load_plan()
        budget=float(Fraction(plan['shared']['rmse_policy']['error_contract']['A_policy']['evolution_rmse']))
        anchor=json.loads(previous.blob(planner.previous.ANCHOR,planner.previous.ANCHOR_DIR+'EIGENSYSTEM.json'))
        anchor_binding=anchor['operator']['identity']['step_key']['binding']
        if (not previous.P.same(anchor['reference'],reference)
                or anchor['operator']['target_hamiltonian']!=plan['shared']['binding']['hamiltonian']
                or any(anchor_binding.get(k)!=plan['shared']['binding'][k] for k in ('hamiltonian','state','snapshot','rank','rank_policy'))):
            raise ValueError('Anchor/reference SOURCE binding differs')
        cache=dict(source=plan['shared']['cached_reference_eigensystem'],hamiltonian=reference['hamiltonian'],
            eigenvalues_ha=anchor['reference_comparison']['eigenvalues_ha'],eigenvectors=anchor['reference_comparison']['eigenvectors'])
        write(out,'SCOPE.json',dict(plan_source=dict(commit=PREPARATION,path=PLAN_PATH),scientific_sources=fixed,
            approved_cell_indices=[i for i,_ in cells],held_M5_32h_excluded=True,
            plan=plan,reference=reference,tolerances=tolerances,numerical_policy=policy,
            cached_eigensystem_source=cache['source'],evolution_budget_ha=budget))
        # Only the approved old M5/.8 spectrum is numerically rechecked.
        h=planner.previous.exact(plan['shared']['h'])
        saved_cell=next(c for c in plan['step_cells'] if c['formula']=='4th(m5_best)' and planner.previous.exact(c['actual_delta_pf'])==8*h)
        saved=json.loads(previous.blob(saved_cell['saved_source']['commit'],saved_cell['saved_source']['path']))
        op=saved['operator'];identity=op['identity']
        if (op['kind']!='FULL_PF' or identity['formula']!='4th(m5_best)'
                or identity['actual_delta_pf']!=saved_cell['actual_delta_pf']
                or not previous.P.same(identity['binding'],plan['shared']['binding'])
                or identity['signed_weights']!=saved_cell['signed_weights']
                or op['scalar_constant_ha']!=anchor['operator']['scalar_constant_ha']
                or op['target_hamiltonian']!=reference['hamiltonian']
                or op['retained_hamiltonian']!=reference['hamiltonian']
                or op['scalar_policy']!='INCLUDED_ONCE_AT_STEP_TIME'
                or not previous.P.same(saved['reference'],reference)):
            raise ValueError('Saved M5 .8 SOURCE/physical binding differs')
        m5=next(c for _,c in cells if c['formula']=='4th(m5_best)')
        branch={k:deepcopy(v) for k,v in m5['branch'].items() if k!='status'}
        branch['source']=dict(commit=PREPARATION,path=PLAN_PATH)
        result['counts']['saved_M5_08_window_checks']=1
        window=check_saved_window(saved,branch,float(8*h),tolerances)
        window['saved_source']=saved_cell['saved_source'];write(out,'SAVED_M5_08_WINDOW_CHECK.json',window)
        if window['status']!='SAVED_WINDOW_COMPATIBILITY_DIAGNOSTIC_PASS':raise ValueError('Saved M5 .8 window gate')
        api=previous.numerical_api();result['counts']['context_load_attempts']=1
        saved_inputs,binding,metadata=api.load_shared_h4(ROOT,plan['shared']['binding'])
        ctx=api.Context(ROOT,saved_inputs,binding,metadata)
        if ctx.constant!=anchor['operator']['scalar_constant_ha']:raise ValueError('Original scalar differs')
        write(out,'INPUT_PROVENANCE.json',dict(binding=binding,inputs=metadata['source'],sector=[2,2],dimension=len(ctx.state),
            scalar_constant_ha=ctx.constant,original_H='saved_context.full'))
        for i,cell in cells:
            current=cell['label'];directory=out/('cell_'+str(i).zfill(2));print('START '+current,flush=True)
            write(directory,'CELL_SCOPE.json',cell)
            branch={k:deepcopy(v) for k,v in cell['branch'].items() if k!='status'}
            branch['source']=dict(commit=PREPARATION,path=PLAN_PATH,selector=['step_cells',i,'branch'])
            result['counts']['new_PF_step_attempts']+=1
            summary=existing.diagnose_cell(ctx,saved_inputs.full,cell,reference=reference,operator=operator,tolerances=tolerances,
                cache=cache,policy=policy,budget=budget,branch=branch,save=lambda n,v:write(directory,n,v))
            summary.update(path=directory.name,status='LOCAL_DIAGNOSTIC_COMPLETE_NOT_RPE_QUALIFIED',source=source)
            write(directory,'result.json',summary);result['cell_results'].append(summary)
            print('DONE '+current+' '+summary['local_spectral_budget_screen'],flush=True)
        result['status']=COMPLETE
        files=list(metadata['source']['inputs'].values())+[metadata['source']['context']]
        result['input_hashes_after']=[dict(path=f['path'],sha256=hashlib.sha256(Path(f['path']).read_bytes()).hexdigest(),
            expected=f['sha256']) for f in files]
        if any(f['sha256']!=f['expected'] for f in result['input_hashes_after']):raise ValueError('Scientific input changed during run')
    except Exception as error:
        result.update(status='STOP_PARTIAL_NO_RETRY',failed_cell=current,errors=[dict(type=type(error).__name__,message=str(error))])
    result['counts'].update(new_PF_steps_saved=len(list(out.glob('cell_*/OPERATORS.npz'))),
        effective_evaluator_calls=len(list(out.glob('cell_*/EIGENSYSTEM.json'))),
        direct_means=len(list(out.glob('cell_*/round_*.json'))),spectral_means=len(list(out.glob('cell_*/round_*.json'))))
    write(out,'result.json',result)
    import numpy,scipy
    write(out,'EXECUTION.json',dict(source=source,preparation=PREPARATION,started_UTC=started,
        finished_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick,
        python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,status=result['status'],
        counts=result['counts'],automatic_retry=False,mandatory_stop=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();value=execute(a.out,a.source)
    print(json.dumps(dict(status=value['status'],errors=value['errors'],counts=value['counts'])),flush=True)
    sys.exit(0 if value['status']==COMPLETE else 2)
