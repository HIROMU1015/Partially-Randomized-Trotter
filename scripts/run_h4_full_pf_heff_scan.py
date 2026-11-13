"""Approved finite 14-new-cell H_eff scan; cached reference, no retry/search/N/C."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import prepare_h4_full_pf_heff_scan as planner
import run_h4_full_pf_heff as previous
from trottertracks import pf_effective_hamiltonian as E

PREPARATION='878000d8d5d956b1d2f5513986ae40d0612e0bdd'
PLAN_PATH='docs/research/h4_full_pf_heff_scan_preparation_20261114/PLAN.json'
MODULE='scripts/run_h4_full_pf_heff_scan.py'
COMPLETE='FINITE_HEFF_SCAN_COMPLETE_NOT_RPE_QUALIFIED'


def load_plan():
    plan=json.loads(previous.blob(PREPARATION,PLAN_PATH))
    if not previous.P.same(planner.build(),plan):raise ValueError('Fixed canonical scan PLAN differs')
    if plan['metadata_counts']['new_step_cells']!=14 or len(plan['step_cells'])!=15:
        raise ValueError('Fixed finite scope differs')
    _,_,reference,operator,tolerances=previous.metadata()
    p=json.loads(previous.blob('9af1ac8ed8adcaef059f439ce690b22b7a320de0',previous.PLAN_PATH))
    return plan,reference,operator,tolerances,p['numerical_policy']


def local_summary(diagnostic,cell,records,budget):
    rc=diagnostic['reference_comparison'];sp=diagnostic['spectrum']
    overlaps=rc['reference_lowest_group_to_effective_groups_trace_overlap']
    largest=[i for i,v in enumerate(overlaps) if v==max(overlaps)]
    one=(len(largest)==1 and rc['energy_groups'][0]['multiplicity']==1
         and sp['energy_groups'][largest[0]]['multiplicity']==1)
    shifted=(sp['energy_groups'][largest[0]]['minimum_energy']-rc['energy_groups'][0]['minimum_energy']) if one else None
    screen=('UNKNOWN_CORRESPONDENCE' if shifted is None else
        'WITHIN_DECLARED_SPECTRAL_BUDGET_DIAGNOSTIC_ONLY' if abs(shifted)<=budget else
        'EXCEEDS_DECLARED_SPECTRAL_BUDGET_DIAGNOSTIC_ONLY')
    return dict(label=cell['label'],formula=cell['formula'],actual_delta_pf=cell['actual_delta_pf'],
        spectral_shift_diagnostics=deepcopy(diagnostic['spectral_shift_diagnostics']),
        overlap_selected_shift_ha=shifted,selection_scope='UNIQUE_MAX_TRACE_OBSERVATION_NOT_GROUND_OR_BRANCH_CERTIFICATION',
        largest_trace_overlap_group_indices=largest,largest_trace_overlap=max(overlaps),
        effective_minimum_group_state_weight=sp['minimum_energy_group_weight'],
        overlap_selected_group_state_weight=sp['energy_groups'][largest[0]]['weight'] if one else None,
        reference_minimum_group_state_weight=rc['energy_groups'][0]['weight'],
        local_spectral_budget_screen=screen,evolution_budget_ha=budget,
        branch_status=diagnostic['branch_status'],physical_alias_qualified=False,ground_identification_qualified=False,
        minimum_cut_clearance_rad=diagnostic['diagnostics']['selected_cut_clearance_rad'],
        diagnostic_origin=rc.get('eigensystem_origin','SAVED_EIGH_FROM_ORIGINAL_RUN'),
        scalar_constant_ha=diagnostic['operator'].get('scalar_constant_ha'),
        round_checks=records,numerical_gates_pass=True,matched_effective_group=None,
        rpe_bias_ha=None,sufficient_N=None,global_RMSE_bound=None,C=None,G=None)


def diagnose_cell(ctx,full_h,cell,*,reference,operator,tolerances,cache,policy,budget,save,branch=None):
    """Existing PF and E.evaluate plus the two planned power/measure checks."""
    import numpy as np
    from trottertracks.pf_power_stability import power_diagnostics
    delta=float(planner.exact(cell['actual_delta_pf']));first=cell['first_check']
    step=ctx.pf_step(first['physical_candidate'],cell['signed_weights'])
    unitary=np.exp(-1j*delta*ctx.constant)*step
    save('OPERATORS.npz',dict(PF_step_without_global_scalar=step,scalar_included_unitary=unitary))
    op=deepcopy(operator)
    op.update(identity=dict(step_cell=cell['label'],actual_delta_pf=cell['actual_delta_pf'],
        formula=cell['formula'],signed_weights=cell['signed_weights'],
        first_check=first['physical_candidate'],linked_rounds=cell['uses'],binding=operator['identity']['binding']),
        scalar_constant_ha=ctx.constant)
    diagnostic=E.evaluate(unitary,delta,state=ctx.state,operator=op,reference=reference,
        branch=deepcopy(branch) if branch is not None else dict(mode='PRINCIPAL',source=dict(commit=PREPARATION,path=PLAN_PATH)),
        tolerances=tolerances,reference_hamiltonian=full_h,reference_eigensystem=cache)
    save('EIGENSYSTEM.json',diagnostic)
    if diagnostic['status']!='EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE':
        raise ValueError('Effective spectrum/cache gate: '+diagnostic.get('reason',diagnostic['status']))
    energies=np.array(diagnostic['spectrum']['energies_ha']);weights=np.array(diagnostic['spectrum']['state_weights'])
    records=[];z=lambda x:dict(real=float(x.real),imag=float(x.imag))
    for row in (cell['first_check'],cell['last_check']):
        c=row['physical_candidate'];power,metrics=power_diagnostics(step,c['q'],ctx.state)
        propagated=power@ctx.state*np.exp(-1j*c['T']*ctx.constant)
        direct=complex(np.vdot(ctx.state,propagated))
        spectral=complex(np.sum(weights*np.exp(-1j*c['T']*energies)));gap=abs(direct-spectral)
        finite=all(math.isfinite(v) for v in (direct.real,direct.imag,spectral.real,spectral.imag,gap))
        passed=finite and metrics['gate_pass'] and gap<=policy['signal_difference_absolute']
        record=dict(m=row['m'],q_RPE=row['q_RPE'],canonical_candidate_id=row['canonical_candidate_id'],
            physical_candidate=deepcopy(c),direct_PF_mean=z(direct),full_effective_spectrum_mean=z(spectral),
            absolute_difference=gap,numerical=metrics,gate_pass=passed,
            scalar_once_at_total_T_in_direct_route=True,scalar_already_in_effective_spectrum=True,
            raw_equals_corrected_at_B1=True,B=1.,new_reference_actions=0,scientific_PASS_FAIL=None)
        save('round_'+str(row['m'])+'.json',record);records.append(record)
        if not passed:raise ValueError('Two-time power/signal numerical gate at m='+str(row['m']))
    return local_summary(diagnostic,cell,records,budget)


def execute(out,source):
    out=Path(out)
    if out.exists():raise ValueError('Fresh directory required; no retry/resume')
    out.mkdir(parents=True)
    def write(directory,name,value):
        directory.mkdir(parents=True,exist_ok=True)
        if name.endswith('.npz'):
            import numpy as np
            with (directory/name).open('xb') as f:np.savez(f,**value)
        else:
            with (directory/name).open('x') as f:f.write(json.dumps(E._finite_payload(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    start=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    result=dict(status='STARTED',source=source,preparation=PREPARATION,cell_results=[],errors=[],
        counts=dict(context_load_attempts=0,new_PF_steps=0,effective_evaluator_calls=0,direct_means=0,spectral_means=0),
        original_H_eigh=0,new_reference_actions=0,old_saved_signal_reanalysis=0,
        normalizer_N_MC_C_G=0,automatic_retry=False,automatic_grid_extension=False,
        rpe_bias_ha=None,sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True)
    current=None
    try:
        fixed={MODULE:source,'scripts/prepare_h4_full_pf_heff_scan.py':PREPARATION,
            'src/trottertracks/pf_effective_hamiltonian.py':PREPARATION,
            'scripts/run_h4_full_pf_heff.py':'3a26b9ba70f7c5022a7e07356fe1df8b620545e8',
            'src/trottertracks/pf_power_stability.py':'dd24dd329949c255d5e37b90ff8993fa8df7159b'}
        for path,commit in fixed.items():
            if (ROOT/path).read_bytes()!=previous.blob(commit,path):raise ValueError('Scientific SOURCE differs: '+path)
        plan,reference,operator,tolerances,policy=load_plan()
        operator['identity']=dict(binding=plan['shared']['binding'])
        budget=float(Fraction(plan['shared']['rmse_policy']['error_contract']['A_policy']['evolution_rmse']))
        anchor_ref=dict(commit=planner.ANCHOR,path=planner.ANCHOR_DIR+'EIGENSYSTEM.json')
        anchor=json.loads(previous.blob(anchor_ref['commit'],anchor_ref['path']))
        anchor_binding=anchor['operator']['identity']['step_key']['binding']
        if (not previous.P.same(anchor['reference'],reference)
                or anchor['operator']['target_hamiltonian']!=plan['shared']['binding']['hamiltonian']
                or any(anchor_binding.get(k)!=plan['shared']['binding'][k]
                    for k in ('hamiltonian','state','snapshot','rank','rank_policy'))):
            raise ValueError('Anchor/reference SOURCE binding differs')
        cache=dict(source=plan['shared']['cached_reference_eigensystem'],hamiltonian=reference['hamiltonian'],
            eigenvalues_ha=anchor['reference_comparison']['eigenvalues_ha'],
            eigenvectors=anchor['reference_comparison']['eigenvectors'])
        write(out,'SCOPE.json',dict(plan_source=dict(commit=PREPARATION,path=PLAN_PATH),scientific_sources=fixed,
            plan=plan,reference=reference,tolerances=tolerances,numerical_policy=policy,
            cached_eigensystem_source=cache['source'],evolution_budget_ha=budget))
        api=previous.numerical_api();result['counts']['context_load_attempts']=1
        saved,binding,metadata=api.load_shared_h4(ROOT,plan['shared']['binding']);ctx=api.Context(ROOT,saved,binding,metadata)
        if ctx.constant!=anchor['operator']['scalar_constant_ha']:raise ValueError('Original scalar differs')
        write(out,'INPUT_PROVENANCE.json',dict(binding=binding,inputs=metadata['source'],sector=[2,2],dimension=len(ctx.state),
            scalar_constant_ha=ctx.constant,original_H='saved_context.full'))
        for i,cell in enumerate(plan['step_cells']):
            current=cell['label']
            if cell['saved_source'] is not None:
                result['cell_results'].append(dict(label=cell['label'],status='SAVED_REFERENCE_ONLY_NO_REANALYSIS',source=cell['saved_source']))
                continue
            directory=out/('cell_'+str(i).zfill(2));print('START '+cell['label'],flush=True)
            write(directory,'CELL_SCOPE.json',cell)
            summary=diagnose_cell(ctx,saved.full,cell,reference=reference,operator=operator,tolerances=tolerances,
                cache=cache,policy=policy,budget=budget,save=lambda n,v:write(directory,n,v))
            summary.update(path=directory.name,status='LOCAL_DIAGNOSTIC_COMPLETE_NOT_RPE_QUALIFIED',source=source)
            write(directory,'result.json',summary);result['cell_results'].append(summary)
            print('DONE '+cell['label']+' '+summary['local_spectral_budget_screen'],flush=True)
        result['status']=COMPLETE
    except Exception as error:
        result.update(status='STOP_PARTIAL_NO_RETRY',failed_cell=current,
            errors=[dict(type=type(error).__name__,message=str(error))])
    result['counts'].update(new_PF_steps=len(list(out.glob('cell_*/OPERATORS.npz'))),
        effective_evaluator_calls=len(list(out.glob('cell_*/EIGENSYSTEM.json'))),
        direct_means=len(list(out.glob('cell_*/round_*.json'))),spectral_means=len(list(out.glob('cell_*/round_*.json'))))
    write(out,'result.json',result)
    import numpy,scipy
    write(out,'EXECUTION.json',dict(source=source,preparation=PREPARATION,started_UTC=start,
        finished_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick,
        python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,status=result['status'],
        counts=result['counts'],automatic_retry=False,mandatory_stop=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=execute(a.out,a.source)
    print(json.dumps(dict(status=r['status'],errors=r['errors'],counts=r['counts'])),flush=True)
    sys.exit(0 if r['status']==COMPLETE else 2)
