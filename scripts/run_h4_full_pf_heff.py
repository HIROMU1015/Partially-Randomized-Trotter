"""Fixed H4 Full PF effective-Hamiltonian pilot; no retries, N, MC or search."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_effective_hamiltonian as E
from trottertracks import pf_pr_rpe_pipeline as P

PREPARATION='9af1ac8ed8adcaef059f439ce690b22b7a320de0'
PLAN_PATH='docs/research/h4_full_pf_heff_preparation_20261113/PLAN.json'
C_SOURCE='86f37d25ffd5eee3daf7f25ce001b090b1f2cf33'
MODULE='scripts/run_h4_full_pf_heff.py'
COMPLETE='TWO_TIME_HEFF_DIAGNOSTIC_COMPLETE_NOT_RPE_QUALIFIED'


def blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def metadata():
    """Pinned JSON/identity only; no molecular arrays or numerical leaves."""
    plan=json.loads(blob(PREPARATION,PLAN_PATH));desc=plan['sources']['canonical_configuration']
    manifest=json.loads(blob(desc['commit'],desc['path']));candidate=manifest['candidates'][0]
    config=candidate['configuration'];s=config['spec']
    if (config['id']!=plan['canonical_configuration_id'] or candidate['label']!='FULL_PF_S2_D01'
            or (s['method'],s['formula'],s['L_D'],s['M'])!=('PF','2nd',10,12)
            or s['delta_pf']!=plan['step_key']['actual_delta'] or plan['proposed_signal_rounds']!=[0,12]):
        raise ValueError('Fixed Full PF configuration differs')
    exact=lambda v:Fraction(v['numerator'],v['denominator'])
    if exact(s['delta_pf'])!=Fraction(.1) or s['h']!=s['delta_pf']:
        raise ValueError('Fixed exact step/h differs')
    for k,v in plan['binding'].items():
        if not P.same(s['binding'].get(k),v):raise ValueError('Saved binding differs: '+k)
    for r,expected,physical in zip(config['rounds'],plan['canonical_rounds'],candidate['rounds']):
        if (r['m']!=expected['m'] or r['candidate']['id']!=expected['candidate_id']
                or not P.same(physical['physical_candidate'],expected['physical_candidate'])
                or r['T']!=expected['T'] or r['q_PF']!=expected['q_PF']
                or r['q_RPE']!=expected['q_RPE'] or r['q_PF']!=1<<r['m']
                or r['actual_delta_pf']!=s['delta_pf'] or r['structure']['signed_weights']!=[1.]
                or exact(r['T'])!=r['q_PF']*exact(s['delta_pf'])):
            raise ValueError('Canonical round/physical recipe differs')
    if len(config['rounds'])!=13:raise ValueError('Complete fixed metadata required')
    policy=plan['numerical_policy']
    tolerances=dict(unitarity=policy['unitarity_fro'],reconstruction=policy['reconstruction_fro'],
        generator_hermiticity_ha=policy['generator_hermiticity_and_reference_eigen_residual_ha'],
        energy_cluster_ha=policy['energy_cluster_ha'],phase_guard_rad=policy['phase_guard_rad'],
        source=dict(commit=PREPARATION,path=PLAN_PATH,selector=['numerical_policy']))
    target=s['binding']['hamiltonian']
    reference=dict(energy=manifest['shared']['target']['reference_solver_energy_ha'],
        definition=plan['reference']['declared_energy_definition'],hamiltonian=target,
        source=deepcopy(plan['reference']['declared_energy_source']),ground_state_certified=False,
        carrier=dict(commit=desc['commit'],path=desc['path'],selector=['shared','target','reference_solver_energy_ha']))
    operator=dict(kind='FULL_PF',identity=dict(configuration_id=config['id'],step_key=plan['step_key']),
        source=deepcopy(plan['step_key']['source']),target_hamiltonian=target,retained_hamiltonian=target,
        scalar_policy='INCLUDED_ONCE_AT_STEP_TIME')
    return plan,config,reference,operator,tolerances


def numerical_api():
    """Reuse C's frozen loader/Context in an isolated namespace, not C's checkout."""
    import ast
    from trottertracks import h4_pr_m1_rmse as existing
    name='trottertracks._h4_heff_fixed_C86'
    package=types.ModuleType(name);package.__path__=[];sys.modules[name]=package
    sys.modules[name+'.pf_pr_rpe_pipeline']=P
    helper=types.ModuleType(name+'.r_price_phase1_science')
    helper.__dict__.update(ast=ast,m=existing)
    existing.function(ROOT,C_SOURCE,'src/trottertracks/r_price_phase1_science.py','functions',helper.__dict__)
    sys.modules[helper.__name__]=helper
    for short in ('baseline_signal_pipeline','baseline_signal_numerics'):
        path='src/trottertracks/'+short+'.py'
        mod=types.ModuleType(name+'.'+short);mod.__package__=name;mod.__file__=str(ROOT/path)
        sys.modules[mod.__name__]=mod
        exec(compile(blob(C_SOURCE,path),C_SOURCE+':'+path,'exec'),mod.__dict__)
        setattr(package,short,mod)
    return package.baseline_signal_numerics


def diagnose(ctx,full_h,rows,reference,operator,tolerances,plan,save):
    """One supplied step/eigensystem and the two planned powers; synthetic-testable."""
    import numpy as np
    from trottertracks.pf_power_stability import power_diagnostics
    delta=float(Fraction(plan['step_key']['actual_delta']['numerator'],plan['step_key']['actual_delta']['denominator']))
    step=ctx.pf_step(rows[0]['physical_candidate'],[1.])
    unitary=np.exp(-1j*delta*ctx.constant)*step
    save('OPERATORS.npz',dict(PF_step_without_global_scalar=step,scalar_included_unitary=unitary))
    op=deepcopy(operator);op['scalar_constant_ha']=ctx.constant
    branch=dict(mode='PRINCIPAL',source=dict(commit=PREPARATION,path=PLAN_PATH,selector=['branch']))
    diagnostic=E.evaluate(unitary,delta,state=ctx.state,operator=op,reference=reference,
        branch=branch,tolerances=tolerances,reference_hamiltonian=full_h)
    save('EIGENSYSTEM.json',diagnostic)
    if diagnostic['status']!='EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE':
        raise ValueError('H_eff numerical/branch gate: '+diagnostic.get('reason',diagnostic['status']))
    spectrum=diagnostic['spectrum'];energies=np.array(spectrum['energies_ha']);weights=np.array(spectrum['state_weights'])
    records=[]
    for row in rows:
        c=row['physical_candidate'];powered,metrics=power_diagnostics(step,c['q'],ctx.state)
        propagated=powered@ctx.state*np.exp(-1j*c['T']*ctx.constant)
        direct=complex(np.vdot(ctx.state,propagated))
        reconstructed=complex(np.sum(weights*np.exp(-1j*c['T']*energies)))
        gap=abs(direct-reconstructed)
        z=lambda v:dict(real=float(v.real),imag=float(v.imag))
        finite=all(math.isfinite(x) for x in (direct.real,direct.imag,reconstructed.real,reconstructed.imag,gap))
        gate=finite and metrics['gate_pass'] and gap<=plan['numerical_policy']['signal_difference_absolute']
        record=dict(m=row['m'],canonical_candidate_id=row['candidate_id'],physical_candidate=deepcopy(c),
            q_RPE=row['q_RPE'],direct_PF_mean=z(direct),full_effective_spectrum_mean=z(reconstructed),
            absolute_difference=gap,numerical=metrics,gate_pass=gate,
            scalar_once_at_total_T_in_direct_route=True,scalar_already_in_effective_spectrum=True,
            raw_equals_corrected_at_B1=True,B=1.,new_reference_actions=0,scientific_PASS_FAIL=None)
        save('round_'+str(row['m'])+'.json',record);records.append(record)
        if not gate:raise ValueError('Two-time signal/power gate at m='+str(row['m']))
    return diagnostic,records


def execute(out,source):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required; no retry/resume')
    out.mkdir(parents=True)
    def save(name,value):
        if name.endswith('.npz'):
            import numpy as np
            with (out/name).open('xb') as f:np.savez(f,**value)
        else:
            value=E._finite_payload(value)
            with (out/name).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    result=dict(status='STARTED',caller_source=source,preparation_source=PREPARATION,C_source=C_SOURCE,
        counts=dict(context_load_attempts=0,PF_steps=0,effective_evaluator_calls=0,direct_means=0,spectral_means=0),
        new_reference_actions=0,saved_signal_reanalysis=0,new_normalizers=0,new_N_MC_C_G=0,
        rpe_bias_ha=None,sufficient_N=None,global_RMSE_bound=None,G=None,
        errors=[],automatic_retry=False,mandatory_stop=True)
    try:
        for path,commit in ((MODULE,source),('src/trottertracks/pf_effective_hamiltonian.py',PREPARATION),
                ('src/trottertracks/pf_power_stability.py','dd24dd329949c255d5e37b90ff8993fa8df7159b')):
            if (ROOT/path).read_bytes()!=blob(commit,path):raise ValueError('Scientific SOURCE differs: '+path)
        plan,config,reference,operator,tolerances=metadata()
        save('SCOPE.json',dict(plan=plan,caller_source=source,canonical_configuration_source=plan['sources']['canonical_configuration'],
            canonical_configuration_id=config['id'],reference=reference,operator=operator,tolerances=tolerances))
        cpath='src/trottertracks/baseline_signal_numerics.py'
        if blob(C_SOURCE,cpath)!=blob(plan['sources']['numerics']['commit'],cpath):
            raise ValueError('C numerical SOURCE differs between plan aliases')
        api=numerical_api();result['counts']['context_load_attempts']=1
        saved,binding,context_metadata=api.load_shared_h4(ROOT,config['spec']['binding'])
        ctx=api.Context(ROOT,saved,binding,context_metadata)
        save('INPUT_PROVENANCE.json',dict(binding=binding,inputs=context_metadata['source'],sector=[2,2],dimension=len(ctx.state),
            original_H='SAVED_CONTEXT.full = sum(original matrices) + original constant I',scalar_constant_ha=ctx.constant))
        rows=[plan['canonical_rounds'][m] for m in plan['proposed_signal_rounds']]
        diagnostic,records=diagnose(ctx,saved.full,rows,reference,operator,tolerances,plan,save)
        rc=diagnostic['reference_comparison'];spec=diagnostic['spectrum']
        overlap=rc['reference_lowest_group_to_effective_groups_trace_overlap']
        largest=[i for i,x in enumerate(overlap) if x==max(overlap)]
        result.update(status=COMPLETE,rounds=records,spectral_shift_diagnostics=diagnostic['spectral_shift_diagnostics'],
            correspondence_observation=dict(reference_lowest_group_state_weight=rc['energy_groups'][0]['weight'],
                effective_lowest_group_state_weight=spec['minimum_energy_group_weight'],
                largest_trace_overlap_group_indices=largest,largest_trace_overlap=max(overlap),
                matched_effective_group=None,ground_identification_qualified=False),
            branch_status=diagnostic['branch_status'],physical_alias_qualified=False,
            eigensystem_path='EIGENSYSTEM.json',configuration_id=config['id'])
    except Exception as error:
        result.update(status='STOP_PARTIAL_NO_RETRY',errors=[dict(type=type(error).__name__,message=str(error))])
        # Completed records persist even if the next gate failed.
        result['round_files_saved']=[p.name for p in sorted(out.glob('round_*.json'))]
    result['counts'].update(PF_steps=int((out/'OPERATORS.npz').exists()),
        effective_evaluator_calls=int((out/'EIGENSYSTEM.json').exists()),
        direct_means=len(list(out.glob('round_*.json'))),spectral_means=len(list(out.glob('round_*.json'))))
    save('result.json',result)
    import numpy,scipy
    save('EXECUTION.json',dict(source=source,preparation=PREPARATION,started_UTC=started,
        finished_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick,
        python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,status=result['status'],
        automatic_retry=False,mandatory_stop=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();value=execute(a.out,a.source)
    print(json.dumps(dict(status=value['status'],errors=value['errors'],counts=value['counts'])),flush=True)
    sys.exit(0 if value['status']==COMPLETE else 2)
