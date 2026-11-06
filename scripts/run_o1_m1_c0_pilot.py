"""Fixed C0 pilot. A separately approved scientific run; first failure stops.

Uses the saved rank10/original-state input and the fixed resource bridge. No
search, retry, unit-radius default, old prices or proxy. Import does no science.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
BASE='8a13539e51bb113e82cb86ddb608e2e6e27786ca'
INPUT_COMMIT='4591b8d75a2a6a55c74a51225f0c61648023c8ee'
INPUT_PATH='artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz'
INPUT_SHA='8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50'
DELTA=.05
R=2
K=2
Q=tuple(1<<m for m in range(7))
MODEL_BUDGET=.03
IMPLEMENTATION_BUDGET=.01


def modules():
    sys.path.insert(0,str(ROOT/'src'))
    if 'trotterlib' not in sys.modules:
        package=types.ModuleType('trotterlib')
        package.__path__=[str(ROOT/'src/trotterlib')]
        sys.modules['trotterlib']=package
    from trotterlib import o1_m1_matched_split as adapter
    from trotterlib import o1_m1_resource_bridge as bridge
    from trotterlib import o1_m1_circuit_signal_validation as signals
    from trotterlib import rpe_resource_accounting as accounting
    import run_o1_m1_h4_operator_correctness as reference
    return adapter,bridge,signals,accounting,reference


def dump(path,payload):
    # Own incremental result only, never a historical result path.
    import numpy as np
    def convert(value):
        if isinstance(value,np.generic): return value.item()
        raise TypeError(type(value).__name__)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False,default=convert)+'\n')
    temporary.replace(path)


def c0_margins(payload,record):
    """Exact SOURCE-fixed C0 engineering formulas; no observed-delta fit."""
    tol=payload['tolerance']
    eta=payload['conditions']['full_dimension']*max(tol['operator'],tol['basis'],tol['implementation'])
    radius=record['PF_signal']['radius']-eta
    if radius<=0 or 2*eta>=radius:
        raise ValueError('C0 numerical phase/radius allowance domain failed')
    return dict(radius_allowance=eta,complex_signal_allowance=2*eta,
                phase_allowance=math.asin(2*eta/radius))


def phase(first,second):
    import numpy as np
    return float(abs(np.angle(first/second))) if abs(first)*abs(second)>0 else None


def run(input_path,output_dir):
    import numpy as np
    from scipy.linalg import expm
    adapter,bridge,signals,accounting,reference=modules()
    if output_dir.exists(): raise FileExistsError('Fresh C0 output directory required')
    output_dir.mkdir(parents=True)
    files=('scripts/run_o1_m1_c0_pilot.py','src/trotterlib/o1_m1_matched_split.py',
        'src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/o1_m1_circuit_signal_validation.py',
        'scripts/run_o1_m1_h4_operator_correctness.py','src/trottertracks/c1_h4_strong_baseline.py')
    for path in files[1:]:
        fixed=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        if fixed!=(ROOT/path).read_bytes(): raise ValueError('Fixed SOURCE changed: '+path)
    provenance=dict(base=BASE,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        input=dict(commit=INPUT_COMMIT,path=INPUT_PATH,sha256=INPUT_SHA,local_path=str(input_path)),
        host=platform.node(),utc=datetime.now(timezone.utc).isoformat(),
        runtime={p:importlib.metadata.version(p) for p in ('numpy','scipy','openfermion','qiskit','qiskit-nature')})
    if provenance['runtime']['qiskit']!='1.3.0': raise ValueError('C0 Qiskit 1.3.0 required')
    dump(output_dir/'provenance.json',provenance)
    if hashlib.sha256(input_path.read_bytes()).hexdigest()!=INPUT_SHA: raise ValueError('Wrong saved input')
    result=dict(status='RUNNING',configuration=dict(delta=DELTA,r=R,K=K,q_values=Q,
        epsilon_E=.1,alpha_total=.05,beta_RPE=.30,phase_budgets=dict(PF=.06,RTE=.03,stat=.15,model=.03,implementation=.01,reserve=.02),
        alpha_axis=.05/14,energy_interval=(-10.,0.),state_field='state',state_preparation_included=False,
        identity_policy='faithful_identity_in_tail',baseline_substeps=4,
        baseline_basis_policy='exact_spin_lift_spatial_eigh_v1',relative_fusion=True,
        compiler=dict(version='1.3.0',basis=('rz','sx','x','cx'),optimization=1,seed=7,backend=None),
        classical_MC_samples=32,classical_MC_seed='2026110600+100*arm_index+m',price_relative_SE_limit=.1),
        provenance=provenance,signal_rounds=[],baseline_rounds=[],priced_rounds=[],
        total_cost=None,total_cost_comparison='UNASSESSED',price_confidence_interval=None)
    def stop(reason):
        result['status']='C0_NOT_QUALIFIED'; result['stop_reason']=reason
        result['mandatory_stop']=True
        dump(output_dir/'result.json',result)
        print('MANDATORY STOP: '+reason,flush=True)
        return result
    inputs=adapter.load_h4_a_snapshot(input_path)
    a=reference.annihilators(inputs.num_qubits)
    original=(inputs.nuclear_repulsion*np.eye(1<<inputs.num_qubits)
        +reference.one_body(inputs.spin_one,a).toarray()+reference.quartic(inputs.spin_two,a).toarray())
    norm=float(np.vdot(inputs.state,inputs.state).real)
    psi=inputs.state/math.sqrt(norm)
    E=complex(np.vdot(psi,original@psi))
    result['original_state']=dict(squared_norm=norm,rayleigh_energy=E.real,energy_imaginary=E.imag,
        eigen_residual_l2=float(np.linalg.norm(original@psi-E*psi)),
        state_sha256=hashlib.sha256(inputs.state.tobytes()).hexdigest(),assumption='exact ground state of original H; no solve/replacement')
    if not -10.<=E.real<=0.: return stop('original-state energy outside fixed C0 interval')
    split=adapter.prepare_matched_a(inputs)
    preparations={name:bridge.prepare_matched_resources(split,dictionary=name) for name in ('P','DF')}
    laws={name:bridge.configure_law(p,delta_time=DELTA,r=R,K=K) for name,p in preparations.items()}
    allocation=accounting.RPEErrorAllocation(.06,.03,.15,.05/14,.05/14)
    round_bounds={}; raw_payloads={}
    for m,q in enumerate(Q):
        print(f'signal round m={m} q={q}',flush=True)
        payload=signals.validate_matched_circuit_signals(split,reference,step_times=(DELTA,),r=R,K_values=(K,),q=q)
        dump(output_dir/f'signal_m{m}.json',payload)
        if payload['status']!='PASS':
            result['failed_checks']=[c for c in payload['checks'] if not c['passed']]
            return stop(f'm={m}: existing operator/controlled-signal validation failed')
        record=payload['records'][0]
        try: margins=c0_margins(payload,record)
        except ValueError as exc: return stop(f'm={m}: {exc}')
        spec=accounting.RPERoundSpecification(m,DELTA)
        row=dict(round_index=m,q=q,numerical_allowances=margins,arms={})
        for name,p in preparations.items():
            config,_=laws[name]
            try: bounds=bridge.observed_signal_bounds(p,config,spec,record,**margins)
            except ValueError as exc: return stop(f'm={m} {name}: {exc}')
            if bounds['model_phase_bound']>MODEL_BUDGET or bounds['implementation_phase_bound']>IMPLEMENTATION_BUDGET:
                row['arms'][name]=dict(bounds=bounds,account=None)
                result['signal_rounds'].append(row)
                return stop(f'm={m} {name}: fixed C0 model/implementation budget failed')
            account=bridge.account_radius_round(context_hash=p.context_hash,arm_hash=p.preparation_hash,
                specification=spec,allocation=allocation,beta_rpe=.30,cost_metric='cx_count',config=config,**bounds)
            row['arms'][name]=account
            # Preserve every observed result before first scientific failure.
            if not account['feasible']:
                result['signal_rounds'].append(row)
                return stop(f'm={m} {name}: fixed C0 phase/radius condition failed')
            round_bounds[m,name]=bounds
        raw_payloads[m]=payload
        result['signal_rounds'].append(row)
        dump(output_dir/'result.json',result)
    # No circuit-price work starts until all PR signal conditions have passed.
    from qiskit.quantum_info import Statevector,Operator
    from trottertracks import c1_h4_strong_baseline as strong
    generators=[reference.one_body(inputs.df_one_body,a).toarray()]
    generators += [(w*(reference.one_body(g,a)@reference.one_body(g,a))).toarray()
                   for w,g in zip(inputs.df_lambdas,inputs.df_g_matrices,strict=True)]
    schedule=strong.native.ordered_schedule(len(generators),Fraction(DELTA)/4)
    half_evolutions={(index,t):expm(-1j*float(t)*generators[index]) for index,t in schedule}
    substep=np.eye(len(psi),dtype=complex)
    for index,t in schedule: substep=half_evolutions[index,t]@substep
    baseline_step=np.exp(-1j*DELTA*inputs.nuclear_repulsion)*np.linalg.matrix_power(substep,4)
    baseline_circuits={};baseline_bounds={}
    for m,q in enumerate(Q):
        print(f'strong baseline round m={m} q={q}',flush=True)
        circuit=bridge.build_strong_df_baseline(split,delta_time=DELTA,q=q,substeps=4,basis_policy=strong.SPIN)
        expected=np.linalg.matrix_power(baseline_step,q)@psi
        if m==0:
            expected_operator=signals.controlled_reference(baseline_step)
            error=float(np.max(abs(Operator(circuit).data-expected_operator)))
            result['baseline_operator_q1_max_entry']=error
            if error>raw_payloads[m]['tolerance']['implementation']:
                return stop('m=0 baseline: full controlled operator/relative phase failed')
        superposition=np.r_[psi,psi]/math.sqrt(2)
        actual=Statevector(superposition).evolve(circuit).data
        target=np.r_[psi,expected]/math.sqrt(2)
        implementation_error=float(np.linalg.norm(actual-target))
        z=complex(np.vdot(psi,expected))
        raw=raw_payloads[m]['records'][0]
        zdf=complex(raw['DF_exact_signal']['real'],raw['DF_exact_signal']['imag'])
        margins=c0_margins(raw_payloads[m],raw)
        eta=margins['radius_allowance']
        rho=abs(z)-eta
        if rho<=0 or implementation_error>eta: return stop(f'm={m} baseline: controlled state-action/phase/radius failed')
        beta=math.asin(2*eta/rho) if 2*eta<rho else None
        if beta is None: return stop(f'm={m} baseline: numerical phase domain failed')
        bounds=dict(reference_radius_lower_bound=rho,corrected_signal_error_bound=0.,
            pf_phase_bound=phase(z,zdf)+beta,
            model_phase_bound=raw['original_vs_DF_exact_signal_difference']['wrapped_phase_absolute']+beta,
            implementation_phase_bound=math.asin(implementation_error/rho)+beta,
            radius_provenance='same original state / independent all-factor S2, declared C0 allowance')
        account=bridge.account_radius_round(context_hash=preparations['DF'].context_hash,
            arm_hash='strong_DF_PF_4_substeps_spin_fused',specification=accounting.RPERoundSpecification(m,DELTA),
            allocation=allocation,beta_rpe=.30,cost_metric='cx_count',config=None,**bounds)
        result['baseline_rounds'].append(dict(account=account,controlled_state_action_error_l2=implementation_error))
        if (not account['feasible'] or bounds['model_phase_bound']>MODEL_BUDGET
                or bounds['implementation_phase_bound']>IMPLEMENTATION_BUDGET):
            return stop(f'm={m} baseline: fixed C0 phase/radius condition failed')
        baseline_bounds[m]=bounds;baseline_circuits[m]=circuit
        dump(output_dir/'result.json',result)
    from trotterlib.rte import CompilerSettings
    provider=bridge.MatchedNoPrepCostProvider(CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0'))
    complete={name:[] for name in ('P','DF','strong_DF_PF')}
    for m,q in enumerate(Q):
        for name,arm_index in (('P',0),('DF',1),('strong_DF_PF',None)):
            print(f'price round m={m} q={q} arm={name}',flush=True)
            if arm_index is None:
                p=preparations['DF']; config=None; bounds=baseline_bounds[m]
                arm='strong_DF_PF_4_substeps_spin_fused'
                cost=provider.deterministic(baseline_circuits[m],context_hash=p.context_hash,
                    arm_hash=arm,q=q,delta_time=DELTA)
            else:
                p=preparations[name];config,distribution=laws[name];bounds=round_bounds[m,name];arm=p.preparation_hash
                cost=provider.monte_carlo(p,config,distribution,q=q,sample_count=32,seed=2026110600+100*arm_index+m)
            account=bridge.account_radius_round(context_hash=p.context_hash,arm_hash=arm,
                specification=accounting.RPERoundSpecification(m,DELTA),allocation=allocation,beta_rpe=.30,
                cost_metric='cx_count',config=config,cost=cost,**bounds)
            row=dict(arm=name,round_index=m,cost=asdict(cost),account=account)
            if arm_index is not None:
                rse={axis:getattr(cost,axis+'_standard_error').cx_count/getattr(cost,axis+'_expected_cost').cx_count
                     for axis in ('cosine','sine')}
                row['CX_relative_sample_SE']=rse
                if max(rse.values())>.1:
                    result['priced_rounds'].append(row)
                    return stop(f'm={m} {name}: price relative sample SE > fixed 10%')
            result['priced_rounds'].append(row);complete[name].append(account)
            dump(output_dir/'result.json',result)
    result['total_cost']={name:bridge.aggregate_registered_rounds(rows,required_max_round=6,total_alpha_budget=.05)
                          for name,rows in complete.items()}
    result['status']='C0_CONDITIONAL_PILOT_COMPLETE'
    result['total_cost_comparison']='conditional mean estimates only; no price CI or definitive advantage'
    result['mandatory_stop']=True
    dump(output_dir/'result.json',result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.input.resolve(),args.output_dir.resolve())
    return 0 if result['status']=='C0_CONDITIONAL_PILOT_COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())
