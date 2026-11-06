"""Fixed saved B2 P1 bounds -> existing conditional Hoeffding shot accounting.

No input arrays, simulation, decoder, sampling, circuits or prices are executed.
Shot counts and all their sums remain arbitrary-precision Python integers.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

from . import b2_p1_branch_validation as saved
from .b2_rpe_connection import B2SignalRecord, SignalValue
from .o1_m1_resource_bridge import account_radius_round
from .pauli_signal_adapter import digest
from .rpe_resource_accounting import RPEErrorAllocation, RPERoundSpecification
from .rte import RTEConfig, finite_rte_distribution

ROOT=Path(__file__).resolve().parents[2]
BRANCH_COMMIT='d424bb8bec1aaf1224a33513a06e64801749a69f'
BRANCH_PATH='artifacts/track_a_b2_p1_branch/2026-11-07/run01/result.json'
ALLOCATION=RPEErrorAllocation(.06,.04,.24,.05/36,.05/36)
CAPS=dict(model=.02,PF=.06,RTE=.04,statistical=.24,implementation=.02,reserve=.02)
ALPHA_EXACT=Fraction(1,720)


def saved_config(new,end):
    """Canonical scalar law from saved metadata; no Hamiltonian preparation."""
    law=end['normalization_law'];other=new['normalization_law']
    for name in ('lambda_R','tau','b_event','tail_identity'):
        saved.require(other[name]==law[name],'SAVED_LAW_'+name)
    saved.require(other['prior_executable_tail_reference']==law['tail_hash'],'SAVED_TAIL_REFERENCE')
    finite=finite_rte_distribution(law['tau'],2)
    config=RTEConfig(tail_id='B2_P1_saved_tail_shot_accounting_only',tail_hash=law['tail_hash'],
        lambda_r=law['lambda_R'],evolution_time=saved.DELTA,rte_steps=4,
        step_time=saved.DELTA/4,dimensionless_step_time=law['tau'],
        taylor_distribution='rte_even_taylor_paired',finite_taylor_order=2,
        truncation_tolerance=math.nextafter(finite.step_truncation_residual_bound,math.inf),
        distribution_normalization=law['b_event'],truncation_residual_bound=finite.step_truncation_residual_bound,seed=0)
    saved.require(config.distribution_normalization==finite.exact_finite_distribution,'CANONICAL_SAVED_LAW')
    return config


def checked_bounds(row,identity,config):
    """Recheck saved B2 bounds from saved five scalars, never resimulate them."""
    binding='saved_scalar_bounds:'+str(row['m'])
    round_=SimpleNamespace(context=SimpleNamespace(**identity),config=config,q=row['Q'],binding=binding)
    fields=[SignalValue(saved.complex_signal(row['signals'][name]),identity['original_state'],
        identity['original_hamiltonian'] if name=='phys' else identity['df_hamiltonian'])
        for name in ('phys','DF','PF','finite','raw')]
    record=B2SignalRecord(binding,*fields,row['normalization'],row['implementation_phase_observed'])
    bounds=record.bounds(round_,radius_allowance=saved.U,complex_allowance=saved.U,
        phase_allowance=saved.U,normalization_tolerance=saved.U)
    saved.require(bounds==row['qualification']['bounds'],'SAVED_B2_BOUNDS_MISMATCH')
    saved.require(bounds['model_phase_bound']<=CAPS['model'] and
                  bounds['implementation_phase_bound']<=CAPS['implementation'],'P1_MODEL_IMPLEMENTATION_CAP')
    saved.require(ALLOCATION.beta_total+bounds['model_phase_bound']+bounds['implementation_phase_bound']+
                  CAPS['reserve']<=saved.BETA,'P1_TOTAL_WITH_RESERVE')
    return bounds


def account_saved_round(row,bounds,config,*,context_hash,arm_hash):
    """Actual existing shot function call. cost=None prevents price arithmetic."""
    fields=('reference_radius_lower_bound','corrected_signal_error_bound','pf_phase_bound',
            'model_phase_bound','implementation_phase_bound','radius_provenance')
    account=account_radius_round(context_hash=context_hash,arm_hash=arm_hash,
        specification=RPERoundSpecification(row['m'],saved.DELTA),allocation=ALLOCATION,
        beta_rpe=saved.BETA,config=config,cost=None,cost_metric='rz_count',backend_hash='no_backend',
        **{k:bounds[k] for k in fields})
    saved.require(account['feasible'],'ROUND_SHOT_INELIGIBLE:'+','.join(account['reasons']))
    q=row['qualification']
    saved.require(account['rho_observed_lower_bound']==q['rho_observed_lower_bound'] and
        account['attenuation']==row['attenuation'] and
        account['finite_rte_phase_bound']==q['finite_rte_phase_bound'],'SAVED_RADIUS_PHASE_MISMATCH')
    nx,ny=account['shots']['cosine'],account['shots']['sine']
    saved.require(type(nx) is int and type(ny) is int and nx>0 and ny>0,'POSITIVE_INTEGER_SHOTS')
    return dict(m=row['m'],Q=row['Q'],T_Ha_inverse=row['T_Ha_inverse'],
        PF_radius_lower=bounds['reference_radius_lower_bound'],
        corrected_error_disk=bounds['corrected_signal_error_bound'],
        normalization=row['normalization'],attenuation=row['attenuation'],
        raw_radius_lower=account['rho_observed_lower_bound'],
        N_X=nx,N_Y=ny,N_round=nx+ny,beta_stat=ALLOCATION.beta_stat_budget,
        alpha_X=ALLOCATION.alpha_cosine,alpha_Y=ALLOCATION.alpha_sine,
        alpha_each_exact='1/720',phase_bounds=bounds,
        finite_rte_phase_bound=account['finite_rte_phase_bound'],
        P1_total_with_reserve=q['phase_total_with_reserved_statistical'],
        context_hash=context_hash,arm_hash=arm_hash,status='CONDITIONAL_SUFFICIENT_SHOTS',
        guarantee_status=account['guarantee_status'],price=None,total_circuit_cost=None,
        account_API_cost_metric_identifier='rz_count; metadata only, no price evaluation')


def aggregate_shots(rows):
    """Integer-only aggregate; reject missing/duplicate/inconsistent rounds."""
    saved.require([r['m'] for r in rows]==list(range(18)),'EXACTLY_ONCE_ALL18_IN_ORDER')
    for row in rows:
        saved.require(row['status']=='CONDITIONAL_SUFFICIENT_SHOTS','INELIGIBLE_ROUND')
        saved.require(all(type(row[k]) is int and row[k]>0 for k in ('N_X','N_Y','N_round')),
                      'INTEGER_SHOT_COUNTS')
        saved.require(row['N_X']+row['N_Y']==row['N_round'],'ROUND_SUM_MISMATCH')
        saved.require(row['alpha_X']==row['alpha_Y']==float(ALPHA_EXACT),'FIXED_AXIS_ALPHA')
    nx=sum(r['N_X'] for r in rows);ny=sum(r['N_Y'] for r in rows)
    total=sum(r['N_round'] for r in rows)
    saved.require(total==nx+ny,'AXIS_ROUND_TOTAL_MISMATCH')
    # Independent exact integer accumulation in reverse order, via Decimal text.
    with localcontext() as ctx:
        ctx.prec=max(len(str(r[k])) for r in rows for k in ('N_X','N_Y'))+len(str(2*len(rows)))+1
        decimal_total=sum((Decimal(str(r['N_X']))+Decimal(str(r['N_Y'])) for r in reversed(rows)),Decimal(0))
    saved.require(decimal_total==Decimal(total),'INDEPENDENT_DECIMAL_SUM_MISMATCH')
    exact_alpha=2*len(rows)*ALPHA_EXACT
    saved.require(exact_alpha<=Fraction(1,20),'TOTAL_FAILURE_BUDGET')
    return dict(N_X_total=nx,N_Y_total=ny,N_all_rounds=total,round_count=18,
        round_sum_equals_axis_sum=True,independent_decimal_total=str(decimal_total),
        exact_failure_budget_sum=str(exact_alpha),failure_budget_target_exact='1/20',
        binary64_axis_alpha=ALLOCATION.alpha_cosine,
        binary64_fsum_of_axis_budgets=math.fsum(r['alpha_X']+r['alpha_Y'] for r in rows),
        integer_counts_preserved=True,price=None,total_circuit_cost=None,
        status='ALL18_CONDITIONAL_SUFFICIENT_SHOTS',guarantee_status='not_certified')


def verify_serialized_totals(output,expected):
    """Independent readback of saved CSV/JSON counts, without a new shot law."""
    stored=json.loads((output/'result.json').read_text())
    with (output/'rounds.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    saved.require([int(r['m']) for r in rows]==list(range(18)),'SERIALIZED_ROUND_INVENTORY')
    with localcontext() as ctx:
        ctx.prec=max(len(r[k]) for r in rows for k in ('N_X','N_Y'))+len(str(2*len(rows)))+1
        x=sum((Decimal(r['N_X']) for r in rows),Decimal(0))
        y=sum((Decimal(r['N_Y']) for r in reversed(rows)),Decimal(0))
        combined=x+y
    saved.require(x==Decimal(expected['N_X_total']) and y==Decimal(expected['N_Y_total']) and
                  combined==Decimal(expected['N_all_rounds']),'SERIALIZED_SHOT_TOTAL')
    saved.require(stored['totals']==expected,'SERIALIZED_JSON_TOTAL')
    for raw,row in zip(rows,stored['rounds'],strict=True):
        saved.require(all(int(raw[k])==row[k] for k in ('m','N_X','N_Y','N_round')),
                      'CSV_JSON_INTEGER_MISMATCH')
    return dict(status='PASS',CSV_X_sum=str(x),CSV_Y_sum=str(y),CSV_all_sum=str(combined),
                JSON_CSV_integer_match=True)


def run(output):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=saved.git('rev-parse','HEAD').decode().strip()
    paths=['src/trotterlib/b2_p1_shot_accounting.py','scripts/run_b2_p1_shot_accounting.py',
        'src/trotterlib/b2_p1_branch_validation.py','src/trotterlib/b2_rpe_connection.py',
        'src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/finite_rte_signal_validation.py',
        'src/trotterlib/rpe_resource_accounting.py','src/trotterlib/rte.py']
    for path in paths:
        saved.require((ROOT/path).read_bytes()==saved.git('show',source+':'+path),'COMMITTED_SOURCE_REQUIRED')
    result=dict(schema_version='b2_p1_all18_shot_accounting_v1',execution_SOURCE=source,
        SOURCE_SHA256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        started_UTC=datetime.now(timezone.utc).isoformat(),rounds=[],mandatory_stop=True,
        shot_method='finite_rte_signal_validation._hoeffding_shots; unchanged int(math.ceil(...))',
        assumptions=['Saved original-H-state signals/bounds satisfy fixed P1 engineering allowances',
            'Fresh independent canonical RTE trajectory for each Hadamard shot',
            'Independent bounded outcomes in {-1,+1} for each axis',
            'Existing real-radius coordinate-to-phase sufficient condition; beta_stat=.24'],
        not_certified=['Real-device 95% success','Rigorous numerical/implementation enclosure',
            'All-support/long-circuit correctness','Strong-PF cost superiority'],
        guarantee_status='not_certified',forbidden_work=dict(private_input_reads=0,SCF_DF_state_solve=0,
        signal_simulation=0,branch_reanalysis=0,sampling=0,circuit_build_compile=0,price_MC=0,
        price_metrics=0,total_circuit_cost=0,baseline=0,optimization=0,scientific_review=0))
    error_to_raise=None
    try:
        combined,cm=saved.fixed_json(saved.ALL_COMMIT,saved.ALL_PATH)
        new,nm=saved.fixed_json(saved.ALL_COMMIT,saved.NEW_PATH)
        end,em=saved.fixed_json(saved.END_COMMIT,saved.END_PATH)
        branch,bm=saved.fixed_json(BRANCH_COMMIT,BRANCH_PATH)
        selected,identity,conditions=saved.validate_saved_bundle(combined,new,end)
        saved.require(branch['status']=='P1_CONDITIONAL_BRANCH_ENERGY_PASS' and
            branch['input_validation']=='PASS' and branch['scientific_context']==identity and
            branch['fixed_conditions']==conditions,'SAVED_BRANCH_P1_CONTEXT')
        saved.require(branch['saved_inputs']==[cm,nm,em],'BRANCH_SIGNAL_INPUT_HASHES')
        saved.require([r['m'] for r in branch['reconstruction']['rounds']]==list(range(18)) and
            branch['reconstruction']['unique_saved_mean_branches'] and
            branch['correctness']['final_within_energy_target'],'SAVED_BRANCH_ENERGY_STATUS')
        for path,sha in branch['SOURCE_SHA256'].items():
            saved.require(hashlib.sha256(saved.git('show',branch['execution_SOURCE']+':'+path)).hexdigest()==sha,
                          'BRANCH_EXECUTION_SOURCE_HASH')
        result.update(saved_inputs=[cm,nm,em,bm],scientific_context=identity,fixed_conditions=conditions,
            input_status='PASS',saved_branch_energy_status=branch['status'])
        config=saved_config(new,end)
        result['saved_RTE_config']=asdict(config)
        context_hash=digest(dict(identity=identity,P1=conditions,shot_execution_SOURCE=source))
        arm_hash=digest(dict(DF=identity['df_hamiltonian'],L_D=3,PF='partial-S2',r=4,K=2,
            delta=saved.DELTA,canonical_tail_hash=config.tail_hash))
        # Verify all stored bounds before the first shot call. No changed
        # allowance and no use of an already attenuated radius as PF radius.
        bounds=[checked_bounds(row,identity,config) for row in combined['rounds']]
        print('Saved input, SOURCE, P1 bounds and branch evidence: PASS',flush=True)
        for row,bound,provenance in zip(combined['rounds'],bounds,selected,strict=True):
            item=account_saved_round(row,bound,config,context_hash=context_hash,arm_hash=arm_hash)
            item.update(shot_execution_SOURCE=source,signal_execution_SOURCE=provenance['signal_execution_SOURCE'],
                signal_result_commit=provenance['signal_result_commit'],signal_result_path=provenance['signal_result_path'])
            result['rounds'].append(item)
        result['totals']=aggregate_shots(result['rounds'])
        for row in result['rounds']:
            fraction=Fraction(row['N_round'],result['totals']['N_all_rounds'])
            row['total_shot_share_exact']=str(fraction)
        result['status']=result['totals']['status']
        fields=['m','Q','T_Ha_inverse','PF_radius_lower','corrected_error_disk','normalization',
            'attenuation','raw_radius_lower','N_X','N_Y','N_round','beta_stat','alpha_X','alpha_Y',
            'alpha_each_exact','total_shot_share_exact','status','guarantee_status',
            'signal_execution_SOURCE','signal_result_commit','shot_execution_SOURCE']
        with (output/'rounds.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
            writer.writeheader();writer.writerows(result['rounds'])
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        result['serialization_check']=verify_serialized_totals(output,result['totals'])
        print(result['status']+'; total shots='+str(result['totals']['N_all_rounds']),flush=True)
    except Exception as error:
        result['status']='STOP_'+type(error).__name__
        result['stop_reason']=str(error) if isinstance(error,ValueError) else type(error).__name__
        error_to_raise=error
    result['elapsed_seconds']=time.perf_counter()-started
    result['environment']=dict(host=platform.node(),python=platform.python_version(),
        numpy=importlib.metadata.version('numpy'),cpu_affinity=len(os.sched_getaffinity(0)),
        BLAS_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},gpu_used=False)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if error_to_raise is not None:
        raise error_to_raise
    return result
