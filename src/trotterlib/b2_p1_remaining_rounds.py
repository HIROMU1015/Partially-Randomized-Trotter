"""P1 continuation: fixed m=1..16, no circuit preparation/build or shots.

The saved coefficient projection is mean-only. It is never represented as an
executable B2 preparation. Endpoint signals/native checks are reused as evidence.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

from . import b2_p1_signal_validation as p
from .df_trotter.decompose import diag_hermitian
from .df_rte_tail import diagonal_pauli_matrix
from .pauli_signal_adapter import digest
from .rpe_resource_accounting import RPERoundSpecification
from .rte import RTEComponent, CompilerSettings, finite_rte_operator_moments

PRIOR_COMMIT='ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5'
PRIOR_SOURCE='5ef55f923e73e61bada383d0cfe98283e7984e9d'
PRIOR_PATH='artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json'
SCREEN_COMMIT='e69a51de70651ced673c22d2b215683a9f30059b'
SCREEN_PATH='artifacts/track_a_b2_rpe_attenuation_20261106/run01/result.json'
REQUESTED_ROUNDS=tuple(range(1,17))


@dataclass(frozen=True)
class MeanRound:
    """Signal record binding only: no sampled_request/native/price methods."""
    context: object
    specification: object
    config: object
    projection_hash: str
    prior_preparation_hash: str

    @property
    def q(self):
        return self.specification.q_m

    @property
    def binding(self):
        return digest(dict(mode='mean_only_no_circuit',context=self.context.context_hash,
            m=self.specification.round_index,delta=self.specification.delta_time,
            projection=self.projection_hash,prior_preparation=self.prior_preparation_hash,
            law=p.connection.accounting._law_hash(self.config)))


def checked_prior():
    prior=p.public_json(PRIOR_COMMIT,PRIOR_PATH)
    p.require(prior['execution_SOURCE']==PRIOR_SOURCE and
              prior['status']=='TWO_ROUND_CONDITIONAL_PASS','PRIOR_RESULT_STATUS_SOURCE')
    p.require([r['m'] for r in prior['rounds']]==[0,17] and
              all(r['status']=='CONDITIONAL_PASS' for r in prior['rounds']),'PRIOR_ENDPOINTS')
    p.require(prior['controlled_correctness']['status']=='PASS' and
              len(prior['controlled_correctness']['checks'])==16 and
              all(c['passed'] for c in prior['controlled_correctness']['checks']),'PRIOR_CONTROLLED_EVIDENCE')
    fixed=prior['fixed_conditions']
    p.require(fixed['caps_rad']==p.CAPS and fixed['allocation']==p.asdict(p.ALLOCATION) and
              fixed['numerical_allowance']==p.U and fixed['independent_match_tolerance']==p.MATCH and
              (fixed['L_D'],fixed['r'],fixed['K'],fixed['delta'])==(p.LD,p.R,p.K,p.DELTA),'P1_CONDITIONS')
    return prior,p.public_json(SCREEN_COMMIT,SCREEN_PATH)


def mean_preparation(ham,screen,prior):
    """Reuse saved law/coefficients, without Gaussian operations or extraction."""
    projection=screen['coefficient_projection']
    p.require(screen['input']['sha256']==p.DF_SHA and
              screen['input']['df_hamiltonian_hash']==prior['input_identity']['df_hamiltonian'],
              'SAVED_PROJECTION_INPUT')
    coefficients=projection['components']
    lam=projection['lambda_R'];identity=projection['extracted_tail_identity']
    p.require(lam==prior['normalization_law']['lambda_R'] and
              identity==prior['normalization_law']['tail_identity'],'SAVED_PROJECTION_LAW')
    components=tuple(RTEComponent(
        component_id=f"df-fragment-{v['fragment']}:"+''.join(f'Z{k}' for k in v['support']),
        probability=abs(v['coefficient'])/lam,coefficient_abs=abs(v['coefficient']),
        coefficient_sign=1 if v['coefficient']>0 else -1,
        df_fragment_id=f"df-fragment-{v['fragment']}",diagonal_pauli_support=tuple(v['support']))
        for v in coefficients)
    # This is the existing attenuation projection, not the executable tail hash.
    tail=SimpleNamespace(tail_id='B2_LD3_normalization_projection_only',
        tail_hash=projection['projection_hash'],lambda_r=lam,components=components)
    return SimpleNamespace(exact_rte_lambda_r=lam,constant_coefficient=ham.constant,
        extracted_identity_coefficient=identity,rte_preparation=SimpleNamespace(symbolic_tail=tail))


def ideal_program_mean(inputs,screen,config,phase):
    """Numerical exterior-power Gaussian/diagonal mean, not a circuit build.

    This is the ideal semantics already covered by the selected native checks.
    It does not validate every event or float implementation of long circuits.
    """
    ham=inputs['ham'];n=ham.n_qubits;d=1<<n
    occupation=np.asarray([[(j>>bit)&1 for bit in range(n)] for j in range(d)])
    def basis(matrix):
        orbital,eigenvalues=diag_hermitian(matrix,sort='descending_abs',assume_hermitian=True)
        return inputs['ref'].fock_lift(orbital),eigenvalues
    rotation,eps=basis(ham.one_body)
    diagonal=np.exp(-.5j*p.DELTA*(occupation@eps))
    halves=[(rotation*diagonal)@rotation.conj().T]
    tail=np.zeros((d,d),complex)
    saved=screen['coefficient_projection']['components']
    for index,(weight,matrix) in enumerate(zip(ham.lambdas,ham.g_matrices,strict=True)):
        rotation,eta=basis(matrix)
        if index<p.LD:
            diag=np.exp(-.5j*p.DELTA*weight*(occupation@eta)**2)
            halves.append((rotation*diag)@rotation.conj().T)
        else:
            # Only the fixed saved coefficient dictionary is composed. No new
            # support selection, circuit event enumeration, or support search.
            diagonal=sum((v['coefficient']*np.diag(diagonal_pauli_matrix(n,v['support']))
                          for v in saved if v['fragment']==index),np.zeros(d))
            tail+=(rotation*diagonal)@rotation.conj().T
    moments=finite_rte_operator_moments(tail/config.lambda_r,config)
    full=p.compose_step(halves,moments.corrected_operator,phase)
    return full[np.ix_(inputs['indices'],inputs['indices'])]


def evaluate_in_order(evaluate,save):
    """Never evaluate another round after the first failed qualification."""
    rows=[]
    for m in REQUESTED_ROUNDS:
        item=evaluate(m)
        if not item['independent_numerical_match']:
            item['status']='STOP_NUMERICAL_MATCH'
        else:
            item['status']='CONDITIONAL_PASS' if item['qualification']['feasible'] else 'STOP_P1_INELIGIBLE'
        rows.append(item)
        save(rows)
        print('Round m='+str(m)+': '+item['status'],flush=True)
        if item['status']!='CONDITIONAL_PASS':
            return rows,item['status']
    return rows,'REMAINING_16_ROUND_CONDITIONAL_PASS'


def run(private_dir,output):
    started=time.perf_counter();output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source=p.git('rev-parse','HEAD').decode().strip()
    paths=['src/trotterlib/b2_p1_remaining_rounds.py','scripts/run_b2_p1_remaining_rounds.py',
        'src/trotterlib/b2_p1_signal_validation.py','src/trotterlib/o1_m1_resource_bridge.py',
        'src/trotterlib/pr2_matched_accuracy_m1_execution.py','src/trotterlib/b2_rpe_connection.py']
    for path in paths:
        p.require(p.git('show',source+':'+path)==(p.ROOT/path).read_bytes(),'COMMITTED_SOURCE_REQUIRED')
    result=dict(schema_version='b2_p1_remaining_rounds_v1',execution_SOURCE=source,
        SOURCE_SHA256={path:hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest() for path in paths},
        prior_result_commit=PRIOR_COMMIT,prior_result_path=PRIOR_PATH,started_at_UTC=datetime.now(timezone.utc).isoformat(),
        rounds=[],mandatory_stop=True,guarantee_status='not_certified',
        forbidden_work=dict(SCF=0,DF_fit=0,state_solve=0,circuit_build=0,compile=0,
            sampling=0,endpoint_signal_recomputation=0,other_round_signals=0,shot_accounting=0,
            branch_success=0,price_MC=0,total_cost=0,strong_baseline=0,scientific_review=0))
    error_to_raise=None
    try:
        prior,screen=checked_prior()
        result['fixed_conditions']=dict(prior['fixed_conditions'],requested_rounds=list(REQUESTED_ROUNDS))
        result['reused_controlled_correctness']=dict(commit=PRIOR_COMMIT,path=PRIOR_PATH,
            status='PASS',check_count=16,scope=prior['controlled_correctness']['scope'],reexecuted=False)
        inputs=p.load_private_inputs(private_dir)
        result['input_identity']=inputs['public']
        for key in ('original_hamiltonian','original_state','df_hamiltonian','sector','spin_orbital_order'):
            p.require(inputs['public'][key]==prior['input_identity'][key],'PRIOR_CURRENT_INPUT_'+key)
        print('Input identity and saved correctness evidence: PASS',flush=True)
        prep=mean_preparation(inputs['ham'],screen,prior)
        ix=np.ix_(inputs['indices'],inputs['indices'])
        det=[inputs['one_body'][ix],*[b[ix] for b in inputs['fragments'][:p.LD]]]
        deterministic=[(str(i),p.legacy._eigendecomposition(b)) for i,b in enumerate(det)]
        tail=sum((b[ix] for b in inputs['fragments'][p.LD:]),np.zeros_like(det[0]))-prep.extracted_identity_coefficient*np.eye(len(det[0]))
        eig=p.legacy._eigendecomposition(tail/prep.exact_rte_lambda_r)
        config,distribution,kernel,steps=p.legacy_mean_steps(prep,deterministic,eig)
        result['normalization_law']=dict(lambda_R=config.lambda_r,tau=config.dimensionless_step_time,
            b_event=config.distribution_normalization,tail_identity=prep.extracted_identity_coefficient,
            projection_hash=screen['coefficient_projection']['projection_hash'],
            prior_executable_preparation_reference=prior['normalization_law']['preparation_hash'],
            prior_executable_tail_reference=prior['normalization_law']['tail_hash'],
            executable_preparation_generated=False)
        p.require(config.distribution_normalization==screen['canonical_law']['normalization_per_event'], 'FINITE_LAW_NORMALIZATION')
        halves=[expm(-.5j*p.DELTA*b) for b in det]
        independent={'ideal_tail':p.compose_step(halves,expm(-1j*p.DELTA*tail),kernel['phase']),
            'corrected':p.compose_step(halves,p.paired_polynomial_action(tail,np.eye(len(tail)),p.DELTA,p.R,p.K),kernel['phase'])}
        moments=finite_rte_operator_moments(tail/config.lambda_r,config)
        independent['raw']=p.compose_step(halves,moments.attenuated_event_mean_operator,kernel['phase'])
        native_step=ideal_program_mean(inputs,screen,config,kernel['phase'])
        ctx=p.connection.RPEConnectionContext(inputs['spec']['original_hamiltonian'],inputs['spec']['df_hamiltonian'],
            inputs['spec']['original_state'],inputs['spec']['sector'],inputs['spec']['spin_orbital_order'],source,
            CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'),'no_backend','rz_count',
            None,.00015936001019904,.95,'P1 [-128,128] Ha; shift0; beta<pi/3; branch untested')
        def evaluate(m):
            round_=MeanRound(ctx,RPERoundSpecification(m,p.DELTA),config,
                screen['coefficient_projection']['projection_hash'],prior['normalization_law']['preparation_hash'])
            row=p.evaluate_signal_round(inputs,round_,kernel=kernel,steps=steps,
                independent=independent,native_step=native_step)
            row.update(execution_SOURCE=source,context_hash=ctx.context_hash,round_binding=round_.binding,
                input_identity_status='PASS',controlled_evidence_commit=PRIOR_COMMIT,
                implementation_scope='ideal Gaussian exterior-power mean Q power; prior selected native checks reused; no new circuits or all-support correctness')
            return row
        def checkpoint(rows):
            result['rounds']=rows
            (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        result['rounds'],result['status']=evaluate_in_order(evaluate,checkpoint)
        for name,path in inputs['files'].items():
            p.require(hashlib.sha256(path.read_bytes()).hexdigest()==inputs['public']['file_SHA256'][name],'PRIVATE_INPUT_CHANGED')
        result['private_input_unchanged']=True
        all_rows=[dict(r,evidence='reused',result_commit=PRIOR_COMMIT) for r in prior['rounds']]+[
            dict(r,evidence='new',result_commit=None) for r in result['rounds']]
        all_rows.sort(key=lambda r:r['m'])
        result['all_18_round_signal_status']=('ALL_18_P1_CONDITIONAL_SIGNAL_PASS'
            if [r['m'] for r in all_rows]==list(range(18)) and all(r['status']=='CONDITIONAL_PASS' for r in all_rows)
            else 'NOT_ALL_18_QUALIFIED')
        (output/'all_18_rounds.json').write_text(json.dumps(dict(scope='conditional_signal_qualification_only',
            status=result['all_18_round_signal_status'],rounds=all_rows,guarantee_status='not_certified',mandatory_stop=True),
            ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except Exception as error:
        result['status']='STOP_'+type(error).__name__
        result['stop_reason']=str(error) if isinstance(error,ValueError) else type(error).__name__
        error_to_raise=error
    result['elapsed_seconds']=time.perf_counter()-started
    result['environment']=dict(host=platform.node(),python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','qiskit','openfermion','h5py')},
        cpu_affinity=len(os.sched_getaffinity(0)),BLAS_env={k:os.environ.get(k) for k in
        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},gpu_used=False)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if error_to_raise is not None:
        raise error_to_raise
    return result
