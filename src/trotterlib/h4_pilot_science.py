"""H4-specific science and input IO, reached only after the stdlib guard.

Artificial eight-qubit data exercise the same case evaluation/serialization.
No call below runs on import; the molecular builder has a separate entrypoint.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .fixed_signal_evaluation import FixedSignalConfig, prepare_signal_adapter,validate_fixed_signal_result
from .h2_non_df_pilot import check, numerical_evidence, precision_accounting, dense_law_mean
from .h2_non_df_pilot_validation import event_context, trajectory_reference
from .h4_audited_evaluation import evaluate_audited_signal
from .h4_execution_contract import seed_for
from .h4_molecular_inputs import resolve_prefixes
from .isolated_qiskit_execution import read, save, sha, digest
from .pauli_signal_adapter import PauliHamiltonian, array_hash, append_pauli_event
from .rte import _make_event, event_unitary
from .rte_event_audit import TrajectoryAudit, verify_join


def case_config(h, representation, method, plan, receipt):
    e=plan['evolution']
    return FixedSignalConfig(representation, method=method, T=e['T'], q=e['q'],
        L_D=receipt['prefixes'][representation]['L_D'] if method=='PR' else None,
        r=e['r'], K=e['K'], seed=seed_for(representation,method),
        cost_samples=plan['MC']['trajectories_per_PR'])


def legacy_plan(plan):
    return dict(numerical=plan['numerical'], evaluation=dict(alpha_axis=plan['precision']['alpha_axis']))


def save_inputs(destination, pauli, df, state, receipt, *, artificial=False):
    """Seal the snapshot and receipt before any evaluation worker is released."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    snap=destination/'snapshot.npz'
    # Molecular builder already stored all MO integrals and independent evidence.
    if not snap.exists():
        with snap.open('xb') as f:
            np.savez(f,nuclear_repulsion=df.constant,df_one_body=df.one_body,df_lambdas=df.lambdas,
                df_g_matrices=np.asarray(df.g_matrices),state=state)
    save(destination/'pauli.json',dict(num_qubits=pauli.num_qubits,constant=pauli.constant,
        terms=pauli.terms,metadata_json=pauli.metadata_json))
    dfhash=prepare_signal_adapter(df,FixedSignalConfig('df_native')).metadata()['hamiltonian_hash']
    receipt=dict(receipt,status='PASS',num_qubits=8,physical_sector_dimension=36,
        artificial_only=artificial,DF_metadata=df.metadata,state_hash=array_hash(state),
        Pauli_hamiltonian_hash=pauli.hamiltonian_hash,DF_hamiltonian_hash=dfhash,
        snapshot_sha256=sha(snap),pauli_json_sha256=sha(destination/'pauli.json'))
    save(destination/'receipt.json',receipt)
    return receipt


def load_inputs(destination, expected):
    from .df_hamiltonian import DFHamiltonian
    destination=Path(destination)
    if sha(destination/'receipt.json') != expected['receipt_sha256']:
        raise ValueError('H4_INPUT_RECEIPT_CHANGED')
    receipt=read(destination/'receipt.json')
    if (receipt['status']!='PASS' or receipt['num_qubits']!=8 or receipt['physical_sector_dimension']!=36
            or sha(destination/'snapshot.npz')!=receipt['snapshot_sha256']
            or receipt['snapshot_sha256']!=expected['snapshot_sha256']
            or sha(destination/'pauli.json')!=receipt['pauli_json_sha256']):
        raise ValueError('H4_INPUT_SNAPSHOT_CHANGED_OR_DIMENSION')
    p=read(destination/'pauli.json')
    pauli=PauliHamiltonian(p['num_qubits'],p['constant'],tuple(map(tuple,p['terms'])),p['metadata_json'])
    with np.load(destination/'snapshot.npz',allow_pickle=False) as z:
        df=DFHamiltonian(float(z['nuclear_repulsion']),z['df_one_body'].copy(),z['df_lambdas'].copy(),
            tuple(g.copy() for g in z['df_g_matrices']),receipt['DF_metadata'])
        state=z['state'].copy()
    if state.shape!=(256,) or df.n_qubits!=8 or abs(np.linalg.norm(state)-1)>1e-12:
        raise ValueError('H4_INPUT_STATE_DIMENSION')
    if (array_hash(state)!=receipt['state_hash'] or pauli.hamiltonian_hash!=receipt['Pauli_hamiltonian_hash']
            or prepare_signal_adapter(df,FixedSignalConfig('df_native')).metadata()['hamiltonian_hash']!=receipt['DF_hamiltonian_hash']
            or resolve_prefixes(pauli,df)!=receipt['prefixes']):
        raise ValueError('H4_INPUT_IDENTITY_OR_PREFIX_CHANGED')
    return pauli,df,state,receipt


def event_gates(adapter, config, plan):
    """Algebraic order checks, no RNG and no forced-event cost sampling."""
    components, matrices=event_context(adapter)
    tol=plan['numerical']['trajectory_unitary_agreement']
    forced=[]
    if adapter.randomized:
        tail=adapter.lambda_r*sum(c.probability*matrices[c.component_id] for c in components)
        check(np.linalg.norm(tail-adapter.operators()[2].toarray(),2),
            plan['numerical']['Hamiltonian_operator_agreement'],'H4_TAIL_OPERATOR')
        for i,order in enumerate(adapter.distribution.orders):
            event=_make_event(tuple(j%len(components) for j in range(order+1)),components,adapter.distribution,i)
            expected=event_unitary(event,matrices)
            if adapter.representation=='pauli_jw':
                circuit=QuantumCircuit(adapter.num_qubits)
                append_pauli_event(circuit,event,adapter.labels)
            else:
                from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
                prep=adapter.preparation.rte_preparation
                circuit=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.basis_registry).build_event(prep.request_for_event(event)).circuit
            forced.append(dict(order=order,event=event.to_dict(),
                gate=check(np.linalg.norm(Operator(circuit).data-expected,2),tol,'H4_FORCED_ORDER_EVENT'),
                cost_sample=False,compile=False))
    checks=[]
    def validate(a,c,sequences,circuit,index):
        if index>=3:
            return
        expected=trajectory_reference(a,c,sequences,matrices) if a.randomized else dense_law_mean(a,q=c.q)
        actual=Operator(circuit).data
        if a.representation=='df_native':
            n=len(expected);expected=np.block([[np.eye(n),np.zeros((n,n))],[np.zeros((n,n)),expected]])
        gate=check(np.linalg.norm(actual-expected,2),tol,'H4_CAPTURED_PRODUCTION_TRAJECTORY')
        checks.append(dict(trajectory_index=index,gate=gate,replayed_sampling=False,phase_alignment=False))
    return validate,checks,forced


def evaluate_h4_case(hamiltonian,state,representation,method,plan,receipt,provenance,destination,*,audit_enabled=True):
    destination=Path(destination); started=time.time()
    config=case_config(hamiltonian,representation,method,plan,receipt)
    adapter=prepare_signal_adapter(hamiltonian,config)
    evidence=numerical_evidence(adapter,state,config,legacy_plan(plan))
    save(destination/'numerical_evidence.json',evidence)
    validator,checks,forced=event_gates(adapter,config,plan)
    audit=TrajectoryAudit(representation+'_'+method,provenance,directory=destination/'trajectories',
        validator=validator,max_trajectories=config.cost_samples if adapter.randomized else 1) if audit_enabled else None
    result=evaluate_audited_signal(hamiltonian,state,config,epsilon_sig=plan['precision']['epsilon_sig'][0],
        numerical_allowance=evidence['allowance'],log_B_margin=evidence['log_B_margin'],
        evidence_kind='EMPIRICAL',evidence_ref='numerical_evidence.json',verify_compiled=True,
        source_provenance=provenance,event_audit=audit,
        state_metadata=dict(method=plan['state']['method'],molecular=not receipt['artificial_only'],
            physical_sector='Nalpha=Nbeta=2',shared_across_four_cases=True))
    if array_hash(state)!=receipt['state_hash']:
        raise ValueError('H4_STATE_MUTATED')
    precision=[precision_accounting(result,e,legacy_plan(plan)) for e in plan['precision']['epsilon_sig']]
    # Apply the full allowance/headroom guard also to the primary result.
    result['shots'],result['total_RZ']=precision[0]['shots'],precision[0]['total_RZ']
    cost=result['circuit_cost'];xs=[x['RZ']['cosine'] for x in cost['samples']];ys=[x['RZ']['sine'] for x in cost['samples']]
    cost['paired_covariance_RZ']=float(np.cov(xs,ys,ddof=1)[0,1]) if len(xs)>1 else 0.
    cost['paired_covariance_interpretation']='observed finite cost sample only; no rare/population guarantee'
    expected_count=config.cost_samples if method=='PR' else 1
    if cost['sample_count']!=expected_count or (audit is not None and len(audit.rows)!=expected_count):
        raise ValueError('H4_FIXED_CALL_BUDGET')
    save(destination/'actual_call_accounting.json',dict(paired_trajectories=cost['sample_count'],
        wrapper_compile_requests=2*cost['sample_count'],new_compiles=cost['cache_misses'],cache_hits=cost['cache_hits'],
        production_RTE_events=sum(row['event_count'] for row in audit.rows) if audit else None,
        event_audit_additional_draws=0,validation_replay_draws=0,automatic_retry=False))
    result['validation'].update(status='H4_FIXED_CASE_TECHNICAL_PASS',molecular_validation=not receipt['artificial_only'],
        resource_status='CONDITIONAL_CNG_AVAILABLE' if result['total_RZ'] is not None else 'N_G_UNDEFINED')
    validate_fixed_signal_result(result)
    for joined,sample in zip(audit.rows if audit else [],cost['samples'] if audit else [],strict=True):
        verify_join(joined,expected_case=representation+'_'+method,expected_binding=provenance)
        if joined['compiled_RZ']!=sample['RZ'] or joined['seed']!=sample['seed']:
            raise ValueError('H4_AUDIT_PRICE_OR_SEED_JOIN')
    save(destination/'precision_rows.json',precision)
    save(destination/'validation.json',dict(status='PASS',trajectory_operator_checks=checks,
        forced_order_checks=forced,production_sampling_replayed=False,thresholds=plan['numerical']))
    save(destination/'result.json',result)
    save(destination/'status.json',dict(status='COMPLETE_FIXED_CASE',SOURCE=provenance['SOURCE'],
        case=representation+'_'+method,started=started,ended=time.time(),artificial_only=receipt['artificial_only'],
        event_join_count=len(audit.rows) if audit else 0,one_shot=True,automatic_retry=False))
    return result,precision,audit


def integrate(run,plan,expected_binding):
    from .h4_execution_contract import CASES
    run=Path(run);table=[];states={};missing=[]
    for case in CASES:
        folder=run/case
        if not (folder/'result.json').is_file() or not (folder/'status.json').is_file():
            missing.append(case);states[case]='MISSING_OR_FAILED';continue
        status=read(folder/'status.json');result=read(folder/'result.json')
        if status['status']!='COMPLETE_FIXED_CASE' or result['provenance']['source']!=expected_binding:
            raise ValueError('H4_RESULT_IDENTITY')
        if read(folder/'validation.json')['status']!='PASS':
            raise ValueError('H4_RESULT_VALIDATION')
        samples=result['circuit_cost']['samples'];seen=set()
        for index,sample in enumerate(samples):
            tid=sample['trajectory_id']
            if tid in seen:raise ValueError('H4_DUPLICATE_TRAJECTORY_JOIN')
            seen.add(tid)
            joined=read(folder/'trajectories'/tid/'price_join.json')
            verify_join(joined,expected_case=case,expected_binding=expected_binding)
            events=read(folder/'trajectories'/tid/'events.json')
            if events['event_record_hash']!=joined['event_record_hash'] or digest({k:v for k,v in events.items() if k!='event_record_hash'})!=events['event_record_hash']:
                raise ValueError('H4_MISSING_OR_CORRUPT_PRECOMPILE_EVENT_RECORD')
            if (joined['trajectory_index']!=index or joined['seed']!=sample['seed']
                    or joined['compiled_RZ']!=sample['RZ'] or joined['trajectory_record_hash']!=sample['trajectory_record_hash']):
                raise ValueError('H4_RESULT_EVENT_PRICE_JOIN')
        rows=read(folder/'precision_rows.json')
        if [r['epsilon_sig'] for r in rows]!=plan['precision']['epsilon_sig']:
            raise ValueError('H4_PRECISION_COVERAGE')
        for row in rows:
            table.append(dict(case=case,**row,C=result['circuit_cost']['axes'],source=expected_binding))
        states[case]='COMPLETE_FIXED_CASE'
    return dict(status='FOUR_CASES_COMPLETE' if not missing else 'PARTIAL_RESULTS_NOT_FOUR_METHOD_COMPARISON',
        rows=table,case_states=states,missing_cases=missing,scientific_superiority_claimed=False)


def artificial_inputs():
    """Eight spin modes with supplied arrays; no integrals/SCF/DF generation."""
    from .df_hamiltonian import DFHamiltonian
    n=8;one=np.diag(np.linspace(-.1,.1,n));one[0,1]=one[1,0]=.007
    gs=[]
    for i in range(3):
        g=np.diag(np.linspace(.02*(i+1),.09*(i+1),n));g[i,i+1]=g[i+1,i]=.004;gs.append(g)
    df=DFHamiltonian(.05,one,np.array([.2,-.1,.05]),tuple(gs),dict(model_id='supplied_artificial_eight_qubit'))
    terms=[('IIIIIIII',.05),('IIIIIIIZ',.1),('IIIIIIZI',.07),('IIIIIZII',.05),
        ('IIIIXIXI',-.02),('IIIIYIYI',.015),('IZIZIIII',.01)]
    pauli=PauliHamiltonian.from_terms(terms,model_id='supplied_artificial_eight_qubit')
    psi=np.zeros(256,complex);psi[15]=1/np.sqrt(2);psi[51]=1j/np.sqrt(2)
    receipt=dict(prefixes=resolve_prefixes(pauli,df),state_hash=array_hash(psi),artificial_only=True)
    return pauli,df,psi,receipt
