"""Quote stored law metadata and index SOURCE functions. No quantum imports."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
BASE_COMMIT='21fa1d08cf53422bd6e44e832ce8cc453f6d3ebf'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,o):(BASE/name).write_text(json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n')

def inspect():
    functions={
        'src/trotterlib/rte.py':['iter_sample_rte_events','_make_event','event_unitary'],
        'src/trotterlib/pauli_signal_adapter.py':['append_pauli_evolution','append_pauli_event','trajectory'],
        'src/trotterlib/h4_legacy_export.py':['build_export'],
        'src/trotterlib/h4_ball_evidence.py':['grid_masses','verify_probability_mapping','finite_implemented_mean','semantic_bound'],
        'src/trotterlib/h4_d2_pauli_control.py':['_tokens','_rotation','central_control_body'],
        'src/trotterlib/fixed_signal_evaluation.py':['default_compiler','hadamard_wrapper'],
        'src/trotterlib/rte_compiled_cost.py':['transpile_and_measure_cost'],
        'src/trotterlib/h4_arb_pr0_preparation.py':['build_plan','_selected_unitary_enclosure'],
        'src/trotterlib/h4_arb_bridge.py':['_native_unitary_enclosure','_residual_interval'],
        'src/trotterlib/h4_central_rho_plan.py':['drafts'],
    }
    locations=[];files={}
    for name,names in functions.items():
        path=ROOT/name;files[name]=sha(path)
        assert subprocess.check_output(['git','show',BASE_COMMIT+':'+name],cwd=ROOT)==path.read_bytes()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.FunctionDef) and node.name in names:
                locations.append(dict(path=name,function=node.name,line=node.lineno,end_line=node.end_lineno))
    model_path='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json'
    result_path='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json'
    six_path='artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.json'
    model,result,six=map(lambda p:read(ROOT/p),(model_path,result_path,six_path));case=model['cases']['PR'];norm=result['normalization']['PR']
    for p in (model_path,result_path,six_path,'artifacts/non_df_h4_central_rho_preparation/2026-10-24/contracts_DRAFT.json',
        'artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/verification.json',
        'artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/run01/STOP.json',
        'artifacts/non_df_h4_tail_rho_investigation/2026-10-24/verification.json'):
        files[p]=sha(ROOT/p)
    save('law_QUOTED.json',dict(operation='saved fields only; no CDF reconstruction, RNG, mean or residual evaluation',
        C_SOURCE='e80d7510880f8db993b777c627bb8e95b05a070d',model_path=model_path,result_path=result_path,
        fixed=dict(n_qubits=model['n_qubits'],step_time=model['step_time'],config=case['config'],
            Pauli_terms=len(model['hamiltonian']['terms']),retained_blocks=len(case['blocks']),tail_components=len(case['components'])),
        state_hash=model['state_hash'],qubit_order=model['qubit_order'],
        orders=[0,2,4],order_effective_probabilities=norm['order_effective_probabilities'],
        component_effective_probabilities=norm['component_effective_probabilities'],
        component_metadata=case['components'],angles_binary64=case['law']['angles'],
        native_scalar_phases_binary64=case['native_scalar_phases'],saved_normalization=case['saved_normalization'],
        saved_log_B_upper_encloses=norm['saved_log_B_upper_encloses'],
        range_plus_minus_saved_B_certified=norm['range_plus_minus_saved_B_certified'],
        IID_uniform_word_assumption=norm['IID_uniform_word_assumption'],PCG64_independence_proved=False,
        formal_certificate_accepted=False))
    save('local_results_QUOTED.json',dict(operation='copy six existing local rows; no residual reevaluation',
        source=six_path,rows=six['rows'],new_scientific_calls=0,PR_law_rho=None,formal_certificate_accepted=False))
    runtime_root=Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/lib/python3.11/site-packages/qiskit')
    runtime={}
    for rel in ('circuit/quantumcircuit.py','transpiler/preset_passmanagers/level1.py','transpiler/passes/optimization/optimize_1q_decomposition.py'):
        p=runtime_root/rel;runtime[str(p)]=sha(p)
    save('source_evidence.json',dict(status='STATIC_SOURCE_AND_SAVED_FIELDS_ONLY',base_commit=BASE_COMMIT,
        files=files,functions=locations,compiler_source_files=runtime,
        compiler=dict(qiskit='1.3.0',basis=['rz','sx','x','cx'],optimization_level=1,seed=17,
            routing=None,layout=None,backend=None,scope='SOURCE-selected current profile, not proof of every output'),
        unresolved_compiler_edge='Optimize1qGatesDecomposition.run invokes accelerated numerical synthesis; no all-input error or proof trace saved',
        new_H4_numerical_evaluations=0,new_Arb_H4_state_actions=0,new_compile=0,new_sampling=0,new_RNG_replay=0,
        new_scientific_grants=0,N_G_reanalysis=0,formal_certificate_accepted=False))
    policy=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-23.md')
    save('policy_observation.json',dict(path=str(policy),section_10_3_present='### 10.3' in policy.read_text(),
        applied='8.4 / 10 plus current direct user instructions; no newly built approval-proof machinery',
        resource_policy='no arbitrary CPU/RAM/worker cap; no shared or other-user changes',
        scientific_scope='static design and synthetic tests only; no scientific run or grant'))
    print(json.dumps(dict(status='STATIC_QUOTES_SAVED',source_files=len(files),function_locations=len(locations),new_science=0)))

if __name__=='__main__':inspect()
