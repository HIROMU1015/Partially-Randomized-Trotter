#!/usr/bin/env python3
"""Static preparation only. No molecule law, state, sampling, compile or ranking."""
import csv,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_rte_event_extension_preparation/2026-10-26'
BASE='288d2206d689a9c24516844cba2100c2d9845a32'
FUTURE='artifacts/pauli_rte_event_extension/2026-10-26'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def csvsave(p,rows):
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    assert not (ROOT/FUTURE).exists()
    roles={
     'H4_model':'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json',
     'H4_reference':'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json',
     'H6_model':'artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/model.json',
     'H6_reference':'artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/reference.json',
     'H4_comparison':'artifacts/pauli_hybrid_rte_comparison/2026-10-25/comparison_by_precision.json',
     'H6_comparison':'artifacts/h6_rte_fairness/2026-10-25/comparison_by_precision.json',
     'prior_inventory':'artifacts/pauli_hybrid_rte_preparation/2026-10-25/candidate_evidence_inventory.csv',
     'H6_prior_registry':'artifacts/h6_rte_fairness/2026-10-25/registered_RTE_candidates.csv'}
    for m,n,M in [('H4',8,184),('H6',12,918)]:
        model=read(ROOT/roles[m+'_model']);ref=read(ROOT/roles[m+'_reference'])
        assert model['n_qubits']==n and len(model['state'])==1<<n and len(model['hamiltonian']['terms'])==M
        assert model['qubit_order']=='qiskit_little_endian' and 'reference' in ref
    with (ROOT/roles['prior_inventory']).open() as f:inventory=list(csv.DictReader(f))
    with (ROOT/roles['H6_prior_registry']).open() as f:h6_registry=list(csv.DictReader(f))
    old={(x['molecule'],int(x['q']),int(x['r']),int(x['K'])) for x in inventory if x['group']=='RTE'}
    old|={('H6',int(x['q']),int(x['r']),int(x['K'])) for x in h6_registry}
    candidates=[]
    for molecule,rs in [('H6',[16,32,64,128]),('H4',[16,32])]:
        for r in rs:
            for K in [2,4]:
                assert (molecule,8,r,K) not in old
                initial=(molecule=='H6' and r in [16,32,64] and K==2) or (molecule=='H4' and r==16)
                candidates.append(dict(candidate_id=f'{molecule}_p000_q8_r{r}_K{K}',molecule=molecule,L_D=0,q=8,r=r,K=K,s=8*r,
                    stage='initial' if initial else 'conditional',trigger='INITIAL' if initial else 'H4_S256' if molecule=='H4' else 'H6_S1024' if K==2 else 'H6_K4',
                    status='NOT_EXECUTED_AWAITING_PERMISSION',N_cos=None,N_sin=None,G_exact=None,old_duplicate=False))
    assert len(candidates)==12 and sum(x['stage']=='initial' for x in candidates)==5
    csvsave(OUT/'candidate_plan.csv',candidates)
    plan=dict(schema='pauli_rte_event_extension_preparation_v1',status='DRAFT_DISABLED_AWAITING_SCIENTIFIC_PERMISSION',execute_enabled=False,
      base_commit=BASE,prepared_branch='codex/pauli-rte-event-extension-preparation-20261026',new_output_namespace=FUTURE,
      conditions=dict(molecules=['H4','H6'],geometry='linear; adjacent1.00A',basis='STO-3G',charge=0,spin=0,
        Pauli_nonidentity_terms=dict(H4=184,H6=918),L_D=0,q=8,T='0.8',formula='S2',event_K='maximum even event order; polynomial degree K+1',
        compiler='Qiskit1.3.0 rz/sx/x/cx opt1 seed17 no topology/routing override',control='central-rotation full Hadamard cosine/+Im sine',
        cost='compiled RZ excluding state preparation',epsilon_sig=['0.05','0.01','0.005','0.001','0.0005'],alpha_axis='0.025',
        rho='hypothetical zero',G='N_cos*C_cos+N_sin*C_sin',state_reference='same saved raw ray/full-H reference and exact common norm; no regeneration or renormalization'),
      candidates=candidates,
      adaptive_rules=dict(
        H6_S1024='After initial K2 s128/256/512: add K2 s1024 iff s512 is the finite-G exploration minimum for any epsilon OR its difference from the minimum is <=2 combined SE. Do not assert global optimality if the best remains at the upper boundary.',
        H6_K4='At a currently lowest or <=2combinedSE K2 r among16/32/64/128, add same-r K4 only if any epsilon is bias-ineligible/width-ambiguous OR (abs(signed_bias)+u)>=5%epsilon_axis OR Hybrid/RTE difference<=2combinedSE. Record the trigger before the batch. At most four K4 candidates; no unconditional K4 grid.',
        H4_S256='Add s256 K2/K4 only if s128 improves any corresponding epsilon over stored s64, is <=2combinedSE from s64, changes the observed Hybrid/RTE order, or the new Hybrid/RTE comparison is <=2combinedSE. If s128 is clearly worse and order unchanged in all relevant cells, stop H4.',
        ambiguity='No computable comparator or finite allowance: retain unresolved. No treating null as0/infinity; no forced scientific ranking.',
        stop='Stop after decrease-to-increase trend is resolved for relevant epsilons, H4 clear worsening without changed order, no material scientific value, source/numerical mismatch, or the finite candidate cap. Only the listed 12 candidate universe is allowed.'),
      target_precision=dict(H4=dict(schedule=[64,96,128],max_state_actions_per_candidate=3,early_stop='No WIDTH_AMBIGUITY in the five existing precision decisions; never repeat a completed precision unnecessarily',backend='fixed d3fdfad FLINT mean-matrix/vector contraction path'),
         H6=dict(scalar_bits=128,vector_bits=53,max_state_actions_per_candidate=1,backend='existing tight L2 matrix-free enclosure',unresolved_policy='Keep ambiguity; no new numerical model/allowance/threshold or state retry')),
      pricing=dict(exploration_n=8,confirmation_schedule=[48,96,192,384],
        confirmation_nomination='Freeze union of per-epsilon new RTE exploration minima before confirmation. Add already-sampled adjacent-s candidates only when their <=2combinedSE difference prevents locating the empirical minimum/turning trend; freeze these IDs before their confirmation. No old successful cohort rerun.',
        confirmation_extension='Extend a new candidate only when its G difference from fixed saved Hybrid or adjacent sampled RTE is <=2combinedSE AND nominal t95 halfwidth>1%G AND its own uncertainty exceeds the saved comparator floor. At384 keep near-tie/unresolved. No repeated old successful confirmations.',
        cohorts='No exploration/confirmation pooling. Frozen-nominee result and post-confirmation-set min remain distinct. Same trajectory supplies both axes; use paired covariance; five eps share prices.',
        seeds='Existing H4 [20261025,3101,L_D,q,r,K,tag,index] and H6 [20261025,6,L_D,q,r,K,tag,index]; exploration101/confirmation202; fresh r>8 candidate IDs, no unused-index or old-cohort replay assumption without checking.'),
      limits=dict(initial_candidates=5,total_new_candidates_max=12,law_descriptor_evaluations_max=12,
        initial_target_state_actions_max=9,total_target_state_actions_max=20,initial_exploration_trajectories_max=40,
        exploration_trajectories_max=96,confirmation_distinct_candidates_max=12,confirmation_trajectories_max=4608,
        total_new_trajectories_max=4704,total_new_axis_compiles_max=9408,new_Hamiltonian_state_reference=0,new_H2_scientific_calls=0,new_Hybrid_scientific_calls=0),
      comparisons=dict(primary='Finite sampled RTE family versus saved independently confirmed Hybrid, per molecule/epsilon. Reuse s64 and lower baselines without reranking before execution permission.',
        matched='Only quote existing exact q/r/K matched controls. No new Hybrid r>8 is in scope, so matched Hybrid at new r16/32/64/128 is unavailable/null; do not label these family comparisons matched.',
        H2='Saved results only; no state action/sampling/compile/reanalysis/reranking.'),
      provenance='Before each permitted scientific batch freeze exact IDs, SOURCE/code/runtime/input, seeds/index ranges, call counts and stop. Use a fresh driver/output; never reopen old fixed CLI output or STOP. This disabled preparation plan remains unchanged.',
      resources='Choose workers/memory from actual free resources and workload; no arbitrary CPU/RAM caps or shared/other-user modifications. Matrix-free H6; native FLINT H4. Do not treat s=q*r as sufficient to discard actual q/r/seed/phase/compiler provenance.',
      forbidden=['science before explicit permission','new Hamiltonian/state/reference','new H2 or Hybrid science','new geometry/algorithm/law/shot formula','old SOURCE/raw/grant/ledger/marker/STOP modification','formal certificate acceptance','population-tail guarantee','major-claim adoption','detailed GPT review','paper change'],
      expected_outputs=['candidate/precision/cohort CSV with s,q,r,K,eligibility,bias,B,u,N,C,SE,G,ratios,n','log-s log-G figure with saved Hybrid baselines','B and shot-count versus s','optional C/N tradeoff','technical report, SOURCE/input/raw/native preservation, commit/remote SHA'],
      approval='User chat permission is sufficient after review of this concrete scope; no approval certificate or grant machinery. Preparation stops awaiting that permission.')
    save(OUT/'execution_plan_DRAFT_DISABLED.json',plan)
    baseline=[]
    for molecule in ['H4','H6']:
        for x in read(ROOT/roles[molecule+'_comparison']):
            if x.get('molecule',molecule)!=molecule:continue
            for group in ['Hybrid','RTE']:
                selected=x['confirmation_'+group]['selected']
                baseline.append(dict(molecule=molecule,epsilon_sig=x['epsilon_sig'],group=group,**{k:selected.get(k) for k in ['candidate_id','L_D','q','r','K','B_exact','bias_cos_exact','bias_sin_exact','u_cos_exact','u_sin_exact','N_cos','N_sin','C_cos_exact','C_sin_exact','G_exact','G_SE','price_n']},source_file=roles[molecule+'_comparison'],source_commit='90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3' if molecule=='H4' else BASE,classification='LITERAL_QUOTE_OF_ALREADY_SELECTED_CONFIRMATION_ROW_NO_NEW_RANKING'))
    csvsave(OUT/'saved_baseline_quotes.csv',baseline)
    save(OUT/'inputs.json',dict(roles=roles,sha256={p:sha(ROOT/p) for p in roles.values()},raw_data_copied=False,state_reference_regenerated=False,source_ref='Existing fixed Git objects and separately verified GitHub reports; H4/H6 scientific input namespaces remain separate'))
    code=['src/trotterlib/pauli_rte_event_extension.py','scripts/prepare_pauli_rte_event_extension.py','tests/test_pauli_rte_event_extension.py','tests/test_pauli_rte_event_extension_mean.py',
      'src/trotterlib/pauli_signal_adapter.py','src/trotterlib/rte.py','src/trotterlib/h4_pauli_pr_search.py','src/trotterlib/rte_event_audit.py','src/trotterlib/h4_d2_pauli_control.py',
      'src/trotterlib/fixed_signal_evaluation.py','src/trotterlib/rte_compiled_cost.py','src/trotterlib/pauli_vector_enclosure.py','src/trotterlib/pauli_vector_tight_enclosure.py',
      'src/trotterlib/h4_conditional_cng.py','src/trotterlib/h6_rte_fairness.py','scripts/pauli_h2_h6_study.py',
      'src/trotterlib/h4_pauli_b0_comparison.py','src/trotterlib/h4_pauli_b0_execution.py',
      'src/trotterlib/h4_arb_bridge.py','src/trotterlib/h4_arb_pr0_preparation.py','src/trotterlib/h4_ball_evidence.py',
      'src/trotterlib/pauli_hybrid_rte_comparison.py',str((OUT/'inputs/h4_mean_d3.py').relative_to(ROOT))]
    save(OUT/'prospective_SOURCE_inputs.json',dict(base_commit=BASE,code_sha256={p:sha(ROOT/p) for p in code},input_sha256={p:sha(ROOT/p) for p in roles.values()},
      disabled_contract_sha256=sha(OUT/'execution_plan_DRAFT_DISABLED.json'),future_output=FUTURE,new_scientific_run_started=False,
      execution_driver='Kernel connections prepared; fresh orchestration/active batch SOURCE to be fixed after scientific permission, without changing old SOURCEs or this disabled plan.'))
    for name,n in [('artificial_compiler_tests.txt',6),('artificial_mean_tests.txt',1),('reused_integer_tests.txt',5)]:
        text=(OUT/name).read_text();assert f'Ran {n} test' in text and text.rstrip().endswith('OK'),name
    save(OUT/'verification.json',dict(status='PREPARATION_TECHNICAL_PASS',artificial_tests_passed=12,new_artificial_tests=7,reused_artificial_tests=5,
      artificial_mean_initial_import_failure='Arb environment lacks Qiskit; preserved log. Generated artificial law in compiler environment and passed it to Arb, matching production environment separation. No shared packages changed.',
      molecular_law_evaluations=0,molecular_state_actions=0,molecular_sampling=0,molecular_compiles=0,new_saved_value_reanalysis=0,new_N_G_or_ranking=0,real_figures=0,
      candidate_duplicates=0,initial_candidates=5,max_candidates=12,old_scientific_kernels_changed=False,execution_enabled=False,
      static_input_shapes_verified=True,saved_baseline_quotes=len(baseline),quote_scope='Existing selected rows only, no new min/G/ratio/statistic',preparation_stop=True))
    save(OUT/'PREPARATION_STOP.json',dict(status='AWAITING_EXPLICIT_SCIENTIFIC_EXECUTION_PERMISSION',preparation_complete=True,execution_enabled=False,new_scientific_calls=0))
    print('Static preparation PASS; 12 artificial tests; molecular scientific calls 0; execution disabled.')
if __name__=='__main__':main()
