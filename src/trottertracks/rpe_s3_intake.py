"""Published A/C/D evidence -> V2/S3 readiness. Metadata copies, no simulation."""
from copy import deepcopy
from pathlib import Path

from . import rpe_diagnostic_intake as M
from . import rpe_decoder_statistics as S
from . import compiled_residual as R

B,p,H=M.B,M.p,M.H
ROOT=Path(__file__).resolve().parents[2]
A_RELEASE='169d3316dd008eb98dd9be56fe4ecefc5c1a7d60'
A_DIR='docs/research/h4_pr_s1_s2_rmse_results_20261113/'
A_SCIENCE='c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e'
M1_RELEASE='3e035c0d61386010f59ad7ea792bc7a5cefd18ac'
M1_DIR='docs/research/h4_pr_m1_rmse_results_20261113/'
PREVIOUS=dict(commit=M1_RELEASE,path=M1_DIR+'run01/N_HANDOFF.json')
PHYSICAL_RELEASE='bba150834ca30bdb186fbfd58c55a8d51177b855'
PHYSICAL_PATH='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json'
C_RELEASE='83d219cc68c25540507a5539ebd2da94c29d06b3'
C_DIR='docs/research/h4_structural_kernel_20261113/run02/'
C_SCIENCE='19b95d8b3c88063d2dbae4e5bb47455b52b00903'
D_RELEASE='af6f27d7a6467bb355686421acd067544edcbbd2'
NOMINAL=[64,60,56,52,47,43,38,34,30,25,21,17,12]
NUMERICAL=dict(commit=A_SCIENCE,path='scripts/run_h4_pr_rmse_completion.py',stage='S1_S2')
ADAPTER=dict(module='src/trottertracks/h4_pr_rmse_completion.py',previous=PREVIOUS,
    B='3216d27a69ac8532d2a22e78f5f94491d202d238',power_SOURCE='e417ffc9f52b13c18f5493193ce510d858664259',
    scalar_SOURCE='7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e',
    law=dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',path='src/trotterlib/rte.py'))


def accept_a(config, packet, previous, result, physical, *, store):
    """Associate all 13 copied means with their original source records."""
    same=H.same
    M.validate_completion_packet(config,packet,previous,numerical_source=NUMERICAL,
        adapter_source=ADAPTER,previous_source=PREVIOUS)
    for key,value in dict(configuration_id=config['id'],N_packet_source=dict(commit=A_RELEASE,path=A_DIR+'run01/N_HANDOFF.json'),
            binding=config['spec']['A_input']['design']['binding'],profile=config['spec']['profile'],policy=packet['policy'],
            conditions=config['spec']['A_input'],
            sufficient_N_qualified=False,global_RMSE_bound=None,common_effective_H_qualified=False,
            numerical_allowances_certified=False,saved_values_reanalyzed=False,new_science_calls=0).items():
        same(physical[key],value,'A_PHYSICAL_PACKET_SCOPE_SOURCE_OR_QUALIFICATION:'+key)
    same(result['source'],A_SCIENCE,'A_SCIENCE_SOURCE_CHANGED')
    same(result['stage'],'S1_S2','A_STAGE_CHANGED')
    same(result['global_RMSE_bound'],None,'A_GLOBAL_BOUND_NOT_NULL')
    new=packet['diagnostic_scope']['newly_evaluated_rounds']
    same(sorted(r['m'] for r in result['diagnostics']),new,'A_RESULT_DIAGNOSTIC_SCOPE_CHANGED')
    same(sorted(r['identity']['m'] for r in result['model_records']),new,'A_RESULT_MODEL_SCOPE_CHANGED')
    same([r['m'] for r in physical['rounds']],list(range(13)),'A_ALL_PHYSICAL_ROUNDS_REQUIRED')
    models={r['identity']['m']:r for r in result['model_records']}
    diag={r['m']:r['diagnostics'] for r in result['diagnostics']}
    for r,n,canonical in zip(physical['rounds'],packet['rounds'],config['rounds']):
        m=r['m'];d=r['diagnostics']
        for key,value in dict(physical_candidate=canonical['physical_candidate'],structure=canonical['structure'],
                canonical_candidate_id=canonical['candidate']['id'],source_request_id=n['source_request_id'],
                law_binding=canonical['law_binding'],scientific_source=n['diagnostic_source']['commit'],
                nominal_N_X=n['nominal_N_X'],nominal_N_Y=n['nominal_N_Y'],sufficient_N_X=None,sufficient_N_Y=None,
                energy_bias_ha=None,energy_RMSE_bound_ha=None,qualification='SAVED_LOCAL_EMPIRICAL_ONLY',
                scientific_values_recomputed=False).items():
            same(r[key],value,'A_ROUND_PHYSICAL_SOURCE_OR_QUALIFICATION:'+key)
        for axis in p.AXES:same(n['nominal_N_'+axis],NOMINAL[m],'A_NOMINAL_SAVED_VALUE_CHANGED')
        ref=r['diagnostics_source'];source_record=store.read(dict(commit=ref['commit'],path=ref['path']))
        same(ref['commit'],H.A_COMMIT if m in (0,6) else M1_RELEASE if m==1 else A_RELEASE,
            'A_ORIGINAL_DIAGNOSTIC_CARRIER_CHANGED')
        for key in ref['selector']:source_record=source_record[key]
        same(d,source_record,'A_DIAGNOSTIC_VALUE_NOT_ORIGINAL_COPY')
        same(d['candidate'],canonical['physical_candidate'],'A_DIAGNOSTIC_PHYSICAL_ID_MISMATCH')
        same(d['rigorous_error_bound'],False,'A_EMPIRICAL_BOUND_PROMOTED')
        for key in ('raw_error','law_error','ground_identification_error','RMSE_bound_ha'):
            same(d['actual_implementation'][key],None,'A_UNKNOWN_ERROR_FILLED')
        for key in ('raw','corrected','exact_tail','reference','total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias'):
            if set(d['signal'][key])!={'real','imag'}:raise ValueError('A_REQUIRED_COMPLEX_MEAN_OR_BIAS_MISSING')
        same(d['signal']['log_B'],d['normalization']['log_B'],'A_NORMALIZATION_ASSOCIATION_MISMATCH')
        if m in new:
            model=models[m]
            for key,value in dict(request_id=n['source_request_id'],nominal_N_X=n['nominal_N_X'],nominal_N_Y=n['nominal_N_Y'],
                    status=n['model_status'],signal=d['signal'],sufficient_N_X=None,sufficient_N_Y=None,rmse_guaranteed=False).items():
                same(model[key],value,'A_MODEL_PACKET_ASSOCIATION_CHANGED:'+key)
            same(model['identity']['candidate'],canonical['physical_candidate'],'A_MODEL_PHYSICAL_CANDIDATE_CHANGED')
            for key,value in dict(binding=physical['binding'],profile=physical['profile'],structure=canonical['structure'],
                    q_RPE=canonical['q_RPE'],q_PF=canonical['q_PF']).items():
                same(model['identity'][key],value,'A_MODEL_IDENTITY_CHANGED:'+key)
            same(d,diag[m],'A_PHYSICAL_RESULT_DIAGNOSTIC_DIFFERS')
        if m in (3,4):
            same(d['norm_evidence']['kind'],'SOURCE_GATE_LIMIT_NOT_OBSERVED','A_M3_M4_NORM_EVIDENCE_CHANGED')
            same(d['norm_evidence']['observed_error'],None,'A_M3_M4_NORM_OBSERVATION_FILLED')
            same(d['signal']['allowance_evidence'].get('exact_tail_norm_error'),None,'A_M3_M4_MISSING_NORM_FILLED')
    same(physical['target']['ground_state_certified'],False,'A_GROUND_QUALIFICATION_PROMOTED')
    same(physical['target']['truth_proxy_adopted_for_new_simulation'],False,'A_TARGET_PROXY_AUTO_ADOPTED')
    same(physical['target']['energy_shift'],0,'A_TARGET_ENERGY_SHIFT_CHANGED')
    truth_ref=physical['target']['truth_proxy_source'];truth=store.read(dict(commit=truth_ref['commit'],path=truth_ref['path']))
    for key in truth_ref['selector']:truth=truth[key]
    same(truth['ideal_numerical_pilot']['binding'],physical['binding'],'A_TARGET_STATE_BINDING_CHANGED')
    for key in ('reference_solver_energy_ha','state_residual_ha','ground_state_certified'):
        same(physical['target'][key],truth[key],'A_TARGET_SAVED_EVIDENCE_CHANGED:'+key)
    return dict(configuration_id=config['id'],accepted=True,new_scope=new,preserved_rounds=[0,1,6],
        nominal_N_XY=deepcopy(NOMINAL),raw_mean_rounds=13,nominal_axis_cells=26,observed_norm_missing_rounds=[3,4],
        sufficient_N=None,global_RMSE_bound=None,scientific_qualification='UNQUALIFIED',G=None)


def accept_k1(config, k0, k1):
    """C's native saved K0/K1 -> existing B structural diagnostic envelope."""
    same=H.same;B._validate(config)
    for key,value in dict(status='PASS',execution_source=C_SCIENCE,exact_local_and_terminal_parity=True).items():
        same(k0[key],value,'C_K0_SOURCE_OR_PARITY_MISMATCH:'+key)
    same(k0['B_binding']['configuration_id'],config['id'],'C_K0_CONFIGURATION_MISMATCH')
    same(k0['B_binding']['configuration_source']['commit'],M.S0.B_RELEASE,'C_K0_CONFIGURATION_SOURCE_MISMATCH')
    same(k0['B_binding']['configuration_source']['path'],M.S0.BDIR+'CONFIGURATION_V2.json','C_K0_CONFIGURATION_PATH_MISMATCH')
    same(k0['preparation_hash'],'b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7','C_K0_PREPARATION_CHANGED')
    for key,value in dict(source=C_SCIENCE,status='FOUR_IDEAL_FINITE_LAW_MEAN_STRUCTURAL_PRICES_COMPLETE',
            quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',
            mean_compiled_C=None,qualified_for_compiled_price=False,compiled_price_records_produced=0,full_rpe_resources=None).items():
        same(k1[key],value,'C_K1_QUANTITY_SOURCE_OR_COMPILED_QUALIFICATION:'+key)
    source=dict(commit=C_RELEASE,path=C_DIR+'K1.json',execution_source=C_SCIENCE)
    rows=[]
    for saved in k1['values']:
        if saved['axis'] not in ('cosine','sine') or saved['m'] not in (0,6):raise ValueError('C_FIXED_K1_DOMAIN_CHANGED')
        m=saved['m'];axis={'cosine':'X','sine':'Y'}[saved['axis']];row=config['rounds'][m]
        for key,value in dict(candidate_id=row['physical_candidate']['id'],q_PF=row['q_PF'],R_outer=row['R_outer'],
                identity=row['identity'],law_binding=row['law_binding'],quantity=k1['quantity'],
                expectation='IDEAL_FINITE_K_SOURCE_LAW',kernel_source=C_SCIENCE,
                mean_compiled_C=None,compiled_price_qualified=False,sampler_ideal_distance=None,
                compiler_residual=None,joint_covariance=None).items():
            same(saved[key],value,'C_K1_IDENTITY_LAW_SOURCE_OR_QUALIFICATION:'+key)
        S._finite(saved['mean_S'],'saved structural mean')
        interval=saved['numerical_enclosure']
        if len(interval)!=2 or not interval[0]<=saved['mean_S']<=interval[1]:
            raise ValueError('C_SAVED_NUMERICAL_INTERVAL_MISMATCH')
        rows.append(dict(m=m,axis=axis,identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
            physical_candidate_id=row['physical_candidate']['id'],profile=deepcopy(config['spec']['profile']),
            source=deepcopy(source),quantity=M.STRUCTURAL,original_quantity=k1['quantity'],
            expectation=saved['expectation'],mean_S=saved['mean_S'],
            law_context=dict(law=saved['expectation'],N_law=config['spec']['law']['model'],actual_sampler_distance=None),
            checks=dict(K0='PASS',source=dict(commit=C_RELEASE,path=C_DIR+'K0.json'),
                preparation_hash=k0['preparation_hash'],parity=deepcopy(k0['all_legal_continuations'])),
            uncertainty=dict(numerical_interval=deepcopy(saved['numerical_enclosure']),
                probability_factorization_relative_bound=saved['source_probability_factorization_relative_bound'],
                numerical_contract=dict(commit=C_SCIENCE,path='docs/research/h4_structural_kernel_20261113/NUMERICAL_CONTRACT.md'),
                systematic=None,joint_covariance=None,total_CI=None)))
    if {(r['m'],r['axis']) for r in rows}!={(m,a) for m in (0,6) for a in p.AXES} or len(rows)!=4:
        raise ValueError('C_ALL_FOUR_STRUCTURAL_CELLS_REQUIRED')
    packet=dict(configuration_id=config['id'],source=source,kernel_source=C_SCIENCE,stage='K1',K0_status='PASS',
        quantity=M.STRUCTURAL,original_quantity=k1['quantity'],qualified_for_candidate=False,
        qualified_for_compiled_price=False,mean_compiled_C=None,mean_C_minus_S=None,rows=rows,
        K0_evidence=dict(source=dict(commit=C_RELEASE,path=C_DIR+'K0.json'),
            state_representation=deepcopy(k0['state_representation']),
            probability_contract=deepcopy(k0['probability_contract']),
            law_metadata_parity=deepcopy(k0['law_metadata_parity']),
            kernel_npz_sha256=k0['kernel_npz_sha256']))
    return M.accept_structural_packet(config,packet)


def residual_price(config, structural, residual, *, cohort, source):
    """One supplied C-S component through D's unchanged combine/record APIs.

    Nonnull paired estimates need same-path proof metadata. This function does
    not estimate a residual mean, variance, fit or scientific qualification.
    """
    if residual is None:
        row=config['rounds'][structural['m']]
        for key,value in dict(identity=row['identity'],law_binding=row['law_binding'],
                physical_candidate_id=row['physical_candidate']['id'],profile=config['spec']['profile']).items():
            H.same(structural[key],value,'RESIDUAL_MISSING_PRICE_IDENTITY_MISMATCH:'+key)
        if structural['axis'] not in p.AXES or not source:raise ValueError('Price axis/SOURCE required')
        combined=dict(m=structural['m'],axis=structural['axis'],mean=None,status='MISSING_COMPONENT',
            reasons=['MEAN_D_MISSING','RESIDUAL_METHOD_AND_LAW_NOT_SELECTED'],qualified_for_candidate=False,
            components=dict(structural=dict(mean_S=structural['mean_S'],source=deepcopy(structural['source'])),residual=None))
        return dict(component=combined,price_record=R.b_price_record(combined,config=config,cohort=cohort,source=source),
            residual_source=dict(commit=D_RELEASE,path='src/trottertracks/compiled_residual.py'),qualified_for_compiled_price=False)
    if isinstance(residual,(list,tuple)):
        if len(residual)!=1:raise ValueError('Only one residual component is allowed')
        residual=residual[0]
    if not source or not residual.get('source'):raise ValueError('Residual SOURCE required')
    if residual.get('mode')=='PAIRED_DIRECT_FINITE_COHORT' and residual.get('mean_D') is not None:
        proof=residual.get('paired_paths') or []
        if len(proof)!=residual.get('n_paths') or residual.get('role') not in R.ROLES:
            raise ValueError('Paired C-S requires path references and a single role')
        seen=set()
        for path in proof:
            key=(path.get('index'),path.get('event_digest'))
            if (key in seen or not path.get('event_digest') or not path.get('structural_source') or not path.get('compiled_source')
                    or path.get('S_event_digest')!=path['event_digest'] or path.get('C_event_digest')!=path['event_digest']
                    or path.get('axis')!=structural['axis'] or path.get('candidate_id')!=structural['physical_candidate_id']
                    or path.get('compiler')!=config['spec']['profile']['compiler'] or path.get('forced_conditional') is not False):
                raise ValueError('Residual same-path/axis/compiler/normal-cohort proof mismatch')
            seen.add(key)
    combined=R.combine(structural,residual,config=config)
    # D keeps a diagnostic sum when laws differ. B keeps a missing price for
    # that case; it never pools ideal S with an unqualified sampler-law D.
    if 'S_AND_D_LAW_DIFFER_OR_UNDECLARED' in combined['reasons']:
        combined=dict(combined,mean=None,status='UNQUALIFIED_LAW_MISMATCH_PRICE_WITHHELD')
    return dict(component=combined,price_record=R.b_price_record(combined,config=config,cohort=cohort,source=source),
        residual_source=dict(commit=D_RELEASE,path='src/trottertracks/compiled_residual.py'),qualified_for_compiled_price=False)


def prepare_statistics(config, packet, physical, models):
    """Connect copied A evidence to S.prepare; never materialize probabilities."""
    target=dict(binding=deepcopy(config['spec']['binding']),quantity='ORIGINAL_DF_H_TARGET_ENERGY',energy_shift_ha=0,
        source=deepcopy(physical['target']['truth_proxy_source']),energy_ha=None,
        saved_truth_proxy=deepcopy(physical['target']),truth_proxy_adopted=False)
    bym={r['identity']['m']:r for r in models}
    if len(bym)!=13 or len(models)!=13:raise ValueError('All thirteen original A model records required')
    means=[];shots=[]
    for row,n in zip(physical['rounds'],packet['rounds']):
        m=row['m'];d=row['diagnostics'];model=bym[m]
        H.same(model['signal'],d['signal'],'A_MODEL_MEAN_AND_DIAGNOSTIC_SOURCE_DIFFERS')
        for axis in p.AXES:H.same(model['nominal_N_'+axis],n['nominal_N_'+axis],'A_MODEL_NOMINAL_DIFFERS')
        mean=S.mean_from_a(config,m,model,source_request_id=n['source_request_id'],
            law_binding=config['rounds'][m]['round_law_definition']['A_declaration'],source=row['diagnostics_source'],
            target=target,qualification='SAVED_LOCAL_EMPIRICAL_ONLY',
            uncertainty=dict(allowance=deepcopy(d['allowance']),norm_evidence=deepcopy(d.get('norm_evidence')),
                reference_pair=deepcopy(d['reference_pair']),actual_implementation=deepcopy(d['actual_implementation']),
                rigorous_error_bound=d['rigorous_error_bound']))
        mean.update(normalization=deepcopy(d['normalization']),scientific_source=row['scientific_source'],
            mean_provenance=deepcopy(row['mean_provenance']),norm_evidence=deepcopy(d.get('norm_evidence')))
        means.append(mean);shots.append(dict(m=m,N_X=n['nominal_N_X'],N_Y=n['nominal_N_Y']))
    prepared=S.prepare(config,means,shots,target=target,
        shot_policy=dict(meaning='NOMINAL',source=dict(commit=A_RELEASE,path=A_DIR+'run01/N_HANDOFF.json')))
    prepared.update(scientific_execution_ready=False,probabilities_evaluated=False,
        S3_conditions=dict(count_caller=dict(status='NOT_ADOPTED',available=S.COUNT_RULE),
            independent_binomial=dict(status='CONDITIONAL_IDEAL_LAW_ASSUMPTION',evidence=deepcopy(physical['measurement_contract'])),
            unresolved_loss=dict(status='UNDECIDED',energy_fallback=None),
            target=dict(status='UNDECIDED_SAVED_SOLVER_PROXY_GROUND_UNQUALIFIED',source=physical['target']['truth_proxy_source']),
            PF_finite_K_state_law_errors=dict(status='EMPIRICAL_LOCAL_ONLY_NO_GLOBAL_L2_TRANSFER',
                common_effective_H_qualified=physical['common_effective_H_qualified']),
            Minkowski=dict(status='L2_COMPARISON_COUPLING_TARGET_LOSS_UNDECIDED'),
            MSE_precision=None,confidence=None,loss_support=None,simulation_repetitions=None),
        physical_contract_source=dict(commit=PHYSICAL_RELEASE,path=PHYSICAL_PATH),
        scientific_systematic_errors_zero_filled=False,S3_executed=False,empirical_H4_RMSE=None)
    return prepared


def load_published(root=ROOT):
    """Read fixed published packets only, preserving the old acceptance chain."""
    store=p.SavedArtifacts(root)
    def read(commit,path):return store.read(dict(commit=commit,path=path))
    config=read(M.S0.B_RELEASE,M.S0.BDIR+'CONFIGURATION_V2.json')
    old_s0=read(M.S0.A_RELEASE,M.S0.ADIR+'V2_N_HANDOFF.json')
    previous=store.read(PREVIOUS)
    m1_adapter=read(M.A_PREPARATION,M.A_PREFLIGHT)['source']
    M.validate_m1_packet(config,previous,old_s0,numerical_source=previous['source'],adapter_source=m1_adapter)
    packet=read(A_RELEASE,A_DIR+'run01/N_HANDOFF.json')
    result=read(A_RELEASE,A_DIR+'run01/result.json')
    physical=read(PHYSICAL_RELEASE,PHYSICAL_PATH)
    k0=read(C_RELEASE,C_DIR+'K0.json');k1=read(C_RELEASE,C_DIR+'K1.json')
    s0=store.read(dict(commit=H.A_COMMIT,path=H.A_DIR+'result.json'))
    m1=read(M1_RELEASE,M1_DIR+'run01/result.json')
    models=s0['model_records']+[m1['model_record']]+result['model_records']
    return dict(config=config,packet=packet,previous=previous,result=result,physical=physical,k0=k0,k1=k1,models=models),store
