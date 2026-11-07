"""B6b metadata contracts and future accounting adapter; no molecular runs."""
import csv
import hashlib
import io
import json
import math
import subprocess

from . import c1_stage3_accuracy as accounting

B4='07d18bfe1d5145d6f8de026c0d994d5cb1c783c8'
B5='0a8b953aea1be20b110784960763d7f36ebccac8'
C74='64072ba81bb70db1df1b34e94adb08e78926f24b'
BDIR='artifacts/track_d_b4_2_rpe_shots/2026-11-06'
PACKET='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
CDF='a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
LAW='c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5'
SCOPE='single_hadamard_interrogation_without_state_preparation'
METRICS=accounting.METRICS
METHODS=('symmetric_directional','reference')
P0=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,basis_gates=['rz','sx','x','cx'],
        approximation_degree=1.0,qubits_initially_zero=False)
PRICE_SETS=(
 (2,16,'e9d4e2f602d5f9e014507ca7e391719bab017b88','track_d_c6a_paired_price/2026-11-06/run01'),
 (4,16,'e9d4e2f602d5f9e014507ca7e391719bab017b88','track_d_c6a_paired_price/2026-11-06/run01'),
 (8,32,'fed5f458be2f625de4e1b8c98b8cc5da6e551b14','track_d_c6b_q8/2026-11-07/run01_accelerated_completion'),
 (16,16,'86c5c9d48dbddc085f388b6521be8cd92ee49d0d','track_d_c7_1_q16/2026-11-07/run01'),
 (32,8,'acea43987edae0d5eb0c186a89aa8cd7139fbd7b','track_d_c7_3_q32/2026-11-07/run01_accelerated_completion'))


def blob(root,commit,path,bindings=None):
    data=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
    if bindings is not None:bindings[commit+':'+path]=dict(commit=commit,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return data


def binding():
    return dict(original_h=PACKET,cdf=CDF,law_sha256=LAW,geometry='linear H4 / 1 Angstrom',
        basis='STO-3G',sector='Nalpha2_Nbeta2_sector36',frame='original LD2',
        PR='ordered S2/s1/r32/K0',h=.1,state='exact_original_H_ground',state_preparation='EXCLUDED',
        measurement='X=Re(z),Y=Im(z); H/core/(Y:Sdg)/H; bit0:+1,bit1:-1',
        normalization='raw path unitary, attenuation is an ensemble property; not a price multiplier')


def read_b4(root,bindings):
    get=lambda p:json.loads(blob(root,B4,p,bindings))
    handoff=get(BDIR+'/Stage_C_handoff.json');result=get(BDIR+'/run01/result.json')
    stop=get(BDIR+'/run01/STOP.json');source=get(BDIR+'/run01/SOURCE.json')
    axes=list(csv.DictReader(io.StringIO(blob(root,B4,BDIR+'/run01/axis_shots.csv',bindings).decode())))
    if handoff['qualification']!='CONDITIONAL_IDEAL_13_ROUND_PASS' or not stop['mandatory_stop']:
        raise ValueError('B4_FIXED_COMPLETION')
    if (handoff['binding']['original_h'],handoff['binding']['cdf'])!=(PACKET,CDF):raise ValueError('B4_IDENTITY')
    target=handoff['target']
    expected=dict(energy_ha=.00159,base_time=.1,beta_total=math.pi/6,beta_stat=math.pi/12,
        alpha_axis=.01/26,alpha_total=.01,maximum_round_index=12,energy_window_ha=[-10.,10.])
    if target!=expected:raise ValueError('B4_FIXED_TARGET')
    if len(axes)!=26 or len(handoff['rows'])!=13 or handoff['total_shots']!=14738:raise ValueError('B4_FIXED_COUNTS')
    for m,r in enumerate(handoff['rows']):
        if r['m']!=m or r['q']!=2**m or not math.isclose(r['time'],.1*2**m,rel_tol=0,abs_tol=1e-13):raise ValueError('B4_SCHEDULE')
        for a in ('X','Y'):
            matches=[v for v in axes if int(v['m'])==m and v['axis']==a]
            if len(matches)!=1:raise ValueError('B4_AXIS_COVERAGE')
            # Exact field spelling is the existing public axis_shots schema.
            if int(matches[0]['N'])!=r['N_'+a]:raise ValueError('B4_CSV_HANDOFF_N')
    # This identifies the existing law descriptor, not its ideal/actual distance.
    law_path='artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json'
    data=blob(root,'3649b75af191916daf44abb91e14d34c43b83e8f',law_path,bindings)
    if hashlib.sha256(data).hexdigest()!=LAW:raise ValueError('LAW_DESCRIPTOR_IDENTITY')
    law=json.loads(data)
    if handoff['normalization']['lambda_R']!=law['lambda_R'] or handoff['normalization']['r']!=32:
        raise ValueError('NORMALIZATION_LAW_BINDING')
    if (handoff['measurement']!=binding()['measurement'] or handoff['state_preparation']!='EXCLUDED'
        or handoff['energy']['decoder']!='consecutive_dyadic_nearest_branch_principal_energy_v1'
        or handoff['energy']['energy_phase_convention']!='phi=-h*E; E=-phi/h'
        or handoff['energy']['state_model']!='exact_original_H_ground_state_oracle'):
        raise ValueError('MEASUREMENT_STATE_DECODER')
    return handoff,dict(status='SAVED_B4_METADATA_MATCHED',target=target,binding=binding(),rows=handoff['rows'],
        stored_total_shots=14738,N_recomputed=False,actual_N=None,actual_qualification='UNQUALIFIED',
        execution_source_local=handoff['execution_source_local_commit'],published_source=handoff['published_source_commit'],
        source_tree=handoff['published_source_tree'],measurement=handoff['measurement'],
        decoder=handoff['energy']['decoder'],energy_phase_convention=handoff['energy']['energy_phase_convention'])


def common_N(premises,*,lane='ideal'):
    """Assess declared model conditions; booleans do not prove external facts."""
    fields=('same_V','actual_adjoint','vacuum_eigenray_with_chi','same_initial_system_state',
        'same_whole_path_law','same_XY_convention','fresh_independent_axis_trials','same_target_decoder')
    missing=[k for k in fields if premises.get(k) is not True]
    if missing:return dict(status='COMMON_N_UNQUALIFIED',missing=missing,common_N=False)
    if lane=='ideal':
        if premises.get('law_is_B4_ideal_product') is not True:return dict(status='SAME_LAW_BUT_B4_N_TRANSFER_UNQUALIFIED',common_N=False)
        return dict(status='CONDITIONAL_IDEAL_COMMON_N',common_N=True,actual_success_guaranteed=False,
                    qualification='conditional on stated all-path ideal blueprint and ideal trial law')
    if lane not in ('implemented','actual'):raise ValueError('LANE')
    needed=('law_bias_and_concentration_qualified','both_measurement_error_budgets_qualified',
            'saved_N_radius_phase_coordinate_conditions_checked')
    missing=[k for k in needed if premises.get(k) is not True]
    return dict(status='ACTUAL_COMMON_N_UNQUALIFIED' if missing else 'CONDITIONAL_IMPLEMENTED_COMMON_N',
        common_N=not missing,missing=missing,actual_success_guaranteed=False)


def price_catalog(root,bindings):
    catalog=[]
    for q,n,commit,directory in PRICE_SETS:
        path='artifacts/'+directory
        rows=json.loads(blob(root,commit,path+'/prices.json',bindings))
        rows=[r for r in rows if r['q']==q]
        expected={(i,m,a) for i in range(n) for m in METHODS for a in ('X','Y')}
        if {(r['sample'],r['method'],r['axis']) for r in rows}!=expected or len(rows)!=4*n:raise ValueError('PRICE_FIXED_COHORT')
        for r in rows:
            req=r['request']
            if (req['input_packet_sha256'],req['endpoint'],req['law_sha256'],req['cut_after_complete_step'])!=(PACKET,CDF,LAW,q//2):
                raise ValueError('PRICE_IDENTITY_OR_CUT')
            if (r['cost_scope']!=SCOPE or r['compiler']!=P0 or r['topology']!='free' or r['state_preparation']!='EXCLUDED'
                or r['precision']!=dict(component_abs=1e-10,observable_frobenius=2e-10,rtol=0,global_phase_fit=False)):
                raise ValueError('PRICE_SCOPE_OR_COMPILER')
            if r['peak_qubits']!=(17 if r['method']=='reference' else 9):raise ValueError('FIXED_WIDTH_SCOPE')
        for leaf in ('SOURCE.json','result.json','STOP.json','paired_statistics.json'):
            blob(root,commit,path+'/'+leaf,bindings)
        catalog.append(dict(q=q,n_whole_paths=n,cut=dict(kind='complete_step',after=q//2),commit=commit,
            price_path=path+'/prices.json',classification='DIRECT_COMPILED_FINITE_SAMPLE',
            population_expectation_qualified=False,XY_independent_samples=False,compiler=P0,scope=SCOPE,
            peak_width=dict(symmetric_directional=9,reference=17),width_is_additive_cost=False,
            covariance='same whole-path paired method/resource/axis covariance retained in saved paired_statistics',
            sampling_sources=sorted({r.get('sampling_source',r.get('provenance',{}).get('sampling_source',r['execution_source'])) for r in rows}),
            compile_post_sources=sorted({r.get('compile_post_source',r.get('provenance',{}).get('compile_and_postcheck_source',r['execution_source'])) for r in rows})))
    # q1 is a saved finite cohort with an event cut, not a complete-step model.
    commit='3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0'
    path='artifacts/track_d_h4_paired32_reference/2026-11-06/run01'
    for leaf in ('SOURCE.json','result.json','STOP.json','paired_resources.csv'):
        blob(root,commit,path+'/'+leaf,bindings)
    catalog.insert(0,dict(q=1,n_whole_paths=32,cut=dict(kind='complete_event_within_step',after=16),commit=commit,
        price_path=path+'/paired_resources.csv',classification='DIRECT_COMPILED_FINITE_SAMPLE',
        complete_step_proxy_transfer='SCOPE_MISMATCH',population_expectation_qualified=False,
        cohort='near_practical_PR_r32_fixed32_v1 (saved, not fresh RPE shot transcript)',
        XY_independent_samples=False,scope=SCOPE,compiler=P0,
        peak_width=dict(symmetric_directional=9,reference=17),width_is_additive_cost=False,
        sampling_sources=['4df8eca29c59ffd55e2dd898f05da16bb246ddbe'],
        compile_post_sources=['2dd012dc77dcc877d5918c7d212819eb79b97557']))
    return catalog


def coverage(handoff,catalog,*,c74_metadata=None):
    byq={r['q']:r for r in catalog};rows=[]
    for b in handoff['rows']:
        q=b['q'];direct=byq.get(q)
        rows.append(dict(m=b['m'],q=q,N_X=b['N_X'],N_Y=b['N_Y'],
            classification=direct['classification'] if direct else ('EXPLORATORY_PROXY_ONLY' if c74_metadata else 'PRICE_UNQUALIFIED'),
            direct_evidence=direct,proxy_version=c74_metadata if q>=64 else None,
            frozen_proxy_alternative=None if q==1 else dict(classification='FROZEN_PROXY_AVAILABLE',
                commit='e9d4e2f602d5f9e014507ca7e391719bab017b88',
                path='artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json',
                population_or_long_q_guarantee=False),
            expectation_qualification='UNQUALIFIED',formal_total_input_ready=False,
            missing=['qualified expectation model or explicit conditional proxy adoption']+
                (['q1 event-cut domain handled separately'] if q==1 else [])+
                (['independent long-q price domain evidence'] if q>=64 else [])))
    return dict(rows=rows,rounds=13,axes=26,full_H4_total_evaluated=False,total=None,
        actual_qualification_required_for_ideal_accounting=False,
        note='future explicitly adopted proxy totals may be conditional estimates; actual success qualification is separate')


def aggregate_registered(handoff,price_rows,premises,*,claim='ideal_conditional',covariance_by_metric=None):
    """Future formal runner hook. Current runner never calls this on H4 prices.

    The fixed B4 schedule has NX=NY. Do not synthesize a new shot formula or
    populate legacy PF candidates with invented fields to force reuse.
    Covariance inputs describe estimates of expected prices, not single-path
    sample covariance, physical spending variance or quantum outcome variance.
    """
    if claim!='ideal_conditional':raise ValueError('ACTUAL_TOTAL_REQUIRES_SEPARATE_QUALIFICATION')
    if handoff['binding']['original_h']!=PACKET or handoff['binding']['cdf']!=CDF:raise ValueError('SHOT_BINDING')
    if len(handoff['rows'])!=13 or any(r['m']!=m or r['q']!=2**m for m,r in enumerate(handoff['rows'])):raise ValueError('SHOT_SCHEDULE')
    assessment=common_N(premises,lane='ideal')
    if not assessment['common_N']:raise ValueError('COMMON_N_UNQUALIFIED')
    expected={(m,a,method) for m in range(13) for a in ('X','Y') for method in METHODS}
    keys=[(r['m'],r['axis'],r['method']) for r in price_rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('COMPLETE_13_ROUND_PRICE_COVERAGE_REQUIRED')
    for r in price_rows:
        if r['q']!=2**r['m']:raise ValueError('PRICE_Q_ROUND')
        if not r.get('source') or r.get('estimate_unit')!='whole_path_one_interrogation':raise ValueError('COST_SOURCE_OR_SAMPLE_UNIT')
        if r['binding']!=binding() or r['scope']!=SCOPE or r['compiler']!=P0:raise ValueError('COST_IDENTITY_SCOPE_COMPILER')
        if r.get('expectation_basis') not in ('qualified_expectation','explicit_conditional_model'):raise ValueError('FINITE_SAMPLE_OR_UNKNOWN_IS_NOT_EXPECTED_PRICE')
        if r['expectation_basis']=='explicit_conditional_model' and not r.get('model_assumptions'):raise ValueError('MODEL_ASSUMPTIONS_REQUIRED')
        if r.get('mean') is None or any(r['mean'].get(k) is None for k in METRICS):raise ValueError('MISSING_EXPECTED_PRICE')
        if set(r['mean'])!=set(METRICS):raise ValueError('ADDITIVE_METRICS_ONLY_NO_QUBIT_SUM')
    plans=[]
    for r in handoff['rows']:
        if any(type(r.get('N_'+a)) is not int or r['N_'+a]<=0 for a in ('X','Y')):raise ValueError('MISSING_VALID_SHOTS')
        if r['N_X']!=r['N_Y']:raise ValueError('FIXED_B4_EQUAL_AXIS_N_REQUIRED')
        plans.append(dict(m=r['m'],shots=r['N_X'],target_id='B6b-fixed-B4-original-H'))
    totals={}
    for method in METHODS:
        costs=[dict(m=r['m'],axis=r['axis'],target_id='B6b-fixed-B4-original-H',scope=SCOPE,
            mean=r['mean'],long_round_coverage=True,simultaneous_interval_qualified=False,
            lower=None,upper=None) for r in price_rows if r['method']==method]
        totals[method]=accounting.aggregate_totals(dict(same_accuracy=True),plans,costs,cost_scope=SCOPE,expected_scope=SCOPE)
    delta={g:totals['reference']['totals'][g]-totals['symmetric_directional']['totals'][g] for g in METRICS}
    # Optional full covariance is over (method, round, axis), never independent
    # by default. Shared calibration and paired methods may be correlated.
    uncertainty=None
    if covariance_by_metric is not None:
        import numpy as np
        order=[(method,m,a) for method in METHODS for m in range(13) for a in ('X','Y')]
        weights=np.array([(-1 if method==METHODS[0] else 1)*handoff['rows'][m]['N_'+a] for method,m,a in order],float)
        uncertainty={}
        for g in METRICS:
            c=np.asarray(covariance_by_metric[g],float)
            if c.shape!=(52,52) or not np.isfinite(c).all() or not np.allclose(c,c.T,rtol=0,atol=1e-12):raise ValueError('FULL_COVARIANCE_SCHEMA')
            v=float(weights@c@weights)
            if v < -1e-8:raise ValueError('NEGATIVE_TOTAL_VARIANCE')
            uncertainty[g]=max(0.,v)
    return dict(status='CONDITIONAL_MODEL_ACCOUNTING_ONLY',methods=totals,paired_delta=delta,
        paired_total_covariance_diagonal=uncertainty,peak_qubits='separate max-width constraint; never multiplied by N',
        covariance_missing_is_zero=False,depth_is_hardware_runtime=False,attenuation_price_multiplier=False,
        actual_success_guaranteed=False)
