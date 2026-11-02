"""Quote fixed evidence and inspect control availability; no numerical dispatcher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import h246_order_cost_analysis as prior

ROOT=Path(__file__).resolve().parents[1]
BASE='d2536c4f959bfb4edd7d2deeceb155585fb53eb0'
ADOPTED=prior.old6.ADOPTED
SAVED=prior.REFERENCE_SOURCE
CONFIRM='3eca46148070ba4d4b0f5815ef45943407cc959e'
OLD_H6='3cccfcb2051dacf29e91473e2e1152acef1574bd'
INDEX='6dafed86cc82149d55b4e6395382953d5f3631d7'
ADIR=prior.old6.h4.ADOPTED_DIR
SDIR=prior.old6.h4.SAVED_DIR
RDIR='artifacts/resource_applicability/track_a_h246_order_cost_analysis_run01/2026-10-24/'
CDIR='artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/'
IDIR='docs/data_collection/2026-10-23_three_environment_integration/'
OUT=ROOT/'artifacts/resource_applicability/track_a_h246_reversal_evidence_controls/2026-10-24'
ASSETS=ROOT.parent/'h246-reversal-evidence-controls-assets-20261024'
require,read,sha,digest=prior.require,prior.read,prior.sha,prior.digest
SOURCE_FILES=('scripts/h246_reversal_evidence.py','tests/test_h246_reversal_evidence.py')


def classify_control(a,b):
    """Only metadata comparisons; never compare or calculate cost values."""
    fields=('system','method','prefix','q','rank','rank_policy','basis','snapshot','hamiltonian','state',
            'compiler','native','unit','control','T','P','epsilon','cohort')
    require(a['formula']=='2nd' and b['formula']=='4th','STANDARD_S2_YOSHIDA4_ONLY')
    require(all(a[k]==b[k] for k in fields),'SAME_Q_PARTITION_AND_INPUT')
    if a['method']=='PR':
        require(all(a[k]==b[k] for k in ('K','r_policy','r_value')),'RTE_LAW_SCOPE')
    require(a['reference']==b['reference'],'WITHIN_METHOD_REFERENCE')
    require(a['u']==b['u'],'ALLOWANCE_POLICY')
    complete=all(x['status']=='ELIGIBLE' and x['G'] is not None and
                 all(x['N'][axis] is not None and x['C'][axis] is not None for axis in ('cosine','sine')) for x in (a,b))
    return dict(status='COMPLETE_SAVED_ELIGIBLE_PAIR' if complete else 'SAVED_INELIGIBILITY_NO_COST_RATIO',
                complete_saved_pair=complete,new_acquisition_required=False,new_comparison_computed=False,
                causal_order_only_identified=False,cohorts_pooled=False,
                same_q_does_not_equal_same_piece_allocation=True)


def confirmation_matches(target,record):
    """Candidate-law matching only, not endpoint-policy or statistical certification."""
    keys=('system','method','formula','prefix','q','K','r_policy','r_value')
    return record.get('independent') is True and record.get('cohort')!='EXPLORATION' and all(
        target.get(k)==record.get(k) for k in keys)


def reject_execution(execute=False):
    require(execute is False,'THIS_METADATA_TOOL_HAS_NO_SCIENCE_EXECUTION_PATH')


def public(name,value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    require(all(s not in text for s in ('/home/','/usr/','/tmp/','/opt/')),'NO_PRIVATE_ABSOLUTE_PATH')
    (OUT/name).write_text(text)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def collect(source):
    reject_execution();ASSETS.mkdir(exist_ok=False);OUT.mkdir(parents=True,exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    catalog=[];protected={}
    def protect(p,expected=None):
        p=Path(p).resolve();s=sha(p);require(expected is None or s==expected,'FIXED_INPUT_SHA')
        if str(p) in protected:require(protected[str(p)]==s,'INPUT_REOBSERVATION_CHANGED')
        protected[str(p)]=s
    def fixed(cm,path,expected=None):
        p=ASSETS/'fixed'/cm/path
        if p.exists():raw=p.read_bytes();protect(p,expected)
        else:
            raw=prior.old.recovery.git_bytes(cm,path);p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(raw);protect(p,expected)
            catalog.append(dict(commit=cm,path=path,sha256=sha(p),bytes=len(raw)))
        return json.loads(gzip.decompress(raw)) if path.endswith('.gz') else json.loads(raw) if path.endswith('.json') else raw.decode()
    for rel in SOURCE_FILES:
        require((ROOT/rel).read_bytes()==prior.old.recovery.git_bytes(source,rel),'COMMITTED_METADATA_SOURCE')
        protect(ROOT/rel)
    cfgp=prior.ASSETS/'ANALYSIS_contract_PRIVATE.json';cfg=read(cfgp);protect(cfgp)
    require(digest(cfg)=='2478152414dce3c468e48c17e04d0addf3143794190be686593bc94eae1708e8','PRIOR_ACCEPTED_CONTRACT')
    for row in cfg['files']:protect(row['path'],row['sha256'])
    inputs=read(cfg['manifest'])
    for p in Path(cfg['run_root']).rglob('*'):
        if p.is_file():protect(p)
    for name in ('approval_run01_PRIVATE.json','grant_run01_PRIVATE.json'):protect(prior.ASSETS/name)
    # Known publication files only; no projects/tmp/history-wide scans.
    patterns=['artifacts/resource_applicability/track_a_h246_order_cost_analysis*',
              'artifacts/resource_applicability/track_a_h468_paper*',
              'artifacts/resource_applicability/track_a_mechanism_figure2*']
    paths=subprocess.check_output(['git','ls-files','-z','--']+patterns,cwd=ROOT,text=True).split('\0')
    for p in paths:
        if p:protect(ROOT/p)
    # Hash baseline precedes all new evidence extraction and document edits.
    (OUT/'cache/preservation_before_PRIVATE.json').write_text(json.dumps(protected,indent=2)+'\n')
    result=fixed(BASE,RDIR+'analysis.json','3614edf7afb06891428b48a0a8969a56b7bf699b3647cf7a1cdb3cd0df3ab6c0')
    for name in ('final_verification.json','native_completion.json','REPORT.md'):fixed(BASE,RDIR+name)
    dataset=fixed(ADOPTED,ADIR+'dataset.json.gz')
    fullcat=fixed(ADOPTED,ADIR+'source_catalog.json')
    evaluation=fixed(SAVED,SDIR+'evaluation.json')
    for name in ('input_provenance.json','end_audit.json'):fixed(SAVED,SDIR+name)
    for name in ('comparisons_within_formula_45.json','precision_rankings_15.json'):fixed(ADOPTED,ADIR+name)
    h4cfg=read(ROOT.parent/'h4-four-method-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json')
    pp=next(r['path'] for r in h4cfg['files'] if r['role']=='PROTECTED:original_analysis_authority_and_results'
            and r['path'].endswith('plan_APPROVED_PRIVATE.json'))
    protect(pp);plan=read(pp)
    refs=list(csv.DictReader(fixed(INDEX,IDIR+'references.csv').splitlines()))
    index=list(csv.DictReader(fixed(INDEX,IDIR+'research_index.csv').splitlines()))
    fixed(INDEX,IDIR+'scientific_profiles.json')
    fixed(INDEX,IDIR+'README.md')
    mirrors={}
    for rid in ('R0017','R0124','R0139'):
        ref=next(r for r in refs if r['ref_id']==rid);mirrors[rid]=fixed(ref['commit'],ref['path'],ref['sha256'])
    confirmations=[]
    contracts=fixed(CONFIRM,CDIR+'stage_contracts.json')['contracts']
    for system in ('H2','H4','H6'):
        r=fixed(CONFIRM,CDIR+'fixed_evaluation/'+system+'_fixed_comparison.json')
        contract=next(x for x in contracts if x['stage']==system+'_CONFIRMATION')
        for g in r['groups']:
            record=dict(g['candidate'],independent=True,cohort='RECONSTRUCTED_INDEPENDENT_CONFIRMATION',
                epsilon=contract['analysis_scope']['epsilon'],statistics=g['statistics'],
                observed_strata=g['observed_strata'],input_binding=contract['input_binding'],
                source=dict(commit=CONFIRM,path=CDIR+'fixed_evaluation/'+system+'_fixed_comparison.json',SOURCE=r['SOURCE']),
                conditional_on_saved_shots=True,complete_historical_seed_noncollision_guaranteed=False,
                parent_comparison_point_status=r['point_status'],parent_comparison_high_formula=r['groups'][1]['candidate']['formula'])
            confirmations.append(record)
    fixed(CONFIRM,CDIR+'end_audit.json')
    fixed(CONFIRM,'docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md')
    prior_h6=fixed(OLD_H6,'artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json')
    prior_receipts=list(csv.DictReader(fixed(OLD_H6,'artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/trajectory_receipts.csv').splitlines()))
    fixed(OLD_H6,'docs/research/track_a_h6_pr_confirmation_science_results_20261019.md')
    fixed('5a448d94ea389ada6e2b2755de6a215ed2034679','scripts/h46_pr_exploration_design.py')
    # Full/PR identities, reference values and source locations are compared as metadata only.
    bindings=inputs['bindings'];main=[];controls=[]
    def group_for(cid):return next(g for g in plan['groups'] if g['record']['candidate']['id']==cid)
    def endpoint(system,method,q,formula,ep=.001):
        prefix={'H4':10,'H6':19}[system] if method=='FULL' else {'H4':3,'H6':5}[system]
        t=dict(system=system,method=method,prefix=prefix,q=q,formula=formula)
        binding=bindings[system]
        if method=='FULL':
            item=next(x for x in dataset['candidates'] if all(x['candidate_status_saved'].get(k)==v for k,v in t.items()))
            target=item['candidate_status_saved'];e=next(e for e in item['endpoints_saved'] if e['epsilon']==ep)
            signal=item['signal_record_saved']['signal'];n=1;stats=None;strata=None
            C={a:e['price'][a] for a in ('cosine','sine')};N=dict(cosine=e['N_cos'],sine=e['N_sin']);G=e['total_RZ_exact']
            u=e['u'];margin=e['log_B_margin'];se=None;pricekind=e['price']['kind']
            ref=fullcat[system+'_identity'];identity=fixed(ref['commit'],ref['path'],ref['sha256'])['binding']
            require(all(binding[k]==identity[k] for k in ('hamiltonian','state','rank','rank_policy')),'FULL_IDENTITY')
            pointer=dict(commit=ADOPTED,path=ADIR+'dataset.json.gz',candidate_id=target['id'],epsilon=ep)
        else:
            saved=next(g for g in evaluation['groups'] if all(g['candidate'].get(k)==v for k,v in t.items()))
            target=saved['candidate'];e=next(e for e in saved['rows'] if e['epsilon']==ep and e['mode']=='USUAL_SIGNED')
            original=group_for(target['id']);signal=original['record']['signal'];stats=saved['unit_statistics'];strata=saved['strata'];n=e['n']
            require(all(binding[k]==original['binding'][k] for k in ('hamiltonian','state','rank','rank_policy')),'PR_IDENTITY')
            require(target['K']==4 and target['r_policy']=='ABS_TIME_PER_OUTER' and target['r_value']==32,'ORIGINAL_RTE_LAW')
            C=dict(cosine=e['mean_cos_RZ'],sine=e['mean_sin_RZ']);N=dict(cosine=e['shot']['N_cos'],sine=e['shot']['N_sin']);G=e['total_RZ']
            u=e['shot']['u'];margin=e['shot']['log_B_margin'];se=e['total_RZ_SE'];pricekind=e['unit_cost_kind']
            pointer=dict(commit=SAVED,path=SDIR+'evaluation.json',candidate_id=target['id'],epsilon=ep)
        return dict(candidate=target['id'],**t,rank=binding['rank'],hamiltonian=binding['hamiltonian'],
            state=binding['state'],snapshot=binding['snapshot'],rank_policy=binding['rank_policy'],basis='STO-3G',
            compiler=prior.old6.h4.COMPILER,native=prior.old6.h4.NATIVE,unit=prior.UNIT,
            control='symmetric_directional',T=.8,P=0,
            cohort='EXPLORATION' if method=='PR' else 'COMMON_PF_DETERMINISTIC',
            K=4 if method=='PR' else None,r_policy='ABS_TIME_PER_OUTER' if method=='PR' else None,
            r_value=32 if method=='PR' else None,epsilon=ep,n=n,status=e['status'],C=C,N=N,G=G,u=u,
            log_B=signal['log_B'],log_B_margin=margin,reference=signal['reference'],bias=signal['total_signed_bias'],
            paired_G_SE=se,saved_unit_statistics=stats,saved_strata=strata,price_kind=pricekind,source=pointer,
            selection='Exploration/post-selection; fixed for accepted analysis, not independent confirmation',
            population_guarantee=False,independent_standard_Y4_reversal_confirmed=False)
    for system in ('H4','H6'):
        for method in ('FULL','PR'):
            for q in (1,2,4,8):
                a,b=[endpoint(system,method,q,f) for f in ('2nd','4th')]
                controls.append(dict(system=system,method=method,q=q,epsilon=.001,S2=a,Yoshida4=b,
                                     **classify_control(a,b)))
            for formula,q in (('2nd',8),('4th',4)):main.append(endpoint(system,method,q,formula))
    weak=[endpoint('H4','PR',8,'2nd',.005),endpoint('H4','PR',4,'4th',.005)]
    for r in main+weak:
        accepted=next(x for x in inputs['rows'] if x['candidate']==r['candidate'] and x['epsilon']==r['epsilon'])
        require(r['G']==accepted['G_saved'] and r['C']==accepted['C'] and r['N']==accepted['N'] and
                r['paired_G_SE']==accepted['uncertainty']['G_SE'],'ACCEPTED_RESULT_JOIN')
    confirmation_scope=[]
    for t in prior.TARGETS:
        if t['method']!='PR':continue
        query=dict(t,K=4,r_policy='ABS_TIME_PER_OUTER',r_value=32)
        matches=[r for r in confirmations if confirmation_matches(query,r)]
        confirmation_scope.append(dict(target=query,main_epsilon=.0005 if t['system']=='H2' else .001,
            independent_candidate_law_matches=matches,
            same_main_epsilon=any(r['epsilon']==(.0005 if t['system']=='H2' else .001) for r in matches),
            matched_candidates_do_not_certify_same_allowance_or_shots=True,
            matched_standard_Y4=bool(matches) if t['formula']=='4th' else None,
            candidate_references_or_seed_history_limits_preserved=True))
    public('primary_and_weak_evidence.json',dict(primary_epsilon=.001,primary_candidates=main,
        H4_weak_epsilon_005_candidates=weak,
        saved_comparisons=[x for x in result['comparisons'] if x['epsilon']==.001 or
                          (x['system']=='H4' and x['method']=='PR' and x['epsilon']==.005) or
                          (x['system']=='H2' and x['epsilon']==.0005)],
        no_new_significance_test_CI_gap_or_ratio=True))
    public('same_q_controls.json',dict(target_epsilon=.001,controls=controls,new_accounting_or_ranking=False,
        primary_different_q_comparison_retained=True,
        never_substitute_ineligible_cost=None))
    public('independent_confirmation_scope.json',dict(candidate_matches=confirmation_scope,
        inspected_reconstructed_records=confirmations,
        older_H6_cell_ids=list(dict.fromkeys(r['cell_id'] for r in prior_receipts)),
        older_H6_scope='Second-order PR prefix4 K2/r4, B0I deterministic baselines; not current PR prefix5 K4/outer32 Yoshida4',
        original_run01_result_mirror=dict(commit='b8bd64339169ecdf61b90fb7b575a3937fdc0c8c',
            path=next(r for r in refs if r['ref_id']=='R0017')['path'],original_commit='0d3578a4584a924ed10bdd124db6093e02d90c94',
            original_commit_locally_unavailable=True,original_commit_GitHub_422=True,mirror_SHA_verified=True,
            scope='H2 stdY4; H4 m5; H6 original SIGBUS not-evaluated; later reconstructed result separately identified'),
        exact_H4_H6_standard_Y4_confirmation_found=False,
        conclusion='Not found in inspected fixed confirmation records and bounded integration indexes; not a global absence proof',
        search_scope_commits=[CONFIRM,OLD_H6,INDEX,'b8bd64339169ecdf61b90fb7b575a3937fdc0c8c'],
        population_rank_guaranteed=False))
    # Quote accepted H2 main-epsilon inputs; leave all uncomputed endpoints null.
    h2=[]
    h2cfg=read(ROOT.parent/'h2-archived-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json')
    for row in inputs['rows']:
        if row['system']!='H2':continue
        dp=Path(h2cfg['origin_run_root'])/'results'/(row['candidate']+'_diagnostics.json')
        protect(dp);diagnostic=read(dp)
        require(diagnostic['reference_binding']['reference_recomputed'] is False,'H2_SAVED_REFERENCE')
        h2.append(dict(candidate=row['candidate'],target=row['target'],binding=row['binding'],
            bias=row['explanation']['signed_bias'],allowance=row['explanation']['allowance'],
            log_B=row['explanation']['log_B_base'],log_B_margin=row['explanation']['log_B_margin'],
            C=row['C'],paired_prices=row['pairs'],saved_reference_available=True,
            reference_binding=diagnostic['reference_binding'],saved_signal_columns=diagnostic['signal_columns'],
            accepted_main_endpoint=dict(epsilon=.0005,N=row['N'],G=row['G_saved'],source=row['source']),
            missing_endpoints=[dict(epsilon=ep,N=None,G=None,status='NOT_YET_COMPUTED_UNDER_ACCEPTED_POLICY')
                               for ep in (.05,.01,.005,.001)],
            source_input_manifest_sha256=cfg['manifest_sha256']))
    h2manifest=ASSETS/'H2_missing_precision_inputs_PRIVATE.json';prior.save(h2manifest,h2);protect(h2manifest)
    public('H2_missing_precision_inputs.json',dict(candidates=[{k:v for k,v in r.items() if k!='paired_prices'} for r in h2],
        policy_matched_saved_endpoints_missing=16,all_required_saved_inputs_available=True,
        reference_or_Hamiltonian_recomputation_required=False,new_shot_or_G_evaluation_needed=True,
        original_kernel=dict(commit=prior.old.PROVIDER,path='src/trottertracks/resource_applicability/ax2b_numerical_accounting.py',
            function='u_aware_shots',internal='axis_headroom once per axis'),
        account_source=dict(commit='2e59a55beec8159c7832ff4ff02513b8976c0eb9',path='scripts/h2_accepted_compare_backend.py',function='Backend.account'),
        minimal_adapter_change='Make epsilon explicit and output names candidate+epsilon unique; retain exact policy/arithmetic',
        empirical_allowance_not_recomputed=True,no_alternative_policy_substitution=True))
    complete=[r for r in controls if r['complete_saved_pair']]
    plans=dict(status='FIXED_SCOPE_DRAFTS_NOT_EXECUTABLE_NOT_APPROVED',evidence_SOURCE=source,
        evidence_input_catalog_sha256=digest(catalog),
        same_q=dict(epsilon=.001,targets=[dict(system=r['system'],method=r['method'],q=r['q'],
                      S2=r['S2']['candidate'],Yoshida4=r['Yoshida4']['candidate']) for r in complete],
            existing_inputs_complete=True,candidate_selection=False,
            proposed_calls=dict(bootstrap=1,bundle_load=1,cost_decomposition=8,ranking=0,sampling=0,compile=0,
                                simulation=0,new_shot=0,new_statistics=0),
            reuse=dict(commit='2e59a55beec8159c7832ff4ff02513b8976c0eb9',
                       function='h246_order_cost_analysis_backend.decompose'),
            required_adapter='New metadata/permission/target binding only; numerical body reusable'),
        H2_missing_4_eps=dict(candidates=[r['candidate'] for r in h2],epsilons=[.05,.01,.005,.001],
            input_manifest_sha256=sha(h2manifest),allowance_policy_unchanged=True,
            proposed_calls=dict(bootstrap=1,bundle_load=1,endpoint_accounting=16,u_aware_shots=16,
                axis_headroom=32,cost_decomposition=8,ranking=0,sampling=0,compile=0,simulation=0,
                reference_recompute=0,allowance_recompute=0,new_statistics=0),
            additional_optional_anchor_reproduction_calls=0,
            output='16 new endpoints and 8 within-family contrasts; no sample re-estimation',
            required_adapter='Explicit epsilon parameter and unique per-endpoint paths; reuse accepted bias/u/log_B and paired prices'),
        acceptance=['Exact targets/q/prefix/cohort/source/input policy', 'No zero for missing/ineligible fields',
                    'No new uncertainty independence assumptions or population claim','Same-q auxiliary; primary comparison unchanged'],
        execution_readiness='Not a runnable grant: production adapter and runtime seal must be fixed before the one execution approval',
        numerical_production_SOURCE=None,valid_grants_issued=0,old_grant_reuse=False,retry=False,mandatory_STOP=True)
    public('additional_validation_DRAFT.json',plans)
    public('source_catalog.json',dict(evidence_SOURCE=source,base_commit=BASE,inputs=catalog,
                                     private_input_count=len(protected),numerical_calls=0))
    public('metadata_preflight.json',dict(status='PASS_METADATA_ONLY',primary_candidates=8,weak_candidates=2,
        same_q_controls=16,complete_saved_pairs=len(complete),missing_H2_endpoints=16,
        H2_inputs_complete=True,current_H4_H6_independent_standard_Y4_confirmed=False,
        numerical_calls=0,valid_grants=0,strict_COMPARE='BLOCKED',mandatory_STOP=True))
    public('STOP.json',dict(status='TECHNICAL_EVIDENCE_COMPLETE_NO_NEW_SCIENCE',numeric_calls=0,
        grant_issued=False,new_sampling=False,new_compile=False,detailed_scientific_review=False,
        strict_COMPARE='BLOCKED',mandatory_STOP=True))
    # Flat values are literal saved fields, never new means, deltas or ratios.
    write_csv('primary_candidates_8.csv',[dict(system=r['system'],method=r['method'],formula=r['formula'],
        candidate=r['candidate'],prefix=r['prefix'],q=r['q'],n=r['n'],epsilon=r['epsilon'],
        C_cos=r['C']['cosine'],C_sin=r['C']['sine'],N_cos=r['N']['cosine'],N_sin=r['N']['sine'],G=r['G'],
        paired_G_SE=r['paired_G_SE'],u=r['u'],log_B=r['log_B'],cohort=r['cohort']) for r in main])
    write_csv('same_q_availability_16.csv',[dict(system=r['system'],method=r['method'],q=r['q'],epsilon=r['epsilon'],
        S2=r['S2']['candidate'],Yoshida4=r['Yoshida4']['candidate'],S2_status=r['S2']['status'],Y4_status=r['Yoshida4']['status'],
        availability=r['status'],S2_n=r['S2']['n'],Y4_n=r['Yoshida4']['n'],new_comparison_computed=False) for r in controls])
    # Final input protection includes newly recovered fixed copies and private metadata manifest.
    (OUT/'cache/all_inputs_before_PRIVATE.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(json.dumps(dict(status='METADATA_COMPLETE',evidence_SOURCE=source,protected_files=len(protected),
        complete_same_q_pairs=len(complete),new_scientific_calls=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    reject_execution(args.execute);collect(args.source)
