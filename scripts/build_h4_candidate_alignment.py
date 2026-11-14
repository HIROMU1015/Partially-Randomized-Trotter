"""Join published candidate/evidence metadata only; no signal/price/risk arithmetic.

A's saved-signal audit is authoritative for signal availability. Reuse C's
receipt registry and 52-axis DF coverage and B's published decoder bindings.
This is an evidence index, not a price/N producer or a new scientific guard.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from functools import lru_cache
import gzip
import io
import json
from pathlib import Path
import sys

from trottertracks import baseline_signal_pipeline as C
from trottertracks import price_implementation_guard as G

# scripts precedes tests to select the metadata reader, not its test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m5_m4_receipt_admission as M

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'docs/research/h4_five_method_candidate_alignment_20261115'
A = '714247619afb4714fd663eb15c0470ef32bdc7af'
AUDIT = 'c212cee857d639d2400819bbca3afa1dc052a9f4'
B = '99930491819cabd94609c42005a8637ac06a787d'
ADMIT = '855573064b2c17cfe20eb94a01ece1a554141b0b'
D = 'acea1118a4df4e727e13398ce0f8ae11511c7396'
B_LATEST = '9c0e12d87bf83429e3411d295b16c47d913283d2'
SOURCES = {
    'A_audit': dict(commit=AUDIT, path='docs/research/a_signal_candidate_identity_coverage_20261114/SIGNAL_COVERAGE.csv'),
    'A_audit_report': dict(commit=AUDIT, path='docs/research/a_signal_candidate_identity_coverage_20261114/REPORT.md'),
    'B_readiness': dict(commit=B, path='docs/research/rpe_required_n_cross_method_preparation_20261114/READINESS.json'),
    'B_pauli_binding': dict(commit=B, path='docs/research/b_total_q3_canonical_integration_20261114/CANONICAL_BINDING_METADATA.json'),
    'B_DF_inputs': dict(commit=B, path='docs/research/h4_dp_risk_fixed_results_20261114/INPUTS.json'),
    'B_pauli_inputs': dict(commit='caf1417ef20d0a5ab2030a2d957ee4f09443be43', path='docs/research/pauli_total_q3_accelerated_results_20261114/INPUTS.json'),
    'B_current_readiness': dict(commit=B_LATEST,path='docs/research/rpe_pr_df_total_bridge_20261115/READINESS.json'),
    'B_DF_mapping': dict(commit=B_LATEST,path='docs/research/rpe_pr_df_total_bridge_20261115/CANONICAL_MAPPING.json'),
    'C_df': dict(commit=ADMIT, path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/PR_DF_COVERAGE.json'),
    'C_saved': dict(commit=ADMIT, path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/SAVED_RECEIPT_SOURCES.json'),
    'C_cells': dict(commit='43c15bfa34a22f313e55b824223b200332b7f23f', path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json'),
    'D_update': dict(commit=D, path='docs/research/d_pf_price_saved4_revision_20261114/ADMISSION_UPDATE.md'),
    'A_price_design': dict(commit='7a0c9468b2499fac9fa1cd279ae2980c9dab048e', path='docs/research/a_full_h_p2_pauli_b1_price_design_20261114/DESIGNS.json'),
    'roadmap': dict(commit='fa84c0d6bce0c8e8b8accbe6009069074b466d32', path='docs/research/RESEARCH_ROADMAP.md'),
    'legacy_prices': dict(commit=M.CARRIER, path=M.BASE+'exploration_costs.json.gz'),
    'C_baseline_manifest': M.MANIFEST,
}
REMEDIES = {
    '1': 'SAVED_RESULT_ADMISSION', '2': 'METADATA_OR_DECODER_BRIDGE',
    '3': 'NEW_SIGNAL_SCIENCE', '4': 'DIRECT_COMPILED_PRICE_SCIENCE',
    '5': 'INDEPENDENT_PRICE_MODEL_VALIDATION', '6': 'MSE_OR_SUFFICIENT_N_SCIENCE',
    '7': 'USER_COMPARISON_OR_SCIENTIFIC_CONTRACT_DECISION',
}


def selected(root, ref):
    doc = json.loads(C.blob(root, ref['commit'], ref['path']))
    for key in ref.get('selector', []):
        doc = doc[key]
    return doc


def at(ref, *selector):
    return dict(ref, selector=list(ref.get('selector', []))+list(selector))


def check_price_link(context, identity, law, physical, price, axis):
    """Metadata equality using the unchanged C guard comparator.

    Published guard/sidecars remain the implementation evidence. This check
    cannot upgrade a finite path, a model, or a numerical qualification.
    """
    G._same(price['identity'], identity, 'Original canonical identity differs')
    G._same(price['law_binding'], law, 'Original law binding differs')
    G._same(price['physical_context'], context, 'Physical SOURCE/split/profile differs')
    G._same(price['physical_context']['physical_candidate'], physical, 'Physical candidate differs')
    if (price['axis'] != axis
            or price['measurement_axis'] != {'X':'cosine','Y':'sine'}[axis]
            or price['observable'] != {'X':'Re(z_raw)','Y':'Im(z_raw)'}[axis]
            or price['outcome_convention'] != 'P0_minus_P1'):
        raise ValueError('Measured X/Y differs')
    if price['P'] != 0 or price.get('normalization_is_price_multiplier', False):
        raise ValueError('P0 or price unit differs')
    return 'MATCHED_EVIDENCE'


def classify(signal, bridge, prices):
    """Correspondence only. Multiple blockers are preserved, not hidden."""
    flags = []
    if signal.startswith('S3'):
        flags.append('SIGNAL_EVIDENCE_MISSING')
    elif signal.startswith(('S2_', 'S2L_')):
        flags.append('INSUFFICIENT_SOURCE_EVIDENCE' if signal.startswith('S2L_')
                     else 'DECODER_OR_CANONICAL_BRIDGE_PENDING')
    if bridge != 'SOURCE_BOUND_TOTAL_VIEW':
        flags.append('DECODER_OR_CANONICAL_BRIDGE_PENDING')
    if not any(p['kind'] in ('DIRECT_COMPILED_DETERMINISTIC', 'DIRECT_COMPILED_FINITE_SAMPLE',
                             'SAVED_FINITE_DEVELOPMENT_COHORT') for p in prices):
        flags.append('PRICE_EVIDENCE_MISSING')
    flags = list(dict.fromkeys(flags))
    return dict(status=flags[0] if flags else 'MATCHED_BUT_UNQUALIFIED', blockers=flags,
                qualified_for_G=False, G=None)


def price_metadata(price, ref, *, guard, sidecars):
    """Copy provenance/meaning only. Deliberately omit mean/metrics/CI values."""
    return dict(kind=price['classification'], expectation=price['expectation'],
                cohort=price['cohort'], n_paths=price['n_paths'], source=ref,
                original_source=deepcopy(price['source']), guard=guard, sidecars=sidecars,
                price_profile=price.get('price_profile'), profile_extension=price.get('profile_extension'),
                diagnostic=price.get('diagnostic'), join_status='MATCHED_EVIDENCE',
                deterministic_receipt_admitted=price['expectation']=='DETERMINISTIC',
                qualified_law_expectation=None if price['expectation']=='DETERMINISTIC' else False,
                tier=('DETERMINISTIC_DIRECT_AT_FIXED_CONDITION'
                      if price['expectation']=='DETERMINISTIC' else 'WIRING_SINGLE_PATH_ONLY'),
                E_C=None)


@lru_cache(maxsize=2)
def build(root):
    root = str(root)
    audit = list(csv.DictReader(io.StringIO(C.blob(root, AUDIT, SOURCES['A_audit']['path']).decode())))
    ready = selected(root, SOURCES['B_readiness'])
    ready_by = {s['series']: s for s in ready['series']}
    current=selected(root,SOURCES['B_current_readiness'])
    current_by={c['configuration_id']:(i,c) for i,c in enumerate(current['cases'])}
    df_mapping=selected(root,SOURCES['B_DF_mapping'])['cases']
    df = selected(root, SOURCES['C_df'])
    df_by = {(r['label'].replace('PR-', 'PR-DF '), r['m'], r['axis']): (i, r)
             for i, r in enumerate(df['coverage'])}
    bindings = selected(root, SOURCES['B_pauli_binding'])['configurations']
    pauli = selected(root, dict(commit=A, path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json'))['packets']
    pinputs = selected(root, SOURCES['B_pauli_inputs'])
    dinputs = selected(root, SOURCES['B_DF_inputs'])
    registry = selected(root, SOURCES['C_saved'])
    manifest = selected(root, SOURCES['C_baseline_manifest'])
    cells = selected(root, SOURCES['C_cells'])
    cell_by = {(r['configuration_id'], r['m'], r['axis']): (i, r) for i, r in enumerate(cells)}
    # Existing D metadata functions only. No science factory/runner is called.
    t, f = M.d_callers(root)
    templates = {x['configuration_id']: x for x in f.plans(root)+t.plans(root)}
    legacy = json.loads(gzip.decompress(C.blob(root, M.CARRIER, SOURCES['legacy_prices']['path'])))
    rows, candidates = [], {}
    for ar in audit:
        name, m, axis, cfg = ar['candidate'], int(ar['m']), ar['axis'], ar['configuration_id']
        row = dict(candidate=name, route=ar['method'], configuration_id=cfg, m=m, axis=axis,
                   physical_id=ar['physical_id'], original_canonical_id=ar['canonical_id'] or None,
                   signal=dict(status=ar['signal_status'], source_index=at(SOURCES['A_audit'], audit.index(ar)),
                               original_sources=ar['signal_sources'], new_signal=False,
                               qualification='PRESERVE_ORIGINAL_CONDITIONAL_NUMERICAL_QUALIFICATION'),
                   decoder=dict(status='PENDING', source=SOURCES['B_readiness'],
                                total_configuration_id=None, total_canonical_id=None),
                   MSE=dict(status='NOT_EVALUATED_FOR_TOTAL_VIEW', source=None, qualified=False),
                   N=dict(used=None, vector_source=None, allocation_id=None, sufficient_N=None),
                   price=[], price_plan=None, E_C=None, G=None)
        config, physical, identity, law, context = None, None, None, None, None
        if cfg in templates:
            template = templates[cfg]
            config = template['plan']['configuration']
            req = template['plan']['pending']['requests'][m]
            physical, identity, law = req['physical_candidate'], req['identity'], req['law_binding']
            # Reuse C's exact price context (scalar/profile/source included).
            context = M.PN.I.price_context(template['plan'], m)
        elif name=='FULL_PF S2_0.1 (C manifest FULL_PF_S2_D01)':
            proposal=next(c for c in manifest['candidates'] if c['label']=='FULL_PF_S2_D01')
            config=proposal['configuration']; physical=proposal['rounds'][m]['physical_candidate']
            identity=config['rounds'][m]['identity'];law=config['rounds'][m]['law_binding']
            row['configuration_source']=at(SOURCES['C_baseline_manifest'],'candidates',0,'configuration')
        elif name.startswith('PR-DF '):
            s = ready_by['PR_DF_'+name.split()[-1]]
            config = selected(root, s['configuration_source'])
            packet = selected(root, s['bundle_source'])
            mean = packet['means'][m]
            identity, law = mean['identity'], mean['law_binding']
            index, evidence = df_by[(name, m, axis)]
            physical=deepcopy(config['rounds'][m]['physical_candidate'])
            G._same((physical['q'],physical['R_outer'],physical['K'],physical['prefix']),
                    (evidence['q_PF'],evidence['R_outer'],evidence['K'],evidence['prefix']),
                    'A/C DF physical q/R/K/prefix differs')
            G._same(ar['physical_id'], evidence['physical_id'], 'A/C DF physical id differs')
            G._same(identity, config['rounds'][m]['identity'], 'DF signal identity differs')
            bm=next(x for x in df_mapping if x['old_configuration_id']==cfg);br=bm['rows'][m]
            G._same(br['physical_candidate'],physical,'B fixed-grid physical fields differ')
            G._same(br['old_canonical_candidate_id'],identity['candidate']['id'],'B original canonical differs')
            if not (br['law_definition_unchanged'] and bm['physical_ids_unchanged'] and bm['old_spec_preserved']):
                raise ValueError('B bridge does not preserve original physical law/spec')
            row['decoder']=dict(status='SOURCE_BOUND_TOTAL_VIEW', source=SOURCES['B_DF_mapping'],
                binding_id=bm['binding_id'], original_configuration_id=cfg,
                total_configuration_id=bm['total_configuration_id'],
                total_canonical_id=br['total_canonical_candidate_id'],
                origin_adapter=bm['origin_adapter'],view_adapter=bm['view_adapter'],
                scientific_adoption=False,physical_circuit_changed=False)
            row['signal']['packet_source'] = s['bundle_source']
            row['price_coverage_source'] = at(SOURCES['C_df'], 'coverage', index)
            for part_index,part in enumerate(evidence['finite_cohort']):
                row['price'].append(dict(kind='SAVED_FINITE_DEVELOPMENT_COHORT',
                    expectation=part['expectation'], cohort=part['cohort'], n_paths=part['n_paths'],
                    role=part.get('role', part.get('consumer_role')), source=at(SOURCES['C_df'], 'coverage', index, 'finite_cohort',part_index),
                    original_source=df['sources'][part['source']], lowering=part['lowering_id'],
                    join_status='MATCHED_BUT_UNQUALIFIED', consumer_bridge='PENDING_SAVED_PRICE_ADMISSION',tier=evidence['G_paper_tier'],
                    qualified_law_expectation=False, E_C=None))
            row['structural_mean'] = dict(available=evidence['structural_expectation_available'],
                source=at(SOURCES['C_df'], 'coverage', index, 'structural_mean'),
                quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT', E_C=False,
                consumer_bridge='PENDING_NO_ADMISSION')
            row['models'] = dict(C1=evidence['C1'], additive=evidence['additive'],
                residual_model_in_current_domain=evidence['residual_model_in_current_domain'],
                qualified_compiled_expectation_available=False)
            case = next(c for c in dinputs['cases'] if c['label']==name.split()[-1])
            row['N'] = dict(used=case['allocation']['shots'][m]['N_'+axis],
                vector_source=at(SOURCES['B_DF_inputs'], 'cases', dinputs['cases'].index(case), 'allocation'),
                allocation_id=case['allocation']['allocation_id'], shot_policy='USER_SUPPLIED', sufficient_N=None)
            row['MSE'] = dict(status='SAVED_PARTIAL_DP_OLD_DECODER_CONDITIONAL_MODEL_ONLY', qualified=False,
                source=dict(commit=B, path='docs/research/h4_dp_risk_fixed_results_20261114/run01/RESULT_'+case['label']+'.json'))
        elif name.startswith('PR-Pauli '):
            prefix = int(name[-1]); packet = pauli[prefix-1]; config=packet['configuration']
            item=packet['rounds'][m]; mean=item['mean_record']
            context=M.PN.I.price_context(dict(configuration=config,provider='A_PR_PAULI',pending=dict(
                jobs=[dict(physical_context=x['mean_record']['physical_context']) for x in packet['rounds']])),m)
            physical=context['physical_candidate']
            identity, law = item['identity'], item['law_binding']
            G._same(identity, mean['B_identity'], 'Pauli saved identity differs')
            G._same(law, mean['B_law_binding'], 'Pauli saved law differs')
            bind=bindings[prefix-1]; br=bind['rounds'][m]
            G._same(identity['candidate']['id'], br['original_candidate_id'], 'Pauli bridge original id differs')
            row['decoder'] = dict(status='SOURCE_BOUND_TOTAL_VIEW', source=SOURCES['B_pauli_binding'],
                binding_id=bind['binding_id'], original_configuration_id=cfg,
                total_configuration_id=bind['total_configuration_id'], total_canonical_id=br['total_candidate_id'],
                decoder_source=bind['decoder_source'], risk_source=bind['risk_source'],
                scientific_adoption=False, physical_circuit_changed=False)
            case=pinputs['cases'][prefix-1]
            row['N']=dict(used=case['allocation']['shots'][m]['N_'+axis],
                vector_source=at(SOURCES['B_pauli_inputs'], 'cases', prefix-1, 'allocation'),
                allocation_id=case['allocation']['allocation_id'], shot_policy='USER_SUPPLIED', sufficient_N=None)
            row['MSE']=dict(status=ready['prior_Pauli_results'][prefix-1]['status'], qualified=False,
                source=ready['prior_Pauli_results'][prefix-1]['raw_source'],
                qualification=ready['prior_Pauli_results'][prefix-1]['qualification'])
            row['signal']['packet_source']=dict(commit=A, path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json',
                                               selector=['packets',prefix-1,'rounds',m])
            row['dictionary']=packet['dictionary_identity']
        elif name.startswith('FULL_H') or name=='TRUNCATED_B0I_P7 S2_0.1':
            if name.startswith('FULL_H R448'):
                pref=dict(commit='5fb2c32e39e63267ea1748861269c362a93a1b86',
                          path='docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/PLAN.json')
                plan=selected(root,pref); packet_ref=dict(pref,path=pref['path'].replace('PLAN.json','B_SIGNAL_INPUT.json'))
                readyrow=None
            else:
                series='FULL_H_KAPPA2' if name.startswith('FULL_H') else 'TRUNCATED_PF_P7_B0I'
                s=ready_by[series]; pref=s['configuration_source']; packet_ref=s['packet_source']
                plan=selected(root,dict(pref,selector=[])); readyrow=s['rows'][m]
                row['decoder']=dict(status='SOURCE_BOUND_TOTAL_VIEW', source=SOURCES['B_readiness'],
                    binding_id=s['binding_id'], original_configuration_id=cfg,
                    total_configuration_id=s['total_configuration_id'],
                    total_canonical_id=readyrow['total_canonical_candidate_id'], scientific_adoption=False)
            config=plan['configuration'];req=plan['requests'][m]
            physical,identity,law=req['physical_candidate'],req['identity'],req['law_binding']
            mean=selected(root,packet_ref)['means'][m]
            G._same(mean['identity'], identity, 'Baseline saved identity differs')
            G._same(mean['law_binding'], law, 'Baseline saved law differs')
            context=M.PN.I.price_context(dict(configuration=config,provider='C_BASELINE',pending=plan),m)
            row['signal']['packet_source']=packet_ref
        if physical is not None:
            G._same(physical['id'], ar['physical_id'], 'A/consumer physical id differs')
            row['physical_candidate']=deepcopy(physical)
        else:
            row['physical_candidate']=None
        if identity:
            if ar['canonical_id']:
                G._same(identity['candidate']['id'], ar['canonical_id'], 'A/consumer canonical id differs')
            row['original_canonical_id']=identity['candidate']['id']
            row['structure']=dict(canonical=identity['structure'],
                                   physical=(context or {}).get('structure',identity['structure']))
        if config:
            G._same(config['id'], cfg, 'A/consumer schedule differs')
        if cfg in current_by:
            ci,cur=current_by[cfg]
            row['current_B_readiness']=at(SOURCES['B_current_readiness'],'cases',ci)
            row['MSE']['total_evaluation_status']=cur['total_model_MSE']
            row['N']['total_vector_status']=cur.get('total_shot_vector')
            if row['decoder']['total_configuration_id'] is not None:
                G._same(row['decoder']['total_configuration_id'],cur['total_configuration_id'],
                        'Published total configuration differs')
            elif cur['total_configuration_id'] is not None:
                row['decoder'].update(source=at(SOURCES['B_current_readiness'],'cases',ci),
                    total_configuration_id=cur['total_configuration_id'],
                    status=('SOURCE_BOUND_TOTAL_VIEW' if cur['total_metadata'].startswith('SOURCE_BOUND_VIEW_ACCEPTED')
                            else cur['total_metadata']))
            row['decoder']['latest_published_state']=cur['total_metadata']
        # Reuse the four already admitted deterministic receipts and guard results.
        for saved in registry['receipts']:
            if (saved['physical_id'],saved['m'],saved['axis'])==(row['physical_id'],m,axis):
                ref=saved['references']['PRICES.json'];price=selected(root,ref)
                check_price_link(context,identity,law,physical,price,axis)
                row['price'].append(price_metadata(price,ref,guard=saved['references']['GUARD_CHECK.json'],
                                                  sidecars=saved['references']['SIDECARS.json']))
        # Published single-path wiring receipts, with original canonical view.
        pilot=None
        if name=='PR-Pauli LD1' and m==0:
            pilot=('00f74f86d16cc6878e1d4c5977eb9abd9ce9c076',
                   'docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02/', 'C_GUARD_CHECK.json','SIDECARS_PUBLISHED.json')
        elif name=='FULL_H kappa2 K2 q=2^m' and m==3:
            pilot=('a04470c0ad489eea1798856e648f863f9092cfb1',
                   'docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/', 'GUARD_CHECK.json','SIDECARS.json')
        if pilot:
            commit,base,guard,sidecar=pilot
            prices=selected(root,dict(commit=commit,path=base+'PRICES.json'))
            idx=next(i for i,p in enumerate(prices) if p['axis']==axis)
            price=prices[idx]
            check_price_link(context,identity,law,physical,price,axis)
            row['price'].append(price_metadata(price,dict(commit=commit,path=base+'PRICES.json',selector=[idx]),
                guard=dict(commit=commit,path=base+guard), sidecars=dict(commit=commit,path=base+sidecar)))
        # Legacy price locator is not an accepted full-wrapper receipt.
        old=legacy.get(row['physical_id'])
        if old is not None and ar['method'] in ('FULL','B0I'):
            if physical: G._same(old['candidate'], physical, 'Legacy physical fields differ')
            row['legacy_price_candidate']=dict(source=at(SOURCES['legacy_prices'],row['physical_id']),
                kind=old.get('kind','SAVED_PRICE_METADATA'), origin=old.get('origin'),
                reuse_provenance=old.get('reuse_provenance'), result_sha256=old.get('result_sha256'),
                status='INSUFFICIENT_SOURCE_EVIDENCE', accepted_receipt=False,
                full_measured_wrapper_qualified=False)
        key=(cfg,m,axis)
        if key in cell_by:
            i,cell=cell_by[key]
            G._same(cell['physical_candidate'],physical,'CELL_PLAN physical fields differ')
            row['price_plan']=dict(source=at(SOURCES['C_cells'],i),role=cell['role'],action=cell['action'],
                                   model_group=cell['model_group'],current_D_source=SOURCES['D_update'])
        row['price_coverage']=dict(
            admitted_deterministic=any(p['kind']=='DIRECT_COMPILED_DETERMINISTIC' for p in row['price']),
            finite_path_or_cohort=any(p['kind'] in ('DIRECT_COMPILED_FINITE_SAMPLE','SAVED_FINITE_DEVELOPMENT_COHORT') for p in row['price']),
            structural_expectation=row.get('structural_mean',{}).get('available',False),
            residual_model_in_current_domain=False,qualified_stochastic_compiled_expectation=False,
            new_model_prediction=None,legacy_unadmitted_locator=bool(row.get('legacy_price_candidate')))
        row['compatibility']=classify(row['signal']['status'],row['decoder']['status'],row['price'])
        remedies={'6','7'}
        if row['signal']['status'].startswith('S3'): remedies.add('3')
        if row['signal']['status'].startswith(('S2_','S2L_')): remedies.update(('1','2'))
        if row['signal']['status'].startswith('S2L_'): remedies.add('7')
        if row['decoder']['status']!='SOURCE_BOUND_TOTAL_VIEW': remedies.add('2')
        if not any(p['kind']=='DIRECT_COMPILED_DETERMINISTIC' for p in row['price']):
            remedies.update(('4','5'))
        if row.get('legacy_price_candidate') or row.get('price_coverage_source'): remedies.add('1')
        if row.get('price_coverage_source') and (row['price'] or row['price_coverage']['structural_expectation']):
            remedies.add('2')
        row['next']=[REMEDIES[x] for x in sorted(remedies)]
        row['missing']=['SUFFICIENT_N','SCIENTIFIC_COMPARISON_CONTRACT']
        if not any(p['kind']=='DIRECT_COMPILED_DETERMINISTIC' for p in row['price']):
            row['missing'].append('QUALIFIED_COMPILED_EXPECTATION')
        if row['decoder']['status']!='SOURCE_BOUND_TOTAL_VIEW':row['missing'].append('TOTAL_DECODER_BINDING')
        if not row['signal']['status'].startswith(('S1_','S1B_')):row['missing'].append('ACCEPTED_ROUND_SIGNAL_PACKET')
        if any(p.get('consumer_bridge')=='PENDING_SAVED_PRICE_ADMISSION' for p in row['price']):
            row['missing'].append('SAVED_PRICE_CONSUMER_ADMISSION')
        if row['price_coverage']['structural_expectation']:
            row['missing'].append('STRUCTURAL_EXPECTATION_CONSUMER_ADMISSION')
        rows.append(row)
        if name not in candidates:
            spec=config['spec'] if config else {}
            candidates[name]=dict(candidate=name,route=ar['method'],configuration_id=cfg,
                formula=spec.get('formula'), prefix=spec.get('L_D'), requested_delta_pf=spec.get('delta_pf'),
                finite_K=spec.get('finite_K'), kappa=spec.get('kappa'),
                binding=spec.get('binding'), law=spec.get('law'), split=spec.get('split'),
                compiler_profile=spec.get('profile'), physical_SOURCE=spec.get('binding',{}).get('source'),
                signal_audit=SOURCES['A_audit'], decoder=row['decoder'], MSE=row['MSE'],
                N_vector_source=row['N']['vector_source'], sufficient_N=None, E_C=None,G=None)
    for name,candidate in candidates.items():
        rr=[r for r in rows if r['candidate']==name]
        candidate['signal_status_counts_axes']=dict(Counter(r['signal']['status'] for r in rr))
        candidate['saved_price_axes']=[dict(m=r['m'],axis=r['axis'],kinds=[p['kind'] for p in r['price']])
                                      for r in rr if r['price']]
        candidate['legacy_price_candidate_axes']=[dict(m=r['m'],axis=r['axis']) for r in rr if r.get('legacy_price_candidate')]
        candidate['missing']=sorted({v for r in rr for v in r['missing']})
    # Reject tempting cross-candidate transfers. A one-round physical coincidence
    # is explicitly distinct from whole-schedule/canonical/N equivalence.
    def find(name,m): return next(r for r in rows if r['candidate']==name and r['m']==m and r['axis']=='X')
    cross=[]
    for left,right,m in [
        ('TRUNCATED_B0I_P7 S2_0.1','TRUNCATED_B0I_P7 S2_16h_over_7',0),
        ('TRUNCATED_B0I_P7 S2_0.1','TRUNCATED_B0I_P7 S2_16h_over_7',4),
        ('TRUNCATED_B0I_P7 S2_0.1','TRUNCATED_B0I_P7 M5_16h',4),
        ('FULL_H kappa2 K2 q=2^m','FULL_H R448 K2 q=2^m (separate candidate)',3),
        ('PR-DF LD1','PR-Pauli LD1',0),
        ('PR-DF LD1','PR-DF LD2',0),
    ]:
        l,r=find(left,m),find(right,m)
        same=C.p.same(l['physical_candidate'],r['physical_candidate'])
        cross.append(dict(left=left,right=right,m=m,axes=['X','Y'],
            left_physical_id=l['physical_id'],right_physical_id=r['physical_id'],
            status='DECODER_OR_CANONICAL_BRIDGE_PENDING' if same else 'PHYSICAL_CANDIDATE_MISMATCH',
            physical_round_equal=same,whole_schedule_equal=False,signal_transfer_adopted=False,
            N_or_price_transfer=False))
    return dict(schema='H4_CANDIDATE_EVIDENCE_ALIGNMENT_V1',status='TECHNICAL_METADATA_ONLY_MANDATORY_STOP',
                sources=SOURCES,remedy_classes=REMEDIES,candidates=list(candidates.values()),rows=rows,cross_candidate_links=cross,
                scientific_calls=0, numerical_reanalysis=0, N_sufficient=None,E_C=None,G=None,winner=None)


def validate(data):
    keys=[(r['configuration_id'],r['m'],r['axis']) for r in data['rows']]
    if len(keys)!=364 or len(set(keys))!=364 or len(data['candidates'])!=14:
        raise ValueError('Expected 14 distinct schedules x13 rounds x2 axes')
    for r in data['rows']:
        if r['N']['sufficient_N'] is not None or r['E_C'] is not None or r['G'] is not None:
            raise ValueError('Index cannot produce qualified N/E[C]/G')
        if r.get('physical_candidate') and r['physical_candidate']['id']!=r['physical_id']:
            raise ValueError('Physical fields and id differ')
        if r['compatibility']['qualified_for_G']:
            raise ValueError('Metadata index is not a scientific qualification')
        for p in r['price']:
            if p['qualified_law_expectation'] or p['E_C'] is not None:
                raise ValueError('Finite/path/model evidence cannot be promoted')
    return dict(status='PASS',candidates=14,round_axis_rows=364,
                reused_PR_DF_rows=52,scientific_calls=0,mandatory_stop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/OUTPUT)
    args=ap.parse_args();data=build(str(ROOT));checks=validate(data)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('ALIGNMENT.json',data),('METADATA_CHECKS.json',checks)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    cols=['candidate','route','configuration_id','m','axis','physical_id','canonical_id','signal_status',
          'decoder','MSE','used_N','sufficient_N','price_kinds','price_plan','compatibility','missing','next']
    with (args.output/'ALIGNMENT.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=cols,lineterminator='\n');writer.writeheader()
        for r in data['rows']:
            writer.writerow(dict(candidate=r['candidate'],route=r['route'],configuration_id=r['configuration_id'],
                m=r['m'],axis=r['axis'],physical_id=r['physical_id'],canonical_id=r['original_canonical_id'],
                signal_status=r['signal']['status'],decoder=r['decoder']['status'],MSE=r['MSE']['status'],
                used_N=r['N']['used'],sufficient_N='',price_kinds=';'.join(p['kind'] for p in r['price']),
                price_plan=(r['price_plan'] or {}).get('action',''),compatibility=r['compatibility']['status'],
                missing=';'.join(r['missing']),next=';'.join(r['next'])))
    print(json.dumps(checks))


if __name__=='__main__': main()
