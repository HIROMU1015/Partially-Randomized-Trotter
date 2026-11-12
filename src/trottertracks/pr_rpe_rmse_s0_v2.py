"""Source-bound S0 V1 -> B canonical V2 and field-only evidence coverage.

No mean/normalizer/N calculation, parameter materialization or simulation.
Canonical identities and the evidence planner are delegated to pinned B code.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
import subprocess
from types import SimpleNamespace

from . import pr_rpe_rmse_pilot as legacy
A=legacy.A;p=legacy.p
S0='7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
S0_SCIENCE='7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e'
B_RELEASE='af284d853890155bacd19e61850c196bd1d5c660'
SCHEMA='PR_RPE_RMSE_V2'
S0_DIR='docs/research/pr_rpe_rmse_s0_results_20261113/run01/'
RAW_RELEASE='155529e20a8fac39e00786c40b57830b71203c4f'
RAW_DIR='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
SOURCE=dict(module='src/trottertracks/pr_rpe_rmse_s0_v2.py',B=B_RELEASE,S0=S0,S0_science=S0_SCIENCE)


def b_metadata(root):
    """Unchanged B metadata/qualification functions only; cost/C providers absent."""
    def load(path,names,scope):
        data=subprocess.check_output(['git','show',B_RELEASE+':'+path],cwd=root)
        nodes=[n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=set(names):raise ValueError('Fixed B metadata functions missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),B_RELEASE+':'+path,'exec'),scope)
    exact=dict(Fraction=Fraction,math=math)
    load('src/trottertracks/h4_df10_rpe_connection.py',('exact_scalar','exact_value'),exact)
    b=SimpleNamespace(p=p,a=SimpleNamespace(**{k:exact[k] for k in ('exact_scalar','exact_value')}),
        SCHEMA=SCHEMA,COMPARISON_METHODS=('FULL_PF','TRUNCATED_PF','PR','FULL_H_RTE'))
    bs=dict(p=p,a=b.a,deepcopy=deepcopy,hashlib=hashlib,json=json,SCHEMA=SCHEMA,
            METHODS=('PF','PR','FULL_H_RTE'),COMPARISON_METHODS=b.COMPARISON_METHODS,DRAFT_MAPPER=A.MAPPING)
    load(legacy.previous.B_PATH,('_id','_positive','comparison_method','_coverage','_assemble','_validate','assess','select'),bs)
    b._assemble=bs['_assemble']
    def validate(config):
        if config.get('schema')!=SCHEMA or config['spec'].get('schedule_adapter')!=legacy.ADAPTER:
            raise ValueError('Canonical V2 A-adapter configuration required')
        bs['_validate'](config)
    b._validate=validate;b.assess=bs['assess'];b.select=bs['select'];b.comparison_method=bs['comparison_method']
    scope=dict(A=A,B=b,p=p,deepcopy=deepcopy,Fraction=Fraction,ADAPTER=legacy.ADAPTER,A_RELEASE=legacy.previous.A_SOURCE)
    load('src/trottertracks/pr_rpe_rmse_integration.py',('_exact','_schedule','a_spec','generate_from_a_spec','n_from_a'),scope)
    bs['generate']=scope['generate_from_a_spec']
    evidence=dict(B=b,deepcopy=deepcopy)
    load('src/trottertracks/pr_rpe_rmse_evidence.py',('_source_constraints','_foreign_prices','plan'),evidence)
    return SimpleNamespace(validate=validate,generate=scope['generate_from_a_spec'],a_spec=scope['a_spec'],
        n_from_a=scope['n_from_a'],plan=evidence['plan'])


def connect(configuration,packet,*,root,canonical_v2=None):
    """Copy published nominal results only, through B's own packet translator."""
    store=p.SavedArtifacts(root)
    if any(not p.same(value,store.read(dict(commit=S0,path=S0_DIR+name)))
           for value,name in ((configuration,'CONFIGURATION.json'),(packet,'N_HANDOFF.json'))):
        raise ValueError('Published S0 configuration/packet SOURCE or content differs')
    inputs=configuration['inputs'];old=configuration['B_configuration'];schedule=configuration['A_schedule']
    if inputs['M']!=12 or inputs['design']['system']!='H4':raise ValueError('Fixed S0 H4/13-round scope required')
    v1=legacy.b_metadata(root);v1.validate(old)
    if not p.same(v1.schedule(inputs),schedule):raise ValueError('S0 whole schedule/exact kappa/SOURCE mismatch')
    scope=packet['pilot_scope'];expected_ids=[r['request_id'] for r in schedule['rows']
        if r['identity']['m'] in (0,6) and p.arm(r['identity']['candidate'])=='PR']
    if scope!=dict(rounds=[0,6],completed_request_ids=expected_ids,whole_schedule_RMSE_qualified=False):
        raise ValueError('S0 scope changed')
    old_base=deepcopy(packet);old_base.pop('pilot_scope')
    expected=v1.n_from_a(old,packet['A_original_packet'],inputs=inputs)
    if not p.same(old_base,expected):raise ValueError('Legacy packet/source/nominal values or identity changed')
    for row in packet['rounds']:
        if row['N_X'] is not None or row['N_Y'] is not None:raise ValueError('S0 has no sufficient N')
        if row['m'] not in (0,6) and (row['nominal_N_X'] is not None or row['nominal_N_Y'] is not None or row['model_status']!='NOT_EVALUATED'):
            raise ValueError('Outside S0 nominal science scope')
    b=b_metadata(root);generated=b.generate(b.a_spec(inputs,'PR'))
    config=generated if canonical_v2 is None else deepcopy(canonical_v2)
    b.validate(config)
    if not p.same(config,generated):raise ValueError('B V2 configuration differs from fixed S0 inputs')
    links=[]
    for before,after in zip(old['rounds'],config['rounds']):
        for key in ('m','q_RPE','q_PF','T','actual_delta_pf','structure','mapping','R_outer','physical_candidate','round_law_definition'):
            if not p.same(before[key],after[key]):raise ValueError('Physical schedule or law differs: '+key)
        for key in ('definition','kappa','event_mapping'):
            if not p.same(before['law_binding'][key],after['law_binding'][key]):raise ValueError('Law/SOURCE bridge mismatch')
        links.append(dict(m=after['m'],source_candidate_id=after['physical_candidate']['id'],
            legacy_B_candidate_id=before['candidate']['id'],canonical_B_candidate_id=after['candidate']['id']))
    out=b.n_from_a(config,packet['A_original_packet'],inputs=inputs)
    out.update(schema=SCHEMA,pilot_scope=deepcopy(scope),adapter_source=deepcopy(SOURCE),
        connection_provenance=dict(legacy_B_configuration_id=old['id'],canonical_B_configuration_id=config['id'],
            original_packet=dict(commit=S0,path=S0_DIR+'N_HANDOFF.json'),physical_links=links,
            nominal_recalculated=False,legacy_relabelled=False))
    for row,request in zip(out['rounds'],config['rounds']):row['source_candidate_id']=request['physical_candidate']['id']
    return dict(configuration=config,N_packet=out,source=deepcopy(SOURCE),science_calls=0,
                full_RPE_qualified=False,mandatory_stop=True)


def saved_inventory(root):
    """Extend existing field inventory with exact raw candidates and provenance."""
    store=p.SavedArtifacts(root);out=legacy.pilot_inventory(root)
    fixed=store.read(dict(commit=S0,path=S0_DIR+'CONFIGURATION.json'))
    target=fixed['inputs']['design']['binding'];wanted={r['physical_candidate']['id'] for r in fixed['B_configuration']['rounds']}
    prov=store.read(dict(commit=RAW_RELEASE,path=RAW_DIR+'scientific_provenance.json'))
    sp=store.read(dict(commit=RAW_RELEASE,path=RAW_DIR+'signal_provenance.json'))
    binding=prov['inputs']['H4']['binding'];compiler=prov['compiler']
    if not p.same(binding,target) or compiler['hash']!=fixed['inputs']['design']['profile']['compiler']:
        raise ValueError('Original raw carrier H/state/profile binding differs')
    raw_impl=dict(commit=sp['signal_SOURCE'],path='src/trotterlib/rte.py')
    if not legacy.previous.same_source(raw_impl,dict(commit=legacy.P1,path='src/trotterlib/rte.py'),root):
        raise ValueError('Original raw finite-RTE law SOURCE differs')
    for filename in ('signal_records.json.gz','extension_signal_records.json.gz'):
        path=RAW_DIR+filename;data=subprocess.check_output(['git','show',RAW_RELEASE+':'+path],cwd=root)
        store.sources[RAW_RELEASE+':'+path]=hashlib.sha256(data).hexdigest()
        for cid,record in json.loads(gzip.decompress(data)).items():
            if cid not in wanted:continue
            c=record['candidate'];sig=record['signal']
            if c['id']!=p.candidate_id(c) or record.get('phase_alignment') is not False:
                raise ValueError('Raw candidate or phase convention differs')
            out['means'].append(dict(candidate=deepcopy(c),binding=deepcopy(binding),
                profile=deepcopy(fixed['inputs']['design']['profile']),structure=p.structure(c),
                source=dict(commit=RAW_RELEASE,path=path,selector=cid,
                    binding_source=dict(commit=RAW_RELEASE,path=RAW_DIR+'scientific_provenance.json')),
                saved_status='ARCHIVED_FINITE_MEAN_NOT_RMSE_QUALIFIED',
                fields=legacy.previous.present(sig,legacy.previous.MEAN_FIELDS),
                error_fields=legacy.previous.present(sig.get('allowance_evidence',{}),legacy.previous.ERROR_FIELDS),
                source_norm_gate=dict(source=dict(commit=sp['signal_SOURCE'],path='scripts/h46_long_time_signal.py',function='evolve_batch'),
                    limit=1e-9,observed_error=None,status='SOURCE_GATE_ONLY_NOT_NEWLY_RECHECKED'),
                law_implementation_source=raw_impl,phase_alignment=False,
                numerical_values_copied=False,RMSE_qualification='NOT_EVALUATED'))
    # Exact m1 carrier already exists in the old bound catalog. No numeric fields read.
    for mean in out['means']:
        if mean['source']['commit']=='89c91c692e1d35de94d2e7afbe890aabe6d80b7e':
            mean['law_implementation_source']=dict(commit='90dcc3ed20f30dfab6b22a68cc4adb9461485a98',path='src/trotterlib/rte.py')
    out['additional_sources']=store.sources
    return out


def coverage(connection,inventory,*,root):
    """Round-level local evidence availability only, not B's eligibility planner."""
    config=connection['configuration'];packet=connection['N_packet'];out=[]
    a_binding=config['spec']['A_input']['design']['binding'];profile=config['spec']['profile']
    for row,n in zip(config['rounds'],packet['rounds']):
        c=row['physical_candidate'];matches=[];foreign=[]
        for saved in inventory.get('means',[]):
            if saved['candidate'].get('id')!=c['id']:continue
            if (not p.same(saved['candidate'],c) or not p.same(saved['binding'],a_binding)
                    or not p.same(saved['profile'],profile) or not p.same(saved['structure'],row['structure']) or not saved.get('source')
                    or not legacy.previous.same_source(saved.get('law_implementation_source'),
                        row['round_law_definition']['A_declaration']['source'],root)):
                foreign.append(deepcopy(saved.get('source')));continue
            matches.append(saved)
        refs=[v for v in inventory.get('references',[]) if v['T']==c['T'] and p.same(v['binding'],a_binding) and v.get('source')]
        pairs=[v for v in refs if set(legacy.previous.REFERENCE_FIELDS)<=set(v['fields'])]
        is_s0=row['m'] in (0,6)
        fields=set().union(*(set(v['fields']) for v in matches)) if matches else set()
        error_fields=set().union(*(set(v['error_fields']) for v in matches)) if matches else set()
        out.append(dict(m=row['m'],physical_candidate=deepcopy(c),structure=deepcopy(row['structure']),
            canonical_candidate_id=row['candidate']['id'],source_candidate_id=c['id'],law_binding=deepcopy(row['law_binding']),
            saved_mean_present=bool(matches),mean_sources=[deepcopy(v['source']) for v in matches],
            saved_statuses=[v['saved_status'] for v in matches],
            blocking_saved_statuses=[v['saved_status'] for v in matches if any(w in v['saved_status']
                for w in ('FAIL','STOP','UNSTABLE','UNRESOLVED','INFEASIBLE'))],
            rejected_same_id_sources=foreign,
            saved_reference_signal_present=bool(pairs) or 'reference' in fields,
            independent_reference_pair_present=bool(pairs),reference_pair_sources=[deepcopy(v['source']) for v in pairs],
            signal_fields=sorted(fields),error_fields=sorted(error_fields),
            missing_error_fields=[k for k in legacy.previous.ERROR_FIELDS if k not in error_fields],
            source_gate_evidence=[deepcopy(v['source_norm_gate']) for v in matches if v.get('source_norm_gate')],
            allowance_status='S0_PRESERVED_RECHECKED' if is_s0 else 'LEGACY_SOURCE_ALLOWANCE_BRIDGE_REQUIRED' if matches else 'NOT_EVALUATED',
            normalization_status='SAVED_LOG_B_PRESENT_NOT_REEVALUATED' if 'log_B' in fields else 'MISSING',
            local_diagnostics_status='S0_PUBLISHED_ONLY' if is_s0 else 'NOT_EVALUATED',
            nominal_N_X=n['nominal_N_X'],nominal_N_Y=n['nominal_N_Y'],
            nominal_status='S0_PUBLISHED_ONLY' if is_s0 else 'NOT_EVALUATED',
            sufficient_N_X=None,sufficient_N_Y=None,RMSE_bound_ha=None,
            scientific_gate_rechecked_now=False,law_implementation_sources=[deepcopy(v.get('law_implementation_source')) for v in matches if v.get('law_implementation_source')]))
    return dict(schema='A_FIXED13_LOCAL_EVIDENCE_FIELDS_ONLY_V1',source=deepcopy(SOURCE),configuration_id=config['id'],rounds=out,
        nominal_evaluated_rounds=[0,6],additional_saved_mean_rounds=[r['m'] for r in out if r['saved_mean_present'] and r['m'] not in (0,6)],
        missing_mean_rounds=[r['m'] for r in out if not r['saved_mean_present']],
        missing_reference_pair_rounds=[r['m'] for r in out if not r['independent_reference_pair_present']],
        all_round_RMSE_qualified=False,science_calls=0,mandatory_stop=True)


def build(root):
    store=p.SavedArtifacts(root)
    config=store.read(dict(commit=S0,path=S0_DIR+'CONFIGURATION.json'))
    packet=store.read(dict(commit=S0,path=S0_DIR+'N_HANDOFF.json'))
    connected=connect(config,packet,root=root)
    inventory=saved_inventory(root);local=coverage(connected,inventory,root=root)
    refs=[dict(commit=S0,path=S0_DIR+'result.json',m=m,kind='S0_LOCAL_DIAGNOSTICS_ONLY') for m in (0,6)]
    entry=dict(configuration=connected['configuration'],N_packet=connected['N_packet'],prices=[],cohorts=[],diagnostic_references=refs)
    # Delegates to B's actual metadata-only planner, never cost/provider methods.
    bplan=b_metadata(root).plan([entry],price_policy='qualified_expectation',required_methods=('PR',),
        unexplored_scope='FIXED_S0_PR_ONLY_NO_NEW_SCIENTIFIC_MENU')
    return dict(connection=connected,local_coverage=local,B_evidence_plan=bplan,
                source_files=dict(store.sources,**inventory.get('additional_sources',{})),science_calls=0,mandatory_stop=True)
