"""H6 scalar/JSON analysis only. No quantum/science library imports or sampling.

All probability/mixture calculations are diagnostics of the SAVED finite law.
An observed maximum, instruction guard, or plug-in mean is never a cost bound.
"""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-hash-reuse-eight-worker-20261012')
RUN=PARENT/'artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/launch_confirmation_fast_v1'
SUMMARY=ROOT/'artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12'
EXPLORATION=ROOT/'artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12'
ARCHIVE=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261012/roundoff-execution-v1/h6-roundoff-preparations-v1.json')
REVIEW=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track_a_h6_independent_confirmation_scientific_review_20261013.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:
        raise ValueError('METADATA_PATH_OR_SIZE')
    return json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(name,obj):
    data=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>16*2**20:raise ValueError('ANALYSIS_OUTPUT_LIMIT')
    with (HERE/name).open('xb') as f:f.write(data)


def max_order_probabilities(R,p2,p4=0.):
    if type(R) is not int or R<1 or min(p2,p4)<0 or p2+p4>=1:
        raise ValueError('FINITE_LAW_DOMAIN')
    log0=R*math.log1p(-math.fsum((p2,p4)))
    log_no4=R*math.log1p(-p4)
    return {'0':math.exp(log0),'2':math.exp(log_no4)*-math.expm1(log0-log_no4),
        '4':-math.expm1(log_no4)}


def histogram_probabilities(R,p2,p4=0.):
    # Complement arithmetic differs from saved p0 only by binary64 roundoff.
    p0=1-math.fsum((p2,p4));result={}
    for n4 in range(R+1):
        if p4==0 and n4:continue
        for n2 in range(R-n4+1):
            n0=R-n2-n4
            result[(n2,n4)]=math.comb(R,n4)*math.comb(R-n4,n2)*p0**n0*p2**n2*p4**n4
    return result


def stats(values):
    if not values:return dict(n=0,mean=None,min=None,max=None,sample_sd=None,standard_error=None)
    n=len(values);sd=statistics.stdev(values) if n>1 else None
    return dict(n=n,mean=float(Fraction(sum(values),n)),min=min(values),max=max(values),
        sample_sd=sd,standard_error=sd/math.sqrt(n) if sd is not None else None)


def mixture(probabilities,values,baseline=None):
    """Missing conditional means stay null; breakpoint is explicitly plug-in."""
    records={str(k):dict(probability=p,**stats(values.get(k,[]))) for k,p in probabilities.items()}
    unknown=math.fsum(p for k,p in probabilities.items() if p>0 and not values.get(k))
    known=math.fsum(p*stats(values[k])['mean'] for k,p in probabilities.items() if values.get(k))
    return dict(strata=records,known_mass_plugin=known,unobserved_probability_mass=unknown,
        full_plugin_mean=known if unknown==0 else None,
        unknown_common_conditional_mean_break_even=(baseline-known)/unknown if baseline is not None and unknown>0 else None,
        known_conditional_means_are_sample_estimates=True,unobserved_means_imputed_as_zero=False,
        population_mean_certified=False,cost_tail_bound=None)


def scalar_codec(codec):
    nodes=codec['nodes']
    def node(x):return nodes[x['ref']]
    def scalar(x):
        if isinstance(x,dict) and 'float_hex' in x:return float.fromhex(x['float_hex'])
        if isinstance(x,dict) and 'ref' in x:
            n=node(x)
            if n['type'] not in ('tuple','list'):raise ValueError('SCALAR_ONLY_CODEC')
            return [scalar(v) for v in n['items']]
        if x is None or isinstance(x,(str,bool,int,float)):return x
        raise ValueError('ARRAY_OR_RUNTIME_OBJECT_NOT_DECODED')
    return node,scalar


def component_catalog(archive,prefix):
    c=archive['codec'];node,scalar=scalar_codec(c)
    f=node(c['roots'][f'faithful:p{prefix}'])['fields'];r=node(f['rte_preparation'])['fields']
    tail=node(r['symbolic_tail'])['fields'];catalog=[]
    for ref in node(tail['components'])['items']:
        x=node(ref)['fields'];catalog.append({k:scalar(x[k]) for k in
            ('component_id','coefficient_abs','coefficient_sign','probability','basis_id','basis_hash',
             'df_fragment_id','is_identity','diagonal_pauli_support')})
        catalog[-1]['basis_operation_count']=len(node(x['basis_change_operations'])['items'])
    blocks=[]
    for ref in node(f['deterministic_blocks'])['items']:
        n=node(ref);x=n['fields'];q=x['num_system_qubits'];pairs=0 if n['type']=='DFDeterministicOneBodySpec' else q*(q-1)//2
        basis=2*len(node(x['runtime_basis_operations'])['items'])
        blocks.append(dict(block_type=n['type'],basis_operations=basis//2,
            uncontrolled_instruction_bound=basis+q+pairs,directional_instruction_bound=basis+3*q+5*pairs+1))
    product=max(2*x['basis_operation_count']+len(x['diagonal_pauli_support']) for x in catalog if not x['is_identity'])
    rotation=max(2*x['basis_operation_count']+1 for x in catalog if not x['is_identity'])
    return dict(prefix=prefix,lambda_r=scalar(f['exact_rte_lambda_r']),components=catalog,
        component_probability_sum=math.fsum(x['probability'] for x in catalog),blocks=blocks,
        maximum_product_instruction_bound=product,maximum_rotation_instruction_bound=rotation,
        backbone_per_q_instruction_bound=sum(x['uncontrolled_instruction_bound']+x['directional_instruction_bound'] for x in blocks),
        structural_bounds_are_not_compiled_RZ_costs=True)


def extract(directory,expected,source,phase,selected,inventory=None):
    directory=Path(directory);rp=directory/'cost_result.json';tp=directory/'trajectory.json'
    if sha(rp)!=expected:raise ValueError('SAVED_COST_SHA_CHANGED')
    if inventory is not None:
        item=inventory[str(tp)]
        if tp.stat().st_size!=item['bytes'] or sha(tp)!=item['sha256']:raise ValueError('SAVED_TRAJECTORY_SHA_CHANGED')
    tr=read(tp);cost=read(rp);task=cost['task'];events=tr['events']
    if task!=tr['task'] or task['phase']!=phase or task['cell']['method']!='B2' or digest(events)!=tr['event_digest']:
        raise ValueError('SAVED_TASK_EVENT_IDENTITY')
    orders=[e['taylor_order'] for step in events for e in step]
    if orders!=tr['observed_orders'] or len(orders)!=task['cell']['R']:raise ValueError('SAVED_ORDER_IDENTITY')
    paired=[x for x in cost['rows'] if x['control']=='symmetric_directional']
    if len(paired)!=2 or {x['axis'] for x in paired}!={'cosine','sine'}:raise ValueError('PAIRED_AXES')
    for x in paired:
        if any(x[k]!=task[k] for k in ('cell_id','seed','phase','replica')) or x['event_digest']!=tr['event_digest']:
            raise ValueError('SAVED_WRAPPER_IDENTITY')
        if x!=read(directory/f'wrapper_symmetric_directional_{x["axis"]}.json'):raise ValueError('SAVED_WRAPPER_FILE')
    probabilities=dict(zip(tr['finite_distribution']['orders'],tr['finite_distribution']['order_probabilities']))
    record=dict(cell_id=task['cell_id'],cell=task['cell'],phase=phase,replica=task['replica'],seed=task['seed'],
        source_commit=source,event_digest=tr['event_digest'],orders=orders,n2=orders.count(2),n4=orders.count(4),
        max_order=max(orders),finite_distribution=tr['finite_distribution'],
        metrics={x['axis']:x['metrics'] for x in paired},
        compiler_hashes={x['axis']:x['compiler_hash'] for x in paired},
        classical_wall_seconds=sum(x['classical_wall_seconds'] for x in paired),
        origin_directory=str(directory),cost_result_sha256=sha(rp),trajectory_sha256=sha(tp))
    component_variants={};compact=[]
    for step in events:
        st=[]
        for e in step:
            applications=[]
            for a in e['application_sequence']:
                key=a['component_id']+'@'+str(a['basis_hash'])
                v={k:a[k] for k in ('component_id','coefficient_abs','coefficient_sign','basis_id','basis_hash','df_fragment_id','is_identity','diagonal_pauli_support')}
                v['basis_operation_count']=len(a['basis_change_operations'])
                component_variants[key]=v;applications.append(dict(variant_key=key,role=a['role']))
            st.append(dict(taylor_order=e['taylor_order'],event_probability=e['event_probability'],
                rotation_angle=e['rotation_angle'],applications=applications))
        compact.append(st)
    if selected or 4 in orders:record['event_sequence']=compact
    return record,component_variants,probabilities


def main():
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));os.nice(10)
    manifest=read(RUN/'frozen_manifest.json');source_hashes=manifest['source_hashes']
    for p,h in source_hashes.items():
        if sha(ROOT/p)!=h:raise ValueError('FIXED_RUNTIME_SOURCE_CHANGED:'+p)
    assert sha(ARCHIVE)=='ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010'
    selection=manifest['exploration_binding']['selection'];selected=set(selection['selected_ids'])
    archive=read(ARCHIVE);catalogs={p:component_catalog(archive,p) for p in (5,10,15)}
    records=[];variants={};other_order4_variants={}
    for g in read(EXPLORATION/'integrated_exploration_v1.json')['groups']:
        task=manifest['base_manifest']['reuse_registry']['original_tasks'][g['index']]
        if task['cell']['method']!='B2':continue
        rec,var,_=extract(Path(g['cost_result_path']).parent,g['cost_result_sha256'],g['source_commit'],
            'exploration',task['cell_id'] in selected,manifest['exploration_binding']['saved_group_metadata_inventory'])
        records.append(rec)
        if rec['cell_id'] in selected:variants.update(var)
        elif rec['n4']:other_order4_variants.update(var)
    provenance=read(SUMMARY/'integrated_confirmation_v1.json')
    for g in provenance['groups']:
        directory=Path(g['directory']);expected=g['file_sha256']
        for name in ('trajectory.json','cost_result.json','worker_terminal.json'):
            if sha(directory/name)!=expected[name]:raise ValueError('CONFIRMATION_METADATA_CHANGED')
        rec,var,_=extract(directory,expected['cost_result.json'],g['source_commit'],'confirmation',True)
        records.append(rec);variants.update(var)
    assert len(records)==832
    by=defaultdict(list)
    for r in records:by[(r['cell_id'],r['phase'])].append(r)
    for (cid,phase),rs in by.items():
        count=32 if phase=='confirmation' else 8
        assert len(rs)==count and {x['replica'] for x in rs}==set(range(count))
    assert len({(x['phase'],x['cell_id'],x['replica']) for x in records})==832
    assert len({x['seed'] for x in records})==832
    maps=read(SUMMARY/'confirmation_resource_map_v1.json');oldmaps=read(EXPLORATION/'exploration_resource_map_v1.json')
    indexed={(r['cell_id'],r['epsilon']):r for r in maps['rows'] if r['metric']=='RZ' and r['allowance_factor']=='1.0'}
    scalar_rows=[];diagnostics=[];histogram_rows=[];comparison=[]
    for cid in sorted(selected):
        cell=by[(cid,'confirmation')][0]['cell'];R=cell['R'];catalog=catalogs[cell['prefix']]
        saved=by[(cid,'confirmation')][0]['finite_distribution'];law=dict(zip(saved['orders'],saved['order_probabilities']))
        p2,p4=law.get(2,0.),law.get(4,0.);prob=max_order_probabilities(R,p2,p4);hist=histogram_probabilities(R,p2,p4)
        assert math.isclose(math.fsum(hist.values()),1.,abs_tol=2e-14)
        I0=cell['q']*catalog['backbone_per_q_instruction_bound']+R*catalog['maximum_rotation_instruction_bound']+cell['q']+5
        Ip=catalog['maximum_product_instruction_bound']
        expected_I=I0+R*(2*p2+4*p4)*Ip
        for phase in ('exploration','confirmation'):
            rs=by[(cid,phase)];counts=Counter((r['n2'],r['n4']) for r in rs)
            for (n2,n4),p in hist.items():
                histogram_rows.append(dict(cell_id=cid,phase=phase,n2=n2,n4=n4,probability=p,
                    observed_trajectories=counts[(n2,n4)],native_instruction_bound=I0+(2*n2+4*n4)*Ip))
            maxvalues={s:[r['metrics']['cosine']['RZ'] for r in rs if str(r['max_order'])==s] for s in prob}
            histvalues={s:[r['metrics']['cosine']['RZ'] for r in rs if (r['n2'],r['n4'])==s] for s in hist}
            observed_components={a['variant_key'].split('@')[0] for r in rs for step in r['event_sequence'] for e in step for a in e['applications']}
            diagnostics.append(dict(cell_id=cid,phase=phase,R=R,saved_order_probabilities=law,
                p0_complement_roundoff_difference=(1-math.fsum((p2,p4)))-law[0],
                max_order_probabilities=prob,max_order_RZ_cosine=mixture(prob,maxvalues),
                histogram_RZ_cosine=mixture(hist,histvalues),
                component_support_observed=len(observed_components),component_support_total=len(catalog['components']),
                component_probability_mass_seen=math.fsum(c['probability'] for c in catalog['components'] if c['component_id'] in observed_components),
                source_structural_native_bound_all_orders=I0+R*cell['K']*Ip,
                expected_structural_native_instruction_bound=expected_I,
                compiled_RZ_bound_proven=False,exact_component_sequence_support_covered=False))
            for eps in (.05,.01,.005,.001):
                row=indexed.get((cid,eps))
                if row is None:continue
                N={a:row['axes'][a]['shots'] for a in ('cosine','sine')}
                values=[sum(N[a]*r['metrics'][a]['RZ'] for a in N) for r in rs]
                st=stats(values)
                original_map=maps if phase=='confirmation' else oldmaps
                old=next(r for r in original_map['rows'] if r['cell_id']==cid and r['epsilon']==eps and r['metric']=='RZ' and r['allowance_factor']=='1.0')
                assert math.isclose(st['mean'],old['G_point'],rel_tol=2e-15)
                pf=min((r for r in maps['rows'] if r['method']=='B1' and r['epsilon']==eps and r['metric']=='RZ' and r['allowance_factor']=='1.0'),key=lambda r:r['G_point'])
                maxD={s:[v for r,v in zip(rs,values) if str(r['max_order'])==s] for s in prob}
                histD={s:[v for r,v in zip(rs,values) if (r['n2'],r['n4'])==s] for s in hist}
                max_mix=mixture(prob,maxD,pf['G_point']);hist_mix=mixture(hist,histD,pf['G_point'])
                zero=stats(maxD['0'])['mean'];prare=math.fsum((prob['2'],prob['4']))
                rare_break=(pf['G_point']-prob['0']*zero)/prare
                M=hist_mix['unobserved_probability_mass'];known=hist_mix['known_mass_plugin']
                scenarios=[dict(unobserved_histogram_conditional_mean_multiplier_of_zero=k,
                    resulting_PR_PF_ratio=(known+M*k*zero)/pf['G_point'],is_bound=False) for k in (1,2,10,100)]
                scalar_rows.append(dict(cell_id=cid,phase=phase,epsilon=eps,trajectories=len(rs),G=st,
                    shot_counts=N,N_total=sum(N.values()),PF_cell_id=pf['cell_id'],PF_G=pf['G_point'],PR_PF_ratio=st['mean']/pf['G_point'],
                    max_order_mixture=max_mix,histogram_mixture=hist_mix,
                    rare_union_break_even_G_conditional=rare_break,
                    rare_union_break_even_RZ_per_shot=rare_break/sum(N.values()),
                    rare_union_break_even_multiple_of_zero=rare_break/zero,
                    empirical_zero_G=zero,expected_rare_quantum_execution_count=sum(N.values())*prare,
                    unobserved_histogram_sensitivity=scenarios,
                    hypothetical_RZ_per_native_instruction_needed_for_global_expected_bound=pf['G_point']/(sum(N.values())*expected_I),
                    hypothetical_instruction_conversion_is_proven=False))
    for eps in (.05,.01,.005,.001):
        prior=next(x for x in selection['rankings'] if x['epsilon']==eps)['top2'][0]['cell_id']
        available=[x for x in scalar_rows if x['epsilon']==eps and x['phase']=='confirmation'];best=min(available,key=lambda r:r['G']['mean'])
        fixed=next(x for x in available if x['cell_id']==prior)
        comparison.append(dict(epsilon=eps,preselected_exploration_winner=prior,
            fixed_winner_confirmation_G=fixed['G']['mean'],fixed_winner_PR_PF_ratio=fixed['PR_PF_ratio'],
            postconfirmation_point_minimum=best['cell_id'],postconfirmation_PR_PF_ratio=best['PR_PF_ratio'],
            confirmation_minimum_is_postselection=True,rare_union_break_even_RZ_per_shot=best['rare_union_break_even_RZ_per_shot'],
            rare_union_break_even_multiple_of_zero=best['rare_union_break_even_multiple_of_zero'],
            unknown_histogram_mass=best['histogram_mixture']['unobserved_probability_mass'],
            unknown_histogram_break_even_RZ_per_shot=best['histogram_mixture']['unknown_common_conditional_mean_break_even']/best['N_total']))
    all_explore=[]
    for (cid,phase),rs in sorted(by.items()):
        if phase!='exploration':continue
        observed=Counter(k for r in rs for k in r['orders'])
        all_explore.append(dict(cell_id=cid,trajectories=len(rs),observed_order_counts=dict(observed),
            trajectories_containing_order2=sum(r['n2']>0 for r in rs),trajectories_containing_order4=sum(r['n4']>0 for r in rs),
            other_cell_costs_are_not_a_bound_for_selected_cells=True))
    write('selected_trajectory_cost_evidence_v1.json',dict(schema='saved_trajectory_order_component_cost_extract_v1',
        records=[r for r in records if r['cell_id'] in selected],observed_component_variants=variants,
        fresh_trajectories_generated=0,scientific_arrays_decoded=0))
    write('component_and_structure_v1.json',dict(catalogs=catalogs,archive_sha256=sha(ARCHIVE),arrays_decoded=False))
    write('other_cell_order4_evidence_v1.json',dict(records=[r for r in records if r['n4'] and r['cell_id'] not in selected],
        component_variants=other_order4_variants,role='Context only; different q/r/time; not reused in selected-cell estimates or as a bound'))
    write('cost_reliability_analysis_v1.json',dict(schema='h6_rare_order_cost_analysis_v1',
        base_result_commit='321cd8c3066194fe98741cbebbcdba1515d1deaa',science_source=manifest['base_manifest']['science_source'],
        comparison=comparison,cell_phase_diagnostics=diagnostics,epsilon_cell_phase_rows=scalar_rows,
        all_72_exploration_order_coverage=all_explore,
        method='Exact saved-law mixture identity; empirical conditional plug-ins; missing means remain null; breakpoints and multiplier scenarios are assumptions, not bounds',
        population_winner='UNDETERMINED_POPULATION_MEAN',compiled_cost_population_bound=None,
        new_trajectory_calls=0,new_transpiles=0,new_grant=False,next_stage_authorized=False))
    with (HERE/'order_histogram_probability_v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(histogram_rows[0]),lineterminator='\n');w.writeheader();w.writerows(histogram_rows)
    with (HERE/'trajectory_metrics_v1.csv').open('x',newline='') as f:
        fields=['cell_id','phase','replica','seed','n2','n4','max_order','RZ_cosine','RZ_sine','event_digest','cost_result_sha256','trajectory_sha256','source_commit']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in records:
            w.writerow({k:r['metrics'][k[3:]]['RZ'] if k.startswith('RZ_') else r[k] for k in fields})
    write('analysis_audit_v1.json',dict(status='LOCAL_SAVED_METADATA_AND_SCALAR_ANALYSIS_COMPLETE',
        saved_groups_read=832,selected_groups_published=320,exploration_trajectories=576,confirmation_trajectories=256,
        source_closure_unchanged=len(source_hashes),science_source_unchanged=True,
        exploration_confirmation_pooled=False,selected_before_confirmation=selection['selected_before_confirmation'],
        designated_scientific_review_sha256=sha(REVIEW),scientific_review_repeated=False,
        analysis_process=dict(AS_bytes=2*2**30,threads=1,nice=10),new_science_calls=0,new_transpiles=0,
        scientific_arrays_decoded=0,CI=False,formal_cost_bound_proven=False,
        input_artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            (SUMMARY/'result_summary_v1.json',SUMMARY/'confirmation_resource_map_v1.json',SUMMARY/'integrated_confirmation_v1.json',
             EXPLORATION/'integrated_exploration_v1.json',EXPLORATION/'exploration_resource_map_v1.json')},
        parent_manifest_sha256=sha(RUN/'frozen_manifest.json'),
        generated_file_hashes={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}))
    print(json.dumps(dict(groups=832,selected=320,comparison=comparison,
        observed_order4_exploration_cells=[x['cell_id'] for x in all_explore if x['trajectories_containing_order4']]),ensure_ascii=False))


if __name__=='__main__':main()
