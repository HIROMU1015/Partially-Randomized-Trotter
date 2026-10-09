"""G4 STOP後の保存値・sequence identity・provenance照合だけ。Acquisitionなし。"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import csv
import importlib.util
import json
import subprocess

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('b_saved',Path(__file__).with_name('g4_matched_cts.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
P=b.OUT
OLD=ROOT/'artifacts/track_b_g3_finite_law/2026-10-09'
APPEND_ONLY={'PROJECT_MAP.md','docs/README.md','docs/tracks/algorithm_codesign/README.md',
    'docs/research/研究概要・現状.md','scripts/README.md','scripts/tracks/algorithm_codesign/README.md',
    'src/trotterlib/README.md','docs/research/研究ノート/README.md','tests/tracks/algorithm_codesign/README.md'}

def load(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()

def audit():
    hashes=load(P/'prior_evidence_hashes.json')
    appended=[]
    for n,h in hashes.items():
        if digest(ROOT/n)==h:continue
        if n not in APPEND_ONLY:raise ValueError('prior evidence changed: '+n)
        base=subprocess.check_output(['git','show','3b0fa70b47848e72ef9a9c9e13afc3164962f7cd:'+n],cwd=ROOT)
        if sha256(base).hexdigest()!=h or not (ROOT/n).read_bytes().startswith(base):raise ValueError('old index body changed: '+n)
        appended.append(n)
    for scope in ('scope_A.json','scope_B.json'):
        if any(digest(ROOT/n)!=h for n,h in load(P/scope)['fixed_hashes'].items()):raise ValueError('frozen source changed')
    a=load(P/'result_A.json');r=load(P/'result_B.json');defs=load(P/'cts_definition.json')
    if a['status']!='G4_A_STATED_CLASS_SEPARATION_CERTIFIED' or r['status']!='G4_B_MATCHED_CTS_COMPLETE':raise ValueError('completed status missing')
    Aprofiles=load(P/'independent_profiles.json')
    for xs,s in a['summary'].items():
        for name,axis,readout in (('T_native','T',False),('1Q_native','1Q',False),('1Q_readout','1Q',True)):
            ps=[p for p in Aprofiles if (p['x'],p['axis'],p['readout_price_added'])==(xs,axis,readout)]
            assert len(ps)==63
            lower=min(F(p['ratio']['lo']) for p in ps)
            assert lower==F(s[name]['r_enclosure']['lo'])
            assert F(s[name]['budget_policy_cost_lower'])==37*lower*lower
            assert F(s[name]['digital_bridge_min_slack'])>=0
        c=s['J1_finite_certificate'];t=F(c['resource_totals']['T']);one=F(c['resource_totals']['1Q'])
        assert s['strict_T_separation']==(t<F(s['T_native']['budget_policy_cost_lower']))
        assert s['strict_1Q_separation']==(one<F(s['1Q_readout']['budget_policy_cost_lower']))
    keys=b.planned_keys(defs);cache={z['key']:z for z in r['synthesis_rows']}
    assert len(cache)==12 and set(cache)==set(keys)
    for key,(angle,eps) in keys.items():b.native_cost.__globals__['validate_saved'](cache[key],angle,eps)
    # Source-derived additive cost formula; no circuit builder or matrix guard.
    for e in r['event_rows']:
        d=defs[e['x']];ref=next(z for z in d['events'] if z['label']==e['label'])
        c=e['native_cost']
        if ref['real_event']:
            T,CX,one,err,gates=(0,0,1,F(0),1) if ref['axis']=='II' else (0,2,5,F(0),7)
        else:
            ratio=F(d['fixed_rational_rotation_ratio'])
            rows=[cache[b.synthesis_key(b.Angle('atan',ratio,F(sign)),e['epsilon'])] for sign in (-1,1)]
            T=sum(z['T_count'] for z in rows);one=sum(z['one_qubit_count'] for z in rows)
            err=sum((F(z['strict_operator_error_upper']) for z in rows),F(0))
            if ref['axis'] in ('ZI','IZ'):CX,gates=2,4
            else:CX,gates=4,12;one+=6
        assert (c['T'],c['CX'],c['1Q'],F(c['strict_event_error_upper']),c['IR_gate_count'])==(T,CX,one,err,gates)
    selected=load(P/'CTS_selected_complete_laws.json')
    for c in selected.values():assert b.certify(c,defs[c['x']],r['event_rows'])==c['certificate']
    best=load(P/'CTS_best_per_profile.json');assert len(best)==486
    for c in best:
        s=F(c['remaining']);m2=F(c['m2_exact']);L=F(c['L_exact']);n=c['shots_per_axis']
        margin=n*s*s-b.ELL*(2*m2+F(4,3)*L*s)
        assert margin==F(c['certificate']['confidence_margin']) and margin>=0
        assert n==(b.ELL*(2*m2/(s*s)+F(4,3)*L/s)).__ceil__()
        for k in b.AXES:assert F(c['resource_total'][k])==n*(2*F(c['expected_cost'][k])+(5 if k=='1Q' else 0))
    old=load(OLD/'phase_A_best_per_profile.json');returns=load(OLD/'phase_B_best_laws.json')
    records=[];comparison={};dominance={}
    old_complete=load(OLD/'phase_A_selected_laws.json')
    for xs in defs:
        comparison[xs]={}
        for arm,pool in [(arm,old) for arm in ('ordinary','PTSC_K0','A','J1')]+[('return',returns),('CTS_literal_M3',best)]:
            comparison[xs][arm]={}
            for axis in b.AXES:
                eligible=[c for c in pool if c['x']==xs and c['optimized_axis']==axis and (arm in ('return','CTS_literal_M3') or c['arm']==arm)]
                c=min(eligible,key=lambda c:F(c['resource_total'][axis]))
                comparison[xs][arm][axis]={k:c[k] for k in ('precision','shots_per_axis','resource_total','expected_cost','m2_exact','L_exact','workspace_peak')}
                records.append({'x':xs,'arm':arm,'selection_axis':axis,'shots':c['shots_per_axis'],**{k:float(F(z)) for k,z in c['resource_total'].items()}})
        J=next(c for c in old_complete if (c['x'],c['arm'],c['optimized_axis'])==(xs,'J1','1Q'))
        C=selected[xs+'/1Q']
        dominance[xs]={'CTS_1Q_selected_vs_G3_same_J1_1Q_selected':
            {k:F(C['resource_total'][k])<F(J['resource_total'][k]) for k in b.AXES},
            'workspace_equal':C['workspace_peak']==J['workspace_peak'],'same_law_coordinates_not_axis_minima_combined':True,
            'CTS_T_min_vs_J1_T_min_ratio':str(F(comparison[xs]['CTS_literal_M3']['T']['resource_total']['T'])/F(comparison[xs]['J1']['T']['resource_total']['T']))}
    A=next(c for c in old_complete if (c['x'],c['arm'],c['optimized_axis'])==('1/4','A','1Q'))
    J=next(c for c in old_complete if (c['x'],c['arm'],c['optimized_axis'])==('1/4','J1','1Q'))
    dn=2*(J['shots_per_axis']-A['shots_per_axis']);assert dn>0
    crossover={k:str((F(A['resource_total'][k])-F(J['resource_total'][k]))/dn) for k in b.AXES}
    usage=r['usage'];caps=load(P/'scope_B.json')['caps']
    assert usage['wall_seconds']<caps['wall_seconds'] and usage['cpu_seconds']<caps['cpu_seconds']
    assert usage['peak_RSS_KiB']<=caps['RSS_MiB']*1024
    assert r['retries']==0 and r['synthesis_attempts']==12 and r['new_native_event_conditions']==28 and r['profiles']==162
    markerA=load(P/'phase_A_consumed.json');markerB=load(P/'phase_B_consumed.json')
    assert markerA['source']==a['source_commit'] and markerB['source']==r['source_commit']
    assert load(P/'STOP.json')['next_science_stage_authorized'] is False
    payload={'status':'PASS','old_protected_paths_checked':len(hashes),
        'old_protected_paths_exactly_unchanged':len(hashes)-len(appended),'old_indexes_append_only_prefix_preserved':appended,'A_priced_profiles':len(Aprofiles),
        'B_sequences_identity_counts_strict_bound_binding':12,'B_native_event_cost_formula_checks':28,
        'B_complete_selected_laws_recertified':6,'B_saved_profile_axis_winners_accounting_checks':486,
        'new_synthesis_circuit_matrix_LP_signal_sampling_GPU_calls':0,'retries':0,
        'source_A':a['source_commit'],'source_B':r['source_commit'],
        'one_shot_markers_sha256':{n:digest(P/n) for n in ('phase_A_consumed.json','phase_B_consumed.json')},
        'old_scientific_source_result_contract_authorization_marker_unchanged':True,'comparisons_same_optimized_axis_only':comparison,
        'complete_law_dominance':dominance,'fixed_A_J1_common_per_shot_overhead_crossover':crossover,
        'crossover_no_optimization_or_new_overhead_grid':True,'mandatory_STOP':True,'research_owner':'GPT'}
    b.v.write(P/'saved_values_audit.json',payload)
    with (P/'resource_comparison_display.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['x','arm','selection_axis','shots','T','CX','1Q'],lineterminator='\n');w.writeheader();w.writerows(records)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('comparisons_same_optimized_axis_only',)}))

if __name__=='__main__':audit()
