"""Read-only saved-value certificate, sequence identity, and comparison audit.

No candidate generation, synthesis, circuit lowering, matrix, LP or old runner.
New report artifacts only. Recomputed comparisons do not alter frozen results.
"""
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('g3_saved_verifier',Path(__file__).with_name('g3_finite_law.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
O=a.OUT
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
from trottertracks.algorithm_codesign.rte_reallocation.numeric import validate_saved,synthesis_key

def read(name):return json.loads((O/name).read_bytes())
def sha(path):return sha256(path.read_bytes()).hexdigest()
def subset_record(row):
    return {k:row[k] for k in ('id','x','arm','precision','optimized_axis','proposal_rule',
        'resource_total','expected_cost','shots_per_axis','m2_exact','L_exact','synthesis_bias_upper')}

def run():
    # Immutable source/inputs verify before arithmetic; no execution controller.
    for scope in ('scope_v1.json','phase_B_scope.json'):
        for path,h in read(scope)['fixed_hashes'].items():
            if sha(ROOT/path)!=h:raise PermissionError('binding changed: '+path)
    ar,br=read('phase_A_result.json'),read('phase_B_result.json')
    if ar['status']!='G3_PHASE_A_FINITE_LAW_DIAGNOSIS_COMPLETE' or br['status']!='G3_KNOWN_RETURN_COMPARATOR_COMPLETE':
        raise PermissionError('partial run not used as final comparison')
    old=json.loads((ROOT/a.g.RAW).read_bytes());cache={v['key']:v for v in old['synthesis_rows']}
    for v in br['synthesis_rows']:
        kind,value,_,scale=v['angle_key'].split(':')
        validate_saved(v,Angle(kind,F(value),F(scale)),v['epsilon']);cache[v['key']]=v
    if len(br['synthesis_rows'])!=12 or len(br['event_rows'])!=12:raise ValueError('return inventory')
    # Native counts independently from two CRZ primitives and saved conjugators.
    for row in br['event_rows']:
        eps=row['epsilon'];ratio=F(row['returned_ratio']);label=row['label']
        seqs=[cache[synthesis_key(Angle('atan',ratio,F(sign)),eps)] for sign in (-1,1)]
        if label==1:seqs += [cache[synthesis_key(Angle('pi',F(1,8),F(sign)),eps)] for sign in (-1,1)]
        T=sum(z['T_count'] for z in seqs);one=sum(z['one_qubit_count'] for z in seqs)+(8 if label else 0)
        error=sum(F(z['strict_operator_error_upper']) for z in seqs)
        cost=row['native_cost']
        if (cost['T'],cost['CX'],cost['1Q'],F(cost['strict_event_error_upper']))!=(T,6 if label else 2,one,error):
            raise ValueError('returned native saved count mismatch')
        if len(cost['IR_sha256'])!=64:raise ValueError('missing IR identity')
    table=json.loads((ROOT/a.g.TABLE).read_bytes())
    oldcols={xs:{c['id']:c for c in t['columns']} for xs,t in table['tables'].items()}
    cols=read('phase_B_return_columns.json')
    # Same-word returns removed; remaining source IR/phase/error unchanged.
    reused=0
    for xs,cs in cols.items():
        for c in cs.values():
            if c['prototype']!='RET2':continue
            source=oldcols[xs][c['source_O2_id']]
            if c['source_O2_identity']!=source['implementation_identity_sha256']:raise ValueError('O2 identity')
            for e in c['events']:
                z=next(z for z in source['events'] if z['source_label']==e['source_label'])
                if e['word'][0]==e['word'][1] or F(e['label_probability'])!=F(z['label_probability'])/F(3,8):raise ValueError('conditional support')
                if any(e[k]!=z[k] for k in ('native_cost','word','rotation','phase_i_power','rotation_sign','complement')):raise ValueError('O2 phase/IR changed')
                reused+=1
    saved_A=read('phase_A_best_per_profile.json');saved_B=read('phase_B_best_laws.json')
    complete=read('phase_A_selected_laws.json')+saved_B
    max_mean=F(0);max_bits=0
    for c in complete:
        proof=a.certify(c,source_columns=cols[c['x']] if c['arm']=='known_return' else oldcols[c['x']])
        if any(proof[k]!=c['certificate'][k] for k in ('resource_vector_exact','confidence_margin_exact','degree_mean_residual_upper','m2_exact','L_exact')):raise ValueError('certificate receipt mismatch')
        max_mean=max(max_mean,F(proof['degree_mean_residual_upper']))
        for v in map(F,c['weights_exact']):max_bits=max(max_bits,v.numerator.bit_length(),v.denominator.bit_length())
        if c['arm']!='known_return':
            ref=next(z for z in saved_A if z['id']==c['id'])
            if ref['law_digest']!=a.g.digest({'events':c['event_source'],'q':c['q_exact'],'weights':c['weights_exact']}):raise ValueError('A law digest')
    # Two parallel saved-value summaries: reported per-axis-winner pool, plus
    # stricter same-optimized-axis pool. Neither is all-B2/all-IS optimality.
    comparisons={}
    for mode in ('saved_axis_winner_pool','same_optimized_axis_only'):
        comparisons[mode]={}
        for xs in ('1/8','1/4'):
            comparisons[mode][xs]={}
            for axis in a.AXES:
                rows=[z for z in saved_A+saved_B if z['x']==xs and (mode=='saved_axis_winner_pool' or z['optimized_axis']==axis)]
                winners={arm:min([z for z in rows if z['arm']==arm],key=lambda z:F(z['resource_total'][axis])) for arm in a.ARMS+('known_return',)}
                oldwin=min([winners[k] for k in a.ARMS if k!='J1'],key=lambda z:F(z['resource_total'][axis]))
                j=winners['J1'];ret=winners['known_return'];strong=min((oldwin,ret),key=lambda z:F(z['resource_total'][axis]))
                comparisons[mode][xs][axis]={'winners':{arm:subset_record(z) for arm,z in winners.items()},
                    'original_best_arm':oldwin['arm'],'strongest_known_in_pool_arm':strong['arm'],
                    'J1_over_original_exact':str(F(j['resource_total'][axis])/F(oldwin['resource_total'][axis])),
                    'J1_over_strongest_known_exact':str(F(j['resource_total'][axis])/F(strong['resource_total'][axis])),
                    'return_over_J1_exact':str(F(ret['resource_total'][axis])/F(j['resource_total'][axis])),
                    'J1_nondominated_by_known_pool':not any(a.dominates(z,j) for z in rows if z['arm']!='J1'),
                    'no_global_B2_or_sampler_optimality':True,'no_scientific_materiality_threshold_applied':True}
    a.dump(O/'saved_comparison_audit.json',{'status':'G3_SAVED_COMPARISON_AUDIT_PASS',
        'full_laws_reverified':len(complete),'A_saved_axis_winners':len(saved_A),'B_saved_axis_winners':len(saved_B),
        'new_sequences_identity_count_verified':12,'new_native_saved_cost_rows_verified':12,
        'offdiagonal_source_event_identities_reused':reused,'largest_mean_residual_upper':str(max_mean),
        'selected_max_corrected_weight_numerator_or_denominator_bits':max_bits,
        'comparisons':comparisons,'new_candidates_synthesis_circuit_matrix_LP_GPU_calls':0,
        'prior_results_status_not_changed':True,'mandatory_STOP':True,'research_owner':'GPT_G3'})
    print(json.dumps({'status':'SAVED_AUDIT_PASS','full_laws':len(complete),'return_sequences':12,'O2_events':reused}))

if __name__=='__main__':run()
