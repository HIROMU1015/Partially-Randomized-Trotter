"""Copy saved selections and compare their stored totals; stdlib only.

No project imports, raw sample analysis, shot/cost evaluation, minima search,
candidate sorting, coefficient evaluation, circuit construction, or compile.
"""
import ast
import copy
import csv
from decimal import Decimal
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD='5d53698534a1f325c8b7ed7be1bca6fa5192a77d'
HIGH='ee160395e87ed6d46cd4993572457afd0aa18590'
COST='8613b3ecda2030a677b1690fe7aa3da23dec4873'
PROVIDER='217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
O='docs/progress/track_a/figures/all_methods_hchain_20261022/'
H='docs/progress/track_a/figures/all_methods_hchain_with_higher_pr_20261022/'
C='artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/'
EPS=['0.05','0.01','0.005','0.001','0.0005']
SYSTEMS=['H2','H4','H6']

def file(commit,path):return ROOT/'sources'/commit/path
def read(commit,path):
    p=file(commit,path)
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if path.endswith('.gz') else b)
def rows(commit,path):
    with file(commit,path).open(newline='') as f:return list(csv.DictReader(f))
def dump(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(name,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rr:w.writerow({k:'null' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def pointer(x,p):
    for k in p.lstrip('/').split('/'):
        k=k.replace('~1','/').replace('~0','~')
        x=x[int(k)] if isinstance(x,list) else x[k]
    return x
def ep(x):return str(Decimal(str(x)).normalize())
def exact(s):return str(s['numerator'])+'/'+str(s['denominator'])
def url(c,p):return 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+c+'/'+p

old=read(OLD,O+'figure_data.json')
high=read(HIGH,H+'figure_data.json')
evaluation=read(COST,C+'evaluation.json')
formula=rows(COST,C+'formula_comparison_30.csv')
candidate=rows(COST,C+'candidate_precision_costs_1000.csv')
decomposition=rows(COST,C+'cost_decomposition_60.csv')
h2c='829d697ae7905b3ea90a61da9d2d4da15d8b757a'
h2p='artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_completed_run02/'
h4c='406f70570323366da185355550a3900388717fbb'
h4p='artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/'
h6c='7430a386bcd4dc545107e413cd022ab7cb261968'
h6p='artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/candidate_accounting.json.gz'
h2=read(h2c,h2p+'result.json')
h4=read(h4c,h4p+'result.json')
h4signals=read(h4c,h4p+'signals.json.gz')['signals']
h4pairs=read(h4c,h4p+'paired_measurements.json.gz')['rows']
h4reuse=read(h4c,'artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/FP_reuse_evidence.json')
h6=read(h6c,h6p)
records=[]
lookup={}

for i,r in enumerate(old['rows']):
    if r['system'] not in SYSTEMS or not r['series'].startswith('full PF'):continue
    src=r['result_source'];origin=pointer(read(src['commit'],src['path']),src['locator']) if r['system']!='H6' else rows(src['commit'],src['path'])[int(src['locator'].split()[3].rstrip(';'))-1]
    cid=r['candidate_id'];epsilon=r['epsilon']
    assert origin.get('cell_id',origin.get('candidate_id'))==cid
    if r['system']!='H6':
        assert origin['eligible'] and ep(origin['epsilon'])==ep(epsilon)
        assert origin['G_RZ_exact']==r['cost_exact']
    else:
        assert Fraction(origin['replacement_G_exact'])==Fraction(r['cost_exact'])
    row=dict(comparison='A',system=r['system'],epsilon=epsilon,method='Full PF',formula='S2' if r['PF_order']=='S2' else 'Yoshida4',candidate_id=cid,prefix=r['L_D'],DF_rank=r['DF_rank'],q=r['q'],delta=r['delta'],K=None,r_policy=None,r_value=None,n=r['n'],saved_cost_exact=r['cost_exact'],saved_cost_display=r['cost_display'],eligibility='ELIGIBLE',scientific_SOURCE=r['scientific_SOURCE'],result_commit=src['commit'],result_path=src['path'],record_locator=src['locator'],figure_commit=OLD,figure_locator='/rows/'+str(i),retrieval='ACQUIRED_FIXED_BLOB_HASH_VERIFIED',identity_status='PARTIAL_NOT_SCIENTIFICALLY_CERTIFIED',cost_class=r['cost_class'],exact_ties=r['exact_ties'],evidence_details=copy.deepcopy(r['evidence_details']))
    if r['system'] in ('H2','H4'):
        ss=h2['signals'] if r['system']=='H2' else h4signals
        si,s=next((j,s) for j,s in enumerate(ss) if s['cell']['id']==cid)
        pr=s['precision_rows'][epsilon]['1.0']
        assert pr['cost_acquisition_eligible']
        row.update(N_cos=origin['shots']['cosine'],N_sin=origin['shots']['sine'],N_total=pr['N_total'],signed_bias_cos=s['error_decomposition']['total_signed']['real'],signed_bias_sin=s['error_decomposition']['total_signed']['imag'],allowance_cos=s['allowance']['real'],allowance_sin=s['allowance']['imag'],log_B=s['log_B'],log_B_margin=s['allowance']['log_B_margin'],headroom_cos=pr['axes']['real']['headroom'],headroom_sin=pr['axes']['imag']['headroom'],signal_commit=h2c if r['system']=='H2' else h4c,signal_path=h2p+'result.json' if r['system']=='H2' else h4p+'signals.json.gz',signal_locator='/signals/'+str(si))
        row['signal_scientific_SOURCE']=h2['original_SOURCE'] if r['system']=='H2' else h4reuse['SOURCE'] if cid in h4reuse['signal_records'] else h4['SOURCE']
        if r['system']=='H2':
            bucket='measurements' if any(m['cell_id']==cid for m in h2['measurements']) else 'reused_FP_measurements'
            mi,m=next((j,m) for j,m in enumerate(h2[bucket]) if m['cell_id']==cid)
            row.update(mean_cos_RZ=m['axes']['cosine']['RZ'],mean_sin_RZ=m['axes']['sine']['RZ'],unit_locator='/'+bucket+'/'+str(mi),unit_commit=h2c,unit_path=h2p+'result.json',unit_scientific_SOURCE=h2['SOURCE'] if bucket=='measurements' else h2['original_SOURCE'])
        else:
            matches=[(j,m) for j,m in enumerate(h4pairs) if m['cell_id']==cid]
            if matches:
                mi,m=matches[0];unit={a['axis']:a['metrics']['RZ'] for a in m['rows']}
                row.update(mean_cos_RZ=unit['cosine'],mean_sin_RZ=unit['sine'],unit_locator='/rows/'+str(mi),unit_commit=h4c,unit_path=h4p+'paired_measurements.json.gz',unit_scientific_SOURCE=r['scientific_SOURCE'])
            else:
                unit={a['axis']:a['metrics']['RZ'] for a in next(m['axes'] for m in h4reuse['measured_axes'] if m['cell_id']==cid)}
                row.update(mean_cos_RZ=unit.get('cosine'),mean_sin_RZ=unit.get('sine'),unit_locator='/measured_axes (cell_id filter)',unit_commit=h4c,unit_path='artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/FP_reuse_evidence.json',unit_scientific_SOURCE=h4reuse['SOURCE'])
    else:
        loc=r['evidence_details']['original_accounting_locator'];saved=pointer(h6,loc);ac=saved['accounting']
        assert saved['cell_id']==cid and saved['eligible'] and ep(saved['epsilon'])==ep(epsilon)
        ev=r['evidence_details']['unit_evidence'];unit={u['axis']:u['replacement_RZ_literal'] for u in ev}
        row.update(N_cos=ac['axes']['real']['shots'],N_sin=ac['axes']['imag']['shots'],N_total=saved['N_total'],allowance_cos=ac['axes']['real']['allowance'],allowance_sin=ac['axes']['imag']['allowance'],headroom_cos=ac['axes']['real']['headroom'],headroom_sin=ac['axes']['imag']['headroom'],mean_cos_RZ=unit['cosine'],mean_sin_RZ=unit['sine'],signed_bias_cos=None,signed_bias_sin=None,log_B=None,log_B_margin=None,bias_metadata_status='NOT_PRESENT_IN_CITED_H6_ACCOUNTING_ROW; NO_RECONSTRUCTION_FROM_HEADROOM',accounting_commit=h6c,accounting_path=h6p,accounting_locator=loc)
    records.append(row);lookup[('A',row['system'],epsilon,row['formula'])]=row

selected_copies=[]
for rid,r in high['records'].items():
    if r['domain']!='OUTER32' or r['system'] not in SYSTEMS or r['mode']!='USUAL_SIGNED' or r['series'] not in ['PR S2','PR Yoshida4','PR m5_best']:continue
    ff={'PR S2':'S2','PR Yoshida4':'Yoshida4','PR m5_best':'m5_best'}[r['series']]
    fi=next(j for j,f in enumerate(formula) if f['mode']=='USUAL_SIGNED' and f['system']==r['system'] and f['epsilon']==r['epsilon'])
    f=formula[fi];key='m5' if ff=='m5_best' else ff
    assert f[key+'_candidate']==r['candidate_id']
    ci,can=next((j,z) for j,z in enumerate(candidate) if z['candidate_id']==r['candidate_id'] and z['epsilon']==r['epsilon'] and z['mode']=='USUAL_SIGNED')
    assert can['status']=='ELIGIBLE'
    assert Fraction(can['total_RZ_numerator']+'/'+can['total_RZ_denominator'])==Fraction(r['cost_exact'])
    src=r['source'];s=pointer(evaluation,src['locator']);assert s['candidate_id']==r['candidate_id']
    assert Fraction(exact(s['total_RZ_exact']))==Fraction(r['cost_exact'])
    row=dict(comparison='B',system=r['system'],epsilon=r['epsilon'],method='PR',formula=ff,candidate_id=r['candidate_id'],prefix=r['prefix'],DF_rank=r['DF_rank'],q=r['q'],delta=r['delta_exact'],K=r['K'],r_policy=r['r_policy'],r_value=r['r_value'],n=r['n'],saved_cost_exact=r['cost_exact'],saved_cost_display=can['total_RZ'],eligibility=can['status'],scientific_SOURCE=r['scientific_SOURCE'],acquisition_SOURCE=r['acquisition_SOURCE'],result_commit=src['commit'],result_path=src['path'],record_locator=src['locator'],figure_commit=HIGH,figure_locator='/records/'+rid,csv_data_row=ci+1,csv_path=C+'candidate_precision_costs_1000.csv',selection_csv_data_row=fi+1,retrieval='ACQUIRED_FIXED_BLOB_HASH_VERIFIED',identity_status='INTRA_OUTER32_FIXED_SET_ONLY',cost_class='DIRECT_MEASURED_SAMPLE_EXPLORATORY_MEAN',exact_ties=r['exact_ties'])
    for k in ['N_cos','N_sin','N_total','mean_cos_RZ','mean_sin_RZ','signed_bias_cos','signed_bias_sin','u','log_B','log_B_margin','log_B_for_shots','headroom_cos','headroom_sin','effective_unit_RZ','total_RZ_SE']:
        row[k]=can[k]
    records.append(row);lookup[('B',row['system'],row['epsilon'],ff)]=row
    selected_copies.append(dict(record_id=rid,figure_record=copy.deepcopy(r),saved_evaluation_selection=copy.deepcopy(s),saved_candidate_csv_row=copy.deepcopy(can),saved_formula_csv_row=copy.deepcopy(f)))
assert len(records)==75
csvout('evidence_rows_75.csv',records)
dump('selected_record_copies.json',selected_copies)

matrix=[]
for sy in SYSTEMS:
    for e in EPS:
        a0=lookup['A',sy,e,'S2'];a1=lookup['A',sy,e,'Yoshida4']
        b0=lookup['B',sy,e,'S2'];b1=lookup['B',sy,e,'Yoshida4']
        a='S2_LOWER' if Fraction(a0['saved_cost_exact'])<Fraction(a1['saved_cost_exact']) else 'YOSHIDA4_LOWER'
        b='S2_LOWER' if Fraction(b0['saved_cost_exact'])<Fraction(b1['saved_cost_exact']) else 'YOSHIDA4_LOWER'
        classification='REVERSAL_CANDIDATE' if a=='S2_LOWER' and b=='YOSHIDA4_LOWER' else 'SAME_DIRECTION' if a==b else 'OPPOSITE_DIRECTION_OTHER_THAN_REQUESTED_C'
        row=dict(system=sy,epsilon=e,A_saved_order=a,B_saved_order=b,classification=classification,compatibility='PARTIAL_IDENTITY; ALLOWANCE_POLICY_DIFFERENCE; H2_COMPILER_GUARD_DIFFERENCE' if sy=='H2' else 'PARTIAL_IDENTITY; ALLOWANCE_POLICY_DIFFERENCE; NATIVE_RUNTIME_NOT_FULLY_JOINED' if sy=='H4' else 'PARTIAL_IDENTITY; CONDITIONAL_PROFILE_RUNTIME; S4_COMPOSITION_HYPOTHESIS',scientific_reversal_C='NOT_CERTIFIED',evidence_acquired='COMPLETE_SELECTED_VALUES; H6_BIAS_METADATA_PARTIAL' if sy=='H6' else 'COMPLETE_SELECTED_VALUES',A_S2_cost=a0['saved_cost_exact'],A_S2_q=a0['q'],A_Y4_cost=a1['saved_cost_exact'],A_Y4_q=a1['q'],B_S2_cost=b0['saved_cost_exact'],B_S2_q=b0['q'],B_Y4_cost=b1['saved_cost_exact'],B_Y4_q=b1['q'])
        matrix.append(row)
csvout('comparison_ABC_15.csv',matrix)

aux=[]
for i,r in enumerate(old['rows']):
    if r['system'] in SYSTEMS and r['series'].startswith(('B0,','B0+I,')):
        aux.append(dict(system=r['system'],epsilon=r['epsilon'],series=r['series'],candidate_id=r['candidate_id'],q=r['q'],cost_exact=r['cost_exact'],status=r['status'],locator='/rows/'+str(i),source=copy.deepcopy(r.get('result_source')),reason=r.get('missing_reason',r.get('reason'))))
assert len(aux)==60
csvout('B0_B0I_auxiliary_60.csv',aux)
csvout('saved_cost_decomposition_usual_30.csv',[x for x in decomposition if x['mode']=='USUAL_SIGNED'])

identity=[]
binding=read(COST,C+'input_provenance.json')['input_metadata']
for sy in SYSTEMS:
    a=next(r for r in old['rows'] if r['system']==sy and r['series']=='full PF, S2')
    b=next(r['binding'] for r in binding if r['system']==sy)
    for ak,bk in [('hamiltonian_hash','hamiltonian'),('state_hash','state'),('snapshot_hash','snapshot'),('compiler_settings_hash','compiler')]:
        identity.append(dict(system=sy,field=bk,A=a[ak],B=b[bk],equal=a[ak]==b[bk],scope='METADATA_HASH_MATCH_ONLY'))
assert all(x['equal'] for x in identity)
csvout('identity_hashes_12.csv',identity)

def function(c,p,name):
    tree=ast.parse(file(c,p).read_text())
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.dump(n),n.lineno,n.end_lineno
oldshot='src/trottertracks/resource_applicability/ax2b_numerical_accounting.py'
newshot='scripts/d0_frozen_shot_rule_v1.py'
checks=[]
for name,p in [('u_aware_shots',oldshot),('_number',oldshot),('axis_headroom','src/trottertracks/resource_applicability/ax2a_preparation.py')]:
    a,a1,a2=function(PROVIDER,p,name);b,b1,b2=function('9a77d2b0a0b87e246e85f085b37f5923c18e27bb',newshot,name)
    checks.append(dict(function=name,AST_equal=a==b,A_commit=PROVIDER,A_path=p,A_line=a1,B_commit='9a77d2b0a0b87e246e85f085b37f5923c18e27bb',B_path=newshot,B_line=b1))
dump('shot_static_checks.json',checks)

out=['# 比較A/B/C：保存採用点の対応表','','費用は保存compiled RZ総費用。括弧内は保存q。小数は保存表示値を転記し、新しい費用・比率は計算していない。Y4=Yoshida4。全行の科学的比較Cは未認定。','','| 系 | ε | A Full S2（q） | A Full Y4（q） | B PR S2（q） | B PR Y4（q） | 保存順位分類 |','|---|---:|---:|---:|---:|---:|---|']
for m in matrix:
    sy,e=m['system'],m['epsilon']
    vals=[]
    for side,ff in [('A','S2'),('A','Yoshida4'),('B','S2'),('B','Yoshida4')]:
        r=lookup[side,sy,e,ff];vals.append(str(r['saved_cost_display'])+' (q'+str(r['q'])+')')
    out.append('| '+' | '.join([sy,e,*vals,m['classification']])+' |')
out+=['','全75採用recordのID・prefix・rank・δ・K・outer32・n・shot・bias・u・normalization・単価・exact総費用・元commit/locatorは [evidence_rows_75.csv](evidence_rows_75.csv)。',
      '15条件の順位とidentity適合性を別列にした表は [comparison_ABC_15.csv](comparison_ABC_15.csv)。m5採用15行も75行表に別formulaとして含む。',
      'B0/B0+Iの60枠は [B0_B0I_auxiliary_60.csv](B0_B0I_auxiliary_60.csv) へ分離。H2のε=.01/.005/.001/.0005の16枠は元のnullを保持した。']
(ROOT/'TABLES.md').write_text('\n'.join(out)+'\n')
print(json.dumps(dict(selected_records=len(records),comparison_conditions=len(matrix),classifications={s:sum(m['classification']==s for m in matrix) for s in {m['classification'] for m in matrix}},auxiliary_slots=len(aux),static_shot_checks=checks),ensure_ascii=False,indent=2))
