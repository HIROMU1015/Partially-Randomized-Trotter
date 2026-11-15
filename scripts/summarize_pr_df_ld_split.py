"""Persist admitted baseline comparison and held vectors at the SOURCE gate STOP.

Only exact weighting of already frozen prices and reuse of saved Q3 results.
No evaluator, signal, fit, sampling, circuit or compiler imports.
"""
from decimal import Decimal,localcontext
from fractions import Fraction
import csv
import json
from pathlib import Path
import subprocess
import sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[1];PUBLIC='docs/research/d_pr_df_ld_split_results_20261115'
def read(commit,path):return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))
def resolve(commit):return subprocess.check_output(['git','rev-parse',commit],cwd=ROOT,text=True).strip()
def dec(v):return Fraction(v['numerator'],v['denominator'])if isinstance(v,dict)else Fraction(v)
def enc(v):v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)
def display(v):
    with localcontext()as c:c.prec=24;return str(Decimal(v.numerator)/Decimal(v.denominator))
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    base=ROOT/PUBLIC;source=resolve('HEAD')
    if (ROOT/'scripts/summarize_pr_df_ld_split.py').read_bytes()!=subprocess.check_output(['git','show',source+':scripts/summarize_pr_df_ld_split.py'],cwd=ROOT):raise ValueError('Unfixed summary SOURCE')
    inputs=json.loads((base/'ALLOCATION_INPUTS.json').read_text());prices=json.loads((base/'AXIS_PRICES.json').read_text())['rows'];acq=json.loads((base/'acquisition/RESULT.json').read_text())
    completion=json.loads((base/'q3_calls/COMPLETION.json').read_text())
    if completion['receipts'] or len(completion['errors'])!=4:raise ValueError('Unexpected science disposition; do not invent missing risk')
    for c in inputs['cases']:
        error=json.loads((base/'q3_calls'/c['label']/'WORKER_PERSISTENCE_ERROR.json').read_text())
        if error['scientific_evaluation_calls']!=0:raise ValueError('Unexpected started scientific call')
    rows=[];terms=[];best=[]
    for item in inputs['menu']:
        L=item['L_D'];r=dict(item);r['cost_display']=display(dec(r['cost']))
        selected=[]
        for p in prices:
            if p['L_D']!=L:continue
            m,a=p['m'],p['axis'];N=r['shots'][m]['N_'+a];value=N*dec(p['C_ref'])
            t=dict(L_D=L,rule=r['rule'],m=m,axis=a,physical_id=p['physical_id'],N=N,C_ref=p['C_ref'],weighted=enc(value),weighted_display=display(value),
                price_source=dict(commit=resolve('3ac3e09b7'),path=PUBLIC+'/AXIS_PRICES.json',selector=['rows',prices.index(p)]))
            terms.append(t);selected.append(t)
        if sum(dec(t['weighted'])for t in selected)!=dec(r['cost']):raise ValueError('Weighted total mismatch')
        for label,cond in [('m0_m6',lambda m:m<=6),('m7_m12',lambda m:m>=7)]:r[label]=enc(sum(dec(t['weighted'])for t in selected if cond(t['m'])))
        if r['rule']=='V0':
            ref=r['Q3_source'];old=read(ref['commit'],ref['path']);risk=old['record']['diagnostic']['deterministic_risk']
            r.update(status=old['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],MSE_lower_display=display(dec(risk['MSE_lower'])),MSE_upper_display=display(dec(risk['MSE_upper'])),Q3_disposition='REUSED_SAVED_ADMITTED_V0')
            best.append(r)
        else:r.update(status='HELD_BEFORE_SCIENCE_SOURCE_GATE',MSE_lower=None,MSE_upper=None,Q3_disposition='NO_EVALUATION_NO_CLOSURE_NO_DP',reason='Python3.12 AST type_params mismatch with fixed accelerator AST hash; guard not relaxed')
        rows.append(r)
    differences=[]
    for m in range(13):
        v={L:sum(dec(t['weighted'])for t in terms if t['L_D']==L and t['rule']=='V0'and t['m']==m)for L in (1,2)}
        differences.append(dict(m=m,LD1=enc(v[1]),LD2=enc(v[2]),LD2_minus_LD1=enc(v[2]-v[1]),difference_display=display(v[2]-v[1]),scope='SAVED_ADMITTED_V0_BASELINES_ONLY'))
    paths=[]
    for row in acq['rows']:
        rec=row['record'];m,i=row['m'],row['index']
        for j,a in enumerate(rec['records']):
            axis=a['axis'];paths.append(dict(L_D=2,m=m,path_index=i,axis=axis,physical_id=rec['request']['physical_context']['physical_candidate']['id'],cohort=rec['cohort'],seeds=rec['seeds'],S=a['S'],C=a['C'],D=a['D'],event_digest=rec['event_digest'],events=row['events'],fingerprint=a['fingerprint'],gate_status=row['gate_status'],numerical_action_gate=row['numerical_action_gate'],full_wrapper_statevector_validation=False,
                receipt=dict(commit=resolve('b030ea97a'),path=PUBLIC+f'/acquisition/m{m}/p{i}/{dict(X="cosine",Y="sine")[axis]}.json'),
                path_receipt=dict(commit=resolve('b030ea97a'),path=PUBLIC+f'/acquisition/m{m}/p{i}/PATH_RECEIPT.json'),execution_source=resolve('ea9e66092')))
    usage=dict(acquisition_wall_seconds=acq['wall_seconds'],summed_path_CPU_seconds=sum(r['CPU_seconds']for r in acq['rows']),maximum_path_peak_RSS_KiB=max(r['peak_RSS_KiB']for r in acq['rows']),path_wall_seconds=[dict(m=r['m'],index=r['index'],seconds=r['wall_seconds'])for r in acq['rows']],
        scope='Independent 12 path workers. Sum of worker CPU; max RSS is largest single worker, not simultaneous tree peak. Per-worker OS PID/exit signal was not separately persisted by acquisition runner; successful Future and receipt returns recorded.',
        new_Q3_CPU='No scientific work; dispatch 0.017s and SOURCE-gate 0.043s supervisory walls in preserved completion receipts')
    result=dict(schema='PR_DF_LD_SPLIT_FINITE_COMPARISON_PARTIAL_STOP_V1',status='PARTIAL_SOURCE_GATE_STOP',source=source,raw_publication=resolve('b030ea97a'),model_publication=resolve('3ac3e09b7'),allocation_publication=resolve('4ef702742'),Q3_input_publication=resolve('4d3aa8a66'),
        counts=dict(new_paths=12,new_wrapper_compiles=24,valid_compile_receipts=24,reused_target_paths=0,new_axis_models=2,new_LD2_references=26,reused_LD1_references=26,new_vectors=4,reused_V0_Q3=2,new_Q3_calls=0,new_Q3_held=4,closure_calls=0,total_DP_calls=0,science_retries=0),
        menu=rows,minimum_admitted_per_LD=best,minimum_scope='V0 only has Q3 evidence; A_SHAPE/B_PRICE remain unassessed. These are not minima of a fully evaluated six-entry menu.',round_differences=differences,weighted_terms=terms,resources=usage,
        research_recommendation='CHECK_LD2_LONG_ROUND_PRICE_BEFORE_ADDING_LD3',operational_next='Resolve actual accelerator/runtime AST contract and obtain separate resume instruction before four still-unstarted Q3 calls. Preserve all frozen inputs and old STOPs.',
        price_scope='Exploratory model references; LD1 saved E_I[S]+residual; LD2 full-wrapper three-point model, zero offset; no E[C]/CI/G/winner qualification',
        PF_references=dict(Full_PF=615304502,B0I_p7=435481090,use='Reference only, not allocation budget or certified PR ranking'),mandatory_stop=True)
    write(base/'SUMMARY.json',result);write(base/'COMPILE_ROWS.json',dict(rows=paths,raw_publication=resolve('b030ea97a')))
    with (base/'MENU.csv').open('x')as f:
        names=['L_D','rule','status','cost_display','MSE_lower_display','MSE_upper_display','N_X','N_Y'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in rows:w.writerow({k:r.get(k)for k in names[:-2]}|dict(N_X=' '.join(str(s['N_X'])for s in r['shots']),N_Y=' '.join(str(s['N_Y'])for s in r['shots'])))
    print('MINIMA_SCOPE',result['minimum_scope'])
    for r in best:print(r['L_D'],r['cost_display'],r['MSE_lower_display'],r['MSE_upper_display'],'early',display(dec(r['m0_m6'])),'late',display(dec(r['m7_m12'])))
    print('DIFFERENCES',[(r['m'],r['difference_display'])for r in differences]);print('RESOURCES',usage)

if __name__=='__main__':main()
