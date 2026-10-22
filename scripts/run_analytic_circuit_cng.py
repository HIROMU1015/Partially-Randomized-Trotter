"""One-shot, source-bound saved accounting, one unit per method/candidate/epsilon."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trotterlib.analytic_circuit_cng import account

OLD=ROOT/'docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022'
RAW=ROOT/'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22'
BASE=ROOT/'artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22'
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def saved_prices(row,role):
    cid=row[role+'_candidate_id']
    if row['system']=='H2':
        src=ROOT/'docs/progress/track_a/figures/pr_pf_theory_compiled_ratio_20261022/sources'
        p=src/('h2_a.json' if role=='PR' else 'h2_d.json');d=read(p)
        records=[x for x in d['measurements']+d.get('reused_FP_measurements',[]) if x['cell_id']==cid]
        return [dict(cosine=x['axes']['cosine']['RZ'],sine=x['axes']['sine']['RZ']) for x in records],p
    p=RAW/'units'/row['system']/cid/('MC_main' if role=='PR' else 'compiled_cost')/'result.json'
    stop=read(p.with_name('STOP.json'))
    assert stop['status']=='COMPLETE_MANDATORY_STOP_UNIT' and stop['result_files']['result.json']==sha(p)
    return read(p)['values'],p


def main():
    assert not (BASE/'STOP.json').exists(),'Overall STOP exists'
    source=read(BASE/'SOURCE.json')
    for p,h in source['files'].items():assert sha(ROOT/p)==h,'SOURCE mismatch: '+p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source['commit']
    data=read(OLD/'figure_data.json');results=[]
    from scipy.stats import t
    for row in data['rows']:
        for role in ['PR','PF']:
            cid=row[role+'_candidate_id'];eps=row['epsilon'];prices,price_path=saved_prices(row,role)
            unit=BASE/'units'/row['cohort']/row['system']/cid/('saved_CNG_accounting_'+eps)
            assert not unit.exists(),'Consumed unit; no retry: '+str(unit)
            plan=dict(cohort=row['cohort'],system=row['system'],candidate=cid,role=role,
                stage='saved_CNG_accounting',epsilon=eps,SOURCE=source['commit'],
                input_hashes={'paired_figure_data':sha(OLD/'figure_data.json'),'saved_prices':sha(price_path)},
                price_path=str(price_path.relative_to(ROOT)),random_seed=None,sampling=False,
                conditions='Exact saved C,N,G accounting; no new estimator, signal, compilation or shot evaluation')
            write(unit/'plan.json',plan)
            grant=dict(grant_id=digest(plan),plan_sha256=digest(plan),one_shot=True,automatic_retry=False,
                authorization_sha256=sha(BASE/'user_instruction.txt'),source_manifest_sha256=sha(BASE/'SOURCE.json'))
            write(unit/'grant.json',grant)
            # Exclusive marker: neither failures nor interrupts can be relabelled as a new attempt.
            with (unit/'marker.json').open('x') as f:json.dump(dict(grant_id=grant['grant_id'],state='CONSUMED_NONREFUNDABLE',time=time.time()),f)
            try:
                critical=float(t.ppf(.975,len(prices)-1)) if len(prices)>1 else 0.
                value=account(row,role,prices,critical)
                write(unit/'result.json',value);results.append(value)
                write(unit/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP_UNIT',SOURCE=source['commit'],
                    result_sha256=sha(unit/'result.json'),automatic_retry=False))
            except BaseException as exc:
                write(unit/'STOP.json',dict(status='FAILED_NO_RETRY',SOURCE=source['commit'],error=repr(exc)))
                raise
    write(OUT/'figure_data.json',dict(schema='SAVED_AXIS_CNG_V1',rows=results,
        old_result_commit='a590214d6d70962aa02b9e0a2bce56810859522f',SOURCE=source['commit'],
        C_paired_conditions=20,C_method_sides=40,N_independent_conditions=0,
        G_comparison='N_FIXED_C_ONLY_REPLACEMENT',G11_obtained=False,
        conditions=data['conditions'],new_physical_science_calls=0,new_shot_evaluations=0))
    write(BASE/'execution_receipt.json',dict(status='SAVED_ACCOUNTING_COMPLETE',units=len(results),
        SOURCE=source['commit'],new_sampling=0,new_compile=0,new_signal=0,new_shot_evaluation=0,
        saved_accounting_units=len(results),failed_units_retried=False))
    print(json.dumps(dict(completed_units=len(results),N_independent=0)))


if __name__=='__main__':main()
