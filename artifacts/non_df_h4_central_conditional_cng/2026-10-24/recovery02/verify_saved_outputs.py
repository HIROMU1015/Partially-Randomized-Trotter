"""Metadata, SHA and saved integer/moment identities only.

No shot evaluator, scientific entrypoint, statistics estimator or quantile call.
"""
import csv
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
NS=BASE/'run01'
pkg=types.ModuleType('audit_central_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['audit_central_private']=pkg
from audit_central_private import h4_certificate_control as C


def main():
    checks={}
    def ok(name,value):
        if not value:raise ValueError(name)
        checks[name]='PASS'
    seal=C.read(BASE/'seal.json')
    for p,h in seal['bound_manifests'].items():ok('bound_manifest:'+p,C.sha(BASE/p)==h)
    for p,h in seal['code_files'].items():
        ok('code_SHA:'+p,C.sha(ROOT/p)==h)
        b=subprocess.check_output(['git','show',seal['SOURCE']+':'+p],cwd=ROOT)
        ok('committed_SOURCE:'+p,hashlib.sha256(b).hexdigest()==h)
    for name,files in [('inputs',{str(ROOT/k):v for k,v in C.read(BASE/'inputs.json')['files'].items()}),
                       ('runtime',C.read(BASE/'runtime.json')['files']),
                       ('old_preservation',{str(ROOT/k):v for k,v in C.read(BASE/'preservation_before.json').items()})]:
        C.hashes(files);ok(name+'_SHA',True)
    contract=C.read(BASE/'active_contract.json');g=C.read(NS/'grant.json')
    ok('grant_digest',g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}))
    ok('grant_bound',g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json') and g['namespace']==str(NS))
    ok('consumed_one_shot',C.read(NS/'CONSUMED.json')['grant_id']==g['grant_id'])
    ok('exact_batch_limits',C.read(NS/'call_ledger.json')['used']==contract['authority_batch_limits'])
    stop=C.read(NS/'STOP.json');native=C.read(NS/'native/worker_native_exit.json');nstop=C.read(NS/'native/STOP.json')
    ok('new_native_exit_reap_STOP',stop['status']=='COMPLETE_MANDATORY_STOP' and native['exit_code']==0 and native['reaped'] and nstop['all_owned_reaped'] and not nstop['descendants_terminated'])
    identity=C.read(NS/'native/worker_identity.json')
    ok('no_remaining_owned_worker',C._proc_identity(identity['pid'])!=identity)
    prior=BASE.parent
    ok('failed_run_preserved',C.read(prior/'run01/STOP.json')['status']=='FAILED_NO_AUTOMATIC_RETRY' and C.read(prior/'run01/native/worker_native_exit.json')['exit_code']==1)
    used=C.read(prior/'run01/call_ledger.json')['used']
    ok('failed_run_no_N_G',all(v==0 for k,v in used.items() if k!='input_binding_batch'))
    for p,d in C.read(BASE/'recovery_history.json')['prior_source_archives'].items():
        ok('failed_SOURCE_archive:'+p,C.sha(ROOT/d['archive'])==d['sha256'])
    rows=C.read(NS/'conditional_results.json');axes=C.read(NS/'axis_results.json');comparisons=C.read(NS/'comparison_regions.json')
    ok('10_20_15_cardinalities',(len(rows),len(axes),len(comparisons))==(10,20,15))
    ok('unique_conditions_axes_comparisons',len({(r['case'],r['epsilon_sig']) for r in rows})==10 and len({(r['case'],r['epsilon_sig'],r['axis']) for r in axes})==20 and len({(r['left'],r['right'],r['epsilon_sig']) for r in comparisons})==15)
    ok('actual_values_unknown',all(r['actual_rho_new'] is None and r['actual_N'] is None and r['actual_G'] is None and not r['formal_certificate_accepted'] for r in rows))
    for r in axes:
        for k in ('rho_zero','guard_face'):
            s=r[k];n=s['N'];lo,hi=F(s['bound_lower_exact']),F(s['bound_upper_exact'])
            ok('saved_scalar:'+r['case']+r['epsilon_sig']+r['axis']+k,
               type(n) is int and 0<n<2**52 and n-1<lo<=hi<=n and s['ceiling_interval_width']==0 and s['sufficient_inequality_verified'])
    prices=C.read(ROOT/'artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.json')['rows']
    gvals=C.read(NS/'paired_G_values.json')
    for r in rows:
        pair=next(x for x in gvals if (x['case'],x['epsilon_sig'])==(r['case'],r['epsilon_sig']))
        pp=[p for p in prices if p['case']==r['case'].rsplit('_',1)[1]]
        z=pair['G_i'];n=len(z);mu=F(r['G_exact'])
        ok('G_integer_identity:'+r['case']+r['epsilon_sig'],
           pair['trajectory_ids']==[p['trajectory_id'] for p in pp] and
           z==[r['N_cos']*p['cosine']['RZ']+r['N_sin']*p['sine']['RZ'] for p in pp] and sum(z)==mu*n)
        var=sum((F(x)-mu)**2 for x in z)/(n-1)/n if n>1 else F(0)
        ok('G_saved_covariance_moment:'+r['case']+r['epsilon_sig'],var==F(r['paired_variance_of_mean_exact']))
        ok('G_saved_SD_SE:'+r['case']+r['epsilon_sig'],math.isclose(r['G_SD'],r['G_SE']*math.sqrt(n)))
    old=C.read(ROOT/'artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/conditional_results.json')
    ok('DF_generic_rows_copied_exactly',C.read(NS/'reused_reference_rows.json')==[r for r in old if r['theta']=='0'])
    for label,data in [('conditional_results',rows),('axis_results',axes),('comparison_regions',comparisons)]:
        with (NS/(label+'.csv')).open() as f: rr=list(csv.DictReader(f))
        ok('CSV_JSON:'+label,len(rr)==len(data) and all(a['epsilon_sig']==b['epsilon_sig'] for a,b in zip(rr,data)))
    manifest=C.read(ROOT/'artifacts/validation_manifest.json');oldmanifest=C.read(prior/'manifest_before.json')
    ok('all_180_prior_manifest_rows_unchanged',all(manifest[k]==v for k,v in oldmanifest.items()))
    output={str(p.relative_to(NS)):C.sha(p) for p in sorted(NS.rglob('*')) if p.is_file()}
    C.save(BASE/'output_inventory.json',output)
    result=dict(status='PASS',SOURCE=seal['SOURCE'],checks=checks,
        counts=dict(code=len(seal['code_files']),inputs=len(C.read(BASE/'inputs.json')['files']),
                    runtime=len(C.read(BASE/'runtime.json')['files']),preserved=len(C.read(BASE/'preservation_before.json')),
                    output_files=len(output),prior_manifest_rows=len(oldmanifest)),
        scope='Primary-agent metadata/SHA/saved-integer-moment checks; no new scientific evaluator or statistics call',
        old_native_not_repaired=True,remaining_owned_processes=[],mandatory_STOP=True)
    C.save(BASE/'verification.json',result)
    print(json.dumps(dict(status='PASS',checks=len(checks),counts=result['counts'])))


if __name__=='__main__':main()
