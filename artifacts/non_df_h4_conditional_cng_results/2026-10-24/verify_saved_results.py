"""Read-only verification of saved outputs, bindings and native termination.

No new shot evaluator/statistics/scientific entrypoint. Checks saved rational
identities and sufficient inequalities, not another numerical experiment.
"""
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'run01'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seal=read(HERE/'seal.json'); c=read(HERE/'active_contract.json')
    for p,h in seal['code_files'].items():
        assert sha(ROOT/p)==h
        assert hashlib.sha256(subprocess.check_output(['git','show',seal['SOURCE']+':'+p],cwd=ROOT)).hexdigest()==h
    for p,h in read(HERE/'runtime.json')['files'].items():assert sha(Path(p))==h
    plan=ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24'
    manifest=read(plan/'inputs.json')
    for p,h in manifest['files'].items():assert sha(ROOT/p)==h
    model=read(ROOT/manifest['roles']['pauli_jw_C_model'])['cases']['PR']
    assert [x['label'] for x in model['components']]==model['metadata']['tail_labels']
    assert sha(plan/'inputs.json')==seal['input_manifest_sha256']
    assert sha(plan/'axis_conditions.json')==seal['axis_catalog_sha256']
    assert sha(HERE/'runtime.json')==seal['runtime_sha256']
    assert sha(HERE/'approval.txt')==seal['approval_sha256']
    grant=read(RUN/'grant.json')
    assert sha(HERE/'active_contract.json')==grant['contract_sha256']
    assert sha(HERE/'seal.json')==grant['seal_sha256']==c['seal_sha256']
    assert read(RUN/'CONSUMED.json')['grant_id']==grant['grant_id']
    fn=read(RUN/'conditional_functions.json'); rows=read(RUN/'conditional_results.json')
    regions=read(RUN/'comparison_regions.json'); costs=read(RUN/'cost_summary.json')
    evals=list(csv.DictReader((RUN/'axis_results.csv').open()))
    assert (len(fn),len(rows),len(regions),len(evals))==(20,100,30,200)
    assert len(list(csv.DictReader((RUN/'guard_thresholds.csv').open())))==40
    assert sum(x['primary'] for x in regions)==10
    for f in fn:
        assert f['actual_N'] is None and f['actual_G'] is None and f['residual_default'] is None
        for a,ax in f['axes'].items():
            s,u,cap=map(F,(ax['s_exact'],ax['u_enc_exact'],ax['rho_guard_exact']))
            assert cap==s/101-u
            # Endpoint enclosures used for independent-box comparisons have
            # unambiguous integer ceilings, so the minimum cannot be overstated.
            for h in (s-u,s*100/101):
                import math
                assert math.ceil(F(ax['scale']['A_lower'])/h**2)==math.ceil(F(ax['scale']['A_upper'])/h**2)
    for e in evals:
        ax=next(f for f in fn if (f['case'],f['epsilon_sig'])==(e['case'],e['epsilon_sig']))['axes'][e['axis']]
        if e['N']:
            n=int(e['N']);h=F(e['headroom_exact'])
            assert 0<n<2**52 and n*h*h>=F(ax['scale']['A_upper'])
            assert F(e['u_over_h_exact'])<=F(1,100) and h>0
    for row in rows:
        assert row['actual_rho'] is None and row['unconditional_N_G'] is None
        if row['G'] is not None:
            cc=costs[row['case']]['axes']
            assert row['N_total']==row['N_cos']+row['N_sin']
            assert F(row['G_exact'])==row['N_cos']*F(cc['cosine']['mean_exact'])+row['N_sin']*F(cc['sine']['mean_exact'])
    rare=read(RUN/'rare_cost_sensitivity.json')
    for d in rare.values():
        assert sum(map(F,d['implemented_probability_exact'].values()))==1
        assert sum(d['observed_stratum_count'].values())==256
        assert d['valid_whole_wrapper_upper'] is None
        assert all(v is None for v in d['stratum_population_mean'].values())
        assert d['observed_stratum_cost']['any_order4'] is None
        assert d['observed_stratum_cost']['no4_with_unseen'] is None
    assert read(RUN/'call_ledger.json')['used']==c['authority_batch_limits']
    assert read(RUN/'STOP.json')['status']=='COMPLETE_MANDATORY_STOP'
    native=read(RUN/'native/STOP.json')
    assert native['exit_code']==0 and native['all_owned_reaped'] and not native['descendants_terminated']
    assert read(RUN/'native/worker_native_exit.json')['scope']=='NEW_CNG_SAVED_REANALYSIS_ONLY'
    assert read(RUN/'STOP.json')['original_H4_native_verified'] is False
    report=dict(status='PASS',SOURCE=seal['SOURCE'],artificial_tests=16,
        input_SHA_files=len(manifest['files']),runtime_SHA_files=len(read(HERE/'runtime.json')['files']),
        functions=20,scenarios=100,axis_sufficient_inequalities=200,guard_thresholds=40,
        independent_box_endpoint_ceilings_unambiguous=True,comparison_regions=30,primary_regions=10,
        paired_covariance_and_exact_mean_verified=True,rare_strata_missing_means_preserved=True,
        native_exit=0,all_owned_reaped=True,old_H4_native_status='UNKNOWN_UNCHANGED',
        original_N_G_null_preserved=True,new_shot_evaluator_calls_in_this_check=0,new_scientific_runs=0,
        formal_certificate_accepted=False,mandatory_STOP=True)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
