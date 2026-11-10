"""Check saved Q3-C/D tables without rerunning campaigns or quantum calls."""
from fractions import Fraction as F
import hashlib, io, json, lzma, subprocess, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01'
def load(name):return json.loads((OUT/name).read_text())
def p(value):return tuple(F(v) for v in value['exact'])
def sum_pairs(ps):
    ps=list(ps)
    return tuple(sum((r[i] for r in ps),F()) for i in (0,1))


def main():
    src=load('SOURCE.json');cost=load('COST.json');d=load('D.json');native=load('C0.json')
    for path,sha in src['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,(path,'SOURCE')
    for path,meta in src['inputs'].items():
        raw=subprocess.check_output(['git','show',meta['commit']+':'+path],cwd=ROOT)
        assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
    assert len(native['checks'])==4 and all(r['core_lowerings_identical'] for r in native['checks'])
    extra_inputs={}
    def pinned(path):
        data=subprocess.check_output(['git','show',src['base_commit']+':'+path],cwd=ROOT)
        extra_inputs[path]=dict(commit=src['base_commit'],sha256=hashlib.sha256(data).hexdigest())
        return data
    for stem in ('pr_reference_feedback_q1','pr_reference_feedback_q3b','pr_reference_v6q',
                 'pr_reference_v6q_consumer','pr_reference_v6q_producer','pr_reference_v6q_schedule',
                 'track_c_compiler_a'):
        path='src/trottertracks/'+stem+'.py'
        assert (ROOT/path).read_bytes()==pinned(path)
    inverse=json.loads(lzma.decompress(pinned('artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/STAGE_A.json.xz')))
    with tarfile.open(fileobj=io.BytesIO(pinned('artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz')),mode='r:xz') as stream:
        catalogue=[json.load(stream.extractfile(m)) for m in stream if m.isfile()]
    assert inverse['passed'] and inverse['actual_inverse_records']==20
    for context in inverse['rows']:
        old=next(x for x in catalogue if x['request']==['reference',True,*context['context']])
        records={x['name']:x for x in old['records']}
        for record in context['records']:
            assert record['inverse_frontier']['ops']==records[record['name']]['ops']
            assert record['span_upper_holds']
    symbolic=load('SYMBOLIC_CONTEXT.json')
    assert symbolic['artificial_compiles']==4 and symbolic['scientific_compiles']==0
    assert len(symbolic['rows'])==4
    assert all(row['all_parameters_unbound'] and row['before_last_ops'].get('x',0)==0 for row in symbolic['rows'])
    assert src['scientific_calls']['C1_build_compile_operator']==0
    assert len(cost['rows'])==36
    checked=0
    for arm in ('Bprime','Reference_B1'):
        rows=sorted((r for r in cost['rows'] if r['arm']==arm),key=lambda x:x['m'])
        assert [r['m'] for r in rows]==list(range(18))
        dr=next(x for x in d['rows'] if x['arm']==arm)
        assert dr['prefix_trial_count']==7346 and dr['all_trial_count']==7686
        assert F(dr['hull_rounding_charge'])<F(dr['hull_rounding_reserve'])
        assert F(dr['final_energy_halfwidth_upper_Ha'])<F('0.00015936')
        for row in rows:
            n=row['old_N_per_axis'];is17=row['m']==17
            assert (row['q'],row['r'],row['bits'])==(131072,32,88) if is17 else row['q']==1<<row['m']
            for axis in 'XY':
                ps=row['per_shot'][axis]
                for metric in ps:
                    assert 0<p(ps[metric])[0]<=p(ps[metric])[1]
                sx=F(row['SX_per_shot_exact']);rz,cx,size=(p(ps[k]) for k in ('RZ','CX','size'))
                assert size==tuple(rz[i]+cx[i]+sx+1+(8 if arm=='Reference_B1' else 0) for i in (0,1))
            for metric in ('RZ','CX','size','depth'):
                x,y=(p(row['per_shot'][axis][metric]) for axis in 'XY')
                before=tuple(n*(a+b) for a,b in zip(x,y))
                assert before==p(row['XY_cost'][metric])
                after=tuple(340*b for b in y) if is17 else before
                assert after==p(row['Q_cost'][metric])
                saving=tuple(n*a+(n-340)*b for a,b in zip(x,y)) if is17 else (F(),F())
                assert saving==p(row['paired_saving'][metric])
                if is17:
                    assert p(row['Q_per_shot'][metric])==y
                    assert all(before[i]-after[i]==saving[i] for i in (0,1))
                else:assert row['new_physical_shots']==2*n
                checked+=1
        for metric in ('RZ','CX','size','depth'):
            totals=cost['totals'][arm]
            for mode in ('XY','Q'):
                assert sum_pairs(p(r[mode+'_cost'][metric]) for r in rows)==p(totals[mode][metric])
            saving=sum_pairs(p(r['paired_saving'][metric]) for r in rows)
            comp=cost['comparisons'][arm][metric]
            assert p(comp['all18_saving'])==saving and saving[0]>0
            before=p(totals['XY'][metric])
            assert p(comp['reduction_fraction_all18'])==(saving[0]/before[1],saving[1]/before[0])
    for metric in ('RZ','CX','size','depth'):
        rb=p(cost['totals']['Reference_B1']['Q'][metric]);bb=p(cost['totals']['Bprime']['Q'][metric])
        cross=cost['Reference_Q_vs_Bprime_Q'][metric]
        assert p(cross['reference_minus_Bprime'])==(rb[0]-bb[1],rb[1]-bb[0])
        assert rb[1]<bb[0]
    checks=dict(status='Q3CD_POSTRUN_CHECKS_PASS',fixed_arm_round_metric_cells=checked,
        all_input_SOURCE_hashes_match=True,paired_price_correlations_preserved=True,
        full_round_sums_and_comparisons_checked=True,zero_unknown_price_imputation=False,
        new_scientific_campaign_or_compile_or_Q0_scan=0,
        author_check_not_independent_review=True,
        fixed_consumer_provider_SOURCE_match=True,inverse_D_saved_native_counts_match=20,
        symbolic_before_cancellation_X_free_contexts=4,additional_provenance=extra_inputs,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        symbolic_test_sha256=hashlib.sha256((ROOT/'tests/test_pr_reference_feedback_q3cd_context.py').read_bytes()).hexdigest())
    (OUT/'CHECKS.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
