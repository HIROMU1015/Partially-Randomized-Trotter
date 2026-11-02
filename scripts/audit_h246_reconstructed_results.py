"""Read-only independent integer-moment checks after full fixed evaluation.

No quantum/scientific provider import, resampling, shot recomputation or fit.
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def moments(values):
    n=len(values)
    assert n>=2 and all(type(x) is int for x in values)
    total=sum(values);squares=sum(x*x for x in values)
    mean=Fraction(total,n)
    # Independent route: uncentered integer sums; production uses centered
    # Fraction deviations and a separate within-trajectory covariance identity.
    variance=Fraction(n*squares-total*total,n*(n-1))
    return mean,variance


def exact(x):return dict(numerator=x.numerator,denominator=x.denominator)


def close(a,b):assert math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-6),(a,b)


def audit(private,destination):
    private=Path(private).resolve();attempt=private/'confirmation_run02'
    fixed=read(ROOT/'references/h246_confirmation_design_v1.json')
    assert sha(ROOT/'references/h246_confirmation_design_v1.json')=='91abf0d4c57a165d376c43644847e005250a63e5ab52a41700590306bb37a4f5'
    paths=read(attempt/'configs_PRIVATE.json')['configs']
    # All stages and their authorized fixed evaluations must have finished.
    for path in paths:
        cfg=read(path);r=Path(cfg['run_root'])
        assert read(r/'end_audit.json')['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP'
        assert read(r/'fixed_comparison.json')['status']=='FIXED_CONFIRMATION_COMPARISON_COMPLETE'
    reports=[];sample_rows=[];group_rows=[];raw_inventory=[]
    for path in paths:
        cfg=read(path);root=Path(cfg['run_root']);system=cfg['stage'][:2]
        end=read(root/'end_audit.json');ev=read(root/'fixed_comparison.json')
        comp=next(c for c in fixed['comparisons'] if c['id']==system+'_PRIMARY')
        assert cfg['comparison']==comp
        assert ev['SOURCE']==cfg['SOURCE']==end['SOURCE'] and ev['end_audit_sha256']==sha(root/'end_audit.json')
        assert not end['errors'] and end['byte_SHA_roundtrip_PASS']
        accepted={r['id']:r for r in end['results']};by={g['candidate']['id']:[] for g in comp['groups']}
        kinds=Counter();rte_events=0;maxima={}
        for p in Path(cfg['ledger']).glob('*.json'):
            row=read(p);assert row['state']=='RESERVED_NONREFUNDABLE' and row['SOURCE']==cfg['SOURCE']
            kinds[row['kind']]+=1
        for task in cfg['tasks']:
            out=Path(cfg['output'])/task['id'];p=out/'result.json';r=read(p)
            assert sha(p)==accepted[task['id']]['result_sha256'] and r['task']==task
            assert r['SOURCE']==cfg['SOURCE'] and r['plan_sha256']==cfg['plan_sha256']
            assert r['seed_cohort']=='CONFIRMATION' and not r['conditional_sampling'] and not r['cohort_pooling']
            assert [a['axis'] for a in r['axis_costs']]==['cosine','sine']
            pair=[a['metrics']['rz_count'] for a in r['axis_costs']]
            assert all(type(x) is int and x>0 for x in pair)
            by[task['candidate']['id']].append(dict(pair=pair,stratum=r['events']['stratum']))
            events=read(out/'events_PRIVATE.json')['events'];count=sum(map(len,events));rte_events+=count
            assert count==32*task['candidate']['q']
            for check in r['checks']:
                assert check['PASS'] and not check['phase_alignment'] and 0<=check['error']<check['strictly_below']
                maxima[check['check']]=max(maxima.get(check['check'],0),check['error'])
            sample_rows.append(dict(system=system,candidate_id=task['candidate']['id'],sample=task['sample'],
                cosine_RZ=pair[0],sine_RZ=pair[1],stratum=r['events']['stratum'],result_sha256=sha(p),SOURCE=cfg['SOURCE']))
            for mf in out.glob('quarantine_*/*.archive.json'):
                m=read(mf);archive=mf.parent/m['archive_file']
                assert archive.exists() and archive.stat().st_size==m['codec_receipt']['stored']['bytes']
                raw_inventory.append(dict(path=str(archive.relative_to(private)),bytes=archive.stat().st_size,
                    archive_sha256=m['codec_receipt']['stored']['sha256'],
                    original_sha256=m['codec_receipt']['original']['sha256']))
        assert kinds['TASK']==len(cfg['tasks'])
        for kind in ('trajectories','request_sampling_calls','native_piece_compositions','native_evolution',
                'native_low_actions','native_high_actions','independent_oracle_actions','uncompiled_wrapper_actions',
                'compiled_wrapper_actions','compile_axes'):
            assert kinds[kind]==cfg['call_scope'][kind],kind
        assert rte_events==cfg['call_scope']['rte_events']
        groups=[]
        for group,n,stored in zip(comp['groups'],comp['recommended']['n_C'],ev['groups']):
            candidate=group['candidate'];items=by[candidate['id']]
            assert len(items)==n and stored['candidate']==candidate
            nc,ns=(group['cost_row']['shot'][k] for k in ('N_cos','N_sin'))
            xs=[x['pair'][0] for x in items];ys=[x['pair'][1] for x in items]
            values=[nc*x+ns*y for x,y in zip(xs,ys)]
            mean,var=moments(values);mx,vx=moments(xs);my,vy=moments(ys)
            cov=Fraction(n*sum(x*y for x,y in zip(xs,ys))-sum(xs)*sum(ys),n*(n-1))
            assert var==nc*nc*vx+ns*ns*vy+2*nc*ns*cov
            stats=stored['statistics'];assert exact(mean)==stats['mean_exact'] and exact(var)==stats['variance_exact']
            se=math.sqrt(float(var/n));close(se,stats['SE'])
            observed=dict(Counter(x['stratum'] for x in items))
            for key,value in stored['observed_strata'].items():assert observed.get(key,0)==value
            assert stored['unobserved_costs'] is None and stats['population_variance_bound'] is None
            row=dict(system=system,candidate_id=candidate['id'],formula=candidate['formula'],q=candidate['q'],n=n,
                N_cos=nc,N_sin=ns,mean_total_RZ=float(mean),unbiased_variance_total_RZ=float(var),SE_total_RZ=se,
                mean_cos_RZ=float(mx),mean_sin_RZ=float(my),mean_exact=exact(mean),variance_exact=exact(var),
                observed_strata=stored['observed_strata'],unknown_stratum_costs=None,old_frozen_shots_used=True)
            groups.append(row);group_rows.append(row)
        delta=Fraction(groups[0]['mean_exact']['numerator'],groups[0]['mean_exact']['denominator'])-Fraction(groups[1]['mean_exact']['numerator'],groups[1]['mean_exact']['denominator'])
        se=math.hypot(groups[0]['SE_total_RZ'],groups[1]['SE_total_RZ'])
        z=NormalDist().inv_cdf(1-.05/6)
        close(float(delta),ev['difference']);close(se,ev['difference_SE'])
        for x,y in zip([float(delta)-z*se,float(delta)+z*se],ev['approximate_interval']):close(x,y)
        improvement=float(delta)/groups[0]['mean_total_RZ'];close(improvement,ev['relative_point_improvement'])
        assert ev['point_status']==('REPRODUCED' if delta>0 else 'REVERSED' if delta<0 else 'TIED')
        assert ev['ratio_interval'] is None and ev['population_guarantee'] is False
        external=read(cfg['supervisor']['exit']);done=read(root/'completion_witness.json')
        assert external['native_exit']==0 and external['reaped'] and all(w['exit']==0 and w['reaped'] for w in done['workers'].values())
        assert cfg['complete_historical_seed_noncollision_guaranteed'] is False
        reports.append(dict(system=system,SOURCE=cfg['SOURCE'],input_identity=cfg['input_identity'],
            input_sha256=cfg['binding']['snapshot'],hamiltonian_hash=cfg['binding']['hamiltonian'],state_hash=cfg['binding']['state'],
            grant_id=cfg['grant']['grant_id'],plan_sha256=cfg['plan_sha256'],call_scope=cfg['call_scope'],ledger_counts=dict(kinds),
            executed_rte_events=rte_events,groups=groups,difference=float(delta),difference_SE=se,
            normal_Bonferroni_approximate_interval=ev['approximate_interval'],relative_point_improvement=improvement,
            point_status=ev['point_status'],approximate_direction=ev['approximate_direction'],
            ratio_interval=None,raw_numeric_error_maxima=maxima,native_exit=external['native_exit'],all_workers_reaped=True,
            acquisition_wall_seconds=done['wall_seconds'],worker_count=len(done['workers']),
            end_audit_sha256=sha(root/'end_audit.json'),fixed_comparison_sha256=sha(root/'fixed_comparison.json')))
    assert sum(r['call_scope']['trajectories'] for r in reports)==118
    assert sum(r['call_scope']['compile_axes'] for r in reports)==236
    assert len(raw_inventory)==708
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (destination/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    save('results.json',dict(status='FIXED_CONFIRMATION_AND_INTEGER_MOMENT_AUDIT_PASS_MANDATORY_STOP',
        comparisons=reports,independent_check='UNCENTERED_INTEGER_SUMS_AND_SUMS_OF_SQUARES_PLUS_PAIRED_COVARIANCE',
        trajectories=118,compile_axes=236,raw_archives=708,unknown_rare_tail_cost_bound=None,
        complete_historical_seed_noncollision_guaranteed=False,quantum_shots_executed=0,
        additional_candidate_search=False,statistical_family_size=3,statistical_family_alpha=.05,
        normal_intervals_are_rigorous=False,ratio_confidence_intervals=None,immutable_CI=False,
        separate_reviewer=False,mandatory_STOP=True,push=False))
    save('raw_archive_inventory.json',dict(files=raw_inventory,byte_decompression_SHA_audit='ALREADY_PASSED_BY_STAGE_END_AUDITS_NO_SCIENTIFIC_REPLAY'))
    for path in paths:
        cfg=read(path);s=cfg['stage'][:2];r=Path(cfg['run_root'])
        save(s+'_fixed_comparison.json',read(r/'fixed_comparison.json'))
        save(s+'_end_audit.json',read(r/'end_audit.json'))
    for name,rows in [('paired_samples_118.csv',sample_rows),('group_statistics_6.csv',group_rows)]:
        with (destination/name).open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(status='PASS',comparisons=[dict(system=r['system'],improvement_percent=100*r['relative_point_improvement'],groups=r['groups']) for r in reports])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.private,a.output)))
