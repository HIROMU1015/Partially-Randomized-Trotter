"""Saved-result audit and arithmetic only. No NPZ decode, sampler or compiler."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from h6_fixed_delta_plan_v1 import cells, registrations, EPSILONS, digest
from d0_frozen_shot_rule_v1 import u_aware_shots

ROOT=Path(__file__).resolve().parents[1]
METRICS=('RZ','CX','RZ_depth','CX_depth','depth','size')
OLD_COMMIT='321cd8c3066194fe98741cbebbcdba1515d1deaa'
OLD_BASE='artifacts/resource_applicability/'
OLD_WORKTREE=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-hash-reuse-eight-worker-20261012')


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def committed_csv(path):
    raw=subprocess.check_output(['git','show',OLD_COMMIT+':'+path],cwd=ROOT)
    assert raw==(OLD_WORKTREE/path).read_bytes(), 'OLD_PUBLIC_CSV_CHANGED'
    return list(csv.DictReader(raw.decode().splitlines()))


def paired_metrics(rows,shots,metric):
    pairs={}
    for r in rows:
        if type(r['metrics'][metric]) is not int or r['metrics'][metric]<0:raise ValueError('METRIC')
        key=r['replica'];pair=pairs.setdefault(key,{})
        if r['axis'] in pair:raise ValueError('DUPLICATE_AXIS')
        pair[r['axis']]=r['metrics'][metric]
    if not pairs or any(set(x)!={'cosine','sine'} for x in pairs.values()):raise ValueError('INCOMPLETE_PAIR')
    means={a:sum(Fraction(x[a]) for x in pairs.values())/len(pairs) for a in ('cosine','sine')}
    out=dict(n=len(pairs),axis_means={a:float(v) for a,v in means.items()},G=None,G_exact=None,SE=None)
    if shots is not None:
        totals=[shots['real']*x['cosine']+shots['imag']*x['sine'] for x in pairs.values()]
        exact=Fraction(sum(totals),len(totals))
        out.update(G=float(exact),G_exact=dict(numerator=str(exact.numerator),denominator=str(exact.denominator)),
            SE=statistics.stdev(totals)/math.sqrt(len(totals)) if len(totals)>1 else None)
    return out


def account(signal,epsilon,factor):
    b=signal['error_decomposition']['total_signed'];u=signal['allowance']
    r=u_aware_shots({a:abs(b[a]) for a in ('real','imag')},{a:factor*u[a] for a in ('real','imag')},
        epsilon=epsilon,log_B_upper=signal['log_B']+factor*u['log_B_margin'],evidence_kind='EMPIRICAL',
        evidence_ref=signal['cell']['id']+'_signal.json')
    ratios=[v['u_over_headroom'] for v in r['axes'].values()]
    r['cost_acquisition_eligible']=(r['eligibility_under_declared_allowance']=='ELIGIBLE' and
        r['N_total'] is not None and all(v is not None and v<=.01 for v in ratios))
    return r


def signed_parts(s):
    z=lambda x:complex(x['real'],x['imag'])
    corrected=z(s['signals']['corrected']);ref=z(s['reference'])
    if s['cell']['method'] in ('B0','B0I'):
        middle=z(s['signals']['exact_truncated']);parts=dict(discard=middle-ref,PF=corrected-middle)
    elif s['cell']['method']=='B2':
        middle=z(s['signals']['exact_tail']);parts=dict(outer_PF=middle-ref,finite_RTE=corrected-middle)
    else:parts=dict(PF=corrected-ref)
    assert abs(sum(parts.values())-(corrected-ref))<1e-14
    return dict(total=abs(corrected-ref),parts_abs={k:abs(v) for k,v in parts.items()},
        parts_signed={k:dict(real=v.real,imag=v.imag) for k,v in parts.items()})


def verify_bundle(path):
    """Reproduce the published arithmetic without private inputs or scientific imports."""
    path=Path(path)
    signals={(s['T'],s['cell']['id']):s for s in read(path/'signal_summary_v1.json')}
    rows=read(path/'wrapper_projection_v1.json');entries=read(path/'accounting_v1.json')
    assert len(signals)==15 and len(rows)==216 and len(entries)==150
    checked=0
    for entry in entries:
        key=(entry['T'],entry['cell_id']);s=deepcopy(signals[key])
        s['error_decomposition']={'total_signed':dict(
            real=s['signals']['corrected']['real']-s['reference']['real'],
            imag=s['signals']['corrected']['imag']-s['reference']['imag'])}
        a=account(s,entry['epsilon'],entry['factor'])
        assert a==entry['accounting'], 'SHOT_ARITHMETIC'
        shots={axis:a['axes'][axis]['shots'] for axis in ('real','imag')} if a['cost_acquisition_eligible'] else None
        subset=[r for r in rows if (r['T'],r['cell_id'])==key]
        for metric in METRICS:
            assert paired_metrics(subset,shots,metric)==entry['metrics'][metric], 'PAIRED_COST_ARITHMETIC'
            checked+=1
    return dict(status='PASS',accounting_rows=len(entries),paired_metric_checks=checked,
        new_scientific_calls=0,private_inputs_required=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--output');p.add_argument('--verify');a=p.parse_args()
    if a.verify:
        if a.run or a.output:p.error('--verify is exclusive with --run/--output')
        print(json.dumps(verify_bundle(a.verify)));return
    if not a.run or not a.output:p.error('--run and --output are required')
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=read(run/'config.json');terminal=read(run/'control01/terminal.json');wait=read(run/'launch02/native_wait_v1.json')
    assert terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['first_stop'] is None
    assert terminal['SOURCE']==cfg['SOURCE']==wait['SOURCE'] and wait['exit_code']==0 and wait['wait_completed']
    inventory=[]
    def receipt(path):
        inventory.append(dict(path=str(path.relative_to(run)) if path.is_relative_to(run) else str(path),
            bytes=path.stat().st_size,sha256=sha(path)))
    for f in (run/'config.json',run/'authorization_approved.json',run/'control01/terminal.json',run/'launch02/native_wait_v1.json'):receipt(f)
    signals={};wrappers={};orders={};calls=Counter();native_errors=[];primitive_errors=[];all_retired=[]
    stage_stats={};group_count=0;resource_rows=[]
    for mode,n in (('signal',2),('heavy',2),('normal',70)):
        summary=read(run/'control01'/('summary_'+mode+'.json'));receipt(run/'control01'/('summary_'+mode+'.json'))
        assert len(summary['completed_indices'])==len(summary['dispatched'])==n and summary['pending']==0
        assert summary['all_dispatched_reaped'] and not summary['cleanup_errors'] and summary['first_stop_reason'] is None
        assert all(r['exit_code']==0 and r['reaped'] for r in summary['retired'])
        all_retired+=summary['retired'];peak=0;size=0;gap=0;duration=0;psi=0;last_oom=None;samples=0
        journal=run/'output01'/mode/'resource_observations.jsonl';receipt(journal)
        for line in journal.open():
            r=json.loads(line);assert r['first_stop_reason'] is None
            peak=max(peak,r['aggregate_rss_bytes']);gap=max(gap,r['staleness_seconds']);duration=max(duration,r['observation_seconds'])
            psi=max(psi,*r['memory']['full_psi_avg10'].values());samples+=1
            if last_oom is not None:assert r['memory']['oom']==last_oom
            last_oom=r['memory']['oom']
            for worker in r['workers']:size=max(size,worker['memory']['VmSize_bytes'])
        assert max(gap,duration)<=5
        stage_stats[mode]=dict(wall_seconds=summary['wall_seconds'],peak_aggregate_RSS_bytes=max(peak,summary['aggregate_peak_rss_bytes']),
            sampled_worker_max_AS_bytes=size,monitor_samples=samples,max_interval=gap,max_observation_seconds=duration,max_PSI=psi)
        for d in sorted((run/'output01'/mode).glob('cost_*')):
            t=read(d/'worker_terminal.json');b=read(d/'task_binding.json');task=b['task'];receipt(d/'worker_terminal.json')
            assert t['status']=='COMPLETE' and t['task']==task and t['SOURCE']==cfg['SOURCE']
            if mode=='signal':
                s=read(d/'panel_signals.json');receipt(d/'panel_signals.json');calls.update(s['calls'])
                assert s['T']==task['T'] and len(s['signals'])==5
                signals[s['T']]=s['signals'];v=read(d/'primitive_validation.json');receipt(d/'primitive_validation.json')
                assert v['actual_time_count']==125 and v['probe_count']==3
                primitive_errors.append(v)
                continue
            result=read(d/'cost_result.json');receipt(d/'cost_result.json');group_count+=1
            assert result['task']==task and result['calls_attempted']['compile']==2
            expected=registrations()[b['index']]['task'];assert {k:v for k,v in task.items() if k!='expected_signal'}==expected
            calls.update(result['calls_attempted']);assert [x['axis'] for x in result['rows']]==['cosine','sine']
            tr=read(d/'trajectory.json');receipt(d/'trajectory.json');assert tr['task']==task
            assert digest(tr['events'])==tr['event_digest']
            if tr['events'] is not None:
                seq=[e['taylor_order'] for st in tr['events'] for e in st];assert len(seq)==task['cell']['R']
                orders.setdefault(task['T'],Counter()).update(seq)
            for row in result['rows']:
                assert row['task']==task and row['compiler_hash']==cfg['compiler_hash'] and row['event_digest']==tr['event_digest']
                assert read(d/('wrapper_'+row['axis']+'.json'))==row
                receipt(d/('wrapper_'+row['axis']+'.json'))
                wrappers.setdefault((task['T'],task['cell']['id']),[]).append(dict(replica=task['replica'],seed=task['seed'],
                    axis=row['axis'],metrics=row['metrics'],fingerprint=row['fingerprint'],event_digest=row['event_digest'],
                    compiler_hash=row['compiler_hash'],phase=task['phase'],origin_SOURCE=cfg['SOURCE']))
                resource_rows.append(dict(T=task['T'],cell_id=task['cell']['id'],replica=task['replica'],
                    axis=row['axis'],stage=mode,wall_seconds=row['wall_seconds'],
                    peak_RSS_bytes=row['peak_rss_bytes'],**row['metrics']))
            if task['validate']:
                v=read(d/'control_validation.json');assert v['max_error']<1e-9;native_errors.append(v['max_error']);receipt(d/'control_validation.json')
    assert group_count==72 and calls['compile']==144 and calls['trajectory']==64 and calls['occurrence']==320
    assert calls['control_probe']==40 and calls['primitive']==750 and calls['reference_matvec']==800
    assert len(all_retired)==74 and not Path('/proc/'+str(wait['PID'])).exists()
    for r in all_retired:assert not Path('/proc/'+str(r['pid'])).exists(), 'PRESENT_HISTORICAL_PID_NOT_ASSUMED_OWN'
    # T=.8: byte-bound old signals and separate exploration/confirmation tables.
    old_signal=read(Path(next(r['path'] for r in cfg['files'] if r['logical_name']=='signal_summary.json')))['signals']
    wanted={c['id'] for c in cells(4) if c['method']!='B0I'};signals[.8]=[deepcopy(s) for s in old_signal if s['cell']['id'] in wanted]
    saved=read(ROOT/'artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/T08_saved_panel_v1.json')
    overlay=deepcopy(saved['H6_B0I_p15_q4']['signal']);overlay.update(cell=cells(4)[1],log_B=0.);signals[.8].append(overlay)
    exploration=committed_csv(OLD_BASE+'track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv')
    confirmation=committed_csv(OLD_BASE+'track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv')
    for row in exploration+confirmation:
        if row['cell_id'] not in wanted:continue
        if (row['cell_id'].startswith('B2') and row['phase']!='confirmation') or (not row['cell_id'].startswith('B2') and row['phase']!='exploration'):continue
        assert row['compiler_hash']==cfg['compiler_hash'] and row['control']=='symmetric_directional'
        wrappers.setdefault((.8,row['cell_id']),[]).append(dict(replica=int(row['replica']),seed=int(row['seed']),axis=row['axis'],
            metrics={m:int(row[m]) for m in METRICS},event_digest=row['event_digest'],fingerprint=row['fingerprint'],
            compiler_hash=row['compiler_hash'],phase=row['phase'],origin_result_commit=OLD_COMMIT))
    axis_path=ROOT/'artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv'
    for row in csv.DictReader(axis_path.open()):
        if (int(row['prefix']),int(row['q']))!=(15,4):continue
        assert row['compiler_hash']==cfg['compiler_hash']
        wrappers.setdefault((.8,cells(4)[1]['id']),[]).append(dict(replica=0,seed=None,axis=row['axis'],metrics={m:int(row[m]) for m in METRICS},
            fingerprint=row['numeric_fingerprint'],compiler_hash=row['compiler_hash'],phase='deterministic',origin_SOURCE=row['origin_SOURCE']))
    accounting=[];signal_rows=[];checks=0
    for T in sorted(signals):
        assert {s['cell']['id'] for s in signals[T]}=={c['id'] for c in cells(int(T/.2))}
        for s in signals[T]:
            key=(T,s['cell']['id']);rows=wrappers[key];n=32 if s['cell']['method']=='B2' else 1
            assert len(rows)==2*n and {r['replica'] for r in rows}==set(range(n))
            dec=signed_parts(s);signal_rows.append(dict(T=T,cell=s['cell'],reference=s['reference'],signals=s['signals'],
                decomposition=dec,log_B=s['log_B'],B=math.exp(s['log_B']),allowance=s['allowance']))
            for eps in EPSILONS:
                for factor in (1.,10.):
                    acc=account(s,eps,factor);shots={a:acc['axes'][a]['shots'] for a in ('real','imag')} if acc['cost_acquisition_eligible'] else None
                    stats={m:paired_metrics(rows,shots,m) for m in METRICS}
                    entry=dict(T=T,cell_id=s['cell']['id'],method=s['cell']['method'],epsilon=eps,factor=factor,
                        accounting=acc,metrics=stats,population_mean_certified=False);accounting.append(entry)
                    if T in (.4,1.6):
                        old=s['precision_rows'][str(eps)][str(factor)]
                        assert acc['N_total']==old['N_total'] and acc['cost_acquisition_eligible']==old['cost_acquisition_eligible'];checks+=1
                    elif s['cell']['method']!='B0I':
                        old=next(r for r in saved['H6_CONFIRMATION_UNION8'] if r['cell_id']==s['cell']['id'] and r['epsilon']==eps and r['factor']==str(factor))
                        assert acc['N_total']==old['accounting']['N_total']
                        if shots:
                            for m in METRICS:assert stats[m]['G_exact']==old['metrics'][m]['G_exact'];checks+=1
                    else:
                        old=next(r for r in saved['H6_B0I_p15_q4']['precision'] if r['epsilon']==eps and r['factor']==factor)
                        assert acc['N_total']==old['accounting']['N_total'];checks+=1
    comparisons=[]
    for T in sorted(signals):
        for eps in EPSILONS:
            for factor in (1.,10.):
                rs=[r for r in accounting if (r['T'],r['epsilon'],r['factor'])==(T,eps,factor)]
                pr=next(r for r in rs if r['method']=='B2')
                for m in METRICS:
                    ds=[r for r in rs if r['method']!='B2' and r['metrics'][m]['G'] is not None]
                    best=min(ds,key=lambda r:Fraction(int(r['metrics'][m]['G_exact']['numerator']),int(r['metrics'][m]['G_exact']['denominator']))) if ds else None
                    Gp=pr['metrics'][m]['G'];Gd=best['metrics'][m]['G'] if best else None
                    ratio=Gp/Gd if Gp is not None and Gd is not None else None
                    comparisons.append(dict(T=T,epsilon=eps,factor=factor,metric=m,PR_cell=pr['cell_id'],
                        deterministic_cell=best['cell_id'] if best else None,PR_G=Gp,deterministic_G=Gd,ratio=ratio,
                        reduction_percent=100*(1-ratio) if ratio is not None else None,
                        PR_SE=pr['metrics'][m]['SE'],ratio_SE=pr['metrics'][m]['SE']/Gd if ratio is not None else None,
                        PR_eligible=pr['accounting']['cost_acquisition_eligible'],PR_shots=pr['accounting']['N_total'],
                        deterministic_shots=best['accounting']['N_total'] if best else None,
                        population_guarantee=False,scope='fixed five-cell panel; confirmation32 only'))
    # Same delta/r and input => identical local law; old saved order diagnostic fixes it.
    rare_path=OLD_BASE+'track_a_h6_confirmation_summary_v1/2026-10-12/rare_order_diagnostics_v1.json'
    rare=json.loads(subprocess.check_output(['git','show',OLD_COMMIT+':'+rare_path],cwd=ROOT))
    oldrare=next(r for r in rare['rows'] if r['cell_id']=='B2_p5_q4_2nd_r4_K2')
    orders[.8]=Counter({int(k):v for k,v in oldrare['observed_counts'].items()});p2=oldrare['orders']['2'];rare_rows=[]
    for T in sorted(signals):
        R=int(T/.2)*4;prob=-math.expm1(R*math.log1p(-p2))
        assert sum(orders[T].values())==R*32
        rare_rows.append(dict(T=T,R=R,n=32,observed_orders=dict(orders[T]),order2_probability=p2,
            probability_trajectory_contains_order2=prob,probability_no_order2_in32=math.exp(R*32*math.log1p(-p2)),
            mean_cost_tail_bound=None,zero_observed_is_zero_contribution=False))
    for name,data in dict(signal_summary=signal_rows,accounting=accounting,comparisons=comparisons,rare_orders=rare_rows,
            wrapper_projection=[dict(T=T,cell_id=cid,**r) for (T,cid),rows in sorted(wrappers.items()) for r in rows],
            input_receipts=inventory).items():
        (out/(name+'_v1.json')).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    primary=[r for r in comparisons if r['metric']=='RZ' and r['factor']==1.]
    for name,rows in (('primary_RZ_P0_v1.csv',primary),('all_metric_comparisons_v1.csv',comparisons),
            ('execution_resources_v1.csv',resource_rows)):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    audit=dict(status='COMPLETE_SAVED_RESULTS_AUDITED_MANDATORY_STOP',SOURCE=cfg['SOURCE'],run_id=wait['run_id'],
        finish_UTC=wait['finish_UTC'],native_exit=0,total_wall_seconds=wait['wall_seconds'],stages=stage_stats,
        groups=72,wrappers=144,PR_trajectories=64,signal_jobs=2,all_74_workers_exit0_reaped=True,
        own_recorded_PIDs_present=0,cleanup_errors=[],first_stop=None,calls=dict(calls),
        cumulative_caps=cfg['cumulative_caps'],carry_preserved=True,old_T08_trajectory32_not_new=True,
        saved_arithmetic_checks=checks,primitive_validation=primitive_errors,max_control_probe_error=max(native_errors),
        runtime_SOURCE_files_unchanged=all(sha(ROOT/n)==h for n,h in cfg['source_hashes'].items()),
        bound_input_files_unchanged=all(sha(r['path'])==r['sha256'] for r in cfg['files']),
        new_science_during_analysis=0,independent_review=False,subagents=0,formal_population_claim=False,
        source_input_profile_verified_at_launch=True,profile_rechecked_in_this_script=False,
        receipt_count=len(inventory),receipt_digest=digest(inventory),old_result_commit=OLD_COMMIT)
    assert audit['runtime_SOURCE_files_unchanged'] and audit['bound_input_files_unchanged']
    (out/'completion_audit_v1.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(primary=primary,rare=rare_rows,audit=audit),ensure_ascii=False))


if __name__=='__main__':main()
