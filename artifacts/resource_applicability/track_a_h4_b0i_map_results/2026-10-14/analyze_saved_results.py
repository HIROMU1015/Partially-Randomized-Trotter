"""B0+I measured map: stdlib-only posthoc analysis, never builds or compiles.

Collection additionally checks native result bindings against the fixed runner.
Public verification reads only committed projections and the original saved map.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
PREP = REPO / 'artifacts/resource_applicability/track_a_h4_b0i_map_preparation/2026-10-14-parallel-run02'
OLD = REPO / 'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json'
OLD_GRID = REPO / 'artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/precision_grid_1812.csv'
AXES = ('cosine', 'sine')
GEOMETRIES = ('0.70', '0.80', '0.90', '1.10', '1.40', '1.60')


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def byte_identity(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            sha.update(block)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest())


def shots(bias, normalization, epsilon):
    h = [epsilon/math.sqrt(2)-bias[a] for a in AXES]
    return None if min(h) <= 0 else [math.ceil(2*normalization**2/x**2*math.log(80)) for x in h]


def paired_work(pairs, counts):
    values = [sum(n*c for n, c in zip(counts, pair)) for pair in pairs]
    n = len(values)
    mean = Fraction(sum(values), n)
    variance_of_mean = (Fraction(sum(v*v for v in values), n*(n-1))
                        - Fraction(sum(values)**2, n*n*(n-1))) if n > 1 else Fraction(0)
    return mean, math.sqrt(variance_of_mean)


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(wrappers):
    old = load(OLD)
    old_candidates = {(r['geometry'], r['template_id']): r for r in old['candidates']}
    old_by_geometry = defaultdict(list)
    for r in old['candidates']:
        pairs = r['saved']['paired_costs']
        # Means have denominator 1 or 32; these integer metrics are exactly binary-representable.
        means = [sum(p[i] for p in pairs)/len(pairs) for i in (0,1)]
        old_by_geometry[r['geometry']].append((r,means))
    assert len(wrappers) == 240
    expected = {(g, p, q, a) for g in GEOMETRIES for p in (3,4,5,6,9) for q in (1,2,4,8) for a in AXES}
    assert {(r['geometry'],r['prefix'],r['q'],r['axis']) for r in wrappers} == expected
    groups = {}
    metric_deltas = defaultdict(Counter)
    for r in wrappers:
        original = old_candidates[(r['geometry'], r['original_B0_template'])]
        for metric, value in r['metrics'].items():
            baseline = original['paired_metric_axis_means'][r['axis']][metric]
            assert baseline == int(baseline)
            metric_deltas[metric][value-int(baseline)] += 1
        key = (r['geometry'], r['prefix'], r['q'])
        c = groups.setdefault(key, dict(geometry=key[0], prefix=key[1], q=key[2],
            template_id=r['template_id'], bias=r['saved_bias_B0I'], normalization=1., costs={}))
        assert c['bias'] == r['saved_bias_B0I']
        c['costs'][r['axis']] = r['metrics']['rz_count']
    candidates = sorted(groups.values(), key=lambda c:(c['geometry'],c['prefix'],c['q']))
    assert len(candidates) == 120
    with OLD_GRID.open(newline='') as f:
        old_grid = list(csv.DictReader(f))
    grid, selected = [], []
    counts = {g: Counter() for g in GEOMETRIES}
    ranges = {g: [] for g in GEOMETRIES}
    for row in old_grid:
        g, eps = row['geometry'], float(row['epsilon'])
        original_eligible = []
        for r,means in old_by_geometry[g]:
            ns = shots(r['saved']['bias'],r['saved']['normalization'],eps)
            if ns is not None:
                original_eligible.append((sum(n*m for n,m in zip(ns,means)),r['template_id']))
        old_min = min(original_eligible)
        assert old_min[1] == row['winner'] == row['B2_template']
        assert math.isclose(old_min[0],float(row['B2_G']),rel_tol=1e-12)
        eligible = []
        for c in candidates:
            if c['geometry'] != g: continue
            ns = shots(c['bias'], 1., eps)
            if ns is not None:
                work = sum(n*c['costs'][a] for n,a in zip(ns,AXES))
                eligible.append((work, c['prefix'], c['q'], c, ns))
        best = min(eligible, key=lambda x:x[:3]) if eligible else None
        b2 = old_candidates[(g,row['B2_template'])]
        ns2 = shots(b2['saved']['bias'], b2['saved']['normalization'], eps)
        assert ns2 is not None
        work2, se2 = paired_work(b2['saved']['paired_costs'], ns2)
        assert math.isclose(float(work2),float(row['B2_G']),rel_tol=1e-12)
        assert math.isclose(se2,float(row['SE']),rel_tol=1e-10,abs_tol=1e-6)
        assert row['winner'] == row['B2_template']
        winner = 'B0_PLUS_I' if best and best[0]<work2 else 'B2'
        tie = bool(best and best[0]==work2)
        counts[g][winner] += 1
        separated = bool(best and best[0]<work2-2*se2)
        counts[g]['B0I_below_B2_point_minus_2SE'] += int(separated)
        counts[g]['ties'] += int(tie)
        if winner == 'B0_PLUS_I': ranges[g].append(eps)
        grid.append(dict(geometry=g, epsilon=eps, B0I_eligible_candidates=len(eligible),
            B0I_template=best[3]['template_id'] if best else '',
            B0I_prefix=best[1] if best else '', B0I_q=best[2] if best else '',
            B0I_N_cos=best[4][0] if best else '', B0I_N_sin=best[4][1] if best else '',
            B0I_RZ_cos=best[3]['costs']['cosine'] if best else '',
            B0I_RZ_sin=best[3]['costs']['sine'] if best else '',
            B0I_work=best[0] if best else '', B2_template=row['B2_template'],
            B2_N_cos=ns2[0], B2_N_sin=ns2[1], B2_work=float(work2),
            B2_work_numerator=work2.numerator, B2_work_denominator=work2.denominator,
            B2_paired_SE=se2, winner=winner, tie=tie,
            B0I_reduction_percent=100*(1-best[0]/float(work2)) if best else '',
            B0I_below_B2_point_minus_2SE=separated))
        if eps in (.005,.05,.1): selected.append(grid[-1])
    assert len(grid) == 1812 and len(selected) == 18
    summary = dict(schema_version='h4-b0i-measured-map-posthoc-v1',
        scope='saved complex-signal accuracy, shots times measured RZ, state preparation P=0',
        model='linear H4 STO-3G, frozen DF rank12, structured Gaussian, ordinary control, S2',
        T=.8, delta_window=[.1,.8], prefixes=[3,4,5,6,9], q=[1,2,4,8],
        execution_source=wrappers[0]['source_commit'],
        dataset_identity_source=old['dataset_identity_source'],
        wrapper_count=240, new_compile_calls=sum(r['new_compile_invocations'] for r in wrappers),
        verified_pilot_reuses=sum(r['verified_pilot6_reuse'] for r in wrappers),
        actual_B0I_candidates=120, geometry_precision_points=1812,
        selected_comparisons=selected, point_minimum_counts={g:dict(v) for g,v in counts.items()},
        B0I_winning_grid_ranges={g:dict(count=len(v), minimum=min(v) if v else None,
            maximum=max(v) if v else None) for g,v in ranges.items()},
        metric_delta_histograms={m:{str(k):v for k,v in c.items()} for m,c in metric_deltas.items()},
        B0I_point_minimum_total=sum(c['B0_PLUS_I'] for c in counts.values()),
        B2_point_minimum_total=sum(c['B2'] for c in counts.values()),
        B0I_below_B2_point_minus_2SE_total=sum(c['B0I_below_B2_point_minus_2SE'] for c in counts.values()),
        uncertainty='B2 uses the same 32 paired trajectories for selection and estimation; descriptive SE, no simultaneous or population winner guarantee',
        continuous_precision_boundaries=False, energy_RPE_or_hardware_cost=False,
        additional_compile_calls_in_posthoc_analysis=0, scientific_array_reads_in_posthoc_analysis=0,
        independent_scientific_review=False, automatic_next_stage=False)
    return candidates, grid, selected, summary


def collect(run_root):
    sys.path.insert(0,str(REPO/'src'))
    from trottertracks.resource_applicability.h4_b0i_map.runtime import accept_result, source_gate
    from trottertracks.resource_applicability.h4_b0i_map.contract import authorize, digest
    plan, auth, review = [load(PREP/n) for n in ('plan.json','authorization.json','review.json')]
    authorize(plan,auth,review);source_gate(REPO,plan)
    closure_git = []
    for item in plan['source_closure']:
        raw = subprocess.check_output(['git','-C',str(REPO),'show',plan['source_commit']+':'+item['path']])
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        closure_git.append(item)
    for item in load(PREP/'launch_manifest.json')['files']:
        assert byte_identity(REPO/item['path']) == {k:item[k] for k in ('bytes','sha256')}
    output, control = Path(plan['output']), Path(plan['control'])
    assert output.parent.parent == run_root
    terminal=load(output/'terminal.json');started=load(output/'started.json')
    assert terminal['status']=='MAP240_COMPLETE_STOP' and terminal['first_stop_reason'] is None
    assert terminal['cleanup_complete'] is True and terminal['native_worker_exit_codes']==[0]*12
    assert terminal['native_observer_exit_code']==-15 and terminal['records']==240
    assert started['source']==plan['source_commit'] and started['plan_digest']==digest(plan)
    assert terminal['carry']==plan['carry'] and terminal['actual_consumed_or_reserved']==234
    assert len(list(output.glob('reservation-*.json')))==234
    assert {p.name for p in output.glob('task-*.json')} == {'task-'+t['task_id']+'.json' for t in plan['tasks']}
    wrappers=[]; native=[]
    for t in plan['tasks']:
        path=output/('task-'+t['task_id']+'.json')
        r=accept_result(plan,t,load(path))
        public={k:r[k] for k in ('task_id','source_commit','plan_digest','numeric_circuit_fingerprint',
            'validation_wrapper_fingerprint','metrics','new_compile_invocations','verified_pilot6_reuse',
            'build_fingerprint_wall_seconds','transpile_wall_seconds','worker_lifetime_peak_RSS_bytes')}
        public.update(geometry=t['geometry'], prefix=t['prefix'], q=t['template']['q'],axis=t['axis'],
            template_id=t['template']['template_id'],original_B0_template=t['original_B0_template'],
            saved_bias_B0I=t['saved_bias_B0I'], input_identity=t['input_identity'],
            frozen_input_byte_identity={k:t['input_file'][k] for k in ('bytes','sha256')})
        wrappers.append(public)
        native.append(dict(path=path.name,**byte_identity(path)))
    rows=[json.loads(line) for line in (control/'observer.jsonl').read_text().splitlines()]
    obs=[r for r in rows if r.get('kind')=='observation']
    assert obs and all(r.get('first_failure') is None for r in obs)
    retirement=load(output/'observer-retirement.json')
    assert set(retirement['complete_task_ids']) == {t['task_id'] for t in plan['tasks']}
    workers=load(output/'workers.json')['identities'];launch=load(run_root/'launch_attempt.json')
    identities=workers+[retirement['observer_identity'],dict(pid=launch['driver_pid'],start=launch['driver_start'],uid=launch['uid'])]
    residual=[]
    for ident in identities:
        proc=Path('/proc')/str(ident['pid'])
        if proc.exists():
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[19]==str(ident['start']) and proc.stat().st_uid==ident['uid']:residual.append(ident)
    assert not residual
    files=[output,control]+list(output.rglob('*'))+list(control.rglob('*'))+[Path(plan['one_shot_lock']),run_root/'driver.log',run_root/'launch_attempt.json']
    assert not any(p.is_symlink() for p in files)
    charged=sum(p.stat().st_blocks*512 for p in files)
    native_receipt=dict(terminal=terminal,source_closure_match_count=len(closure_git),
        launch_manifest_unchanged=True, runtime_binding_checks='PASS_240_ACCEPT_RESULT',
        native_result_identities=native, observer_observations=len(obs),
        max_observation_interval_seconds=max(r['interval_seconds'] for r in obs),
        max_observation_duration_seconds=max(r['observation_duration_seconds'] for r in obs),
        max_observation_staleness_seconds=max(r['staleness_seconds'] for r in obs),
        max_psi_full_avg10=max(v for r in obs for v in r['memory']['psi_full_by_scope'].values()),
        oom_counter_change={k:obs[-1]['memory']['oom_events'][k]-obs[0]['memory']['oom_events'][k] for k in obs[0]['memory']['oom_events']},
        peak_worker_native_VmHWM_bytes=max(r['worker_lifetime_peak_RSS_bytes'] for r in wrappers),
        VmHWM_scope='worker lifetime up to each COMPLETE receipt, not isolated task or final-after-cleanup peak',
        max_observer_sampled_RSS_bytes=max(r['observer']['rss'] for r in obs),
        observed_owned_identities_absent=len(identities), native_driver_exit_code=None,
        driver_exit_code_note='original tool session unavailable; do not infer native exit code from successful terminal JSON',
        storage_scope='output/control tree entries, one-shot lock, driver log, launch receipt; frozen inputs excluded',
        stage_storage_logical_bytes=sum(p.stat().st_size for p in files if p.is_file()),
        stage_storage_allocated_bytes=charged, stage_storage_entries=len(files),
        cumulative_actual_invocations=plan['carry']['actual_invocations']+234,
        cumulative_charge_bytes=plan['carry']['charged_bytes']+charged,
        cumulative_recorded_wall_seconds=plan['carry']['recorded_wall_seconds']+terminal['stage_wall_seconds'],
        scientific_array_reads_in_posthoc_analysis=0)
    for path in (output/'terminal.json', output/'workers.json', output/'observer-retirement.json',control/'observer.jsonl',run_root/'driver.log'):
        native_receipt.setdefault('native_provenance',[]).append(dict(path=path.name,**byte_identity(path)))
    return wrappers,native_receipt


def verify():
    manifest=load(BUNDLE/'artifact_manifest.json')
    for name, ident in manifest['files'].items(): assert byte_identity(BUNDLE/name)==ident,name
    for name, ident in manifest['dependencies'].items(): assert byte_identity(REPO/name)==ident,name
    _,grid,selected,summary=analyze(load(BUNDLE/'compiled_wrapper_metrics.json'))
    assert summary==load(BUNDLE/'summary.json')
    assert load(BUNDLE/'measured_candidates.json')==analyze(load(BUNDLE/'compiled_wrapper_metrics.json'))[0]
    with (BUNDLE/'precision_grid_1812.csv').open(newline='') as f:
        saved=list(csv.DictReader(f))
    assert saved==[{k:str(v) for k,v in r.items()} for r in grid]
    with (BUNDLE/'selected_precision18.csv').open(newline='') as f:
        assert list(csv.DictReader(f))==[{k:str(v) for k,v in r.items()} for r in selected]
    return dict(status='PASS_PUBLIC_SAVED_VALUE_REPRODUCTION',wrappers=240,candidates=120,
        precision_points=1812,exact_fraction_B2_comparisons=1812,
        private_runtime_access=0,NPZ_access=0,new_compile_calls=0,
        independent_scientific_review=False)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    signal.alarm(60)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-run-root',type=Path)
    args=parser.parse_args()
    if args.collect_run_root:
        wrappers,native=collect(args.collect_run_root)
        candidates,grid,selected,summary=analyze(wrappers)
        for name,value in [('compiled_wrapper_metrics.json',wrappers),('native_resource_receipt.json',native),
                ('measured_candidates.json',candidates),('summary.json',summary)]:dump(BUNDLE/name,value)
        table(BUNDLE/'precision_grid_1812.csv',grid);table(BUNDLE/'selected_precision18.csv',selected)
        deps=[OLD,OLD_GRID,PREP/'plan.json',PREP/'authorization.json',PREP/'review.json',PREP/'launch_manifest.json']
        names=['compiled_wrapper_metrics.json','native_resource_receipt.json','measured_candidates.json',
            'summary.json','precision_grid_1812.csv','selected_precision18.csv','analyze_saved_results.py']
        dump(BUNDLE/'artifact_manifest.json',dict(files={n:byte_identity(BUNDLE/n) for n in names},
            dependencies={str(p.relative_to(REPO)):byte_identity(p) for p in deps}))
    result=verify()
    print(json.dumps(result))


if __name__=='__main__':main()
