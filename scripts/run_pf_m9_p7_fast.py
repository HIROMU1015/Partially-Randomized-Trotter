"""Same three p7 m9 cells, alongside existing disjoint Full workers.

Original m9 selector/child, scientific gates, price bindings and publication
guard are reused. Only scheduling, fast canonical hashing and exact repeated
native-part reuse differ. The running Full SOURCE is never modified.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_pf_m9_prices as M

MODULE = 'scripts/run_pf_m9_p7_fast.py'
ACCEL = 'scripts/pf_native_price_fast.py'
FAST_HASH = 'scripts/h46_b3_fast_fingerprint.py'
ROUTE = 'TRUNCATED_PF'
SOURCES = (MODULE, ACCEL, FAST_HASH, M.MODULE, M.F.CALLER, M.T.CALLER, M.S.MODULE, 'scripts/pf_m6_holdout_evidence.py')


def full_failure_check():
    rel, private = M.paths('FULL_PF')
    for ident, _ in M.TARGETS['FULL_PF'].values():
        M.check(not (private/'cost'/M.F.COHORT/ident/'0/STOP.json').exists(), 'Full gate failed; stop new p7 work')
    if (private/'child.json').exists():
        data=json.loads((private/'child.json').read_text())
        M.check(len(data['results'])==3 and all(r['ok'] for r in data['results']), 'Full child failed; stop new p7 work')
    if (ROOT/rel/'RESULT.json').exists():
        M.check(json.loads((ROOT/rel/'RESULT.json').read_text())['status']=='M9_PRICES_COMPLETE','Full acquisition failed; stop new p7 work')


def disjoint_preflight(cells):
    full_failure_check()
    base = M.S.duplicate_preflight(ROOT, M.T.stage_spec(), cells)
    target = {c['row']['id'] for c in cells}; jobs = []
    for pid in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid='],text=True).split():
        if int(pid) == os.getpid(): continue
        try: args = Path('/proc',pid,'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, PermissionError): continue
        if not args or not Path(args[0]).name.startswith('python'): continue
        if not any(Path(a).name in (Path(M.MODULE).name,Path(MODULE).name) for a in args[1:]): continue
        if '--child' in args:
            job_path = Path(args[args.index('--child')+1]); job = json.loads(job_path.read_text())
            ids = {c['row']['id'] for c in job['cells']}
            M.check(not target & ids, 'Overlapping m9 worker already running')
            jobs.append(dict(pid=int(pid),job=str(job_path),physical_ids=sorted(ids),status='DISJOINT_FROM_P7_TARGETS'))
        elif '--execute-approved-route' in args:
            route = args[args.index('--execute-approved-route')+1]
            M.check(route == 'FULL_PF', 'Unresolved m9 acquisition parent')
            jobs.append(dict(pid=int(pid),route=route,status='DISJOINT_FULL_ROUTE_PARENT'))
        elif '--execute-approved-p7' in args:
            raise ValueError('Another p7 fast caller is running')
    rel, private = M.paths(ROUTE)
    M.check(not (ROOT/rel).exists() and not private.exists(), 'p7 namespace already entered; no retry')
    return dict(base, m9_jobs=jobs, scheduling='FULL_AND_P7_DISJOINT_CELLS_MAY_RUN_CONCURRENTLY')


def child(job_path):
    job = json.loads(Path(job_path).read_text())
    M.check(job['route'] == ROUTE, 'Only the three fixed p7 cells supported')
    for path in SOURCES:
        M.check((ROOT/path).read_bytes() == subprocess.check_output(['git','show',job['execution_SOURCE']+':'+path],cwd=ROOT), 'Fast execution SOURCE changed: '+path)
    # Provider installed before importing the hasher or scientific libraries.
    import h46_long_time_signal as signal
    signal.install(); M.T.provider_checks()
    import h46_long_time_native as native
    from pf_native_price_fast import accelerated_native
    full_failure_check()
    with accelerated_native(native):
        return M.child(job_path)


def execute():
    gate = M.preopen(); doc, templates, _ = M.templates_and_selection()
    cells = [c for p in templates[ROUTE] for c in p['cells']]
    source_commit = M.S.source_gate(ROOT, SOURCES)
    duplicate = disjoint_preflight(cells)
    # Preserve the original saved-tail gate and runtime input contract.
    extra = M.T.saved_tail_scalar(ROOT)
    rel, private = M.paths(ROUTE); out = ROOT/rel
    out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    scalar = next(r['scalar']['value'] for r in doc['rows'] if r['model_group'][0] == ROUTE)
    job = dict(route=ROUTE,cells=cells,public=str(out/'child_public'),private=str(private),
        result=str(private/'child.json'),execution_SOURCE=source_commit,expected_scalar=scalar,**extra)
    M.write(private/'job.json',job)
    started=time.monotonic()
    code=subprocess.call([sys.executable,'-B',str(ROOT/MODULE),'--child',str(private/'job.json')],cwd=ROOT)
    data=json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else dict(results=[],resources=None,error='Child result missing')
    good=[r for r in data['results'] if r['ok']]
    source=dict(M.T.stage_spec()['source'],commit=source_commit,caller=MODULE,underlying_caller=M.T.CALLER,
        stage='M9_HOLDOUT',predictions=M.PRED,policy_source=M.POLICY,route=ROUTE,
        acceleration=dict(source=dict(commit=source_commit,path=ACCEL),fast_hash=dict(commit=source_commit,path=FAST_HASH),scope='ONE_COMPOSE_EXACT_ARGUMENT_TEMPLATE_REUSE',canonical_SHA_unchanged=True))
    prices,sides,proofs=[],[],{};error=None
    try:
        for result in good:
            p=next(p for p in templates[ROUTE] if p['formula']==result['cell']['formula'])
            a,b,c=M.T.price_bindings(ROOT,p['plan'],result['cell'],result['trial'],price_source=dict(source,record='RESULT.json'),evidence_commit=M.S.PLACEHOLDER,evidence_path=rel+'/IMPLEMENTATION_EVIDENCE.json')
            for price in a:
                price['diagnostic'].update(stage='M9_HOLDOUT',compiled_wrapper=result['wrappers']['cosine' if price['axis']=='X' else 'sine'])
                pred=next(r for r in doc['rows'] if r['configuration_id']==price['configuration_id'] and r['axis']==price['axis'])
                M.check(price['m']==9 and price['physical_context']['physical_candidate']==pred['physical_candidate'] and price['price_profile']==pred['domain']['profile'],'Acquired domain changed')
            prices+=a;sides+=b;proofs.update(c)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);prices,sides,proofs=[],[],{}
    M.write(out/'PRICES.json',prices);M.write(out/'SIDECARS_UNBOUND.json',sides);M.write(out/'IMPLEMENTATION_EVIDENCE.json',proofs)
    after=M.preopen()['protected_snapshots']
    ok=code==0 and len(good)==3 and error is None and after==gate['protected_snapshots']
    M.write(out/'RESULT.json',dict(status='M9_PRICES_COMPLETE' if ok else 'STOP_M9_ACQUISITION',route=ROUTE,source=source,
        predictions_sha256=M.PRED_SHA,duplicate_preflight=duplicate,context_gate=data.get('context_gate'),results=data['results'],error=data.get('error'),record_error=error,child_exit=code,
        resources=data['resources'],elapsed_seconds=time.monotonic()-started,protected_before=gate['protected_snapshots'],protected_after=after,
        valid_axis_records=len(prices),finished_UTC=datetime.now(timezone.utc).isoformat(),retry=False,fit_calls=0,prediction_recalculations=0,mandatory_stop=True))
    print(json.dumps(dict(route=ROUTE,status='M9_PRICES_COMPLETE' if ok else 'STOP_M9_ACQUISITION',output=rel)))
    return 0 if ok else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-approved-p7',action='store_true');p.add_argument('--child',type=Path);a=p.parse_args()
    if a.child:sys.exit(child(a.child))
    if a.execute_approved_p7:sys.exit(execute())
    p.error('Fixed p7 execute or child operation required')
