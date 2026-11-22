"""Run only the stock worker readiness entry on saved requests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(os.environ.get('M13_SERVER2_EVIDENCE_ROOT', Path(__file__).resolve().parent)).resolve()
FLEET = HERE/'fleet'
PROJECTS = HERE.parents[3]
sys.path.insert(0, str(FLEET/'src'))
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
from trottertracks import rpe_host_profiles as P, rpe_q3_coordinator as Q

REAL = 'ac869a3dc239cf65489aacaf200cb4f113880c28'
DOCS = 'docs/research/rpe_m13_real_validation_20261122/'
SOURCE_SUMMARY = DOCS+'SUMMARY.json'
SPOOL = PROJECTS/'rpe-romutail-worker/server2/spool'
CURRENT = PROJECTS/'rpe-release-fleet/metadata/romutail-h4-rerun4-1355ad65fa/PROFILE.json'
RISK = '89fcde14f6ac5f4b97be234f9b0822091e7d9ac7'
FAMILIES = ('FULL_PF', 'TRUNCATED_PF', 'PR_DF')
PYTHON = str(PROJECTS/'h6-environments/20261012/native-dependency-v2/venv/bin/python')


def get(path):
    return json.loads(path.read_bytes())


def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def git(*args):
    return subprocess.check_output(['git','-C',str(FLEET),*args])


def published(path):
    return git('show', REAL+':'+path)


def old_assignments():
    paths = {}
    m13 = []
    for path in sorted(SPOOL.glob('*/*/ASSIGNMENT.json'), key=lambda p:p.stat().st_mtime, reverse=True):
        assignment = get(path)
        req = assignment.get('request',{})
        method, n = req.get('method'), len(req.get('N_X') or [])
        if n==14 or req.get('route',{}).get('horizon_risk_source'):
            m13.append(str(path))
        if method in FAMILIES and n in (12,13) and path.with_name('WORKER_PROFILE.json').is_file():
            if assignment.get('host_execution_profile'):
                paths.setdefault((method,n-1), path)
    return paths,m13


def phase_radii(request):
    loc, = request['route']['templates']
    blob = json.loads(git('show',loc['commit']+':'+loc['path']))
    for key in loc.get('inputs_selector', []):
        blob=blob[key]
    return blob['options']['phase_radii']


def prepare(run):
    run.mkdir()
    old, actual_m13 = old_assignments()
    assert not actual_m13, f'Use the actual central assignments first: {actual_m13}'
    base = get(CURRENT)
    assert base['host_id']=='server2'
    base['work_dir'] = str(run/'unused-spool')
    base['prefix_cache_directory'] = str(run/'unused-prefix')
    manifest = dict(commit=git('rev-parse','HEAD').decode().strip(),files={})
    for path in ('scripts/run_rpe_optimizer_q3_job.py',
                 'src/trottertracks/rpe_q3_evidence.py','src/trottertracks/rpe_dp_prefix_reuse.py',
                 'src/trottertracks/rpe_q3_round_tables.py','src/trottertracks/rpe_large_shot_bounds.py'):
        data=(FLEET/path).read_bytes()
        assert data==git('show','HEAD:'+path)
        manifest['files'][path]=hashlib.sha256(data).hexdigest()
    risk_view = PROJECTS/'partially-randomized-trotter-worktrees/rpe-q3-m13-risk-89fcde14f'
    rows = json.loads(published(SOURCE_SUMMARY))['rows']
    cases=[]
    for row in rows:
        method=row['method']
        dest=run/f'{method}-M13'
        dest.mkdir()
        name=method+'_Q3_REQUEST'+('_V2' if method=='PR_DF' else '')+'.json'
        raw=published(DOCS+name)
        (dest/'PUBLISHED_REQUEST.json').write_bytes(raw)
        req=json.loads(raw)
        original=published(row['original_outcome'])
        assert hashlib.sha256(original).hexdigest()==row['original_sha256']
        original_identity=json.loads(original)['identity']
        assert original_identity['configuration_id']==req['configuration_id']==row['configuration_id']
        assert original_identity['M']==13
        # The earlier standalone caller carried its M=12 expected values. Bind the
        # saved M=13 outcome identity for a worker-envelope acceptance check.
        req['expected']={k:original_identity[k] for k in ('configuration_id','binding_id','total_configuration_id')}
        req['expected']['signal_source']=original_identity['signal_packet']
        assert len(req['N_X'])==len(req['N_Y'])==len(phase_radii(req))==14
        loc=req['expected']['signal_source']
        signal=json.loads(git('show',loc['commit']+':'+loc['path']))
        for key in loc.get('selector',[]):
            signal=signal[key]
        if 'means' not in signal and 'signal_packet' in signal:
            signal=signal['signal_packet']
        assert [r['m'] for r in signal['means']]==list(range(14)), signal.keys()
        r=req['route']
        staged=deepcopy(base)
        selected=P.select_host_profile(staged, {'request':{'method':method, 'route':{k:r[k] for k in ('kind','view_commit')}}})
        selected.update(adapter_root=str(FLEET),adapter_manifest=manifest,interpreter=r['interpreter'])
        selected['route_views'][RISK]=str(risk_view)
        for key in ('work_dir','prefix_cache_directory'):
            selected[key]=str(dest/key)
        staged.setdefault('route_profiles',[]).append(dict(
            route_kind=r['kind'],view_commit=r['view_commit'],horizon_risk_commit=RISK,profile=selected))
        a=dict(job_id=req['request_id'],attempt_id='local-readiness-only',method=method,request=req,
               request_sha256=P.digest(req),
               source_manifest=Q.source_manifest(Path(base['root']),git('-C',base['root'],'rev-parse','HEAD').decode().strip()),
               input_hashes=Q.input_hashes(Path(base['root']),req))
        a['host_execution_profile']=P.profile_core(P.select_host_profile(staged,a))
        a['host_execution_profile_sha256']=P.digest(a['host_execution_profile'])
        put(dest/'ASSIGNMENT.json',a)
        put(dest/'PROFILE.json',staged)
        cases.append(dict(directory=str(dest),origin='SAVED_PUBLIC_REQUEST_STAGED_LOCAL_ASSIGNMENT',
                     source=REAL+':'+DOCS+name,n_signals=14,phase_radii=14,terminal_XY_price=None))
    for (method,M), path in old.items():
        dest=run/f'{method}-M{M}'
        dest.mkdir()
        shutil.copyfile(path,dest/'ASSIGNMENT.json')
        shutil.copyfile(path.with_name('WORKER_PROFILE.json'),dest/'PROFILE.json')
        cases.append(dict(directory=str(dest),origin='PRESERVED_CENTRAL_ASSIGNMENT',source=str(path)))
    put(run/'CASES.json',cases)
    return cases


def check(case):
    directory=Path(case['directory'])
    command=[PYTHON,'-B',str(FLEET/'scripts/check_rpe_q3_host_readiness.py'),
             '--assignment',str(directory/'ASSIGNMENT.json'),'--profile',str(directory/'PROFILE.json'),
             '--output',str(directory/'dry')]
    with (directory/'RUN.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return directory.name, result.returncode


if __name__=='__main__':
    run=Path(sys.argv[1]).resolve()
    cases=prepare(run)
    with ThreadPoolExecutor() as pool:
        results=dict(pool.map(check,cases))
    put(run/'EXIT_CODES.json',results)
    print(results,flush=True)
