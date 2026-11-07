"""One metadata-only attempt. Existing signed kernel is imported, never changed.

No state/engine, RNG, shots, fit, circuit or price. Tests inject invented files.
"""
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from . import scv_signed_support as signed
from . import scv_scalar as scalar
from . import a1_control_variates as cv
from . import a1_n3_connection as n3
from . import a1_connections

DOC=Path('docs/research/scv_signed_support_h4_metadata_science_20261107')
SETTINGS=DOC/'EXECUTION_SETTINGS.json'
PROFILE=DOC/'PROFILE_LOCK.json'
ENTRY=Path('scripts/tracks/algorithm_codesign/run_scv_signed_h4_metadata.py')


def digest(data):return hashlib.sha256(data).hexdigest()


def encode(x):
    if isinstance(x,F):return str(x)
    raise TypeError(type(x).__name__)


def dumps(x):return json.dumps(x,default=encode,sort_keys=True,ensure_ascii=False,allow_nan=False)


def save(path,x):
    """Same atomic/fsync pattern as a1_n3_campaign.save, without engine import."""
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:f.write(dumps(x)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def append(path,x):
    with Path(path).open('a') as f:f.write(dumps(x)+'\n');f.flush();os.fsync(f.fileno())


class Calls:
    """Local durable call ledger; not an authorization or restart service."""
    def __init__(self,out):self.path=Path(out)/'CALLS.jsonl';self.counts={}

    def call(self,name,fn,*args,**kwargs):
        v=self.counts.setdefault(name,dict(started=0,returned=0,exceptions=0))
        if v['started']:raise RuntimeError('one-call limit: '+name)
        v['started']+=1;append(self.path,dict(event='CALL_START',name=name,monotonic=time.monotonic()))
        try:result=fn(*args,**kwargs)
        except BaseException as e:
            v['exceptions']+=1;append(self.path,dict(event='CALL_EXCEPTION',name=name,error=repr(e)));raise
        v['returned']+=1;append(self.path,dict(event='CALL_RETURN',name=name,monotonic=time.monotonic()))
        return result


class Observe:
    """Read-only Python call observation. No monkeypatch of scientific kernel."""
    def __init__(self,law,catalog):
        self.raw=(law,catalog);self.counts={};self.forbidden={}
        self.functions={
            scalar.metadata_from_bytes.__code__:('scalar_binder',1),
            n3.ConditionalSampler.from_record.__func__.__code__:('from_record',1),
            n3.ConditionalSampler.__init__.__code__:('sampler',1),
            cv.CategoricalLaw.__post_init__.__code__:('categorical_law',2),
            cv.OccurrenceLaw.__post_init__.__code__:('occurrence_law',1),
            cv.ControlBank.__post_init__.__code__:('control_bank',3),
            a1_connections.proposed_control_banks.__code__:('proposed_banks',1),
            scalar.Metadata.__post_init__.__code__:('scalar_metadata',1),
            signed.Metadata.__post_init__.__code__:('signed_metadata',1),
            signed.Component.__post_init__.__code__:('component',None),
            signed.SignedControl.__post_init__.__code__:('signed_control_post_init',1),
            n3.ConditionalSampler.cells.__code__:('cells',2),
            cv.ControlBank.pattern_mass.__code__:('pattern_mass',10),
            cv.ControlBank.joint_mean.__code__:('joint_mean',80),
        }

    def observe(self,frame,event,arg):
        if event!='call':return
        code=frame.f_code;module=frame.f_globals.get('__name__','');name=code.co_name
        banned=(module.startswith(('trotterlib','numpy','scipy','qiskit')) or
            (module.startswith('trottertracks.algorithm_codesign.') and
             (module.rsplit('.',1)[-1].startswith(('a1_engine','a1_new_target_n2','scv224','a1_n3_campaign')) or
              name in {'draw','below','word53','pcg64_streams','load_certified_services',
                       'inspect_certified_bundle','selected_full_prices','_price_dependencies',
                       'fit_and_freeze','fit_h1','fit_cells','paired_evaluate','calibrate',
                       'sufficient_shots','plan_production','residual_range','select_fragment_control',
                       'connect_sampler_inputs','law_from_choice_cdf'})))
        if banned:
            key=module+'.'+name;self.forbidden[key]=self.forbidden.get(key,0)+1
            raise RuntimeError('metadata-only forbidden call blocked: '+key)
        if code is json.loads.__code__:
            value=frame.f_locals.get('s')
            key='json_law' if value is self.raw[0] else 'json_catalog' if value is self.raw[1] else 'json_other'
            limit=2 if key!='json_other' else 0
        elif code in self.functions:key,limit=self.functions[code]
        else:return
        self.counts[key]=self.counts.get(key,0)+1
        if limit is not None and self.counts[key]>limit:raise RuntimeError('internal call limit: '+key)

    @contextmanager
    def active(self):
        if sys.getprofile() is not None:raise RuntimeError('unexpected profiling owner')
        sys.setprofile(self.observe)
        try:yield
        finally:sys.setprofile(None)

    def verify(self,components):
        expected={name:limit for name,limit in self.functions.values()}
        expected.update(component=components,json_law=2,json_catalog=2)
        if self.counts!=expected or self.forbidden:raise ValueError('observed internal call mismatch')


def check_law(md,record,settings,cells):
    patterns=tuple(tuple(x) for x in settings['patterns'])
    if (tuple(tuple(x) for x in record['five_patterns'])!=patterns or md.sampler.patterns!=patterns
        or tuple(h for h,_ in cells)!=tuple(map(str,range(5))) or min(p for _,p in cells)<=0
        or sum(p for _,p in cells)!=1 or tuple(p for _,p in cells)!=tuple(F(x['exact']) for x in record['five_strata'])
        or tuple(x['label'] for x in record['label_masses'])!=tuple(l for l,_ in md.base.catalog)):
        raise ValueError('saved five strata / label order mismatch')


def check_moments(control):
    """Enclosure/identity checks only; never recompute the label finite sums."""
    nu,m2,var=control.means,control.second_moments,control.variances
    if (not len(nu)==len(m2)==len(var)==5 or any(not -1<=v<=1 for v in nu)
        or any(not 0<=v<=1 for v in m2) or any(v!=s-m*m or v<0 for m,s,v in zip(nu,m2,var))
        or var[0]!=0 or len(set(nu[2:]))!=1 or len(set(m2[2:]))!=1 or len(set(var[2:]))!=1):
        raise ValueError('inconsistent exact conditional moments')


def classify(variance):
    if variance<0:raise ValueError('negative variance')
    return 'ELIGIBLE' if variance>0 else 'INELIGIBLE_CONSTANT_H1'


def report(result,source):
    text=['# Signed support H4 metadata-only result','',
        'Status: '+result['metadata_structural_eligibility'],
        'SOURCE: '+source.get('source_commit','UNKNOWN'),
        'Scope: '+dumps(source.get('scope',{})),
        'Reason: '+result['reason'],'',
        '| h | exact mass | exact mean | exact second moment | exact variance |',
        '|---|---|---|---|---|']
    for c in result['cells']:text.append('| '+ ' | '.join(str(c[k]) for k in ('cell','mass_exact','mean_exact','second_moment_exact','variance_exact'))+' |')
    text+=['','Metadata identity: '+str(result['metadata_identity']),
        'Control identity: '+str(result['control_identity']),
        'Statistical effectiveness / correlation / residual reduction / RZ benefit: UNKNOWN.',
        'No state, engine, RNG, shots, circuits, compile, fit, train, paired or calibration.',
        'MANDATORY STOP. No retry/resume or next science stage.','']
    return '\n'.join(text)


def execute_once(root,out,settings,source):
    """Called only after production SOURCE preflight, or with invented test files."""
    root=Path(root);out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()  # exclusive, no overwrite/resume
    calls=Calls(out);observer=None;started=time.monotonic();checks={}
    result=dict(schema='scv_signed_support_h4_metadata_result_v1',metadata_identity=None,control_identity=None,
        feature_version=settings['feature_version'],feature_bounds_exact=('-1','1'),cells=[],h1_nonconstant=None,
        metadata_structural_eligibility='UNKNOWN_ERROR',reason='Not completed',validation_checks=checks,
        statistical_effectiveness='UNKNOWN',science_ready=False,next_stage_started=False,
        state_loaded=False,engine_loaded=False,old_measurement_data_used=False)
    error=None;finalizer_errors=[]
    try:
        save(out/'SOURCE.json',source)
        if settings['feature_version']!=signed.FEATURE:raise ValueError('feature version mismatch')
        law=calls.call('read_LAW', (root/settings['inputs']['law']['path']).read_bytes)
        cat=calls.call('read_catalog',(root/settings['inputs']['catalog']['path']).read_bytes)
        for name,data in (('law',law),('catalog',cat)):
            if digest(data)!=settings['inputs'][name]['sha256']:raise ValueError('input byte identity mismatch: '+name)
        checks['input_bytes']='PASS'
        save(out/'INPUT_IDENTITY.json',dict(input_commit=settings['input_commit'],inputs=settings['inputs'],
            identity=settings['identity'],a=settings['a'],b=settings['b'],system_qubits=settings['system_qubits'],
            patterns=settings['patterns'],feature_version=settings['feature_version']))
        observer=Observe(law,cat)
        def compute():
            with observer.active():
                ids=settings['identity']
                md=calls.call('signed_binder',signed.metadata_from_bytes,law,cat,
                    system_qubits=settings['system_qubits'],law_sha256=settings['inputs']['law']['sha256'],
                    catalog_sha256=settings['inputs']['catalog']['sha256'],
                    hamiltonian_hash=ids['hamiltonian_hash'],preparation_hash=ids['preparation_hash'],a=settings['a'],b=settings['b'])
                result['metadata_identity']=md.identity
                record=json.loads(law)
                cells=calls.call('external_cells',md.sampler.cells,'B1-OL')
                check_law(md,record,settings,cells);checks['saved_strata_label_order']='PASS'
                control=calls.call('signed_control',signed.SignedControl,md)
                result.update(control_identity=control.identity,snapshot=control.snapshot())
                result['cells']=[dict(cell=h,pattern=settings['patterns'][i],mass_exact=p,
                    mean_exact=control.means[i],second_moment_exact=control.second_moments[i],variance_exact=control.variances[i])
                    for i,(h,p) in enumerate(cells)]
                calls.call('moment_snapshot',lambda:None)  # one acquired set; no new arithmetic
                check_moments(control);checks['moments']='PASS'
                decision=calls.call('h1_classification',classify,control.variances[1])
                observer.verify(len(md.components));checks['call_scope']='PASS'
                result.update(metadata_structural_eligibility=decision,h1_nonconstant=(decision=='ELIGIBLE'),
                    reason='All metadata checks passed; exact Sigma1 '+('> 0' if decision=='ELIGIBLE' else '= 0'))
        calls.call('metadata_attempt',compute)
        checks['completion']='PASS'
        # Save result now; finalization failures must not silently become success.
        save(out/'METADATA_RESULT.json',result)
    except BaseException as e:
        error=repr(e);result.update(metadata_structural_eligibility='UNKNOWN_ERROR',reason=error,h1_nonconstant=None)
        checks['completion']='FAIL'
    finally:
        if observer is not None and observer.forbidden:checks['forbidden_calls']='FAIL'
        counts=dict(observed_top_level=calls.counts,
            observed_internal_entries={} if observer is None else observer.counts,
            forbidden_calls_attempted={} if observer is None else observer.forbidden,
            static_expected_normal_path=dict(LAW_JSON_decodes=2,catalog_JSON_decodes=2,cells=2,pattern_mass=10,joint_mean=80),
            forbidden_scope=dict(state_solve_eigh=0,engine_Born=0,sampling_shots_RNG=0,build_compile_price=0,OLS_train_paired_calibration=0),
            note='Internal entries are observed via sys.setprofile; static expectations are separately labelled. Forbidden calls are blocked before their body.',
            error=error,elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,retry=0)
        for name,obj in (('CALL_COUNTS.json',counts),('METADATA_RESULT.json',result)):
            if finalizer_errors:result.update(metadata_structural_eligibility='UNKNOWN_ERROR',reason='Finalization incomplete: '+repr(finalizer_errors),h1_nonconstant=None)
            try:save(out/name,obj)
            except BaseException as e:finalizer_errors.append(dict(file=name,error=repr(e)))
        try:
            p=out/'REPORT.md'
            with p.open('w') as f:f.write(report(result,source));f.flush();os.fsync(f.fileno())
        except BaseException as e:finalizer_errors.append(dict(file='REPORT.md',error=repr(e)))
        if finalizer_errors:result.update(metadata_structural_eligibility='UNKNOWN_ERROR',reason='Finalization incomplete: '+repr(finalizer_errors),h1_nonconstant=None)
        save(out/'STOP.json',dict(status='MANDATORY_STOP',result_status=result['metadata_structural_eligibility'],
            error=error,finalizer_errors=finalizer_errors,no_retry=True,next_stage_started=False,
            science_ready=False,attempt_started=calls.counts.get('metadata_attempt',{}).get('started',0)))
    return result


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


def environment():
    exe=Path(sys.executable).resolve()
    return dict(python=sys.version,executable=str(exe),executable_sha256=digest(exe.read_bytes()),
        mpmath=importlib.metadata.version('mpmath'),platform=platform.platform(),
        bytecode_disabled=sys.dont_write_bytecode,pythonpath=os.environ.get('PYTHONPATH'),
        cache={k:os.environ.get(k) for k in ('MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME')})


def preflight(root,source_commit):
    root=Path(root);lock=json.loads((root/PROFILE).read_bytes());settings=json.loads((root/SETTINGS).read_bytes())
    if git(root,'rev-parse','HEAD')!=source_commit or git(root,'status','--porcelain=v1'):
        raise ValueError('SOURCE commit / clean worktree mismatch')
    if environment()!=lock['environment']:raise ValueError('environment identity mismatch')
    for name,h in lock['source_sha256'].items():
        if digest((root/name).read_bytes())!=h:raise ValueError('SOURCE hash mismatch: '+name)
    for name in lock['kernel_paths']:
        blob=subprocess.check_output(['git','show',settings['kernel_commit']+':'+name],cwd=root)
        if digest(blob)!=lock['source_sha256'][name]:raise ValueError('fixed kernel mismatch: '+name)
    for item in settings['inputs'].values():
        if git(root,'rev-parse',settings['input_commit']+':'+item['path'])!=item['git_blob']:
            raise ValueError('N1 Git blob identity mismatch')
    out=root/settings['output']
    if out.exists():raise FileExistsError('output collision / retry forbidden')
    return settings,dict(source_commit=source_commit,contract_commit=settings['contract_commit'],
        kernel_commit=settings['kernel_commit'],environment=lock['environment'],source_sha256=lock['source_sha256'],
        profile_sha256=digest((root/PROFILE).read_bytes()),settings_sha256=digest((root/SETTINGS).read_bytes()),
        scope=settings['scope'],references=settings['references'],attempt_id=settings['attempt_id'],output=settings['output'],
        input_commit=settings['input_commit'],feature_version=settings['feature_version'],host=platform.node(),uid=os.getuid(),
        resource=dict(loadavg=Path('/proc/loadavg').read_text().strip(),
            memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))],
            own_cgroup=Path('/proc/self/cgroup').read_text()),
        preflight='SOURCE/environment/kernel/Git input identity PASS; direct bytes checked once before metadata_attempt',
        no_state_engine_or_old_measurements=True)


def run_fixed(root,source_commit):
    settings,source=preflight(root,source_commit)
    old=signal.getsignal(signal.SIGTERM)
    def stop(signum,frame):raise InterruptedError('SIGTERM: preserve and STOP without retry')
    signal.signal(signal.SIGTERM,stop)
    try:return execute_once(root,Path(root)/settings['output'],settings,source)
    finally:signal.signal(signal.SIGTERM,old)
