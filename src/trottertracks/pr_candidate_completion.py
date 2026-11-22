"""Complete B2-selected PR inputs; never choose LD, kappa, width or shot rules.

Scientific leaves are the published method-specific providers/acquisition paths.
The public API accepts the optimizer's input request/canonical configuration.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from fractions import Fraction
import hashlib
import ast
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time

from . import pr_candidate_selection as Selection
from .pr_step_price_registry import PRICE_SOURCE_CONFLICT, PRICE_EXTRAPOLATION_HOLD, QUOTE_POLICY_REVISION

ROOT=Path(globals().get('FIXED_RUN_CODE_ROOT', Path(__file__).resolve().parents[2]))
NATIVE='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
PRICE='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'
D2_VIEW=ROOT.parent/'b2-route-view-d2-c1f95e080'
B_VIEW=ROOT.parent/'d2-auto-b-intake-33e324b'
DF_Q3_VIEW=ROOT.parent/'b2-route-view-d2-23160cc87'
DF_VIEW=ROOT.parent/'d2-pr-df-ld1-m3-acquisition-20261115'
SIGNAL_VIEW=ROOT.parent/'d2-pr-df-ld7-signal-source-fefd-20261116'
A_WORKSPACE=ROOT.parent/'d2-pauli-auto-nodrop-evidence-20261116'
GIT_LOCK=threading.RLock()
IDENTITY=['-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp']
_DF_PRICE_BOOTSTRAP="""
import importlib.util, json, sys
from pathlib import Path
runner, view = sys.argv[1:3]
sys.argv = [runner, *sys.argv[3:]]
sys.path.insert(0, str(Path(runner).resolve().parents[1]/'src'))
import trottertracks
trottertracks.__path__.insert(0, str(Path(view)/'src'/'trottertracks'))
if '--input' in sys.argv:
    from trottertracks.rpe_preparation_paths import resolve
    document = Path(sys.argv[sys.argv.index('--input')+1])
    fixed_inputs = json.loads(document.read_bytes())
    provider = resolve(fixed_inputs['runtime']['provider_root'])
    native = Path(provider)/'src'/'trottertracks'
    if native.is_dir():
        import trottertracks.resource_applicability as resource_applicability
        resource_applicability.__path__.insert(0, str(native/'resource_applicability'))
# Pool jobs resolve functions by module name even with the fork start method.
spec = importlib.util.spec_from_file_location('_rpe_fixed_df_price_runner', runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.FIXED_VIEW = Path(view)
if '--input' in sys.argv and ('freeze' in sys.argv[1:] or 'compare-holdout' in sys.argv[1:]):
    # Retain the original paths for Git receipt reads. Return new evidence
    # under the owned input's directory, including on continuation workers.
    original = (module.ROOT / fixed_inputs['output_directory']).resolve()
    target = document.resolve().parent
    write = module.write
    audit_outputs = {'FROZEN_MODELS.json', 'AXIS_PRICES.json', 'HOLDOUT_COMPARISON.json'}
    def write_audit(path, value):
        path = Path(path)
        if path.parent.resolve() == original and path.name in audit_outputs:
            path = target / path.name
        write(path, value)
    module.write = write_audit
raise SystemExit(module.main())
"""

DF_PRICE_FLANKS = Path(__file__).resolve().with_name('rpe_df_price_flanks.py')


def accelerate_df_price_command(command, stage, helper):
    """Wrap newly owned native acquisition commands, leaving audits and saved owners intact."""
    if stage not in ('DF_CALIBRATION', 'DF_HOLDOUT') or _DF_PRICE_BOOTSTRAP not in command:
        return command
    position = command.index(_DF_PRICE_BOOTSTRAP)
    command = list(command)
    command[position] = _DF_PRICE_BOOTSTRAP.removesuffix('raise SystemExit(module.main())\n') + """
import runpy
helper = sys.argv.pop()
with runpy.run_path(helper)['reuse_df_price_flanks'](module):
    raise SystemExit(module.main())
"""
    command.append(str(helper))
    return command


def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def read(p):return json.loads(Path(p).read_text())
def value(loc):
    x=json.loads(subprocess.check_output(['git','show',loc['commit']+':'+loc['path']],cwd=ROOT))
    for k in loc.get('selector',[]):x=x[k]
    return x

def intake_locators(record):
    """Keep the locators covered by the binding receipt after an input relocation."""
    intake = record['intake']
    signal = record['signal']
    configuration = intake['configuration']
    packet = intake['signal_packet']
    for supplied, checked in (
            (signal.get('configuration_source') or record['configuration'], configuration),
            (signal['packet_source'], packet)):
        if supplied == checked:
            continue
        expected = json.dumps(value(supplied), sort_keys=True, allow_nan=False)
        observed = json.dumps(value(checked), sort_keys=True, allow_nan=False)
        if observed != expected:
            raise ValueError('Prepared input differs from its canonical intake receipt')
    return configuration, packet

def dec(v):return Fraction(int(v['numerator']),int(v['denominator']))if isinstance(v,dict)else Fraction(v)
def enc(v):v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists()and p.read_text()==s:return
    tmp=p.with_name(p.name+'.tmp-'+str(os.getpid())+'-'+str(threading.get_ident()));tmp.write_text(s);tmp.replace(p)
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:20]
def rel(p):return str(Path(p).resolve().relative_to(ROOT))


def publish(paths,message):
    with GIT_LOCK:
        selected=[rel(p)for p in paths if Path(p).exists()]
        if selected:
            git('add','--sparse','--',*selected)
            if git('diff','--cached','--name-only','--',*selected):
                git(*IDENTITY,'commit','--only','-q','-m',message+' [skip ci]','--',*selected)
        return git('rev-parse','HEAD')


def physical_evaluation_contract(config):
    """Exact schedule/family contract, independent of declared width and kappa.

    Requested parameters act through the mapper's physical rounds. Administrative
    configuration IDs are removed; H/state, dictionary and law SOURCE, target,
    error/decoder/profile contracts and actual steps remain part of equality.
    """
    spec=deepcopy(config['spec'])
    for field in ('kappa','delta_pf','delta_over_h'):spec.pop(field,None)
    # DF keeps the original mapper input as provenance as well as the resolved
    # spec. Its declared width/kappa are not extra physical conditions.
    if isinstance(spec.get('A_input'),dict):
        spec['A_input'].pop('kappa_paper',None)
        design=spec['A_input'].get('design',{})
        for field in ('delta','delta_pf','delta_over_h'):design.pop(field,None)
    rows=[]
    for row in sorted(config['rounds'],key=lambda r:r['m']):
        physical=deepcopy(row.get('physical_candidate',{}));physical.pop('id',None)
        candidate=deepcopy(row.get('candidate',{}));candidate.pop('id',None)
        rows.append(dict(m=row['m'],T=row['T'],q=row['q_PF'],R=row['R_outer'],
            actual_delta_pf=row['actual_delta_pf'],structure=row['structure'],
            law_binding=row.get('law_binding'),physical_candidate=physical,candidate=candidate))
    return dict(spec=spec,rounds=rows)


class Held(Exception):pass

class PreparationWait(Held):
    """A stage can be checked again through its original request and owner."""

class DFOriginalPriceAttempt(Held):
    def __init__(self, marker):
        self.marker = marker
        super().__init__(DF_PRICE_UNAVAILABLE+' ORIGINAL_PRICE_ATTEMPT_REQUIRES_RECEIPT:'+str(marker))


class PreparationYield(Exception):
    """No new stage started; an orchestration failure requested a boundary yield."""




PRICE_HOLD_FIELDS = {
    PRICE_SOURCE_CONFLICT: 'conflicting_price_axes',
    PRICE_EXTRAPOLATION_HOLD: 'extrapolated_price_axes',
}
DF_PRICE_UNAVAILABLE = 'DF_PRICE_ACTION_UNAVAILABLE:'
DF_PRICE_RECHECK_STATES = frozenset(('PRICE_ACTION_UNAVAILABLE', 'PRICE_NOT_EVALUATED'))


def price_source_holds(quote):
    holds = {}
    for row in quote.get('rows', []):
        if row.get('unit_price_RZ') is not None:
            continue
        needs = row.get('needs') or {}
        for reason in needs.get('reasons', []):
            if reason in PRICE_HOLD_FIELDS:
                item = dict(m=row['m'], axis=row['axis'], entry_ids=needs.get('entry_ids', []))
                if 'extrapolation' in needs:
                    item['extrapolation'] = needs['extrapolation']
                holds.setdefault(reason, []).append(item)
    return holds


def price_hold_reason(holds):
    return 'PRICE_EVIDENCE_REQUIRED:' + ','.join(sorted(holds))


def record_price_holds(record, holds):
    for reason, field in PRICE_HOLD_FIELDS.items():
        if reason in holds:
            record[field] = holds[reason]
        else:
            record.pop(field, None)


class Completion:
    def path(self,name):
        value=getattr(self,"settings",{}).get("controller_paths",{}).get(name,globals()[name])
        return str(value) if name in ("NATIVE","PRICE") else Path(value)

    def __init__(self,directory,settings):
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.settings=settings
        self.state_path=self.directory/'STATE.json'
        prior=read(self.state_path)if self.state_path.exists()else None
        self.source=settings.get('source')or (prior or{}).get('source')or git('rev-parse','HEAD')
        self.state=prior or dict(schema='PR_CANDIDATE_COMPLETION_V1',source=self.source,records={})
        imported=settings.get('resume_preparation_from')
        if not prior and imported:
            saved=value(imported)
            # Only terminal READY evidence crosses a technical source epoch.
            # Failed stages and their original STOPs remain in the old state.
            self.state['records']={k:dict(r,reused_from=imported) for k,r in saved['records'].items()
                                   if r['status']=='READY_FOR_SEARCH'}
        if not prior:
            for locator in settings.get('protected_stops_from',()):
                for k,record in value(locator)['records'].items():
                    if record.get('scientific_failure'):
                        self.state['records'].setdefault(k,dict(record,reused_from=locator))
            for repair in settings.get('metadata_repair_from', ()):
                path=Path(repair['state_file']).resolve()
                if hashlib.sha256(path.read_bytes()).hexdigest()!=repair['sha256']:
                    raise ValueError('Metadata repair state changed')
                failed=read(path)
                for identifier in repair['record_keys']:
                    record=failed['records'][identifier]
                    directory=path.parent/'candidates'/identifier
                    if not self.metadata_path_failure(record,directory):
                        raise ValueError('Metadata repair would retry a scientific or ambiguous failure: '+identifier)
                    carried=self.state['records'].get(identifier)
                    if carried is not None and {k:v for k,v in carried.items() if k!='reused_from'}!={k:v for k,v in record.items() if k!='reused_from'}:
                        raise ValueError('Metadata repair does not match the preserved failure: '+identifier)
                    self.state.setdefault('metadata_repairs',{})[identifier]=dict(
                        original_record=deepcopy(record),state_file=str(path),sha256=repair['sha256'],
                        reason='NONSCIENTIFIC_DF_PLAN_PATH_REPAIR')
                    self.state['records'].pop(identifier,None)
            from .rpe_preparation_repair import retry_records
            self.state['records'], repairs = retry_records(
                self.state['records'], settings.get('preparation_retry_from', ()))
            if repairs:
                self.state['preparation_repairs'] = repairs
        if self.state['source']!=self.source:raise ValueError('Completion SOURCE changed; preserve prior state')
        self.lock=threading.RLock()
        self.pauli_price_lock=threading.RLock()
        self.pauli_metadata_lock=threading.RLock()
        self.stage_executor=None
        self.shared_preparation_locks={}
        self.registry=self.directory/'PRICE_RULES_REGISTRY.json'
        if not self.registry.exists():
            provided=settings.get('price_registry_source')or settings.get('price_registry_file')
            if provided:
                initial=value(provided)if isinstance(provided,dict)else read(Path(provided))
                extra=dict(entries=[])
            else:
                initial=value(dict(commit='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa',path='configs/pr_price_rules/h4_saved_rules.json'))
                extra=value(dict(commit='54b4353b9',path='docs/research/b2_pauli_calibration_20261116/PRICE_RULES_REGISTRY_EXTENDED.json'))
            by={e['name']:e for e in initial['entries']};by.update({e['name']:e for e in extra['entries']})
            if imported:
                previous=Path(imported['path']).with_name('PRICE_RULES_REGISTRY.json')
                saved_registry=value(dict(commit=imported['commit'],path=str(previous)))
                if provided:
                    for entry in saved_registry['entries']:by.setdefault(entry['name'],entry)
                else:by.update({e['name']:e for e in saved_registry['entries']})
            put(self.registry,dict(initial,entries=list(by.values())))
        self.step_registry_path=self.directory/'STEP_PRICE_RULES.json'
        if settings.get('price_evidence_results'):
            self.ingest_price_results()

    def saved_signal_packets(self,method):
        name='saved_pauli_packets'if method=='PR_PAULI'else'saved_df_bundles'
        packets=list(getattr(self,'settings',{}).get(name,[]))
        for record in list(getattr(self,'state',{}).get('records',{}).values()):
            loc=(record.get('signal')or{}).get('packet_source')
            if record.get('method')==method and loc and loc not in packets:packets.append(loc)
        return packets

    @staticmethod
    def metadata_path_failure(record,directory):
        """Only a recorded DF metadata lookup failure may start fresh metadata."""
        if (record.get('method')=='PR_DF' and record.get('status')=='HELD_TECHNICAL_FAILURE'
                and str(record.get('reason','')).startswith('CalledProcessError: Command ')):
            return Completion.metadata_git_input_failure(record,directory)
        if (record.get('method')!='PR_DF' or record.get('status')!='HELD_WITH_REASON'
                or not str(record.get('reason','')).startswith('STAGE_FAILED:df-plan:')):
            return False
        try:
            if not (directory/'df-plan.log').is_file():
                # An inherited hold retains its original absolute log path.
                # Resolve it only from a normally stopped owner; never inspect
                # past a new local scientific attempt.
                if any(directory.rglob('*.STARTED.json')):return False
                original=Path(record['reason'].split(':',4)[-1])
                if (not original.is_absolute() or original.name!='df-plan.log'
                        or original.parent.name!=directory.name or original.parent.parent.name!='candidates'):return False
                parent=original.parents[3]
                launch=read(parent/'LAUNCH.json');ended=read(parent/'EXIT.json')
                if (read(parent/'STOP_REQUESTED.json').get('reason')!='USER_REQUESTED_STOP'
                        or launch.get('invocation_id')!=ended.get('invocation_id')
                        or ended.get('returncode')not in(0,-2)
                        or read(parent/'distributed/relay/STATUS.json').get('stopped')is not True):return False
                directory=original.parent
            markers=list(directory.rglob('*.STARTED.json'))
            descriptor=directory/'DF_TAIL_DESCRIPTOR.STARTED.json'
            if descriptor in markers:
                # A completed descriptor is immutable evidence, not a failed
                # science leaf to retry. Only the later metadata lookup is
                # being repaired; the descriptor must remain reusable.
                from .pr_df_descriptor_extension import validate
                started=read(descriptor);document=read(directory/'DF_TAIL_DESCRIPTOR.json')
                returned=read(directory/'DF_DESCRIPTOR.RETURN.json')
                if ((directory/'DF_TAIL_DESCRIPTOR.STOP.json').exists()
                        or returned.get('returncode')!=0 or returned.get('signal')is not None
                        or returned.get('automatic_retry')is not False or not returned.get('job_id')
                        or started.get('source')!=document.get('execution_source')
                        or started.get('configuration')!=document.get('configuration_source')
                        or started.get('prefix')!=document.get('prefix')
                        or returned.get('source')!=document['execution_source']['commit']):return False
                validate(document,{},read(directory/'CONFIGURATION.json'))
                markers.remove(descriptor)
            if not markers or any(p.parent!=directory or read(p).get('science') is not False for p in markers):return False
            for marker in markers:
                stage=marker.name.removesuffix('.STARTED.json')
                if stage not in ('df-generate','DF_GENERATE_WITH_DESCRIPTOR','INITIAL_QUOTE','df-plan'):return False
                receipt=read(directory/(stage+'.RETURN.json'))
                if receipt.get('signal') is not None or receipt.get('returncode')!=(1 if stage=='df-plan'else 0):return False
            if not {'df-generate.STARTED.json','INITIAL_QUOTE.STARTED.json','df-plan.STARTED.json'} <= {p.name for p in markers}:
                return False
            if (directory/'df-plan.json').exists():return False
            log=(directory/'df-plan.log').read_text()
            return ("FileNotFoundError: [Errno 2] No such file or directory:" in log
                    or ("KeyError: 'means'" in log and 'validated_packet_index' in log))
        except (OSError,ValueError,KeyError,TypeError):
            return False

    @staticmethod
    def metadata_git_input_failure(record,directory):
        """Re-resolve a missing committed physical twin before any science leaf."""
        expected={'df-generate.STARTED.json','df-generate.RETURN.json','df-generate.json',
                  'df-generate.INPUT.json','df-generate.log','CONFIGURATION.json'}
        if {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}!=expected:
            return False
        try:
            reason=record['reason'];prefix='CalledProcessError: Command ';suffix=' returned non-zero exit status 128.'
            if not reason.endswith(suffix):return False
            literal=reason[len(prefix):-len(suffix)]
            if not literal.startswith("'[") or not literal.endswith("]'"):return False
            command=ast.literal_eval(literal[1:-1])
            if (not isinstance(command,list) or len(command)!=3 or command[:2]!=['git','show']
                    or not isinstance(command[2],str) or ':' not in command[2]):return False
            commit,path=command[2].split(':',1)
            if (len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)
                    or Path(path).is_absolute() or '..' in Path(path).parts
                    or Path(path).name!='CONFIGURATION.json'):return False
            started=read(directory/'df-generate.STARTED.json');receipt=read(directory/'df-generate.RETURN.json')
            generated=read(directory/'df-generate.json');config=read(directory/'CONFIGURATION.json')
            if (started.get('science') is not False or receipt.get('returncode')!=0 or receipt.get('signal') is not None
                    or generated.get('status')!='CONFIGURATION_READY' or generated.get('configuration')!=config):return False
            design=record['request']['design'];spec=config['spec']
            if (spec['L_D']!=design['L_D'] or spec['formula']!=design['formula']
                    or dec(spec['kappa']['value'])!=dec(design['kappa'])
                    or dec(spec['delta_pf'])!=dec(design['delta_pf'])):return False
            recovered=json.loads(git('show',command[2]))
            state=read(directory.parent.parent/'STATE.json')
            matched=[r for r in state['records'].values() if r.get('method')=='PR_DF'
                     and r.get('configuration')==dict(commit=commit,path=path)]
            if not matched:return False
            for previous in matched:
                wanted=previous['request']['design'];actual=recovered['spec']
                if (previous.get('configuration_id')!=recovered.get('id')
                        or actual['L_D']!=wanted['L_D'] or actual['formula']!=wanted['formula']
                        or dec(actual['kappa']['value'])!=dec(wanted['kappa'])
                        or dec(actual['delta_pf'])!=dec(wanted['delta_pf'])):return False
            return True
        except (OSError,ValueError,KeyError,TypeError,SyntaxError,subprocess.CalledProcessError):
            return False

    def _step_rules(self):
        """The partial-step registry is independent of the legacy 26-axis rule."""
        from .pr_price_rules import GitReader
        from .pr_step_price_registry import StepPriceRegistry
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        return StepPriceRegistry(GitReader(ROOT),read(self.registry),state=read(path)if path.exists()else None)

    def ingest_price_results(self,packages=None):
        """Accept fixed audited results once; never launch a calibration here."""
        observe_prior = packages is None
        packages=(getattr(self,'settings',{}).get('price_evidence_results',[])if observe_prior else packages)
        roots=self.df_price_owner_roots() if observe_prior else []
        if not packages and not roots:return []
        with self.lock:
            pending = {action['group_id'] for row in self.state['records'].values()
                       if row.get('method')=='PR_DF' and row['status'] in DF_PRICE_RECHECK_STATES
                       for action in row.get('price_actions', ()) if action.get('group_id')} if roots else set()
            if not packages and not pending:return []
            rules=self._step_rules();answers=[rules.ingest(item)for item in packages]
            if pending:
                known={key(e['provenance']) for e in rules.state['entries']}
                for root in roots:
                    for gid in pending:
                        owner=root/'df-price-owners'/(gid+'.json')
                        try:
                            saved=read(Path(read(owner)['directory'])/'PRICE_EVIDENCE.json')
                            fields=('inputs_source','model_source','price_source','holdout_source')
                            token=key({k:saved.get(k) for k in fields})
                            if token not in known:
                                answers.append(rules.ingest(saved))
                                known.add(token)
                        except FileNotFoundError:
                            continue
                        except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as error:
                            self.log(event='DF_PRIOR_PRICE_EVIDENCE_UNAVAILABLE',group=gid,reason=str(error))
            if answers:put(getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json'),rules.state)
        return answers

    def common_pauli_model_terms(self, configurations):
        from .pauli_calibration_request import implementation
        step=self._step_rules()
        rules=step.api['PriceRules'](step.reader,step.registry)
        out={}
        for key,config in configurations.items():
            try:
                terms=step.common_pauli_model_terms(config,implementation(),rules=rules)
            except (KeyError,ValueError,TypeError):
                terms=None
            if terms is not None:
                out[key]=terms
        return out

    def price_registry_digest(self):
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        return key(dict(legacy=read(self.registry),steps=read(path)if path.exists()else None,
                        quote_policy_revision=QUOTE_POLICY_REVISION))

    def price_deferred_keys(self):
        with self.lock:
            distributed = getattr(self, 'stage_executor', None) is not None
            waiting = [record for record in self.state['records'].values()
                       if (record.get('signal_deferred_for_price') and record['status']=='WAITING_INPUTS'
                           and not (distributed and record['method']=='PR_PAULI' and not record.get('signal')))
                       or record['status'] in DF_PRICE_RECHECK_STATES]
            if not waiting:
                return set()
            digest = self.price_registry_digest()
            return {record['key'] for record in waiting if self.price_evidence_unchanged(record, digest)}

    def deferred_stage_keys(self):
        """Check whether saved exact UNKNOWN owners are still unresolved."""
        runtime = getattr(self, 'stage_executor', None)
        store = getattr(getattr(runtime, 'executor', None), 'store', None)
        if store is None:
            return set()
        with self.lock:
            waiting = [(record['key'], record['preparation_recovery'])
                       for record in self.state['records'].values()
                       if record.get('signal') and record.get('intake')
                       and record['status'] == 'WAITING_DISTRIBUTED_PREPARATION'
                       and (record.get('preparation_recovery') or {}).get('state') == 'UNKNOWN']
        if not waiting:
            return set()
        with store._connection() as db:
            return {key for key, evidence in waiting if evidence.get('attempt_id') and
                    db.execute("SELECT 1 FROM jobs WHERE job_id=? AND state='UNKNOWN' AND attempt_id=?",
                               (evidence['job_id'], evidence['attempt_id'])).fetchone()}

    def price_evidence_unchanged(self, record, registry_digest):
        if record.get('registry_at_hold') != registry_digest:
            return False
        if record.get('method') != 'PR_DF':
            return True
        return record.get('df_price_evidence_at_decision') == self.df_price_evidence_snapshot(record)

    def quote_with_steps(self,config,quote):
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        if not path.exists()and not(quote.get('rows')and config.get('spec',{}).get('method')=='PR'):return quote
        try:return self._step_rules().quote(config,quote['implementation'],base_quote=quote)
        except Exception:
            if path.exists():raise  # preserve actual-step application guards
            return quote  # optional metadata cannot invalidate an existing quote

    def save(self):
        with self.lock:put(self.state_path,self.state)

    def log(self,**event):
        with self.lock:
            with(self.directory/'PROGRESS.jsonl').open('a')as f:f.write(json.dumps(dict(epoch=time.time(),**event),default=str)+'\n')

    def bind_stage_executor(self,runtime):
        """Use the same durable host queue as Q3; metadata remains central."""
        self.stage_executor=runtime
        return self

    def parallel_pauli_preparation(self):
        runtime=getattr(self,'stage_executor',None)
        handles=getattr(runtime,'handles',None)
        return handles is not None and all(handles(stage) for stage in
                                          ('PAULI_SIGNALS','PAULI_COMPILE','PAULI_AUDIT'))

    def shared_preparation_lock(self,identity):
        """One producer for an exact physical contract or calibration group."""
        token=key(dict(source=self.source,identity=identity))
        with self.lock:return self.shared_preparation_locks.setdefault(token,threading.RLock())

    def stage_source(self, stage):
        if stage == 'DF_DESCRIPTOR':
            return self.settings.get('df_descriptor_source', self.source)
        if stage == 'PAULI_SIGNALS':
            return self.settings.get('pauli_signal_source', self.source)
        if stage in ('PAULI_COMPILE', 'PAULI_AUDIT'):
            return self.settings.get('pauli_price_wrapper_source', self.source)
        if stage == 'DF_SIGNALS':
            return self.settings.get('df_signal_source', self.source)
        if stage in ('DF_PRICE_PREPARE', 'DF_CALIBRATION', 'DF_MODEL_FREEZE',
                     'DF_HOLDOUT', 'DF_HOLDOUT_COMPARE'):
            return self.settings.get('df_price_source', self.source)
        return self.source

    def call(self,cmd,d,stage,output=None,science=False,*,poll=False):
        d=Path(d);d.mkdir(parents=True,exist_ok=True);done=d/(stage+'.RETURN.json');started=d/(stage+'.STARTED.json')
        if done.exists():
            r=read(done)
            if r['returncode']!=0:raise Held('PRIOR_STAGE_FAILED_NO_RETRY:'+stage)
            return read(output)if output and Path(output).exists()else r
        runtime=getattr(self,'stage_executor',None)
        if runtime is not None and science and runtime.handles(stage):
            # Recovery is reconciled with the original host/attempt by the
            # shared executor. A local STARTED marker is never a retry grant.
            if poll:
                return runtime.call(self,cmd,d,stage,output,science=science,wait=False)
            return runtime.call(self,cmd,d,stage,output,science=science)
        if poll and runtime is not None:
            raise PreparationWait('NO_DISTRIBUTED_PREPARATION_PROVIDER:'+stage)
        if started.exists():raise Held('IN_PROGRESS_OR_AMBIGUOUS_NO_DUPLICATE:'+stage)
        if getattr(self,'stop_requested',lambda:False)():
            self.log(event='STAGE_YIELDED_AFTER_ORCHESTRATOR_FAILURE',stage=stage,directory=rel(d))
            raise PreparationYield(stage)
        source=self.stage_source(stage)
        cmd = accelerate_df_price_command(cmd, stage, DF_PRICE_FLANKS)
        put(started,dict(command=cmd,source=source,science=science,automatic_retry=False))
        self.log(event='STAGE_START',stage=stage,science=science,directory=rel(d));t=time.time()
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        mapping=json.loads(environment.get('RPE_PREPARATION_PATH_MAP','{}'))
        mapping.update(self.settings.get('pauli_saved_inputs') or {})
        mapping.update(self.settings.get('preparation_path_map') or {})
        if mapping:environment['RPE_PREPARATION_PATH_MAP']=json.dumps(mapping)
        with(d/(stage+'.log')).open('w')as f:
            p=subprocess.Popen(cmd,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
            put(started,dict(command=cmd,source=source,pid=p.pid,science=science,automatic_retry=False))
            code=p.wait()
        put(done,dict(returncode=code if code>=0 else None,signal=-code if code<0 else None,seconds=time.time()-t,source=source))
        self.log(event='STAGE_RETURN',stage=stage,returncode=code,seconds=time.time()-t)
        if code!=0:
            detail=read(output)if output and Path(output).exists()else{}
            raise Held('STAGE_FAILED:'+stage+':'+str(detail.get('status'))+':'+str(detail.get('reason'))+':'+str(d/(stage+'.log')))
        return read(output)if output and Path(output).exists()else read(done)

    def service(self,mode,request,d,tag=None,python=None):
        python=python or self.path("NATIVE")
        request=dict(request,controller_paths=self.settings.get("controller_paths",{}))
        request['preparation_path_map'] = dict(self.settings.get('pauli_saved_inputs') or {},
                                              **(self.settings.get('preparation_path_map') or {}))
        if mode in ('df-plan','df-stage','df-bundle'):
            request['saved_index_cache']=str(self.directory/'cache'/'df-saved-index')
        tag=tag or mode;inp=d/(tag+'.INPUT.json');out=d/(tag+'.json');put(inp,request)
        command = [python, '-B', str(ROOT/'scripts/pr_candidate_services.py')]
        metadata_entry = globals().get('FIXED_RUN_PR_PRICE_SERVICE')
        terminal_request = (request.get('terminal_reference') is not None
                            or (mode in ('df-generate','intake') and request.get('M') == 13))
        if metadata_entry and (mode == 'df-price-design' or terminal_request):
            command = [python, '-B', '-c',
                       'import runpy,sys\n'
                       'root,entry=sys.argv[1:3]\n'
                       'sys.argv=sys.argv[2:]\n'
                       'runpy.run_path(entry,run_name="__main__",init_globals={"FIXED_RUN_CODE_ROOT":root})',
                       str(ROOT), metadata_entry]
        return self.call([*command,mode,'--input',str(inp),'--output',str(out)],d,tag,out)

    def saved_df_descriptor(self,request):
        if request.get('tail_descriptor'):return request['tail_descriptor']
        cfg=request.get('configuration')
        if cfg:
            cfg=value(cfg)if 'commit'in cfg else cfg
            source=cfg.get('spec',{}).get('split',{}).get('lambda_source',{})
            if source.get('commit')and source.get('path'):
                # The descriptor validator still checks its H/state/prefix.
                return source
        prefix=(request.get('design')or{}).get('L_D')
        supplied=getattr(self,'settings',{}).get('saved_df_descriptors',{}).get(str(prefix))
        if supplied:
            return supplied
        for record in list(self.state['records'].values()):
            source=record.get('request',{}).get('tail_descriptor')
            if record['method']=='PR_DF'and source and (record.get('design')or record.get('request',{}).get('design',{})).get('L_D')==prefix:
                return source
        for record in self.state.get('family_metadata',{}).values():
            if record.get('method')=='PR_DF'and record.get('L_D')==prefix and record.get('tail_descriptor'):
                return record['tail_descriptor']
        return None

    def verify_continuation_inputs(self, record):
        """Rebind a saved packet and verify its original price evidence."""
        directory = self.directory/'continuation-inputs'/record['key']
        directory.mkdir(parents=True, exist_ok=True)
        config = value(record['configuration'])
        configuration, packet = intake_locators(record)
        method = record['method']
        inputs = dict(method=method, configuration=configuration, packet=packet,
                      view=str(self.path('B_VIEW') if method == 'PR_PAULI' else self.path('DF_Q3_VIEW')))
        if record.get('request', {}).get('horizon_parent'):
            inputs['M'] = config['spec']['M']
        intake = self.service('intake', inputs, directory)
        if intake != record['intake']:
            raise ValueError('CONTINUATION_INTAKE_BINDING_DIFFERS')
        registry = directory/'PRICE_RULES.json'
        put(registry, read(self.registry))
        current = self.service('quote', dict(
            configuration=record['configuration'], view=str(self.path('D2_VIEW')),
            registry_file=str(registry), policy='MODEL'), directory, python=self.path('PRICE'))
        current = self.quote_with_steps(config, current)
        saved = value(record['quote'])
        if (not current.get('complete') or current.get('rows') != saved.get('rows')
                or [row['unit_price_RZ'] for row in current['rows']] != record['prices']):
            raise ValueError('CONTINUATION_PRICE_EVIDENCE_DIFFERS')

    def refresh_price_hold(self,record):
        """Requery after a registry or quote policy update, without scientific retry."""
        reason=str(record.get('reason',''))
        if record.get('scientific_failure')or record.get('status','').startswith('STOP')or any(
                token in reason for token in ('STAGE_FAILED:','PRIOR_STAGE_FAILED','NUMERICAL_GATE','NO_AUTOMATIC_RETRY')):
            return None
        if not all(record.get(k)for k in ('configuration','signal','intake')):return None
        digest=self.price_registry_digest()
        if record.get('registry_at_hold')==digest:return None
        d=self.directory/'refresh'/record['key']/digest
        registry=d/'PRICE_RULES.json';put(registry,read(self.registry))
        quote=self.service('quote',dict(configuration=record['configuration'],view=str(self.path('D2_VIEW')),
            registry_file=str(registry),policy='MODEL'),d,tag='AVAILABLE_EVIDENCE_QUOTE',python=self.path('PRICE'))
        quote=self.quote_with_steps(value(record['configuration']),quote)
        put(d/'AVAILABLE_EVIDENCE_QUOTE.json',quote)
        if record.get('method') == 'PR_DF':
            holds = price_source_holds(quote)
            record_price_holds(record, holds)
            if not quote.get('complete') and reason.startswith(('PRICE_EVIDENCE_REQUIRED:', 'PRICE_RULE_INCOMPLETE:')):
                updated = price_hold_reason(holds) if holds else 'PRICE_RULE_INCOMPLETE:'+str(quote.get('classifications'))
                if updated != reason:
                    record['previous_hold'] = dict(status=record['status'], reason=reason)
                    record['reason'] = updated
        if not quote.get('complete'):
            record['registry_at_hold']=digest;self.save();return None
        fixed=publish([d],'Reuse newly available candidate prices without scientific retry')
        config=value(record['configuration'])
        record.update(previous_hold=dict(status=record['status'],reason=record.get('reason')),
            status='READY_FOR_SEARCH',reason=None,price_policy='MODEL',prices=[r['unit_price_RZ']for r in quote['rows']],
            quote=dict(commit=fixed,path=rel(d/'AVAILABLE_EVIDENCE_QUOTE.json')),
            design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),delta_pf=config['spec']['delta_pf']),
            schedule=[dict(m=r['m'],T=r['T'],q=r['q_PF'],R=r['R_outer'],actual_delta_pf=r['actual_delta_pf'],structure=r['structure'],physical_id=r['candidate']['id'])for r in config['rounds']],
            recovered_by='METADATA_REQUERY_NO_SCIENTIFIC_RETRY')
        record.pop('price_actions', None)
        self.save();return record

    def reconcile_committed_df_holdout(self, record):
        """Reopen a publication mismatch only after the original results pass intake."""
        if (record.get('method') != 'PR_DF'
                or record.get('reason') != 'ValueError: DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS'
                or not all(record.get(key) for key in ('configuration','signal','intake','directory'))):
            return False
        directory = (ROOT/record['directory']).resolve()
        if not directory.is_relative_to(self.directory.resolve()):
            return False
        packages = [read(path) for path in directory.glob('df_price_groups/*/PRICE_EVIDENCE.json')]
        if not packages or any(p['configuration_source'] != record['configuration'] for p in packages):
            return False
        try:
            self.ingest_price_results(packages)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            self.log(event='DF_PUBLISHED_HOLDOUT_STILL_UNRESOLVED', key=record['key'], reason=str(error))
            return False
        self.state.setdefault('metadata_repairs', {})[record['key']+':df-holdout'] = dict(
            action='RECOVER_ORIGINAL_FROZEN_DF_PRICE_EVIDENCE', previous=deepcopy(record))
        record.update(status='WAITING_INPUTS',reason='PRICE_RULE_INCOMPLETE:REQUERY_AFTER_FROZEN_DF_PRICE_RECONCILIATION',
                      scientific_failure=False)
        self.save()
        return True

    def recoverable_pauli_input_hold(self, record, directory):
        """An old empty metadata response can be rechecked with fixed saved inputs."""
        if (record.get('method') != 'PR_PAULI' or record.get('configuration')
                or record.get('reason') != 'IndexError: list index out of range'):
            return False
        paths = getattr(self, 'settings', {}).get('pauli_saved_inputs') or {}
        if not paths or not all(Path(path).is_file() for path in paths.values()):
            return False
        try:
            # Any additional stage evidence keeps the old no-retry decision.
            files = {p.name for p in directory.iterdir()}
            if not files <= {'A_BASE.json', 'A_BASE.REQUEST.json', 'A_BASE.STARTED.json',
                             'A_BASE.RETURN.json', 'A_BASE.log', 'saved-input-recheck'}:
                return False
            result = read(directory/'A_BASE.json')
            started = read(directory/'A_BASE.STARTED.json')
            receipt = read(directory/'A_BASE.RETURN.json')
        except (OSError, ValueError):
            return False
        return (result.get('status') == 'NEEDS_INPUT_SOURCE' and result.get('items') == []
                and result.get('signal_price_Q3_evaluation_performed') is False
                and started.get('science') is False and receipt.get('returncode') == 0
                and receipt.get('signal') is None and started.get('source') == receipt.get('source'))

    def saved_pauli_recheck_directory(self, record, directory):
        """Resolve a copied metadata recheck using its original candidate and receipts."""
        missing_source = 'A_PREPARATION:NEEDS_INPUT_SOURCE:'
        history = (record.get('reason'), (record.get('previous_hold') or {}).get('reason'))
        if (record.get('method') != 'PR_PAULI' or record.get('scientific_failure')
                or record.get('status') == 'READY_FOR_SEARCH'
                or not any(str(reason).startswith(missing_source) for reason in history)):
            return None
        if not self.recoverable_pauli_input_hold(
                dict(method='PR_PAULI', reason='IndexError: list index out of range'), directory):
            return None
        paths = self.settings['pauli_saved_inputs']
        original = read(directory/'A_BASE.json')
        if not original.get('missing_inputs') or any(
                not any(repr(path) in error for path in paths) for error in original['missing_inputs']):
            return None
        checked = directory/'saved-input-recheck'
        try:
            if not checked.is_dir() or checked.resolve() != checked:
                return None
            workspace = Path(self.settings.get('pauli_workspace', self.path('A_WORKSPACE')))
            from .rpe_pauli_saved_inputs import saved_inputs
            saved_inputs(workspace, paths)
            for stage in ('A_BASE', 'A_DESIGN'):
                started = read(checked/(stage+'.STARTED.json'))
                receipt = read(checked/(stage+'.RETURN.json'))
                command = started['command']
                item, = read(checked/(stage+'.json'))['items']
                if (started['science'] is not False or receipt['returncode'] != 0
                        or receipt['signal'] is not None or started['source'] != receipt['source']
                        or receipt['source'] != self.stage_source(stage)
                        or Path(command[2]).name != 'rpe_pauli_saved_inputs.py'
                        or json.loads(command[3]) != paths
                        or Path(command[4]) != workspace/'scripts/prepare_pauli_candidate.py'
                        or item['status'] != 'GENERATED'):
                    return None
            config = item['configuration']
            design = record['request']['design']
            spec = config['spec']
            if (any(spec[k] != design[k] for k in ('L_D', 'formula'))
                    or dec(spec['kappa']['value']) != dec(design['kappa'])
                    or (design.get('delta_pf') is not None
                        and dec(spec['delta_pf']) != dec(design['delta_pf']))
                    or (record.get('configuration') and value(record['configuration']) != config)
                    or config != read(checked/'CONFIGURATION.json')):
                return None
        except (OSError, KeyError, TypeError, ValueError, IndexError, subprocess.CalledProcessError):
            return None
        return checked

    def priced_pauli_recheck_keys(self):
        """Bring previously priced receipt recoveries forward in the preparation lanes."""
        with self.lock:
            records = list(self.state['records'].items())
        ready = set()
        for identifier, record in records:
            checked = self.saved_pauli_recheck_directory(record, self.directory/'candidates'/identifier)
            if checked is None or not (checked/'FINAL_QUOTE.json').is_file():
                continue
            quote = read(checked/'FINAL_QUOTE.json')
            if quote.get('complete') is True and quote.get('configuration_id') == read(checked/'CONFIGURATION.json')['id']:
                ready.add(identifier)
        return ready

    def complete(self,request, *, existing_signal_only=False, saved_packets=()):
        request=deepcopy(request)  # internal descriptor resolution must not change the caller's cache key
        if request['method']not in ('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        if request.get('plan_only')or getattr(self,'settings',{}).get('plan_only'):
            return self.plan(request)
        replacement = getattr(self, 'settings', {}).get('df_descriptor_replacement')
        if replacement and request['method'] == 'PR_DF' and request.get('design') == replacement['design']:
            previous = self.state['records'].get(replacement['prior_key'])
            if (previous is None or previous['method'] != 'PR_DF'
                    or previous['request']['design'] != replacement['design']
                    or not previous.get('scientific_failure') or previous['status'] != 'HELD_WITH_REASON'
                    or not previous['reason'].startswith('STAGE_FAILED:DF_CALIBRATION:')):
                raise ValueError('DF_DESCRIPTOR_REPLACEMENT_REQUIRES_PRESERVED_FAILURE')
            old_split = value(previous['configuration'])['spec']['split']
            descriptor = replacement['descriptor']
            updated = value(descriptor)
            if (updated['prefix'] != replacement['design']['L_D']
                    or descriptor == old_split['lambda_source']
                    or dec(updated['descriptor']['lambda_R']) == dec(old_split['lambda_R'])):
                raise ValueError('DF_DESCRIPTOR_REPLACEMENT_REQUIRES_DISTINCT_FIXED_INPUT')
            request['tail_descriptor'] = descriptor
        identifier=key(request);d=self.directory/'candidates'/identifier
        with self.lock:
            old=self.state['records'].get(identifier)
            if old is None:
                # Historical callers could persist a request after descriptor
                # injection under its original key. Keep that record/STOP and
                # directory; never turn the changed key into a fresh attempt.
                fixed_request=json.dumps(request,sort_keys=True,default=str)
                matches=[(k,r)for k,r in self.state['records'].items()
                    if json.dumps(r.get('request'),sort_keys=True,default=str)==fixed_request]
                if len(matches)>1:
                    return dict(status='HELD_WITH_REASON',method=request['method'],
                        reason='AMBIGUOUS_EXISTING_COMPLETION_REQUEST_NO_DUPLICATE',key=None)
                if matches:
                    identifier,old=matches[0];d=self.directory/'candidates'/identifier
            recheck = self.saved_pauli_recheck_directory(old, d) if old else None
            if old and old.get('directory') and recheck is None:
                previous=(ROOT/old['directory']).resolve()
                if previous.is_relative_to(self.directory.resolve()):
                    d=previous
            if old and old['status']=='READY_FOR_SEARCH':return dict(old,resumed=True)
            recovered_input = (old is not None and old['status']=='HELD_TECHNICAL_FAILURE'
                               and self.recoverable_pauli_input_hold(old, d))
            if recheck is not None:
                d = recheck
            elif recovered_input:
                d = d/'saved-input-recheck'
            df_recheck = False
            received_preparation = False
            if old and (old['status'].startswith(('HELD','STOP','WAITING')) or
                        old['status'] in DF_PRICE_RECHECK_STATES) and not (recovered_input or recheck):
                if old.get('scientific_failure'):return dict(old,resumed=True)
                if not old.get('recoverable_preparation') and old['status'] == 'HELD_WITH_REASON':
                    bridge = getattr(self, 'stage_executor', None)
                    if not getattr(self, 'stop_requested', lambda: False)():
                        inspect_stage = getattr(bridge, 'hold_evidence', None)
                        evidence = inspect_stage(d, old.get('reason') or '') if inspect_stage else None
                        if evidence and evidence.get('action') == 'CHECK_ORIGINAL_OUTPUT':
                            old['preparation_recovery'] = evidence
                            old['reason'] = ('PREPARATION_RESULT_MISMATCH:ORIGINAL_AUDIT_OUTPUT_MISSING:'
                                             + evidence['job_id'] + ':' + evidence['reason'])
                            self.save()
                            return dict(old, resumed=True)
                        received_preparation = any(
                            reconcile(d, old.get('reason') or '')
                            for action in ('received_after_reservation_stop', 'resume_unreserved_stage')
                            if (reconcile := getattr(bridge, action, None)) is not None)
                if not (old.get('recoverable_preparation') or received_preparation):
                    self.ingest_price_results()
                    price_unresolved = str(old.get('reason','')).startswith(
                        ('PRICE_EVIDENCE_REQUIRED:', 'PRICE_RULE_INCOMPLETE:', DF_PRICE_UNAVAILABLE))
                    if old.get('signal_deferred_for_price'):
                        digest=self.price_registry_digest()
                        can_prepare_signal = (request['method']=='PR_PAULI' and not old.get('signal')
                                              and getattr(self, 'stage_executor', None) is not None)
                        if old.get('registry_at_hold')==digest and not can_prepare_signal:
                            return dict(old,resumed=True)
                        d=self.directory/'candidates'/identifier/'price-refresh'/digest
                        df_recheck = request['method']=='PR_DF'
                    elif request['method']=='PR_DF' and price_unresolved and old.get('configuration'):
                        digest=self.price_registry_digest()
                        if self.price_evidence_unchanged(old, digest):return dict(old,resumed=True)
                        refreshed = self.refresh_price_hold(old)
                        if refreshed:return dict(refreshed,resumed=True)
                        df_recheck = True
                        if (not old.get('signal') and not any(d.rglob('*.DISTRIBUTED.json'))
                                and not any(d.rglob('*.STARTED.json'))):
                            d=self.directory/'candidates'/identifier/'price-refresh'/digest
                    else:
                        return dict(self.refresh_price_hold(old)or old,resumed=True)
            rec=(dict(old,previous_hold=dict(status=old['status'],reason=old.get('reason')),status='SELECTED',
                      reason=None,recoverable_preparation=False)if old and (old.get('recoverable_preparation') or received_preparation or df_recheck) else
                 dict(method=request['method'],selection=request.get('selection'),request=deepcopy(request),status='SELECTED',key=identifier,directory=rel(d)))
            if df_recheck:
                rec.update(directory=rel(d), signal_deferred_for_price=False,
                           deferred_configuration=old.get('deferred_configuration') or old['configuration'],
                           deferred_tail_descriptor=old.get('deferred_tail_descriptor') or
                                                    old['request'].get('tail_descriptor'))
            if recovered_input or recheck:
                previous_hold = old.get('previous_hold') or {}
                if not str(previous_hold.get('reason')).startswith('A_PREPARATION:NEEDS_INPUT_SOURCE:'):
                    previous_hold = dict(status=old['status'], reason=old['reason'])
                rec['previous_hold'] = dict(previous_hold, metadata_receipt=rel(d.parent/'A_BASE.RETURN.json'))
                rec['directory'] = rel(d)
            if old and (old.get('signal_deferred_for_price') or old.get('deferred_configuration')):
                rec.update(deferred_configuration=old.get('deferred_configuration') or old['configuration'],
                           deferred_tail_descriptor=old.get('deferred_tail_descriptor'))
            self.state['records'][identifier]=rec;self.save()
        request=deepcopy(request)  # descriptor resolution must not change the caller's stable request key
        self.log(event='SELECTED_BY_B2',method=rec['method'],selection=rec['selection'])
        physical_guard=None
        quoted_registry=None
        try:
            if request.get('selection_error'):raise Held('B2_SELECTION_NORMALIZATION:'+request['selection_error'])
            if request['method']=='PR_PAULI':
                pauli_request=dict(request,configuration=rec['deferred_configuration']) if rec.get('deferred_configuration') else request
                config,prepared=self.pauli_configuration(pauli_request,d)
            elif rec.get('deferred_configuration'):
                config,prepared=value(rec['deferred_configuration']),None
                descriptor=rec.get('deferred_tail_descriptor') or self.saved_df_descriptor(request)
                if descriptor:request['tail_descriptor']=descriptor
            else:
                saved_descriptor=self.saved_df_descriptor(request)
                if saved_descriptor:request['tail_descriptor']=saved_descriptor
                generated=self.service('df-generate',request,d)
                if generated['status']=='NEEDS_DF_TAIL_DESCRIPTOR':
                    prefix=request['design']['L_D'];descriptor=d/'DF_TAIL_DESCRIPTOR.json'
                    self.call([self.path('NATIVE'),'-B',str(ROOT/'scripts/derive_pr_df_candidate_descriptor.py'),'--prefix',str(prefix),'--configuration','0defa286c53a1b9acf513be453907bcad83102b3:docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json','--output',str(descriptor),'--source',self.stage_source('DF_DESCRIPTOR'),'--signal-view',str(self.path('SIGNAL_VIEW'))],d,'DF_DESCRIPTOR',descriptor,science=True)
                    dc=publish([d],'Fix derived DF tail descriptor and scientific SOURCE')
                    request['tail_descriptor']=dict(commit=dc,path=rel(descriptor))
                    generated=self.service('df-generate',request,d,tag='DF_GENERATE_WITH_DESCRIPTOR')
                if generated['status']!='CONFIGURATION_READY':raise Held('CONFIGURATION:'+str(generated))
                config=generated['configuration'];prepared=None
            physical=physical_evaluation_contract(config)
            physical_guard=self.shared_preparation_lock(dict(method=request['method'],physical=physical))
            physical_guard.acquire()
            if request.get('design')and dec(config['spec']['kappa']['value'])!=dec(request['design']['kappa']):raise Held('ABSOLUTE_KAPPA_MISMATCH')
            put(d/'CONFIGURATION.json',config);commit=publish([d],'Fix selected PR candidate configuration')
            cref=dict(commit=commit,path=rel(d/'CONFIGURATION.json'));rec.update(configuration=cref,configuration_id=config['id'])
            # A declaration can differ while the complete physical schedule and
            # evaluation contract are identical. Reuse READY inputs before any
            # scientific leaf; install() retains both declarations as aliases.
            with self.lock:
                records=list(self.state['records'].values())
            twin=next((old for old in records
                if old is not rec and old.get('method')==request['method']and old.get('status')=='READY_FOR_SEARCH'
                and old.get('configuration')and physical_evaluation_contract(value(old['configuration']))==physical),None)
            if twin is not None:
                rec.update(status='READY_FOR_SEARCH',signal=deepcopy(twin['signal']),intake=deepcopy(twin['intake']),
                    quote=deepcopy(twin['quote']),prices=deepcopy(twin['prices']),price_policy=twin.get('price_policy','MODEL'),
                    reused_physical_configuration=twin['configuration'],new_scientific_calls=0,
                    design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),delta_pf=config['spec']['delta_pf']),
                    schedule=deepcopy(twin.get('schedule',[])))
                self.save();return rec
            failed_twin=next((old for old in records
                if old is not rec and old.get('method')==request['method']and old.get('scientific_failure')
                and old.get('configuration')and physical_evaluation_contract(value(old['configuration']))==physical),None)
            if failed_twin is not None:
                raise Held('PRIOR_PHYSICAL_PREPARATION_FAILED_NO_AUTOMATIC_RETRY:'+failed_twin['key'])
            initial_registry=self.price_registry_digest()
            initial_quote=self.initial_quote(config,cref,d)
            holds=price_source_holds(initial_quote)
            if request['method']=='PR_PAULI':
                already_started = (bool(old and old.get('signal')) or
                    any(d.glob('PAULI_SIGNALS.*.json')) or any(d.glob('PAULI_COMPILE.*.json')) or
                    any(d.glob('PAULI_PRICES.*.json')) or (d/'PAULI_SIGNAL_REQUEST.json').exists())
                if holds and not already_started and getattr(self, 'stage_executor', None) is None:
                    # Another scoped Pauli model cannot resolve the existing
                    # per-axis choice. Ordinary missing axes still calibrate.
                    put(d/'FINAL_QUOTE.json',initial_quote)
                    record_price_holds(rec,holds)
                    rec.update(signal_deferred_for_price=True,registry_at_hold=initial_registry)
                    raise Held(price_hold_reason(holds))
            if request['method']=='PR_DF':
                descriptor=request.get('tail_descriptor')
                rec['deferred_tail_descriptor'] = descriptor
                rec['deferred_configuration'] = cref
                if not (rec.get('signal') and rec.get('intake')):
                    df_bundles=self.saved_signal_packets('PR_DF')
                    if parent := request.get('horizon_parent'):
                        for packet in (parent['signal_packet'], *saved_packets):
                            if packet not in df_bundles:
                                df_bundles = [*df_bundles, packet]
                    plan_args = dict(configuration=config, tail_descriptor=descriptor,
                                     saved_bundles=df_bundles, terminal_reference=request.get('terminal_reference'))
                    if existing_signal_only:
                        # A newly arrived packet must invalidate only the metadata plan.
                        df_plan = self.service('df-plan', plan_args, d, tag='DF_SAVED_PLAN_'+key(plan_args))
                    else:
                        df_plan = self.service('df-plan', plan_args, d)
                    conflicts=df_plan.get('saved_signal_conflicts',[])
                    if conflicts:
                        rec['conflicting_signal_rounds']=conflicts
                        rounds=','.join(str(row['m']) for row in conflicts)
                        raise Held('DF_SAVED_SIGNAL_CONFLICT:rounds='+rounds)
                    if existing_signal_only and any(row['decision'] != 'REUSE' for row in df_plan['rounds']):
                        raise PreparationWait('AWAIT_ORIGINAL_DF_HORIZON_SIGNAL_OR_SAVED_PACKET')
                    if df_plan.get('unsupported_rounds') or any(row['decision']=='UNSUPPORTED' for row in df_plan['rounds']):
                        raise Held('DF_PROVIDER_UNSUPPORTED:'+str(df_plan))
            def intake_work(signal):
                if rec.get('intake'):
                    return rec['intake']
                intake = dict(method=request['method'],configuration=signal.get('configuration_source',cref),
                              packet=signal['packet_source'],view=str(self.path(
                                  'B_VIEW' if request['method']=='PR_PAULI' else 'DF_Q3_VIEW')))
                if request.get('horizon_parent'):
                    intake['M'] = request['M']
                return self.service('intake',intake,d)
            def signal_work():
                if rec.get('signal') and rec.get('intake'):
                    return rec['signal']
                return (self.pauli_signals(config,prepared,d)if request['method']=='PR_PAULI'
                        else self.df_signals(config,cref,d,descriptor,plan=df_plan,bundles=df_bundles))
            # Both branches depend on the frozen configuration, not on each
            # other's numerical result. They share host slots through the
            # executor rather than independently reserving CPU resources.
            if getattr(self,'stage_executor',None)is not None:
                with ThreadPoolExecutor(max_workers=2)as pool:
                    signal_future=pool.submit(signal_work)
                    price_future=pool.submit(self.price,config,cref,prepared,request,d,initial_quote)
                    signal=signal_future.result();rec.update(signal=signal);self.save()
                    intake=intake_work(signal)
                    rec['intake']=intake;self.save();quote=price_future.result()
            else:
                signal=signal_work();rec.update(signal=signal);self.save()
                intake=intake_work(signal)
                rec['intake']=intake;self.save()
                quote=self.price(config,cref,prepared,request,d,initial_quote=initial_quote)
            if not quote['complete']:
                quoted_registry = quote.get('price_registry_at_quote', initial_registry)
                if quote.get('price_decisions'):
                    rec['price_actions']=quote['price_decisions']
                holds = price_source_holds(quote)
                record_price_holds(rec, holds)
                if request['method']=='PR_DF':
                    count = sum(row.get('unit_price_RZ') is None for row in quote['rows'])
                    raise Held(f'{DF_PRICE_UNAVAILABLE}{count} unresolved price axes (DF_PRICE_DECISIONS.json)')
                if holds:
                    rec['price_actions'] = [dict(m=row['m'], axis=row['axis'],
                        physical_id=row['physical_id'], quote_needs=row['needs'],
                        action='RECHECK_REGISTERED_PRICE_EVIDENCE', new_calculation=False,
                        recheck_on='PRICE_REGISTRY_CHANGED') for row in quote['rows']
                        if row.get('unit_price_RZ') is None and row.get('needs')]
                    raise Held(price_hold_reason(holds))
                raise Held('PRICE_RULE_INCOMPLETE:'+str(quote.get('classifications')))
            quote_commit=publish([d],'Publish completed candidate prices and receipt bindings')
            with self.lock:
                local_registry=d/'PRICE_RULES.json'
                # DF publishes step rules through the locked registry writer.
                if request['method'] == 'PR_PAULI' and local_registry.exists():
                    current=read(self.registry);merged={e['name']:e for e in current['entries']};merged.update({e['name']:e for e in read(local_registry)['entries']});put(self.registry,dict(current,entries=list(merged.values())))
            rec.pop('preparation_recovery', None)
            if quote.get('price_decisions'):
                rec['price_actions'] = quote['price_decisions']
            else:
                rec.pop('price_actions', None)
            rec.update(status='READY_FOR_SEARCH',quote=dict(commit=quote_commit,path=rel(d/'FINAL_QUOTE.json')),
                price_policy='MODEL',prices=[r['unit_price_RZ']for r in quote['rows']],
                design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),
                    delta_pf=config['spec']['delta_pf']),schedule=[dict(m=r['m'],T=r['T'],q=r['q_PF'],R=r['R_outer'],actual_delta_pf=r['actual_delta_pf'],structure=r['structure'],physical_id=r['candidate']['id'])for r in config['rounds']])
        except PreparationYield as exc:
            rec.update(status='SELECTED',reason='ORCHESTRATOR_YIELD_BEFORE_STAGE:'+str(exc),
                interrupted_before_stage=str(exc),scientific_failure=False)
        except Held as exc:
            text=str(exc)
            df_decisions = d/'DF_PRICE_DECISIONS.json'
            if request['method'] == 'PR_DF' and df_decisions.is_file():
                rec['price_actions'] = read(df_decisions)
            inspect = getattr(getattr(self, 'stage_executor', None), 'hold_evidence', None)
            if inspect is not None and (evidence := inspect(d, text)) is not None:
                rec['preparation_recovery'] = evidence
            waiting=text.startswith(('PRICE_RULE_INCOMPLETE:','PRICE_EVIDENCE_REQUIRED:'))
            operational=isinstance(exc, PreparationWait)
            unavailable = (request['method']=='PR_DF' and text.startswith(DF_PRICE_UNAVAILABLE) or
                           request['method']=='PR_PAULI' and bool(rec.get('price_actions')) and waiting)
            rec.update(status='WAITING_DISTRIBUTED_PREPARATION'if operational else
                       'PRICE_NOT_EVALUATED'if unavailable else'WAITING_INPUTS'if waiting else'HELD_WITH_REASON',reason=text,
                       recoverable_preparation=operational,
                       scientific_failure=any(t in text for t in ('STAGE_FAILED:','NUMERICAL_GATE','NO_AUTOMATIC_RETRY')))
        except Exception as exc:rec.update(status='HELD_TECHNICAL_FAILURE',reason=type(exc).__name__+': '+str(exc),scientific_failure=True)
        finally:
            if physical_guard is not None:physical_guard.release()
        if (rec['status'].startswith(('HELD','STOP','WAITING')) or rec['status'] in DF_PRICE_RECHECK_STATES) and not rec.get('signal_deferred_for_price'):
            rec['registry_at_hold']=quoted_registry or self.price_registry_digest()
            if rec['method'] == 'PR_DF' and rec['status'] in DF_PRICE_RECHECK_STATES:
                rec['df_price_evidence_at_decision'] = self.df_price_evidence_snapshot(rec)
        self.save();self.log(event='CANDIDATE_COMPLETED',method=request['method'],status=rec['status'],reason=rec.get('reason'));return rec

    def plan(self,request):
        """Normal preparation entry without a scientific producer call.

        h/T and the requested width are passed to each existing mapper. A missing
        descriptor/dictionary or an unrepresentable step is returned as needs;
        another candidate can continue independently.
        """
        if request.get('method')not in ('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        identifier=key(request);prior=self.state['records'].get(identifier)
        if prior and prior['status']=='READY_FOR_SEARCH':return dict(prior,resumed=True,plan_only=True)
        d=self.directory/'plans'/identifier/self.price_registry_digest()
        answer=d/'PLAN.json'
        if answer.exists():return dict(read(answer),resumed=True)
        d.mkdir(parents=True,exist_ok=True)
        out=dict(status='WAITING_INPUTS',method=request['method'],request=deepcopy(request),plan_only=True,
                 source=self.source,configuration=None,rounds=[],calibration_requests=[],new_scientific_calls=0)
        try:
            if request['method']=='PR_PAULI':
                config,prepared=self.pauli_configuration(request,d,allow_science=False)
                signals=prepared.get('signal_plan')or dict(status='SOURCE_UNRESOLVED',rounds=[],unsupported_rounds=list(range(13)))
            else:
                selected=deepcopy(request);descriptor=self.saved_df_descriptor(selected)
                if descriptor:selected['tail_descriptor']=descriptor
                generated=self.service('df-generate',selected,d)
                if generated.get('status')!='CONFIGURATION_READY':
                    out.update(reason=generated.get('reason',generated.get('status')),configuration_needs=generated)
                    put(answer,out);return out
                config=generated['configuration'];prepared=None
                signals=self.service('df-plan',dict(configuration=config,tail_descriptor=selected.get('tail_descriptor'),
                    saved_bundles=self.saved_signal_packets('PR_DF'),terminal_reference=selected.get('terminal_reference')),d)
            quote=self.initial_quote(config,config,d)
            holds = price_source_holds(quote) if request['method']=='PR_DF' else {}
            if request['method']=='PR_PAULI':
                cal=self.service('pauli-price-plan',dict(configuration=config,prepared_item=prepared,price_quote=quote,
                    workspace=str(self.settings.get('pauli_workspace',self.path('A_WORKSPACE')))),d)
            else:
                cal=self.df_price_plan(config,quote,d)
                if holds:
                    out['reason'] = price_hold_reason(holds)
                    record_price_holds(out, holds)
            signal_rows={r['m']:r for r in signals.get('rounds',[])}
            out.update(configuration=config,configuration_id=config['id'],signal_plan=signals,price_quote=quote,
                calibration_plan=cal,calibration_requests=cal.get('calibration_requests',cal.get('groups',[])),
                rounds=[dict(m=r['m'],physical_id=r['candidate']['id'],T=r['T'],q=r['q_PF'],R=r['R_outer'],
                    actual_delta_pf=r['actual_delta_pf'],signal=signal_rows.get(r['m'],dict(decision='SOURCE_UNRESOLVED')),
                    price_axes=[row for row in quote['rows']if row['m']==r['m']])for r in config['rounds']],
                requested_delta_pf=config['spec']['delta_pf'],readiness=dict(configuration=True,
                    signal=not signals.get('request_rounds')and not signals.get('unsupported_rounds')
                        and len(signals.get('reuse_rounds',[]))==len(config['rounds']),
                    prices=quote['complete']))
        except Held as exc:out['reason']=str(exc)
        put(answer,out)
        return out

    def prepare_family(self,request,allow_science=False):
        """Prepare a chosen split's mapper metadata only, never signal or price.

        The caller supplies a same-H template kappa solely to obtain that split's
        constants. Each split still derives its own descriptor/dictionary; the
        template never supplies another split's coefficients.
        """
        if request.get('method')not in('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        d=self.directory/'family_metadata'/key(request);d.mkdir(parents=True,exist_ok=True)
        out=dict(status='NEEDS_FAMILY_PREPARATION',configuration=None,needs=[],
                 new_signal_calls=0,new_price_calls=0,scope='DICTIONARY_DESCRIPTOR_AND_MAPPER_METADATA_ONLY')
        try:
            if request['method']=='PR_PAULI':
                config,prepared=self.pauli_configuration(request,d,allow_science=allow_science)
                out.update(configuration=config,configuration_source=prepared.get('configuration_source'))
            else:
                selected=deepcopy(request);descriptor=self.saved_df_descriptor(selected)
                if descriptor:selected['tail_descriptor']=descriptor
                generated=self.service('df-generate',selected,d)
                if generated.get('status')=='NEEDS_DF_TAIL_DESCRIPTOR'and allow_science:
                    prefix=selected['design']['L_D'];target=d/'DF_TAIL_DESCRIPTOR.json'
                    template=selected.get('template_source')or dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json')
                    self.call([self.path('NATIVE'),'-B',str(ROOT/'scripts/derive_pr_df_candidate_descriptor.py'),'--prefix',str(prefix),
                        '--configuration',template['commit']+':'+template['path'],'--output',str(target),'--source',self.stage_source('DF_DESCRIPTOR'),
                        '--signal-view',str(self.path('SIGNAL_VIEW'))],
                        d,'DF_DESCRIPTOR',target,science=True)
                    fixed=publish([d],'Fix chosen split descriptor for mapper metadata')
                    selected['tail_descriptor']=dict(commit=fixed,path=rel(target))
                    generated=self.service('df-generate',selected,d,tag='DF_GENERATE_WITH_DESCRIPTOR')
                if generated.get('status')!='CONFIGURATION_READY':
                    out.update(reason=generated.get('reason',generated.get('status')),needs=[generated])
                    if generated.get('status') == 'UNSUPPORTED':
                        out['status'] = 'UNSUPPORTED_MAPPER'
                    put(d/'FAMILY.json',out);return out
                out.update(configuration=generated['configuration'],configuration_source=generated.get('configuration_source'),
                           tail_descriptor=selected.get('tail_descriptor'))
            out.update(status='CONFIGURATION_READY',needs=[])
            if out.get('tail_descriptor'):
                self.state.setdefault('family_metadata',{})[key(dict(method=request['method'],L_D=request['design']['L_D']))]=dict(
                    method=request['method'],L_D=request['design']['L_D'],tail_descriptor=out['tail_descriptor'])
                self.save()
        except Held as exc:out.update(reason=str(exc),needs=[str(exc)])
        put(d/'FAMILY.json',out);return out

    def pauli_configuration(self,r,d,allow_science=True):
        workspace=Path(self.settings.get('pauli_workspace',self.path('A_WORKSPACE')));d.mkdir(parents=True,exist_ok=True)
        saved=[dict(commit='d7a976a629a6db68197b24f8f86734e109c8bf5d',path=p)for p in (
            'docs/research/rpe_split_preparation_20261116/LD4/PREPARATION_P1.0.CONFIGURATION.json',
            'docs/research/rpe_split_preparation_20261116/LD6/PREPARATION.0.CONFIGURATION.json')]
        def prepare(item,tag,science=False):
            inp=d/(tag+'.REQUEST.json');out=d/(tag+'.json')
            put(inp,dict(kind='pauli_native',items=[item],saved_configurations=saved,saved_packets=self.saved_signal_packets('PR_PAULI')))
            cache=(self.directory/'PAULI_SPLITS.json')if hasattr(self,'directory')else d/'SPLITS.json'
            cmd=[self.path('NATIVE'),'-B',str(workspace/'scripts/prepare_pauli_candidate.py'),'--request',str(inp),'--output',str(out),'--cache',str(cache)]
            if self.settings.get('pauli_saved_inputs') and not science:
                cmd[2:2]=[str(Path(__file__).with_name('rpe_pauli_saved_inputs.py')),
                          json.dumps(self.settings['pauli_saved_inputs'])]
            requested_prefix=(item.get('configuration')or{}).get('spec',{}).get('L_D',(item.get('configuration')or{}).get('L_D'))
            if requested_prefix in (7,8,9):cmd+=['--p0-builder','no-drop']
            if science:
                publish([d],'Fix selected Pauli P0 request')
                cmd+=['--prepare-missing','--source',git('rev-parse','HEAD',cwd=workspace),'--p0-directory',str(workspace/'docs/research/d2_auto_p0_20261116')]
            lock=getattr(self,'pauli_metadata_lock',None)
            with (lock if lock is not None else nullcontext()):
                response=self.call(cmd,d,tag,out,science=science)
            if not response.get('items'):
                raise Held('A_PREPARATION:'+str(response.get('status'))+':'+str(response.get('missing_inputs',[])))
            return response['items'][0]
        if r.get('configuration'):
            config=value(r['configuration'])if 'commit'in r['configuration']else r['configuration']
            item=prepare(dict(item_id='B2_SELECTED',configuration=config),'A_CANONICAL')
        else:
            design=r['design'];item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE')
            if item['status']=='NEEDS_SPLIT_PREPARATION':
                if not allow_science:raise Held('CONFIGURATION_PREPARATION_REQUIRED_NOT_EXECUTED:'+str(item))
                if design['L_D']in(7,8,9):raise Held('REGISTERED_NO_DROP_CONFIGURATION_UNRESOLVED_NO_P0_RETRY:'+str(design['L_D']))
                item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE_P0',True)
            if item['status']!='GENERATED':raise Held('A_PREPARATION:'+str(item))
            base=item['configuration'];changes=dict(formula=design['formula'],K=design.get('K',2),kappa_multiplier=str(dec(design['kappa'])/dec(base['spec']['kappa']['value'])))
            if design.get('delta_pf')is not None:changes['delta_pf']=design['delta_pf']
            else:changes['delta_over_h']=design['delta_over_h']
            item=prepare(dict(item_id='B2_SELECTED',configuration=base,changes=changes),'A_DESIGN')
        if item['status']!='GENERATED':raise Held('A_CONFIG_NOT_READY:'+str(item))
        return item['configuration'],item

    @staticmethod
    def pauli_signal_inputs(workspace, relative, config, descriptor, packets):
        base = workspace/relative
        frozen = base/'SAVED_PACKETS.json'
        if not frozen.is_file() or read(frozen) == packets:
            return relative, packets
        if read(base/'CONFIGURATION.json') != config:
            raise Held('FIXED_PAULI_SIGNAL_CONFIGURATION_DIFFERS')
        old_descriptor = read(base/'DESCRIPTOR.json') if (base/'DESCRIPTOR.json').is_file() else None
        if descriptor != old_descriptor:
            raise Held('FIXED_PAULI_SIGNAL_DESCRIPTOR_DIFFERS')
        state = base/'SIGNAL_STATE.json'
        if state.is_file() and read(state).get('status') == 'SIGNALS_READY':
            # The fixed wrapper reuses the original complete packet.
            return relative, read(frozen)
        if any(base.rglob('SIGNALS_STARTED.json')) or any(base.rglob('STOP*.json')):
            raise Held('PRIOR_PAULI_SIGNALS_STARTED_OR_STOPPED_NO_AUTOMATIC_RETRY')
        # Acquire only missing rounds using the updated evidence. The old
        # metadata and the fixed science contract remain independently readable.
        return relative/('saved-packets-'+key(packets)), packets

    def pauli_signals(self,config,prepared,d):
        packets=self.saved_signal_packets('PR_PAULI')
        source=prepared['descriptor']['source']
        if 'commit'in source and 'path'not in source:raise Held('PAULI_DESCRIPTOR_SOURCE_UNRESOLVED')
        # Saved LD1/2/3/5 descriptors use A's index/context locator, not a
        # commit/path override. Let that same fixed entry resolve its saved
        # descriptor. Derived prefixes still require the explicit envelope.
        descriptor=(dict(prefix=config['spec']['L_D'],binding=config['spec']['binding'],
            descriptor=prepared['descriptor']['descriptor'],source=source)if 'commit'in source else None)
        if descriptor is None and(config['spec']['L_D']not in(1,2,3,5)or not all(k in source for k in('index','context_sha256'))):
            raise Held('PAULI_DESCRIPTOR_SOURCE_UNRESOLVED')
        workspace=Path(self.settings.get('pauli_workspace',self.path('A_WORKSPACE')))
        relative=Path('docs/research/d2_automatic_pr_20261116')/config['id'][:20]/self.settings.get('signal_epoch','descriptor_envelope_v2')
        relative,packets=self.pauli_signal_inputs(workspace,relative,config,descriptor,packets)
        request=dict(configuration=config,tail_descriptor=descriptor,saved_packets=packets,
            workspace=str(workspace),directory=str(relative),interpreter=self.path('NATIVE'))
        if self.settings.get('pauli_saved_inputs'):
            request['pauli_saved_inputs']=self.settings['pauli_saved_inputs']
        inp=d/'PAULI_SIGNAL_REQUEST.json';out=d/'PAULI_SIGNAL_RESULT.json';put(inp,request)
        result=self.call([self.path('NATIVE'),'-B',str(ROOT/'scripts/run_pr_pauli_candidate_signals.py'),'--request',str(inp),'--output',str(out),'--execute'],d,'PAULI_SIGNALS',out,science=True)
        if result['status']!='SIGNALS_READY':raise Held('PAULI_SIGNAL:'+str(result.get('reason')or result['status']))
        return result

    def df_signals(self,config,cref,d,descriptor,*,plan,bundles):
        result=dict(rows=[],new_provider_calls=0);obtained={}
        terminal = plan.get('terminal_reference')
        marker = d/'DF_SIGNALS.DISTRIBUTED.json'
        recovering = terminal and marker.is_file()
        if recovering or any(row['decision']=='REQUEST' for row in plan['rounds']):
            source=self.stage_source('DF_SIGNALS')
            request=dict(configuration=cref,plan=plan,tail_descriptor=descriptor,execution_source=source,
                         output_directory=rel(d/'df_signals'),runtime=dict(signal_view=str(self.path('SIGNAL_VIEW'))))
            inp = d/'DF_SIGNAL_REQUEST.json'
            if recovering:
                frozen = read(inp)
                if (frozen['execution_source'] != source
                        or frozen['plan'].get('terminal_reference') != terminal
                        or value(frozen['configuration']) != config):
                    raise Held('Existing DF terminal attempt has different physical inputs')
                input_commit = git('log','-1','--format=%H','--',rel(inp))
            else:
                put(inp,request)
                input_commit=publish([d],'Fix missing DF round jobs')
            output=d/'df_signals/RESULT.json'
            transport_root = (Path(globals()['FIXED_RUN_PR_PRICE_SERVICE']).parents[1]
                              if terminal and globals().get('FIXED_RUN_PR_PRICE_SERVICE') else ROOT)
            result=self.call([self.path('NATIVE'),'-B',str(transport_root/'scripts/run_pr_df_candidate_signal_jobs.py'),
                '--input',str(inp),'--inputs-commit',input_commit,'--source',source],d,'DF_SIGNALS',output,science=True,
                **({'poll':True} if terminal else {}))
            if result['status']not in ('REQUESTED_ROUNDS_COMPLETE','SIGNALS_READY','MISSING_ROUNDS_COMPLETE'):raise Held('DF_SIGNAL_RESULT:'+str(result['status']))
            raw=publish([d],'Publish missing DF round signals')
            obtained={str(x['m']):dict(commit=raw,path=x['raw_path'])for x in result['rows']if x.get('raw_path')}
        staged=d/'df_carrier'
        st=self.service('df-stage',dict(configuration=cref,directory=rel(staged),obtained=obtained,
            tail_descriptor=descriptor,saved_bundles=bundles,terminal_reference=terminal),d)
        if st['status']!='CARRIERS_READY':raise Held('DF_CARRIER:'+str(st))
        carrier=publish([d],'Fix unified DF raw carrier')
        configref=dict(commit=carrier,path=rel(staged/'CONFIGURATION.json'))
        self.service('df-bundle',dict(configuration_source=configref,tail_descriptor=descriptor,
            saved_bundles=bundles,terminal_reference=terminal,
            raw_sources={str(row['m']):dict(commit=carrier,path=rel(staged/'raw'/('m'+str(row['m'])+'.json')))
                         for row in config['rounds']}),d)
        pub=publish([d],'Assemble DF bundle for normal intake')
        return dict(status='SIGNALS_READY',configuration_source=configref,packet_source=dict(commit=pub,path=rel(d/'df-bundle.json')),new_signal_calls=result.get('new_provider_calls'))

    def initial_quote(self,config,cref,d):
        registry=d/'PRICE_RULES.json'
        if not registry.exists():put(registry,read(self.registry))
        req=dict(configuration=cref,view=str(self.path('D2_VIEW')),registry_file=str(registry),policy='MODEL')
        quote=self.service('quote',req,d,tag='INITIAL_QUOTE',python=self.path('PRICE'))
        return self.quote_with_steps(config,quote)

    def price(self,config,cref,prepared,r,d,initial_quote=None):
        lock=getattr(self,'pauli_price_lock',None)
        if r['method']=='PR_PAULI' and self.parallel_pauli_preparation():
            # Distributed leaves use their own attempt workspaces. Pauli
            # calibration groups within one split can still overlap.
            lock=self.shared_preparation_lock(dict(stage='PAULI_PRICE',L_D=config['spec']['L_D']))
        with (lock if r['method']=='PR_PAULI'and lock is not None else nullcontext()):
            return self._price(config,cref,prepared,r,d,initial_quote)

    @staticmethod
    def pauli_price_preparation(path, prepared):
        # The calibration entry reads the fixed configuration, dictionary,
        # schedule and implementation. Its predecessor's signal acquisition
        # plan may differ after more saved means become available.
        if not path.is_file() or not isinstance(prepared, dict):
            return prepared
        frozen = read(path)
        metadata = {'configuration_file', 'signal_plan', 'signal_plan_command'}
        if {frozen.get('preparation_status'), prepared.get('preparation_status')} == {
                'GENERATED_CONFIGURATION', 'REUSED_LOCAL_METADATA_CONFIGURATION'}:
            metadata.add('preparation_status')
        scientific = lambda item: {k: v for k, v in item.items() if k not in metadata}
        return frozen if scientific(frozen) == scientific(prepared) else prepared

    def _price(self,config,cref,prepared,r,d,initial_quote=None):
        registry=d/'PRICE_RULES.json'
        req=dict(configuration=cref,view=str(self.path('D2_VIEW')),registry_file=str(registry),policy='MODEL')
        quote=self.initial_quote(config,cref,d)if initial_quote is None else initial_quote
        registry_digest=self.price_registry_digest()
        quote=self.quote_with_steps(config,quote)
        if r['method']=='PR_PAULI' and price_source_holds(quote):
            quote['price_registry_at_quote'] = registry_digest
            put(d/'FINAL_QUOTE.json',quote)
            return quote
        if r.get('horizon_parent') and not quote['complete']:
            # The terminal extension inherits the parent's model policy. A
            # missing model quote does not request a large direct circuit here.
            quote['price_registry_at_quote'] = registry_digest
            put(d/'FINAL_QUOTE.json',quote)
            return quote
        if not quote['complete']:
            if r['method']=='PR_PAULI':
                inp=d/'PAULI_PRICE_REQUEST.json';out=d/'PAULI_PRICE_RESULT.json'
                workspace=Path(self.settings.get('pauli_workspace',self.path('A_WORKSPACE')))
                relative='docs/research/d2_automatic_price_20261116/'+config['id'][:20]
                price_prepared=self.pauli_price_preparation(workspace/relative/'PREPARATION.json',prepared)
                put(inp,dict(configuration_source=cref,configuration=config,prepared_item=price_prepared,
                    price_quote=quote,registry_file=str(registry),workspace=str(workspace),
                    directory=relative,private_root=str(ROOT.parent/'d2-auto-private'/config['id'][:20]),
                    d2_view=str(self.path('D2_VIEW')),interpreter=self.path('PRICE'),
                    scoped_registration=True,execution_source=self.source))
                try:
                    cmd=[self.path('PRICE'),'-B',str(ROOT/'scripts/run_pr_pauli_candidate_prices.py'),'--request',str(inp),'--output',str(out),'--execute']
                    if getattr(self,'stage_executor',None)is not None:
                        compiled=self.call(cmd+['--phase','compile','--no-push'],d,'PAULI_COMPILE',out,science=True)
                        if compiled.get('status')not in('COMPILE_READY_FOR_AUDIT','STEP_COMPILES_READY_FOR_AUDIT','PRICES_READY','SCOPED_STEP_MODEL_READY','STEP_MODELS_READY'):
                            raise Held('PAULI_COMPILE:'+str(compiled))
                        result=self.call(cmd+['--phase','audit','--no-push'],d,'PAULI_AUDIT',out,science=True)
                    else:result=self.call(cmd,d,'PAULI_PRICES',out,science=True)
                except Held:
                    # Valid completed groups remain useful to independent
                    # candidates even if a later group reaches its own STOP.
                    if out.exists()and read(out).get('evidence_packages'):
                        self.ingest_price_results(read(out)['evidence_packages'])
                    raise
                if result.get('evidence_packages'):
                    self.ingest_price_results(result['evidence_packages'])
                if result['status']not in('PRICES_READY','PRICE_READY','COMPLETE_MODEL_26_AXES','STEP_MODELS_READY'):
                    raise Held(('PRICE_EVIDENCE_REQUIRED:'if result['status']in('WAITING_PRICE_EVIDENCE','CALIBRATION_PLANNED_NOT_EXECUTED')else'PAULI_PRICE:')+str(result))
            else:self.df_price(cref,quote,d,registry,r.get('tail_descriptor'))
            registry_digest=self.price_registry_digest()
            quote=self.quote_with_steps(config,quote)
            if not quote['complete']:
                quote=self.service('quote',req,d,tag='REQUOTE',python=self.path('PRICE'))
                quote=self.quote_with_steps(config,quote)
        decisions=d/'DF_PRICE_DECISIONS.json'
        if r['method']=='PR_DF' and decisions.is_file():
            priced = {(row['m'], row['axis']) for row in quote['rows'] if row.get('unit_price_RZ') is not None}
            actions = {(row['m'],row['axis']):row for row in read(decisions)}
            for index in priced & actions.keys():
                if actions[index]['action'] not in ('REUSE_PRICE', 'AUDIT_ORIGINAL_COMPILE'):
                    actions[index]=dict(actions[index],action='PRICE_ADMITTED')
            quote=dict(quote,price_decisions=list(actions.values()))
        if not quote['complete']:
            quote['price_registry_at_quote'] = registry_digest
        put(d/'FINAL_QUOTE.json',quote)
        return quote

    def df_price_plan(self,config,quote,d):
        plan = self.service('df-price-design',dict(configuration=config,price_quote=quote,resolve_axes=True),
                            d,tag='DF_PRICE_PLAN_'+key(quote['rows']))
        prior = d/'DF_PRICE_GROUPS.json'
        if not prior.is_file():
            return plan
        owned = []
        for group in read(prior).get('groups', []):
            owner = d/'df_price_groups'/group['group_id'][:20]
            leaf = self.df_execution_leaf(owner)
            if ((leaf/'df_prices/INPUTS.json').is_file() and not (owner/'PRICE_EVIDENCE.json').is_file()
                    and any(leaf.glob('DF_CALIBRATION.*.json'))):
                owned.append(group)
        if not owned:
            return plan
        # Complete the original INPUTS and its queue request first.
        return dict(plan,groups=[*owned, *[g for g in plan['groups']
                    if not any(set(g['covered_rounds']) & set(o['covered_rounds']) for o in owned)]])

    def df_price_owner_roots(self):
        executor=getattr(getattr(self,'stage_executor',None),'executor',None)
        return [Path(run)/'completion' for run in getattr(executor,'prior_run_dirs', ())
                if Path(run).resolve()!=self.directory.parent]

    @staticmethod
    def df_execution_leaf(owner):
        marker = owner/'EXECUTION_LEAF.json'
        if not marker.is_file():
            return owner
        name = read(marker)['name']
        if Path(name).name != name or not name.startswith('evidence-'):
            raise ValueError('Malformed DF price owner leaf')
        return owner/name

    def df_attempt_observation(self, marker):
        """Read the matched run's ledger without claiming or retrying its work."""
        if not marker.name.endswith('.DISTRIBUTED.json'):
            return {}
        from .rpe_preparation_bridge import DistributedPreparationBridge
        from .rpe_preparation_queue import normalize_preparation_request
        from .rpe_q3_coordinator import digest
        for root in (self.directory, *self.df_price_owner_roots()):
            if not marker.resolve().is_relative_to(root.resolve()):
                continue
            database = root.parent/'distributed/QUEUE.sqlite3'
            try:
                submitted = read(marker)
                job_id = submitted['job_id']
                stage = marker.name.removesuffix('.DISTRIBUTED.json')
                dependencies = DistributedPreparationBridge._dependencies(marker.parent, stage)
                request = normalize_preparation_request(submitted['request'], dependencies=dependencies)
                if request['request_id'] != job_id:
                    return {}
                with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=1) as db:
                    row = db.execute('''SELECT state,attempt_id,host_id,completion_sha256 FROM jobs
                                        WHERE job_id=? AND request_sha256=?''',
                                     (job_id,digest(request))).fetchone()
                return (dict(zip(('state','attempt_id','host_id','completion_sha256'), row), job_id=job_id)
                        if row else {})
            except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
                return {}
        return {}

    def df_price_waiting_for_owner(self):
        """Only live ledger owners keep the unattended recovery loop active."""
        with self.lock:
            markers = {a['original_marker'] for r in self.state['records'].values()
                       if r['status'] in DF_PRICE_RECHECK_STATES for a in r.get('price_actions', ())
                       if a.get('original_marker') and a.get('action') == 'RECONCILE_ORIGINAL_ATTEMPT'}
        return any(self.df_attempt_observation(Path(path)).get('state') in
                   ('QUEUED','ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')
                   for path in markers)

    def df_price_evidence_snapshot(self, record):
        """Wake only when a registered receipt or observed owner's evidence changes."""
        files = set()
        groups = {a['group_id'] for a in record.get('price_actions', ()) if a.get('group_id')}
        owners = {Path(a['original_marker']).parent for a in record.get('price_actions', ())
                  if a.get('original_marker')}
        for root in [self.directory, *self.df_price_owner_roots()]:
            for gid in groups:
                marker = root/'df-price-owners'/(gid+'.json')
                files.add(marker)
                try:
                    owner = Path(read(marker)['directory']).resolve()
                    if not owner.is_relative_to(root.resolve()):
                        continue
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                owners.add(owner)
        for owner in owners:
            try:
                leaf = self.df_execution_leaf(owner)
            except (OSError, KeyError, TypeError, ValueError):
                continue
            files.update((owner/'PRICE_EVIDENCE.json', owner/'EXECUTION_LEAF.json',
                          leaf/'df_prices/INPUTS.json'))
            for stage, name in (('DF_CALIBRATION', 'calibration'), ('DF_HOLDOUT', 'holdout')):
                files.update(leaf/(stage+suffix) for suffix in ('.DISTRIBUTED.json', '.STARTED.json', '.RETURN.json'))
                files.add(leaf/'df_prices'/name/'acquisition/RESULT.json')
            files.update(leaf/'df_prices'/name for name in ('FROZEN_MODELS.json','HOLDOUT_COMPARISON.json'))
        observed = []
        for path in sorted(files):
            try:
                stat = path.stat()
                observed.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                observed.append((str(path), None, None))
        configured = self.settings.get('saved_df_price_axis_quotes', ())
        try:
            objects = self.df_saved_axis_objects(configured)
        except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError):
            objects = None
        attempts = {str(path): self.df_attempt_observation(path) for path in files
                    if path.name.endswith('.DISTRIBUTED.json') and path.is_file()}
        return key(dict(files=observed, configured=configured, objects=objects, attempts=attempts,
                        source=self.stage_source('DF_PRICE_PREPARE')))

    @staticmethod
    def df_saved_axis_objects(locators):
        from .pf_saved_prices import receipt_objects
        return receipt_objects(ROOT, [loc['commit']+':'+loc['path'] for loc in locators])

    def df_original_axis_quotes(self, owner):
        """Expose finished immutable raw wrappers to the existing DF receipt validator."""
        leaf = self.df_execution_leaf(owner)
        base = leaf/'df_prices'
        rows = []
        for stage, name, role in (('DF_CALIBRATION', 'calibration', 'CALIBRATION'),
                                  ('DF_HOLDOUT', 'holdout', 'HOLDOUT')):
            try:
                recovery = getattr(getattr(self, 'stage_executor', None), 'import_received_stage', None)
                marker = leaf/(stage+'.DISTRIBUTED.json')
                if (recovery is not None and marker.is_file()
                        and leaf.resolve().is_relative_to(self.directory.resolve())
                        and not (leaf/(stage+'.RETURN.json')).is_file()):
                    recovery(self, marker, stage)
                ended = read(leaf/(stage+'.RETURN.json'))
                path = base/name/'acquisition/RESULT.json'
                result = read(path)
                if (ended['returncode'] != 0 or ended.get('signal') is not None
                        or result['status'] != 'ACQUIRED' or result['source'] != ended['source']):
                    continue
                working_inputs = read(base/'INPUTS.json')
                fixed_output = Path(working_inputs['output_directory'])
                if fixed_output.is_absolute() or '..' in fixed_output.parts:
                    continue
                relative = (fixed_output/name/'acquisition/RESULT.json').as_posix()
                audit = base/('FROZEN_MODELS.json' if role == 'CALIBRATION' else 'HOLDOUT_COMPARISON.json')
                snapshot = read(audit).get('raw_commit') if audit.is_file() else None
                if not snapshot:
                    for head in ended.get('git_heads', ()):
                        try:
                            if value(dict(commit=head, path=relative)) == result:
                                snapshot = head
                                break
                        except (OSError, ValueError, subprocess.CalledProcessError):
                            continue
                if not snapshot:
                    snapshot = git('log', '-1', '--format=%H', '--', relative, cwd=ROOT)
                if not snapshot or value(dict(commit=snapshot, path=relative)) != result:
                    continue
                inputs = value(dict(commit=result['inputs_commit'], path=(fixed_output/'INPUTS.json').as_posix()))
                if inputs != working_inputs or inputs['source'] != result['source']:
                    continue
                observed = {(row['m'], row['index']): row for row in result['rows']}
                recovered = []
                for cell in inputs['cells']:
                    if cell['role'] != role:
                        continue
                    physical = cell['request']['physical_context']['physical_candidate']
                    for axis in ('X', 'Y'):
                        witnesses = []
                        for index in cell['indices']:
                            row = observed.get((cell['m'], index), {})
                            witness = deepcopy(row.get('axes', {}).get(axis, {}))
                            if (row.get('status') != 'VALID' or witness.get('status') != 'REUSE' or
                                    not all(isinstance(witness.get(field), dict) and
                                            witness[field].get('commit') and witness[field].get('path')
                                            for field in ('receipt', 'path_receipt'))):
                                continue
                            for field in ('receipt', 'path_receipt'):
                                if witness[field]['commit'] == 'WORKING':
                                    witness[field]['commit'] = snapshot
                            witnesses.append(witness)
                        if not witnesses:
                            continue
                        recovered.append(dict(m=cell['m'], axis=axis, q=physical['q'], R=physical['R_outer'],
                            physical_id=physical['id'], direct_alternatives=[dict(role=role, cohort=cell['cohort'],
                                source=dict(commit=snapshot, path=relative), path_witnesses=witnesses)]))
                rows.extend(recovered)
            except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError):
                continue
        return dict(schema='PR_DF_PARTIAL_AXIS_QUOTES_V1', rows=rows) if rows else None

    def df_existing_price_attempt(self, own_input, cell, missing_axes):
        """Read scientific ownership of a matching circuit in known run trees."""
        patterns = ('*/df_prices/INPUTS.json',
                    '*/df_price_groups/*/df_prices/INPUTS.json',
                    '*/df_price_groups/*/evidence-*/df_prices/INPUTS.json',
                    '*/price-refresh/*/df_price_groups/*/df_prices/INPUTS.json',
                    '*/price-refresh/*/df_price_groups/*/evidence-*/df_prices/INPUTS.json')
        for root in (self.directory, *self.df_price_owner_roots()):
            for path in (p for pattern in patterns for p in (root/'candidates').glob(pattern)):
                directory=path.parent.parent
                if path==own_input or not any(
                        (directory/(stage+suffix)).exists()
                        for stage in ('DF_CALIBRATION','DF_HOLDOUT')
                        for suffix in ('.DISTRIBUTED.json','.STARTED.json','.RETURN.json')):
                    continue
                for previous in read(path)['cells']:
                    old=previous['request']['physical_context']
                    now=cell['request']['physical_context']
                    fields=('physical_candidate','structure','binding','split','price_profile',
                            'implementation','expected_scalar')
                    if not all(now.get(k)==old.get(k) for k in fields):
                        continue
                    stage='DF_CALIBRATION' if previous['role']=='CALIBRATION' else 'DF_HOLDOUT'
                    for suffix in ('.DISTRIBUTED.json','.STARTED.json','.RETURN.json'):
                        marker=directory/(stage+suffix)
                        if marker.exists() and missing_axes.intersection(previous['request']['axes']):
                            state = self.df_attempt_observation(marker)
                            if state.get('state') == 'CANCELLED' and state.get('attempt_id') is None:
                                continue
                            return marker
        return None

    def df_price(self,cref,quote,d,registry,descriptor=None):
        config=value(cref)
        planned=self.df_price_plan(config,quote,d)
        put(d/'DF_PRICE_GROUPS.json', planned)
        groups=[g for g in planned.get('groups',[]) if g['status']=='CALIBRATION_DESIGN_READY']
        packages=[];decisions=[];waiting=None
        for group in groups:
            gid=group['group_id'];directory=d/'df_price_groups'/gid[:20]
            design=group.get('design',{})
            rounds=list(design.get('calibration_rounds',group['covered_rounds']))
            if design.get('holdout_round') is not None:
                rounds.append(design['holdout_round'])
            family={k:config['spec'][k] for k in ('binding','split','profile')}
            cells={key(dict(family=family,physical=config['rounds'][m]['physical_candidate'])) for m in rounds}
            with ExitStack() as owners:
                owners.enter_context(self.shared_preparation_lock(dict(stage='DF_PRICE_GROUP',group_id=gid)))
                for cell in sorted(cells):
                    owners.enter_context(self.shared_preparation_lock(dict(stage='DF_PRICE_PHYSICAL',cell=cell)))
                # A different candidate may have completed this exact group
                # while we waited. Requote from its verified registry first.
                refreshed=self.quote_with_steps(config,quote)
                missing={(r['m'],r['axis'])for r in refreshed['rows']if r.get('unit_price_RZ')is None}
                if not any((r['m'],r['axis'])in missing for r in group['axis_keys']):
                    decisions.extend(dict(axis, action='REUSE_PRICE', new_calculation=False)
                                     for axis in group['axis_keys'])
                    continue
                name=Path('df-price-owners')/(gid+'.json')
                owner_file=self.directory/name
                prior=next((root/name for root in self.df_price_owner_roots() if (root/name).is_file()),None)
                recovered_axes = None
                if not owner_file.exists() and prior is not None:
                    original=Path(read(prior)['directory']).resolve()
                    if not original.is_relative_to(prior.parents[1].resolve()):
                        decisions.extend(dict(axis, action='NO_SAFE_PRICE_ACTION', group_id=gid,
                            reason='ORIGINAL_DF_OWNER_PATH_DIFFERS', new_calculation=False)
                            for axis in group['axis_keys'])
                        continue
                    if (original/'PRICE_EVIDENCE.json').is_file():
                        package=read(original/'PRICE_EVIDENCE.json')
                        self.ingest_price_results([package]);packages.append(package)
                        decisions.extend(dict(axis, action='REUSE_PRICE', group_id=gid, new_calculation=False)
                                         for axis in group['axis_keys'])
                        continue
                    recovered_axes = self.df_original_axis_quotes(original)
                    original_leaf = self.df_execution_leaf(original)
                    marker = next((original_leaf/('DF_CALIBRATION'+suffix) for suffix in
                                   ('.DISTRIBUTED.json', '.STARTED.json', '.RETURN.json')
                                   if (original_leaf/('DF_CALIBRATION'+suffix)).is_file()), None)
                    observation = self.df_attempt_observation(marker) if marker else {}
                    cancelled = observation.get('state') == 'CANCELLED' and observation.get('attempt_id') is None
                    if not recovered_axes and not cancelled and (marker or (original/'EXECUTION_LEAF.json').exists()):
                        evidence = dict(original_marker=str(marker), original_attempt=observation,
                                        recheck_on='ORIGINAL_ATTEMPT_RECEIPT') if marker else {}
                        decisions.extend(dict(axis,action='RECONCILE_ORIGINAL_ATTEMPT',group_id=gid,
                            new_calculation=False,
                            reason='PRIOR_RUN_DF_PRICE_OWNER_UNRESOLVED:'+str(original), **evidence)
                            for axis in group['axis_keys'])
                        continue
                owner=Path(read(owner_file)['directory']).resolve() if owner_file.is_file() else directory.resolve()
                if not owner.is_relative_to(self.directory.resolve()):
                    decisions.extend(dict(axis, action='NO_SAFE_PRICE_ACTION', group_id=gid,
                        reason='ORIGINAL_DF_OWNER_PATH_DIFFERS', new_calculation=False) for axis in group['axis_keys'])
                    continue
                saved=owner/'PRICE_EVIDENCE.json'
                reused = saved.is_file()
                if reused:
                    package=read(saved)
                elif owner != directory.resolve():
                    if (owner/'NO_SAFE_ACTION.json').exists():
                        original = read(owner/'NO_SAFE_ACTION.json')
                        action = ('RECONCILE_ORIGINAL_ATTEMPT' if original.get('original_marker')
                                  else 'NO_SAFE_PRICE_ACTION')
                        decisions.extend(dict(axis,action=action,group_id=gid,new_calculation=False,
                            **original) for axis in group['axis_keys'])
                    else:
                        waiting=PreparationWait('DF_PRICE_ORIGINAL_OWNER:'+str(owner))
                        decisions.extend(dict(axis,action='RECONCILE_ORIGINAL_ATTEMPT',group_id=gid,
                                              new_calculation=False,reason=str(waiting))
                                         for axis in group['axis_keys'])
                    continue
                else:
                    put(owner_file,dict(group_id=gid,directory=str(directory)))
                    try:
                        if group.get('direct_resolution') and not (directory/'EXECUTION_LEAF.json').exists():
                            protected=[r['axis'] for r in refreshed['rows']
                                       if r['m'] in group['covered_rounds'] and r['unit_price_RZ'] is not None]
                            group=dict(group,required_saved_companions=protected)
                        options = {'recovered_axes': recovered_axes} if recovered_axes else {}
                        package=self.df_price_group(cref,refreshed,directory,registry,group,descriptor,**options)
                    except Held as error:
                        if isinstance(error, PreparationWait):
                            waiting=error
                            decisions.extend(dict(axis,action='RECONCILE_ORIGINAL_ATTEMPT',group_id=gid,
                                                  new_calculation=False,reason=str(error))
                                             for axis in group['axis_keys'])
                            continue
                        disposition = dict(reason=str(error))
                        if isinstance(error, DFOriginalPriceAttempt):
                            disposition.update(original_marker=str(error.marker), recheck_on='ORIGINAL_ATTEMPT_RECEIPT',
                                               original_attempt=self.df_attempt_observation(error.marker))
                        put(directory/'NO_SAFE_ACTION.json',disposition)
                        if not str(error).startswith(DF_PRICE_UNAVAILABLE):
                            raise
                        decisions.extend(dict(axis,action='RECONCILE_ORIGINAL_ATTEMPT' if isinstance(error, DFOriginalPriceAttempt)
                                              else 'NO_SAFE_PRICE_ACTION',group_id=gid,new_calculation=False, **disposition)
                                         for axis in group['axis_keys'])
                        continue
                    put(saved,package)
                self.ingest_price_results([package]);packages.append(package)
                new_axes = {(r['m'], r['axis']) for r in package.get('calibration_compile_axes', ())} if not reused else set()
                action = 'REUSE_PRICE' if reused else 'AUDIT_ORIGINAL_COMPILE' if recovered_axes and not new_axes else 'PRICE_ADMITTED'
                decisions.extend(dict(axis, action=action, group_id=gid,
                    new_calculation=(axis['m'], axis['axis']) in new_axes)
                    for axis in group['axis_keys'])
        resolved = {(r['m'], r['axis']): dict(r, new_calculation=False)
                    for r in planned.get('round_actions', [])}
        for row in decisions:
            index = row['m'], row['axis']
            resolved[index] = dict(resolved.get(index, {}), **row)
        put(d/'DF_PRICE_DECISIONS.json', list(resolved.values()))
        if waiting is not None:
            raise waiting
        return packages

    def df_price_group(self,cref,quote,d,registry,group,descriptor=None,*,recovered_axes=None):
        """One existing DF calibration/holdout campaign per exact actual step."""
        owner=d
        direct=group.get('direct_resolution')
        cohort='D2_AUTO_DF_'+d.name
        leaf=d/'EXECUTION_LEAF.json'
        if direct:
            if leaf.is_file():
                d=d/read(leaf)['name']
            else:
                evidence = dict(rows=quote['rows'], companions=group['required_saved_companions'],
                                saved=self.settings.get('saved_df_price_axis_quotes', []))
                if recovered_axes:
                    evidence['recovered_axes'] = key(recovered_axes)
                d=d/('evidence-'+key(evidence))
        d.mkdir(parents=True,exist_ok=True);base=d/'df_prices';inp=base/'INPUTS.json'
        args=[self.path('PRICE'),'-B','-c',_DF_PRICE_BOOTSTRAP,
              str(ROOT/'scripts/run_pr_df_candidate_prices.py'),str(self.path('DF_VIEW'))]
        if core := self.settings.get('df_price_openblas_coretype'):
            args.extend(['--openblas-coretype',core])
        descriptor_file=d/'DF_DESCRIPTOR_LOCATOR.json'
        if descriptor and not descriptor_file.exists():put(descriptor_file,descriptor)
        quote_file=d/'AVAILABLE_PRICE_QUOTE.json'
        if not quote_file.exists():
            prepared_quote=deepcopy(quote)
            if direct:
                for row in prepared_quote['rows']:
                    for alt in row.get('direct_alternatives', ()):
                        paths=alt.get('paths')
                        cohorts={p.get('cohort') for p in paths or ()}
                        if len(cohorts)==1 and None not in cohorts and not alt.get('role'):
                            alt.update(cohort=next(iter(cohorts)),role='CALIBRATION',path_witnesses=paths,
                                       source=(row.get('source') or {}).get('price_source'))
            put(quote_file,prepared_quote)
        parallel=['--parallel-axes'] if self.settings.get('parallel_df_price_axes') is True else []
        price_source=self.stage_source('DF_PRICE_PREPARE')
        saved_axes=[]
        locators = list(self.settings.get('saved_df_price_axis_quotes', ()))
        if recovered_axes and not inp.exists():
            found = d/'RECOVERED_AXIS_QUOTES.json'
            put(found,recovered_axes)
            commit = publish([found], 'Bind original DF path and wrapper receipt locators')
            locators.append(dict(commit=commit,path=rel(found)))
        saved_file=d/'SAVED_AXIS_QUOTES.json'
        if locators and not saved_file.exists() and not inp.exists():
            put(saved_file,locators)
        if saved_file.is_file():
            if not inp.exists():
                try:
                    unavailable = [ref for ref, status in self.df_saved_axis_objects(read(saved_file)).items()
                                   if status.rstrip().endswith((' missing', ' ambiguous'))]
                except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
                    raise Held(DF_PRICE_UNAVAILABLE+' SAVED_AXIS_RECEIPT_INDEX_UNREADABLE: '+str(error)) from error
                if unavailable:
                    raise Held(DF_PRICE_UNAVAILABLE+' SAVED_AXIS_RECEIPT_OBJECT_UNAVAILABLE: '+str(unavailable))
            saved_axes=['--saved-axis-quotes',str(saved_file)]
        self.call(args+['prepare']+parallel+saved_axes+(['--tail-descriptor',str(descriptor_file)]if descriptor else[])+['--configuration',cref['commit']+':'+cref['path'],'--output',rel(base),'--cohort',cohort,'--source',price_source,'--price-quote',str(quote_file),'--covered-rounds',json.dumps(group['covered_rounds'])],d,'DF_PRICE_PREPARE')
        if not inp.exists():
            raise Held(DF_PRICE_UNAVAILABLE+' PRICE_RULE_OR_RECEIPT_REUSE_REQUIRED:'+str(read(base/'NEEDS.json')))
        frozen_inputs=read(inp)
        compile_axes = {'CALIBRATION': [], 'HOLDOUT': []}
        for cell in frozen_inputs['cells']:
            role = cell.get('role', 'CALIBRATION')
            evidence = cell.get('axis_evidence')
            if evidence is not None and any(item['status'] not in ('MISSING','REUSE')
                                            for path in evidence.values() for item in path.values()):
                raise Held(DF_PRICE_UNAVAILABLE+' EXISTING_AXIS_RECEIPT_REQUIRES_RECONCILIATION')
            axes = ({axis for path in evidence.values() for axis, item in path.items()
                     if item['status'] == 'MISSING'} if evidence is not None else {'X', 'Y'})
            compile_axes[role].extend(dict(m=cell['m'], axis=axis) for axis in sorted(axes))
            if recovered_axes and axes:
                stage = 'DF_CALIBRATION' if role == 'CALIBRATION' else 'DF_HOLDOUT'
                if not any((d/(stage+suffix)).exists() for suffix in
                           ('.DISTRIBUTED.json', '.STARTED.json', '.RETURN.json')):
                    original = self.df_existing_price_attempt(inp, cell, axes)
                    if original is not None:
                        raise DFOriginalPriceAttempt(original)
        compile_needed=bool(compile_axes['CALIBRATION'])
        if direct:
            [cell]=frozen_inputs['cells']
            paths=list(cell['axis_evidence'].values())
            for axis in group['required_saved_companions']:
                if any(path[axis]['status']!='REUSE' for path in paths):
                    raise Held(DF_PRICE_UNAVAILABLE+' DIRECT_PAIR_REQUIRES_SAVED_COMPANION:'+axis)
                existing=next(r for r in read(quote_file)['rows'] if (r['m'],r['axis'])==(cell['m'],axis))
                if sum(dec(path[axis]['C']) for path in paths)/len(paths)!=dec(existing['unit_price_RZ']):
                    raise Held(DF_PRICE_UNAVAILABLE+' SAVED_COMPANION_PRICE_DIFFERS:'+axis)
            if not leaf.exists():
                missing={a for path in paths for a,v in path.items() if v['status']=='MISSING'}
                original=self.df_existing_price_attempt(inp,cell,missing) if missing else None
                if original is not None:
                    # Another plan can have reserved the same physical axis
                    # under a different group ID. Recover exact old wrappers
                    # before choosing a new cohort or reserving any science.
                    if recovered_axes is None:
                        recovered = self.df_original_axis_quotes(original.parent)
                        if recovered is not None:
                            return self.df_price_group(cref, quote, owner, registry, group, descriptor,
                                                       recovered_axes=recovered)
                    raise DFOriginalPriceAttempt(original)
            if compile_needed and not getattr(getattr(self,'stage_executor',None),'handles',lambda _:False)('DF_CALIBRATION'):
                raise Held(DF_PRICE_UNAVAILABLE+' DISTRIBUTED_DF_CALIBRATION_UNAVAILABLE')
            if not leaf.exists():
                put(leaf,dict(name=d.name))
        inputs=publish([d],'Fix candidate DF calibration and independent holdout inputs')
        common=['--input',str(inp),'--inputs-commit',inputs,'--source',price_source]
        calibration_dispatch = compile_needed or (d/'DF_CALIBRATION.DISTRIBUTED.json').is_file()
        self.call(args+['acquire',*common,'--stage','calibration'],d,'DF_CALIBRATION',science=calibration_dispatch)
        raw=publish([d],'Save DF calibration wrapper receipts')
        self.call(args+['freeze',*common,'--raw-commit',raw],d,'DF_MODEL_FREEZE',science=True)
        frozen=publish([d],'Freeze candidate DF full-wrapper model before holdout')
        holdout=None
        if frozen_inputs['model_design'].get('holdout_round')is not None:
            holdout_dispatch = bool(compile_axes['HOLDOUT']) or (d/'DF_HOLDOUT.DISTRIBUTED.json').is_file()
            self.call(args+['acquire',*common,'--stage','holdout','--model-commit',frozen],d,'DF_HOLDOUT',
                      science=holdout_dispatch)
            hold=publish([d],'Save independent DF holdout receipts')
            self.call(args+['compare-holdout',*common,'--raw-commit',hold,'--model-commit',frozen],d,'DF_HOLDOUT_COMPARE',science=True)
            compared=publish([d],'Publish scoped DF independent holdout comparison')
            holdout=dict(commit=compared,path=rel(base/'HOLDOUT_COMPARISON.json'))
        reg=read(registry)
        imp=next(e['implementation']for e in reg['entries']if e['name']=='DF_LD7')
        return dict(kind='PR_DF_STEP_MODEL',configuration_source=cref,implementation=imp,
            inputs_source=dict(commit=inputs,path=rel(inp)),model_source=dict(commit=frozen,path=rel(base/'FROZEN_MODELS.json')),
            price_source=dict(commit=frozen,path=rel(base/'AXIS_PRICES.json')),holdout_source=holdout,
            calibration_compile_axes=compile_axes['CALIBRATION'])

    def common_route(self,method,cid,configuration,packet,binding):
        """The same fixed worker/contract for newly prepared and saved candidates."""
        view=self.path('B_VIEW') if method=='PR_PAULI'else self.path('DF_Q3_VIEW')
        return dict(kind='pauli_b_common_entry'if method=='PR_PAULI'else'prdf_d2_common_entry',interpreter=self.path('NATIVE'),
            view=str(view),view_commit=git('rev-parse','HEAD',cwd=view),configuration_ids=[cid],
            contract_template=(dict(commit='b6d4dea49c39f364048af422bc2928ee7c169261',path='docs/research/pauli_ld5_two_vectors_q3_results_20261116/INPUTS.json')if method=='PR_PAULI'else dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/Q3_INPUTS.json',inputs_selector=['inputs'])),
            configurations={cid:dict(configuration=configuration,signal_packet=packet,
                binding_id=binding['binding_id'],total_configuration_id=binding['total_configuration_id'])})

    def install_saved_routes(self,context,routes):
        """Reconnect catalog-only PR cases through their own source-bound metadata.

        Existing usable routes remain unchanged, preserving prior-job reuse.
        Missing/conflicting metadata is a candidate-specific preparation need;
        neither signals nor numerical results are synthesized here.
        """
        from . import rpe_method_optimizer as O
        reader=context.get('reader')or O.Reader(ROOT)
        points=context.get('catalog',{}).get('candidates',[])
        for c in context['candidates']:
            method=c['method'];cid=c.get('configuration_id')
            if method not in('PR_PAULI','PR_DF')or not cid or not c.get('ready'):continue
            usable=False
            for rule in routes.get(method,[]):
                try:usable=O.route_supports_configuration(rule,cid,reader)
                except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError):continue
                if usable:break
            if usable:continue
            saved=[p for p in points if p.get('method')==method and p.get('configuration_id')==cid]
            try:
                if not saved:continue
                p=saved[0]
                fields=('configuration_source','signal_source','binding_id','total_configuration_id')
                if any(any(row.get(k)!=p.get(k)for k in fields)for row in saved):
                    raise ValueError('SAVED_ROUTE_BINDINGS_CONFLICT')
                if any(p.get(k)is None for k in fields):raise ValueError('SAVED_ROUTE_BINDING_INCOMPLETE')
                if c.get('signal_source')!=p['signal_source']:raise ValueError('SAVED_ROUTE_CANDIDATE_SIGNAL_DIFFERS')
                config=reader.read(p['configuration_source'])
                if config['id']!=cid:raise ValueError('SAVED_ROUTE_CONFIGURATION_ID_DIFFERS')
                inputs=reader.read(p['case_source'])
                cases=[row for row in inputs['cases']if(row.get('original_configuration_id')or row.get('configuration_id'))==cid]
                if not cases:raise ValueError('SAVED_ROUTE_CASE_MISSING')
                rule=self.common_route(method,cid,p['configuration_source'],p['signal_source'],p)
                loc=rule['contract_template'];contract=reader.read(dict(loc,selector=loc.get('inputs_selector',[])))
                if inputs['adopted_evaluation']!=contract['adopted_evaluation']:
                    raise ValueError('SAVED_ROUTE_EVALUATION_CONTRACT_DIFFERS')
                for case in cases:
                    if any(case.get(k)!=p[v]for k,v in [('configuration','configuration_source'),('signal_packet','signal_source'),('binding_id','binding_id'),('total_configuration_id','total_configuration_id')]):
                        raise ValueError('SAVED_ROUTE_CASE_BINDING_DIFFERS')
                    if case.get('evaluation_target')!=contract['cases'][0].get('evaluation_target'):
                        raise ValueError('SAVED_ROUTE_EVALUATION_TARGET_DIFFERS')
                expected=context['expected'].get(cid,{})
                if any(expected.get(k)!=p[k]for k in('binding_id','total_configuration_id')):
                    raise ValueError('SAVED_ROUTE_EXPECTED_BINDING_DIFFERS')
                routes.setdefault(method,[]).insert(0,rule)
                c['execution_binding_source']=p['case_source'];c.pop('execution_binding_hold',None)
            except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError)as exc:
                c['execution_binding_hold']=str(exc)
        return routes

    def install_snapshot(self,context,problem,keys):
        """Install immutable READY rows while the producer continues elsewhere.

        Do not mutate/save the producer's state from the queue thread. The
        existing installer and all physical/binding checks remain unchanged.
        """
        with self.lock:
            records=deepcopy([r for r in self.state['records'].values()if r.get('key')in keys and r['status']=='READY_FOR_SEARCH'])
        snapshot=object.__new__(type(self));snapshot.__dict__.update(self.__dict__)
        snapshot.state=dict(self.state,records={r['key']:r for r in records});snapshot.save=lambda:None
        return snapshot.install(context,problem,records),records

    def install(self,context,problem,records):
        from . import rpe_method_optimizer as O,rpe_candidate_preparation as R
        routes=deepcopy(problem['execution_routes'])
        for rec in records:
            if rec['status']!='READY_FOR_SEARCH':continue
            config=value(rec['configuration']);method=rec['method'];binding=rec['intake'];cid=config['id']
            requested=(problem.get('methods',{}).get(method,{})or{}).get('requested_delta_pf_values')
            if requested is not None and dec(config['spec']['delta_pf'])not in{dec(v)for v in requested}:
                continue
            try:
                configuration, packet = intake_locators(rec)
            except (KeyError, ValueError, subprocess.CalledProcessError) as error:
                rec['installation_hold'] = str(error)
                continue
            rec.pop('installation_hold', None)
            # Populate original B2 physical keys from saved metadata only. R.register
            # owns the equivalence/alias decision; no new schedule rule is defined.
            for candidate in context['candidates']:
                if candidate['method']!=method or not candidate.get('configuration_id')or candidate.get('schedule_keys')is not None:continue
                try:existing,_=Selection.configuration_for(context,candidate['configuration_id'],value)
                except ValueError:continue
                candidate['schedule_keys']=O.schedule_keys(existing)
            def same_placeholder(candidate):
                if candidate.get('configuration_id')is not None:return False
                k=candidate.get('key',{})
                if rec.get('request', {}).get('horizon_parent') and candidate.get('M', k.get('M')) != config['spec']['M']:
                    return False
                if k.get('L_D')!=config['spec']['L_D']or k.get('formula',config['spec']['formula'])!=config['spec']['formula']:
                    return False
                width=(dec(k['delta_pf'])if k.get('delta_pf')is not None else
                       dec(config['spec']['h'])*dec(k.get('delta_over_h','1')))
                return width==dec(config['spec']['delta_pf'])
            old=next((c for c in context['candidates']if c['method']==method and
                (c.get('configuration_id')==cid or same_placeholder(c))),None)
            ready_before={c['configuration_id']:c for c in context['candidates']if c['method']==method and c.get('ready')and c.get('configuration_id')}
            item=dict(status='GENERATED',configuration=deepcopy(config),configuration_id=cid,
                configuration_source=rec['configuration'],L_D=config['spec']['L_D'],
                readiness=dict(configuration_complete=True),requested_changes={})
            # Legacy register() uses a smaller schedule key. Do not let that
            # key alias different dictionaries, targets, SOURCE or law contracts.
            hidden=[];new_keys=O.schedule_keys(config)
            for candidate in context['candidates']:
                if candidate['method']!=method or candidate.get('schedule_keys')!=new_keys:continue
                try:existing,_=Selection.configuration_for(context,candidate['configuration_id'],value)
                except (ValueError,KeyError):existing=None
                if existing is None or physical_evaluation_contract(existing)!=physical_evaluation_contract(config):
                    if candidate.get('configuration_id')==cid:raise ValueError('CANONICAL_CONFIGURATION_PHYSICAL_CONTRACT_DIFFERS')
                    hidden.append((candidate,candidate.get('schedule_keys')));candidate['schedule_keys']=None
            try:
                disposition=R.register(method,old,item,context['candidates'],rec.get('directory',self.directory),
                    key=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],
                        delta_over_h=str(dec(config['spec']['delta_pf'])/dec(config['spec']['h']))))
            finally:
                for candidate,keys in hidden:candidate['schedule_keys']=keys
            alias=disposition.split(':',1)[1]if disposition.startswith('PHYSICALLY_IDENTICAL_TO:')else None
            installed=alias or cid
            rec.update(installed_configuration_id=installed,installation_disposition=disposition)
            if alias is not None and installed in ready_before:
                # Keep the original candidate, its own prices/evaluations and route.
                # The new declaration remains visible as B2's physical_alias record.
                continue
            if alias:
                rec.update(status='HELD_PHYSICAL_ALIAS_CANONICAL_NOT_READY',
                    reason='Preserve existing partial prices/Q3; prepared metadata needs rebinding to the original canonical view')
                continue
            cand=next(c for c in context['candidates']if c['method']==method and c.get('configuration_id')==installed)
            cand.update(kappa_value=str(dec(config['spec']['kappa']['value'])),ready=True,needs=[],
                generated=True,generated_configuration=config,configuration_source=rec['configuration'],
                signal_source=packet,prices=[dec(x)for x in rec['prices']],
                price_status='MODEL_REFERENCE',price_quote=value(rec['quote']),
                schedule_keys=O.schedule_keys(config),prepared_by='D2_COMMON_CANDIDATE_COMPLETION')
            context['expected'][cid]=dict(configuration_id=cid,binding_id=binding['binding_id'],total_configuration_id=binding['total_configuration_id'])
            rule=self.common_route(method,cid,configuration,packet,binding)
            parent = rec.get('request',{}).get('horizon_parent')
            if parent:
                from run_rpe_optimizer_q3_job import M13_RISK
                source_id = value(parent['configuration'])['id']
                if config['spec']['M'] != 13 or len(cand['prices']) != 2*(config['spec']['M']+1):
                    raise ValueError('Native DF horizon requires complete per-axis prices')
                cand.update(M=config['spec']['M'],source_configuration_id=source_id)
                context['expected'][cid]['M'] = config['spec']['M']
                rule.update(horizon_parent=parent,horizon_risk_source=deepcopy(M13_RISK))
            routes[method]=[rule]+[old for old in routes.get(method,[])if old!=rule]
        routes=self.install_saved_routes(context,routes)
        self.save()
        return routes


def prepare_for_search(root,problem,context,spec,run,*,selection_results=None,completion=None,stage_executor=None):
    """V2 chooses missing inputs; this callback completes them and returns ready inputs.

    A finite demonstration may bound completions per method; unknown/failed
    stages are recorded and independent candidates retain their V2 request order.
    """
    from . import rpe_method_optimizer as O
    settings=deepcopy(spec['candidate_completion'])
    if not(settings.get('price_registry_source')or settings.get('price_registry_file')):
        supplied=(problem.get('price_rules')or{}).get('registry_file')
        if supplied:settings['price_registry_file']=str((Path(root)/supplied).resolve())
    if spec.get('science_approved')is not True:raise ValueError('Candidate acquisition requires the adopted science scope')
    old_state=Path(run)/'completion/STATE.json'
    settings.setdefault('source',read(old_state)['source']if old_state.exists()else git('rev-parse','HEAD'))
    completion=completion or Completion(Path(run)/'completion',settings)
    if stage_executor is not None:completion.bind_stage_executor(stage_executor)
    completion.ingest_price_results()
    for record in list(completion.state['records'].values()):
        if record['status'].startswith(('HELD','STOP','WAITING')):completion.refresh_price_hold(record)
    ready=[r for r in completion.state['records'].values()if r['status']=='READY_FOR_SEARCH']
    routes=completion.install(context,problem,ready)
    search=dict(problem['search'],**spec.get('search_overrides',{}))
    frozen=dict(methods=spec['methods'],method_scope={m:problem['methods'][m]for m in spec['methods']},
        design=settings.get('new_candidate_design'),max_completed_per_method=settings.get('max_completed_per_method'),source=settings['source'])
    put(completion.directory/'INPUTS.json',frozen);publish([completion.directory/'INPUTS.json'],'Fix finite PR preparation scope and SOURCE')
    def one(method):
        current=[r for r in completion.state['records'].values()if r['method']==method]
        if getattr(completion,'stage_executor',None)is not None:
            # Reconcile the original durable task before counting attempted
            # candidates. It may have finished while this controller was down.
            for record in current:
                if not record.get('scientific_failure')and(record['status']=='SELECTED'or record.get('recoverable_preparation')):
                    completion.complete(deepcopy(record['request']))
            current=[r for r in completion.state['records'].values()if r['method']==method]
        maximum=settings.get('max_completed_per_method')
        if isinstance(maximum,dict):maximum=maximum.get(method)
        if maximum is not None and sum(r['status']=='READY_FOR_SEARCH'for r in current)>=maximum:return current
        deferred=[r for r in current if not r.get('scientific_failure') and (
                  r.get('signal_deferred_for_price') and r['status']=='WAITING_INPUTS'
                  or r['status'] in DF_PRICE_RECHECK_STATES)]
        if deferred:
            blocked=completion.price_deferred_keys()
            for record in deferred:
                if record['key'] in blocked:continue
                resumed=completion.complete(deepcopy(record['request']))
                if resumed['status']=='READY_FOR_SEARCH':
                    return [r for r in completion.state['records'].values()if r['method']==method]
        attempted={Selection.identity(r['request']['selection'])for r in current if r['request'].get('selection')}
        limit=settings.get('max_attempted_per_method')
        remaining=None if limit is None else max(0,limit-len(current))
        while remaining is None or remaining>0:
            if remaining is not None:remaining-=1
            # Uses the published B2 rule verbatim; no local LD/kappa priority implementation.
            answer=(selection_results or{}).get(method)
            if answer is None:
                answer=O.optimize(root,dict(problem,search=search),context=context,methods=[method],max_new={method:0},science_approved=False,
                    initial=spec.get('initial','all'))[method]
            requests=answer.get('input_requests')or[]
            selection=next((r for r in requests if r['kind']in('NEW_L_D','CONFIGURATION_INPUTS_MISSING','KAPPA_PROPOSAL')and Selection.identity(r)not in attempted),None)
            if selection is None:break
            attempted.add(Selection.identity(selection))
            req=Selection.normalize(method,selection,context,settings.get('new_candidate_design',{}),value)
            put(completion.directory/(method+'_LATEST_SELECTION.json'),req)
            result=completion.complete(req);current.append(result)
            if result['status']=='READY_FOR_SEARCH':break
        return current
    preparation_methods=[m for m in spec['methods']if m in ('PR_PAULI','PR_DF')]
    if preparation_methods:
        with ThreadPoolExecutor(max_workers=len(preparation_methods))as pool:
            for _ in pool.map(one,preparation_methods):pass
    records=list(completion.state['records'].values())
    routes=completion.install(context,problem,records)
    put(completion.directory/'RETURN_TO_SEARCH.json',dict(source=completion.source,records=records,
        ready=[r['configuration_id']for r in records if r['status']=='READY_FOR_SEARCH'],selection_rule=search.get('rule')))
    if settings.get('evaluate_completed_only',False):
        # Integration acceptance scope: V2 still chooses both missing candidates
        # and their shot proposals. Existing ready baselines stay out of new Q3.
        allowed={r.get('installed_configuration_id',r['configuration_id'])for r in records if r['status']=='READY_FOR_SEARCH'}
        context['candidates']=[c for c in context['candidates']if c.get('configuration_id')in allowed]
    return context,routes,completion


def continue_selected_preparation(root,problem,context,spec,run,completion,executor,result,extra_results=(),optimizer=None):
    """Consume subsequent V2 requests with the same cumulative Q3 budget.

    No new chooser or independent scientific worker loop: each continuation is
    the normal optimizer with the same Executor/job keys. Failed requests are
    retained and skipped, while independent missing inputs remain eligible.
    """
    from . import rpe_method_optimizer as O
    cycles=[]
    while True:
        cycles.append(dict(ready=[r.get('configuration_id')for r in completion.state['records'].values()if r['status']=='READY_FOR_SEARCH'],
            methods={m:dict(stop=r['stop_reason'],new_calls=r['new_calls'],launched=executor.launched(m))for m,r in result.items()}))
        put(completion.directory/'SEARCH_CYCLES.json',dict(cycles=cycles,cumulative_budget=spec['max_new_q3_per_method']))
        # Continuation shares the original search's STOP, including a fatal
        # answer returned by an early READY candidate or a later cycle.
        if any(r.get('stop_reason')in {'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}for r in result.values()):
            break
        remaining=[]
        for method in spec['methods']:
            maximum=spec['max_new_q3_per_method'][method]
            if maximum in (None,'UNLIMITED')or executor.launched(method)<maximum:remaining.append(method)
        pending={m:result[m]if m in remaining else dict(input_requests=[])for m in spec['methods']}
        if not any(r.get('input_requests')for r in pending.values()):break
        before={k for k,v in completion.state['records'].items()if v['status']=='READY_FOR_SEARCH'}
        context,routes,_=prepare_for_search(root,problem,context,spec,run,selection_results=pending,completion=completion)
        after={k for k,v in completion.state['records'].items()if v['status']=='READY_FOR_SEARCH'}
        if after==before:break
        executor.routes=routes;executor.expected=context['expected']
        O.apply_published(context['candidates'],routes)
        result=(optimizer or O.optimize)(root,problem,context=context,methods=spec['methods'],initial=spec.get('initial','all'),
            max_new=spec['max_new_q3_per_method'],executor=executor,extra_results=extra_results,science_approved=True)
    return result


def publish_comparison_exports(completion,run,context):
    """Publish each acquired candidate's actual Q3 answer for unchanged C sync."""
    from .rpe_method_optimizer import job_outcome
    from . import rpe_q3_evidence as evidence
    run=Path(run);exports=[]
    for record in completion.state['records'].values():
        if record['status']!='READY_FOR_SEARCH':continue
        cid=record['configuration_id'];method=record['method']
        for job in sorted((run/'jobs'/method).glob('*')):
            if not(job/'OUTCOME.json').exists()or not(job/'REQUEST.json').exists():continue
            request=read(job/'REQUEST.json');outcome,outcome_path=job_outcome(job)
            if request['configuration_id']!=cid or outcome['status']not in ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD','INCONCLUSIVE'):continue
            leaf=job/job.name;result=leaf/'RESULT.json'
            if not result.exists():continue
            api=leaf/'API_RETURN.json'
            witness=leaf/evidence.WITNESS
            if not api.exists() and not witness.exists() and (leaf/evidence.CHECKPOINTS).is_file():
                # A successfully validated job whose retained sidecars differ
                # from the result still publishes its actual worker API bytes.
                preserved_api=evidence.checkpoints(leaf)['API_RETURN.json']
                with api.open('xb') as stream:
                    stream.write(preserved_api)
            field, artifact = ('api',api) if api.is_file() else ('compact_evidence',witness)
            if not artifact.is_file():raise ValueError('Published Q3 evidence is missing for '+job.name)
            raw=publish([job],'Publish automated candidate Q3 raw and individual receipts')
            d=completion.directory/'comparisons'/method/job.name;d.mkdir(parents=True,exist_ok=True)
            req=dict(configuration=record['intake']['configuration'],case_inputs=dict(commit=raw,path=rel(job/'CASE_INPUTS.json')),
                result=dict(commit=raw,path=rel(result)),outcome=dict(commit=raw,path=rel(outcome_path)),
                quote=record['quote'],comparison_label=method+'_AUTO_'+cid[:12]+'_'+job.name)
            req[field]=dict(commit=raw,path=rel(artifact))
            inp=d/'EXPORT_REQUEST.json';put(inp,req);script=str(ROOT/'scripts/export_pr_candidate_comparison.py')
            completion.call([completion.path("NATIVE"),'-B',script,'prices','--request',str(inp),'--output',str(d/'AXIS_PRICES.json')],d,'EXPORT_PRICES')
            pc=publish([d],'Bind automated candidate 26 axis prices for C')
            completion.call([completion.path("NATIVE"),'-B',script,'inputs','--request',str(inp),'--prices',json.dumps(dict(commit=pc,path=rel(d/'AXIS_PRICES.json'))),'--output',str(d/'INPUTS.json')],d,'EXPORT_INPUTS')
            ic=publish([d],'Fix automated candidate cost/Q3 consumer inputs')
            completion.call([completion.path("NATIVE"),'-B',script,'join','--inputs',json.dumps(dict(commit=ic,path=rel(d/'INPUTS.json'))),'--output-directory',str(d)],d,'EXPORT_JOIN')
            jc=publish([d],'Publish automated PR comparison through existing C schema')
            check=d/'C_SYNC_CHECK.json'
            completion.call([completion.path("NATIVE"),'-B',str(ROOT/'scripts/check_pr_candidate_c_sync.py'),'--join',jc+':'+rel(d/'C_RESULT_JOIN.json'),'--comparison-only','--output',str(check)],d,'C_READ_SYNC_CHECK',check)
            exports.append(dict(method=method,configuration_id=cid,job=job.name,source=dict(commit=jc,path=rel(d/'C_RESULT_JOIN.json')),C_sync_check=rel(check)))
    put(completion.directory/'C_EXPORTS.json',dict(exports=exports,source=completion.source,scientific_calls_in_export=0))
    return exports
