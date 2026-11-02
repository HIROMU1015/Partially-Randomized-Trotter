"""Two future one-shot stages. No execution authorization is emitted here."""
import io
import json
import os
from pathlib import Path
import time
import threading
from .identity import require, Stop, sha, fingerprint
from .gates import authorize, checkout_gate, reexecution_metadata, DISTANCES, OUTPUT, Permit
from .resources import OutputBudget, OUTPUT_CAP, WallBudget, Monitor, admission, observe_memory, limit_owned_address_space
from .ledger import Ledger, wire, json_bytes


def _generate_worker(permit,distance):
    from .inputs import generate_input,freeze_input
    arrays=generate_input(permit,distance,lambda:None)
    return freeze_input(permit,distance,arrays)


def _compile_worker(circuit,options):
    from qiskit import transpile
    from .circuits import metrics
    return metrics(transpile(circuit,**options))


class OwnedRun:
    def __init__(self,permit,authorization,*,prior_wall=0,handoff=False,prior_charge=0,prepared_budget=None):
        run_started=time.monotonic()
        # A new role cannot inherit old-host approval. This is a future path;
        # all new-host preparation drafts keep runtime_authorization=false.
        from .observer import IndependentObserver, AS_CAP, RSS_CAP, FRAME_CAP, TERMINAL_RESERVE
        role = authorization.get('observer_role', {})
        require(role.get('runtime_authorization') is True and role.get('approved') is True and
                role.get('AS_bytes') == AS_CAP and role.get('RSS_bytes') == RSS_CAP,
                'new independent observer role needs separate production approval')
        observation=observe_memory()
        newhost=permit.plan.get('schema_version')=='h4-newhost-plan-v2'
        if newhost:
            from .launch_binding import authorize as authorize_new, FILE_LIMITS, CONTROL_LOG_CAP, roles
            authorize_new(permit.plan,authorization,permit.review,explicit_launch=True)
            self.workers=permit.plan['requested_workers']
            require(permit.launch_observation is not None and
                    observation['oom_events']==permit.launch_observation['memory']['oom_events'] and
                    0<=time.monotonic()-permit.launch_observation['observed_monotonic']<=5,
                    'OOM/cgroup changed or startup observation stale')
            require(observation['available'] >= (8+8*self.workers+16)*2**30+AS_CAP,'newhost observer admission')
            require(set(os.sched_getaffinity(0))==set(permit.plan['cpu_proposal']['driver']),'newhost driver CPU binding')
        else:
            self.workers=admission(observation['available']-AS_CAP,observation['observed_at'],permit.plan['requested_workers'],
                                   authorization['allowed_cpus'],observation['process_cpus'])
        # No affinity edits. A narrower CPU permission requires a separately scoped launch context.
        require(set(os.sched_getaffinity(0)) <= set(authorization['allowed_cpus']), 'process can use unpermitted CPU')
        self.wall=WallBudget(prior_wall)
        self.wall.start=run_started
        limit_owned_address_space()
        self.budget=prepared_budget or OutputBudget(permit.plan['output_root'] if newhost else OUTPUT,handoff=handoff,prior_charge=prior_charge,
                                cap=permit.plan['caps']['output_bytes'] if newhost else OUTPUT_CAP,
                                file_limits=FILE_LIMITS if newhost else None)
        self.close_budget=prepared_budget is None
        trace_cap = (72*3600+2)*FRAME_CAP+TERMINAL_RESERVE
        # Full append-only trace is reserved before the observer can write;
        # reserve conservatively charges temp+final+journal even for one file.
        import sys
        try:
            if prepared_budget is None:
                self.budget.reserve(trace_cap+FRAME_CAP)  # includes driver first-stop file
                if newhost:self.budget.reserve(CONTROL_LOG_CAP+65536)
            self.monitor=IndependentObserver(sys.executable, self.budget.root/'observer.jsonl',
                scope='PRODUCTION', runtime_authorization=True, workers=self.workers,
                prior_wall=prior_wall, wall_started=self.wall.start, output_cap=trace_cap,
                allowed_cpus=authorization['allowed_cpus'] if newhost else None,
                role_cpus=permit.plan['cpu_proposal']['observer'] if newhost else None)
        except BaseException:
            self.budget.close(); raise
        self.finished=threading.Event();self.failure=None
        self.stage=permit.stage
        self.wall_index=0
        self.thread=threading.Thread(target=self._watch,name='owned-h4-monitor',daemon=True)
        self.thread.start()
        try:
            from .workers import OwnedPool
            self.pool=OwnedPool(self.workers,permit,self.monitor,self.budget)
        except BaseException:
            self.finished.set();self.thread.join(timeout=6)
            self.monitor.stop_children()
            self.monitor.close(abort=True)
            self.budget.close();raise

    def _watch(self):
        try:
            while not self.finished.wait(1):
                self.monitor.poll()
                self.wall.consumed()
                # Independent trace records cumulative wall, including observer
                # startup/shutdown; no second one-file-per-tick wall inventory.
                self.wall_index+=1
        except BaseException as exc:
            self.failure=exc
            # Retain the first cause before cleanup can replace it with a pipe
            # or interrupt failure. This is the driver's existing bounded log.
            try:print('H4 MONITOR STOP: '+type(exc).__name__+': '+str(exc),flush=True)
            finally:
                self.monitor.stop_children()
                # Cleanup still occurs if the bounded driver log is exhausted.
                import _thread
                _thread.interrupt_main()

    def pulse(self):
        require(self.failure is None,'owned monitoring failed STOP: '+str(self.failure))
        pool_failure = getattr(getattr(self,'pool',None),'failure',None)
        require(pool_failure is None,'owned pool failed STOP: '+str(pool_failure))
        self.wall.consumed()
        if hasattr(self.monitor, 'poll'):
            self.monitor.poll()

    def phase(self, name):
        self.pulse()
        self.monitor.phase(name)

    def wait(self,future):
        from concurrent.futures import TimeoutError
        while True:
            self.pulse()
            try:
                return future.result(timeout=1)
            except TimeoutError:
                pass

    def wait_any(self,futures):
        from concurrent.futures import wait,FIRST_COMPLETED
        while True:
            self.pulse()
            done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
            if done:
                self.pulse()
                return done

    def abort(self):
        self.finished.set()
        self.monitor.stop_children()

    def close(self):
        self.finished.set();self.thread.join(timeout=6)
        self.monitor.stop_children()
        try:
            try:self.pool.shutdown(wait=True,cancel_futures=True)
            finally:self.monitor.close()
            self.wall.consumed()  # charge through pool + observer reap and FD cleanup
            require(not self.thread.is_alive(), 'owned monitor watchdog did not end')
        finally:
            if self.close_budget:self.budget.close()
        require(self.failure is None,'monitor STOP; no retry/resume: '+str(self.failure))


def generation_stage(permit,authorization,options):
    require(permit.stage=='input_generation','generation stage')
    run=OwnedRun(permit,authorization)
    records={};identities={}
    try:
        futures={d:run.pool.submit(_generate_worker,permit,d) for d in DISTANCES}
        for d in DISTANCES:
            data,record,identity=run.wait(futures[d]);run.pulse()
            run.budget.write(record['file'],data)
            records[d],identities[d]=record,identity
        freeze={'stage':'INPUTS_FROZEN_STOP','run_id':permit.plan['run_id'],
                'source_commit':permit.plan['source_commit'],'generation_review_digest':permit.review_digest,
                'inputs':records,'identities':identities,'consumed_seconds':run.wall.consumed(),
                'research_decision':None,'next_stage_authorized':False,'mandatory_stop':True}
        run.budget.write('generation-freeze.json',json_bytes(freeze))
        return freeze  # no signal continuation and no plan/authorization emitted
    finally:
        run.close()


def input_boundary(permit):
    require(permit.stage=='signal_compile','separate signal permit')
    if permit.plan.get('schema_version')=='h4-newhost-plan-v2':
        from .launch_binding import verify_frozen_receipts
        return verify_frozen_receipts(permit.plan)
    reuse = reexecution_metadata(permit) if 'reexecution' in permit.plan else None
    input_root = Path(reuse['input_root']) if reuse else Path(OUTPUT)
    require(not any(p.is_symlink() for p in [input_root,*input_root.parents]), 'input root symlink')
    freeze_path=input_root/'generation-freeze.json'
    require(not freeze_path.is_symlink(),'freeze symlink')
    freeze=json.loads(freeze_path.read_bytes())
    require(fingerprint('h4-generation-freeze-v1',freeze)==permit.plan['generation_freeze_digest'],'generation freeze binding')
    expected_source = reuse['generation_source_commit'] if reuse else permit.plan['source_commit']
    require(freeze['stage']=='INPUTS_FROZEN_STOP' and freeze['inputs']==permit.plan['inputs'] and
            freeze['source_commit']==expected_source and freeze['mandatory_stop'] is True,'frozen inputs')
    if reuse:
        require(sha(freeze_path.read_bytes()) == reuse['generation_freeze_file_sha256'], 'original freeze bytes')
        require(sha((input_root/'byte-budget.journal').read_bytes()) == reuse['prior_journal_sha256'], 'stopped predecessor budget bytes')
        return {**freeze, 'input_root':str(input_root), 'consumed_seconds':reuse['prior_wall_seconds'],
                'prior_charge':reuse['prior_cumulative_charge_bytes'], 'prior_invocations':reuse['prior_actual_invocations']}
    return freeze


def load_new_input(permit,distance,freeze):
    require(permit.stage=='signal_compile','signal input gate')
    import numpy as np
    from .inputs import array_identity
    entry=permit.plan['inputs'][distance]
    require(entry['file']=='input-'+distance+'.npz','input path scope')
    path=Path(freeze.get('input_root',OUTPUT))/entry['file']
    require(not path.is_symlink(),'input symlink')
    data=path.read_bytes();require(sha(data)==entry['bytes_sha256'],'frozen input bytes')
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    expected=freeze['identities'][distance]
    require({k:array_identity(v) for k,v in sorted(arrays.items())}==expected['arrays'],'complete input array identities')
    require(fingerprint('h4-input-freeze-v1',expected)==entry['input'],'input fingerprint')
    return arrays


def candidate_wrapper_jobs(run,identity,seeds,preparation,template):
    """Generate one evolution and its two axes lazily in trajectory order."""
    from .signal import sample_events
    from .circuits import build_evolution,wrapper,numerical_fingerprint
    for index,seed in enumerate(seeds):
        run.pulse()
        if hasattr(run,'phase'):run.phase('candidate_events_build')
        events=sample_events(preparation['components'],template,seed)[0] if seed is not None else [[] for _ in range(template['q'])]
        evolution=build_evolution(preparation,template,events)
        for axis in ('cosine','sine'):
            if hasattr(run,'phase'):run.phase('wrapper_build:'+axis)
            run.pulse();circuit=wrapper(evolution,axis)
            if hasattr(run,'phase'):run.phase('numerical_serialization:'+axis)
            numerical=numerical_fingerprint(circuit,axis)
            run.pulse()
            if hasattr(run,'phase'):run.phase('wrapper_ready:'+axis)
            yield wire(identity,axis,seed,index if seed is not None else None,numerical),circuit


def signal_stage(permit,authorization,options,*,launch_started=None,prepared_budget=None):
    require(permit.stage=='signal_compile','separate signal stage')
    freeze=input_boundary(permit)
    reused='reexecution' in permit.plan
    newhost=permit.plan.get('schema_version')=='h4-newhost-plan-v2'
    if launch_started is not None:freeze={**freeze,'consumed_seconds':freeze['consumed_seconds']+time.monotonic()-launch_started}
    run=OwnedRun(permit,authorization,prior_wall=freeze['consumed_seconds'],handoff=not reused and not newhost,
                 prior_charge=freeze.get('prior_charge',0),prepared_budget=prepared_budget)
    ledger=None
    from .signal import GeometryPreparation,prepare,corrected_signal,candidate_identity,trajectory_seeds,display_map
    from .parallel import compile_candidates
    from .inputs import validate_frozen_state
    signals=[];reuse={}

    def candidates():
        import numpy as np
        for distance in DISTANCES:
            run.pulse();arrays=load_new_input(permit,distance,freeze)
            validate_frozen_state(arrays)
            common=GeometryPreparation(arrays)
            inp={'geometry':distance,**{k:permit.plan['inputs'][distance][k] for k in ('input','H','DF','state')}}
            for template in permit.plan['templates']:
                run.pulse()
                identity=candidate_identity(inp,template,permit.plan['source_commit'],
                    permit.plan['compiler_fingerprint'],permit.plan['environment_fingerprint'])
                preparation,det,tail=prepare(arrays,template,common=common)
                signal_record=corrected_signal(det,tail,preparation['constant'],arrays['qiskit_state'],template)
                target=complex(np.exp(-1j*float(arrays['energy'])*template['T']))
                seeds=trajectory_seeds(identity) if template['method'] in ('B2','B3') else [None]
                metadata={'geometry':distance,'template':template,'identity':identity,
                          'signal_record':signal_record,'target':target}
                yield metadata,candidate_wrapper_jobs(run,identity,seeds,preparation,template),2*len(seeds)

    def publish(metadata,records):
        distance,template,identity=(metadata[k] for k in ('geometry','template','identity'))
        signal_record,target=metadata['signal_record'],metadata['target']
        all_metrics=[[records[i]['metrics'],records[i+1]['metrics']] for i in range(0,len(records),2)]
        costs=[[m['rz_count'] for m in pair] for pair in all_metrics]
        saved={'normalization':signal_record['normalization'],
               'bias':{'cosine':abs(signal_record['corrected'].real-target.real),
                       'sine':abs(signal_record['corrected'].imag-target.imag)},'paired_costs':costs}
        result={'geometry':distance,'template_id':template['template_id'],'identity':identity,
                'candidate_fingerprint':fingerprint('h4-candidate-v1',identity),
                'signal':{'corrected':[signal_record['corrected'].real,signal_record['corrected'].imag],
                          'raw':[signal_record['raw'].real,signal_record['raw'].imag],'normalization':saved['normalization']},
                'saved':saved,'paired_metrics':all_metrics,'display':display_map(saved),
                'research_decision':None,'next_stage_authorized':False}
        run.budget.write('signal-'+distance+'-'+template['template_id']+'.json',json_bytes(result))
        signals.append(result)
        print('H4 candidate COMPLETE %d/1308 %s %s'%(len(signals),distance,template['template_id']),flush=True)

    try:
        invocation_cap=permit.plan['caps']['actual_invocations'] if newhost else 74784
        ledger=Ledger(run.budget,cap=invocation_cap,prior_invocations=freeze.get('prior_invocations',0))
        compile_candidates(run,ledger,candidates(),_compile_worker,options,reuse,
                           expected_candidates=1308,expected_wrappers=74784,on_candidate=publish)
        accounting=ledger.audit()
        require(len(signals)==1308 and accounting['logical_wrappers']==74784 and accounting['actual_invocations']<=invocation_cap,'fixed campaign accounting')
        final={'status':'MAP_COMPLETE_STOP','signal_records':len(signals),**accounting,
               'prior_actual_invocations':freeze.get('prior_invocations',0),
               'ledger_head_digest':ledger.chain,'consumed_seconds':run.wall.consumed(),
               'research_decision':None,'next_stage_authorized':False,'mandatory_stop':True}
        run.budget.write('map-complete.json',json_bytes(final))
        return final
    finally:
        if ledger is not None:ledger.close()
        run.close()


def launch(stage,plan,authorization,review,*,explicit_launch=False):
    if plan.get('schema_version')=='h4-newhost-plan-v2':
        require(stage=='signal_compile','newhost frozen-input signal stage only')
        from .launch_binding import launch as new_launch
        return new_launch(plan,authorization,review,explicit_launch=explicit_launch)
    permit=authorize(stage,plan,authorization,review,explicit_launch=explicit_launch)
    contract,options=checkout_gate(permit)
    if stage=='input_generation':
        return generation_stage(permit,authorization,options)
    return signal_stage(permit,authorization,options)
